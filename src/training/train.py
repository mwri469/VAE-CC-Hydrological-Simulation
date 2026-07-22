from models.Models import ClimateVAE
from models.LossFunctions import compute_loss, masked_l1_loss
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List
import torch
import torch.nn.functional as F
from tqdm import tqdm

def train_epoch(model: ClimateVAE, loader: DataLoader, optimizer: torch.optim.Optimizer,
                beta: float, config, device: str, mask: torch.Tensor,
                scaler: torch.amp.GradScaler = None) -> Dict:
    """Train one epoch"""
    model.train()
    losses = {'total': 0, 'recon': 0, 'kl': 0, 'rollout': 0}
    use_amp = device == 'cuda'
    if scaler is None:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Move mask to device once
    mask = mask.to(device)
    
    for x_seq, c_seq in tqdm(loader):
        x_seq, c_seq = x_seq.to(device), c_seq.to(device)
        
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            loss_dict = compute_loss(model, x_seq, c_seq, beta,
                                    config.LAMBDA_ROLLOUT, config.K_ROLLOUT, mask)

        scaler.scale(loss_dict['total']).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        for k, v in loss_dict.items():
            losses[k] += v.item()
    
    for k in losses:
        losses[k] /= len(loader)
    
    return losses


def train_epoch_vq(model, loader, optimizer, config, device, mask,
                   scaler: torch.amp.GradScaler = None):
    """Train one epoch of VQ-VAE-2

    Each dataset item is a sequence of T frames, so a DataLoader batch of
    shape (B, T, C, H, W) is flattened to B*T independent images. With a
    large model (wide channels) this flattened batch can make a single
    conv2d kernel run long enough to trigger a Windows TDR (driver
    watchdog) timeout. To avoid that while keeping the larger model, the
    flattened batch is processed in smaller micro-batches with gradient
    accumulation, so gradients still reflect the full batch.
    """
    model.train()
    losses = {'total': 0, 'recon': 0, 'vq': 0, 'perp_top': 0, 'perp_bottom': 0}
    use_amp = device == 'cuda'
    if scaler is None:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    mask = mask.to(device)
    micro_bs = config.VQ_MICRO_BATCH_SIZE

    for x_seq, _ in tqdm(loader):
        x_seq = x_seq.to(device)
        B, T = x_seq.shape[:2]
        x = x_seq.reshape(B * T, *x_seq.shape[2:])
        N = x.shape[0]

        optimizer.zero_grad(set_to_none=True)

        # Suppress EMA updates during gradient accumulation.  The loop runs
        # ceil(N/micro_bs) times per optimizer step; with decay=0.99 that
        # compresses the effective EMA memory ~6x, destabilising the codebook
        # within a few epochs.  EMA is applied exactly once below, after the
        # gradient step, over all encoder outputs concatenated.
        batch_totals = {'total': 0.0, 'recon': 0.0, 'vq': 0.0, 'perp_top': 0.0, 'perp_bottom': 0.0}
        for start in range(0, N, micro_bs):
            x_chunk = x[start:start + micro_bs]
            weight = x_chunk.shape[0] / N

            with torch.amp.autocast('cuda', enabled=use_amp):
                out = model(x_chunk, update_codebook=False)
                recon_loss = masked_l1_loss(x_chunk, out['recon'], mask)
                total = recon_loss + out['vq_loss']

            scaler.scale(total * weight).backward()

            batch_totals['total'] += total.item() * weight
            batch_totals['recon'] += recon_loss.item() * weight
            batch_totals['vq'] += out['vq_loss'].item() * weight
            batch_totals['perp_top'] += out['perplexity_top'].item() * weight
            batch_totals['perp_bottom'] += out['perplexity_bottom'].item() * weight

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        # Single EMA update over the full flattened batch.  Only the encoder
        # and VQ paths are needed here (no decoder), so this is cheap relative
        # to a full forward.  Running without autocast keeps all arithmetic in
        # float32, which is what the EMA buffers expect.
        with torch.no_grad():
            enc_b_chunks, enc_t_chunks = [], []
            for start in range(0, N, micro_bs):
                eb, et = model.encode(x[start:start + micro_bs])
                enc_b_chunks.append(eb.float())
                enc_t_chunks.append(et.float())
            enc_b_all = torch.cat(enc_b_chunks, 0)
            enc_t_all = torch.cat(enc_t_chunks, 0)

            quant_t, _, _, _ = model.vq_top(enc_t_all, update_codebook=True)
            dec_t_all = model.dec_top(quant_t)
            if dec_t_all.shape[2:] != enc_b_all.shape[2:]:
                dec_t_all = F.interpolate(dec_t_all, size=enc_b_all.shape[2:], mode='nearest')
            enc_b_combined_all = model.enc_bottom_combine(
                torch.cat([enc_b_all, dec_t_all], dim=1))
            model.vq_bottom(enc_b_combined_all, update_codebook=True)

        for k in losses:
            losses[k] += batch_totals[k]

    n = len(loader)
    return {k: v / n for k, v in losses.items()}