from models.Models import ClimateVAE
from models.LossFunctions import compute_loss, masked_l1_loss
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List
import torch
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
    """Train one epoch of VQ-VAE-2"""
    model.train()
    losses = {'total': 0, 'recon': 0, 'vq': 0, 'perp_top': 0, 'perp_bottom': 0}
    use_amp = device == 'cuda'
    if scaler is None:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    mask = mask.to(device)

    for x_seq, _ in tqdm(loader):
        x_seq = x_seq.to(device)
        B, T = x_seq.shape[:2]
        x = x_seq.reshape(B * T, *x_seq.shape[2:])

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            out = model(x)
            recon_loss = masked_l1_loss(x, out['recon'], mask)
            total = recon_loss + out['vq_loss']

        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        losses['total'] += total.item()
        losses['recon'] += recon_loss.item()
        losses['vq'] += out['vq_loss'].item()
        losses['perp_top'] += out['perplexity_top'].item()
        losses['perp_bottom'] += out['perplexity_bottom'].item()

    n = len(loader)
    return {k: v / n for k, v in losses.items()}