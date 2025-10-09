from models.Models import ClimateVAE
from models.LossFunctions import compute_loss
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List
import torch
from tqdm import tqdm

def train_epoch(model: ClimateVAE, loader: DataLoader, optimizer: torch.optim.Optimizer,
                beta: float, config, device: str, mask: torch.Tensor) -> Dict:
    """Train one epoch"""
    model.train()
    losses = {'total': 0, 'recon': 0, 'kl': 0, 'rollout': 0}
    
    # Move mask to device once
    mask = mask.to(device)
    
    for x_seq, c_seq in tqdm(loader):
        x_seq, c_seq = x_seq.to(device), c_seq.to(device)
        
        optimizer.zero_grad()
        loss_dict = compute_loss(model, x_seq, c_seq, beta, 
                                config.LAMBDA_ROLLOUT, config.K_ROLLOUT, mask)
        
        loss_dict['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()
        
        for k, v in loss_dict.items():
            losses[k] += v.item()
    
    for k in losses:
        losses[k] /= len(loader)
    
    return losses