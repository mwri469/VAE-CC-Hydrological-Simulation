import math
import torch
from typing import Tuple, Dict, List
from models.Models import ClimateVAE

def reconstruction_loss(x_true: torch.Tensor, x_mean: torch.Tensor,
                       x_logvar: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """Negative log-likelihood (Gaussian) with optional masking"""
    var = torch.exp(x_logvar)
    loss = 0.5 * (torch.log(2 * math.pi * var) + (x_true - x_mean) ** 2 / var)
    
    if mask is not None:
        # mask shape: (1, H, W) or (B, 1, H, W)
        # loss shape: (B, C, H, W)
        B, C = loss.shape[:2]
        
        # Expand mask to match loss dimensions: (B, C, H, W)
        if mask.dim() == 3:  # (1, H, W)
            mask_expanded = mask.unsqueeze(1).expand(B, C, -1, -1)
        else:  # (B, 1, H, W)
            mask_expanded = mask.expand(-1, C, -1, -1)
        
        # Apply mask and normalize by number of valid pixels
        masked_loss = (loss * mask_expanded).sum()
        num_valid = mask_expanded.sum() + 1e-8  # Avoid division by zero
        return masked_loss / num_valid
    else:
        return loss.sum(dim=[1, 2, 3]).mean()

def kl_divergence(mu_q: torch.Tensor, logvar_q: torch.Tensor,
                  mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
    """KL(q||p) for Gaussians"""
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
   
    kl = 0.5 * (logvar_p - logvar_q + var_q / var_p +
                (mu_q - mu_p) ** 2 / var_p - 1)
    return kl.sum(dim=-1).mean()

def compute_loss(model: ClimateVAE, x_seq: torch.Tensor, c_seq: torch.Tensor,
                 beta: float, lambda_rollout: float, k_rollout: int, 
                 mask: torch.Tensor = None) -> Dict:
    """
    Complete loss computation with optional masking
    
    Args:
        model: ClimateVAE model
        x_seq: (B, T, C, H, W) input sequence
        c_seq: (B, T, 3) conditioning sequence
        beta: KL weight
        lambda_rollout: Rollout loss weight
        k_rollout: Number of rollout steps
        mask: (1, H, W) or (B, 1, H, W) - binary mask (1=land, 0=ocean)
    """
    outputs = model(x_seq, c_seq)
    T = x_seq.shape[1]
   
    # Reconstruction loss
    recon_loss = 0
    for t in range(T):
        x_mean, x_logvar = outputs['recons'][t]
        recon_loss += reconstruction_loss(x_seq[:, t], x_mean, x_logvar, mask)
    recon_loss /= T
   
    # KL loss (no masking needed - it's in latent space)
    kl_loss = 0
    for t in range(T):
        kl_loss += kl_divergence(
            outputs['mu_post'][:, t], outputs['logvar_post'][:, t],
            outputs['mu_prior'][:, t], outputs['logvar_prior'][:, t]
        )
    kl_loss /= T
   
    # Rollout loss (optional)
    rollout_loss = 0
    if lambda_rollout > 0 and k_rollout > 0:
        z = outputs['z_post'][:, 0]
        h = None
        for k in range(1, min(k_rollout, T)):
            c_k = c_seq[:, k]
            mu_p, logvar_p, h = model.transition(z, c_k, h)
            z = model.reparameterize(mu_p, logvar_p)
            x_mean, x_logvar = model.decoder(z, c_k)
            rollout_loss += reconstruction_loss(x_seq[:, k], x_mean, x_logvar, mask)
        rollout_loss /= k_rollout
   
    total_loss = recon_loss + beta * kl_loss + lambda_rollout * rollout_loss
   
    return {
        'total': total_loss,
        'recon': recon_loss,
        'kl': kl_loss,
        'rollout': rollout_loss
    }