"""
Climate VAE with Latent Dynamics for Long-term Weather Generation
PyTorch implementation for training and generating multi-variable climate fields
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List
import math

# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Data parameters
    BBOX = {
        'lon_min': 174.253,
        'lon_max': 175.440,
        'lat_min': -37.275,
        'lat_max': -36.589
    }
    VARIABLES = ['tasmax', 'pr', 'PETsrad']
    MODELS = ['ACCESS-CM2']
    
    # Model parameters
    LATENT_DIM = 64
    HIDDEN_DIM = 128
    SEQ_LEN = 64  # days per training sequence
    SPATIAL_SIZE = 32  # downsample to 32x32
    
    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    N_EPOCHS = 100
    BETA_START = 0.0
    BETA_END = 1.0
    BETA_WARMUP_EPOCHS = 30
    LAMBDA_ROLLOUT = 0.1
    K_ROLLOUT = 7
    GRAD_CLIP = 1.0
    
    # Generation parameters
    GEN_YEARS = 1000
    GEN_DAYS = 365 * GEN_YEARS

# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

class ClimateDataset(Dataset):
    """Dataset for loading and preprocessing NetCDF climate data"""
    
    def __init__(self, csv_path: str, config: Config, scenario: str = 'historical'):
        self.config = config
        self.scenario = scenario
        
        # Load file list
        df = pd.read_csv(csv_path)
        df = df[df['scenario'] == scenario]
        self.files = df.groupby('model').apply(
            lambda x: {row['variable']: row['fpath'] for _, row in x.iterrows()}
        ).to_dict()
        
        # Load and preprocess data
        self.data = self._load_data()
        self.conditioning = self._create_conditioning()
        
        print(f"Loaded {len(self.data)} days of data")
        
    def _load_data(self) -> torch.Tensor:
        """Load NetCDF files and crop to bbox"""
        all_data = []
        
        for model, paths in self.files.items():
            model_data = []
            
            for var in self.config.VARIABLES:
                if var not in paths:
                    continue
                    
                # Load NetCDF
                ds = xr.open_dataset(paths[var])
                
                # Crop to bbox
                ds_crop = ds.sel(
                    lon=slice(self.config.BBOX['lon_min'], self.config.BBOX['lon_max']),
                    lat=slice(self.config.BBOX['lat_max'], self.config.BBOX['lat_min'])
                )
                
                # Get variable data
                var_data = ds_crop[var].values  # (time, lat, lon)
                
                # Transform variables
                if var == 'pr':
                    # Handle precipitation: log(x+1) transform
                    var_data = np.log1p(var_data)
                elif var == 'tasmax':
                    # Convert to Celsius if needed
                    if var_data.mean() > 100:
                        var_data = var_data - 273.15
                
                # Normalize (per-pixel)
                mean = np.nanmean(var_data, axis=0, keepdims=True)
                std = np.nanstd(var_data, axis=0, keepdims=True) + 1e-6
                var_data = (var_data - mean) / std
                
                model_data.append(var_data)
            
            # Stack variables: (time, var, lat, lon)
            model_data = np.stack(model_data, axis=1)
            all_data.append(model_data)
        
        # Concatenate all models along time dimension
        all_data = np.concatenate(all_data, axis=0)
        
        # Downsample spatially using interpolation
        all_data = self._downsample(all_data)
        
        return torch.FloatTensor(all_data)
    
    def _downsample(self, data: np.ndarray) -> np.ndarray:
        """Downsample spatial dimensions"""
        T, C, H, W = data.shape
        target = self.config.SPATIAL_SIZE
        
        # Simple interpolation
        data_tensor = torch.FloatTensor(data)
        data_down = F.interpolate(
            data_tensor, size=(target, target), mode='bilinear', align_corners=False
        )
        return data_down.numpy()
    
    def _create_conditioning(self) -> torch.Tensor:
        """Create conditioning vectors for each day"""
        n_days = len(self.data)
        
        # Day of year (cyclic encoding)
        doy = np.arange(n_days) % 365
        doy_sin = np.sin(2 * np.pi * doy / 365)
        doy_cos = np.cos(2 * np.pi * doy / 365)
        
        # Year (normalized)
        year = np.arange(n_days) // 365
        year_norm = year / year.max()
        
        # Stack conditioning
        cond = np.stack([doy_sin, doy_cos, year_norm], axis=1)
        return torch.FloatTensor(cond)
    
    def __len__(self) -> int:
        return len(self.data) - self.config.SEQ_LEN
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return sequence of days and conditioning"""
        seq = self.data[idx:idx + self.config.SEQ_LEN]
        cond = self.conditioning[idx:idx + self.config.SEQ_LEN]
        return seq, cond

# ============================================================================
# Model Components
# ============================================================================

class Encoder(nn.Module):
    """Convolutional encoder: spatial field -> latent distribution"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Conditioning embedding
        self.cond_embed = nn.Linear(3, 32)
        
        # Conv layers
        self.conv1 = nn.Conv2d(len(config.VARIABLES), 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # down
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)  # down
        
        # Bottleneck
        flat_size = 128 * (config.SPATIAL_SIZE // 4) ** 2
        self.fc = nn.Linear(flat_size + 32, config.HIDDEN_DIM)
        
        # Latent parameters
        self.fc_mu = nn.Linear(config.HIDDEN_DIM, config.LATENT_DIM)
        self.fc_logvar = nn.Linear(config.HIDDEN_DIM, config.LATENT_DIM)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) spatial field
            c: (B, 3) conditioning
        Returns:
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        # Encode conditioning
        c_embed = F.relu(self.cond_embed(c))
        
        # Convolutional encoding
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        
        # Flatten and concatenate with conditioning
        h = h.flatten(1)
        h = torch.cat([h, c_embed], dim=1)
        
        # Bottleneck
        h = F.relu(self.fc(h))
        
        # Latent parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, -10, 2)  # stability
        
        return mu, logvar


class Decoder(nn.Module):
    """Decoder: latent -> spatial field"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Conditioning embedding
        self.cond_embed = nn.Linear(3, 32)
        
        # Initial projection
        self.fc = nn.Linear(config.LATENT_DIM + 32, config.HIDDEN_DIM)
        
        init_size = config.SPATIAL_SIZE // 4
        self.fc_reshape = nn.Linear(config.HIDDEN_DIM, 128 * init_size ** 2)
        self.init_size = init_size
        
        # Transposed conv layers
        self.deconv1 = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        
        # Output heads
        self.out_mean = nn.Conv2d(32, len(config.VARIABLES), 3, padding=1)
        self.out_logvar = nn.Conv2d(32, len(config.VARIABLES), 3, padding=1)
    
    def forward(self, z: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, latent_dim)
            c: (B, 3) conditioning
        Returns:
            mean: (B, C, H, W)
            logvar: (B, C, H, W)
        """
        B = z.shape[0]
        
        # Conditioning
        c_embed = F.relu(self.cond_embed(c))
        
        # Project
        h = torch.cat([z, c_embed], dim=1)
        h = F.relu(self.fc(h))
        h = F.relu(self.fc_reshape(h))
        
        # Reshape
        h = h.view(B, 128, self.init_size, self.init_size)
        
        # Deconvolve
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = F.relu(self.deconv3(h))
        
        # Output
        mean = self.out_mean(h)
        logvar = self.out_logvar(h)
        logvar = torch.clamp(logvar, -10, 2)
        
        return mean, logvar


class TransitionModel(nn.Module):
    """Latent dynamics: p(z_t | z_{t-1}, c_t)"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Conditioning embedding
        self.cond_embed = nn.Linear(3, 32)
        
        # GRU-based transition
        self.gru = nn.GRUCell(config.LATENT_DIM + 32, config.HIDDEN_DIM)
        
        # Output parameters
        self.fc_mu = nn.Linear(config.HIDDEN_DIM, config.LATENT_DIM)
        self.fc_logvar = nn.Linear(config.HIDDEN_DIM, config.LATENT_DIM)
    
    def forward(self, z_prev: torch.Tensor, c: torch.Tensor, 
                h: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_prev: (B, latent_dim)
            c: (B, 3)
            h: (B, hidden_dim) GRU hidden state
        Returns:
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
            h_new: (B, hidden_dim)
        """
        B = z_prev.shape[0]
        
        if h is None:
            h = torch.zeros(B, self.config.HIDDEN_DIM, device=z_prev.device)
        
        # Conditioning
        c_embed = F.relu(self.cond_embed(c))
        
        # Transition
        inp = torch.cat([z_prev, c_embed], dim=1)
        h_new = self.gru(inp, h)
        
        # Parameters
        mu = self.fc_mu(h_new)
        logvar = self.fc_logvar(h_new)
        logvar = torch.clamp(logvar, -10, 2)
        
        return mu, logvar, h_new


class ClimateVAE(nn.Module):
    """Complete VAE with latent dynamics"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.transition = TransitionModel(config)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x_seq: torch.Tensor, c_seq: torch.Tensor) -> Dict:
        """
        Args:
            x_seq: (B, T, C, H, W)
            c_seq: (B, T, 3)
        Returns:
            Dictionary with losses and reconstructions
        """
        B, T = x_seq.shape[:2]
        device = x_seq.device
        
        # Storage
        z_post = []
        mu_post = []
        logvar_post = []
        mu_prior = []
        logvar_prior = []
        recons = []
        
        # Prior for t=0
        mu_p = torch.zeros(B, self.config.LATENT_DIM, device=device)
        logvar_p = torch.zeros(B, self.config.LATENT_DIM, device=device)
        h = None
        
        # Encode sequence
        for t in range(T):
            x_t = x_seq[:, t]
            c_t = c_seq[:, t]
            
            # Posterior
            mu_q, logvar_q = self.encoder(x_t, c_t)
            z_t = self.reparameterize(mu_q, logvar_q)
            
            # Reconstruction
            x_mean, x_logvar = self.decoder(z_t, c_t)
            
            # Store
            z_post.append(z_t)
            mu_post.append(mu_q)
            logvar_post.append(logvar_q)
            mu_prior.append(mu_p)
            logvar_prior.append(logvar_p)
            recons.append((x_mean, x_logvar))
            
            # Next prior
            if t < T - 1:
                mu_p, logvar_p, h = self.transition(z_t, c_t, h)
        
        return {
            'z_post': torch.stack(z_post, 1),
            'mu_post': torch.stack(mu_post, 1),
            'logvar_post': torch.stack(logvar_post, 1),
            'mu_prior': torch.stack(mu_prior, 1),
            'logvar_prior': torch.stack(logvar_prior, 1),
            'recons': recons
        }

# ============================================================================
# Loss Functions
# ============================================================================

def reconstruction_loss(x_true: torch.Tensor, x_mean: torch.Tensor, 
                       x_logvar: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood (Gaussian)"""
    var = torch.exp(x_logvar)
    loss = 0.5 * (torch.log(2 * math.pi * var) + (x_true - x_mean) ** 2 / var)
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
                 beta: float, lambda_rollout: float, k_rollout: int) -> Dict:
    """Complete loss computation"""
    outputs = model(x_seq, c_seq)
    T = x_seq.shape[1]
    
    # Reconstruction loss
    recon_loss = 0
    for t in range(T):
        x_mean, x_logvar = outputs['recons'][t]
        recon_loss += reconstruction_loss(x_seq[:, t], x_mean, x_logvar)
    recon_loss /= T
    
    # KL loss
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
            rollout_loss += reconstruction_loss(x_seq[:, k], x_mean, x_logvar)
        rollout_loss /= k_rollout
    
    total_loss = recon_loss + beta * kl_loss + lambda_rollout * rollout_loss
    
    return {
        'total': total_loss,
        'recon': recon_loss,
        'kl': kl_loss,
        'rollout': rollout_loss
    }

# ============================================================================
# Training
# ============================================================================

def train_epoch(model: ClimateVAE, loader: DataLoader, optimizer: torch.optim.Optimizer,
                beta: float, config: Config, device: str) -> Dict:
    """Train one epoch"""
    model.train()
    losses = {'total': 0, 'recon': 0, 'kl': 0, 'rollout': 0}
    
    for x_seq, c_seq in loader:
        x_seq, c_seq = x_seq.to(device), c_seq.to(device)
        
        optimizer.zero_grad()
        loss_dict = compute_loss(model, x_seq, c_seq, beta, 
                                config.LAMBDA_ROLLOUT, config.K_ROLLOUT)
        
        loss_dict['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()
        
        for k, v in loss_dict.items():
            losses[k] += v.item()
    
    for k in losses:
        losses[k] /= len(loader)
    
    return losses


def train(config: Config, csv_path: str, device: str = 'cuda'):
    """Main training loop"""
    # Dataset
    dataset = ClimateDataset(csv_path, config)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, 
                       shuffle=True, num_workers=4)
    
    # Model
    model = ClimateVAE(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                  weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    # Training
    for epoch in range(config.N_EPOCHS):
        # Beta annealing
        if epoch < config.BETA_WARMUP_EPOCHS:
            beta = config.BETA_START + (config.BETA_END - config.BETA_START) * \
                   epoch / config.BETA_WARMUP_EPOCHS
        else:
            beta = config.BETA_END
        
        # Train
        losses = train_epoch(model, loader, optimizer, beta, config, device)
        scheduler.step(losses['total'])
        
        print(f"Epoch {epoch+1}/{config.N_EPOCHS} | Beta: {beta:.3f} | "
              f"Loss: {losses['total']:.4f} | Recon: {losses['recon']:.4f} | "
              f"KL: {losses['kl']:.4f} | Rollout: {losses['rollout']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f'checkpoint_epoch_{epoch+1}.pt')

# ============================================================================
# Generation
# ============================================================================

def generate_long_sequence(model: ClimateVAE, config: Config, 
                          n_days: int, device: str = 'cuda') -> torch.Tensor:
    """Generate long sequence (1000+ years)"""
    model.eval()
    
    generated = []
    
    with torch.no_grad():
        # Initial latent
        z = torch.randn(1, config.LATENT_DIM, device=device)
        h = None
        
        for day in range(n_days):
            # Conditioning for this day
            doy = day % 365
            year = day // 365
            c = torch.FloatTensor([[
                np.sin(2 * np.pi * doy / 365),
                np.cos(2 * np.pi * doy / 365),
                year / 1000  # normalize
            ]]).to(device)
            
            # Transition
            mu_p, logvar_p, h = model.transition(z, c, h)
            z = model.reparameterize(mu_p, logvar_p)
            
            # Decode
            x_mean, _ = model.decoder(z, c)
            generated.append(x_mean.cpu())
            
            if (day + 1) % 365 == 0:
                print(f"Generated {(day + 1) // 365} years")
    
    return torch.cat(generated, dim=0)


if __name__ == '__main__':
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Train
    print("Starting training...")
    climate_data_path = r"C:\Users\mawr\OneDrive - Tonkin + Taylor Group Ltd\Documents\VAE-GAN for Hydrological Simulation\src\VAE-GAN-Hydrological-Simulation\data\climatedata.environment.govt.nz_daily_metadata.csv"
    train(config, climate_data_path, device)
    
    # Load best model and generate
    print("\nGenerating 1000 years...")
    model = ClimateVAE(config).to(device)
    checkpoint = torch.load('checkpoint_epoch_100.pt')
    model.load_state_dict(checkpoint['model'])
    
    generated = generate_long_sequence(model, config, config.GEN_DAYS, device)
    torch.save(generated, 'generated_1000yr.pt')
    print(f"Generated shape: {generated.shape}")