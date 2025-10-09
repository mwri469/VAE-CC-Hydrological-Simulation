import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List

class Encoder(nn.Module):
    """Convolutional encoder: spatial field -> latent distribution"""
    
    def __init__(self, config, input_height, input_width):
        super().__init__()
        self.config = config
        
        # Conditioning embedding
        self.cond_embed = nn.Linear(3, 32)
        
        # Conv layers
        self.conv1 = nn.Conv2d(len(config.VARIABLES), 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # down by 2
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)  # down by 2
        
        # Calculate actual flattened size by doing a test forward pass
        with torch.no_grad():
            test_input = torch.zeros(1, len(config.VARIABLES), input_height, input_width)
            test_output = self._conv_forward(test_input)
            flat_size = test_output.flatten(1).shape[1]
        
        print(f"Encoder: input={input_height}x{input_width}, "
              f"conv output shape={test_output.shape[2]}x{test_output.shape[3]}, "
              f"flat_size={flat_size}")
        
        # Bottleneck
        self.fc = nn.Linear(flat_size + 32, config.HIDDEN_DIM)
        
        # Latent parameters
        self.fc_mu = nn.Linear(config.HIDDEN_DIM, config.LATENT_DIM)
        self.fc_logvar = nn.Linear(config.HIDDEN_DIM, config.LATENT_DIM)
    
    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Helper to run just the convolutional part"""
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        return h
    
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
        h = self._conv_forward(x)
        
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
    
    def __init__(self, config, output_height, output_width):
        super().__init__()
        self.config = config
        self.output_height = output_height
        self.output_width = output_width
        
        # Conditioning embedding
        self.cond_embed = nn.Linear(3, 32)
        
        # Initial projection
        self.fc = nn.Linear(config.LATENT_DIM + 32, config.HIDDEN_DIM)
        
        # We need to figure out what initial size will give us the right output
        # Work backwards from output through the transposed convs
        # deconv3: stride=2, so input should be roughly output_h//2, output_w//2
        # deconv1: stride=2, so we need to go back another factor of 2
        self.init_h = output_height // 4
        self.init_w = output_width // 4
        self.fc_reshape = nn.Linear(config.HIDDEN_DIM, 128 * self.init_h * self.init_w)
        
        print(f"Decoder: output={output_height}x{output_width}, "
              f"initial={self.init_h}x{self.init_w}")
        
        # Transposed conv layers (will upsample by 4x total)
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
        h = h.view(B, 128, self.init_h, self.init_w)
        
        # Deconvolve
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = F.relu(self.deconv3(h))
        
        # Crop/pad to exact output size if needed (handles rounding issues)
        if h.shape[2] != self.output_height or h.shape[3] != self.output_width:
            h = F.interpolate(h, size=(self.output_height, self.output_width), 
                            mode='bilinear', align_corners=False)
        
        # Output
        mean = self.out_mean(h)
        logvar = self.out_logvar(h)
        logvar = torch.clamp(logvar, -10, 2)
        
        return mean, logvar


class TransitionModel(nn.Module):
    """Latent dynamics: p(z_t | z_{t-1}, c_t)"""
    
    def __init__(self, config):
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
    
    def __init__(self, config, input_height=None, input_width=None):
        super().__init__()
        self.config = config
        
        # If dimensions not provided, use SPATIAL_SIZE (assume square)
        if input_height is None:
            input_height = config.SPATIAL_SIZE
        if input_width is None:
            input_width = config.SPATIAL_SIZE
            
        self.input_height = input_height
        self.input_width = input_width
        
        self.encoder = Encoder(config, input_height, input_width)
        self.decoder = Decoder(config, input_height, input_width)
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