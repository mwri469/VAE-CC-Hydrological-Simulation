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


# ── VQ-VAE-2 Components ────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_embed_sum', self.embedding.weight.data.clone())

    def forward(self, z: torch.Tensor):
        # z: (B, D, H, W)
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)

        # Nearest-code lookup is non-differentiable; keep it (and the EMA
        # bookkeeping below) out of the autograd graph so no reference to the
        # encoder's activations is retained past this call.
        with torch.no_grad():
            z_flat_detached = z_flat.detach()
            dist = (z_flat_detached.pow(2).sum(1, keepdim=True)
                    + self.embedding.weight.pow(2).sum(1)
                    - 2 * z_flat_detached @ self.embedding.weight.t())

            indices = dist.argmin(dim=1)

            if self.training:
                encodings = F.one_hot(indices, self.num_embeddings).float()
                self.ema_cluster_size.mul_(self.decay).add_(
                    encodings.sum(0), alpha=1 - self.decay)
                embed_sum = encodings.t() @ z_flat_detached
                self.ema_embed_sum.mul_(self.decay).add_(
                    embed_sum, alpha=1 - self.decay)

                n = self.ema_cluster_size.sum()
                cluster_size = ((self.ema_cluster_size + 1e-5)
                                / (n + self.num_embeddings * 1e-5) * n)
                self.embedding.weight.data.copy_(
                    self.ema_embed_sum / cluster_size.unsqueeze(1))

        quantized = self.embedding(indices).view(B, H, W, D).permute(0, 3, 1, 2)

        commitment_loss = self.commitment_cost * F.mse_loss(
            z_flat, quantized.permute(0, 2, 3, 1).reshape(-1, D).detach())

        # Straight-through estimator
        quantized_st = z + (quantized - z).detach()

        avg_probs = F.one_hot(indices, self.num_embeddings).float().mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized_st, commitment_loss, perplexity, indices.view(B, H, W)


    def replace_dead_codes(self, z_flat: torch.Tensor) -> int:
        dead = self.ema_cluster_size < 1.0
        n_dead = dead.sum().item()
        if n_dead > 0 and z_flat.shape[0] > 0:
            replace_idx = torch.randint(0, z_flat.shape[0], (int(n_dead),),
                                        device=z_flat.device)
            self.embedding.weight.data[dead] = z_flat[replace_idx].detach()
            self.ema_embed_sum[dead] = z_flat[replace_idx].detach()
            self.ema_cluster_size[dead] = 1.0
        return int(n_dead)


class VQVAE2(nn.Module):
    def __init__(self, config, input_height, input_width):
        super().__init__()
        self.config = config
        self.input_height = input_height
        self.input_width = input_width

        C = len(config.VARIABLES)
        D = config.VQ_EMBED_DIM
        H = config.VQ_HIDDEN_DIM
        K = config.VQ_NUM_EMBEDDINGS
        beta = config.VQ_COMMITMENT_COST
        decay = config.VQ_DECAY

        # Bottom encoder: input → /4 features
        self.enc_bottom = nn.Sequential(
            nn.Conv2d(C, H // 2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(H // 2, H, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(H, H, 3, padding=1),
            ResBlock(H),
            ResBlock(H),
            nn.Conv2d(H, D, 1),
        )

        # Top encoder: /4 features → /8 features
        self.enc_top = nn.Sequential(
            nn.Conv2d(D, H, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(H, H, 3, padding=1),
            ResBlock(H),
            ResBlock(H),
            nn.Conv2d(H, D, 1),
        )

        self.vq_top = VectorQuantizerEMA(K, D, beta, decay)
        self.vq_bottom = VectorQuantizerEMA(K, D, beta, decay)

        # Upsample quantized top → bottom resolution
        self.dec_top = nn.Sequential(
            nn.Conv2d(D, H, 3, padding=1),
            ResBlock(H),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(H, D, 3, padding=1),
        )

        # Combine bottom encoder output with decoded top for bottom quantization
        self.enc_bottom_combine = nn.Conv2d(D * 2, D, 1)

        # Full decoder: bottom + top → output
        self.decoder = nn.Sequential(
            nn.Conv2d(D * 2, H, 3, padding=1),
            ResBlock(H),
            ResBlock(H),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(H, H, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(H, C, 3, padding=1),
        )

        # Compute internal spatial dims
        with torch.no_grad():
            test = torch.zeros(1, C, input_height, input_width)
            eb = self.enc_bottom(test)
            et = self.enc_top(eb)
            self.bottom_h, self.bottom_w = eb.shape[2], eb.shape[3]
            self.top_h, self.top_w = et.shape[2], et.shape[3]
            print(f"VQVAE2: input={input_height}x{input_width}, "
                  f"bottom={self.bottom_h}x{self.bottom_w}, "
                  f"top={self.top_h}x{self.top_w}")

    def encode(self, x: torch.Tensor):
        enc_b = self.enc_bottom(x)
        enc_t = self.enc_top(enc_b)
        return enc_b, enc_t

    def decode(self, quant_b: torch.Tensor, quant_t: torch.Tensor):
        dec_t = self.dec_top(quant_t)
        if dec_t.shape[2:] != quant_b.shape[2:]:
            dec_t = F.interpolate(dec_t, size=quant_b.shape[2:], mode='nearest')

        x_recon = self.decoder(torch.cat([quant_b, dec_t], dim=1))
        if x_recon.shape[2:] != (self.input_height, self.input_width):
            x_recon = F.interpolate(x_recon, size=(self.input_height, self.input_width),
                                    mode='bilinear', align_corners=False)
        return x_recon

    def decode_from_ids(self, ids_b: torch.Tensor, ids_t: torch.Tensor):
        quant_t = self.vq_top.embedding(ids_t).permute(0, 3, 1, 2)
        quant_b = self.vq_bottom.embedding(ids_b).permute(0, 3, 1, 2)
        return self.decode(quant_b, quant_t)

    def replace_dead_codes(self, x: torch.Tensor):
        with torch.no_grad():
            enc_b, enc_t = self.encode(x)
            D = self.config.VQ_EMBED_DIM

            n_dead_t = self.vq_top.replace_dead_codes(
                enc_t.permute(0, 2, 3, 1).reshape(-1, D))

            quant_t, _, _, _ = self.vq_top(enc_t)
            dec_t = self.dec_top(quant_t)
            if dec_t.shape[2:] != enc_b.shape[2:]:
                dec_t = F.interpolate(dec_t, size=enc_b.shape[2:], mode='nearest')
            enc_b_combined = self.enc_bottom_combine(torch.cat([enc_b, dec_t], dim=1))

            n_dead_b = self.vq_bottom.replace_dead_codes(
                enc_b_combined.permute(0, 2, 3, 1).reshape(-1, D))

        return n_dead_t, n_dead_b

    def forward(self, x: torch.Tensor) -> Dict:
        enc_b, enc_t = self.encode(x)

        # Top level quantization
        quant_t, loss_t, perp_t, ids_t = self.vq_top(enc_t)

        # Upsample top codes to bottom resolution
        dec_t = self.dec_top(quant_t)
        if dec_t.shape[2:] != enc_b.shape[2:]:
            dec_t = F.interpolate(dec_t, size=enc_b.shape[2:], mode='nearest')

        # Bottom level quantization (conditioned on top)
        enc_b_combined = self.enc_bottom_combine(torch.cat([enc_b, dec_t], dim=1))
        quant_b, loss_b, perp_b, ids_b = self.vq_bottom(enc_b_combined)

        # Full decode
        x_recon = self.decoder(torch.cat([quant_b, dec_t], dim=1))
        if x_recon.shape[2:] != x.shape[2:]:
            x_recon = F.interpolate(x_recon, size=x.shape[2:],
                                    mode='bilinear', align_corners=False)

        return {
            'recon': x_recon,
            'vq_loss': loss_t + loss_b,
            'perplexity_top': perp_t,
            'perplexity_bottom': perp_b,
            'ids_top': ids_t,
            'ids_bottom': ids_b,
        }