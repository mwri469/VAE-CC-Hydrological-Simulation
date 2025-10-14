# Model Weights Directory

This directory contains trained VAE model checkpoints for climate data synthesis and extreme weather scenario generation.

## Overview

The Climate VAE model is designed to learn latent representations of climate variables (precipitation, temperature, potential evapotranspiration) and enable controllable synthesis of weather fields, including extreme weather scenarios. The model architecture consists of:

- **Encoder**: Convolutional network that maps spatial climate fields to latent distributions
- **Decoder**: Transposed convolutional network that reconstructs climate fields from latent codes
- **Transition Model**: GRU-based dynamics model for temporal coherence across sequences

## Checkpoint Schema

Each checkpoint file follows the naming convention:
```
checkpoint_{scenario}_epoch_{epoch_number}.pt
```

### Checkpoint Contents

Checkpoints are PyTorch state dictionaries containing:

```python
{
    'epoch': int,              # Training epoch number
    'scenario': str,           # Climate scenario ('historical' or 'ssp370')
    'model': OrderedDict,      # Model state dictionary
    'optimizer': dict,         # Optimizer state dictionary
    'config': Config,          # Configuration object with hyperparameters
}
```

### Configuration Parameters

The `config` object contains all model and training hyperparameters:

**Data Parameters:**
- `BBOX`: Bounding box coordinates (lon_min, lon_max, lat_min, lat_max)
- `VARIABLES`: List of climate variables ['tasmax', 'pr', 'PETsrad']
- `MODELS`: GCM models used ['ACCESS-CM2']
- `SSP`: Climate scenarios ['historical', 'ssp370']

**Model Parameters:**
- `LATENT_DIM`: Dimensionality of latent space (default: 64)
- `HIDDEN_DIM`: Hidden layer dimensions (default: 128)
- `SEQ_LEN`: Sequence length in days (default: 64)
- `SPATIAL_SIZE`: Spatial resolution of input fields

**Training Parameters:**
- `BATCH_SIZE`: Training batch size (default: 16)
- `LEARNING_RATE`: Learning rate (default: 1e-4)
- `BETA_START/END`: KL divergence annealing schedule
- `LAMBDA_ROLLOUT`: Weight for rollout consistency loss
- `K_ROLLOUT`: Number of rollout steps

## Loading Checkpoints

### Basic Loading

```python
import torch
from models.Models import ClimateVAE

# Load checkpoint
checkpoint = torch.load('checkpoint_historical_epoch_100.pt', 
                       map_location='cpu')

# Extract configuration
config = checkpoint['config']

# Initialize model with saved spatial dimensions
model = ClimateVAE(config, 
                   input_height=config.SPATIAL_SIZE,
                   input_width=config.SPATIAL_SIZE)

# Load trained weights
model.load_state_dict(checkpoint['model'])
model.eval()
```

### Resuming Training

```python
import torch
from models.Models import ClimateVAE

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load checkpoint
checkpoint = torch.load('checkpoint_historical_epoch_50.pt',
                       map_location=device)

# Initialize model and optimizer
config = checkpoint['config']
model = ClimateVAE(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), 
                             lr=config.LEARNING_RATE)

# Restore states
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
start_epoch = checkpoint['epoch'] + 1

# Continue training from start_epoch
```

## Generating Weather Fields

### Standard Generation

```python
import torch
import numpy as np

# Sample from standard normal distribution
z = torch.randn(1, config.LATENT_DIM)

# Create conditioning (e.g., summer day)
doy = 180  # Day 180 = approximately June 29
doy_sin = np.sin(2 * np.pi * doy / 365)
doy_cos = np.cos(2 * np.pi * doy / 365)
year_norm = 0.5  # Mid-range year
conditioning = torch.tensor([[doy_sin, doy_cos, year_norm]], 
                           dtype=torch.float32)

# Generate
with torch.no_grad():
    x_mean, x_logvar = model.decoder(z, conditioning)
    
# x_mean shape: (1, 3, H, W) for [tasmax, pr, PETsrad]
```

### Extreme Scenario Generation

Following the methodology from Oliveira et al. (2021), sample from distribution tails for extreme events:

```python
import torch

# For more extreme events, sample from tails (higher std deviation)
sigma = 1.5  # Increase for more extreme scenarios
z_extreme = torch.randn(1, config.LATENT_DIM) * sigma

# For less extreme events, sample from center (lower std deviation)
sigma = 0.5  # Decrease for average scenarios
z_average = torch.randn(1, config.LATENT_DIM) * sigma

# Generate with same conditioning as before
with torch.no_grad():
    x_extreme, _ = model.decoder(z_extreme, conditioning)
    x_average, _ = model.decoder(z_average, conditioning)
```

### Multi-Day Sequence Generation

```python
import torch

# Initialize
B = 1
T = 30  # Generate 30 days
z_sequence = []
h = None  # Hidden state for GRU

# Create conditioning sequence
doy_start = 150  # Start from day 150
conditioning_seq = []
for t in range(T):
    doy = (doy_start + t) % 365
    doy_sin = np.sin(2 * np.pi * doy / 365)
    doy_cos = np.cos(2 * np.pi * doy / 365)
    year_norm = 0.5
    conditioning_seq.append([doy_sin, doy_cos, year_norm])
conditioning_seq = torch.tensor(conditioning_seq, dtype=torch.float32)

# Generate sequence
with torch.no_grad():
    # Initial latent state
    z_t = torch.randn(B, config.LATENT_DIM)
    
    for t in range(T):
        c_t = conditioning_seq[t:t+1]
        
        # Decode current state
        x_t, _ = model.decoder(z_t, c_t)
        
        # Transition to next state
        mu_next, logvar_next, h = model.transition(z_t, c_t, h)
        z_t = model.reparameterize(mu_next, logvar_next)
        
        z_sequence.append(x_t)
    
    generated_sequence = torch.stack(z_sequence, dim=1)
    # Shape: (B, T, 3, H, W)
```

## Post-Processing Generated Data

### De-normalize and Transform Back

```python
import numpy as np

# Extract generated precipitation (variable index 1)
pr_normalized = x_mean[0, 1].numpy()  # (H, W)

# Reverse normalization (you'll need stored statistics)
pr_scaled = pr_normalized * std + mean

# Reverse log1p transform
pr_final = np.expm1(pr_scaled)  # mm/day

# Apply land mask (set ocean to NaN)
pr_final[~land_mask] = np.nan
```

**Note:** You'll need to save and load normalization statistics (mean, std) for each variable computed over land pixels during preprocessing.

## Model Specifications

### Input Dimensions
- Variables: 3 channels (tasmax, pr, PETsrad)
- Spatial: Variable based on bounding box (typically ~50x50 to 100x100 pixels at 5km resolution)
- Temporal: Sequences of 64 days

### Output Dimensions
- Same as input dimensions
- Includes both mean and log-variance for probabilistic predictions

### Latent Space
- 64-dimensional continuous latent space
- Regularized to approximate standard normal distribution
- Enables interpolation and controlled sampling

## Best Practices

1. **Extreme Weather Generation**: Use σ ∈ [1.0, 1.5] for sampling to generate more extreme scenarios
2. **Average Conditions**: Use σ ∈ [0.5, 0.85] for typical weather patterns
3. **Temporal Consistency**: Use the transition model for multi-day sequences rather than independent sampling
4. **Land Masking**: Always apply land mask before computing statistics or losses
5. **Validation**: Compare quantile-quantile plots of generated vs. historical data

## References

- Oliveira, D. A. B., et al. (2021). "Controlling Weather Field Synthesis Using Variational Autoencoders". ICML Workshop on Tackling Climate Change with Machine Learning.
- Kingma, D. P., & Welling, M. (2014). "Auto-Encoding Variational Bayes". ICLR.

## Support

For questions or issues with loading checkpoints, please refer to the main project README or create an issue in the repository.
