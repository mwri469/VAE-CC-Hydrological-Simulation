# Climate VAE: Controllable Weather Field Synthesis for Hydrological Simulation

A deep learning framework for generating synthetic climate data with controllable extreme weather scenarios using Variational Autoencoders (VAEs) with latent dynamics. This project enables the synthesis of long-term climate sequences for water resource planning and climate change impact assessment.

## Overview

This project implements a temporal VAE architecture that learns to generate realistic multi-day climate sequences while maintaining spatial coherence and temporal consistency. By mapping climate data to a structured latent space, the model enables controlled generation of extreme weather scenarios - a critical capability for climate change adaptation planning.

### Key Features

- **Temporal Dynamics**: GRU-based transition model captures day-to-day evolution of weather patterns
- **Multi-variable Support**: Simultaneously models precipitation, maximum temperature, and potential evapotranspiration
- **Spatial Coherence**: Convolutional architecture preserves spatial relationships in weather fields
- **Controllable Generation**: Sample from different regions of latent space to generate varying climate scenarios
- **Climate Scenario Support**: Train on historical data and future climate projections (SSP scenarios)

## Architecture

The model consists of three main components:

1. **Encoder**: Maps spatial weather fields to a latent distribution
   - Convolutional layers for spatial feature extraction
   - Conditioning on day-of-year and temporal information
   - Outputs mean and variance for latent distribution

2. **Decoder**: Reconstructs weather fields from latent representations
   - Transposed convolutional layers for spatial upsampling
   - Generates both mean and uncertainty estimates
   - Incorporates temporal conditioning

3. **Transition Model**: Models latent space dynamics over time
   - GRU-based recurrent architecture
   - Predicts next-day latent distribution from current state
   - Enables multi-day rollout generation

## Installation

### Requirements

```bash
python >= 3.8
torch >= 1.9.0
xarray
numpy
pandas
matplotlib
```

### Setup

```bash
git clone https://github.com/yourusername/VAE-CC-Hydrological-Simulation.git
cd VAE-CC-Hydrological-Simulation
pip install -r requirements.txt
```

## Data

The model expects climate data in NetCDF format with the following structure:

- **Variables**: `pr` (precipitation), `tasmax` (maximum temperature), `PETsrad` (potential evapotranspiration)
- **Dimensions**: `(time, latitude, longitude)`
- **Format**: Daily timesteps with spatial resolution of ~5km
- **Scenarios**: Historical (1980-2014) and future projections (SSP370: 2015-2099)

Data should be organized as:
```
data/
├── historical/
│   └── ACCESS-CM2/
│       ├── pr_historical_ACCESS-CM2_CCAM_daily_NZ5km_bc.nc
│       ├── tasmax_historical_ACCESS-CM2_CCAM_daily_NZ5km_bc.nc
│       └── PETsrad_historical_ACCESS-CM2_CCAM_daily_NZ5km_bc.nc
└── ssp370/
    └── ACCESS-CM2/
        ├── pr_ssp370_ACCESS-CM2_CCAM_daily_NZ5km_bc.nc
        ├── tasmax_ssp370_ACCESS-CM2_CCAM_daily_NZ5km_bc.nc
        └── PETsrad_ssp370_ACCESS-CM2_CCAM_daily_NZ5km_bc.nc
```

## Usage

### Training

Configure parameters in `main.py` and run:

```python
from main import main, Config

config = Config()
# Adjust parameters as needed
config.BBOX = {
    'lon_min': 174.25,
    'lon_max': 177,
    'lat_min': -39,
    'lat_max': -36
}
config.N_EPOCHS = 100
config.BATCH_SIZE = 16

main()
```

Training will automatically:
- Load and preprocess NetCDF data
- Create land masks for the region
- Train the VAE with beta-annealing
- Save checkpoints every 10 epochs

### Generation

Generate synthetic climate sequences:

```python
from models.Models import ClimateVAE
from postprocessing.postprocess import generate_sequences

# Load trained model
checkpoint = torch.load('checkpoint_historical_epoch_100.pt')
model = ClimateVAE(checkpoint['config'])
model.load_state_dict(checkpoint['model'])

# Generate 1000 years of daily data
synthetic_data = generate_sequences(
    model, 
    n_days=365000,
    std_multiplier=1.0  # Adjust for extreme scenarios
)
```

### Controlling Extremes

Sample from different regions of the latent distribution:

```python
# Normal scenarios (center of distribution)
normal_data = generate_sequences(model, std_multiplier=0.5)

# Extreme scenarios (tails of distribution)
extreme_data = generate_sequences(model, std_multiplier=1.5)
```

## Configuration

Key parameters in `Config` class:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LATENT_DIM` | 64 | Dimensionality of latent space |
| `HIDDEN_DIM` | 128 | Hidden dimension for GRU transition |
| `SEQ_LEN` | 64 | Training sequence length (days) |
| `BATCH_SIZE` | 16 | Batch size for training |
| `LEARNING_RATE` | 1e-4 | Adam learning rate |
| `BETA_WARMUP_EPOCHS` | 30 | KL annealing warmup period |
| `LAMBDA_ROLLOUT` | 0.1 | Weight for temporal consistency loss |
| `K_ROLLOUT` | 7 | Number of rollout steps |

## Model Outputs

The trained model generates:

- **Synthetic weather fields**: Multi-day sequences with spatial coherence
- **Uncertainty estimates**: Per-pixel variance for each variable
- **Temporal consistency**: Realistic day-to-day transitions
- **Statistical properties**: Distributions matching or exceeding training data extremes

## Applications

- **Water resource planning**: Generate long synthetic records for reservoir yield analysis
- **Climate change impact assessment**: Synthesize future climate scenarios
- **Drought/flood analysis**: Create extreme event catalogs for risk assessment
- **Infrastructure design**: Provide climate inputs for resilience planning

## References

This work builds on methodologies from:

- Oliveira et al. (2021) - "Controlling Weather Field Synthesis Using Variational Autoencoders"
- Kingma & Welling (2014) - "Auto-Encoding Variational Bayes"
- NIWA Climate Projections - Statistical downscaling framework

## Project Structure

```
├── src/
│   ├── models/
│   │   ├── Models.py           # VAE architecture
│   │   └── LossFunctions.py    # Training objectives
│   ├── preprocessing/
│   │   └── preprocess.py       # Data loading and normalization
│   ├── training/
│   │   └── train.py            # Training loop
│   ├── postprocessing/
│   │   └── postprocess.py      # Generation and analysis
│   └── main.py                 # Main entry point
├── data/                       # Climate datasets (not included)
├── model_weights/              # Saved checkpoints
├── outputs/                    # Generated visualizations
└── report/                     # Technical documentation
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NIWA for climate projection datasets
- Research inspired by IBM Research's work on climate VAEs
- Climate change impact modeling for Auckland water supply (Tonkin + Taylor / Watercare)

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{climate_vae_2024,
  title={Climate VAE: Controllable Weather Field Synthesis for Hydrological Simulation},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/yourusername/VAE-CC-Hydrological-Simulation}
}
```

## Contact

For questions or collaboration opportunities, please open an issue on GitHub or contact the maintainers.