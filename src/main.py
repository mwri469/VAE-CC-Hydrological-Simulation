import os
# Reduce CUDA allocator fragmentation on small/constrained GPUs; must be set before torch initializes CUDA
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

from models.Models import *
from training.train import *
from preprocessing.preprocess import *
from postprocessing.postprocess import *
import argparse

class Config:
    # Data parameters
    BBOX = {
        'lon_min': 174.25,
        'lon_max': 177,
        'lat_min': -39,
        'lat_max': -36
    }
    VARIABLES = ['pr']
    MODELS = ['ACCESS-CM2', 'EC-Earth3', 'NorESM2-MM']
    SSP = ['historical']
    DATA_PATH = r"C:\Users\mawr\OneDrive - Tonkin + Taylor Group Ltd\Documents\00 VAE CC Hydrological Simulation\src\VAE-CC-Hydrological-Simulation\data".replace("\\", "/")
    
    # Model parameters
    LATENT_DIM = 64
    HIDDEN_DIM = 128
    SEQ_LEN = 12  # days per training sequence
    SPATIAL_SIZE = None  # Will be set from actual data dimensions
    
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

    # VQ-VAE-2 parameters
    VQ_EMBED_DIM = 64
    VQ_HIDDEN_DIM = 128
    VQ_NUM_EMBEDDINGS = 512
    VQ_COMMITMENT_COST = 0.25
    VQ_DECAY = 0.99
    VQ_N_EPOCHS = 500
    VQ_LEARNING_RATE = 3e-4

def load_checkpoint(checkpoint_path, device='cpu'):
    """
    Load a checkpoint and return model, optimizer, config, and start epoch
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: Device to load model to
    
    Returns:
        model: Loaded model
        optimizer: Loaded optimizer
        config: Configuration object
        start_epoch: Epoch to resume from
        scenario: Training scenario
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle config (could be object or dict)
    config_data = checkpoint['config']
    if isinstance(config_data, dict):
        # Config saved as dict - convert back to object
        config = Config()
        for key, value in config_data.items():
            setattr(config, key, value)
    else:
        # Config saved as object
        config = config_data
    
    scenario = checkpoint['scenario']
    start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
    
    print(f"Checkpoint info:")
    print(f"  - Scenario: {scenario}")
    print(f"  - Completed epoch: {checkpoint['epoch']}")
    print(f"  - Will resume from epoch: {start_epoch}")
    
    # Load dataset to get dimensions
    dataset = ClimateDataset(config, scenario, load_npy=True)
    
    # Initialize model with correct dimensions
    model = ClimateVAE(
        config,
        input_height=dataset.height,
        input_width=dataset.width
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model'])
    print("Model weights loaded successfully!")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Load optimizer state if available
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Optimizer state loaded successfully!")
    else:
        print("No optimizer state found, using fresh optimizer")
    
    return model, optimizer, config, start_epoch, scenario, dataset


def load_vq_checkpoint(checkpoint_path, device='cpu'):
    """Load a VQ-VAE-2 checkpoint"""
    print(f"Loading VQ-VAE-2 checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config_data = checkpoint['config']
    if isinstance(config_data, dict):
        config = Config()
        for key, value in config_data.items():
            setattr(config, key, value)
    else:
        config = config_data

    scenario = checkpoint['scenario']
    start_epoch = checkpoint['epoch'] + 1

    print(f"Checkpoint info:")
    print(f"  - Scenario: {scenario}")
    print(f"  - Completed epoch: {checkpoint['epoch']}")
    print(f"  - Will resume from epoch: {start_epoch}")

    dataset = ClimateDataset(config, scenario, load_npy=True)

    model = VQVAE2(
        config,
        input_height=dataset.height,
        input_width=dataset.width
    ).to(device)
    model.load_state_dict(checkpoint['model'])
    print("Model weights loaded successfully!")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.VQ_LEARNING_RATE
    )
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Optimizer state loaded successfully!")

    return model, optimizer, config, start_epoch, scenario, dataset


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Climate VAE')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from (e.g., checkpoint_historical_epoch_100.pt)')
    parser.add_argument('--scenario', type=str, default=None,
                       help='Scenario to train (overrides config if not resuming)')
    parser.add_argument('--model', type=str, default='vae', choices=['vae', 'vqvae2'],
                       help='Model architecture to use')
    args = parser.parse_args()
    
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
        
    # Determine scenarios to train
    if args.resume:
        if args.model == 'vqvae2':
            model, optimizer, config, start_epoch, scenario, dataset = load_vq_checkpoint(args.resume, device)
        else:
            model, optimizer, config, start_epoch, scenario, dataset = load_checkpoint(args.resume, device)
        scenarios_to_train = [scenario]
        print(f"\nResuming training from epoch {start_epoch}")
    else:
        scenarios_to_train = [args.scenario] if args.scenario else config.SSP
        start_epoch = 0
        model = None
        optimizer = None
        dataset = None

    if args.model == 'vqvae2':
        _train_vqvae2(config, device, scenarios_to_train, start_epoch, model, optimizer, dataset, args)
    else:
        _train_vae(config, device, scenarios_to_train, start_epoch, model, optimizer, dataset, args)


def _train_vae(config, device, scenarios_to_train, start_epoch, model, optimizer, dataset, args):
    for scenario in scenarios_to_train:
        print(f"\n{'='*60}")
        print(f"Training VAE on scenario: {scenario}")
        print(f"{'='*60}")

        if not args.resume or dataset is None:
            dataset = ClimateDataset(config, scenario, load_npy=True)
            config.SPATIAL_SIZE = max(dataset.height, dataset.width)
            model = ClimateVAE(config, input_height=dataset.height, input_width=dataset.width).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                        weight_decay=config.WEIGHT_DECAY)
            start_epoch = 0

        loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

        for epoch in range(start_epoch, config.N_EPOCHS):
            if epoch < config.BETA_WARMUP_EPOCHS:
                beta = config.BETA_START + (config.BETA_END - config.BETA_START) * \
                    epoch / config.BETA_WARMUP_EPOCHS
            else:
                beta = config.BETA_END

            try:
                losses = train_epoch(model, loader, optimizer, beta, config, device, dataset.mask, scaler)
                scheduler.step(losses['total'])

                print(f"Epoch {epoch+1}/{config.N_EPOCHS} | Beta: {beta:.3f} | "
                    f"Loss: {losses['total']:.4f} | Recon: {losses['recon']:.4f} | "
                    f"KL: {losses['kl']:.4f} | Rollout: {losses['rollout']:.4f}")

                if (epoch + 1) % 10 == 0:
                    checkpoint_path = f'checkpoint_{scenario}_epoch_{epoch+1}.pt'
                    torch.save({
                        'epoch': epoch,
                        'scenario': scenario,
                        'model_type': 'vae',
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                        'losses': losses,
                    }, checkpoint_path)
                    print(f"Saved checkpoint: {checkpoint_path}")

            except RuntimeError as e:
                print(f"\nError during training: {e}")
                print(f"Dataset spatial dimensions: {dataset.height} x {dataset.width}")
                print(f"Config SPATIAL_SIZE: {config.SPATIAL_SIZE}")
                raise

        if len(scenarios_to_train) > 1:
            model = None
            optimizer = None
            dataset = None
            start_epoch = 0
            if device == 'cuda':
                torch.cuda.empty_cache()


def _train_vqvae2(config, device, scenarios_to_train, start_epoch, model, optimizer, dataset, args):
    for scenario in scenarios_to_train:
        print(f"\n{'='*60}")
        print(f"Training VQ-VAE-2 on scenario: {scenario}")
        print(f"{'='*60}")

        if not args.resume or dataset is None:
            dataset = ClimateDataset(config, scenario, load_npy=True)
            config.SPATIAL_SIZE = max(dataset.height, dataset.width)
            model = VQVAE2(config, input_height=dataset.height, input_width=dataset.width).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.VQ_LEARNING_RATE)
            start_epoch = 0

        loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

        for epoch in range(start_epoch, config.VQ_N_EPOCHS):
            losses = train_epoch_vq(model, loader, optimizer, config, device, dataset.mask, scaler)
            scheduler.step(losses['total'])

            print(f"Epoch {epoch+1}/{config.VQ_N_EPOCHS} | "
                  f"Loss: {losses['total']:.4f} | Recon: {losses['recon']:.4f} | "
                  f"VQ: {losses['vq']:.4f} | Perp_T: {losses['perp_top']:.1f} | "
                  f"Perp_B: {losses['perp_bottom']:.1f}")

            # Dead code replacement every 50 epochs
            if (epoch + 1) % 50 == 0:
                x_sample, _ = next(iter(loader))
                x_sample = x_sample.to(device).reshape(-1, *x_sample.shape[2:])
                n_dead_t, n_dead_b = model.replace_dead_codes(x_sample)
                if n_dead_t + n_dead_b > 0:
                    print(f"  Replaced {n_dead_t} dead top codes, {n_dead_b} dead bottom codes")

            if (epoch + 1) % 10 == 0:
                checkpoint_path = f'checkpoint_vqvae2_{scenario}_epoch_{epoch+1}.pt'
                torch.save({
                    'epoch': epoch,
                    'scenario': scenario,
                    'model_type': 'vqvae2',
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'losses': losses,
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

        if len(scenarios_to_train) > 1:
            model = None
            optimizer = None
            dataset = None
            start_epoch = 0
            if device == 'cuda':
                torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
