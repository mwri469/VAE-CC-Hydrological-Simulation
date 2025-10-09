from models.Models import *
from training.train import *
from preprocessing.preprocess import *
from postprocessing.postprocess import *

class Config:
    # Data parameters
    BBOX = {
        'lon_min': 174,
        'lon_max': 177,
        'lat_min': -39,
        'lat_max': -36
    }
    VARIABLES = ['pr', 'PETsrad']
    MODELS = ['ACCESS-CM2']
    SSP = ['historical', 'ssp370']
    # DATA_PATH = "C:/Users/.../VAE-GAN-Hydrological-Simulation/data"
    DATA_PATH = r"C:\Users\mawr\OneDrive - Tonkin + Taylor Group Ltd\Documents\VAE-GAN for Hydrological Simulation\src\VAE-CC-Hydrological-Simulation\data".replace("\\", "/")
    
    # Model parameters
    LATENT_DIM = 64
    HIDDEN_DIM = 128
    SEQ_LEN = 32  # days per training sequence
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

def main():
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
        
    for scenario in config.SSP:
        print(f"\n{'='*60}")
        print(f"Training on scenario: {scenario}")
        print(f"{'='*60}")
        
        # Load dataset and get actual spatial dimensions
        dataset = ClimateDataset(config, scenario, True)
        mask = dataset.mask
        
        # Update config with actual spatial dimensions
        config.SPATIAL_SIZE = max(dataset.height, dataset.width)
        print(f"Set SPATIAL_SIZE to: {config.SPATIAL_SIZE}")
        
        loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
        
        # Initialize model with correct dimensions
        model = ClimateVAE(config, input_height=dataset.height, input_width=dataset.width).to(device)
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
            try:
                losses = train_epoch(model, loader, optimizer, beta, config, device, mask)
                scheduler.step(losses['total'])
                
                print(f"Epoch {epoch+1}/{config.N_EPOCHS} | Beta: {beta:.3f} | "
                    f"Loss: {losses['total']:.4f} | Recon: {losses['recon']:.4f} | "
                    f"KL: {losses['kl']:.4f} | Rollout: {losses['rollout']:.4f}")
                
                # Save checkpoint
                if (epoch + 1) % 10 == 0:
                    checkpoint_path = f'../model_weights/checkpoint_{scenario}_epoch_{epoch+1}.pt'
                    torch.save({
                        'epoch': epoch,
                        'scenario': scenario,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                    }, checkpoint_path)
                    print(f"Saved checkpoint: {checkpoint_path}")
                    
            except RuntimeError as e:
                print(f"\nError during training: {e}")
                print(f"Dataset spatial dimensions: {dataset.height} x {dataset.width}")
                print(f"Config SPATIAL_SIZE: {config.SPATIAL_SIZE}")
                raise

if __name__ == '__main__':

    main()
