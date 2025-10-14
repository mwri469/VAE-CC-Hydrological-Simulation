"""
Script to load VAE checkpoint and test generation quality
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from models.Models import ClimateVAE
from main import Config
from preprocessing.preprocess import ClimateDataset
import seaborn as sns

class GenerationTester:
    def __init__(self, checkpoint_path: str):
        """
        Args:
            checkpoint_path: Path to saved checkpoint (.pt file)
            climate_data_path: Path to climate metadata CSV
        """
        # Load checkpoint
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Handle config (could be object or dict)
        config_data = checkpoint['config']
        if isinstance(config_data, dict):
            # Config saved as dict (new format) - convert back to object
            from main import Config
            self.config = Config()
            for key, value in config_data.items():
                setattr(self.config, key, value)
        else:
            # Config saved as object (old format)
            self.config = config_data
        
        self.epoch = checkpoint['epoch']
        self.scenario = checkpoint['scenario']
        
        print(f"Loaded checkpoint from epoch {self.epoch}, scenario: {self.scenario}")
        
        # Setup device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load dataset to get normalization stats and mask
        self.dataset = ClimateDataset(self.config, self.scenario, load_npy=True)
        
        # Initialize model with correct dimensions
        self.model = ClimateVAE(
            self.config, 
            input_height=self.dataset.height, 
            input_width=self.dataset.width
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        print("Model loaded successfully!")
        
        # Store land mask for visualization (convert to boolean)
        self.land_mask = self.dataset.mask.squeeze().numpy().astype(bool)
        
    @torch.no_grad()
    def generate_unconditional(self, n_days: int = 365, temperature: float = 1.0):
        """
        Generate weather fields unconditionally from prior
        
        Args:
            n_days: Number of days to generate
            temperature: Sampling temperature (higher = more extreme)
        
        Returns:
            generated: (n_days, n_vars, H, W) numpy array
        """
        print(f"\nGenerating {n_days} days unconditionally (temperature={temperature})...")
        
        # Create conditioning (synthetic year)
        doy = np.arange(n_days) % 365
        doy_sin = np.sin(2 * np.pi * doy / 365)
        doy_cos = np.cos(2 * np.pi * doy / 365)
        year_norm = np.zeros(n_days)  # Year 0
        
        cond = np.stack([doy_sin, doy_cos, year_norm], axis=1)
        cond = torch.FloatTensor(cond).to(self.device)
        
        # Generate sequence
        generated = []
        h = None  # GRU hidden state
        z = torch.randn(1, self.config.LATENT_DIM, device=self.device) * temperature
        
        for t in range(n_days):
            c_t = cond[t:t+1]
            
            # Decode
            x_mean, x_logvar = self.model.decoder(z, c_t)
            
            # Sample from decoder distribution
            x_std = torch.exp(0.5 * x_logvar)
            x = x_mean + torch.randn_like(x_mean) * x_std * temperature
            
            generated.append(x.cpu().numpy())
            
            # Next latent state
            if t < n_days - 1:
                mu_next, logvar_next, h = self.model.transition(z, c_t, h)
                z = mu_next + torch.randn_like(mu_next) * torch.exp(0.5 * logvar_next) * temperature
        
        generated = np.concatenate(generated, axis=0)  # (n_days, n_vars, H, W)
        
        return generated
    
    @torch.no_grad()
    def generate_from_extremes(self, n_samples: int = 10, std_threshold: float = 2.0):
        """
        Generate samples by sampling from distribution tails (for extreme events)
        
        Args:
            n_samples: Number of samples to generate
            std_threshold: Standard deviations from mean (higher = more extreme)
        
        Returns:
            samples: (n_samples, n_vars, H, W) numpy array
        """
        print(f"\nGenerating {n_samples} extreme samples (threshold={std_threshold} std)...")
        
        samples = []
        
        for i in range(n_samples):
            # Sample from tail of distribution
            z = torch.randn(1, self.config.LATENT_DIM, device=self.device)
            z = torch.sign(z) * (torch.abs(z) * std_threshold)  # Push to tails
            
            # Random day in monsoon season (day 150-300)
            doy = np.random.randint(150, 300)
            doy_sin = np.sin(2 * np.pi * doy / 365)
            doy_cos = np.cos(2 * np.pi * doy / 365)
            c = torch.FloatTensor([[doy_sin, doy_cos, 0.5]]).to(self.device)
            
            # Decode
            x_mean, _ = self.model.decoder(z, c)
            samples.append(x_mean.cpu().numpy())
        
        samples = np.concatenate(samples, axis=0)
        
        return samples
    
    @torch.no_grad()
    def reconstruct_real_samples(self, n_samples: int = 5, start_idx: int = None):
        """
        Reconstruct real samples from dataset to check reconstruction quality
        
        Args:
            n_samples: Number of samples to reconstruct
            start_idx: Starting index in dataset (random if None)
        
        Returns:
            originals: (n_samples, n_vars, H, W)
            reconstructions: (n_samples, n_vars, H, W)
        """
        print(f"\nReconstructing {n_samples} real samples...")
        
        if start_idx is None:
            start_idx = np.random.randint(0, len(self.dataset) - n_samples)
        
        originals = []
        reconstructions = []
        
        for i in range(n_samples):
            # Get real sample
            x_seq, c_seq = self.dataset[start_idx + i]
            x = x_seq[0:1].to(self.device)  # Take first day of sequence
            c = c_seq[0:1].to(self.device)
            
            # Encode
            mu, logvar = self.model.encoder(x, c)
            z = self.model.reparameterize(mu, logvar)
            
            # Decode
            x_recon, _ = self.model.decoder(z, c)
            
            originals.append(x.cpu().numpy())
            reconstructions.append(x_recon.cpu().numpy())
        
        originals = np.concatenate(originals, axis=0)
        reconstructions = np.concatenate(reconstructions, axis=0)
        
        return originals, reconstructions
    
    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """
        Denormalize data back to original units
        Note: This is approximate since we normalized per-variable
        
        Args:
            data: (n_samples, n_vars, H, W) normalized data
        
        Returns:
            denorm: Same shape, in original units
        """
        # For precipitation, reverse log1p transform
        # For other variables, data is already in normalized form
        denorm = data.copy()
        
        # Precipitation is first variable (index 0) and was log1p transformed
        if 'pr' in self.config.VARIABLES:
            pr_idx = self.config.VARIABLES.index('pr')
            # Reverse normalization (approximate - we don't store original stats)
            # This is just for visualization
            denorm[:, pr_idx] = np.expm1(denorm[:, pr_idx] * 2)  # Rough rescaling
        
        return denorm
    
    def visualize_samples(self, samples: np.ndarray, title: str = "Generated Samples", 
                         n_display: int = 5, save_path: str = None, fixed_scale: bool = True):
        """
        Visualize generated weather fields with optional shared scales
        
        Args:
            samples: (n_samples, n_vars, H, W) array
            title: Plot title
            n_display: Number of samples to display
            save_path: Path to save figure (if provided)
            fixed_scale: If True, use same scale for all samples of each variable
        """
        n_display = min(n_display, len(samples))
        n_vars = len(self.config.VARIABLES)
        
        fig, axes = plt.subplots(n_display, n_vars, figsize=(4*n_vars, 4*n_display))
        if n_display == 1:
            axes = axes.reshape(1, -1)
        
        # Calculate shared scales per variable if requested
        vmin_max = {}
        if fixed_scale:
            for j, var_name in enumerate(self.config.VARIABLES):
                all_data = []
                for i in range(n_display):
                    data = samples[i, j].copy()
                    data[~self.land_mask] = np.nan
                    valid = data[~np.isnan(data)]
                    all_data.extend(valid.flatten())
                vmin_max[j] = (np.min(all_data), np.max(all_data))
        
        for i in range(n_display):
            for j, var_name in enumerate(self.config.VARIABLES):
                ax = axes[i, j]
                
                # Get data and mask ocean
                data = samples[i, j].copy()
                data[~self.land_mask] = np.nan
                
                # Set scale
                if fixed_scale:
                    vmin, vmax = vmin_max[j]
                    im = ax.imshow(data, cmap='viridis', interpolation='nearest',
                                  vmin=vmin, vmax=vmax)
                else:
                    im = ax.imshow(data, cmap='viridis', interpolation='nearest')
                
                ax.set_title(f"{var_name} - Sample {i+1}")
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)
        
        if fixed_scale:
            title += " (Shared Scales)"
        fig.suptitle(title, fontsize=16, y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved figure to: {save_path}")
        
        plt.show()
    
    def compare_reconstruction_with_difference(self, originals: np.ndarray, reconstructions: np.ndarray,
                                              n_display: int = 3, save_path: str = None):
        """
        Compare original, reconstructed, and difference maps side-by-side
        """
        n_display = min(n_display, len(originals))
        n_vars = len(self.config.VARIABLES)
        
        fig, axes = plt.subplots(n_display, n_vars*3, figsize=(5*n_vars*3, 4*n_display))
        if n_display == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_display):
            for j, var_name in enumerate(self.config.VARIABLES):
                # Prepare data with masking
                data_orig = originals[i, j].copy()
                data_recon = reconstructions[i, j].copy()
                data_orig[~self.land_mask] = np.nan
                data_recon[~self.land_mask] = np.nan
                
                # Calculate difference
                data_diff = data_recon - data_orig
                
                # Shared vmin/vmax for orig and recon
                valid_orig = data_orig[~np.isnan(data_orig)]
                valid_recon = data_recon[~np.isnan(data_recon)]
                vmin = min(valid_orig.min(), valid_recon.min())
                vmax = max(valid_orig.max(), valid_recon.max())
                
                # Difference scale (symmetric around 0)
                valid_diff = data_diff[~np.isnan(data_diff)]
                diff_abs_max = max(abs(valid_diff.min()), abs(valid_diff.max()))
                
                # Original
                ax_orig = axes[i, j*3]
                im = ax_orig.imshow(data_orig, cmap='viridis', interpolation='nearest',
                                   vmin=vmin, vmax=vmax)
                ax_orig.set_title(f"{var_name} - Original {i+1}")
                ax_orig.axis('off')
                plt.colorbar(im, ax=ax_orig, fraction=0.046)
                
                # Reconstruction
                ax_recon = axes[i, j*3+1]
                im = ax_recon.imshow(data_recon, cmap='viridis', interpolation='nearest',
                                    vmin=vmin, vmax=vmax)
                ax_recon.set_title(f"{var_name} - Reconstructed {i+1}")
                ax_recon.axis('off')
                plt.colorbar(im, ax=ax_recon, fraction=0.046)
                
                # Difference (Recon - Original)
                ax_diff = axes[i, j*3+2]
                im = ax_diff.imshow(data_diff, cmap='RdBu_r', interpolation='nearest',
                                   vmin=-diff_abs_max, vmax=diff_abs_max)
                ax_diff.set_title(f"{var_name} - Difference {i+1}")
                ax_diff.axis('off')
                plt.colorbar(im, ax=ax_diff, fraction=0.046)
        
        fig.suptitle("Original | Reconstructed | Difference (Shared Scales)", fontsize=16, y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison to: {save_path}")
        
        plt.show()
    
    def compare_reconstruction(self, originals: np.ndarray, reconstructions: np.ndarray,
                              n_display: int = 3, save_path: str = None):
        """
        Compare original and reconstructed samples side-by-side with shared color scales
        """
        n_display = min(n_display, len(originals))
        n_vars = len(self.config.VARIABLES)
        
        fig, axes = plt.subplots(n_display, n_vars*2, figsize=(4*n_vars*2, 4*n_display))
        if n_display == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_display):
            for j, var_name in enumerate(self.config.VARIABLES):
                # Prepare data with masking
                data_orig = originals[i, j].copy()
                data_recon = reconstructions[i, j].copy()
                data_orig[~self.land_mask] = np.nan
                data_recon[~self.land_mask] = np.nan
                
                # Calculate shared vmin/vmax for this variable across both plots
                valid_orig = data_orig[~np.isnan(data_orig)]
                valid_recon = data_recon[~np.isnan(data_recon)]
                vmin = min(valid_orig.min(), valid_recon.min())
                vmax = max(valid_orig.max(), valid_recon.max())
                
                # Original
                ax_orig = axes[i, j*2]
                im = ax_orig.imshow(data_orig, cmap='viridis', interpolation='nearest',
                                   vmin=vmin, vmax=vmax)
                ax_orig.set_title(f"{var_name} - Original {i+1}")
                ax_orig.axis('off')
                plt.colorbar(im, ax=ax_orig, fraction=0.046)
                
                # Reconstruction
                ax_recon = axes[i, j*2+1]
                im = ax_recon.imshow(data_recon, cmap='viridis', interpolation='nearest',
                                    vmin=vmin, vmax=vmax)
                ax_recon.set_title(f"{var_name} - Reconstructed {i+1}")
                ax_recon.axis('off')
                plt.colorbar(im, ax=ax_recon, fraction=0.046)
        
        fig.suptitle("Original vs Reconstructed (Shared Scales)", fontsize=16, y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison to: {save_path}")
        
        plt.show()
    
    def plot_temporal_sequence(self, sequence: np.ndarray, var_idx: int = 0, 
                              sample_every: int = 30, save_path: str = None):
        """
        Plot temporal evolution of a generated sequence with shared color scale
        
        Args:
            sequence: (n_days, n_vars, H, W) generated sequence
            var_idx: Which variable to plot
            sample_every: Show every Nth day
            save_path: Path to save figure
        """
        var_name = self.config.VARIABLES[var_idx]
        days_to_plot = sequence[::sample_every]
        n_days = len(days_to_plot)
        
        # Calculate global min/max across all time steps for consistent scale
        all_data = []
        for i in range(n_days):
            data = days_to_plot[i, var_idx].copy()
            data[~self.land_mask] = np.nan
            all_data.append(data)
        
        # Get valid (non-NaN) values for vmin/vmax
        valid_values = np.concatenate([d[~np.isnan(d)].flatten() for d in all_data])
        vmin = valid_values.min()
        vmax = valid_values.max()
        
        fig, axes = plt.subplots(1, n_days, figsize=(4*n_days, 4))
        if n_days == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            im = ax.imshow(all_data[i], cmap='viridis', interpolation='nearest',
                          vmin=vmin, vmax=vmax)
            ax.set_title(f"Day {i*sample_every}")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        fig.suptitle(f"Temporal Evolution - {var_name} (Shared Scale: {vmin:.2f} to {vmax:.2f})", 
                    fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved temporal plot to: {save_path}")
        
        plt.show()
    
    def analyze_statistics(self, generated: np.ndarray, var_idx: int = 1):
        """
        Compare statistics of generated data vs real data
        
        Args:
            generated: (n_samples, n_vars, H, W) generated data
            var_idx: Which variable to analyze (default: 1 = precipitation)
        """
        var_name = self.config.VARIABLES[var_idx]
        print(f"\n{'='*60}")
        print(f"Statistical Analysis - {var_name}")
        print(f"{'='*60}")
        
        # Get real data statistics (over land pixels only)
        real_data = self.dataset.data[:, var_idx].numpy()
        real_land = real_data[:, self.land_mask]
        
        # Get generated statistics (over land pixels only)
        gen_land = generated[:, var_idx][:, self.land_mask]
        
        # Compute statistics
        print(f"\n{'Statistic':<20} {'Real Data':<15} {'Generated':<15} {'Difference'}")
        print(f"{'-'*60}")
        
        stats = {
            'Mean': (np.mean(real_land), np.mean(gen_land)),
            'Std': (np.std(real_land), np.std(gen_land)),
            'Min': (np.min(real_land), np.min(gen_land)),
            'Max': (np.max(real_land), np.max(gen_land)),
            'Median': (np.median(real_land), np.median(gen_land)),
            'P5': (np.percentile(real_land, 5), np.percentile(gen_land, 5)),
            'P95': (np.percentile(real_land, 95), np.percentile(gen_land, 95)),
        }
        
        for stat_name, (real_val, gen_val) in stats.items():
            diff = gen_val - real_val
            print(f"{stat_name:<20} {real_val:<15.4f} {gen_val:<15.4f} {diff:+.4f}")
        
        # Plot distributions
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram
        axes[0].hist(real_land.flatten(), bins=50, alpha=0.5, label='Real', density=True)
        axes[0].hist(gen_land.flatten(), bins=50, alpha=0.5, label='Generated', density=True)
        axes[0].set_xlabel(var_name)
        axes[0].set_ylabel('Density')
        axes[0].set_title('Distribution Comparison')
        axes[0].legend()
        
        # Q-Q plot
        real_sorted = np.sort(real_land.flatten())
        gen_sorted = np.sort(gen_land.flatten())
        # Sample to same size
        n_points = min(len(real_sorted), len(gen_sorted), 10000)
        real_sample = real_sorted[np.linspace(0, len(real_sorted)-1, n_points).astype(int)]
        gen_sample = gen_sorted[np.linspace(0, len(gen_sorted)-1, n_points).astype(int)]
        
        axes[1].scatter(real_sample, gen_sample, alpha=0.3, s=1)
        axes[1].plot([real_sample.min(), real_sample.max()], 
                     [real_sample.min(), real_sample.max()], 
                     'r--', label='Perfect match')
        axes[1].set_xlabel(f'Real {var_name}')
        axes[1].set_ylabel(f'Generated {var_name}')
        axes[1].set_title('Q-Q Plot')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()


# Example usage
def main():
    # Configuration
    checkpoint_path = "../model_weights/pt_files/checkpoint_historical_epoch_300.pt"
    
    # Initialize tester
    tester = GenerationTester(checkpoint_path)
    
    # Test 1: Reconstruction quality
    print("\n" + "="*60)
    print("TEST 1: Reconstruction Quality")
    print("="*60)
    originals, reconstructions = tester.reconstruct_real_samples(n_samples=3)
    
    # Show both comparison types
    tester.compare_reconstruction(originals, reconstructions, 
                                  save_path="reconstruction_comparison.png")
    tester.compare_reconstruction_with_difference(originals, reconstructions,
                                                 save_path="reconstruction_with_difference.png")
    
    # Test 2: Unconditional generation
    print("\n" + "="*60)
    print("TEST 2: Unconditional Generation")
    print("="*60)
    generated_normal = tester.generate_unconditional(n_days=365, temperature=1.0)
    tester.visualize_samples(generated_normal[:5], title="Normal Generation (T=1.0)", 
                            save_path="generated_normal.png")
    
    # Test 3: Extreme event generation
    print("\n" + "="*60)
    print("TEST 3: Extreme Event Generation")
    print("="*60)
    generated_extreme = tester.generate_from_extremes(n_samples=10, std_threshold=2.5)
    tester.visualize_samples(generated_extreme[:5], title="Extreme Events (2.5σ)", 
                            save_path="generated_extreme.png")
    
    # Test 4: Temporal coherence
    print("\n" + "="*60)
    print("TEST 4: Temporal Coherence")
    print("="*60)
    sequence = tester.generate_unconditional(n_days=365, temperature=1.0)
    tester.plot_temporal_sequence(sequence, var_idx=1, sample_every=30, 
                                  save_path="temporal_sequence.png")
    
    # Test 5: Statistical comparison
    print("\n" + "="*60)
    print("TEST 5: Statistical Analysis")
    print("="*60)
    tester.analyze_statistics(generated_normal, var_idx=1)
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)


if __name__ == "__main__":
    main()