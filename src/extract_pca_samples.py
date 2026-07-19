"""
Extract representative latent space points and rainfall maps for the PCA web explorer.

Run from the src/ directory:
    python extract_pca_samples.py \
        --checkpoint "path/to/checkpoint.pt" \
        --data "path/to/historical_ACCESS-CM2_FilteredGrid.npy" \
        --mask "path/to/historical_ACCESS-CM2_LandMask.npy" \
        --output-dir "path/to/mwri469.github.io/assets/pca-samples" \
        --n-points 6 \
        --n-sample 500
"""

import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from models.Models import ClimateVAE, VQVAE2


# -- Config matching real main.py ---------------------------------------------

class Config:
    BBOX = {'lon_min': 174, 'lon_max': 177, 'lat_min': -39, 'lat_max': -36}
    VARIABLES = ['pr', 'tasmax', 'tasmin']
    MODELS = ['ACCESS-CM2', 'AWI-CM-1-1-MR', 'CNRM-CM6-1', 'EC-Earth3', 'GFDL-ESM4', 'NZESM', 'NorESM2-MM']
    SSP = ['historical']
    DATA_PATH = ""
    LATENT_DIM = 64
    HIDDEN_DIM = 128
    SEQ_LEN = 24
    SPATIAL_SIZE = None
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    N_EPOCHS = 10_000
    BETA_START = 0.0
    BETA_END = 1.0
    BETA_WARMUP_EPOCHS = 25
    LAMBDA_ROLLOUT = 0.1
    K_ROLLOUT = 7
    GRAD_CLIP = 1.0
    GEN_YEARS = 1000
    GEN_DAYS = 365 * 1000

    # VQ-VAE-2 parameters
    VQ_EMBED_DIM = 256
    VQ_HIDDEN_DIM = 512
    VQ_NUM_EMBEDDINGS = 2048
    VQ_COMMITMENT_COST = 0.25
    VQ_DECAY = 0.99
    VQ_N_EPOCHS = 500
    VQ_LEARNING_RATE = 3e-4


def load_config_from_checkpoint(ckpt):
    """Extract config from checkpoint, handling both object and dict formats."""
    cfg_raw = ckpt.get('config', None)
    if cfg_raw is None:
        print("  No config in checkpoint, using default Config")
        return Config()
    if isinstance(cfg_raw, dict):
        cfg = Config()
        for k, v in cfg_raw.items():
            setattr(cfg, k, v)
        return cfg
    # Config saved as object — use it directly
    return cfg_raw


# -- Conditioning (mirrors ClimateDataset._create_conditioning) ---------------

def make_conditioning(day_indices: np.ndarray) -> torch.Tensor:
    doy = day_indices % 365
    doy_sin = np.sin(2 * np.pi * doy / 365)
    doy_cos = np.cos(2 * np.pi * doy / 365)
    year = day_indices // 365
    year_norm = year / max(year.max(), 1)
    cond = np.stack([doy_sin, doy_cos, year_norm], axis=1)
    return torch.FloatTensor(cond)


# -- Encoding -----------------------------------------------------------------

def encode_days_vae(model, data, day_indices, device, batch_size=32):
    """Encode a set of day indices -> (N, latent_dim) mu array (ClimateVAE)."""
    model.eval()
    all_mu = []
    n = len(day_indices)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            idx = day_indices[start:start + batch_size]
            x_batch = torch.FloatTensor(data[idx]).to(device)   # (B, C, H, W)
            c_batch = make_conditioning(idx).to(device)          # (B, 3)
            mu, _ = model.encoder(x_batch, c_batch)
            all_mu.append(mu.cpu().numpy())
    return np.concatenate(all_mu, axis=0)


def encode_days_vqvae2(model, data, day_indices, device, batch_size=32):
    """Encode a set of day indices -> (N, embed_dim) array (VQVAE2).

    VQVAE2 has no single continuous latent vector per day, so we use the
    pre-quantization top-level encoding (enc_t), global-average-pooled over
    its spatial dimensions, as a continuous summary vector suitable for PCA.
    """
    model.eval()
    all_vec = []
    n = len(day_indices)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            idx = day_indices[start:start + batch_size]
            x_batch = torch.FloatTensor(data[idx]).to(device)   # (B, C, H, W)
            _, enc_t = model.encode(x_batch)                     # (B, D, h, w)
            vec = enc_t.mean(dim=[2, 3])                         # (B, D)
            all_vec.append(vec.cpu().numpy())
    return np.concatenate(all_vec, axis=0)


# -- Decoding -----------------------------------------------------------------

def decode_pr_vae(model, mu_np, day_index, device):
    """Decode a mu vector -> (H, W) precipitation mean field (channel 0)."""
    model.eval()
    with torch.no_grad():
        z = torch.FloatTensor(mu_np).unsqueeze(0).to(device)
        c = make_conditioning(np.array([day_index])).to(device)
        mean, _ = model.decoder(z, c)   # (1, C, H, W)
    return mean[0, 0].cpu().numpy()     # channel 0 = pr, shape (H, W)


def decode_pr_vqvae2(model, data, day_index, device):
    """Reconstruct the precipitation field (channel 0) for a specific day.

    VQVAE2 decodes from full spatial quantized code maps, not a single
    vector, so we run the actual day's field through the full
    encode -> quantize -> decode pipeline.
    """
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(data[day_index:day_index + 1]).to(device)  # (1, C, H, W)
        out = model(x)
    return out['recon'][0, 0].cpu().numpy()   # channel 0 = pr, shape (H, W)


# -- PNG export ---------------------------------------------------------------

def save_rainfall_png(field: np.ndarray, mask: np.ndarray, output_path: Path,
                      vmin: float = None, vmax: float = None):
    display = field.copy().astype(float)
    display[~mask.astype(bool)] = np.nan

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor('#F0EEE5')
    ax.set_facecolor('#F0EEE5')
    ax.imshow(display, cmap='Blues', vmin=vmin, vmax=vmax, origin='upper')
    ax.axis('off')
    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='#F0EEE5')
    plt.close(fig)


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True, help='*_FilteredGrid.npy (T, C, H, W) for one model')
    parser.add_argument('--mask', required=True, help='*_LandMask.npy (H, W) for same model')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--n-points', type=int, default=6)
    parser.add_argument('--n-sample', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Load data --------------------------------------------------------------
    print(f"Loading data from {args.data}")
    data = np.load(args.data)       # (T, C, H, W)
    mask = np.load(args.mask)       # (H, W)
    T, C, H, W = data.shape
    print(f"  data shape: {data.shape}  mask shape: {mask.shape}")

    # -- Load checkpoint --------------------------------------------------------
    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = load_config_from_checkpoint(ckpt)
    model_type = ckpt.get('model_type', 'vae')
    print(f"  Config: VARIABLES={getattr(config, 'VARIABLES', '?')}  LATENT_DIM={getattr(config, 'LATENT_DIM', 64)}  model_type={model_type}")

    # Pickled Config instances only store overridden attributes, so VARIABLES
    # can silently pick up the *current* main.Config class default instead of
    # the value used at training time. Trust the checkpoint's actual first-conv
    # weight shape for the true channel count, and slice the data to match.
    conv1_key = 'enc_bottom.0.weight' if model_type == 'vqvae2' else 'encoder.conv1.weight'
    n_channels_ckpt = ckpt['model'][conv1_key].shape[1]
    if n_channels_ckpt != C:
        print(f"  Checkpoint expects {n_channels_ckpt} channel(s) but data has {C}; "
              f"using the first {n_channels_ckpt} channel(s) of data ({config.VARIABLES[:n_channels_ckpt]}).")
        config.VARIABLES = config.VARIABLES[:n_channels_ckpt]
        data = data[:, :n_channels_ckpt]

    if model_type == 'vqvae2':
        model = VQVAE2(config, input_height=H, input_width=W).to(device)
    else:
        model = ClimateVAE(config, input_height=H, input_width=W).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print("  Model loaded")

    # -- Sample + encode --------------------------------------------------------
    n_sample = min(args.n_sample, T)
    sample_idx = np.sort(rng.choice(T, size=n_sample, replace=False))
    print(f"Encoding {n_sample} days...")
    if model_type == 'vqvae2':
        embed_all = encode_days_vqvae2(model, data, sample_idx, device)
    else:
        embed_all = encode_days_vae(model, data, sample_idx, device)
    print(f"  embedding shape: {embed_all.shape}")

    # -- PCA -------------------------------------------------------------------
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(embed_all)
    variance_explained = pca.explained_variance_ratio_.tolist()
    print(f"  Variance explained: PC1={variance_explained[0]:.3f}  PC2={variance_explained[1]:.3f}")

    cloud_min = coords_2d.min(axis=0)
    cloud_max = coords_2d.max(axis=0)
    def normalize(pts):
        return 2 * (pts - cloud_min) / (cloud_max - cloud_min) - 1

    coords_norm = normalize(coords_2d)

    # -- K-Means representative points -----------------------------------------
    print(f"Selecting {args.n_points} representative points via K-Means...")
    km = KMeans(n_clusters=args.n_points, random_state=args.seed, n_init='auto')
    km.fit(coords_2d)

    selected = []
    for center in km.cluster_centers_:
        nearest = int(np.argmin(np.linalg.norm(coords_2d - center, axis=1)))
        selected.append({
            'day_index': int(sample_idx[nearest]),
            'embed': embed_all[nearest],
            'pcaX': float(coords_2d[nearest, 0]),
            'pcaY': float(coords_2d[nearest, 1]),
            'pcaNormX': float(coords_norm[nearest, 0]),
            'pcaNormY': float(coords_norm[nearest, 1]),
        })

    selected.sort(key=lambda p: p['pcaNormX'])

    # -- Decode all fields first, compute shared colorscale, then save --------
    print("Decoding rainfall maps...")
    fields = []
    for pt in selected:
        if model_type == 'vqvae2':
            fields.append(decode_pr_vqvae2(model, data, pt['day_index'], device))
        else:
            fields.append(decode_pr_vae(model, pt['embed'], pt['day_index'], device))

    # Shared vmin/vmax across all decoded fields (land pixels only)
    all_land_vals = np.concatenate([
        f[mask.astype(bool)] for f in fields
    ])
    vmin = float(np.percentile(all_land_vals, 2))
    vmax = float(np.percentile(all_land_vals, 98))
    print(f"  Shared colorscale: vmin={vmin:.3f}  vmax={vmax:.3f}")

    print("Saving rainfall maps...")
    points_out = []
    for i, (pt, field) in enumerate(zip(selected, fields)):
        img_name = f"point-{i+1}.png"
        save_rainfall_png(field, mask, output_dir / img_name, vmin=vmin, vmax=vmax)
        print(f"  point-{i+1}.png  day={pt['day_index']}  norm=({pt['pcaNormX']:.2f}, {pt['pcaNormY']:.2f})")

        points_out.append({
            'id': i + 1,
            'pcaNormX': pt['pcaNormX'],
            'pcaNormY': pt['pcaNormY'],
            # Rename these by hand after inspecting the PNGs:
            'label': f"Cluster {i+1}",
            'imgSrc': f"../assets/pca-samples/{img_name}",
            'dayIndex': pt['day_index'],
        })

    # -- Background cloud (downsample to ~200 pts for JSON) -------------------
    step = max(1, len(coords_norm) // 200)
    cloud_out = [
        {'x': float(coords_norm[j, 0]), 'y': float(coords_norm[j, 1])}
        for j in range(0, len(coords_norm), step)
    ]

    manifest = {
        'pcaVarianceExplained': variance_explained,
        'cloud': cloud_out,
        'points': points_out,
    }
    json_path = output_dir / 'points.json'
    with open(json_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {json_path}")
    print("Done — open each PNG and rename the 'label' fields in points.json.")


if __name__ == '__main__':
    main()
