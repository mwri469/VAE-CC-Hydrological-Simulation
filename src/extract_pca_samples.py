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

from models.Models import ClimateVAE


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

def encode_days(model, data, day_indices, device, batch_size=32):
    """Encode a set of day indices -> (N, latent_dim) mu array."""
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


# -- Decoding -----------------------------------------------------------------

def decode_pr(model, mu_np, day_index, device):
    """Decode a mu vector -> (H, W) precipitation mean field (channel 0)."""
    model.eval()
    with torch.no_grad():
        z = torch.FloatTensor(mu_np).unsqueeze(0).to(device)
        c = make_conditioning(np.array([day_index])).to(device)
        mean, _ = model.decoder(z, c)   # (1, C, H, W)
    return mean[0, 0].cpu().numpy()     # channel 0 = pr, shape (H, W)


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
    print(f"  Config: VARIABLES={getattr(config, 'VARIABLES', '?')}  LATENT_DIM={getattr(config, 'LATENT_DIM', 64)}")

    model = ClimateVAE(config, input_height=H, input_width=W).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print("  Model loaded")

    # -- Sample + encode --------------------------------------------------------
    n_sample = min(args.n_sample, T)
    sample_idx = np.sort(rng.choice(T, size=n_sample, replace=False))
    print(f"Encoding {n_sample} days...")
    mu_all = encode_days(model, data, sample_idx, device)
    print(f"  mu shape: {mu_all.shape}")

    # -- PCA -------------------------------------------------------------------
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(mu_all)
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
            'mu': mu_all[nearest],
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
        fields.append(decode_pr(model, pt['mu'], pt['day_index'], device))

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


import argparse
import json
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from models.Models import ClimateVAE


# -- Config reconstruction ----------------------------------------------------

class Config:
    VARIABLES = ['pr']
    MODELS = ['ACCESS-CM2']
    SSP = ['historical']
    LATENT_DIM = 64
    HIDDEN_DIM = 128
    SEQ_LEN = 12
    SPATIAL_SIZE = None
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
    GEN_YEARS = 1000
    GEN_DAYS = 365 * 1000


def config_from_checkpoint(ckpt_config):
    cfg = Config()
    if isinstance(ckpt_config, dict):
        for k, v in ckpt_config.items():
            setattr(cfg, k, v)
    else:
        cfg = ckpt_config
    return cfg


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

def encode_days(model, data, day_indices, device, batch_size=32):
    """Encode a set of day indices -> (N, latent_dim) mu array."""
    model.eval()
    all_mu = []
    n = len(day_indices)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            idx = day_indices[start:start + batch_size]
            x_batch = data[idx]                          # (B, C, H, W)
            c_batch = make_conditioning(idx)             # (B, 3)
            x_batch = torch.FloatTensor(x_batch).to(device)
            c_batch = c_batch.to(device)
            mu, _ = model.encoder(x_batch, c_batch)
            all_mu.append(mu.cpu().numpy())

    return np.concatenate(all_mu, axis=0)


# -- Decoding -----------------------------------------------------------------

def decode_mu(model, mu_np, day_index, device):
    """Decode a single mu vector -> (H, W) mean field."""
    model.eval()
    with torch.no_grad():
        z = torch.FloatTensor(mu_np).unsqueeze(0).to(device)    # (1, latent_dim)
        c = make_conditioning(np.array([day_index])).to(device)  # (1, 3)
        mean, _ = model.decoder(z, c)                            # (1, C, H, W)
    return mean.squeeze().cpu().numpy()                          # (H, W)


# -- PNG export ---------------------------------------------------------------

def save_rainfall_png(field: np.ndarray, mask: np.ndarray, output_path: Path):
    """Save a spatial rainfall field as a clean PNG image."""
    display = field.copy().astype(float)
    display[~mask.astype(bool)] = np.nan

    # Clamp colorscale to 5th–95th percentile of land pixels
    land_vals = display[mask.astype(bool)]
    vmin = np.nanpercentile(land_vals, 5)
    vmax = np.nanpercentile(land_vals, 95)

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor('#F0EEE5')
    ax.set_facecolor('#F0EEE5')

    im = ax.imshow(display, cmap='Blues', vmin=vmin, vmax=vmax, origin='upper')
    ax.axis('off')
    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='#F0EEE5')
    plt.close(fig)


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True, help='*_FilteredGrid.npy (T, C, H, W)')
    parser.add_argument('--mask', required=True, help='*_LandMask.npy (H, W)')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--n-points', type=int, default=6)
    parser.add_argument('--n-sample', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Load data -------------------------------------------------------------
    print(f"Loading data from {args.data}")
    data = np.load(args.data)           # (T, C, H, W)
    mask = np.load(args.mask)           # (H, W)
    T, C, H, W = data.shape
    print(f"  data shape: {data.shape}, mask shape: {mask.shape}")

    # -- Load checkpoint -------------------------------------------------------
    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = config_from_checkpoint(ckpt['config'])

    model = ClimateVAE(config, input_height=H, input_width=W).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print("  Model loaded successfully")

    # -- Sample days + encode --------------------------------------------------
    n_sample = min(args.n_sample, T)
    sample_idx = rng.choice(T, size=n_sample, replace=False)
    sample_idx.sort()
    print(f"Encoding {n_sample} randomly sampled days...")
    mu_all = encode_days(model, data, sample_idx, device)       # (N, 64)
    print(f"  mu shape: {mu_all.shape}")

    # -- PCA ------------------------------------------------------------------
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(mu_all)                       # (N, 2)
    variance_explained = pca.explained_variance_ratio_.tolist()
    print(f"  PCA variance explained: {[f'{v:.3f}' for v in variance_explained]}")

    # Normalize full cloud to [-1, 1]
    cloud_min = coords_2d.min(axis=0)
    cloud_max = coords_2d.max(axis=0)
    def normalize(pts):
        return 2 * (pts - cloud_min) / (cloud_max - cloud_min) - 1

    coords_norm = normalize(coords_2d)

    # -- K-Means: pick representative points -----------------------------------
    print(f"Selecting {args.n_points} representative points via K-Means...")
    km = KMeans(n_clusters=args.n_points, random_state=args.seed, n_init='auto')
    km.fit(coords_2d)

    selected = []
    for cluster_id, center in enumerate(km.cluster_centers_):
        distances = np.linalg.norm(coords_2d - center, axis=1)
        nearest_local = int(np.argmin(distances))      # index in sample_idx
        actual_day = int(sample_idx[nearest_local])

        selected.append({
            'cluster_id': cluster_id,
            'local_idx': nearest_local,
            'day_index': actual_day,
            'mu': mu_all[nearest_local],
            'pcaX': float(coords_2d[nearest_local, 0]),
            'pcaY': float(coords_2d[nearest_local, 1]),
            'pcaNormX': float(coords_norm[nearest_local, 0]),
            'pcaNormY': float(coords_norm[nearest_local, 1]),
        })

    # Sort by PCA x for consistent labelling order
    selected.sort(key=lambda p: p['pcaNormX'])

    # -- Decode + save PNGs ----------------------------------------------------
    print("Decoding and saving rainfall maps...")
    points_out = []
    for i, pt in enumerate(selected):
        field = decode_mu(model, pt['mu'], pt['day_index'], device)
        img_name = f"point-{i+1}.png"
        save_rainfall_png(field, mask, output_dir / img_name)
        print(f"  Saved {img_name}  (day {pt['day_index']}, "
              f"norm=({pt['pcaNormX']:.2f}, {pt['pcaNormY']:.2f}))")

        points_out.append({
            'id': i + 1,
            'pcaNormX': pt['pcaNormX'],
            'pcaNormY': pt['pcaNormY'],
            'label': f"Cluster {i+1}",      # <-- rename these by hand after inspection
            'imgSrc': f"../assets/pca-samples/{img_name}",
            'dayIndex': pt['day_index'],
        })

    # -- Background cloud (downsample for JSON size) ---------------------------
    cloud_step = max(1, len(coords_norm) // 200)
    cloud_out = [
        {'x': float(coords_norm[j, 0]), 'y': float(coords_norm[j, 1])}
        for j in range(0, len(coords_norm), cloud_step)
    ]

    # -- Write points.json -----------------------------------------------------
    manifest = {
        'pcaVarianceExplained': variance_explained,
        'normMin': cloud_min.tolist(),
        'normMax': cloud_max.tolist(),
        'cloud': cloud_out,
        'points': points_out,
    }
    json_path = output_dir / 'points.json'
    with open(json_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {json_path}")
    print("\nDone. Open each PNG, then rename the 'label' fields in points.json to describe each scenario.")


if __name__ == '__main__':
    main()
