import math
import numpy as np
import pandas as pd
import xarray as xr

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from typing import Tuple, Dict, List
from torch.utils.data import Dataset, DataLoader

class ClimateDataset(Dataset):
    """Dataset for loading and preprocessing NetCDF climate data"""
    
    def __init__(self, csv_path: str, config, scenario: str = 'historical', load_npy: bool=False):
        self.config = config
        self.scenario = scenario
        
        # Load file list
        df = pd.read_csv(csv_path)
        df = df[df['scenario'] == scenario]
        self.models = config.MODELS
        
        # Load and preprocess data
        self.data, self.mask = self._load_data(load_npy)
        self.conditioning = self._create_conditioning()
        
        # Store actual spatial dimensions for model initialization
        _, _, self.height, self.width = self.data.shape
        print(f"Loaded {len(self.data)} days of data")
        print(f"Spatial dimensions: {self.height} x {self.width}")
        
    def _load_data(self, use_npy: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load NetCDF files and crop to bbox"""
        all_data = []
        
        for model in self.models:
            model_data = []

            model_path = self.config.DATA_PATH + '/' + self.scenario + '/' + model + '/' + self.scenario + '_' + model + '_FilteredGrid.npy'

            land_mask = self._create_mask(model, use_npy)            
            
            if use_npy:
                # netCDF files have been previously processed, so just load them in from their .npy
                model_data = np.load(model_path)
                all_data.append(model_data)
            else:
                for var in self.config.VARIABLES:                    
                    # Load NetCDF
                    path = self.config.DATA_PATH + '/' + self.scenario + '/' + model + '/' + \
                        var + '_' + self.scenario + '_' + model + '_CCAM_daily_NZ5km_bc.nc'
                    ds = xr.open_dataset(path)
                    
                    # Crop to bbox
                    ds_crop = ds.sel(
                        longitude=slice(self.config.BBOX['lon_min'], self.config.BBOX['lon_max']),
                        latitude=slice(self.config.BBOX['lat_max'], self.config.BBOX['lat_min'])
                    )
                    
                    var_data = ds_crop[var].values  # (time, lat, lon)
                    
                    # Transform
                    if var == 'pr':
                        var_data = np.log1p(np.maximum(var_data, 0))  # Ensure non-negative
                    
                    # Normalize only over LAND pixels
                    land_values = var_data[:, land_mask]  # (time, n_land_pixels)
                    mean = np.mean(land_values)  # Single value
                    std = np.std(land_values) + 1e-6
                    
                    # Normalize all data
                    var_data = (var_data - mean) / std
                    
                    # Set ocean pixels to 0 (they'll be masked later)
                    var_data[:, ~land_mask] = 0
                    
                    model_data.append(var_data)
                
                model_data = np.stack(model_data, axis=1)
                np.save(model_path, model_data)
                all_data.append(model_data)
        
        all_data = np.concatenate(all_data, axis=0)
        
        # Convert mask to tensor (will be broadcast during training)
        land_mask_tensor = torch.FloatTensor(land_mask).unsqueeze(0)  # (1, H, W)
        
        return torch.FloatTensor(all_data), land_mask_tensor
    
    def _create_mask(self, model: str, use_npy: bool) -> np.ndarray:
        """Exported creation/loading of mask to this private method"""

        mask_path = self.config.DATA_PATH + '/' + self.scenario + '/' + model + '/' + self.scenario + '_' + model + '_LandMask.npy'

        if use_npy:
            mask = np.load(mask_path)
        else:
            # Create land mask from first variable (True where land exists)
            first_var_path = self.config.DATA_PATH + '/' + self.scenario + '/' + model + '/' + \
                            self.config.VARIABLES[0] + '_' + self.scenario + '_' + model + '_CCAM_daily_NZ5km_bc.nc'
            ds_first = xr.open_dataset(first_var_path)
            ds_first_crop = ds_first.sel(
                longitude=slice(self.config.BBOX['lon_min'], self.config.BBOX['lon_max']),
                latitude=slice(self.config.BBOX['lat_max'], self.config.BBOX['lat_min'])
            )
            # Land mask: True where we have valid data
            mask = ~np.isnan(ds_first_crop[self.config.VARIABLES[0]].values[0])

            # Save as npy
            np.save(mask_path, mask)

        return mask
    
    def _create_conditioning(self) -> torch.Tensor:
        """Create conditioning vectors for each day"""
        n_days = len(self.data)
        
        # Day of year (cyclic encoding)
        doy = np.arange(n_days) % 365
        doy_sin = np.sin(2 * np.pi * doy / 365)
        doy_cos = np.cos(2 * np.pi * doy / 365)
        
        # Year (normalized)
        year = np.arange(n_days) // 365
        year_norm = year / max(year.max(), 1)  # Avoid division by zero
        
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
