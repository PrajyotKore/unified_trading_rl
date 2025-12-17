"""
Dataset for world model training.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """
    Dataset for world model training on financial time series.
    """
    
    def __init__(
        self,
        data_path: str,
        sequence_length: int = 50,
        prediction_horizon: int = 5,
        feature_columns: Optional[list] = None,
        normalize: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to processed parquet file
            sequence_length: Number of timesteps in lookback window
            prediction_horizon: Number of steps to predict ahead
            feature_columns: List of feature column names (if None, auto-detect)
            normalize: Normalize features
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.normalize = normalize
        
        # Load data
        logger.info(f"Loading data from {data_path}")
        self.df = pd.read_parquet(data_path)
        
        # Identify OHLCV columns
        self.ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # Identify feature columns
        if feature_columns is None:
            # Exclude datetime, OHLCV, and regime columns
            exclude = self.ohlcv_cols + ['datetime', 'date', 'time', 'regime', 'regime_name']
            self.feature_cols = [col for col in self.df.columns if col not in exclude]
        else:
            self.feature_cols = feature_columns
        
        logger.info(f"OHLCV columns: {self.ohlcv_cols}")
        logger.info(f"Feature columns: {len(self.feature_cols)} features")
        
        # Extract arrays
        self.ohlcv = self.df[self.ohlcv_cols].values.astype(np.float32)
        self.features = self.df[self.feature_cols].values.astype(np.float32)
        
        # Check for NaNs/Infs in loaded data
        if np.isnan(self.ohlcv).any() or np.isinf(self.ohlcv).any():
            logger.warning("OHLCV data contains NaNs or Infs. replacing with 0.")
            self.ohlcv = np.nan_to_num(self.ohlcv)
            
        if np.isnan(self.features).any() or np.isinf(self.features).any():
            logger.warning("Features contain NaNs or Infs. replacing with 0.")
            self.features = np.nan_to_num(self.features)

        if 'regime' in self.df.columns:
            self.regimes = self.df['regime'].values.astype(np.int64)
        else:
            self.regimes = np.zeros(len(self.df), dtype=np.int64)  # default regime
        
        # Compute returns and volume changes for targets
        # Handle zero volume for log
        vol_safe = np.maximum(self.df['volume'].values, 1e-8)
        self.returns = np.diff(np.log(self.df['close'].values), prepend=0).astype(np.float32)
        self.volume_changes = np.diff(np.log(vol_safe + 1), prepend=0).astype(np.float32)
        
        # Normalize if requested
        if normalize:
            self._compute_normalization_stats()
            self.features = self._normalize_features(self.features)
        
        # Valid indices (need sequence_length before and prediction_horizon after)
        self.valid_indices = range(
            sequence_length,
            len(self.df) - prediction_horizon
        )
        
        logger.info(f"Dataset loaded: {len(self.valid_indices)} valid samples")
    
    def _compute_normalization_stats(self):
        """Compute mean and std for normalization."""
        self.feature_mean = np.mean(self.features, axis=0)
        self.feature_std = np.std(self.features, axis=0)
        
        # OHLCV normalization (price-based, so normalize relative to close)
        close_prices = self.ohlcv[:, 3]  # close column
        self.price_scale = np.maximum(close_prices, 1.0)
    
    def _normalize_features(self, features):
        """Normalize features safely."""
        # Handle constant features (std ~ 0)
        mask = self.feature_std > 1e-6
        normalized = np.zeros_like(features)
        
        if mask.any():
            normalized[:, mask] = (features[:, mask] - self.feature_mean[mask]) / (self.feature_std[mask] + 1e-8)
            
        return normalized
    
    def _normalize_ohlcv(self, ohlcv, idx):
        """Normalize OHLCV by current close price."""
        # Get close price at start of sequence
        # Use max to avoid division by zero
        close_price = max(float(self.price_scale[idx - self.sequence_length]), 1.0)
        
        normalized = ohlcv.copy()
        normalized[:, :4] = normalized[:, :4] / close_price  # normalize OHLC
        
        # Normalize volume relative to sequence mean
        vol_mean = np.mean(normalized[:, 4])
        if vol_mean > 1e-6:
            normalized[:, 4] = normalized[:, 4] / vol_mean
        else:
            # If volume is all zero or effectively zero, set to 0
            normalized[:, 4] = 0.0
            
        return normalized
    
    def __len__(self) -> int:
        """Dataset length."""
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[dict, dict]:
        """
        Get item.
        
        Args:
            idx: Index
            
        Returns:
            (inputs, targets) tuple
        """
        actual_idx = self.valid_indices[idx]
        
        # Input sequence: [actual_idx - sequence_length : actual_idx]
        ohlcv_seq = self.ohlcv[actual_idx - self.sequence_length:actual_idx]
        features_seq = self.features[actual_idx - self.sequence_length:actual_idx]
        
        # Normalize OHLCV
        if self.normalize:
            ohlcv_seq = self._normalize_ohlcv(ohlcv_seq, actual_idx)
        
        # Targets: predict next 'prediction_horizon' steps
        target_returns = self.returns[actual_idx:actual_idx + self.prediction_horizon]
        target_volumes = self.volume_changes[actual_idx:actual_idx + self.prediction_horizon]
        target_regimes = self.regimes[actual_idx:actual_idx + self.prediction_horizon]
        
        inputs = {
            'ohlcv': torch.from_numpy(ohlcv_seq.T),  # [channels=5, seq_len]
            'features': torch.from_numpy(features_seq),  # [seq_len, feature_dim]
        }
        
        targets = {
            'returns': torch.from_numpy(target_returns),  # [prediction_horizon]
            'volumes': torch.from_numpy(target_volumes),  # [prediction_horizon]
            'regimes': torch.from_numpy(target_regimes),  # [prediction_horizon]
        }
        
        return inputs, targets
    
    def get_sample_dimensions(self) -> dict:
        """Get sample dimensions for model initialization."""
        sample_inputs, sample_targets = self[0]
        return {
            'ohlcv_channels': sample_inputs['ohlcv'].shape[0],
            'sequence_length': sample_inputs['ohlcv'].shape[1],
            'feature_dim': sample_inputs['features'].shape[1],
            'prediction_horizon': len(sample_targets['returns'])
        }
