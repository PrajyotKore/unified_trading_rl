"""
Data loader for NIFTY 50 minute-level data from Kaggle.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load and preprocess NIFTY 50 minute data.
    """
    
    def __init__(
        self,
        data_path: str,
        sequence_length: int = 50,
        prediction_horizon: int = 5
    ):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to raw data file
            sequence_length: Number of timesteps in lookback window
            prediction_horizon: Forward-looking prediction window
        """
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.data = None
        
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw NIFTY 50 minute data.
        
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Loading data from {self.data_path}")
        
        # Try different file formats
        if self.data_path.endswith('.csv'):
            df = pd.read_csv(self.data_path)
        elif self.data_path.endswith('.parquet'):
            df = pd.read_parquet(self.data_path)
        elif self.data_path.endswith('.xlsx'):
            df = pd.read_excel(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path}")
        
        logger.info(f"Loaded {len(df)} rows")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Date range: {df.iloc[0]['date'] if 'date' in df.columns else 'N/A'} to {df.iloc[-1]['date'] if 'date' in df.columns else 'N/A'}")
        
        return df
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess raw data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing data...")
        
        # Standardize column names
        column_mapping = {
            'Date': 'date',
            'Time': 'time',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Symbol': 'symbol'
        }
        df = df.rename(columns=column_mapping)
        
        # Ensure we have datetime
        if 'date' in df.columns:
            if 'time' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
            else:
                df['datetime'] = pd.to_datetime(df['date'])
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        else:
            # Try to find a date-like column
            date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
            if date_cols:
                df['datetime'] = pd.to_datetime(df[date_cols[0]])
            else:
                raise ValueError("No datetime column found")

        # Required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        # If any missing, check for case-insensitive matches
        for req in required_cols:
            if req not in df.columns:
                matches = [c for c in df.columns if c.lower() == req]
                if matches:
                    df = df.rename(columns={matches[0]: req})
        
        # Symbol handling
        if 'symbol' not in df.columns:
            df['symbol'] = 'NIFTY'
            
        df = df.sort_values(['symbol', 'datetime']).reset_index(drop=True)
        
        # Handle missing values
        logger.info(f"Missing values before cleaning:\n{df[required_cols].isnull().sum()}")
        
        # Forward fill missing values (up to 5 periods)
        df[required_cols] = df[required_cols].ffill(limit=5)
        
        # Drop remaining NaNs
        df = df.dropna(subset=required_cols)
        
        # Remove outliers (3-sigma rule on returns) symbol-wise to avoid cross-stock returns
        def remove_outliers(symbol_df):
            symbol_df = symbol_df.copy()
            rets = symbol_df['close'].pct_change().fillna(0)
            mean_ret = rets.mean()
            std_ret = rets.std()
            return symbol_df[np.abs(rets - mean_ret) < 3 * std_ret]

        df = df.groupby('symbol', group_keys=False).apply(remove_outliers)
        
        # Ensure positive prices and volumes
        df = df[(df[['open', 'high', 'low', 'close']] > 0).all(axis=1)]
        df = df[df['volume'] >= 0]
        
        logger.info(f"Data after preprocessing: {len(df)} rows")
        
        self.data = df
        return df
    
    def aggregate_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate minute data to daily (optional, for comparison with paper).
        
        Args:
            df: Minute-level DataFrame
            
        Returns:
            Daily DataFrame
        """
        logger.info("Aggregating to daily data...")
        
        df['date'] = df['datetime'].dt.date
        
        daily = df.groupby('date').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'datetime': 'first'
        }).reset_index(drop=True)
        
        logger.info(f"Daily data: {len(daily)} rows")
        return daily
    
    def create_train_test_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test splits.
        
        Args:
            df: DataFrame to split
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            
        Returns:
            (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        logger.info(f"Train: {len(train_df)} rows ({train_df['datetime'].min()} to {train_df['datetime'].max()})")
        logger.info(f"Val: {len(val_df)} rows ({val_df['datetime'].min()} to {val_df['datetime'].max()})")
        logger.info(f"Test: {len(test_df)} rows ({test_df['datetime'].min()} to {test_df['datetime'].max()})")
        
        return train_df, val_df, test_df
    
    def create_walk_forward_windows(
        self,
        df: pd.DataFrame,
        num_windows: int = 6,
        train_days: int = 365,
        test_days: int = 90
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create walk-forward validation windows.
        
        Args:
            df: DataFrame to split
            num_windows: Number of walk-forward windows
            train_days: Days in training window
            test_days: Days in test window
            
        Returns:
            List of (train_df, test_df) tuples
        """
        logger.info(f"Creating {num_windows} walk-forward windows...")
        
        df['date'] = df['datetime'].dt.date
        unique_dates = sorted(df['date'].unique())
        
        windows = []
        window_size = train_days + test_days
        
        for i in range(num_windows):
            start_idx = i * test_days  # Overlapping training windows
            
            # Check if we have enough data remaining
            if start_idx >= len(unique_dates):
                logger.warning(f"Not enough data for window {i+1}. Stopping.")
                break
                
            train_start = unique_dates[start_idx]
            
            # Ensure we have enough data for training + testing
            if start_idx + train_days + test_days > len(unique_dates):
                logger.warning(f"Not enough data for full window {i+1}. Stopping.")
                break
                
            train_end_idx = start_idx + train_days
            train_end = unique_dates[train_end_idx]
            
            test_start = unique_dates[train_end_idx]
            test_end_idx = min(train_end_idx + test_days, len(unique_dates))
            test_end = unique_dates[test_end_idx - 1]
            
            train_df = df[(df['date'] >= train_start) & (df['date'] < train_end)].copy()
            test_df = df[(df['date'] >= test_start) & (df['date'] <= test_end)].copy()
            
            if len(train_df) > 0 and len(test_df) > 0:
                logger.info(f"Window {i+1}: Train {train_start} to {train_end} ({len(train_df)} rows), "
                           f"Test {test_start} to {test_end} ({len(test_df)} rows)")
                windows.append((train_df, test_df))
        
        return windows
    
    def save_processed(self, df: pd.DataFrame, save_path: str):
        """
        Save processed data.
        
        Args:
            df: DataFrame to save
            save_path: Path to save file
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_parquet(save_path, index=False)
        logger.info(f"Saved processed data to {save_path}")
    
    def load_processed(self, data_path: str) -> pd.DataFrame:
        """
        Load processed data.
        
        Args:
            data_path: Path to processed data
            
        Returns:
            DataFrame
        """
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded processed data from {data_path}: {len(df)} rows")
        return df


def download_kaggle_dataset(dataset_name: str, download_path: str):
    """
    Download dataset from Kaggle.
    
    Args:
        dataset_name: Kaggle dataset name (e.g., "debashis74017/nifty-50-minute-data")
        download_path: Directory to download to
    """
    import kaggle
    
    logger.info(f"Downloading dataset {dataset_name}...")
    os.makedirs(download_path, exist_ok=True)
    
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        dataset_name,
        path=download_path,
        unzip=True
    )
    
    logger.info(f"Downloaded to {download_path}")
