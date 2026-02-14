
import pandas as pd
import numpy as np
from pathlib import Path

def inspect_data():
    data_path = "data/processed/train.parquet"
    print(f"Loading {data_path}...")
    try:
        df = pd.read_parquet(data_path)
    except Exception as e:
        print(f"Error loading: {e}")
        return

    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())
    
    # Check OHLCV
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    print("\nOHLCV Stats:")
    print(df[ohlcv_cols].describe())
    
    print("\nNaNs in OHLCV:")
    print(df[ohlcv_cols].isna().sum())

    # Check Features (excluding OHLCV, datetime, regime)
    exclude = ohlcv_cols + ['datetime', 'date', 'regime', 'regime_name']
    feature_cols = [c for c in df.columns if c not in exclude]
    
    if not feature_cols:
        print("No feature columns found!")
    else:
        print(f"\nFeature Columns ({len(feature_cols)}):")
        print(feature_cols[:5], "...")
        
        feat_df = df[feature_cols]
        print("\nFeature Stats:")
        # Check global min/max
        print(f"Global Max: {feat_df.max().max()}")
        print(f"Global Min: {feat_df.min().min()}")
        print(f"Global Mean: {feat_df.mean().mean()}")
        
        # Check for NaNs
        nan_counts = feat_df.isna().sum()
        cols_with_nans = nan_counts[nan_counts > 0]
        if not cols_with_nans.empty:
            print("\nFeatures with NaNs:")
            print(cols_with_nans)
        else:
            print("\nNo NaNs in features.")
            
        # Check for Infs
        inf_counts = np.isinf(feat_df).sum()
        cols_with_infs = inf_counts[inf_counts > 0]
        if not cols_with_infs.empty:
             print("\nFeatures with Infs:")
             print(cols_with_infs)

    # Check Regime distribution if exists
    if 'regime' in df.columns:
        print("\nRegime Distribution:")
        print(df['regime'].value_counts())

if __name__ == "__main__":
    inspect_data()
