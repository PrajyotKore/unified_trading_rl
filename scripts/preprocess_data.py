"""
Preprocess NIFTY 50/100 minute data for UNIFIED trading system.
This script loads raw data, computes technical indicators, labels regimes, 
and creates train/val/test splits for meta-learning.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from data.regime_labeling import RegimeLabeler
from utils.utils import load_config, setup_logging

def main():
    # Setup
    project_root = Path(__file__).parent.parent
    config = load_config(project_root / 'config' / 'config.yaml')
    logger = setup_logging(
        log_dir=project_root / config['logging']['log_dir'],
        name='preprocess_data',
        level=config['logging']['verbosity']
    )
    
    logger.info("=" * 80)
    logger.info("UNIFIED Trading System - Data Preprocessing")
    logger.info("=" * 80)
    
    # Paths
    raw_data_dir = project_root / config['data']['raw_path']
    processed_data_dir = project_root / config['data']['processed_path']
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Find data file
    csv_files = list(raw_data_dir.glob("*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in {raw_data_dir}!")
        logger.info("Please run scripts/download_data.py first.")
        return
    
    # Prioritize NIFTY data
    nifty_files = [f for f in csv_files if 'NIFTY' in f.name.upper()]
    if nifty_files:
        data_file = nifty_files[0]
    else:
        # Fallback to whatever is there (e.g. VIX if NIFTY missing)
        # But prefer larger files as VIX might be small
        csv_files.sort(key=lambda x: x.stat().st_size, reverse=True)
        data_file = csv_files[0]
        
    logger.info(f"Using data file: {data_file.name}")
    
    # Step 1: Load and preprocess data
    logger.info("\n[Step 1/4] Loading and cleaning data...")
    loader = DataLoader(
        data_path=str(data_file),
        sequence_length=config['data']['sequence_length'],
        prediction_horizon=config['data']['prediction_horizon']
    )
    
    raw_df = loader.load_raw_data()
    df = loader.preprocess(raw_df)
    logger.info(f"✓ Data loaded: {len(df)} rows")
    
    # Step 2: Feature engineering
    logger.info("\n[Step 2/4] Computing technical indicators...")
    feature_engineer = FeatureEngineer()
    df = feature_engineer.compute_all_features(df)
    logger.info(f"✓ Computed {len(feature_engineer.get_feature_columns())} features")
    logger.info(f"✓ Rows after feature engineering: {len(df)}")
    
    if len(df) == 0:
        logger.error("No data remaining after feature engineering! Check your indicators/data.")
        return
    
    # Step 3: Regime labeling
    logger.info("\n[Step 3/4] Labeling market regimes...")
    regime_labeler = RegimeLabeler(num_regimes=config['world_model']['num_regimes'])
    df = regime_labeler.label_regimes(df)
    logger.info("✓ Regime labeling complete")
    
    # Step 4: Save processed data
    logger.info("\n[Step 4/4] Saving processed data...")
    
    # Save full processed dataset
    output_file = processed_data_dir / 'nifty_processed.parquet'
    loader.save_processed(df, str(output_file))
    
    # Save train/val/test splits
    train_df, val_df, test_df = loader.create_train_test_split(df)
    loader.save_processed(train_df, str(processed_data_dir / 'train.parquet'))
    loader.save_processed(val_df, str(processed_data_dir / 'val.parquet'))
    loader.save_processed(test_df, str(processed_data_dir / 'test.parquet'))
    
    # Save walk-forward windows
    logger.info("\nCreating walk-forward windows...")
    windows = loader.create_walk_forward_windows(
        df,
        num_windows=config['validation']['num_windows'],
        train_days=config['validation']['train_window_days'],
        test_days=config['validation']['test_window_days']
    )
    
    if not windows:
        logger.warning("No walk-forward windows created! Check dataset size vs window parameters.")
    
    for i, (train_window, test_window) in enumerate(windows):
        loader.save_processed(train_window, str(processed_data_dir / f'wf_train_{i}.parquet'))
        loader.save_processed(test_window, str(processed_data_dir / f'wf_test_{i}.parquet'))
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Preprocessing Summary:")
    logger.info(f"  Input File: {data_file.name}")
    logger.info(f"  Total samples: {len(df):,}")
    logger.info(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    logger.info(f"  Features: {len(feature_engineer.get_feature_columns())}")
    logger.info(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    logger.info(f"  Walk-forward windows: {len(windows)}")
    logger.info(f"  Output directory: {processed_data_dir}")
    logger.info("=" * 80)
    logger.info("\n✓ Preprocessing complete! Ready for training.\n")


if __name__ == '__main__':
    main()
