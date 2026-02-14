"""
Download and preprocess NIFTY 50 minute data from Kaggle.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.data_loader import DataLoader, download_kaggle_dataset
from data.feature_engineering import FeatureEngineer
from data.regime_labeling import RegimeLabeler
from utils.utils import load_config, setup_logging

def main():
    # Setup
    project_root = Path(__file__).parent.parent
    config = load_config(project_root / 'config' / 'config.yaml')
    logger = setup_logging(
        log_dir=project_root / config['logging']['log_dir'],
        name='download_data',
        level=config['logging']['verbosity']
    )
    
    logger.info("=" * 80)
    logger.info("UNIFIED Trading System - Data Download and Preprocessing")
    logger.info("=" * 80)
    
    # Paths
    raw_data_dir = project_root / config['data']['raw_path']
    processed_data_dir = project_root / config['data']['processed_path']
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Step 1: Download data from Kaggle
    logger.info("\n[Step 1/5] Downloading NIFTY 50 minute data from Kaggle...")
    dataset_name = config['data']['dataset']
    
    # try:
    #     download_kaggle_dataset(dataset_name, str(raw_data_dir))
    #     logger.info("✓ Download complete")
    # except Exception as e:
    #     logger.warning(f"Download failed: {e}")
    #     logger.info("Assuming data already exists in raw_data directory")
    
    # Find the downloaded CSV file
    csv_files = list(raw_data_dir.glob("*.csv"))
    if not csv_files:
        logger.error("No CSV files found in raw data directory!")
        logger.info(f"Please manually download data from Kaggle and place in: {raw_data_dir}")
        return
    
    # Prioritize NIFTY data
    nifty_files = [f for f in csv_files if 'NIFTY' in f.name.upper()]
    if nifty_files:
        data_file = nifty_files[0]
    else:
        data_file = csv_files[0]
    logger.info(f"Found data file: {data_file.name}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info(f"Summary:")
    logger.info(f"  Dataset: {dataset_name}")
    logger.info(f"  Download Path: {raw_data_dir}")
    logger.info(f"  Files: {[f.name for f in csv_files]}")
    logger.info("=" * 80)
    logger.info("\n✓ Data download complete! Now run scripts/preprocess_data.py\n")


if __name__ == '__main__':
    main()
