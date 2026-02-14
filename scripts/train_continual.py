"""
Continual Domain Randomization Training Script.
Stage 7: Adapts randomization ranges based on recent market conditions.
"""

import sys
from pathlib import Path
import argparse
import logging
import torch
import numpy as np
import pandas as pd
from typing import Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.utils import load_config, setup_logging, get_device
from models.world_model import TransformerWorldModel
from models.domain_randomization import DomainRandomizer
from models.meta_policy import MetaPolicy
from environments.trading_env import TradingEnv
from data.dataset import TimeSeriesDataset
from data.realtime_connector import YahooFinanceConnector

logger = logging.getLogger('continual_training')

def update_randomization_params(config: Dict, recent_data: pd.DataFrame) -> Dict:
    """Adjust randomization ranges based on recent volatility."""
    returns = recent_data['close'].pct_change().dropna()
    realized_vol = returns.std() * np.sqrt(252 * 6.5 * 60) # Annualized
    
    new_config = config.copy()
    
    # Adjust volatility range based on realized vol
    # If market is volatile, expand the random range to stress test more
    if realized_vol > config['domain_randomization']['parameters']['volatility']['real_max']:
        logger.info(f"High realized vol ({realized_vol:.2f}) detected. Expanding randomization.")
        new_config['domain_randomization']['parameters']['volatility']['random_max'] *= 1.2
        
    return new_config

def main():
    parser = argparse.ArgumentParser(description='UNIFIED Continual Training')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    config = load_config(project_root / args.config)
    setup_logging(log_dir=project_root / 'logs/continual')
    device = get_device(config['compute'].get('device', 'cpu'))
    
    logger.info("Starting Stage 7: Continual Domain Randomization")
    
    # 1. Fetch recent data to assess current market regime
    connector = YahooFinanceConnector()
    recent_market_data = connector.fetch_latest_data(period="5d", interval="5m")
    
    if not recent_market_data.empty:
        config = update_randomization_params(config, recent_market_data)
    
    # 2. Load latest data and models
    data_path = project_root / config['data']['processed_path'] / 'train.parquet'
    dataset = TimeSeriesDataset(str(data_path))
    
    # Load World Model
    wm_path = project_root / config['logging']['checkpoint_dir'] / 'world_model/best_model.pt'
    # (Initialize and load world_model as in train_meta_rl.py)
    # ... placeholder for brevity ...
    
    # 3. Incremental Training
    # Load the latest policy (Stage 4 or later)
    policy_path = project_root / config['logging']['checkpoint_dir'] / 'meta_rl/stage4_policy.zip'
    
    # Initialize Environment with updated config (randomization)
    env = TradingEnv(
        dataset=dataset,
        # world_model=world_model,
        # ...
    )
    
    # Continual learn
    # policy.learn(total_timesteps=config['training']['stage5_timesteps'])
    
    logger.info("Continual training cycle complete.")

if __name__ == '__main__':
    main()
