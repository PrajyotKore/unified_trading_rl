"""
Meta-RL Training Script for UNIFIED System.
Stages 2-5: Offline Meta-RL -> Online Fine-tuning -> Validation.
"""

import sys
from pathlib import Path
import argparse
import logging
import torch
import pandas as pd
import numpy as np
from typing import Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.utils import load_config, setup_logging, get_device
from models.world_model import WorldModel, TransformerWorldModel
from models.domain_randomization import DomainRandomizer
from models.meta_policy import MetaPolicy
from environments.synthetic_env import SyntheticTradingEnv
from environments.trading_env import TradingEnv
from data.dataset import TimeSeriesDataset

logger = logging.getLogger('train_meta_rl')

def main():
    parser = argparse.ArgumentParser(description='Train UNIFIED Meta-RL Policy')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume')
    parser.add_argument('--stage', type=int, default=2, help='Starting stage (2-5)')
    args = parser.parse_args()
    
    # Setup
    project_root = Path(__file__).parent.parent
    config = load_config(project_root / args.config)
    
    setup_logging(log_dir=project_root / 'logs/meta_rl')
    device = get_device(config['compute'].get('device', 'cpu'))
    
    logger.info("Initializing Meta-RL Training Pipeline")
    logger.info(f"Starting at Stage {args.stage}")
    
    # Load Data
    data_dir = project_root / config['data']['processed_path']
    data_path = data_dir / 'train.parquet'
    
    # Use dataset class for sampling
    dataset = TimeSeriesDataset(
        str(data_path),
        sequence_length=config['data']['sequence_length'],
        prediction_horizon=config['data']['prediction_horizon']
    )
    
    # helper to sample initial states for synthetic env
    def initial_state_sampler():
        idx = np.random.randint(0, len(dataset))
        inputs, _ = dataset[idx]
        return {
            'ohlcv': inputs['ohlcv'], # [5, seq]
            'features': inputs['features'] # [seq, dim]
        }
        
    # Load World Model (Common for all stages)
    wm_checkpoint_dir = project_root / config['logging']['checkpoint_dir'] / 'world_model'
    wm_path = wm_checkpoint_dir / 'best_model.pt'
    
    if not wm_path.exists():
        logger.warning(f"best_model.pt not found at {wm_path}. Looking for latest checkpoint...")
        checkpoints = sorted(wm_checkpoint_dir.glob("checkpoint_*.pt"))
        if checkpoints:
            wm_path = checkpoints[-1] 
            logger.info(f"Using latest checkpoint: {wm_path}")
        else:
            logger.error("World Model not found! Please run Stage 1.")
            return

    # Get dimensions from dataset
    sample_input, _ = dataset[0]
    input_channels = sample_input['ohlcv'].shape[0]
    feature_dim = sample_input['features'].shape[1]

    # Initialize TransformerWorldModel
    logger.info("Loading Transformer World Model...")
    d_model = config['world_model'].get('d_model', 128) # Default to 128 if not in config
    
    world_model = TransformerWorldModel(
        input_channels=input_channels,
        feature_dim=feature_dim,
        d_model=d_model,
        nhead=4,
        num_layers=2,
        dim_feedforward=d_model * 4,
        dropout=config['world_model']['dropout'],
        num_regimes=config['world_model']['num_regimes'],
        sequence_length=config['data']['sequence_length']
    ).to(device)
    
    checkpoint = torch.load(wm_path, map_location=device)
    missing_keys, unexpected_keys = world_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if missing_keys:
        logger.warning(f"Missing keys in state_dict: {missing_keys}. Using default initialization.")
    if unexpected_keys:
        logger.warning(f"Unexpected keys in state_dict: {unexpected_keys}.")
    world_model.eval()
    logger.info("World Model loaded")
    
    # STAGE 2: Offline Meta-RL on Synthetic Data
    if args.stage <= 2:
        logger.info("\n=== STAGE 2: Offline Meta-RL on Synthetic Data ===")
        
        # Domain Randomizer
        dr = DomainRandomizer(config['domain_randomization'], device=device)
        
        # Create Synthetic Environment
        env = SyntheticTradingEnv(
            world_model=world_model,
            domain_randomizer=dr,
            initial_state_sampler=initial_state_sampler,
            episode_length=config['training']['episode_length'],
            device=device
        )
        
        # Initialize Policy
        policy = MetaPolicy(env, config, device=device)
        
        # Train
        logger.info("Training policy on synthetic data...")
        policy.learn(total_timesteps=config['training']['stage2_timesteps'])
        
        # Save Stage 2 Checkpoint
        save_path = project_root / config['logging']['checkpoint_dir'] / 'meta_rl/stage2_policy.zip'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        policy.save(str(save_path))
        logger.info(f"Stage 2 (Synthetic) complete. Saved to {save_path}")

    # STAGE 3: Online Fine-tuning on Real Data
    if args.stage <= 3:
        logger.info("\n=== STAGE 3: Online Fine-tuning on Real Data ===")
        
        # Create Real Environment
        # Pass dataset and world_model for Latent State
        # df is also needed usually for internal step logic, which dataset wraps
        
        real_env = TradingEnv(
            dataset=dataset,
            world_model=world_model,
            window_size=config['data']['sequence_length'],
            transaction_cost_bps=5.0,
            risk_config=config['reward'],
            device=device
        )
        
        # Load Stage 2 policy if needed
        policy_path = project_root / config['logging']['checkpoint_dir'] / 'meta_rl/stage2_policy.zip'
        
        if policy_path.exists():
            logger.info(f"Loading Stage 2 policy from {policy_path}")
            policy = MetaPolicy(real_env, config, device=device)
            policy.load(str(policy_path))
        else:
            logger.warning("Stage 2 policy not found. Initializing from scratch (Not recommended).")
            policy = MetaPolicy(real_env, config, device=device)
            
        # Update Environment (set logic specific to real env)
        # policy.model.set_env(real_env) 
        # Note: sb3 set_env works, but policy helper wrapper might not expose it directly.
        # But policy.model is the RecurrentPPO object.
        policy.model.set_env(real_env)
        
        # Train (Fine-tune)
        logger.info("Fine-tuning policy on real data...")
        policy.learn(total_timesteps=config['training']['stage3_timesteps'])
        
        # Save Stage 3 Checkpoint
        save_path = project_root / config['logging']['checkpoint_dir'] / 'meta_rl/stage3_policy.zip'
        policy.save(str(save_path))
        logger.info(f"Stage 3 (Real) complete. Saved to {save_path}")

    # STAGE 4-5 (Placeholders or Extensions)
    # ...

if __name__ == '__main__':
    main()
