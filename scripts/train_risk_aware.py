
"""
Stage 4: Risk-Aware Multi-Task Fine-Tuning.

This script fine-tunes the policy on a distribution of risk penalties.
The goal is to make the agent robust to varying risk preferences or to find a balanced strategy.
"""

import sys
from pathlib import Path
import argparse
import logging
import numpy as np
import gymnasium as gym
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.utils import load_config, setup_logging, get_device
from models.meta_policy import MetaPolicy
from environments.trading_env import TradingEnv
from data.dataset import TimeSeriesDataset
from models.world_model import WorldModel
import torch

logger = logging.getLogger('train_risk_aware')

class RiskAwareWrapper(gym.Wrapper):
    """
    Wrapper that randomizes risk penalties (Volatility & Drawdown) at the start of each episode.
    This simulates 'Multi-Task' learning where the task is a specific risk profile.
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # Sample new risk penalties
        # Range: Volatility Penalty [0.0, 0.5], Drawdown Penalty [0.0, 1.0]
        # These ranges should be tuned based on business logic
        new_vol_penalty = np.random.uniform(0.0, 0.5)
        new_dd_penalty = np.random.uniform(0.0, 1.0)
        
        # Apply to base environment
        # Access unwrapped to get to the real TradingEnv
        base_env = self.env.unwrapped
        base_env.vol_penalty = new_vol_penalty
        base_env.dd_penalty = new_dd_penalty
        
        # Log occasionally
        # print(f"Risk Profile: Vol={new_vol_penalty:.2f}, DD={new_dd_penalty:.2f}")
        
        return self.env.reset(seed=seed, options=options)

def main():
    parser = argparse.ArgumentParser(description='Stage 4: Risk-Aware Training')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--model_input', type=str, default='models/checkpoints/meta_rl/stage3_policy.zip', help='Input policy')
    parser.add_argument('--model_output', type=str, default='models/checkpoints/meta_rl/stage4_policy.zip', help='Output policy')
    args = parser.parse_args()
    
    # Setup
    project_root = Path(__file__).parent.parent
    config = load_config(project_root / args.config)
    setup_logging(log_dir=project_root / 'logs/risk_aware')
    device = get_device(config['compute'].get('device', 'cpu'))
    
    logger.info("=== STAGE 4: Risk-Aware Multi-Task Fine-Tuning ===")
    
    # Load Data & World Model (Same as Stage 3)
    data_path = project_root / config['data']['processed_path'] / 'train.parquet'
    dataset = TimeSeriesDataset(
        str(data_path),
        sequence_length=config['data']['sequence_length'],
        prediction_horizon=config['data']['prediction_horizon']
    )
    
    wm_path = project_root / config['logging']['checkpoint_dir'] / 'world_model/best_model.pt'
    world_model = WorldModel(
        input_channels=5, # Standard
        feature_dim=dataset[0][0]['features'].shape[1],
        lstm_hidden_size=config['world_model']['lstm_hidden_size'],
        lstm_num_layers=config['world_model']['lstm_num_layers'],
        num_regimes=config['world_model']['num_regimes']
    ).to(device)
    
    checkpoint = torch.load(wm_path, map_location=device)
    world_model.load_state_dict(checkpoint['model_state_dict'])
    world_model.eval()
    
    # Create Environment
    env = TradingEnv(
        dataset=dataset,
        world_model=world_model,
        window_size=config['data']['sequence_length'],
        transaction_cost_bps=5.0,
        risk_config=config['reward'], # Base config
        device=device
    )
    
    # Wrap for Risk Randomization
    env = RiskAwareWrapper(env)
    
    # Load Policy
    policy_path = project_root / args.model_input
    if not policy_path.exists():
        logger.error(f"Input policy not found at {policy_path}. Please complete Stage 3.")
        return
        
    logger.info(f"Loading policy from {policy_path}")
    policy = MetaPolicy(env, config, device=device)
    policy.load(str(policy_path))
    
    # Train
    steps = config['training'].get('stage4_timesteps', 100000)
    logger.info(f"Training for {steps} timesteps with randomized risk penalties...")
    
    policy.learn(total_timesteps=steps)
    
    # Save
    save_path = project_root / args.model_output
    save_path.parent.mkdir(parents=True, exist_ok=True)
    policy.save(str(save_path))
    logger.info(f"Stage 4 complete. Saved to {save_path}")

if __name__ == '__main__':
    main()
