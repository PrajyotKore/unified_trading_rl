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
from typing import Dict, Optional
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

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
        # Use TensorBoard log if enabled
        tb_log = str(project_root / config['logging']['log_dir'] / 'meta_rl/stage2') if config['logging'].get('tensorboard') else None
        
        policy = MetaPolicy(env, config, device=device, tensorboard_log=tb_log)
        
        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=config['training'].get('save_frequency', 5000),
            save_path=str(project_root / config['logging']['checkpoint_dir'] / 'meta_rl/stage2_checkpoints'),
            name_prefix='stage2_model'
        )
        
        # Train
        logger.info(f"Training policy on synthetic data for {config['training']['stage2_timesteps']} steps...")
        policy.learn(
            total_timesteps=config['training']['stage2_timesteps'],
            callback=checkpoint_callback
        )
        
        # Save Stage 2 Checkpoint
        save_path = project_root / config['logging']['checkpoint_dir'] / 'meta_rl/stage2_policy.zip'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        policy.save(str(save_path))
        logger.info(f"Stage 2 (Synthetic) complete. Saved to {save_path}")

    # STAGE 3: Online Fine-tuning on Real Data
    if args.stage <= 3:
        logger.info("\n=== STAGE 3: Online Fine-tuning on Real Data ===")
        
        real_env = TradingEnv(
            dataset=dataset,
            world_model=world_model,
            window_size=config['data']['sequence_length'],
            transaction_cost_bps=5.0,
            risk_config=config['reward'],
            device=device
        )
        
        policy_path = project_root / config['logging']['checkpoint_dir'] / 'meta_rl/stage2_policy.zip'
        
        if args.resume:
            policy_path = Path(args.resume)

        tb_log = str(project_root / config['logging']['log_dir'] / 'meta_rl/stage3') if config['logging'].get('tensorboard') else None

        if policy_path.exists():
            logger.info(f"Loading policy from {policy_path}")
            policy = MetaPolicy(real_env, config, device=device, tensorboard_log=tb_log)
            policy.load(str(policy_path))
        else:
            logger.warning("Policy checkpoint not found. Initializing from scratch.")
            policy = MetaPolicy(real_env, config, device=device, tensorboard_log=tb_log)
            
        policy.model.set_env(real_env)
        
        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=config['training'].get('save_frequency', 5000),
            save_path=str(project_root / config['logging']['checkpoint_dir'] / 'meta_rl/stage3_checkpoints'),
            name_prefix='stage3_model'
        )
        
        logger.info(f"Fine-tuning policy on real data for {config['training']['stage3_timesteps']} steps...")
        policy.learn(
            total_timesteps=config['training']['stage3_timesteps'],
            callback=checkpoint_callback
        )
        
        save_path = project_root / config['logging']['checkpoint_dir'] / 'meta_rl/stage3_policy.zip'
        policy.save(str(save_path))
        logger.info(f"Stage 3 (Real) complete. Saved to {save_path}")

    # STAGE 4: Risk-aware Multi-task Fine-tuning
    if args.stage <= 4:
        logger.info("\n=== STAGE 4: Risk-aware Multi-task Fine-tuning ===")
        
        # For Stage 4, we might vary risk parameters (lambda/mu)
        risk_configs = [
            {'volatility_penalty': 0.1, 'drawdown_penalty': 0.5}, # Balanced
            {'volatility_penalty': 0.5, 'drawdown_penalty': 1.0}, # Conservative
            {'volatility_penalty': 0.05, 'drawdown_penalty': 0.2} # Aggressive
        ]
        
        # Load Stage 3 policy
        policy_path = project_root / config['logging']['checkpoint_dir'] / 'meta_rl/stage3_policy.zip'
        
        # In multi-task, we can cycle through different risk profiles
        # For simplicity, we'll use the environment with config rewards
        # but one could wrap it in a VecEnv with different parameters.
        
        tb_log = str(project_root / config['logging']['log_dir'] / 'meta_rl/stage4') if config['logging'].get('tensorboard') else None

        real_env = TradingEnv(
            dataset=dataset,
            world_model=world_model,
            window_size=config['data']['sequence_length'],
            transaction_cost_bps=5.0,
            risk_config=config['reward'],
            device=device
        )

        if policy_path.exists():
            policy = MetaPolicy(real_env, config, device=device, tensorboard_log=tb_log)
            policy.load(str(policy_path))
            logger.info(f"Stage 4: Loaded Stage 3 policy from {policy_path}")
        else:
            policy = MetaPolicy(real_env, config, device=device, tensorboard_log=tb_log)
            logger.warning("Stage 4: Stage 3 policy not found. Starting from scratch.")

        policy.model.set_env(real_env)
        
        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=config['training'].get('save_frequency', 5000),
            save_path=str(project_root / config['logging']['checkpoint_dir'] / 'meta_rl/stage4_checkpoints'),
            name_prefix='stage4_model'
        )

        logger.info(f"Training Stage 4: Risk-aware adaptation for {config['training']['stage4_timesteps']} steps...")
        policy.learn(
            total_timesteps=config['training']['stage4_timesteps'],
            callback=checkpoint_callback
        )
        
        save_path = project_root / config['logging']['checkpoint_dir'] / 'meta_rl/stage4_policy.zip'
        policy.save(str(save_path))
        logger.info(f"Stage 4 complete. Saved to {save_path}")

    # STAGE 5: Validation & Final Evaluation
    if args.stage <= 5:
        logger.info("\n=== STAGE 5: Final Evaluation ===")
        
        # Load Test Data
        test_path = project_root / config['data']['processed_path'] / 'test.parquet'
        if not test_path.exists():
            logger.error(f"Test data not found at {test_path}")
            return
            
        test_dataset = TimeSeriesDataset(
            str(test_path),
            sequence_length=config['data']['sequence_length'],
            prediction_horizon=config['data']['prediction_horizon']
        )
        
        eval_env = TradingEnv(
            dataset=test_dataset,
            world_model=world_model,
            window_size=config['data']['sequence_length'],
            transaction_cost_bps=5.0,
            risk_config=config['reward'],
            device=device
        )
        
        # Load best policy (Stage 4)
        policy_path = project_root / config['logging']['checkpoint_dir'] / 'meta_rl/stage4_policy.zip'
        if not policy_path.exists():
            policy_path = project_root / config['logging']['checkpoint_dir'] / 'meta_rl/stage3_policy.zip'
            
        if policy_path.exists():
            policy = MetaPolicy(eval_env, config, device=device)
            policy.load(str(policy_path))
            logger.info(f"Evaluating policy from {policy_path}")
        else:
            logger.error("No policy found for evaluation!")
            return

        # Run Evaluation Loop
        obs, _ = eval_env.reset()
        done = False
        portfolio_values = []
        
        logger.info("Running evaluation on test set...")
        while not done:
            action, _states = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            portfolio_values.append(info['portfolio_value'])
            done = terminated or truncated
            
        # Calculate Metrics
        returns = pd.Series(portfolio_values).pct_change().dropna()
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 6.5 * 60) if len(returns) > 0 else 0 # Minutely to Annualized
        
        # Drawdown
        peak = pd.Series(portfolio_values).expanding().max()
        drawdown = (pd.Series(portfolio_values) - peak) / peak
        max_drawdown = drawdown.min()
        
        logger.info(f"Final Evaluation Metrics:")
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"Annualized Sharpe: {sharpe:.2f}")
        logger.info(f"Max Drawdown: {max_drawdown:.2%}")
        logger.info(f"Final Portfolio Value: {portfolio_values[-1]:.2f}")

if __name__ == '__main__':
    main()
