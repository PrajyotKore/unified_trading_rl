"""
Evaluation Script for UNIFIED System.
Stage 6: Walk-Forward Validation and Metrics.
"""

import sys
from pathlib import Path
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from gymnasium.wrappers import TimeLimit

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.utils import load_config, setup_logging, get_device
from models.meta_policy import MetaPolicy
from environments.trading_env import TradingEnv
from models.world_model import WorldModel, TransformerWorldModel
from data.dataset import TimeSeriesDataset
import empyrical

logger = logging.getLogger('evaluate')

def evaluate_policy(policy, env, episodes=1):
    """Evaluate policy on environment."""
    trajectory_returns = []
    trajectory_values = []
    
    for _ in tqdm(range(episodes), desc="Evaluating"):
        obs, _ = env.reset()
        done = False
        truncated = False
        
        # Track portfolio value
        values = [env.unwrapped.portfolio_value]
        
        # RNN state
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)
        
        while not (done or truncated):
            action, lstm_states = policy.predict(
                obs, 
                state=lstm_states, 
                episode_start=episode_start,
                deterministic=True
            )
            obs, reward, done, truncated, info = env.step(action)
            episode_start[0] = False
            values.append(env.unwrapped.portfolio_value)
        
        # Calculate returns from values
        values = np.array(values)
        if len(values) > 1:
            returns = np.diff(values) / values[:-1]
            trajectory_returns.append(returns)
            trajectory_values.append(values)
        
    return trajectory_returns, trajectory_values

def main():
    parser = argparse.ArgumentParser(description='Evaluate UNIFIED Policy')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--model_path', type=str, help='Path to policy checkpoint')
    parser.add_argument('--cost_bps', type=float, default=5.0, help='Transaction cost in basis points')
    parser.add_argument('--episodes', type=int, default=20, help='Number of episodes to evaluate')

    args = parser.parse_args()
    
    # Setup
    project_root = Path(__file__).parent.parent
    config = load_config(project_root / args.config)
    global logger
    logger = setup_logging(log_dir=project_root / 'logs/evaluation', name='evaluate')
    device = get_device(config['compute'].get('device', 'cpu'))
    
    logger.info("Starting Evaluation (Stage 6)")
    logger.info(f"Transaction Cost: {args.cost_bps} bps")
    
    # Load Data via Dataset
    data_path_str = str(project_root / config['data']['processed_path'] / 'val.parquet')
    if not Path(data_path_str).exists():
        data_path_str = str(project_root / config['data']['processed_path'] / 'train.parquet')
        logger.warning(f"Validation data not found, using {data_path_str} for evaluation.")
    
    dataset = TimeSeriesDataset(
        data_path=data_path_str,
        sequence_length=config['data']['sequence_length'],
        prediction_horizon=config['data']['prediction_horizon']
    )
    
    # Load World Model
    wm_checkpoint_dir = project_root / config['logging']['checkpoint_dir'] / 'world_model'
    wm_path = wm_checkpoint_dir / 'best_model.pt'
    
    if not wm_path.exists():
        logger.warning(f"best_model.pt not found at {wm_path}. Looking for latest checkpoint...")
        checkpoints = sorted(wm_checkpoint_dir.glob("checkpoint_*.pt"))
        if checkpoints:
            wm_path = checkpoints[-1] 
            logger.info(f"Using latest checkpoint: {wm_path}")
        else:
             logger.error("World Model not found! Cannot evaluate latent policy.")
             return
 
    # Get dimensions from dataset
    sample_input, _ = dataset[0]
    input_channels = sample_input['ohlcv'].shape[0]
    feature_dim = sample_input['features'].shape[1]
 
    # Initialize TransformerWorldModel
    logger.info("Loading Transformer World Model...")
    d_model = config['world_model'].get('d_model', 128) 
    
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
    
    # Create Environment
    env = TradingEnv(
        dataset=dataset,
        world_model=world_model,
        window_size=config['data']['sequence_length'],
        transaction_cost_bps=args.cost_bps,
        device=device
    )
    
    # Wrap with TimeLimit to ensure finite evaluation
    env = TimeLimit(env, max_episode_steps=5000)
    
    # Load Policy
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        # Default to Stage 3 model
        model_path = project_root / config['logging']['checkpoint_dir'] / 'meta_rl/stage3_policy.zip'
        
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return
        
    logger.info(f"Loading policy from {model_path}")
    policy = MetaPolicy(env, config, device=device)
    policy.load(str(model_path))
    
    # Run Evaluation

    episodes = args.episodes
    returns_list, values_list = evaluate_policy(policy, env, episodes=episodes)
    
    if not returns_list:
        logger.error("No returns collected (episode too short?)")
        return

    # Metrics
    # Metrics aggregation
    all_metrics = {
        'sharpe': [],
        'sortino': [],
        'max_dd': [],
        'cagr': [],
        'total_return': []
    }
    
    for returns, values in zip(returns_list, values_list):
        all_metrics['sharpe'].append(empyrical.sharpe_ratio(returns))
        all_metrics['sortino'].append(empyrical.sortino_ratio(returns))
        all_metrics['max_dd'].append(empyrical.max_drawdown(returns))
        all_metrics['cagr'].append(empyrical.cagr(returns))
        all_metrics['total_return'].append((values[-1]/values[0] - 1) * 100)
        
    # Compute stats
    logger.info("\n" + "="*50)
    logger.info(f"Performance Metrics ({episodes} Episodes)")
    logger.info("="*50)
    
    for metric, values in all_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Scaling correction for % metrics if needed (Sharpe/Sortino are ratios)
        is_pct = metric in ['max_dd', 'cagr', 'total_return']
        
        display_mean = mean_val * 100 if metric == 'max_dd' else mean_val
        display_std = std_val * 100 if metric == 'max_dd' else std_val
        
        if metric == 'max_dd': # Drawdown is usually negative
            suffix = "%"
        elif is_pct:
            suffix = "%" 
        else:
            suffix = ""
            
        logger.info(f"{metric.replace('_', ' ').title():<15}: {display_mean:>8.4f}{suffix} ± {display_std:.4f}")
        
    logger.info("="*50)
    
    # Save results
    results_dir = project_root / 'logs/evaluation'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot equity curve
    plt.figure(figsize=(10, 6))
    
    # Plot individual episodes
    for values in values_list:
        plt.plot(values, alpha=0.3, color='gray') # Faint lines for individual runs
        
    # Plot mean
    min_len = min(len(v) for v in values_list)
    truncated_values = [v[:min_len] for v in values_list]
    mean_values = np.mean(truncated_values, axis=0)
    plt.plot(mean_values, color='blue', linewidth=2, label='Mean Equity')
    
    plt.title(f"Equity Curve ({episodes} Episodes)")
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.savefig(results_dir / 'equity_curve.png')
    logger.info(f"Results saved to {results_dir}")

if __name__ == '__main__':
    main()
