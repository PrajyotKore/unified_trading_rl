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
import seaborn as sns
from tqdm import tqdm
import torch
from gymnasium.wrappers import TimeLimit
from typing import List, Dict

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
    """Evaluate policy on environment and return detailed info."""
    trajectory_returns = []
    trajectory_values = []
    trajectory_infos = []
    
    for _ in tqdm(range(episodes), desc="Evaluating"):
        obs, _ = env.reset()
        done = False
        truncated = False
        
        # Track portfolio value
        values = [env.unwrapped.portfolio_value]
        infos = []
        
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
            
            # Store regime and returns for attribution
            # Get regime from env directly
            regime = env.unwrapped.df.iloc[env.unwrapped.current_step]['regime_name'] if hasattr(env.unwrapped, 'df') else 'unknown'
            info['regime_name'] = regime
            infos.append(info)
            
        # Calculate returns from values
        values = np.array(values)
        if len(values) > 1:
            returns = np.diff(values) / values[:-1]
            trajectory_returns.append(returns)
            trajectory_values.append(values)
            trajectory_infos.append(infos)
        
    return trajectory_returns, trajectory_values, trajectory_infos

def plot_regime_performance(infos: List[List[Dict]], save_path: Path):
    """Plot performance attribution by market regime."""
    all_data = []
    for episode in infos:
        for step in episode:
            all_data.append({
                'Regime': step['regime_name'],
                'Return': step['return'],
                'Position': step['position']
            })
    
    df = pd.DataFrame(all_data)
    if df.empty: return

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.barplot(data=df, x='Regime', y='Return', estimator=np.mean)
    plt.title('Average Return by Regime')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x='Regime', y='Position')
    plt.title('Position Sizing by Regime')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def run_evaluation_on_data(data_path: Path, config: dict, world_model, policy_path: Path, args: argparse.Namespace, device: str, results_suffix: str = ""):
    """Helper to run evaluation on a specific data file."""
    logger.info(f"Evaluating on {data_path.name}")
    
    dataset = TimeSeriesDataset(
        data_path=str(data_path),
        sequence_length=config['data']['sequence_length'],
        prediction_horizon=config['data']['prediction_horizon']
    )
    
    env = TradingEnv(
        dataset=dataset,
        world_model=world_model,
        window_size=config['data']['sequence_length'],
        transaction_cost_bps=args.cost_bps,
        device=device
    )
    env = TimeLimit(env, max_episode_steps=5000)
    
    policy = MetaPolicy(env, config, device=device)
    policy.load(str(policy_path))
    
    returns_list, values_list, infos_list = evaluate_policy(policy, env, episodes=args.episodes)
    
    if not returns_list:
        return None

    metrics = {
        'sharpe': [],
        'sortino': [],
        'calmar': [],
        'max_dd': [],
        'total_return': [],
        'win_rate': []
    }
    
    for returns, values, infos in zip(returns_list, values_list, infos_list):
        metrics['sharpe'].append(empyrical.sharpe_ratio(returns))
        metrics['sortino'].append(empyrical.sortino_ratio(returns))
        metrics['calmar'].append(empyrical.calmar_ratio(returns))
        metrics['max_dd'].append(empyrical.max_drawdown(returns))
        metrics['total_return'].append((values[-1]/values[0] - 1) * 100)
        metrics['win_rate'].append(np.mean(returns > 0) * 100)
        
    # Aggregate
    agg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    
    # Save artifacts
    results_dir = Path(__file__).parent.parent / 'logs/evaluation' / results_suffix
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot Mean Equity
    plt.figure(figsize=(10, 6))
    min_len = min(len(v) for v in values_list)
    truncated_values = [v[:min_len] for v in values_list]
    mean_values = np.mean(truncated_values, axis=0)
    plt.plot(mean_values)
    plt.title(f"Mean Equity Curve - {results_suffix}")
    plt.savefig(results_dir / 'mean_equity.png')
    plt.close()
    
    plot_regime_performance(infos_list, results_dir / 'regime_performance.png')
    
    return agg_metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate UNIFIED Policy')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--model_path', type=str, help='Path to policy checkpoint')
    parser.add_argument('--cost_bps', type=float, default=5.0, help='Transaction cost in basis points')
    parser.add_argument('--episodes', type=int, default=20, help='Number of episodes to evaluate')
    parser.add_argument('--walk_forward', action='store_true', help='Run walk-forward validation')

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
    
    # Load Policy path
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = project_root / config['logging']['checkpoint_dir'] / 'meta_rl/stage3_policy.zip'
        if not model_path.exists():
            model_path = project_root / config['logging']['checkpoint_dir'] / 'meta_rl/stage2_policy.zip'
            
    if not model_path.exists():
        logger.error(f"Policy not found at {model_path}")
        return

    if args.walk_forward:
        logger.info("Running Walk-Forward Validation...")
        wf_files = sorted(list((project_root / config['data']['processed_path']).glob('wf_test_*.parquet')))
        if not wf_files:
            logger.error("No walk-forward files found!")
            return
            
        wf_results = []
        for i, wf_file in enumerate(wf_files):
            res = run_evaluation_on_data(wf_file, config, world_model, model_path, args, device, f"wf_window_{i}")
            if res:
                wf_results.append(res)
        
        # Summary of WF
        if wf_results:
            wf_df = pd.DataFrame(wf_results)
            logger.info("\n" + "="*50)
            logger.info("WALK-FORWARD SUMMARY")
            logger.info("="*50)
            for col in wf_df.columns:
                logger.info(f"{col.title():<15}: {wf_df[col].mean():>8.4f}")
            logger.info("="*50)
    else:
        # Standard Evaluation on val.parquet
        res = run_evaluation_on_data(Path(data_path_str), config, world_model, model_path, args, device, "standard_eval")
        if res:
            logger.info("\n" + "="*50)
            logger.info("EVALUATION SUMMARY")
            logger.info("="*50)
            for k, v in res.items():
                logger.info(f"{k.title():<15}: {v:>8.4f}")
            logger.info("="*50)

if __name__ == '__main__':
    main()
