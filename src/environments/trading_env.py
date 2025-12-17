"""
Trading Environment for UNIFIED System.
Compatible with Gymnasium interface.
Handles portfolio management, transaction costs, and reward calculation.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

import torch

class TradingEnv(gym.Env):
    """
    Trading Environment with realistic market mechanics.
    Supports both Raw Features and World Model Latent States.
    
    Features:
    - Continuous action space [-1, 1] for target position
    - Transaction costs (spread, slippage, market impact)
    - Margin trading support
    - Vectorized for speed (optional)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        dataset = None,  # TimeSeriesDataset
        world_model = None, # WorldModel
        device: str = 'cpu',
        initial_balance: float = 100_000.0,
        max_position: float = 1.0,  # Max generic position size (fraction of portfolio)
        transaction_cost_bps: float = 5.0,  # Basis points
        reward_scaling: float = 1.0,
        window_size: int = 50,  # Observation window
        risk_config: Optional[dict] = None  # New: Risk parameters
    ):
        """
        Initialize Trading Environment.
        
        Args:
            df: DataFrame containing market data (must have 'close', 'returns')
            initial_balance: Starting cash
            max_position: Maximum generic leverage (1.0 = 100%)
            transaction_cost_bps: Base transaction cost in bps
            reward_scaling: Scale factor for rewards
            window_size: Lookback window for observation
            feature_cols: Columns to use as observation features
        """
        super().__init__()
        
        self.dataset = dataset
        self.world_model = world_model
        self.device = device
        
        # Fallback to DF if dataset not provided
        if self.dataset is None and df is not None:
            self.df = df.reset_index(drop=True)
            self.max_steps = len(self.df)
        elif self.dataset is not None:
            self.df = dataset.df
            self.max_steps = len(self.dataset)
        else:
            raise ValueError("Must provide either df or dataset")

        self.initial_balance = initial_balance
        self.max_position = max_position
        self.transaction_cost_pct = transaction_cost_bps / 10000.0
        self.reward_scaling = reward_scaling
        self.window_size = window_size
        
        # Risk Configuration
        self.risk_config = risk_config or {}
        self.vol_penalty = self.risk_config.get('volatility_penalty', 0.0)
        self.dd_penalty = self.risk_config.get('drawdown_penalty', 0.0)
        
        # Action space: Target position [-1, 1]
        # -1 = Full Short, 0 = Cash, 1 = Full Long
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation Space
        # Observation Space
        if self.world_model is not None:
            # Latent obs + Raw Feature obs (Hybrid)
            latent_dim = self.world_model.lstm_hidden_size * (2 if self.world_model.bidirectional else 1)
            # We append the *current* technical features (size: self.dataset.feature_dim)
            # Assuming dataset is available via accessors.
            # Usually self.df has ~69 columns, features are a subset.
            # We will grab the correct feature dim from the dataset or df.
            
            # Identify feature columns count
            exclude = ['datetime', 'date', 'regime', 'regime_name', 'open', 'high', 'low', 'close', 'volume']
            self.feature_cols = [c for c in self.df.columns if c not in exclude]
            self.feature_dim = len(self.feature_cols)

            obs_dim = latent_dim + self.feature_dim + 3
            self.use_latent = True
            # Ensure WM is in eval mode
            self.world_model.eval()
        else:
            # Raw feature obs (flattened window)
            exclude = ['datetime', 'date', 'regime', 'regime_name']
            self.feature_cols = [c for c in self.df.columns if c not in exclude]
            self.data_matrix = self.df[self.feature_cols].values.astype(np.float32)
            obs_dim = (len(self.feature_cols) * window_size) + 3
            self.use_latent = False
            
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # State variables
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0  # Current position fraction
        self.portfolio_value = initial_balance
        
        # Risk tracking
        self.peak_value = initial_balance
        self.return_history = []
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Random start
        # Leave enough data for window at start and some steps at end
        min_step = self.window_size
        max_step = self.max_steps - 100
        
        if max_step <= min_step:
            self.current_step = min_step
        else:
            self.current_step = self.np_random.integers(min_step, max_step)
            
        self.balance = self.initial_balance
        self.position = 0.0
        self.portfolio_value = self.initial_balance
        
        self.peak_value = self.initial_balance
        self.return_history = []
        
        return self._get_observation(), {}
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step."""
        # Clean action
        target_position = float(np.clip(action[0], -1, 1))
        
        # Get current price data
        current_price = self.df.iloc[self.current_step]['close']
        
        # Check bounds
        if self.current_step + 1 >= len(self.df):
            return self._get_observation(), 0.0, True, False, {}
            
        next_price = self.df.iloc[self.current_step + 1]['close']
        
        # Calculate position change
        position_change = target_position - self.position
        
        # Transaction costs
        # Cost applied to traded volume (change in position * portfolio value)
        trade_value = abs(position_change) * self.portfolio_value
        cost = trade_value * self.transaction_cost_pct
        
        # Update position
        self.position = target_position
        
        # Calculate return
        # Simple return: position * percentage_price_change
        price_return = (next_price - current_price) / current_price
        gross_return = self.position * price_return
        
        # Update portfolio value
        # NAV_new = NAV_old * (1 + gross_return) - cost
        prev_value = self.portfolio_value
        self.portfolio_value = self.portfolio_value * (1 + gross_return) - cost
        
        # Check Bankruptcy
        if self.portfolio_value <= 1e-6:
            self.portfolio_value = 1e-6
            reward = -10.0
            terminated = True
            drawdown = 1.0
            volatility = 0.0 # Irrelevant
        else:
            # Risk Metrics
            if self.portfolio_value > self.peak_value:
                self.peak_value = self.portfolio_value
            drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
            
            self.return_history.append(gross_return)
            if len(self.return_history) > 20: self.return_history.pop(0)
            volatility = np.std(self.return_history) if len(self.return_history) > 5 else 0.0
            
            # Differential Sharpe Ratio (DSR)
            # Update running stats
            eta = 1/1000  # Learning rate for running stats
            
            # 1. Update A (First moment: Mean return)
            if not hasattr(self, 'running_A'): self.running_A = 0.0
            self.running_A = (1 - eta) * self.running_A + eta * gross_return
            
            # 2. Update B (Second moment: Mean squared return)
            if not hasattr(self, 'running_B'): self.running_B = 0.0
            self.running_B = (1 - eta) * self.running_B + eta * (gross_return ** 2)
            
            # 3. Compute DSR
            # Delta_S_t ~ (R_t - A_{t-1}) / (B_{t-1} - A_{t-1}^2)^(1/2)
            # Actually standard DSR formula derived from gradient
            
            std_dev = np.sqrt(self.running_B - self.running_A**2 + 1e-6)
            
            if std_dev > 1e-6:
                # D_t = (B_{t-1} * (R_t - A_{t-1}) - 0.5 * A_{t-1} * (R_t^2 - B_{t-1})) / (B_{t-1} - A_{t-1}^2)^(3/2)
                # Simplified robust version:
                dsr = (gross_return - self.running_A) / std_dev
            else:
                dsr = 0.0
                
            reward = dsr * self.reward_scaling
            
            # Apply penalties (if configured)
            reward -= self.vol_penalty * volatility
            reward -= self.dd_penalty * drawdown
            
            terminated = self.current_step >= self.max_steps - 1
            
        self.current_step += 1
        truncated = False
        
        # info
        info = {
            'portfolio_value': self.portfolio_value,
            'return': gross_return,
            'cost': cost,
            'position': self.position,
            'drawdown': drawdown,
            'volatility': volatility
        }
        
        return self._get_observation(), reward, terminated, truncated, info
        
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        if self.use_latent:
            # Use World Model to get latent state
            # Get window from dataset
            # TimeSeriesDataset valid_indices maps idx -> actual_index
            # We used current_step as index.
            # Assuming current_step matches dataset logic valid range?
            # dataset[i] returns data at valid_indices[i].
            # This mapping might differ from simple current_step if not aligned.
            # But dataset handles normalization.
            
            # If we passed 'dataset' we should use `dataset[idx]` where idx corresponds to current_step?
            # Creating dataset with valid_indices is tricky if we just random sample.
            # Let's assumes we use the dataset's getitem directly via some mapping
            # or we construct input tensors manually normalization.
            
            # Better: Use dataset.
            # We need to map self.current_step (index in DF) to dataset index.
            # dataset.valid_indices contains DF indices.
            # We need to find `i` such that `dataset.valid_indices[i] == self.current_step`.
            # This is slow.
            
            # Alternate: Manually normalize using dataset stats.
            # dataset._normalize_ohlcv expects dataframe slice.
            
            start = self.current_step - self.window_size
            end = self.current_step
            
            ohlcv = self.df.iloc[start:end][['open','high','low','close','volume']].values.astype(np.float32)
            features = self.df.iloc[start:end][self.dataset.feature_cols].values.astype(np.float32)
            
            # Normalize
            if self.dataset:
                ohlcv = self.dataset._normalize_ohlcv(ohlcv, self.current_step)
                features = self.dataset._normalize_features(features)
            
            # To Tensor
            ohlcv_t = torch.from_numpy(ohlcv.T).unsqueeze(0).to(self.device) # [1, 5, seq]
            features_t = torch.from_numpy(features).unsqueeze(0).to(self.device) # [1, seq, dim]
            
            with torch.no_grad():
                # Get latent
                # We need latent at END of window (representing state at current_step)
                latent = self.world_model.get_latent_representation(ohlcv_t, features_t)
                # latent shape [1, seq, dim]. Take last.
                latent_vec = latent[0, -1].cpu().numpy()
            
            # Extract Raw Features for the LAST step
            # features_t shape is [1, seq, dim]. We want normalized features at t.
            # However features_t is tensor. We can just take the last element from numpy features array.
            # features shape [window, dim]
            raw_features_vec = features[-1] # Normalized features at t
            
            # Concatenate Latent + Raw Features
            obs = np.concatenate([latent_vec, raw_features_vec])
            
        else:
            # Raw
            start = self.current_step - self.window_size
            end = self.current_step
            market_features = self.data_matrix[start:end].flatten()
            obs = market_features
            
        # Portfolio state
        unrealized_pnl = 0.0  # Simplification for continuous allocation
        portfolio_state = np.array([
            self.position,
            unrealized_pnl,
            self.balance / self.initial_balance  # Normalize
        ], dtype=np.float32)
        
        return np.concatenate([obs, portfolio_state])
