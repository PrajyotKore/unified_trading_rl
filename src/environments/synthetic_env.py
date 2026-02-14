"""
Synthetic Trading Environment for UNIFIED System.
Uses World Model for transition dynamics.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from typing import Dict, Optional, Tuple, Callable
import logging

from models.world_model import WorldModel
from models.domain_randomization import DomainRandomizer

logger = logging.getLogger(__name__)

class SyntheticTradingEnv(gym.Env):
    """
    Synthetic Trading Environment using World Model.
    
    Dynamics:
    s_{t+1}, r_{t+1} ~ WorldModel(s_t)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        world_model: WorldModel,
        domain_randomizer: DomainRandomizer,
        initial_state_sampler: Callable[[], Dict[str, torch.Tensor]],
        episode_length: int = 100,
        initial_balance: float = 100_000.0,
        transaction_cost_bps: float = 5.0,
        reward_scaling: float = 1.0,
        device: str = 'cpu'
    ):
        """
        Initialize Synthetic Environment.
        
        Args:
            world_model: Pre-trained world model
            domain_randomizer: Domain randomization module
            initial_state_sampler: Function returning initial (ohlcv, features)
            episode_length: Steps per episode
            initial_balance: Starting cash
            transaction_cost_bps: Base cost
            reward_scaling: Reward scale
            device: 'cpu' or 'cuda'
        """
        super().__init__()
        
        self.world_model = world_model
        self.domain_randomizer = domain_randomizer
        self.initial_state_sampler = initial_state_sampler
        self.episode_length = episode_length
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_bps / 10000.0
        self.reward_scaling = reward_scaling
        self.device = device
        
        # Dimensions from world model
        self.input_channels = world_model.input_channels
        self.feature_dim = world_model.feature_dim
        
        # Action space: Target position [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation space:
        # Latent state (z_t) + Raw Features (f_t) + Portfolio State
        
        # We need to know latent dimension.
        lstm_out_dim = world_model.lstm_hidden_size * (2 if world_model.bidirectional else 1)
        
        # Match TradingEnv: Latent + Features + Portfolio(3) + Penalties(2) + Cost(1) = 194
        obs_dim = lstm_out_dim + self.feature_dim + 3 + 2 + 1
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # Internal state
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0
        self.portfolio_value = initial_balance
        self.current_ohlcv = None
        self.current_features = None
        self.wm_hidden = None
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment."""
        super().reset(seed=seed)
        
        # Sample initial state (window)
        init_data = self.initial_state_sampler()
        
        # To device
        self.current_ohlcv = init_data['ohlcv'].to(self.device).unsqueeze(0)  # [1, 5, seq]
        self.current_features = init_data['features'].to(self.device).unsqueeze(0)  # [1, seq, dim]
        
        # Init portfolio
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.portfolio_value = self.initial_balance
        
        # Generate entire synthetic trajectory for this episode
        # Use WorldModel.sample_trajectory which handles auto-regression
        
        with torch.no_grad():
            self.world_model.eval()
            
            # Predict next step first to get initial latent
            preds = self.world_model.predict_next_step(self.current_ohlcv, self.current_features)
            initial_latent = preds['latent'][0].cpu().numpy()
            
            # We need to store latents and features for the whole episode.
            # Since sample_trajectory doesn't return latents/features, using manual loop again here
            # But incorporating the window update logic we added to WorldModel.
            
            self.episode_latents = []
            self.episode_features = [] # Store features for observation
            self.episode_returns = []
            self.episode_vols = []
            
            curr_ohlcv = self.current_ohlcv.clone()
            curr_features = self.current_features.clone()
            
            # Generating trajectory loop
            for i in range(self.episode_length):
                # Predict
                pred = self.world_model.predict_next_step(curr_ohlcv, curr_features)
                
                # Store Latent & Feature (for observation at this step)
                lat = pred['latent'][0].cpu().numpy()
                if np.isnan(lat).any():
                    logger.error(f"NaN Latent detected at step {i} of reset!")
                    lat = np.nan_to_num(lat)
                self.episode_latents.append(lat)
                
                # Extract feature vector at current timestep (last in window)
                # features: [1, seq, dim] -> take last -> [dim]
                feat_vec = curr_features[0, -1].cpu().numpy()
                if np.isnan(feat_vec).any():
                    logger.error(f"NaN Feature detected at step {i} of reset!")
                    feat_vec = np.nan_to_num(feat_vec)
                self.episode_features.append(feat_vec)
                
                # Sample return
                mu = pred['price_pred'][0, 0]
                sigma = pred['price_pred'][0, 1]
                
                # Randomize sigma and clip returns to prevent blowout
                sigma = self.domain_randomizer.randomize_volatility(sigma.unsqueeze(0)).item()
                step_ret = torch.normal(mu, sigma).item()
                step_ret = np.clip(step_ret, -0.2, 0.2) # Clip to +/- 20% per step
                
                sampled_volume = torch.normal(pred['volume_pred'][0, 0], pred['volume_pred'][0, 1]).item()
                sampled_volume = np.clip(sampled_volume, -10.0, 10.0)
                
                self.episode_returns.append(step_ret)
                self.episode_vols.append(sigma)
                
                # UPDATE WINDOW (Auto-regression logic)
                # 1. Shift window left
                curr_ohlcv = torch.roll(curr_ohlcv, -1, dims=2)
                curr_features = torch.roll(curr_features, -1, dims=1)
                
                # 2. Update candle (Use exponential update for log-returns)
                prev_close = curr_ohlcv[:, 3, -2]
                # prev_close is normalized price. step_ret is log-return.
                new_close = prev_close * torch.exp(torch.tensor(step_ret, device=self.device, dtype=torch.float32))
                new_close = torch.clamp(new_close, 0.1, 10.0) # Keep normalized price in [0.1x, 10x] range
                
                curr_ohlcv[:, 0, -1] = prev_close
                curr_ohlcv[:, 1, -1] = new_close
                curr_ohlcv[:, 2, -1] = new_close
                curr_ohlcv[:, 3, -1] = new_close
                curr_ohlcv[:, 4, -1] = sampled_volume
                
                # Check for NaNs immediately to prevent loop corruption
                if torch.isnan(curr_ohlcv).any():
                    logger.error(f"NaN detected in OHLCV window at step {i}! Aborting simulation loop.")
                    # Fill rest with last good values to prevent policy crash
                    for j in range(i, self.episode_length):
                         self.episode_returns.append(0.0)
                         self.episode_vols.append(0.01)
                         self.episode_latents.append(self.episode_latents[-1] if self.episode_latents else np.zeros(self.feature_dim))
                         self.episode_features.append(self.episode_features[-1] if self.episode_features else np.zeros(self.feature_dim))
                    break
                
                # Safety: Clip OHLCV values to prevent exponential explosion in training
                curr_ohlcv = torch.clamp(curr_ohlcv, -100.0, 100.0)
                
                # 3. Update features (Repeat last)
                curr_features[:, -1, :] = curr_features[:, -2, :]
        
        # Placeholder for penalties in synthetic (could randomize them if wanted)
        self.vol_penalty = 0.1
        self.dd_penalty = 0.5
        
        return self._get_observation(self.episode_latents[0], self.episode_features[0]), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute step in simulator."""
        target_position = float(np.clip(action[0], -1, 1))
        
        # Get pre-generated values for THIS step (prediction of next state)
        # Note: current_step index corresponds to action at t, leading to return at t+1
        if self.current_step >= len(self.episode_returns):
            # Safe catch
            return self._get_observation(self.episode_latents[-1], self.episode_features[-1]), 0.0, True, False, {}
            
        step_return = self.episode_returns[self.current_step]
        vol = self.episode_vols[self.current_step]
        
        # Calculate Position Change
        position_change = target_position - self.position
        
        # Apply Transaction Costs
        # Using domain randomizer logic manually
        # cost ~ impact * sqrt(change) * vol + spread * change
        # Simplified:
        ret_tensor = torch.tensor([step_return], device=self.device, dtype=torch.float32)
        pos_change_tensor = torch.tensor([position_change], device=self.device, dtype=torch.float32)
        vol_tensor = torch.tensor([vol], device=self.device, dtype=torch.float32)
        
        net_return = self.domain_randomizer.apply_randomized_costs(
            ret_tensor, pos_change_tensor, vol_tensor
        ).item()
        
        # Update Portfolio
        # PnL = Position * NetReturn
        gross_return = self.position * net_return
        prev_value = self.portfolio_value
        self.portfolio_value *= (1 + gross_return)
        
        # Update balance (assuming cash for costs etc)
        cost_abs = abs(position_change) * self.transaction_cost_pct * self.portfolio_value
        self.balance -= cost_abs
        
        self.position = target_position
        
        # Increment curriculum
        self.domain_randomizer.step()
        
        terminated = False
        
        # Reward
        if self.portfolio_value <= 1e-6:
            self.portfolio_value = 1e-6
            reward = -10.0  # Bankruptcy penalty
            terminated = True
        else:
            # Clip ratio to avoid log(inf) or log(0)
            ratio = np.clip(self.portfolio_value / prev_value, 0.01, 100.0)
            reward = np.log(ratio) * self.reward_scaling
        
        # Next State
        self.current_step += 1
        
        # Check if already done (max steps)
        if not terminated:
            terminated = self.current_step >= self.episode_length
        # Use next latent state if available, else last
        next_idx = min(self.current_step, len(self.episode_latents) - 1)
        next_latent = self.episode_latents[next_idx]
        next_feature = self.episode_features[next_idx]
        
        info = {
            'portfolio_value': self.portfolio_value,
            'return': gross_return,
            'position': self.position
        }
        
        return self._get_observation(next_latent, next_feature), reward, terminated, False, info

    def _get_observation(self, latent: np.ndarray, feature: np.ndarray) -> np.ndarray:
        unrealized_pnl = 0.0
        portfolio_state = np.array([
            self.position,
            unrealized_pnl,
            self.balance / max(self.initial_balance, 1e-6),
            self.vol_penalty,
            self.dd_penalty,
            self.transaction_cost_pct * 100.0 # to pct
        ], dtype=np.float32)
        
        obs = np.concatenate([latent, feature, portfolio_state])
        if np.isnan(obs).any():
             logger.error(f"NaN Observation! Latent: {np.isnan(latent).any()}, Feature: {np.isnan(feature).any()}, Port: {np.isnan(portfolio_state).any()}")
             obs = np.nan_to_num(obs)
        
        # Final safety check: Clip observations to reasonable range for policy stability
        obs = np.clip(obs, -1e6, 1e6)
             
        return obs
