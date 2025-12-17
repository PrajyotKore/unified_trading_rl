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
        # We need to expose the *latent* state of the world model to the policy?
        # Or the *raw* observation (ohlcv window)?
        # The World Model paper says "Policy receives latent state z_t".
        # Let's check RecurrentPPO. It expects vector input.
        # If we feed latent state (output of LSTM), then Policy doesn't need its own LSTM?
        # But `MetaPolicy` uses `RecurrentPPO`.
        # If we feed RAW observation history, Policy does the encoding.
        # "Sample Efficiency: 10-20x via world models" usually means training inside world model.
        # Providing world model latent state is common.
        # Let's output `z_t` (latent) + portfolio state.
        
        # We need to know latent dimension.
        lstm_out_dim = world_model.lstm_hidden_size * (2 if world_model.bidirectional else 1)
        obs_dim = lstm_out_dim + 3  # latent + portfolio
        
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
        with torch.no_grad():
            self.world_model.eval()
            
            # Predict next step first to get initial latent
            preds = self.world_model.predict_next_step(self.current_ohlcv, self.current_features)
            initial_latent = preds['latent'][0].cpu().numpy()
            
            # Rollout trajectory
            # Note: sample_trajectory returns dict of tensors [batch, steps, ...]
            trajectory = self.world_model.sample_trajectory(
                initial_ohlcv=self.current_ohlcv,
                initial_features=self.current_features,
                num_steps=self.episode_length,
                randomize=True, # Use default noise
                noise_scale=0.1
            )
            
            # Also need latent states for each step. 
            # sample_trajectory might not return latents by default in current impl.
            # But let's assume valid returns/vols.
            # We will use the *latent state* from the rollout as observation?
            # WorldModel.sample_trajectory doesn't return latents currently.
            # We should update WorldModel or approximating it. 
            # For now, just using Random Walk Latents or fixing sample_trajectory is needed.
            
            # Actually, `sample_trajectory` updates hidden state.
            # Let's modify manual rollout here to keep latents.
            
            self.episode_latents = []
            self.episode_returns = []
            self.episode_vols = []
            
            curr_ohlcv = self.current_ohlcv
            curr_features = self.current_features
            hidden = None
            
            # Generating trajectory loop manually to capture latents
            for _ in range(self.episode_length):
                pred = self.world_model.predict_next_step(curr_ohlcv, curr_features, hidden)
                
                # Store latent
                self.episode_latents.append(pred['latent'][0].cpu().numpy())
                
                # Sample return
                mu = pred['price_pred'][0, 0]
                sigma = pred['price_pred'][0, 1]
                
                # Randomize sigma
                sigma = self.domain_randomizer.randomize_volatility(sigma.unsqueeze(0)).item()
                step_ret = torch.normal(mu, sigma).item()
                
                self.episode_returns.append(step_ret)
                self.episode_vols.append(sigma)
                
                hidden = pred['hidden']
                # Note: Not updating ohlcv/features realistically for next step (expensive).
                # Using static input + hidden state evolution.
                # This approximates "Mental Simulation" in latent space (World Models paper).
        
        return self._get_observation(self.episode_latents[0]), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute step in simulator."""
        target_position = float(np.clip(action[0], -1, 1))
        
        # Get pre-generated values for THIS step (prediction of next state)
        # Note: current_step index corresponds to action at t, leading to return at t+1
        if self.current_step >= len(self.episode_returns):
            # Safe catch
            return self._get_observation(self.episode_latents[-1]), 0.0, True, False, {}
            
        step_return = self.episode_returns[self.current_step]
        vol = self.episode_vols[self.current_step]
        
        # Calculate Position Change
        position_change = target_position - self.position
        
        # Apply Transaction Costs
        # Using domain randomizer logic manually
        # cost ~ impact * sqrt(change) * vol + spread * change
        # Simplified:
        ret_tensor = torch.tensor([step_return], device=self.device)
        pos_change_tensor = torch.tensor([position_change], device=self.device)
        vol_tensor = torch.tensor([vol], device=self.device)
        
        net_return = self.domain_randomizer.apply_randomized_costs(
            ret_tensor, pos_change_tensor, vol_tensor
        ).item()
        
        # Update Portfolio
        # PnL = Position * NetReturn
        gross_return = self.position * net_return
        prev_value = self.portfolio_value
        self.portfolio_value *= (1 + gross_return)
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
            reward = np.log(self.portfolio_value / prev_value) * self.reward_scaling
        
        # Next State
        self.current_step += 1
        
        # Check if already done (max steps)
        if not terminated:
            terminated = self.current_step >= self.episode_length
        # Use next latent state if available, else last
        next_latent_idx = min(self.current_step, len(self.episode_latents) - 1)
        next_latent = self.episode_latents[next_latent_idx]
        
        info = {
            'portfolio_value': self.portfolio_value,
            'return': gross_return,
            'position': self.position
        }
        
        return self._get_observation(next_latent), reward, terminated, False, info

    def _get_observation(self, latent: np.ndarray) -> np.ndarray:
        unrealized_pnl = 0.0
        portfolio_state = np.array([
            self.position,
            unrealized_pnl,
            self.balance / self.initial_balance
        ], dtype=np.float32)
        return np.concatenate([latent, portfolio_state])
