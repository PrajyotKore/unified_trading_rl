"""
Meta-RL Policy for UNIFIED System.
Wraps RecurrentPPO from sb3-contrib.
"""

import torch
import torch.nn as nn
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MetaPolicy:
    """
    Meta-RL Policy based on Recurrent PPO (GRU/LSTM).
    Adapts to non-stationary regimes via hidden state.
    """
    
    def __init__(
        self,
        env,
        config: Dict,
        device: str = 'auto',
        tensorboard_log: Optional[str] = None
    ):
        """
        Initialize policy.
        
        Args:
            env: Trading environment
            config: Configuration dictionary
            device: 'cpu', 'cuda', 'mps' or 'auto'
            tensorboard_log: Path for logs
        """
        self.config = config
        self.env = env
        
        # Parse policy config
        policy_kwargs = {
            'net_arch': dict(pi=config['policy']['hidden_sizes'], vf=config['policy']['hidden_sizes']),
            'lstm_hidden_size': config['policy']['rnn_hidden_size'],
            'enable_critic_lstm': True,
        }
        
        self.model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=config['policy']['learning_rate'],
            n_steps=config['policy']['n_steps'],
            batch_size=config['policy']['batch_size'],
            n_epochs=config['policy']['n_epochs'],
            gamma=config['policy']['gamma'],
            gae_lambda=config['policy']['gae_lambda'],
            clip_range=config['policy']['clip_range'],
            ent_coef=config['policy']['ent_coef'],
            vf_coef=config['policy']['vf_coef'],
            max_grad_norm=config['policy']['max_grad_norm'],
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=device,
            tensorboard_log=tensorboard_log
        )
        
        logger.info(f"MetaPolicy initialized on {self.model.device}")
        
    def learn(self, total_timesteps: int, callback=None):
        """Train the policy."""
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        
    def save(self, path: str):
        """Save policy."""
        self.model.save(path)
        
    def load(self, path: str):
        """Load policy."""
        self.model = RecurrentPPO.load(path, env=self.env)
        return self

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        """Predict action."""
        return self.model.predict(obs, state=state, episode_start=episode_start, deterministic=deterministic)
