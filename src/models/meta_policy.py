"""
Multi-Head Meta-RL Policy for UNIFIED System.
Implements Recurrent Policy Core with Risk and Regime auxiliary heads.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import logging

logger = logging.getLogger(__name__)

class MultiHeadRecurrentNetwork(nn.Module):
    """
    Custom Network for RecurrentPPO that includes:
    1. Shared Recurrent Core (LSTM/GRU)
    2. Action Head (Actor)
    3. Value Head (Critic)
    4. Risk Head (Auxiliary)
    5. Regime Head (Auxiliary)
    """
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_mlp_size: int = 256,
        risk_head_dim: int = 3,
        regime_head_dim: int = 4
    ):
        super().__init__()
        
        self.lstm_hidden_size = lstm_hidden_size
        
        # 1. Initial MLP processing (z_t + features -> MLP)
        self.shared_mlp = nn.Sequential(
            nn.Linear(feature_dim, shared_mlp_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 2. Recurrent Core
        self.lstm = nn.LSTM(
            input_size=shared_mlp_size,
            hidden_size=lstm_hidden_size,
            num_layers=n_lstm_layers,
            batch_first=False # SB3 RecurrentPPO expects [seq_len, batch, dim]
        )
        
        # 3. Policy (Action) Head
        self.action_net = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # 4. Value Head
        self.value_net = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 5. Risk Head (Auxiliary)
        self.risk_net = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, risk_head_dim) # [DD, Vol, VaR]
        )
        
        # 6. Regime Head (Auxiliary)
        self.regime_net = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, regime_head_dim)
        )

    def forward(
        self,
        features: torch.Tensor,
        lstm_states: Tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        Args:
            features: [seq_len * batch, feature_dim]
            lstm_states: (h, c)
            episode_starts: [seq_len * batch]
        """
        # Features are already flattened by SB3. We need to handle the recurrent part.
        # This custom network structure requires careful integration with SB3's ActorCriticPolicy.
        # For now, we'll keep the multi-head structure ready for a custom trainer.
        
        x = self.shared_mlp(features)
        
        # LSTM rollout logic would go here if not using SB3's built-in RecurrentPPO.
        # Since we want to leverage SB3, we might need to wrap these heads differently.
        
        return x # Placeholder

class UnifiedMetaPolicy(RecurrentActorCriticPolicy):
    """
    Custom Recurrent Policy for SB3 that adds Risk and Regime heads.
    """
    def __init__(self, *args, **kwargs):
        # Extract auxiliary dims from kwargs or use defaults
        self.risk_head_dim = kwargs.pop("risk_head_dim", 3)
        self.regime_head_dim = kwargs.pop("regime_head_dim", 4)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        """
        Override to add multi-head structure.
        In SB3 RecurrentPPO, the LSTM sits between features and heads.
        """
        super()._build_mlp_extractor()
        # The base class creates self.mlp_extractor, self.action_net, self.value_net
        # We add our auxiliary heads here, branching from the same latent as action/value
        
        # Recurrent policies in SB3 use self.lstm_output_dim
        self.risk_net = nn.Sequential(
            nn.Linear(self.lstm_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.risk_head_dim)
        )
        
        self.regime_net = nn.Sequential(
            nn.Linear(self.lstm_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.regime_head_dim)
        )

    def forward_auxiliary(
        self,
        obs: torch.Tensor,
        lstm_states: Tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get risk and regime predictions.
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf, lstm_states = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor, self.lstm_critic)
        
        # Usually latent_pi is used for auxiliary heads
        risk_pred = self.risk_net(latent_pi)
        regime_logits = self.regime_net(latent_pi)
        
        return risk_pred, regime_logits

class MetaPolicy:
    """
    Wrapper for Meta-RL Policy.
    Uses UnifiedMetaPolicy (Multi-Head Recurrent PPO).
    """
    
    def __init__(
        self,
        env,
        config: Dict,
        device: str = 'auto',
        tensorboard_log: Optional[str] = None
    ):
        self.config = config
        self.env = env
        
        policy_kwargs = {
            'net_arch': dict(pi=config['policy'].get('hidden_sizes', [128, 128]), 
                           vf=config['policy'].get('hidden_sizes', [128, 128])),
            'lstm_hidden_size': config['policy']['rnn_hidden_size'],
            'enable_critic_lstm': True,
            'risk_head_dim': config['policy'].get('risk_head_dim', 3),
            'regime_head_dim': config['policy'].get('regime_head_dim', 4)
        }
        
        self.model = RecurrentPPO(
            UnifiedMetaPolicy, # Use our custom multi-head policy
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
        
        logger.info(f"UnifiedMetaPolicy initialized on {self.model.device}")
        
    def learn(self, total_timesteps: int, callback=None):
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        
    def save(self, path: str):
        self.model.save(path)
        
    def load(self, path: str):
        self.model = RecurrentPPO.load(path, env=self.env)
        return self

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        return self.model.predict(obs, state=state, episode_start=episode_start, deterministic=deterministic)

    def predict_auxiliary(self, obs, state=None, episode_start=None):
        """
        Predict risk and regime alongside action.
        """
        # Ensure model is in eval mode
        self.model.policy.set_training_mode(False)
        
        # Prepare inputs
        obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
        
        # Handle LSTM states (Simplified for single inference)
        if state is None:
             state = (torch.zeros((1, 1, self.config['policy']['rnn_hidden_size'])).to(self.model.device),
                     torch.zeros((1, 1, self.config['policy']['rnn_hidden_size'])).to(self.model.device))
        
        episode_start = torch.tensor([episode_start or False]).to(self.model.device)
        
        with torch.no_grad():
            risk, regime = self.model.policy.forward_auxiliary(obs_tensor, state, episode_start)
            
        return risk.cpu().numpy(), regime.cpu().numpy()
