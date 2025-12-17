"""
World Model for UNIFIED Trading System.

Learns market dynamics from historical data using Conv1D encoder + BiLSTM.
Provides multi-head predictions: price, volume, and regime classification.
Implements Equations 2-4 from the paper.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class WorldModel(nn.Module):
    """
    World model for learning market dynamics.
    
    Architecture (from paper Section 3.1):
        z_t = BiLSTM(Conv1D(OHLCV_{t-T:t} + features_t))
        
    Multi-head predictions:
        - Price head: (return_mean, return_std)
        - Volume head: (volume_mean, volume_std)
        - Regime head: 4-class classification (bull, bear, crash, sideways)
    """
    
    def __init__(
        self,
        input_channels: int = 5,  # OHLCV
        feature_dim: int = 45,     # Technical indicators
        conv_channels: list = [64, 128],
        conv_kernel_sizes: list = [5, 3],
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
        num_regimes: int = 4
    ):
        """
        Initialize world model.
        
        Args:
            input_channels: Number of OHLCV channels (5)
            feature_dim: Dimension of engineered features
            conv_channels: List of conv layer output channels
            conv_kernel_sizes: List of conv kernel sizes
            lstm_hidden_size: LSTM hidden dimension
            lstm_num_layers: Number of LSTM layers
            bidirectional: Use bidirectional LSTM
            dropout: Dropout rate
            num_regimes: Number of market regimes
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.bidirectional = bidirectional
        self.num_regimes = num_regimes
        
        # Conv1D encoder for OHLCV sequences
        conv_layers = []
        in_channels = input_channels
        
        for out_channels, kernel_size in zip(conv_channels, conv_kernel_sizes):
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.conv_encoder = nn.Sequential(*conv_layers)
        
        # BiLSTM for temporal dynamics
        # Input: conv outputs + technical features concatenated
        lstm_input_size = conv_channels[-1] + feature_dim
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_num_layers > 1 else 0
        )
        
        # Output dimension after LSTM
        lstm_output_size = lstm_hidden_size * (2 if bidirectional else 1)
        
        # Multi-head prediction layers
        # Price head: predict (return_mean, return_std)
        self.price_head = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # mean, std
        )
        
        # Volume head: predict (volume_mean, volume_std)
        self.volume_head = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # mean, std
        )
        
        # Regime classification head: 4 regimes
        self.regime_head = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_regimes)
        )
        
        logger.info(f"WorldModel initialized with {self._count_parameters():,} parameters")
    
    def forward(
        self,
        ohlcv: torch.Tensor,
        features: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through world model.
        
        Args:
            ohlcv: OHLCV sequences [batch, channels=5, seq_len]
            features: Technical features [batch, seq_len, feature_dim]
            hidden: Optional LSTM hidden state (h, c)
            
        Returns:
            Dictionary with predictions:
                - latent: Latent state z_t [batch, seq_len, lstm_output_size]
                - price_pred: (return_mean, return_std) [batch, seq_len, 2]
                - volume_pred: (volume_mean, volume_std) [batch, seq_len, 2]
                - regime_logits: Regime classification logits [batch, seq_len, num_regimes]
                - hidden: LSTM hidden state for next step
        """
        batch_size, channels, seq_len = ohlcv.shape
        
        # 1. Conv1D encoding of OHLCV
        # Input: [batch, 5, seq_len]
        conv_out = self.conv_encoder(ohlcv)  # [batch, conv_channels[-1], seq_len]
        
        # Transpose for LSTM: [batch, seq_len, conv_channels[-1]]
        conv_out = conv_out.transpose(1, 2)
        
        # 2. Concatenate with technical features
        # features: [batch, seq_len, feature_dim]
        lstm_input = torch.cat([conv_out, features], dim=-1)  # [batch, seq_len, conv+feature]
        
        # 3. BiLSTM processing
        if hidden is not None:
            lstm_out, hidden_new = self.lstm(lstm_input, hidden)
        else:
            lstm_out, hidden_new = self.lstm(lstm_input)
        
        # lstm_out: [batch, seq_len, lstm_output_size]
        
        # 4. Multi-head predictions
        price_pred = self.price_head(lstm_out)      # [batch, seq_len, 2]
        volume_pred = self.volume_head(lstm_out)    # [batch, seq_len, 2]
        regime_logits = self.regime_head(lstm_out)  # [batch, seq_len, num_regimes]
        
        # Apply softplus to std predictions to ensure positivity
        price_pred = torch.cat([
            price_pred[..., :1],  # mean (no constraint)
            nn.functional.softplus(price_pred[..., 1:]) + 1e-6  # std (positive)
        ], dim=-1)
        
        volume_pred = torch.cat([
            volume_pred[..., :1],
            nn.functional.softplus(volume_pred[..., 1:]) + 1e-6
        ], dim=-1)
        
        return {
            'latent': lstm_out,
            'price_pred': price_pred,
            'volume_pred': volume_pred,
            'regime_logits': regime_logits,
            'hidden': hidden_new
        }
    
    def predict_next_step(
        self,
        ohlcv: torch.Tensor,
        features: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict next step only (for rollout during training).
        
        Args:
            ohlcv: Current OHLCV sequence [batch, 5, seq_len]
            features: Current features [batch, seq_len, feature_dim]
            hidden: LSTM hidden state
            
        Returns:
            Predictions for next timestep
        """
        outputs = self.forward(ohlcv, features, hidden)
        
        # Extract last timestep predictions
        return {
            'latent': outputs['latent'][:, -1],  # [batch, lstm_output_size]
            'price_pred': outputs['price_pred'][:, -1],  # [batch, 2]
            'volume_pred': outputs['volume_pred'][:, -1],  # [batch, 2]
            'regime_logits': outputs['regime_logits'][:, -1],  # [batch, num_regimes]
            'hidden': outputs['hidden']
        }
    
    def sample_trajectory(
        self,
        initial_ohlcv: torch.Tensor,
        initial_features: torch.Tensor,
        num_steps: int,
        randomize: bool = False,
        noise_scale: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Generate synthetic trajectory by rolling out world model.
        Used for domain randomization (Section 3.1).
        
        Args:
            initial_ohlcv: Starting OHLCV [batch, 5, seq_len]
            initial_features: Starting features [batch, seq_len, feature_dim]
            num_steps: Number of steps to generate
            randomize: Apply domain randomization noise
            noise_scale: Scale of noise if randomizing
            
        Returns:
            Generated trajectory
        """
        self.eval()
        
        trajectories = {
            'returns': [],
            'volatilities': [],
            'volumes': [],
            'regimes': []
        }
        
        ohlcv = initial_ohlcv
        features = initial_features
        hidden = None
        
        with torch.no_grad():
            for _ in range(num_steps):
                pred = self.predict_next_step(ohlcv, features, hidden)
                
                # Sample from predicted distributions
                return_mean, return_std = pred['price_pred'][:, 0], pred['price_pred'][:, 1]
                volume_mean, volume_std = pred['volume_pred'][:, 0], pred['volume_pred'][:, 1]
                
                if randomize:
                    return_std = return_std * (1 + noise_scale * torch.randn_like(return_std))
                    volume_std = volume_std * (1 + noise_scale * torch.randn_like(volume_std))
                
                sampled_return = torch.normal(return_mean, return_std)
                sampled_volume = torch.normal(volume_mean, volume_std)
                regime_probs = torch.softmax(pred['regime_logits'], dim=-1)
                
                trajectories['returns'].append(sampled_return)
                trajectories['volatilities'].append(return_std)
                trajectories['volumes'].append(sampled_volume)
                trajectories['regimes'].append(regime_probs)
                
                hidden = pred['hidden']
                
                # For next step, we'd need to update ohlcv and features
                # This is simplified - in practice, integrate predicted returns
        
        # Stack trajectories
        for key in trajectories:
            trajectories[key] = torch.stack(trajectories[key], dim=1)  # [batch, num_steps, ...]
        
        return trajectories
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_latent_representation(
        self,
        ohlcv: torch.Tensor,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract latent representation z_t for policy input.
        
        Args:
            ohlcv: OHLCV sequences [batch, 5, seq_len]
            features: Technical features [batch, seq_len, feature_dim]
            
        Returns:
            Latent state [batch, seq_len, lstm_output_size]
        """
        outputs = self.forward(ohlcv, features)
        return outputs['latent']


class WorldModelLoss(nn.Module):
    """
    Combined loss for world model training.
    
    L_total = L_price + L_volume + λ_regime * L_regime
    """
    
    def __init__(
        self,
        regime_weight: float = 1.0,
        price_weight: float = 1000.0,
        volume_weight: float = 1000.0,
        use_mse: bool = True
    ):
        """
        Initialize loss.
        
        Args:
            regime_weight: Weight for regime classification loss
            price_weight: Weight for price prediction loss
            volume_weight: Weight for volume prediction loss
            use_mse: Use MSE for regression (vs negative log-likelihood)
        """
        super().__init__()
        self.regime_weight = regime_weight
        self.price_weight = price_weight
        self.volume_weight = volume_weight
        self.use_mse = use_mse
        
        self.regime_criterion = nn.CrossEntropyLoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
                - returns: [batch, seq_len]
                - volumes: [batch, seq_len]
                - regimes: [batch, seq_len] (class indices)
            
        Returns:
            Dictionary with total loss and components
        """
        # Price prediction loss (returns)
        # Use last step of prediction to predict first step of target (next step)
        if self.use_mse:
            price_loss = nn.functional.mse_loss(
                predictions['price_pred'][:, -1, 0],  # predicted mean (last step)
                targets['returns'][:, 0]              # target (first step ahead)
            )
        else:
            # Negative log likelihood
            pred_mean = predictions['price_pred'][:, -1, 0]
            pred_std = predictions['price_pred'][:, -1, 1]
            target = targets['returns'][:, 0]
            price_loss = -self._gaussian_nll(target, pred_mean, pred_std).mean()
        
        # Volume prediction loss
        if self.use_mse:
            volume_loss = nn.functional.mse_loss(
                predictions['volume_pred'][:, -1, 0],
                targets['volumes'][:, 0]
            )
        else:
            pred_mean = predictions['volume_pred'][:, -1, 0]
            pred_std = predictions['volume_pred'][:, -1, 1]
            target = targets['volumes'][:, 0]
            volume_loss = -self._gaussian_nll(target, pred_mean, pred_std).mean()
        
        # Regime classification loss
        # Use last step prediction
        regime_logits = predictions['regime_logits'][:, -1, :]  # [batch, num_classes]
        regime_targets = targets['regimes'][:, 0].long()       # [batch]
        
        # Filter out invalid regime labels (-1)
        valid_mask = regime_targets >= 0
        if valid_mask.any():
            regime_loss = self.regime_criterion(
                regime_logits[valid_mask],
                regime_targets[valid_mask]
            )
        else:
            regime_loss = torch.tensor(0.0, device=regime_logits.device)
        
        # Total loss
        # Total loss
        total_loss = (self.price_weight * price_loss) + \
                    (self.volume_weight * volume_loss) + \
                    (self.regime_weight * regime_loss)
        
        return {
            'total': total_loss,
            'price': price_loss,
            'volume': volume_loss,
            'regime': regime_loss
        }
    
    def _gaussian_nll(self, x, mean, std):
        """Gaussian negative log-likelihood."""
        return 0.5 * torch.log(2 * np.pi * std**2) + 0.5 * ((x - mean) / std)**2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerWorldModel(nn.Module):
    """
    Transformer-based World Model.
    Replaces Conv1D+LSTM with a pure Transformer Encoder architecture.
    """
    def __init__(
        self,
        input_channels: int = 5,
        feature_dim: int = 45,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_regimes: int = 4,
        sequence_length: int = 50
    ):
        super().__init__()
        
        self.d_model = d_model
        self.lstm_hidden_size = d_model # API compatibility with TradingEnv (latent dim)
        self.bidirectional = False # Transformers are bidirectional by nature (contextual)
        
        # Input embedding: Project (OHLCV + Features) -> d_model
        input_dim = input_channels + feature_dim
        self.input_projection = nn.Linear(input_dim, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length + 100)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output Heads
        self.price_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # mean, std
        )
        
        self.volume_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_regimes)
        )
        
        logger.info(f"TransformerWorldModel initialized with {self._count_parameters():,} parameters")
        
    def _count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        ohlcv: torch.Tensor,
        features: torch.Tensor,
        hidden = None # Argument kept for API compatibility, unused
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
            ohlcv: [batch, 5, seq_len] -> We need to transpose to [batch, seq, 5]
            features: [batch, seq, feature_dim]
        """
        # Prepare Input
        # ohlcv: [batch, 5, seq] -> [batch, seq, 5]
        ohlcv_t = ohlcv.transpose(1, 2)
        
        # Concatenate: [batch, seq, 5 + feature_dim]
        x = torch.cat([ohlcv_t, features], dim=-1)
        
        # Embed
        x = self.input_projection(x) # [batch, seq, d_model]
        x = self.pos_encoder(x)
        
        # Transform
        # Transformer mask could be added here for causal attention if we were doing autoregressive generation.
        # But for "understanding the window", full attention (bidirectional) is fine/better.
        latent = self.transformer_encoder(x) # [batch, seq, d_model]
        
        # Heads
        price_pred = self.price_head(latent)
        volume_pred = self.volume_head(latent)
        regime_logits = self.regime_head(latent)
        
        # Softplus for std
        price_pred = torch.cat([
            price_pred[..., :1],
            nn.functional.softplus(price_pred[..., 1:]) + 1e-6
        ], dim=-1)
        
        volume_pred = torch.cat([
            volume_pred[..., :1],
            nn.functional.softplus(volume_pred[..., 1:]) + 1e-6
        ], dim=-1)
        
        return {
            'latent': latent,
            'price_pred': price_pred,
            'volume_pred': volume_pred,
            'regime_logits': regime_logits,
            'hidden': None # No hidden state in Transformer
        }
        
    def get_latent_representation(self, ohlcv, features):
        outputs = self.forward(ohlcv, features)
        return outputs['latent']
