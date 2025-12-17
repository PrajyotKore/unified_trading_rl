"""
Domain Randomization for UNIFIED Trading System.

Implements domain randomization (DR) to improve sim-reality transfer.
Randomizes transaction costs and market dynamics beyond real ranges.
From paper Table 1 and Section 3.1.
"""

import torch
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DomainRandomizer:
    """
    Domain randomization for world model training.
    
    Randomizes 10+ parameters beyond real-world ranges to create
    robust policies that handle extreme market conditions.
    
    Parameters (from Table 1):
        - Volatility: Real 10-60%, Random 5-80%
        - Spread: Real 1-50 bps, Random 0.5-100 bps
        - Market impact: Real 0.01-0.1, Random 0.001-0.2
        - Slippage, volume shocks, correlation shifts
    """
    
    def __init__(
        self,
        config: Dict,
        curriculum: bool = True,
        device: str = 'cpu'
    ):
        """
        Initialize domain randomizer.
        
        Args:
            config: Configuration dictionary with randomization ranges
            curriculum: Use curriculum learning (gradual increase)
            device: Device for tensors
        """
        self.config = config
        self.curriculum = curriculum
        self.device = device
        
        # Curriculum parameters
        self.current_step = 0
        self.curriculum_steps = config.get('curriculum_steps', 10000)
        self.initial_range_mult = config.get('initial_range_multiplier', 1.0)
        self.final_range_mult = config.get('final_range_multiplier', 2.0)
        
        # Extract parameter ranges
        self.params = config.get('parameters', {})
        
        logger.info(f"DomainRandomizer initialized with {len(self.params)} parameters")
        logger.info(f"Curriculum: {curriculum}, Steps: {self.curriculum_steps}")
    
    def get_curriculum_progress(self) -> float:
        """Get current curriculum progress [0, 1]."""
        if not self.curriculum:
            return 1.0
        return min(1.0, self.current_step / self.curriculum_steps)
    
    def get_randomization_multiplier(self) -> float:
        """Get current randomization range multiplier."""
        progress = self.get_curriculum_progress()
        return self.initial_range_mult + progress * (self.final_range_mult - self.initial_range_mult)
    
    def randomize_volatility(self, base_vol: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
        """
        Randomize volatility.
        
        Args:
            base_vol: Base volatility [batch]
            batch_size: Batch size
            
        Returns:
            Randomized volatility [batch]
        """
        if 'volatility' not in self.params:
            return base_vol
        
        vol_config = self.params['volatility']
        mult = self.get_randomization_multiplier()
        
        # Sample from randomized range
        min_vol = vol_config['random_min'] / mult
        max_vol = vol_config['random_max'] * mult
        
        random_vol = torch.rand(batch_size, device=self.device) * (max_vol - min_vol) + min_vol
        
        return random_vol
    
    def randomize_spread(self, base_spread: Optional[torch.Tensor] = None, batch_size: int = 1) -> torch.Tensor:
        """
        Randomize bid-ask spread (in basis points).
        
        Args:
            base_spread: Base spread [batch] (optional)
            batch_size: Batch size
            
        Returns:
            Randomized spread in bps [batch]
        """
        if 'spread_bps' not in self.params:
            return base_spread if base_spread is not None else torch.ones(batch_size, device=self.device) * 5
        
        spread_config = self.params['spread_bps']
        mult = self.get_randomization_multiplier()
        
        min_spread = spread_config['random_min'] / mult
        max_spread = spread_config['random_max'] * mult
        
        random_spread = torch.rand(batch_size, device=self.device) * (max_spread - min_spread) + min_spread
        
        return random_spread
    
    def randomize_market_impact(self, base_impact: Optional[torch.Tensor] = None, batch_size: int = 1) -> torch.Tensor:
        """
        Randomize market impact coefficient.
        
        Market impact = impact_coef * sqrt(order_size) * volatility
        
        Args:
            base_impact: Base impact coefficient [batch]
            batch_size: Batch size
            
        Returns:
            Randomized impact coefficient [batch]
        """
        if 'market_impact' not in self.params:
            return base_impact if base_impact is not None else torch.ones(batch_size, device=self.device) * 0.05
        
        impact_config = self.params['market_impact']
        mult = self.get_randomization_multiplier()
        
        min_impact = impact_config['random_min'] / mult
        max_impact = impact_config['random_max'] * mult
        
        random_impact = torch.rand(batch_size, device=self.device) * (max_impact - min_impact) + min_impact
        
        return random_impact
    
    def randomize_slippage(self, base_slippage: Optional[torch.Tensor] = None, batch_size: int = 1) -> torch.Tensor:
        """
        Randomize slippage (in basis points).
        
        Args:
            base_slippage: Base slippage [batch]
            batch_size: Batch size
            
        Returns:
            Randomized slippage in bps [batch]
        """
        if 'slippage_bps' not in self.params:
            return base_slippage if base_slippage is not None else torch.ones(batch_size, device=self.device) * 5
        
        slippage_config = self.params['slippage_bps']
        mult = self.get_randomization_multiplier()
        
        min_slippage = slippage_config['random_min'] / mult
        max_slippage = slippage_config['random_max'] * mult
        
        random_slippage = torch.rand(batch_size, device=self.device) * (max_slippage - min_slippage) + min_slippage
        
        return random_slippage
    
    def apply_randomized_costs(
        self,
        returns: torch.Tensor,
        position_change: torch.Tensor,
        volatility: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply randomized transaction costs to returns.
        
        Args:
            returns: Raw returns [batch, seq_len] or [batch]
            position_change: Absolute position change [batch, seq_len] or [batch]
            volatility: Current volatility [batch, seq_len] or [batch] (optional)
            
        Returns:
            Returns after randomized costs [batch, seq_len] or [batch]
        """
        batch_size = returns.shape[0]
        
        # Get randomized parameters
        spread = self.randomize_spread(batch_size=batch_size)
        slippage = self.randomize_slippage(batch_size=batch_size)
        impact_coef = self.randomize_market_impact(batch_size=batch_size)
        
        # Reshape for broadcasting if needed
        if len(returns.shape) > 1:
            spread = spread.unsqueeze(-1)
            slippage = slippage.unsqueeze(-1)
            impact_coef = impact_coef.unsqueeze(-1)
        
        # Calculate total costs (in decimal, not bps)
        spread_cost = spread / 10000  # bps to decimal
        slippage_cost = slippage / 10000
        
        # Market impact (proportional to sqrt(position_change) * volatility)
        if volatility is not None:
            market_impact = impact_coef * torch.sqrt(torch.abs(position_change) + 1e-8) * volatility
        else:
            market_impact = impact_coef * torch.sqrt(torch.abs(position_change) + 1e-8) * 0.02  # assume 2% vol
        
        # Total cost applied when position changes
        total_cost = (spread_cost + slippage_cost + market_impact) * torch.abs(position_change)
        
        # Apply costs (reduce returns)
        returns_after_costs = returns - total_cost
        
        return returns_after_costs
    
    def randomize_trajectory(
        self,
        trajectory: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply domain randomization to an entire trajectory.
        
        Args:
            trajectory: Dictionary with returns, volatilities, volumes, etc.
            
        Returns:
            Randomized trajectory
        """
        randomized = trajectory.copy()
        batch_size = trajectory['returns'].shape[0]
        
        # Randomize volatility
        if 'volatilities' in trajectory:
            base_vol = trajectory['volatilities']
            random_mult = self.randomize_volatility(base_vol, batch_size)
            if len(random_mult.shape) == 1 and len(base_vol.shape) > 1:
                random_mult = random_mult.unsqueeze(-1)
            randomized['volatilities'] = base_vol * (random_mult / base_vol.mean(dim=-1, keepdim=True))
        
        # Apply randomized costs to returns
        if 'returns' in trajectory and 'position_changes' in trajectory:
            randomized['returns'] = self.apply_randomized_costs(
                trajectory['returns'],
                trajectory['position_changes'],
                trajectory.get('volatilities')
            )
        
        return randomized
    
    def step(self):
        """Increment curriculum step."""
        self.current_step += 1
    
    def get_current_ranges(self) -> Dict[str, Dict[str, float]]:
        """Get current randomization ranges (for logging/monitoring)."""
        mult = self.get_randomization_multiplier()
        ranges = {}
        
        for param_name, param_config in self.params.items():
            ranges[param_name] = {
                'min': param_config['random_min'] / mult,
                'max': param_config['random_max'] * mult,
                'real_min': param_config['real_min'],
                'real_max': param_config['real_max']
            }
        
        return ranges


class SyntheticDataGenerator:
    """
    Generate synthetic training data using world model + domain randomization.
    Used for Stage 2 (offline meta-RL) in training pipeline.
    """
    
    def __init__(
        self,
        world_model: torch.nn.Module,
        domain_randomizer: DomainRandomizer,
        device: str = 'cpu'
    ):
        """
        Initialize synthetic data generator.
        
        Args:
            world_model: Trained world model
            domain_randomizer: Domain randomization module
            device: Device for computation
        """
        self.world_model = world_model
        self.domain_randomizer = domain_randomizer
        self.device = device
    
    def generate_episode(
        self,
        initial_state: Dict[str, torch.Tensor],
        episode_length: int,
        apply_randomization: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Generate synthetic episode.
        
        Args:
            initial_state: Initial OHLCV and features
            episode_length: Number of steps
            apply_randomization: Apply domain randomization
            
        Returns:
            Synthetic episode trajectory
        """
        self.world_model.eval()
        
        # Generate trajectory using world model
        trajectory = self.world_model.sample_trajectory(
            initial_ohlcv=initial_state['ohlcv'],
            initial_features=initial_state['features'],
            num_steps=episode_length,
            randomize=apply_randomization
        )
        
        # Apply additional domain randomization if enabled
        if apply_randomization:
            # Add position_changes for cost calculation (random for now)
            trajectory['position_changes'] = torch.rand_like(trajectory['returns']) * 0.2
            trajectory = self.domain_randomizer.randomize_trajectory(trajectory)
        
        return trajectory
    
    def generate_batch(
        self,
        num_episodes: int,
        episode_length: int,
        real_data_sampler: callable
    ) -> Dict[str, torch.Tensor]:
        """
        Generate batch of synthetic episodes.
        
        Args:
            num_episodes: Number of episodes
            episode_length: Steps per episode
            real_data_sampler: Function to sample initial states from real data
            
        Returns:
            Batch of episodes
        """
        episodes = []
        
        for _ in range(num_episodes):
            initial_state = real_data_sampler()
            episode = self.generate_episode(initial_state, episode_length)
            episodes.append(episode)
        
        # Stack episodes
        batch = {}
        for key in episodes[0].keys():
            batch[key] = torch.stack([ep[key] for ep in episodes], dim=0)
        
        return batch
