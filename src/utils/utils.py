"""
Utility functions for UNIFIED trading system.
"""

import os
import random
import numpy as np
import torch
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json


def set_seed(seed: int = 42, deterministic: bool = False):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
        deterministic: If True, use deterministic algorithms (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """
    Get PyTorch device.
    
    Args:
        device: Device type ("auto", "cpu", "cuda", "mps")
        
    Returns:
        torch.device
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save config
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def setup_logging(log_dir: str, name: str = "unified", level: str = "INFO"):
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory for log files
        name: Logger name
        level: Logging level
        
    Returns:
        Logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level))
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    
    # File handler
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'{name}.log')
    )
    file_handler.setLevel(getattr(logging, level))
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_format)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


class Checkpointer:
    """
    Model checkpointing utility.
    """
    
    def __init__(self, checkpoint_dir: str, keep_n_best: int = 3):
        """
        Initialize checkpointer.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_n_best: Number of best checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_n_best = keep_n_best
        self.best_scores = []
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metric: float,
        metadata: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None
    ):
        """
        Save checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metric: Metric value (for tracking best models)
            metadata: Additional metadata
            filename: Custom filename (if None, use epoch number)
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metric': metric,
            'metadata': metadata or {}
        }
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        # Track best checkpoints
        self.best_scores.append((metric, checkpoint_path))
        self.best_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Remove old checkpoints if exceeding limit
        if len(self.best_scores) > self.keep_n_best:
            _, old_path = self.best_scores.pop()
            if old_path.exists() and old_path != checkpoint_path:
                old_path.unlink()
    
    def load(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        filename: Optional[str] = None,
        load_best: bool = False
    ) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            filename: Checkpoint filename (if None, load latest)
            load_best: If True, load best checkpoint by metric
            
        Returns:
            Checkpoint dictionary
        """
        if load_best and self.best_scores:
            checkpoint_path = self.best_scores[0][1]
        elif filename:
            checkpoint_path = self.checkpoint_dir / filename
        else:
            # Load latest checkpoint
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
            if not checkpoints:
                raise FileNotFoundError("No checkpoints found")
            checkpoint_path = checkpoints[-1]
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num: float, precision: int = 2) -> str:
    """
    Format large numbers with K/M/B suffix.
    
    Args:
        num: Number to format
        precision: Decimal precision
        
    Returns:
        Formatted string
    """
    for unit in ['', 'K', 'M', 'B']:
        if abs(num) < 1000.0:
            return f"{num:.{precision}f}{unit}"
        num /= 1000.0
    return f"{num:.{precision}f}T"


def save_json(data: Dict[str, Any], filepath: str):
    """Save dictionary to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)
