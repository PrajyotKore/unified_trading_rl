"""
Training script for world model (Stage 1 of pipeline).

Supervised pre-training of world model on historical data.
From paper Section 5: Stage 1 training.
"""

import os
import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from models.world_model import WorldModel, WorldModelLoss, TransformerWorldModel
from data.dataset import TimeSeriesDataset
from utils.utils import (
    load_config, setup_logging, set_seed, get_device,
    Checkpointer, count_parameters
)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_price_loss = 0
    total_volume_loss = 0
    total_regime_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        # Move to device
        ohlcv = inputs['ohlcv'].to(device)
        features = inputs['features'].to(device)
        
        target_returns = targets['returns'].to(device)
        target_volumes = targets['volumes'].to(device)
        target_regimes = targets['regimes'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(ohlcv, features)
        
        # Prepare targets for loss
        loss_targets = {
            'returns': target_returns,
            'volumes': target_volumes,
            'regimes': target_regimes
        }
        
        # Compute loss
        losses = criterion(outputs, loss_targets)
        loss = losses['total']
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_price_loss += losses['price'].item()
        total_volume_loss += losses['volume'].item()
        total_regime_loss += losses['regime'].item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'price': f"{losses['price'].item():.4f}",
            'regime': f"{losses['regime'].item():.4f}"
        })
    
    return {
        'loss': total_loss / num_batches,
        'price_loss': total_price_loss / num_batches,
        'volume_loss': total_volume_loss / num_batches,
        'regime_loss': total_regime_loss / num_batches
    }


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> dict:
    """Validate for one epoch."""
    model.eval()
    
    total_loss = 0
    total_price_loss = 0
    total_volume_loss = 0
    total_regime_loss = 0
    num_batches = 0
    
    # For regime accuracy
    correct_regimes = 0
    total_regimes = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    
    with torch.no_grad():
        for inputs, targets in pbar:
            # Move to device
            ohlcv = inputs['ohlcv'].to(device)
            features = inputs['features'].to(device)
            
            target_returns = targets['returns'].to(device)
            target_volumes = targets['volumes'].to(device)
            target_regimes = targets['regimes'].to(device)
            
            # Forward pass
            outputs = model(ohlcv, features)
            
            # Prepare targets
            loss_targets = {
                'returns': target_returns,
                'volumes': target_volumes,
                'regimes': target_regimes
            }
            
            # Compute loss
            losses = criterion(outputs, loss_targets)
            
            # Accumulate losses
            total_loss += losses['total'].item()
            total_price_loss += losses['price'].item()
            total_volume_loss += losses['volume'].item()
            total_regime_loss += losses['regime'].item()
            num_batches += 1
            
            # Regime accuracy
            # Use last step prediction vs first step target
            regime_preds = torch.argmax(outputs['regime_logits'][:, -1, :], dim=-1)  # [batch]
            target = target_regimes[:, 0].long()  # [batch]
            
            valid_mask = target >= 0
            if valid_mask.any():
                correct_regimes += (regime_preds[valid_mask] == target[valid_mask]).sum().item()
                total_regimes += valid_mask.sum().item()
            
            pbar.set_postfix({'loss': f"{losses['total'].item():.4f}"})
    
    regime_accuracy = correct_regimes / total_regimes if total_regimes > 0 else 0.0
    
    return {
        'loss': total_loss / num_batches,
        'price_loss': total_price_loss / num_batches,
        'volume_loss': total_volume_loss / num_batches,
        'regime_loss': total_regime_loss / num_batches,
        'regime_accuracy': regime_accuracy
    }


def main():
    parser = argparse.ArgumentParser(description='Train world model')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Setup
    project_root = Path(__file__).parent.parent
    config = load_config(project_root / args.config)
    
    logger = setup_logging(
        log_dir=project_root / config['logging']['log_dir'],
        name='train_world_model',
        level=config['logging']['verbosity']
    )
    
    logger.info("=" * 80)
    logger.info("UNIFIED Trading System - Stage 1: World Model Training")
    logger.info("=" * 80)
    
    # Set seed
    set_seed(config['compute']['seed'])
    device = get_device(config['compute']['device'])
    logger.info(f"Device: {device}")
    
    # Load datasets
    logger.info("\nLoading datasets...")
    processed_dir = project_root / config['data']['processed_path']
    
    train_dataset = TimeSeriesDataset(
        data_path=str(processed_dir / 'train.parquet'),
        sequence_length=config['data']['sequence_length'],
        prediction_horizon=config['data']['prediction_horizon'],
        normalize=True
    )
    
    val_dataset = TimeSeriesDataset(
        data_path=str(processed_dir / 'val.parquet'),
        sequence_length=config['data']['sequence_length'],
        prediction_horizon=config['data']['prediction_horizon'],
        normalize=True
    )
    
    # Get dimensions
    dims = train_dataset.get_sample_dimensions()
    logger.info(f"Data dimensions: {dims}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['world_model']['batch_size'],
        shuffle=True,
        num_workers=config['compute']['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['world_model']['batch_size'],
        shuffle=False,
        num_workers=config['compute']['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    logger.info("\nInitializing model (TransformerWorldModel)...")
    # We use config['world_model']['lstm_hidden_size'] as d_model for now
    d_model = config['world_model'].get('d_model', 128) 
    
    model = TransformerWorldModel(
        input_channels=dims['ohlcv_channels'],
        feature_dim=dims['feature_dim'],
        d_model=d_model,
        nhead=4,
        num_layers=2,
        dim_feedforward=d_model * 4,
        dropout=config['world_model']['dropout'],
        num_regimes=config['world_model']['num_regimes'],
        sequence_length=config['data']['sequence_length']
    ).to(device)
    
    logger.info(f"Model parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    criterion = WorldModelLoss(regime_weight=1.0)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['world_model']['learning_rate'],
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Checkpointer
    checkpointer = Checkpointer(
        checkpoint_dir=str(project_root / config['logging']['checkpoint_dir'] / 'world_model'),
        keep_n_best=3
    )
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(project_root / 'logs' / 'tensorboard' / 'world_model'))
    
    # Resume if requested
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = checkpointer.load(model, optimizer, filename=args.resume)
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('metric', float('inf'))
    
    # Training loop
    logger.info("\nStarting training...")
    logger.info(f"Epochs: {config['world_model']['num_epochs']}")
    logger.info(f"Batch size: {config['world_model']['batch_size']}")
    logger.info(f"Learning rate: {config['world_model']['learning_rate']}")
    
    for epoch in range(start_epoch, config['world_model']['num_epochs']):
        logger.info(f"\nEpoch {epoch+1}/{config['world_model']['num_epochs']}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch+1)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch+1)
        
        # Log metrics
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                   f"Price: {train_metrics['price_loss']:.4f}, "
                   f"Regime: {train_metrics['regime_loss']:.4f}")
        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                   f"Price: {val_metrics['price_loss']:.4f}, "
                   f"Regime: {val_metrics['regime_loss']:.4f}, "
                   f"Regime Acc: {val_metrics['regime_accuracy']:.3f}")
        
        # TensorBoard
        for key, value in train_metrics.items():
            writer.add_scalar(f'train/{key}', value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f'val/{key}', value, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Save checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            logger.info(f"✓ New best validation loss: {best_val_loss:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        checkpointer.save(
            model, optimizer, epoch,
            metric=-val_metrics['loss'],  # Negative for "higher is better" in checkpointer
            metadata={
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'regime_accuracy': val_metrics['regime_accuracy']
            },
            filename=f"checkpoint_epoch_{epoch}.pt" if not is_best else "best_model.pt"
        )
        
        # Early stopping
        if epochs_no_improve >= config['world_model']['early_stopping_patience']:
            logger.info(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to: {checkpointer.checkpoint_dir}")
    logger.info("=" * 80)
    
    writer.close()


if __name__ == '__main__':
    main()
