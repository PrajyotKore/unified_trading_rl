# UNIFIED Trading System - Quick Start Guide

## 🚀 Getting Started

### Step 1: Set Up Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download and Prepare Data

```bash
# Download NIFTY 50 data and compute features
python scripts/download_data.py
```

This script handles preprocessing, feature engineering (40+ indicators), and regime labeling.

### Step 3: Train World Model (Stage 1)

```bash
python scripts/train_world_model.py
```
- Learns market dynamics using a **Transformer Encoder**.
- Outputs: Price, Volume, **Spread Proxy**, and Regime predictions.

### Step 4: Train Meta-RL Policy (Stages 2-5)

```bash
# Start Stage 2: Offline Meta-Learning (MAML) on Synthetic Data
python scripts/train_meta_rl.py --stage 2
```
- Trains the recurrent core to adapt to new regimes in 1-2 weeks.
- Integrates **Risk and Regime auxiliary heads** for interpretability.

### Step 5: Evaluate (Stage 6)

```bash
# Full walk-forward validation
python scripts/evaluate.py --walk_forward
```

## 📁 Key Components

- **Transformer World Model**: `src/models/world_model.py`
- **MAML Meta-Trainer**: `src/training/meta_trainer.py`
- **Multi-Head Policy**: `src/models/meta_policy.py`
- **Composite Reward Env**: `src/environments/trading_env.py`

## 🔧 Training Pipeline

1. **Stage 1**: Supervised pre-training of the market dynamics.
2. **Stage 2**: Offline Meta-RL on a randomized simulator.
3. **Stage 3**: Online fine-tuning on real historical data.
4. **Stage 4**: Risk-aware multi-task training (Risk Head optimization).
5. **Stage 5**: Robustness stress-testing via domain randomization.
6. **Stage 6**: Final walk-forward evaluation.

## 📈 Success Criteria

- Sharpe Ratio > 1.5
- Max Drawdown < 15%
- Regime Adaptation < 2 weeks
