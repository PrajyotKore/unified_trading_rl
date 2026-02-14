# UNIFIED: Meta-Reinforcement Learning for Financial Trading

Implementation of the UNIFIED architecture for robust financial trading in non-stationary markets using NIFTY 50 minute-level data.

## 🎯 Overview

UNIFIED addresses six critical challenges in RL-based trading:
1. **Sample Efficiency**: 10-20x improvement via world models
2. **Non-Stationarity**: 1-2 week regime adaptation via meta-learning
3. **Interpretability**: 95%+ fidelity with multi-head outputs
4. **Risk Management**: Composite rewards with drawdown/volatility penalties
5. **Sim-Reality Gap**: 15-20% realistic cost modeling via domain randomization
6. **Overfitting Prevention**: Walk-forward validation with 5+ windows

**Target Performance**: Sharpe 1.5-2.0, Max Drawdown 12-15%

## 🏗️ Architecture

```
UNIFIED = World Model + Meta-RL + Multi-Head + Domain Randomization + Composite Rewards
```

### Components

1. **World Model**: **Transformer-based** (or Conv1D+BiLSTM) encoder with multi-head predictions (**price, volume, spread proxy, regime**).
2. **Meta-RL Policy**: Recurrent policy core with **MAML** for fast adaptation and auxiliary **Risk/Regime heads**.
3. **Domain Randomization**: 4-phase curriculum-based cost/volatility randomization (Easy to Extreme).
4. **Trading Environment**: Realistic transaction costs with portfolio management and soft risk constraints.
5. **Composite Reward**: Risk-aware objective: $R = P\&L - \lambda \sigma - \mu \text{max\_dd}$.

## 📁 Project Structure

```
unified_trading/
├── config/
│   └── config.yaml              # Hyperparameters
├── src/
│   ├── data/                    # Data loading, feature engineering, real-time connectors
│   ├── models/                  # Transformer World model, Multi-head policy, DR
│   ├── environments/            # Trading env with composite rewards
│   ├── api/                     # Model serving API
│   ├── training/                # MAML MetaTrainer and SB3 loops
│   ├── evaluation/              # Backtesting & walk-forward validation
│   └── utils/                   # Utilities
├── scripts/                     # Executable scripts
├── data/                        # Raw & processed data
├── models/checkpoints/          # Saved models
└── logs/                        # TensorBoard & execution logs
```

## 🚀 Quick Start (Step-by-Step)

### 1. Installation & Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Acquisition & Preprocessing
```bash
# Downloads, cleans, and labels regimes
python scripts/download_data.py
```

### 3. Stage 1: World Model Training
Train the Transformer-based market dynamics model:
```bash
python scripts/train_world_model.py --config config/config.yaml
```

### 4. Stages 2-5: Meta-RL Training Pipeline
Train the trading policy through offline meta-learning (**MAML**) and online fine-tuning:
```bash
# Start from Stage 2 (Offline Meta-RL with MAML)
python scripts/train_meta_rl.py --stage 2
```

### 5. Stage 6: Evaluation & Walk-Forward Validation
```bash
python scripts/evaluate.py --walk_forward
```

## 📊 Training Pipeline (6 Stages)

| Stage | Description | Duration (GPU) | Output |
|-------|-------------|----------------|--------|
| 1 | World model pre-training | 2-3 days | Learned market dynamics |
| 2 | Offline meta-RL on synthetic data | 3-4 days | MAML Meta-policy |
| 3 | Online fine-tuning on real data | 2-3 days | Adapted to real market |
| 4 | Risk-aware multi-task training | 1-2 days | Risk-conscious trading |
| 5 | Continual domain randomization | 1-2 days | Robust to costs |
| 6 | Walk-forward validation | 1 day | Performance metrics |

## 📈 Expected Results

| Metric | Target | Baseline RL |
|--------|--------|-------------|
| Sharpe Ratio | 1.5-2.0 | 1.2-1.5 |
| Max Drawdown | 12-15% | 20-35% |
| Regime Adaptation | 1-2 weeks | 6+ months |
| Interpretability | 95%+ | 0% |

## 📝 Key Features

### World Model
- Multi-head architecture: **price, volume, spread proxy**, regime predictions
- **Transformer Encoder** (Primary) or BiLSTM for temporal dynamics
- Domain randomization for bridging the sim-reality gap

### Meta-RL Policy
- **MAML-style** meta-learning for 1-2 week regime adaptation
- Recurrent policy core for handling non-stationarity
- Multi-head outputs: action, **risk (DD, Vol, VaR)**, and regime

### Trading Environment
- Realistic transaction costs and **soft risk constraints**
- **Composite Reward** function ($P\&L$ - penalties)

## 🧪 Testing
```bash
pytest tests/
```

## 📄 License
MIT License
