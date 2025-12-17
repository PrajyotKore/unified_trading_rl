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

1. **World Model**: Conv1D + BiLSTM with multi-head predictions (price, volume, regime)
2. **Meta-RL Policy**: GRU-based policy with MAML for fast adaptation
3. **Domain Randomization**: Curriculum-based cost/volatility randomization
4. **Trading Environment**: Realistic transaction costs with portfolio management
5. **Composite Reward**: Risk-adjusted P&L with drawdown/volatility penalties

## 📁 Project Structure

```
unified_trading/
├── config/
│   └── config.yaml              # Hyperparameters
├── src/
│   ├── data/                    # Data loading & feature engineering
│   ├── models/                  # World model, policy, domain randomization
│   ├── environments/            # Trading env & portfolio manager
│   ├── training/                # 6-stage training pipeline
│   ├── evaluation/              # Backtesting & walk-forward validation
│   └── utils/                   # Utilities
├── tests/                       # Unit & integration tests
├── scripts/                     # Executable scripts
├── notebooks/                   # EDA & analysis
├── data/                        # Raw & processed data
├── models/checkpoints/          # Saved models
└── logs/                        # TensorBoard logs
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

```bash
# Configure Kaggle API credentials
mkdir -p ~/.kaggle
# Place your kaggle.json in ~/.kaggle/

# Download NIFTY 50 minute data
python scripts/download_data.py
```

### 3. Train World Model

```bash
# Stage 1: Pre-train world model
python scripts/train_world_model.py --config config/config.yaml
```

### 4. Train Meta-RL Policy

```bash
# Stages 2-5: Meta-RL training pipeline
python scripts/train_meta_rl.py --config config/config.yaml
```

### 5. Evaluate

```bash
# Stage 6: Walk-forward validation
python scripts/evaluate.py --config config/config.yaml
```

## 📊 Training Pipeline (6 Stages)

| Stage | Description | Duration (GPU) | Output |
|-------|-------------|----------------|--------|
| 1 | World model pre-training | 2-3 days | Learned market dynamics |
| 2 | Offline meta-RL on synthetic data | 3-4 days | Sample-efficient policy |
| 3 | Online fine-tuning on real data | 2-3 days | Adapted to real market |
| 4 | Risk-aware multi-task training | 1-2 days | Risk-conscious trading |
| 5 | Continual domain randomization | 1-2 days | Robust to costs |
| 6 | Walk-forward validation | 1 day | Performance metrics |

**Total**: ~10-15 days on GPU (RTX 3090 / A100)

## 📈 Expected Results

Based on paper targets for NSE/BSE data (2015-2025):

| Metric | Target | Baseline RL |
|--------|--------|-------------|
| Sharpe Ratio | 1.5-2.0 | 1.2-1.5 |
| Max Drawdown | 12-15% | 20-35% |
| Regime Adaptation | 1-2 weeks | 6+ months |
| Sample Efficiency | 10-20x | 1x |
| Interpretability | 95%+ | 0% |

## 🔧 Configuration

Edit `config/config.yaml` to customize:
- Data parameters (sequence length, train/test splits)
- Model architecture (hidden sizes, layers)
- Training hyperparameters (learning rates, batch sizes)
- Domain randomization ranges
- Reward function weights

## 📝 Key Features

### World Model
- Multi-head architecture: price, volume, regime predictions
- BiLSTM for temporal dynamics
- Domain randomization for robustness

### Meta-RL Policy
- MAML-based fast adaptation
- Recurrent GRU core for non-stationarity
- Multi-head outputs: action, risk, regime

### Trading Environment
- Realistic transaction costs (spread, slippage, market impact)
- Portfolio management with position tracking
- Risk metrics (Sharpe, drawdown, VaR)

### Evaluation
- Walk-forward validation (5+ windows)
- Regime-specific analysis
- Bonferroni correction for multiple testing

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific component tests
pytest tests/test_world_model.py
pytest tests/test_trading_env.py
pytest tests/test_meta_policy.py
```

## 📚 Citation

```bibtex
@article{unified2025,
  title={UNIFIED: A Meta-Reinforcement Learning Architecture for Robust Financial Trading in Non-Stationary Markets},
  author={Architecture Research Collective},
  year={2025}
}
```

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

## ⚠️ Disclaimer

This is a research implementation. Past performance does not guarantee future results. Use at your own risk. Not financial advice.
