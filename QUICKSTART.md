# UNIFIED Trading System - Quick Start Guide

## 🚀 Getting Started

### Step 1: Set Up Environment

```bash
# Navigate to project directory
cd /Users/prajyotkore/.gemini/antigravity/scratch/unified_trading

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Configure Kaggle API

The dataset needs to be downloaded from Kaggle. Set up your Kaggle API credentials:

```bash
# Create Kaggle directory
mkdir -p ~/.kaggle

# Place your kaggle.json file in ~/.kaggle/
# You can download kaggle.json from: https://www.kaggle.com/settings
# (Go to Account -> API -> Create New Token)

# Set permissions
chmod 600 ~/.kaggle/kaggle.json
```

### Step 3: Download and Prepare Data

```bash
# Download NIFTY 50 minute data and compute features
python scripts/download_data.py
```

This script will:
- Download data from Kaggle (debashis74017/nifty-50-minute-data)
- Preprocess and clean the data
- Compute 40+ technical indicators
- Label market regimes (bull, bear, crash, sideways)
- Create train/val/test splits
- Generate walk-forward validation windows

**Expected output:**
- Processed data in `data/processed/`
- Log files in `logs/`
- Summary statistics

### Step 4: Verify Installation

Create a simple test script to verify everything works:

```bash
# Create test script
cat > test_installation.py << 'EOF'
import torch
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, 'src')

from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from utils.utils import get_device

print("✓ Imports successful")
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ Device: {get_device()}")

# Check if processed data exists
processed_path = Path('data/processed/nifty50_processed.parquet')
if processed_path.exists():
    df = pd.read_parquet(processed_path)
    print(f"✓ Loaded {len(df):,} samples")
    print(f"✓ Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"✓ Features: {len(df.columns)} columns")
else:
    print("⚠ Run 'python scripts/download_data.py' first")

print("\n✓ Installation verified!")
EOF

python test_installation.py
```

## 📊 What's Next?

After data preparation is complete, you can proceed with training:

### Option A: Train Individual Components (Recommended for Learning)

1. **Train World Model** (Stage 1):
   ```bash
   python scripts/train_world_model.py
   ```
   - Learns market dynamics
   - Duration: ~2-3 days on GPU
   - Output: Checkpoints in `models/checkpoints/`

2. **Train Meta-RL Policy** (Stages 2-5):
   ```bash
   python scripts/train_meta_rl.py
   ```
   - Trains policy with meta-learning
   - Duration: ~7-10 days on GPU
   - Output: Trained policy checkpoints

3. **Evaluate** (Stage 6):
   ```bash
   python scripts/evaluate.py
   ```
   - Walk-forward validation
   - Generates performance reports

### Option B: Full Pipeline (Coming Soon)

```bash
# End-to-end training pipeline
python scripts/train_full_pipeline.py
```

## 📁 Project Structure

```
unified_trading/
├── config/
│   └── config.yaml              # ✓ Hyperparameters
├── src/
│   ├── data/
│   │   ├── data_loader.py      # ✓ Data loading
│   │   ├── feature_engineering.py  # ✓ 40+ indicators
│   │   └── regime_labeling.py  # ✓ Regime detection
│   ├── models/                  # ⏳ World model, policy (next)
│   ├── environments/            # ⏳ Trading env (next)
│   ├── training/                # ⏳ Training pipeline (next)
│   └── utils/
│       └── utils.py            # ✓ Utilities
├── scripts/
│   └── download_data.py        # ✓ Data preparation
├── data/
│   ├── raw/                    # Downloaded CSV
│   └── processed/              # ⏳ Processed parquet (after Step 3)
└── requirements.txt            # ✓ Dependencies

✓ = Completed
⏳ = In Progress / Next Steps
```

## 🔧 Configuration

Edit `config/config.yaml` to customize:

- **Data parameters**: Sequence length, prediction horizon
- **Model architecture**: Hidden sizes, number of layers
- **Training**: Learning rates, batch sizes, epochs
- **Domain randomization**: Volatility/spread/impact ranges
- **Validation**: Number of walk-forward windows

## 🐛 Troubleshooting

### Issue: Kaggle API not working
```bash
# Check if kaggle.json exists
ls -la ~/.kaggle/kaggle.json

# Verify credentials
kaggle datasets list
```

### Issue: Missing dependencies
```bash
# Install specific packages
pip install torch pandas ta scikit-learn

# For Apple Silicon Macs
pip install torch torchvision torchaudio
```

### Issue: Out of memory
Edit `config/config.yaml`:
```yaml
world_model:
  batch_size: 64  # Reduce from 128
  
training:
  num_workers: 2  # Reduce from 4
```

## 📖 Documentation

- **Paper**: See your LaTeX document for architecture details
- **Config**: All hyperparameters in `config/config.yaml`
- **Logs**: Check `logs/` for detailed execution logs

## 🎯 Current Status

**Phase 1: Data Pipeline** ✓ COMPLETE
- [x] Project structure
- [x] Dependencies
- [x] Data loader
- [x] Feature engineering (40+ indicators)
- [x] Regime labeling
- [x] Train/test splits

**Next: Phase 2 - World Model**
- [ ] Implement Conv1D + BiLSTM architecture
- [ ] Multi-head predictions
- [ ] Domain randomization
- [ ] Training loop

## 💡 Tips

1. **Start Small**: Test on a subset of data first (e.g., 1 year)
2. **Monitor GPU**: Use `nvidia-smi` to check GPU usage
3. **TensorBoard**: Track training with `tensorboard --logdir logs/`
4. **Checkpoints**: Models are saved every 10K steps by default

## 📧 Need Help?

Check logs in `logs/` for detailed error messages.

---

**Ready to train? Run `python scripts/download_data.py` to get started!**
