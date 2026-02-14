from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import sys
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.utils import load_config, get_device
from models.meta_policy import MetaPolicy
from models.world_model import TransformerWorldModel
from data.feature_engineering import FeatureEngineer
from environments.trading_env import TradingEnv

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("unified_api")

app = FastAPI(title="UNIFIED Trading Strategy API")

# Global State
config = None
world_model = None
policy = None
feature_engineer = None
device = None

class MarketState(BaseModel):
    ohlcv_window: List[Dict[str, float]] # Last N minutes OHLCV
    current_position: float

class TradeAction(BaseModel):
    target_position: float
    confidence: float
    regime: str

@app.on_event("startup")
def startup_event():
    global config, world_model, policy, feature_engineer, device
    
    project_root = Path(__file__).parent.parent.parent
    config = load_config(project_root / 'config/config.yaml')
    device = get_device(config['compute'].get('device', 'cpu'))
    
    # Initialize components (Load best models)
    logger.info("Initializing UNIFIED components...")
    
    # 1. Feature Engineer
    feature_engineer = FeatureEngineer()
    
    # 2. World Model
    wm_path = project_root / config['logging']['checkpoint_dir'] / 'world_model/best_model.pt'
    if wm_path.exists():
        # Quick dummy init to get dimensions
        input_channels = 5
        feature_dim = config['features']['num_features']
        d_model = config['world_model'].get('d_model', 128)
        
        world_model = TransformerWorldModel(
            input_channels=input_channels,
            feature_dim=feature_dim,
            d_model=d_model,
            nhead=4,
            num_layers=2,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            num_regimes=config['world_model']['num_regimes'],
            sequence_length=config['data']['sequence_length']
        ).to(device)
        
        checkpoint = torch.load(wm_path, map_location=device)
        world_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        world_model.eval()
        logger.info(f"World Model loaded from {wm_path}")
    else:
        logger.error(f"World Model not found at {wm_path}!")

    # 3. Policy
    policy_path = project_root / config['logging']['checkpoint_dir'] / 'meta_rl/stage4_policy.zip'
    if not policy_path.exists():
         policy_path = project_root / config['logging']['checkpoint_dir'] / 'meta_rl/stage3_policy.zip'
         
    if policy_path.exists():
        # We need a dummy env to init policy structure
        # In production, we'd use a more lightweight policy wrapper
        policy = MetaPolicy(None, config, device=device) # Env=None works for some PPO loads if using policy.load
        policy.model = policy.model.load(str(policy_path))
        logger.info(f"Policy loaded from {policy_path}")
    else:
        logger.error("No policy found!")

@app.get("/")
def read_root():
    return {"status": "UNIFIED API Online"}

@app.post("/predict", response_model=TradeAction)
def predict(state: MarketState):
    """
    Get trading action for current market state.
    Expected ohlcv_window size matches config.data.sequence_length
    """
    if policy is None or world_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    try:
        # 1. Convert window to DataFrame
        df_window = pd.DataFrame(state.ohlcv_window)
        
        # 2. Feature Engineering
        df_features = feature_engineer.compute_all_indicators(df_window)
        
        # 3. Prep Tensors for World Model
        # (Assuming normalization stats are handled or features are pre-normalized)
        # For production, we'd load dataset stats
        
        # 4. Get Action
        # This is a simplified placeholder for the full latent processing
        # In a real setup, we'd use the TradingEnv logic to construct the observation
        
        # Mocking observation for now to show flow
        obs_dim = 194 # From TradingEnv
        obs = np.zeros(obs_dim, dtype=np.float32)
        obs[-6] = state.current_position # position
        
        action, _ = policy.predict(obs, deterministic=True)
        
        return TradeAction(
            target_position=float(action[0]),
            confidence=0.85, # Example
            regime="bull" # Example
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
