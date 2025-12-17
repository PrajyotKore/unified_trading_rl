"""
Feature engineering for NIFTY 50 trading data.
Implements 40-50 technical indicators as specified in the paper.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

# Try to import ta library (technical analysis)
try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False
    logging.warning("ta library not found, using custom implementations")

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for financial time series.
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.features_computed = []
    
    def compute_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute various return metrics."""
        df = df.copy()
        
        # Simple returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Multi-period returns
        for period in [5, 10, 20]:
            df[f'returns_{period}'] = df['close'].pct_change(period)
        
        self.features_computed.extend(['returns', 'log_returns', 'returns_5', 'returns_10', 'returns_20'])
        return df
    
    def compute_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volatility metrics."""
        df = df.copy()
        
        # Rolling volatility
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['returns'].rolling(window).std()
        
        # Parkinson volatility (uses high-low range)
        df['parkinson_vol'] = np.sqrt(
            np.log(df['high'] / df['low']) ** 2 / (4 * np.log(2))
        )
        
        # Garman-Klass volatility
        df['gk_vol'] = np.sqrt(
            0.5 * np.log(df['high'] / df['low']) ** 2 -
            (2 * np.log(2) - 1) * np.log(df['close'] / df['open']) ** 2
        )
        
        self.features_computed.extend(['volatility_5', 'volatility_10', 'volatility_20', 
                                      'parkinson_vol', 'gk_vol'])
        return df
    
    def compute_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute momentum indicators."""
        df = df.copy()
        
        if HAS_TA:
            # RSI (Relative Strength Index)
            df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # CCI (Commodity Channel Index)
            df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
            
            # ROC (Rate of Change)
            df['roc'] = ta.momentum.ROCIndicator(df['close'], window=12).roc()
            
            self.features_computed.extend(['rsi_14', 'macd', 'macd_signal', 'macd_diff',
                                          'stoch_k', 'stoch_d', 'cci', 'roc'])
        else:
            # Custom RSI implementation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # Simple momentum
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
            
            self.features_computed.extend(['rsi_14', 'momentum_10', 'momentum_20'])
        
        return df
    
    def compute_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute trend indicators."""
        df = df.copy()
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        
        # MA crossovers
        df['sma_cross_5_20'] = (df['sma_5'] > df['sma_20']).astype(int)
        df['ema_cross_5_20'] = (df['ema_5'] > df['ema_20']).astype(int)
        
        if HAS_TA:
            # ADX (Average Directional Index)
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
            df['adx'] = adx.adx()
            df['adx_pos'] = adx.adx_pos()
            df['adx_neg'] = adx.adx_neg()
            
            # Aroon
            aroon = ta.trend.AroonIndicator(df['high'], df['low'])
            df['aroon_up'] = aroon.aroon_up()
            df['aroon_down'] = aroon.aroon_down()
            
            self.features_computed.extend(['adx', 'adx_pos', 'adx_neg', 'aroon_up', 'aroon_down'])
        
        self.features_computed.extend([f'sma_{w}' for w in [5, 10, 20, 50]] +
                                      [f'ema_{w}' for w in [5, 10, 20, 50]] +
                                      ['sma_cross_5_20', 'ema_cross_5_20'])
        return df
    
    def compute_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volatility-based indicators."""
        df = df.copy()
        
        if HAS_TA:
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'])
            df['bb_high'] = bb.bollinger_hband()
            df['bb_mid'] = bb.bollinger_mavg()
            df['bb_low'] = bb.bollinger_lband()
            df['bb_width'] = bb.bollinger_wband()
            df['bb_pct'] = bb.bollinger_pband()
            
            # ATR (Average True Range)
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            
            # Keltner Channel
            kc = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
            df['kc_high'] = kc.keltner_channel_hband()
            df['kc_low'] = kc.keltner_channel_lband()
            
            self.features_computed.extend(['bb_high', 'bb_mid', 'bb_low', 'bb_width', 'bb_pct',
                                          'atr', 'kc_high', 'kc_low'])
        else:
            # Custom Bollinger Bands
            for window in [20]:
                rolling_mean = df['close'].rolling(window).mean()
                rolling_std = df['close'].rolling(window).std()
                df[f'bb_high_{window}'] = rolling_mean + 2 * rolling_std
                df[f'bb_low_{window}'] = rolling_mean - 2 * rolling_std
                df[f'bb_width_{window}'] = (df[f'bb_high_{window}'] - df[f'bb_low_{window}']) / rolling_mean
            
            # Custom ATR
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift(1))
            tr3 = abs(df['low'] - df['close'].shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr'] = tr.rolling(14).mean()
            
            self.features_computed.extend(['bb_high_20', 'bb_low_20', 'bb_width_20', 'atr'])
        
        return df
    
    def compute_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volume indicators."""
        df = df.copy()
        
        # Check if volume data is available (indices might have 0 volume)
        if df['volume'].sum() == 0:
            logger.warning("Volume is all zero (likely an index). Skipping volume indicators.")
            # Set volume features to 0
            df['volume_ratio_5'] = 0.0
            df['volume_ratio_20'] = 0.0
            df['obv'] = 0.0
            df['vwap'] = df['close']  # fallback to price
            df['cmf'] = 0.0
            df['mfi'] = 50.0  # neutral
            df['volume_trend'] = 0.0
            
            features = ['volume_ratio_5', 'volume_ratio_20', 'obv', 'vwap', 'cmf', 'mfi', 'volume_trend']
            if HAS_TA:
                self.features_computed.extend(['obv', 'vwap', 'cmf', 'mfi'])
            else:
                self.features_computed.append('volume_trend')
            self.features_computed.extend(['volume_ratio_5', 'volume_ratio_20'])
            
            return df

        # Volume ratios
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        df['volume_ratio_5'] = df['volume'] / (df['volume'].rolling(5).mean() + epsilon)
        df['volume_ratio_20'] = df['volume'] / (df['volume'].rolling(20).mean() + epsilon)
        
        if HAS_TA:
            # OBV (On-Balance Volume)
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            
            # VWAP (Volume Weighted Average Price) - approximate
            # Handle zero volume sum in rolling window
            vol_sum = df['volume'].rolling(20).sum() + epsilon
            df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / vol_sum
            
            # CMF (Chaikin Money Flow)
            df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
            
            # MFI (Money Flow Index)
            df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
            
            self.features_computed.extend(['obv', 'vwap', 'cmf', 'mfi'])
        else:
            # Custom volume features
            df['volume_trend'] = df['volume'].pct_change(5).fillna(0)
            self.features_computed.append('volume_trend')
        
        self.features_computed.extend(['volume_ratio_5', 'volume_ratio_20'])
        return df
    
    def compute_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute price pattern features."""
        df = df.copy()
        
        # Candle patterns
        df['body'] = df['close'] - df['open']
        df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
        df['body_pct'] = abs(df['body']) / (df['high'] - df['low'] + 1e-8)
        
        # Gap detection
        df['gap_up'] = ((df['open'] > df['close'].shift(1)) & 
                       (df['low'] > df['high'].shift(1))).astype(int)
        df['gap_down'] = ((df['open'] < df['close'].shift(1)) & 
                         (df['high'] < df['low'].shift(1))).astype(int)
        
        self.features_computed.extend(['body', 'upper_shadow', 'lower_shadow', 'body_pct',
                                      'gap_up', 'gap_down'])
        return df
    
    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all features
        """
        logger.info("Computing all features...")
        self.features_computed = []
        
        df = self.compute_returns(df)
        df = self.compute_volatility(df)
        df = self.compute_momentum_indicators(df)
        df = self.compute_trend_indicators(df)
        df = self.compute_volatility_indicators(df)
        df = self.compute_volume_indicators(df)
        df = self.compute_price_patterns(df)
        
        # Drop NaN values from rolling calculations
        initial_len = len(df)
        df = df.dropna()
        logger.info(f"Dropped {initial_len - len(df)} rows with NaN values")
        
        logger.info(f"Computed {len(self.features_computed)} features: {self.features_computed[:10]}...")
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of computed feature column names."""
        return self.features_computed
