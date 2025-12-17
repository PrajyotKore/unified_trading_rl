"""
Regime labeling for market conditions.
Identifies bull, bear, crash, and sideways regimes for meta-learning.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class RegimeLabeler:
    """
    Automatically label market regimes for meta-RL task sampling.
    """
    
    def __init__(self, num_regimes: int = 4):
        """
        Initialize regime labeler.
        
        Args:
            num_regimes: Number of regimes (default 4: bull, bear, crash, sideways)
        """
        self.num_regimes = num_regimes
        self.kmeans = None
        self.scaler = StandardScaler()
        self.regime_names = {
            0: 'sideways',
            1: 'bull',
            2: 'bear', 
            3: 'crash'
        }
    
    def compute_regime_features(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Compute features for regime classification.
        
        Args:
            df: DataFrame with OHLCV data
            window: Rolling window for feature calculation
            
        Returns:
            DataFrame with regime features
        """
        df = df.copy()
        
        # Returns statistics
        df['rolling_return'] = df['close'].pct_change(window)
        df['rolling_volatility'] = df['close'].pct_change().rolling(window).std()
        
        # Trend strength
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['trend'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['trend_strength'] = (df['sma_20'] / df['sma_50']) - 1
        
        # Volatility metrics
        df['atr_pct'] = self._compute_atr(df) / df['close']
        
        # Volume trend
        if df['volume'].sum() == 0:
            df['volume_trend'] = 0.0
        else:
            df['volume_trend'] = df['volume'] / (df['volume'].rolling(window).mean() + 1e-8) - 1
        
        # Drawdown
        df['drawdown'] = (df['close'] / df['close'].rolling(window).max()) - 1
        
        return df
    
    def _compute_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Compute Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window).mean()
        
        return atr
    
    def fit_kmeans(self, df: pd.DataFrame) -> pd.Series:
        """
        Fit K-means clustering on regime features.
        
        Args:
            df: DataFrame with regime features
            
        Returns:
            Cluster labels
        """
        feature_cols = ['rolling_return', 'rolling_volatility', 'trend', 
                       'atr_pct', 'volume_trend', 'drawdown']
        
        # Ensure all features exist
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        X = df[feature_cols].dropna()
        
        # Check if we have enough samples
        if len(X) == 0:
            logger.warning("No valid samples after dropping NaN values. Using simple regime labeling.")
            # Fallback: use simple return-based labeling
            return self._simple_regime_labeling(df)
        
        indices = X.index
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit K-means
        logger.info(f"Fitting K-means with {self.num_regimes} clusters on {len(X)} samples...")
        self.kmeans = KMeans(n_clusters=self.num_regimes, random_state=42, n_init=20)
        labels = self.kmeans.fit_predict(X_scaled)
        
        # Map cluster IDs to regime names based on characteristics
        labels = self._map_clusters_to_regimes(X, labels)
        
        # Create Series with index from X (which has dropped NaNs)
        label_series = pd.Series(data=labels, index=X.index, name='regime')
        
        # Reindex to match original df, filling missing with -1
        full_series = label_series.reindex(df.index, fill_value=-1)
        
        return full_series
    
    def _simple_regime_labeling(self, df: pd.DataFrame) -> np.ndarray:
        """
        Simple regime labeling based on returns when K-means fails.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Regime labels
        """
        logger.info("Using simple return-based regime labeling...")
        
        # Calculate rolling returns
        returns = df['close'].pct_change(20).fillna(0)
        volatility = df['close'].pct_change().rolling(20).std().fillna(0)
        
        labels = np.zeros(len(df), dtype=int)
        
        # Simple classification based on returns and volatility
        for i in range(len(df)):
            ret = returns.iloc[i]
            vol = volatility.iloc[i]
            
            if vol > volatility.quantile(0.75):  # High volatility
                labels[i] = 3  # crash
            elif ret > 0.02:  # Positive returns
                labels[i] = 1  # bull
            elif ret < -0.02:  # Negative returns
                labels[i] = 2  # bear
            else:  # Low returns
                labels[i] = 0  # sideways
        
        return labels
    
    def _map_clusters_to_regimes(self, X: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
        """
        Map cluster IDs to meaningful regime names based on cluster characteristics.
        
        Args:
            X: Feature DataFrame
            labels: Cluster labels
            
        Returns:
            Remapped labels
        """
        cluster_stats = {}
        for i in range(self.num_regimes):
            mask = labels == i
            cluster_stats[i] = {
                'mean_return': X.loc[mask, 'rolling_return'].mean(),
                'mean_volatility': X.loc[mask, 'rolling_volatility'].mean(),
                'mean_drawdown': X.loc[mask, 'drawdown'].mean()
            }
        
        # Create mapping based on cluster characteristics
        mapping = {}
        
        # Find crash regime (high volatility, negative returns, large drawdowns)
        crash_cluster = max(cluster_stats.items(), 
                           key=lambda x: x[1]['mean_volatility'] - x[1]['mean_return'])[0]
        mapping[crash_cluster] = 3  # crash
        
        remaining = [i for i in range(self.num_regimes) if i != crash_cluster]
        
        # Find bull regime (positive returns, low-medium volatility)
        bull_cluster = max([i for i in remaining], 
                          key=lambda x: cluster_stats[x]['mean_return'])
        mapping[bull_cluster] = 1  # bull
        
        remaining.remove(bull_cluster)
        
        # Find bear regime (negative returns, medium volatility)
        bear_cluster = min([i for i in remaining],
                          key=lambda x: cluster_stats[x]['mean_return'])
        mapping[bear_cluster] = 2  # bear
        
        remaining.remove(bear_cluster)
        
        # Remaining is sideways (low returns, low volatility)
        if remaining:
            mapping[remaining[0]] = 0  # sideways
        
        # Remap labels
        remapped = np.array([mapping[label] for label in labels])
        
        # Log cluster statistics
        logger.info("Regime statistics:")
        for old_id, new_id in mapping.items():
            regime_name = self.regime_names[new_id]
            stats = cluster_stats[old_id]
            logger.info(f"  {regime_name}: return={stats['mean_return']:.4f}, "
                       f"volatility={stats['mean_volatility']:.4f}, "
                       f"drawdown={stats['mean_drawdown']:.4f}")
        
        return remapped
    
    def label_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label market regimes in DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with 'regime' column
        """
        logger.info("Labeling market regimes...")
        
        # Compute regime features
        df = self.compute_regime_features(df)
        
        # Fit K-means
        labels = self.fit_kmeans(df)
        df['regime'] = labels
        df['regime_name'] = df['regime'].map(self.regime_names)
        
        # Log regime distribution
        regime_counts = df['regime_name'].value_counts()
        logger.info(f"Regime distribution:\n{regime_counts}")
        
        return df
    
    def get_regime_periods(
        self, 
        df: pd.DataFrame, 
        min_period_length: int = 20
    ) -> Dict[str, List[tuple]]:
        """
        Extract contiguous periods for each regime.
        
        Args:
            df: DataFrame with 'regime' column
            min_period_length: Minimum length of regime period
            
        Returns:
            Dictionary mapping regime names to list of (start_idx, end_idx) tuples
        """
        regime_periods = {name: [] for name in self.regime_names.values()}
        
        current_regime = None
        start_idx = 0
        
        for idx, regime in enumerate(df['regime']):
            if regime < 0:  # Skip invalid regimes
                continue
                
            regime_name = self.regime_names[regime]
            
            if regime != current_regime:
                # End of previous regime period
                if current_regime is not None and (idx - start_idx) >= min_period_length:
                    prev_regime_name = self.regime_names[current_regime]
                    regime_periods[prev_regime_name].append((start_idx, idx - 1))
                
                # Start of new regime period
                current_regime = regime
                start_idx = idx
        
        # Handle final period
        if current_regime is not None and (len(df) - start_idx) >= min_period_length:
            regime_name = self.regime_names[current_regime]
            regime_periods[regime_name].append((start_idx, len(df) - 1))
        
        # Log regime periods
        for regime_name, periods in regime_periods.items():
            logger.info(f"{regime_name}: {len(periods)} periods")
        
        return regime_periods
