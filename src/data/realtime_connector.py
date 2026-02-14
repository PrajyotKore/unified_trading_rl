import yfinance as yf
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class YahooFinanceConnector:
    """
    Real-time data connector using Yahoo Finance (yfinance).
    Provides free market data for development.
    """
    
    def __init__(self, symbol: str = "^NSEI"): # Default to NIFTY 50
        self.symbol = symbol
        
    def fetch_latest_data(self, period: str = "1d", interval: str = "1m") -> pd.DataFrame:
        """
        Fetch the latest market data.
        
        Args:
            period: Data period (e.g., '1d', '5d')
            interval: Data interval (e.g., '1m', '5m', '1h')
        """
        try:
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data returned for {self.symbol}")
                return pd.DataFrame()
                
            # Clean column names to match project convention
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={'stock splits': 'splits'})
            
            # Ensure index is datetime and named correctly
            df.index.name = 'datetime'
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from yfinance: {e}")
            return pd.DataFrame()

    def get_realtime_quote(self) -> Dict:
        """Get the most recent single quote."""
        df = self.fetch_latest_data(period="1d", interval="1m")
        if not df.empty:
            latest = df.iloc[-1]
            return {
                'symbol': self.symbol,
                'datetime': df.index[-1],
                'open': latest['open'],
                'high': latest['high'],
                'low': latest['low'],
                'close': latest['close'],
                'volume': latest['volume']
            }
        return {}

if __name__ == "__main__":
    # Test
    connector = YahooFinanceConnector()
    quote = connector.get_realtime_quote()
    print(f"Latest NIFTY 50 Quote: {quote}")
