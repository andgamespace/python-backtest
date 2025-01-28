import pandas as pd
import numpy as np
from numba import jit
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataHandler:
    """
    Handles loading and processing of financial data with high-performance operations.
    """
    def __init__(self):
        self.data: Dict[str, pd.DataFrame] = {}
        self.required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    
    @staticmethod
    @jit(nopython=True) # Use Numba for performance
    def _calculate_technical_indicators(close_prices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate technical indicators using Numba for performance.
        """
        n = len(close_prices)
        sma_20 = np.zeros(n)
        
        # Calculate 20-day SMA
        for i in range(19, n):
            sma_20[i] = np.mean(close_prices[i-19:i+1])
            
        return {
            'sma_20': sma_20
        }
    
    def load_data(self, 
                  file_paths: List[str], 
                  symbol: str,
                  datetime_format: Optional[str] = None) -> pd.DataFrame:
        """
        Load and process multiple CSV files for a given symbol.
        """
        dfs = []
        
        for file_path in file_paths:
            try:
                df = pd.read_csv(
                    file_path,
                    parse_dates=['datetime'],
                    date_parser=lambda x: pd.to_datetime(x, format=datetime_format) 
                                       if datetime_format else pd.to_datetime(x)
                )
                
                # Validate columns
                if not all(col in df.columns for col in self.required_columns):
                    missing = set(self.required_columns) - set(df.columns)
                    raise ValueError(f"Missing required columns: {missing}")
                
                df['symbol'] = symbol
                dfs.append(df)
                logger.info(f"Loaded {file_path}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
                continue
        
        if not dfs:
            return pd.DataFrame()
        
        # Combine and sort data
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
        
        # Add technical indicators
        close_prices = combined_df['close'].values
        indicators = self._calculate_technical_indicators(close_prices)
        for name, values in indicators.items():
            combined_df[name] = values
        
        self.data[symbol] = combined_df
        return combined_df