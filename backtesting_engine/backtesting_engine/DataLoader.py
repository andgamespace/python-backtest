from typing import List, Dict, Union, Optional, Tuple
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

class DataLoader:
    def __init__(self, cache_data: bool = True):
        self.data: Dict[str, pd.DataFrame] = {}
        self.column_mappings: Dict[str, Dict[str, str]] = {}
        self.logger = self._setup_logger()
        self.cache_data = cache_data
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('DataLoader')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Map file columns to standardized names"""
        required = {'open', 'high', 'low', 'close', 'volume'}
        mapping = {}
        
        # Convert all column names to lowercase for comparison
        cols = {col.lower(): col for col in df.columns}
        
        # Direct matches
        for req in required:
            if req in cols:
                mapping[req] = cols[req]
                
        # Common variations
        variations = {
            'open': ['open_price', 'opening', 'op'],
            'high': ['high_price', 'highest', 'hp'],
            'low': ['low_price', 'lowest', 'lp'],
            'close': ['close_price', 'closing', 'cp'],
            'volume': ['vol', 'quantity', 'qty']
        }
        
        # Try variations for unmapped columns
        for req, vars in variations.items():
            if req not in mapping:
                for var in vars:
                    if var in cols:
                        mapping[req] = cols[var]
                        break
                        
        return mapping

    def load_data(self, symbol: str, paths: Union[str, List[str]], interval: str = '1min') -> pd.DataFrame:
        """Load and process data for a single symbol"""
        if isinstance(paths, str):
            paths = [paths]
            
        dfs = []
        first_file = True
        
        for path in paths:
            df = pd.read_csv(path)
            
            if first_file:
                # Detect column structure from first file
                self.column_mappings[symbol] = self._detect_columns(df)
                first_file = False
                
            # Rename columns using detected mapping
            df = df.rename(columns=self.column_mappings[symbol])
            
            # Ensure datetime column exists
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                df['datetime'] = pd.to_datetime(df[date_cols[0]])
            
            dfs.append(df)
            
        # Combine all files
        final_df = pd.concat(dfs, ignore_index=True)
        
        # Standardize the dataframe
        final_df = self._standardize_dataframe(final_df, interval)
        
        if self.cache_data:
            self.data[symbol] = final_df
            
        return final_df
        
    def _standardize_dataframe(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """Standardize dataframe format"""
        # Ensure required columns exist
        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
                
        # Set datetime as index
        df = df.set_index('datetime')
        
        # Sort by datetime
        df = df.sort_index()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Resample to desired interval
        df = df.resample(interval).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return df

    def load_multiple_symbols(self, symbol_paths: Dict[str, Union[str, List[str]]], 
                            interval: str = '1min') -> Dict[str, pd.DataFrame]:
        """Load data for multiple symbols in parallel"""
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.load_data, symbol, paths, interval): symbol 
                for symbol, paths in symbol_paths.items()
            }
            
            results = {}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    self.logger.error(f"Failed to load {symbol}: {str(e)}")
                    
        return results

    def get_aligned_data(self, symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Get data aligned to common timestamps"""
        if symbols is None:
            symbols = list(self.data.keys())
            
        # Find common index
        common_index = None
        for symbol in symbols:
            if symbol not in self.data:
                raise KeyError(f"No data loaded for symbol {symbol}")
            if common_index is None:
                common_index = self.data[symbol].index
            else:
                common_index = common_index.intersection(self.data[symbol].index)
                
        # Align all dataframes
        aligned_data = {
            symbol: self.data[symbol].loc[common_index] 
            for symbol in symbols
        }
        
        return aligned_data