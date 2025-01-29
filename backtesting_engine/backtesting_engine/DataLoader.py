from typing import List, Dict, Union, Optional, Tuple
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

class DataLoader:
    """
    Loads and stores data in self.data, a dictionary of:
        symbol -> pd.DataFrame with DateTimeIndex
    Each DataFrame contains columns: open, high, low, close, volume,
    resampled to the specified interval, ensuring consistency across symbols.
    """
    def __init__(self, cache_data: bool = True):
        self.data: Dict[str, pd.DataFrame] = {}
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

    def load_data(self, ticker: str, file_paths: Union[str, List[str]], interval: str = '5m',
                 file_structure: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load and process data for a single ticker from one or multiple CSV files.
        Uses the provided file_structure to standardize data.
        :param ticker: The stock ticker symbol.
        :param file_paths: List of CSV file paths containing the data.
        :param interval: Resampling interval (default is '5m').
        :param file_structure: List defining the order and names of columns.
                               Default is ['datetime', 'open', 'high', 'low', 'close', 'volume'].
        """
        file_structure = file_structure or ['datetime', 'open', 'high', 'low', 'close', 'volume']
        
        if isinstance(file_paths, str):
            file_paths = [file_paths]
            
        dfs = []
        
        for path in file_paths:
            try:
                df = pd.read_csv(path)
            except FileNotFoundError:
                self.logger.error(f"File not found: {path}")
                continue
            except Exception as e:
                self.logger.error(f"Error reading {path}: {str(e)}")
                continue
            
            # Rename columns based on provided file_structure
            if len(df.columns) >= len(file_structure):
                rename_mapping = {original: new for original, new in zip(df.columns[:len(file_structure)], file_structure)}
                df = df.rename(columns=rename_mapping)
            else:
                self.logger.warning(f"Column mismatch in {path} for ticker {ticker}")
                continue  # Skip files that don't match the expected structure
            
            # Ensure datetime column is parsed
            try:
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                invalid_count = df['datetime'].isna().sum()
                if invalid_count > 0:
                    self.logger.warning(f"{invalid_count} invalid datetime entries found in {path}, setting them to NaT.")
            except Exception as e:
                self.logger.error(f"Datetime parsing failed for {path}: {str(e)}")
                continue  # Skip files with parsing errors

            dfs.append(df)
            
        if not dfs:
            raise ValueError(f"No valid data loaded for ticker {ticker}")
            
        # Combine all files
        final_df = pd.concat(dfs, ignore_index=True)
        
        # Standardize the dataframe
        final_df = self._standardize_dataframe(final_df, interval)
        
        if self.cache_data:
            self.data[ticker] = final_df
            
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
                              interval: str = '1m',
                              file_structure: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Load data for multiple tickers in parallel"""
        file_structure = file_structure or ['datetime', 'open', 'high', 'low', 'close', 'volume']
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.load_data, ticker, paths, interval, file_structure): ticker 
                for ticker, paths in symbol_paths.items()
            }
            
            results = {}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    results[ticker] = future.result()
                except Exception as e:
                    self.logger.error(f"Failed to load {ticker}: {str(e)}")
                    
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