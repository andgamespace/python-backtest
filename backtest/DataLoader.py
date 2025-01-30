import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict
import talib  # Ensure you have TA-Lib installed

class DataLoader:
    # Utility class for loading financial data
    def __init__(self, cache_data: bool = True):
        self.data: Dict[str, pd.DataFrame] = {}
        self.logger = self._setup_logger()
        self.cache_data = cache_data

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('DataLoader')
        # Create console handler with a higher log level
        ch = logging.StreamHandler()
        logger.setLevel(logging.INFO)
        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        # Add the handlers to the logger
        if not logger.handlers:
            logger.addHandler(ch)
        return logger

    def read_stock_data(
        self, 
        file_paths: List[str], 
        stock_symbol: str, 
        structure: List[str] = ['datetime', 'open', 'high', 'low', 'close', 'volume'],
        sep: str = ';'
        ) -> pd.DataFrame:
        """
        Reads and concatenates multiple CSV files for a given stock symbol.

        Args:
            file_paths (List[str]): List of CSV file paths.
            stock_symbol (str): Stock symbol ('AAPL' or 'AMD') for logging purposes.
            structure (List[str]): Order of columns in the CSV files.
            sep (str): Separator used in the CSV files.

        Returns:
            pd.DataFrame: Combined DataFrame sorted by datetime in ascending order."""
        dfs = []
        for file_path in file_paths:
            try:
                df = pd.read_csv(
                    file_path,
                    sep=sep,
                    parse_dates=['datetime'],
                    usecols=structure
                )
                # Convert 'datetime' to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
                    df['datetime'] = pd.to_datetime(df['datetime'])
                self.logger.info(f"Successfully read {file_path}")
                dfs.append(df)
            except Exception as e:
                self.logger.error(f"Error reading {file_path}: {e}")
                # Do not append df since it was not successfully read
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            # Sort by 'datetime' in ascending order
            combined_df = combined_df.sort_values(by='datetime', ascending=True).reset_index(drop=True)
            return combined_df
        else:
            self.logger.warning(f"No dataframes to concatenate for {stock_symbol}.")
            return pd.DataFrame()  # Return empty DataFrame if no dataframes were read

    def load_ticker(self, stock_symbol: str, file_paths: List[str], structure: List[str] = ['datetime', 'open', 'high', 'low', 'close', 'volume'], sep: str = ';') -> None:
        """
        Loads data for a specific stock ticker and stores it in the data dictionary.
        """
        if self.cache_data and stock_symbol in self.data:
            self.logger.info(f"Data for {stock_symbol} is already loaded and cached.")
            return

        self.logger.info(f"Loading data for {stock_symbol}...")
        combined_df = self.read_stock_data(file_paths, stock_symbol, structure, sep)
        
        if not combined_df.empty:
            try:
                self.data[stock_symbol] = combined_df  # Store the raw data first
                features_df = self.get_features(combined_df)  # Pass DataFrame directly
                if features_df is not None and not features_df.empty:
                    self.data[stock_symbol] = features_df  # Update with features
                    self.logger.info(f"Data with features for {stock_symbol} loaded successfully.")
                else:
                    self.logger.warning(f"Feature generation failed for {stock_symbol}, using raw data.")
            except Exception as e:
                self.logger.error(f"Error during feature generation for {stock_symbol}: {e}")
                # Keep the raw data if feature generation fails
        else:
            self.logger.warning(f"No data loaded for {stock_symbol}.")

    def get_latest_price(self, ticker: str) -> float:
        """Get most recent price for a ticker."""
        if ticker in self.data:
            return self.data[ticker]['close'].iloc[-1]
        return None

    def get_price_history(self, ticker: str, lookback: int = None) -> pd.DataFrame:
        """Get price history for a ticker with optional lookback period."""
        if ticker not in self.data:
            return None
        if lookback:
            return self.data[ticker].tail(lookback)
        return self.data[ticker]

    def get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get feature matrix suitable for ML models.
        """
        if df.empty:
            return None
            
        df = df.copy()
        
        try:
            # Calculate returns
            df['returns'] = df['close'].pct_change()
            
            # Add basic technical indicators
            df['SMA_5'] = df['close'].rolling(window=5).mean()
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # Add RSI
            df['RSI'] = talib.RSI(df['close'].values, timeperiod=14)
            
            # Add MACD
            macd, macdsignal, macdhist = talib.MACD(
                df['close'].values, 
                fastperiod=12, 
                slowperiod=26, 
                signalperiod=9
            )
            df['MACD'] = macd
            df['MACD_Signal'] = macdsignal
            
            # Add Bollinger Bands
            upperband, middleband, lowerband = talib.BBANDS(
                df['close'].values, 
                timeperiod=20,
                nbdevup=2,
                nbdevdn=2,
                matype=0
            )
            df['BB_upper'] = upperband
            df['BB_middle'] = middleband
            df['BB_lower'] = lowerband
            
            # Ensure datetime is correct
            if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
                df['datetime'] = pd.to_datetime(df['datetime'])
            
            return df
        except Exception as e:
            self.logger.error(f"Error calculating features: {e}")
            return None