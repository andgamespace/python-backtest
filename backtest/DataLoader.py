import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional
import talib  # Ensure you have TA-Lib installed
from sklearn.preprocessing import StandardScaler, MinMaxScaler # Import for scaling

class DataLoader:
    # Utility class for loading financial data
    def __init__(self, cache_data: bool = True, scaler_type: Optional[str] = None): # Added scaler_type
        self.data: Dict[str, pd.DataFrame] = {}
        self.logger = self._setup_logger()
        self.cache_data = cache_data
        self.scaler_type = scaler_type # Store scaler type
        self.scalers: Dict[str, Any] = {} # Dictionary to store scalers for each ticker

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
        Reads and concatenates multiple CSV files for a given stock symbol with data validation.
        """
        dfs = []
        for file_path in file_paths:
            try:
                df = pd.read_csv(
                    file_path,
                    sep=sep,
                    parse_dates=['datetime'],
                    usecols=structure
                )
                # Data Validation
                if df.empty:
                    self.logger.warning(f"File {file_path} is empty for {stock_symbol}.")
                    continue

                for col in structure:
                    if col not in df.columns:
                        self.logger.error(f"Column '{col}' missing in {file_path} for {stock_symbol}.")
                        return pd.DataFrame() # Return empty DataFrame if essential column is missing

                # Type validation and correction
                if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
                    try:
                        df['datetime'] = pd.to_datetime(df['datetime'])
                    except ValueError:
                        self.logger.error(f"Invalid datetime format in {file_path} for {stock_symbol}.")
                        return pd.DataFrame()

                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_cols:
                    if col in df.columns:
                        try:
                            df[col] = pd.to_numeric(df[col])
                            if (df[col] < 0).any(): # Check for negative values in price/volume columns
                                self.logger.warning(f"Negative values found in '{col}' column in {file_path} for {stock_symbol}. Clipping to 0.")
                                df[col] = df[col].clip(lower=0) # Clip negative values to 0
                        except ValueError:
                            self.logger.error(f"Non-numeric values in '{col}' column in {file_path} for {stock_symbol}.")
                            return pd.DataFrame()

                # Handle missing values - Forward fill then backward fill
                df.fillna(method='ffill', inplace=True)
                df.fillna(method='bfill', inplace=True)
                if df.isnull().any().any(): # Final check for any remaining NaNs
                    self.logger.warning(f"Still missing values after fill in {file_path} for {stock_symbol}. Consider more robust data handling.")


                self.logger.info(f"Successfully read and validated {file_path}")
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
            self.logger.warning(f"No valid dataframes to concatenate for {stock_symbol}.")
            return pd.DataFrame()  # Return empty DataFrame if no dataframes were read

    def load_ticker(self, stock_symbol: str, file_paths: List[str], structure: List[str] = ['datetime', 'open', 'high', 'low', 'close', 'volume'], sep: str = ';', return_numpy: bool = False, scale_features: bool = True) -> None: # Added scale_features
        """
        Loads data for a specific stock ticker, applies feature engineering and scaling, and stores it.
        """
        if self.cache_data and stock_symbol in self.data:
            self.logger.info(f"Data for {stock_symbol} is already loaded and cached.")
            return

        self.logger.info(f"Loading data for {stock_symbol}...")
        combined_df = self.read_stock_data(file_paths, stock_symbol, structure, sep)

        if not combined_df.empty:
            try:
                features_df = self.get_features(combined_df)  # Pass DataFrame directly
                if features_df is not None and not features_df.empty:
                    processed_df = features_df
                    if scale_features and self.scaler_type: # Apply scaling if requested and scaler_type is set
                        processed_df = self._scale_data(stock_symbol, features_df)

                    if return_numpy:
                        self.data[stock_symbol] = processed_df.drop(columns=['datetime']).values # Store numpy array, exclude datetime
                        self.logger.info(f"Data with features for {stock_symbol} loaded as NumPy array.")
                    else:
                        self.data[stock_symbol] = processed_df  # Update with processed data, store as DataFrame
                        self.logger.info(f"Data with features for {stock_symbol} loaded as DataFrame.")
                else:
                    self.logger.warning(f"Feature generation failed for {stock_symbol}, using raw data.")
                    if return_numpy:
                        self.data[stock_symbol] = combined_df.drop(columns=['datetime']).values # Store raw data as numpy, exclude datetime
                        self.logger.warning(f"Raw data for {stock_symbol} loaded as NumPy array.")
                    else:
                        self.data[stock_symbol] = combined_df # Store raw data as DataFrame
                        self.logger.warning(f"Raw data for {stock_symbol} loaded as DataFrame.")

            except Exception as e:
                self.logger.error(f"Error during feature generation for {stock_symbol}: {e}")
                if return_numpy:
                    self.data[stock_symbol] = combined_df.drop(columns=['datetime']).values # Store raw data as numpy, exclude datetime
                    self.logger.warning(f"Raw data for {stock_symbol} loaded as NumPy array due to error.")
                else:
                    self.data[stock_symbol] = combined_df # Store raw data as DataFrame
                    self.logger.warning(f"Raw data for {stock_symbol} loaded as DataFrame due to error.")
        else:
            self.logger.warning(f"No data loaded for {stock_symbol}.")

    def _scale_data(self, stock_symbol: str, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Scales the features DataFrame based on self.scaler_type.
        """
        df_scaled = features_df.copy()
        scaler = None
        if self.scaler_type == 'standard':
            scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            return df_scaled # No scaling

        numerical_cols = ['open', 'high', 'low', 'close', 'volume', 'returns', 'SMA_5', 'SMA_20', 'volatility', 'RSI', 'MACD', 'MACD_Signal', 'BB_upper', 'BB_middle', 'BB_lower']
        cols_to_scale = [col for col in numerical_cols if col in df_scaled.columns] # Scale only available columns
        if cols_to_scale:
            df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
            self.scalers[stock_symbol] = scaler # Store scaler for potential inverse transform later
            self.logger.info(f"Features for {stock_symbol} scaled using {self.scaler_type} scaler.")
        else:
            self.logger.warning(f"No numerical columns to scale for {stock_symbol}.")
        return df_scaled

    def get_latest_price(self, ticker: str) -> float:
        """Get most recent price for a ticker."""
        if ticker in self.data:
            if isinstance(self.data[ticker], pd.DataFrame):
                return self.data[ticker]['close'].iloc[-1]
            elif isinstance(self.data[ticker], np.ndarray):
                close_col_index = self.get_feature_columns().index('close') # Assuming 'close' is still a feature
                return self.data[ticker][-1, close_col_index] # Access by index
        return None

    def get_price_history(self, ticker: str, lookback: int = None) -> pd.DataFrame or np.ndarray:
        """Get price history for a ticker with optional lookback period."""
        if ticker not in self.data:
            return None
        data = self.data[ticker]
        if lookback:
            if isinstance(data, pd.DataFrame):
                return data.tail(lookback)
            elif isinstance(data, np.ndarray):
                return data[-lookback:]
        return data

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

    def get_feature_columns(self):
        """Returns a list of feature column names, assuming features are generated."""
        # Define the feature columns in the order they are created in get_features
        return ['datetime', 'open', 'high', 'low', 'close', 'volume', 'returns', 'SMA_5', 'SMA_20', 'volatility', 'RSI', 'MACD', 'MACD_Signal', 'BB_upper', 'BB_middle', 'BB_lower']