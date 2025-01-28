from typing import List, Dict, Union, Optional, Tuple
import pandas as pd
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

class DataLoader:
    """
    Data loading interface for backtesting engine.
    Handles multiple data files per ticker, maintaining organized dataframes.
    
    Attributes:
        data (Dict[str, pd.DataFrame]): Dictionary of dataframes for each ticker/symbol
        logger (logging.Logger): Custom logger for DataLoader
    example:
        data_loader = DataLoader(data_paths='data/TICKER.csv')
    """
   
    def __init__(self, data_paths, logger: Optional[logging.Logger] = None):
        # Convert single path to list if necessary
        self.data_paths = [data_paths] if not isinstance(data_paths, list) else data_paths
        
        # Initialize storage for loaded data
        self.data: Dict[str, pd.DataFrame] = {}
        self.logger = logger or self._setup_logger()
        
        # Required columns for data validation (case-insensitive)
        self._required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        
        # Supported file extensions (based on pandas' capabilities)
        self._readable_formats = {
            '.csv': pd.read_csv,
            '.xlsx': pd.read_excel,
            '.xls': pd.read_excel,
            '.parquet': pd.read_parquet,
            '.feather': pd.read_feather,
            '.pkl': pd.read_pickle,
            '.h5': pd.read_hdf
        }

    def _setup_logger(self) -> logging.Logger:
        """Configure default logger with proper formatting."""
        logger = logging.getLogger('DataLoader')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            #Create console handler
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False #prevent double logging
        return logger

    def _validate_paths(self, paths: Union[str, List[str]]) -> List[Path]:
        """
        Validate and convert file paths to Path objects.
        
        Args:
            paths: Single path or list of paths to data files
            
        Returns:
            List of validated Path objects
        todo: [ ] test and make sure it works 
        """
        # Convert single path to list if necessary
        if isinstance(paths, (str, Path)):
            paths = [paths]
            
        # Convert to Path objects
        path_objects = [Path(p) for p in paths]
        
        # Validate paths
        for path in path_objects:
            if not path.exists():
                self.logger.error(f"File not found: {path}")
                raise FileNotFoundError(f"File not found: {path}")
                
        return path_objects

    def _read_file(self, path: Path) -> Optional[pd.DataFrame]:
        """
        Read data file using appropriate pandas reader based on file extension.
        
        Args:
            path: Path to data file
            
        Returns:
            DataFrame or None if reading fails
        todo; [ ] test and make sure it works
        """
        try:
            file_extension = path.suffix.lower()
            # Get appropriate reader function
            reader_func = self._readable_formats.get(file_extension)
            
            if reader_func is None:
                supported_formats = ", ".join(self._readable_formats.keys())
                self.logger.warning(
                    f"Unsupported file format: {file_extension}. "
                    f"Attempting to use pd.read_csv. "
                    f"Supported formats are: {supported_formats}"
                )
                reader_func = pd.read_csv
            # Read the file
            df = reader_func(path)
            # Try to parse datetime column if it exists
            if 'datetime' in df.columns:
                try:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                except Exception as e:
                    self.logger.warning(f"Failed to parse datetime column: {e}")
            
            self.logger.info(f"Successfully read {path}")
            return df
        
        except Exception as e:
            self.logger.error(f"Error reading {path}: {str(e)}")
            return None

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """  Validate and clean loaded data.
        Args:
            df: DataFrame to validate   
        Returns:
            Validated and cleaned DataFrame  """
        if df is None or df.empty:
            raise ValueError("No data loaded or empty DataFrame")
        # Convert column names to lowercase for case-insensitive comparison
        df.columns = df.columns.str.lower()
        # Check for required columns (case-insensitive)
        missing_cols = set(self._required_columns) - set(df.columns)
        if missing_cols:
            self.logger.warning(f"Missing required columns: {missing_cols}")
            return df
        
        # Ensure numeric columns are properly typed
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

    def load_path_list(self, path_list: Union[str, List[str]], symbol: str = None) -> pd.DataFrame:
        """Load financial data from specified path(s).
        Args:
            path_list: Path or list of paths to data files
            symbol: Optional symbol/ticker for the data
        Returns:
            Loaded and validated DataFrame """
        try:
            self.logger.info(f"Starting to load data from {path_list}")
            
            # Validate paths
            paths = self._validate_paths(path_list)
            
            # Read and combine all files
            dfs = []
            for path in paths:
                df = self._read_file(path)
                if df is not None:
                    dfs.append(df)
            
            if not dfs:
                raise ValueError("No valid data loaded from provided paths")
            
            # Combine all dataframes
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Validate and clean data
            combined_df = self._validate_data(combined_df)
            
            # Add symbol if provided
            if symbol:
                combined_df['symbol'] = symbol
                self.data[symbol] = combined_df
            
            self.logger.info(
                f"Successfully loaded {len(paths)} files. "
                f"Total rows: {len(combined_df)}"
            )
            
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise