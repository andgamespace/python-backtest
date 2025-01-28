from typing import List, Dict, Union, Optional, Tuple
import pandas as pd
import logger
import numpy as np
from pathlib import Path

class DataLoader(object, logger: Optional[logging.Logger] = None):
    """Data loading interface for backtesting engine
    Example: 
    data_loader = DataLoader('data.csv')
    data = data_loader.load_data("path/to/data.csv)
    """
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None

    def load(self, path: Union[str, Path]) -> pd.DataFrame:
        # Load financial data from the specified path
        path = Path(path)

        if not path.exist():
            raise FileNotFoundError(f"File not found: {path}")
        
        if path.suffix not in self._supported_formats:
            raise ValueError(f"Unsupported format. Use {self._supported_formats}")
        
        if path.suffix == '.csv':
            self.data = pd.read_csv(path, parse_dates=['datetime'])
        elif path.suffix == '.parquet':
            self.data = pd.read_parquet(path)
        
        self._validate_data()
        return self.data
    def _validate_data(self) -> None:
        #Ensure data meets required format
        required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']

        if not all(col in self.data.columns for col in required_columns):
            missing = set(required_columns) - set(self.data.columns)
            raise ValueError(f"Missing required columns: {missing}")
