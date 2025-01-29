import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict

class DataLoader:
    #utility class for loading in financial data
    def __init__(self, cache_data: bool = True):
        self.data: Dict[str, pd.DataFrame] = {}
        self.logger = self._setup_logger()
        self.cache_data = cache_data
        self.logger = self.__setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('DataLoader')
        #Create console handler with a higher log level
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
            pd.DataFrame: Combined DataFrame sorted by datetime in ascending order.
        """
        dfs = []
        for file_path infile_