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
                # might need to convert to pd.datetime if not already
                self.logger.info(f"Succesfully read {file_path}")
                dfs.append(df)
            except Exception as e:
                self.logger.error(f"Error reading {file_path}")
                dfs.append(df)
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            # Sort by 'datetime' in ascending order
            combined_df = combined_df.sort_values(by='datetime', ascending=True).reset_index(drop=True)
            return combined_df
        else:
            self.logger.warning(f"No dataframes to concatenate for {stock_symbol}.")
            return pd.DataFrame()  # Return empty DataFrame if no dataframes were read

    def load_ticker(
        self, 
        stock_symbol: str, 
        file_paths: List[str], 
        structure: List[str] = ['datetime', 'open', 'high', 'low', 'close', 'volume'], 
        sep: str = ';'
    ) -> None:
        """
        Loads data for a specific stock ticker and stores it in the data dictionary.

        Args:
            stock_symbol (str): Stock symbol (e.g., 'AAPL').
            file_paths (List[str]): List of CSV file paths for the stock.
            structure (List[str], optional): Order of columns in the CSV files.
                Defaults to ['datetime', 'open', 'high', 'low', 'close', 'volume'].
            sep (str, optional): Separator used in the CSV files. Defaults to ';'.
        """
        if self.cache_data and stock_symbol in self.data:
            self.logger.info(f"Data for {stock_symbol} is already loaded and cached.")
            return

        self.logger.info(f"Loading data for {stock_symbol}...")
        combined_df = self.read_stock_data(file_paths, stock_symbol, structure, sep)

        if not combined_df.empty:
            self.data[stock_symbol] = combined_df
            self.logger.info(f"Data for {stock_symbol} loaded successfully.")
        else:
            self.logger.warning(f"No data loaded for {stock_symbol}.")