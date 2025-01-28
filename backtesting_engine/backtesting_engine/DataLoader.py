from typing import List, Dict, Union, Optional, Tuple
import pandas as pd
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

class DataLoader():
    # Data loading interface for backtesting engine
    # handles multliple csv files per ticker, maintaining organized dataframes
    # Attributes: 
    # data (Dict[str, pd.DataFrame]): Dictionary of dataframes for each ticker/symbol
    # logger(logging.Logger): Custom logger for DataLoader
    # Example: 
    # data_loader = DataLoader('data.csv')
    # data = data_loader.load_data("path/to/data.csv)
    # 
    def __init__(self, data_paths, logger: Optional[logging.Logger] = None):
        # make sure data_paths is a list
        if not isinstance(data_paths, list):
            data_paths = [data_paths]
        # Initialize Datloader with optional custom logger
        # Args: Optional custom logger, file path to data
        # todo: [ ] add support for multiple data sources
        # todo: [ ] *test* make sure that loading data works primarily for csv
        # todo: [ ] add support for different data formats
        # todo: [ ] keep track of loaded data
        # todo: [ ] *test* make sure that other parts of the code can access the loaded data
        self.data: Dict[str, pd.DataFrame] = {}
        self.logger = logger or self.setup_logger() # set up logging
         # required columns for data validation
        self.__required_columns__ = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        # should ingnore case when checking for required columns  
        self.data_paths = data_paths
        self.data = None

    def setup_logger(self) -> logging.Logger:
        #Configure default logger here
        logger = logging.getLogger('DataLoader')
        logger.setLevel(logging.INFO) # set minimum logging level

        if not logger.handlers:
            #create console handler 
            handler = logging.StreamHandler()
            #create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
                )
            #Add formatter to handler
            formatter = logging.Formatter(formatter)
            #add handler to logger
            logger.addHandler(handler)
            logger.propagate = False # prevent double logging
        return logger
    


    def _validate_paths(self, paths: Union[str, List[str]]) -> List[Path]:
        # Validate paths
        # Args: list of paths to data files
        # Returns: list of Path objects
        # [ ] todo: make sure that different data formats are supported
        if isinstance(paths, (str, Path)):
            paths = [paths]
            # convert single path to a list
        #convert strings to path objects
        path_objects = [Path(p) for p in paths]
        #Validate that the paths exist
        for path in path_objects:
            if not path.exists():
                self.logger.error(f"File not found: {path}")
                raise FileNotFoundError(f"File not found: {path}")
            if path.suffix != '.csv':
                self.logger.error(f"Unsupported format: {path}")
                raise FileNotFoundError(f"Unsupported format. Use .csv")
        return path_objects

    def load_path_list(self, path_list: Union[str, Path]) -> pd.DataFrame:
        # Load financial data from the specified path
        # primary method for loading data
        # Args: list of paths to data file
        try:
            # Info level - General Information
            self.logger.info(f"Starting to load data from {path_list}")
            path_list = Path(path)

        if not path_list.exist():
            raise FileNotFoundError(f"File not found: {path}")
        
        if path_list.suffix not in self._supported_formats:
            raise ValueError(f"Unsupported format. Use {self._supported_formats}")
        
        if path_list.suffix == '.csv':
            self.data = pd.read_csv(path_list, parse_dates=['datetime'])
        elif path_list.suffix == '.parquet':
            self.data = pd.read_parquet(path_list)
        
        self._validate_data()
        return self.data
    def _validate_data(self) -> None:
        #Ensure data meets required format
        required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']

        if not all(col in self.data.columns for col in required_columns):
            missing = set(required_columns) - set(self.data.columns)
            raise ValueError(f"Missing required columns: {missing}")
