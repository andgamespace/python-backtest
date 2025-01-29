from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass

@dataclass
class StrategyState:
    """Track strategy state for resuming/checkpointing"""
    parameters: Dict[str, Any]
    positions: Dict[str, float]
    last_signals: Dict[str, float]
    last_update: pd.Timestamp

class Strategy(ABC):
    def __init__(self, symbols: List[str], parameters: Dict = None):
        self.symbols = symbols
        self.parameters = parameters or {}
        self.position_size = 1.0
        self.logger = self._setup_logger()
        self.positions: Dict[str, float] = {symbol: 0.0 for symbol in symbols}
        self.state = StrategyState(
            parameters=self.parameters.copy(),
            positions=self.positions.copy(),
            last_signals={symbol: 0.0 for symbol in symbols},
            last_update=pd.Timestamp.now()
        )
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
        
    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        Generate trading signals for each symbol in 'data'. Return a dict of symbol -> pd.Series,
        where each Series index matches the DataFrame dates, and the values are -1 / 0 / +1.
        """
        pass
        
    def calculate_position_sizes(self, portfolio_value: float) -> Dict[str, float]:
        """Calculate position sizes for each symbol using risk management"""
        # Get volatility for each symbol
        volatilities = {}
        risk_budget = portfolio_value * 0.02  # 2% risk per trade
        
        for symbol, pos in self.positions.items():
            if pos != 0:
                # Calculate ATR or standard deviation based volatility
                vol = self.calculate_volatility(symbol)
                volatilities[symbol] = vol
                
        # Adjust position sizes based on volatility
        if volatilities:
            total_vol = sum(volatilities.values())
            return {
                symbol: (risk_budget * vol / total_vol) 
                for symbol, vol in volatilities.items()
            }
        else:
            # Equal position sizes if no volatility data
            return {symbol: portfolio_value * self.position_size / len(self.symbols)
                    for symbol in self.symbols}
                    
    def calculate_volatility(self, symbol: str, window: int = 20) -> float:
        """Calculate volatility measure for position sizing"""
        # Implement your preferred volatility metric
        return 1.0  # Placeholder
        
    def update_parameters(self, new_parameters: Dict) -> None:
        """Update strategy parameters and save state"""
        self.parameters.update(new_parameters)
        self.state.parameters = self.parameters.copy()
        self.state.last_update = pd.Timestamp.now()
        
    def save_state(self) -> StrategyState:
        """Save strategy state for checkpointing"""
        return self.state
        
    def load_state(self, state: StrategyState) -> None:
        """Load strategy state from checkpoint"""
        self.parameters = state.parameters.copy()
        self.positions = state.positions.copy()
        self.state = state

    def preprocess_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Preprocess data before signal generation"""
        processed = {}
        for symbol, df in data.items():
            # Add technical indicators
            df = self.add_indicators(df)
            processed[symbol] = df
        return processed
        
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe"""
        # Override in concrete strategies
        return df

    def finalize_signals(self, signals: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Optionally refine or filter signals after initial generation.
        Override this in concrete strategies if needed.
        """
        return signals