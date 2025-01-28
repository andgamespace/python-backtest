from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

class Strategy(ABC):
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.position_size = 1.0  # Default full position size
        
    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        Generate trading signals for each symbol
        Returns:
            Dict[str, pd.Series]: Dictionary of signals per symbol
            where signals are:
            1 = Buy
            0 = Hold
            -1 = Sell
        """
        pass
        
    def set_position_size(self, size: float) -> None:
        """Set position size as percentage (0.0 to 1.0)"""
        self.position_size = max(0.0, min(1.0, size))

class MovingAverageCrossover(Strategy):
    def __init__(self, symbols: List[str], short_window: int = 20, long_window: int = 50):
        super().__init__(symbols)
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        signals = {}
        for symbol, df in data.items():
            short_ma = df['close'].rolling(window=self.short_window).mean()
            long_ma = df['close'].rolling(window=self.long_window).mean()
            
            signals[symbol] = pd.Series(0, index=df.index)
            signals[symbol][short_ma > long_ma] = 1
            signals[symbol][short_ma < long_ma] = -1
            
        return signals