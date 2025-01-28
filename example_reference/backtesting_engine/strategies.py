from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from numba import jit
from typing import Dict, List, Tuple

class BaseStrategy(ABC):
    """
    Base class for all trading strategies.
    """
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.positions: Dict[str, float] = {symbol: 0.0 for symbol in symbols}
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate trading signals for the given data.
        Returns array of signals: 1 (buy), -1 (sell), 0 (hold)
        """
        pass

class MovingAverageCrossover(BaseStrategy):
    """
    Example strategy using Moving Average Crossover with Numba optimization.
    """
    def __init__(self, symbols: List[str], short_window: int = 20, long_window: int = 50):
        super().__init__(symbols)
        self.short_window = short_window
        self.long_window = long_window
    
    @staticmethod
    @jit(nopython=True)
    def _compute_signals(prices: np.ndarray, short_window: int, long_window: int) -> np.ndarray:
        """
        Compute MA crossover signals using Numba for performance.
        """
        n = len(prices)
        signals = np.zeros(n)
        
        if n < long_window:
            return signals
            
        # Calculate moving averages
        short_ma = np.zeros(n)
        long_ma = np.zeros(n)
        
        for i in range(short_window-1, n):
            short_ma[i] = np.mean(prices[i-short_window+1:i+1])
        
        for i in range(long_window-1, n):
            long_ma[i] = np.mean(prices[i-long_window+1:i+1])
            
        # Generate signals
        for i in range(long_window, n):
            if short_ma[i] > long_ma[i] and short_ma[i-1] <= long_ma[i-1]:
                signals[i] = 1  # Buy signal
            elif short_ma[i] < long_ma[i] and short_ma[i-1] >= long_ma[i-1]:
                signals[i] = -1  # Sell signal
                
        return signals
    
    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        prices = data['close'].values
        return self._compute_signals(prices, self.short_window, self.long_window)