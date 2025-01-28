import numpy as np
import pandas as pd
from numba import jit
from typing import Dict, List, Optional
from .data_handler import DataHandler
from .strategies import BaseStrategy
from .portfolio import Portfolio

class BacktestEngine:
    """
    Main backtesting engine with performance optimizations.
    """
    def __init__(self,
                 data_handler: DataHandler,
                 strategy: BaseStrategy,
                 initial_capital: float = 100000.0,
                 commission: float = 0.001):
        self.data_handler = data_handler
        self.strategy = strategy
        self.commission = commission
        self.portfolio = Portfolio(initial_capital)
        self.history: List[Dict] = []

    def _execute_trade(self, symbol: str, signal: float, price: float) -> None:
        """
        Execute a trade based on the signal and update portfolio.
        """
        if signal == 0:
            return

        current_position = self.portfolio.positions.get(symbol, 0)
        trade_value = self.portfolio.cash if signal > 0 else abs(current_position * price)
        quantity = (trade_value / price) * (1 - self.commission) * np.sign(signal)
        
        self.portfolio.cash -= quantity * price * (1 + self.commission)
        self.portfolio.update_position(symbol, quantity, price)

    def run(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Run the backtest and return results.
        """
        results = {}
        
        for symbol in self.strategy.symbols:
            data = self.data_handler.data[symbol]
            
            if start_date:
                data = data[data['datetime'] >= start_date]
            if end_date:
                data = data[data['datetime'] <= end_date]
                
            signals = self.strategy.generate_signals(data)
            returns, positions = self._calculate_returns(
                data['close'].values,
                signals,
                self.initial_capital,
                self.commission
            )
            
            results[symbol] = pd.DataFrame({
                'datetime': data['datetime'],
                'close': data['close'],
                'signal': signals,
                'position': positions,
                'returns': returns
            })
            
        return results