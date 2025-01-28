from abc import ABC, abstractmethod
from typing import Dict
from .portfolio import Portfolio

class RiskManager(ABC):
    @abstractmethod
    def check_trade(self, portfolio: Portfolio, symbol: str, quantity: float, price: float) -> bool:
        pass

class BasicRiskManager(RiskManager):
    def __init__(self, max_position_size: float = 0.2, max_drawdown: float = 0.1):
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown

    def check_trade(self, portfolio: Portfolio, symbol: str, quantity: float, price: float) -> bool:
        # Check position size
        trade_value = abs(quantity * price)
        if trade_value / portfolio.total_value > self.max_position_size:
            return False
            
        # Check drawdown
        if (portfolio.initial_capital - portfolio.total_value) / portfolio.initial_capital > self.max_drawdown:
            return False
            
        return True
