from dataclasses import dataclass
from typing import Dict, List
import numpy as np

@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.current_price - self.entry_price)

class Portfolio:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
    
    def update_position(self, symbol: str, quantity: float, price: float) -> None:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol, quantity, price, price)
        else:
            pos = self.positions[symbol]
            new_quantity = pos.quantity + quantity
            if new_quantity == 0:
                del self.positions[symbol]
            else:
                # Calculate new average entry price if adding to position
                if quantity > 0:
                    total_cost = (pos.quantity * pos.entry_price) + (quantity * price)
                    new_entry_price = total_cost / new_quantity
                    self.positions[symbol] = Position(symbol, new_quantity, new_entry_price, price)
                else:
                    self.positions[symbol] = Position(symbol, new_quantity, pos.entry_price, price)

    @property
    def total_value(self) -> float:
        return self.cash + sum(pos.market_value for pos in self.positions.values())
