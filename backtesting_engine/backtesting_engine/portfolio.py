from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class Position:
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float = 0.0

class Portfolio:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.history: List[Dict] = []
        
    def update_position(self, symbol: str, quantity: float, price: float) -> None:
        """Update or create a position for a symbol"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol, quantity, price)
        else:
            position = self.positions[symbol]
            new_quantity = position.quantity + quantity
            if new_quantity == 0:
                del self.positions[symbol]
            else:
                # Update average entry price only for buys
                if quantity > 0:
                    total_cost = (position.quantity * position.avg_entry_price) + (quantity * price)
                    position.avg_entry_price = total_cost / new_quantity
                position.quantity = new_quantity
                
    def get_position_value(self, symbol: str) -> float:
        """Get current value of a position"""
        if symbol in self.positions:
            position = self.positions[symbol]
            return position.quantity * position.current_price
        return 0.0
        
    def get_total_value(self) -> float:
        """Get total portfolio value including cash"""
        return self.cash + sum(self.get_position_value(symbol) for symbol in self.positions)
    
    def record_state(self) -> None:
        """Record current portfolio state"""
        self.history.append({
            'timestamp': pd.Timestamp.now(),
            'cash': self.cash,
            'total_value': self.get_total_value(),
            'positions': {s: p.quantity for s, p in self.positions.items()}
        })