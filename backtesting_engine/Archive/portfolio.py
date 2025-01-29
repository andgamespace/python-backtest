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
    def __init__(self, initial_capital: float = 100000, commission: float = 0.0, 
                 slippage: float = 0.0, daily_factor: int = 252):
        """
        :param initial_capital: starting cash
        :param commission: fixed percentage for each trade
        :param slippage: artificial price change to simulate real market conditions
        :param daily_factor: factor to adjust Sharpe ratio calculation
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.daily_factor = daily_factor
        self.positions: Dict[str, Position] = {}
        self.history: List[Dict] = []
        
    def update_position(self, symbol: str, quantity: float, price: float) -> None:
        """
        Update position with slippage and commission.
        :param symbol: symbol to trade
        :param quantity: number of shares to buy (positive) or sell (negative)
        :param price: reference price for the trade
        """
        trade_cost = quantity * price
        commission_cost = abs(trade_cost) * self.commission
        self.cash -= commission_cost
        
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

    def get_statistics(self) -> dict:
        """
        Compute and return portfolio performance metrics such as Sharpe ratio,
        maximum drawdown, and total return. Assumes 'history' contains timestamps
        and 'total_value' entries for the entire backtest.
        """
        # Create a DataFrame from history
        df = pd.DataFrame(self.history).set_index('timestamp')
        df['returns'] = df['total_value'].pct_change().fillna(0.0)

        # Calculate cumulative returns
        df['cumulative'] = (1 + df['returns']).cumprod()

        # Sharpe ratio (assuming daily data, adjust if needed)
        # Using a simple approach: (mean(returns) / std(returns)) * sqrt(periods)
        # For intraday data (like 5m), adapt the scale factor accordingly
        returns_mean = df['returns'].mean()
        returns_std = df['returns'].std() if df['returns'].std() != 0 else 1e-9
        sharpe_ratio = (returns_mean / returns_std) * np.sqrt(self.daily_factor)

        # Max drawdown
        rolling_max = df['cumulative'].cummax()
        drawdown = (df['cumulative'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Total return
        total_return = df['cumulative'].iloc[-1] - 1.0 if len(df) > 0 else 0.0

        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return': total_return
        }

    def reset_portfolio(self) -> None:
        """
        Reset the portfolio to its initial state, clearing all positions and history.
        Useful when running multiple sequential backtests in one session.
        """
        self.cash = self.initial_capital
        self.positions.clear()
        self.history.clear()