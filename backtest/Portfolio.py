import logging
from typing import Dict, Optional, List
import pandas as pd
from .Position import Position
import numpy as np

class Portfolio:
    """
    Holds multiple Positions, tracks account value, PnL, cash, etc.
    """

    def __init__(self, initial_cash: float = 100_000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.history = []
        self.trade_log: List[tuple] = []  # (ticker, 'BUY'/'SELL', quantity, price, index)
        self.data_loader = None  # Reference to DataLoader for visualization
        self.logger = self._setup_logger()
        self.logger.info(f"Portfolio initialized with initial_cash={self.initial_cash}")

    def _setup_logger(self):
        logger = logging.getLogger('Portfolio')
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(ch)
            logger.setLevel(logging.INFO)
        return logger

    def set_data_loader(self, data_loader):
        """
        Set the DataLoader reference for accessing data during visualization.
        """
        self.data_loader = data_loader

    def handle_signal(self, ticker, signal, current_price, index):
        """
        Takes a signal from the Engine and updates positions accordingly.
        """
        self.logger.info(f"Handling signal '{signal}' for ticker '{ticker}' at price {current_price}")
        if signal == 'BUY':
            self._open_or_add_position(ticker, current_price, index)
        elif signal == 'SELL':
            self._close_or_reduce_position(ticker, current_price, index)

    def _open_or_add_position(self, ticker, price, index):
        """
        Example logic for a market buy.
        """
        quantity = 10  # Simplified logic: Buy 10 shares on each signal
        cost = price * quantity

        if cost > self.cash:
            self.logger.info(f"Not enough cash to buy {quantity} shares of {ticker}.")
            return

        if ticker not in self.positions:
            self.positions[ticker] = Position(ticker=ticker, quantity=0, entry_price=0)
            self.logger.info(f"Opened new position for {ticker}.")

        # Weighted average price if adding to existing position
        existing_qty = self.positions[ticker].quantity
        new_qty = existing_qty + quantity
        new_cost = (existing_qty * self.positions[ticker].entry_price) + cost
        avg_price = new_cost / new_qty

        self.positions[ticker].quantity = new_qty
        self.positions[ticker].entry_price = avg_price
        self.cash -= cost
        self.trade_log.append((ticker, 'BUY', quantity, price, index))

        self.logger.info(f"Bought {quantity} shares of {ticker} at {price}. "
                         f"New quantity: {new_qty}, average price: {avg_price}")

    def _close_or_reduce_position(self, ticker, price, index):
        """
        Example logic for a market sell, selling entire position by default.
        """
        if ticker not in self.positions or self.positions[ticker].quantity <= 0:
            self.logger.info(f"No existing position in {ticker} to sell.")
            return

        quantity_to_sell = self.positions[ticker].quantity
        proceeds = price * quantity_to_sell

        self.cash += proceeds
        self.trade_log.append((ticker, 'SELL', quantity_to_sell, price, index))

        self.logger.info(f"Sold {quantity_to_sell} shares of {ticker} at {price}. "
                         f"Cash += {proceeds}")
        # Reset position
        self.positions[ticker].quantity = 0
        self.positions[ticker].entry_price = 0
        self.logger.info(f"Position for {ticker} closed.")

    def calculate_final_metrics(self):
        """
        Print out or return final portfolio metrics, e.g., total PnL, final holdings, etc.
        """
        total_portfolio_value = self.cash
        for ticker, position in self.positions.items():
            if position.quantity > 0:
                current_price = position.entry_price  # Assuming current price equals entry price for simplicity
                total_portfolio_value += position.quantity * current_price

        pnl = total_portfolio_value - self.initial_cash
        self.logger.info(f"Final Portfolio Value: {total_portfolio_value}")
        self.logger.info(f"Total PnL: {pnl}")

    def total_value(self) -> float:
        """Calculate total portfolio value."""
        total = self.cash + sum(pos.market_value() for pos in self.positions.values())
        self.logger.info(f"Total portfolio value calculated: {total}")
        return total

    def can_trade(self, ticker: str, quantity: int, price: float) -> bool:
        """Check if trade is possible given current cash."""
        cost = quantity * price
        can_trade = self.cash >= cost
        self.logger.info(f"Can trade {'Yes' if can_trade else 'No'} for {quantity} shares of {ticker} at {price}.")
        return can_trade

    def execute_trade(self, ticker: str, quantity: int, price: float, index: int) -> bool:
        """Execute a trade (positive quantity for buy, negative for sell)."""
        cost = quantity * price

        if quantity > 0 and not self.can_trade(ticker, quantity, price):
            self.logger.info(f"Trade aborted: Not enough cash to buy {quantity} shares of {ticker}.")
            return False

        if ticker not in self.positions:
            self.positions[ticker] = Position(ticker=ticker)
            self.logger.info(f"Opened new position for {ticker}.")

        position = self.positions[ticker]
        
        # Update position
        new_quantity = position.quantity + quantity
        if new_quantity == 0:
            del self.positions[ticker]
            self.logger.info(f"Position for {ticker} closed.")
        else:
            position.quantity = new_quantity
            position.entry_price = price  # Simplified - could use average price
            self.logger.info(f"Updated position for {ticker}: quantity={new_quantity}, entry_price={price}")

        # Update cash
        self.cash -= cost
        self.logger.info(f"Executed trade for {ticker}: quantity={quantity}, price={price}. New cash balance: {self.cash}")

        # Record trade with index
        self.trade_log.append((ticker, 'BUY' if quantity > 0 else 'SELL', quantity, price, index))

        # Record history with proper datetime
        self.history.append({
            'timestamp': pd.to_datetime('now'),
            'ticker': ticker,
            'quantity': quantity,
            'price': price,
            'cash': self.cash,
            'portfolio_value': self.total_value()
        })
        
        return True

    def get_historical_value(self) -> pd.DataFrame:
        """Get historical portfolio value as DataFrame."""
        self.logger.info("Retrieving historical portfolio values.")
        return pd.DataFrame(self.history)