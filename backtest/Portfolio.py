import logging
from typing import Dict, Optional, List
import pandas as pd
from .Position import Position
import numpy as np
import random
from .utils import risk_management
from .Orders import Order, OrderType # Import Order and OrderType

class Portfolio:
    """
    Holds multiple Positions, tracks account value, PnL, cash, etc.
    """

    def __init__(self, initial_cash: float = 100_000, slippage_rate: float = 0.0025, max_drawdown: Optional[float] = None, volatility_threshold: Optional[float] = None, risk_free_rate: float = 0.02): # Added risk_free_rate
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.history: List[dict] = []
        self.portfolio_value_history: pd.Series = pd.Series()
        self.trade_log: List[tuple] = []
        self.data_loader = None
        self.logger = self._setup_logger()
        self.slippage_rate = slippage_rate
        self.max_drawdown = max_drawdown
        self.volatility_threshold = volatility_threshold
        self.risk_free_rate = risk_free_rate # Annual risk-free rate
        # New: Order management
        self.pending_orders: List[Order] = [] # List to hold pending orders
        self.logger.info(f"Portfolio initialized with initial_cash={self.initial_cash}, slippage_rate={self.slippage_rate}, max_drawdown={self.max_drawdown}, volatility_threshold={self.volatility_threshold}, risk_free_rate={self.risk_free_rate}")

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

    def handle_signal(self, ticker, signal, current_price, index, order_type=OrderType.MARKET, limit_price=None, stop_price=None): # Added order_type, limit_price, stop_price
        """
        Takes a signal from the Engine and updates positions accordingly.
        Now accepts order_type and order prices.
        """
        self.logger.info(f"Handling signal '{signal}' for ticker '{ticker}' at price {current_price}, order_type: {order_type}")
        if signal == 'BUY':
            order_price = limit_price if order_type == OrderType.LIMIT else stop_price if order_type == OrderType.STOP else None # Determine order price based on order type
            order = Order(order_type=order_type, ticker=ticker, quantity=10, price=order_price, stop_price=stop_price) # Create Order object, use order_price
            if order_type == OrderType.MARKET:
                self._execute_market_order(order, current_price, index) # Execute market order immediately
            else:
                self.pending_orders.append(order) # Add limit/stop order to pending orders
        elif signal == 'SELL':
            order_price = limit_price if order_type == OrderType.LIMIT else stop_price if order_type == OrderType.STOP else None # Determine order price based on order type
            order = Order(order_type=order_type, ticker=ticker, quantity=-10, price=order_price, stop_price=stop_price) # Negative quantity for sell, use order_price
            if order_type == OrderType.MARKET:
                self._execute_market_order(order, current_price, index) # Execute market order immediately
            else:
                self.pending_orders.append(order) # Add limit/stop order to pending orders

    def _execute_market_order(self, order: Order, current_price, index): # New method to execute market orders
        """
        Executes a market order immediately.
        """
        if order.quantity > 0: # Buy order
            self._open_or_add_position(order.ticker, current_price, order.quantity, index)
        elif order.quantity < 0: # Sell order
            self._close_or_reduce_position(order.ticker, current_price, abs(order.quantity), index) # Use abs for quantity to sell

    def _apply_slippage(self, price, order_type):
        """
        Applies slippage to the order price based on slippage rate and order type.
        """
        random_factor = random.uniform(-1, 1)
        slippage_amount = price * self.slippage_rate * random_factor
        execution_price = price + slippage_amount

        if order_type == 'BUY':
            execution_price = max(0, execution_price)
        elif order_type == 'SELL':
            execution_price = max(0, execution_price)

        self.logger.debug(f"Slippage applied: Order Type: {order_type}, Base Price: {price}, Slippage Rate: {self.slippage_rate}, Random Factor: {random_factor:.4f}, Slippage Amount: {slippage_amount:.4f}, Execution Price: {execution_price:.4f}")
        return execution_price

    def _open_or_add_position(self, ticker, price, quantity, index): # Modified to accept quantity
        """
        Logic for opening or adding to a position. Now accepts quantity from order.
        """
        execution_price = self._apply_slippage(price, 'BUY')
        cost = execution_price * quantity

        # Risk Management Check before opening position
        if not risk_management(
            position_size=quantity,
            account_balance=self.cash,
            portfolio_history=self.portfolio_value_history,
            max_drawdown=self.max_drawdown,
            volatility_threshold=self.volatility_threshold,
            current_price=execution_price,
            entry_price=self.positions.get(ticker, Position(ticker)).entry_price if ticker in self.positions else None
        ):
            return

        if cost > self.cash:
            self.logger.info(f"Not enough cash to buy {quantity} shares of {ticker} at execution price {execution_price}.")
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
        self.trade_log.append((ticker, 'BUY', quantity, price, execution_price, index))

        self.logger.info(f"Bought {quantity} shares of {ticker} at price {price}, execution price {execution_price}. "
                         f"New quantity: {new_qty}, average price: {avg_price}")
        self._update_portfolio_history()

    def _close_or_reduce_position(self, ticker, price, quantity, index): # Modified to accept quantity
        """
        Logic for closing or reducing a position. Now accepts quantity from order.
        """
        if ticker not in self.positions or self.positions[ticker].quantity <= 0:
            self.logger.info(f"No existing position in {ticker} to sell.")
            return

        quantity_to_sell = min(self.positions[ticker].quantity, quantity) # Ensure not selling more than owned
        execution_price = self._apply_slippage(price, 'SELL')
        proceeds = execution_price * quantity_to_sell

        # Risk Management Check before closing position
        if not risk_management(
            position_size=quantity_to_sell,
            account_balance=self.cash + proceeds,
            portfolio_history=self.portfolio_value_history,
            max_drawdown=self.max_drawdown,
            volatility_threshold=self.volatility_threshold,
            current_price=execution_price,
            entry_price=self.positions[ticker].entry_price
        ):
            return

        self.cash += proceeds
        self.trade_log.append((ticker, 'SELL', quantity_to_sell, price, execution_price, index))

        self.logger.info(f"Sold {quantity_to_sell} shares of {ticker} at price {price}, execution price {execution_price}. "
                         f"Cash += {proceeds}")

        self.positions[ticker].quantity -= quantity_to_sell
        if self.positions[ticker].quantity == 0:
            self.positions[ticker].entry_price = 0
            self.logger.info(f"Position for {ticker} closed.")
        else:
            self.logger.info(f"Position for {ticker} reduced, remaining quantity: {self.positions[ticker].quantity}")
        self._update_portfolio_history()

    def calculate_final_metrics(self):
        """
        Print out or return final portfolio metrics.
        """
        total_portfolio_value = self.total_value() # Use total_value method to get current portfolio value including positions
        pnl = total_portfolio_value - self.initial_cash

        # Calculate portfolio returns
        portfolio_values = pd.Series([item['portfolio_value'] for item in self.history])
        if len(portfolio_values) < 2: # Need at least two points to calculate returns
            self.logger.warning("Insufficient portfolio history to calculate metrics.")
            return

        portfolio_returns = portfolio_values.pct_change().dropna()
        if portfolio_returns.empty:
            self.logger.warning("No portfolio returns to calculate metrics.")
            return

        # Annualize returns and risk-free rate (assuming daily data, adjust if needed)
        annualization_factor = 252 # Trading days in a year
        annual_return = portfolio_returns.mean() * annualization_factor
        excess_returns = portfolio_returns - (self.risk_free_rate / annualization_factor) # Daily excess returns

        # Sharpe Ratio
        sharpe_ratio = excess_returns.mean() / portfolio_returns.std() * np.sqrt(annualization_factor)

        # Sortino Ratio (Downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(annualization_factor) if not downside_returns.empty else np.nan
        sortino_ratio = excess_returns.mean() / downside_deviation if downside_deviation else np.nan

        # Maximum Drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown_val = drawdown.min()

        # Calmar Ratio
        max_drawdown_abs = abs(max_drawdown_val) if not pd.isna(max_drawdown_val) else np.nan
        calmar_ratio = annual_return / max_drawdown_abs if max_drawdown_abs != 0 and not pd.isna(max_drawdown_abs) else np.nan


        self.logger.info(f"Final Portfolio Value: {total_portfolio_value:.2f}")
        self.logger.info(f"Total PnL: {pnl:.2f}")
        self.logger.info(f"Annual Return: {annual_return:.4f}")
        self.logger.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        self.logger.info(f"Sortino Ratio: {sortino_ratio:.4f}")
        self.logger.info(f"Maximum Drawdown: {max_drawdown_val:.4f}")
        self.logger.info(f"Calmar Ratio: {calmar_ratio:.4f}")

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

    def execute_trade(self, ticker: str, quantity: int, price: float, index: int, order_type=OrderType.MARKET, limit_price=None, stop_price=None) -> bool: # Added order_type, limit_price, stop_price
        """Execute a trade (positive quantity for buy, negative for sell) with slippage and risk management."""
        if quantity == 0:
            self.logger.info("Trade aborted: Quantity is zero.")
            return False

        if quantity > 0:
            execution_price = self._apply_slippage(price, 'BUY')
            trade_type = 'BUY'
        else:
            execution_price = self._apply_slippage(price, 'SELL')
            trade_type = 'SELL'

        cost = abs(quantity) * execution_price

        # Risk Management Check before executing trade
        if not risk_management(
            position_size=abs(quantity),
            account_balance=self.cash if quantity > 0 else self.cash + cost,
            portfolio_history=self.portfolio_value_history,
            max_drawdown=self.max_drawdown,
            volatility_threshold=self.volatility_threshold,
            current_price=execution_price,
            entry_price=self.positions.get(ticker, Position(ticker)).entry_price if ticker in self.positions and quantity > 0 else price
        ):
            return False

        if quantity > 0 and not self.can_trade(ticker, quantity, execution_price):
            self.logger.info(f"Trade aborted: Not enough cash to buy {quantity} shares of {ticker} at execution price {execution_price}.")
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
            position.entry_price = price  # Simplified - could use average price, but using original price for entry point
            self.logger.info(f"Updated position for {ticker}: quantity={new_quantity}, entry_price={price}")

        # Update cash
        self.cash -= cost if quantity > 0 else -cost
        self.logger.info(f"Executed trade for {ticker} (Order Type: {order_type}): quantity={quantity}, price={price}, execution_price={execution_price}. New cash balance: {self.cash}")

        # Record trade with execution_price in trade_log
        self.trade_log.append((ticker, trade_type, quantity, price, execution_price, index))

        # Record history with proper datetime
        self.history.append({
            'timestamp': pd.to_datetime('now'),
            'ticker': ticker,
            'quantity': quantity,
            'price': price,
            'execution_price': execution_price,
            'cash': self.cash,
            'portfolio_value': self.total_value()
        })
        self._update_portfolio_history()

        return True

    def get_historical_value(self) -> pd.DataFrame:
        """Get historical portfolio value as DataFrame."""
        self.logger.info("Retrieving historical portfolio values.")
        return pd.DataFrame(self.history)

    def _update_portfolio_history(self):
        """Updates the portfolio value history."""
        historical_df = self.get_historical_value()
        if not historical_df.empty:
            self.portfolio_value_history = historical_df.set_index('timestamp')['portfolio_value']
        else:
            self.portfolio_value_history = pd.Series()

    def process_orders(self, current_time, current_prices): # Placeholder for order processing logic
        """
        Process pending orders (LIMIT, STOP). To be implemented.
        This would be called at each time step to check and execute pending orders.
        """
        orders_to_execute = []
        remaining_pending_orders = []

        for order in self.pending_orders:
            ticker_price = current_prices.get(order.ticker)
            if ticker_price is None:
                remaining_pending_orders.append(order) # Keep pending if no price data
                continue

            if order.order_type == OrderType.LIMIT:
                if order.quantity > 0 and ticker_price <= order.price: # Limit BUY order triggered
                    orders_to_execute.append(order)
                elif order.quantity < 0 and ticker_price >= order.price: # Limit SELL order triggered
                    orders_to_execute.append(order)
                else:
                    remaining_pending_orders.append(order) # Not triggered yet

            elif order.order_type == OrderType.STOP:
                if order.quantity > 0 and ticker_price >= order.price: # Stop BUY order triggered
                    orders_to_execute.append(order)
                elif order.quantity < 0 and ticker_price <= order.price: # Stop SELL order triggered
                    orders_to_execute.append(order)
                else:
                    remaining_pending_orders.append(order) # Not triggered yet
            else: # MARKET orders should not be in pending_orders, but just in case
                orders_to_execute.append(order)

        self.pending_orders = remaining_pending_orders # Update pending orders

        for order in orders_to_execute:
            execution_price = current_prices[order.ticker] # Use current price for execution
            self._execute_market_order(order, execution_price, -1) # Assuming index doesn't matter here, using -1

        if orders_to_execute:
            executed_tickers = ", ".join([order.ticker for order in orders_to_execute])
            self.logger.info(f"Processed and executed {len(orders_to_execute)} pending orders for tickers: {executed_tickers} at time {current_time}.")