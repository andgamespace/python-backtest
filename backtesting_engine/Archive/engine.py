from typing import Dict, List, Optional, Type
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .DataLoader import DataLoader
from .portfolio import Portfolio
from .strategy import Strategy
import numba

@numba.njit
def process_signals_jit(signal_array, price_array, position_array, cash_array, position_size, slippage):
    """
    JIT-accelerated helper function to process signals.
    signal_array: numpy array of signals
    price_array: numpy array of prices
    position_array: current position quantity
    cash_array: current cash
    position_size: fraction of capital per trade
    slippage: slippage factor
    """
    # This is a simplified example, real usage will involve arrays for each symbol
    for i in range(signal_array.shape[0]):
        s = signal_array[i]
        p = price_array[i]
        tf = p * (1.0 + slippage if s > 0 else 1.0 - slippage)
        # ...simple logic example, real code must handle position logic thoroughly...
        if s > 0 and cash_array[0] > 0:
            trade_val = cash_array[0] * position_size
            shares = trade_val / tf
            position_array[0] += shares
            cash_array[0] -= shares * tf
        elif s < 0 and position_array[0] > 0:
            shares_to_sell = min(position_array[0], cash_array[0] * position_size / tf)
            position_array[0] -= shares_to_sell
            cash_array[0] += shares_to_sell * tf

    return position_array[0], cash_array[0]

class BacktestEngine:
    def __init__(self, data_loader: DataLoader, strategy: Strategy, initial_capital: float = 100000, commission: float = 0.0, interval: str = '5m'):
        self.data_loader = data_loader
        self.strategy = strategy
        self.interval = interval
        self.portfolio = Portfolio(initial_capital, commission)
        self.results = {}
        
    def run(self) -> Dict[str, pd.DataFrame]:
        """
        Run backtest over all symbols provided by the Strategy. The system will:
        1. Extract data for each symbol from the DataLoader.
        2. Generate signals via the Strategy.
        3. For each timestamp in the primary symbol's index, apply signals for all symbols.
        4. Store and return final results in a dictionary of DataFrames.

        If data is missing for the primary symbol or the index is empty,
        the backtest will terminate early.
        """
        data = {}  # Store aligned data for all symbols
        
        # Initialize results storage
        self.results = {symbol: pd.DataFrame() for symbol in self.strategy.symbols}
        
        # Get data for all symbols and align dates
        for symbol in self.strategy.symbols:
            if symbol in self.data_loader.data:
                data[symbol] = self.data_loader.data[symbol]
        
        # Generate signals using strategy
        initial_signals = self.strategy.generate_signals(data)
        signals = self.strategy.finalize_signals(initial_signals)
        
        # Store signals alongside data for plotting
        for symbol in self.strategy.symbols:
            if symbol in data:
                self.results[symbol] = data[symbol].copy()
                self.results[symbol]['signal'] = signals[symbol]
        
        # Ensure the primary symbol has data
        if not data[self.strategy.symbols[0]].index.size:
            print("No data found for the primary symbol. Unable to run backtest.")
            return {}
        
        # Process signals for each timestamp across all symbols
        for timestamp in data[self.strategy.symbols[0]].index:
            for symbol in self.strategy.symbols:
                if symbol not in data:
                    continue
                if signals[symbol][timestamp] != 0:
                    price = data[symbol].loc[timestamp, 'close']
                    signal = signals[symbol][timestamp]
                    self.execute_trade(symbol, signal, price)
            self.portfolio.record_state()
        
        # Example usage of jit function (optional):
        # Convert one symbol's signals to arrays and run JIT for demonstration
        symbol = self.strategy.symbols[0]
        arr_signals = self.results[symbol]['signal'].values.astype(np.int32)
        arr_prices = self.results[symbol]['close'].values.astype(np.float64)
        position_array = np.array([0.0], dtype=np.float64)
        cash_array = np.array([self.portfolio.cash], dtype=np.float64)

        position_array[0], cash_array[0] = process_signals_jit(
            arr_signals,
            arr_prices,
            position_array,
            cash_array,
            self.strategy.position_size,
            self.portfolio.slippage
        )
        # Update portfolio with final results from JIT function
        self.portfolio.cash = float(cash_array[0])
        # This is a simplified example; in a complete system, we'd incorporate
        # JIT across all symbols and timestamps more comprehensively.
        
        results_dict = self.calculate_results()

        # After final data processing, retrieve and display statistics
        stats = self.portfolio.get_statistics()
        print("Performance Summary:")
        for k, v in stats.items():
            print(f"{k.capitalize()}: {v}")

        return results_dict
    
    def execute_trade(self, symbol: str, signal: int, price: float) -> None:
        """
        Execute a trade after applying slippage to the price.
        :param symbol: symbol to trade
        :param signal: positive for buy, negative for sell
        :param price: current market price
        """
        # Apply slippage to simulate real fills
        slippage_factor = 1 + self.portfolio.slippage if signal > 0 else 1 - self.portfolio.slippage
        trade_price = price * slippage_factor

        position_value = self.portfolio.get_total_value() * self.strategy.position_size * abs(signal)
        shares = position_value / trade_price
        
        if signal > 0:  # Buy
            if self.portfolio.cash >= position_value:
                self.portfolio.update_position(symbol, shares, trade_price)
                self.portfolio.cash -= shares * trade_price
        elif signal < 0 and symbol in self.portfolio.positions:  # Sell
            position = self.portfolio.positions[symbol]
            sell_amount = min(position.quantity, shares)
            self.portfolio.cash += sell_amount * trade_price
            self.portfolio.update_position(symbol, -sell_amount, trade_price)
    
    def calculate_results(self) -> Dict[str, pd.DataFrame]:
        """
        Calculate and return backtest results with simpler performance columns.
        """
        results = pd.DataFrame(self.portfolio.history)
        results.set_index('timestamp', inplace=True)
        
        # Calculate returns and metrics
        results['returns'] = results['total_value'].pct_change()
        results['cumulative_returns'] = (1 + results['returns']).cumprod()
        results['cash'] = results['cash'].fillna(method='ffill')
        
        return {'summary': results}
    
    def plot_results(self, summary: pd.DataFrame) -> None:
        """
        Plot price with buy/sell signals for each symbol,
        and plot portfolio cash over time.
        """
        for symbol, df in self.results.items():
            fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            ax[0].plot(df.index, df['close'], label=f'{symbol} Price')
            buys = df[df['signal'] > 0]
            sells = df[df['signal'] < 0]
            ax[0].scatter(buys.index, buys['close'], marker='^', color='g', label='Buy', alpha=0.8)
            ax[0].scatter(sells.index, sells['close'], marker='v', color='r', label='Sell', alpha=0.8)
            ax[0].legend()
            ax[0].set_ylabel('Price')
            
            ax[1].plot(summary.index, summary['cash'], label='Portfolio Cash', color='orange')
            ax[1].legend()
            ax[1].set_ylabel('Cash')
            
            plt.suptitle(f'Results for {symbol}')
            plt.tight_layout()
            plt.show()
    
    def plot_combined_results(self, summary: pd.DataFrame) -> None:
        """
        Plot all symbols on one figure with buy/signals, plus
        a separate figure for total portfolio value.
        """
        import matplotlib.pyplot as plt
        
        # One figure for all symbols
        fig, ax = plt.subplots(figsize=(10, 6))
        for symbol, df in self.results.items():
            ax.plot(df.index, df['close'], label=f'{symbol} Price')
            buys = df[df['signal'] > 0]
            sells = df[df['signal'] < 0]
            ax.scatter(buys.index, buys['close'], marker='^', color='g', alpha=0.6)
            ax.scatter(sells.index, sells['close'], marker='v', color='r', alpha=0.6)
        ax.legend()
        ax.set_title("Combined Stock Prices with Signals")
        ax.set_ylabel("Price")
        plt.tight_layout()
        plt.show()

        # Another figure for total portfolio value
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(summary.index, summary['total_value'], label='Total Portfolio Value', color='blue')
        ax.legend()
        ax.set_ylabel("Value")
        ax.set_title("Portfolio Value Over Time")
        plt.tight_layout()
        plt.show()

    def plot_performance(self, summary: pd.DataFrame) -> None:
        """
        Plot portfolio total value vs. time in one figure,
        and each symbol's price plus buy/sell signals in another.
        """
        import matplotlib.pyplot as plt

        # 1) Portfolio value over time
        fig, ax = plt.subplots(figsize=(10, 4))
        if 'total_value' not in summary.columns:
            print("No 'total_value' in summary, cannot plot portfolio performance.")
            return
        ax.plot(summary.index, summary['total_value'], label='Portfolio Value', color='blue')
        ax.set_title("Portfolio Value Over Time")
        ax.set_ylabel("Value")
        ax.legend()
        plt.tight_layout()
        plt.show()

        # 2) Each symbol's price with signals
        fig, ax = plt.subplots(figsize=(10, 6))
        for symbol, df in self.results.items():
            if 'close' in df.columns:
                ax.plot(df.index, df['close'], label=f'{symbol} Price')
                buys = df[df['signal'] > 0]
                sells = df[df['signal'] < 0]
                ax.scatter(buys.index, buys['close'], marker='^', color='g', alpha=0.6)
                ax.scatter(sells.index, sells['close'], marker='v', color='r', alpha=0.6)
        ax.set_title("Symbol Prices with Buy/Sell Signals")
        ax.set_ylabel("Price")
        ax.legend()
        plt.tight_layout()
        plt.show()