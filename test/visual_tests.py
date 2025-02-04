import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import backtest
from backtest import DataLoader, SimpleMovingAverageStrategy, Portfolio, Engine
from backtest import RSIStrategy, MACDStrategy, BollingerBandsStrategy
from backtest.visuals import plot_signals, plot_portfolio, plot_strategy_results, plot_portfolio_over_time, plot_all_strategies_results

# Force a specific backend (Try 'TkAgg' or 'QtAgg' if 'MacOSX' doesn't work)
plt.switch_backend('MacOSX')  # Or try 'TkAgg', 'QtAgg' - see comment below

def run_visual_tests():
    try: # Added try-except block to catch any errors during plotting
        np.random.seed(42)  # For reproducibility
        data_loader = DataLoader()
        structure = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        sep = ';'

        # Define file paths (same as in tests.py)
        stock_file_paths = {
            'AMD': [
                '/Users/anshc/repos/python-backtest/test/stock_data/time-series-AMD-5min.csv',
                '/Users/anshc/repos/python-backtest/test/stock_data/time-series-AMD-5min(1).csv',
                '/Users/anshc/repos/python-backtest/test/stock_data/time-series-AMD-5min(2).csv',
            ],
            'NVDA': [
                '/Users/anshc/repos/python-backtest/test/stock_data/time-series-NVDA-5min.csv',
                '/Users/anshc/repos/python-backtest/test/stock_data/time-series-NVDA-5min(1).csv',
                '/Users/anshc/repos/python-backtest/test/stock_data/time-series-NVDA-5min(2).csv',
            ],
            'AAPL': [
                '/Users/anshc/repos/python-backtest/test/stock_data/time-series-AAPL-5min.csv',
                '/Users/anshc/repos/python-backtest/test/stock_data/time-series-AAPL-5min(1).csv',
                '/Users/anshc/repos/python-backtest/test/stock_data/time-series-AAPL-5min(2).csv',
            ],
            'MSFT': [
                '/Users/anshc/repos/python-backtest/test/stock_data/time-series-MSFT-5min.csv',
                '/Users/anshc/repos/python-backtest/test/stock_data/time-series-MSFT-5min(1).csv',
                '/Users/anshc/repos/python-backtest/test/stock_data/time-series-MSFT-5min(2).csv',
            ],
        }

        # Load data for all tickers
        for ticker, paths in stock_file_paths.items():
            data_loader.load_ticker(ticker, paths, structure, sep)

        # Initialize strategies (same as in tests.py)
        strategies = [
            SimpleMovingAverageStrategy(short_window=5, long_window=20),
            RSIStrategy(rsi_low=30, rsi_high=70),
            MACDStrategy(),
            BollingerBandsStrategy(),
        ]

        # Initialize a single portfolio for all strategies
        portfolio = Portfolio(initial_cash=100000, max_drawdown=0.1, volatility_threshold=0.05, risk_free_rate=0.02)
        portfolio.set_data_loader(data_loader)

        # Initialize engines for each strategy, sharing the same portfolio
        engines = {
            strategy.__class__.__name__: Engine(
                data_loader=data_loader,
                portfolio=portfolio,
                strategy=strategy
            ) for strategy in strategies
        }

        tickers = list(stock_file_paths.keys())

        # Run backtest for all strategies and tickers (like in test_run_backtest_all_strategies)
        for strategy_name, engine in engines.items():
            engine.run_backtest(tickers)

        # --- Visualizations ---

        # 1. Example plot_signals (using dummy data or first ticker data)
        if 'AMD' in data_loader.data:
            df_amd = data_loader.data['AMD']
            sample_signals = [(50, 'BUY'), (100, 'SELL'), (150, 'BUY')] # Example signals
            plot_signals(df_amd.iloc[20:200].reset_index(drop=True), sample_signals) # Plot a slice of data

        # 2. Example plot_portfolio
        plot_portfolio(portfolio.portfolio_value_history)

        # 3. Example plot_strategy_results (for SMAStrategy and AMD)
        if 'SMAStrategy' in engines and 'AMD' in data_loader.data:
            plot_strategy_results(engines['SMAStrategy'].portfolio, 'AMD', 'SMAStrategy')

        # 4. Example plot_portfolio_over_time (for SMAStrategy)
        if 'SMAStrategy' in engines:
            plot_portfolio_over_time(engines['SMAStrategy'].portfolio, 'SMAStrategy')

        # 5. Example plot_all_strategies_results
        plot_all_strategies_results(engines, tickers)

        plt.show() # Ensure all plots are shown before pausing - moved outside loop

        input("Press Enter to close plot windows and exit...") # Pause script to keep plots open

    except Exception as e:
        print(f"An error occurred during visual tests: {e}") # Catch and print any errors

if __name__ == '__main__':
    run_visual_tests()
    print("Visual tests script execution finished.") # Message after execution ends