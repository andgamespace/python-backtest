import unittest
import pandas as pd
import numpy as np
from backtest import DataLoader, SimpleMovingAverageStrategy, Portfolio, Engine
from backtest import RSIStrategy, MACDStrategy, BollingerBandsStrategy

class TestBacktestingFramework(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)  # For reproducibility
        self.data_loader = DataLoader()
        self.structure = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        self.sep = ';'

        # Define file paths
        self.stock_file_paths = {
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
        for ticker, paths in self.stock_file_paths.items():
            self.data_loader.load_ticker(ticker, paths, self.structure, self.sep)

        # Initialize strategies
        self.strategies = [
            SimpleMovingAverageStrategy(short_window=5, long_window=20),
            RSIStrategy(rsi_low=30, rsi_high=70),
            MACDStrategy(),
            BollingerBandsStrategy(),
        ]

        # Initialize a single portfolio for all strategies
        self.portfolio = Portfolio(initial_cash=100000)
        self.portfolio.set_data_loader(self.data_loader)

        # Initialize engines for each strategy, sharing the same portfolio
        self.engines = {
            strategy.__class__.__name__: Engine(
                data_loader=self.data_loader, 
                portfolio=self.portfolio, 
                strategy=strategy
            ) for strategy in self.strategies
        }

    def test_run_backtest_all_strategies(self):
        """
        Test the complete backtesting workflow for all strategies and tickers.
        """
        tickers = list(self.stock_file_paths.keys())
        for strategy_name, engine in self.engines.items():
            try:
                engine.run_backtest(tickers)
            except Exception as e:
                self.fail(f"Backtest run failed for {strategy_name} with exception: {e}")

            # Assertions to verify portfolio updates
            final_cash = self.portfolio.cash
            self.assertTrue(final_cash <= 100000, f"Final cash for {strategy_name} should not exceed initial cash without profits.")

    def test_data_loading_real_tickers(self):
        """
        Test if data is loaded correctly for real tickers.
        """
        for ticker in self.stock_file_paths.keys():
            self.assertIn(ticker, self.data_loader.data)
            df = self.data_loader.data[ticker]
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0)
            for column in self.structure:
                self.assertIn(column, df.columns)

    def test_portfolio_initialization(self):
        """
        Test Portfolio initialization.
        """
        portfolio = Portfolio(initial_cash=50000)
        self.assertEqual(portfolio.cash, 50000)
        self.assertEqual(len(portfolio.positions), 0)

    def test_strategy_generation(self):
        """
        Test signal generation of the strategies.
        """
        # Create sample data with consistent lengths
        sample_length = 5
        dates = pd.date_range(start='2020-01-01', periods=sample_length)
        
        # Simulate market data up to a specific point for RSIStrategy
        rsi_market_data = {
            'close': 25,
            'df': pd.DataFrame({
                'datetime': dates,
                'close': [28, 27, 26, 25, 24],
                'RSI': [35, 32, 30, 28, 25]
            })
        }
        rsi_signal = self.strategies[1].generate_signal('AMD', rsi_market_data)
        self.assertIn(rsi_signal, ['BUY', 'SELL', None])

        # Simulate market data for MACDStrategy
        macd_market_data = {
            'close': 150,
            'df': pd.DataFrame({
                'datetime': dates,
                'close': [150, 151, 152, 153, 154],
                'MACD': [1.2, 1.3, 1.4, 1.5, 1.6],
                'MACD_Signal': [1.1, 1.2, 1.3, 1.4, 1.5]
            })
        }
        macd_signal = self.strategies[2].generate_signal('NVDA', macd_market_data)
        self.assertIn(macd_signal, ['BUY', 'SELL', None])

        # Simulate market data for BollingerBandsStrategy
        bb_market_data = {
            'close': 95,
            'df': pd.DataFrame({
                'datetime': dates,
                'close': [90, 92, 94, 96, 98],
                'BB_upper': [100, 101, 102, 103, 104],
                'BB_middle': [90, 91, 92, 93, 94],
                'BB_lower': [80, 81, 82, 83, 84]
            })
        }
        bb_signal = self.strategies[3].generate_signal('AAPL', bb_market_data)
        self.assertIn(bb_signal, ['BUY', 'SELL', None])

if __name__ == '__main__':
    unittest.main()
    # After tests, visualize the results.
    tickers = list(self.stock_file_paths.keys())
    plot_all_strategies_results(self.portfolios, tickers)