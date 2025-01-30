import unittest
import pandas as pd
import numpy as np
from backtest import DataLoader, SimpleMovingAverageStrategy, Portfolio, Engine

class TestBacktestingFramework(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)  # For reproducibility
        self.data_loader = DataLoader()
        self.structure = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        self.sep = ';'
        # Load mock data for testing
        self.data_loader.data['TEST'] = pd.DataFrame({
            'datetime': pd.date_range(start='2020-01-01', periods=50, freq='D'),
            'open': np.random.uniform(100, 200, 50),
            'high': np.random.uniform(200, 300, 50),
            'low': np.random.uniform(50, 100, 50),
            'close': np.random.uniform(100, 200, 50),
            'volume': np.random.randint(1000, 10000, 50),
        })

        self.strategy = SimpleMovingAverageStrategy(short_window=5, long_window=20)
        self.portfolio = Portfolio(initial_cash=100000)
        self.engine = Engine(data_loader=self.data_loader, portfolio=self.portfolio, strategy=self.strategy)

    def test_run_backtest(self):
        """
        Test the complete backtesting workflow.
        """
        tickers = ['TEST']
        try:
            self.engine.run_backtest(tickers)
        except Exception as e:
            self.fail(f"Backtest run failed with exception: {e}")

        # Assertions to verify portfolio updates
        final_cash = self.portfolio.cash
        self.assertTrue(final_cash <= 100000, "Final cash should not exceed initial cash without profits.")

        # Since data is random, we can't predict positions. Just ensure no errors.

    def test_data_loading(self):
        """
        Test if data is loaded correctly.
        """
        self.assertIn('TEST', self.data_loader.data)
        df = self.data_loader.data['TEST']
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 50)
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
        Test signal generation of the strategy.
        """
        # Simulate market data up to a specific point
        market_data = {
            'close': 150,
            'df': pd.DataFrame({
                'close': [140, 145, 147, 149, 150]
            })
        }
        signal = self.strategy.generate_signal('TEST', market_data)
        self.assertIn(signal, ['BUY', 'SELL', None])

if __name__ == '__main__':
    unittest.main()