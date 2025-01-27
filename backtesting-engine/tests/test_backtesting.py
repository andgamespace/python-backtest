import unittest
from src.backtesting import Backtester

class TestBacktester(unittest.TestCase):

    def setUp(self):
        self.backtester = Backtester()

    def test_run_backtest(self):
        result = self.backtester.run_backtest()
        self.assertIsNotNone(result)
        # Add more assertions based on expected output

    def test_evaluate_performance(self):
        self.backtester.run_backtest()
        performance = self.backtester.evaluate_performance()
        self.assertIsInstance(performance, dict)
        # Add more assertions based on expected performance metrics

if __name__ == '__main__':
    unittest.main()