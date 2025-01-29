import backtest
import unittest
import pandas as pd
from backtest import data_loader

amd_file_path = [
    '/Users/anshc/repos/python-backtest/test/stock_data/time-series-AMD-5min.csv',
    '/Users/anshc/repos/python-backtest/test/stock_data/time-series-AMD-5min(1).csv',
    '/Users/anshc/repos/python-backtest/test/stock_data/time-series-AMD-5min(2).csv',
]
nvda_file_path = [
    '/Users/anshc/repos/python-backtest/test/stock_data/time-series-NVDA-5min.csv',
    '/Users/anshc/repos/python-backtest/test/stock_data/time-series-NVDA-5min(1).csv',
    '/Users/anshc/repos/python-backtest/test/stock_data/time-series-NVDA-5min(2).csv',
]
aapl_file_path = [
    '/Users/anshc/repos/python-backtest/test/stock_data/time-series-AAPL-5min.csv',
    '/Users/anshc/repos/python-backtest/test/stock_data/time-series-AAPL-5min(1).csv',
    '/Users/anshc/repos/python-backtest/test/stock_data/time-series-AAPL-5min(2).csv',
]
msft_file_path = [
    '/Users/anshc/repos/python-backtest/test/stock_data/time-series-MSFT-5min.csv',
    '/Users/anshc/repos/python-backtest/test/stock_data/time-series-MSFT-5min(1).csv',
    '/Users/anshc/repos/python-backtest/test/stock_data/time-series-MSFT-5min(2).csv',
]

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.data_loader = data_loader
        self.structure = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        self.sep = ';'

    def test_load_amd_data(self):
        self.data_loader.load_ticker('AMD', amd_file_path, self.structure, self.sep)
        self.assertIn('AMD', self.data_loader.data)
        self.assertIsInstance(self.data_loader.data['AMD'], pd.DataFrame)
        self.assertTrue(len(self.data_loader.data['AMD']) > 0)

    def test_load_nvda_data(self):
        self.data_loader.load_ticker('NVDA', nvda_file_path, self.structure, self.sep)
        self.assertIn('NVDA', self.data_loader.data)
        self.assertIsInstance(self.data_loader.data['NVDA'], pd.DataFrame)
        self.assertTrue(len(self.data_loader.data['NVDA']) > 0)

    def test_load_aapl_data(self):
        self.data_loader.load_ticker('AAPL', aapl_file_path, self.structure, self.sep)
        self.assertIn('AAPL', self.data_loader.data)
        self.assertIsInstance(self.data_loader.data['AAPL'], pd.DataFrame)
        self.assertTrue(len(self.data_loader.data['AAPL']) > 0)

    def test_load_msft_data(self):
        self.data_loader.load_ticker('MSFT', msft_file_path, self.structure, self.sep)
        self.assertIn('MSFT', self.data_loader.data)
        self.assertIsInstance(self.data_loader.data['MSFT'], pd.DataFrame)
        self.assertTrue(len(self.data_loader.data['MSFT']) > 0)

    def test_data_structure(self):
        # Test for one stock to verify data structure
        self.data_loader.load_ticker('AMD', amd_file_path, self.structure, self.sep)
        df = self.data_loader.data['AMD']
        
        # Check if all required columns exist
        for column in self.structure:
            self.assertIn(column, df.columns)
        
        # Check if datetime is properly parsed
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['datetime']))

if __name__ == '__main__':
    unittest.main()