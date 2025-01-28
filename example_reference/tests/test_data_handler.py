import unittest
import pandas as pd
import numpy as np
from backtesting_engine.data_handler import DataHandler
from pathlib import Path

class TestDataHandler(unittest.TestCase):
    def setUp(self):
        self.data_handler = DataHandler()
        
    def test_load_data(self):
        # Create a test DataFrame
        dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
        test_data = pd.DataFrame({
            'datetime': dates,
            'open': np.random.rand(100) * 100,
            'high': np.random.rand(100) * 100,
            'low': np.random.rand(100) * 100,
            'close': np.random.rand(100) * 100,
            'volume': np.random.randint(1000, 100000, 100)
        })
        
        # Save test data to CSV
        test_file = 'test_data.csv'
        test_data.to_csv(test_file, index=False)
        
        # Test loading
        df = self.data_handler.load_data([test_file], 'TEST')
        
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 100)
        self.assertTrue('sma_20' in df.columns)
        
        # Cleanup
        Path(test_file).unlink()