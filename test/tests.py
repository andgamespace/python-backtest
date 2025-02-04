import unittest
import pandas as pd
import numpy as np
from backtest import DataLoader, SimpleMovingAverageStrategy, Portfolio, Engine
from backtest import RSIStrategy, MACDStrategy, BollingerBandsStrategy
from backtest.utils import risk_management
from backtest.Orders import Order, OrderType # Import Order and OrderType for tests
import io # Import io for testing CSV data
from unittest.mock import patch
import matplotlib.pyplot as plt  # Import pyplot for visual tests
from backtest.visuals import plot_signals, plot_portfolio, plot_strategy_results, plot_portfolio_over_time, plot_all_strategies_results # Import visual functions


class TestBacktestingFramework(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)  # For reproducibility
        self.data_loader = DataLoader() # Default DataLoader without scaling for most tests
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
        self.portfolio = Portfolio(initial_cash=100000, max_drawdown=0.1, volatility_threshold=0.05, risk_free_rate=0.02) # Initialize with risk_free_rate
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

            # Check if metrics are calculated (basic check, more detailed tests below)
            self.portfolio.calculate_final_metrics() # Call metrics calculation
            self.assertIsNotNone(self.portfolio.portfolio_value_history, "Portfolio value history should be recorded.")

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

    def test_risk_management_position_size(self):
        """
        Test risk management based on position size.
        """
        allowed = risk_management(position_size=1000, account_balance=3000)
        self.assertTrue(allowed, "Trade should be allowed with reasonable position size.")
        disallowed = risk_management(position_size=2000, account_balance=500)
        self.assertFalse(disallowed, "Trade should be disallowed when position size is too large.")

    def test_risk_management_max_drawdown(self):
        """
        Test risk management based on maximum drawdown.
        """
        portfolio_history = pd.Series([100000, 95000, 90000])
        max_drawdown_limit = 0.05
        allowed = risk_management(position_size=100, account_balance=10000, portfolio_history=portfolio_history, max_drawdown=0.15)
        self.assertTrue(allowed, "Trade should be allowed if drawdown is within limit.")
        disallowed = risk_management(position_size=100, account_balance=10000, portfolio_history=portfolio_history, max_drawdown=0.08)
        self.assertFalse(disallowed, "Trade should be disallowed if drawdown exceeds limit.")

    def test_risk_management_volatility_stop(self):
        """
        Test risk management based on volatility stop.
        """
        allowed = risk_management(position_size=100, account_balance=10000, current_price=105, entry_price=100, volatility_threshold=0.10)
        self.assertTrue(allowed, "Trade should be allowed if volatility is within threshold.")
        disallowed = risk_management(position_size=100, account_balance=10000, current_price=120, entry_price=100, volatility_threshold=0.15)
        self.assertFalse(disallowed, "Trade should be disallowed if volatility exceeds threshold.")

    def test_portfolio_risk_management_integration(self):
        """
        Test if portfolio integrates risk management and prevents trades based on drawdown.
        """
        portfolio = Portfolio(initial_cash=100000, max_drawdown=0.05)
        portfolio.history = [{'timestamp': pd.to_datetime('now'), 'portfolio_value': 100000}, {'timestamp': pd.to_datetime('now'), 'portfolio_value': 94000}]
        portfolio._update_portfolio_history()
        initial_cash = portfolio.cash
        portfolio.execute_trade('AMD', 10, 100, 0)
        final_cash = portfolio.cash
        self.assertEqual(initial_cash, final_cash, "Trade should not be executed if max drawdown is exceeded.")

    def test_order_creation(self):
        """
        Test the creation of different order types.
        """
        market_order = Order(order_type=OrderType.MARKET, ticker='AMD', quantity=10)
        self.assertEqual(market_order.order_type, OrderType.MARKET)
        self.assertEqual(market_order.ticker, 'AMD')
        self.assertEqual(market_order.quantity, 10)
        self.assertIsNone(market_order.price)
        self.assertIsNone(market_order.stop_price)

    def test_portfolio_handle_signal_order_types(self):
        """
        Test if portfolio.handle_signal correctly creates different order types.
        """
        portfolio = Portfolio(initial_cash=100000)
        portfolio.set_data_loader(self.data_loader) # Need to set dataloader to avoid error in visuals later if tests are extended

        # Handle BUY MARKET signal
        portfolio.handle_signal('AMD', 'BUY', current_price=100, index=0, order_type=OrderType.MARKET)
        self.assertEqual(len(portfolio.pending_orders), 0, "Market order should not be pending.")
        self.assertEqual(len(portfolio.trade_log), 1, "Market BUY order should be in trade log.")
        self.assertEqual(portfolio.trade_log[0][1], 'BUY')

        # Reset trade_log and pending_orders
        portfolio.trade_log = []
        portfolio.pending_orders = []

        # Handle SELL LIMIT signal
        portfolio.handle_signal('NVDA', 'SELL', current_price=160, index=1, order_type=OrderType.LIMIT, limit_price=155.00)
        self.assertEqual(len(portfolio.pending_orders), 1, "Limit SELL order should be pending.")
        self.assertEqual(portfolio.pending_orders[0].order_type, OrderType.LIMIT)
        self.assertEqual(portfolio.pending_orders[0].ticker, 'NVDA')
        self.assertEqual(portfolio.pending_orders[0].quantity, -10)
        self.assertEqual(portfolio.pending_orders[0].price, 155.00)
        self.assertEqual(len(portfolio.trade_log), 0, "Limit SELL order should not be in trade log yet.")

        # Reset pending_orders
        portfolio.pending_orders = []

        # Handle BUY STOP signal
        portfolio.handle_signal('AAPL', 'BUY', current_price=170, index=2, order_type=OrderType.STOP, stop_price=175.00)
        self.assertEqual(len(portfolio.pending_orders), 1, "Stop BUY order should be pending.")
        self.assertEqual(portfolio.pending_orders[0].order_type, OrderType.STOP)
        self.assertEqual(portfolio.pending_orders[0].ticker, 'AAPL')
        self.assertEqual(portfolio.pending_orders[0].quantity, 10)
        self.assertEqual(portfolio.pending_orders[0].price, 175.00) # price in Order for STOP is stop_price provided in handle_signal, now asserted correctly
        self.assertEqual(len(portfolio.trade_log), 0, "Stop BUY order should not be in trade log yet.")

    def test_data_loader_validation(self):
        """
        Test data loader validation functionalities.
        """
        data_loader = DataLoader()

        # Test for missing columns
        csv_data_missing_columns = io.StringIO("""datetime;open;high;low;volume
2024-01-01 09:30:00;100;102;99;1000
2024-01-01 09:35:00;102;103;101;1500""")
        df_missing_columns = data_loader.read_stock_data([csv_data_missing_columns], 'TEST_MISSING', structure=['datetime', 'open', 'high', 'low', 'close', 'volume'], sep=';')
        self.assertTrue(df_missing_columns.empty, "DataLoader should return empty DataFrame for missing 'close' column.")

        # Test for invalid datetime format
        csv_data_invalid_datetime = io.StringIO("""datetime;open;high;low;close;volume
01-01-2024 09:30:00;100;102;99;101;1000
2024-01-01 09:35:00;102;103;101;103;1500""")
        df_invalid_datetime = data_loader.read_stock_data([csv_data_invalid_datetime], 'TEST_DATETIME', structure=['datetime', 'open', 'high', 'low', 'close', 'volume'], sep=';')
        self.assertTrue(df_invalid_datetime.empty, "DataLoader should return empty DataFrame for invalid datetime format.")

        # Test for non-numeric values in price
        csv_data_non_numeric_price = io.StringIO("""datetime;open;high;low;close;volume
2024-01-01 09:30:00;100;102;99;INVALID;1000
2024-01-01 09:35:00;102;103;101;103;1500""")
        df_non_numeric_price = data_loader.read_stock_data([csv_data_non_numeric_price], 'TEST_NON_NUMERIC', structure=['datetime', 'open', 'high', 'low', 'close', 'volume'], sep=';')
        self.assertTrue(df_non_numeric_price.empty, "DataLoader should return empty DataFrame for non-numeric price.")

        # Test for negative volume (should be clipped to 0)
        csv_data_negative_volume = io.StringIO("""datetime;open;high;low;close;volume
2024-01-01 09:30:00;100;102;99;101;-1000
2024-01-01 09:35:00;102;103;101;103;1500""")
        df_negative_volume = data_loader.read_stock_data([csv_data_negative_volume], 'TEST_NEGATIVE_VOLUME', structure=['datetime', 'open', 'high', 'low', 'close', 'volume'], sep=';')
        self.assertTrue((df_negative_volume['volume'] >= 0).all(), "DataLoader should clip negative volume to 0.")
        self.assertFalse(df_negative_volume.empty, "DataLoader should not return empty df if only volume has negative values and clipping is applied")

        # Test for handling NaN values (fillna - ffill/bfill) - simple check, more thorough testing might be needed
        csv_data_nan_values = io.StringIO("""datetime;open;high;low;close;volume
2024-01-01 09:30:00;NaN;102;99;101;1000
2024-01-01 09:35:00;102;NaN;101;103;NaN""")
        df_nan_values = data_loader.read_stock_data([csv_data_nan_values], 'TEST_NAN_VALUES', structure=['datetime', 'open', 'high', 'low', 'close', 'volume'], sep=';')
        self.assertFalse(df_nan_values.isnull().any().any(), "DataLoader should fill NaN values.")
        self.assertFalse(df_nan_values.empty, "DataLoader should still return df after filling NaNs.")

    @patch('matplotlib.pyplot.show')  # Mock plt.show to prevent plots from displaying during tests
    def test_plot_signals_visual(self, mock_show):
        """
        Test plot_signals function from visuals.py.
        """
        # Create dummy data
        data = {'datetime': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']),
                'close': [150, 152, 148, 155, 153]}
        df = pd.DataFrame(data)
        signals = [(1, 'BUY'), (3, 'SELL')]

        try:
            plot_signals(df, signals)
            self.assertTrue(True, "plot_signals ran without errors") # If it reaches here, no error occurred
            self.assertTrue(mock_show.called, "plt.show was called") # Check if plt.show was called (plot attempted to display)
        except Exception as e:
            self.fail(f"plot_signals raised an exception: {e}")

    @patch('matplotlib.pyplot.show')
    def test_plot_portfolio_visual(self, mock_show):
        """
        Test plot_portfolio function from visuals.py.
        """
        # Create dummy portfolio value series
        dates = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'])
        portfolio_value_series = pd.Series([100000, 101000, 99000, 102000, 103000], index=dates)

        try:
            plot_portfolio(portfolio_value_series)
            self.assertTrue(True, "plot_portfolio ran without errors")
            self.assertTrue(mock_show.called, "plt.show was called")
        except Exception as e:
            self.fail(f"plot_portfolio raised an exception: {e}")

    @patch('matplotlib.pyplot.show')
    def test_plot_strategy_results_visual(self, mock_show):
        """
        Test plot_strategy_results function."""
        portfolio = Portfolio(initial_cash=100000)
        portfolio.set_data_loader(self.data_loader)
        # Dummy trade log and data (adjust as needed for a realistic scenario)
        portfolio.trade_log = [('AMD', 'BUY', 10, 150, 150, 1), ('AMD', 'SELL', 10, 160, 160, 3)]
        data = {'datetime': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']),
                'close': [150, 152, 148, 155, 153]}
        self.data_loader.data['AMD'] = pd.DataFrame(data) # Mock data in data_loader for test
        portfolio.data_loader = self.data_loader # Set data loader in portfolio

        try:
            plot_strategy_results(portfolio, 'AMD', 'SMAStrategy')
            self.assertTrue(True, "plot_strategy_results ran without errors")
            self.assertTrue(mock_show.called, "plt.show was called")
        except Exception as e:
            self.fail(f"plot_strategy_results raised an exception: {e}")

    @patch('matplotlib.pyplot.show')
    def test_plot_portfolio_over_time_visual(self, mock_show):
        """Test plot_portfolio_over_time function."""
        portfolio = Portfolio(initial_cash=100000)
        # Dummy historical data
        portfolio.history = [
            {'timestamp': pd.to_datetime('2024-01-01'), 'portfolio_value': 100000},
            {'timestamp': pd.to_datetime('2024-01-02'), 'portfolio_value': 101000},
            {'timestamp': pd.to_datetime('2024-01-03'), 'portfolio_value': 102000}
        ]
        portfolio._update_portfolio_history()

        try:
            plot_portfolio_over_time(portfolio, 'SMAStrategy')
            self.assertTrue(True, "plot_portfolio_over_time ran without errors")
            self.assertTrue(mock_show.called, "plt.show was called")
        except Exception as e:
            self.fail(f"plot_portfolio_over_time raised an exception: {e}")

    @patch('matplotlib.pyplot.show')
    def test_plot_all_strategies_results_visual(self, mock_show):
        """Test plot_all_strategies_results function."""
        portfolios = {
            'SMAStrategy': Portfolio(initial_cash=100000),
            'RSIStrategy': Portfolio(initial_cash=100000)
        }
        tickers = ['AMD', 'NVDA']
        # Mock data in data_loader (similar to test_plot_strategy_results_visual if needed)
        for strat_name, port in portfolios.items():
            port.set_data_loader(self.data_loader)
            port.trade_log = [('AMD', 'BUY', 10, 150, 150, 1), ('AMD', 'SELL', 10, 160, 160, 3)] # Dummy trade logs
            port.history = [{'timestamp': pd.to_datetime('2024-01-01'), 'portfolio_value': 100000}] # Dummy history
            port._update_portfolio_history()
        data = {'datetime': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']),
                'close': [150, 152, 148, 155, 153]}
        self.data_loader.data['AMD'] = pd.DataFrame(data)
        self.data_loader.data['NVDA'] = pd.DataFrame(data)


        try:
            plot_all_strategies_results(portfolios, tickers)
            self.assertTrue(True, "plot_all_strategies_results ran without errors")
            self.assertTrue(mock_show.called, "plt.show was called at least once (as it's called internally)")
        except Exception as e:
            self.fail(f"plot_all_strategies_results raised an exception: {e}")


if __name__ == '__main__':
    unittest.main()