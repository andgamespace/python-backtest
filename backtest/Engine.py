import logging
from typing import Dict, Any
import pandas as pd

class Engine:
    """
    Engine orchestrates the entire backtest loop.
    It:
      1. Takes a loaded DataLoader.
      2. Accepts a Strategy instance.
      3. Iterates over each timestamp in the loaded data.
      4. Sends signals to Portfolio.
      5. Calculates metrics and logs stats.
    """

    def __init__(self, data_loader, portfolio, strategy, logger=None):
        self.data_loader = data_loader
        self.portfolio = portfolio
        self.strategy = strategy
        self.logger = logger or self._setup_logger()
        self.logger.info("Engine initialized.")

    def _setup_logger(self):
        logger = logging.getLogger('Engine')
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(ch)
            logger.setLevel(logging.INFO)
        return logger

    def run_backtest(self, tickers):
        """
        Main loop over each timestamp across all tickers to simulate trades concurrently.

        Args:
            tickers (list): List of stock symbols to run the strategy on.
        """
        self.logger.info(f"Starting backtest for tickers: {tickers}")

        # Determine the maximum length of data among all tickers
        max_length = max(len(self._get_data(ticker)) for ticker in tickers)

        # Iterate over each timestamp index
        for idx in range(max_length):
            for ticker in tickers:
                df = self._get_data(ticker)
                if df.empty or idx >= len(df):
                    continue

                current_row = df.iloc[idx]
                current_data = df.iloc[:idx+1]

                # Ensure 'datetime' is datetime type
                if not pd.api.types.is_datetime64_any_dtype(current_data['datetime']):
                    current_data['datetime'] = pd.to_datetime(current_data['datetime'])

                market_data = {'close': current_row['close'], 'df': current_data}

                # Generate signal
                signal = self.strategy.generate_signal(ticker, market_data)

                # Execute trade if signal is present
                if signal:
                    self.portfolio.handle_signal(ticker, signal, current_price=current_row['close'], index=idx)

        # After the loop, print final metrics or logs
        self.logger.info("Backtest completed.")
        self.portfolio.calculate_final_metrics()

    def _get_data(self, ticker):
        """
        Fetch data for the given ticker from the DataLoader.
        """
        if ticker not in self.data_loader.data:
            self.logger.warning(f"Ticker {ticker} not loaded.")
            return pd.DataFrame()
        return self.data_loader.data[ticker]