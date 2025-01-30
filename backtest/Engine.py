import logging
from typing import Dict, Any

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
        Main loop over each ticker's data to simulate trades.

        Args:
            tickers (list): List of stock symbols to run the strategy on.
        """
        self.logger.info(f"Starting backtest for tickers: {tickers}")
        for ticker in tickers:
            df = self._get_data(ticker)
            if df.empty:
                self.logger.warning(f"No data for ticker {ticker}, skipping.")
                continue

            self.logger.info(f"Running strategy for ticker {ticker}")
            # Iterate over each row (timestamp) in the data
            for index in range(1, len(df)):
                current_data = df.iloc[:index+1]
                row = df.iloc[index]
                market_data = {'close': row['close'], 'df': current_data}
                
                # 1. Generate signals based on current market data
                signal = self.strategy.generate_signal(ticker, market_data)

                # 2. Execute trade in the Portfolio (buy, sell, hold)
                if signal:
                    self.portfolio.handle_signal(ticker, signal, current_price=row['close'])

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