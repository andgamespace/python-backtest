import logging
from typing import Dict, Any
import pandas as pd
import multiprocessing

class Engine:
    """
    Engine orchestrates the entire backtest loop, now with concurrency and order processing.
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

    def _run_backtest_single_ticker(self, ticker):
        """
        Run backtest for a single ticker, including order processing at each step.
        """
        self.logger.info(f"Starting backtest for ticker: {ticker} in process {multiprocessing.current_process().name}")
        df = self._get_data(ticker)
        if df.empty:
            return

        for idx in range(len(df)):
            current_row = df.iloc[idx]
            current_data = df.iloc[:idx+1]
            current_time = current_row['datetime'] # Get current datetime for order processing
            current_price = current_row['close']

            # Ensure 'datetime' is datetime type
            if not pd.api.types.is_datetime64_any_dtype(current_data['datetime']):
                current_data['datetime'] = pd.to_datetime(current_data['datetime'])

            market_data = {'close': current_price, 'df': current_data}

            # Process pending orders before generating new signals
            current_prices_for_processing = {ticker: current_price} # For now, process orders based on current ticker price only
            self.portfolio.process_orders(current_time, current_prices_for_processing) # Process orders at each time step

            # Generate signal
            signal = self.strategy.generate_signal(ticker, market_data)

            # Execute trade if signal is present (default Market order for now)
            if signal:
                self.portfolio.handle_signal(ticker, signal, current_price=current_price, index=idx) # Default Market order

        # Process any remaining pending orders at the end of backtest - optional, depends on strategy
        # current_prices_end = {ticker: self._get_data(ticker)['close'].iloc[-1]} # Get last prices - careful with look-ahead bias
        # self.portfolio.process_orders("End of Backtest", current_prices_end)

        self.logger.info(f"Backtest for ticker {ticker} completed in process {multiprocessing.current_process().name}")


    def run_backtest(self, tickers):
        """
        Main loop to run backtest for all tickers concurrently using multiprocessing.
        """
        self.logger.info(f"Starting concurrent backtest for tickers: {tickers}")
        processes = []

        for ticker in tickers:
            process = multiprocessing.Process(target=self._run_backtest_single_ticker, args=(ticker,))
            processes.append(process)
            process.start()

        for process in processes:
            process.join() # Wait for all processes to complete

        self.logger.info("Concurrent backtest completed for all tickers.")
        self.portfolio.calculate_final_metrics()


    def _get_data(self, ticker):
        """
        Fetch data for the given ticker from the DataLoader.
        """
        if ticker not in self.data_loader.data:
            self.logger.warning(f"Ticker {ticker} not loaded.")
            return pd.DataFrame()
        return self.data_loader.data[ticker]