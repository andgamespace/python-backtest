from typing import Any
import logging

def _setup_logger():
    logger = logging.getLogger('Strategy')
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)
        logger.setLevel(logging.INFO)
    return logger

logger = _setup_logger()

class Strategy:
    """
    Base Strategy class. Child classes should override generate_signal().
    """

    def __init__(self, parameters: dict = None):
        self.parameters = parameters or {}
        logger.info(f"{self.__class__.__name__} initialized with parameters: {self.parameters}")

    def generate_signal(self, ticker: str, market_data: Any) -> str:
        """
        Return 'BUY', 'SELL', or None. 
        market_data can be a row of the DataFrame or other relevant info.
        """
        # Example: Always return None, to be overridden by actual strategies.
        return None

class SimpleMovingAverageStrategy(Strategy):
    """
    Example strategy that calculates short-term and long-term moving averages
    to generate buy/sell signals.
    """
    def __init__(self, short_window: int = 5, long_window: int = 20):
        super().__init__({'short_window': short_window, 'long_window': long_window})
        self.short_window = short_window
        self.long_window = long_window
        self.previous_short_ma = None
        self.previous_long_ma = None
        logger.info(f"{self.__class__.__name__} created with short_window={self.short_window} and long_window={self.long_window}")

    def generate_signal(self, ticker: str, market_data: Any) -> str:
        """
        Generate 'BUY' or 'SELL' signals based on moving average crossover.
        """
        current_close = market_data['close']
        df = market_data['df']  # Assume market_data includes the DataFrame up to current point

        short_ma = df['close'].rolling(window=self.short_window).mean().iloc[-1]
        long_ma = df['close'].rolling(window=self.long_window).mean().iloc[-1]

        signal = None

        if self.previous_short_ma is not None and self.previous_long_ma is not None:
            if self.previous_short_ma <= self.previous_long_ma and short_ma > long_ma:
                signal = 'BUY'
                logger.info(f"BUY signal generated for {ticker} at price {current_close}.")
            elif self.previous_short_ma >= self.previous_long_ma and short_ma < long_ma:
                signal = 'SELL'
                logger.info(f"SELL signal generated for {ticker} at price {current_close}.")

        self.previous_short_ma = short_ma
        self.previous_long_ma = long_ma

        return signal