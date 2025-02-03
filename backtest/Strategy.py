from typing import Any, Optional
import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression  # Example ML model
from typing import List

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
        df = market_data['df']  # Ensure 'df' is a pd.DataFrame

        if not isinstance(df, pd.DataFrame):
            logger.error(f"Market data for {ticker} is not a DataFrame.")
            return None

        short_ma = df['SMA_5'].iloc[-1]
        long_ma = df['SMA_20'].iloc[-1]

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

class RSIStrategy(Strategy):
    """
    Strategy based on Relative Strength Index (RSI).
    Generates 'BUY' signal when RSI crosses above rsi_low.
    Generates 'SELL' signal when RSI crosses below rsi_high.
    """
    def __init__(self, rsi_low: int = 30, rsi_high: int = 70):
        super().__init__({'rsi_low': rsi_low, 'rsi_high': rsi_high})
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high
        self.previous_rsi = None
        logger.info(f"{self.__class__.__name__} created with rsi_low={self.rsi_low} and rsi_high={self.rsi_high}")

    def generate_signal(self, ticker: str, market_data: Any) -> str:
        df = market_data['df']
        current_rsi = df['RSI'].iloc[-1]
        
        if self.previous_rsi is None:
            self.previous_rsi = current_rsi
            return None

        signal = None
        if self.previous_rsi < self.rsi_low and current_rsi >= self.rsi_low:
            signal = 'BUY'
            logger.info(f"BUY signal generated for {ticker} based on RSI crossing above {self.rsi_low}.")
        elif self.previous_rsi > self.rsi_high and current_rsi <= self.rsi_high:
            signal = 'SELL'
            logger.info(f"SELL signal generated for {ticker} based on RSI crossing below {self.rsi_high}.")

        self.previous_rsi = current_rsi
        return signal

class MACDStrategy(Strategy):
    """
    Strategy based on Moving Average Convergence Divergence (MACD).
    Generates 'BUY' signal when MACD crosses above the MACD signal line.
    Generates 'SELL' signal when MACD crosses below the MACD signal line.
    """
    def __init__(self, fastperiod=12, slowperiod=26, signalperiod=9):
        super().__init__({'fastperiod': fastperiod, 'slowperiod': slowperiod, 'signalperiod': signalperiod})
        self.fastperiod = fastperiod
        self.slowperiod = slowperiod
        self.signalperiod = signalperiod
        self.previous_macd = None
        self.previous_macd_signal = None
        logger.info(f"{self.__class__.__name__} created with fastperiod={self.fastperiod}, slowperiod={self.slowperiod}, signalperiod={self.signalperiod}")

    def generate_signal(self, ticker: str, market_data: Any) -> str:
        df = market_data['df']
        current_macd = df['MACD'].iloc[-1]
        current_macd_signal = df['MACD_Signal'].iloc[-1]

        if self.previous_macd is None or self.previous_macd_signal is None:
            self.previous_macd = current_macd
            self.previous_macd_signal = current_macd_signal
            return None

        signal = None
        if self.previous_macd <= self.previous_macd_signal and current_macd > current_macd_signal:
            signal = 'BUY'
            logger.info(f"BUY signal generated for {ticker} based on MACD crossover.")
        elif self.previous_macd >= self.previous_macd_signal and current_macd < current_macd_signal:
            signal = 'SELL'
            logger.info(f"SELL signal generated for {ticker} based on MACD crossover.")

        self.previous_macd = current_macd
        self.previous_macd_signal = current_macd_signal
        return signal

class BollingerBandsStrategy(Strategy):
    """
    Strategy based on Bollinger Bands.
    Generates 'BUY' signal when price crosses below the lower band.
    Generates 'SELL' signal when price crosses above the upper band.
    """
    def __init__(self, window=20, num_std=2):
        super().__init__({'window': window, 'num_std': num_std})
        self.window = window
        self.num_std = num_std
        self.previous_close = None
        self.previous_bb_lower = None
        self.previous_bb_upper = None
        logger.info(f"{self.__class__.__name__} created with window={self.window}, num_std={self.num_std}")

    def generate_signal(self, ticker: str, market_data: Any) -> str:
        df = market_data['df']
        current_close = df['close'].iloc[-1]
        current_bb_lower = df['BB_lower'].iloc[-1]
        current_bb_upper = df['BB_upper'].iloc[-1]

        if self.previous_close is None:
            self.previous_close = current_close
            self.previous_bb_lower = current_bb_lower
            self.previous_bb_upper = current_bb_upper
            return None

        signal = None
        if self.previous_close >= self.previous_bb_lower and current_close < current_bb_lower:
            signal = 'BUY'
            logger.info(f"BUY signal generated for {ticker} based on price crossing below BB_lower.")
        elif self.previous_close <= self.previous_bb_upper and current_close > current_bb_upper:
            signal = 'SELL'
            logger.info(f"SELL signal generated for {ticker} based on price crossing above BB_upper.")

        self.previous_close = current_close
        self.previous_bb_lower = current_bb_lower
        self.previous_bb_upper = current_bb_upper
        return signal

class MLStrategy(Strategy):
    """
    Machine Learning Strategy - expects a pre-trained model to be passed during initialization.
    """
    def __init__(self, model, feature_columns: List[str]): # Expects a pre-trained model and feature columns
        super().__init__()
        self.model = model # Now expects a pre-trained model to be passed
        self.feature_columns = feature_columns
        logger.info(f"{self.__class__.__name__} initialized with pre-trained model, using features: {self.feature_columns}")
        if not hasattr(model, 'predict_proba'):
            logger.error("Provided model does not have 'predict_proba' method. MLStrategy requires a model with probability predictions.")
            raise ValueError("Model must have 'predict_proba' method for MLStrategy.")

    def generate_signal(self, ticker: str, market_data: Any) -> Optional[str]:
        """
        Generate 'BUY' or 'SELL' signals based on ML model prediction.
        Uses the pre-trained model passed during initialization.

        Important:
          1. Ensure the 'model' passed is a PRE-TRAINED model.
          2. Features used for training MUST be the same as 'feature_columns'.
          3. Feature scaling used during training MUST be applied to 'market_data' here.
        """
        df = market_data['df']

        # Check if feature columns are available in market data
        for col in self.feature_columns:
            if col not in df.columns:
                logger.warning(f"Feature column '{col}' missing in market data for {ticker}. ML strategy cannot generate signal.")
                return None

        features = df[self.feature_columns].iloc[[-1]] # Get the latest row's features

        try:
            prediction_proba = self.model.predict_proba(features) # Get probabilities
            # Assuming binary classification (BUY/SELL) and index 1 corresponds to 'BUY' probability
            buy_probability = prediction_proba[0][1]

            if buy_probability > 0.6: # Example threshold - adjust as needed
                signal = 'BUY'
                logger.info(f"ML Strategy: BUY signal generated for {ticker} with probability {buy_probability:.2f}.")
            elif buy_probability < 0.4: # Example threshold for SELL
                signal = 'SELL'
                logger.info(f"ML Strategy: SELL signal generated for {ticker} with probability {buy_probability:.2f}.")
            else:
                signal = None # Neutral signal if probability is within the threshold
                logger.info(f"ML Strategy: Neutral signal generated for {ticker} with probability {buy_probability:.2f}.")

        except Exception as e:
            logger.error(f"Error during model prediction for {ticker}: {e}. No signal generated.")
            return None

        return signal