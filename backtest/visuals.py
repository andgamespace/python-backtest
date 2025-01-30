import matplotlib.pyplot as plt
import logging

def _setup_logger():
    logger = logging.getLogger('Visuals')
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)
        logger.setLevel(logging.INFO)
    return logger

logger = _setup_logger()

def plot_signals(df, signals):
    """
    Plot the stock price and overlay buy/sell signals.
    
    df: DataFrame containing columns 'datetime' and 'close'
    signals: list of (index, signal_type) or similar
    """
    logger.info("Plotting signals.")
    plt.figure(figsize=(10, 6))
    plt.plot(df['datetime'], df['close'], label='Price', color='blue')

    for (idx, signal_type) in signals:
        if signal_type == 'BUY':
            plt.scatter(df.loc[idx, 'datetime'], df.loc[idx, 'close'], color='green', marker='v', s=100, label='BUY Signal')
        elif signal_type == 'SELL':
            plt.scatter(df.loc[idx, 'datetime'], df.loc[idx, 'close'], color='red', marker='^', s=100, label='SELL Signal')
    plt.legend()
    plt.title("Price With Buy/Sell Signals")
    plt.show()
    logger.info("Finished plotting signals.")

def plot_portfolio(portfolio_value_series):
    """
    Plot the portfolio value over time.
    """
    logger.info("Plotting portfolio value.")
    plt.figure(figsize=(10, 4))
    plt.plot(portfolio_value_series.index, portfolio_value_series.values, label='Portfolio Value', color='purple')
    plt.legend()
    plt.title("Portfolio Equity Curve")
    plt.show()
    logger.info("Finished plotting portfolio value.")