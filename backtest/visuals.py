import matplotlib.pyplot as plt
import logging
from typing import Dict, List
import pandas as pd
from .Portfolio import Portfolio

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
            plt.scatter(df.loc[idx, 'datetime'], df.loc[idx, 'close'], color='green', marker='^', s=100, label='BUY Signal')
        elif signal_type == 'SELL':
            plt.scatter(df.loc[idx, 'datetime'], df.loc[idx, 'close'], color='red', marker='v', s=100, label='SELL Signal')
    plt.legend()
    plt.title("Price With Buy/Sell Signals")
    plt.xlabel("Datetime")
    plt.ylabel("Price")
    plt.grid(True)
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
    plt.xlabel("Datetime")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.show()
    logger.info("Finished plotting portfolio value.")

def plot_strategy_results(portfolio: Portfolio, ticker: str, strategy_name: str):
    """
    Plot buy/sell signals and portfolio value for a specific strategy and ticker.
    
    Args:
        portfolio (Portfolio): The portfolio instance containing trade logs.
        ticker (str): The stock ticker.
        strategy_name (str): Name of the strategy.
    """
    # Retrieve trade log for the specific ticker and strategy
    trades = [trade for trade in portfolio.trade_log if trade[0] == ticker]
    df = portfolio.data_loader.data[ticker]

    buy_signals = [trade for trade in trades if trade[1] == 'BUY']
    sell_signals = [trade for trade in trades if trade[1] == 'SELL']

    plt.figure(figsize=(14, 7))
    plt.plot(df['datetime'], df['close'], label='Close Price', color='blue')

    # Plot Buy signals
    if buy_signals:
        buy_dates = [df['datetime'].iloc[idx] for idx, _ in buy_signals]
        buy_prices = [df['close'].iloc[idx] for idx, _ in buy_signals]
        plt.scatter(buy_dates, buy_prices, marker='^', color='green', label='BUY Signal', s=100)

    # Plot Sell signals
    if sell_signals:
        sell_dates = [df['datetime'].iloc[idx] for idx, _ in sell_signals]
        sell_prices = [df['close'].iloc[idx] for idx, _ in sell_signals]
        plt.scatter(sell_dates, sell_prices, marker='v', color='red', label='SELL Signal', s=100)
    
    plt.title(f"{ticker} Price with Buy/Sell Signals - {strategy_name}")
    plt.xlabel("Datetime")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_portfolio_over_time(portfolio: Portfolio, strategy_name: str):
    """
    Plot the portfolio value over time for a specific strategy.
    
    Args:
        portfolio (Portfolio): The portfolio instance containing historical values.
        strategy_name (str): Name of the strategy.
    """
    historical = portfolio.get_historical_value()
    if historical.empty:
        logger.warning(f"No historical data to plot for {strategy_name}.")
        return

    plt.figure(figsize=(14, 7))
    plt.plot(historical['timestamp'], historical['portfolio_value'], label='Portfolio Value', color='purple')
    plt.title(f"Portfolio Equity Curve - {strategy_name}")
    plt.xlabel("Datetime")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_all_strategies_results(portfolios: Dict[str, Portfolio], tickers: List[str]):
    """
    Plot buy/sell signals and portfolio value for all strategies and tickers.
    
    Args:
        portfolios (Dict[str, Portfolio]): Dictionary of portfolio instances keyed by strategy name.
        tickers (List[str]): List of stock tickers.
    """
    for strategy_name, portfolio in portfolios.items():
        for ticker in tickers:
            plot_strategy_results(portfolio, ticker, strategy_name)
        plot_portfolio_over_time(portfolio, strategy_name)