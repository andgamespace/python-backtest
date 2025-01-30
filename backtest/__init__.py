"""Initialization of the Python backtesting package."""

from .DataLoader import DataLoader
from .Strategy import Strategy, SimpleMovingAverageStrategy, RSIStrategy, MACDStrategy, BollingerBandsStrategy
from .Portfolio import Portfolio
from .Engine import Engine
from .Orders import Order, OrderType
from .Position import Position
from .utils import risk_management  # Example usage
from .visuals import plot_signals, plot_portfolio  # Example usage

__version__ = '0.1.0'

# Create a single instance of DataLoader
data_loader = DataLoader()

# Plan: 
# 1. User loads data via data_loader.load_ticker().
# 2. User sets up a Strategy instance.
# 3. Engine runs the Strategy across the loaded data.
# 4. Portfolio is updated with the resulting trades.
# 5. Final metrics and plots are shown through visuals.

__all__ = [
    'DataLoader',
    'Strategy',
    'SimpleMovingAverageStrategy',
    'RSIStrategy',
    'MACDStrategy',
    'BollingerBandsStrategy',
    'Portfolio',
    'Engine',
    'Order',
    'OrderType',
    'Position',
    'data_loader',
    'risk_management',       # from utils
    'plot_signals',          # from visuals
    'plot_portfolio',        # from visuals
    '__version__'
]