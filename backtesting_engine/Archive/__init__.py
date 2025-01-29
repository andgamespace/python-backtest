"""
Initialization module for the backtesting_engine package.
This makes classes and functions available at package level.
"""

# This is the initialization file for the backtesting_engine package.
# You can import necessary modules and initialize package-level variables here.

# Example import statements (uncomment and modify as needed):
# from .module1 import Class1, function1
# from .module2 import Class2, function2
from .engine import BacktestEngine
from .portfolio import Portfolio
from .strategy import Strategy
from .DataLoader import DataLoader

__all__ = ["BacktestEngine", "Portfolio", "Strategy", "DataLoader"]

# Example package-level variable (modify as needed):
# package_variable = "default_value"

# You can also define package-level functions or classes here if needed.
