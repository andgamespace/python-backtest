from setuptools import setup, find_packages

setup(
    name="backtesting_engine",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "numba>=0.58.0",
    ],
    author="andgamespace",
    description="A flexible backtesting engine for trading strategies",
    python_requires=">=3.8",
)