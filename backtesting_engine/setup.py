from setuptools import setup, find_packages

setup(
    name="backtesting_engine",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "numba",
        # "typing",  # Removed typing as it's part of the standard library in Python 3.7+
    ],
    python_requires=">=3.7",
)
