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
    ],
    python_requires=">=3.7",
)
