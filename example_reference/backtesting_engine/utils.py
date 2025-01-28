import numpy as np
import pandas as pd
from numba import jit
from typing import Dict, List

@jit(nopython=True)
def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.01) -> float:
    """
    Calculate Sharpe ratio with Numba optimization.
    """
    excess_returns = returns - risk_free_rate/252
    return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

@jit(nopython=True)
def calculate_max_drawdown(returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown with Numba optimization.
    """
    cumulative = np.zeros(len(returns))
    running_max = np.zeros(len(returns))
    drawdowns = np.zeros(len(returns))
    
    cumulative[0] = 1 + returns[0]
    running_max[0] = cumulative[0]
    
    for i in range(1, len(returns)):
        cumulative[i] = cumulative[i-1] * (1 + returns[i])
        running_max[i] = max(running_max[i-1], cumulative[i])
        drawdowns[i] = (cumulative[i] - running_max[i]) / running_max[i]
        
    return np.min(drawdowns)