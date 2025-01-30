from dataclasses import dataclass
from typing import Optional
import logging

def _setup_logger():
    logger = logging.getLogger('Position')
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)
        logger.setLevel(logging.INFO)
    return logger

logger = _setup_logger()

@dataclass
class Position:
    """Represents a position in a single asset."""
    ticker: str
    quantity: int = 0
    entry_price: float = 0.0
    current_price: Optional[float] = None

    def __post_init__(self):
        logger.info(f"Position created for {self.ticker} with quantity={self.quantity} and entry_price={self.entry_price}")

    def market_value(self) -> float:
        """Calculate current market value of position."""
        if self.current_price:
            return self.quantity * self.current_price
        return self.quantity * self.entry_price

    def unrealized_pnl(self) -> float:
        """Calculate unrealized profit/loss."""
        if self.current_price:
            return (self.current_price - self.entry_price) * self.quantity
        return 0.0

    def update_price(self, new_price: float):
        """Update current price."""
        self.current_price = new_price