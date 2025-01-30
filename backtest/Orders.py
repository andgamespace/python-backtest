import logging

def _setup_logger():
    logger = logging.getLogger('Orders')
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)
        logger.setLevel(logging.INFO)
    return logger

logger = _setup_logger()

class OrderType:
    MARKET = 'MARKET'
    LIMIT = 'LIMIT'
    STOP = 'STOP'

class Order:
    """
    Basic representation of an order with type, price, quantity, etc.
    """

    def __init__(self, order_type, ticker, quantity, price=None):
        self.order_type = order_type
        self.ticker = ticker
        self.quantity = quantity
        self.price = price  # For limit or stop orders
        self.filled = False
        logger.info(f"Created {self.order_type} order for {self.quantity} shares of {self.ticker} at price {self.price}.")

    def fill(self, fill_price):
        """
        Mark this order as filled at a given price.
        """
        self.filled = True
        self.price = fill_price
        logger.info(f"Order for {self.ticker} filled at price {self.price}.")