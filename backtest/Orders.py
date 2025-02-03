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

    def __init__(self, order_type, ticker, quantity, price=None, stop_price=None): # Added price and stop_price
        self.order_type = order_type
        self.ticker = ticker
        self.quantity = quantity
        self.price = price  # For limit orders: limit price, for stop orders: trigger price, for market orders: None
        self.stop_price = stop_price # For stop-loss orders, could be extended for more complex stop orders
        self.filled = False
        log_message = f"Created {self.order_type} order for {self.quantity} shares of {self.ticker}"
        if price is not None:
            log_message += f" at price {self.price}"
        if stop_price is not None:
            log_message += f" with stop price {self.stop_price}"
        logger.info(log_message + ".")

    def fill(self, fill_price):
        """
        Mark this order as filled at a given price.
        """
        self.filled = True
        self.price = fill_price # Update order price to fill price when filled
        logger.info(f"Order for {self.ticker} filled at price {self.price}.")