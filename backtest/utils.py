import logging

def risk_management(position_size, account_balance):
    """
    Example function for risk management. Adjust to your needs.
    """
    logger = logging.getLogger('Utils')
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)
        logger.setLevel(logging.INFO)
    if position_size * 2.0 > account_balance:
        logger.info("Risk management: Position size too large, trade disallowed.")
        return False  # disallow big trades
    logger.info("Risk management: Trade allowed.")
    return True

def concurrency_example(data):
    """
    Placeholder function to demonstrate concurrency usage (multiprocessing/threading).
    """
    logger = logging.getLogger('Utils')
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)
        logger.setLevel(logging.INFO)
    logger.info("Starting concurrency example.")
    # E.g., process data in parallel. Not implemented.
    pass

def setup_logger(name='BacktestLogger'):
    logger = logging.getLogger(name)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)
        logger.setLevel(logging.INFO)
    return logger