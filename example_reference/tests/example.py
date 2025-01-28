from backtesting_engine import DataHandler, BacktestEngine
from backtesting_engine.strategies import MovingAverageCrossover

def main():
    # Initialize components
    data_handler = DataHandler()
    
    # Load data for multiple symbols
    symbols = ['AAPL', 'GOOGL']
    for symbol in symbols:
        files = [f'data/{symbol}_*.csv']  # Your CSV files
        data_handler.load_data(files, symbol)
    
    # Create and run strategy
    strategy = MovingAverageCrossover(symbols)
    engine = BacktestEngine(data_handler, strategy)
    results = engine.run(start_date='2024-01-01')
    
    # Print results
    for symbol, df in results.items():
        print(f"\nResults for {symbol}:")
        print(f"Final Return: {df['returns'].iloc[-1]:.2%}")
        print(f"Number of Trades: {(df['signal'] != 0).sum()}")

if __name__ == '__main__':
    main()