"""
Example script demonstrating how to use the CopulaPairTradingStrategy with user-provided data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from copulastuff import CopulaPairTradingStrategy, DataFetcher

# Function to generate sample OHLCV data for demonstration
def generate_sample_data(symbol, days=100, start_date=None):
    """Generate sample OHLCV data for demonstration purposes"""
    if start_date is None:
        start_date = datetime.now() - timedelta(days=days)
    
    # Generate dates
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # Generate random price data with some trend and volatility
    base_price = 100 if 'BTC' in symbol else 50  # Different starting prices for different symbols
    trend = np.random.choice([-0.0001, 0.0001])  # Small trend component
    
    # Generate close prices with random walk
    closes = [base_price]
    for i in range(1, days):
        # Random daily return with some autocorrelation
        daily_return = trend + np.random.normal(0, 0.02)
        closes.append(closes[-1] * (1 + daily_return))
    
    # Generate other OHLCV data based on close prices
    data = []
    for i, close in enumerate(closes):
        # Daily volatility varies
        daily_vol = np.random.uniform(0.01, 0.03) * close
        
        # Generate OHLCV data
        high = close + np.random.uniform(0, daily_vol)
        low = close - np.random.uniform(0, daily_vol)
        open_price = low + np.random.uniform(0, high - low)
        volume = np.random.uniform(100, 1000)
        
        data.append({
            'timestamp': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)

def main():
    # Option 1: Generate sample data for demonstration
    print("Generating sample data...")
    btc_data = generate_sample_data('BTC/USDT', days=100)
    eth_data = generate_sample_data('ETH/USDT', days=100)
    ltc_data = generate_sample_data('LTC/USDT', days=100)
    
    # Create a dictionary of user-provided data
    user_data = {
        'BTC/USDT': btc_data,
        'ETH/USDT': eth_data,
        'LTC/USDT': ltc_data
    }
    
    # Option 2: Load data from CSV files (commented out, for reference)
    """
    user_data = {
        'BTC/USDT': DataFetcher.load_data_from_csv('btc_usdt_data.csv', 'BTC/USDT'),
        'ETH/USDT': DataFetcher.load_data_from_csv('eth_usdt_data.csv', 'ETH/USDT'),
        'LTC/USDT': DataFetcher.load_data_from_csv('ltc_usdt_data.csv', 'LTC/USDT')
    }
    """
    
    # Initialize the strategy with user-provided data
    print("Initializing strategy with user data...")
    strategy = CopulaPairTradingStrategy(user_data=user_data)
    
    # Run the strategy
    print("Running strategy...")
    symbols = ['BTC/USDT', 'ETH/USDT', 'LTC/USDT']
    benchmark = 'BTC/USDT'
    
    results = strategy.run(symbols=symbols, benchmark=benchmark)
    
    # Print results
    print("\nStrategy Results:")
    print(f"Initial Balance: ${results['initial_balance']:.2f}")
    print(f"Final Balance: ${results['final_balance']:.2f}")
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"Win Rate: {results['win_rate']*100:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    
    # Get signal summary
    signals = strategy.get_signals_summary()
    print(f"\nGenerated {len(signals)} trading signals")
    
    return results

if __name__ == "__main__":
    main()