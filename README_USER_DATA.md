# Using Your Own Data with the Copula Pair Trading Strategy

This document explains how to use your own data with the Copula Pair Trading Strategy instead of fetching data from Binance.

## Changes Made

The following changes have been made to the codebase to support user-provided data:

1. Modified the `DataFetcher` class to accept user-provided data
2. Added a method to load user data from CSV files
3. Updated the `CopulaPairTradingStrategy` class to pass user data to the `DataFetcher`
4. Created an example script demonstrating how to use the strategy with user-provided data

## How to Use Your Own Data

There are several ways to use your own data with the strategy:

### Option 1: Provide Data as DataFrames

```python
import pandas as pd
from copulastuff import CopulaPairTradingStrategy

# Create or load your data as pandas DataFrames
btc_data = pd.DataFrame({
    'timestamp': [...],  # datetime objects
    'open': [...],       # float values
    'high': [...],       # float values
    'low': [...],        # float values
    'close': [...],      # float values
    'volume': [...]      # float values
})

eth_data = pd.DataFrame({
    # Similar structure as btc_data
})

# Create a dictionary with symbol names as keys and DataFrames as values
user_data = {
    'BTC/USDT': btc_data,
    'ETH/USDT': eth_data
}

# Initialize the strategy with your data
strategy = CopulaPairTradingStrategy(user_data=user_data)

# Run the strategy
results = strategy.run(
    symbols=['BTC/USDT', 'ETH/USDT'],
    benchmark='BTC/USDT'
)
```

### Option 2: Load Data from CSV Files

```python
from copulastuff import CopulaPairTradingStrategy, DataFetcher

# Load data from CSV files
user_data = {
    'BTC/USDT': DataFetcher.load_data_from_csv('btc_usdt_data.csv', 'BTC/USDT'),
    'ETH/USDT': DataFetcher.load_data_from_csv('eth_usdt_data.csv', 'ETH/USDT')
}

# Initialize the strategy with your data
strategy = CopulaPairTradingStrategy(user_data=user_data)

# Run the strategy
results = strategy.run(
    symbols=['BTC/USDT', 'ETH/USDT'],
    benchmark='BTC/USDT'
)
```

### CSV File Format

Your CSV files should have the following columns:
- A timestamp column (named 'timestamp', 'date', 'time', or 'datetime')
- 'open', 'high', 'low', 'close', and 'volume' columns (case-insensitive)

Example CSV format:
```
timestamp,open,high,low,close,volume
2023-01-01 00:00:00,50000,51000,49500,50500,1000
2023-01-02 00:00:00,50500,52000,50000,51500,1200
...
```

If your timestamp format is non-standard, you can specify the format:
```python
DataFetcher.load_data_from_csv('btc_usdt_data.csv', 'BTC/USDT', timestamp_format='%Y-%m-%d')
```

## Example Script

An example script `example_user_data.py` is provided that demonstrates how to use the strategy with user-provided data. It includes:

1. A function to generate sample data for demonstration
2. Code to initialize the strategy with the sample data
3. Code to run the strategy and display the results

You can run this script to see how the strategy works with user-provided data:

```
python example_user_data.py
```

## Data Requirements

For the strategy to work correctly, your data should:

1. Have sufficient history (at least `config.lookback_window` data points, default is 1000)
2. Be sorted by timestamp in ascending order
3. Have no missing values
4. Have valid OHLCV values (high >= open >= low, high >= close >= low)
5. Have positive values for open, high, low, close, and volume

The `DataFetcher.load_data_from_csv` method and the `DataFetcher._validate_data` method will check these requirements and raise errors if they are not met.

## Troubleshooting

If you encounter issues with your data:

1. Check that your data has the required columns
2. Ensure your timestamps are in a format that pandas can parse
3. Verify that your data has sufficient history
4. Check for missing or invalid values

If you still have issues, you can enable debug logging to see more detailed error messages:

```python
import logging
logging.getLogger('copulastuff').setLevel(logging.DEBUG)
```