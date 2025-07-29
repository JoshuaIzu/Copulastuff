# -*- coding: utf-8 -*-

import numpy as np

import pandas as pd

import ccxt

from scipy import stats

from statsmodels.tsa.stattools import adfuller, kpss

from statsmodels.stats.diagnostic import het_arch

from arch import arch_model

from copulas.bivariate import Clayton

from sklearn.exceptions import NotFittedError
import logging
from typing import Dict, Tuple, Optional, List

import warnings

from dataclasses import dataclass

from datetime import datetime

import time

# Configure numerical stability

np.seterr(all='raise')

pd.options.mode.chained_assignment = 'raise'


# Configuration

@dataclass
class StrategyConfig:
    # Trading parameters

    quantile_threshold: float = 0.95

    vol_threshold: float = 1.5

    stop_loss_pct: float = 0.05

    trade_fee: float = 0.001  # 0.1%

    max_position_size: float = 0.1  # 10% of portfolio

    max_concurrent_trades: int = 3

    # Statistical testing

    adf_critical: float = 0.05

    kpss_critical: float = 0.05

    arch_critical: float = 0.05

    # Data parameters

    lookback_window: int = 1000  # ~40 days of hourly data

    volatility_window: int = 20  # 20 periods for vol calculation

    min_volatility: float = 0.001  # Avoid division by zero

    # Backtest

    initial_balance: float = 10000.0

    slippage: float = 0.0005  # 5bps

    shock_prob: float = 0.05  # For perturbation tests

    max_shock: float = 0.15  # 15% max price shock


config = StrategyConfig()

# Logging setup

logging.basicConfig(

    level=logging.INFO,

    format='%(asctime)s - %(levelname)s - %(message)s',

    handlers=[

        logging.FileHandler('pair_trading.log'),

        logging.StreamHandler()

    ]

)

logger = logging.getLogger(__name__)


class DataFetcher:
    """Robust data fetching with error handling and retries"""

    def __init__(self, exchange_id='binance'):

        self.exchange = getattr(ccxt, exchange_id)({

            'enableRateLimit': True,

            'options': {'adjustForTimeDifference': True}

        })

        self.last_fetch_time = 0

    def fetch_ohlcv(self, symbol: str, timeframe='1h', limit=1000) -> pd.DataFrame:

        """Fetch OHLCV data with rate limiting and validation"""

        max_retries = 5

        for attempt in range(max_retries):

            try:

                # Rate limiting

                elapsed = time.time() - self.last_fetch_time

                if elapsed < 1.0:  # Binance rate limit

                    time.sleep(1.0 - elapsed)

                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

                df = pd.DataFrame(

                    ohlcv,

                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']

                )

                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                df.set_index('timestamp', inplace=True)

                df = self._validate_data(df, symbol)

                self.last_fetch_time = time.time()

                return df

            except Exception as e:

                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")

                if attempt == max_retries - 1:
                    logger.error(f"Failed after {max_retries} attempts for {symbol}")

                    raise

                time.sleep(2 ** attempt)  # Exponential backoff

    def _validate_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:

        """Validate and clean OHLCV data"""

        if df.isnull().values.any():

            logger.warning(f"NaN values in {symbol}, filling gaps")

            df.ffill(inplace=True)

            if df.isnull().any():
                df.bfill(inplace=True)

        if len(df) < config.lookback_window:
            raise ValueError(f"Insufficient data for {symbol} (got {len(df)}, need {config.lookback_window})")

        # Validate price sanity

        if (df['close'] <= 0).any():
            raise ValueError(f"Invalid close prices (<=0) in {symbol}")

        return df


class StationarityTester:
    """Robust stationarity and heteroskedasticity testing"""

    def __init__(self):

        self.cache = {}

    def test_series(self, series: pd.Series, series_name: str) -> dict:

        """Run battery of statistical tests with caching"""

        cache_key = f"{series_name}_{hash(tuple(series.values[~np.isnan(series.values)])):x}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        try:

            results = {

                'adf': adfuller(series.dropna())[1],

                'kpss': kpss(series.dropna())[1],

                'arch': het_arch(series.dropna())[1]

            }

            self.cache[cache_key] = results

            return results

        except Exception as e:

            logger.error(f"Statistical testing failed for {series_name}: {str(e)}")

            return {'adf': 1.0, 'kpss': 0.0, 'arch': 1.0}  # Conservative defaults

    def needs_standardization(self, test_results: dict) -> bool:

        """Determine if GARCH standardization is needed"""

        return (

                test_results['adf'] > config.adf_critical or

                test_results['kpss'] < config.kpss_critical or

                test_results['arch'] < config.arch_critical

        )


class GARCHStandardizer:
    """GARCH modeling with robust error handling"""

    def __init__(self):

        self.models = {}

    def standardize(self, series: pd.Series, series_name: str) -> pd.Series:

        """Standardize series using GARCH(1,1) with fallbacks"""

        if series_name in self.models:
            return self._predict_with_model(self.models[series_name], series)

        try:

            # Scale for numerical stability

            scaled = series / series.std()

            # Fit GARCH with multiple start params

            for p in [1, 2]:

                for q in [1, 2]:

                    try:

                        model = arch_model(

                            scaled,

                            vol='Garch',

                            p=p,

                            q=q,

                            mean='Constant',

                            dist='normal'

                        )

                        res = model.fit(disp='off')

                        if np.isfinite(res.params).all():
                            self.models[series_name] = res

                            return self._predict_with_model(res, scaled)

                    except Exception:

                        continue

            # Fallback to simple standardization if GARCH fails

            logger.warning(f"GARCH failed for {series_name}, using simple standardization")

            return scaled / scaled.std()

        except Exception as e:

            logger.error(f"Standardization failed for {series_name}: {str(e)}")

            return series / max(series.std(), config.min_volatility)

    def _predict_with_model(self, model, series: pd.Series) -> pd.Series:

        """Get standardized residuals from fitted model"""

        try:

            return model.std_resid

        except Exception:

            # Handle cases where model exists but prediction fails

            return series / max(series.std(), config.min_volatility)


class CopulaModel:
    """Clayton copula implementation with robust error handling"""

    def __init__(self):

        self.fitted_copulas = {}

    def fit_copula(self, U: np.ndarray, V: np.ndarray, pair_name: str) -> Clayton:

        """Fit Clayton copula with validation and fallbacks"""

        cache_key = f"{pair_name}_{hash(tuple(U))}_{hash(tuple(V))}"

        if cache_key in self.fitted_copulas:
            return self.fitted_copulas[cache_key]

        try:

            # Check for perfect correlation

            if np.allclose(U, V, atol=1e-6):
                raise ValueError("Perfect correlation detected")

            # Check for sufficient unique values

            if len(np.unique(U)) < 10 or len(np.unique(V)) < 10:
                raise ValueError("Insufficient unique values")

            copula = Clayton()

            copula.fit(np.column_stack((U, V)))

            # Validate fit

            if not hasattr(copula, 'theta') or not np.isfinite(copula.theta):
                raise ValueError("Invalid theta estimate")

            self.fitted_copulas[cache_key] = copula

            return copula

        except Exception as e:

            logger.warning(f"Copula fit failed for {pair_name}: {str(e)}")

            # Return dummy copula that generates neutral signals

            class DummyCopula:

                def partial_derivative(self, x):
                    return np.full(x.shape[0], 0.5)

            return DummyCopula()

    def generate_signals(

            self,

            U: np.ndarray,

            V: np.ndarray,

            copula: Clayton,

            quantile_threshold: float

    ) -> Dict[str, np.ndarray]:

        """Generate trading signals with boundary checks"""

        try:

            # Clip probabilities to avoid edge cases

            cond_U_given_V = np.clip(

                copula.partial_derivative(np.column_stack((U, V))),

                0.01, 0.99

            )

            cond_V_given_U = np.clip(

                copula.partial_derivative(np.column_stack((V, U))),

                0.01, 0.99

            )

            return {

                'long_X': cond_U_given_V < (1 - quantile_threshold),

                'short_X': cond_U_given_V > quantile_threshold,

                'long_Y': cond_V_given_U < (1 - quantile_threshold),

                'short_Y': cond_V_given_U > quantile_threshold

            }

        except Exception as e:

            logger.error(f"Signal generation failed: {str(e)}")

            # Return neutral signals on failure

            neutral = np.zeros_like(U, dtype=bool)

            return {

                'long_X': neutral,

                'short_X': neutral,

                'long_Y': neutral,

                'short_Y': neutral

            }


class PortfolioManager:
    """Handles position sizing and risk management"""

    def __init__(self, initial_balance: float):

        self.balance = initial_balance

        self.positions = {}  # {pair: {'direction': int, 'size': float, ...}}

        self.trade_history = []

        self.portfolio_values = []

        self.volatility_cache = {}

    def calculate_position_size(

            self,

            price_btc: float,

            price_alt: float,

            alt_volatility: float,

            btc_volatility: float

    ) -> float:

        """Dynamic position sizing based on volatility and current exposure"""

        # Calculate current risk exposure

        current_risk = sum(

            pos['size'] for pos in self.positions.values()

        )

        available_risk = min(

            config.max_position_size,

            (config.max_position_size - current_risk) / 2  # Conservative

        )

        if available_risk <= 0:
            return 0.0

        # Volatility-adjusted sizing

        combined_vol = np.sqrt(alt_volatility ** 2 + btc_volatility ** 2)

        if combined_vol < config.min_volatility:
            combined_vol = config.min_volatility

        max_size = available_risk / combined_vol

        return min(config.max_position_size, max_size)

    def get_volatility(self, prices: pd.Series) -> float:

        """Calculate rolling volatility with caching"""

        cache_key = hash(tuple(prices.values))

        if cache_key in self.volatility_cache:
            return self.volatility_cache[cache_key]

        returns = np.log(prices).diff().dropna()

        if len(returns) < 10:  # Minimum data points

            vol = config.min_volatility

        else:

            vol = returns.rolling(config.volatility_window).std().iloc[-1]

            if not np.isfinite(vol):
                vol = config.min_volatility

        self.volatility_cache[cache_key] = max(vol, config.min_volatility)

        return self.volatility_cache[cache_key]


class Backtester:
    """Event-driven backtesting engine"""

    def __init__(self, initial_balance: float = config.initial_balance):

        self.portfolio = PortfolioManager(initial_balance)

        self.current_prices = {}

    def execute_trade(

            self,

            pair: str,

            direction: int,

            price_btc: float,

            price_alt: float,

            timestamp: pd.Timestamp

    ) -> None:

        """Execute trade with slippage and fees"""

        # Update current prices

        self.current_prices[pair] = (price_btc, price_alt)

        # Exit existing position if direction is 0 or changing

        if pair in self.portfolio.positions:
            self._exit_position(pair, price_btc, price_alt, timestamp)

        # Enter new position if direction is not 0

        if direction != 0:
            self._enter_position(pair, direction, price_btc, price_alt, timestamp)

        # Record portfolio value

        self.portfolio.portfolio_values.append({

            'timestamp': timestamp,

            'value': self.portfolio.balance,

            'n_positions': len(self.portfolio.positions)

        })

    def _exit_position(

            self,

            pair: str,

            price_btc: float,

            price_alt: float,

            timestamp: pd.Timestamp

    ) -> None:

        """Close existing position and calculate PnL"""

        position = self.portfolio.positions.pop(pair)

        # Apply slippage (exit trades incur slippage)

        exit_price_btc = price_btc * (1 + config.slippage * position['direction'])

        exit_price_alt = price_alt * (1 - config.slippage * position['direction'])

        # Calculate returns

        btc_return = (exit_price_btc / position['entry_btc'] - 1) * position['direction']

        alt_return = (position['entry_alt'] / exit_price_alt - 1) * position['direction']

        net_return = btc_return + alt_return

        # Calculate PnL

        pnl = net_return * position['size'] * position['entry_balance']

        # Apply fees

        pnl -= 2 * config.trade_fee * position['size'] * position['entry_balance']

        # Update balance

        self.portfolio.balance += pnl

        # Record trade

        self.portfolio.trade_history.append({

            'timestamp': timestamp,

            'pair': pair,

            'action': 'exit',

            'direction': position['direction'],

            'size': position['size'],

            'pnl': pnl,

            'balance': self.portfolio.balance,

            'return': net_return

        })

    def _enter_position(

            self,

            pair: str,

            direction: int,

            price_btc: float,

            price_alt: float,

            timestamp: pd.Timestamp

    ) -> None:

        """Open new position with risk management"""

        # Get volatilities

        btc_vol = self.portfolio.get_volatility(

            pd.Series([p[0] for p in self.current_prices.values()])

        )

        alt_vol = self.portfolio.get_volatility(

            pd.Series([p[1] for p in self.current_prices.values()])

        )

        # Calculate position size

        position_size = self.portfolio.calculate_position_size(

            price_btc, price_alt, alt_vol, btc_vol

        )

        if position_size <= 0:
            return

        # Apply slippage (entry trades incur slippage)

        entry_price_btc = price_btc * (1 - config.slippage * direction)

        entry_price_alt = price_alt * (1 + config.slippage * direction)

        # Open position

        self.portfolio.positions[pair] = {

            'direction': direction,

            'entry_btc': entry_price_btc,

            'entry_alt': entry_price_alt,

            'size': position_size,

            'entry_balance': self.portfolio.balance,

            'stop_loss': config.stop_loss_pct,

            'timestamp': timestamp

        }

        # Record trade

        self.portfolio.trade_history.append({

            'timestamp': timestamp,

            'pair': pair,

            'action': 'enter',

            'direction': direction,

            'size': position_size,

            'balance': self.portfolio.balance

        })

    def check_stop_losses(self, timestamp: pd.Timestamp) -> None:

        """Check all positions for stop-loss triggers"""

        for pair, position in list(self.portfolio.positions.items()):

            if pair not in self.current_prices:
                continue

            price_btc, price_alt = self.current_prices[pair]

            # Calculate current return

            btc_return = (price_btc / position['entry_btc'] - 1) * position['direction']

            alt_return = (position['entry_alt'] / price_alt - 1) * position['direction']

            net_return = btc_return + alt_return

            # Check stop loss

            if net_return < -position['stop_loss']:
                logger.info(f"Stop loss triggered for {pair} at {timestamp}")

                self.execute_trade(pair, 0, price_btc, price_alt, timestamp)


class CopulaPairTradingStrategy:
    """Complete strategy implementation"""

    def __init__(self):

        self.data_fetcher = DataFetcher()

        self.stationarity_tester = StationarityTester()

        self.garch_standardizer = GARCHStandardizer()

        self.copula_model = CopulaModel()

        self.backtester = Backtester()

    def run(

            self,

            symbols: List[str],

            benchmark: str,

            timeframe: str = '1h',

            perturb: bool = False

    ) -> Dict:

        """Run complete strategy pipeline"""

        try:

            # Fetch and prepare data

            data = self._fetch_and_prepare_data(symbols, timeframe, perturb)

            # Process each pair

            for symbol in symbols:

                if symbol == benchmark:
                    continue

                try:

                    self._process_pair(benchmark, symbol, data)

                except Exception as e:

                    logger.error(f"Failed to process {symbol}: {str(e)}")

                    continue

            # Generate performance report

            return self._generate_report()

        except Exception as e:

            logger.critical(f"Strategy failed: {str(e)}", exc_info=True)

            return {'error': str(e)}

    def _fetch_and_prepare_data(

            self,

            symbols: List[str],

            timeframe: str,

            perturb: bool

    ) -> Dict[str, pd.DataFrame]:

        """Fetch and optionally perturb market data"""

        data = {}

        for symbol in symbols:
            data[symbol] = self.data_fetcher.fetch_ohlcv(

                symbol,

                timeframe=timeframe,

                limit=config.lookback_window

            )

        if perturb:
            return self._perturb_data(data)

        return data

    def _perturb_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:

        """Apply realistic market shocks"""

        perturbed = {}

        shock_dates = sorted([

            d for d in data[list(data.keys())[0]].index

            if np.random.random() < config.shock_prob

        ])

        for symbol, df in data.items():

            p_df = df.copy()

            for date in shock_dates:

                # Correlated shocks with persistence

                shock = np.random.uniform(

                    -config.max_shock,

                    config.max_shock

                )

                for col in ['open', 'high', 'low', 'close']:
                    p_df.loc[date:, col] = p_df.loc[date:, col] * (1 + shock)

                # Add volatility cluster

                idx = p_df.index.get_loc(date)

                window = p_df.iloc[idx:idx + 10]

                p_df.loc[window.index, 'close'] = window['close'] * np.cumprod(

                    1 + np.random.normal(0, 0.01, len(window))

                )

            perturbed[symbol] = p_df

        return perturbed

    def _process_pair(

            self,

            benchmark: str,

            symbol: str,

            data: Dict[str, pd.DataFrame]

    ) -> None:

        """Process a single trading pair"""

        pair_name = f"{benchmark[:3]}_{symbol[:3]}"

        logger.info(f"Processing pair: {pair_name}")

        # Prepare price series

        pair_data = pd.DataFrame({

            benchmark: data[benchmark]['close'],

            symbol: data[symbol]['close']

        }).dropna()

        # Calculate log returns

        log_returns = np.log(pair_data).diff().dropna()

        log_X = log_returns[benchmark]

        log_Y = log_returns[symbol]

        # Stationarity and GARCH processing

        test_X = self.stationarity_tester.test_series(log_X, f"{benchmark}_returns")

        test_Y = self.stationarity_tester.test_series(log_Y, f"{symbol}_returns")

        if self.stationarity_tester.needs_standardization(test_X):
            log_X = self.garch_standardizer.standardize(log_X, f"{benchmark}_returns")

        if self.stationarity_tester.needs_standardization(test_Y):
            log_Y = self.garch_standardizer.standardize(log_Y, f"{symbol}_returns")

        # Transform to uniform margins

        U = stats.rankdata(log_X) / (len(log_X) + 1)

        V = stats.rankdata(log_Y) / (len(log_Y) + 1)

        # Fit copula

        copula = self.copula_model.fit_copula(U, V, pair_name)

        # Generate signals

        signals = self.copula_model.generate_signals(

            U, V, copula, config.quantile_threshold

        )

        # Calculate volatility filter

        returns_Y = log_Y.diff()

        volatility = returns_Y.rolling(config.volatility_window).std().shift(1)

        vol_condition = volatility > (

                volatility.ewm(span=100).mean() * config.vol_threshold

        )

        # Execute trades

        for i in range(1, len(pair_data)):

            timestamp = pair_data.index[i]

            price_btc = pair_data[benchmark].iloc[i]

            price_alt = pair_data[symbol].iloc[i]

            # Update current prices

            self.backtester.current_prices[pair_name] = (price_btc, price_alt)

            # Check stop losses first

            self.backtester.check_stop_losses(timestamp)

            # Generate trade signal if volatility condition met

            if vol_condition.iloc[i - 1] if i > 0 else False:

                signal_idx = i - 1  # Signals are offset by 1

                if signals['long_Y'][signal_idx] and signals['short_X'][signal_idx]:

                    # Long altcoin, short BTC

                    self.backtester.execute_trade(

                        pair_name, 1, price_btc, price_alt, timestamp

                    )

                elif signals['short_Y'][signal_idx] and signals['long_X'][signal_idx]:

                    # Short altcoin, long BTC

                    self.backtester.execute_trade(

                        pair_name, -1, price_btc, price_alt, timestamp

                    )

                elif pair_name in self.backtester.portfolio.positions:

                    # Close position if no signal

                    self.backtester.execute_trade(

                        pair_name, 0, price_btc, price_alt, timestamp

                    )

            # Record portfolio value

            self.backtester.portfolio.portfolio_values.append({

                'timestamp': timestamp,

                'value': self.backtester.portfolio.balance,

                'n_positions': len(self.backtester.portfolio.positions)

            })

    def _generate_report(self) -> Dict:

        """Generate performance analytics report"""

        if not self.backtester.portfolio.trade_history:
            return {'error': 'No trades executed'}

        trades = pd.DataFrame(self.backtester.portfolio.trade_history)

        portfolio = pd.DataFrame(self.backtester.portfolio.portfolio_values)

        # Calculate metrics

        winning_trades = trades[trades['pnl'] > 0]

        losing_trades = trades[trades['pnl'] < 0]

        total_return_pct = (

                (self.backtester.portfolio.balance / config.initial_balance - 1) * 100

        )

        return {

            'initial_balance': config.initial_balance,

            'final_balance': self.backtester.portfolio.balance,

            'total_return_pct': total_return_pct,

            'total_trades': len(trades),

            'win_rate': len(winning_trades) / len(trades),

            'profit_factor': (

                winning_trades['pnl'].sum() / abs(losing_trades['pnl'].sum())

                if len(losing_trades) > 0 else np.inf

            ),

            'max_drawdown_pct': self._calculate_max_drawdown(portfolio) * 100,

            'sharpe_ratio': self._calculate_sharpe(portfolio),

            'avg_trade_duration': self._calculate_avg_trade_duration(trades),

            'trades': trades.to_dict('records'),

            'portfolio_values': portfolio.to_dict('records')

        }

    def _calculate_max_drawdown(self, portfolio: pd.DataFrame) -> float:

        """Calculate maximum drawdown"""

        peak = portfolio['value'].cummax()

        drawdown = (portfolio['value'] - peak) / peak

        return drawdown.min()

    def _calculate_sharpe(self, portfolio: pd.DataFrame) -> float:

        """Calculate annualized Sharpe ratio"""

        returns = portfolio['value'].pct_change().dropna()

        if len(returns) < 2:
            return 0.0

        return (returns.mean() / returns.std()) * np.sqrt(365 * 24)  # Hourly data

    def _calculate_avg_trade_duration(self, trades: pd.DataFrame) -> str:

        """Calculate average trade duration"""

        entry_exit = {}

        durations = []

        for _, trade in trades.iterrows():

            if trade['action'] == 'enter':

                entry_exit[trade['pair']] = trade['timestamp']

            elif trade['action'] == 'exit' and trade['pair'] in entry_exit:

                durations.append((trade['timestamp'] - entry_exit[trade['pair']]).total_seconds())

        if not durations:
            return "0 days"

        avg_seconds = np.mean(durations)

        return f"{avg_seconds / 86400:.1f} days"


if __name__ == "__main__":

    # Initialize and run strategy

    strategy = CopulaPairTradingStrategy()

    # Define assets

    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT']

    benchmark = 'BTC/USDT'

    # Run on normal data

    logger.info("Running strategy on normal data")

    normal_results = strategy.run(symbols, benchmark)

    # Run with perturbations

    logger.info("Running strategy with market shocks")

    perturbed_results = strategy.run(symbols, benchmark, perturb=True)

    # Print comparison

    print("\n=== Performance Comparison ===")

    print(f"{'Metric':<25} {'Normal':>10} {'Perturbed':>10}")

    for metric in ['total_return_pct', 'win_rate', 'max_drawdown_pct', 'sharpe_ratio']:
        print(f"{metric:<25} {normal_results.get(metric, 0):>10.2f} {perturbed_results.get(metric, 0):>10.2f}")
