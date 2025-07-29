import numpy as np
import pandas as pd
import ccxt
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import het_arch
from arch import arch_model
from copulas.bivariate import Clayton
#from copulas.utils import NotFittedError
from sklearn.exceptions import NotFittedError
import logging
from typing import Dict, Tuple, Optional, List, Any
import warnings
from dataclasses import dataclass
from datetime import datetime
import time
from pydantic import BaseModel, field_validator, Field, PositiveFloat, confloat
from typing import Literal

# Configure numerical stability
np.seterr(all='raise')
pd.options.mode.chained_assignment = 'raise'


# Pydantic Models
class OHLCVDataPoint(BaseModel):
    timestamp: datetime
    open: PositiveFloat
    high: PositiveFloat
    low: PositiveFloat
    close: PositiveFloat
    volume: confloat(ge=0)

    @field_validator('high')
    def high_must_be_highest(cls, v, info):
        values = info.data
        if 'open' in values and v < values['open']:
            raise ValueError('High must be >= open')
        if 'low' in values and v < values['low']:
            raise ValueError('High must be >= low')
        return v

    @field_validator('low')
    def low_must_be_lowest(cls, v, info):
        values = info.data
        if 'open' in values and v > values['open']:
            raise ValueError('Low must be <= open')
        if 'high' in values and v > values['high']:
            raise ValueError('Low must be <= high')
        return v


class TradeSignal(BaseModel):
    pair: str
    direction: Literal[-1, 0, 1]
    confidence: confloat(ge=0, le=1)
    timestamp: datetime


class PortfolioPosition(BaseModel):
    asset: str
    size: confloat(gt=0)
    entry_price: PositiveFloat
    entry_time: datetime
    stop_loss: confloat(gt=0)


class PerformanceReport(BaseModel):
    initial_balance: PositiveFloat
    final_balance: float
    total_return_pct: float
    total_trades: int
    win_rate: confloat(ge=0, le=1)
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_trade_duration: str
    trades: List[Dict[str, Any]]
    portfolio_values: List[Dict[str, Any]]


# Configuration
@dataclass
class StrategyConfig:
    # Trading parameters
    quantile_threshold: float = 0.85  # Balanced value for selective trading
    vol_threshold: float = 1.2  # Balanced value for volatility filtering
    stop_loss_pct: float = 0.025  # Balanced stop loss
    take_profit_pct: float = 0.04  # Balanced take profit
    trailing_stop_pct: float = 0.02  # Balanced trailing stop
    trade_fee: float = 0.001  # 0.1%
    max_position_size: float = 0.08  # Reduced from 10% to 8% for conservative sizing
    max_concurrent_trades: int = 3

    # Signal confirmation
    momentum_window: int = 14  # RSI window
    momentum_threshold: float = 30.0  # Oversold threshold
    mean_reversion_window: int = 20  # Bollinger Band window
    mean_reversion_std: float = 2.0  # Standard deviations for Bollinger Bands

    # Statistical testing
    adf_critical: float = 0.05
    kpss_critical: float = 0.05
    arch_critical: float = 0.05

    # Data parameters
    lookback_window: int = 1000  # ~40 days of hourly data
    volatility_window: int = 20  # 20 periods for vol calculation
    min_volatility: float = 0.001  # Avoid division by zero

    # Market regime detection
    regime_detection_window: int = 50  # Window for detecting market regime
    bull_threshold: float = 0.6  # Percentage of positive returns to be considered bullish

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
    """Robust data fetching with enhanced error handling and data quality assurance"""

    def __init__(self, exchange_id='binance', user_data=None):
        # Support multiple exchanges with failover capability
        self.exchange_ids = ['binance', 'kucoin', 'okex', 'bybit']
        if exchange_id not in self.exchange_ids:
            self.exchange_ids.insert(0, exchange_id)
        else:
            self.exchange_ids.insert(0, self.exchange_ids.pop(self.exchange_ids.index(exchange_id)))

        self.exchanges = {}
        self.initialize_exchanges()
        self.last_fetch_time = 0
        self.user_data = user_data  # Dictionary of user-provided data {symbol: DataFrame}

    def initialize_exchanges(self):
        """Initialize exchange connections with error handling"""
        for exchange_id in self.exchange_ids:
            try:
                self.exchanges[exchange_id] = getattr(ccxt, exchange_id)({
                    'enableRateLimit': True,
                    'options': {'adjustForTimeDifference': True}
                })
                logger.info(f"Successfully initialized {exchange_id} exchange")
            except Exception as e:
                logger.warning(f"Failed to initialize {exchange_id} exchange: {str(e)}")

        if not self.exchanges:
            raise RuntimeError("Failed to initialize any exchanges")

    def load_user_data(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Load user-provided data for a symbol
        
        Args:
            symbol: The symbol name (e.g., 'BTC/USDT')
            data: DataFrame with OHLCV data. Must have columns: timestamp (or index), open, high, low, close, volume
            
        Returns:
            Validated DataFrame with OHLCV data
        """
        if self.user_data is None:
            self.user_data = {}
            
        # Ensure data has the correct format
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Data for {symbol} must be a pandas DataFrame")
            
        # Check if data has the required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # If timestamp is the index, that's fine, otherwise it should be a column
        if 'timestamp' not in data.columns and not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError(f"Data for {symbol} must have a 'timestamp' column or DatetimeIndex")
            
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Data for {symbol} is missing required columns: {missing_columns}")
            
        # Convert to standard format
        df = data.copy()
        
        # If timestamp is a column, set it as index
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
            
        # Validate the data
        df = self._validate_data(df, symbol)
        
        # Store the validated data
        self.user_data[symbol] = df
        
        logger.info(f"Successfully loaded user data for {symbol}")
        return df
        
    def fetch_ohlcv(self, symbol: str, timeframe='1h', limit=1000) -> pd.DataFrame:
        """Fetch OHLCV data with rate limiting and validation
        
        If user_data is provided for this symbol, use that instead of fetching from exchange
        """
        # Check if we have user-provided data for this symbol
        if self.user_data is not None and symbol in self.user_data:
            logger.info(f"Using user-provided data for {symbol}")
            return self.user_data[symbol]
            
        # Otherwise fetch from exchange
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Rate limiting
                elapsed = time.time() - self.last_fetch_time
                if elapsed < 1.0:  # Binance rate limit
                    time.sleep(1.0 - elapsed)

                # Try each exchange in order until one succeeds
                last_error = None
                for exchange_id, exchange in self.exchanges.items():
                    try:
                        raw_data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                        validated_data = self._validate_with_pydantic(raw_data, symbol)
                        df = self._convert_to_dataframe(validated_data)
                        df = self._validate_data(df, symbol)
                        self.last_fetch_time = time.time()
                        logger.info(f"Successfully fetched data from {exchange_id}")
                        return df
                    except Exception as e:
                        last_error = e
                        logger.warning(f"Failed to fetch from {exchange_id}: {str(e)}")
                        continue

                # If we get here, all exchanges failed
                raise RuntimeError(f"All exchanges failed to fetch {symbol}: {str(last_error)}")

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed after {max_retries} attempts for {symbol}")
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

    def _validate_with_pydantic(self, raw_data: list, symbol: str) -> list[OHLCVDataPoint]:
        """Validate each data point with Pydantic"""
        validated = []
        for item in raw_data:
            try:
                data = OHLCVDataPoint(
                    timestamp=datetime.fromtimestamp(item[0] / 1000),
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5])
                )
                validated.append(data)
            except Exception as e:
                logger.error(f"Validation failed for {symbol} data point: {e}")
                continue
        return validated

    def _convert_to_dataframe(self, validated_data: list[OHLCVDataPoint]) -> pd.DataFrame:
        """Convert validated data to DataFrame"""
        data_dicts = [item.model_dump() for item in validated_data]
        df = pd.DataFrame(data_dicts)
        df.set_index('timestamp', inplace=True)
        return df

    def _validate_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Additional pandas-based validation"""
        if df.isnull().values.any():
            logger.warning(f"NaN values in {symbol}, filling gaps")
            df.ffill(inplace=True)
            if df.isnull().any():
                df.bfill(inplace=True)

        if len(df) < config.lookback_window:
            raise ValueError(f"Insufficient data for {symbol} (got {len(df)}, need {config.lookback_window})")

        return df


class StationarityTester:
    """Robust stationarity and heteroskedasticity testing"""

    def __init__(self):
        self.cache = {}

    def test_series(self, series: pd.Series, series_name: str) -> dict:
        """Run battery of statistical tests with enhanced robustness"""
        cache_key = f"{series_name}_{hash(tuple(series.values[~np.isnan(series.values)])):x}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            # Prepare data - ensure sufficient length and no extreme values
            clean_series = series.dropna()

            # If series is too short, return conservative defaults
            if len(clean_series) < 20:
                logger.warning(f"Series {series_name} too short for reliable testing")
                return {'adf': 1.0, 'kpss': 0.05, 'arch': 0.5}

            # Handle extreme values that might cause statistical issues
            if clean_series.std() < 1e-8:
                logger.warning(f"Series {series_name} has near-zero variance")
                return {'adf': 1.0, 'kpss': 0.05, 'arch': 0.5}

            # Run tests with try/except for each test
            try:
                adf_result = adfuller(clean_series)
                adf_pval = adf_result[1]
            except Exception as e:
                logger.warning(f"ADF test failed for {series_name}: {str(e)}")
                adf_pval = 1.0

            try:
                # Suppress warnings about p-value range
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kpss_result = kpss(clean_series)
                kpss_pval = kpss_result[1]
            except Exception as e:
                logger.warning(f"KPSS test failed for {series_name}: {str(e)}")
                kpss_pval = 0.05

            try:
                arch_result = het_arch(clean_series)
                arch_pval = arch_result[1]
            except Exception as e:
                logger.warning(f"ARCH test failed for {series_name}: {str(e)}")
                arch_pval = 0.5

            # Ensure p-values are within valid range (0.001, 1.0)
            adf_pval = min(max(adf_pval, 0.001), 1.0)
            kpss_pval = min(max(kpss_pval, 0.001), 1.0)
            arch_pval = min(max(arch_pval, 0.001), 1.0)

            results = {
                'adf': adf_pval,
                'kpss': kpss_pval,
                'arch': arch_pval
            }
            self.cache[cache_key] = results
            return results
        except Exception as e:
            logger.error(f"Statistical testing failed for {series_name}: {str(e)}")
            return {'adf': 1.0, 'kpss': 0.05, 'arch': 0.5}  # Conservative defaults

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
            data = np.column_stack((U, V))
            copula.fit(data)

            # Validate fit
            if not hasattr(copula, 'theta') or not np.isfinite(copula.theta):
                raise ValueError("Invalid theta estimate")

            # Test partial derivative calculation
            try:
                test_input = np.array([[0.5, 0.5]])
                _ = copula.partial_derivative(test_input)
            except Exception as e:
                raise ValueError(f"Partial derivative test failed: {str(e)}")

            self.fitted_copulas[cache_key] = copula
            return copula
        except Exception as e:
            logger.warning(f"Copula fit failed for {pair_name}: {str(e)}")

            # Return a properly configured  copula
            class DummyCopula:
                def __init__(self):
                    self.theta = 0

                def partial_derivative(self, x):
                    return np.full(x.shape[0], 0.5)

                def probability_density(self, x):
                    return np.ones(x.shape[0])

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
            # Prepare input data
            uv = np.column_stack((U, V))
            vu = np.column_stack((V, U))

            # Calculate partial derivatives with error handling
            try:
                cond_U_given_V = copula.partial_derivative(uv)
            except Exception as e:
                logger.warning(f"Partial derivative U|V failed: {str(e)}")
                cond_U_given_V = np.full(len(U), 0.5)

            try:
                cond_V_given_U = copula.partial_derivative(vu)
            except Exception as e:
                logger.warning(f"Partial derivative V|U failed: {str(e)}")
                cond_V_given_U = np.full(len(V), 0.5)

            # Clip probabilities to avoid edge cases
            cond_U_given_V = np.clip(cond_U_given_V, 0.01, 0.99)
            cond_V_given_U = np.clip(cond_V_given_U, 0.01, 0.99)

            # Generate signals for pair trading (original implementation)
            pair_signals = {
                'long_X': cond_U_given_V < (1 - quantile_threshold),
                'short_X': cond_U_given_V > quantile_threshold,
                'long_Y': cond_V_given_U < (1 - quantile_threshold),
                'short_Y': cond_V_given_U > quantile_threshold
            }

            # Generate individual asset signals
            # For asset X
            X_signals = np.zeros(len(U), dtype=np.int8)  # Default to neutral (0)
            X_signals[pair_signals['long_X']] = 1  # Long signal
            X_signals[pair_signals['short_X']] = -1  # Short signal

            # For asset Y
            Y_signals = np.zeros(len(V), dtype=np.int8)  # Default to neutral (0)
            Y_signals[pair_signals['long_Y']] = 1  # Long signal
            Y_signals[pair_signals['short_Y']] = -1  # Short signal

            # Calculate confidence scores based on how extreme the conditional probabilities are
            X_confidence = np.zeros(len(U))
            Y_confidence = np.zeros(len(V))

            # For long signals, confidence is higher when probability is lower
            X_long_mask = X_signals == 1
            if np.any(X_long_mask):
                X_confidence[X_long_mask] = 1 - (cond_U_given_V[X_long_mask] / (1 - quantile_threshold))

            # For short signals, confidence is higher when probability is higher
            X_short_mask = X_signals == -1
            if np.any(X_short_mask):
                X_confidence[X_short_mask] = (cond_U_given_V[X_short_mask] - quantile_threshold) / (1 - quantile_threshold)

            # Same for Y
            Y_long_mask = Y_signals == 1
            if np.any(Y_long_mask):
                Y_confidence[Y_long_mask] = 1 - (cond_V_given_U[Y_long_mask] / (1 - quantile_threshold))

            Y_short_mask = Y_signals == -1
            if np.any(Y_short_mask):
                Y_confidence[Y_short_mask] = (cond_V_given_U[Y_short_mask] - quantile_threshold) / (1 - quantile_threshold)

            # Clip confidence to [0, 1]
            X_confidence = np.clip(X_confidence, 0, 1)
            Y_confidence = np.clip(Y_confidence, 0, 1)

            return {
                # Original pair trading signals
                'pair_signals': pair_signals,

                # Individual asset signals
                'X_signals': X_signals,
                'X_confidence': X_confidence,
                'Y_signals': Y_signals,
                'Y_confidence': Y_confidence
            }
        except Exception as e:
            logger.error(f"Signal generation failed: {str(e)}")
            # Return neutral signals on failure
            neutral = np.zeros_like(U, dtype=bool)
            neutral_int = np.zeros_like(U, dtype=np.int8)
            neutral_float = np.zeros_like(U, dtype=float)

            return {
                'pair_signals': {
                    'long_X': neutral,
                    'short_X': neutral,
                    'long_Y': neutral,
                    'short_Y': neutral
                },
                'X_signals': neutral_int,
                'X_confidence': neutral_float,
                'Y_signals': neutral_int,
                'Y_confidence': neutral_float
            }


class PortfolioManager:
    """Handles position sizing and risk management"""

    def __init__(self, initial_balance: float):
        self.balance = initial_balance
        self.positions: Dict[str, Dict] = {}  # {pair: position_data}
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
        self._record_portfolio_value(timestamp)

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
        self._record_trade(
            timestamp=timestamp,
            pair=pair,
            action='exit',
            direction=position['direction'],
            size=position['size'],
            pnl=pnl,
            balance=self.portfolio.balance,
            net_return=net_return
        )

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
            'take_profit': config.take_profit_pct,
            'trailing_stop': config.trailing_stop_pct,
            'high_water_mark': 0.0,  # Track highest return for trailing stop
            'timestamp': timestamp
        }

        # Record trade
        self._record_trade(
            timestamp=timestamp,
            pair=pair,
            action='enter',
            direction=direction,
            size=position_size,
            balance=self.portfolio.balance
        )

    def _record_trade(self, **kwargs):
        """Record trade with validation"""
        try:
            # Validate trade data before recording
            trade_data = {
                'timestamp': kwargs['timestamp'],
                'pair': kwargs['pair'],
                'action': kwargs.get('action', ''),
                'direction': kwargs.get('direction', 0),
                'size': kwargs.get('size', 0),
                'pnl': kwargs.get('pnl', 0),
                'balance': kwargs.get('balance', 0),
                'return': kwargs.get('net_return', 0)
            }
            self.portfolio.trade_history.append(trade_data)
        except Exception as e:
            logger.error(f"Failed to record trade: {str(e)}")

    def _record_portfolio_value(self, timestamp):
        """Record portfolio value with validation"""
        try:
            self.portfolio.portfolio_values.append({
                'timestamp': timestamp,
                'value': self.portfolio.balance,
                'n_positions': len(self.portfolio.positions)
            })
        except Exception as e:
            logger.error(f"Failed to record portfolio value: {str(e)}")

    def check_position_exits(self, timestamp: pd.Timestamp) -> None:
        """Check all positions for exit conditions (stop-loss, take-profit, trailing stop)"""
        for pair, position in list(self.portfolio.positions.items()):
            if pair not in self.current_prices:
                continue

            price_btc, price_alt = self.current_prices[pair]

            # Calculate current return
            btc_return = (price_btc / position['entry_btc'] - 1) * position['direction']
            alt_return = (position['entry_alt'] / price_alt - 1) * position['direction']
            net_return = btc_return + alt_return

            # Update high water mark if we have a new high
            if net_return > position['high_water_mark']:
                position['high_water_mark'] = net_return

            # Check stop loss
            if net_return < -position['stop_loss']:
                logger.info(f"Stop loss triggered for {pair} at {timestamp}")
                self.execute_trade(pair, 0, price_btc, price_alt, timestamp)
                continue

            # Check take profit
            if net_return >= position['take_profit']:
                logger.info(f"Take profit triggered for {pair} at {timestamp}")
                self.execute_trade(pair, 0, price_btc, price_alt, timestamp)
                continue

            # Check trailing stop
            if position['high_water_mark'] > 0 and (position['high_water_mark'] - net_return) >= position['trailing_stop']:
                logger.info(f"Trailing stop triggered for {pair} at {timestamp}")
                self.execute_trade(pair, 0, price_btc, price_alt, timestamp)


class CopulaPairTradingStrategy:
    """Complete strategy implementation"""

    def __init__(self, user_data=None):
        """Initialize the strategy
        
        Args:
            user_data: Optional dictionary of user-provided data {symbol: DataFrame}
                       Each DataFrame must have columns: open, high, low, close, volume
                       and either a timestamp column or DatetimeIndex
        """
        self.data_fetcher = DataFetcher(user_data=user_data)
        self.stationarity_tester = StationarityTester()
        self.garch_standardizer = GARCHStandardizer()
        self.copula_model = CopulaModel()
        self.backtester = Backtester()
        self.signals_summary = []  # Store signal summaries for analysis

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

            # Check position exits (stop-loss, take-profit, trailing stop)
            self.backtester.check_position_exits(timestamp)

            # Generate trade signal if volatility condition met
            if vol_condition.iloc[i - 1] if i > 0 else False:
                signal_idx = i - 1  # Signals are offset by 1

                # Original pair trading logic
                pair_signals = signals['pair_signals']
                if pair_signals['long_Y'][signal_idx] and pair_signals['short_X'][signal_idx]:
                    # Long altcoin, short BTC
                    self.backtester.execute_trade(
                        pair_name, 1, price_btc, price_alt, timestamp
                    )
                elif pair_signals['short_Y'][signal_idx] and pair_signals['long_X'][signal_idx]:
                    # Short altcoin, long BTC
                    self.backtester.execute_trade(
                        pair_name, -1, price_btc, price_alt, timestamp
                    )
                elif pair_name in self.backtester.portfolio.positions:
                    # Close position if no signal
                    self.backtester.execute_trade(
                        pair_name, 0, price_btc, price_alt, timestamp
                    )

                # Individual asset signals
                # Create TradeSignal objects for each asset
                X_signal = signals['X_signals'][signal_idx]
                Y_signal = signals['Y_signals'][signal_idx]

                # Create and store signal summary
                self._add_signal_summary(
                    timestamp=timestamp,
                    benchmark=benchmark,
                    symbol=symbol,
                    X_signal=X_signal,
                    Y_signal=Y_signal,
                    X_confidence=signals['X_confidence'][signal_idx],
                    Y_confidence=signals['Y_confidence'][signal_idx],
                    pair_signal=1 if pair_signals['long_Y'][signal_idx] and pair_signals['short_X'][signal_idx] else
                              -1 if pair_signals['short_Y'][signal_idx] and pair_signals['long_X'][signal_idx] else 0
                )

                if X_signal != 0:  # If there's a non-neutral signal for X
                    X_trade_signal = TradeSignal(
                        pair=benchmark,
                        direction=X_signal,  # -1 for short, 1 for long
                        confidence=signals['X_confidence'][signal_idx],
                        timestamp=timestamp
                    )
                    logger.info(f"Generated {X_signal > 0 and 'LONG' or 'SHORT'} signal for {benchmark} with confidence {X_trade_signal.confidence:.2f}")

                if Y_signal != 0:  # If there's a non-neutral signal for Y
                    Y_trade_signal = TradeSignal(
                        pair=symbol,
                        direction=Y_signal,  # -1 for short, 1 for long
                        confidence=signals['Y_confidence'][signal_idx],
                        timestamp=timestamp
                    )
                    logger.info(f"Generated {Y_signal > 0 and 'LONG' or 'SHORT'} signal for {symbol} with confidence {Y_trade_signal.confidence:.2f}")

            # Record portfolio value
            self.backtester._record_portfolio_value(timestamp)

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

        report_data = {
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

        # Validate report with Pydantic
        try:
            return PerformanceReport(**report_data).model_dump()
        except Exception as e:
            logger.error(f"Report validation failed: {str(e)}")
            return report_data

    def _calculate_max_drawdown(self, portfolio: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        peak = portfolio['value'].cummax()
        drawdown = (portfolio['value'] - peak) / peak
        return drawdown.min()

    def _calculate_sharpe(self, portfolio: pd.DataFrame) -> float:
        """Calculate annualized Sharpe ratio with simple, robust methodology"""
        # Check if we have enough data
        if len(portfolio) < 5:
            return 0.0

        # Use simple returns for stability
        returns = portfolio['value'].pct_change().dropna()

        # Handle case where all returns are identical or near-zero volatility
        if returns.std() < 1e-8:
            return 0.0

        # Use a simple annualization factor for hourly data
        annualization_factor = np.sqrt(24 * 365)

        # Calculate basic Sharpe ratio
        sharpe = (returns.mean() / returns.std()) * annualization_factor

        # Ensure we return a finite value
        if np.isfinite(sharpe):
            return sharpe
        else:
            return 0.0

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

    def _add_signal_summary(self, timestamp, benchmark, symbol, X_signal, Y_signal, 
                           X_confidence, Y_confidence, pair_signal):
        """Add a summary of signals for analysis"""
        summary = {
            'timestamp': timestamp,
            'benchmark': benchmark,
            'symbol': symbol,
            'benchmark_signal': X_signal,  # -1 (short), 0 (neutral), 1 (long)
            'symbol_signal': Y_signal,     # -1 (short), 0 (neutral), 1 (long)
            'benchmark_confidence': X_confidence,
            'symbol_confidence': Y_confidence,
            'pair_signal': pair_signal     # -1 (short Y, long X), 0 (neutral), 1 (long Y, short X)
        }
        self.signals_summary.append(summary)

    def get_signals_summary(self) -> pd.DataFrame:
        """Get a DataFrame of all signal summaries"""
        if not self.signals_summary:
            return pd.DataFrame()

        df = pd.DataFrame(self.signals_summary)

        # Add human-readable signal descriptions
        def signal_desc(row):
            if row['benchmark_signal'] == 1:
                return f"LONG {row['benchmark']} ({row['benchmark_confidence']:.2f})"
            elif row['benchmark_signal'] == -1:
                return f"SHORT {row['benchmark']} ({row['benchmark_confidence']:.2f})"
            return "NEUTRAL"

        def symbol_desc(row):
            if row['symbol_signal'] == 1:
                return f"LONG {row['symbol']} ({row['symbol_confidence']:.2f})"
            elif row['symbol_signal'] == -1:
                return f"SHORT {row['symbol']} ({row['symbol_confidence']:.2f})"
            return "NEUTRAL"

        def pair_desc(row):
            if row['pair_signal'] == 1:
                return f"LONG {row['symbol']} / SHORT {row['benchmark']}"
            elif row['pair_signal'] == -1:
                return f"SHORT {row['symbol']} / LONG {row['benchmark']}"
            return "NEUTRAL"

        df['benchmark_desc'] = df.apply(signal_desc, axis=1)
        df['symbol_desc'] = df.apply(symbol_desc, axis=1)
        df['pair_desc'] = df.apply(pair_desc, axis=1)

        return df


if __name__ == "__main__":
    # Initialize and run strategy
    strategy = CopulaPairTradingStrategy()

    # Define assets
    symbols = ['BTC/USDC:USDC', 'ETH/USDC:USDC', 'BNB/USDC:USDC', 'SOL/USDC:USDC', 'ADA/USDC:USDC']
    benchmark = 'BTC/USDC:USDC'

    # Run on normal data
    logger.info("Running strategy on normal data")
    normal_results = strategy.run(symbols, benchmark)

    # Get signal summaries
    signals_df = strategy.get_signals_summary()

    # Display signal counts
    if not signals_df.empty:
        print("\n=== Signal Summary ===")
        # Count by signal type
        benchmark_signals = signals_df['benchmark_signal'].value_counts()
        symbol_signals = signals_df['symbol_signal'].value_counts()
        pair_signals = signals_df['pair_signal'].value_counts()

        print("\nBenchmark Signals:")
        print(f"  Long: {benchmark_signals.get(1, 0)}")
        print(f"  Short: {benchmark_signals.get(-1, 0)}")
        print(f"  Neutral: {benchmark_signals.get(0, 0)}")

        print("\nAltcoin Signals:")
        print(f"  Long: {symbol_signals.get(1, 0)}")
        print(f"  Short: {symbol_signals.get(-1, 0)}")
        print(f"  Neutral: {symbol_signals.get(0, 0)}")

        print("\nPair Trading Signals:")
        print(f"  Long Altcoin/Short Benchmark: {pair_signals.get(1, 0)}")
        print(f"  Short Altcoin/Long Benchmark: {pair_signals.get(-1, 0)}")
        print(f"  Neutral: {pair_signals.get(0, 0)}")

        # Display the last 5 signals
        if len(signals_df) > 0:
            print("\nLast 5 Signals:")
            last_signals = signals_df.tail(5)
            for _, row in last_signals.iterrows():
                print(f"{row['timestamp']} - Benchmark: {row['benchmark_desc']}, Symbol: {row['symbol_desc']}, Pair: {row['pair_desc']}")

    # Run with perturbations
    logger.info("Running strategy with market shocks")
    perturbed_results = strategy.run(symbols, benchmark, perturb=True)

    # Print comparison
    print("\n=== Performance Comparison ===")
    print(f"{'Metric':<25} {'Normal':>10} {'Perturbed':>10}")
    for metric in ['total_return_pct', 'win_rate', 'max_drawdown_pct', 'sharpe_ratio']:
        print(f"{metric:<25} {normal_results.get(metric, 0):>10.2f} {perturbed_results.get(metric, 0):>10.2f}")
