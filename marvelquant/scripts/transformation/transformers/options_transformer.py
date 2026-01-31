"""
Options Bar Transformer with Greeks Calculation.

Transforms options data to Nautilus Bar format + calculates Greeks (IV, Delta, Gamma, Theta, Vega, Rho).
Implements Phase 2 fixes: analytic Greeks fallback, relaxed price validation, IV exception handling.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Optional, Tuple, Dict, Set, List
import gc

import pandas as pd
import numpy as np
from nautilus_trader.model.data import BarType
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.persistence.wranglers import BarDataWrangler

# Import contract generators and utilities - handle both absolute and direct imports
try:
    from marvelquant.utils.contract_generators import (
        create_options_contract,
        parse_nse_option_symbol
    )
    from marvelquant.data.types import OptionOI
    from marvelquant.data.custom.option_greeks import OptionGreeks
    from marvelquant.utils.greeks import OptionPricing
    from marvelquant.utils.rates import InterestRateProvider
except ImportError:
    # Fallback to direct imports
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    utils_path = project_root / "utils"
    data_path = project_root / "data" / "types"
    custom_path = project_root / "data" / "custom"
    if str(utils_path) not in sys.path:
        sys.path.insert(0, str(utils_path))
    if str(data_path) not in sys.path:
        sys.path.insert(0, str(data_path))
    if str(custom_path) not in sys.path:
        sys.path.insert(0, str(custom_path))
    from contract_generators import (
        create_options_contract,
        parse_nse_option_symbol
    )
    from option_oi import OptionOI
    from option_greeks import OptionGreeks
    from greeks import OptionPricing
    from rates import InterestRateProvider

# Import common utilities - handle both relative and absolute imports
try:
    from .common import (
        get_venue,
        yyyymmdd_seconds_to_datetime,
        detect_price_unit,
        convert_paise_to_rupees,
        time_aware_forward_fill,
        IST_OFFSET,
        PROJECT_ROOT,
    )
except ImportError:
    # Fallback to absolute import when run directly
    from common import (
        get_venue,
        yyyymmdd_seconds_to_datetime,
        detect_price_unit,
        convert_paise_to_rupees,
        time_aware_forward_fill,
        IST_OFFSET,
        PROJECT_ROOT,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# TTE validation thresholds
MIN_TTE_DAYS = 1 / (365 * 24 * 12)  # ~5 minutes
MIN_TTE_WARNING = 1 / 365  # 1 day

# Risk-free rate validation (Indian market reasonable range)
R_MIN = 0.03  # 3% minimum
R_MAX = 0.10  # 10% maximum
R_DEFAULT = 0.06  # 6% default fallback

# Price validation tolerance for upper bounds
PRICE_UPPER_BOUND_TOLERANCE = 0.05  # 5% tolerance above theoretical max

# Trading session hours (IST) - NSE/BSE: 9:15-15:30, MCX: varies
# For session-aware forward-fill
MARKET_START_HOUR = 9
MARKET_START_MINUTE = 15
MARKET_END_HOUR = 15
MARKET_END_MINUTE = 30

# 2-hour lookback window for spot fallback (within same session)
SPOT_LOOKBACK_HOURS = 2

# Feature flags for rollback safety
# These flags allow disabling advanced features while keeping critical fixes enabled.
# To rollback: Set ENABLE_UPPER_BOUND_CHECKS = False (keeps analytic fallbacks, unit-safe conversion)
ENABLE_UPPER_BOUND_CHECKS = False  # Disabled for stock options - prices can be more volatile than indices
ENABLE_FUTURES_AS_SPOT_PROXY = False  # Future enhancement: use futures as spot proxy when index spot unavailable

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _log_greeks_debug(enabled: bool, message: str):
    """Utility to guard verbose Greeks diagnostics behind a CLI flag."""
    if enabled:
        logger.debug(f"[GREEKS DEBUG] {message}")


def _calculate_analytic_greeks(
    spot: float,
    strike: float,
    r: float,
    tte: float,
    option_type: str,
    intrinsic_value: float
) -> Tuple[float, float, float, float, float, float]:
    """
    Calculate analytic Greeks for σ→0 limit (deep OTM options or IV solver failures).
    
    This implements the zero-volatility limit behavior:
    - IV = 0 (by definition)
    - Delta: 1.0 (call) or -1.0 (put) if ITM, else 0.0
    - Gamma = 0 (no curvature at zero vol)
    - Theta = 0 (no time decay at zero vol)
    - Vega = 0 (no volatility sensitivity at zero vol)
    - Rho: based on discounted strike and intrinsic value
    
    Args:
        spot: Spot price
        strike: Strike price
        r: Risk-free rate
        tte: Time to expiry in years
        option_type: 'CE' (Call) or 'PE' (Put)
        intrinsic_value: Pre-calculated intrinsic value
        
    Returns:
        Tuple of (iv, delta, gamma, theta, vega, rho)
    """
    iv = 0.0
    gamma = 0.0
    theta = 0.0
    vega = 0.0
    
    discounted_strike = strike * np.exp(-r * tte)
    
    if option_type == 'CE':
        delta = 1.0 if spot > discounted_strike else 0.0
        if intrinsic_value > 0:
            rho = strike * np.exp(-r * tte) * tte * 1.0 / 100.0
        else:
            rho = strike * np.exp(-r * tte) * tte * 0.0 / 100.0
    else:  # PE
        delta = -1.0 if spot < discounted_strike else 0.0
        if intrinsic_value > 0:
            rho = -strike * np.exp(-r * tte) * tte * 1.0 / 100.0
        else:
            rho = -strike * np.exp(-r * tte) * tte * 0.0 / 100.0
    
    return (iv, delta, gamma, theta, vega, rho)


def _price_consistent_with_moneyness(
    price: float,
    spot: float,
    strike: float,
    option_type: str,
    tte: float,
    tolerance: float = PRICE_UPPER_BOUND_TOLERANCE,
    enable_upper_bounds: Optional[bool] = None
) -> Tuple[bool, str]:
    """
    Validate option price using simplified bounds with upper-bound guards.
    
    Note: Intrinsic value check removed - European options (especially deep ITM puts
    with positive interest rates) can legitimately trade below intrinsic value due to
    time value of money and bid-ask spreads. Market prices are valid even if below
    theoretical intrinsic value.
    
    Validation enforces:
    - Hard constraint: price >= 0 (non-negative)
    - Lightweight upper-bound guards:
      - Calls: price <= spot * (1 + tolerance)
      - Puts: price <= strike * (1 + tolerance)
    
    Args:
        price: Option price to validate
        spot: Spot price
        strike: Strike price
        option_type: 'CE' (Call) or 'PE' (Put)
        tte: Time to expiry in years
        tolerance: Upper bound tolerance (default: 5%)
        
    Returns:
        (True, "") if valid, (False, reason) if invalid
    """
    # Hard constraint: non-negative price
    if price < 0:
        return False, "price_below_zero"
    
    # Lightweight upper-bound guards (can be disabled via feature flag for rollback)
    # Use parameter if provided, otherwise use module-level flag
    use_upper_bounds = enable_upper_bounds if enable_upper_bounds is not None else ENABLE_UPPER_BOUND_CHECKS
    if use_upper_bounds:
        if option_type == 'CE':
            max_price = spot * (1 + tolerance)
            if price > max_price:
                return False, "price_above_upper_bound"
        else:  # PE
            max_price = strike * (1 + tolerance)
            if price > max_price:
                return False, "price_above_upper_bound"
    
    return True, ""


def _session_aware_forward_fill(
    df: pd.DataFrame,
    column: str,
    max_gap_minutes: int = 90,
    use_interpolation: bool = True,
    venue: str = "NSE"
) -> pd.DataFrame:
    """
    Session-aware forward-fill that respects trading session boundaries.
    
    Does not bleed prices across session boundaries (e.g., from previous day's close
    to next day's open). Uses time_aware_forward_fill within each session slice.
    
    Args:
        df: DataFrame with datetime index and column to fill
        column: Name of column to forward-fill
        max_gap_minutes: Maximum time gap (in minutes) to forward-fill within session
        use_interpolation: Whether to use interpolation for larger gaps
        venue: Venue string for session hours (NSE/BSE/MCX)
        
    Returns:
        DataFrame with filled column (no cross-session contamination)
    """
    if df.empty or column not in df.columns:
        return df
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"DataFrame index must be DatetimeIndex, got {type(df.index)}")
    
    result_df = df.copy()
    
    # Group by date (trading day)
    # Handle DatetimeIndex - reset_index will create a column from the index
    df_with_date = result_df.reset_index()
    
    # Find the timestamp column (could be index name or default 'index')
    timestamp_col = None
    if 'timestamp' in df_with_date.columns:
        timestamp_col = 'timestamp'
    elif result_df.index.name == 'timestamp' and result_df.index.name in df_with_date.columns:
        timestamp_col = result_df.index.name
    else:
        # Find the first datetime column (should be the index column after reset)
        for col in df_with_date.columns:
            if pd.api.types.is_datetime64_any_dtype(df_with_date[col]):
                timestamp_col = col
                break
        
        # Fallback to first column if no datetime column found
        if timestamp_col is None and len(df_with_date.columns) > 0:
            timestamp_col = df_with_date.columns[0]
    
    if timestamp_col is None or timestamp_col not in df_with_date.columns:
        raise ValueError(f"Could not find timestamp column in DataFrame. Columns: {df_with_date.columns.tolist()}, Index name: {result_df.index.name}")
    
    df_with_date['date'] = pd.to_datetime(df_with_date[timestamp_col]).dt.date
    
    # Apply time-aware forward-fill within each session
    filled_groups = []
    for date_val, group in df_with_date.groupby('date'):
        group_df = group.set_index(timestamp_col)
        filled_group = time_aware_forward_fill(
            group_df,
            column,
            max_gap_minutes=max_gap_minutes,
            use_interpolation=use_interpolation
        )
        filled_groups.append(filled_group)
    
    result_df = pd.concat(filled_groups).sort_index()
    
    return result_df


def _get_spot_with_lookback(
    spot_df: pd.DataFrame,
    timestamp: pd.Timestamp,
    lookback_hours: float = SPOT_LOOKBACK_HOURS
) -> Optional[float]:
    """
    Get spot price with 2-hour lookback fallback within the same session.
    
    If spot is missing or invalid at the given timestamp, looks back up to
    lookback_hours within the same trading session to find a valid spot price.
    
    Args:
        spot_df: DataFrame with DatetimeIndex and 'spot_price' column
        timestamp: Timestamp to get spot for (must be timezone-aware)
        lookback_hours: Maximum hours to look back (default 2)
        
    Returns:
        Spot price if found, None otherwise
    """
    if spot_df.empty or 'spot_price' not in spot_df.columns:
        return None
    
    # Ensure timestamp is timezone-aware (UTC)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize('UTC')
    else:
        timestamp = timestamp.tz_convert('UTC')
    
    # Ensure spot_df index is timezone-aware (UTC) for proper comparison
    if spot_df.index.tzinfo is None:
        spot_df = spot_df.copy()
        spot_df.index = spot_df.index.tz_localize('UTC')
    else:
        spot_df = spot_df.copy()
        spot_df.index = spot_df.index.tz_convert('UTC')
    
    # Get spot at exact timestamp first
    if timestamp in spot_df.index:
        spot = spot_df.loc[timestamp, 'spot_price']
        if pd.notna(spot) and spot > 0 and 1000 <= spot <= 100000:
            return float(spot)
    
    # Look back within the same session (same date)
    session_start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    lookback_start = timestamp - pd.Timedelta(hours=lookback_hours)
    
    # Ensure lookback doesn't go before session start
    lookback_start = max(lookback_start, session_start)
    
    # Filter spot_df to same session and lookback window
    session_mask = (spot_df.index >= lookback_start) & (spot_df.index <= timestamp)
    session_spot = spot_df[session_mask]
    
    if session_spot.empty:
        return None
    
    # Find most recent valid spot price
    valid_spots = session_spot[
        (session_spot['spot_price'].notna()) &
        (session_spot['spot_price'] > 0) &
        (session_spot['spot_price'] >= 1000) &
        (session_spot['spot_price'] <= 100000)
    ]
    
    if valid_spots.empty:
        return None
    
    # Return the most recent valid spot
    return float(valid_spots['spot_price'].iloc[-1])


def _update_date_metrics(
    date_metrics: Dict[date, Dict],
    record_date: date,
    metric_type: str,
    value: Optional[float] = None
):
    """
    Update per-date metrics dictionary for quality report generation.
    
    Expanded skip reasons per Phase 2 requirements:
    - missing_spot_data
    - spot_price_unreasonable
    - tte_too_small
    - tte_negative
    - price_above_upper_bound
    - failed_iv_* (various IV solver failures)
    - parse_error
    
    Note: price_below_intrinsic removed - European options can legitimately trade below intrinsic value.
    """
    if record_date not in date_metrics:
        date_metrics[record_date] = {
            'records_processed': 0,
            'records_skipped': 0,
            'skip_reasons': {
                'missing_spot_data': 0,
                'spot_price_unreasonable': 0,
                'tte_too_small': 0,
                'tte_negative': 0,
                'price_above_upper_bound': 0,
                'failed_iv_solve': 0,
                'failed_iv_valueerror': 0,
                'failed_iv_keyerror': 0,
                'failed_iv_generic': 0,
                'iv_at_upper_bound': 0,
                'parse_error': 0,
            },
            'iv_values': [],
            'theta_values': [],
            'records_at_bounds': 0,
            'records_zero_iv': 0,
            'records_analytic_fallback': 0,
        }
    
    if metric_type == 'processed':
        date_metrics[record_date]['records_processed'] += 1
        if value is not None:  # value is IV
            date_metrics[record_date]['iv_values'].append(value)
    elif metric_type == 'theta':
        if value is not None:
            date_metrics[record_date]['theta_values'].append(value)
    elif metric_type == 'skipped':
        date_metrics[record_date]['records_skipped'] += 1
        if value:  # value is skip reason string
            date_metrics[record_date]['skip_reasons'][value] = (
                date_metrics[record_date]['skip_reasons'].get(value, 0) + 1
            )
    elif metric_type == 'at_bounds':
        date_metrics[record_date]['records_at_bounds'] += 1
    elif metric_type == 'zero_iv':
        date_metrics[record_date]['records_zero_iv'] += 1
    elif metric_type == 'analytic_fallback':
        date_metrics[record_date]['records_analytic_fallback'] += 1


def _load_spot_index_series(
    input_dir: Path,
    symbol: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Load and prepare spot/index series for Greeks calculation.
    
    Applies median-based price unit detection, deduplication (keep='last'),
    and session-aware forward-fill.
    
    For stock options, loads equity data from nested structure: cash/YYYY/MM/{symbol}/

    Returns:
        DataFrame with 'timestamp' index and 'spot_price' column
    """
    # Try both uppercase and lowercase symbol directories
    symbol_lower = symbol.lower()
    index_files = []
    
    # First try index directory (for indices like NIFTY, BANKNIFTY)
    index_dir = input_dir / "index" / symbol_lower
    if index_dir.exists():
        index_files = list(index_dir.rglob("*.parquet"))
    
    # If not found, try cash directory (flat structure)
    if not index_files:
        cash_dir = input_dir / "cash" / symbol_lower
        if cash_dir.exists():
            index_files = list(cash_dir.rglob("*.parquet"))
    
    # If still not found, try nested structure: cash/YYYY/MM/{symbol}/ (for stock equity data)
    if not index_files:
        cash_base = input_dir / "cash"
        if cash_base.exists():
            # Look for nested structure: cash/YYYY/MM/{symbol}/
            for year_dir in cash_base.iterdir():
                if year_dir.is_dir() and year_dir.name.isdigit() and len(year_dir.name) == 4:
                    for month_dir in year_dir.iterdir():
                        if month_dir.is_dir() and month_dir.name.isdigit() and len(month_dir.name) <= 2:
                            symbol_dir = month_dir / symbol_lower
                            if symbol_dir.exists():
                                index_files.extend(list(symbol_dir.glob("*.parquet")))
    
    spot_df = pd.DataFrame()
    
    if not index_files:
        logger.warning(f"No index/cash files found for {symbol} spot data (checked index/{symbol_lower}, cash/{symbol_lower}, and cash/YYYY/MM/{symbol_lower}/)")
        return spot_df
    
    dfs = []
    s_start = pd.to_datetime(start_date) - pd.Timedelta(hours=6)
    s_end = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    
    for f in index_files:
        try:
            # Read parquet file - handle both old format (date, time, close) and new format
            try:
                temp_df = pd.read_parquet(f, columns=['date', 'time', 'close'])
            except (KeyError, ValueError):
                # Try reading all columns and extract what we need
                temp_df = pd.read_parquet(f)
                if 'date' not in temp_df.columns or 'time' not in temp_df.columns or 'close' not in temp_df.columns:
                    logger.debug(f"Skipping {f.name}: missing required columns (date, time, close)")
                    continue
                temp_df = temp_df[['date', 'time', 'close']]
            
            # Apply median-based price unit detection
            # For stocks, expected range is lower (e.g., 100-10000), for indices it's higher (1000-100000)
            # But we need to be careful - if prices are already in rupees, don't convert again
            # Check median first to avoid double conversion
            median_close = temp_df['close'].median()
            if median_close > 10000:
                # Definitely in paise - convert to rupees
                logger.debug(f"Detected paise prices in {f.name} (median={median_close:.2f}), converting to rupees")
                temp_df = convert_paise_to_rupees(temp_df, ['close'])
            elif median_close > 1000 and median_close < 10000:
                # Could be in paise for stocks - check if min is also high
                if temp_df['close'].min() > 5000:
                    logger.debug(f"Detected paise prices in {f.name} (range: {temp_df['close'].min():.2f}-{temp_df['close'].max():.2f}), converting to rupees")
                    temp_df = convert_paise_to_rupees(temp_df, ['close'])
                else:
                    # Already in rupees
                    temp_df['close'] = temp_df['close'].astype('float64')
            else:
                # Already in rupees (median < 1000)
                temp_df['close'] = temp_df['close'].astype('float64')
            
            dfs.append(temp_df)
        except Exception as e:
            logger.debug(f"Error reading spot file {f}: {e}")
            continue
    
    if not dfs:
        return spot_df
    
    spot_raw = pd.concat(dfs, ignore_index=True)
    spot_raw['timestamp'] = spot_raw.apply(
        lambda row: yyyymmdd_seconds_to_datetime(row['date'], row['time']),
        axis=1
    )
    
    # Filter spot data to relevant range
    spot_raw = spot_raw[(spot_raw['timestamp'] >= s_start) & (spot_raw['timestamp'] < s_end)]
    
    if spot_raw.empty:
        return spot_df
    
    # Deduplicate by timestamp (keep='last' per Phase 2 requirements)
    spot_df = spot_raw.groupby('timestamp')['close'].last().reset_index()
    spot_df.columns = ['timestamp', 'spot_price']
    spot_df = spot_df.set_index('timestamp').sort_index()
    
    # Validate spot price range - DO NOT convert again if already converted in file reading step
    # The conversion should have happened in the file reading loop above
    median_spot = spot_df['spot_price'].median()
    
    # Only convert if prices are clearly in paise (median > 10000)
    # If median is between 1000-10000, check if it's reasonable for the symbol type
    # For stocks, prices can be 10-10000 rupees, so don't auto-convert
    if median_spot > 10000:
        # Definitely in paise - convert to rupees (only if not already converted)
        logger.warning(f"Detected spot prices appear to be in paise (median={median_spot:.2f}). Converting to rupees.")
        spot_df['spot_price'] = spot_df['spot_price'] / 100.0
        median_spot = spot_df['spot_price'].median()
        logger.info(f"After conversion: median spot price = {median_spot:.2f} rupees")
    elif median_spot < 10:
        # Too low - might have been double-converted or data issue
        logger.error(f"Spot prices seem unusually low (median={median_spot:.2f}). This might indicate a double conversion issue. Check data source.")
    elif median_spot >= 10 and median_spot < 1000:
        # Reasonable stock price range (10-1000 rupees) - already in rupees
        logger.debug(f"Spot prices appear to be in rupees (median={median_spot:.2f})")
    elif median_spot >= 1000 and median_spot < 10000:
        # Could be in paise for stocks OR already in rupees for higher-priced stocks
        # Only convert if prices are consistently very high (suggesting paise)
        if spot_df['spot_price'].min() > 50000:
            # Very high prices - likely in paise
            logger.warning(f"Detected spot prices appear to be in paise (range: {spot_df['spot_price'].min():.2f}-{spot_df['spot_price'].max():.2f}). Converting to rupees.")
            spot_df['spot_price'] = spot_df['spot_price'] / 100.0
            median_spot = spot_df['spot_price'].median()
            logger.info(f"After conversion: median spot price = {median_spot:.2f} rupees")
        else:
            # Prices in reasonable range - already in rupees
            logger.debug(f"Spot prices appear to be in rupees (median={median_spot:.2f}, range: {spot_df['spot_price'].min():.2f}-{spot_df['spot_price'].max():.2f})")
    
    # Session-aware forward-fill with 2-hour lookback fallback
    venue_str = get_venue(symbol)
    spot_df = _session_aware_forward_fill(
        spot_df,
        'spot_price',
        max_gap_minutes=90,  # Increased from 30 per Phase 2
        use_interpolation=True,
        venue=venue_str
    )
    
    logger.info(
        f"Loaded {len(spot_df)} spot price records for Greeks calculation "
        f"(deduplicated, session-aware forward-filled)"
    )
    logger.info(
        f"Spot price range: {spot_df['spot_price'].min():.2f} to {spot_df['spot_price'].max():.2f} "
        f"(median: {spot_df['spot_price'].median():.2f})"
    )
    
    return spot_df


def _validate_risk_free_rate(r: float, query_date: date) -> float:
    """
    Validate and clamp risk-free rate to reasonable Indian market band.
    
    Args:
        r: Risk-free rate (decimal, e.g., 0.0691 for 6.91%)
        query_date: Date for logging context
        
    Returns:
        Validated rate (clamped to [R_MIN, R_MAX] or R_DEFAULT if invalid)
    """
    if r < R_MIN or r > R_MAX:
        logger.warning(
            f"Risk-free rate {r:.4f} ({r*100:.2f}%) out of reasonable range "
            f"[{R_MIN*100:.0f}%, {R_MAX*100:.0f}%] for {query_date}. "
            f"Using default {R_DEFAULT*100:.0f}%"
        )
        return R_DEFAULT
    
    return r


# ---------------------------------------------------------------------------
# Main Transformer Function
# ---------------------------------------------------------------------------

def transform_options_bars(
    input_dir: Path,
    catalog: ParquetDataCatalog,
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: Optional[Path] = None,
    expiry_filter: Optional[Set[date]] = None,
    strike_min: Optional[float] = None,
    strike_max: Optional[float] = None,
    log_greeks_debug: bool = False,
) -> Tuple[int, Dict, Dict]:
    """
    Transform options data to Nautilus Bar format + calculate Greeks (OFFICIAL PATTERN).
    
    Implements Phase 2 fixes:
    - Analytic Greeks fallback for IV solver failures
    - Relaxed price validation with upper-bound guards
    - Session-aware forward-fill for spot data
    - Enhanced metrics tracking
    
    Args:
        input_dir: Directory containing raw parquet files
        catalog: Nautilus ParquetDataCatalog instance
        symbol: Symbol name (e.g., "NIFTY", "BANKNIFTY")
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Optional output directory for reports
        expiry_filter: Optional set of expiry dates to filter
        strike_min: Optional minimum strike filter
        strike_max: Optional maximum strike filter
        log_greeks_debug: Enable verbose Greeks diagnostics
        
    Returns:
        Tuple of (bars_count, metrics_tracker, date_metrics):
        - bars_count: Number of bars created
        - metrics_tracker: Dictionary with file/contract/Greeks metrics
        - date_metrics: Dictionary with per-date skip reasons and statistics
    """
    logger.info(f"Transforming {symbol} options bars with Greeks (Phase 2 fixes)...")
    
    # Initialize Interest Rate Provider
    ir_xml_path = PROJECT_ROOT / "data/static/interest_rates/india_91day_tbill_rates_2018_2025_nautilus.xml"
    if not ir_xml_path.exists():
        raise FileNotFoundError(
            f"Interest rate file not found: {ir_xml_path}\n"
            f"Full path: {ir_xml_path.resolve()}"
        )
    logger.info(f"Loading interest rates from: {ir_xml_path}")
    ir_provider = InterestRateProvider(ir_xml_path)
    
    # Load spot data
    spot_df = _load_spot_index_series(input_dir, symbol, start_date, end_date)
    
    if spot_df.empty:
        logger.warning(f"No spot data available for {symbol}, Greeks calculation will be skipped")
    
    # Initialize metrics tracker
    metrics_tracker = {
        'files_attempted': 0,
        'files_succeeded': 0,
        'files_failed': 0,
        'contracts_attempted': 0,
        'contracts_succeeded': 0,
        'contracts_parse_failed': 0,
        'greeks_write_failures': 0,
        'records_analytic_fallback': 0,
        'spot_fallback_forward_fill': 0,
        'spot_fallback_lookback': 0,
    }
    
    # Track skip details by strike/contract for debugging
    skip_details = {}  # {(strike, opt_type): {'count': int, 'price_range': (min, max), 'intrinsic_range': (min, max)}}
    
    # Per-date metrics
    date_metrics = {}

    # Process options files
    symbol_lower = symbol.lower()  # Keep for file matching
    
    # Setup expiry/strike filters
    expiry_filter_set: Optional[Set[date]] = None
    allowed_expiry_strings: List[str] = []
    if expiry_filter:
        expiry_filter_set = set(expiry_filter)
        allowed_expiry_strings = sorted(d.isoformat() for d in expiry_filter_set)
        logger.info(f"Expiry filter enabled for {len(expiry_filter_set)} date(s): {allowed_expiry_strings}")
    
    if strike_min is not None or strike_max is not None:
        strike_parts = []
        if strike_min is not None:
            strike_parts.append(f"min={strike_min:.2f}")
        if strike_max is not None:
            strike_parts.append(f"max={strike_max:.2f}")
        logger.info("Strike filter enabled (%s)", ", ".join(strike_parts))
    
    filtered_by_expiry = 0
    filtered_by_strike = 0
    
    # Validate and normalize date range
    start_dt = pd.to_datetime(start_date).normalize()
    end_dt = pd.to_datetime(end_date).normalize()
    
    if start_dt > end_dt:
        swapped_range_days = (start_dt - end_dt).days
        if swapped_range_days > 30:
            logger.error(f"Start date ({start_date}) is after end date ({end_date}). Swapping would create a {swapped_range_days}-day range, which is likely a mistake.")
            return 0, {}, {}
        logger.warning(f"Start date ({start_date}) is after end date ({end_date}). Swapping dates.")
        start_dt, end_dt = end_dt, start_dt
    
    # Construct file paths - support both old structure (option/NIFTY/call/year=2025/month=11/) 
    # and new structure (options/nifty_call/2025/11/)
    parquet_files = []
    dates_with_files = set()  # Track which dates had files found
    date_range = pd.date_range(start_dt, end_dt, freq='D')
    
    for current_date in date_range:
        year = current_date.strftime('%Y')
        month = current_date.strftime('%m')
        date_str = current_date.strftime('%Y%m%d')
        
        # Primary structure: option/{symbol}_call/{year}/{month}/
        call_dir = input_dir / "option" / f"{symbol_lower}_call" / year / month
        if call_dir.exists():
            # Try various filename patterns
            call_file = call_dir / f"{symbol_lower}_call_{date_str}.parquet"
            if not call_file.exists():
                call_file = call_dir / f"{symbol_lower}_{date_str}.parquet"
            if call_file.exists():
                parquet_files.append(call_file)
                dates_with_files.add(current_date.date())
            else:
                # Try to find any parquet file in this directory for this date
                for file in call_dir.glob(f"*{date_str}*.parquet"):
                    parquet_files.append(file)
                    dates_with_files.add(current_date.date())
                    break
        
        # Primary structure: option/{symbol}_put/{year}/{month}/
        put_dir = input_dir / "option" / f"{symbol_lower}_put" / year / month
        if put_dir.exists():
            # Try various filename patterns
            put_file = put_dir / f"{symbol_lower}_put_{date_str}.parquet"
            if not put_file.exists():
                put_file = put_dir / f"{symbol_lower}_{date_str}.parquet"
            if put_file.exists():
                parquet_files.append(put_file)
                dates_with_files.add(current_date.date())
            else:
                # Try to find any parquet file in this directory for this date
                for file in put_dir.glob(f"*{date_str}*.parquet"):
                    parquet_files.append(file)
                    dates_with_files.add(current_date.date())
                    break
        
        # Fallback: Try nested structure: option/call/YYYY/MM/{symbol}/{symbol}_YYYYMMDD.parquet
        if not parquet_files:
            call_file_nested = input_dir / "option" / "call" / year / month / symbol_lower / f"{symbol_lower}_{date_str}.parquet"
            if call_file_nested.exists():
                parquet_files.append(call_file_nested)
                dates_with_files.add(current_date.date())
            
            put_file_nested = input_dir / "option" / "put" / year / month / symbol_lower / f"{symbol_lower}_{date_str}.parquet"
            if put_file_nested.exists():
                parquet_files.append(put_file_nested)
                dates_with_files.add(current_date.date())
        
        # Fallback: Try new structure: options/nifty_call/2025/11/nifty_call_20251117.parquet
        if not parquet_files:
            call_file_new = input_dir / "options" / f"{symbol_lower}_call" / year / month / f"{symbol_lower}_call_{date_str}.parquet"
            if call_file_new.exists():
                parquet_files.append(call_file_new)
                dates_with_files.add(current_date.date())
            
            put_file_new = input_dir / "options" / f"{symbol_lower}_put" / year / month / f"{symbol_lower}_put_{date_str}.parquet"
            if put_file_new.exists():
                parquet_files.append(put_file_new)
                dates_with_files.add(current_date.date())
        
        # Fallback: Try old structure: option/NIFTY/call/year=2025/month=11/nifty_call_20251117.parquet
        if not parquet_files:
            call_file_old = input_dir / "option" / symbol / "call" / f"year={year}" / f"month={month}" / f"{symbol_lower}_call_{date_str}.parquet"
            if call_file_old.exists():
                parquet_files.append(call_file_old)
                dates_with_files.add(current_date.date())
            
            put_file_old = input_dir / "option" / symbol / "put" / f"year={year}" / f"month={month}" / f"{symbol_lower}_put_{date_str}.parquet"
            if put_file_old.exists():
                parquet_files.append(put_file_old)
                dates_with_files.add(current_date.date())
    
    if not parquet_files:
        logger.warning(f"No option files found for date range {start_dt.date()} to {end_dt.date()}")
        return 0, {}, {}
    
    logger.info(f"Using {len(parquet_files)} dated option files for {start_dt.date()} to {end_dt.date()}")
    
    ist_start_utc = start_dt - IST_OFFSET
    ist_end_utc = (end_dt + pd.Timedelta(days=1)) - IST_OFFSET
    logger.info(f"Date filter range (IST to UTC): {start_dt.date()} 00:00 IST to {end_dt.date()} 23:59 IST = UTC {ist_start_utc} to {ist_end_utc}")
    
    total_bars = 0
    total_greeks = 0
    
    # Import tqdm for progress bars
    try:
        from tqdm import tqdm
        USE_TQDM = True
    except ImportError:
        USE_TQDM = False
        logger.warning("tqdm not available, progress bars disabled")
    
    # Process dated option files
    file_iter = tqdm(parquet_files, desc=f"Processing {symbol} options") if USE_TQDM else parquet_files
    for file in file_iter:
        metrics_tracker['files_attempted'] += 1
        try:
            df = pd.read_parquet(file)
            
            if 'symbol' not in df.columns or df.empty:
                continue
            
            # Convert option prices from paise to rupees
            price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in df.columns]
            if price_cols:
                for col in price_cols:
                    # Explicitly convert to float64 first to avoid FutureWarning
                    df[col] = df[col].astype('float64')
                    df[col] = df[col] / 100.0
            
            # Convert timestamp
            df['timestamp'] = df.apply(
                lambda row: yyyymmdd_seconds_to_datetime(row['date'], row['time']),
                axis=1
            )
            
            # Filter by date
            df = df[(df['timestamp'] >= ist_start_utc) & (df['timestamp'] < ist_end_utc)]
            
            if df.empty:
                continue
            
            metrics_tracker['files_succeeded'] += 1
            
            # Group by option symbol
            symbol_iter = tqdm(df.groupby('symbol'), desc=f"Processing {symbol} contracts") if USE_TQDM else df.groupby('symbol')
            for option_symbol, group in symbol_iter:
                metrics_tracker['contracts_attempted'] += 1
                try:
                    # Prepare Bars
                    bar_df = group[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
                    
                    # Ensure prices are in rupees
                    for col in ['open', 'high', 'low', 'close']:
                        bar_df[col] = bar_df[col].astype('float64')
                    bar_df['volume'] = bar_df['volume'].astype('float64').clip(lower=0)
                    bar_df['high'] = bar_df[['high', 'close', 'open']].max(axis=1)
                    bar_df['low'] = bar_df[['low', 'close', 'open']].min(axis=1)
                    bar_df = bar_df.drop_duplicates(subset=['timestamp'], keep='last')
                    bar_df = bar_df.set_index('timestamp').sort_index()
                    
                    # Parse contract details
                    try:
                        parsed = parse_nse_option_symbol(option_symbol)
                        expiry_date = parsed['expiry']
                        strike_price = parsed['strike']
                        option_type = parsed['option_type']
                        opt_type_code = 'CE' if option_type == 'CALL' else 'PE'
                        underlying_symbol = parsed['underlying']
                        
                        # Use expiry from dataframe if available
                        if 'expiry' in group.columns:
                            exp_int = group['expiry'].iloc[0]
                            try:
                                exp_str = str(int(exp_int))
                                if len(exp_str) == 6:  # YYMMDD
                                    year = 2000 + int(exp_str[:2])
                                    month = int(exp_str[2:4])
                                    day = int(exp_str[4:6])
                                    expiry_date = date(year, month, day)
                                elif len(exp_str) == 8:  # YYYYMMDD
                                    year = int(exp_str[:4])
                                    month = int(exp_str[4:6])
                                    day = int(exp_str[6:8])
                                    expiry_date = date(year, month, day)
                            except Exception:
                                pass
                        
                        venue_str = get_venue(underlying_symbol)
                        contract = create_options_contract(
                            symbol=option_symbol,
                            underlying=underlying_symbol,
                            strike=strike_price,
                            expiry=expiry_date,
                            option_kind=option_type,
                            venue=venue_str
                        )
                    except ValueError as e:
                        # Handle missing lot size - skip this option contract
                        if "Lot size not found" in str(e):
                            metrics_tracker['contracts_parse_failed'] += 1
                            _update_date_metrics(date_metrics, pd.Timestamp.now().date(), 'skipped', 'missing_lot_size')
                            logger.error(f"Skipping {option_symbol}: {e}")
                            continue
                        else:
                            # Re-raise other ValueErrors
                            raise
                    except Exception as e:
                        metrics_tracker['contracts_parse_failed'] += 1
                        _update_date_metrics(date_metrics, pd.Timestamp.now().date(), 'skipped', 'parse_error')
                        # Log first few parse errors to understand the format
                        if metrics_tracker['contracts_parse_failed'] <= 5:
                            logger.warning(f"Failed to parse option symbol '{option_symbol}': {e}")
                            # Check if there are other columns that might help
                            if 'expiry' in group.columns:
                                logger.warning(f"  Available columns: {list(group.columns)}")
                                logger.warning(f"  Sample expiry value: {group['expiry'].iloc[0] if len(group) > 0 else 'N/A'}")
                        continue
                    
                    # Apply filters
                    expiry_date_obj = expiry_date if isinstance(expiry_date, date) else pd.to_datetime(expiry_date).date()
                    if expiry_filter_set and expiry_date_obj not in expiry_filter_set:
                        filtered_by_expiry += 1
                        continue
                    if strike_min is not None and strike_price < strike_min:
                        filtered_by_strike += 1
                        continue
                    if strike_max is not None and strike_price > strike_max:
                        filtered_by_strike += 1
                        continue
                    
                    # Create Bars
                    if bar_df.empty:
                        continue
                    
                    bar_type = BarType.from_str(f"{contract.id}-1-MINUTE-LAST-EXTERNAL")
                    wrangler = BarDataWrangler(bar_type, contract)
                    bars = wrangler.process(bar_df)
                    
                    catalog.write_data([contract])
                    catalog.write_data(bars, skip_disjoint_check=True)
                    
                    metrics_tracker['contracts_succeeded'] += 1
                    
                    # Calculate Greeks
                    processed = 0
                    skipped_missing_spot = 0
                    skipped_tte = 0
                    skipped_invalid_price = 0
                    records_analytic_fallback = 0
                    
                    if not spot_df.empty:
                        # CRITICAL FIX: Use merge_asof to ensure all strikes at same timestamp get same spot
                        # Round timestamps to nearest minute to ensure consistency across strikes
                        bar_df_reset = bar_df.reset_index()
                        bar_df_reset['timestamp_rounded'] = bar_df_reset['timestamp'].dt.round('1min')
                        
                        spot_df_reset = spot_df.reset_index()
                        spot_df_reset['timestamp_rounded'] = spot_df_reset['timestamp'].dt.round('1min')
                        
                        # Use merge_asof to get nearest spot price (within 1 minute tolerance)
                        # This ensures all strikes at the same rounded timestamp get the same spot
                        calc_df = pd.merge_asof(
                            bar_df_reset.sort_values('timestamp_rounded'),
                            spot_df_reset[['timestamp_rounded', 'spot_price']].sort_values('timestamp_rounded'),
                            on='timestamp_rounded',
                            direction='nearest',
                            tolerance=pd.Timedelta('1min')
                        )
                        
                        # Drop the rounded timestamp column, keep original
                        calc_df = calc_df.drop(columns=['timestamp_rounded'])
                        calc_df = calc_df.drop_duplicates(subset=['timestamp'], keep='last')
                        calc_df = calc_df.set_index('timestamp').sort_index()
                        
                        # Validate spot prices after merge - filter out invalid values
                        # Spot should be in reasonable range (10-10000 for stocks, 1000-100000 for indices)
                        median_spot_check = calc_df['spot_price'].median()
                        if median_spot_check > 0:
                            # Filter out obviously wrong spot prices (likely from bad forward-fill)
                            # Keep only spot prices within 10x of median (reasonable range)
                            spot_lower_bound = max(10, median_spot_check * 0.1)
                            spot_upper_bound = min(100000, median_spot_check * 10)
                            invalid_mask = (calc_df['spot_price'] < spot_lower_bound) | (calc_df['spot_price'] > spot_upper_bound)
                            if invalid_mask.sum() > 0:
                                logger.debug(f"Filtering {invalid_mask.sum()} records with invalid spot prices (outside {spot_lower_bound:.2f}-{spot_upper_bound:.2f} range)")
                                calc_df.loc[invalid_mask, 'spot_price'] = np.nan
                        
                        # Session-aware forward-fill for missing spot (only if merge_asof didn't find match)
                        n_missing_before = calc_df['spot_price'].isna().sum()
                        if n_missing_before > 0:
                            calc_df = _session_aware_forward_fill(
                                calc_df, 'spot_price', max_gap_minutes=90, use_interpolation=False, venue=venue_str
                            )
                            n_missing_after = calc_df['spot_price'].isna().sum()
                            if n_missing_after < n_missing_before:
                                metrics_tracker['spot_fallback_forward_fill'] += int(n_missing_before - n_missing_after)
                        
                        calc_df = calc_df.reset_index()
                        greeks_list = []
                        
                        # Fix expiry time: NSE expiry is 15:30 IST
                        expiry_ts = pd.Timestamp(expiry_date).tz_localize('Asia/Kolkata').replace(hour=15, minute=30).tz_convert('UTC')
                        
                        for idx, row in calc_df.iterrows():
                            ts = row['timestamp']
                            spot = row['spot_price']
                            price = row['close']
                            
                            if hasattr(ts, 'date'):
                                record_date = ts.date()
                            else:
                                record_date = pd.Timestamp(ts).date()
                            
                            # Ensure timestamp is tz-aware (UTC) for lookback
                            if ts.tzinfo is None:
                                ts_utc = ts.tz_localize('UTC')
                            else:
                                ts_utc = ts.tz_convert('UTC')
                            
                            # Validate spot price - check for reasonable range
                            # For stocks: 10-10000 rupees is reasonable
                            # For indices: 1000-100000 is reasonable
                            # Reject obviously invalid: <= 0, > 1,000,000, or NaN
                            # Also reject if spot is way too low compared to strike (suggests unit mismatch)
                            spot_invalid = False
                            if pd.isna(spot) or spot <= 0 or spot > 1000000:
                                spot_invalid = True
                            elif spot < 10:
                                # Too low for any reasonable stock/index price
                                spot_invalid = True
                                logger.debug(f"Spot price too low: {spot:.2f} for {option_symbol} at {ts}")
                            
                            # Try 2-hour lookback fallback if spot is missing or invalid
                            if spot_invalid:
                                lookback_spot = _get_spot_with_lookback(spot_df, ts_utc, SPOT_LOOKBACK_HOURS)
                                if lookback_spot is not None:
                                    spot = lookback_spot
                                    metrics_tracker['spot_fallback_lookback'] += 1
                                    logger.debug(f"Used 2-hour lookback spot for {option_symbol} at {ts}: {spot:.2f}")
                                else:
                                    # Still missing/invalid after lookback - skip
                                    skipped_missing_spot += 1
                                    if pd.isna(spot) or spot <= 0:
                                        _update_date_metrics(date_metrics, record_date, 'skipped', 'missing_spot_data')
                                    else:
                                        _update_date_metrics(date_metrics, record_date, 'skipped', 'spot_price_unreasonable')
                                    continue
                            
                            # Use ts_utc from above (already converted)
                            ts = ts_utc
                            
                            # Calculate TTE
                            tte = (expiry_ts - ts).total_seconds() / (365.25 * 24 * 3600)
                            
                            # Validate TTE
                            if tte <= 0:
                                skipped_tte += 1
                                _update_date_metrics(date_metrics, record_date, 'skipped', 'tte_negative')
                                continue
                            
                            if tte < MIN_TTE_DAYS:
                                skipped_tte += 1
                                _update_date_metrics(date_metrics, record_date, 'skipped', 'tte_too_small')
                                continue
                            
                            # Get risk-free rate with validation
                            try:
                                r = ir_provider.get_risk_free_rate(ts.date())
                                r = _validate_risk_free_rate(r, ts.date())
                            except Exception as e:
                                logger.debug(f"Failed to get risk-free rate for {ts.date()}: {e}, using default")
                                r = R_DEFAULT
                            
                            # Calculate intrinsic value
                            if opt_type_code == 'CE':
                                intrinsic_value = max(0.0, spot - strike_price * np.exp(-r * tte))
                            else:
                                intrinsic_value = max(0.0, strike_price * np.exp(-r * tte) - spot)
                            
                            # Validate price
                            price_ok, price_reason = _price_consistent_with_moneyness(
                                price, spot, strike_price, opt_type_code, tte
                            )
                            if not price_ok:
                                skipped_invalid_price += 1
                                _update_date_metrics(date_metrics, record_date, 'skipped', price_reason)
                                
                                # Track skip details for debugging
                                skip_key = (strike_price, opt_type_code)
                                if skip_key not in skip_details:
                                    skip_details[skip_key] = {
                                        'count': 0,
                                        'price_min': float('inf'),
                                        'price_max': float('-inf'),
                                        'intrinsic_min': float('inf'),
                                        'intrinsic_max': float('-inf'),
                                        'spot_min': float('inf'),
                                        'spot_max': float('-inf'),
                                    }
                                skip_details[skip_key]['count'] += 1
                                skip_details[skip_key]['price_min'] = min(skip_details[skip_key]['price_min'], price)
                                skip_details[skip_key]['price_max'] = max(skip_details[skip_key]['price_max'], price)
                                skip_details[skip_key]['intrinsic_min'] = min(skip_details[skip_key]['intrinsic_min'], intrinsic_value)
                                skip_details[skip_key]['intrinsic_max'] = max(skip_details[skip_key]['intrinsic_max'], intrinsic_value)
                                skip_details[skip_key]['spot_min'] = min(skip_details[skip_key]['spot_min'], spot)
                                skip_details[skip_key]['spot_max'] = max(skip_details[skip_key]['spot_max'], spot)
                                
                                # Log detailed skip info if debug enabled
                                if log_greeks_debug:
                                    logger.debug(
                                        f"Skipped {option_symbol} at {ts}: {price_reason} "
                                        f"(price={price:.2f}, intrinsic={intrinsic_value:.2f}, spot={spot:.2f}, strike={strike_price:.2f}, tte={tte:.4f})"
                                    )
                                continue
                            
                            # PHASE 2 FIX: Handle prices at/below intrinsic with analytic Greeks
                            if price <= intrinsic_value or price <= 0:
                                iv, delta, gamma, theta, vega, rho = _calculate_analytic_greeks(
                                    spot, strike_price, r, tte, opt_type_code, intrinsic_value
                                )
                                _update_date_metrics(date_metrics, record_date, 'zero_iv')
                                path_label = "analytic_iv0"
                            else:
                                # Normal IV calculation with PHASE 2 FIX: Analytic fallback on exceptions
                                try:
                                    pricer = OptionPricing(S=spot, K=strike_price, r=r, T=tte)
                                    iv = pricer.ImplVolWithBrent(price, opt_type_code)
                                    
                                    if iv is None or iv >= pricer.IV_UPPER_BOUND - 1e-5:
                                        # IV solver failed or hit upper bound - use analytic fallback
                                        records_analytic_fallback += 1
                                        metrics_tracker['records_analytic_fallback'] += 1
                                        _update_date_metrics(date_metrics, record_date, 'analytic_fallback')
                                        if iv is None:
                                            _update_date_metrics(date_metrics, record_date, 'skipped', 'failed_iv_solve')
                                        else:
                                            _update_date_metrics(date_metrics, record_date, 'skipped', 'iv_at_upper_bound')
                                        logger.debug(
                                            f"IV solver failed or hit upper bound for {option_symbol} at {ts} "
                                            f"(iv={iv}, S={spot:.2f}, K={strike_price:.2f}, price={price:.2f}), "
                                            f"using analytic fallback"
                                        )
                                        iv, delta, gamma, theta, vega, rho = _calculate_analytic_greeks(
                                            spot, strike_price, r, tte, opt_type_code, intrinsic_value
                                        )
                                        path_label = "analytic_fallback"
                                    else:
                                        # Check IV bounds (log but don't skip)
                                        if iv <= pricer.IV_LOWER_BOUND + 1e-5:
                                            _update_date_metrics(date_metrics, record_date, 'at_bounds')
                                        
                                        # Calculate Greeks
                                        delta = pricer.Delta(iv, opt_type_code)
                                        gamma = pricer.Gamma(iv)
                                        theta_annual = pricer.Theta(iv, opt_type_code)
                                        vega = pricer.Vega(iv)
                                        rho = pricer.Rho(iv, opt_type_code)
                                        theta = theta_annual / 365.25
                                        path_label = "solver"
                                
                                except ValueError as e:
                                    # PHASE 2 FIX: Fall back to analytic instead of skipping
                                    records_analytic_fallback += 1
                                    metrics_tracker['records_analytic_fallback'] += 1
                                    _update_date_metrics(date_metrics, record_date, 'analytic_fallback')
                                    _update_date_metrics(date_metrics, record_date, 'skipped', 'failed_iv_valueerror')
                                    logger.debug(f"IV solver ValueError for {option_symbol} at {ts}: {e}, using analytic fallback")
                                    iv, delta, gamma, theta, vega, rho = _calculate_analytic_greeks(
                                        spot, strike_price, r, tte, opt_type_code, intrinsic_value
                                    )
                                    path_label = "analytic_fallback_valueerror"
                                
                                except KeyError as e:
                                    # PHASE 2 FIX: Fall back to analytic instead of skipping
                                    records_analytic_fallback += 1
                                    metrics_tracker['records_analytic_fallback'] += 1
                                    _update_date_metrics(date_metrics, record_date, 'analytic_fallback')
                                    _update_date_metrics(date_metrics, record_date, 'skipped', 'failed_iv_keyerror')
                                    logger.debug(f"IV solver KeyError for {option_symbol} at {ts}: {e}, using analytic fallback")
                                    iv, delta, gamma, theta, vega, rho = _calculate_analytic_greeks(
                                        spot, strike_price, r, tte, opt_type_code, intrinsic_value
                                    )
                                    path_label = "analytic_fallback_keyerror"
                                
                                except Exception as e:
                                    # PHASE 2 FIX: Fall back to analytic instead of skipping
                                    records_analytic_fallback += 1
                                    metrics_tracker['records_analytic_fallback'] += 1
                                    _update_date_metrics(date_metrics, record_date, 'analytic_fallback')
                                    _update_date_metrics(date_metrics, record_date, 'skipped', 'failed_iv_generic')
                                    logger.debug(f"IV solver exception for {option_symbol} at {ts}: {e}, using analytic fallback", exc_info=True)
                                    iv, delta, gamma, theta, vega, rho = _calculate_analytic_greeks(
                                        spot, strike_price, r, tte, opt_type_code, intrinsic_value
                                    )
                                    path_label = "analytic_fallback_generic"
                            
                            # Create OptionGreeks object
                            ts_ns = int(ts.timestamp() * 1_000_000_000)
                            greeks = OptionGreeks(
                                instrument_id=contract.id,
                                iv=iv,
                                delta=delta,
                                gamma=gamma,
                                theta=theta,
                                vega=vega,
                                rho=rho,
                                ts_event=ts_ns,
                                ts_init=ts_ns
                            )
                            greeks_list.append(greeks)
                            processed += 1
                            
                            _update_date_metrics(date_metrics, record_date, 'processed', iv)
                            _update_date_metrics(date_metrics, record_date, 'theta', theta)
                            
                            _log_greeks_debug(
                                log_greeks_debug,
                                f"{option_symbol} {ts}: {path_label} spot={spot:.2f} strike={strike_price:.2f} "
                                f"price={price:.2f} tte={tte:.6f} iv={iv:.4f} delta={delta:.4f}"
                            )
                        
                        # Write Greeks
                        if greeks_list:
                            try:
                                catalog.write_data(greeks_list)
                                total_greeks += len(greeks_list)
                            except Exception as e:
                                metrics_tracker['greeks_write_failures'] += 1
                                logger.error(f"Failed to write Greeks for {option_symbol}: {e}", exc_info=True)
                    
                    logger.info(
                        f"{option_symbol}: Processed {processed} records, "
                        f"skipped: missing_spot={skipped_missing_spot}, tte={skipped_tte}, "
                        f"invalid_price={skipped_invalid_price}, analytic_fallback={records_analytic_fallback}"
                    )
                    
                    # Warn if all records were skipped due to missing spot
                    if processed == 0 and skipped_missing_spot > 0:
                        total_records = skipped_missing_spot + skipped_tte + skipped_invalid_price
                        if skipped_missing_spot == total_records:
                            logger.warning(
                                f"⚠️  {option_symbol}: All {skipped_missing_spot} records skipped due to missing spot price. "
                                f"Check if spot data exists for underlying '{symbol}' in date range {start_date} to {end_date}"
                            )
                    
                    # Option OI
                    if "oi" in group.columns:
                        oi_data_list = []
                        prev_oi = 0
                        for idx, oi_row in group.iterrows():
                            current_oi = int(oi_row["oi"])
                            coi = current_oi - prev_oi
                            prev_oi = current_oi
                            ts_ns = int(oi_row["timestamp"].timestamp() * 1_000_000_000)
                            oi_data = OptionOI(
                                instrument_id=contract.id,
                                oi=current_oi,
                                coi=coi,
                                ts_event=ts_ns,
                                ts_init=ts_ns
                            )
                            oi_data_list.append(oi_data)
                        
                        if oi_data_list:
                            oi_data_list.sort(key=lambda x: x.ts_init)
                            catalog.write_data(oi_data_list)
                    
                    total_bars += len(bars)
                    
                except Exception as e:
                    logger.error(f"Error processing option {option_symbol}: {e}", exc_info=True)
                    continue
        
        except Exception as e:
            metrics_tracker['files_failed'] += 1
            logger.warning(f"Error reading {file}: {e}")
            continue
    
    # Log summary
    if expiry_filter_set:
        logger.info(f"Expiry filter skipped {filtered_by_expiry} contract(s) outside {allowed_expiry_strings}")
    if strike_min is not None or strike_max is not None:
        logger.info(f"Strike filter skipped {filtered_by_strike} contract(s)")
    
    logger.info(f"✅ {symbol} options: Created {total_bars:,} bars + {total_greeks:,} Greeks records")
    logger.info(f"Metrics: {metrics_tracker}")
    
    # Log per-date metrics
    if date_metrics or dates_with_files:
        logger.info("Per-date Greeks metrics:")
        # Log all dates in range, showing which had files but no records
        for current_date in pd.date_range(start_dt, end_dt, freq='D'):
            date_key = current_date.date()
            if date_key in date_metrics:
                metrics = date_metrics[date_key]
            else:
                # Date had files but no records processed
                metrics = {
                    'records_processed': 0,
                    'records_skipped': 0,
                    'skip_reasons': {},
                    'iv_values': [],
                    'records_analytic_fallback': 0
                }
            iv_values = metrics['iv_values']
            skip_reasons = ', '.join([f"{k}={v}" for k, v in metrics['skip_reasons'].items() if v > 0]) or 'none'
            # Handle empty IV values to avoid RuntimeWarning
            if iv_values and len(iv_values) > 0:
                iv_min = np.min(iv_values)
                iv_max = np.max(iv_values)
            else:
                iv_min = 0.0
                iv_max = 0.0
            logger.info(
                f"  {date_key}: processed={metrics['records_processed']} skipped={metrics['records_skipped']} "
                f"IV[{iv_min:.4f}, {iv_max:.4f}] "
                f"skip_reasons={skip_reasons} analytic_fallback={metrics['records_analytic_fallback']}"
            )
    
    
    # Log skip details summary if there were skips
    if skip_details:
        logger.info("")
        logger.info("SKIP DETAILS BY STRIKE/CONTRACT:")
        logger.info("-" * 80)
        # Sort by skip count (descending)
        sorted_skips = sorted(skip_details.items(), key=lambda x: x[1]['count'], reverse=True)
        for (strike, opt_type), details in sorted_skips[:20]:  # Top 20
            if details['count'] > 0:
                logger.info(
                    f"  {opt_type} Strike {strike:.0f}: {details['count']} skips | "
                    f"Price range: [{details['price_min']:.2f}, {details['price_max']:.2f}] | "
                    f"Intrinsic range: [{details['intrinsic_min']:.2f}, {details['intrinsic_max']:.2f}] | "
                    f"Spot range: [{details['spot_min']:.2f}, {details['spot_max']:.2f}]"
                )
        if len(sorted_skips) > 20:
            logger.info(f"  ... and {len(sorted_skips) - 20} more strike/contract combinations")
    
    # Return bars count and metrics for CLI summary
    return total_bars, metrics_tracker, date_metrics

