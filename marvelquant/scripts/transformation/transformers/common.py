"""
Shared utilities for Nautilus data transformation.

This module contains common helpers used across index, futures, and options transformers:
- Time/venue helpers (IST offset, venue mapping, timestamp conversion)
- Price conversion utilities
- Logging configuration
- Quote-tick conversion helpers
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List

import pandas as pd
import numpy as np
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.objects import Price, Quantity

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# IST offset (5 hours 30 minutes)
IST_OFFSET = timedelta(hours=5, minutes=30)

# Venue Mapping Configuration
# Maps underlying symbols to their primary exchange venue
VENUE_MAP = {
    # NSE Indices
    "NIFTY": "NSE",
    "BANKNIFTY": "NSE",
    "FINNIFTY": "NSE",
    "MIDCPNIFTY": "NSE",
    
    # BSE Indices
    "SENSEX": "BSE",
    "BANKEX": "BSE",
    
    # MCX Commodities
    "CRUDEOIL": "MCX",
    "NATURALGAS": "MCX",
    "GOLD": "MCX",
    "SILVER": "MCX",
    "COPPER": "MCX",
    "ZINC": "MCX",
    "LEAD": "MCX",
    "ALUMINIUM": "MCX",
    "NICKEL": "MCX",
}

# Default fallback venue
DEFAULT_VENUE = "NSE"

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

def configure_logging(log_level: str, log_file: Optional[Path]) -> Optional[Path]:
    """
    Configure console + optional file logging at runtime with daily rotation.
    Returns the resolved log file path if a file handler is configured.
    
    Args:
        log_level: Logging level (e.g., "INFO", "DEBUG")
        log_file: Optional path to log file. If provided, logs will rotate daily
                 at midnight with date suffix (e.g., log_2025-01-15.log)
        
    Returns:
        Resolved log file path if configured, None otherwise
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    resolved_log_file: Optional[Path] = None

    root_logger = logging.getLogger()
    # Clear any existing handlers so CLI reconfiguration works reliably
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file:
        resolved_log_file = log_file.expanduser()
        if not resolved_log_file.is_absolute():
            resolved_log_file = (PROJECT_ROOT.parent / resolved_log_file).resolve()
        resolved_log_file.parent.mkdir(parents=True, exist_ok=True)

        # Use TimedRotatingFileHandler for daily rotation
        # Rotate at midnight, keep 30 days of backup logs
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=str(resolved_log_file),
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.suffix = '%Y-%m-%d'
        root_logger.addHandler(file_handler)

    root_logger.setLevel(level)
    return resolved_log_file

# ---------------------------------------------------------------------------
# Venue Helpers
# ---------------------------------------------------------------------------

def get_venue(symbol: str) -> str:
    """
    Get the correct exchange venue for a symbol.
    
    Args:
        symbol: Underlying symbol (e.g., "NIFTY", "SENSEX", "SBIN")
        
    Returns:
        Venue string (e.g., "NSE", "BSE", "MCX")
    """
    # Normalize symbol for lookup
    norm_symbol = symbol.upper()
    return VENUE_MAP.get(norm_symbol, DEFAULT_VENUE)

def get_instrument_id_string(symbol: str, instrument_type: str) -> str:
    """
    Generate standardized InstrumentId string based on type and venue.
    
    Format:
    - Index: {SYMBOL}-INDEX.{VENUE} (e.g., NIFTY-INDEX.NSE)
    - Futures: {SYMBOL}-I.{VENUE} (e.g., NIFTY-I.NSE for continuous)
    - Equity: {SYMBOL}.{VENUE} (e.g., SBIN.NSE)
    - Options: handled by contract generators using specific option symbols
    
    Args:
        symbol: Base symbol (e.g., "NIFTY", "SBIN")
        instrument_type: "index", "future", "equity"
        
    Returns:
        Formatted InstrumentId string
    """
    venue = get_venue(symbol)
    norm_symbol = symbol.upper()
    
    if instrument_type == "index":
        return f"{norm_symbol}-INDEX.{venue}"
    elif instrument_type == "future":
        return f"{norm_symbol}-I.{venue}"
    elif instrument_type == "equity":
        return f"{norm_symbol}.{venue}"
    else:
        return f"{norm_symbol}.{venue}"  # Default fallthrough

# ---------------------------------------------------------------------------
# Time Conversion Helpers
# ---------------------------------------------------------------------------

def yyyymmdd_seconds_to_datetime(date_int, time_int: int) -> datetime:
    """
    Convert YYYYMMDD integer + seconds to datetime in UTC.
    
    Args:
        date_int: Date as YYYYMMDD (e.g., 20240102) or datetime.date object
        time_int: Time as seconds since midnight (e.g., 33300 = 09:15:00)
    
    Returns:
        datetime in UTC
    """
    # Handle both int and datetime.date types
    if isinstance(date_int, (datetime, pd.Timestamp)):
        date_int = int(date_int.strftime('%Y%m%d'))
    elif hasattr(date_int, 'year'):  # datetime.date object
        date_int = date_int.year * 10000 + date_int.month * 100 + date_int.day
    
    # Parse date (ensure integer conversion)
    date_int = int(date_int)
    
    # Handle both YYYYMMDD (8 digits) and YYMMDD (6 digits) formats
    if date_int < 1000000:  # YYMMDD format (6 digits)
        # Interpret 2-digit year as 20YY (e.g., 251117 -> 2025-11-17)
        year_2digit = int(date_int // 10000)
        month = int((date_int % 10000) // 100)
        day = int(date_int % 100)
        # Convert 2-digit year to 4-digit (assume 2000-2099 range)
        year = 2000 + year_2digit if year_2digit < 100 else year_2digit
    else:  # YYYYMMDD format (8 digits)
        year = int(date_int // 10000)
        month = int((date_int % 10000) // 100)
        day = int(date_int % 100)
    
    # Parse time (ensure integer conversion)
    time_int = int(time_int)
    hours = int(time_int // 3600)
    minutes = int((time_int % 3600) // 60)
    seconds = int(time_int % 60)
    
    # Create IST datetime (naive)
    ist_dt = datetime(year, month, day, hours, minutes, seconds)
    
    # Convert to UTC
    utc_dt = ist_dt - IST_OFFSET
    
    return utc_dt

# ---------------------------------------------------------------------------
# Price Conversion Helpers
# ---------------------------------------------------------------------------

def detect_price_unit(df: pd.DataFrame, price_column: str = 'close', 
                      expected_range: tuple = (1000, 100000)) -> bool:
    """
    Detect if prices are in paise (×100) instead of rupees using median-based detection.
    
    Args:
        df: DataFrame with price column
        price_column: Name of price column to check
        expected_range: (min, max) expected range for prices in rupees
        
    Returns:
        True if prices appear to be in paise (need conversion), False otherwise
    """
    if df.empty or price_column not in df.columns:
        return False
    
    median_price = df[price_column].median()
    min_expected, max_expected = expected_range
    
    # If median is > max_expected, likely in paise
    if median_price > max_expected:
        return True
    
    # If median is < min_expected, might be wrong units but don't auto-convert
    if median_price < min_expected:
        logger.warning(
            f"Prices seem unusually low (median={median_price:.2f}). "
            f"Expected range: {min_expected}-{max_expected}"
        )
    
    return False

def convert_paise_to_rupees(df: pd.DataFrame, price_columns: List[str]) -> pd.DataFrame:
    """
    Convert price columns from paise to rupees (divide by 100).
    
    Args:
        df: DataFrame with price columns
        price_columns: List of column names to convert
        
    Returns:
        DataFrame with converted prices
    """
    result_df = df.copy()
    for col in price_columns:
        if col in result_df.columns:
            result_df[col] = result_df[col].astype('float64') / 100.0
    return result_df

# ---------------------------------------------------------------------------
# Quote-Tick Conversion Helpers
# ---------------------------------------------------------------------------

def bars_to_quote_ticks(bars, instrument):
    """
    Convert Bar data to QuoteTicks for Greeks calculation.

    Creates QuoteTicks where bid=ask=close price.
    This is required for NautilusTrader Greeks calculator.
    
    Args:
        bars: List of Bar objects
        instrument: Instrument object
        
    Returns:
        List of QuoteTick objects
    """
    quote_ticks = []

    for bar in bars:
        # Create QuoteTick using close price as both bid and ask
        price = Price(bar.close.as_double(), instrument.price_precision)
        size = Quantity(1, instrument.size_precision)

        tick = QuoteTick(
            instrument_id=instrument.id,
            bid_price=price,
            ask_price=price,
            bid_size=size,
            ask_size=size,
            ts_event=bar.ts_event,
            ts_init=bar.ts_init,
        )
        quote_ticks.append(tick)

    return quote_ticks

# ---------------------------------------------------------------------------
# Forward Fill Helpers
# ---------------------------------------------------------------------------

def time_aware_forward_fill(
    df: pd.DataFrame,
    column: str,
    max_gap_minutes: int = 90,
    use_interpolation: bool = True
) -> pd.DataFrame:
    """
    Intelligently forward-fill missing values in a time series DataFrame.
    
    Uses time-aware forward fill with gap limits and optional interpolation:
    1. Forward-fills gaps up to max_gap_minutes (default 90 minutes)
    2. For larger gaps, uses linear interpolation if enabled
    3. Falls back to backward-fill for remaining gaps
    
    This prevents filling from stale data (e.g., filling from hours/days ago) and
    provides more accurate imputation for financial time series data.
    
    Args:
        df: DataFrame with datetime index and column to fill
        column: Name of column to forward-fill
        max_gap_minutes: Maximum time gap (in minutes) to forward-fill (default: 90)
        use_interpolation: Whether to use interpolation for larger gaps (default: True)
    
    Returns:
        DataFrame with filled column
    """
    if df.empty or column not in df.columns:
        return df
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"DataFrame index must be DatetimeIndex, got {type(df.index)}")
    
    result_df = df.copy()
    original_missing = result_df[column].isna().sum()
    
    if original_missing == 0:
        return result_df
    
    # Step 1: Time-aware forward fill with gap limit
    max_gap = pd.Timedelta(minutes=max_gap_minutes)
    
    # Track which values were originally missing (to only clear forward-filled ones)
    originally_missing = result_df[column].isna()
    
    # Forward fill normally first
    filled_series = result_df[column].ffill()
    
    # Identify where gaps exceed max_gap and clear those forward-filled values
    time_diffs = result_df.index.to_series().diff()
    
    # Find indices where gap > max_gap (these shouldn't be forward-filled)
    large_gap_mask = time_diffs > max_gap
    
    # For rows after large gaps, clear the forward-filled value (set back to NaN)
    shifted_mask = large_gap_mask.shift(1)
    rows_after_large_gaps = shifted_mask.fillna(False).infer_objects(copy=False).astype(bool)
    
    # Clear forward-filled values that came from data before a large gap
    clear_mask = rows_after_large_gaps & originally_missing
    filled_series.loc[clear_mask] = np.nan
    
    result_df[column] = filled_series
    
    # Step 2: Interpolate remaining gaps if enabled
    if use_interpolation:
        remaining_missing = result_df[column].isna().sum()
        if remaining_missing > 0:
            result_df[column] = result_df[column].interpolate(method='time', limit_direction='both')
    
    # Step 3: Backward-fill as final fallback (for leading missing values)
    remaining_missing = result_df[column].isna().sum()
    if remaining_missing > 0:
        result_df[column] = result_df[column].bfill()
    
    final_missing = result_df[column].isna().sum()
    filled_count = original_missing - final_missing
    
    if filled_count > 0:
        logger.debug(
            f"Time-aware forward-fill: filled {filled_count} missing values "
            f"(max_gap={max_gap_minutes}min, interpolation={'on' if use_interpolation else 'off'}), "
            f"{final_missing} still missing"
        )
    
    return result_df

