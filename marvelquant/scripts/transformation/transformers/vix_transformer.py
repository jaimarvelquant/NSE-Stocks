"""
VIX Bar Transformer.

Transforms India VIX CSV or Parquet data to Nautilus Bar format.
"""

import logging
from pathlib import Path

import pandas as pd
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.data import BarType
from nautilus_trader.model.instruments import IndexInstrument
from nautilus_trader.model.objects import Price, Quantity, Currency
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.persistence.wranglers import BarDataWrangler

from .common import (
    bars_to_quote_ticks,
)

logger = logging.getLogger(__name__)


def transform_vix_bars(
    input_file: Path,
    catalog: ParquetDataCatalog,
    start_date: str,
    end_date: str
) -> int:
    """
    Transform India VIX CSV or Parquet data to Nautilus Bar format.

    Args:
        input_file: Path to VIX CSV or Parquet file
        catalog: Nautilus ParquetDataCatalog instance
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Number of bars created
    """
    logger.info(f"Transforming India VIX bars from {input_file}...")

    if not input_file.exists():
        logger.warning(f"VIX data file not found: {input_file}")
        return 0

    # Detect file format and read accordingly
    file_ext = input_file.suffix.lower()
    if file_ext == '.parquet':
        df = pd.read_parquet(input_file)
        logger.info(f"Loaded {len(df):,} rows from VIX Parquet file")
    elif file_ext == '.csv':
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df):,} rows from VIX CSV file")
    else:
        logger.error(f"Unsupported file format: {file_ext}. Expected .csv or .parquet")
        return 0

    if df.empty:
        logger.warning(f"No data in VIX file: {input_file}")
        return 0

    # Parse date column - it has timezone info like "2018-01-01 09:15:00+05:30"
    # For Parquet files, the date column may already be a datetime with timezone
    if pd.api.types.is_datetime64_any_dtype(df['date']):
        # Already a datetime, check if it has timezone
        if df['date'].dt.tz is not None:
            df['timestamp'] = df['date'].dt.tz_convert('UTC')
        else:
            # Assume it's already UTC if no timezone
            df['timestamp'] = pd.to_datetime(df['date'])
    else:
        # String format, parse it
        df['timestamp'] = pd.to_datetime(df['date'])

    # Convert to UTC if not already (this handles the +05:30 offset correctly)
    # IST 09:15:00 -> UTC 03:45:00
    if df['timestamp'].dt.tz is not None:
        df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')

    # Remove timezone info to get naive UTC (required by Nautilus)
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)

    logger.info(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()} (UTC)")

    # Filter by date range
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    df = df[(df['timestamp'] >= start) & (df['timestamp'] < end)]

    if df.empty:
        logger.warning(f"No data in date range {start_date} to {end_date}")
        return 0

    logger.info(f"Filtered to {len(df):,} rows in date range")

    # Prepare DataFrame for BarDataWrangler
    # Required: columns ['open', 'high', 'low', 'close', 'volume'] with 'timestamp' as INDEX
    bar_df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()

    # Cleanse numeric columns
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    price_cols = ['open', 'high', 'low', 'close']
    for col in numeric_cols:
        bar_df[col] = pd.to_numeric(bar_df[col], errors='coerce')

    # Drop any rows with NaN price values
    invalid_price_mask = bar_df[price_cols].isna().any(axis=1)
    invalid_count = int(invalid_price_mask.sum())
    if invalid_count:
        logger.warning(f"Dropping {invalid_count} rows with NaN OHLC values")
        bar_df = bar_df[~invalid_price_mask]

    if bar_df.empty:
        logger.error("All rows dropped due to invalid price data")
        return 0

    # Validate OHLC relationships: low <= open/close <= high
    # Nautilus requires: low <= open, low <= close, high >= open, high >= close
    invalid_ohlc_mask = (
        (bar_df['low'] > bar_df['open']) |
        (bar_df['low'] > bar_df['close']) |
        (bar_df['high'] < bar_df['open']) |
        (bar_df['high'] < bar_df['close'])
    )
    invalid_ohlc_count = int(invalid_ohlc_mask.sum())
    if invalid_ohlc_count:
        logger.warning(
            "Dropping %d rows with invalid OHLC relationships (low > open/close or high < open/close)",
            invalid_ohlc_count,
        )
        bar_df = bar_df[~invalid_ohlc_mask]

    if bar_df.empty:
        logger.error("All rows dropped due to invalid OHLC data")
        return 0

    # VIX data has no real volume; ensure it's float
    bar_df['volume'] = bar_df['volume'].fillna(0.0).astype('float64')

    # Deduplicate by timestamp (keep last value)
    bar_df = bar_df.drop_duplicates(subset=['timestamp'], keep='last')

    # Set timestamp as index (CRITICAL for BarDataWrangler)
    bar_df = bar_df.set_index('timestamp')
    bar_df = bar_df.sort_index()

    # Create InstrumentId
    instrument_id = InstrumentId.from_str("INDIA-VIX.NSE")

    # Create IndexInstrument (VIX is a volatility index)
    instrument = IndexInstrument(
        instrument_id=instrument_id,
        raw_symbol=Symbol("INDIA-VIX"),
        currency=Currency.from_str("INR"),
        price_precision=2,
        price_increment=Price(0.01, 2),  # VIX has finer precision (values like 12.67)
        size_precision=0,
        size_increment=Quantity.from_int(1),
        ts_event=0,
        ts_init=0,
    )

    # Create bar type
    bar_type = BarType.from_str(f"{instrument_id}-1-MINUTE-LAST-EXTERNAL")

    # Use BarDataWrangler (official Nautilus pattern)
    wrangler = BarDataWrangler(bar_type, instrument)
    bars = wrangler.process(
        data=bar_df,
        default_volume=0.0,  # VIX has no real volume
        ts_init_delta=0
    )

    # Write to catalog
    catalog.write_data([instrument])  # Write instrument first
    catalog.write_data(bars, skip_disjoint_check=True)

    # Generate and write QuoteTicks for Greeks calculation
    quote_ticks = bars_to_quote_ticks(bars, instrument)
    catalog.write_data(quote_ticks, skip_disjoint_check=True)

    logger.info("✅ INDIA-VIX: Created %s bars + %s QuoteTicks", f"{len(bars):,}", f"{len(quote_ticks):,}")

    return len(bars)

