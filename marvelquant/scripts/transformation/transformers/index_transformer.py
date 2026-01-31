"""
Index Bar Transformer.

Transforms index data (NIFTY, BANKNIFTY, etc.) to Nautilus Bar format.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.data import BarType
from nautilus_trader.model.instruments import IndexInstrument
from nautilus_trader.model.objects import Price, Quantity, Currency
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.persistence.wranglers import BarDataWrangler

# Import common utilities - handle both relative and absolute imports
try:
    from .common import (
        get_venue,
        get_instrument_id_string,
        yyyymmdd_seconds_to_datetime,
        bars_to_quote_ticks,
        detect_price_unit,
        convert_paise_to_rupees,
        IST_OFFSET,
    )
except ImportError:
    # Fallback to absolute import when run directly
    from common import (
        get_venue,
        get_instrument_id_string,
        yyyymmdd_seconds_to_datetime,
        bars_to_quote_ticks,
        detect_price_unit,
        convert_paise_to_rupees,
        IST_OFFSET,
    )

logger = logging.getLogger(__name__)


def transform_index_bars(
    input_dir: Path,
    catalog: ParquetDataCatalog,
    symbol: str,
    start_date: str,
    end_date: str
) -> int:
    """
    Transform index data to Nautilus Bar format (OFFICIAL PATTERN).
    
    Args:
        input_dir: Directory containing raw parquet files
        catalog: Nautilus ParquetDataCatalog instance
        symbol: Symbol name (e.g., "NIFTY", "BANKNIFTY")
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        Number of bars created
    """
    logger.info(f"Transforming {symbol} index bars...")

    # Find all parquet files for this symbol
    symbol_dir = input_dir / "index" / symbol
    parquet_files = list(symbol_dir.rglob("*.parquet"))
    
    if not parquet_files:
        logger.warning(f"No parquet files found in {symbol_dir}")
        return 0
    
    # Read all files into one DataFrame
    dfs = []
    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            
            # CRITICAL FIX: Apply price unit detection and conversion at most once
            # Use median-based detection to determine if prices are in paise
            price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in df.columns]
            if price_cols:
                # Check if prices need conversion (median-based detection)
                if detect_price_unit(df, price_column='close', expected_range=(1000, 100000)):
                    logger.debug(f"Detected paise prices in {file.name}, converting to rupees")
                    df = convert_paise_to_rupees(df, price_cols)
                else:
                    # Ensure float64 type even if no conversion needed
                    for col in price_cols:
                        df[col] = df[col].astype('float64')
            
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Error reading {file}: {e}")
            continue
    
    if not dfs:
        logger.error("No data loaded")
        return 0
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Convert date + time to datetime timestamp
    combined_df['timestamp'] = combined_df.apply(
        lambda row: yyyymmdd_seconds_to_datetime(row['date'], row['time']),
        axis=1
    )
    
    logger.info(f"Data range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    
    # Filter by date range (account for IST->UTC conversion: IST dates start at UTC-5:30)
    start = pd.to_datetime(start_date) - pd.Timedelta(hours=6)  # Buffer for IST conversion
    end = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    combined_df = combined_df[(combined_df['timestamp'] >= start) & 
                               (combined_df['timestamp'] < end)]
    
    if combined_df.empty:
        logger.warning(f"No data in date range {start_date} to {end_date}")
        return 0
    
    # OFFICIAL PATTERN: Prepare DataFrame for BarDataWrangler
    # Required: columns ['open', 'high', 'low', 'close', 'volume'] with 'timestamp' as INDEX
    bar_df = combined_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()

    # CRITICAL: Deduplicate by timestamp (keep last value) before setting index
    bar_df = bar_df.drop_duplicates(subset=['timestamp'], keep='last')
    
    bar_df = bar_df.set_index('timestamp')  # CRITICAL: Set timestamp as index!
    bar_df = bar_df.sort_index()  # Sort by timestamp
    
    # Determine Venue
    venue_str = get_venue(symbol)
    venue = Venue(venue_str)
    
    # Create InstrumentId
    # Use helper to ensure consistent naming (e.g. NIFTY-INDEX.NSE)
    instrument_id_str = get_instrument_id_string(symbol, "index")
    instrument_id = InstrumentId.from_str(instrument_id_str)
    
    # Use IndexInstrument (not Equity) for index instruments
    instrument = IndexInstrument(
        instrument_id=instrument_id,
        raw_symbol=Symbol(symbol),
        currency=Currency.from_str("INR"),
        price_precision=2,
        price_increment=Price(0.05, 2),
        size_precision=0,
        size_increment=Quantity.from_int(1),
        ts_event=0,
        ts_init=0,
    )
    
    # Create bar type
    bar_type = BarType.from_str(f"{instrument_id}-1-MINUTE-LAST-EXTERNAL")
    
    # OFFICIAL PATTERN: Use BarDataWrangler
    wrangler = BarDataWrangler(bar_type, instrument)
    bars = wrangler.process(
        data=bar_df,
        default_volume=0.0,  # Index data has no real volume
        ts_init_delta=0
    )
    
    # OFFICIAL PATTERN: Write to catalog
    catalog.write_data([instrument])  # Write instrument first
    catalog.write_data(bars, skip_disjoint_check=True)  # Skip check for overlapping data

    # Generate and write QuoteTicks for Greeks calculation
    quote_ticks = bars_to_quote_ticks(bars, instrument)
    catalog.write_data(quote_ticks, skip_disjoint_check=True)
    logger.info(f"✅ {symbol}: Created {len(bars):,} bars + {len(quote_ticks):,} QuoteTicks")

    return len(bars)

