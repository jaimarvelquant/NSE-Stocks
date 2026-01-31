"""
Futures Bar Transformer.

Transforms futures data to Nautilus Bar format + FutureOI records.
"""

import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from nautilus_trader.model.data import BarType
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.persistence.wranglers import BarDataWrangler

# Import contract generators and data types - handle both absolute and direct imports
try:
    from marvelquant.utils.contract_generators import create_futures_contract
    from marvelquant.data.types import FutureOI
except ImportError:
    # Fallback to direct imports
    import sys
    from pathlib import Path
    utils_path = Path(__file__).parent.parent.parent.parent / "utils"
    data_path = Path(__file__).parent.parent.parent.parent / "data" / "types"
    if str(utils_path) not in sys.path:
        sys.path.insert(0, str(utils_path))
    if str(data_path) not in sys.path:
        sys.path.insert(0, str(data_path))
    from contract_generators import create_futures_contract
    from future_oi import FutureOI

# Import common utilities - handle both relative and absolute imports
try:
    from .common import (
        get_venue,
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
        yyyymmdd_seconds_to_datetime,
        bars_to_quote_ticks,
        detect_price_unit,
        convert_paise_to_rupees,
        IST_OFFSET,
    )

logger = logging.getLogger(__name__)


def transform_futures_bars(
    input_dir: Path,
    catalog: ParquetDataCatalog,
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: Optional[Path] = None
) -> Tuple[int, None]:
    """
    Transform futures data to Nautilus Bar format + separate OI DataFrame.
    
    Args:
        input_dir: Directory containing raw parquet files
        catalog: Nautilus ParquetDataCatalog instance
        symbol: Symbol name (e.g., "NIFTY", "BANKNIFTY")
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Optional output directory (unused, kept for compatibility)
    
    Returns:
        (bar_count, None) - OI DataFrame no longer returned
    """
    logger.info(f"Transforming {symbol} futures bars...")

    symbol_lower = symbol.lower()
    
    parquet_files = []
    
    # Try structure: futures/{symbol}/{year}/
    symbol_dir = input_dir / "futures" / symbol_lower
    if symbol_dir.exists():
        # Look for year subdirectories
        for year_dir in symbol_dir.iterdir():
            if year_dir.is_dir() and year_dir.name.isdigit():
                year_files = list(year_dir.rglob("*.parquet"))
                parquet_files.extend(year_files)
        if parquet_files:
            logger.info(f"Found {len(parquet_files)} files in structure: futures/{symbol_lower}/YYYY/")
    
    # If not found, try flat structure: futures/{symbol}/
    if not parquet_files:
        symbol_dir = input_dir / "futures" / symbol
        if not symbol_dir.exists():
            # Try alternative name: future/{symbol}/
            symbol_dir = input_dir / "future" / symbol
        
        if symbol_dir.exists():
            parquet_files = list(symbol_dir.rglob("*.parquet"))
    
    # If still not found, try nested structure: future/YYYY/MM/{symbol}/
    if not parquet_files:
        future_dir = input_dir / "future"
        if not future_dir.exists():
            future_dir = input_dir / "futures"
        if future_dir.exists():
            # Use rglob to find all parquet files for this symbol in nested structure
            parquet_files = list(future_dir.rglob(f"{symbol_lower}/*.parquet"))
            if parquet_files:
                logger.info(f"Found {len(parquet_files)} files in nested structure: future/YYYY/MM/{symbol_lower}/")
    
    if not parquet_files:
        logger.warning(f"No parquet files found for symbol {symbol} in futures/future directories")
        return 0, None

    # CRITICAL: Only use dated files (symbol_YYYYMMDD.parquet) which are in RUPEES
    # Pattern: {symbol}_YYYYMMDD.parquet (e.g., abb_20250901.parquet)
    # Also support: {symbol}_future_YYYYMMDD.parquet (e.g., nifty_future_20250901.parquet)
    dated_files = [
        f for f in parquet_files 
        if f.stem.lower().startswith(f"{symbol_lower}_") and 
        (f.stem.lower().endswith('_future_') or '_' in f.stem.lower()[-9:])  # Has date pattern
    ]
    
    # If no files found with strict pattern, try any file starting with symbol
    if not dated_files:
        dated_files = [f for f in parquet_files if f.stem.lower().startswith(f"{symbol_lower}_")]

    if not dated_files:
        logger.warning(f"No dated futures files found in {symbol_dir}")
        return 0, None

    logger.info(f"Using {len(dated_files)} dated futures files (already in rupees)")

    dfs = []
    for file in dated_files:
        try:
            df = pd.read_parquet(file)
            
            # CRITICAL: Apply price unit detection and conversion
            price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in df.columns]
            if price_cols:
                # Check if prices need conversion (median-based detection)
                if detect_price_unit(df, price_column='close', expected_range=(10, 100000)):
                    logger.debug(f"Detected paise prices in {file.name}, converting to rupees")
                    df = convert_paise_to_rupees(df, price_cols)
                else:
                    # Ensure float64 type even if no conversion needed
                    for col in price_cols:
                        df[col] = df[col].astype('float64')
            
            # Handle mixed date formats
            if df['date'].dtype == 'object':
                # Try to parse dates with explicit formats to avoid warnings
                # Common formats: YYYYMMDD (int as string), YYYY-MM-DD, DD/MM/YYYY, etc.
                try:
                    # First check if all dates are already in YYYYMMDD format (as string)
                    if df['date'].str.len().eq(8).all() and df['date'].str.isdigit().all():
                        df['date'] = df['date'].astype(int)
                    else:
                        # Try parsing with explicit format first (most common: YYYYMMDD as string)
                        parsed_dates = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
                        # If that didn't work for all, try other formats for remaining NaN values
                        if parsed_dates.isna().any():
                            mask = parsed_dates.isna()
                            # Try YYYY-MM-DD format
                            parsed_dates[mask] = pd.to_datetime(df['date'][mask], format='%Y-%m-%d', errors='coerce')
                            # Try DD/MM/YYYY format
                            if parsed_dates.isna().any():
                                mask = parsed_dates.isna()
                                parsed_dates[mask] = pd.to_datetime(df['date'][mask], format='%d/%m/%Y', errors='coerce')
                            # Final fallback: let pandas infer but suppress warning
                            if parsed_dates.isna().any():
                                mask = parsed_dates.isna()
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    parsed_dates[mask] = pd.to_datetime(df['date'][mask], errors='coerce')
                        # Convert to YYYYMMDD integer format
                        df['date'] = parsed_dates.dt.strftime('%Y%m%d').astype(int)
                except Exception as e:
                    logger.warning(f"Error parsing dates in {file.name}: {e}, using fallback method")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y%m%d').astype(int)
            elif df['date'].dtype in ['int64', 'int32', 'int']:
                # Already in integer format (YYYYMMDD), ensure it's int64
                df['date'] = df['date'].astype('int64')
            # Ensure time is int
            if df['time'].dtype == 'object':
                df['time'] = df['time'].astype(int)
            elif df['time'].dtype not in ['int64', 'int32', 'int']:
                df['time'] = df['time'].astype('int64')
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Error reading {file}: {e}")
            continue
    
    if not dfs:
        return 0, None
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Convert to timestamp
    combined_df['timestamp'] = combined_df.apply(
        lambda row: yyyymmdd_seconds_to_datetime(row['date'], row['time']),
        axis=1
    )
    
    logger.info(f"Futures data range (before filtering): {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    
    # Filter by date range (account for IST->UTC conversion)
    start = pd.to_datetime(start_date) - pd.Timedelta(hours=6)
    end = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    combined_df = combined_df[(combined_df['timestamp'] >= start) & 
                               (combined_df['timestamp'] < end)]
    
    if combined_df.empty:
        logger.warning(f"No futures data found in date range {start_date} to {end_date}")
        return 0, None
    
    logger.info(f"Futures data range (after filtering): {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    logger.info(f"Filtered to {len(combined_df):,} records for date range {start_date} to {end_date}")
    
    # Prepare for BarDataWrangler (OHLCV only, NO OI!)
    # Ensure volume column exists
    if 'volume' not in combined_df.columns:
        combined_df['volume'] = 0.0
        logger.warning(f"Volume column not found in futures data, using default volume=0.0")
    
    bar_df = combined_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()

    # NOTE: Dated futures files (nifty_future_YYYYMMDD.parquet) are ALREADY in RUPEES
    # No conversion needed!
    
    # Ensure all price columns are float64 (required by Nautilus)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in bar_df.columns:
            bar_df[col] = bar_df[col].astype('float64')

    # Data quality fixes
    bar_df['volume'] = bar_df['volume'].clip(lower=0)  # Handle negative volumes
    
    # Fix invalid OHLC relationships (Nautilus validates: high >= close, low <= close)
    bar_df['high'] = bar_df[['high', 'close']].max(axis=1)
    bar_df['low'] = bar_df[['low', 'close']].min(axis=1)
    
    # CRITICAL: Deduplicate by timestamp before bar creation
    dedup_before = len(bar_df)
    bar_df = bar_df.drop_duplicates(subset=['timestamp'], keep='last')
    dedup_after = len(bar_df)
    if dedup_before > dedup_after:
        logger.info(f"Deduplicated {dedup_before - dedup_after} duplicate futures records (kept {dedup_after:,} unique records)")
    
    bar_df = bar_df.set_index('timestamp')
    bar_df = bar_df.sort_index()
    
    # Additional check: ensure no overlapping intervals (same timestamp)
    if bar_df.index.duplicated().any():
        logger.warning(f"Found {bar_df.index.duplicated().sum()} duplicate timestamps after deduplication, removing...")
        bar_df = bar_df[~bar_df.index.duplicated(keep='last')]
        logger.info(f"Final bar count: {len(bar_df):,}")
    
    # Determine Venue
    venue_str = get_venue(symbol)
    
    # Create FuturesContract (use proper Nautilus instrument type)
    # Symbol for continuous future: {SYMBOL}-I (e.g. NIFTY-I)
    future_symbol = f"{symbol.upper()}-I"
    
    try:
        instrument = create_futures_contract(
            symbol=future_symbol,  # -I for continuous futures
            expiry_date="continuous",  # Continuous contract
            underlying=symbol,
            venue=venue_str
        )
    except ValueError as e:
        # Skip symbol if lot size is not found (no default lot size)
        if "Lot size not found" in str(e):
            logger.error(f"Skipping {symbol} futures: {e}")
            return 0, {}
        else:
            raise  # Re-raise other ValueErrors
    
    bar_type = BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL")
    
    # Create bars
    wrangler = BarDataWrangler(bar_type, instrument)
    bars = wrangler.process(bar_df)
    
    # Write to catalog
    try:
        catalog.write_data([instrument])
    except AssertionError as e:
        if "Intervals are not disjoint" in str(e):
            logger.warning(f"Non-disjoint intervals in instrument write (may already exist), continuing...")
        else:
            raise
    
    try:
        catalog.write_data(bars, skip_disjoint_check=True)
        logger.info(f"✅ Wrote {len(bars):,} futures bars to catalog")
    except AssertionError as e:
        if "Intervals are not disjoint" in str(e):
            logger.error(f"Failed to write futures bars due to non-disjoint intervals. Try using --clean flag to clear output directory.")
            raise
        else:
            raise

    # Generate and write QuoteTicks for Greeks calculation
    quote_ticks = bars_to_quote_ticks(bars, instrument)
    try:
        catalog.write_data(quote_ticks, skip_disjoint_check=True)
        logger.info(f"✅ Wrote {len(quote_ticks):,} QuoteTicks to catalog")
    except AssertionError as e:
        if "Intervals are not disjoint" in str(e):
            logger.warning(f"Non-disjoint intervals in QuoteTicks write, skipping...")
        else:
            raise

    # Create FutureOI custom data (Arrow serialization registered)
    oi_data_list = []
    prev_oi = 0
    for idx, row in combined_df.iterrows():
        current_oi = int(row["oi"])
        coi = current_oi - prev_oi
        prev_oi = current_oi
        ts_ns = int(row["timestamp"].timestamp() * 1_000_000_000)
        
        oi_data = FutureOI(
            instrument_id=instrument.id,
            oi=current_oi,
            coi=coi,
            ts_event=ts_ns,
            ts_init=ts_ns
        )
        oi_data_list.append(oi_data)
    
    # Write FutureOI to catalog (Arrow registered)
    if oi_data_list:
        oi_data_list.sort(key=lambda x: x.ts_init)
        try:
            catalog.write_data(oi_data_list, skip_disjoint_check=True)
            logger.info(f"✅ Saved {len(oi_data_list):,} FutureOI records")
        except AssertionError as e:
            if "Intervals are not disjoint" in str(e):
                logger.warning(f"Non-disjoint intervals in FutureOI data, skipping write")
            else:
                raise
    
    logger.info(f"✅ {symbol} futures: Created {len(bars):,} bars + {len(quote_ticks):,} QuoteTicks")
    return len(bars), None  # No longer returning DataFrame

