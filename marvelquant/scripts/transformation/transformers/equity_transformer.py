"""
Equity Bar Transformer.

Transforms equity/stock data (NSE stocks) to Nautilus Bar format.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.data import BarType
from nautilus_trader.model.objects import Price, Quantity, Currency

# Try to import EquityInstrument or other equity instrument types
# Nautilus Trader may use different class names for equity instruments
# Import Equity instrument - Nautilus uses 'Equity' class for equity instruments
# This ensures data goes to 'equity' folder, not 'index_instrument' folder
try:
    from nautilus_trader.model.instruments import Equity as EquityInstrument
except ImportError:
    try:
        from nautilus_trader.model.instruments import EquityInstrument
    except ImportError:
        try:
            from nautilus_trader.model.instruments import Security as EquityInstrument
        except ImportError:
            # Fallback: Use IndexInstrument (will create 'index_instrument' folder)
            from nautilus_trader.model.instruments import IndexInstrument
            EquityInstrument = IndexInstrument

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

# Import lot size loader - handle both absolute and direct imports
try:
    from marvelquant.utils.lot_size_loader import get_lot_size
    from marvelquant.utils.contract_generators import get_stock_lot_size
except ImportError:
    # Fallback: try direct imports
    try:
        import sys
        from pathlib import Path
        # Add utils to path if not already there
        utils_path = Path(__file__).parent.parent.parent.parent / "utils"
        if str(utils_path) not in sys.path:
            sys.path.insert(0, str(utils_path))
        from lot_size_loader import get_lot_size
        from contract_generators import get_stock_lot_size
    except ImportError:
        # Final fallback: define stub functions
        def get_lot_size(symbol: str, csv_path: Optional[Path] = None, column_name: str = "Nov 2025") -> Optional[int]:
            return None
        def get_stock_lot_size(underlying: str) -> Optional[int]:
            return None

logger = logging.getLogger(__name__)


def transform_equity_bars(
    input_dir: Path,
    catalog: ParquetDataCatalog,
    symbol: str,
    start_date: str,
    end_date: str,
    csv_path: Optional[Path] = None
) -> int:
    """
    Transform equity/stock data to Nautilus Bar format.
    
    Args:
        input_dir: Directory containing raw parquet files
        catalog: Nautilus ParquetDataCatalog instance
        symbol: Stock symbol (e.g., "ABB", "SBIN")
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        csv_path: Optional path to CSV file containing lot sizes
    
    Returns:
        Number of bars created
    """
    logger.info(f"Transforming {symbol} equity bars...")

    # Find all parquet files for this symbol
    # Try both "equity" and "cash" directories
    # Handle both structures: cash/{symbol}/ and cash/YYYY/MM/{symbol}/
    symbol_lower = symbol.lower()
    
    # Try flat structure first: equity/{symbol}/ or cash/{symbol}/
    symbol_dir = input_dir / "equity" / symbol_lower
    if not symbol_dir.exists():
        symbol_dir = input_dir / "cash" / symbol_lower
    
    parquet_files = []
    
    # If flat structure exists, use it
    if symbol_dir.exists():
        parquet_files = list(symbol_dir.rglob("*.parquet"))
    else:
        # Try nested structure: cash/YYYY/MM/{symbol}/
        cash_dir = input_dir / "cash"
        if cash_dir.exists():
            # Use rglob to find all parquet files for this symbol in nested structure
            # Pattern: cash/YYYY/MM/{symbol}/*.parquet
            parquet_files = list(cash_dir.rglob(f"{symbol_lower}/*.parquet"))
            if parquet_files:
                logger.info(f"Found {len(parquet_files)} files in nested structure: cash/YYYY/MM/{symbol_lower}/")
    
    if not parquet_files:
        logger.warning(f"No parquet files found for symbol {symbol} in equity/cash directories")
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
                if detect_price_unit(df, price_column='close', expected_range=(10, 100000)):
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
    
    # Check for required columns
    required_price_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_price_cols if col not in combined_df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}. Available columns: {list(combined_df.columns)}")
        return 0
    
    # Handle missing volume column - use 0 as default if not present
    if 'volume' not in combined_df.columns:
        logger.warning(f"Volume column not found in data, using default volume=0.0")
        combined_df['volume'] = 0.0
    
    # OFFICIAL PATTERN: Prepare DataFrame for BarDataWrangler
    # Required: columns ['open', 'high', 'low', 'close', 'volume'] with 'timestamp' as INDEX
    bar_df = combined_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()

    # Cleanse numeric columns - convert to numeric and handle NaN values
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    price_cols = ['open', 'high', 'low', 'close']
    
    # Convert all numeric columns to numeric (errors='coerce' converts invalid to NaN)
    for col in numeric_cols:
        bar_df[col] = pd.to_numeric(bar_df[col], errors='coerce')
    
    # Drop any rows with NaN price values (OHLC must be valid)
    invalid_price_mask = bar_df[price_cols].isna().any(axis=1)
    invalid_count = int(invalid_price_mask.sum())
    if invalid_count:
        logger.warning(f"Dropping {invalid_count} rows with NaN OHLC values")
        bar_df = bar_df[~invalid_price_mask]
    
    if bar_df.empty:
        logger.error("All rows dropped due to invalid price data")
        return 0
    
    # Fill NaN volume with 0.0 (volume can be 0, but not NaN)
    bar_df['volume'] = bar_df['volume'].fillna(0.0)
    
    # Ensure all numeric columns are float64 (required by Nautilus)
    for col in numeric_cols:
        bar_df[col] = bar_df[col].astype('float64')
    
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
            f"Dropping {invalid_ohlc_count} rows with invalid OHLC relationships (low > open/close or high < open/close)"
        )
        bar_df = bar_df[~invalid_ohlc_mask]
    
    if bar_df.empty:
        logger.error("All rows dropped due to invalid OHLC data")
        return 0
    
    # Ensure volume is non-negative
    bar_df['volume'] = bar_df['volume'].clip(lower=0.0)
    
    # CRITICAL: Deduplicate by timestamp (keep last value) before setting index
    bar_df = bar_df.drop_duplicates(subset=['timestamp'], keep='last')
    
    bar_df = bar_df.set_index('timestamp')  # CRITICAL: Set timestamp as index!
    bar_df = bar_df.sort_index()  # Sort by timestamp
    
    # Determine Venue
    venue_str = get_venue(symbol)
    venue = Venue(venue_str)
    
    # Create InstrumentId
    # Use helper to ensure consistent naming (e.g. ABB.NSE)
    instrument_id_str = get_instrument_id_string(symbol, "equity")
    instrument_id = InstrumentId.from_str(instrument_id_str)
    
    # Get lot size from CSV only (no default fallback for equity)
    lot_size = None
    if csv_path is not None:
        # Try "Nov 2025" first (matches "Lot Size (Nov 2025)"), fallback to "November 2025")
        lot_size = get_lot_size(symbol, csv_path, "Nov 2025")
        if lot_size is None:
            lot_size = get_lot_size(symbol, csv_path, "November 2025")
        if lot_size is not None:
            logger.info(f"Using lot size {lot_size} from CSV for {symbol}")
        else:
            logger.error(f"Lot size not found in CSV for {symbol}. Equity symbols must have lot size in CSV. Skipping {symbol}.")
            return 0
    else:
        # Try using the global CSV path from contract_generators
        lot_size = get_stock_lot_size(symbol)
        if lot_size is not None:
            logger.info(f"Using lot size {lot_size} from CSV for {symbol}")
        else:
            logger.error(f"Lot size not found in CSV for {symbol}. Equity symbols must have lot size in CSV. Skipping {symbol}.")
            return 0
    
    if lot_size is None or lot_size <= 0:
        logger.error(f"Invalid lot size for {symbol}. Must be a positive integer. Skipping {symbol}.")
        return 0
    
    # Use EquityInstrument for equity/stock instruments
    # Note: Equity class may have different parameters than IndexInstrument
    try:
        # Try with size_precision and size_increment (for IndexInstrument fallback)
        instrument = EquityInstrument(
            instrument_id=instrument_id,
            raw_symbol=Symbol(symbol),
            currency=Currency.from_str("INR"),
            price_precision=2,
            price_increment=Price(0.05, 2),
            size_precision=0,
            size_increment=Quantity.from_int(1),
            lot_size=Quantity.from_int(lot_size),
            ts_event=0,
            ts_init=0,
        )
    except TypeError:
        # Equity class doesn't accept size_precision/size_increment, use without them
        instrument = EquityInstrument(
            instrument_id=instrument_id,
            raw_symbol=Symbol(symbol),
            currency=Currency.from_str("INR"),
            price_precision=2,
            price_increment=Price(0.05, 2),
            lot_size=Quantity.from_int(lot_size),
            ts_event=0,
            ts_init=0,
        )
    
    # Create bar type
    bar_type = BarType.from_str(f"{instrument_id}-1-MINUTE-LAST-EXTERNAL")
    
    # OFFICIAL PATTERN: Use BarDataWrangler
    wrangler = BarDataWrangler(bar_type, instrument)
    bars = wrangler.process(
        data=bar_df,
        default_volume=0.0,
        ts_init_delta=0
    )
    
    # OFFICIAL PATTERN: Write to catalog
    catalog.write_data([instrument])  # Write instrument first
    catalog.write_data(bars, skip_disjoint_check=True)  # Skip check for overlapping data

    # Generate and write QuoteTicks for Greeks calculation (if needed)
    quote_ticks = bars_to_quote_ticks(bars, instrument)
    catalog.write_data(quote_ticks, skip_disjoint_check=True)
    logger.info(f"✅ {symbol}: Created {len(bars):,} bars + {len(quote_ticks):,} QuoteTicks")

    return len(bars)

