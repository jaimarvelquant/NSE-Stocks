"""
Contract Metadata Generators for Nautilus Instruments.

Generates proper OptionsContract and FuturesContract metadata following
Nautilus standards for NSE instruments.

Handles:
- Symbol parsing (BANKNIFTY28OCT2548000CE → components)
- InstrumentId generation ({SYMBOL}.NSE format)
- Lot size mapping per underlying
- Contract metadata population

Usage:
    >>> from marvelquant.utils.contract_generators import create_options_contract
    >>> from datetime import date
    >>> 
    >>> contract = create_options_contract(
    ...     symbol_str="BANKNIFTY28OCT2548000CE",
    ...     strike=48000.0,
    ...     expiry=date(2024, 10, 28),
    ...     option_type="CALL",
    ...     underlying="BANKNIFTY"
    ... )
"""

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional
import pandas as pd
import pytz

from nautilus_trader.model.instruments import OptionContract, FuturesContract
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.objects import Price, Quantity, Currency
from nautilus_trader.model.enums import AssetClass, OptionKind

# Import lot size loader
try:
    from marvelquant.utils.lot_size_loader import get_lot_size
except ImportError:
    # Fallback if module not available
    def get_lot_size(symbol: str, csv_path: Optional[Path] = None, column_name: str = "November 2025") -> Optional[int]:
        return None


# NSE lot sizes per underlying (Latest SEBI/NSE Status as of Dec 2025)
# Enforcing CURRENT lot sizes for backtesting consistency across historical data.
# For stocks, lot sizes are loaded from CSV file (see lot_size_loader.py)
NSE_LOT_SIZES = {
    # Index Futures/Options
    "NIFTY": 75,          # Revised to 75 (Nov 2024)
    "BANKNIFTY": 30,      # Revised to 30 (Nov 2024)
    "FINNIFTY": 65,       # Revised to 65 (Nov 2024)
    "MIDCPNIFTY": 120,    # Revised to 120 (Nov 2024)
    "NIFTYNXT50": 25,     # Unchanged
    "SENSEX": 10,         # BSE Sensex
    "BANKEX": 15,         # BSE Bankex
    
    # Commodity Futures/Options (MCX)
    "CRUDEOIL": 100,      # 100 Barrels
    "NATURALGAS": 1250,   # 1250 mmBtu
    "GOLD": 100,          # 1 kg (Quoted per 10g, so 100 units)
    "SILVER": 30,         # 30 kg
    "COPPER": 2500,       # 2.5 MT
    "ZINC": 5000,         # 5 MT
    "LEAD": 5000,         # 5 MT
    "ALUMINIUM": 5000,    # 5 MT
    "NICKEL": 1500,       # 1.5 MT
}

# Global CSV path for lot sizes (set by main script)
_LOT_SIZE_CSV_PATH: Optional[Path] = None
_DEFAULT_LOT_SIZE: int = 1  # Default lot size when not found in CSV


def set_lot_size_csv_path(csv_path: Optional[Path]):
    """Set the CSV path for loading stock lot sizes."""
    global _LOT_SIZE_CSV_PATH
    if csv_path is not None:
        # Convert to absolute path to ensure it's found
        _LOT_SIZE_CSV_PATH = Path(csv_path).resolve()
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Set lot size CSV path to: {_LOT_SIZE_CSV_PATH}")
    else:
        _LOT_SIZE_CSV_PATH = None


def set_default_lot_size(default_lot_size: int):
    """Set the default lot size to use when not found in CSV."""
    global _DEFAULT_LOT_SIZE
    _DEFAULT_LOT_SIZE = default_lot_size
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Set default lot size to: {_DEFAULT_LOT_SIZE}")


def get_default_lot_size() -> int:
    """Get the default lot size."""
    return _DEFAULT_LOT_SIZE


def get_stock_lot_size(underlying: str) -> Optional[int]:
    """
    Get lot size for a stock from CSV file.
    
    Args:
        underlying: Stock symbol (e.g., "ABB", "SBIN")
    
    Returns:
        Lot size as integer, or None if not found
    """
    global _LOT_SIZE_CSV_PATH
    
    if _LOT_SIZE_CSV_PATH is not None:
        # Ensure CSV path exists
        csv_path_resolved = Path(_LOT_SIZE_CSV_PATH).resolve()
        if not csv_path_resolved.exists():
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"CSV path does not exist: {csv_path_resolved}")
            return None
        
        # Try "Nov 2025" first (matches "Lot Size (Nov 2025)"), fallback to "November 2025"
        # Use resolved path to ensure cache works correctly
        lot_size = get_lot_size(underlying, csv_path_resolved, "Nov 2025")
        if lot_size is None:
            lot_size = get_lot_size(underlying, csv_path_resolved, "November 2025")
        
        # If still None, try direct load as fallback (bypasses potential cache issues)
        if lot_size is None:
            import logging
            logger = logging.getLogger(__name__)
            from lot_size_loader import load_lot_sizes_from_csv
            try:
                # Direct load to bypass any cache issues
                direct_lot_sizes = load_lot_sizes_from_csv(csv_path_resolved, "Nov 2025")
                underlying_upper = underlying.upper()
                # Try multiple symbol variations to handle mismatches
                # Common variations: M_M ↔ M&M, BAJAJ_AUTO ↔ BAJAJ-AUTO
                symbol_variations = [
                    underlying_upper,  # Exact match
                    underlying_upper.replace("_", "&"),  # Replace underscore with ampersand (M_M -> M&M)
                    underlying_upper.replace("&", "_"),  # Replace ampersand with underscore (M&M -> M_M)
                    underlying_upper.replace("_", "-"),  # Replace underscore with hyphen (BAJAJ_AUTO -> BAJAJ-AUTO)
                    underlying_upper.replace("-", "_"),  # Replace hyphen with underscore (BAJAJ-AUTO -> BAJAJ_AUTO)
                    underlying_upper.replace("&", "-"),  # Replace ampersand with hyphen (M&M -> M-M)
                    underlying_upper.replace("-", "&"),  # Replace hyphen with ampersand (M-M -> M&M)
                ]
                # Remove duplicates while preserving order
                seen = set()
                unique_variations = []
                for var in symbol_variations:
                    if var not in seen:
                        seen.add(var)
                        unique_variations.append(var)
                symbol_variations = unique_variations
                for symbol_var in symbol_variations:
                    lot_size = direct_lot_sizes.get(symbol_var)
                    if lot_size is not None:
                        if symbol_var != underlying_upper:
                            logger.info(f"✅ Found lot size {lot_size} for {underlying} via direct CSV load (matched as {symbol_var})")
                        else:
                            logger.info(f"✅ Found lot size {lot_size} for {underlying} via direct CSV load")
                        break  # Found a match, exit loop
        
                if lot_size is None:
                    available_symbols = [s for s in direct_lot_sizes.keys() if underlying_upper in s or s in underlying_upper][:5]
                    logger.warning(f"Lot size not found for {underlying} in CSV. Available similar symbols: {available_symbols}")
            except Exception as e:
                logger.warning(f"Error loading CSV directly: {e}")
        
        if lot_size is not None:
            import logging
            logger = logging.getLogger(__name__)
            if not hasattr(get_stock_lot_size, '_logged'):
                logger.info(f"✅ Found lot size {lot_size} for {underlying} from CSV")
        
        return lot_size
    else:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"CSV path not set for lot size lookup. Underlying: {underlying}")
    return None

# Asset class mapping
ASSET_CLASS_MAP = {
    # Commodities
    "CRUDEOIL": AssetClass.COMMODITY,
    "NATURALGAS": AssetClass.COMMODITY,
    "GOLD": AssetClass.COMMODITY,
    "SILVER": AssetClass.COMMODITY,
    "COPPER": AssetClass.COMMODITY,
    "ZINC": AssetClass.COMMODITY,
    "LEAD": AssetClass.COMMODITY,
    "ALUMINIUM": AssetClass.COMMODITY,
    "NICKEL": AssetClass.COMMODITY,
    
    # Everything else defaults to EQUITY (indices, stocks)
    # NIFTY, BANKNIFTY, FINNIFTY, etc.
}


def create_options_contract(
    symbol: str,
    strike: float,
    expiry: date,
    option_kind: str,
    underlying: str,
    lot_size: int = None,
    venue: str = "NSE"
) -> OptionContract:
    """
    Create Nautilus OptionContract following official pattern.
    
    Based on: nautilus_trader/test_kit/providers.py::aapl_option()
    
    Args:
        symbol: NSE option symbol (without .NSE suffix)
        strike: Strike price
        expiry: Expiry date
        option_kind: "CALL" or "PUT"
        underlying: Underlying symbol (NIFTY, BANKNIFTY, etc.)
        lot_size: Lot size (optional, auto-detected)
        venue: Exchange venue (default: NSE)
    
    Returns:
        OptionContract instance
    """
    # Determine lot size
    if lot_size is None:
        import logging
        logger = logging.getLogger(__name__)
        # First try CSV for stocks
        csv_lot_size = get_stock_lot_size(underlying)
        if csv_lot_size is not None:
            lot_size = csv_lot_size
            logger.info(f"✅ Using lot size {lot_size} from CSV for underlying {underlying}")
        else:
            # Fallback to hardcoded lot sizes for indices only
            # For stocks, we must have lot size in CSV - skip if not found
            if underlying.upper() in NSE_LOT_SIZES:
                lot_size = NSE_LOT_SIZES[underlying.upper()]
                logger.info(f"Using hardcoded lot size {lot_size} for index {underlying}")
            else:
                # Stock symbol not in hardcoded list and not in CSV - skip it
                logger.error(f"❌ Lot size not found for {underlying} in CSV. CSV path: {_LOT_SIZE_CSV_PATH}. Skipping symbol.")
                raise ValueError(f"Lot size not found for {underlying}. Symbol must have lot size in CSV or be a known index.")
    
    # Determine asset class (Commodity vs Equity)
    asset_class = ASSET_CLASS_MAP.get(underlying.upper(), AssetClass.EQUITY)
    
    # Create InstrumentId
    instrument_id = InstrumentId(
        symbol=Symbol(symbol),
        venue=Venue(venue)
    )
    
    # Parse option kind
    kind = OptionKind.CALL if option_kind.upper() in ["CALL", "CE"] else OptionKind.PUT
    
    # Convert expiry to UTC timestamp (nanoseconds)
    expiry_utc = pd.Timestamp(expiry, tz=pytz.utc)

    # Set activation to 30 days before expiry (options should be tradeable before expiry!)
    activation_utc = expiry_utc - pd.Timedelta(days=30)

    # Create OptionContract (following Nautilus test provider pattern)
    # Use SPOT INDEX as underlying for NSE index options Greeks calculation
    # NSE index options (NIFTY, BANKNIFTY) reference spot index, NOT futures
    # Futures price includes carry cost (interest - dividend), options reference spot
    # See: docs/bmad/ENTERPRISE_SOLUTION_OPTIONS_GREEKS_AND_DATA_ARCHITECTURE.md
    underlying_spot = f"{underlying}-INDEX" if underlying.upper() in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "BANKEX"] else underlying

    # Log lot size being used
    import logging
    logger = logging.getLogger(__name__)
    if lot_size != 1 or underlying.upper() in NSE_LOT_SIZES:
        logger.info(f"Creating OptionContract for {symbol} (underlying={underlying}) with lot_size={lot_size}")
    
    contract = OptionContract(
        instrument_id=instrument_id,
        raw_symbol=Symbol(symbol),
        asset_class=asset_class,  # Proper classification
        exchange=venue,  # Exchange as string from venue argument
        underlying=underlying_spot,  # Use spot index for Greeks calculation
        option_kind=kind,
        activation_ns=activation_utc.value,  # 30 days before expiry
        expiration_ns=expiry_utc.value,
        strike_price=Price.from_str(f"{strike:.2f}"),
        currency=Currency.from_str("INR"),
        price_precision=2,
        price_increment=Price.from_str("0.05"),
        multiplier=Quantity.from_int(1),  # Multiplier 1 as prices are per unit (but lot_size defines contract size)
        lot_size=Quantity.from_int(lot_size),
        ts_event=0,
        ts_init=0,
    )
    
    # Verify lot_size was set correctly
    if hasattr(contract, 'lot_size'):
        actual_lot_size = contract.lot_size
        if actual_lot_size != Quantity.from_int(lot_size):
            logger.warning(f"Lot size mismatch! Expected {lot_size}, got {actual_lot_size} for {symbol}")
    else:
        logger.warning(f"OptionContract does not have lot_size attribute for {symbol}")
    
    return contract


def create_futures_contract(
    symbol: str,
    expiry_date: str | date,
    underlying: str,
    lot_size: int = None,
    venue: str = "NSE"
) -> FuturesContract:
    """
    Create Nautilus FuturesContract following official pattern.
    
    Based on: nautilus_trader/test_kit/providers.py::es_future()
    
    Args:
        symbol: Futures symbol (e.g., "NIFTY-I" or "NIFTY28MAR24")
        expiry_date: Expiry date or "continuous" for continuous contract
        underlying: Underlying symbol (NIFTY, BANKNIFTY, etc.)
        lot_size: Lot size (optional, auto-detected)
        venue: Exchange venue (default: NSE)
    
    Returns:
        FuturesContract instance
    """
    # Determine lot size
    if lot_size is None:
        # First try CSV for stocks
        csv_lot_size = get_stock_lot_size(underlying)
        if csv_lot_size is not None:
            lot_size = csv_lot_size
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Using lot size {lot_size} from CSV for underlying {underlying}")
        else:
            # Fallback to hardcoded lot sizes for indices only
            # For stocks, we must have lot size in CSV - skip if not found
            if underlying.upper() in NSE_LOT_SIZES:
                lot_size = NSE_LOT_SIZES[underlying.upper()]
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Using hardcoded lot size {lot_size} for index {underlying}")
            else:
                # Stock symbol not in hardcoded list and not in CSV - skip it
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"❌ Lot size not found for {underlying} in CSV. CSV path: {_LOT_SIZE_CSV_PATH}. Skipping symbol.")
                raise ValueError(f"Lot size not found for {underlying}. Symbol must have lot size in CSV or be a known index.")
    
    # Determine asset class (Commodity vs Equity)
    asset_class = ASSET_CLASS_MAP.get(underlying.upper(), AssetClass.EQUITY)
    
    # Create InstrumentId
    instrument_id = InstrumentId(
        symbol=Symbol(symbol),
        venue=Venue(venue)
    )
    
    # Handle expiry timestamp
    if isinstance(expiry_date, str) and expiry_date == "continuous":
        # For continuous contracts, use far future date
        expiry_utc = pd.Timestamp("2099-12-31", tz=pytz.utc)
        activation_utc = pd.Timestamp("2020-01-01", tz=pytz.utc)  # Far past for continuous
    elif isinstance(expiry_date, date):
        expiry_utc = pd.Timestamp(expiry_date, tz=pytz.utc)
        activation_utc = expiry_utc - pd.Timedelta(days=90)  # 90 days before expiry for futures
    else:
        expiry_utc = pd.Timestamp(expiry_date, tz=pytz.utc)
        activation_utc = expiry_utc - pd.Timedelta(days=90)  # 90 days before expiry for futures

    # Create FuturesContract (following Nautilus test provider pattern)
    contract = FuturesContract(
        instrument_id=instrument_id,
        raw_symbol=Symbol(symbol),
        asset_class=asset_class,  # Proper classification
        exchange=venue,  # Exchange as string from venue argument
        underlying=underlying,
        activation_ns=activation_utc.value,  # Set before expiry for trading
        expiration_ns=expiry_utc.value,
        currency=Currency.from_str("INR"),
        price_precision=2,
        price_increment=Price.from_str("0.05"),
        multiplier=Quantity.from_int(1),  # Keep 1 for index futures unless specific mapping
        lot_size=Quantity.from_int(lot_size),
        ts_event=0,
        ts_init=0,
    )
    
    return contract


def parse_nse_option_symbol(symbol: str) -> dict:
    """
    Parse NSE option symbol into components.
    
    Format: {UNDERLYING}{DDMMMYY}{STRIKE}{CE|PE}
    Examples: 
        - Index: BANKNIFTY28OCT2548000CE (5-digit integer strike)
        - Stock: ABB25NOV255000CE (4-digit integer strike) or ABB25NOV25500CE (3-digit integer strike)
        - Stock with decimal: WIPRO30DEC24222.5CE (decimal strike: 222.5)
    
    Args:
        symbol: NSE option symbol
    
    Returns:
        Dictionary with components:
        - underlying: Underlying symbol
        - expiry: Expiry date
        - strike: Strike price (float, can be integer or decimal)
        - option_type: "CALL" or "PUT"
    
    Example:
        >>> parse_nse_option_symbol("BANKNIFTY28OCT2548000CE")
        {
            'underlying': 'BANKNIFTY',
            'expiry': date(2024, 10, 28),
            'strike': 48000.0,
            'option_type': 'CALL'
        }
        >>> parse_nse_option_symbol("ABB25NOV255000CE")
        {
            'underlying': 'ABB',
            'expiry': date(2025, 11, 25),
            'strike': 5000.0,
            'option_type': 'CALL'
        }
        >>> parse_nse_option_symbol("WIPRO30DEC24222.5CE")
        {
            'underlying': 'WIPRO',
            'expiry': date(2024, 12, 30),
            'strike': 222.5,
            'option_type': 'CALL'
        }
    """
    import re
    from datetime import datetime
    
    # Extract option type (last 2 chars: CE or PE)
    option_type_code = symbol[-2:]
    if option_type_code not in ['CE', 'PE']:
        raise ValueError(f"Invalid option type code: {option_type_code}. Expected CE or PE")
    
    option_type = "CALL" if option_type_code == "CE" else "PUT"
    
    # Remove option type from symbol
    symbol_without_type = symbol[:-2]
    
    # Find the date pattern (DDMMMYY format: e.g., "25NOV25", "28OCT25")
    # Date is always 7 characters: 2 digits + 3 letter month + 2 digit year
    date_pattern = r'(\d{2}[A-Z]{3}\d{2})'
    date_match = re.search(date_pattern, symbol_without_type)
    
    if not date_match:
        raise ValueError(f"Could not find date pattern in symbol: {symbol}")
    
    date_str = date_match.group(1)
    date_start_idx = date_match.start()
    date_end_idx = date_match.end()
    
    # Parse expiry date
    try:
        expiry = datetime.strptime(date_str, "%d%b%y").date()
    except ValueError:
        # Try alternative format if needed
        raise ValueError(f"Could not parse date '{date_str}' from symbol: {symbol}")
    
    # Extract strike (numeric value between date and option type)
    # Can be integer (e.g., "2225") or decimal (e.g., "222.5")
    strike_str = symbol_without_type[date_end_idx:]
    if not strike_str:
        raise ValueError(f"Could not extract strike from symbol: {symbol} (strike part is empty)")
    
    # Validate and parse strike - can be integer or decimal
    # Use regex to match valid numeric format (digits with optional decimal point and digits)
    strike_pattern = r'^\d+(\.\d+)?$'
    if not re.match(strike_pattern, strike_str):
        raise ValueError(f"Invalid strike format in symbol: {symbol} (strike part: '{strike_str}'). Expected numeric value (integer or decimal)")
    
    try:
        strike = float(strike_str)
    except ValueError:
        raise ValueError(f"Could not convert strike to float from symbol: {symbol} (strike part: '{strike_str}')")
    
    # Extract underlying (everything before date)
    underlying = symbol_without_type[:date_start_idx]
    
    if not underlying:
        raise ValueError(f"Could not extract underlying from symbol: {symbol}")
    
    return {
        'underlying': underlying,
        'expiry': expiry,
        'strike': strike,
        'option_type': option_type
    }
