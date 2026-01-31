"""
Lot Size Loader from CSV.

Loads lot sizes for NSE stocks from a CSV file.
The CSV should contain stock names and lot sizes in a "November 2025" column.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Global cache for lot sizes
_lot_size_cache: Optional[Dict[str, int]] = None
_csv_path: Optional[Path] = None


def load_lot_sizes_from_csv(csv_path: Path, column_name: str = "November 2025") -> Dict[str, int]:
    """
    Load lot sizes from CSV file.
    
    Args:
        csv_path: Path to CSV file containing stock names and lot sizes
        column_name: Name of the column containing lot sizes (default: "November 2025")
                     Can also be a partial match like "Nov 2025" or "November"
    
    Returns:
        Dictionary mapping stock symbol (uppercase) to lot size
    """
    global _lot_size_cache, _csv_path
    
    # Return cached data if same file is requested (compare resolved paths)
    csv_path_resolved = Path(csv_path).resolve()
    if _lot_size_cache is not None and _csv_path is not None:
        _csv_path_resolved = Path(_csv_path).resolve()
        if str(_csv_path_resolved) == str(csv_path_resolved):
            logger.debug(f"Using cached lot sizes from {csv_path_resolved}")
            return _lot_size_cache
        else:
            logger.debug(f"Cache path mismatch: cached={_csv_path_resolved}, requested={csv_path_resolved}")
    
    if not csv_path.exists():
        logger.warning(f"CSV file not found: {csv_path}")
        return {}
    
    try:
        df = pd.read_csv(csv_path)
        
        # Find the stock name column (could be "Symbol", "Stock", "Name", etc.)
        stock_column = None
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['symbol', 'stock', 'name', 'instrument']):
                stock_column = col
                break
        
        if stock_column is None:
            # Try first column as stock name
            stock_column = df.columns[0]
            logger.info(f"Using first column '{stock_column}' as stock name column")
        
        # Find lot size column - try exact match first, then partial match
        lot_size_column = None
        column_name_lower = column_name.lower()
        
        # Try exact match
        if column_name in df.columns:
            lot_size_column = column_name
        else:
            # Try partial match (e.g., "November 2025" matches "Lot Size (Nov 2025)")
            for col in df.columns:
                col_lower = col.lower()
                # Check if column name contains the search term or vice versa
                if (column_name_lower in col_lower or 
                    col_lower in column_name_lower or
                    any(term in col_lower for term in column_name_lower.split() if len(term) > 2)):
                    # Prefer columns with "lot" and "size" keywords
                    if 'lot' in col_lower and 'size' in col_lower:
                        lot_size_column = col
                        break
            
            # If still not found, try any column with "nov" and "2025"
            if lot_size_column is None:
                for col in df.columns:
                    col_lower = col.lower()
                    if 'nov' in col_lower and '2025' in col_lower:
                        lot_size_column = col
                        break
        
        if lot_size_column is None:
            logger.error(f"Lot size column matching '{column_name}' not found in CSV. Available columns: {list(df.columns)}")
            return {}
        
        logger.info(f"Using column '{lot_size_column}' for lot sizes")
        
        # Create mapping: stock symbol -> lot size
        lot_sizes = {}
        for _, row in df.iterrows():
            stock_name = str(row[stock_column]).strip().upper()
            lot_size = row[lot_size_column]
            
            # Handle NaN or empty values
            if pd.isna(lot_size) or lot_size == '':
                continue
            
            # Convert to int
            try:
                lot_size_int = int(float(lot_size))
                if lot_size_int > 0:
                    lot_sizes[stock_name] = lot_size_int
            except (ValueError, TypeError):
                logger.warning(f"Invalid lot size for {stock_name}: {lot_size}")
                continue
        
        logger.info(f"Loaded {len(lot_sizes)} lot sizes from {csv_path}")
        
        # Cache the results (store resolved path for comparison)
        _lot_size_cache = lot_sizes
        _csv_path = csv_path_resolved
        
        return lot_sizes
    
    except Exception as e:
        logger.error(f"Error loading lot sizes from CSV {csv_path}: {e}", exc_info=True)
        return {}


def get_lot_size(symbol: str, csv_path: Optional[Path] = None, column_name: str = "Nov 2025") -> Optional[int]:
    """
    Get lot size for a symbol from CSV.
    
    Handles symbol normalization to match variations like:
    - M_M (underscore) vs M&M (ampersand)
    - Case-insensitive matching
    
    Args:
        symbol: Stock symbol (e.g., "ABB", "SBIN", "M_M")
        csv_path: Optional path to CSV file (if not provided, uses cached data)
        column_name: Name of the column containing lot sizes
    
    Returns:
        Lot size as integer, or None if not found
    """
    symbol_upper = symbol.upper()
    
    # Try multiple symbol variations to handle mismatches
    # Common variations: M_M ↔ M&M, BAJAJ_AUTO ↔ BAJAJ-AUTO
    symbol_variations = [
        symbol_upper,  # Exact match
        symbol_upper.replace("_", "&"),  # Replace underscore with ampersand (M_M -> M&M)
        symbol_upper.replace("&", "_"),  # Replace ampersand with underscore (M&M -> M_M)
        symbol_upper.replace("_", "-"),  # Replace underscore with hyphen (BAJAJ_AUTO -> BAJAJ-AUTO)
        symbol_upper.replace("-", "_"),  # Replace hyphen with underscore (BAJAJ-AUTO -> BAJAJ_AUTO)
        symbol_upper.replace("&", "-"),  # Replace ampersand with hyphen (M&M -> M-M)
        symbol_upper.replace("-", "&"),  # Replace hyphen with ampersand (M-M -> M&M)
    ]
    # Remove duplicates while preserving order
    seen = set()
    unique_variations = []
    for var in symbol_variations:
        if var not in seen:
            seen.add(var)
            unique_variations.append(var)
    symbol_variations = unique_variations
    
    if csv_path is not None:
        lot_sizes = load_lot_sizes_from_csv(csv_path, column_name)
        # Try each variation
        for symbol_var in symbol_variations:
            result = lot_sizes.get(symbol_var)
            if result is not None:
                if symbol_var != symbol_upper:
                    logger.debug(f"Found {symbol_upper} in CSV as {symbol_var} with lot size {result}")
                else:
                    logger.debug(f"Found {symbol_upper} in CSV with lot size {result}")
                return result
        logger.debug(f"{symbol_upper} not found in CSV. Available symbols: {list(lot_sizes.keys())[:10]}...")
        return None
    elif _lot_size_cache is not None:
        lot_sizes = _lot_size_cache
        # Try each variation
        for symbol_var in symbol_variations:
            result = lot_sizes.get(symbol_var)
            if result is not None:
                if symbol_var != symbol_upper:
                    logger.debug(f"Found {symbol_upper} in cache as {symbol_var} with lot size {result}")
                else:
                    logger.debug(f"Found {symbol_upper} in cache with lot size {result}")
                return result
        logger.debug(f"{symbol_upper} not found in cache. Available symbols: {list(lot_sizes.keys())[:10]}...")
        return None
    else:
        logger.warning("No CSV file loaded. Call load_lot_sizes_from_csv() first or provide csv_path.")
        return None


def clear_cache():
    """Clear the cached lot sizes."""
    global _lot_size_cache, _csv_path
    _lot_size_cache = None
    _csv_path = None

