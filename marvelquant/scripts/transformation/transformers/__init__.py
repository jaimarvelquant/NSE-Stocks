"""
Nautilus Data Transformation Modules.

This package contains modular transformers for converting raw market data
to NautilusTrader catalog format.

Modules:
    - common: Shared utilities (time/venue helpers, price conversion, logging)
    - index_transformer: Index bar transformation
    - futures_transformer: Futures bar transformation  
    - options_transformer: Options bar transformation with Greeks calculation
"""

from .common import (
    configure_logging,
    get_venue,
    get_instrument_id_string,
    yyyymmdd_seconds_to_datetime,
    detect_price_unit,
    convert_paise_to_rupees,
    bars_to_quote_ticks,
    time_aware_forward_fill,
    IST_OFFSET,
    VENUE_MAP,
    DEFAULT_VENUE,
    PROJECT_ROOT,
)

from .index_transformer import transform_index_bars
from .futures_transformer import transform_futures_bars
from .options_transformer import transform_options_bars

__all__ = [
    # Common utilities
    'configure_logging',
    'get_venue',
    'get_instrument_id_string',
    'yyyymmdd_seconds_to_datetime',
    'detect_price_unit',
    'convert_paise_to_rupees',
    'bars_to_quote_ticks',
    'time_aware_forward_fill',
    'IST_OFFSET',
    'VENUE_MAP',
    'DEFAULT_VENUE',
    'PROJECT_ROOT',
    # Transformer functions
    'transform_index_bars',
    'transform_futures_bars',
    'transform_options_bars',
]

