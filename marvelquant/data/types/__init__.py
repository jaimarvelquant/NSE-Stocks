"""
Custom Data Types for MarvelQuant Nautilus Integration.

Provides custom data types that extend Nautilus core Data class:
- OptionOI: Option open interest data
- FutureOI: Future open interest data

These types are stored separately from Bar data because Nautilus Bar class
does NOT have an open_interest field.

Usage:
    >>> from marvelquant.data.types import OptionOI, FutureOI
    >>> from nautilus_trader.model.identifiers import InstrumentId
    >>> 
    >>> # Option OI
    >>> option_oi = OptionOI(
    ...     instrument_id=InstrumentId.from_str("BANKNIFTY28OCT2548000CE.NSE"),
    ...     oi=1_500_000,
    ...     coi=50_000,
    ...     ts_event=1704177900000000000,
    ...     ts_init=1704177900000000000
    ... )
    >>> 
    >>> # Future OI
    >>> future_oi = FutureOI(
    ...     instrument_id=InstrumentId.from_str("BANKNIFTY-I.NSE"),
    ...     oi=2_500_000,
    ...     coi=100_000,
    ...     ts_event=1704177900000000000,
    ...     ts_init=1704177900000000000
    ... )
"""

from .option_oi import OptionOI
from .future_oi import FutureOI

__all__ = [
    "OptionOI",
    "FutureOI",
]
