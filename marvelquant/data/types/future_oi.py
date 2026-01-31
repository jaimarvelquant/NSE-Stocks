"""
Custom data class for futures Open Interest (OI) data.

This class stores Open Interest and Change in OI for futures contracts
for use in NautilusTrader backtesting and live trading.

Uses NautilusTrader's @customdataclass decorator for automatic serialization support.
"""

from nautilus_trader.core.data import Data
from nautilus_trader.core.datetime import unix_nanos_to_iso8601
from nautilus_trader.model.custom import customdataclass
from nautilus_trader.model.identifiers import InstrumentId


@customdataclass
class FutureOI(Data):
    """
    Custom data class for futures Open Interest.
    
    This class uses NautilusTrader's @customdataclass decorator which automatically:
    - Creates to_dict/from_dict methods
    - Creates to_arrow/from_arrow methods for Parquet serialization
    - Registers Arrow serialization schema
    - Handles ts_event and ts_init properties
    - Adds 'type' column with class name
    
    Attributes
    ----------
    instrument_id : InstrumentId
        The instrument ID for the futures contract
    oi : int
        Current open interest (number of contracts)
    coi : int
        Change in open interest from previous period
    ts_event : int
        Unix timestamp (nanoseconds) when OI was recorded
    ts_init : int
        Unix timestamp (nanoseconds) when data was initialized
    
    Examples
    --------
    >>> from nautilus_trader.model.identifiers import InstrumentId
    >>> instrument_id = InstrumentId.from_str("NIFTY-I.NSE")
    >>> future_oi = FutureOI(
    ...     ts_event=1704067200000000000,
    ...     ts_init=1704067200000000000,
    ...     instrument_id=instrument_id,
    ...     oi=1500000,
    ...     coi=25000,
    ... )
    >>> print(future_oi)
    FutureOI[NIFTY-I.NSE]: OI=1,500,000, COI=+25,000
    """
    
    # Field definitions with type annotations (required for @customdataclass)
    # The decorator will use these to create the PyArrow schema
    instrument_id: InstrumentId = InstrumentId.from_str("DEFAULT.VENUE")
    oi: int = 0
    coi: int = 0
    
    def __repr__(self) -> str:
        """Return detailed string representation of FutureOI."""
        return (
            f"FutureOI("
            f"instrument_id={self.instrument_id}, "
            f"oi={self.oi:,}, "
            f"coi={self.coi:+,}, "
            f"ts_event={unix_nanos_to_iso8601(self.ts_event)}, "
            f"ts_init={unix_nanos_to_iso8601(self.ts_init)})"
        )
    
    def __str__(self) -> str:
        """Return user-friendly string representation."""
        return (
            f"FutureOI[{self.instrument_id}]: "
            f"OI={self.oi:,}, "
            f"COI={self.coi:+,}"
        )
