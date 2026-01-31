"""
Custom data class for options Open Interest (OI) data.

This class stores Open Interest and Change in OI for option contracts
for use in NautilusTrader backtesting and live trading.

Uses NautilusTrader's @customdataclass decorator for automatic serialization support.
"""

from nautilus_trader.core.data import Data
from nautilus_trader.core.datetime import unix_nanos_to_iso8601
from nautilus_trader.model.custom import customdataclass
from nautilus_trader.model.identifiers import InstrumentId


@customdataclass
class OptionOI(Data):
    """
    Custom data class for options Open Interest.
    
    This class uses NautilusTrader's @customdataclass decorator which automatically:
    - Creates to_dict/from_dict methods
    - Creates to_arrow/from_arrow methods for Parquet serialization
    - Registers Arrow serialization schema
    - Handles ts_event and ts_init properties
    - Adds 'type' column with class name
    
    Attributes
    ----------
    instrument_id : InstrumentId
        The instrument ID for the option contract
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
    >>> instrument_id = InstrumentId.from_str("NIFTY01FEB2419500CE.NSE")
    >>> option_oi = OptionOI(
    ...     ts_event=1704067200000000000,
    ...     ts_init=1704067200000000000,
    ...     instrument_id=instrument_id,
    ...     oi=150000,
    ...     coi=2500,
    ... )
    >>> print(option_oi)
    OptionOI[NIFTY01FEB2419500CE.NSE]: OI=150,000, COI=+2,500
    """
    
    # Field definitions with type annotations (required for @customdataclass)
    # The decorator will use these to create the PyArrow schema
    instrument_id: InstrumentId = InstrumentId.from_str("DEFAULT.VENUE")
    oi: int = 0
    coi: int = 0
    
    def __repr__(self) -> str:
        """Return detailed string representation of OptionOI."""
        return (
            f"OptionOI("
            f"instrument_id={self.instrument_id}, "
            f"oi={self.oi:,}, "
            f"coi={self.coi:+,}, "
            f"ts_event={unix_nanos_to_iso8601(self.ts_event)}, "
            f"ts_init={unix_nanos_to_iso8601(self.ts_init)})"
        )
    
    def __str__(self) -> str:
        """Return user-friendly string representation."""
        return (
            f"OptionOI[{self.instrument_id}]: "
            f"OI={self.oi:,}, "
            f"COI={self.coi:+,}"
        )
