"""
Custom data class for option Greeks and implied volatility.

This class stores option Greeks (delta, gamma, theta, vega, rho) and implied volatility
for use in NautilusTrader backtesting and live trading.

Uses NautilusTrader's @customdataclass decorator for automatic serialization support.
"""

from nautilus_trader.core.data import Data
from nautilus_trader.core.datetime import unix_nanos_to_iso8601
from nautilus_trader.model.custom import customdataclass
from nautilus_trader.model.identifiers import InstrumentId


@customdataclass
class OptionGreeks(Data):
    """
    Custom data class for option Greeks and implied volatility.

    This class uses NautilusTrader's @customdataclass decorator which automatically:
    - Creates to_dict/from_dict methods
    - Creates to_arrow/from_arrow methods for Parquet serialization
    - Registers Arrow serialization schema
    - Handles ts_event and ts_init properties

    Greeks Calculation Methodology
    --------------------------------
    All Greeks are calculated using the Black-Scholes analytical formulas:
    - Method: Brent's root-finding algorithm for implied volatility
    - Risk-free rate: India 91-day T-Bill rates from RBI (annualized)
    - Dividend yield: Zero (not applicable for index options)
    - IV bounds: [0.01%, 500%]

    Units and Conventions
    ---------------------
    iv : Decimal format (0.15 = 15% volatility)
    delta : Dimensionless ratio
        - Calls: 0 to 1
        - Puts: -1 to 0
    gamma : Delta change per 1-point move in underlying (dimensionless)
    theta : Rupees per day (DAILY time decay)
        - Always negative (time decay reduces option value)
        - Already in daily units (matching Nautilus Trader convention)
        - Convert to per-lot: theta × lot_size
    vega : Rupees per 1% volatility change
        - Always positive (higher volatility increases option value)
        - Already scaled: vega = S·√T·φ(d1) / 100
    rho : Rupees per 1% interest rate change
        - Positive for calls, negative for puts

    Attributes
    ----------
    instrument_id : InstrumentId
        The instrument ID for the option
    iv : float
        Implied volatility (decimal, e.g., 0.15 for 15%)
    delta : float
        Option delta (-1 to 1 for puts/calls)
    gamma : float
        Option gamma (rate of change of delta per 1-point underlying move)
    theta : float
        Option theta (daily time decay in rupees/day)
    vega : float
        Option vega (sensitivity to 1% volatility change in rupees)
    rho : float
        Option rho (sensitivity to 1% interest rate change in rupees)
    ts_event : int
        Unix timestamp (nanoseconds) when the Greeks were calculated
    ts_init : int
        Unix timestamp (nanoseconds) when the data was initialized

    Metadata
    --------
    calculation_method : Black-Scholes analytical formulas
    iv_solver : Brent's method (scipy.optimize.brentq)
    iv_bounds : [0.0001, 5.0]
    risk_free_rate_source : RBI 91-day T-Bill rates (India)
    theta_units : daily (rupees per day)
    vega_units : per 1% volatility change (rupees)
    rho_units : per 1% rate change (rupees)

    Examples
    --------
    >>> from nautilus_trader.model.identifiers import InstrumentId
    >>> instrument_id = InstrumentId.from_str("NIFTY24JAN18000CE.NSE")
    >>> greeks = OptionGreeks(
    ...     ts_event=1704067200000000000,
    ...     ts_init=1704067200000000000,
    ...     instrument_id=instrument_id,
    ...     iv=0.15,
    ...     delta=0.5,
    ...     gamma=0.001,
    ...     theta=-0.05,
    ...     vega=0.2,
    ...     rho=0.01,
    ... )
    >>> print(greeks)
    OptionGreeks(instrument_id=NIFTY24JAN18000CE.NSE, iv=0.1500, delta=0.5000, ...)

    """

    # Field definitions with type annotations (required for @customdataclass)
    # The decorator will use these to create the PyArrow schema
    instrument_id: InstrumentId = InstrumentId.from_str("DEFAULT.VENUE")
    iv: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0

    def __repr__(self) -> str:
        """Return detailed string representation of OptionGreeks."""
        return (
            f"OptionGreeks("
            f"instrument_id={self.instrument_id}, "
            f"iv={self.iv:.4f}, "
            f"delta={self.delta:.4f}, "
            f"gamma={self.gamma:.6f}, "
            f"theta={self.theta:.4f}, "
            f"vega={self.vega:.4f}, "
            f"rho={self.rho:.4f}, "
            f"ts_event={unix_nanos_to_iso8601(self.ts_event)}, "
            f"ts_init={unix_nanos_to_iso8601(self.ts_init)})"
        )

    def __str__(self) -> str:
        """Return user-friendly string representation."""
        return (
            f"Greeks[{self.instrument_id}]: "
            f"IV={self.iv*100:.2f}%, "
            f"Δ={self.delta:.3f}, "
            f"Γ={self.gamma:.5f}, "
            f"Θ={self.theta:.3f}, "
            f"V={self.vega:.3f}, "
            f"ρ={self.rho:.3f}"
        )

