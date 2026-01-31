"""MarvelQuant: Custom trading strategies and extensions for NautilusTrader.

This package contains custom wrapper layer for NautilusTrader-based strategies.
All custom strategy implementations should go in this package, keeping the
/nautilus_trader/ fork clean and sync-friendly with upstream.

Architectural Boundaries:
- /nautilus_trader/: Protected fork (no custom strategies)
- /marvelquant/: Custom wrapper layer (strategies, risk, execution, config, registry, scripts, notebooks)
"""

__version__ = "0.1.0"
__author__ = "MarvelQuant Team"
