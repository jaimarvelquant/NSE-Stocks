"""
Interest Rate Provider Utility.

Parses interest rate XML files (Nautilus/SDMX format) to provide risk-free rates
for Option Greeks calculations.
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, date
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class InterestRateProvider:
    """
    Provides interest rates from parsed XML data.
    """
    
    def __init__(self, xml_path: Path):
        """
        Initialize provider with path to XML file.
        
        Args:
            xml_path: Path to the interest rate XML file.
        """
        self.xml_path = xml_path
        self.rates: Dict[str, float] = {}
        self.load()
        
    def load(self):
        """Load and parse the XML file."""
        if not self.xml_path.exists():
            logger.error(f"Interest rate XML not found: {self.xml_path}")
            return
            
        try:
            tree = ET.parse(self.xml_path)
            root = tree.getroot()
            
            # Handle namespaced XML (SDMX format)
            # Nautilus XML usually has namespaces like xmlns:message="http://www.sdmx.org/..."
            # We can search using local tags or handle namespaces
            
            # Strategy: Find all 'Obs' elements which contain rates
            # Structure: GenericData -> DataSet -> Series -> Obs
            
            # Define namespaces map if needed, or use broad search
            # In the provided file:
            # <generic:Obs>
            #   <generic:ObsDimension id="TIME_PERIOD" value="2024-01" />
            #   <generic:ObsValue value="6.91" />
            # </generic:Obs>
            
            # We'll use a namespace agnostic approach for simplicity using xpath with local-name()
            # or just string parsing if elementtree namespace handling is tricky
            
            # Using iter to traverse everything
            for elem in root.iter():
                if elem.tag.endswith('Obs'):
                    date_str = None
                    value_str = None
                    
                    for child in elem:
                        if child.tag.endswith('ObsDimension'):
                            if child.get('id') == 'TIME_PERIOD':
                                date_str = child.get('value')
                        elif child.tag.endswith('ObsValue'):
                            value_str = child.get('value')
                    
                    if date_str and value_str:
                        try:
                            rate = float(value_str)
                            # Store as YYYY-MM string for easy lookup
                            self.rates[date_str] = rate
                        except ValueError:
                            pass
                            
            logger.info(f"Loaded {len(self.rates)} interest rate records from {self.xml_path.name}")
            
        except Exception as e:
            logger.error(f"Error parsing interest rate XML: {e}")

    def get_risk_free_rate(self, query_date: date) -> float:
        """
        Get annualized risk-free rate for a specific date.
        
        Args:
            query_date: Date to look up
            
        Returns:
            Rate as decimal (e.g. 0.0691 for 6.91%)
        """
        if not self.rates:
            return 0.06  # Default fallback 6%
            
        # Format date as YYYY-MM (since XML is monthly)
        key = query_date.strftime("%Y-%m")
        
        if key in self.rates:
            # XML values are in percent (e.g. 6.91), return decimal
            return self.rates[key] / 100.0
            
        # If exact month missing, find closest previous month
        # Simple fallback: use the latest available rate
        # (Since rates are sorted in dict usually if python >= 3.7, or we sort keys)
        sorted_keys = sorted(self.rates.keys())
        
        # Find index
        idx = -1
        for i, k in enumerate(sorted_keys):
            if k > key:
                break
            idx = i
            
        if idx >= 0:
            closest_key = sorted_keys[idx]
            return self.rates[closest_key] / 100.0
            
        return 0.06  # Default fallback

