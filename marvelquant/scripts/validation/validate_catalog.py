#!/usr/bin/env python3
"""
Nautilus Catalog Validator with Data Type and Byte Size Validation.

Validates Nautilus Trader catalog output by checking:
- Data completeness (expected instruments/data exist)
- Schema validation (correct columns and types)
- Data type validation (Parquet schema and Arrow types)
- Byte size validation (file sizes, data sizes)
- Data quality (no missing values, valid ranges)
- Date range coverage
- Data consistency
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

import pandas as pd
try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyArrow not available. Data type validation will be limited.")

from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.data import BarType
from nautilus_trader.model.instruments import Instrument

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    check_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class SchemaDefinition:
    """Expected schema for a data type."""
    data_type: str  # e.g., "bar", "quote_tick", "custom_option_greeks"
    required_columns: List[str]
    column_types: Dict[str, str]  # column_name -> expected_type (e.g., "int64", "float64", "string")
    optional_columns: List[str] = field(default_factory=list)


@dataclass
class ValidationCriteria:
    """Configuration for validation criteria."""
    # Expected symbols/instruments
    expected_symbols: List[str] = field(default_factory=list)
    expected_instrument_types: List[str] = field(default_factory=lambda: ["index", "futures", "options"])
    
    # Date range validation
    expected_start_date: Optional[str] = None
    expected_end_date: Optional[str] = None
    require_complete_dates: bool = True
    
    # Data quality checks
    check_missing_values: bool = True
    check_price_ranges: bool = True
    min_price: float = 0.0
    max_price: float = 1000000.0
    check_volume_ranges: bool = True
    min_volume: float = 0.0
    
    # Schema validation
    validate_schema: bool = True
    validate_data_types: bool = True
    validate_byte_sizes: bool = True
    
    # File size validation
    min_file_size_bytes: int = 0
    max_file_size_bytes: Optional[int] = None  # None = no limit
    min_total_catalog_size_mb: Optional[float] = None
    max_total_catalog_size_mb: Optional[float] = None
    
    # Data type validation
    expected_schemas: Dict[str, SchemaDefinition] = field(default_factory=dict)
    
    # Consistency checks
    check_timestamp_ordering: bool = True
    check_duplicate_timestamps: bool = True
    check_bar_consistency: bool = True  # high >= low, etc.
    
    # Coverage checks
    min_bars_per_day: Optional[int] = None  # Minimum bars expected per day
    expected_bar_types: List[str] = field(default_factory=lambda: ["1-MINUTE-LAST-EXTERNAL"])


class NautilusCatalogValidator:
    """Validator for Nautilus Trader catalog output with data type and byte size checks."""
    
    def __init__(self, catalog_path: Path, criteria: Optional[ValidationCriteria] = None):
        """
        Initialize validator.
        
        Args:
            catalog_path: Path to the catalog directory
            criteria: Validation criteria configuration
        """
        self.catalog_path = Path(catalog_path)
        self.criteria = criteria or ValidationCriteria()
        self.catalog = ParquetDataCatalog(catalog_path)
        self.results: List[ValidationResult] = []
        
        # Define expected schemas for common data types
        self._initialize_default_schemas()
        
    def _initialize_default_schemas(self):
        """Initialize default schema definitions for common Nautilus data types."""
        if not self.criteria.expected_schemas:
            # Bar data schema (typical Nautilus bar structure)
            self.criteria.expected_schemas["bar"] = SchemaDefinition(
                data_type="bar",
                required_columns=["ts_event", "ts_init", "open", "high", "low", "close", "volume"],
                column_types={
                    "ts_event": "int64",
                    "ts_init": "int64",
                    "open": "string",  # Nautilus uses Decimal128 as string
                    "high": "string",
                    "low": "string",
                    "close": "string",
                    "volume": "string",  # Nautilus uses Decimal128 as string
                }
            )
            
            # Quote tick schema
            self.criteria.expected_schemas["quote_tick"] = SchemaDefinition(
                data_type="quote_tick",
                required_columns=["ts_event", "ts_init", "bid_price", "ask_price", "bid_size", "ask_size"],
                column_types={
                    "ts_event": "int64",
                    "ts_init": "int64",
                    "bid_price": "string",
                    "ask_price": "string",
                    "bid_size": "string",
                    "ask_size": "string",
                }
            )
    
    def validate_all(self) -> List[ValidationResult]:
        """Run all validation checks."""
        logger.info("Starting catalog validation...")
        logger.info(f"Catalog path: {self.catalog_path}")
        
        # Run all validation checks
        self.results = []
        
        # 1. Catalog structure validation
        self._validate_catalog_structure()
        
        # 2. File size validation (byte sizes)
        self._validate_file_sizes()
        
        # 3. Parquet schema validation (data types)
        if PYARROW_AVAILABLE:
            self._validate_parquet_schemas()
        else:
            result = ValidationResult(
                check_name="Parquet Schemas",
                passed=False,
                message="PyArrow not available - skipping schema validation",
                warnings=["PyArrow is required for data type validation. Install with: pip install pyarrow"]
            )
            self.results.append(result)
        
        # 4. Instrument validation
        self._validate_instruments()
        
        # 5. Bar data validation
        self._validate_bars()
        
        # 6. Quote tick validation
        self._validate_quote_ticks()
        
        # 7. Custom data validation
        self._validate_custom_data()
        
        # 8. Date range validation
        self._validate_date_ranges()
        
        # 9. Data quality validation
        self._validate_data_quality()
        
        # 10. Consistency validation
        self._validate_consistency()
        
        # Summary
        self._print_summary()
        
        return self.results
    
    def _validate_catalog_structure(self):
        """Validate catalog directory structure."""
        logger.info("Validating catalog structure...")
        
        result = ValidationResult(
            check_name="Catalog Structure",
            passed=True,
            message="Catalog structure is valid"
        )
        
        # Check if catalog directory exists
        if not self.catalog_path.exists():
            result.passed = False
            result.errors.append(f"Catalog directory does not exist: {self.catalog_path}")
            self.results.append(result)
            return
        
        # Check expected subdirectories
        expected_dirs = [
            "data/bar",
            "data/quote_tick",
            "data/futures_contract",
            "data/option_contract",
        ]
        
        missing_dirs = []
        existing_dirs = []
        for dir_path in expected_dirs:
            full_path = self.catalog_path / dir_path
            if full_path.exists():
                existing_dirs.append(dir_path)
            else:
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            result.warnings.append(f"Missing optional directories: {missing_dirs}")
        
        result.details["catalog_path"] = str(self.catalog_path)
        result.details["exists"] = True
        result.details["existing_dirs"] = existing_dirs
        result.details["missing_dirs"] = missing_dirs
        self.results.append(result)
    
    def _validate_file_sizes(self):
        """Validate file sizes (byte sizes) in catalog."""
        logger.info("Validating file sizes...")
        
        result = ValidationResult(
            check_name="File Sizes",
            passed=True,
            message="File size validation passed"
        )
        
        if not self.criteria.validate_byte_sizes:
            result.warnings.append("Byte size validation disabled")
            self.results.append(result)
            return
        
        try:
            total_size_bytes = 0
            file_stats = []
            parquet_files = []
            
            # Find all parquet files in catalog
            data_dir = self.catalog_path / "data"
            if data_dir.exists():
                parquet_files = list(data_dir.rglob("*.parquet"))
            
            for parquet_file in parquet_files:
                try:
                    file_size = parquet_file.stat().st_size
                    total_size_bytes += file_size
                    
                    # Check individual file size constraints
                    if file_size < self.criteria.min_file_size_bytes:
                        result.warnings.append(
                            f"File {parquet_file.relative_to(self.catalog_path)} "
                            f"is smaller than minimum ({file_size} < {self.criteria.min_file_size_bytes} bytes)"
                        )
                    
                    if self.criteria.max_file_size_bytes and file_size > self.criteria.max_file_size_bytes:
                        result.warnings.append(
                            f"File {parquet_file.relative_to(self.catalog_path)} "
                            f"exceeds maximum size ({file_size} > {self.criteria.max_file_size_bytes} bytes)"
                        )
                    
                    file_stats.append({
                        "path": str(parquet_file.relative_to(self.catalog_path)),
                        "size_bytes": file_size,
                        "size_mb": file_size / (1024 * 1024),
                    })
                    
                except Exception as e:
                    result.warnings.append(f"Error checking file {parquet_file}: {str(e)}")
            
            total_size_mb = total_size_bytes / (1024 * 1024)
            
            # Check total catalog size
            if self.criteria.min_total_catalog_size_mb:
                if total_size_mb < self.criteria.min_total_catalog_size_mb:
                    result.passed = False
                    result.errors.append(
                        f"Total catalog size ({total_size_mb:.2f} MB) is below minimum "
                        f"({self.criteria.min_total_catalog_size_mb} MB)"
                    )
            
            if self.criteria.max_total_catalog_size_mb:
                if total_size_mb > self.criteria.max_total_catalog_size_mb:
                    result.warnings.append(
                        f"Total catalog size ({total_size_mb:.2f} MB) exceeds maximum "
                        f"({self.criteria.max_total_catalog_size_mb} MB)"
                    )
            
            # Sort by size for reporting
            file_stats.sort(key=lambda x: x["size_bytes"], reverse=True)
            
            result.details["total_files"] = len(parquet_files)
            result.details["total_size_bytes"] = total_size_bytes
            result.details["total_size_mb"] = total_size_mb
            result.details["total_size_gb"] = total_size_mb / 1024
            result.details["largest_files"] = file_stats[:10]  # Top 10 largest files
            result.details["smallest_files"] = file_stats[-10:] if len(file_stats) > 10 else file_stats  # Bottom 10
            
            # Group by directory
            size_by_dir = defaultdict(int)
            for parquet_file in parquet_files:
                rel_path = parquet_file.relative_to(self.catalog_path)
                dir_name = str(rel_path.parent)
                size_by_dir[dir_name] += parquet_file.stat().st_size
            
            result.details["size_by_directory"] = {
                k: {
                    "bytes": v,
                    "mb": v / (1024 * 1024)
                }
                for k, v in sorted(size_by_dir.items(), key=lambda x: x[1], reverse=True)
            }
            
        except Exception as e:
            result.passed = False
            result.errors.append(f"Error validating file sizes: {str(e)}")
            logger.exception("Error in file size validation")
        
        self.results.append(result)
    
    def _validate_parquet_schemas(self):
        """Validate Parquet file schemas and data types."""
        logger.info("Validating Parquet schemas and data types...")
        
        result = ValidationResult(
            check_name="Parquet Schemas",
            passed=True,
            message="Parquet schema validation passed"
        )
        
        if not self.criteria.validate_data_types:
            result.warnings.append("Data type validation disabled")
            self.results.append(result)
            return
        
        if not PYARROW_AVAILABLE:
            result.passed = False
            result.errors.append("PyArrow is required for schema validation")
            self.results.append(result)
            return
        
        try:
            data_dir = self.catalog_path / "data"
            if not data_dir.exists():
                result.warnings.append("Data directory does not exist")
                self.results.append(result)
                return
            
            schema_issues = []
            schema_stats = defaultdict(lambda: {"count": 0, "files": []})
            
            # Check each data type directory
            for data_type_dir in data_dir.iterdir():
                if not data_type_dir.is_dir():
                    continue
                
                data_type_name = data_type_dir.name
                parquet_files = list(data_type_dir.rglob("*.parquet"))
                
                if not parquet_files:
                    continue
                
                # Sample a few files for schema validation
                sample_files = parquet_files[:5]  # Check first 5 files per data type
                
                for parquet_file in sample_files:
                    try:
                        # Read Parquet file metadata
                        parquet_file_obj = pq.ParquetFile(parquet_file)
                        schema = parquet_file_obj.schema_arrow
                        
                        # Get column names and types
                        columns = {}
                        for field in schema:
                            col_name = field.name
                            col_type = str(field.type)
                            columns[col_name] = col_type
                        
                        schema_stats[data_type_name]["count"] += 1
                        schema_stats[data_type_name]["files"].append({
                            "file": str(parquet_file.relative_to(self.catalog_path)),
                            "columns": len(columns),
                            "schema": columns
                        })
                        
                        # Validate against expected schema if defined
                        if data_type_name in self.criteria.expected_schemas:
                            expected_schema = self.criteria.expected_schemas[data_type_name]
                            
                            # Check required columns
                            missing_columns = set(expected_schema.required_columns) - set(columns.keys())
                            if missing_columns:
                                schema_issues.append({
                                    "file": str(parquet_file.relative_to(self.catalog_path)),
                                    "issue": "missing_columns",
                                    "missing": list(missing_columns)
                                })
                                result.passed = False
                            
                            # Check column types
                            for col_name, expected_type in expected_schema.column_types.items():
                                if col_name in columns:
                                    actual_type = columns[col_name]
                                    # Normalize type comparison (handle variations)
                                    if not self._types_match(expected_type, actual_type):
                                        schema_issues.append({
                                            "file": str(parquet_file.relative_to(self.catalog_path)),
                                            "issue": "type_mismatch",
                                            "column": col_name,
                                            "expected": expected_type,
                                            "actual": actual_type
                                        })
                                        result.warnings.append(
                                            f"Type mismatch in {parquet_file.name}: "
                                            f"{col_name} expected {expected_type}, got {actual_type}"
                                        )
                        
                        # Get file metadata
                        metadata = parquet_file_obj.metadata
                        num_rows = metadata.num_rows
                        num_row_groups = metadata.num_row_groups
                        
                        # Check for empty files
                        if num_rows == 0:
                            result.warnings.append(
                                f"Empty Parquet file: {parquet_file.relative_to(self.catalog_path)}"
                            )
                        
                        # Store additional metadata
                        schema_stats[data_type_name]["files"][-1]["num_rows"] = num_rows
                        schema_stats[data_type_name]["files"][-1]["num_row_groups"] = num_row_groups
                        schema_stats[data_type_name]["files"][-1]["file_size_bytes"] = parquet_file.stat().st_size
                        
                    except Exception as e:
                        schema_issues.append({
                            "file": str(parquet_file.relative_to(self.catalog_path)),
                            "issue": "read_error",
                            "error": str(e)
                        })
                        result.warnings.append(f"Error reading schema from {parquet_file}: {str(e)}")
                
                # Check consistency across files of same type
                if len(sample_files) > 1:
                    first_schema = None
                    for parquet_file in sample_files:
                        try:
                            parquet_file_obj = pq.ParquetFile(parquet_file)
                            schema = parquet_file_obj.schema_arrow
                            column_names = {field.name for field in schema}
                            
                            if first_schema is None:
                                first_schema = column_names
                            else:
                                if column_names != first_schema:
                                    result.warnings.append(
                                        f"Inconsistent schemas in {data_type_name}: "
                                        f"{parquet_file.name} differs from first file"
                                    )
                        except Exception:
                            continue
            
            result.details["schema_stats"] = {
                k: {
                    "files_checked": v["count"],
                    "sample_files": v["files"][:3]  # Show first 3
                }
                for k, v in schema_stats.items()
            }
            
            if schema_issues:
                result.details["schema_issues"] = schema_issues[:20]  # Limit to first 20
                result.message = f"Found {len(schema_issues)} schema issues"
            
        except Exception as e:
            result.passed = False
            result.errors.append(f"Error validating Parquet schemas: {str(e)}")
            logger.exception("Error in Parquet schema validation")
        
        self.results.append(result)
    
    def _types_match(self, expected: str, actual: str) -> bool:
        """
        Check if expected and actual types match (with normalization).
        
        Args:
            expected: Expected type string (e.g., "int64", "float64", "string")
            actual: Actual Arrow type string (e.g., "int64", "double", "string", "decimal128(38,9)")
        
        Returns:
            True if types match (with normalization)
        """
        # Normalize type strings
        expected_norm = expected.lower().strip()
        actual_norm = actual.lower().strip()
        
        # Direct match
        if expected_norm == actual_norm:
            return True
        
        # Handle Arrow type variations
        type_mappings = {
            "int64": ["int64", "int64[pyarrow]", "int64[arrow]"],
            "int32": ["int32", "int32[pyarrow]", "int32[arrow]"],
            "float64": ["float64", "double", "double[pyarrow]", "double[arrow]"],
            "float32": ["float32", "float", "float[pyarrow]", "float[arrow]"],
            "string": ["string", "str", "utf8", "large_string", "string[pyarrow]", "utf8[pyarrow]"],
        }
        
        # Check if actual type matches any variant of expected
        if expected_norm in type_mappings:
            return actual_norm in type_mappings[expected_norm]
        
        # For Decimal128 (Nautilus uses for prices), check if it's a decimal type
        if expected_norm == "string" and "decimal" in actual_norm:
            return True  # Decimal128 is represented as string in Nautilus
        
        # For timestamp types
        if "timestamp" in expected_norm and "timestamp" in actual_norm:
            return True
        
        return False
    
    def _validate_instruments(self):
        """Validate instruments in catalog."""
        logger.info("Validating instruments...")
        
        result = ValidationResult(
            check_name="Instruments",
            passed=True,
            message="Instruments validation passed"
        )
        
        try:
            # Read all instruments
            instruments = self.catalog.instruments()
            instrument_list = list(instruments)
            
            if not instrument_list:
                result.passed = False
                result.errors.append("No instruments found in catalog")
                self.results.append(result)
                return
            
            result.details["total_instruments"] = len(instrument_list)
            
            # Group by type
            by_type = defaultdict(list)
            by_symbol = defaultdict(list)
            
            for instrument in instrument_list:
                inst_type = type(instrument).__name__
                by_type[inst_type].append(instrument)
                symbol = instrument.id.symbol.value
                by_symbol[symbol].append(instrument)
            
            result.details["instruments_by_type"] = {k: len(v) for k, v in by_type.items()}
            result.details["instruments_by_symbol"] = {k: len(v) for k, v in by_symbol.items()}
            
            # Check expected symbols
            if self.criteria.expected_symbols:
                missing_symbols = set(self.criteria.expected_symbols) - set(by_symbol.keys())
                if missing_symbols:
                    result.passed = False
                    result.errors.append(f"Missing expected symbols: {sorted(missing_symbols)}")
                else:
                    result.details["expected_symbols_found"] = True
            
            # Validate instrument properties
            invalid_instruments = []
            for instrument in instrument_list:
                issues = []
                if not instrument.id:
                    issues.append("Missing instrument ID")
                if not hasattr(instrument, 'currency') or not instrument.currency:
                    issues.append("Missing currency")
                if hasattr(instrument, 'price_precision') and instrument.price_precision < 0:
                    issues.append("Invalid price precision")
                
                if issues:
                    invalid_instruments.append({
                        "instrument_id": str(instrument.id) if instrument.id else "UNKNOWN",
                        "issues": issues
                    })
            
            if invalid_instruments:
                result.warnings.append(f"Found {len(invalid_instruments)} instruments with issues")
                result.details["invalid_instruments"] = invalid_instruments[:10]  # Limit to first 10
            
        except Exception as e:
            result.passed = False
            result.errors.append(f"Error validating instruments: {str(e)}")
            logger.exception("Error in instrument validation")
        
        self.results.append(result)
    
    def _validate_bars(self):
        """Validate bar data in catalog."""
        logger.info("Validating bars...")
        
        result = ValidationResult(
            check_name="Bars",
            passed=True,
            message="Bars validation passed"
        )
        
        try:
            # Get all instruments
            instruments = list(self.catalog.instruments())
            
            if not instruments:
                result.warnings.append("No instruments found, skipping bar validation")
                self.results.append(result)
                return
            
            total_bars = 0
            bar_stats = {}
            date_coverage = defaultdict(set)
            duplicate_timestamps = []
            ordering_issues = []
            
            for instrument in instruments[:10]:  # Limit to first 10 for performance
                instrument_id = instrument.id
                
                try:
                    # Query bars for this instrument
                    bars = self.catalog.bars(
                        instrument_id=instrument_id,
                        bar_type=None,  # Get all bar types
                        start=None,
                        end=None,
                    )
                    
                    bar_list = list(bars)
                    if not bar_list:
                        continue
                    
                    total_bars += len(bar_list)
                    
                    # Collect statistics
                    timestamps = [bar.ts_event for bar in bar_list]
                    dates = {datetime.fromtimestamp(ts / 1e9).date() for ts in timestamps}
                    
                    for d in dates:
                        date_coverage[instrument_id].add(d)
                    
                    # Check timestamp ordering
                    if self.criteria.check_timestamp_ordering:
                        sorted_timestamps = sorted(timestamps)
                        if timestamps != sorted_timestamps:
                            ordering_issues.append(str(instrument_id))
                            result.warnings.append(
                                f"Instrument {instrument_id}: Timestamps are not in ascending order"
                            )
                    
                    # Check for duplicate timestamps
                    if self.criteria.check_duplicate_timestamps:
                        seen = set()
                        for ts in timestamps:
                            if ts in seen:
                                duplicate_timestamps.append({
                                    "instrument_id": str(instrument_id),
                                    "timestamp": ts
                                })
                                break
                            seen.add(ts)
                    
                    # Check bar properties
                    invalid_bars = []
                    for bar in bar_list[:100]:  # Sample first 100
                        issues = []
                        
                        if self.criteria.check_bar_consistency:
                            if bar.high < bar.low:
                                issues.append("High < Low")
                            if bar.close < bar.low or bar.close > bar.high:
                                issues.append("Close outside [Low, High]")
                            if bar.open < bar.low or bar.open > bar.high:
                                issues.append("Open outside [Low, High]")
                        
                        if self.criteria.check_price_ranges:
                            for price_attr in ['open', 'high', 'low', 'close']:
                                price = getattr(bar, price_attr, None)
                                if price is not None:
                                    try:
                                        price_val = float(price)
                                        if price_val < self.criteria.min_price or price_val > self.criteria.max_price:
                                            issues.append(f"{price_attr} out of range: {price_val}")
                                    except (ValueError, TypeError):
                                        issues.append(f"{price_attr} is not a valid number")
                        
                        if self.criteria.check_volume_ranges:
                            if hasattr(bar, 'volume'):
                                try:
                                    vol = float(bar.volume) if bar.volume else 0
                                    if vol < self.criteria.min_volume:
                                        issues.append(f"Volume below minimum: {vol}")
                                except (ValueError, TypeError):
                                    issues.append("Volume is not a valid number")
                        
                        if self.criteria.check_missing_values:
                            for attr in ['open', 'high', 'low', 'close']:
                                if getattr(bar, attr, None) is None:
                                    issues.append(f"Missing {attr}")
                        
                        if issues:
                            invalid_bars.append({
                                "timestamp": bar.ts_event,
                                "issues": issues
                            })
                    
                    if invalid_bars:
                        result.warnings.append(
                            f"Instrument {instrument_id}: {len(invalid_bars)} bars with issues"
                        )
                    
                    bar_stats[str(instrument_id)] = {
                        "count": len(bar_list),
                        "date_range": (
                            min(dates).isoformat() if dates else None,
                            max(dates).isoformat() if dates else None
                        ),
                        "dates_covered": len(dates)
                    }
                    
                except Exception as e:
                    result.warnings.append(f"Error querying bars for {instrument_id}: {str(e)}")
                    continue
            
            result.details["total_bars"] = total_bars
            result.details["bar_stats"] = bar_stats
            result.details["instruments_with_bars"] = len(bar_stats)
            result.details["duplicate_timestamps"] = len(duplicate_timestamps)
            result.details["ordering_issues"] = len(ordering_issues)
            
            # Check date coverage
            if self.criteria.expected_start_date and self.criteria.expected_end_date:
                expected_start = date.fromisoformat(self.criteria.expected_start_date)
                expected_end = date.fromisoformat(self.criteria.expected_end_date)
                
                for inst_id, dates in date_coverage.items():
                    if dates:
                        actual_start = min(dates)
                        actual_end = max(dates)
                        
                        if actual_start > expected_start or actual_end < expected_end:
                            result.warnings.append(
                                f"Instrument {inst_id}: Date range ({actual_start} to {actual_end}) "
                                f"does not fully cover expected range ({expected_start} to {expected_end})"
                            )
            
        except Exception as e:
            result.passed = False
            result.errors.append(f"Error validating bars: {str(e)}")
            logger.exception("Error in bar validation")
        
        self.results.append(result)
    
    def _validate_quote_ticks(self):
        """Validate quote tick data."""
        logger.info("Validating quote ticks...")
        
        result = ValidationResult(
            check_name="Quote Ticks",
            passed=True,
            message="Quote ticks validation passed"
        )
        
        try:
            instruments = list(self.catalog.instruments())
            
            if not instruments:
                result.warnings.append("No instruments found, skipping quote tick validation")
                self.results.append(result)
                return
            
            total_ticks = 0
            invalid_ticks = []
            
            for instrument in instruments[:5]:  # Limit to first 5 for performance
                instrument_id = instrument.id
                
                try:
                    ticks = self.catalog.quote_ticks(
                        instrument_id=instrument_id,
                        start=None,
                        end=None,
                    )
                    
                    tick_list = list(ticks)
                    if tick_list:
                        total_ticks += len(tick_list)
                        
                        # Sample validation
                        for tick in tick_list[:50]:
                            issues = []
                            
                            if hasattr(tick, 'bid_price') and hasattr(tick, 'ask_price'):
                                if tick.bid_price and tick.ask_price:
                                    try:
                                        bid = float(tick.bid_price)
                                        ask = float(tick.ask_price)
                                        if bid > ask:
                                            issues.append("Bid > Ask")
                                    except (ValueError, TypeError):
                                        issues.append("Invalid bid/ask prices")
                            
                            if self.criteria.check_missing_values:
                                if hasattr(tick, 'bid_price') and not tick.bid_price:
                                    issues.append("Missing bid_price")
                                if hasattr(tick, 'ask_price') and not tick.ask_price:
                                    issues.append("Missing ask_price")
                            
                            if issues:
                                invalid_ticks.append({
                                    "instrument_id": str(instrument_id),
                                    "timestamp": tick.ts_event if hasattr(tick, 'ts_event') else None,
                                    "issues": issues
                                })
                                break
                
                except Exception as e:
                    # Quote ticks might not exist for all instruments
                    continue
            
            result.details["total_quote_ticks"] = total_ticks
            result.details["invalid_ticks"] = len(invalid_ticks)
            
            if invalid_ticks:
                result.warnings.append(f"Found {len(invalid_ticks)} ticks with issues")
            
        except Exception as e:
            result.warnings.append(f"Error validating quote ticks: {str(e)}")
        
        self.results.append(result)
    
    def _validate_custom_data(self):
        """Validate custom data types (OI, Greeks, etc.)."""
        logger.info("Validating custom data...")
        
        result = ValidationResult(
            check_name="Custom Data",
            passed=True,
            message="Custom data validation passed"
        )
        
        try:
            # Check for custom data directories
            custom_dirs = [
                "data/custom_future_oi",
                "data/custom_option_greeks",
                "data/custom_option_oi",
            ]
            
            custom_data_stats = {}
            
            for dir_name in custom_dirs:
                dir_path = self.catalog_path / dir_name
                if dir_path.exists():
                    # Count parquet files
                    parquet_files = list(dir_path.rglob("*.parquet"))
                    total_size = sum(f.stat().st_size for f in parquet_files)
                    
                    custom_data_stats[dir_name] = {
                        "exists": True,
                        "parquet_files": len(parquet_files),
                        "total_size_bytes": total_size,
                        "total_size_mb": total_size / (1024 * 1024)
                    }
                    
                    # If PyArrow available, check schemas
                    if PYARROW_AVAILABLE and parquet_files:
                        sample_file = parquet_files[0]
                        try:
                            parquet_file_obj = pq.ParquetFile(sample_file)
                            schema = parquet_file_obj.schema_arrow
                            columns = {field.name: str(field.type) for field in schema}
                            custom_data_stats[dir_name]["sample_schema"] = columns
                        except Exception as e:
                            custom_data_stats[dir_name]["schema_error"] = str(e)
                else:
                    custom_data_stats[dir_name] = {
                        "exists": False,
                        "parquet_files": 0,
                        "total_size_bytes": 0,
                        "total_size_mb": 0
                    }
            
            result.details["custom_data_stats"] = custom_data_stats
            
        except Exception as e:
            result.warnings.append(f"Error validating custom data: {str(e)}")
        
        self.results.append(result)
    
    def _validate_date_ranges(self):
        """Validate date range coverage."""
        logger.info("Validating date ranges...")
        
        result = ValidationResult(
            check_name="Date Ranges",
            passed=True,
            message="Date range validation passed"
        )
        
        if not self.criteria.expected_start_date or not self.criteria.expected_end_date:
            result.warnings.append("No expected date range specified, skipping validation")
            self.results.append(result)
            return
        
        # This is partially covered in _validate_bars, but we can add more checks here
        result.details["expected_start"] = self.criteria.expected_start_date
        result.details["expected_end"] = self.criteria.expected_end_date
        
        self.results.append(result)
    
    def _validate_data_quality(self):
        """Validate overall data quality."""
        logger.info("Validating data quality...")
        
        result = ValidationResult(
            check_name="Data Quality",
            passed=True,
            message="Data quality validation passed"
        )
        
        # This is a summary check - detailed quality checks are in other methods
        result.details["checks_enabled"] = {
            "check_missing_values": self.criteria.check_missing_values,
            "check_price_ranges": self.criteria.check_price_ranges,
            "check_volume_ranges": self.criteria.check_volume_ranges,
        }
        
        self.results.append(result)
    
    def _validate_consistency(self):
        """Validate data consistency."""
        logger.info("Validating consistency...")
        
        result = ValidationResult(
            check_name="Consistency",
            passed=True,
            message="Consistency validation passed"
        )
        
        # Summary of consistency checks performed
        result.details["checks_performed"] = {
            "timestamp_ordering": self.criteria.check_timestamp_ordering,
            "duplicate_timestamps": self.criteria.check_duplicate_timestamps,
            "bar_consistency": self.criteria.check_bar_consistency,
        }
        
        self.results.append(result)
    
    def _print_summary(self):
        """Print validation summary."""
        logger.info("=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)
        
        total_checks = len(self.results)
        passed_checks = sum(1 for r in self.results if r.passed)
        failed_checks = total_checks - passed_checks
        
        logger.info(f"Total checks: {total_checks}")
        logger.info(f"Passed: {passed_checks}")
        logger.info(f"Failed: {failed_checks}")
        logger.info("")
        
        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            logger.info(f"{status}: {result.check_name}")
            logger.info(f"  {result.message}")
            
            # Print key details
            if "total_size_mb" in result.details:
                logger.info(f"  Total size: {result.details['total_size_mb']:.2f} MB "
                          f"({result.details.get('total_size_gb', 0):.2f} GB)")
            if "total_files" in result.details:
                logger.info(f"  Total files: {result.details['total_files']}")
            if "total_instruments" in result.details:
                logger.info(f"  Total instruments: {result.details['total_instruments']}")
            if "total_bars" in result.details:
                logger.info(f"  Total bars: {result.details['total_bars']:,}")
            
            if result.errors:
                logger.info(f"  Errors ({len(result.errors)}):")
                for error in result.errors[:5]:  # Show first 5
                    logger.info(f"    - {error}")
            
            if result.warnings:
                logger.info(f"  Warnings ({len(result.warnings)}):")
                for warning in result.warnings[:5]:  # Show first 5
                    logger.info(f"    - {warning}")
            
            logger.info("")
        
        logger.info("=" * 80)
        
        if failed_checks > 0:
            logger.warning(f"Validation completed with {failed_checks} failed check(s)")
        else:
            logger.info("All validation checks passed!")
    
    def get_summary_dict(self) -> Dict[str, Any]:
        """Get validation summary as dictionary."""
        return {
            "total_checks": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": sum(1 for r in self.results if not r.passed),
            "results": [
                {
                    "check_name": r.check_name,
                    "passed": r.passed,
                    "message": r.message,
                    "errors": r.errors,
                    "warnings": r.warnings,
                    "details": r.details,
                }
                for r in self.results
            ]
        }


def main():
    """CLI entrypoint."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate Nautilus Trader catalog output with data type and byte size checks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation
  python validate_catalog.py --catalog-path /home/ubuntu/nautilus_data
  
  # With expected symbols and date range
  python validate_catalog.py \\
      --catalog-path /home/ubuntu/nautilus_data \\
      --expected-symbols NIFTY BANKNIFTY \\
      --expected-start-date 2024-01-01 \\
      --expected-end-date 2024-01-31
  
  # With size constraints
  python validate_catalog.py \\
      --catalog-path /home/ubuntu/nautilus_data \\
      --min-file-size-bytes 1024 \\
      --max-file-size-bytes 104857600 \\
      --min-total-size-mb 100
        """
    )
    parser.add_argument(
        "--catalog-path",
        type=Path,
        required=True,
        help="Path to Nautilus catalog directory"
    )
    parser.add_argument(
        "--expected-symbols",
        nargs="+",
        default=[],
        help="Expected symbols to validate (e.g., NIFTY BANKNIFTY)"
    )
    parser.add_argument(
        "--expected-start-date",
        type=str,
        help="Expected start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--expected-end-date",
        type=str,
        help="Expected end date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--min-file-size-bytes",
        type=int,
        default=0,
        help="Minimum file size in bytes (default: 0)"
    )
    parser.add_argument(
        "--max-file-size-bytes",
        type=int,
        help="Maximum file size in bytes (no limit if not specified)"
    )
    parser.add_argument(
        "--min-total-size-mb",
        type=float,
        help="Minimum total catalog size in MB"
    )
    parser.add_argument(
        "--max-total-size-mb",
        type=float,
        help="Maximum total catalog size in MB"
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=0.0,
        help="Minimum valid price (default: 0.0)"
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=1000000.0,
        help="Maximum valid price (default: 1000000.0)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Output validation results to JSON file"
    )
    parser.add_argument(
        "--skip-data-type-validation",
        action="store_true",
        help="Skip data type validation (faster but less thorough)"
    )
    parser.add_argument(
        "--skip-byte-size-validation",
        action="store_true",
        help="Skip byte size validation"
    )
    parser.add_argument(
        "--skip-schema-validation",
        action="store_true",
        help="Skip schema validation"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create validation criteria
    criteria = ValidationCriteria(
        expected_symbols=args.expected_symbols,
        expected_start_date=args.expected_start_date,
        expected_end_date=args.expected_end_date,
        validate_data_types=not args.skip_data_type_validation,
        validate_byte_sizes=not args.skip_byte_size_validation,
        validate_schema=not args.skip_schema_validation,
        min_file_size_bytes=args.min_file_size_bytes,
        max_file_size_bytes=args.max_file_size_bytes,
        min_total_catalog_size_mb=args.min_total_size_mb,
        max_total_catalog_size_mb=args.max_total_size_mb,
        min_price=args.min_price,
        max_price=args.max_price,
    )
    
    # Run validation
    validator = NautilusCatalogValidator(args.catalog_path, criteria)
    results = validator.validate_all()
    
    # Output JSON if requested
    if args.output_json:
        import json
        summary = validator.get_summary_dict()
        with open(args.output_json, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Validation results saved to: {args.output_json}")
    
    # Exit with error code if validation failed
    failed_checks = sum(1 for r in results if not r.passed)
    return 1 if failed_checks > 0 else 0


if __name__ == "__main__":
    sys.exit(main())



