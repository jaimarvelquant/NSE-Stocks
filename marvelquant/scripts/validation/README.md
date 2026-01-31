# Nautilus Catalog Validator

A comprehensive validator for Nautilus Trader catalog output that checks data types, byte sizes, schemas, and data quality.

## Features

The validator performs the following checks:

### 1. **Catalog Structure Validation**
   - Verifies catalog directory exists
   - Checks for expected subdirectories (bar, quote_tick, futures_contract, option_contract, etc.)

### 2. **File Size Validation (Byte Sizes)**
   - Calculates total catalog size (bytes, MB, GB)
   - Validates individual file sizes against min/max constraints
   - Groups sizes by directory
   - Identifies largest and smallest files
   - Reports size distribution

### 3. **Parquet Schema & Data Type Validation**
   - Reads Parquet file schemas using PyArrow
   - Validates column names and data types
   - Checks against expected schemas
   - Detects schema inconsistencies across files
   - Reports Arrow data types (int64, float64, Decimal128, etc.)
   - Validates type compatibility

### 4. **Instrument Validation**
   - Verifies instruments exist in catalog
   - Groups instruments by type and symbol
   - Checks for expected symbols
   - Validates instrument properties (ID, currency, precision)

### 5. **Bar Data Validation**
   - Validates bar data quality and consistency
   - Checks price ranges (open, high, low, close)
   - Validates volume ranges
   - Checks bar consistency (high >= low, close/open within range)
   - Validates timestamp ordering
   - Detects duplicate timestamps
   - Checks for missing values
   - Validates date range coverage

### 6. **Quote Tick Validation**
   - Validates quote tick data
   - Checks bid/ask price relationships (bid <= ask)
   - Validates missing values

### 7. **Custom Data Validation**
   - Validates custom data types (OI, Greeks, etc.)
   - Reports file counts and sizes
   - Checks schemas for custom data

### 8. **Date Range Validation**
   - Verifies date coverage matches expected ranges
   - Checks for missing dates

### 9. **Data Quality Validation**
   - Checks for missing values
   - Validates price and volume ranges
   - Ensures data completeness

### 10. **Consistency Validation**
   - Validates timestamp ordering
   - Checks for duplicate timestamps
   - Validates bar consistency

## Installation

Ensure required dependencies are installed:

```bash
pip install pyarrow pandas nautilus-trader
```

Or install from requirements:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Validation

```bash
python validate_catalog.py --catalog-path /home/ubuntu/nautilus_data
```

### With Expected Symbols and Date Range

```bash
python validate_catalog.py \
    --catalog-path /home/ubuntu/nautilus_data \
    --expected-symbols NIFTY BANKNIFTY \
    --expected-start-date 2024-01-01 \
    --expected-end-date 2024-01-31
```

### With Size Constraints

```bash
python validate_catalog.py \
    --catalog-path /home/ubuntu/nautilus_data \
    --min-file-size-bytes 1024 \
    --max-file-size-bytes 104857600 \
    --min-total-size-mb 100 \
    --max-total-size-mb 10000
```

### With Price Range Validation

```bash
python validate_catalog.py \
    --catalog-path /home/ubuntu/nautilus_data \
    --min-price 0.0 \
    --max-price 100000.0
```

### Output Results to JSON

```bash
python validate_catalog.py \
    --catalog-path /home/ubuntu/nautilus_data \
    --output-json validation_results.json
```

### Skip Specific Validations (Faster)

```bash
python validate_catalog.py \
    --catalog-path /home/ubuntu/nautilus_data \
    --skip-data-type-validation \
    --skip-byte-size-validation
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--catalog-path` | Path to Nautilus catalog directory | Required |
| `--expected-symbols` | Expected symbols to validate | None |
| `--expected-start-date` | Expected start date (YYYY-MM-DD) | None |
| `--expected-end-date` | Expected end date (YYYY-MM-DD) | None |
| `--min-file-size-bytes` | Minimum file size in bytes | 0 |
| `--max-file-size-bytes` | Maximum file size in bytes | No limit |
| `--min-total-size-mb` | Minimum total catalog size in MB | None |
| `--max-total-size-mb` | Maximum total catalog size in MB | None |
| `--min-price` | Minimum valid price | 0.0 |
| `--max-price` | Maximum valid price | 1000000.0 |
| `--log-level` | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO |
| `--output-json` | Output validation results to JSON file | None |
| `--skip-data-type-validation` | Skip data type validation | False |
| `--skip-byte-size-validation` | Skip byte size validation | False |
| `--skip-schema-validation` | Skip schema validation | False |

## Output

The validator provides:

1. **Console Output**: Detailed validation results with pass/fail status for each check
2. **JSON Output** (optional): Structured validation results in JSON format

### Example Output

```
================================================================================
VALIDATION SUMMARY
================================================================================
Total checks: 10
Passed: 9
Failed: 1

✅ PASS: Catalog Structure
  Catalog structure is valid
  Total files: 1250

✅ PASS: File Sizes
  File size validation passed
  Total size: 1250.50 MB (1.22 GB)
  Total files: 1250

❌ FAIL: Parquet Schemas
  Found 5 schema issues
  Errors (1):
    - Missing required column: ts_event
  Warnings (4):
    - Type mismatch in bar_001.parquet: open expected string, got double
```

## Validation Criteria

The validator checks the following criteria:

- **Data Completeness**: All expected instruments and data exist
- **Schema Validation**: Correct columns and types in Parquet files
- **Data Type Validation**: Arrow types match expected types
- **Byte Size Validation**: File sizes within expected ranges
- **Data Quality**: No missing values, valid ranges
- **Date Range Coverage**: Data covers expected date ranges
- **Data Consistency**: Timestamps ordered, no duplicates, bar consistency

## Integration

The validator can be integrated into CI/CD pipelines:

```bash
#!/bin/bash
python validate_catalog.py \
    --catalog-path /path/to/catalog \
    --expected-symbols NIFTY BANKNIFTY \
    --expected-start-date 2024-01-01 \
    --expected-end-date 2024-01-31 \
    --output-json validation_results.json

# Exit with error if validation failed
if [ $? -ne 0 ]; then
    echo "Validation failed!"
    exit 1
fi
```

## Notes

- PyArrow is required for data type validation. If not available, schema validation will be skipped with a warning.
- The validator samples a subset of files for performance (first 5-10 files per data type).
- Large catalogs may take time to validate. Use skip flags for faster validation of specific aspects.



