# Zerodha Auto Login NSE - Documentation

## Overview

`zerodha_auto_login_nse.py` is the main automation script for fetching, transforming, and processing NSE (National Stock Exchange) stock data from Zerodha Kite Connect API. The script automates the entire workflow from authentication to data transformation, cloud upload, and Nautilus format conversion.

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Configuration](#configuration)
4. [Key Components](#key-components)
5. [Workflow](#workflow)
6. [Functions Reference](#functions-reference)
7. [Usage](#usage)
8. [Data Structures](#data-structures)
9. [Error Handling](#error-handling)
10. [Dependencies](#dependencies)
11. [File Outputs](#file-outputs)
12. [Troubleshooting](#troubleshooting)

---

## Features

- **Automatic Authentication**: OAuth2 login with TOTP (Time-based One-Time Password) support
- **1-Minute Historical Data**: Fetches 1-minute interval data for stocks, options, futures, and equity
- **Data Transformation**: Converts raw data to standardized parquet format with OHLC in paise
- **Local Storage**: Saves transformed data to `/home/ubuntu/raw_data` with organized folder structure
- **DigitalOcean Spaces Upload**: Uploads raw parquet data to cloud storage with automatic retry logic (3 retries with exponential backoff)
- **Nautilus Integration**: Automatically runs Nautilus format transformation for NautilusTrader
- **Nautilus Data Sync**: Syncs Nautilus output to DigitalOcean Spaces using rclone
- **Trading Day Validation**: Skips weekends and NSE holidays
- **Telegram Alerts**: Real-time notifications for all workflow stages
- **Lot Size Filtering**: Only processes symbols with valid lot sizes from CSV
- **Cron Integration**: Designed for automated daily execution
- **Market Hours Filter**: Only keeps data between 9:15 AM and 3:30 PM IST

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    zerodha_auto_login_nse.py                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Authentication (OAuth2 + TOTP)                          │
│     ↓                                                       │
│  2. Data Fetching (ZerodhaDataFetcher)                      │
│     - Cash (Equity)                                         │
│     - Futures                                               │
│     - Options (CE/PE)                                       │
│     ↓                                                       │
│  3. Data Transformation (Parquet Format)                    │
│     - OHLC to paise (×100)                                  │
│     - Date to YYMMDD integer                                │
│     - Time to seconds since midnight                        │
│     - Calculate COI (Change in Open Interest)               │
│     ↓                                                       │
│  4. Local Storage (/home/ubuntu/raw_data)                   │
│     ↓                                                       │
│  5. DigitalOcean Spaces Upload (with 3x retry logic)        │
│     - Space: historical-nse-1-min                           │
│     - Prefix: raw/parquet_data                              │
│     ↓                                                       │
│  6. Nautilus Transformation (transform_nautilus.py)         │
│     - Filter by lot size CSV                                │
│     - Convert to NautilusTrader format                      │
│     ↓                                                       │
│  7. Nautilus Data Sync (rclone to DigitalOcean)             │
│     - Remote: nexus-lake                                    │
│     ↓                                                       │
│  8. Final Output (/media/ubuntu/Dataspace/nautilus_data)    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Configuration

### Credentials (Lines 32-38)

```python
CREDENTIALS = {
    "user_id": "ZM9343",
    "password": "Chetti@23456",
    "totp_key": "WYFGJCB4XNSJNAPTGXI26BGO4A2PTGHF",
    "api_key": "90mo8gkxxg2p0wm0",
    "api_secret": "hy7impfdog70ctuwkx76fuo8h5dincmb"
}
```

⚠️ **Security Note**: Credentials are hardcoded. For production, move to environment variables or secure vault.

### Telegram Settings (Lines 41-43)

```python
TELEGRAM_BOT_TOKEN = "8471418378:AAERXNyXK_SLuAWETfVLInuxIodnQ16MMGU"
TELEGRAM_CHAT_ID = "-1003599016958"
TELEGRAM_BOT_NAME = "@ajai_fullalert_bot"
```

### DigitalOcean Spaces Configuration

```python
DO_ENDPOINT = "https://blr1.digitaloceanspaces.com"
DO_REGION = "blr1"
SPACE_NAME = "historical-nse-1-min"
SPACE_PREFIX = "raw/parquet_data"
```

**Environment Variables** (recommended):
- `DO_SPACES_KEY`: DigitalOcean Spaces access key
- `DO_SPACES_SECRET`: DigitalOcean Spaces secret key

**Local Upload Structure** (`/home/ubuntu/raw_data`):
```
raw_data/
├── cash/
│   └── {symbol}/{year}/{month}/{symbol}_cash_{YYYYMMDD}.parquet
├── futures/
│   └── {symbol}/{year}/{month}/{symbol}_future_{YYYYMMDD}.parquet
└── option/
    ├── {symbol}_call/{year}/{month}/{symbol}_call_{YYYYMMDD}.parquet
    └── {symbol}_put/{year}/{month}/{symbol}_put_{YYYYMMDD}.parquet
```

**DigitalOcean Spaces Upload Structure**:
```
historical-nse-1-min/
├── raw/parquet_data/
│   ├── cash/
│   │   └── {SYMBOL}/{YEAR}/{MONTH}/{symbol}_cash_{YYYYMMDD}.parquet
│   ├── futures/
│   │   └── {SYMBOL}/{YEAR}/{MONTH}/{symbol}_futures_{YYYYMMDD}.parquet
│   └── options/
│       ├── {SYMBOL}-CALL/{YEAR}/{MONTH}/{symbol}_call_{YYYYMMDD}.parquet
│       └── {SYMBOL}-PUT/{YEAR}/{MONTH}/{symbol}_put_{YYYYMMDD}.parquet
```

### Nautilus Sync Configuration

```python
RCLONE_BIN = "/usr/bin/rclone"
REMOTE_NAME = "nexus-lake"
REMOTE_PATH = "nexus-lake/historical/nautilus_trader/data"
```

### File Paths

| Path | Description |
|------|-------------|
| `{script_dir}/stock_namess.csv` | Stock symbols to process |
| `{script_dir}/nse_holidays.csv` | NSE holiday dates (DD-MM-YYYY format) |
| `/home/ubuntu/Stock_automation/marvelquant/NseLotSize.csv` | Lot size CSV for filtering |
| `/home/ubuntu/raw_data` | Raw transformed data output |
| `/media/ubuntu/Dataspace/nautilus_data` | Nautilus transformation output |
| `{script_dir}/logfile_auto_login_{YYYY-MM-DD}.log` | Daily log files |
| `{script_dir}/.env` | Environment file with access token |

---

## Key Components

### 1. Authentication Module

#### `auto_login(user_id, password, totp_key, api_key, api_secret)`

Handles complete OAuth2 authentication flow:

**Step 1: Login**
- POST to `https://kite.zerodha.com/api/login`
- Sends user_id and password
- Receives request_id for 2FA

**Step 2: Two-Factor Authentication (2FA)**
- Generate TOTP using `pyotp.TOTP(totp_key).now()`
- POST to `https://kite.zerodha.com/api/twofa`
- Retry with exponential backoff (2s, 4s, 6s) on failure

**Step 3: Authorization**
- GET `https://kite.trade/connect/login?api_key={api_key}`
- Follow redirects to extract request_token
- Multiple fallback methods:
  - Extract from redirect URL
  - Extract from Location header
  - POST to authorize endpoint
  - Open browser for manual authorization

**Step 4: Session Generation**
- Call `kite.generate_session(request_token, api_secret)`
- Obtain access_token

**Returns**: `(request_token, access_token, kite_instance)` or `None` on failure

#### `start_callback_server(port=5001, timeout=120)`

HTTP server to catch OAuth callback:
- Listens on `http://127.0.0.1:5001/zerodha/callback`
- Extracts request_token from callback URL
- Times out after 120 seconds

#### `update_env_file(api_key, access_token)`

Updates `.env` file with new access token for other scripts to use.

---

### 2. ZerodhaDataFetcher Class

Main class for fetching and transforming 1-minute historical data.

#### `__init__(self, kite_instance)`

- Initializes with KiteConnect instance
- Loads all NSE and NFO instruments
- Creates instruments DataFrame

#### `get_instruments()`

- Fetches all instruments from Zerodha API
- Filters for NSE and NFO exchanges only
- Parses expiry dates

#### `fetch_historical_data(symbol, from_date, to_date)`

Fetches all data types for a symbol:

1. **Cash (Equity)**: EQ instrument type
2. **Futures**: FUT instrument type, multiple expiries
3. **Options (CE/PE)**: All strikes and expiries

#### `fetch_1min_data(instrument_token, from_date, to_date)`

- Calls `kite.historical_data()` with interval="minute"
- Filters for market hours (9:15 AM - 3:30 PM)
- Returns DataFrame with OHLCV and OI data

#### `save_data(df, symbol, instrument_type, base_path, strike=None, expiry=None)`

Saves raw data to parquet files in date-organized folders:
```
raw_parquet/{YEAR}/{MONTH}/{YYYYMMDD}/{TYPE}/{filename}.parquet
```

---

### 3. Data Transformation

#### `_transform_dataframe(df, stock_name, instrument_type, strike=None, expiry=None)`

Applies transformations:

| Field | Transformation |
|-------|----------------|
| `date` | Convert to YYMMDD integer format |
| `time` | Convert to seconds since midnight (integer) |
| `open`, `high`, `low`, `close` | Multiply by 100 (convert to paise) |
| `oi` | Remove commas, convert to integer |
| `symbol` | Create formatted symbol (e.g., `RELIANCE30JAN251000CE`) |
| `strike` | Add strike price for options |
| `expiry` | Add expiry date in YYMMDD format |

**Output Columns by Type**:

- **CASH**: `date, time, symbol, open, high, low, close`
- **FUTURES**: `date, time, symbol, open, high, low, close, volume, oi, coi`
- **OPTIONS**: `date, time, symbol, strike, expiry, open, high, low, close, volume, oi, coi`

#### `_calculate_coi(merged_df, instrument_type)`

Calculates Change in Open Interest (COI):
- Groups by symbol
- Sorts by date and time
- `coi = oi.diff()` (difference from previous row)

#### `_process_futures_transformed(symbol, future_dataframes, output_folder)`

Special handling for futures:
- Groups by trade date
- Keeps only nearest expiry per date
- Merges and saves

#### `_process_options_transformed(symbol, options_dataframes, option_type, output_folder)`

Processes options (CE or PE):
- Merges all strikes and expiries
- Calculates COI
- Saves by date

---

### 4. DigitalOcean Spaces Upload

#### `upload_to_digitalocean_spaces(source_root, instrument_type, max_workers=10)`

Uploads raw data to DigitalOcean Spaces.

**Parameters**:
- `source_root`: Directory containing data (e.g., `/home/ubuntu/raw_data/cash`)
- `instrument_type`: One of `cash`, `futures`, `call`, `put`
- `max_workers`: Parallel upload threads (default: 10)

**Returns**: `(success: bool, uploaded: int, skipped: int, failed: int)`

**Features**:
- Filters to upload only today's files
- Skips existing files with same size
- Parallel uploads with ThreadPoolExecutor

#### `upload_file(local_path, s3_key, max_retries=3)` (Internal)

Uploads a single file with **retry logic**:

**Retry Logic**:
- **Max Retries**: 3 attempts per file
- **Exponential Backoff**: 1s → 2s → 4s between retries
- **Retryable Errors**:
  - Empty error code (connection dropped)
  - Rate limiting (`SlowDown`)
  - Server errors (`ServiceUnavailable`, `InternalError`, `500`, `503`)
  - Timeout/connection errors in message

**Error Handling**:
- Returns `('uploaded', None)` on success
- Returns `('skipped', None)` if file exists with same size
- Returns `('failed', error_details)` on failure

#### `run_digitalocean_upload(raw_data_dir="/home/ubuntu/raw_data")`

Orchestrates upload for all data types:

1. Verify raw data directory exists
2. Upload CASH data
3. Upload FUTURES data
4. Upload CALL options (iterates all `{symbol}_call` directories)
5. Upload PUT options (iterates all `{symbol}_put` directories)
6. Send Telegram alerts for each type

---

### 5. Nautilus Transformation

#### `filter_symbols_with_lot_size(symbols, csv_path)`

Filters symbols to only include those with valid lot sizes:

- Reads `NseLotSize.csv`
- Finds stock name and lot size columns
- Returns only symbols with lot size > 0
- Returns empty list if CSV not found or on error

#### `run_nautilus_transformation(symbols, current_date)`

Runs the Nautilus transformation script:

**Command**:
```bash
python3 /home/ubuntu/Stock_automation/marvelquant/scripts/transformation/transformers/transform_nautilus.py \
  --input-dir /home/ubuntu/raw_data \
  --output-dir /data/nautilus_data \
  --symbols SYMBOL1 SYMBOL2 ... \
  --start-date YYYY-MM-DD \
  --end-date YYYY-MM-DD \
  --types options equity futures \
  --lot-size-csv /home/ubuntu/Stock_automation/marvelquant/NseLotSize.csv \
  --log-level INFO
```

**Timeout**: 8 hours (for processing all symbols)

#### Step-by-Step: Nautilus Transformation (What Happens Internally)
- Prepare inputs
  - Input root: /home/ubuntu/raw_data (produced earlier in this script)
  - Output root (catalog): /data/nautilus_data
  - Symbols list: filtered by lot size from NseLotSize.csv
- Orchestration
  - Entry script: [transform_nautilus.py](file:///home/ubuntu/Stock_automation/marvelquant/scripts/transformation/transformers/transform_nautilus.py)
  - Builds a ParquetDataCatalog at the output directory and configures logging
  - Routes work to modular transformers based on --types (index, futures, options, equity)
- Options transformation and Greeks
  - Transformer: [options_transformer.py](file:///home/ubuntu/Stock_automation/marvelquant/scripts/transformation/transformers/options_transformer.py)
  - Loads spot series from raw_data for the underlying (index or cash) [options_transformer.py:L438-L603](file:///home/ubuntu/Stock_automation/marvelquant/scripts/transformation/transformers/options_transformer.py#L438-L603)
  - Loads India 91‑day T‑bill monthly rate (risk‑free) via provider [rates.py](file:///home/ubuntu/Stock_automation/marvelquant/utils/rates.py) using XML [india_91day_tbill_rates_2018_2025_nautilus.xml](file:///home/ubuntu/Stock_automation/marvelquant/data/static/interest_rates/india_91day_tbill_rates_2018_2025_nautilus.xml)
  - Merges option bars with spot using minute rounding and merge_asof to ensure consistent spot across strikes at the same timestamp [options_transformer.py:L984-L1007](file:///home/ubuntu/Stock_automation/marvelquant/scripts/transformation/transformers/options_transformer.py#L984-L1007)
  - Applies session-aware forward fill and a 2‑hour lookback fallback for missing/invalid spot [options_transformer.py:L1021-L1072](file:///home/ubuntu/Stock_automation/marvelquant/scripts/transformation/transformers/options_transformer.py#L1021-L1072)
  - Sets expiry time to 15:30 IST and computes Time‑To‑Expiry (TTE) in years; filters out non‑positive or too‑small TTE [options_transformer.py:L1034-L1097](file:///home/ubuntu/Stock_automation/marvelquant/scripts/transformation/transformers/options_transformer.py#L1034-L1097)
  - Fetches risk‑free rate for the record date; clamps to 3%–10% or uses 6% default if out of band [options_transformer.py:L1099-L1106](file:///home/ubuntu/Stock_automation/marvelquant/scripts/transformation/transformers/options_transformer.py#L1099-L1106), [options_transformer.py:L606-L625](file:///home/ubuntu/Stock_automation/marvelquant/scripts/transformation/transformers/options_transformer.py#L606-L625)
  - Validates prices with lightweight upper‑bound guards; tracks skip reasons for quality reporting [options_transformer.py:L1113-L1149](file:///home/ubuntu/Stock_automation/marvelquant/scripts/transformation/transformers/options_transformer.py#L1113-L1149)
  - Calculates Greeks and writes Bars + Contracts to the catalog; uses analytic fallback when IV fails [options_transformer.py](file:///home/ubuntu/Stock_automation/marvelquant/scripts/transformation/transformers/options_transformer.py)
- Futures and index transformation
  - Futures: nearest expiry per trade day and bar writing [futures_transformer.py](file:///home/ubuntu/Stock_automation/marvelquant/scripts/transformation/transformers/futures_transformer.py)
  - Index: bar transformation and writing [index_transformer.py](file:///home/ubuntu/Stock_automation/marvelquant/scripts/transformation/transformers/index_transformer.py)
- Outputs
  - NautilusTrader catalog on /data/nautilus_data
  - Log summary includes per‑type totals and options skip‑rate analysis

#### `sync_nautilus_data_to_spaces(nautilus_data_dir)`

Syncs Nautilus output to DigitalOcean Spaces using rclone:

**Command**:
```bash
rclone copy /media/ubuntu/Dataspace/nautilus_data/data nexus-lake:nexus-lake/historical/nautilus_trader/data \
  --update --checksum --checkers 8 --transfers 4 --progress --verbose
```

**Timeout**: 2 hours

---

### 6. Utility Functions

#### `is_trading_day(date_to_check=None)`

Checks if given date is a trading day:
- Returns `False` if weekend (Saturday/Sunday)
- Returns `False` if date is in `nse_holidays.csv`
- Returns `True` otherwise

#### `send_telegram_alert(message)`

Sends notification to Telegram:
- Uses bot token and chat ID from configuration
- 10 second timeout
- Logs warning on failure but doesn't break flow

#### `get_access_token_only()`

Gets access token without fetching data:
- Calls `auto_login()`
- Updates `.env` file
- Useful for manual token refresh

---

## Workflow

### Complete Daily Workflow

```
1. Start (via cron at 4:00 PM on weekdays)
   │
   ├─ Check if trading day
   │  ├─ Weekend? → Exit
   │  └─ Holiday? → Exit
   │
2. Authentication
   │  ├─ Login with user_id/password
   │  ├─ 2FA with TOTP
   │  ├─ Get request_token
   │  └─ Generate access_token
   │
3. Initialize ZerodhaDataFetcher
   │  └─ Load all NSE/NFO instruments
   │
4. Read stock symbols from stock_namess.csv
   │
5. For each symbol:
   │  ├─ Fetch CASH data
   │  ├─ Fetch FUTURES data (all expiries)
   │  └─ Fetch OPTIONS data (CE & PE, all strikes/expiries)
   │
6. Transform downloaded files
   │  ├─ Apply OHLC transformations
   │  ├─ Calculate COI
   │  └─ Save to /home/ubuntu/raw_data
   │
7. Upload to DigitalOcean Spaces
   │  ├─ Upload CASH → historical-nse-1-min/raw/parquet_data/cash/
   │  ├─ Upload FUTURES → historical-nse-1-min/raw/parquet_data/futures/
   │  ├─ Upload CALL OPTIONS → historical-nse-1-min/raw/parquet_data/options/
   │  └─ Upload PUT OPTIONS → historical-nse-1-min/raw/parquet_data/options/
   │
8. Run Nautilus Transformation
   │  ├─ Filter symbols by lot size
   │  ├─ Run transform_nautilus.py
   │  └─ Output to /media/ubuntu/Dataspace/nautilus_data
   │
9. Sync Nautilus Data to DigitalOcean
   │  └─ rclone copy to nexus-lake remote
   │
10. Complete
    └─ Send final Telegram alert
```

---

## Functions Reference

### Authentication Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `auto_login()` | Complete OAuth2 authentication with 2FA | `(request_token, access_token, kite)` or `None` |
| `start_callback_server()` | HTTP server for OAuth callback on port 5001 | `request_token` or `None` |
| `update_env_file()` | Updates .env with access token | `None` |
| `get_access_token_only()` | Gets token without data fetching | `bool` |

### Data Fetching Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `ZerodhaDataFetcher.__init__()` | Initialize fetcher with kite instance | `None` |
| `get_instruments()` | Fetch all NSE/NFO instruments | `DataFrame` |
| `fetch_historical_data()` | Fetch all data types for symbol | `None` |
| `fetch_1min_data()` | Fetch 1-min data for instrument token | `DataFrame` |
| `save_data()` | Save raw data to parquet | `None` |

### Transformation Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `_transform_dataframe()` | Apply OHLC/date/time transformations | `DataFrame` |
| `_calculate_coi()` | Calculate change in OI | `DataFrame` |
| `_read_parquet_files_from_date_folders()` | Read parquet files from folder structure | `list[str]` |
| `_extract_info_from_filename()` | Parse stock, strike, expiry from filename | `(stock, strike, expiry)` |
| `_process_futures_transformed()` | Process futures with nearest expiry logic | `None` |
| `_process_options_transformed()` | Process and merge options data | `None` |
| `process_saved_files_for_transformation()` | Transform all saved files | `None` |
| `save_transformed_data()` | Save transformed data to raw_data | `None` |

### Storage Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `copy_transformed_data_to_local()` | Copy transformed files to local directory | `bool` |

### DigitalOcean Spaces Upload Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `upload_to_digitalocean_spaces()` | Upload raw data to DigitalOcean Spaces | `(success, uploaded, skipped, failed)` |
| `run_digitalocean_upload()` | Upload all data types (cash, futures, options) | `bool` |
| `upload_file()` | Internal: Upload single file with 3x retry | `('uploaded'/'skipped'/'failed', error_details)` |

### Nautilus Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `filter_symbols_with_lot_size()` | Filter symbols by lot size CSV | `list[str]` |
| `run_nautilus_transformation()` | Run Nautilus transformation script | `bool` |
| `sync_nautilus_data_to_spaces()` | Sync Nautilus data to DigitalOcean using rclone | `bool` |

### Utility Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `is_trading_day()` | Check if date is trading day | `bool` |
| `send_telegram_alert()` | Send Telegram notification | `None` |
| `run_data_fetching_session()` | Main execution function | `None` |

---

## Usage

### Basic Usage

```bash
# Run for current day (default)
python3 zerodha_auto_login_nse.py

# Run for specific date range
python3 zerodha_auto_login_nse.py --from-date 2026-01-06 --to-date 2026-01-06

# Skip holiday check (for manual runs)
python3 zerodha_auto_login_nse.py --skip-holiday-check

# Get access token only (no data fetching)
python3 zerodha_auto_login_nse.py --token-only
```

### Command Line Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--token-only` | Only get access token, don't fetch data | `--token-only` |
| `--from-date` | Start date (YYYY-MM-DD) | `--from-date 2026-01-06` |
| `--to-date` | End date (YYYY-MM-DD) | `--to-date 2026-01-06` |
| `--skip-holiday-check` | Skip trading day validation | `--skip-holiday-check` |

### Cron Job Setup

```bash
# Run setup script
chmod +x setup_cron.sh
./setup_cron.sh

# Manual cron entry (4:00 PM on weekdays)
0 16 * * 1-5 /usr/bin/python3 /home/ubuntu/Stock_automation/zerodha_auto_login_nse.py
```

**Schedule**: 4:00 PM (16:00) IST on weekdays (Monday-Friday)

---

## Data Structures

### Input Files

#### stock_namess.csv

```csv
stock_name
RELIANCE
TCS
INFY
HDFCBANK
...
```

#### nse_holidays.csv

```csv
date,description
26-01-2026,Republic Day
15-08-2026,Independence Day
02-10-2026,Gandhi Jayanti
...
```

Format: DD-MM-YYYY (also supports YYYY-MM-DD as fallback)

#### NseLotSize.csv

```csv
Symbol,Nov 2025
RELIANCE,250
TCS,150
INFY,300
...
```

### Output Data Formats

#### Raw Parquet (Before Transformation)

```
raw_parquet/{YEAR}/{MONTH}/{YYYYMMDD}/{TYPE}/{filename}.parquet
```

Columns: `date, open, high, low, close, volume, oi` (datetime format)

#### Transformed Parquet (After Transformation)

**CASH**:
| Column | Type | Description |
|--------|------|-------------|
| date | int | YYMMDD format |
| time | int | Seconds since midnight |
| symbol | str | Stock symbol |
| open | int | Open price in paise |
| high | int | High price in paise |
| low | int | Low price in paise |
| close | int | Close price in paise |

**FUTURES**:
| Column | Type | Description |
|--------|------|-------------|
| date | int | YYMMDD format |
| time | int | Seconds since midnight |
| symbol | str | Stock symbol |
| open | int | Open price in paise |
| high | int | High price in paise |
| low | int | Low price in paise |
| close | int | Close price in paise |
| volume | int | Volume |
| oi | int | Open Interest |
| coi | int | Change in OI |

**OPTIONS (CE/PE)**:
| Column | Type | Description |
|--------|------|-------------|
| date | int | YYMMDD format |
| time | int | Seconds since midnight |
| symbol | str | e.g., RELIANCE30JAN251000CE |
| strike | float | Strike price |
| expiry | int | Expiry date YYMMDD |
| open | int | Open price in paise |
| high | int | High price in paise |
| low | int | Low price in paise |
| close | int | Close price in paise |
| volume | int | Volume |
| oi | int | Open Interest |
| coi | int | Change in OI |

---

## Error Handling

### Retry Logic

#### 2FA Authentication
- 3 retries with exponential backoff (2s, 4s, 6s)
- Handles connection errors and timeouts

#### DigitalOcean Upload
- 3 retries per file with exponential backoff (1s, 2s, 4s)
- Retryable errors:
  - Empty error code (connection dropped)
  - `SlowDown` (rate limiting)
  - `ServiceUnavailable`, `InternalError`, `500`, `503`
  - Timeout/connection errors

### Telegram Alerts

Alerts are sent at every major step:
- Job start
- Authentication success/failure
- Download completion
- Transformation completion
- Upload start/completion (per type)
- Nautilus transformation start/completion
- Nautilus sync completion
- Errors and warnings

### Logging

- Daily log files: `logfile_auto_login_{YYYY-MM-DD}.log`
- Uses `logzero` library
- Logs include timestamps, log levels, and detailed messages

---

## Dependencies

### Python Packages

```
requests
pyotp
pandas
pyarrow
kiteconnect
logzero
boto3
botocore
```

### System Requirements

- Python 3.x
- rclone (for Nautilus sync)
- Network access to:
  - Zerodha API (`kite.zerodha.com`, `kite.trade`)
  - DigitalOcean Spaces (`blr1.digitaloceanspaces.com`)
  - Telegram API (`api.telegram.org`)

### External Scripts

- `transform_nautilus.py` at `marvelquant/scripts/transformation/transformers/`

---

## File Outputs

### Local Storage

```
/home/ubuntu/raw_data/
├── cash/
│   └── {symbol}/
│       └── {year}/
│           └── {month}/
│               └── {symbol}_cash_{YYYYMMDD}.parquet
├── futures/
│   └── {symbol}/
│       └── {year}/
│           └── {month}/
│               └── {symbol}_future_{YYYYMMDD}.parquet
└── option/
    ├── {symbol}_call/
    │   └── {year}/
    │       └── {month}/
    │           └── {symbol}_call_{YYYYMMDD}.parquet
    └── {symbol}_put/
        └── {year}/
            └── {month}/
                └── {symbol}_put_{YYYYMMDD}.parquet
```

### DigitalOcean Spaces

```
historical-nse-1-min/raw/parquet_data/
├── cash/{SYMBOL}/{YEAR}/{MONTH}/{symbol}_cash_{YYYYMMDD}.parquet
├── futures/{SYMBOL}/{YEAR}/{MONTH}/{symbol}_futures_{YYYYMMDD}.parquet
└── options/
    ├── {SYMBOL}-CALL/{YEAR}/{MONTH}/{symbol}_call_{YYYYMMDD}.parquet
    └── {SYMBOL}-PUT/{YEAR}/{MONTH}/{symbol}_put_{YYYYMMDD}.parquet
```

### Nautilus Output

- **Location**: `/data/nautilus_data/`
- **Format**: NautilusTrader catalog format
- **Synced to**: `nexus-lake:nexus-lake/historical/nautilus_trader/data`

---

## Troubleshooting

### Common Issues

#### 1. Authentication Failures

**Problem**: Login fails or token extraction fails

**Solutions**:
- Verify credentials are correct
- Check TOTP key is valid (generate new if needed)
- Ensure API key is active in Zerodha developer console
- Verify redirect URL: `http://127.0.0.1:5001/zerodha/callback`
- Check network connectivity to Zerodha

#### 2. Module Not Found Errors

**Problem**: `ModuleNotFoundError: No module named 'X'`

**Solution**:
```bash
pip3 install <package_name>
# Or use virtual environment
source venv/bin/activate
pip install -r requirements.txt
```

#### 3. Lot Size CSV Not Found

**Problem**: `Lot size CSV file not found`

**Solution**:
- Verify file exists at: `/home/ubuntu/Stock_automation/marvelquant/NseLotSize.csv`
- Check file permissions
- Verify file name is exactly `NseLotSize.csv` (no spaces)

#### 4. Symbols Skipped

**Problem**: Symbols are being skipped during Nautilus transformation

**Solution**:
- Check if symbols have lot sizes in `NseLotSize.csv`
- Verify lot size column is correctly identified (looks for "Nov 2025" or "lot size")
- Check logs for filtering details

#### 5. Trading Day Check Fails

**Problem**: Script exits on valid trading days

**Solution**:
- Use `--skip-holiday-check` flag for manual runs
- Verify `nse_holidays.csv` format (DD-MM-YYYY)
- Check date parsing logic

#### 6. Nautilus Transformation Timeout

**Problem**: Transformation times out after 8 hours

**Solution**:
- Check if too many symbols are being processed
- Verify input data exists at `/home/ubuntu/raw_data`
- Check disk space
- Review transformation script logs

#### 7. Telegram Alerts Not Sending

**Problem**: No Telegram notifications received

**Solution**:
- Verify bot token is correct
- Check chat ID is correct (include `-` for group chats)
- Verify network connectivity
- Check Telegram API status

#### 8. DigitalOcean Spaces Upload Failures

**Problem**: `An error occurred () when calling the PutObject operation`

**Cause**: Network connection dropped or DigitalOcean returned empty error code (transient issue).

**Solution**: The script has automatic retry logic (3 attempts with exponential backoff: 1s, 2s, 4s). If still failing:
- Check network connectivity to DigitalOcean
- Verify credentials: `DO_SPACES_KEY` and `DO_SPACES_SECRET`
- Check DigitalOcean Spaces quota/limits
- Check [status.digitalocean.com](https://status.digitalocean.com) for service issues
- Check file size (large files may timeout)

**Problem**: `ConnectionError` after multiple attempts

**Solution**:
- Check internet connection stability
- Reduce parallel workers (default: 10) to lower network load
- Check if DigitalOcean is rate-limiting requests

**Problem**: `AccessDenied` error

**Solution**:
- Verify Space name is correct: `historical-nse-1-min`
- Check API key permissions in DigitalOcean console
- Ensure bucket/Space exists

#### 9. Nautilus Sync Fails

**Problem**: rclone sync to DigitalOcean fails

**Solution**:
- Verify rclone is installed at `/usr/bin/rclone`
- Check rclone configuration for `nexus-lake` remote
- Verify source directory exists: `/media/ubuntu/Dataspace/nautilus_data/data`
- Check rclone logs for detailed errors

### Debug Mode

Enable debug logging by modifying log level:

```python
from logzero import logger, logfile, setup_logger
import logging

logger = setup_logger(name="zerodha", level=logging.DEBUG)
```

### Manual Testing

```bash
# Test authentication only
python3 zerodha_auto_login_nse.py --token-only

# Test with single date
python3 zerodha_auto_login_nse.py --from-date 2026-01-06 --to-date 2026-01-06 --skip-holiday-check

# Check logs (full path)
tail -f /home/ubuntu/Stock_automation/logfile_auto_login_2026-01-20.log

# Check DigitalOcean Spaces connectivity
aws s3 ls s3://historical-nse-1-min/ --endpoint-url https://blr1.digitaloceanspaces.com

# Test rclone connection
rclone lsd nexus-lake:nexus-lake/historical/nautilus_trader/
```

---

## Security Recommendations

1. **Environment Variables**: Move credentials to environment variables:
   ```python
   import os
   CREDENTIALS = {
       "user_id": os.getenv("ZERODHA_USER_ID"),
       "password": os.getenv("ZERODHA_PASSWORD"),
       # ...
   }
   ```

2. **Secure Vault**: Use AWS Secrets Manager, HashiCorp Vault, etc.

3. **Encrypted Storage**: Store credentials in encrypted files

4. **Access Control**: Restrict file permissions:
   ```bash
   chmod 600 zerodha_auto_login_nse.py
   ```

5. **Credential Rotation**: Regularly rotate API keys and passwords

---

## Version History

- **Latest (January 2026)**:
  - Added DigitalOcean Spaces upload with automatic retry logic (3 retries with exponential backoff)
  - Added `upload_to_digitalocean_spaces()` and `run_digitalocean_upload()` functions
  - Added Nautilus data sync to DigitalOcean Spaces using rclone
  - Updated Telegram bot configuration
  - 8-hour timeout for Nautilus transformation
  - Integrated lot size filtering for Nautilus transformation
- Removed APScheduler dependency (using cron instead)
- Added lot size CSV filtering (skips symbols without lot sizes)
- Added Telegram alerts for all workflow stages
- Added trading day validation
- Market hours filtering (9:15 AM - 3:30 PM)

---

## Support

For issues or questions:
1. Check log files: `/home/ubuntu/Stock_automation/logfile_auto_login_{YYYY-MM-DD}.log`
2. Review Telegram alerts for error messages
3. Verify all input files exist and are correctly formatted
4. Test authentication separately: `--token-only`
5. Review Nautilus transformation logs: `/home/ubuntu/Stock_automation/marvelquant/logs/transform_nautilus.log`

**Quick Log Commands**:
```bash
# View today's log
tail -f /home/ubuntu/Stock_automation/logfile_auto_login_$(date +%Y-%m-%d).log

# View last 100 lines
tail -n 100 /home/ubuntu/Stock_automation/logfile_auto_login_2026-01-20.log

# Search for errors in all logs
grep -i "error\|failed\|exception" /home/ubuntu/Stock_automation/logfile_auto_login_*.log

# Count log files
ls -1 /home/ubuntu/Stock_automation/logfile_auto_login_*.log | wc -l
```

---

**Last Updated**: January 2026  
**Script Version**: Current  
**Python Version**: 3.x  
**Maintainer**: Stock Automation Team 
