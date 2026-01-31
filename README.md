# Stock Automation Project

## Overview

This project automates the process of fetching, transforming, and processing NSE (National Stock Exchange) stock data from Zerodha Kite Connect API. The system runs daily to collect 1-minute historical data for stocks, options, futures, and equity instruments, transforms the data into multiple formats, and stores it locally for further processing.

## Project Structure

```
/home/ubuntu/Stock_automation/
├── zerodha_auto_login_nse.py      # Main automation script
├── stock_namess.csv                # List of stock symbols to process (196 symbols)
├── nse_holidays.csv                # NSE trading holidays calendar
├── setup_cron.sh                   # Cron job setup script
├── transformed_data.py             # Data transformation utilities
├── fetch_historical_data_cu.py     # Historical data fetcher (alternative)
├── upload_parquet_ssh.py            # SSH upload utility (legacy)
├── upload_to_digitalocean_spaces.py # DigitalOcean Spaces upload utility
├── marvelquant/                    # Data processing and transformation modules
│   ├── NseLotSize.csv              # Lot sizes for NSE instruments
│   ├── scripts/
│   │   ├── transformation/         # Data transformation scripts
│   │   │   └── transformers/
│   │   │       └── transform_nautilus.py  # Nautilus format transformer
│   │   └── validation/             # Data validation scripts
│   └── utils/                      # Utility functions
├── raw_parquet/                    # Raw parquet data storage
│   ├── raw_parquet/                # Original downloaded data
│   └── finaloutput/                # Transformed data output
└── venv/                           # Python virtual environment
```

## Main Components

### 1. `zerodha_auto_login_nse.py` (Main Script)

**Purpose**: Main automation script that orchestrates the entire data fetching and transformation workflow.

**Key Features**:
- Automatic Zerodha Kite Connect authentication with OAuth2
- TOTP (Time-based One-Time Password) handling
- 1-minute historical data fetching for stocks, options, futures, and equity
- Data transformation to parquet format
- Local data storage at `/home/ubuntu/raw_data`
- Nautilus format transformation integration
- Telegram alerts for workflow status
- Trading day validation (skips weekends and holidays)

**Workflow**:
1. **Authentication**: Auto-login to Zerodha using credentials and TOTP
2. **Data Fetching**: Downloads 1-minute historical data for all symbols in `stock_namess.csv`
3. **Transformation**: Converts raw data to parquet format with standardized structure
4. **Local Storage**: Copies transformed data to `/home/ubuntu/raw_data` with folder structure:
   ```
   /home/ubuntu/raw_data/
   ├── options/
   │   ├── {SYMBOL}-CALL/
   │   │   └── YYYY/MM/{symbol}_call_{YYYYMMDD}.parquet
   │   └── {SYMBOL}-PUT/
   │       └── YYYY/MM/{symbol}_put_{YYYYMMDD}.parquet
   ├── futures/
   │   └── {SYMBOL}/
   │       └── YYYY/MM/{symbol}_futures_{YYYYMMDD}.parquet
   └── equity/
       └── {SYMBOL}/
           └── YYYY/MM/{symbol}_equity_{YYYYMMDD}.parquet
   ```
5. **Nautilus Transformation**: Runs Nautilus format transformation for compatible symbols
6. **Telegram Notifications**: Sends status updates at each stage

**Command Line Arguments**:
```bash
# Get access token only
python3 zerodha_auto_login_nse.py --token-only

# Fetch data for specific date range
python3 zerodha_auto_login_nse.py --from-date 2026-01-06 --to-date 2026-01-06

# Skip holiday check (for manual runs on non-trading days)
python3 zerodha_auto_login_nse.py --skip-holiday-check
```

**Configuration**:
- **Credentials**: Hardcoded in script (lines 28-34)
  - User ID, Password, TOTP Key
  - API Key and Secret
- **Telegram**: Bot token and chat ID configured (lines 37-39)
- **Logging**: Daily log files (`logfile_auto_login_YYYY-MM-DD.log`)

### 2. `setup_cron.sh`

**Purpose**: Sets up a cron job to run the main script automatically.

**Schedule**: Runs at 4:00 PM (16:00) on weekdays (Monday-Friday)

**Usage**:
```bash
chmod +x setup_cron.sh
./setup_cron.sh
```

**Cron Entry**:
```
0 16 * * 1-5 /usr/bin/python3 /home/ubuntu/Stock_automation/zerodha_auto_login_nse.py
```

### 3. `stock_namess.csv`

**Purpose**: Contains the list of stock symbols to process.

**Format**:
- Single column: `stock_name`
- Contains 196 stock symbols (e.g., AARTIIND, ABB, ABBOTINDIA, etc.)

**Usage**: The main script reads this file to determine which symbols to fetch data for.

### 4. `nse_holidays.csv`

**Purpose**: Contains NSE trading holidays for the year.

**Format**:
- Single column: `date` (format: DD-MM-YYYY)
- Used by `is_trading_day()` function to skip data fetching on holidays

### 5. Nautilus Transformation

**Location**: `marvelquant/scripts/transformation/transformers/transform_nautilus.py`

**Purpose**: Transforms data into NautilusTrader catalog format for backtesting and trading.

**Input**: `/home/ubuntu/raw_data` (transformed parquet files)
**Output**: `/media/ubuntu/Dataspace/nautilus_modal`

**Features**:
- Processes options, equity, and futures data types
- Filters symbols based on lot size availability in `NseLotSize.csv`
- Skips symbols without lot sizes (no default values)
- Calculates Greeks for options data
- Creates Nautilus contract objects

**Manual Execution**:
```bash
cd /home/ubuntu/Stock_automation
SYMBOLS=$(python3 -c "import pandas as pd; df = pd.read_csv('stock_namess.csv'); print(' '.join(df['stock_name'].dropna().str.upper().tolist()))")
/usr/bin/python3 /home/ubuntu/Stock_automation/marvelquant/scripts/transformation/transformers/transform_nautilus.py \
  --input-dir /home/ubuntu/raw_data \
  --output-dir /media/ubuntu/Dataspace/nautilus_data \
  --symbols $SYMBOLS \
  --start-date 2026-01-06 \
  --end-date 2026-01-06 \
  --types options equity futures \
  --lot-size-csv "/home/ubuntu/Stock_automation/marvelquant/NseLotSize.csv" \
  --log-level INFO
```

### 6. `NseLotSize.csv`

**Location**: `/home/ubuntu/Stock_automation/marvelquant/NseLotSize.csv`

**Purpose**: Contains lot sizes for NSE instruments.

**Usage**: 
- Used by Nautilus transformation to determine contract sizes
- Symbols without lot sizes in this CSV are skipped during transformation
- Contains 214 lot size entries

## Data Flow

```
1. Zerodha Kite Connect API
   ↓
2. Raw Data Download (1-minute intervals)
   ↓
3. Transformation to Parquet Format
   ↓
4. Local Storage (/home/ubuntu/raw_data)
   ↓
5. Nautilus Format Transformation
   ↓
6. Nautilus Output (/media/ubuntu/Dataspace/nautilus_modal)
```

## Key Functions

### `zerodha_auto_login_nse.py`

- **`auto_login()`**: Handles OAuth2 authentication with Zerodha
- **`ZerodhaDataFetcher`**: Class for fetching historical data
  - `fetch_historical_data()`: Downloads 1-minute data for a symbol
  - `process_saved_files_for_transformation()`: Transforms raw data to parquet
- **`copy_transformed_data_to_local()`**: Copies transformed data to local directory
- **`filter_symbols_with_lot_size()`**: Filters symbols that have lot sizes in CSV
- **`run_nautilus_transformation()`**: Executes Nautilus transformation subprocess
- **`is_trading_day()`**: Validates if a date is a trading day
- **`send_telegram_alert()`**: Sends notifications via Telegram

## Dependencies

### Python Packages
- `pyotp` - TOTP authentication
- `pandas` - Data manipulation
- `pyarrow` - Parquet file handling
- `kiteconnect` - Zerodha API client
- `logzero` - Logging
- `requests` - HTTP requests (Telegram API)
- `pathlib` - Path handling
- `shutil` - File operations
- `subprocess` - Process execution

### Installation
```bash
pip3 install pyotp pandas pyarrow kiteconnect logzero requests
```

Or use virtual environment:
```bash
cd /home/ubuntu/Stock_automation
source venv/bin/activate
pip install -r requirements.txt  # If requirements.txt exists
```

## Logging

- **Location**: `/home/ubuntu/Stock_automation/logfile_auto_login_YYYY-MM-DD.log`
- **Format**: Daily log files with timestamp and log level
- **Nautilus Logs**: `/home/ubuntu/Stock_automation/marvelquant/logs/transform_nautilus.log`

## Telegram Alerts

The system sends Telegram notifications for:
- ✅ Workflow start/completion
- 🔄 Data fetching progress
- 📤 Data copy operations
- ✅ Transformation completion
- ❌ Errors and failures
- ⚠️ Warnings (missing lot sizes, skipped symbols)

## File Locations

### Input Files
- Stock symbols: `/home/ubuntu/Stock_automation/stock_namess.csv`
- Lot sizes: `/home/ubuntu/Stock_automation/marvelquant/NseLotSize.csv`
- Holidays: `/home/ubuntu/Stock_automation/nse_holidays.csv`

### Output Directories
- Transformed data: `/home/ubuntu/raw_data/`
- Nautilus output: `/media/ubuntu/Dataspace/nautilus_modal/`
- Raw parquet: `/home/ubuntu/Stock_automation/raw_parquet/`

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install missing Python packages
   ```bash
   pip3 install <package_name>
   ```

2. **Lot Size CSV Not Found**: Verify file exists at:
   `/home/ubuntu/Stock_automation/marvelquant/NseLotSize.csv`

3. **Authentication Failures**: 
   - Check credentials in script
   - Verify TOTP key is correct
   - Ensure API key/secret are valid

4. **Trading Day Check**: Use `--skip-holiday-check` for manual runs on non-trading days

5. **Symbols Skipped**: Check if symbols have lot sizes in `NseLotSize.csv`

## Manual Execution

### Run for specific date:
```bash
cd /home/ubuntu/Stock_automation
python3 zerodha_auto_login_nse.py --from-date 2026-01-06 --to-date 2026-01-06 --skip-holiday-check
```

### Run Nautilus transformation only:
```bash
# Get all symbols from CSV
SYMBOLS=$(python3 -c "import pandas as pd; df = pd.read_csv('stock_namess.csv'); print(' '.join(df['stock_name'].dropna().str.upper().tolist()))")

# Run transformation
/usr/bin/python3 /home/ubuntu/Stock_automation/marvelquant/scripts/transformation/transformers/transform_nautilus.py \
  --input-dir /home/ubuntu/raw_data \
  --output-dir /media/ubuntu/Dataspace/nautilus_data \
  --symbols $SYMBOLS \
  --start-date 2026-01-06 \
  --end-date 2026-01-06 \
  --types options equity futures \
  --lot-size-csv "/home/ubuntu/Stock_automation/marvelquant/NseLotSize.csv" \
  --log-level INFO
```

## Cron Job Management

### View current cron jobs:
```bash
crontab -l
```

### Edit cron jobs:
```bash
crontab -e
```

### Remove cron job:
```bash
crontab -l | grep -v "zerodha_auto_login_nse.py" | crontab -
```

## Security Notes

⚠️ **Important**: The script contains hardcoded credentials. For production use:
- Move credentials to environment variables
- Use `.env` file with proper access controls
- Never commit credentials to version control
- Rotate API keys regularly

## Support Files

- **`transformed_data.py`**: Utility functions for data transformation
- **`fetch_historical_data_cu.py`**: Alternative data fetcher (uses .env for credentials)
- **`upload_parquet_ssh.py`**: Legacy SSH upload utility
- **`upload_to_digitalocean_spaces.py`**: DigitalOcean Spaces upload utility

## Version History

- **Latest**: Integrated Nautilus transformation with lot size filtering
- Removed APScheduler dependency (using cron instead)
- Removed SSH upload functionality (using local storage)
- Added lot size CSV filtering (skips symbols without lot sizes)

## Contact & Maintenance

For issues or questions, check:
- Log files in `/home/ubuntu/Stock_automation/`
- Telegram alerts for real-time status
- Nautilus transformation logs in `marvelquant/logs/`

---

**Last Updated**: January 2026
**Main Script**: `zerodha_auto_login_nse.py`
**Automation**: Cron job at 4:00 PM on weekdays

