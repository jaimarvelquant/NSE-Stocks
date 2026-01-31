# Equity Transformation Command

## Paths
- **Input Folder**: `C:\Users\Admin\Desktop\db_parquet\parquet_data`
- **Output Folder**: `C:\Users\Admin\Desktop\us_indices\marvelquant\nautilus_data`
- **Lot Size CSV**: `C:\Users\Admin\Desktop\us_indices\marvelquant\Nse  Lot Size.csv`

## Example Commands

### Transform a single stock (ABB):
```bash
py scripts/transformation/transformers/transform_nautilus.py `
    --input-dir "C:\Users\Admin\Desktop\db_parquet\parquet_data" `
    --output-dir "C:\Users\Admin\Desktop\us_indices\marvelquant\nautilus_data" `
    --symbols ABB `
    --types equity futures options`
    --lot-size-csv "Nse  Lot Size.csv" `
    --start-date 2025-09-01 `
    --end-date 2025-09-30
```

### Transform multiple stocks:
```bash
python scripts/transformation/transformers/transform_nautilus.py `
    --input-dir "C:\Users\Admin\Desktop\db_parquet\parquet_data" `
    --output-dir "C:\Users\Admin\Desktop\us_indices\marvelquant\nautilus_data" `
    --symbols ABB AARTIIND ABBOTINDIA `
    --types equity `
    --lot-size-csv "Nse  Lot Size.csv" `
    --start-date 2025-09-01 `
    --end-date 2025-09-30
```

### Transform all available stocks (you'll need to list them):
```bash
python scripts/transformation/transformers/transform_nautilus.py `
    --input-dir "C:\Users\Admin\Desktop\db_parquet\parquet_data" `
    --output-dir "C:\Users\Admin\Desktop\us_indices\marvelquant\nautilus_data" `
    --symbols ABB AARTIIND ABBOTINDIA ABCAPITAL ABFRL ACC ADANIENT ADANIPORTS `
    --types equity `
    --lot-size-csv "Nse  Lot Size.csv" `
    --start-date 2025-09-01 `
    --end-date 2025-09-30
```

## Notes
- The input folder structure is: `cash/2025/09/{symbol}/`
- The output will be in: `nautilus_data/equity/{symbol}.nse/`
- Lot sizes are automatically picked from the "Lot Size (Nov 2025)" column in the CSV
- The system handles the nested folder structure automatically

