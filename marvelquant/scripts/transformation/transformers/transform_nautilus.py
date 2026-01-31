#!/usr/bin/env python3
"""
Nautilus Data Transformation Orchestration CLI.

Thin wrapper that parses CLI arguments and delegates to modular transformers.
No business logic - only wiring and argument parsing.
"""

import sys
import shutil
from pathlib import Path
from datetime import date
from typing import Optional, Set
import argparse
import logging
import os
import posixpath
import tempfile

import paramiko

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add transformers directory to path for direct imports
TRANSFORMERS_DIR = Path(__file__).parent
if str(TRANSFORMERS_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSFORMERS_DIR))

# Add utils directory to path
UTILS_DIR = PROJECT_ROOT / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

# Import transformers - use direct imports since we added paths
from common import configure_logging
from index_transformer import transform_index_bars
from futures_transformer import transform_futures_bars
from options_transformer import transform_options_bars
from equity_transformer import transform_equity_bars

# Import contract_generators
from contract_generators import set_lot_size_csv_path

logger = logging.getLogger(__name__)


def _sftp_mkdir_p(sftp: paramiko.SFTPClient, remote_directory: str) -> None:
    """
    Recursively create remote directories if they do not exist.
    """
    if remote_directory in ("", "/"):
        return
    try:
        sftp.stat(remote_directory)
        return
    except IOError:
        parent, _ = posixpath.split(remote_directory.rstrip("/"))
        if parent and parent != remote_directory:
            _sftp_mkdir_p(sftp, parent)
        sftp.mkdir(remote_directory)


def download_input_from_ssh(remote_dir: str, local_dir: Path) -> None:
    """
    Download input data from the remote SSH server to a local temporary directory.
    
    Args:
        remote_dir: Remote directory path on SSH server (e.g., "/home/ubuntu/raw_data")
        local_dir: Local temporary directory to download files to
    """
    # SSH server configuration
    host = "192.168.173.175"
    port = 22
    username = "ubuntu"
    password = "data"
    
    # Normalize remote path (ensure forward slashes)
    remote_dir = remote_dir.replace("\\", "/")
    if not remote_dir.startswith("/"):
        remote_dir = "/" + remote_dir
    
    logger.info(f"Downloading input data from {host}:{remote_dir} to {local_dir}")
    
    transport = paramiko.Transport((host, port))
    try:
        transport.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        
        # Check if remote directory exists
        try:
            sftp.stat(remote_dir)
        except IOError as e:
            logger.error(f"Remote directory does not exist on SSH server: {remote_dir}")
            logger.error(f"Please verify the path exists on {host} as user {username}")
            logger.error(f"Error details: {e}")
            raise FileNotFoundError(f"Remote directory not found: {remote_dir}") from e
        
        # Recursively download directory
        def _download_recursive(remote_path: str, local_path: Path):
            try:
                attrs = sftp.listdir_attr(remote_path)
                local_path.mkdir(parents=True, exist_ok=True)
                
                for attr in attrs:
                    remote_item = posixpath.join(remote_path, attr.filename)
                    local_item = local_path / attr.filename
                    
                    if attr.st_mode & 0o040000:  # Directory
                        _download_recursive(remote_item, local_item)
                    else:  # File
                        logger.debug(f"Downloading {remote_item} -> {local_item}")
                        sftp.get(remote_item, str(local_item))
            except IOError as e:
                logger.error(f"Error accessing {remote_path}: {e}")
                raise
        
        _download_recursive(remote_dir, local_dir)
        sftp.close()
        logger.info(f"Completed download of input data from {host}:{remote_dir}")
    finally:
        transport.close()


def upload_file_to_ssh(local_file: Path, remote_path: str) -> None:
    """
    Upload a single file to the remote SSH server.
    
    Args:
        local_file: Local file path to upload
        remote_path: Remote file path on SSH server
    """
    # SSH server configuration
    host = "192.168.173.175"
    port = 22
    username = "ubuntu"
    password = "data"
    
    if not local_file.exists():
        logger.warning(f"Local file does not exist, nothing to upload: {local_file}")
        return
    
    logger.info(f"Uploading log file {local_file} to {host}:{remote_path}")
    
    transport = paramiko.Transport((host, port))
    try:
        transport.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        
        # Create remote directory if it doesn't exist
        remote_dir = posixpath.dirname(remote_path)
        if remote_dir:
            _sftp_mkdir_p(sftp, remote_dir)
        
        sftp.put(str(local_file), remote_path)
        sftp.close()
        logger.info(f"Successfully uploaded log file to {host}:{remote_path}")
    finally:
        transport.close()


def upload_output_to_ssh(local_dir: Path, remote_dir: str) -> None:
    """
    Upload the generated Nautilus catalog to the remote SSH server.

    Args:
        local_dir: Local directory containing files to upload
        remote_dir: Remote directory path on SSH server
    """
    # SSH server configuration
    host = "192.168.173.175"
    port = 22
    username = "ubuntu"
    password = "data"

    if not local_dir.exists():
        logger.warning(f"Local output directory does not exist, nothing to upload: {local_dir}")
        return

    logger.info(f"Uploading Nautilus catalog from {local_dir} to {host}:{remote_dir}")

    transport = paramiko.Transport((host, port))
    try:
        transport.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)

        local_root_str = str(local_dir)
        for root, dirs, files in os.walk(local_root_str):
            rel_path = os.path.relpath(root, local_root_str)
            if rel_path == ".":
                target_remote_dir = remote_dir
            else:
                target_remote_dir = posixpath.join(remote_dir, rel_path.replace("\\", "/"))

            _sftp_mkdir_p(sftp, target_remote_dir)

            for filename in files:
                local_path = os.path.join(root, filename)
                remote_path = posixpath.join(target_remote_dir, filename)
                logger.debug(f"Uploading {local_path} -> {remote_path}")
                sftp.put(local_path, remote_path)

        sftp.close()
        logger.info(f"Completed upload of Nautilus catalog to {host}:{remote_dir}")
    finally:
        transport.close()


def parse_expiry_dates(expiry_strings: list[str]) -> Optional[Set[date]]:
    """Parse expiry date strings into set of date objects."""
    if not expiry_strings:
        return None
    
    expiry_set = set()
    for exp_str in expiry_strings:
        try:
            expiry_set.add(date.fromisoformat(exp_str))
        except ValueError:
            logger.warning(f"Invalid expiry date format: {exp_str}, skipping")
    
    return expiry_set if expiry_set else None


def main():
    """Main CLI entrypoint."""
    # Input and output paths are on the server (server-to-server only)
    default_input_dir = Path("/home/ubuntu/raw_data")
    default_output_dir = Path("/home/ubuntu/nautilus_data")
    
    parser = argparse.ArgumentParser(
        description="Transform NSE data to Nautilus catalog (Modular Pattern) - Server-to-Server Only"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir,
        help=f"Input directory with raw data on server (default: {default_input_dir})"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Output directory for Nautilus catalog on server (default: {default_output_dir})"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["NIFTY", "BANKNIFTY"],
        help="Symbols to transform"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2024-01-01",
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-01-31",
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean output directory before starting"
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["index", "futures", "options", "equity"],
        default=["index", "futures", "options"],
        help="Data types to transform"
    )
    parser.add_argument(
        "--lot-size-csv",
        type=Path,
        default=None,
        help="Path to CSV file containing stock lot sizes (must have 'November 2025' column)"
    )
    parser.add_argument(
        "--default-lot-size",
        type=int,
        default=1,
        help="Default lot size to use for equity symbols when not found in CSV (default: 1)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level for console/file handlers (default: INFO)"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Log file path. Use '-' to disable file logging. Default: PROJECT_ROOT/logs/transform_nautilus.log (rotates daily)"
    )
    parser.add_argument(
        "--log-greeks-debug",
        action="store_true",
        help="Enable verbose per-record Greeks diagnostics (intended for narrow debug runs)"
    )
    parser.add_argument(
        "--expiry-dates",
        nargs="+",
        help="Limit options processing to these expiry dates (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--strike-min",
        type=float,
        help="Minimum strike (rupees) to process for options Greeks"
    )
    parser.add_argument(
        "--strike-max",
        type=float,
        help="Maximum strike (rupees) to process for options Greeks"
    )
    parser.add_argument(
        "--strike-center",
        type=float,
        help="Center strike (rupees) for symmetric strike band filtering"
    )
    parser.add_argument(
        "--strike-band",
        type=float,
        default=1000.0,
        help="Strike band width (rupees) for symmetric filtering (default: 1000)"
    )
    
    args = parser.parse_args()

    # Server paths (normalize to use forward slashes)
    input_dir = Path(str(args.input_dir).replace("\\", "/"))
    output_dir = Path(str(args.output_dir).replace("\\", "/"))
    
    # Ensure paths are absolute
    if not input_dir.is_absolute():
        input_dir = Path("/") / input_dir
    if not output_dir.is_absolute():
        output_dir = Path("/") / output_dir
    
    # Verify input directory exists on server
    if not input_dir.exists():
        logger.error(f"Input directory does not exist on server: {input_dir}")
        return 1
    
    if not input_dir.is_dir():
        logger.error(f"Input path is not a directory: {input_dir}")
        return 1
    
    logger.info(f"Working directly on server (server-to-server only):")
    logger.info(f"  Input directory: {input_dir}")
    logger.info(f"  Output directory: {output_dir}")
    
    # Configure logging
    # Set default log file if not provided and not disabled
    if args.log_file == Path('-'):
        log_file = None
    elif args.log_file is None:
        # Default log file location
        default_log_dir = PROJECT_ROOT / "logs"
        default_log_dir.mkdir(parents=True, exist_ok=True)
        log_file = default_log_dir / "transform_nautilus.log"
    else:
        log_file = args.log_file
    
    resolved_log_file = configure_logging(args.log_level, log_file)
    if resolved_log_file:
        logger.info(f"Logging to file: {resolved_log_file}")
    
    # Store resolved log file for summary
    final_log_file = resolved_log_file
    
    # Clean output directory if requested
    if args.clean:
        if output_dir.exists():
            logger.info(f"Cleaning output directory: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            logger.info(f"Output directory does not exist yet: {output_dir}")
    
    # Ensure output directory exists on server
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct ParquetDataCatalog using the server output directory
    catalog = ParquetDataCatalog(output_dir)
    logger.info(f"Using catalog on server: {output_dir}")
    
    # Load lot sizes from CSV if provided
    resolved_csv_path = None
    if args.lot_size_csv:
        # Normalize the path - remove quotes if present and handle spaces
        csv_path_str = str(args.lot_size_csv).strip().strip('"').strip("'")
        csv_path_input = Path(csv_path_str)
        csv_path = None
        
        logger.info(f"Looking for lot size CSV: {csv_path_input}")
        logger.debug(f"  Input string: {args.lot_size_csv}")
        logger.debug(f"  Normalized string: {csv_path_str}")
        logger.debug(f"  Path object: {csv_path_input}")
        logger.debug(f"  Is absolute: {csv_path_input.is_absolute()}")
        logger.debug(f"  Exists (direct): {csv_path_input.exists()}")
        
        # Strategy 1: Try the exact path as provided (handle both absolute and relative)
        if csv_path_input.is_absolute():
            if csv_path_input.exists():
                csv_path = csv_path_input.resolve()
                logger.info(f"✅ Found CSV at provided absolute path: {csv_path}")
            else:
                # Try with resolved path in case of symlinks
                resolved = csv_path_input.resolve()
                if resolved.exists() and resolved != csv_path_input:
                    csv_path = resolved
                    logger.info(f"✅ Found CSV at resolved absolute path: {csv_path}")
                else:
                    logger.debug(f"Absolute path does not exist: {csv_path_input} (resolved: {resolved})")
        else:
            # Try resolving relative path
            resolved_relative = csv_path_input.resolve()
            if resolved_relative.exists():
                csv_path = resolved_relative
                logger.info(f"✅ Found CSV at resolved relative path: {csv_path}")
            else:
                logger.debug(f"Resolved relative path does not exist: {resolved_relative}")
        
        # Strategy 2: If not found, try with just the filename in common locations
        if csv_path is None or not csv_path.exists():
            filename = csv_path_input.name
            # Calculate Stock_automation root (one level up from PROJECT_ROOT which is marvelquant)
            stock_automation_root = PROJECT_ROOT.parent if PROJECT_ROOT.name == "marvelquant" else PROJECT_ROOT
            search_locations = [
                Path.cwd() / filename,
                PROJECT_ROOT / filename,  # marvelquant directory
                stock_automation_root / filename,  # Stock_automation directory
                stock_automation_root / "marvelquant" / filename,  # Stock_automation/marvelquant
                Path(csv_path_input).parent / filename if csv_path_input.parent else None,
            ]
            
            for location in search_locations:
                if location and location.exists():
                    csv_path = location.resolve()
                    logger.info(f"Found CSV at search location: {csv_path}")
                    break
        
        # Strategy 3: Fuzzy matching - search for any file with "lot" and "size" in name
        if csv_path is None or not csv_path.exists():
            filename_base = csv_path_input.name.lower().replace(" ", "").replace("_", "").replace("-", "")
            # Calculate Stock_automation root (one level up from PROJECT_ROOT which is marvelquant)
            stock_automation_root = PROJECT_ROOT.parent if PROJECT_ROOT.name == "marvelquant" else PROJECT_ROOT
            search_dirs = [
                Path.cwd(),
                PROJECT_ROOT,  # marvelquant directory
                stock_automation_root,  # Stock_automation directory
                stock_automation_root / "marvelquant",  # Stock_automation/marvelquant
                Path(csv_path_input).parent if csv_path_input.parent and csv_path_input.parent.exists() else None,
            ]
            
            for search_dir in search_dirs:
                if search_dir and search_dir.exists():
                    for file in search_dir.rglob("*lot*size*.csv"):
                        file_base = file.name.lower().replace(" ", "").replace("_", "").replace("-", "")
                        if filename_base in file_base or file_base in filename_base:
                            csv_path = file.resolve()
                            logger.info(f"Found CSV with fuzzy match: {csv_path}")
                            break
                    if csv_path and csv_path.exists():
                        break
        
        # Final check and set
        if csv_path and csv_path.exists():
            resolved_csv_path = csv_path.resolve()
            set_lot_size_csv_path(resolved_csv_path)
            logger.info(f"✅ Successfully loaded lot sizes from CSV: {resolved_csv_path}")
        else:
            logger.error(f"❌ Lot size CSV file not found: {args.lot_size_csv}")
            logger.error(f"   Searched in:")
            logger.error(f"     - Provided path: {csv_path_input}")
            logger.error(f"     - Current directory: {Path.cwd()}")
            logger.error(f"     - Project root: {PROJECT_ROOT}")
            stock_automation_root = PROJECT_ROOT.parent if PROJECT_ROOT.name == "marvelquant" else PROJECT_ROOT
            logger.error(f"     - Stock_automation root: {stock_automation_root}")
            logger.error(f"     - Marvelquant directory: {stock_automation_root / 'marvelquant'}")
            
            # Try one more time with a direct file check as fallback
            direct_path = Path("/home/ubuntu/Stock_automation/marvelquant/NseLotSize.csv")
            if direct_path.exists():
                logger.warning(f"⚠️  Found file at direct path: {direct_path}")
                logger.warning(f"   Using this path instead...")
                resolved_csv_path = direct_path.resolve()
                set_lot_size_csv_path(resolved_csv_path)
                logger.info(f"✅ Successfully loaded lot sizes from direct path: {resolved_csv_path}")
            else:
                logger.error(f"   Direct path also not found: {direct_path}")
                logger.error(f"   CSV path will not be set - symbols without lot sizes will be skipped")
    
    # Parse expiry dates if provided
    expiry_filter = None
    if args.expiry_dates:
        expiry_filter = parse_expiry_dates(args.expiry_dates)
        if expiry_filter:
            logger.info(f"Expiry filter: {sorted([d.isoformat() for d in expiry_filter])}")
    
    # Handle symmetric strike filtering
    strike_min = args.strike_min
    strike_max = args.strike_max
    if args.strike_center is not None:
        strike_min = args.strike_center - args.strike_band
        strike_max = args.strike_center + args.strike_band
        logger.info(f"Symmetric strike filter: center={args.strike_center}, band=±{args.strike_band}")
    
    # Aggregate metrics across all transformations
    total_metrics = {
        'index_bars': 0,
        'futures_bars': 0,
        'options_bars': 0,
        'equity_bars': 0,
        'options_greeks': 0,
    }
    
    # Aggregate options metrics for skip rate calculation
    all_options_metrics = []
    all_date_metrics = {}
    
    # Process each symbol
    for symbol in args.symbols:
        logger.info(f"Processing symbol: {symbol}")
        
        # Transform index bars
        if "index" in args.types:
            try:
                bars_count = transform_index_bars(
                    input_dir=input_dir,
                    catalog=catalog,
                    symbol=symbol,
                    start_date=args.start_date,
                    end_date=args.end_date
                )
                total_metrics['index_bars'] += bars_count
            except Exception as e:
                logger.error(f"Failed to transform {symbol} index bars: {e}", exc_info=True)
        
        # Transform futures bars
        if "futures" in args.types:
            try:
                bars_count, _ = transform_futures_bars(
                    input_dir=input_dir,
                    catalog=catalog,
                    symbol=symbol,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    output_dir=output_dir
                )
                total_metrics['futures_bars'] += bars_count
            except Exception as e:
                logger.error(f"Failed to transform {symbol} futures bars: {e}", exc_info=True)
        
        # Transform options bars + Greeks
        if "options" in args.types:
            try:
                bars_count, options_metrics, date_metrics = transform_options_bars(
                    input_dir=input_dir,
                    catalog=catalog,
                    symbol=symbol,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    output_dir=output_dir,
                    expiry_filter=expiry_filter,
                    strike_min=strike_min,
                    strike_max=strike_max,
                    log_greeks_debug=args.log_greeks_debug
                )
                total_metrics['options_bars'] += bars_count
                all_options_metrics.append(options_metrics)
                # Merge date metrics
                for date_key, metrics in date_metrics.items():
                    if date_key not in all_date_metrics:
                        all_date_metrics[date_key] = {
                            'records_processed': 0,
                            'records_skipped': 0,
                            'skip_reasons': {}
                        }
                    all_date_metrics[date_key]['records_processed'] += metrics['records_processed']
                    all_date_metrics[date_key]['records_skipped'] += metrics['records_skipped']
                    for reason, count in metrics['skip_reasons'].items():
                        all_date_metrics[date_key]['skip_reasons'][reason] = (
                            all_date_metrics[date_key]['skip_reasons'].get(reason, 0) + count
                        )
            except Exception as e:
                logger.error(f"Failed to transform {symbol} options bars: {e}", exc_info=True)
        
        # Transform equity bars
        if "equity" in args.types:
            try:
                bars_count = transform_equity_bars(
                    input_dir=input_dir,
                    catalog=catalog,
                    symbol=symbol,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    csv_path=resolved_csv_path
                )
                total_metrics['equity_bars'] += bars_count
            except Exception as e:
                logger.error(f"Failed to transform {symbol} equity bars: {e}", exc_info=True)
    
    # Print final summary
    logger.info("=" * 80)
    logger.info("TRANSFORMATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Symbols processed: {', '.join(args.symbols)}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info(f"Types processed: {', '.join(args.types)}")
    if final_log_file:
        logger.info(f"Log file: {final_log_file}")
    logger.info(f"Total index bars: {total_metrics['index_bars']:,}")
    logger.info(f"Total futures bars: {total_metrics['futures_bars']:,}")
    logger.info(f"Total options bars: {total_metrics['options_bars']:,}")
    logger.info(f"Total equity bars: {total_metrics['equity_bars']:,}")
    
    # Calculate and display skip rate for options (if available)
    if all_date_metrics and "options" in args.types:
        total_processed = sum(m['records_processed'] for m in all_date_metrics.values())
        total_skipped = sum(m['records_skipped'] for m in all_date_metrics.values())
        total_records = total_processed + total_skipped
        
        if total_records > 0:
            skip_rate = (total_skipped / total_records) * 100.0
            logger.info("")
            logger.info("OPTIONS GREEKS SKIP RATE SUMMARY")
            logger.info("-" * 80)
            logger.info(f"Total records: {total_records:,}")
            logger.info(f"Processed: {total_processed:,}")
            logger.info(f"Skipped: {total_skipped:,}")
            logger.info(f"Skip rate: {skip_rate:.2f}%")
            
            # Aggregate skip reasons across all dates
            aggregated_skip_reasons = {}
            for date_metrics in all_date_metrics.values():
                for reason, count in date_metrics['skip_reasons'].items():
                    aggregated_skip_reasons[reason] = aggregated_skip_reasons.get(reason, 0) + count
            
            # Display top N skip reasons (default 10)
            if aggregated_skip_reasons:
                top_skip_reasons = sorted(
                    aggregated_skip_reasons.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                logger.info("")
                logger.info("Top skip reasons:")
                for reason, count in top_skip_reasons:
                    if count > 0:
                        pct = (count / total_records) * 100.0
                        logger.info(f"  {reason}: {count:,} ({pct:.2f}%)")
            
            # Warning if skip rate exceeds threshold (default 5%)
            skip_rate_threshold = 5.0
            if skip_rate > skip_rate_threshold:
                logger.warning("")
                logger.warning(f"⚠️  WARNING: Skip rate ({skip_rate:.2f}%) exceeds threshold ({skip_rate_threshold}%)")
                logger.warning("   This may indicate data quality issues. Review logs for details.")
    
    logger.info("=" * 80)
    logger.info(f"Transformation complete. Output saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

