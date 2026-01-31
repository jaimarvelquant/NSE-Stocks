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

# Add project root to path (one level above marvelquant package)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

# Use absolute imports to support running as a script
from marvelquant.scripts.transformation.transformers.common import configure_logging
from marvelquant.scripts.transformation.transformers.index_transformer import transform_index_bars
from marvelquant.scripts.transformation.transformers.futures_transformer import transform_futures_bars
from marvelquant.scripts.transformation.transformers.options_transformer import transform_options_bars
from marvelquant.scripts.transformation.transformers.vix_transformer import transform_vix_bars

logger = logging.getLogger(__name__)


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
    default_input_dir = Path("/home/raw_data")
    default_output_dir = Path("/home/nautilus_new")
    
    parser = argparse.ArgumentParser(
        description="Transform NSE data to Nautilus catalog (Modular Pattern)"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir,
        help=f"Input directory with raw data (default: {default_input_dir})"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Output directory for Nautilus catalog (default: {default_output_dir})"
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
        choices=["index", "futures", "options", "vix"],
        default=["index", "futures", "options"],
        help="Data types to transform (vix is separate from index)"
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
    if args.clean and args.output_dir.exists():
        logger.info(f"Cleaning output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct ParquetDataCatalog
    catalog = ParquetDataCatalog(args.output_dir)
    logger.info(f"Using catalog: {args.output_dir}")
    
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
        'options_greeks': 0,
    }
    
    # Aggregate options metrics for skip rate calculation
    all_options_metrics = []
    all_date_metrics = {}
    
    # Process each symbol (only if index/futures/options are requested)
    symbol_types = {"index", "futures", "options"}
    if symbol_types & set(args.types):
        for symbol in args.symbols:
            logger.info(f"Processing symbol: {symbol}")

            # Transform index bars
            if "index" in args.types:
                try:
                    bars_count = transform_index_bars(
                        input_dir=args.input_dir,
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
                        input_dir=args.input_dir,
                        catalog=catalog,
                        symbol=symbol,
                        start_date=args.start_date,
                        end_date=args.end_date,
                        output_dir=args.output_dir
                    )
                    total_metrics['futures_bars'] += bars_count
                except Exception as e:
                    logger.error(f"Failed to transform {symbol} futures bars: {e}", exc_info=True)

            # Transform options bars + Greeks
            if "options" in args.types:
                try:
                    bars_count, options_metrics, date_metrics = transform_options_bars(
                        input_dir=args.input_dir,
                        catalog=catalog,
                        symbol=symbol,
                        start_date=args.start_date,
                        end_date=args.end_date,
                        output_dir=args.output_dir,
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

    # Transform VIX bars (special case - not per-symbol, uses CSV input)
    if "vix" in args.types:
        try:
            vix_file = args.input_dir / "cash" / "vix" / "VIX_data.csv"
            bars_count = transform_vix_bars(
                input_file=vix_file,
                catalog=catalog,
                start_date=args.start_date,
                end_date=args.end_date
            )
            total_metrics['vix_bars'] = bars_count
        except Exception as e:
            logger.error(f"Failed to transform VIX bars: {e}", exc_info=True)

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
    if 'vix_bars' in total_metrics:
        logger.info(f"Total VIX bars: {total_metrics['vix_bars']:,}")
    
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
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

