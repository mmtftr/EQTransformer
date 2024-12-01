#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path
import logging
import re

def setup_logging(log_file: str = None):
    """Setup logging configuration"""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def organize_files(input_dir: Path, output_dir: Path, dry_run: bool = False):
    """
    Organize GCF files into station directories

    Parameters
    ----------
    input_dir : Path
        Directory containing GCF files
    output_dir : Path
        Directory where organized files will be placed
    dry_run : bool
        If True, only print what would be done without moving files
    """
    # GCF filename pattern: STATION_DATE_TIME_RATE_B_NETWORK_COMPONENT.GCF
    pattern = r'^([A-Z0-9]+)\s_\d{8}_\d{4}_\d+_B_[A-Z]+_[ENZ]\.GCF$'

    # Track statistics
    stats = {
        'total_files': 0,
        'organized_files': 0,
        'skipped_files': 0,
        'stations': set()
    }

    # Process each GCF file
    for file in input_dir.glob('*.GCF'):
        stats['total_files'] += 1
        match = re.match(pattern, file.name)

        if match:
            station_code = match.group(1)
            stats['stations'].add(station_code)

            # Create station directory
            station_dir = output_dir / station_code
            if not dry_run:
                station_dir.mkdir(parents=True, exist_ok=True)

            # Move file
            dest_file = station_dir / file.name
            if dest_file.exists():
                logging.warning(f"File already exists: {dest_file}")
                stats['skipped_files'] += 1
                continue

            if dry_run:
                logging.info(f"Would move {file} to {dest_file}")
            else:
                try:
                    shutil.move(str(file), str(dest_file))
                    logging.info(f"Moved {file.name} to {station_code}/")
                    stats['organized_files'] += 1
                except Exception as e:
                    logging.error(f"Failed to move {file}: {e}")
                    stats['skipped_files'] += 1
        else:
            logging.warning(f"Skipping file with invalid name format: {file.name}")
            stats['skipped_files'] += 1

    return stats

def main():
    parser = argparse.ArgumentParser(description='Organize GCF files into station directories')

    parser.add_argument('input_dir', type=str,
                        help='Directory containing GCF files')
    parser.add_argument('output_dir', type=str,
                        help='Directory where organized files will be placed')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be done without moving files')
    parser.add_argument('--log-file', type=str,
                        help='Log file path (optional)')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_file)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        return 1

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Organize files
    logging.info(f"{'Dry run: ' if args.dry_run else ''}Organizing files from {input_dir} to {output_dir}")
    stats = organize_files(input_dir, output_dir, args.dry_run)

    # Print summary
    logging.info("\nSummary:")
    logging.info(f"Total files processed: {stats['total_files']}")
    logging.info(f"Files organized: {stats['organized_files']}")
    logging.info(f"Files skipped: {stats['skipped_files']}")
    logging.info(f"Stations found: {len(stats['stations'])}")
    logging.info(f"Station list: {', '.join(sorted(stats['stations']))}")

    return 0

if __name__ == '__main__':
    exit(main())