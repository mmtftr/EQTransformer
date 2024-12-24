#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import Dict
import logging
from datetime import datetime
from multiprocessing import Pool, cpu_count
import functools
import logging.handlers

from tqdm import tqdm

from gcf_to_hdf5 import GcfPreprocessor, StationMetadata, process_station

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

def load_station_metadata(metadata_file: str) -> Dict[str, StationMetadata]:
    """Load station metadata from JSON file"""
    with open(metadata_file, 'r') as f:
        data = json.load(f)

    stations = {}
    for station_code, info in data.items():
        stations[station_code] = StationMetadata(
            network=info['network'],
            station=station_code,
            latitude=info['coords'][0],
            longitude=info['coords'][1],
            elevation=info['coords'][2]
        )
    return stations

def setup_station_logger(station_code: str, log_dir: Path) -> logging.Logger:
    """Setup a separate logger for each station"""
    logger = logging.getLogger(f"station_{station_code}")
    logger.setLevel(logging.INFO)

    # Create station log file
    log_file = log_dir / f"{station_code}.log"
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

    return logger

def process_station_wrapper(args):
    """Wrapper function for multiprocessing"""
    station_dir, output_dir, station_metadata, config, log_dir = args

    # Setup station-specific logger
    logger = setup_station_logger(station_dir.name, log_dir)

    try:
        logger.info(f"Starting processing for station {station_dir.name}")
        hdf5_path, csv_path = process_station(
            str(station_dir),
            output_dir,
            station_metadata,
            config
        )
        logger.info(f"Successfully processed station {station_dir.name}")
        return (station_dir.name, True, None)
    except Exception as e:
        logger.error(f"Failed to process station {station_dir.name}: {str(e)}")
        return (station_dir.name, False, str(e))

def main():
    parser = argparse.ArgumentParser(description='Process GCF files into EQTransformer-compatible HDF5 files')

    parser.add_argument('input_dir', type=str,
                        help='Directory containing GCF files (organized by station)')
    parser.add_argument('output_dir', type=str,
                        help='Directory where processed files will be saved')
    parser.add_argument('metadata_file', type=str,
                        help='JSON file containing station metadata')
    parser.add_argument('--sampling-rate', type=float, default=100.0,
                        help='Target sampling rate in Hz (default: 100.0)')
    parser.add_argument('--window-size', type=int, default=6000,
                        help='Window size in samples (default: 6000)')
    parser.add_argument('--overlap', type=float, default=0.3,
                        help='Overlap between windows (default: 0.3)')
    parser.add_argument('--log-file', type=str,
                        help='Log file path (optional)')
    parser.add_argument('--station', type=str,
                        help='Process only this station (optional)')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_file)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Load station metadata
    try:
        stations = load_station_metadata(args.metadata_file)
        logging.info(f"Loaded metadata for {len(stations)} stations")
    except Exception as e:
        logging.error(f"Failed to load station metadata: {e}")
        return 1

    # Preprocessing configuration
    config = {
        'sampling_rate': args.sampling_rate,
        'window_size': args.window_size,
        'overlap': args.overlap
    }

    # Process stations
    success_count = 0
    error_count = 0

    station_dirs = []
    if args.station:
        if args.station not in stations:
            logging.error(f"Station {args.station} not found in metadata")
            return 1
        station_dirs = [Path(args.input_dir) / args.station]
    else:
        station_dirs = [d for d in Path(args.input_dir).iterdir() if d.is_dir()]

    # Process stations in parallel
    num_workers = cpu_count()  # Or specify a different number
    logging.info(f"Using {num_workers} workers for parallel processing")

    process_args = [
        (station_dir, args.output_dir, stations[station_dir.name], config, log_dir)
        for station_dir in station_dirs
        if station_dir.name in stations
    ]

    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_station_wrapper, process_args),
            total=len(process_args),
            desc="Processing stations"
        ))

    # Process results
    for station_code, success, error in results:
        if success:
            logging.info(f"Successfully processed {station_code}")
            success_count += 1
        else:
            logging.error(f"Failed to process station {station_code}: {error}")
            error_count += 1

    logging.info(f"\nProcessing complete:")
    logging.info(f"Successful: {success_count}")
    logging.info(f"Failed: {error_count}")

    return 0 if error_count == 0 else 1

if __name__ == '__main__':
    exit(main())