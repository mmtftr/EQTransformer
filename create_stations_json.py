#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, Set
import logging
from obspy.clients.fdsn import Client
from datetime import datetime
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

def get_station_codes(data_dir: Path) -> Set[str]:
    """Extract unique station codes from GCF filenames"""
    station_codes = set()

    # GCF filename pattern: STATION_DATE_TIME_RATE_B_NETWORK_COMPONENT.GCF
    pattern = r'^([A-Z0-9]+)\s?_\d{8}_\d{4}_\d+_B_[A-Z]+_[ENZ]\.GCF$'

    for file in data_dir.glob('**/*.GCF'):
        match = re.match(pattern, file.name)
        if match:
            station_codes.add(match.group(1))
        else:
            logging.warning(f"Couldn't parse station code from filename: {file.name}")

    return station_codes

def get_station_metadata(station_code: str,
                        network_code: str,
                        start_date: datetime,
                        client: Client) -> Dict:
    """Get station metadata from FDSN client"""
    try:
        inventory = client.get_stations(
            network=network_code,
            station=station_code,
            starttime=start_date,
            endtime=start_date,
            level="station"
        )

        if not inventory:
            raise ValueError("No station data found")

        station = inventory[0][0]
        return {
            "network": network_code,
            "coords": [
                station.latitude,
                station.longitude,
                station.elevation
            ]
        }
    except Exception as e:
        logging.error(f"Error getting metadata for {station_code}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Create stations.json from GCF files using FDSN')

    parser.add_argument('--data_dir', type=str, default="data",
                        help='Directory containing GCF files')
    parser.add_argument('--output_file', type=str, default="stations.json",
                        help='Output JSON file path')
    parser.add_argument('--network', type=str, default='KO',
                        help='Network code (default: KO)')
    parser.add_argument('--date', type=str, default=None,
                        help='Date for metadata query (default: from filenames)')
    parser.add_argument('--fdsn-server', type=str,
                        default='KOERI',
                        help='FDSN server (default: KOERI)')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Log file path (optional)')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_file)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logging.error(f"Data directory does not exist: {data_dir}")
        return 1

    # Get unique station codes
    logging.info("Scanning for station codes...")
    station_codes = get_station_codes(data_dir)
    logging.info(f"Found {len(station_codes)} stations: {', '.join(sorted(station_codes))}")

    # Get date from first file if not provided
    if args.date is None:
        for file in data_dir.glob('**/*.GCF'):
            try:
                date_str = file.name.split('_')[1]
                args.date = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
                break
            except:
                continue
        if args.date is None:
            logging.error("Could not determine date from filenames and no date provided")
            return 1

    # Initialize FDSN client
    try:
        logging.info(f"Connecting to {args.fdsn_server} FDSN server...")
        client = Client(args.fdsn_server)
    except Exception as e:
        logging.error(f"Failed to connect to FDSN server: {e}")
        return 1

    # Get metadata for each station
    stations_data = {}
    query_date = datetime.strptime(args.date, '%Y-%m-%d')

    for station_code in sorted(station_codes):
        logging.info(f"Getting metadata for {station_code}...")
        metadata = get_station_metadata(station_code, args.network, query_date, client)

        if metadata:
            stations_data[station_code] = metadata
            logging.info(f"Successfully got metadata for {station_code}")

    # Save metadata to JSON file
    with open(args.output_file, 'w') as f:
        json.dump(stations_data, f, indent=4)

    logging.info(f"Saved metadata to {args.output_file}")

if __name__ == "__main__":
    main()