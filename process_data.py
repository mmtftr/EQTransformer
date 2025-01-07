#!/usr/bin/env python3

import argparse
import logging
import subprocess
from pathlib import Path
import sys

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

def run_command(cmd: list, description: str) -> bool:
    """Run a command and log its output"""
    logging.info(f"Starting {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info(f"Successfully completed {description}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed during {description}: {e}")
        logging.error(f"Command output: {e.output}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Process GCF files for EQTransformer')

    parser.add_argument('input_dir', type=str,
                        help='Directory containing input GCF files')
    parser.add_argument('--log-file', type=str,
                        help='Log file path (optional)')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_file)

    # Create necessary directories
    base_dir = Path(args.input_dir)
    organized_dir = base_dir / "organized_data"
    processed_dir = base_dir / "processed_data"
    organized_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Define the pipeline steps
    steps = [
        {
            'cmd': ['python', 'create_stations_json.py',
                   '--data_dir', str(base_dir / "gcf_files"),
                   '--output_file', str(base_dir / "stations.json")],
            'description': 'creating stations.json'
        },
        {
            'cmd': ['python', 'organize_gcf_files.py',
                   str(base_dir / "gcf_files"),
                   str(organized_dir)],
            'description': 'organizing GCF files'
        },
        {
            'cmd': ['python', 'preprocess_gcf.py',
                   str(organized_dir),
                   str(processed_dir),
                   str(base_dir / "stations.json")],
            'description': 'preprocessing GCF files'
        },
        {
            'cmd': ['python', 'predict.py',
                    str(processed_dir),
                    'EqT_original_model',
                    '--output_dir', 'detections'],
            'description': 'running EQTransformer prediction'
        }
    ]

    # Execute each step
    for step in steps:
        if not run_command(step['cmd'], step['description']):
            logging.error("Pipeline failed. Exiting.")
            return 1

    logging.info("Pipeline completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())