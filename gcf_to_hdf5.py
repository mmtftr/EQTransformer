from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime
import obspy
import h5py
import os
import csv
from pathlib import Path
import logging
obspy.UTCDateTime

@dataclass
class StationMetadata:
    """Station metadata container"""
    network: str
    station: str
    latitude: float
    longitude: float
    elevation: float

class GcfPreprocessor:
    """Preprocessor for GCF files to create EQTransformer compatible HDF5 files"""

    def __init__(self,
                 sampling_rate: float = 100.0,
                 window_size: int = 6000,
                 overlap: float = 0.3):
        """
        Initialize preprocessor with configuration

        Parameters
        ----------
        sampling_rate : float
            Target sampling rate for the data (Hz)
        window_size : int
            Number of samples per window
        overlap : float
            Overlap between consecutive windows (0-1)
        """
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.overlap = overlap

    def align_components(self,
                        streams: Dict[str, obspy.Stream]
                        ) -> Tuple[np.ndarray, List[datetime]]:
        """
        Align different components and return time-aligned data

        Parameters
        ----------
        streams : Dict[str, obspy.Stream]
            Dictionary of streams for each component (E/N/Z)

        Returns
        -------
        data : np.ndarray
            Aligned data array of shape (samples, 3)
        timestamps : List[datetime]
            List of timestamps for each sample
        """
        # Find common time range
        start_times = []
        end_times = []
        for stream in streams.values():
            start_times.append(stream[0].stats.starttime)
            end_times.append(stream[0].stats.endtime)

        start_time = max(start_times)
        end_time = min(end_times)

        # Calculate expected number of samples including the endpoint
        # Use ceil to ensure we include the last sample
        duration = end_time - start_time
        n_samples = int(np.ceil((duration * self.sampling_rate))) + 1

        # Initialize output array
        aligned_data = np.zeros((n_samples, 3))

        # Process each component
        for idx, (comp, pos) in enumerate({'E': 0, 'N': 1, 'Z': 2}.items()):
            if comp in streams:
                tr = streams[comp][0].copy()  # Make a copy to avoid modifying original

                # Ensure correct sampling rate first
                if tr.stats.sampling_rate != self.sampling_rate:
                    tr.filter('lowpass', freq=self.sampling_rate/2.1, zerophase=True)
                    tr.resample(self.sampling_rate)

                # Trim to exact time window
                tr.trim(start_time, end_time, pad=True, fill_value=0)

                # Ensure exact number of samples
                if len(tr.data) > n_samples:
                    aligned_data[:, pos] = tr.data[:n_samples]
                else:
                    aligned_data[:len(tr.data), pos] = tr.data

        # Create timestamps including the endpoint
        timestamps = [
            start_time + t/self.sampling_rate
            for t in range(n_samples)
        ]

        return aligned_data, timestamps

    def create_sliding_windows(self,
                             data: np.ndarray,
                             timestamps: List[datetime]
                             ) -> Tuple[np.ndarray, List[datetime]]:
        """
        Create overlapping windows from continuous data

        Parameters
        ----------
        data : np.ndarray
            Continuous data array of shape (samples, 3)
        timestamps : List[datetime]
            List of timestamps for each sample

        Returns
        -------
        windows : np.ndarray
            Array of windows, shape (n_windows, window_size, 3)
        window_times : List[datetime]
            List of start times for each window
        """
        # Calculate step size in samples
        step_size = int(self.window_size * (1 - self.overlap))

        # Calculate number of windows
        # Use ceil to ensure we get all possible windows
        n_windows = int(np.ceil((len(data) - self.window_size) / step_size)) + 1

        # Initialize output arrays
        windows = np.zeros((n_windows, self.window_size, data.shape[1]))
        window_times = []

        # Create windows
        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + self.window_size

            # If we would exceed the data length, take the last possible window
            if end_idx > len(data):
                start_idx = len(data) - self.window_size
                end_idx = len(data)

                # Only add this window if it's different from the previous one
                if i > 0 and start_idx == (i-1) * step_size:
                    continue

            windows[i] = data[start_idx:end_idx]
            window_times.append(timestamps[start_idx])

        # Trim any unused windows
        if len(window_times) < n_windows:
            windows = windows[:len(window_times)]

        return windows, window_times

    def write_hdf5(self,
                   windows: np.ndarray,
                   timestamps: List[datetime],
                   metadata: StationMetadata,
                   output_file: str):
        """Write processed windows to HDF5 file"""
        with h5py.File(output_file, 'w') as f:
            f.create_group("data")

            for i, (window, timestamp) in enumerate(zip(windows, timestamps)):
                # Create trace name
                trace_name = f"{metadata.station}_{metadata.network}_HH_{timestamp.strftime('%Y-%m-%dT%H:%M:%S.%f')}"

                # Create dataset
                ds = f.create_dataset(f"data/{trace_name}",
                                    data=window,
                                    dtype=np.float32)

                # Set attributes
                ds.attrs['trace_name'] = trace_name
                ds.attrs['network_code'] = metadata.network
                ds.attrs['receiver_code'] = metadata.station
                ds.attrs['receiver_latitude'] = metadata.latitude
                ds.attrs['receiver_longitude'] = metadata.longitude
                ds.attrs['receiver_elevation_m'] = metadata.elevation
                ds.attrs['trace_start_time'] = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")

    def write_csv(self,
                  timestamps: List[datetime],
                  metadata: StationMetadata,
                  output_file: str):
        """Write metadata to CSV file"""
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['trace_name', 'start_time'])

            for timestamp in timestamps:
                trace_name = f"{metadata.station}_{metadata.network}_HH_{timestamp.strftime('%Y-%m-%dT%H:%M:%S.%f')}"
                writer.writerow([
                    trace_name,
                    timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")
                ])

def process_station(input_dir: str,
                   output_dir: str,
                   metadata: StationMetadata,
                   config: Optional[Dict] = None) -> Tuple[str, str]:
    """
    Process all GCF files for a single station

    Parameters
    ----------
    input_dir : str
        Directory containing GCF files for the station
    output_dir : str
        Directory where processed files will be saved
    metadata : StationMetadata
        Station metadata
    config : Dict, optional
        Configuration parameters for preprocessor

    Returns
    -------
    hdf5_path : str
        Path to output HDF5 file
    csv_path : str
        Path to output CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize preprocessor with config
    proc = GcfPreprocessor(**(config or {}))

    # Group files by hour
    hour_files = {}
    logging.info(f"\nProcessing directory: {input_dir}")

    for file in os.listdir(input_dir):
        if file.endswith('.GCF') or file.endswith('.mseed'):
            try:
                parts = Path(file).stem.split('_')
                date_hour = None
                component = None
                for i, part in enumerate(parts):
                    if len(part) == 8 and part.isdigit():  # YYYYMMDD
                        date = part
                        if i + 1 < len(parts) and len(parts[i+1]) == 4 and parts[i+1].isdigit():  # HHMM
                            hour = parts[i+1]
                            date_hour = f"{date}_{hour}"
                    elif part in ['E', 'N', 'Z']:
                        component = part

                if date_hour and component:
                    if date_hour not in hour_files:
                        hour_files[date_hour] = {'E': None, 'N': None, 'Z': None}
                    hour_files[date_hour][component] = os.path.join(input_dir, file)
            except Exception as e:
                logging.error(f"Error processing filename {file}: {str(e)}")
                continue

    # Process all hours and combine data
    all_data = []
    all_timestamps = []

    for hour, files in sorted(hour_files.items()):
        logging.info(f"\nProcessing hour: {hour}")
        # Read all components
        streams = {}
        for comp, file in files.items():
            if file is not None:
                try:
                    stream = obspy.read(file)
                    logging.info(f"Successfully read {comp} component from {file}")
                    streams[comp] = stream
                except Exception as e:
                    logging.error(f"Error reading file {file}: {str(e)}")

        # Skip if missing components
        if len(streams) < 3:
            logging.warning(f"Skipping hour {hour} - missing components. Found: {list(streams.keys())}")
            continue

        try:
            # Align components for this hour
            data, timestamps = proc.align_components(streams)
            all_data.append(data)
            all_timestamps.extend(timestamps)
            logging.info(f"Added {len(data)} samples from hour {hour}")
        except Exception as e:
            logging.error(f"Error processing hour {hour}: {str(e)}")
            continue

    # Process all data if we have any
    if all_data:
        try:
            # Combine all hours of data
            combined_data = np.vstack(all_data)
            logging.info(f"Combined data shape: {combined_data.shape}")

            # Create windows from the combined data
            windows, window_times = proc.create_sliding_windows(combined_data, all_timestamps)
            logging.info(f"Created {len(windows)} windows")

            # Write outputs
            hdf5_path = os.path.join(output_dir, f"{metadata.station}.hdf5")
            csv_path = os.path.join(output_dir, f"{metadata.station}.csv")

            proc.write_hdf5(windows, window_times, metadata, hdf5_path)
            proc.write_csv(window_times, metadata, csv_path)

            return hdf5_path, csv_path
        except Exception as e:
            logging.error(f"Error processing combined data: {str(e)}")
            raise
    else:
        raise ValueError(f"No valid data found for station {metadata.station}")
