import pytest
import numpy as np
from datetime import datetime, timedelta
import h5py
import os
from pathlib import Path
import obspy

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gcf_to_hdf5 import GcfPreprocessor, StationMetadata, process_station

@pytest.fixture
def sample_metadata():
    return StationMetadata(
        network="KO",
        station="ADVT",
        latitude=35.5865,
        longitude=-117.4622,
        elevation=694.5
    )

@pytest.fixture
def sample_gcf_data():
    """Create a synthetic GCF-like stream for testing"""
    # Create exactly 5 minutes of synthetic data at 100Hz
    duration = 300  # 5 minutes
    npts = int(100 * duration)  # 5 minutes * 100Hz
    times = np.arange(npts) / 100.0  # Time in seconds

    # Create synthetic signals for each component
    data_e = np.sin(2 * np.pi * 1.0 * times)  # 1 Hz sine wave
    data_n = np.cos(2 * np.pi * 1.0 * times)  # 1 Hz cosine wave
    data_z = np.sin(2 * np.pi * 2.0 * times)  # 2 Hz sine wave

    # Create ObsPy traces with exact timing
    start_time = obspy.UTCDateTime("2023-02-06T01:00:00.000000Z")
    st = obspy.Stream()

    for data, component in [(data_e, 'E'), (data_n, 'N'), (data_z, 'Z')]:
        tr = obspy.Trace(data=data)
        tr.stats.sampling_rate = 100.0
        tr.stats.starttime = start_time
        tr.stats.npts = npts
        tr.stats.channel = f'HH{component}'
        tr.stats.station = 'ADVT'
        tr.stats.network = 'KO'
        st += tr

    return st

def test_init():
    """Test preprocessor initialization"""
    proc = GcfPreprocessor()
    assert proc.sampling_rate == 100.0
    assert proc.window_size == 6000
    assert proc.overlap == 0.3

    # Test custom parameters
    proc = GcfPreprocessor(sampling_rate=200.0, window_size=12000, overlap=0.5)
    assert proc.sampling_rate == 200.0
    assert proc.window_size == 12000
    assert proc.overlap == 0.5

def test_align_components(sample_gcf_data):
    """Test component alignment"""
    proc = GcfPreprocessor()

    # Split stream into component streams
    streams = {
        'E': sample_gcf_data.select(component='E'),
        'N': sample_gcf_data.select(component='N'),
        'Z': sample_gcf_data.select(component='Z')
    }

    data, timestamps = proc.align_components(streams)

    # Check data shape
    expected_samples = int(300 * proc.sampling_rate)  # 5 minutes * 100Hz
    assert data.shape == (expected_samples, 3)
    assert len(timestamps) == expected_samples

    # Check timestamps
    start_time = sample_gcf_data[0].stats.starttime
    assert abs(timestamps[0] - start_time) < 1e-6
    assert abs(timestamps[-1] - (start_time + 300)) < 1e-6

def test_create_sliding_windows(sample_gcf_data):
    """Test creation of sliding windows"""
    proc = GcfPreprocessor()

    # First create aligned data
    streams = {
        'E': sample_gcf_data.select(component='E'),
        'N': sample_gcf_data.select(component='N'),
        'Z': sample_gcf_data.select(component='Z')
    }
    data, timestamps = proc.align_components(streams)

    # Create windows
    windows, window_times = proc.create_sliding_windows(data, timestamps)

    # Check window shapes
    assert windows.shape[1] == proc.window_size  # Each window should be window_size samples
    assert windows.shape[2] == 3  # 3 components

    # Check overlap
    expected_step = int(proc.window_size * (1 - proc.overlap))
    total_windows = (len(data) - proc.window_size) // expected_step + 1
    assert len(window_times) == total_windows
    assert windows.shape[0] == total_windows

def test_write_hdf5(tmp_path, sample_gcf_data, sample_metadata):
    """Test HDF5 file writing"""
    proc = GcfPreprocessor()

    # Create some windows
    streams = {
        'E': sample_gcf_data.select(component='E'),
        'N': sample_gcf_data.select(component='N'),
        'Z': sample_gcf_data.select(component='Z')
    }
    data, timestamps = proc.align_components(streams)
    windows, window_times = proc.create_sliding_windows(data, timestamps)

    # Write to temporary file
    output_file = tmp_path / "test_station.hdf5"
    proc.write_hdf5(windows, window_times, sample_metadata, str(output_file))

    # Verify file contents
    with h5py.File(output_file, 'r') as f:
        assert 'data' in f

        # Check first window
        first_window_name = list(f['data'].keys())[0]
        window_data = f['data'][first_window_name]

        # Check data shape
        assert window_data.shape == (proc.window_size, 3)

        # Check attributes
        attrs = window_data.attrs
        assert attrs['network_code'] == sample_metadata.network
        assert attrs['receiver_code'] == sample_metadata.station
        assert attrs['receiver_latitude'] == sample_metadata.latitude
        assert attrs['receiver_longitude'] == sample_metadata.longitude
        assert attrs['receiver_elevation_m'] == sample_metadata.elevation

def test_write_csv(tmp_path, sample_gcf_data, sample_metadata):
    """Test CSV file writing"""
    proc = GcfPreprocessor()

    # Create some windows
    streams = {
        'E': sample_gcf_data.select(component='E'),
        'N': sample_gcf_data.select(component='N'),
        'Z': sample_gcf_data.select(component='Z')
    }
    data, timestamps = proc.align_components(streams)
    windows, window_times = proc.create_sliding_windows(data, timestamps)

    # Write to temporary file
    output_file = tmp_path / "test_station.csv"
    proc.write_csv(window_times, sample_metadata, str(output_file))

    # Verify file contents
    import pandas as pd
    df = pd.read_csv(output_file)

    assert len(df) == len(window_times)
    assert 'trace_name' in df.columns
    assert 'start_time' in df.columns

    # Check first row format
    first_row = df.iloc[0]
    assert first_row['trace_name'].startswith(f"{sample_metadata.station}_{sample_metadata.network}_")
    # Verify datetime format
    datetime.strptime(first_row['start_time'], "%Y-%m-%d %H:%M:%S.%f")

def test_process_station(tmp_path, sample_metadata):
    """Test full station processing"""
    # Create test directory structure
    station_dir = tmp_path / "raw_data" / sample_metadata.station
    station_dir.mkdir(parents=True)

    # Create synthetic GCF files
    start_time = datetime(2023, 2, 6, 1, 0)
    for component in ['E', 'N', 'Z']:
        filename = f"{sample_metadata.station}_{start_time.strftime('%Y%m%d_%H%M')}_100_B_{sample_metadata.network}_{component}.GCF"
        filepath = station_dir / filename

        # Create a dummy GCF file (we'll use ObsPy's write function)
        tr = obspy.Trace(data=np.zeros(6000))
        tr.stats.sampling_rate = 100
        tr.stats.starttime = obspy.UTCDateTime(start_time)
        tr.stats.channel = f'HH{component}'
        tr.stats.station = sample_metadata.station
        tr.stats.network = sample_metadata.network
        tr.write(str(filepath), format='GCF')  # Using MSEED as a stand-in for F

    # Create test configuration
    config = {
        'sampling_rate': 100.0,
        'window_size': 6000,
        'overlap': 0.3
    }

    # Process station
    hdf5_path, csv_path = process_station(
        str(station_dir),
        str(tmp_path / "processed"),
        sample_metadata,
        config
    )

    # Verify outputs exist
    assert os.path.exists(hdf5_path)
    assert os.path.exists(csv_path)

def test_hour_boundary_handling(tmp_path, sample_metadata):
    """Test handling of data across hour boundaries"""
    proc = GcfPreprocessor()

    # Create two consecutive hours of data
    hour1_start = obspy.UTCDateTime("2023-02-06T01:00:00.000000Z")
    hour2_start = obspy.UTCDateTime("2023-02-06T02:00:00.000000Z")

    # Create synthetic data for both hours
    def create_hour_data(start_time):
        st = obspy.Stream()
        duration = 3600  # 1 hour in seconds
        npts = int(100 * duration)  # 1 hour at 100Hz
        times = np.arange(npts) / 100.0

        for comp, func in [('E', np.sin), ('N', np.cos), ('Z', np.sin)]:
            data = func(2 * np.pi * 1.0 * times)  # 1 Hz waves
            tr = obspy.Trace(data=data)
            tr.stats.sampling_rate = 100.0
            tr.stats.starttime = start_time
            tr.stats.npts = npts
            tr.stats.channel = f'HH{comp}'
            tr.stats.station = sample_metadata.station
            tr.stats.network = sample_metadata.network
            st += tr
        return st

    hour1_stream = create_hour_data(hour1_start)
    hour2_stream = create_hour_data(hour2_start)

    # Process each hour
    streams1 = {comp: hour1_stream.select(component=comp) for comp in 'ENZ'}
    streams2 = {comp: hour2_stream.select(component=comp) for comp in 'ENZ'}

    # Get windows for each hour
    data1, timestamps1 = proc.align_components(streams1)
    data2, timestamps2 = proc.align_components(streams2)
    windows1, times1 = proc.create_sliding_windows(data1, timestamps1)
    windows2, times2 = proc.create_sliding_windows(data2, timestamps2)

    # Check for gap at hour boundary
    last_window_end = times1[-1] + proc.window_size / proc.sampling_rate
    next_window_start = times2[0]

    time_difference = next_window_start - last_window_end
    max_allowed_gap = proc.window_size * (1 - proc.overlap) / proc.sampling_rate

    assert time_difference <= max_allowed_gap, \
        f"Gap at hour boundary ({time_difference}s) exceeds maximum allowed ({max_allowed_gap}s)"

    # Additional test: combine hours and verify window count
    combined_data = np.vstack([data1, data2])
    combined_timestamps = timestamps1 + timestamps2

    windows_combined, times_combined = proc.create_sliding_windows(combined_data, combined_timestamps)

    # The combined windows should have more windows than the sum of individual hours
    # due to windows spanning the boundary
    assert len(windows_combined) > len(windows1) + len(windows2), \
        "Missing windows at hour boundary"

    # Verify continuity of windows
    for i in range(len(times_combined) - 1):
        time_diff = times_combined[i + 1] - times_combined[i]
        assert time_diff <= max_allowed_gap, \
            f"Gap between windows ({time_diff}s) exceeds maximum allowed ({max_allowed_gap}s)"