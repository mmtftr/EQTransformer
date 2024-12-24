#!/usr/bin/env python3
import click
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict

def find_nearest_events(df1: pd.DataFrame,
                       df2: pd.DataFrame,
                       time_column: str,
                       max_time_diff: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Find events that are common between two dataframes (within max_time_diff),
    and events that are unique to each dataframe.
    NaN values are excluded from the results.
    """
    # Filter out NaN values from both dataframes
    df1 = df1.dropna(subset=[time_column])
    df2 = df2.dropna(subset=[time_column])

    common_events_1 = []
    common_events_2 = []
    used_indices_df2 = set()

    # For each event in df1, find the nearest event in df2
    for idx1, row1 in df1.iterrows():
        time1 = pd.to_datetime(row1[time_column])
        if pd.isna(time1):
            continue

        min_diff = float('inf')
        nearest_idx = None

        for idx2, row2 in df2.iterrows():
            if idx2 in used_indices_df2:
                continue

            time2 = pd.to_datetime(row2[time_column])
            if pd.isna(time2):
                continue

            time_diff = abs((time2 - time1).total_seconds())

            if time_diff < min_diff:
                min_diff = time_diff
                nearest_idx = idx2

        if nearest_idx is not None and min_diff <= max_time_diff:
            common_events_1.append(row1)
            common_events_2.append(df2.iloc[nearest_idx])
            used_indices_df2.add(nearest_idx)

    # Create mask for events only in df1
    mask_df1 = ~df1.index.isin([row.name for row in common_events_1])
    only_in_df1 = df1[mask_df1].copy()

    # Create mask for events only in df2
    mask_df2 = ~df2.index.isin([row.name for row in common_events_2])
    only_in_df2 = df2[mask_df2].copy()

    # Combine common events
    common_events = pd.DataFrame({
        'file1_' + time_column: [pd.to_datetime(row[time_column]) for row in common_events_1],
        'file2_' + time_column: [pd.to_datetime(row[time_column]) for row in common_events_2],
        'time_diff': [(pd.to_datetime(row2[time_column]) - pd.to_datetime(row1[time_column])).total_seconds()
                     for row1, row2 in zip(common_events_1, common_events_2)]
    })

    # Check if probability column exists and handle NaNs
    prob_col = time_column.split('_')[0] + '_probability'
    if prob_col in df1.columns and prob_col in df2.columns:
        common_events['file1_probability'] = [row[prob_col] for row in common_events_1]
        common_events['file2_probability'] = [row[prob_col] for row in common_events_2]
        only_in_df1[prob_col] = df1[prob_col]
        only_in_df2[prob_col] = df2[prob_col]

    return common_events, only_in_df1, only_in_df2

def process_wave_type(df1: pd.DataFrame,
                     df2: pd.DataFrame,
                     wave_type: str,
                     output_dir: Path,
                     prefix: str = '') -> None:
    """
    Process a specific wave type (P or S) and save results to files.

    Args:
        df1: First dataframe
        df2: Second dataframe
        wave_type: Type of wave ('p' or 's')
        output_dir: Directory to save output files
        prefix: Prefix for output files
    """
    time_col = f'{wave_type}_arrival_time'

    common, only1, only2 = find_nearest_events(
        df1, df2, time_col
    )

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    common.to_csv(output_dir / f'{prefix}{wave_type}_common.csv', index=False)
    only1.to_csv(output_dir / f'{prefix}{wave_type}_only_file1.csv', index=False)
    only2.to_csv(output_dir / f'{prefix}{wave_type}_only_file2.csv', index=False)

@click.command()
@click.argument('file1', type=click.Path(exists=True))
@click.argument('file2', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), default='output',
              help='Output directory for result files')
@click.option('--prefix', '-p', type=str, default='',
              help='Prefix for output files')
@click.option('--max-time-diff', '-t', type=float, default=0.5,
              help='Maximum time difference (in seconds) to consider events as same')
def main(file1: str, file2: str, output_dir: str, prefix: str, max_time_diff: float):
    """
    Compare seismic wave picks between two CSV files.

    This script takes two CSV files containing P and S wave arrival picks and generates
    three files for each wave type (P and S):
    1. Common events (within specified time difference)
    2. Events only in file 1
    3. Events only in file 2

    The CSV files should contain columns:
    - p_arrival_time
    - s_arrival_time
    - p_probability (optional)
    - s_probability (optional)
    """
    # Read CSV files
    df1 = pd.read_csv(file1, parse_dates=['p_arrival_time', 's_arrival_time'])
    df2 = pd.read_csv(file2, parse_dates=['p_arrival_time', 's_arrival_time'])

    output_path = Path(output_dir)

    # Process P waves
    process_wave_type(df1, df2, 'p', output_path, prefix)

    # Process S waves
    process_wave_type(df1, df2, 's', output_path, prefix)

    click.echo(f"Results have been saved to {output_path}")

    # Print summary statistics
    for wave_type in ['p', 's']:
        common = pd.read_csv(output_path / f'{prefix}{wave_type}_common.csv')
        only1 = pd.read_csv(output_path / f'{prefix}{wave_type}_only_file1.csv')
        only2 = pd.read_csv(output_path / f'{prefix}{wave_type}_only_file2.csv')

        click.echo(f"\n{wave_type.upper()} wave statistics:")
        click.echo(f"Common events: {len(common)}")
        click.echo(f"Only in file 1: {len(only1)}")
        click.echo(f"Only in file 2: {len(only2)}")
        if len(common) > 0:
            click.echo(f"Average time difference: {common['time_diff'].mean():.3f}s")
            click.echo(f"Max time difference: {common['time_diff'].max():.3f}s")

if __name__ == '__main__':
    main()