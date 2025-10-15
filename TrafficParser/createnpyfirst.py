#!/usr/bin/env python
"""
Convert all traffic CSVs to .npy datasets ready for model training.
"""

import os
import glob
import csv
import numpy as np
import re
from sessions_plotter import session_2d_histogram  # make sure this works

# --- Configuration ---
INPUT_DIR = "../raw_csvs/classes_csvs/"
OUTPUT_DIR = "../raw_csvs/classes_npy/"
TPS = 60           # TimePerSession in seconds
DELTA_T = 60       # Delta between sessions
MIN_TPS = 50       # Minimum session length

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper functions ---
def process_csv_file(file_path):
    """Convert a single CSV file into a numpy array of 2D histograms."""
    dataset = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                length = int(row[7])
                ts = np.array(row[8:8+length], dtype=float)
                sizes = np.array(row[8+length:], dtype=int)  # Adjusted indexing
            except ValueError:
                # Skip rows with non-numeric data
                continue

            if length <= 10 or ts[-1] - ts[0] < MIN_TPS:
                continue

            # Split session into chunks of TPS
            for t in range(int(ts[-1]/DELTA_T - TPS/DELTA_T) + 1):
                mask = (ts >= t*DELTA_T) & (ts <= t*DELTA_T + TPS)
                ts_mask = ts[mask]
                sizes_mask = sizes[mask]
                if len(ts_mask) > 10 and ts_mask[-1] - ts_mask[0] > MIN_TPS:
                    h = session_2d_histogram(ts_mask, sizes_mask)
                    dataset.append(h)

    return np.array(dataset)

def process_class_dir(class_dir):
    """Process all CSV files in a class folder and save as a single .npy file."""
    csv_files = glob.glob(os.path.join(class_dir, "*.csv"))
    all_data = []
    for f in csv_files:
        print(f"Processing {f}")
        data = process_csv_file(f)
        all_data.append(data)
    if all_data:
        all_data = np.concatenate(all_data, axis=0)
        # Generate output filename based on class_dir path
        class_name = "_".join(re.findall(r"[\w']+", class_dir)[-2:])
        output_file = os.path.join(OUTPUT_DIR, class_name + ".npy")
        np.save(output_file, all_data)
        print(f"Saved {output_file} with shape {all_data.shape}")
    else:
        print(f"No valid data found in {class_dir}")

# --- Main ---
def main():
    # Iterate over all classes and types
    class_dirs = [d for d in glob.glob(os.path.join(INPUT_DIR, "**", "*"), recursive=True) if os.path.isdir(d)]
    for class_dir in class_dirs:
        process_class_dir(class_dir)

if __name__ == "__main__":
    main()
