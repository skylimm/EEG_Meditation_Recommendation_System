from muselsl import record, list_muses
import subprocess
import sys
import time
import os
import numpy as np
from scipy.signal import welch, butter, filtfilt
from sklearn.decomposition import FastICA
import pywt  # For DWT
import pandas as pd
from start_stream import start_muse_stream



# prev version working
def record_eeg_to_csv(duration, folder_path, filename):
    """
    Records EEG data from the Muse Headband and saves it to a CSV file.

    Parameters:
        duration (int): Recording duration in seconds. Default is 20 seconds.
        filename (str): Name of the CSV file to save the EEG data. Default is "eeg_data.csv".
        :param filename:
        :param duration:
        :param folder_path:
    """
    # Define the folder path
    folder_path = folder_path
    full_path = os.path.join(folder_path, filename)

    # Ensure the directory exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Start the Muse LSL stream using subprocess.Popen
    process = subprocess.Popen([sys.executable, "start_stream.py"])  # Replace with the path to your stream script
    time.sleep(20)  # Wait for the stream to start

    try:
        # Check if the stream process is still running
        if process.poll() is not None:
            print("Stream process has exited. Stopping recording.")
            return

        # Start recording EEG data
        print(f"Recording EEG data for {duration} seconds...")
        record(duration, filename=full_path)
        print(f"EEG data saved to {full_path}")

        return full_path  # Return path for reference or future processing

    except KeyboardInterrupt:
        print("Recording interrupted by user.")
        process.terminate()  # Ensure the stream process is terminated on exit

    finally:
        if process.poll() is None:  # Check if the process is still running
            print("Terminating the stream process.")
            process.terminate()  # Clean up the process if it's still active

