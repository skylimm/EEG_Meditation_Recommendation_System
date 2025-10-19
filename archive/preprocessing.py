from muselsl import record, list_muses
import subprocess
import sys
import time
import os
import numpy as np
from scipy.signal import butter, lfilter, welch, iirnotch, filtfilt
from sklearn.decomposition import FastICA
import pywt  # For DWT
import pandas as pd
from start_stream import start_muse_stream

EEG_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']
FS = 256  # Hz, Muse headband sampling rate
NOTCH_FREQ = 50  # Hz for powerline interference
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# 2. Bandpass Filtering
"""Removes very low-frequency drift (e.g., from sweat/sensors) and high-frequency noise (e.g., EMG).
Keeps only the brain-relevant bands (delta–gamma)"""
def bandpass_filter(data, lowcut=1.0, highcut=50.0, fs=FS, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


# 4. FFT-based Band Power (Absolute & Relative)
def compute_band_powers(signal, fs=FS):
    freqs, psd = welch(signal, fs, nperseg=256)
    band_powers = {}
    total_power = np.sum(psd)

    for band, (low, high) in BANDS.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_power = np.sum(psd[idx])
        band_powers[f'{band}_abs'] = band_power
        band_powers[f'{band}_rel'] = (band_power / total_power) * 100 if total_power > 0 else 0

    return band_powers

# 5. DWT Feature Extraction
def extract_dwt_features(data, wavelet='db4', levels=5):
    coeffs = pywt.wavedec(data, wavelet, level=levels)
    features = {
        'delta_energy': np.sum(coeffs[-1]**2),  # Approx. Delta (0.5-4 Hz)
        'theta_energy': np.sum(coeffs[-2]**2),  # Approx. Theta (4-8 Hz)
        'alpha_energy': np.sum(coeffs[-3]**2),  # Approx. Alpha (8-12 Hz)
        'beta_energy': np.sum(coeffs[-4]**2),   # Approx. Beta (12-30 Hz)
    }
    return features

# 5. Statistical Features
def extract_statistical_features(data):
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'max': np.max(data),
        'min': np.min(data),
        'median': np.median(data)
    }

# def calculate_band_power(data, fs=256, band=None):
#     """
#     Calculates the power in a specific frequency band.
#
#     Parameters:
#         data (array): EEG data for a single electrode.
#         fs (int): Sampling frequency. Default is 256 Hz.
#         band (tuple): Frequency band (low, high).
#
#     Returns:
#         float: Power in the specified frequency band.
#     """
#     filtered = bandpass_filter(data, fs=fs)
#     # Calculate Power Spectral Density (PSD)
#     freqs, psd = welch(filtered, fs, nperseg=1024)
#
#     # Calculate average power in the specified band
#     band_power = np.mean(psd[(freqs >= band[0]) & (freqs <= band[1])])
#     return band_power


def recommend_meditation(theta_power, alpha_power, beta_power):
    """
    Recommends a meditation type using scoring logic across all bands.
    """
    # Define log-space thresholds
    theta_low, theta_high = 25, 28
    alpha_low, alpha_high = 24, 27
    beta_low, beta_high   = 24, 27

    # Categorize band powers
    def level(value, low, high):
        if value < low:
            return "low"
        elif value > high:
            return "high"
        else:
            return "medium"

    theta_level = level(theta_power, theta_low, theta_high)
    alpha_level = level(alpha_power, alpha_low, alpha_high)
    beta_level  = level(beta_power, beta_low, beta_high)

    # Initialize scores
    scores = {
        "Concentration Meditation (Breathwork)": 0,
        "Relaxation Meditation (Guided, TM)": 0,
        "Focused Attention Meditation (FAM)": 0,
        "Mindfulness Meditation (Open Monitoring)": 0,
        "Non-Directive Meditation (Let Thoughts Flow)": 0
    }

    # Score logic
    if beta_level == "low":
        scores["Concentration Meditation (Breathwork)"] += 1
    elif beta_level == "high":
        scores["Relaxation Meditation"] += 1

    if theta_level == "high":
        scores["Focused Attention Meditation (FAM)"] += 1
    elif theta_level == "low":
        scores["Mindfulness Meditation (Open Monitoring)"] += 1
        scores["Non-Directive Meditation (Let Thoughts Flow)"] += 0.5
    if alpha_level == "high":
        scores["Mindfulness Meditation (Open Monitoring)"] += 1
    elif alpha_level == "low":
        scores["Non-Directive Meditation (Let Thoughts Flow)"] += 1

    # Select top scoring meditation
    recommended = max(scores, key=scores.get)

    # (Optional) Debug: print the breakdown
    print(f"[Band Levels] Beta: {beta_level}, Theta: {theta_level}, Alpha: {alpha_level}")
    print("[Score Breakdown]")
    for mtype, score in scores.items():
        print(f" - {mtype}: {score}")

    return recommended

# def recommend_meditation(theta_power, alpha_power, beta_power):
#     """
#     Recommends a meditation type based on Theta, Alpha, and Beta power.
#
#     Based on your mapping:
#     - Alpha low  => Non-directive meditation
#     - Alpha high => Mindfulness meditation
#     - Beta low   => Concentration / Breathwork
#     - Beta high  => Relaxation
#     - Theta high => Focused Attention (FAM)
#     - Theta low  => Mindfulness / Non-directive
#
#     Priority: Beta > Theta > Alpha
#     """
#     # Define thresholds
#     delta_low, delta_high = 2.0, 5.0
#     theta_low, theta_high = 2.5, 6.5
#     alpha_low, alpha_high = 4.0, 10.0
#     beta_low, beta_high = 1.5, 3.5
#     gamma_low, gamma_high = 0.5, 1.0
#
#     # Helper: Categorize into low, medium, high
#     def level(value, low, high):
#         if value < low:
#             return "low"
#         elif value > high:
#             return "high"
#         else:
#             return "medium"
#
#     theta_level = level(theta_power, theta_low, theta_high)
#     alpha_level = level(alpha_power, alpha_low, alpha_high)
#     beta_level = level(beta_power, beta_low, beta_high)
#
#     # Step 1: Prioritize Beta
#     if beta_level == "low":
#         return "Concentration Meditation (Focused Attention, Breathwork)"
#     elif beta_level == "high":
#         return "Relaxation Meditation (Guided Relaxation, TM)"
#
#     # Step 2: Then check Theta
#     elif theta_level == "high":
#         return "Focused Attention Meditation (FAM)"
#     elif theta_level == "low":
#         return "Mindfulness (Open Monitoring) / Non-Directive Meditation (Let Thoughts Flow)"
#
#     # Step 3: Then Alpha
#     elif alpha_level == "high":
#         return "Mindfulness Meditation (Open Monitoring)"
#     elif alpha_level == "low":
#         return "Non-Directive Meditation (Let Thoughts Flow)"
#
#     # Fallback
#     else:
#         return "Maintain Current Meditation Practice"
