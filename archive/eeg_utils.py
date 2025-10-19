import pandas as pd
import numpy as np
from scipy.signal import welch, butter, filtfilt
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

#  for each channel visualization
def bandpass_filter(data, fs=256, lowcut=0.5, highcut=45, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

def plot_raw_signal(df, channel):
    fig, ax = plt.subplots()
    ax.plot(df[channel].dropna().values)
    ax.set_title(f"Raw EEG Signal - {channel}")
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Amplitude")
    return fig

def plot_filtered_signal(df, channel, fs=256):
    raw = df[channel].dropna().values
    filtered = bandpass_filter(raw, fs)
    fig, ax = plt.subplots()
    ax.plot(filtered)
    ax.set_title(f"Filtered EEG Signal - {channel}")
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Amplitude")
    return fig

def plot_fft(df, channel, fs=256):
    raw = df[channel].dropna().values
    filtered = bandpass_filter(raw, fs)
    freqs, psd = welch(filtered, fs, nperseg=1024)
    fig, ax = plt.subplots()
    ax.semilogy(freqs, psd)
    ax.set_xlim([0, 60])
    ax.set_title(f"FFT Power Spectrum - {channel}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density")
    return fig

def plot_spectrogram(df, channel, fs=256):
    from scipy.signal import spectrogram
    raw = df[channel].dropna().values
    filtered = bandpass_filter(raw, fs)
    f, t, Sxx = spectrogram(filtered, fs=fs)
    fig, ax = plt.subplots()
    ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    ax.set_ylim(0, 60)
    ax.set_title(f"Spectrogram - {channel}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    return fig

# EEG bands
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30)
}


def calculate_band_powers(filepath, fs=256):
    df = pd.read_csv(filepath)
    abs_results = {}
    rel_results = {}

    for col in df.columns:
        if col in ['TP9', 'AF7', 'AF8', 'TP10']:
            raw_data = df[col].dropna()
            filtered = bandpass_filter(raw_data, fs=fs)

            freqs, psd = welch(filtered, fs=fs, nperseg=1024)

            band_powers = {}
            for band, (low, high) in BANDS.items():
                idx = np.logical_and(freqs >= low, freqs <= high)
                band_powers[band] = np.trapz(psd[idx], freqs[idx])

            abs_results[col] = band_powers

            # Calculate relative band power
            total_power = sum(band_powers.values())
            if total_power > 0:
                rel_results[col] = {band: val / total_power for band, val in band_powers.items()}
            else:
                rel_results[col] = {band: 0 for band in band_powers}

    return abs_results, rel_results


# avaeraged out accross all channels
def compute_band_power_avg(signal, fs=256):
    freqs, psd = welch(signal, fs=fs, nperseg=1024)
    powers = {}
    total_power = 0

    for band, (low, high) in BANDS.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_power = np.trapz(psd[idx], freqs[idx])
        powers[band] = band_power
        total_power += band_power

    # Compute relative powers
    rel_powers = {f"{band}_rel": powers[band] / total_power if total_power > 0 else 0 for band in BANDS}
    return {**powers, **rel_powers}

def calculate_average_bandpowers(filepath, fs=256):
    df = pd.read_csv(filepath)
    channels = ['TP9', 'AF7', 'AF8', 'TP10']
    band_results = []

    for ch in channels:
        if ch in df.columns:
            filtered = bandpass_filter(df[ch].dropna().values, fs)
            powers = compute_band_power_avg(filtered, fs)
            band_results.append(powers)

    # Average across channels
    avg_results = {}
    for key in band_results[0]:
        avg_results[key] = np.mean([ch[key] for ch in band_results])

    return avg_results

def plot_avg_bandpowers_bar(avg_powers):
    bands = ['delta', 'theta', 'alpha', 'beta']
    values = [avg_powers[band] for band in bands]

    fig, ax = plt.subplots()
    ax.bar(bands, values, color='skyblue')
    ax.set_title("Absolute Averaged Band Powers")
    ax.set_ylabel("Power")
    return fig

def plot_avg_relative_bar(avg_powers):
    bands = ['delta_rel', 'theta_rel', 'alpha_rel', 'beta_rel']
    rel_labels = [b.replace('_rel', '') for b in bands]
    values = [avg_powers[band] * 100 for band in bands]

    fig, ax = plt.subplots()
    ax.bar(rel_labels, values, color='mediumseagreen')
    ax.set_title("Relative Averaged Band Powers")
    ax.set_ylabel("Power (%)")
    return fig

