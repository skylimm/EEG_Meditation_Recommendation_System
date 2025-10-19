# pre_processings.py
# Simple & consistent EEG preprocessing for Muse 2 (pre/post meditation)
# Flow: per-channel -> (optional interp for big spikes) -> 50 Hz notch -> 1–40 Hz band-pass
# -> slice a fixed middle duration -> Welch PSD (2 s, 50% overlap) -> integrate bands
# -> mean across channels -> print-ready dicts (absolute & relative)

from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from scipy.signal import iirnotch, butter, filtfilt, welch

# ------------ Config ------------
EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]

BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1, 4),
    "theta": (4, 7),
    "alpha": (8, 12),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

BANDPASS = (1, 40)       # Hz
NOTCH_FREQ = 50          # Hz (Singapore mains)
NOTCH_Q = 30
ARTIFACT_THRESH = 1000.0 # µV; simple spike interpolation cap
TOTAL_PWR_BAND = (1, 40) # match band-pass

# Welch settings (consistent across files)
WELCH_SEG_SEC = 2.0      # seconds
WELCH_OVERLAP = 0.5      # 50%

@dataclass
class Summary:
    fs: float
    channels_used: int
    abs_overall: Dict[str, float]
    rel_overall_pct: Dict[str, float]


# ------------ Helpers ------------

def read_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def estimate_fs_from_timestamps(series: pd.Series, default_fs: float = 256.0) -> float:
    try:
        ts = series.to_numpy(float)
        ts = ts[np.isfinite(ts)]
        if ts.size < 3:
            return default_fs
        diffs = np.diff(ts)
        diffs = diffs[(np.isfinite(diffs)) & (diffs > 0)]
        if diffs.size == 0:
            return default_fs
        med_dt = np.median(diffs)
        # Heuristic: if someone exported ms ticks
        if med_dt > 10:
            med_dt /= 1000.0
        fs = 1.0 / med_dt
        return float(fs) if 10 < fs < 2048 else default_fs
    except Exception:
        return default_fs

def notch_filter(x: np.ndarray, fs: float, f0: float = NOTCH_FREQ, q: float = NOTCH_Q) -> np.ndarray:
    b, a = iirnotch(w0=f0/(fs/2.0), Q=q)
    return filtfilt(b, a, x)

def bandpass_filter(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, x)

def preprocess_channel(x: np.ndarray, fs: float) -> np.ndarray:
    x = x.astype(float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return x
    # simple spike interpolation (optional but keeps things robust)
    mask = np.abs(x) <= ARTIFACT_THRESH
    if not np.all(mask):
        good = np.where(mask)[0]
        if good.size >= 2:
            x = np.interp(np.arange(x.size), good, x[good])
    x = notch_filter(x, fs)
    x = bandpass_filter(x, fs, BANDPASS[0], BANDPASS[1], order=4)
    return x


def compute_welch_psd(x: np.ndarray, fs: float):
    if x.size < int(2 * fs):
        return np.array([]), np.array([])
    nperseg = int(WELCH_SEG_SEC * fs)
    noverlap = int(nperseg * WELCH_OVERLAP)
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return f, Pxx

def integrate_band(f: np.ndarray, Pxx: np.ndarray, band: Tuple[float, float]) -> float:
    if f.size == 0 or Pxx.size == 0:
        return np.nan
    f1, f2 = band
    idx = (f >= f1) & (f <= f2)
    if not np.any(idx):
        return np.nan
    return float(np.trapz(Pxx[idx], f[idx]))


# ------------ Main per-file summary ------------

def summarize_one_file(path) -> Summary:
    """
    Returns overall (mean across channels) bandpowers:
      - abs_overall: Dict[band] -> absolute power
      - rel_overall_pct: Dict[band] -> relative power (% of total 1–40 Hz)
    """
    df = read_csv(path)

    # Estimate fs
    fs = 256.0
    # for cand in ["timestamps", "time", "Time", "timestamp", "TS"]:
    #     if cand in df.columns:
    #         fs = estimate_fs_from_timestamps(df[cand], default_fs=256.0)
    #         break

    # Pick available EEG channels
    channels = [c for c in EEG_CHANNELS if c in df.columns]
    if not channels:
        return Summary(fs=fs, channels_used=0, abs_overall={}, rel_overall_pct={})

    # Per-channel processing and PSD
    per_ch_abs = {bn: [] for bn in BANDS}
    per_ch_rel = {bn: [] for bn in BANDS}

    for ch in channels:
        x = preprocess_channel(df[ch].dropna().to_numpy(), fs)
        f, Pxx = compute_welch_psd(x, fs)
        if f.size == 0:
            continue
        # absolute per band
        band_abs = {bn: integrate_band(f, Pxx, rng) for bn, rng in BANDS.items()}
        # total in the same range as band-pass
        total = integrate_band(f, Pxx, TOTAL_PWR_BAND)
        # relative %
        band_rel = {bn: (band_abs[bn] / total * 100.0) if (total and np.isfinite(total)) else np.nan
                    for bn in BANDS}

        for bn in BANDS:
            per_ch_abs[bn].append(band_abs[bn])
            per_ch_rel[bn].append(band_rel[bn])

    # Mean across channels → overall
    abs_overall = {bn: float(np.nanmean(per_ch_abs[bn])) if per_ch_abs[bn] else np.nan for bn in BANDS}
    rel_overall = {bn: float(np.nanmean(per_ch_rel[bn])) if per_ch_rel[bn] else np.nan for bn in BANDS}

    return Summary(fs=fs, channels_used=len(channels), abs_overall=abs_overall, rel_overall_pct=rel_overall)
