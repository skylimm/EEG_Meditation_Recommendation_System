import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window, iirnotch, butter, filtfilt, hilbert
import pywt

EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
BANDS = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta":  (13, 30),
    "Gamma": (30, 45),
}

ARTIFACT_THRESH = 1000  # µV
NOTCH_FREQ = 50
NOTCH_Q = 30
BANDPASS = (1, 45)

# ============================Preprocessings============================
def estimate_fs(timestamps: np.ndarray):
    """estimates sampling rate (Hz) from df['timestamps'] by
    taking the median time difference and inverting it"""
    dt = np.median(np.diff(timestamps)) #median safer than mean, in case got one time packet missing
    return float(np.round(1.0 / dt)), dt

def _bandpass_filter(sig, fs, low, high, order=4):
    """keeps only frequencies between low and high (e.g. 1–45 Hz EEG range).
    Removes very slow drift (<1 Hz) and high-frequency noise (>45 Hz)"""
    nyq = fs / 2
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, sig)

def preprocess(sig, fs):
    """Artifact kill + interpolate, notch @ 50Hz, broad bandpass 1–45 Hz."""
    x = sig.astype(float).copy() #convert to float
    x[np.abs(x) >= ARTIFACT_THRESH] = np.nan #artifact removal and replace with Nan(above 1000µV)
    x = pd.Series(x).interpolate(limit_direction="both").to_numpy() #fills NaN gaps smoothly so the signal is continuous again
    b_notch, a_notch = iirnotch(NOTCH_FREQ/(fs/2), Q=NOTCH_Q) #removes single frequency (50 Hz), which is mains electricity hum in sg
    x = filtfilt(b_notch, a_notch, x)
    x = _bandpass_filter(x, fs, BANDPASS[0], BANDPASS[1]) #keeps frequency between 1-45Hz
    return x


# ============================FFT============================
"""find out how much power is in each EEG frequency band (Delta, Theta, Alpha, Beta, Gamma) over time."""
def sliding_fft_bandpower(x, fs, bands=BANDS, win_sec=2.0, step_sec=0.5, window="hann"):
    """Slide a window over the signal → compute FFT-based PSD inside each window
    → integrate PSD inside each band to get absolute power (µV²) over time."""
    nperseg = int(round(win_sec * fs))
    step = int(round(step_sec * fs))
    win = get_window(window, nperseg, fftbins=True)
    U = (win**2).sum()
    times = []
    band_abs = {k: [] for k in bands.keys()}

    for start in range(0, len(x) - nperseg + 1, step):
        seg = x[start:start + nperseg]
        seg = seg - seg.mean()
        segw = seg * win
        X = np.fft.rfft(segw)
        freqs = np.fft.rfftfreq(nperseg, d=1/fs)
        psd = (np.abs(X)**2) / U * (2.0/fs)  # µV²/Hz
        for name, (f1, f2) in bands.items():
            idx = (freqs >= f1) & (freqs < f2)
            band_abs[name].append(np.trapz(psd[idx], freqs[idx]) if np.any(idx) else np.nan)
        times.append((start + nperseg/2) / fs)
    # 0-2sec -> 0.5-2.5sec
    times = np.array(times)
    return times, {k: np.array(v) for k, v in band_abs.items()}

"""
Using FFT-based power for before/after meditation comparison 
Using DWT envelopes to show dynamic visualizations of brainwave activity across time, e.g. illustrate how Beta rises during focus tasks
"""
# ============================DWT============================
def map_dwt_levels_to_bands(fs, levels):
    """shows which DWT "detail level" corresponds to which frequency range at fs"""
    out = {}
    for j in range(1, levels + 1):
        f_high = fs / (2**j)
        f_low  = fs / (2**(j + 1))
        out[j] = (f_low, f_high)
    return out

def dwt_band_envelopes(x, fs, bands=BANDS, wavelet="db4", levels=6, smooth_sec=0.25):
    """Reconstruct band-limited signals from DWT levels, Hilbert envelope, smoothens the amplitude"""
    coeffs = pywt.wavedec(x, wavelet, level=levels)
    n = len(x)

    # one-level reconstructions
    level_to_sig = {}
    for j in range(1, levels + 1):
        keep = [np.zeros_like(c) for c in coeffs]
        keep[j] = coeffs[j].copy()
        rec = pywt.waverec(keep, wavelet)[:n]
        level_to_sig[j] = rec

    # overlap-assign levels to EEG bands
    dwt_levels = map_dwt_levels_to_bands(fs, levels)
    band_signals = {}
    for name, (f1, f2) in bands.items():
        acc = np.zeros(n, dtype=float)
        for j, (fl, fh) in dwt_levels.items():
            if not (fh <= f1 or fl >= f2):  # overlaps
                acc += level_to_sig[j]
        band_signals[name] = acc

    # envelopes
    win = max(1, int(round(smooth_sec * fs)))
    envelopes = {}
    for name, sig in band_signals.items():
        env = np.abs(hilbert(sig))
        if win > 1:
            env = pd.Series(env).rolling(window=win, center=True, min_periods=1).mean().to_numpy()
        envelopes[name] = env

    t = np.arange(n) / fs
    return t, envelopes


# ============================Formatting helpers for Band Summary============================
def _fmt_percent(x):
    try:
        return f"{float(x):.1f}%"
    except Exception:
        return "—"

def _round_units(x):
    # nearest 1 (e.g., 231 -> 231, 383.4 -> 383)
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return "—"

def format_overall_summary(overall: dict, method_used: str) -> str:
    """
    Build the 'Overall summary' and 'Relative percentages' text block
    exactly as requested for FFT and DWT.
    """
    lines = []
    if method_used.lower() == "fft":
        lines.append(f"```\nShowing results for FFT..\n"
                     f"Absolute Power: Power (µV²) in a band\n"
                     f"Relative Power: % of total power\n```")

        for band in ["Delta","Theta","Alpha","Beta","Gamma"]:
            key = f"{band}_abs_mean"
            lines.append(f"{key}: ~{overall.get(key, float('nan'))} µV²\n")
        lines.append("Relative percentages:\n")
        for band in ["Delta","Theta","Alpha","Beta","Gamma"]:
            key = f"{band}_rel_mean"
            lines.append(f"{key}: {_fmt_percent(overall.get(key, float('nan')))}\n")

    else:  # DWT
        lines.append(f"```\nShowing results for DWT..\n"
                     f"Absolute Power: Mean amplitude (µV) of oscillations\n"
                     f"Relative Power: % of total amplitude\n```")

        lines.append("Overall summary:\n")
        for band in ["Delta","Theta","Alpha","Beta","Gamma"]:
            key = f"{band}_amp_mean"
            lines.append(f"{key}: ~{_round_units(overall.get(key, float('nan')))} µV\n")
        lines.append("Relative percentages:\n")
        for band in ["Delta","Theta","Alpha","Beta","Gamma"]:
            key = f"{band}_rel_mean"
            lines.append(f"{key}: {_fmt_percent(overall.get(key, float('nan')))}\n")
    return "\n".join(lines).strip()

def format_per_channel_summary(per_ch: dict, method_used: str) -> str:
    """
    Build the same style block per channel.
    """
    blocks = []
    for ch, metrics in per_ch.items():
        lines = [f"**{ch}**", ""]
        if method_used.lower() == "fft":
            lines.append("Overall summary:\n")
            for band in ["Delta","Theta","Alpha","Beta","Gamma"]:
                key = f"{band}_abs_mean"
                lines.append(f"{key}: ~{metrics.get(key, float('nan'))} µV²\n")
            lines.append("Relative percentages:\n")
            for band in ["Delta","Theta","Alpha","Beta","Gamma"]:
                key = f"{band}_rel_mean"
                lines.append(f"{key}: {_fmt_percent(metrics.get(key, float('nan')))}\n")
        else:
            lines.append("Overall summary:\n")
            for band in ["Delta","Theta","Alpha","Beta","Gamma"]:
                key = f"{band}_amp_mean"
                lines.append(f"{key}: ~{_round_units(metrics.get(key, float('nan')))} µV\n")
            lines.append("Relative percentages:\n")
            for band in ["Delta","Theta","Alpha","Beta","Gamma"]:
                key = f"{band}_rel_mean"
                lines.append(f"{key}: {_fmt_percent(metrics.get(key, float('nan')))}\n")
        blocks.append("\n".join(lines).strip())
    return "\n\n---\n\n".join(blocks)

def format_single_channel_summary(metrics: dict, method_used: str) -> str:
    """
    One-channel version of the formatted summary block.
    Returns a string you can drop into st.markdown inside an expander.
    """
    lines = []
    if method_used.lower() == "fft":
        lines.append("Overall summary:\n")
        for band in ["Delta","Theta","Alpha","Beta","Gamma"]:
            key = f"{band}_abs_mean"
            lines.append(f"{key}: ~{metrics.get(key, float('nan'))} µV²\n")
        lines.append("Relative percentages:\n")
        for band in ["Delta","Theta","Alpha","Beta","Gamma"]:
            key = f"{band}_rel_mean"
            lines.append(f"{key}: {_fmt_percent(metrics.get(key, float('nan')))}\n")
    else:  # DWT
        lines.append("Overall summary:\n")
        for band in ["Delta","Theta","Alpha","Beta","Gamma"]:
            key = f"{band}_amp_mean"
            lines.append(f"{key}: ~{_round_units(metrics.get(key, float('nan')))} µV\n")
        lines.append("Relative percentages:\n")
        for band in ["Delta","Theta","Alpha","Beta","Gamma"]:
            key = f"{band}_rel_mean"
            lines.append(f"{key}: {_fmt_percent(metrics.get(key, float('nan')))}\n")
    return "\n".join(lines).strip()



# ============================Band power summaries (per-channel & overall)============================
def per_channel_summary(df, method="fft", channel_list=None,
                        bands=BANDS, win_sec=2.0, step_sec=0.5, window="hann",
                        wavelet="db4", levels=6, smooth_sec=0.25):
    """
    Returns:
      per_channel: dict(channel -> dict of metrics)
      fs: float sampling rate
    """
    if channel_list is None:
        channel_list = [c for c in EEG_CHANNELS if c in df.columns]

    if "timestamps" in df.columns:
        fs, _ = estimate_fs(df["timestamps"].to_numpy())
    else:
        fs = 256.0

    out = {}
    for ch in channel_list:
        x = df[ch].dropna().to_numpy()
        x = preprocess(x, fs)

        if method.lower() == "fft":
            _, band_abs = sliding_fft_bandpower(x, fs, bands, win_sec, step_sec, window)
            summary = {f"{band}_abs_mean": float(np.nanmean(vals))
                       for band, vals in band_abs.items()}
            stacked = np.vstack([band_abs[b] for b in bands.keys()])
            total = np.nansum(stacked, axis=0)
            for band in bands.keys():
                with np.errstate(invalid="ignore", divide="ignore"):
                    rel = 100.0 * band_abs[band] / total
                summary[f"{band}_rel_mean"] = float(np.nanmean(rel))
        elif method.lower() == "dwt":
            # DWT envelopes (amplitude in µV over time)
            _, envelopes = dwt_band_envelopes(x, fs, bands, wavelet, levels, smooth_sec)

            # Mean amplitude per band (µV)
            summary = {f"{band}_amp_mean": float(np.nanmean(envelopes[band]))
                       for band in bands.keys()}

            stacked = np.vstack([envelopes[b] for b in bands.keys()])  # shape: (B, T)
            total = np.nansum(stacked, axis=0)  # (T,)
            for band in bands.keys():
                with np.errstate(invalid="ignore", divide="ignore"):
                    rel = 100.0 * envelopes[band] / total
                summary[f"{band}_rel_mean"] = float(np.nanmean(rel))

        else:
            raise ValueError("method must be 'fft' or 'dwt'")

        out[ch] = summary

    return out, fs

def overall_summary(per_channel_dict, method="fft", bands=BANDS):
    """Mean of per-channel metrics (across channels)."""
    if method.lower() == "fft":
        overall_abs = {
            f"{band}_abs_mean": float(np.mean([per_channel_dict[ch][f"{band}_abs_mean"] for ch in per_channel_dict]))
            for band in bands.keys()
        }
        overall_rel = {
            f"{band}_rel_mean": float(np.mean([per_channel_dict[ch][f"{band}_rel_mean"] for ch in per_channel_dict]))
            for band in bands.keys()
        }
        return {**overall_abs, **overall_rel}
    else:
        overall_amp = {
            f"{band}_amp_mean": float(np.mean([per_channel_dict[ch][f"{band}_amp_mean"] for ch in per_channel_dict]))
            for band in bands.keys()
        }
        overall_rel = {
            f"{band}_rel_mean": float(np.mean([per_channel_dict[ch][f"{band}_rel_mean"] for ch in per_channel_dict]))
            for band in bands.keys()
        }
        return {**overall_amp, **overall_rel}


# ============================Plot Helpers============================
# for each channel de plot
def plot_band_timeseries_fft(times, band_abs, bands=BANDS, title_suffix=""):
    colors = {"Delta":"tab:blue","Theta":"tab:purple","Alpha":"tab:green","Beta":"tab:orange","Gamma":"tab:red"}
    fig, ax = plt.subplots(figsize=(10, 4))
    for name in bands.keys():
        ax.plot(times, band_abs[name], label=name, color=colors.get(name, None), linewidth=1.5)
    ax.set_title(f"Sliding FFT Band Power (µV²){title_suffix}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Absolute power (µV²)")
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    return fig

def plot_band_timeseries_dwt(t_axis, envelopes, bands=BANDS, title_suffix=""):
    colors = {"Delta":"tab:blue","Theta":"tab:purple","Alpha":"tab:green","Beta":"tab:orange","Gamma":"tab:red"}
    fig, ax = plt.subplots(figsize=(10, 4))
    for name in bands.keys():
        ax.plot(t_axis, envelopes[name], label=name, color=colors.get(name, None), linewidth=1.2)
    ax.set_title(f"DWT Band Envelopes (µV){title_suffix}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    return fig

# for overall

def aggregate_fft_overall(df, channels, fs, bands, win_sec, step_sec, window, preprocess):
    """
    Compute sliding FFT band power per channel, then average across channels
    at each time step. Returns (times, band_abs_mean_dict).
    """
    times = None
    accum = {b: None for b in bands.keys()}
    count = 0
    min_len = None

    for ch in channels:
        x = df[ch].dropna().to_numpy()
        x = preprocess(x, fs)
        t, band_abs = sliding_fft_bandpower(x, fs, bands, win_sec, step_sec, window)

        if times is None:
            times = t
            min_len = len(t)
            for b in bands.keys():
                accum[b] = np.array(band_abs[b], dtype=float)
        else:
            min_len = min(min_len, len(t))
            times = times[:min_len]
            for b in bands.keys():
                accum[b] = accum[b][:min_len] + np.array(band_abs[b][:min_len], dtype=float)

        count += 1

    if count == 0:
        return np.array([]), {b: np.array([]) for b in bands.keys()}

    for b in bands.keys():
        accum[b] = accum[b][:min_len] / float(count)

    return times, accum


def aggregate_dwt_overall(df, channels, fs, bands, wavelet, levels, smooth_sec, preprocess):
    """
    Compute DWT band envelopes per channel, then average across channels
    at each time step. Returns (t_axis, envelopes_mean_dict).
    """
    t_axis = None
    accum = {b: None for b in bands.keys()}
    count = 0
    min_len = None

    for ch in channels:
        x = df[ch].dropna().to_numpy()
        x = preprocess(x, fs)
        t, env = dwt_band_envelopes(
            x, fs, bands, wavelet=wavelet, levels=int(levels), smooth_sec=smooth_sec
        )

        if t_axis is None:
            t_axis = t
            min_len = len(t)
            for b in bands.keys():
                accum[b] = np.array(env[b], dtype=float)
        else:
            min_len = min(min_len, len(t))
            t_axis = t_axis[:min_len]
            for b in bands.keys():
                accum[b] = accum[b][:min_len] + np.array(env[b][:min_len], dtype=float)

        count += 1

    if count == 0:
        return np.array([]), {b: np.array([]) for b in bands.keys()}

    for b in bands.keys():
        accum[b] = accum[b][:min_len] / float(count)

    return t_axis, accum



# ====== Thresholding & Labelling (based on per_channel_summary output) ======

from collections import defaultdict
import numpy as np

# Default rules for RELATIVE power (% of total 1–45 Hz).
# Edit to taste for your study.
REL_RULES_DEFAULT = {
    "Delta": (20, 35),   # Low <20, 20–35 Medium, >=35 High
    "Theta": (15, 25),
    "Alpha": (15, 30),
    "Beta":  (15, 25),
    "Gamma": (5, 15),
}

def _detect_abs_key(method: str) -> str:
    """
    For FFT per_channel_summary: keys are like 'Alpha_abs_mean'
    For DWT per_channel_summary: keys are like 'Alpha_amp_mean'
    """
    if method.lower() == "fft":
        return "_abs_mean"
    elif method.lower() == "dwt":
        return "_amp_mean"
    else:
        raise ValueError("method must be 'fft' or 'dwt'")

def build_absolute_cutoffs_from_summary(per_channel_dict: dict, method: str, bands=BANDS):
    """
    Build absolute 'Low/Medium/High' cutoffs per band by using the distribution
    across the channels currently in per_channel_dict. (33rd/66th percentiles)

    Returns: abs_cuts[band] = (low_cut, high_cut)  in µV² (FFT) or µV (DWT)
    """
    suffix = _detect_abs_key(method)
    abs_cuts = {}
    for band in bands.keys():
        vals = []
        for ch in per_channel_dict:
            key = f"{band}{suffix}"
            if key in per_channel_dict[ch]:
                vals.append(per_channel_dict[ch][key])
        vals = np.array(vals, dtype=float)
        if len(vals) >= 3:
            low_cut = float(np.nanpercentile(vals, 33))
            high_cut = float(np.nanpercentile(vals, 66))
        elif len(vals) > 0:
            # fallback: split min/mid/max
            low_cut = float(np.nanmin(vals))
            high_cut = float(np.nanmedian(vals))
        else:
            low_cut = high_cut = float("nan")
        abs_cuts[band] = (low_cut, high_cut)
    return abs_cuts

def label_relative_value(band: str, rel_percent: float, rel_rules=REL_RULES_DEFAULT) -> str:
    """
    Map a relative percentage to Low/Medium/High using rel_rules[band] = (low, high).
    """
    lo, hi = rel_rules.get(band, (np.nan, np.nan))
    if np.isnan(lo) or np.isnan(hi) or rel_percent is None:
        return "Unknown"
    if rel_percent < lo:   return "Low"
    if rel_percent < hi:   return "Medium"
    return "High"

def label_absolute_value(band: str, abs_value: float, abs_cuts: dict) -> str:
    """
    Map an absolute value to Low/Medium/High using abs_cuts[band] = (low_cut, high_cut).
    """
    low_cut, high_cut = abs_cuts.get(band, (np.nan, np.nan))
    if np.isnan(low_cut) or np.isnan(high_cut) or abs_value is None:
        return "Unknown"
    if abs_value < low_cut:   return "Low"
    if abs_value < high_cut:  return "Medium"
    return "High"

def apply_threshold_labels(per_channel_dict: dict, method: str,
                           bands=BANDS, rel_rules=REL_RULES_DEFAULT):
    """
    Given per_channel_dict from per_channel_summary(...), produce labels and
    include the cutoffs used. Works for both FFT (abs) and DWT (amp).

    Returns: labeled[ch][band] = {
        'abs_or_amp_value': float,
        'abs_or_amp_label': 'Low|Medium|High|Unknown',
        'rel_value_pct': float,
        'rel_label': 'Low|Medium|High|Unknown'
    }, plus a top-level 'ABS_CUTOFFS' entry with the (low, high) used per band.
    """
    suffix_abs = _detect_abs_key(method)
    suffix_rel = "_rel_mean"

    # 1) Build absolute/amp cutoffs from the summary itself (across channels)
    abs_cuts = build_absolute_cutoffs_from_summary(per_channel_dict, method, bands=bands)

    # 2) Label per channel
    labeled = {}
    for ch, metrics in per_channel_dict.items():
        per_band = {}
        for band in bands.keys():
            abs_key = f"{band}{suffix_abs}"
            rel_key = f"{band}{suffix_rel}"

            abs_val = metrics.get(abs_key, None)   # µV² if FFT, µV if DWT
            rel_val = metrics.get(rel_key, None)   # percent

            per_band[band] = {
                "abs_or_amp_value": abs_val,
                "abs_or_amp_label": label_absolute_value(band, abs_val, abs_cuts),
                "rel_value_pct": rel_val,
                "rel_label": label_relative_value(band, rel_val, rel_rules),
            }
        labeled[ch] = per_band

    # 3) Attach the cutoffs used (handy for logs/UIs)
    labeled["ABS_CUTOFFS"] = abs_cuts
    labeled["REL_RULES"] = rel_rules
    labeled["METHOD"] = method.lower()
    return labeled

# ===== Pre/Post comparison plots =====

def _abs_key_for_method(method: str) -> str:
    return "_abs_mean" if method.lower() == "fft" else "_amp_mean"

def plot_overall_abs_prepost(overall_pre: dict, overall_post: dict, method="fft", bands=BANDS, title="Overall Absolute/Amplitude"):
    """
    Bar chart: Pre vs Post for absolute (FFT, µV²) or amplitude (DWT, µV).
    """
    suffix = _abs_key_for_method(method)
    labels = list(bands.keys())
    pre = [overall_pre.get(f"{b}{suffix}", np.nan) for b in labels]
    post = [overall_post.get(f"{b}{suffix}", np.nan) for b in labels]

    x = np.arange(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(x - w/2, pre, width=w, label="Pre")
    ax.bar(x + w/2, post, width=w, label="Post")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("µV²" if method.lower()=="fft" else "µV (envelope)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig

def plot_overall_rel_prepost(overall_pre: dict, overall_post: dict, bands=BANDS, title="Overall Relative Power (%)"):
    """
    Bar chart: Pre vs Post for relative power (% of total) per band.
    """
    labels = list(bands.keys())
    pre = [overall_pre.get(f"{b}_rel_mean", np.nan) for b in labels]
    post = [overall_post.get(f"{b}_rel_mean", np.nan) for b in labels]

    x = np.arange(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(x - w/2, pre, width=w, label="Pre")
    ax.bar(x + w/2, post, width=w, label="Post")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("% of 1–45 Hz total")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig

def plot_overall_rel_change(overall_pre: dict, overall_post: dict, bands=BANDS, title="Change in Relative Power (Post − Pre)"):
    """
    Bar chart: difference in relative power, in percentage points.
    """
    labels = list(bands.keys())
    delta = [ (overall_post.get(f"{b}_rel_mean", np.nan) - overall_pre.get(f"{b}_rel_mean", np.nan)) for b in labels ]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(x, delta)
    ax.axhline(0, linewidth=1, color="k")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("pp (percentage points)")
    ax.set_title(title)
    fig.tight_layout()
    return fig

def plot_rel_change_heatmap(per_pre: dict, per_post: dict, bands=BANDS, title="Per-channel Relative Power Change (pp)"):
    """
    Heatmap of (Post − Pre) for relative power (%), per channel × band.
    """
    chs = list(per_pre.keys())
    bnames = list(bands.keys())
    M = np.zeros((len(chs), len(bnames)))
    for i, ch in enumerate(chs):
        for j, b in enumerate(bnames):
            pre_v = per_pre[ch].get(f"{b}_rel_mean", np.nan)
            post_v = per_post[ch].get(f"{b}_rel_mean", np.nan)
            M[i, j] = post_v - pre_v

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    im = ax.imshow(M, aspect="auto", cmap="coolwarm", vmin=-np.nanmax(np.abs(M)), vmax=np.nanmax(np.abs(M)))
    ax.set_yticks(np.arange(len(chs))); ax.set_yticklabels(chs)
    ax.set_xticks(np.arange(len(bnames))); ax.set_xticklabels(bnames)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("pp")
    fig.tight_layout()
    return fig

# ===== Pre vs Post Line Plots =====

def _available_channels(df):
    return [c for c in EEG_CHANNELS if c in df.columns]

def _series_fft(df, fs, bands=BANDS, scope="Overall", win_sec=2.0, step_sec=0.5, window="hann"):
    chs = _available_channels(df) if scope == "Overall" else [scope]
    t_union = None
    acc = {b: [] for b in bands}
    for ch in chs:
        x = preprocess(df[ch].dropna().to_numpy(), fs)
        t, band_abs = sliding_fft_bandpower(x, fs, bands=bands, win_sec=win_sec, step_sec=step_sec, window=window)
        if t_union is None: t_union = t
        L = min(len(t_union), *(len(band_abs[b]) for b in bands))
        for b in bands:
            acc[b].append(np.asarray(band_abs[b][:L]))
        t_union = t_union[:L]
    out = {b: np.nanmean(np.vstack(acc[b]), axis=0) for b in bands}
    return t_union, out

def _series_dwt(df, fs, bands=BANDS, scope="Overall", wavelet="db4", levels=6, smooth_sec=0.25):
    chs = _available_channels(df) if scope == "Overall" else [scope]
    t_union = None
    acc = {b: [] for b in bands}
    for ch in chs:
        x = preprocess(df[ch].dropna().to_numpy(), fs)
        t, env = dwt_band_envelopes(x, fs, bands=bands, wavelet=wavelet, levels=levels, smooth_sec=smooth_sec)
        if t_union is None: t_union = t
        L = min(len(t_union), *(len(env[b]) for b in bands))
        for b in bands:
            acc[b].append(np.asarray(env[b][:L]))
        t_union = t_union[:L]
    out = {b: np.nanmean(np.vstack(acc[b]), axis=0) for b in bands}
    return t_union, out

def plot_band_timeseries_fft_compare(df_pre, df_post, fs, bands=BANDS, scope="Overall",
                                     win_sec=2.0, step_sec=0.5, window="hann"):
    t_pre,  s_pre  = _series_fft(df_pre,  fs, bands, scope, win_sec, step_sec, window)
    t_post, s_post = _series_fft(df_post, fs, bands, scope, win_sec, step_sec, window)
    colors = {"Delta":"tab:blue","Theta":"tab:purple","Alpha":"tab:green","Beta":"tab:orange","Gamma":"tab:red"}
    fig, ax = plt.subplots(figsize=(10,4))
    for name in bands.keys():
        ax.plot(t_pre,  s_pre[name],  linestyle="--", alpha=0.7, color=colors.get(name), label=f"{name} (Pre)")
        ax.plot(t_post, s_post[name], linewidth=1.6,       color=colors.get(name), label=f"{name} (Post)")
    ax.set_title(f"Sliding FFT Band Power (Before vs After) — {scope}")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Absolute power (µV²)")
    ax.legend(ncol=2, fontsize=8); ax.grid(alpha=0.2); fig.tight_layout()
    return fig

def plot_band_timeseries_dwt_compare(df_pre, df_post, fs, bands=BANDS, scope="Overall",
                                     wavelet="db4", levels=6, smooth_sec=0.25):
    t_pre,  e_pre  = _series_dwt(df_pre,  fs, bands, scope, wavelet, levels, smooth_sec)
    t_post, e_post = _series_dwt(df_post, fs, bands, scope, wavelet, levels, smooth_sec)
    colors = {"Delta":"tab:blue","Theta":"tab:purple","Alpha":"tab:green","Beta":"tab:orange","Gamma":"tab:red"}
    fig, ax = plt.subplots(figsize=(10,4))
    for name in bands.keys():
        ax.plot(t_pre,  e_pre[name],  linestyle="--", alpha=0.7, color=colors.get(name), label=f"{name} (Pre)")
        ax.plot(t_post, e_post[name], linewidth=1.6,       color=colors.get(name), label=f"{name} (Post)")
    ax.set_title(f"DWT Band Envelopes (Before vs After) — {scope}")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude (µV)")
    ax.legend(ncol=2, fontsize=8); ax.grid(alpha=0.2); fig.tight_layout()
    return fig




# ===== Percent-change time-series (Pre vs Post) =====
import numpy as np
import matplotlib.pyplot as plt

def _available_channels(df):
    return [c for c in EEG_CHANNELS if c in df.columns]

def _timeseries_fft_abs_rel(df, fs, bands=BANDS, scope="Overall",
                            win_sec=2.0, step_sec=0.5, window="hann"):
    """
    Returns:
      t: time axis (s)
      abs_series: dict[band] -> absolute power time series (µV²)
      rel_series: dict[band] -> relative power time series (% of total at each time)
    scope: "Overall" or a specific channel (e.g., "TP9")
    """
    chs = _available_channels(df) if scope == "Overall" else [scope]
    t_union = None
    acc_abs = {b: [] for b in bands}
    acc_tot = []

    for ch in chs:
        x = preprocess(df[ch].dropna().to_numpy(), fs)
        t, band_abs = sliding_fft_bandpower(x, fs, bands=bands, win_sec=win_sec, step_sec=step_sec, window=window)

        # Truncate/align to common length
        if t_union is None:
            t_union = t
        L = min(len(t_union), *(len(band_abs[b]) for b in bands))
        t_union = t_union[:L]

        # absolute arrays
        stack = np.vstack([np.asarray(band_abs[b][:L]) for b in bands])  # shape (B, L)
        tot = np.nansum(stack, axis=0)  # (L,)
        acc_tot.append(tot)
        for b in bands:
            acc_abs[b].append(np.asarray(band_abs[b][:L]))

    # average across selected channels
    abs_series = {b: np.nanmean(np.vstack(acc_abs[b]), axis=0) for b in bands}
    tot_series = np.nanmean(np.vstack(acc_tot), axis=0)

    # relative % at each time
    rel_series = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for b in bands:
            rel_series[b] = 100.0 * abs_series[b] / tot_series
    return t_union, abs_series, rel_series

def _percent_change(pre, post, mode="ratio"):
    """
    pre, post: 1D np arrays aligned in time
    mode:
      - "ratio": 100 * (post - pre) / pre           (percent change)
      - "pp":    post - pre                         (percentage points, use for relative %)
    """
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    if mode == "pp":
        return post - pre
    # ratio %
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100.0 * (post - pre) / pre

def percent_change_timeseries_fft(df_pre, df_post, fs, bands=BANDS, scope="Overall",
                                  win_sec=2.0, step_sec=0.5, window="hann",
                                  which="absolute", rel_mode="pp"):
    """
    Compute time-series percent-change lines for each band.
      which: "absolute" -> % change of absolute power (ratio %)
             "relative" -> change of relative power (default percentage points)
      rel_mode: "pp" or "ratio" (only used when which == "relative")
    Returns: t_common, change_dict (dict[band]->1D array), ylabel, title_suffix
    """
    t0, abs_pre, rel_pre = _timeseries_fft_abs_rel(df_pre,  fs, bands, scope, win_sec, step_sec, window)
    t1, abs_post, rel_post = _timeseries_fft_abs_rel(df_post, fs, bands, scope, win_sec, step_sec, window)

    # Align lengths (use min)
    L = min(len(t0), len(t1))
    t = t0[:L]

    out = {}
    if which == "absolute":
        for b in bands:
            out[b] = _percent_change(abs_pre[b][:L], abs_post[b][:L], mode="ratio")
        ylabel = "% change vs Pre (absolute µV²)"
        title = f"Percent Change — Absolute Power (FFT) — {scope}"
    else:
        mode = rel_mode  # "pp" or "ratio"
        for b in bands:
            out[b] = _percent_change(rel_pre[b][:L], rel_post[b][:L], mode=("pp" if mode=="pp" else "ratio"))
        ylabel = "Change (percentage points)" if rel_mode == "pp" else "% change vs Pre (relative %)"
        title = f"Change — Relative Power (FFT) — {scope}"

    return t, out, ylabel, title

def plot_percent_change_lines(t, change_dict, bands_select=None, ylabel="", title=""):
    """
    Line plot: one line per selected band of percent-change over time.
    """
    if bands_select is None:
        bands_select = list(change_dict.keys())
    color_map = {"Delta":"tab:blue","Theta":"tab:purple","Alpha":"tab:green","Beta":"tab:orange","Gamma":"tab:red"}
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for b in bands_select:
        ax.plot(t, change_dict[b], label=b, linewidth=1.6, color=color_map.get(b))
    ax.axhline(0, color="k", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=5, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig
