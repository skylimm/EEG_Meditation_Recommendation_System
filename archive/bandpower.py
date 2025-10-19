import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, welch
from pathlib import Path

# ============ Config ============
CSV_PATH = Path("./data/dingyi/before_eeg.csv")  # change if needed
CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
BANDS = {
    "Delta (1-4)":  (0.5, 4),
    "Theta (4-7)":  (4, 8),
    "Alpha (8-12)": (8, 13),
    "Beta (13-30)": (13, 30),
    "Gamma (30-45)": (30, 45),  # Muse gamma is noisy; optional
}
BANDPASS = (1, 45)      # Hz
NOTCH_FREQ = 50         # Hz (SG mains)
NOTCH_Q = 30
ARTIFACT_THRESH = 1000  # µV (Muse saturation)
# ================================

# --- Load ---
df = pd.read_csv(CSV_PATH)
if "timestamps" not in df.columns:
    raise ValueError("Expected a 'timestamps' column (Unix seconds).")

# Infer sampling rate from timestamps
t = df["timestamps"].to_numpy()
if len(t) < 3:
    raise ValueError("Not enough samples.")
dt = np.median(np.diff(t))
fs = float(np.round(1.0 / dt))  # ~256 for Muse 2
print(f"Estimated sample rate: {fs} Hz (dt ≈ {dt:.6f} s)")

# Convert to datetime for plotting
df["datetime"] = pd.to_datetime(df["timestamps"], unit="s")

# --- Artifact mask (±1000 µV hits) ---
art_mask = (df[CHANNELS].abs() >= ARTIFACT_THRESH).any(axis=1)
n_art = int(art_mask.sum())
print(f"Artifact samples (|x| >= {ARTIFACT_THRESH} µV): {n_art} / {len(df)}")

# --- Filters: 50 Hz notch + 1–40 Hz band-pass ---
def design_bandpass(low, high, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return b, a

def apply_filters(x, fs):
    # handle NaNs by linear interpolation first
    s = pd.Series(x)
    x_clean = s.where(~s.isna()).interpolate(limit_direction="both").to_numpy()

    # notch
    b_notch, a_notch = iirnotch(w0=NOTCH_FREQ/(fs/2), Q=NOTCH_Q)
    x_notched = filtfilt(b_notch, a_notch, x_clean)

    # band-pass
    b_bp, a_bp = design_bandpass(BANDPASS[0], BANDPASS[1], fs)
    x_band = filtfilt(b_bp, a_bp, x_notched)
    return x_band

# Prepare a copy with artifacts set to NaN so they don't pollute PSD
df_clean = df.copy()
df_clean.loc[art_mask, CHANNELS] = np.nan

# Filter each channel
filtered = {}
for ch in CHANNELS:
    filtered[ch] = apply_filters(df_clean[ch].to_numpy(), fs)
filtered_df = pd.DataFrame(filtered)
filtered_df["datetime"] = df["datetime"]

# --- Bandpower via Welch (relative) ---
def bandpower_welch(x, fs, band, nperseg=None, noverlap=None):
    # Remove NaNs (from artifact rejection)
    x = pd.Series(x).dropna().to_numpy()
    if len(x) < 4:
        return np.nan
    if nperseg is None:
        nperseg = int(fs * 2)  # 2-second windows
    if noverlap is None:
        noverlap = nperseg // 2
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    # integrate band
    idx = (f >= band[0]) & (f <= band[1])
    return np.trapz(Pxx[idx], f[idx])

def compute_relative_bandpowers(x, fs, bands):
    total = bandpower_welch(x, fs, BANDPASS)  # total 1–40 Hz power
    out = {}
    for name, (lo, hi) in bands.items():
        p = bandpower_welch(x, fs, (lo, hi))
        out[name] = np.nan if (total is None or total == 0 or np.isnan(total)) else p / total
    return out

# Per-channel relative bandpowers
rel_bp_by_channel = {ch: compute_relative_bandpowers(filtered_df[ch].to_numpy(), fs, BANDS)
                     for ch in CHANNELS}

# Average across channels for a simple overall view
avg_rel_bp = {}
for band_name in BANDS.keys():
    vals = [rel_bp_by_channel[ch][band_name] for ch in CHANNELS]
    avg_rel_bp[band_name] = np.nanmean(vals)

print("\nRelative Bandpowers (per channel):")
for ch, d in rel_bp_by_channel.items():
    print(ch, {k: round(v, 3) if v==v else None for k, v in d.items()})

print("\nRelative Bandpowers (average across channels):")
print({k: round(v, 3) if v==v else None for k, v in avg_rel_bp.items()})


# --- Bandpower via Welch (absolute) -----------------------------------------
def compute_absolute_bandpowers(x, fs, bands):
    """Return absolute band powers (µV²) for each band."""
    out = {}
    for name, (lo, hi) in bands.items():
        out[name] = bandpower_welch(x, fs, (lo, hi))  # integrates PSD ⇒ µV²
    # it's often handy to keep the total bandpass power too
    out["Total (1-40)"] = bandpower_welch(x, fs, BANDPASS)
    return out

# Per-channel absolute bandpowers
abs_bp_by_channel = {
    ch: compute_absolute_bandpowers(filtered_df[ch].to_numpy(), fs, BANDS)
    for ch in CHANNELS
}

# Average across channels (simple overall view)
avg_abs_bp = {}
for band_name in list(BANDS.keys()) + ["Total (1-40)"]:
    vals = [abs_bp_by_channel[ch][band_name] for ch in CHANNELS]
    avg_abs_bp[band_name] = np.nanmean(vals)

print("\nAbsolute Bandpowers (per channel) [µV²]:")
for ch, d in abs_bp_by_channel.items():
    print(ch, {k: (None if pd.isna(v) else round(float(v), 3)) for k, v in d.items()})

print("\nAbsolute Bandpowers (average across channels) [µV²]:")
print({k: (None if pd.isna(v) else round(float(v), 3)) for k, v in avg_abs_bp.items()})

# Optional: dataframe for saving/plotting
abs_bp_df = pd.DataFrame.from_dict(abs_bp_by_channel, orient="index")
# abs_bp_df.to_csv("absolute_bandpowers.csv")

print("absolute power and relative power values we computed are not frequencies — they’re amplitudes of energy in those frequency ranges.")

# --- PLOTS ---

# # 1) Band-pass filtered plot (all channels)
# plt.figure(figsize=(14, 7))
# for ch in CHANNELS:
#     plt.plot(filtered_df["datetime"], filtered_df[ch], label=ch, linewidth=0.8)
# plt.title("Band‑pass Filtered EEG (1–40 Hz) with 50 Hz Notch")
# plt.xlabel("Time")
# plt.ylabel("Amplitude (µV)")
# plt.legend()
# plt.tight_layout()
# plt.show()

# # 2) Bandpower chart (average across channels)
# bands_order = list(BANDS.keys())
# avg_vals = [avg_rel_bp[b] for b in bands_order]
#
# plt.figure(figsize=(10, 5))
# plt.bar(bands_order, avg_vals)
# plt.title("Relative Bandpower (Averaged Across Channels)")
# plt.ylabel("Relative Power (0–1)")
# plt.xticks(rotation=0)
# plt.tight_layout()
# plt.show()

# # (Optional) Per‑channel bandpower as small multiples
# fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
# axes = axes.ravel()
# for i, ch in enumerate(CHANNELS):
#     vals = [rel_bp_by_channel[ch][b] for b in bands_order]
#     axes[i].bar(bands_order, vals)
#     axes[i].set_title(f"{ch} Relative Bandpower")
#     axes[i].set_ylabel("Rel. Power")
#     axes[i].tick_params(axis='x', labelrotation=0)
# plt.suptitle("Relative Bandpower per Channel")
# plt.tight_layout(rect=[0, 0.03, 1, 0.97])
# plt.show()

# --- Plot absolute bandpowers (avg) ---
bands_order = list(BANDS.keys())
avg_abs_vals = [avg_abs_bp[b] for b in bands_order]

plt.figure(figsize=(10, 5))
plt.bar(bands_order, avg_abs_vals)
plt.title("Absolute Bandpower (Averaged Across Channels)")
plt.ylabel("Power (µV²)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
