import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, welch
from tqdm import tqdm  # <--- NEW: for progress bar
from pathlib import Path

# ---------------------- Config ----------------------
EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
BANDS = {
    "Delta (1-4)":  (1, 4),
    "Theta (4-7)":  (4, 7),
    "Alpha (8-12)": (8, 12),
    "Beta (13-30)": (13, 30),
    "Gamma (30-45)": (30, 45),
}
BANDPASS = (1, 40)     # Hz
NOTCH_FREQ = 50        # Hz
NOTCH_Q = 30
ARTIFACT_THRESH = 1000.0
TOTAL_PWR_BAND = (1, 45)

FILENAME_RE = re.compile(
    r'^(?P<seq>\d+)_(?P<subject>[^_]+)_(?P<technique>[^_]+)_(?P<when>pre|post)\.csv$',
    re.IGNORECASE
)
# ---------------------- Helpers ----------------------
def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def estimate_fs_from_timestamps(series: pd.Series, default_fs: float = 256.0) -> float:
    try:
        ts = series.to_numpy(dtype=float)
        ts = ts[np.isfinite(ts)]
        if ts.size < 3:
            return default_fs
        diffs = np.diff(ts)
        diffs = diffs[(np.isfinite(diffs)) & (diffs > 0)]
        if diffs.size == 0:
            return default_fs
        med_dt = np.median(diffs)
        # Heuristic: treat large values as milliseconds
        if med_dt > 10:
            med_dt = med_dt / 1000.0
        fs = 1.0 / med_dt
        if 10 < fs < 2048:
            return float(fs)
        return default_fs
    except Exception:
        return default_fs

def bandpass_filter(x, fs, low, high, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, x)

def notch_filter(x, fs, notch_freq=50.0, q=30.0):
    b, a = iirnotch(w0=notch_freq/(fs/2.0), Q=q)
    return filtfilt(b, a, x)

def preprocess_signal(x: np.ndarray, fs: float) -> np.ndarray:
    x = x.astype(float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return x
    # Artifact interpolation
    mask = np.abs(x) <= ARTIFACT_THRESH
    if not np.all(mask):
        good_idx = np.where(mask)[0]
        if good_idx.size >= 2:
            x = np.interp(np.arange(x.size), good_idx, x[good_idx])
    # Filters
    try:
        x = notch_filter(x, fs, NOTCH_FREQ, NOTCH_Q)
        x = bandpass_filter(x, fs, BANDPASS[0], BANDPASS[1], order=4)
    except Exception:
        pass
    return x

def compute_welch_psd(x: np.ndarray, fs: float):
    if x.size < 64:
        return np.array([]), np.array([])
    nperseg = min(1024, x.size)
    noverlap = nperseg // 2
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return f, Pxx

def bandpower(f: np.ndarray, Pxx: np.ndarray, band: tuple) -> float:
    if f.size == 0 or Pxx.size == 0:
        return np.nan
    f1, f2 = band
    idx = (f >= f1) & (f <= f2)
    if not np.any(idx):
        return np.nan
    return float(np.trapz(Pxx[idx], f[idx]))

def summarize_one_file(path: Path) -> dict:
    df = read_csv(path)
    # Estimate fs
    fs = 256.0
    for cand in ["timestamps", "time", "Time", "timestamp", "TS"]:
        if cand in df.columns:
            fs = estimate_fs_from_timestamps(df[cand])
            break
    # Channels available
    channels = [c for c in EEG_CHANNELS if c in df.columns]
    if not channels:
        return {"fs": fs, "channels_used": 0, "per_band": {}}

    abs_list = {b: [] for b in BANDS.keys()}
    rel_list = {b: [] for b in BANDS.keys()}
    for ch in channels:
        x = preprocess_signal(df[ch].dropna().to_numpy(), fs)
        f, Pxx = compute_welch_psd(x, fs)
        if f.size == 0:
            continue
        band_abs = {bn: bandpower(f, Pxx, rng) for bn, rng in BANDS.items()}
        total = bandpower(f, Pxx, TOTAL_PWR_BAND)
        band_rel = {bn: (band_abs[bn] / total * 100.0) if (total and total > 0 and np.isfinite(total)) else np.nan
                    for bn in BANDS.keys()}
        for bn in BANDS.keys():
            abs_list[bn].append(band_abs[bn])
            rel_list[bn].append(band_rel[bn])

    per_band = {}
    for bn in BANDS.keys():
        per_band[bn] = {
            "Abs": float(np.nanmean(abs_list[bn])) if len(abs_list[bn]) else np.nan,
            "RelPct": float(np.nanmean(rel_list[bn])) if len(rel_list[bn]) else np.nan,
        }
    return {"fs": fs, "channels_used": len(channels), "per_band": per_band}

# ---------------------- Batch logic ----------------------
def parse_filename(p: Path):
    m = FILENAME_RE.match(p.name)
    if not m:
        return None
    d = m.groupdict()
    return {
        "seq": int(d["seq"]),
        "subject": d["subject"],
        "technique": d["technique"],
        "when": d["when"].lower(),
    }







def main():
    ap = argparse.ArgumentParser(description="Batch EEG summary (recursive) per subject & technique + aggregates.")
    ap.add_argument("--input", type=Path, required=True, help="Top-level folder (e.g. data_collection)")
    ap.add_argument("--output", type=Path, required=True, help="Output folder for CSV and Excel summaries")
    args = ap.parse_args()

    in_dir: Path = args.input
    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    # 🔍 recursively search all .csv files
    files = list(in_dir.rglob("*.csv"))
    meta = []
    for f in files:
        m = re.match(r"^(?P<seq>\d+)_(?P<subject>[^_]+)_(?P<technique>[^_]+)_(?P<when>pre|post)\.csv$", f.name, re.IGNORECASE)
        if m:
            d = m.groupdict()
            meta.append({
                "seq": int(d["seq"]),
                "subject": d["subject"],
                "technique": d["technique"],
                "when": d["when"].lower(),
                "path": f
            })

    if not meta:
        raise SystemExit("No valid EEG files found matching 'N_subject_technique_pre/post.csv'.")

    # 🧠 group into pairs
    groups = {}
    for m in meta:
        key = (m["seq"], m["subject"], m["technique"])
        groups.setdefault(key, {})[m["when"]] = m["path"]

    rows = []
    all_keys = list(groups.keys())
    print(f"🧩 Found {len(all_keys)} valid pre/post pairs across all folders.\n")

    for (seq, subject, tech) in tqdm(all_keys, desc="Processing pairs", ncols=90):
        files_dict = groups[(seq, subject, tech)]
        pre = files_dict.get("pre")
        post = files_dict.get("post")
        if not pre or not post:
            continue

        pre_sum = summarize_one_file(pre)
        post_sum = summarize_one_file(post)

        for band in BANDS.keys():
            pre_abs = pre_sum["per_band"].get(band, {}).get("Abs", np.nan)
            post_abs = post_sum["per_band"].get(band, {}).get("Abs", np.nan)
            pre_rel = pre_sum["per_band"].get(band, {}).get("RelPct", np.nan)
            post_rel = post_sum["per_band"].get(band, {}).get("RelPct", np.nan)

            abs_change_pct = float(((post_abs - pre_abs) / pre_abs * 100.0)) if (pre_abs and np.isfinite(pre_abs)) else np.nan
            rel_diff_pp = float(post_rel - pre_rel) if (np.isfinite(pre_rel) and np.isfinite(post_rel)) else np.nan

            rows.append({
                "Sequence": seq,
                "Subject": subject,
                "Technique": tech,
                "Band": band,
                "Abs_Pre": pre_abs,
                "Abs_Post": post_abs,
                "Abs_%Change": abs_change_pct,
                "Rel%_Pre": pre_rel,
                "Rel%_Post": post_rel,
                "Rel_Diff_pp": rel_diff_pp,
                "Pre_fs": pre_sum["fs"],
                "Post_fs": post_sum["fs"],
                "Pre_ChannelsUsed": pre_sum["channels_used"],
                "Post_ChannelsUsed": post_sum["channels_used"],
            })

    print("\n✅ All pairs processed. Writing summary files...")

    # 🧾 Detailed per-subject table
    per_subject = pd.DataFrame.from_records(rows).sort_values(["Technique","Subject","Sequence","Band"])
    per_subject_path = out_dir / "summary_per_subject.csv"
    per_subject.to_csv(per_subject_path, index=False)

    # 📊 Aggregate: by technique
    agg1 = (per_subject
            .groupby(["Technique","Band"], as_index=False)[["Abs_Pre","Abs_Post","Abs_%Change","Rel%_Pre","Rel%_Post","Rel_Diff_pp"]]
            .mean(numeric_only=True))
    agg1_path = out_dir / "summary_overall_by_technique.csv"
    agg1.to_csv(agg1_path, index=False)

    # 📈 Aggregate: by technique + sequence
    agg2 = (per_subject
            .groupby(["Technique","Sequence","Band"], as_index=False)[["Abs_Pre","Abs_Post","Abs_%Change","Rel%_Pre","Rel%_Post","Rel_Diff_pp"]]
            .mean(numeric_only=True))
    agg2_path = out_dir / "summary_overall_by_technique_sequence.csv"
    agg2.to_csv(agg2_path, index=False)

    # 🧮 Export to Excel (optional)
    excel_path = out_dir / "EEG_Summary_All.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        per_subject.to_excel(writer, sheet_name="Per_Subject", index=False)
        agg1.to_excel(writer, sheet_name="By_Technique", index=False)
        agg2.to_excel(writer, sheet_name="By_Technique_Seq", index=False)

    print(f"""
🎉 Done!
📁 Outputs saved in: {out_dir}
- summary_per_subject.csv
- summary_overall_by_technique.csv
- summary_overall_by_technique_sequence.csv
- EEG_Summary_All.xlsx (3 sheets)
""")

#
# def main():
#     ap = argparse.ArgumentParser(description="Batch EEG summary (pre/post) per subject & technique + aggregates.")
#     ap.add_argument("--input", type=Path, required=True, help="Folder with CSVs like '1_subject_tech_pre.csv'")
#     ap.add_argument("--output", type=Path, required=True, help="Output folder for CSV summaries")
#     args = ap.parse_args()
#
#     in_dir: Path = args.input
#     out_dir: Path = args.output
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     files = list(in_dir.glob("*.csv"))
#     meta = []
#     for f in files:
#         info = parse_filename(f)
#         if info:
#             meta.append({**info, "path": f})
#
#     if not meta:
#         raise SystemExit("No files matched 'N_subject_technique_pre/post.csv' in the input folder.")
#
#     # Pair pre/post within (seq, subject, technique)
#     groups = {}
#     for m in meta:
#         key = (m["seq"], m["subject"], m["technique"])
#         groups.setdefault(key, {})[m["when"]] = m["path"]
#
#     rows = []
#     for (seq, subject, tech), files_dict in groups.items():
#         pre = files_dict.get("pre")
#         post = files_dict.get("post")
#         if not pre or not post:
#             continue
#
#         pre_sum = summarize_one_file(pre)
#         post_sum = summarize_one_file(post)
#
#         for band in BANDS.keys():
#             pre_abs = pre_sum["per_band"].get(band, {}).get("Abs", np.nan)
#             post_abs = post_sum["per_band"].get(band, {}).get("Abs", np.nan)
#             pre_rel = pre_sum["per_band"].get(band, {}).get("RelPct", np.nan)
#             post_rel = post_sum["per_band"].get(band, {}).get("RelPct", np.nan)
#
#             abs_change_pct = float(((post_abs - pre_abs) / pre_abs * 100.0)) if (pre_abs and np.isfinite(pre_abs)) else np.nan
#             rel_diff_pp = float(post_rel - pre_rel) if (np.isfinite(pre_rel) and np.isfinite(post_rel)) else np.nan
#
#             rows.append({
#                 "Sequence": seq,
#                 "Subject": subject,
#                 "Technique": tech,
#                 "Band": band,
#                 "Abs_Pre": pre_abs,
#                 "Abs_Post": post_abs,
#                 "Abs_%Change": abs_change_pct,
#                 "Rel%_Pre": pre_rel,
#                 "Rel%_Post": post_rel,
#                 "Rel_Diff_pp": rel_diff_pp,
#                 "Pre_fs": pre_sum["fs"],
#                 "Post_fs": post_sum["fs"],
#                 "Pre_ChannelsUsed": pre_sum["channels_used"],
#                 "Post_ChannelsUsed": post_sum["channels_used"],
#             })
#
#     per_subject = pd.DataFrame.from_records(rows).sort_values(["Technique","Subject","Sequence","Band"])
#     per_subject_path = out_dir / "summary_per_subject.csv"
#     per_subject.to_csv(per_subject_path, index=False)
#
#     # 1) Mean across subjects per technique & band
#     agg1 = (per_subject
#             .groupby(["Technique","Band"], as_index=False)[["Abs_Pre","Abs_Post","Abs_%Change","Rel%_Pre","Rel%_Post","Rel_Diff_pp"]]
#             .mean(numeric_only=True))
#     agg1_path = out_dir / "summary_overall_by_technique.csv"
#     agg1.to_csv(agg1_path, index=False)
#
#     # 2) Mean across subjects per technique, sequence & band
#     agg2 = (per_subject
#             .groupby(["Technique","Sequence","Band"], as_index=False)[["Abs_Pre","Abs_Post","Abs_%Change","Rel%_Pre","Rel%_Post","Rel_Diff_pp"]]
#             .mean(numeric_only=True))
#     agg2_path = out_dir / "summary_overall_by_technique_sequence.csv"
#     agg2.to_csv(agg2_path, index=False)
#
#     print("Done.")
#     print(f"- {per_subject_path}")
#     print(f"- {agg1_path}")
#     print(f"- {agg2_path}")

if __name__ == "__main__":
    main()
