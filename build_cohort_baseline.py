# build_cohort_baseline.py
# Scans data_collection/<subject>/*_pre.csv, computes:
# 1) Per-subject baseline mean (relative bandpower) across all their "pre" files
# 2) Cohort stats (mean, sd, percentiles) across subjects (of those means)
#
# Outputs: outputs/per_subject_baseline.csv, outputs/cohort_baseline.csv, outputs/cohort_baseline.json

import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List
from preprocessings import summarize_one_file, BANDS

def find_subject_dirs(data_root: str) -> List[str]:
    return sorted(
        d for d in (os.path.join(data_root, x) for x in os.listdir(data_root))
        if os.path.isdir(d)
    )

def list_pre_files(subject_dir: str) -> List[str]:
    return sorted(
        os.path.join(subject_dir, f)
        for f in os.listdir(subject_dir)
        if f.endswith(".csv") and "_pre" in f.lower()
    )

def safe_rel_dict(path: str) -> Dict[str, float]:
    """Return relative bandpower dict for a file; empty dict if unusable."""
    try:
        s = summarize_one_file(path)  # uses full recording
        if s.channels_used == 0:
            return {}
        return s.rel_overall_pct
    except Exception as e:
        print(f"[WARN] Skipping {path}: {e}")
        return {}

def main():
    ap = argparse.ArgumentParser("Build cohort baselines from *_pre.csv files")
    ap.add_argument("--data_root", default="data_collection", help="Root folder with subject subfolders")
    ap.add_argument("--out_dir",   default="outputs", help="Where to write baseline tables")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    subject_rows = []  # one row per subject with mean baseline per band
    file_rows = []     # optional: one row per pre file processed (for audit)

    subjects = find_subject_dirs(args.data_root)
    if not subjects:
        raise SystemExit(f"No subject folders found under {args.data_root}")

    for subj_dir in subjects:
        subj_id = os.path.basename(subj_dir)
        pre_files = list_pre_files(subj_dir)
        if not pre_files:
            print(f"[INFO] No *_pre.csv in {subj_id}; skipping subject.")
            continue

        rel_list_per_band: Dict[str, List[float]] = {b: [] for b in BANDS.keys()}

        for fpath in pre_files:
            rel = safe_rel_dict(fpath)
            if not rel:
                continue
            for b in BANDS.keys():
                val = rel.get(b, np.nan)
                if val == val:  # finite check
                    rel_list_per_band[b].append(float(val))

            # keep per-file record (optional)
            file_rows.append({
                "subject": subj_id,
                "file": os.path.basename(fpath),
                **{f"{b}_rel": rel.get(b, np.nan) for b in BANDS.keys()}
            })

        # aggregate to subject mean (baseline profile)
        if any(len(v) > 0 for v in rel_list_per_band.values()):
            row = {"subject": subj_id}
            for b in BANDS.keys():
                vals = rel_list_per_band[b]
                row[f"{b}_rel_mean"] = float(np.nanmean(vals)) if vals else np.nan
                row[f"{b}_rel_sd"]   = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else np.nan
                row[f"{b}_n"]        = int(len(vals))
            subject_rows.append(row)
        else:
            print(f"[INFO] {subj_id}: no usable channels across pre files; skipping.")

    if not subject_rows:
        raise SystemExit("No subject baselines computed. Check your files/data_root.")

    # ---- Save per-subject baseline means ----
    df_subject = pd.DataFrame(subject_rows).sort_values("subject")
    per_subject_csv = os.path.join(args.out_dir, "per_subject_baseline.csv")
    df_subject.to_csv(per_subject_csv, index=False)
    print(f"[OK] Wrote {per_subject_csv}")

    # ---- Build cohort stats across subjects (using each subject's mean) ----
    cohort_records = []
    for b in BANDS.keys():
        col = f"{b}_rel_mean"
        series = df_subject[col].dropna().astype(float)
        if series.empty:
            continue
        stats = {
            "band": b,
            "n_subjects": int(series.shape[0]),
            "mean": float(series.mean()),
            "sd": float(series.std(ddof=1)) if series.shape[0] > 1 else 0.0,
            "median": float(series.median()),
            "p10": float(series.quantile(0.10)),
            "p25": float(series.quantile(0.25)),
            "p75": float(series.quantile(0.75)),
            "p90": float(series.quantile(0.90)),
            "min": float(series.min()),
            "max": float(series.max()),
        }
        cohort_records.append(stats)

    df_cohort = pd.DataFrame(cohort_records).set_index("band").loc[list(BANDS.keys())]
    cohort_csv = os.path.join(args.out_dir, "cohort_baseline.csv")
    df_cohort.to_csv(cohort_csv)
    cohort_json = os.path.join(args.out_dir, "cohort_baseline.json")
    with open(cohort_json, "w") as f:
        json.dump(df_cohort.to_dict(orient="index"), f, indent=2)

    print(f"[OK] Wrote {cohort_csv}")
    print(f"[OK] Wrote {cohort_json}")

    # optional audit table per file
    if file_rows:
        df_files = pd.DataFrame(file_rows)
        df_files.to_csv(os.path.join(args.out_dir, "all_pre_files_rel.csv"), index=False)

if __name__ == "__main__":
    main()
