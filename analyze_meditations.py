# analyze_meditations.py
# Build per-session pre/post metrics, run within-subject stats per meditation,
# run across-meditation Friedman tests, save CSVs + plots.

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.stats import wilcoxon, friedmanchisquare
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

from preprocessings import summarize_one_file, BANDS  # your module

# ---------- Helpers ----------

def find_subject_dirs(root: str) -> List[str]:
    return sorted(
        d for d in (os.path.join(root, x) for x in os.listdir(root))
        if os.path.isdir(d)
    )

def list_csvs(folder: str) -> List[str]:
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".csv")
    )

def parse_med_from_name(fname: str) -> str:
    """
    Try to extract meditation code from filename.
    Looks for one of: lkm, fam, om, mm, bsm (case-insensitive).
    Falls back to the last token between underscores before suffix.
    """
    base = os.path.basename(fname).lower()
    m = re.search(r'_(lkm|fam|om|mm|bsm)_', base)
    if m:
        return m.group(1).upper()
    # fallback heuristic
    toks = base.replace(".csv", "").split("_")
    for t in reversed(toks):
        if t not in {"pre", "post"} and not t.isdigit():
            return t.upper()
    return "UNK"

def safe_rel(path: str) -> Dict[str, float]:
    try:
        s = summarize_one_file(path)   # full recording
        return s.rel_overall_pct       # {'alpha': %, ...}
    except Exception as e:
        print(f"[WARN] {path}: {e}")
        return {}

def bootstrap_ci_mean(x: np.ndarray, nboot=2000, ci=0.95, random_state=42) -> Tuple[float, float]:
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan)
    means = []
    for _ in range(nboot):
        samp = rng.choice(x, size=x.size, replace=True)
        means.append(np.mean(samp))
    lower = np.quantile(means, (1-ci)/2)
    upper = np.quantile(means, 1-(1-ci)/2)
    return float(lower), float(upper)

def rank_biserial_from_wilcoxon(x: np.ndarray) -> float:
    """
    Effect size r_rb = (sum positive ranks - sum negative ranks) / (n(n+1)/2)
    Compute from signed diffs; zero differences are dropped like wilcoxon.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x != 0)]
    n = x.size
    if n == 0:
        return np.nan
    # rank absolute values
    ranks = pd.Series(np.abs(x)).rank(method="average")
    pos = ranks[x > 0].sum()
    neg = ranks[x < 0].sum()
    denom = n*(n+1)/2
    return float((pos - neg) / denom)

# ---------- Build long table of per-session deltas ----------

def build_long_table(data_root: str) -> pd.DataFrame:
    """
    Returns long-form table with columns:
    subject, meditation, band, pre_rel, post_rel, delta_pp
    """
    rows = []
    for subj_dir in find_subject_dirs(data_root):
        subj = os.path.basename(subj_dir)
        files = list_csvs(subj_dir)
        # group files by (session key) = everything except trailing _pre/_post
        # We'll match pairs by matching base without _pre/_post
        base_to_paths = {}
        for f in files:
            base = re.sub(r'_(pre|post)\.csv$', '', os.path.basename(f), flags=re.I)
            base_to_paths.setdefault(base, []).append(f)

        for base, fpaths in base_to_paths.items():
            # expect up to 2 files: *_pre.csv and *_post.csv
            pre = [p for p in fpaths if p.lower().endswith('_pre.csv')]
            post = [p for p in fpaths if p.lower().endswith('_post.csv')]
            if not pre or not post:
                continue
            pre, post = pre[0], post[0]
            med = parse_med_from_name(pre)

            rel_pre = safe_rel(pre)
            rel_post = safe_rel(post)
            if not rel_pre or not rel_post:
                continue

            for band in BANDS.keys():
                pre_v  = rel_pre.get(band, np.nan)
                post_v = rel_post.get(band, np.nan)
                if not np.isfinite(pre_v) or not np.isfinite(post_v):
                    continue
                rows.append({
                    "subject": subj,
                    "session": base,
                    "meditation": med,
                    "band": band,
                    "pre_rel": float(pre_v),
                    "post_rel": float(post_v),
                    "delta_pp": float(post_v - pre_v),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No pre/post pairs found. Check filenames and folders.")
    return df

# ---------- Stats per meditation × band ----------

def per_meditation_stats(df_long: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for med in sorted(df_long.meditation.unique()):
        for band in BANDS.keys():
            d = df_long[(df_long.meditation == med) & (df_long.band == band)]
            # aggregate per subject (in case multiple sessions slip in)
            dsub = d.groupby("subject", as_index=False)["delta_pp"].mean()
            x = dsub["delta_pp"].to_numpy()
            n = np.isfinite(x).sum()
            if n < 1:
                continue
            mean = float(np.nanmean(x))
            sd   = float(np.nanstd(x, ddof=1)) if n > 1 else np.nan
            medn = float(np.nanmedian(x))
            lo, hi = bootstrap_ci_mean(x) if n > 1 else (np.nan, np.nan)
            p = np.nan
            if n >= 1 and np.any(x != 0):
                try:
                    # Wilcoxon signed-rank vs 0; zero-diffs dropped by default
                    p = wilcoxon(x, zero_method='wilcox', alternative='two-sided').pvalue
                except ValueError:
                    p = np.nan
            r_rb = rank_biserial_from_wilcoxon(x)
            recs.append({
                "meditation": med,
                "band": band,
                "n_subjects": int(n),
                "mean_delta_pp": mean,
                "median_delta_pp": medn,
                "sd_delta_pp": sd,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "wilcoxon_p": p,
                "rank_biserial_r": r_rb,
            })
    out = pd.DataFrame(recs)
    # FDR across all Wilcoxon tests
    if not out.empty and out["wilcoxon_p"].notna().any():
        mask = out["wilcoxon_p"].notna()
        q = multipletests(out.loc[mask, "wilcoxon_p"], method="fdr_bh")[1]
        out.loc[mask, "q_fdr"] = q
    else:
        out["q_fdr"] = np.nan
    return out

# ---------- Friedman across meditations (per band) ----------

def friedman_by_band(df_long: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for band in BANDS.keys():
        d = df_long[df_long.band == band]
        # Pivot to subjects × meditations with mean delta per cell
        pivot = d.pivot_table(index="subject", columns="meditation", values="delta_pp", aggfunc="mean")
        # keep subjects who have at least 2 meditations (Friedman needs >=2, ideally all 5)
        pivot = pivot.dropna(axis=0, how="any")  # require full data for fair test
        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            recs.append({"band": band, "n_subjects": int(pivot.shape[0]), "n_meds": int(pivot.shape[1]),
                         "friedman_chi2": np.nan, "friedman_p": np.nan})
            continue
        try:
            stat, p = friedmanchisquare(*[pivot[col].to_numpy() for col in pivot.columns])
        except Exception:
            stat, p = np.nan, np.nan
        recs.append({
            "band": band,
            "n_subjects": int(pivot.shape[0]),
            "n_meds": int(pivot.shape[1]),
            "friedman_chi2": stat,
            "friedman_p": p
        })
    out = pd.DataFrame(recs)
    if not out.empty and out["friedman_p"].notna().any():
        mask = out["friedman_p"].notna()
        out.loc[mask, "q_fdr"] = multipletests(out.loc[mask, "friedman_p"], method="fdr_bh")[1]
    else:
        out["q_fdr"] = np.nan
    return out

# ---------- Plots ----------

def plot_heatmap_mean_delta(df_long: pd.DataFrame, out_path: str):
    mat = df_long.pivot_table(index="band", columns="meditation", values="delta_pp", aggfunc="mean")
    if mat.empty:
        return
    plt.figure(figsize=(8, 5))
    bands = list(BANDS.keys())
    mat = mat.reindex(index=bands)
    im = plt.imshow(mat, aspect="auto")
    plt.xticks(range(mat.shape[1]), mat.columns, rotation=0)
    plt.yticks(range(mat.shape[0]), mat.index)
    plt.colorbar(im, label="Mean Δ (pp)")
    plt.title("Mean post–pre change by meditation and band")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_boxplots(df_long: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for band in BANDS.keys():
        d = df_long[df_long.band == band]
        if d.empty:
            continue
        order = sorted(d.meditation.unique())
        plt.figure(figsize=(8, 5))
        data = [d[d.meditation == m]["delta_pp"].dropna() for m in order]
        plt.boxplot(data, labels=order, showfliers=False)
        plt.axhline(0, color="k", lw=1, ls="--")
        plt.title(f"Δ (post–pre) for {band}")
        plt.ylabel("Δ (percentage points)")
        plt.xlabel("Meditation")
        plt.tight_layout()
        f = os.path.join(out_dir, f"box_{band}.png")
        plt.savefig(f, dpi=200)
        plt.close()

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser("Analyze meditation effects (within-subject)")
    ap.add_argument("--data_root", default="data_collection", help="Folder with subject subfolders")
    ap.add_argument("--out_dir",   default="analysis_outputs", help="Where to save CSVs and plots")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Build long table
    df_long = build_long_table(args.data_root)
    df_long.to_csv(os.path.join(args.out_dir, "per_session_long.csv"), index=False)

    # 2) Per-meditation stats (Wilcoxon + effect sizes)
    df_med = per_meditation_stats(df_long)
    df_med.to_csv(os.path.join(args.out_dir, "per_meditation_stats.csv"), index=False)

    # 3) Across meditations (Friedman per band)
    df_fried = friedman_by_band(df_long)
    df_fried.to_csv(os.path.join(args.out_dir, "friedman_per_band.csv"), index=False)

    # 4) Plots
    plot_heatmap_mean_delta(df_long, os.path.join(args.out_dir, "heatmap_mean_delta.png"))
    plot_boxplots(df_long, os.path.join(args.out_dir, "boxplots"))

    # 5) Console summary
    print("\n=== Per-meditation summary (top 10 by |mean Δ|) ===")
    if not df_med.empty:
        print(df_med.reindex(df_med["mean_delta_pp"].abs().sort_values(ascending=False).index).head(10))
    print("\n=== Friedman per band ===")
    print(df_fried)

if __name__ == "__main__":
    main()
