# main.py
# Print overall bandpowers (absolute & relative %) for PRE and POST,
# plus deltas, using a fixed middle-duration slice for consistency.

import argparse
from preprocessings import summarize_one_file, BANDS

def pct_change(new, old):
    if old and old != 0:
        return (new - old) / old * 100.0
    return float('nan')

def main():
    ap = argparse.ArgumentParser("EEG bandpower summary (pre vs post)")
    ap.add_argument("--pre",  required=True, help="Pre-meditation CSV")
    ap.add_argument("--post", required=True, help="Post-meditation CSV")
    args = ap.parse_args()

    pre = summarize_one_file(args.pre,  analysis_sec=args.duration_sec)
    post = summarize_one_file(args.post, analysis_sec=args.duration_sec)

    print(f"Sampling rate (pre)  ≈ {pre.fs:.2f} Hz | Channels used: {pre.channels_used}")
    print(f"Sampling rate (post) ≈ {post.fs:.2f} Hz | Channels used: {post.channels_used}")

    # ----- Absolute -----
    print("=== Overall Absolute Bandpower ===")
    for bn in BANDS.keys():
        pre_v  = pre.abs_overall.get(bn, float('nan'))
        post_v = post.abs_overall.get(bn, float('nan'))
        d_pct  = pct_change(post_v, pre_v)
        print(f"{bn:>6}: pre={pre_v:10.3f} | post={post_v:10.3f} | Δ%={d_pct:7.2f}")

    # ----- Relative (percent of 1–40 Hz) -----
    print("\n=== Overall Relative Bandpower (%) ===")
    for bn in BANDS.keys():
        pre_v  = pre.rel_overall_pct.get(bn, float('nan'))
        post_v = post.rel_overall_pct.get(bn, float('nan'))
        diff   = post_v - pre_v if (pre_v == pre_v and post_v == post_v) else float('nan')  # pp difference
        print(f"{bn:>6}: pre={pre_v:8.3f}% | post={post_v:8.3f}% | Δpp={diff:7.3f}")

    # ----- Quick recommendation hints (editable rules) -----
    print("\n=== Recommendation Hints (edit thresholds as you wish) ===")
    alpha_pp = post.rel_overall_pct.get("alpha", float('nan')) - pre.rel_overall_pct.get("alpha", float('nan'))
    theta_pp = post.rel_overall_pct.get("theta", float('nan')) - pre.rel_overall_pct.get("theta", float('nan'))
    beta_pp  = post.rel_overall_pct.get("beta",  float('nan')) - pre.rel_overall_pct.get("beta",  float('nan'))

    if alpha_pp == alpha_pp and alpha_pp >= 3.0:
        print("- Alpha ↑ (≥ +3 pp): suggests relaxed attention; maintain **breath/OM** or light-focus meditations.")
    if theta_pp == theta_pp and theta_pp >= 3.0:
        print("- Theta ↑ (≥ +3 pp): suggests deeper internal focus; consider **mindfulness/visualization** practices.")
    if beta_pp == beta_pp and beta_pp <= -3.0:
        print("- Beta ↓ (≤ −3 pp): lower arousal; avoid strong stimulation; keep **calm breathing**.")
    if (alpha_pp == alpha_pp and alpha_pp < 1.0) and (theta_pp == theta_pp and theta_pp < 1.0):
        print("- Small alpha/theta change: try **guided breathing** first, then body-scan.")

if __name__ == "__main__":
    main()
