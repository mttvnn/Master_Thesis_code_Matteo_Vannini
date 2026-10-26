"""
bland_altman.py — Agreement between the sit-to-stand (SiSt) events detected by the algorithms and the
gold-standard events of the TVS (thesis Sections 4.5 and 5.4, Table 5.9, Bland-Altman figure).

Inputs (results/)
  gold_standard_events.xlsx   reference events  (PatientID, Test, Type, T_Start_s, T_End_s, Duration_s)
  events_baseline.xlsx        detected events with the baseline parameters   (PatientID, Test, Algorithm, Type, T_Start_s, T_End_s, Duration_s)
  events_optimized.xlsx       detected events with the LOSO-optimized parameters (same columns)
  Algorithm C (JAS) is evaluated with its baseline parameters, the configuration used in the unsupervised analysis;
  Algorithm A (Sit2StandPy) with its LOSO-optimized parameters (thesis Table 5.9, Figure 21).

Method
  - for every participant and test, each detected SiSt is matched to the unused reference SiSt with the highest
    intersection over union (IoU); the pair is kept when IoU >= IOU_THR (true positive), as in Section 4.5
  - for the matched pairs: differences detected - reference of onset, end and duration
  - Bland-Altman: mean bias and 95% limits of agreement (bias +/- 1.96 SD of the differences)
  - two-way intraclass correlation coefficient for absolute agreement, single measures, ICC(A,1) (McGraw & Wong)

Outputs (results/agreement/)
  bland_altman_pairs.csv      one row per matched pair (algorithm, participant, test, condition, onset/end/duration)
  bland_altman_stats.csv      n, bias, SD, limits of agreement and ICC(A,1) per algorithm and variable
  bland_altman_duration.png/.pdf   Bland-Altman plot of the duration, one panel per algorithm

Environment: conda activate jas
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================= PARAMETERS =============================
RESULTS = Path(__file__).resolve().parent.parent / "results"
GOLD_XLSX = RESULTS / "gold_standard_events.xlsx"
EVENTS_BASELINE_XLSX = RESULTS / "events_baseline.xlsx"
EVENTS_OPTIMIZED_XLSX = RESULTS / "events_optimized.xlsx"
OUT_DIR = RESULTS / "agreement"

IOU_THR = 0.5                                   # true positive threshold (Section 4.5)
FS_IOU = 100                                    # resolution of the time grid used for the IoU (Hz)
# label in the file -> (name in the thesis, which detections to use)
ALGORITHMS = {"JAS": ("Algorithm C", "baseline"), "Adamowicz": ("Algorithm A", "optimized")}
CONDITIONS = {"Test4": "Laboratory", "Test8": "Laboratory", "Test11": "Daily activities"}
VARIABLES = {"onset": "Onset (s)", "end": "End (s)", "duration": "Duration (s)"}

# thesis figure style
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
                     "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "axes.linewidth": 0.8})
CM = 1 / 2.54
COND_STYLE = {"Laboratory": dict(marker="o", color="#3E4A57"),
              "Daily activities": dict(marker="s", color="#8C5A5A")}


# ============================= MATCHING =============================
def iou(a, b, fs=FS_IOU):
    """Intersection over union of two intervals (s), computed on a time grid at fs Hz."""
    t0, t1 = min(a[0], b[0]), max(a[1], b[1])
    n = max(int(round((t1 - t0) * fs)), 1)
    t = np.linspace(t0, t1, n, endpoint=False)
    va = (t >= a[0]) & (t < a[1])
    vb = (t >= b[0]) & (t < b[1])
    union = np.sum(va | vb)
    return float(np.sum(va & vb) / union) if union else 0.0


def match_pairs(gold, det):
    """Greedy matching of detected and reference SiSt within each participant and test (IoU >= IOU_THR)."""
    rows = []
    for (pid, test), d in det.groupby(["PatientID", "Test"]):
        g = gold[(gold["PatientID"] == pid) & (gold["Test"] == test)].reset_index(drop=True)
        used = set()
        for r in d.sort_values("T_Start_s").itertuples():
            best, best_j = 0.0, None
            for j, ref in g.iterrows():
                if j in used:
                    continue
                v = iou((r.T_Start_s, r.T_End_s), (ref.T_Start_s, ref.T_End_s))
                if v > best:
                    best, best_j = v, j
            if best_j is not None and best >= IOU_THR:
                used.add(best_j)
                ref = g.loc[best_j]
                rows.append(dict(PatientID=pid, Test=test, condition=CONDITIONS.get(test, test), iou=round(best, 3),
                                 onset_det=r.T_Start_s, onset_ref=ref.T_Start_s,
                                 end_det=r.T_End_s, end_ref=ref.T_End_s,
                                 duration_det=r.T_End_s - r.T_Start_s, duration_ref=ref.T_End_s - ref.T_Start_s))
    return pd.DataFrame(rows)


# ============================= AGREEMENT =============================
def icc_a1(x, y):
    """Two-way random effects, absolute agreement, single measures: ICC(A,1) (McGraw & Wong, 1996)."""
    data = np.column_stack([x, y])
    n, k = data.shape
    grand = data.mean()
    ms_r = k * np.sum((data.mean(1) - grand) ** 2) / (n - 1)          # rows (transitions)
    ms_c = n * np.sum((data.mean(0) - grand) ** 2) / (k - 1)          # columns (methods)
    sse = np.sum((data - data.mean(1, keepdims=True) - data.mean(0, keepdims=True) + grand) ** 2)
    ms_e = sse / ((n - 1) * (k - 1))
    return (ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k / n * (ms_c - ms_e))


def agreement(pairs):
    rows = []
    for alg, p in pairs.groupby("algorithm"):
        for v in VARIABLES:
            diff = p[f"{v}_det"] - p[f"{v}_ref"]
            bias, sd = diff.mean(), diff.std(ddof=1)
            rows.append(dict(algorithm=alg, variable=v, n=len(p), bias=bias, sd=sd,
                             loa_low=bias - 1.96 * sd, loa_high=bias + 1.96 * sd,
                             icc_a1=icc_a1(p[f"{v}_det"].to_numpy(), p[f"{v}_ref"].to_numpy())
                             if v == "duration" else np.nan))
    return pd.DataFrame(rows)


# ============================= FIGURE =============================
def plot_duration(pairs, stats, fname="bland_altman_duration"):
    algs = [a for a, _ in ALGORITHMS.values() if a in set(pairs["algorithm"])]
    fig, axes = plt.subplots(1, len(algs), figsize=(16 * CM, 7 * CM), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    mean_all = (pairs["duration_det"] + pairs["duration_ref"]) / 2
    diff_all = pairs["duration_det"] - pairs["duration_ref"]
    for ax, alg, let in zip(axes, algs, "abcd"):
        p = pairs[pairs["algorithm"] == alg]
        s = stats[(stats["algorithm"] == alg) & (stats["variable"] == "duration")].iloc[0]
        ax.axhspan(s.loa_low, s.loa_high, color="0.93", lw=0, zorder=0)
        ax.axhline(0, color="0.75", lw=0.6)
        ax.axhline(s.bias, color="0.15", lw=1.0)
        for y in (s.loa_low, s.loa_high):
            ax.axhline(y, color="0.45", lw=0.8, ls="--")
        for cond, st in COND_STYLE.items():
            q = p[p["condition"] == cond]
            ax.scatter((q.duration_det + q.duration_ref) / 2, q.duration_det - q.duration_ref,
                       s=14, lw=0, alpha=0.9, zorder=3, **st)
        x_txt = mean_all.max()
        fmt = lambda v: f"{v:.2f}".replace("-", "\u2212")
        ax.text(x_txt, s.loa_high, f"+1.96 SD: {fmt(s.loa_high)} s", ha="right", va="bottom", fontsize=7)
        ax.text(x_txt, s.bias, f"Bias: {fmt(s.bias)} s", ha="right", va="bottom", fontsize=7)
        ax.text(x_txt, s.loa_low, f"\u22121.96 SD: {fmt(s.loa_low)} s", ha="right", va="top", fontsize=7)
        ax.set_title(f"({let}) {alg}  (n = {len(p)})", loc="left", fontsize=9, fontweight="bold")
        ax.set_xlabel("Mean of detected and reference duration (s)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Detected \u2212 reference duration (s)")
    pad = 0.15 * (diff_all.max() - diff_all.min())
    axes[0].set_ylim(min(diff_all.min(), stats["loa_low"].min()) - pad, max(diff_all.max(), stats["loa_high"].max()) + pad)
    handles = [Line2D([], [], ls="", markersize=5, label=f"{c} ({'Test4, Test8' if c == 'Laboratory' else 'Test11'})", **st)
               for c, st in COND_STYLE.items()]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.subplots_adjust(wspace=0.08)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{fname}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


# ============================= MAIN =============================
def main(gold_xlsx=GOLD_XLSX, baseline_xlsx=EVENTS_BASELINE_XLSX, optimized_xlsx=EVENTS_OPTIMIZED_XLSX):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gold = pd.read_excel(gold_xlsx).iloc[:, :6]
    gold = gold[gold["Type"] == "STS"].dropna(subset=["T_Start_s", "T_End_s"])
    gold["PatientID"] = gold["PatientID"].astype(str)
    events = {"baseline": pd.read_excel(baseline_xlsx), "optimized": pd.read_excel(optimized_xlsx)}
    for df in events.values():
        df["PatientID"] = df["PatientID"].astype(str)

    pairs = []
    for label, (name, config) in ALGORITHMS.items():
        det = events[config]
        d = det[(det["Type"] == "STS") & (det["Algorithm"] == label)]
        m = match_pairs(gold, d)
        if not m.empty:
            pairs.append(m.assign(algorithm=name, configuration=config))
    pairs = pd.concat(pairs, ignore_index=True)
    pairs.round(3).to_csv(OUT_DIR / "bland_altman_pairs.csv", index=False)

    stats = agreement(pairs)
    stats.round(3).to_csv(OUT_DIR / "bland_altman_stats.csv", index=False)
    print(stats.round(2).to_string(index=False))

    plot_duration(pairs, stats)
    print(f"-> results saved in {OUT_DIR}")


if __name__ == "__main__":
    main()
