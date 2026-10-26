"""
analysis_sensitivity_reliability.py — Sensitivity analysis, day-to-day reliability, hourly profile and
check of the low-activity participants, from the final results of the pipeline.

Inputs (ROOT_OUT, produced by run_jas_per_ID.py / filter_weartime.py / aggregate_daily.py):
  daily_aggregates.csv, summary_all_IDs.csv, weartime_P3_P4.csv, <fase>/<ID>/<ID>_<fase>_events.csv

Rules
  - n_sts: computed on ALL valid days (zeros included)
  - kinematic features (p10 / p50 / p90 / CoV of duration, velocities, acceleration, smoothness, ...):
    only valid days with >= MIN_STS_FEATURES SiSt (with fewer transitions percentiles and CoV are unreliable)
    -> since the daily statistics are computed on the KEPT SiSt, this is the same as n_sts >= MIN_STS_FEATURES

Outputs (ROOT_OUT/analysis/):
  1. weekly_main.csv                     weekly values per subject with the rule above (main analysis)
  2. comparison_P3_P4_scenarios.csv      P3 vs P4 for every weekly metric, 3 scenarios:
       main         rule above
       S1_days      days with < MIN_STS_FEATURES SiSt excluded entirely (also from n_sts)
       S2_subjects  participants with mean < MIN_STS_FEATURES SiSt/day excluded
     -> median (p25–p75) per phase, Mann-Whitney U p-value, n per phase
  3. icc_day_to_day.csv                  day-to-day reliability per metric (all, P3, P4):
       ICC(1,1) one-way random (single day), ICC(1,k) (mean of the subject's valid days),
       days needed for ICC = ICC_TARGET (Spearman-Brown)
  4. hourly_profile.csv / hourly_profile_P3_P4.png   SiSt per hour of the day (KEPT, valid days),
       mean per subject-day, then mean and 95% CI over subjects, P3 vs P4
  5. low_activity_participants.csv       participants with mean < MIN_STS_FEATURES SiSt/day: per day n_sts,
       waking wear time, JAS detections, rejections (walking bout / filter), orientation correction
     (the visual check of the signals is then done with inspect_signal.py / inversion_timeline.py)

Not possible with the current outputs: SiSt followed by walking (STS-W) vs isolated (STS-S) — the gait
sequences are computed in run_jas_per_ID.py only for the WB filter and not saved (needs a small change there).

Environment: conda activate jas ; run from ~/tesi after aggregate_daily.py
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================= PARAMETERS =============================
VISITS = ["T1"]                    # must match run_jas_per_ID.py
FILTER_CFG = "default"             # must match run_jas_per_ID.py
ORIENT_FIX = False                 # True only if run_jas_per_ID.py writes to the _orientfix folder
ROOT_OUT = (Path.home() / "tesi" / "risultati" /
            f"JAS_per_ID_{'_'.join(VISITS)}_{FILTER_CFG}{'_orientfix' if ORIENT_FIX else ''}")
OUT_DIR = ROOT_OUT / "analysis"

MIN_STS_FEATURES = 5               # threshold for the kinematic features and for the sensitivity analysis
ICC_TARGET = 0.8                   # target reliability for the number of days needed
COLORS = {"P3": "#6F93BF", "P4": "#D9A066"}
EDGES = {"P3": "#4A6F9C", "P4": "#B07A3E"}
DPI = 200


# ============================= DATA =============================
def load_daily():
    d = pd.read_csv(ROOT_OUT / "daily_aggregates.csv", dtype={"ID": str})
    d = d[d["valid_day"] == 1].copy()
    feats = [c for c in d.columns if c.rsplit("_", 1)[-1] in ("p10", "p50", "p90", "cov")]
    return d, feats


def apply_feature_rule(d, feats):
    """Kinematic features only on days with >= MIN_STS_FEATURES SiSt; n_sts untouched."""
    d = d.copy()
    d.loc[d["n_sts"] < MIN_STS_FEATURES, feats] = np.nan
    return d


def weekly(d, feats):
    """Mean over the valid days per subject (NaN ignored)."""
    g = d.groupby(["ID", "phase"])
    w = g[["n_sts"] + feats].mean()
    w.insert(0, "n_days_valid", g.size())
    w.insert(1, "n_days_features", g[feats[0]].count() if feats else 0)
    return w.reset_index()


# ============================= 2. P3 vs P4 =============================
def compare(w, metrics, scenario):
    rows = []
    for m in metrics:
        a = w.loc[w["phase"] == "P3", m].dropna()
        b = w.loc[w["phase"] == "P4", m].dropna()
        p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue if len(a) and len(b) else np.nan
        f = lambda x: f"{x.median():.2f} ({x.quantile(.25):.2f}–{x.quantile(.75):.2f})" if len(x) else ""
        rows.append(dict(scenario=scenario, metric=m, P3=f(a), P4=f(b), n_P3=len(a), n_P4=len(b),
                         p_value=p))
    return rows


# ============================= 3. ICC =============================
def icc_oneway(df, col):
    """One-way random ICC for unbalanced data (subjects with >= 2 days with a value)."""
    x = df[["ID", col]].dropna()
    x = x[x.groupby("ID")[col].transform("size") >= 2]
    if x["ID"].nunique() < 3:
        return dict(n_subjects=x["ID"].nunique(), mean_days=np.nan, icc_single=np.nan,
                    icc_week=np.nan, days_needed=np.nan)
    g = x.groupby("ID")[col]
    n_i, m_i = g.size(), g.mean()
    N, k, M = len(x), len(n_i), x[col].mean()
    ssb = (n_i * (m_i - M) ** 2).sum()
    ssw = ((x[col] - x["ID"].map(m_i)) ** 2).sum()
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    icc1 = (msb - msw) / (msb + (n0 - 1) * msw)
    icck = (msb - msw) / msb if msb > 0 else np.nan
    need = ICC_TARGET * (1 - icc1) / (icc1 * (1 - ICC_TARGET)) if icc1 > 0 else np.nan
    return dict(n_subjects=k, mean_days=round(n_i.mean(), 1), icc_single=icc1, icc_week=icck,
                days_needed=need)


# ============================= 4. HOURLY PROFILE =============================
def hourly_profile(d_valid):
    rows = []
    for (pid, phase), g in d_valid.groupby(["ID", "phase"]):
        f = ROOT_OUT / phase / pid / f"{pid}_{phase}_events.csv"
        if not f.exists():
            continue
        try:
            ev = pd.read_csv(f, dtype={"ID": str})
        except pd.errors.EmptyDataError:
            ev = pd.DataFrame()
        days = g["day"].astype(int).tolist()
        counts = np.zeros(24)
        if not ev.empty and "clock_start" in ev.columns:
            ev = ev[(ev["status"] == "KEPT") & (ev["day"].isin(days))]
            hrs = pd.to_numeric(ev["clock_start"].astype(str).str[11:13], errors="coerce").dropna().astype(int)
            counts = np.bincount(hrs, minlength=24)[:24].astype(float)
        rows.append(dict(ID=pid, phase=phase, n_days=len(days), **{f"h{h:02d}": counts[h] / len(days)
                                                                  for h in range(24)}))
    per_subj = pd.DataFrame(rows)
    hcols = [f"h{h:02d}" for h in range(24)]
    summ = []
    for ph in ["P3", "P4"]:
        x = per_subj.loc[per_subj["phase"] == ph, hcols]
        n = len(x)
        m, se = x.mean(), x.std(ddof=1) / np.sqrt(max(n, 1))
        for h, c in enumerate(hcols):
            summ.append(dict(phase=ph, hour=h, n_subjects=n, mean=m[c], ci_low=m[c] - 1.96 * se[c],
                             ci_high=m[c] + 1.96 * se[c]))
    summ = pd.DataFrame(summ)

    fig, ax = plt.subplots(figsize=(10, 5.4))
    for ph in ["P3", "P4"]:
        s = summ[summ["phase"] == ph]
        ax.fill_between(s["hour"] + 0.5, s["ci_low"].clip(lower=0), s["ci_high"], color=COLORS[ph], alpha=0.25, lw=0)
        ax.plot(s["hour"] + 0.5, s["mean"], color=EDGES[ph], lw=2, marker="o", ms=3.5,
                label=f"{ph} (n = {int(s['n_subjects'].iloc[0])})")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlabel("Hour of the day", fontsize=11)
    ax.set_ylabel("SiSt per hour (mean per day)", fontsize=11)
    ax.set_title("Hourly profile of SiSt", fontsize=13, color="#333333", pad=10)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, axis="y", color="#E6E6E6", lw=0.7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(0.0, -0.16, "Line = mean over subjects, band = 95% CI; KEPT SiSt on valid days",
            transform=ax.transAxes, fontsize=8, color="#777777")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "hourly_profile_P3_P4.png", dpi=DPI)
    plt.close(fig)
    per_subj.to_csv(OUT_DIR / "hourly_profile_per_subject.csv", index=False)
    summ.round(3).to_csv(OUT_DIR / "hourly_profile.csv", index=False)
    return summ


# ============================= 5. LOW-ACTIVITY CHECK =============================
def local_date(x):
    try:
        return pd.Timestamp(datetime.fromisoformat(str(x).strip()).date())
    except (ValueError, TypeError):
        return pd.NaT


def low_activity(d_all, w_main):
    ids = w_main.loc[w_main["n_sts"] < MIN_STS_FEATURES, "ID"].tolist()
    if not ids:
        return pd.DataFrame()
    summ = pd.read_csv(ROOT_OUT / "summary_all_IDs.csv", dtype={"ID": str})
    summ = summ[summ["ID"].isin(ids)]
    summ["date"] = summ["recording_start"].map(local_date)
    agg = {"date": ("date", "min"), "rec_h": ("duration_h", "sum"), "jas": ("sts_jas", "sum"),
           "rej_wb": ("rejected_wb", "sum"), "rej_filter": ("rejected_filter", "sum")}
    if "flipped_pct" in summ.columns:
        agg["flipped_pct"] = ("flipped_pct", "max")
    s = summ.groupby(["ID", "day"]).agg(**agg).reset_index()

    wt_f = ROOT_OUT / "weartime_P3_P4.csv"
    if wt_f.exists():
        wt = pd.read_csv(wt_f, dtype={"ID": str})
        wt["date"] = pd.to_datetime(wt["measurement_date"], errors="coerce").dt.normalize()
        s = s.merge(wt[["ID", "date", "total_worn_during_waking_h"]], on=["ID", "date"], how="left")

    out = (d_all[d_all["ID"].isin(ids)][["ID", "phase", "day", "valid_day", "n_sts"]]
           .merge(s, on=["ID", "day"], how="left"))
    out = out.merge(w_main[["ID", "n_sts"]].rename(columns={"n_sts": "mean_sts_per_day"}), on="ID")
    out = out.sort_values(["mean_sts_per_day", "ID", "day"]).drop(columns="date")
    out.round(2).to_csv(OUT_DIR / "low_activity_participants.csv", index=False)
    return out


# ============================= MAIN =============================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d_valid, feats = load_daily()
    d_all = pd.read_csv(ROOT_OUT / "daily_aggregates.csv", dtype={"ID": str})
    metrics = ["n_sts"] + feats

    # 1. main weekly values
    d_main = apply_feature_rule(d_valid, feats)
    w_main = weekly(d_main, feats)
    w_main.round(4).to_csv(OUT_DIR / "weekly_main.csv", index=False)
    low_days = (d_valid["n_sts"] < MIN_STS_FEATURES)
    print(f"Valid days: {len(d_valid)} | days with < {MIN_STS_FEATURES} SiSt: {int(low_days.sum())} "
          f"({low_days.mean() * 100:.1f}%) in {d_valid.loc[low_days, 'ID'].nunique()} participants | "
          f"days with 0 SiSt: {int((d_valid['n_sts'] == 0).sum())}")

    # 2. scenarios
    w_s1 = weekly(d_valid[d_valid["n_sts"] >= MIN_STS_FEATURES], feats)
    w_s2 = w_main[w_main["n_sts"] >= MIN_STS_FEATURES]
    rows = compare(w_main, metrics, "main") + compare(w_s1, metrics, "S1_days") + compare(w_s2, metrics, "S2_subjects")
    comp = pd.DataFrame(rows)
    comp["p_value"] = comp["p_value"].map(lambda p: "" if pd.isna(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}"))
    comp.to_csv(OUT_DIR / "comparison_P3_P4_scenarios.csv", index=False, encoding="utf-8-sig")
    print(f"\nParticipants: main {len(w_main)} | S1 {len(w_s1)} | S2 {len(w_s2)} "
          f"(excluded with mean < {MIN_STS_FEATURES} SiSt/day: {len(w_main) - len(w_s2)})")
    key = [m for m in metrics if m == "n_sts" or m.endswith("_p50")]
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(comp[comp["metric"].isin(key)].to_string(index=False))

    # 3. reliability
    icc_rows = []
    for m in metrics:
        for scope, sub in [("All", d_main), ("P3", d_main[d_main["phase"] == "P3"]),
                           ("P4", d_main[d_main["phase"] == "P4"])]:
            icc_rows.append(dict(metric=m, scope=scope, **icc_oneway(sub, m)))
    icc = pd.DataFrame(icc_rows).round(3)
    icc.to_csv(OUT_DIR / "icc_day_to_day.csv", index=False)
    print("\nDay-to-day reliability (All)")
    print(icc[(icc["scope"] == "All") & icc["metric"].isin(key)].to_string(index=False))

    # 4. hourly profile
    hourly_profile(d_valid)

    # 5. low-activity participants
    la = low_activity(d_all, w_main)
    if not la.empty:
        print(f"\nLow-activity participants (mean < {MIN_STS_FEATURES} SiSt/day): "
              f"{la['ID'].nunique()} -> see low_activity_participants.csv")
        print(la.groupby(["ID", "phase"])["mean_sts_per_day"].first().round(2).to_string())
    print(f"\n-> results saved in {OUT_DIR}")


if __name__ == "__main__":
    main()