"""
plot_distributions.py — Histograms P3 vs P4 of the STS metrics + tables by day + weekday pattern.
Thesis-style version: sober colours, serif font, no titles inside the figures.

Thesis figures (ROOT_OUT/figures/):
  Figure_RW_hist_thesis.png/.pdf      (a) STS per day (b) duration (c) max flexion velocity
                                      (d) max extension velocity (e) max vertical acceleration (f) SPARC
  Figure_RW_weekday_thesis.png/.pdf   (a) STS per day and (b) daily median duration by day of the week
  only these, from the terminal:  python -c "import plot_distributions as P; P.thesis_figures()"
Single figures (SAVE_SINGLE): hist_<metric>_P3_P4.png ; weekday_<metric>_P3_P4.png
Tables: table_by_day_*, table_p10_p90_by_day_*, table_cov_by_day_*, table_by_weekday_P3_P4.csv
Data:
  Number of STS       one value per valid day (n_sts of daily_aggregates.csv)
  all other metrics   one value per KEPT STS on the valid days (events CSVs)
Environment: conda activate jas ; run from ~/tesi after aggregate_daily.py
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

# ============================= THESIS STYLE =============================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.linewidth": 0.8,
})
CM = 1 / 2.54
THESIS_DPI = 600

# ============================= PARAMETERS =============================
VISITS = ["T1"]                    # must match run_jas_per_ID.py
FILTER_CFG = "default"             # must match run_jas_per_ID.py
ORIENT_FIX = False                 # True only if run_jas_per_ID.py writes to the _orientfix folder
ROOT_OUT = (Path.home() / "tesi" / "risultati" /
            f"JAS_per_ID_{'_'.join(VISITS)}_{FILTER_CFG}{'_orientfix' if ORIENT_FIX else ''}")
FIG_DIR = ROOT_OUT / "figures"

VALID_ONLY = True                  # only days with valid_day == 1
MIN_STS_PER_DAY = None             # e.g. 5 or 10: exclude days with fewer STS (None = keep all days)
SAVE_SINGLE = True                 # one figure per histogram: hist_<metric>_P3_P4.png
SAVE_COMBINED = False              # True = also the old multi-panel figures
N_BINS_AUTO = 25                   # number of bins when no bin width is given
MIN_EVENTS_STATS = 3               # CoV table: min STS in a subject-day (same as aggregate_daily.py)

# key: (source column, title (single figures only), x label, bin width or None = auto, decimals, show CoV)
PANELS = {
    "duration":  ("duration", "Duration of SiSt", "SiSt duration (s)", 0.25, 2, True),
    "n_sists":   ("n_sts", "Number of Sist", "SiSt per day", 5, 0, True),
    "flexion":   ("sts_max_flexion_vel", "Max flexion velocity", "Max flexion velocity (deg/s)", None, 1, False),
    "extension": ("sts_max_extension_vel", "Max extension velocity", "Max extension velocity (deg/s)", None, 1, False),
    "acc_v":     ("sts_max_accv", "Max vertical acceleration", "Max vertical acceleration (m/s$^2$)", None, 2, False),
    "sparc":     ("sts_sparc_gyroml", "SPARC (ML gyro)", "SPARC (\u2013)", None, 2, False),
    "ldlj":      ("sts_ldlj_gyroml", "LDLJ (ML gyro)", "LDLJ (\u2013)", None, 2, False),
    "tilt":      ("sts_tilt_angle", "Trunk tilt", "Tilt angle (deg)", None, 0, False),
}
# panels of the thesis figure, in order
THESIS_PANELS = ["n_sists", "duration", "flexion", "extension", "acc_v", "sparc"]

# x-range per panel: (x_min or None = data minimum, upper quantile or None = data maximum).
X_RANGE = {
    "n_sists": (0, None),
    "sparc":   (0.01, None),
    "ldlj":    (0.01, None),
    "tilt":    (0.01, 0.99),
}
FLEX_BINS = N_BINS_AUTO
X_MARGIN = False

EVENT_COLS = ["duration", "sts_max_flexion_vel", "sts_max_extension_vel", "sts_max_accv",
              "sts_sparc_gyroml", "sts_ldlj_gyroml", "sts_tilt_angle"]

# sober palette, same as the other thesis figures
COLORS = {"P3": "#3E4A57", "P4": "#8C5A5A"}   # fill (slate / muted burgundy)
EDGES  = {"P3": "#3E4A57", "P4": "#8C5A5A"}   # outline
FILL_ALPHA = 0.22                             # light fill inside the bars
LABELS = {"P3": "Extended recovery (P3)", "P4": "Long-term recovery (P4)"}
DPI = 200


# ============================= DATA =============================
def load_days():
    d = pd.read_csv(ROOT_OUT / "daily_aggregates.csv", dtype={"ID": str})
    if VALID_ONLY:
        d = d[d["valid_day"] == 1]
    if MIN_STS_PER_DAY is not None:
        low = d["n_sts"] < MIN_STS_PER_DAY
        tab = d.assign(low=low).groupby("phase")["low"].agg(["sum", "size"])
        for ph, r in tab.iterrows():
            print(f"[i] {ph}: {int(r['sum'])} of {int(r['size'])} days excluded "
                  f"(< {MIN_STS_PER_DAY} STS, {r['sum'] / r['size'] * 100:.1f}%)")
        d = d[~low]
    return d[["ID", "phase", "day", "n_sts"]]


def load_events(days):
    """All KEPT STS on the selected days, with the event-level metrics."""
    out = []
    for (pid, phase), g in days.groupby(["ID", "phase"]):
        f = ROOT_OUT / phase / pid / f"{pid}_{phase}_events.csv"
        if not f.exists():
            continue
        try:
            ev = pd.read_csv(f, dtype={"ID": str})
        except pd.errors.EmptyDataError:
            continue
        if ev.empty:
            continue
        ev = ev[(ev["status"] == "KEPT") & (ev["day"].isin(g["day"]))]
        cols = [c for c in EVENT_COLS if c in ev.columns]
        out.append(ev[cols].astype(float).assign(phase=phase, ID=pid, day=ev["day"].astype(int).to_numpy()))
    return (pd.concat(out, ignore_index=True) if out
            else pd.DataFrame(columns=EVENT_COLS + ["phase", "ID", "day"]))


def get_sources():
    days = load_days()
    ev = load_events(days)
    return {"n_sists": days, "duration": ev, "flexion": ev, "extension": ev, "acc_v": ev,
            "sparc": ev, "ldlj": ev, "tilt": ev}


# ============================= PLOT =============================
def style(ax, thesis=False):
    ax.grid(True, axis="y", color="#EBEBEB", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#555555" if thesis else "#9A9A9A")
    ax.tick_params(colors="#333333", labelsize=8 if thesis else 10)


def draw_hist(ax, data, key, thesis=False):
    col, title, xlabel, bin_w, dec, show_cov = PANELS[key]
    x_all = data[col].dropna().to_numpy()
    if len(x_all) == 0:
        ax.set_title(f"{title} (no data)")
        return

    q_lo, q_hi = X_RANGE.get(key, (None, None))
    start = np.quantile(x_all, q_lo) if isinstance(q_lo, float) and 0 < q_lo < 1 else (
        q_lo if q_lo is not None else x_all.min())
    top = np.quantile(x_all, q_hi) if isinstance(q_hi, float) and 0 < q_hi < 1 else (
        q_hi if q_hi is not None else x_all.max())
    grouped_left = isinstance(q_lo, float) and 0 < q_lo < 1
    grouped_right = isinstance(q_hi, float) and 0 < q_hi < 1

    if bin_w is None:
        nb = FLEX_BINS if key == "flexion" else N_BINS_AUTO
        bin_w = (top - start) / nb or 1.0
        lo = start
    else:
        lo = np.floor(start / bin_w) * bin_w
    hi = top + bin_w
    bins = np.arange(lo, hi + bin_w / 2, bin_w)
    bw = bin_w / 2

    handles = []
    for k, ph in enumerate(["P3", "P4"]):
        x = data.loc[data["phase"] == ph, col].dropna().to_numpy()
        if len(x) == 0:
            continue
        x_clipped = np.clip(x, bins[0], bins[-1] - 1e-9)       # outliers grouped in the extreme bins
        h, _ = np.histogram(x_clipped, bins=bins)
        y = h / len(x) * 100
        centers = bins[:-1] + bw / 2 + k * bw
        ax.bar(centers, y, width=bw, color=COLORS[ph], alpha=FILL_ALPHA, lw=0)
        ax.bar(centers, y, width=bw, fill=False, edgecolor=EDGES[ph], lw=0.6 if thesis else 0.9)
        handles.append(Patch(facecolor=COLORS[ph], alpha=FILL_ALPHA, edgecolor=EDGES[ph], lw=0.8,
                             label=LABELS[ph] if thesis else ph))

    ax.set_xlim(bins[0], bins[-1] + (bin_w if X_MARGIN and not grouped_right else 0))
    if key in ("sparc", "ldlj", "acc_v"):
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    else:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6, steps=[1, 2, 5, 10]))

    if thesis:
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Proportion (%)")
    else:
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Proportion (%)", fontsize=11)
        ax.set_title(title, fontsize=13, color="#333333", pad=10)
        ax.legend(handles=handles, frameon=False, fontsize=10, loc="upper right")

    notes = []
    if grouped_left:
        notes.append(f"first bin includes values < {bins[1]:.2f}")
    if grouped_right:
        notes.append(f"last bin includes values > {bins[-2]:.2f}")
    if notes and not thesis:
        ax.text(0.0, -0.15, " & ".join(notes), transform=ax.transAxes,
                ha="left", va="top", fontsize=9, color="#666666", style="italic")
    if MIN_STS_PER_DAY is not None and not thesis:
        ax.text(1.0, -0.15, f"days with \u2265 {MIN_STS_PER_DAY} STS", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="#666666", style="italic")
    style(ax, thesis)
    return handles


def figure(panels, sources, fname, width_per_panel=7.5):
    if SAVE_COMBINED:
        fig, axes = plt.subplots(1, len(panels), figsize=(width_per_panel * len(panels), 6.2))
        for ax, key in zip(np.atleast_1d(axes), panels):
            draw_hist(ax, sources[key], key)
        plt.tight_layout(w_pad=4)
        fig.savefig(FIG_DIR / fname, dpi=DPI)
        plt.close(fig)
    if SAVE_SINGLE:
        for key in panels:
            f, ax = plt.subplots(figsize=(7.5, 6))
            draw_hist(ax, sources[key], key)
            plt.tight_layout()
            f.savefig(FIG_DIR / f"hist_{key}_P3_P4.png", dpi=DPI)
            plt.close(f)


# ============================= THESIS FIGURES =============================
def thesis_hist_figure(sources, fname="Figure_RW_hist_thesis"):
    """2 x 3 histograms, P3 vs P4, thesis style (16 cm wide)."""
    fig, axes = plt.subplots(2, 3, figsize=(16 * CM, 10.5 * CM))
    handles = None
    for ax, key, let in zip(axes.ravel(), THESIS_PANELS, "abcdef"):
        h = draw_hist(ax, sources[key], key, thesis=True)
        handles = handles or h
        ax.text(0.0, 1.04, f"({let})", transform=ax.transAxes, fontsize=9, fontweight="bold")
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(hspace=0.55, wspace=0.38, top=0.88, bottom=0.1, left=0.08, right=0.98)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{fname}.{ext}", dpi=THESIS_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {fname}.png/.pdf saved in {FIG_DIR}")


def thesis_weekday_figure(fname="Figure_RW_weekday_thesis"):
    """STS per day and daily median duration by day of the week, P3 vs P4, thesis style."""
    d = pd.read_csv(ROOT_OUT / "daily_aggregates.csv", dtype={"ID": str})
    if VALID_ONLY:
        d = d[d["valid_day"] == 1]
    if MIN_STS_PER_DAY is not None:
        d = d[d["n_sts"] >= MIN_STS_PER_DAY]
    d = add_weekday(d)
    fig, axes = plt.subplots(1, 2, figsize=(16 * CM, 6.5 * CM))
    for ax, (col, ylab), let in zip(axes, [("n_sts", "SiSt per day"), ("duration_p50", "Daily median duration (s)")], "ab"):
        draw_weekday(ax, d, col, ylab, thesis=True)
        ax.text(0.0, 1.04, f"({let})", transform=ax.transAxes, fontsize=9, fontweight="bold")
    handles = [Patch(facecolor=COLORS[ph], alpha=FILL_ALPHA, edgecolor=EDGES[ph], label=LABELS[ph]) for ph in ("P3", "P4")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.subplots_adjust(wspace=0.3, top=0.84, bottom=0.15, left=0.08, right=0.99)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{fname}.{ext}", dpi=THESIS_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {fname}.png/.pdf saved in {FIG_DIR}")


def thesis_figures():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    thesis_hist_figure(get_sources())
    thesis_weekday_figure()


# ============================= TABLE BY DAY =============================
def table_by_day(sources):
    """Mean and SD per day number (and over all days) for P3 and P4, every metric."""
    num, wide = [], {}
    for key, (col, title, _xl, _bw, dec, _cov) in PANELS.items():
        d = sources[key]
        if col not in d.columns:
            continue
        groups = [(f"Day {int(day)}", g) for day, g in d.groupby("day")] + [("All days", d)]
        for label, g in groups:
            for ph in ["P3", "P4"]:
                x = g.loc[g["phase"] == ph, col].dropna()
                m, sd = (x.mean(), x.std(ddof=1)) if len(x) else (np.nan, np.nan)
                num.append(dict(metric=title, day=label, phase=ph, n=len(x), mean=m, sd=sd))
                wide.setdefault(label, {})[f"{title} — {ph}"] = \
                    f"{m:.{dec}f} ± {sd:.{dec}f}" if len(x) > 1 else ""
    num = pd.DataFrame(num)
    wide = pd.DataFrame.from_dict(wide, orient="index")
    wide.index.name = "Day"
    num.round(4).to_csv(FIG_DIR / "table_by_day_P3_P4_numeric.csv", index=False)
    wide.to_csv(FIG_DIR / "table_by_day_P3_P4.csv", encoding="utf-8-sig")
    return wide


def table_percentiles_by_day(sources, lo=10, hi=90):
    """p10 and p90 per day number (and over all days) for P3 and P4, every metric."""
    num, wide = [], {}
    for key, (col, title, _xl, _bw, dec, _cov) in PANELS.items():
        d = sources[key]
        if col not in d.columns:
            continue
        groups = [(f"Day {int(day)}", g) for day, g in d.groupby("day")] + [("All days", d)]
        for label, g in groups:
            for ph in ["P3", "P4"]:
                x = g.loc[g["phase"] == ph, col].dropna().to_numpy()
                a, b = (np.percentile(x, lo), np.percentile(x, hi)) if len(x) else (np.nan, np.nan)
                num.append({"metric": title, "day": label, "phase": ph, "n": len(x), f"p{lo}": a, f"p{hi}": b})
                wide.setdefault(label, {})[f"{title} — {ph}"] = \
                    f"{a:.{dec}f} – {b:.{dec}f}" if len(x) else ""
    num = pd.DataFrame(num)
    wide = pd.DataFrame.from_dict(wide, orient="index")
    wide.index.name = f"Day (p{lo} – p{hi})"
    num.round(4).to_csv(FIG_DIR / f"table_p{lo}_p{hi}_by_day_P3_P4_numeric.csv", index=False)
    wide.to_csv(FIG_DIR / f"table_p{lo}_p{hi}_by_day_P3_P4.csv", encoding="utf-8-sig")
    return wide


def table_cov_by_day(sources):
    """CoV (%) as in daily/weekly_aggregates: within-subject CoV of each subject-day
    (>= MIN_EVENTS_STATS STS), averaged over subjects; All days = mean of the subjects' weekly CoV."""
    ev = sources["duration"]
    num, wide = [], {}
    for key, (col, title, *_rest) in PANELS.items():
        if key == "n_sists" or col not in ev.columns:
            continue
        g = ev.dropna(subset=[col]).groupby(["phase", "ID", "day"])[col]
        sd = g.std(ddof=1)
        m = g.mean().abs()
        n = g.size()
        daily = (sd / m * 100).where((n >= MIN_EVENTS_STATS) & (m > 0)).dropna().rename("cov").reset_index()
        per_day = daily.groupby(["phase", "day"])["cov"].agg(["mean", "size"]).reset_index()
        weekly = daily.groupby(["phase", "ID"])["cov"].mean().groupby("phase").agg(["mean", "size"])
        rows = [(f"Day {int(r.day)}", r.phase, r["mean"], int(r["size"])) for _, r in per_day.iterrows()]
        rows += [("All days", ph, r["mean"], int(r["size"])) for ph, r in weekly.iterrows()]
        for label, ph, c, k in rows:
            num.append(dict(metric=title, day=label, phase=ph, n_subjects=k, cov_pct=c))
            wide.setdefault(label, {})[f"{title} — {ph}"] = f"{c:.1f}%"
    num = pd.DataFrame(num)
    order = [f"Day {d}" for d in sorted({int(x.split()[1]) for x in wide if x.startswith("Day ")})] + ["All days"]
    wide = pd.DataFrame.from_dict(wide, orient="index").reindex(order)
    cols = [f"{PANELS[k][1]} — {ph}" for k in PANELS if k != "n_sists" for ph in ("P3", "P4")]
    wide = wide[[c for c in cols if c in wide.columns]]
    wide.index.name = "Day (CoV %, mean of subjects)"
    num.round(2).to_csv(FIG_DIR / "table_cov_by_day_P3_P4_numeric.csv", index=False)
    wide.to_csv(FIG_DIR / "table_cov_by_day_P3_P4.csv", encoding="utf-8-sig")
    return wide


def tables_only():
    """Only the tables (mean ± SD, p10 – p90 and CoV by day), no figures."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sources = get_sources()
    w1 = table_by_day(sources)
    w2 = table_percentiles_by_day(sources)
    w3 = table_cov_by_day(sources)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print("MEAN ± SD\n", w1.to_string(), "\n\nP10 – P90\n", w2.to_string(),
              "\n\nCoV (%)\n", w3.to_string())
    print(f"\n-> tables saved in {FIG_DIR}")


# ============================= WEEKDAY PATTERN =============================
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
WEEKDAY_METRICS = {
    "n_sts": ("Number of STS by day of the week", "STS per day"),
    "duration_p50": ("Duration of STS by day of the week", "Daily median STS duration (s)"),
}


def local_date(x):
    """Calendar date in the recording's own local time (mixed timezone offsets are fine)."""
    try:
        return pd.Timestamp(datetime.fromisoformat(str(x).strip()).date())
    except (ValueError, TypeError):
        return pd.NaT


def add_weekday(days_full):
    """Weekday of every ID x day from the first recording_start of the day (summary_all_IDs.csv)."""
    summ = pd.read_csv(ROOT_OUT / "summary_all_IDs.csv", dtype={"ID": str})
    summ["date"] = summ["recording_start"].map(local_date)
    dd = summ.groupby(["ID", "day"])["date"].min().reset_index()
    out = days_full.merge(dd, on=["ID", "day"], how="left")
    n_miss = int(out["date"].isna().sum())
    if n_miss:
        print(f"[!] {n_miss} days without recording_start: excluded from the weekday plots")
    out = out.dropna(subset=["date"])
    out["weekday"] = out["date"].dt.dayofweek          # 0 = Monday
    return out


def draw_weekday(ax, d, col, ylabel, thesis=False):
    width = 0.36
    for k, ph in enumerate(["P3", "P4"]):
        data = [d.loc[(d["phase"] == ph) & (d["weekday"] == w), col].dropna().to_numpy() for w in range(7)]
        pos = np.arange(7) + (k - 0.5) * width
        lw = 0.8 if thesis else 1.1
        bp = ax.boxplot(data, positions=pos, widths=width, patch_artist=True, showfliers=False,
                        medianprops=dict(color=EDGES[ph], lw=1.4 if thesis else 1.8),
                        whiskerprops=dict(color=EDGES[ph], lw=lw), capprops=dict(color=EDGES[ph], lw=lw),
                        boxprops=dict(edgecolor=EDGES[ph], lw=lw))
        for b in bp["boxes"]:
            b.set_facecolor(COLORS[ph])
            b.set_alpha(FILL_ALPHA)
        med = [np.median(x) if len(x) else np.nan for x in data]
        ax.plot(pos, med, color=EDGES[ph], lw=1.0 if thesis else 1.6, marker="o",
                ms=2.5 if thesis else 4, zorder=3)
    ax.set_xticks(range(7))
    ax.set_xticklabels(WEEKDAYS, fontsize=8 if thesis else 9)
    ax.set_ylabel(ylabel, fontsize=9 if thesis else 11)
    style(ax, thesis)


def weekday_plot(d, col, title, ylabel, fname):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    draw_weekday(ax, d, col, ylabel)
    labels = []
    for w in range(7):
        n3 = int(((d["phase"] == "P3") & (d["weekday"] == w) & d[col].notna()).sum())
        n4 = int(((d["phase"] == "P4") & (d["weekday"] == w) & d[col].notna()).sum())
        labels.append(f"{WEEKDAYS[w]}\nn = {n3} | {n4}")
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=13, color="#333333", pad=10)
    handles = [Patch(facecolor=COLORS[ph], alpha=FILL_ALPHA, edgecolor=EDGES[ph], label=ph) for ph in ("P3", "P4")]
    ax.legend(handles=handles, frameon=False, fontsize=10, loc="upper right", ncol=2)
    ax.text(0.0, -0.2, "Box = IQR, whiskers = 1.5 IQR (outliers not shown), line = weekly trend of the medians; "
            "n = valid days P3 | P4", transform=ax.transAxes, fontsize=8, color="#777777")
    plt.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=DPI)
    plt.close(fig)


def weekday_plots():
    """Weekly pattern (Monday-Sunday) of the daily metrics, P3 vs P4, valid days only."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(ROOT_OUT / "daily_aggregates.csv", dtype={"ID": str})
    if VALID_ONLY:
        d = d[d["valid_day"] == 1]
    if MIN_STS_PER_DAY is not None:
        d = d[d["n_sts"] >= MIN_STS_PER_DAY]
    d = add_weekday(d)
    rows = []
    for col, (title, ylabel) in WEEKDAY_METRICS.items():
        if col not in d.columns:
            print(f"[!] {col} not in daily_aggregates.csv: skipped")
            continue
        weekday_plot(d, col, title, ylabel, f"weekday_{col}_P3_P4.png")
        t = d.groupby(["phase", "weekday"])[col].agg(["size", "median", "mean", "std"]).round(2).reset_index()
        t.insert(0, "metric", col)
        t["weekday"] = t["weekday"].map(dict(enumerate(WEEKDAYS)))
        rows.append(t)
    if rows:
        tab = pd.concat(rows, ignore_index=True)
        tab.to_csv(FIG_DIR / "table_by_weekday_P3_P4.csv", index=False)
        print(tab.to_string(index=False))
    print(f"-> weekday figures saved in {FIG_DIR}")


# ============================= MAIN =============================
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sources = get_sources()

    figure(["duration", "n_sists"], sources, "hist_duration_n_sists_P3_P4.png")
    figure(["flexion", "extension", "acc_v", "tilt"], sources, "hist_kinematics_P3_P4.png", width_per_panel=6.5)
    figure(["sparc", "ldlj"], sources, "hist_smoothness_P3_P4.png")
    weekday_plots()

    thesis_hist_figure(sources)          # thesis figures
    thesis_weekday_figure()

    wide = table_by_day(sources)
    wide_p = table_percentiles_by_day(sources)
    wide_c = table_cov_by_day(sources)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print("MEAN ± SD\n", wide.to_string(), "\n\nP10 – P90\n", wide_p.to_string(),
              "\n\nCoV (%)\n", wide_c.to_string())
    print(f"\n-> figures and tables saved in {FIG_DIR}")


if __name__ == "__main__":
    main()