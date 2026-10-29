"""
inspect_signal.py — Visual inspection of the full-day signal for selected IDs / days
(e.g. valid days with 0 or very few STS).

For every selected ID x day x recording it saves one figure in ROOT_OUT/inspection/:
  1. per-minute mean acceleration on V / ML / AP (g)  -> posture and sensor orientation
       upright: V ~ +1 g | lying: V ~ 0 g | sensor upside down: V ~ -1 g
  2. vertical acceleration, 1-s min/max envelope, with JAS sit2stand events
  3. mediolateral gyroscope (sagittal trunk rotation), 1-s envelope, with events
  4. events per hour, stacked by status (KEPT / REJECTED_WB / REJECTED_FILTER)
  Events: green = KEPT, orange = REJECTED_WB, red = REJECTED_FILTER.
and a table ROOT_OUT/inspection/inspection_summary.csv with hours, counts and posture percentages.

Optional: SAVE_EVENT_ZOOMS = True saves a zoomed plot of every event (same as SAVE_PLOTS in run_jas_per_ID.py).

Requires run_jas_per_ID.py in the same folder (loading functions, paths and phases are reused from it).
Environment: conda activate jas ; run from ~/tesi
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_all_subject as R

# ============================= PARAMETERS =============================
# ID -> list of day numbers, or "all"   (CHECK the transcription from the handwritten notes)
import json
PARTICIPANTS_FILE = "path/to/participants.json"   # local file with participant IDs, NOT in the repository (see participants_template.json)
with open(PARTICIPANTS_FILE) as _f:
    _P = json.load(_f)

INSPECT = _P.get("INSPECT", {})   # ID -> list of day numbers, or "all"
SAVE_EVENT_ZOOMS = False           # True = also one zoomed plot per event (slow on many events)
OUT_DIR = R.ROOT_OUT / "inspection"
UPRIGHT_G = 0.7                    # per-minute mean V > this  -> upright
LYING_G = 0.5                      # per-minute |mean V| < this -> lying / reclined
COLORS = {"KEPT": "tab:green", "REJECTED_WB": "tab:orange", "REJECTED_FILTER": "tab:red"}


# ============================= HELPERS =============================
def block_stats(x, n):
    """min, max, mean over consecutive blocks of n samples."""
    m = len(x) // n
    if m == 0:
        return np.array([]), np.array([]), np.array([])
    b = x[:m * n].reshape(m, n)
    return b.min(1), b.max(1), b.mean(1)


def load_patient_events(pid, fase):
    f = R.ROOT_OUT / fase / pid / f"{pid}_{fase}_events.csv"
    if not f.exists():
        return pd.DataFrame()
    try:
        ev = pd.read_csv(f, dtype={"ID": str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "reason" in ev.columns:
        ev["reason"] = ev["reason"].fillna("")
    return ev


def load_valid_days():
    f = R.ROOT_OUT / "daily_aggregates.csv"
    if not f.exists():
        return {}
    d = pd.read_csv(f, dtype={"ID": str})
    if "valid_day" not in d.columns:
        return {}
    return {(r.ID, int(r.day)): int(r.valid_day) for r in d.itertuples(index=False)}


# ============================= PLOT =============================
def plot_day(pid, fase, day_name, rec, acc, gyr, fs, start, ev, valid):
    n_s, n_m = int(round(fs)), int(round(60 * fs))
    hours = len(acc) / fs / 3600
    t0h = start.hour + start.minute / 60 + start.second / 3600 if start else 0.0
    xlabel = "Clock time (h)" if start else "Time from recording start (h)"

    vmin, vmax, _ = block_stats(acc[:, 0], n_s)
    gmin, gmax, _ = block_stats(gyr[:, 1], n_s)
    ts = t0h + (np.arange(len(vmin)) + 0.5) / 3600
    means = [block_stats(acc[:, k], n_m)[2] for k in range(3)]
    tm = t0h + (np.arange(len(means[0])) + 0.5) / 60

    mv = means[0]
    pct = lambda mask: float(np.mean(mask) * 100) if len(mv) else np.nan
    upright, lying, inverted = pct(mv > UPRIGHT_G), pct(np.abs(mv) < LYING_G), pct(mv < -UPRIGHT_G)

    counts = ev["status"].value_counts() if not ev.empty else pd.Series(dtype=int)
    n = {s: int(counts.get(s, 0)) for s in COLORS}

    fig, ax = plt.subplots(4, 1, figsize=(16, 11), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.2, 1.5, 1.5, 1]))
    # 1. posture / orientation
    for k, (lab, col) in enumerate(zip(["V", "ML", "AP"], ["black", "tab:purple", "tab:cyan"])):
        ax[0].plot(tm, means[k], color=col, lw=1, label=f"Acc {lab}")
    for y in (1, UPRIGHT_G, 0, -UPRIGHT_G, -1):
        ax[0].axhline(y, color="grey", lw=0.6, ls=":" if abs(y) != 1 else "--")
    ax[0].set_ylabel("1-min mean acc (g)")
    ax[0].set_ylim(-1.3, 1.3)
    ax[0].legend(loc="upper right", ncol=3, fontsize=8)

    # 2-3. signal envelopes
    ax[1].fill_between(ts, vmin * 9.81, vmax * 9.81, color="black", lw=0)
    ax[1].set_ylabel("Vertical acc (m/s²)")
    ax[2].fill_between(ts, gmin, gmax, color="tab:blue", lw=0)
    ax[2].set_ylabel("ML gyro (°/s)")

    # events
    if not ev.empty:
        for _, e in ev.iterrows():
            x = t0h + e["start_time"] / 3600
            for a in (ax[1], ax[2]):
                a.axvline(x, color=COLORS.get(e["status"], "red"), lw=0.8, alpha=0.8)

    # 4. events per hour by status
    edges = np.arange(np.floor(t0h), np.ceil(t0h + max(hours, 1 / 60)) + 1)
    bottom = np.zeros(len(edges) - 1)
    for s, col in COLORS.items():
        x = t0h + ev.loc[ev["status"] == s, "start_time"].to_numpy() / 3600 if not ev.empty else np.array([])
        h, _ = np.histogram(x, bins=edges)
        ax[3].bar(edges[:-1], h, width=1, align="edge", bottom=bottom, color=col, edgecolor="white", label=s)
        bottom += h
    ax[3].set_ylabel("Events / h")
    ax[3].set_xlabel(xlabel)
    ax[3].legend(loc="upper right", ncol=3, fontsize=8)
    for a in ax:
        a.grid(True, alpha=0.3)

    vd = "?" if valid is None else valid
    ax[0].set_title(f"{pid} | {fase} | {day_name} | {rec} | valid_day={vd} | {hours:.1f} h | "
                    f"JAS={len(ev)}  KEPT={n['KEPT']}  WB={n['REJECTED_WB']}  filter={n['REJECTED_FILTER']}\n"
                    f"upright {upright:.0f}%  |  lying/reclined {lying:.0f}%  |  sensor inverted {inverted:.0f}%",
                    fontsize=11, fontweight="bold")
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{pid}_{day_name}_{fase}_{rec}.png", dpi=110)
    plt.close(fig)

    return dict(ID=pid, phase=fase, day=R.day_number(day_name), Recording=rec, valid_day=valid,
                hours=round(hours, 2), jas=len(ev), kept=n["KEPT"], rejected_wb=n["REJECTED_WB"],
                rejected_filter=n["REJECTED_FILTER"], upright_pct=round(upright, 1),
                lying_pct=round(lying, 1), inverted_pct=round(inverted, 1))


# ============================= MAIN =============================
def main():
    valid_map = load_valid_days()
    rows = []
    for pid, days_req in INSPECT.items():
        pid = str(pid)
        fase = R.get_phase(pid)
        cohort, pdir = R.find_patient(pid)
        if pdir is None:
            print(f"[!] {pid}: folder not found")
            continue
        ev_all = load_patient_events(pid, fase)
        days = [(v, d, p) for v, d, p in R.find_days(pdir)
                if days_req == "all" or R.day_number(d) in days_req]
        if not days:
            print(f"[!] {pid}: none of the requested days found ({days_req})")
            continue
        for visit, day_name, path in days:
            dnum = R.day_number(day_name)
            try:
                recs = R.load_day(path, R.TM_CHOICE)
            except Exception as ex:
                print(f"[!] {pid} {day_name}: loading error {type(ex).__name__}: {ex}")
                continue
            for rec, (acc, gyr, fs, start) in recs.items():
                if len(acc) < 60 * fs:
                    print(f"[!] {pid} {day_name} {rec}: recording shorter than 1 min, skipped")
                    continue
                ev = (ev_all[(ev_all["day"] == dnum) & (ev_all["Visit"] == visit) & (ev_all["Recording"] == rec)]
                      if not ev_all.empty else ev_all)
                row = plot_day(pid, fase, day_name, rec, acc, gyr, fs, start, ev, valid_map.get((pid, dnum)))
                rows.append(row)
                print(f"   {pid} {day_name} {rec}: {row['hours']} h | JAS={row['jas']} KEPT={row['kept']} | "
                      f"upright {row['upright_pct']}% lying {row['lying_pct']}% inverted {row['inverted_pct']}%")
                if SAVE_EVENT_ZOOMS and not ev.empty:
                    for _, e in ev.iterrows():
                        R.save_plot(acc, gyr, fs, e, OUT_DIR / "events",
                                    f"{pid}_{day_name}_{fase}_{rec}_ev{int(e['ev']):03d}")
    if rows:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT_DIR / "inspection_summary.csv", index=False)
        print(f"-> figures and inspection_summary.csv saved in {OUT_DIR}")


if __name__ == "__main__":
    main()