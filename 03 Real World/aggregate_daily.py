"""
aggregate_daily.py — Daily aggregation of the JAS sit-to-stand events produced by run_jas_per_ID.py.

Input (in ROOT_OUT, same folder used by run_jas_per_ID.py):
  - summary_all_IDs.csv                        recording hours per ID / visit / day / recording
  - <fase>/<ID>/<ID>_<fase>_events.csv         events per patient
    (fallback: the per-day files <ID>_<visit>_DayN_<fase>_events.csv)

  - ROOT_OUT/weartime_P3_P4.csv                 daily weartime (from filter_weartime.py), optional

Output: ROOT_OUT/daily_aggregates.csv — one row per ID x day:
  ID, phase, day, valid_day, reoriented, n_sts,
  then for each variable, grouped: <var>_p10, <var>_p50, <var>_p90, <var>_cov  (cov in %, = SD / |mean| * 100)

  - only events with status in STATUS_KEEP (default KEPT) are used
  - all days are kept, including 0-hour recordings (MIN_RECORDING_H = None);
    set MIN_RECORDING_H to a number to exclude days with total recording <= that value
  - days with 0 STS are kept (n_sts = 0, statistics empty)
  - percentiles and CoV are computed only if the day has >= MIN_EVENTS_STATS events
  - reoriented = 1 if part of the day was corrected for sensor orientation (flipped_pct > 0 in
    summary_all_IDs.csv), else 0
  - valid_day = 1 if total_worn_during_waking_h > MIN_WAKING_WEAR_H (Mobilise-D criterion), else 0.
    MATCH_MODE = "hybrid" (default): by date (local date of the first recording_start of the day);
                 days without a date match are assigned by order (k-th weartime row = DayK),
                 only for IDs whose number of weartime rows equals the number of days.
    MATCH_MODE = "date" / "order": only one of the two methods.
    Days where the two methods disagree, and days left unmatched (valid_day = 0), are listed on screen.
  - VALID_DAY_OVERRIDE: valid days set by hand for specific IDs (weartime not referring to the recording).

Environment: conda activate jas ; run from ~/tesi after run_jas_per_ID.py
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ============================= PARAMETERS =============================
VISITS = ["T1"]                    # must match run_jas_per_ID.py
FILTER_CFG = "default"             # must match run_jas_per_ID.py
ORIENT_FIX = False                 # True only if run_jas_per_ID.py writes to the _orientfix folder
ROOT_OUT = (Path.home() / "tesi" / "risultati" /
            f"JAS_per_ID_{'_'.join(VISITS)}_{FILTER_CFG}{'_orientfix' if ORIENT_FIX else ''}")
OUT_CSV = ROOT_OUT / "daily_aggregates.csv"

# output name -> JAS column
VARIABLES = {
    "duration": "duration",
    "max_vertical_acc": "sts_max_accv",
    "max_flexion_velocity": "sts_max_flexion_vel",
    "max_extension_velocity": "sts_max_extension_vel",
    "tilt_angle": "sts_tilt_angle",
    "sparc_gyroml": "sts_sparc_gyroml",
    "jerkldlj_gyroml": "sts_ldlj_gyroml",
}
PERCENTILES = [10, 50, 90]
STATUS_KEEP = ["KEPT"]             # events used for n_sts and statistics
MIN_RECORDING_H = None             # None = keep all days (also 0 h); a number = exclude days with recording <= it
MIN_EVENTS_STATS = 3               # min STS in the day to compute percentiles and CoV (n_sts always reported)
ROUND = 4

WEARTIME_CSV = ROOT_OUT / "weartime_P3_P4.csv"   # output of filter_weartime.py; None = no valid_day column
MIN_WAKING_WEAR_H = 12.0                          # valid day: total_worn_during_waking_h > this value
MATCH_MODE = "hybrid"     # "hybrid" = by date, fallback by order (only IDs with same n. of days); "date"; "order"

# Manual valid days, for IDs whose weartime does not refer to the analysed recording
# (e.g. data extracted manually). ID -> list of VALID day numbers; the other days of that ID get 0.
# Overrides the weartime matching for these IDs only. Document it in the Methods.
import json
PARTICIPANTS_FILE = "path/to/participants.json"   # local file with participant IDs, NOT in the repository (see participants_template.json)
with open(PARTICIPANTS_FILE) as _f:
    _P = json.load(_f)

VALID_DAY_OVERRIDE = _P.get("VALID_DAY_OVERRIDE", {})


# ============================= HELPERS =============================
def load_events(root, pid, fase):
    """All events of one patient: per-patient file if present, otherwise the per-day files."""
    pdir = root / fase / pid
    f = pdir / f"{pid}_{fase}_events.csv"
    files = [f] if f.exists() else sorted(pdir.glob(f"{pid}_*_Day*_{fase}_events.csv"))
    dfs = []
    for x in files:
        try:
            dfs.append(pd.read_csv(x, dtype={"ID": str}))
        except pd.errors.EmptyDataError:
            pass
    dfs = [d for d in dfs if not d.empty]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def cov_percent(x):
    """Coefficient of variation in %: SD (ddof=1) / |mean| * 100."""
    if len(x) < 2:
        return np.nan
    m = x.mean()
    return x.std(ddof=1) / abs(m) * 100 if m != 0 else np.nan


def day_stats(e, row):
    for name, col in VARIABLES.items():
        if len(e) and col in e.columns:
            x = pd.to_numeric(e[col], errors="coerce").dropna()
        else:
            x = pd.Series(dtype=float)
        ok = len(x) >= MIN_EVENTS_STATS
        for p in PERCENTILES:
            row[f"{name}_p{p}"] = np.percentile(x, p) if ok else np.nan
        row[f"{name}_cov"] = cov_percent(x) if ok else np.nan
    return row


def print_phase_percentiles(out):
    """Screen only (not saved): per phase, p10/p50/p90 across days of n_sts and of each variable's daily p50."""
    cols = {"n_sts": "n_sts", **{name: f"{name}_p50" for name in VARIABLES}}
    rows = []
    for name, col in cols.items():
        for ph, g in out.groupby("phase"):
            x = g[col].dropna()
            rows.append(dict(variable=name if name == "n_sts" else f"{name} (daily p50)", phase=ph, n_days=len(x),
                             **{f"p{p}": np.percentile(x, p) if len(x) else np.nan for p in PERCENTILES}))
    tab = pd.DataFrame(rows).set_index(["variable", "phase"])
    print("\nPercentiles across days, per phase (screen only, not saved)")
    print(tab.round(3).to_string())


def local_date(x):
    """Calendar date in the recording's own local time ('2021-04-13 00:05:00+02:00' -> 2021-04-13).
    Each value is parsed on its own, so mixed timezone offsets (e.g. +01:00 / +02:00, DST) are fine."""
    try:
        return pd.Timestamp(datetime.fromisoformat(str(x).strip()).date())
    except (ValueError, TypeError):
        return pd.NaT


def parse_dates(col):
    """Robust parsing of measurement_date: ISO (YYYY-MM-DD) or dd/mm/yyyy vs mm/dd/yyyy detected on the
    whole column (a field > 12 tells which is the day). Needed because re-saving the CSV from
    LibreOffice (English USA) rewrites dd/mm/yyyy as mm/dd/yy."""
    s = col.astype(str).str.strip()
    if s.str.match(r"^\d{4}-\d{1,2}-\d{1,2}").all():
        return pd.to_datetime(s.str[:10], format="%Y-%m-%d", errors="coerce")
    parts = s.str.extract(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})")
    a, b = pd.to_numeric(parts[0], errors="coerce"), pd.to_numeric(parts[1], errors="coerce")
    if (b > 12).any() and not (a > 12).any():
        order = "%m/%d"
    else:
        order = "%d/%m"
        if not (a > 12).any():
            print("[!] measurement_date: day/month order ambiguous, assuming dd/mm")
    year = "%Y" if parts[2].str.len().max() == 4 else "%y"
    norm = parts[0] + "/" + parts[1] + "/" + parts[2]
    return pd.to_datetime(norm, format=f"{order}/{year}", errors="coerce")


def load_weartime(weartime_csv):
    """Weartime with calendar date, validity flag and order index (k-th row of the ID/visit = DayK)."""
    wt = pd.read_csv(weartime_csv, dtype={"ID": str})
    wt["wdate"] = parse_dates(wt["measurement_date"]).dt.normalize()
    if wt["wdate"].isna().any():
        print(f"[!] {int(wt['wdate'].isna().sum())} weartime dates could not be parsed")
    wt["wvalid"] = (pd.to_numeric(wt["total_worn_during_waking_h"], errors="coerce") > MIN_WAKING_WEAR_H).astype(int)
    wt = wt.sort_values(["ID", "visit_type", "wdate"])
    wt["wday"] = wt.groupby(["ID", "visit_type"]).cumcount() + 1
    return wt


def match_by_date(summ, wt):
    """{(ID, Visit, day): 0/1 or None} matching the local date of the first recording_start with measurement_date."""
    lut = {(r.ID, r.visit_type, r.wdate): r.wvalid for r in wt.itertuples(index=False)}
    s = summ.dropna(subset=["day"]).copy()
    s["start_date"] = s["recording_start"].map(local_date)
    day_date = s.groupby(["ID", "Visit", "day"])["start_date"].min()
    return {k: (lut.get((k[0], k[1], d)) if pd.notna(d) else None) for k, d in day_date.items()}


def match_by_order(summ, wt):
    """{(ID, Visit, day): 0/1 or None}: k-th weartime row is DayK, only for IDs with the same number of days."""
    lut = {(r.ID, r.visit_type, r.wday): r.wvalid for r in wt.itertuples(index=False)}
    n_wt = wt.groupby(["ID", "visit_type"]).size()
    s = summ.dropna(subset=["day"])
    n_days = s.groupby(["ID", "Visit"])["day"].nunique()
    out, mism = {}, []
    for (pid, visit, day) in s.groupby(["ID", "Visit", "day"]).groups:
        same = n_wt.get((pid, visit), 0) == n_days[(pid, visit)]
        out[(pid, visit, day)] = lut.get((pid, visit, int(day))) if same else None
    for (pid, visit), n in n_days.items():
        if n_wt.get((pid, visit), 0) != n:
            mism.append(f"{pid} {visit}: weartime {n_wt.get((pid, visit), 0)} rows vs {n} days")
    return out, mism


def build_valid_day(summ, weartime_csv, mode=MATCH_MODE):
    """{(ID, Visit, day): 0/1}. mode: "hybrid" (date, then order as fallback), "date" or "order"."""
    wt = load_weartime(weartime_csv)
    by_date = match_by_date(summ, wt)
    by_order, mism = match_by_order(summ, wt)

    valid, fallback, unmatched, disagree = {}, [], [], []
    for k in by_date.keys() | by_order.keys():
        d, o = by_date.get(k), by_order.get(k)
        tag = f"{k[0]} {k[1]} Day{int(k[2])}"
        if d is not None and o is not None and d != o:
            disagree.append(f"{tag}: date={d} order={o}")
        if mode == "order":
            v = o
        elif mode == "date":
            v = d
        else:                                   # hybrid
            v = d if d is not None else o
            if d is None and o is not None:
                fallback.append(tag)
        if v is None:
            unmatched.append(tag)
            v = 0
        valid[k] = v

    print(f"valid_day: match mode = {mode}")
    if mism:
        print(f"[!] {len(mism)} IDs with a different number of days (order match not used for them):")
        for m in mism:
            print(f"    {m}")
    if disagree:
        print(f"[!] {len(disagree)} days where date and order matching disagree (check by hand):")
        for m in sorted(disagree):
            print(f"    {m}")
    if fallback:
        print(f"[i] {len(fallback)} days without date match, assigned by order: {sorted(fallback)}")
    if unmatched:
        print(f"[!] {len(unmatched)} days not matched -> valid_day = 0: {sorted(unmatched)}")
    return valid


# ============================= MAIN =============================
def main(root=ROOT_OUT, out_csv=OUT_CSV):
    root, out_csv = Path(root), Path(out_csv)
    summ = pd.read_csv(root / "summary_all_IDs.csv", dtype={"ID": str})
    days = (summ.dropna(subset=["day"])
                .groupby(["ID", "fase", "Visit", "day"], as_index=False)
                .agg(recording_h=("duration_h", "sum"),
                     flipped_pct=("flipped_pct", "max") if "flipped_pct" in summ.columns else ("duration_h", lambda x: 0.0)))
    n_all = len(days)
    if MIN_RECORDING_H is not None:
        days = days[days.recording_h > MIN_RECORDING_H]
        print(f"Days: {n_all} total, {n_all - len(days)} excluded (recording <= {MIN_RECORDING_H} h)")
    else:
        print(f"Days: {n_all} total, none excluded "
              f"({int((days.recording_h == 0).sum())} with 0 h recording)")

    valid = None
    if WEARTIME_CSV and Path(WEARTIME_CSV).exists():
        valid = build_valid_day(summ, WEARTIME_CSV, MATCH_MODE)
    if VALID_DAY_OVERRIDE:
        valid = {} if valid is None else valid
        for pid, vdays in VALID_DAY_OVERRIDE.items():
            keys = [k for k in summ.dropna(subset=["day"]).groupby(["ID", "Visit", "day"]).groups if k[0] == str(pid)]
            for k in keys:
                valid[k] = int(int(k[2]) in vdays)
            print(f"[i] valid_day set manually for {pid}: valid days {sorted(vdays)} "
                  f"({len(keys)} days in the recording)")
    if valid is None:
        print(f"[!] weartime file not found ({WEARTIME_CSV}): valid_day not added")

    rows = []
    for (pid, fase), d in days.groupby(["ID", "fase"]):
        ev = load_events(root, pid, fase)
        if not ev.empty:
            ev = ev[ev["status"].isin(STATUS_KEEP)] if "status" in ev.columns else ev
            missing = [c for c in VARIABLES.values() if c not in ev.columns]
            if missing:
                print(f"[!] {pid}: missing columns {missing}")
        for r in d.itertuples(index=False):
            e = ev[(ev["Visit"] == r.Visit) & (ev["day"] == r.day)] if not ev.empty else ev
            row = dict(ID=pid, phase=fase, day=int(r.day))
            row["reoriented"] = int(pd.notna(r.flipped_pct) and r.flipped_pct > 0)
            if valid is not None:
                row["valid_day"] = valid.get((pid, r.Visit, r.day), 0)
            row["n_sts"] = len(e)
            rows.append(day_stats(e, row))

    out = pd.DataFrame(rows)
    if out.empty:
        print("[!] No valid days found")
        return out
    if days.groupby("ID")["Visit"].nunique().max() > 1:
        print("[!] some IDs have more than one visit: 'day' is repeated across visits (no Visit column)")
    out = out.sort_values(["phase", "ID", "day"]).reset_index(drop=True)
    # explicit column order: ID, phase, day, n_sts, then each variable followed by its statistics
    stat_cols = [f"{name}_{s}" for name in VARIABLES for s in [f"p{p}" for p in PERCENTILES] + ["cov"]]
    first = ["ID", "phase", "day"] + (["valid_day"] if "valid_day" in out.columns else []) + ["reoriented", "n_sts"]
    out = out[first + stat_cols]
    num = out.select_dtypes("number").columns.difference(["day", "n_sts", "valid_day", "reoriented"])
    out[num] = out[num].round(ROUND)
    out.to_csv(out_csv, index=False)

    # short report
    rep = out.groupby("phase").agg(patients=("ID", "nunique"), days=("day", "size"),
                                   zero_days=("n_sts", lambda s: int((s == 0).sum())),
                                   days_with_stats=(f"{next(iter(VARIABLES))}_p50", lambda s: int(s.notna().sum())),
                                   n_sts_mean=("n_sts", "mean"),
                                   **({"valid_days": ("valid_day", "sum")} if "valid_day" in out.columns else {}))
    print(rep.round(2).to_string())
    print_phase_percentiles(out)
    print(f"-> saved {out_csv}")
    return out


if __name__ == "__main__":
    main()