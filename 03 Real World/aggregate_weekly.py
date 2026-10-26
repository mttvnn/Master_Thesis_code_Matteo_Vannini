"""
aggregate_weekly.py — Weekly (per-subject) aggregation of the daily STS metrics, on valid days only.

Input:  ROOT_OUT/daily_aggregates.csv   (from aggregate_daily.py, must contain valid_day)
Output: ROOT_OUT/weekly_aggregates.csv  — one row per subject:
  ID, phase, n_days_valid, n_days_reoriented (valid days with orientation correction), then the MEAN over the valid days of every daily metric
  (n_sts and <var>_p10 / _p50 / _p90 / _cov), same column names as in the daily file.

  - only days with valid_day == 1 are used
  - subjects without valid days are kept (n_days_valid = 0, metrics empty)
  - n_sts is averaged over all valid days (zeros included); the other metrics only exist on days with
    >= MIN_EVENTS_STATS STS (see aggregate_daily.py), so their mean uses those days only

Environment: conda activate jas ; run from ~/tesi after aggregate_daily.py
"""
from pathlib import Path

import pandas as pd

# ============================= PARAMETERS =============================
VISITS = ["T1"]                    # must match run_jas_per_ID.py
FILTER_CFG = "default"             # must match run_jas_per_ID.py
ORIENT_FIX = False                 # True only if run_jas_per_ID.py writes to the _orientfix folder
ROOT_OUT = (Path.home() / "tesi" / "risultati" /
            f"JAS_per_ID_{'_'.join(VISITS)}_{FILTER_CFG}{'_orientfix' if ORIENT_FIX else ''}")
IN_CSV = ROOT_OUT / "daily_aggregates.csv"
OUT_CSV = ROOT_OUT / "weekly_aggregates.csv"
ROUND = 4


def main(in_csv=IN_CSV, out_csv=OUT_CSV):
    d = pd.read_csv(in_csv, dtype={"ID": str})
    if "valid_day" not in d.columns:
        raise ValueError("valid_day column missing: run filter_weartime.py and then aggregate_daily.py")

    metrics = [c for c in d.columns if c not in ("ID", "phase", "day", "valid_day", "reoriented")]
    subjects = d[["ID", "phase"]].drop_duplicates()

    v = d[d["valid_day"] == 1]
    n_valid = v.groupby(["ID", "phase"]).size().rename("n_days_valid")
    has_reor = "reoriented" in d.columns
    n_reor = (v.groupby(["ID", "phase"])["reoriented"].sum().rename("n_days_reoriented")
              if has_reor else None)
    means = v.groupby(["ID", "phase"])[metrics].mean()

    out = subjects.merge(n_valid.reset_index(), on=["ID", "phase"], how="left")
    if has_reor:
        out = out.merge(n_reor.reset_index(), on=["ID", "phase"], how="left")
        out["n_days_reoriented"] = out["n_days_reoriented"].fillna(0).astype(int)
    out = out.merge(means.reset_index(), on=["ID", "phase"], how="left")
    out["n_days_valid"] = out["n_days_valid"].fillna(0).astype(int)
    out[metrics] = out[metrics].round(ROUND)
    out = out[["ID", "phase", "n_days_valid"] + (["n_days_reoriented"] if has_reor else []) + metrics].sort_values(["phase", "ID"]).reset_index(drop=True)
    out.to_csv(out_csv, index=False)

    # report
    rep = out.groupby("phase").agg(subjects=("ID", "size"),
                                   with_valid_days=("n_days_valid", lambda s: int((s > 0).sum())),
                                   valid_days_median=("n_days_valid", "median"),
                                   n_sts_mean=("n_sts", "mean"))
    print(rep.round(2).to_string())
    print(f"-> saved {out_csv}")
    return out


if __name__ == "__main__":
    main()