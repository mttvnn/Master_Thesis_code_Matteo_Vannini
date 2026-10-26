"""
filter_weartime.py — Keeps only the P3/P4 patients from the CVS daily weartime file.

Input:
  - WEARTIME_CSV                      cvs-T1-daily_weartime-*.csv
                                      (visit_type, participant_id, measurement_date, total_worn_h,
                                       total_worn_during_waking_h)
  - ROOT_OUT/summary_all_IDs.csv      ID -> fase mapping of the patients processed by run_jas_per_ID.py

Output: ROOT_OUT/weartime_P3_P4.csv
  ID, phase, visit_type, measurement_date (ISO YYYY-MM-DD), total_worn_h, total_worn_during_waking_h
  (participant_id renamed to ID; only rows of the selected VISITS and PHASES)

Environment: conda activate jas ; run from ~/tesi
"""
from pathlib import Path

import pandas as pd

# ============================= PARAMETERS =============================
VISITS = ["T1"]                    # must match run_jas_per_ID.py
FILTER_CFG = "default"             # must match run_jas_per_ID.py
ORIENT_FIX = False                 # True only if run_jas_per_ID.py writes to the _orientfix folder
ROOT_OUT = (Path.home() / "tesi" / "risultati" /
            f"JAS_per_ID_{'_'.join(VISITS)}_{FILTER_CFG}{'_orientfix' if ORIENT_FIX else ''}")

WEARTIME_CSV = Path.home() / "tesi" / "cvs-T1-daily_weartime-14-04-2026.csv"
PHASES = ["P3", "P4"]
OUT_CSV = ROOT_OUT / "weartime_P3_P4.csv"


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


def main(root=ROOT_OUT, weartime_csv=WEARTIME_CSV, out_csv=OUT_CSV):
    root = Path(root)
    wt = pd.read_csv(weartime_csv, sep=None, engine="python", dtype={"participant_id": str})
    wt["participant_id"] = wt["participant_id"].str.strip()

    summ = pd.read_csv(root / "summary_all_IDs.csv", dtype={"ID": str})
    id_phase = (summ[summ["fase"].isin(PHASES)][["ID", "fase"]]
                .drop_duplicates().rename(columns={"fase": "phase"}))
    dup = id_phase[id_phase.duplicated("ID", keep=False)]
    if not dup.empty:
        print(f"[!] IDs with more than one phase: {sorted(dup.ID.unique())}")

    out = (wt[wt["visit_type"].isin(VISITS)]
           .rename(columns={"participant_id": "ID"})
           .merge(id_phase, on="ID", how="inner"))
    out = out[["ID", "phase"] + [c for c in out.columns if c not in ("ID", "phase")]]
    out["_date"] = parse_dates(out["measurement_date"])
    out = out.sort_values(["phase", "ID", "_date"]).reset_index(drop=True)
    # unambiguous ISO date in the output (YYYY-MM-DD)
    out["measurement_date"] = out["_date"].dt.strftime("%Y-%m-%d")
    out = out.drop(columns="_date")
    out.to_csv(out_csv, index=False)

    # report
    missing = sorted(set(id_phase.ID) - set(out.ID))
    rep = out.groupby("phase").agg(patients=("ID", "nunique"), days=("ID", "size"))
    print(rep.to_string())
    if missing:
        print(f"[!] P3/P4 IDs without weartime rows ({len(missing)}): {missing}")
    print(f"-> saved {out_csv}")
    return out


if __name__ == "__main__":
    main()