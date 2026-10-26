"""
scan_orientation.py — Orientation check (sensor upside down) on candidate IDs, without running JAS.

1. Candidate selection from ROOT_OUT/daily_aggregates.csv (valid days only, IDs already corrected excluded):
     a day is suspicious if
       (abs)  n_sts < ABS_FRACTION x mean n_sts of the valid days of its phase, or
       (rel)  n_sts < REL_DROP x median n_sts of the other valid days of the same subject
     an ID is a candidate if it has at least one suspicious day.
   (abs) catches subjects inverted for the whole week, (rel) catches inversions starting mid-week.
   With SELECT = "all" every ID is scanned.
2. For every candidate all days are loaded and, per recording, the same logic used by the correction
   (run_jas_per_ID.orientation_mask) gives the inverted stretches.

Output in ROOT_OUT/orientation_scan/:
  candidates.csv          suspicious days with the criterion that flagged them
  orientation_scan.csv    one row per ID x day x recording: hours, upright / lying / inverted %,
                          flipped_pct (% that the correction would flip), first/last flipped time
  At the end the IDs with flipped_pct >= MIN_FLIPPED_PCT on at least one day are printed as a list,
  ready for ORIENT_FIX_IDS in run_jas_per_ID.py.
Restart: IDs already in orientation_scan.csv are skipped.

Requires run_jas_per_ID.py in the same folder.
Environment: conda activate jas ; run from ~/tesi
"""
import time
from datetime import timedelta

import numpy as np
import pandas as pd

import run_all_subject as R

# ============================= PARAMETERS =============================
SELECT = "candidates"            # "candidates" or "all"
ABS_FRACTION = 0.25              # day suspicious if n_sts < 25% of the phase mean (valid days)
REL_DROP = 0.5                   # day suspicious if n_sts < 50% of the subject's median of the other valid days
EXCLUDE_IDS = list(R.ORIENT_FIX_IDS)     # already corrected
MIN_FLIPPED_PCT = 5.0            # % of a day flipped to flag the ID in the final list
ORIENT_G = R.ORIENT_G
OUT_DIR = R.ROOT_OUT / "orientation_scan"


# ============================= CANDIDATES =============================
def select_candidates():
    d = pd.read_csv(R.ROOT_OUT / "daily_aggregates.csv", dtype={"ID": str})
    v = d[(d["valid_day"] == 1) & (~d["ID"].isin(EXCLUDE_IDS))].copy()
    phase_mean = v.groupby("phase")["n_sts"].transform("mean")
    v["abs_flag"] = v["n_sts"] < ABS_FRACTION * phase_mean

    def other_median(g):
        return g["n_sts"].apply(lambda x: np.nan) if len(g) < 2 else \
            pd.Series([g["n_sts"].drop(i).median() for i in g.index], index=g.index)
    v["other_median"] = v.groupby("ID", group_keys=False).apply(other_median)
    v["rel_flag"] = v["n_sts"] < REL_DROP * v["other_median"]

    sus = v[v["abs_flag"] | v["rel_flag"]].copy()
    sus["criterion"] = np.where(sus["abs_flag"] & sus["rel_flag"], "abs+rel",
                                np.where(sus["abs_flag"], "abs", "rel"))
    sus = sus[["ID", "phase", "day", "n_sts", "other_median", "criterion"]].round(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sus.to_csv(OUT_DIR / "candidates.csv", index=False)
    print(f"Valid days: {len(v)} | suspicious days: {len(sus)} | candidate IDs: {sus['ID'].nunique()} "
          f"(of {v['ID'].nunique()})")
    print(sus.groupby("phase")["ID"].nunique().rename("candidate IDs").to_string())
    return sorted(sus["ID"].unique())


# ============================= SCAN =============================
def scan_recording(acc, fs, start):
    n_m = int(round(60 * fs))
    m = len(acc) // n_m
    if m == 0:
        return None
    mv = acc[:m * n_m, 0].reshape(m, n_m).mean(1)
    mask = R.orientation_mask(acc, fs)
    mask_min = mask[:m * n_m].reshape(m, n_m)[:, 0]
    idx = np.flatnonzero(mask_min)
    clock = (lambda k: (start + timedelta(minutes=int(k))).strftime("%H:%M")) if start else (lambda k: f"{k} min")
    return dict(hours=round(len(acc) / fs / 3600, 2),
                upright_pct=round(float(np.mean(mv > ORIENT_G)) * 100, 1),
                lying_pct=round(float(np.mean(np.abs(mv) < 0.5)) * 100, 1),
                inverted_pct=round(float(np.mean(mv < -ORIENT_G)) * 100, 1),
                flipped_pct=round(float(mask.mean()) * 100, 1),
                first_flipped=clock(idx[0]) if len(idx) else "",
                last_flipped=clock(idx[-1] + 1) if len(idx) else "")


def main():
    ids = select_candidates() if SELECT == "candidates" else \
        [p for p in R.ID_TO_PHASE if p not in EXCLUDE_IDS]
    out_csv = OUT_DIR / "orientation_scan.csv"
    done = set(pd.read_csv(out_csv, dtype={"ID": str})["ID"]) if out_csv.exists() else set()
    todo = [p for p in ids if p not in done]
    print(f"IDs to scan: {len(todo)} ({len(done & set(ids))} already done)")

    t_start, n_files = time.time(), 0
    for i, pid in enumerate(todo, 1):
        _, pdir = R.find_patient(pid)
        if pdir is None:
            print(f"[!] {pid}: folder not found")
            continue
        rows = []
        for visit, day_name, path in R.find_days(pdir):
            try:
                recs = R.load_day(path, R.TM_CHOICE)
            except Exception as ex:
                print(f"[!] {pid} {day_name}: loading error {type(ex).__name__}: {ex}")
                continue
            n_files += 1
            for rec, (acc, gyr, fs, start) in recs.items():
                r = scan_recording(acc, fs, start)
                if r is not None:
                    rows.append(dict(ID=pid, phase=R.get_phase(pid), day=R.day_number(day_name),
                                     Recording=rec, **r))
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
            flagged = df[df["flipped_pct"] > 0]
            txt = ", ".join(f"Day{int(r.day)} {r.flipped_pct}%" for r in flagged.itertuples()) or "none"
        else:
            txt = "no data"
        el = time.time() - t_start
        eta = el / i * (len(todo) - i)
        print(f"[{i}/{len(todo)}] {pid}: flipped -> {txt}   "
              f"({el / max(n_files, 1):.1f} s/file, ETA {eta / 60:.0f} min)")

    if out_csv.exists():
        res = pd.read_csv(out_csv, dtype={"ID": str})
        res = res[res["ID"].isin(ids)]
        hit = res[res["flipped_pct"] >= MIN_FLIPPED_PCT]
        summary = (hit.groupby(["ID", "phase"])
                      .agg(days=("day", lambda s: ", ".join(map(str, sorted(s.astype(int).unique())))),
                           max_flipped_pct=("flipped_pct", "max"))
                      .reset_index())
        print("\n" + "=" * 60)
        print(f"IDs with >= {MIN_FLIPPED_PCT}% flipped on at least one day: {len(summary)}")
        print("=" * 60)
        if not summary.empty:
            print(summary.to_string(index=False))
            print("\nTo add to ORIENT_FIX_IDS:")
            print(summary["ID"].tolist())
        print(f"\n-> saved in {OUT_DIR}")


if __name__ == "__main__":
    main()