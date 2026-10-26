"""
run_jas_per_ID.py — Automatic processing per patient of all days for the selected visits.

For each ID in PHASE_IDS (P3, P4, ...):
  - searches for the patient in ROOT/<cohort>/<ID> (PFF11, PFF16, PFF18, ...)
  - processes ALL days of the VISITS (e.g., T1_Day1..T1_DayN) every Recording
  - pipeline: [orientation fix] -> JAS predict -> walking bout filter -> external filter (ext_filter_jas.py)
    orientation fix only for ORIENT_FIX_IDS: V and ML flipped where the sensor is upside down;
    summary column flipped_pct (% of the recording flipped), events column flipped (1 = event in a
    flipped stretch); IDs in REPROCESS_FIX_IDS (True = all ORIENT_FIX_IDS, or a list) are
    reprocessed even when SKIP_DONE = True
  - every events/summary table starts with the columns ID, day, fase (each only once:
    no PatientID / Day duplicates; the visit is in the "Visit" column)
  - saves into ROOT_OUT/<fase>/<ID>/:
      per day (SAVE_PER_DAY):
        <ID>_<day>_<fase>_events.csv      all sit2stand transitions of that day
        <ID>_<day>_<fase>_events.xlsx     sheets "Events", "KEPT", "Summary_days"
      per patient (SAVE_PER_PATIENT):
        <ID>_<fase>_events.xlsx  sheet "Events"          all sit2stand transitions with ALL JAS variables,
                                                         ordered by visit, day, recording, and time
                                 sheet "KEPT"            only the kept sit2stand transitions
                                 sheet "Summary_days"    daily counts
                                 sheet "Summary_visit"   average STS/day per visit
        <ID>_<fase>_events.csv            same content as the "Events" sheet
        <ID>_<fase>_summary_days.csv
  - at the end: summary of all IDs in ROOT_OUT/summary_all_IDs.csv
                 skipped/failed IDs in ROOT_OUT/skipped_IDs.csv (ID, fase, reason, detail)

Restart: with SKIP_DONE = True, already processed IDs are skipped.
Requires ext_filter_jas.py in ~/tesi. Openpyxl is required for Excel (conda install openpyxl).
Environment: conda activate jas ; run from ~/tesi
"""
import os
import re
import glob
import time
import warnings
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jas_detector import JASDetector
from ext_filter_jas import filter_transitions, filter_rules, DEFAULT_THRESHOLDS, IMPAIRED_THRESHOLDS
from mobgap.gait_sequences import GsdIonescu

# ============================= PARAMETERS =============================
ROOT = "path/to/CVS/PFF"                     # contains one folder per recruiting site, then one per participant

# IDs per phase: fill in the lists. An ID must appear in only one phase.
import json
PARTICIPANTS_FILE = "path/to/participants.json"   # local file with participant IDs, NOT in the repository (see participants_template.json)
with open(PARTICIPANTS_FILE) as _f:
    _P = json.load(_f)

PHASE_IDS = {"P3": _P["P3_IDS"], "P4": _P["P4_IDS"]}
PATIENT_IDS = None                            # None = all IDs in PHASE_IDS; or a list to process only a subset

# Optional per-ID folder override, for patients stored outside ROOT/<cohort>/<ID>.
# Value: patient folder path (cohort = name of its parent folder) or a tuple (cohort, path).
# The folder must still contain <visit>_DayN/data.mat (at any depth). IDs not listed use ROOT.
PATIENT_PATHS = {
    # "<ID>": "path/to/<site>/<ID>",
    # "<ID>": ("<site>", "path/to/<ID>_fixed"),
}

VISITS = ["T1"]                               # visits to process (all their days)
TM_CHOICE = "TimeMeasure1"
FS_DEFAULT = 100.0

JAS_PARAMS = dict(val_min_vdisp=0.1, disp_threshold_m=0.1, val_max_vdisp=2.0, val_min_gap=0.5,
                  lp_cutoff_hz=2.0)
FILTER_CFG = "default"                        # "default" or "impaired"
USE_WB_FILTER = True
USE_EXT_FILTER = True
ORDER = "WB_first"                            # "WB_first" or "filter_first"
TAIL_FRACTION = 0.25

SAVE_EXCEL = False
ONLY_KEPT = False                             # True = events CSV/Excel contain only KEPT transitions
SAVE_PER_DAY = True                           # one file per day: <ID>_<day>_<fase>_events.csv/.xlsx
SAVE_PER_PATIENT = True                       # one file per patient with all days: <ID>_<fase>_events.csv/.xlsx
SAVE_PLOTS = False                            # True = one plot per event (slow on many days)
PLOT_PAD_S = 5.0
SKIP_DONE = True                             # skips IDs that already have the output file

ROOT_OUT = Path.home() / "tesi" / "risultati" / f"JAS_per_ID_{'_'.join(VISITS)}_{FILTER_CFG}"

# Sensor orientation correction (only for the IDs below): where the sensor is upside down
# (per-minute mean V < -ORIENT_G), V and ML are sign-flipped on acc and gyro (180° rotation about AP)
# before JAS. Days/stretches with normal orientation are left untouched.
ORIENT_FIX_IDS = _P["ORIENT_FIX_IDS"]
# IDs reprocessed even if SKIP_DONE = True: True = all ORIENT_FIX_IDS, False = none, or a list
REPROCESS_FIX_IDS = _P["REPROCESS_FIX_IDS"]
ORIENT_G = 0.7                                # |per-minute mean V| above this -> upright (+) / upside down (-)
ORIENT_MIN_RUN_MIN = 10                       # an orientation change must last >= this many minutes

RULE_NAMES_EN = {
    "angle_floor": "min_angle", "high_angle": "high_angle", "extension_velocity": "extension_vel",
    "displacement": "displacement", "angle_range": "angle_range_SdS", "base_or_braking": "base_braking_SdS",
}

# Column order in the final file (those not listed will be appended at the end, in alphabetical order)
COLUMN_ORDER = [
    # requested identification columns
    "ID", "day", "fase",
    # identification
    "Cohort", "Visit", "Recording", "ev", "status", "WB", "ext_filter", "flipped", "reason", "filter_check",
    # time
    "start_time", "end_time", "duration", "clock_start", "sts_time", "sts_hour",
    # main STS variables
    "transition_type", "sub_transition_type",
    "sts_vertical_displacement", "sts_max_vertical_vel", "sts_tilt_angle",
    "sts_max_flexion_vel", "sts_max_extension_vel", "sts_partial",
    # acceleration
    "sts_max_accv", "sts_max_accml", "sts_max_accap",
    "sts_range_accv", "sts_range_accml", "sts_range_accap",
    "sts_mean_accv", "sts_mean_accml", "sts_mean_accap",
    "sts_rms_accv", "sts_rms_accml", "sts_rms_accap",
    # gyroscope
    "sts_max_gyrov", "sts_max_gyroml", "sts_max_gyroap",
    "sts_range_gyrov", "sts_range_gyroml", "sts_range_gyroap",
    "sts_mean_gyrov", "sts_mean_gyroml", "sts_mean_gyroap",
    "sts_rms_gyrov", "sts_rms_gyroml", "sts_rms_gyroap",
    # smoothness/fluidity
    "sts_njs_accv", "sts_njs_accml", "sts_njs_accap",
    "sts_ldlj_accv", "sts_ldlj_accml", "sts_ldlj_accap",
    "sts_njs_gyrov", "sts_njs_gyroml", "sts_njs_gyroap",
    "sts_ldlj_gyrov", "sts_ldlj_gyroml", "sts_ldlj_gyroap",
    "sts_sparc_gyrov", "sts_sparc_gyroml", "sts_sparc_gyroap",
    "sts_imu_jerk", "sts_imu_jerk_nogyro",
    # other
    "sts_peak_index", "sts_duration",
]


# ============================= PHASE MAPPING =============================
def build_id_to_phase(phase_ids):
    """{ID: phase}; raises an error if an ID appears in more than one phase."""
    mapping = {}
    for phase, ids in phase_ids.items():
        for pid in ids:
            pid = str(pid).strip()
            if pid in mapping and mapping[pid] != phase:
                raise ValueError(f"ID {pid} appears in both {mapping[pid]} and {phase}")
            mapping[pid] = phase
    return mapping


ID_TO_PHASE = build_id_to_phase(PHASE_IDS)


def get_phase(pid):
    return ID_TO_PHASE.get(str(pid), "NA")


def day_number(day):
    """'T1_Day3' -> 3"""
    m = re.search(r"Day(\d+)$", str(day), re.IGNORECASE)
    return int(m.group(1)) if m else None


# ============================= FILE SEARCH =============================
def _natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


def find_patient(pid):
    """Returns (cohort, patient folder): PATIENT_PATHS override if present, else searches ROOT/*/<pid>."""
    if pid in PATIENT_PATHS:
        val = PATIENT_PATHS[pid]
        cohort, path = val if isinstance(val, (tuple, list)) else (None, val)
        path = os.path.normpath(os.path.expanduser(str(path)))
        if not os.path.isdir(path):
            return None, None
        return cohort or os.path.basename(os.path.dirname(path)), path
    hits = [p for p in glob.glob(os.path.join(ROOT, "*", pid)) if os.path.isdir(p)]
    if not hits:
        return None, None
    return os.path.basename(os.path.dirname(hits[0])), hits[0]


def find_days(pdir):
    """[(visit, day, data.mat path)] for all VISITS, in natural order."""
    out = []
    for p in glob.glob(os.path.join(pdir, "**", "data.mat"), recursive=True):
        day = os.path.basename(os.path.dirname(p))
        m = re.match(r"^(T\d+)_Day\d+$", day, re.IGNORECASE)
        if m and m.group(1).upper() in VISITS:
            out.append((m.group(1).upper(), day, p))
    return sorted(out, key=lambda x: (_natural_key(x[0]), _natural_key(x[1])))


# ============================= LOADING =============================
def _parse_start(s):
    try:
        return datetime.fromisoformat(str(s).strip())
    except Exception:
        return None


def load_day(mat_path, tm):
    """{recording: (acc in g (N,3), gyro in deg/s (N,3), fs, start datetime or None)} — columns [V, ML, AP]."""
    out = {}
    with open(mat_path, "rb") as f:
        is_v73 = f.read(8) == b"\x89HDF\r\n\x1a\n"
    if is_v73:
        with h5py.File(mat_path, "r") as m:
            tmg = m["data"][tm]
            for rec in sorted([k for k in tmg.keys() if k.startswith("Recording")], key=_natural_key):
                r = tmg[rec]
                lb = r["SU"]["LowerBack"]
                fs = FS_DEFAULT
                if "Fs" in lb:
                    fsn = lb["Fs"]
                    fs = float(np.array(fsn["Acc"] if isinstance(fsn, h5py.Group) else fsn).ravel()[0])
                start = "".join(chr(c) for c in np.array(r["StartDateTime"]).ravel() if c) \
                    if "StartDateTime" in r else None
                out[rec] = (np.array(lb["Acc"]), np.array(lb["Gyr"]), fs, _parse_start(start))
    else:
        m = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
        tmg = getattr(m["data"], tm)
        for rec in sorted([k for k in tmg._fieldnames if k.startswith("Recording")], key=_natural_key):
            r = getattr(tmg, rec)
            lb = r.SU.LowerBack
            fs = FS_DEFAULT
            if hasattr(lb, "Fs"):
                fsv = lb.Fs
                fs = float(np.ravel(fsv.Acc if hasattr(fsv, "_fieldnames") else fsv)[0])
            out[rec] = (np.array(lb.Acc), np.array(lb.Gyr), fs, _parse_start(getattr(r, "StartDateTime", None)))
    fixed = {}
    for rec, (acc, gyr, fs, start) in out.items():
        acc = acc.T if (acc.shape[0] == 3 and acc.shape[1] != 3) else acc
        gyr = gyr.T if (gyr.shape[0] == 3 and gyr.shape[1] != 3) else gyr
        fixed[rec] = (acc.astype(float), gyr.astype(float), fs, start)
    return fixed


# ============================= FILTERS =============================
def detect_gait_sequences(acc_g, gyr, fs):
    acc_ms2 = acc_g * 9.81
    data = pd.DataFrame({"acc_x": acc_ms2[:, 0], "acc_y": acc_ms2[:, 1], "acc_z": acc_ms2[:, 2],
                         "gyr_x": gyr[:, 0], "gyr_y": gyr[:, 1], "gyr_z": gyr[:, 2]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs = GsdIonescu().detect(data, sampling_rate_hz=fs).gs_list_
    return [(r.start / fs, r.end / fs) for r in gs.itertuples()]


def in_central_zone(start, end, gait_sequences, tail=TAIL_FRACTION):
    mid = (start + end) / 2
    return any(a + tail * (b - a) <= mid <= b - tail * (b - a) for a, b in gait_sequences)


def apply_wb_filter(events, gs):
    if events.empty or gs is None:
        return events, events.iloc[0:0]
    in_wb = np.array([in_central_zone(s, e, gs) for s, e in zip(events.start_time, events.end_time)], bool)
    return events[~in_wb], events[in_wb]


def apply_ext_filter(events, cfg):
    if events.empty:
        return events, events.iloc[0:0]
    kept, _, removed, _ = filter_transitions(events, cfg)
    return kept, removed


# ============================= ORIENTATION =============================
def orientation_mask(acc_g, fs, thr=ORIENT_G, min_run=ORIENT_MIN_RUN_MIN):
    """Boolean mask (per sample) of the stretches recorded with the sensor upside down.
    Per minute: +1 if mean V > thr, -1 if mean V < -thr, 0 otherwise (lying/ambiguous).
    Ambiguous minutes inherit the state of the surrounding stretch (forward, then backward fill);
    stretches shorter than min_run minutes are merged into the previous state."""
    n_m = int(round(60 * fs))
    m = len(acc_g) // n_m
    if m == 0:
        return np.zeros(len(acc_g), bool)
    mv = acc_g[:m * n_m, 0].reshape(m, n_m).mean(1)
    state = pd.Series(np.where(mv > thr, 1.0, np.where(mv < -thr, -1.0, np.nan)))
    state = state.ffill().bfill().fillna(1.0).to_numpy().copy()
    changed = True
    while changed:
        changed = False
        starts = np.r_[0, np.flatnonzero(np.diff(state)) + 1]
        ends = np.r_[starts[1:], len(state)]
        for a, b in zip(starts, ends):
            if b - a < min_run and len(starts) > 1:
                state[a:b] = state[a - 1] if a > 0 else state[b]
                changed = True
                break
    mask = np.repeat(state < 0, n_m)
    tail = len(acc_g) - len(mask)
    return np.r_[mask, np.full(tail, mask[-1] if len(mask) else False)]


def fix_orientation(acc_g, gyr, fs):
    """Flips V and ML (acc and gyro) where the sensor is upside down. Returns (acc, gyro, mask)."""
    mask = orientation_mask(acc_g, fs)
    if not mask.any():
        return acc_g, gyr, mask
    acc_g, gyr = acc_g.copy(), gyr.copy()
    for c in (0, 1):                      # V, ML
        acc_g[mask, c] *= -1
        gyr[mask, c] *= -1
    return acc_g, gyr, mask


def process_recording(acc_g, gyr, fs, cfg):
    """Returns sit2stand transitions with status / reason / filter_check columns."""
    ts = np.arange(len(acc_g)) / fs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds = JASDetector(verbose=False, **JAS_PARAMS).predict(accel=acc_g, gyro=gyr, time=ts)
    if preds is None or len(preds) == 0:
        return pd.DataFrame()

    sts = preds[preds["transition_type"] == "sit2stand"].sort_values("start_time").reset_index(drop=True)
    if sts.empty:
        return sts
    fails = filter_rules(sts, cfg)
    sts["filter_check"] = fails.apply(
        lambda r: "+".join(RULE_NAMES_EN.get(c, c) for c in fails.columns if r[c]), axis=1)
    sts["ev"] = np.arange(1, len(sts) + 1)

    gs = None
    if USE_WB_FILTER:
        try:
            gs = detect_gait_sequences(acc_g, gyr, fs)
        except Exception as ex:
            print(f"      [!] GsdIonescu failed ({type(ex).__name__}): WB filter not applied")

    empty = sts.iloc[0:0]
    if USE_WB_FILTER and ORDER == "WB_first":
        after_wb, wb = apply_wb_filter(sts, gs)
        kept, removed = apply_ext_filter(after_wb, cfg) if USE_EXT_FILTER else (after_wb, empty)
    else:
        kept, removed = apply_ext_filter(sts, cfg) if USE_EXT_FILTER else (sts, empty)
        kept, wb = apply_wb_filter(kept, gs) if USE_WB_FILTER else (kept, empty)

    # 0/1 flags, each filter evaluated independently on ALL events (1 = the event fails that filter)
    if USE_WB_FILTER and gs is not None:
        sts["WB"] = [int(in_central_zone(s, e, gs)) for s, e in zip(sts.start_time, sts.end_time)]
    else:
        sts["WB"] = 0
    if USE_EXT_FILTER:
        _, removed_all = apply_ext_filter(sts.drop(columns=["WB"]), cfg)
        sts["ext_filter"] = sts.ev.isin(removed_all.ev).astype(int)
    else:
        sts["ext_filter"] = 0

    sts["status"], sts["reason"] = "KEPT", ""
    sts.loc[sts.ev.isin(wb.ev), ["status", "reason"]] = ["REJECTED_WB", "walking_bout"]
    rem = sts.ev.isin(removed.ev)
    sts.loc[rem, "status"] = "REJECTED_FILTER"
    sts.loc[rem, "reason"] = sts.loc[rem, "filter_check"]
    return sts


# ============================= PLOTS (optional) =============================
def save_plot(acc_g, gyr, fs, ev, folder, tag):
    t0, t1 = ev["start_time"], ev["end_time"]
    i0, i1 = max(0, int((t0 - PLOT_PAD_S) * fs)), min(len(acc_g), int((t1 + PLOT_PAD_S) * fs))
    t = np.arange(i0, i1) / fs
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    a1.plot(t, acc_g[i0:i1, 0] * 9.81, color="black", label="Vertical Acc.")
    a2.plot(t, gyr[i0:i1, 1], color="tab:blue", label="Mediolateral Gyro")
    for a in (a1, a2):
        a.axvline(t0, color="red", ls="--", lw=2)
        a.axvline(t1, color="green", ls="--", lw=2)
        a.grid(True)
        a.legend(loc="upper right")
    a1.set_ylabel("Acc (m/s^2)")
    a2.set_ylabel("Gyro (°/s)")
    a2.set_xlabel("Time (s)")
    reason = f" | reason: {ev['reason']}" if ev["reason"] else ""
    a1.set_title(f"{tag} | {ev['status']}{reason} | t={t0:.1f}s\n"
                 f"tilt={ev['sts_tilt_angle']:.1f}°  flex={ev['sts_max_flexion_vel']:.0f}°/s  "
                 f"ext={ev['sts_max_extension_vel']:.0f}°/s  vdisp={ev['sts_vertical_displacement']:.2f} m",
                 color={"KEPT": "green", "REJECTED_WB": "orange"}.get(ev["status"], "red"),
                 fontweight="bold", fontsize=10)
    plt.tight_layout()
    folder.mkdir(parents=True, exist_ok=True)
    fname = f"STS_{tag}_{ev['status']}" + (f"_{ev['reason']}" if ev["reason"] else "") + ".png"
    fig.savefig(folder / fname, dpi=110)
    plt.close(fig)


# ============================= COLUMN ORDERING =============================
def order_columns(df):
    first = [c for c in COLUMN_ORDER if c in df.columns]
    rest = sorted(c for c in df.columns if c not in first)
    return df[first + rest]


# ============================= SAVING =============================
def write_excel(path, ev_df, sum_df, per_visit=None):
    try:
        with pd.ExcelWriter(path) as xw:
            ev_df.to_excel(xw, sheet_name="Events", index=False)
            kept = ev_df[ev_df.status == "KEPT"] if (not ev_df.empty and "status" in ev_df.columns) else ev_df
            kept.to_excel(xw, sheet_name="KEPT", index=False)
            sum_df.to_excel(xw, sheet_name="Summary_days", index=False)
            if per_visit is not None:
                per_visit.to_excel(xw, sheet_name="Summary_visit", index=False)
        return True
    except ImportError:
        print("   [!] openpyxl not installed: saving only CSVs (conda install openpyxl)")
        return False


def save_per_day(pid, fase, days, ev_df, sum_df, out_dir):
    """One file per day: <ID>_<day>_<fase>_events.csv (+ .xlsx)."""
    excel_ok = SAVE_EXCEL
    for visit, day, _ in days:
        tag = f"{pid}_{day}_{fase}"
        dnum = day_number(day)
        ev_d = ev_df[(ev_df.Visit == visit) & (ev_df.day == dnum)] if not ev_df.empty else ev_df
        sum_d = sum_df[(sum_df.Visit == visit) & (sum_df.day == dnum)] if not sum_df.empty else sum_df
        ev_d.to_csv(out_dir / f"{tag}_events.csv", index=False)
        if excel_ok:
            excel_ok = write_excel(out_dir / f"{tag}_events.xlsx", ev_d, sum_d)


# ============================= SINGLE PATIENT =============================
def process_patient(pid, cfg, skipped):
    fase = get_phase(pid)
    cohort, pdir = find_patient(pid)
    if pdir is None:
        where = PATIENT_PATHS.get(pid, f"{ROOT}/*/{pid}")
        print(f"[!] {pid}: folder not found ({where})")
        skipped.append(dict(ID=pid, fase=fase, reason="no_folder", detail=str(where)))
        return None
    out_dir = ROOT_OUT / fase / pid
    out_csv = out_dir / f"{pid}_{fase}_events.csv"
    out_sum = out_dir / f"{pid}_{fase}_summary_days.csv"
    force = (pid in ORIENT_FIX_IDS) if REPROCESS_FIX_IDS is True else \
        (pid in REPROCESS_FIX_IDS if isinstance(REPROCESS_FIX_IDS, (list, tuple, set)) else False)
    if SKIP_DONE and out_sum.exists() and not force:
        print(f"[=] {pid}: already processed, skipping ({out_sum})")
        return pd.read_csv(out_sum, dtype={"ID": str})

    days = find_days(pdir)
    fix = pid in ORIENT_FIX_IDS
    print(f"\n### {pid} ({cohort}, {fase}){' [orientation fix]' if fix else ''} — days found: {len(days)} "
          f"({', '.join(sorted(set(v for v, _, _ in days)))})")
    if not days:
        print(f"[!] {pid}: no days for visits {VISITS}, skipping")
        skipped.append(dict(ID=pid, fase=fase, reason="no_days_for_visits",
                            detail=f"no <visit>_DayN folder with data.mat for {VISITS} in {pdir}"))
        return None
    out_dir.mkdir(parents=True, exist_ok=True)

    events, summary, errors = [], [], []
    for visit, day, path in days:
        t0 = time.time()
        dnum = day_number(day)
        try:
            recs = load_day(path, TM_CHOICE)
        except Exception as ex:
            errors.append(dict(ID=pid, day=dnum, fase=fase, Visit=visit, Recording="",
                               error=f"{type(ex).__name__}: {ex}"))
            print(f"   {day}: [LOADING ERROR] {ex}")
            continue
        for rec, (acc_g, gyr, fs, start_dt) in recs.items():
            flipped = np.zeros(len(acc_g), bool)
            if fix:
                acc_g, gyr, flipped = fix_orientation(acc_g, gyr, fs)
            flipped_pct = round(float(flipped.mean()) * 100, 1) if len(flipped) else 0.0
            try:
                sts = process_recording(acc_g, gyr, fs, cfg)
            except Exception as ex:
                errors.append(dict(ID=pid, day=dnum, fase=fase, Visit=visit,
                                   Recording=rec, error=f"{type(ex).__name__}: {ex}"))
                print(f"   {day} {rec}: [ERROR] {ex}")
                traceback.print_exc()
                continue
            dur_h = round(len(acc_g) / fs / 3600, 2)
            counts = sts["status"].value_counts() if not sts.empty else pd.Series(dtype=int)
            summary.append(dict(ID=pid, day=dnum, fase=fase,
                                Cohort=cohort, Visit=visit, Recording=rec,
                                recording_start=str(start_dt) if start_dt else "", duration_h=dur_h,
                                flipped_pct=flipped_pct,
                                sts_jas=len(sts), rejected_wb=int(counts.get("REJECTED_WB", 0)),
                                rejected_filter=int(counts.get("REJECTED_FILTER", 0)),
                                kept=int(counts.get("KEPT", 0))))
            print(f"   {day} {rec}: {dur_h} h | JAS={len(sts)} | WB={counts.get('REJECTED_WB', 0)} | "
                  f"filter={counts.get('REJECTED_FILTER', 0)} | KEPT={counts.get('KEPT', 0)} "
                  f"| flipped {flipped_pct}% ({time.time() - t0:.0f} s)")
            if sts.empty:
                continue
            sts.insert(0, "Recording", rec)
            sts.insert(0, "Visit", visit)
            sts.insert(0, "Cohort", cohort)
            sts.insert(0, "fase", fase)
            sts.insert(0, "day", dnum)
            sts.insert(0, "ID", pid)
            mid = ((sts.start_time + sts.end_time) / 2 * fs).astype(int).clip(0, len(flipped) - 1)
            sts["flipped"] = flipped[mid.to_numpy()].astype(int)
            sts["duration"] = sts["sts_duration"] if "sts_duration" in sts.columns else sts.end_time - sts.start_time
            sts["clock_start"] = [(start_dt + timedelta(seconds=float(s))).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                                  if start_dt else "" for s in sts.start_time]
            events.append(sts)
            if SAVE_PLOTS:
                for _, ev in sts.iterrows():
                    save_plot(acc_g, gyr, fs, ev, out_dir / "plots",
                              f"{pid}_{day}_{fase}_{rec}_ev{int(ev['ev']):03d}")

    ev_df = pd.concat(events, ignore_index=True) if events else pd.DataFrame()
    if not ev_df.empty:
        num = lambda col: ev_df[col].astype(str).str.extract(r"(\d+)$", expand=False).astype(float)
        ev_df = (ev_df.assign(_v=num("Visit"), _r=num("Recording"))
                      .sort_values(["_v", "day", "_r", "start_time"])
                      .drop(columns=["_v", "_r"]))
        ev_df = order_columns(ev_df.reset_index(drop=True))
        if ONLY_KEPT:
            ev_df = ev_df[ev_df.status == "KEPT"].reset_index(drop=True)
    sum_df = pd.DataFrame(summary)
    per_visit = pd.DataFrame()
    if not sum_df.empty:
        per_day = sum_df.groupby(["Visit", "day"]).agg(duration_h=("duration_h", "sum"),
                                                       sts_jas=("sts_jas", "sum"), kept=("kept", "sum")).reset_index()
        per_visit = per_day.groupby("Visit").agg(days=("day", "nunique"), total_hours=("duration_h", "sum"),
                                                 kept_per_day_mean=("kept", "mean"),
                                                 kept_per_day_sd=("kept", "std"),
                                                 jas_per_day_mean=("sts_jas", "mean")).round(2).reset_index()
        per_visit.insert(0, "fase", fase)
        per_visit.insert(0, "ID", pid)

    # summary per patient is always saved (needed by SKIP_DONE and by the final summary)
    sum_df.to_csv(out_sum, index=False)
    if errors:
        pd.DataFrame(errors).to_csv(out_dir / f"{pid}_{fase}_errors.csv", index=False)

    if SAVE_PER_DAY:
        save_per_day(pid, fase, days, ev_df, sum_df, out_dir)

    if SAVE_PER_PATIENT:
        ev_df.to_csv(out_csv, index=False)
        if SAVE_EXCEL:
            write_excel(out_dir / f"{pid}_{fase}_events.xlsx", ev_df, sum_df, per_visit)

    if not per_visit.empty:
        print("   " + per_visit.to_string(index=False).replace("\n", "\n   "))
    print(f"   -> saved in {out_dir}")
    return sum_df


# ============================= MAIN =============================
def main():
    cfg = DEFAULT_THRESHOLDS if FILTER_CFG == "default" else IMPAIRED_THRESHOLDS
    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    order_txt = ("JAS -> WB -> ext filter" if (USE_WB_FILTER and ORDER == "WB_first")
                 else "JAS -> ext filter -> WB")

    ids = [str(p) for p in PATIENT_IDS] if PATIENT_IDS else list(ID_TO_PHASE.keys())
    missing = [p for p in ids if p not in ID_TO_PHASE]
    if missing:
        print(f"[!] IDs without phase (fase = 'NA'): {missing}")

    print(f"=== Visits: {VISITS} | filter: {FILTER_CFG} | order: {order_txt} ===")
    print(f"=== Phases: { {k: len(v) for k, v in PHASE_IDS.items()} } | IDs to process: {len(ids)} ===")
    print(f"Output: {ROOT_OUT}")

    t_start = time.time()
    summaries, skipped = [], []
    for pid in ids:
        try:
            s = process_patient(pid, cfg, skipped)
        except Exception as ex:
            print(f"[ERROR] {pid}: {type(ex).__name__}: {ex} -> moving to next ID")
            traceback.print_exc()
            skipped.append(dict(ID=pid, fase=get_phase(pid), reason="error",
                                detail=f"{type(ex).__name__}: {ex}"))
            continue
        if s is None:
            continue
        if s.empty:
            skipped.append(dict(ID=pid, fase=get_phase(pid), reason="no_valid_recordings",
                                detail="all days/recordings failed, see <ID>_<fase>_errors.csv"))
            continue
        summaries.append(s)

    skipped_df = pd.DataFrame(skipped, columns=["ID", "fase", "reason", "detail"])
    skipped_df.to_csv(ROOT_OUT / "skipped_IDs.csv", index=False)
    if not skipped_df.empty:
        print("\n" + "=" * 60)
        print(f"SKIPPED / FAILED IDs: {len(skipped_df)} (see {ROOT_OUT / 'skipped_IDs.csv'})")
        print("=" * 60)
        print(skipped_df[["ID", "fase", "reason"]].to_string(index=False))

    if summaries:
        allsum = pd.concat(summaries, ignore_index=True)
        if "flipped_pct" in allsum.columns:      # IDs processed before the orientation fix have no value
            allsum["flipped_pct"] = allsum["flipped_pct"].fillna(0.0)
        allsum.to_csv(ROOT_OUT / "summary_all_IDs.csv", index=False)
        per_day = allsum.groupby(["fase", "Cohort", "ID", "Visit", "day"]).kept.sum().reset_index()
        tab = per_day.groupby(["fase", "ID", "Visit"]).kept.agg(days="size", mean="mean",
                                                                      sd="std").round(1)
        print("\n" + "=" * 60)
        print("STS KEPT per day, per phase, patient and visit")
        print("=" * 60)
        print(tab.to_string())
    print(f"\nTotal time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()