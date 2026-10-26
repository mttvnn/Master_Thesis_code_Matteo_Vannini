"""
run_jas_extfilter.py — JAS detector + walking bout filter + external transition filter (ext_filter_jas.py).

ORDER = "WB_first"     (default)
    1. JAS predict
    2. walking bout filter on ALL sit2stand            -> REJECTED_WB
    3. external filter on the sit2stand left by the WB -> REJECTED_FILTER
    4. remaining                                       -> KEPT
ORDER = "filter_first" (previous order)
    1. JAS predict
    2. external filter                                 -> REJECTED_FILTER
    3. walking bout filter on the survivors            -> REJECTED_WB
    4. remaining                                       -> KEPT

A SINGLE CSV and plots of KEPT, REJECTED_WB and REJECTED_FILTER sit2stand transitions.
The column 'filter_check' always reports which external-filter rules each event fails
(also for events already rejected by the WB filter), so the two filters can be compared.
"""
import os
import re
import glob
import warnings
import traceback
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
BASE_DIR = "path/to/CVS/PFF/<site>"
PATIENT_IDS = []  # [] = all patients in BASE_DIR, or a list of IDs
TARGET_DAY = "T1_Day1"  # None = all days
TM_CHOICE = "TimeMeasure1"
FS_DEFAULT = 100.0

FILTER_CFG = "default"  # "default" or "impaired"
JAS_PARAMS = dict(val_min_vdisp=0.1, disp_threshold_m=0.1, val_max_vdisp=2.0, val_min_gap=0.5,
                  lp_cutoff_hz=2.0)

ORDER = "WB_first"  # "WB_first" = JAS -> WB -> ext filter ; "filter_first" = JAS -> ext filter -> WB
USE_WB_FILTER = True  # True = apply walking bout filter
TAIL_FRACTION = 0.25  # discard if the midpoint falls within the central 50% of a walking bout

SAVE_CSV = True  # True = save the single CSV file with the results
SAVE_PLOTS = True
PLOT_PAD_S = 5.0

patient_str = "_".join(PATIENT_IDS) if PATIENT_IDS else "all"
order_str = "WBfirst" if (USE_WB_FILTER and ORDER == "WB_first") else ""
OUT_DIR = (Path.home() / "tesi" / "risultati" /
           f"JAS_filtro_{FILTER_CFG}{'_WB' if USE_WB_FILTER else ''}{'_' + order_str if order_str else ''}_{patient_str}")

# Regole in italiano per non rompere il filtro esterno
RULE_NAMES = {
    "angle_floor": "angolo_min",
    "high_angle": "angolo_alto",
    "extension_velocity": "vel_estensione",
    "displacement": "spostamento",
    "angle_range": "range_angolo_SdS",
    "base_or_braking": "base_frenata_SdS",
}

# Traduzioni in inglese per l'export CSV e i grafici finali
RULE_NAMES_EN = {
    "angolo_min": "min_angle",
    "angolo_alto": "high_angle",
    "vel_estensione": "extension_vel",
    "spostamento": "displacement",
    "range_angolo_SdS": "angle_range_SdS",
    "base_frenata_SdS": "base_braking_SdS",
    "walking_bout": "walking_bout"
}


def translate_reason(m):
    if not isinstance(m, str) or not m:
        return ""
    return "+".join(RULE_NAMES_EN.get(p, p) for p in m.split("+"))


# ============================= LOADING =============================
def _natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


def find_patient_ids(base_dir):
    return sorted(d for d in os.listdir(base_dir)
                  if os.path.isdir(os.path.join(base_dir, d))
                  and glob.glob(os.path.join(base_dir, d, "**", "data.mat"), recursive=True))


def find_daily_mat_paths(base_dir, pid):
    out = []
    for p in glob.glob(os.path.join(base_dir, pid, "**", "*Day*", "data.mat"), recursive=True):
        day = os.path.basename(os.path.dirname(p))
        if TARGET_DAY is None or day == TARGET_DAY:
            out.append((day, p))
    return sorted(out, key=lambda x: _natural_key(x[0]))


def load_day(mat_path, tm):
    """{recording: (acc in g (N,3), gyro in deg/s (N,3), fs)} — columns [V, ML, AP]."""
    out = {}
    with open(mat_path, "rb") as f:
        is_v73 = f.read(8) == b"\x89HDF\r\n\x1a\n"
    if is_v73:
        with h5py.File(mat_path, "r") as m:
            tmg = m["data"][tm]
            for rec in sorted([k for k in tmg.keys() if k.startswith("Recording")], key=_natural_key):
                lb = tmg[rec]["SU"]["LowerBack"]
                fs = FS_DEFAULT
                if "Fs" in lb:
                    fsn = lb["Fs"]
                    fs = float(np.array(fsn["Acc"] if isinstance(fsn, h5py.Group) else fsn).ravel()[0])
                out[rec] = (np.array(lb["Acc"]), np.array(lb["Gyr"]), fs)
    else:
        m = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
        tmg = getattr(m["data"], tm)
        for rec in sorted([k for k in tmg._fieldnames if k.startswith("Recording")], key=_natural_key):
            lb = getattr(tmg, rec).SU.LowerBack
            fs = FS_DEFAULT
            if hasattr(lb, "Fs"):
                fsv = lb.Fs
                fs = float(np.ravel(fsv.Acc if hasattr(fsv, "_fieldnames") else fsv)[0])
            out[rec] = (np.array(lb.Acc), np.array(lb.Gyr), fs)
    fixed = {}
    for rec, (acc, gyr, fs) in out.items():
        acc = acc.T if (acc.shape[0] == 3 and acc.shape[1] != 3) else acc
        gyr = gyr.T if (gyr.shape[0] == 3 and gyr.shape[1] != 3) else gyr
        fixed[rec] = (acc.astype(float), gyr.astype(float), fs)
    return fixed


# ============================= REJECTION REASONS =============================
def add_reasons_and_numbering(preds, cfg):
    """Mantiene la colonna 'motivo' in italiano per compatibilità con ext_filter_jas.py"""
    preds = preds.copy()
    fails = filter_rules(preds, cfg)
    preds["motivo"] = fails.apply(lambda r: "+".join(RULE_NAMES.get(c, c) for c in fails.columns if r[c]), axis=1)
    preds["ev"] = np.nan
    s2st = preds.index[preds["transition_type"] == "sit2stand"]
    order = preds.loc[s2st].sort_values("start_time").index
    preds.loc[order, "ev"] = np.arange(1, len(order) + 1)
    return preds


# ============================= WALKING BOUT FILTER =============================
def detect_gait_sequences(acc_g, gyr, fs):
    """Walking bout detection with GsdIonescu."""
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


def apply_wb_filter(events, acc_g, gyr, fs):
    """Divides sit2stand transitions into (kept, rejected_WB)."""
    if events.empty:
        return events, events.iloc[0:0]
    try:
        gs = detect_gait_sequences(acc_g, gyr, fs)
    except Exception as ex:
        print(f"   [!] GsdIonescu failed ({type(ex).__name__}: {ex}): WB filter not applied")
        return events, events.iloc[0:0]
    in_wb = np.array([in_central_zone(s, e, gs) for s, e in zip(events.start_time, events.end_time)], bool)
    print(f"   walking bouts detected: {len(gs)}")
    return events[~in_wb].reset_index(drop=True), events[in_wb].reset_index(drop=True)


def apply_ext_filter(events, cfg):
    """Divides sit2stand transitions into (kept, rejected_filter) with ext_filter_jas.filter_transitions."""
    if events.empty:
        return events, events.iloc[0:0]
    kept, _, removed, _ = filter_transitions(events, cfg)
    return (kept.sort_values("start_time").reset_index(drop=True),
            removed.sort_values("start_time").reset_index(drop=True))


# ============================= PLOTS =============================
def save_plot(acc_g, gyr, fs, ev, tag, status, reason):
    t0, t1 = ev["start_time"], ev["end_time"]
    i0, i1 = max(0, int((t0 - PLOT_PAD_S) * fs)), min(len(acc_g), int((t1 + PLOT_PAD_S) * fs))
    t = np.arange(i0, i1) / fs
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    a1.plot(t, acc_g[i0:i1, 0] * 9.81, color="black", label="Vertical Acc.")
    a2.plot(t, gyr[i0:i1, 1], color="tab:blue", label="Mediolateral Gyro")
    for a in (a1, a2):
        a.axvline(t0, color="red", ls="--", lw=2, label="STS Start")
        a.axvline(t1, color="green", ls="--", lw=2, label="STS End")
        a.grid(True)
        a.legend(loc="upper right")
    a1.set_ylabel("Acc (m/s^2)")
    a2.set_ylabel("Gyro (°/s)")
    a2.set_xlabel("Time (s)")
    reason_txt = f" | reason: {reason}" if reason else ""
    check = ev.get("filter_check", "")
    check_txt = f" | ext filter: {check if check else 'pass'}" if status == "REJECTED_WB" else ""
    a1.set_title(f"{tag} | {status}{reason_txt}{check_txt} | t={t0:.1f}s\n"
                 f"tilt={ev.get('sts_tilt_angle', 0):.1f}°  flex={ev.get('sts_max_flexion_vel', 0):.0f}°/s  "
                 f"ext={ev.get('sts_max_extension_vel', 0):.0f}°/s  vdisp={ev.get('sts_vertical_displacement', 0):.2f} m",
                 color={"KEPT": "green", "REJECTED_WB": "orange"}.get(status, "red"),
                 fontweight="bold", fontsize=10)
    plt.tight_layout()
    folder = OUT_DIR / "plots"
    folder.mkdir(parents=True, exist_ok=True)
    fname = f"STS_{tag}_{status}" + (f"_{reason}" if reason else "") + ".png"
    fig.savefig(folder / fname, dpi=110)
    plt.close(fig)


# ============================= MAIN =============================
def main():
    cfg = DEFAULT_THRESHOLDS if FILTER_CFG == "default" else IMPAIRED_THRESHOLDS
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids = PATIENT_IDS or find_patient_ids(BASE_DIR)

    # Raccoglitore unico per il CSV finale
    all_transitions = []

    for pid in ids:
        for day, mat_path in find_daily_mat_paths(BASE_DIR, pid):
            print(f"--- {pid} {day}")
            for rec, (acc_g, gyr, fs) in load_day(mat_path, TM_CHOICE).items():
                try:
                    ts = np.arange(len(acc_g)) / fs
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        preds = JASDetector(verbose=False, **JAS_PARAMS).predict(accel=acc_g, gyro=gyr, time=ts)

                    if preds is None or len(preds) == 0:
                        print(f"   {rec}: no events found")
                        continue

                    # Motivi del filtro esterno calcolati per TUTTI gli eventi (nessuna eliminazione qui)
                    preds = add_reasons_and_numbering(preds, cfg)
                    sts = preds[preds["transition_type"] == "sit2stand"].sort_values("start_time").reset_index(drop=True)
                    sts["filter_check"] = sts["motivo"].apply(translate_reason)
                    n_jas = len(sts)

                    empty = sts.iloc[0:0]
                    if USE_WB_FILTER and ORDER == "WB_first":
                        # ---- 1) walking bout filter su tutte le sit2stand ----
                        after_wb, wb = apply_wb_filter(sts, acc_g, gyr, fs)
                        # ---- 2) filtro esterno sulle sopravvissute al WB ----
                        kept, removed = apply_ext_filter(after_wb, cfg)
                    else:
                        # ---- 1) filtro esterno su tutte le sit2stand ----
                        kept, removed = apply_ext_filter(sts, cfg)
                        # ---- 2) walking bout filter sulle sopravvissute ----
                        if USE_WB_FILTER:
                            kept, wb = apply_wb_filter(kept, acc_g, gyr, fs)
                        else:
                            wb = empty
                    if not wb.empty:
                        wb = wb.copy()
                        wb["motivo"] = "walking_bout"

                    print(f"   {rec}: JAS sit2stand = {n_jas}  |  rejected WB = {len(wb)}  |  "
                          f"rejected filter = {len(removed)}  |  KEPT = {len(kept)}")

                    for df, status in ((kept, "KEPT"), (wb, "REJECTED_WB"), (removed, "REJECTED_FILTER")):
                        if df.empty:
                            continue
                        df = df.copy()

                        # pd.to_numeric gestisce eventuali NaN senza far crashare il programma (usa Int64 pandas)
                        df["ev"] = pd.to_numeric(df["ev"], errors="coerce").astype("Int64")
                        df["status"] = status

                        # ESTRAZIONE DIRETTA DURATION (se JAS la produce), altrimenti la calcola
                        if "duration" in df.columns:
                            dur_col = df["duration"]
                        elif "sts_duration" in df.columns:
                            dur_col = df["sts_duration"]
                        elif "transition_duration" in df.columns:
                            dur_col = df["transition_duration"]
                        else:
                            dur_col = df["end_time"] - df["start_time"]
                        df["duration"] = dur_col

                        # Inserimento metadati (PatientID, Day, Recording)
                        for k, v in reversed([("PatientID", pid), ("Day", day), ("Recording", rec)]):
                            if k not in df.columns:
                                df.insert(0, k, v)

                        # Traduzione stringhe per CSV e grafici in inglese
                        df["reason"] = df["motivo"].apply(translate_reason)

                        # Generazione grafici
                        if SAVE_PLOTS:
                            for _, ev_row in df.iterrows():
                                ev_num = int(ev_row["ev"]) if pd.notna(ev_row["ev"]) else 0
                                save_plot(acc_g, gyr, fs, ev_row, f"{pid}_{day}_{rec}_ev{ev_num:03d}",
                                          status, ev_row["reason"])

                        # Selezione colonne finali per esportazione CSV
                        cols_csv = ["PatientID", "Day", "Recording", "ev", "status", "reason", "filter_check",
                                    "start_time", "duration"]
                        all_transitions.append(df[cols_csv])

                except Exception as e:
                    # Se questo singolo file dà un errore, il programma non si ferma, ma lo notifica!
                    print(f"   [!] ERRORE imprevisto durante l'elaborazione di {rec}: {e}")
                    traceback.print_exc()

    # Unione ed esportazione finale
    if all_transitions:
        all_df = pd.concat(all_transitions, ignore_index=True)
        all_df = all_df.sort_values(["PatientID", "Day", "Recording", "ev"]).reset_index(drop=True)

        if SAVE_CSV:
            csv_path = OUT_DIR / "STS_all_transitions.csv"
            all_df.to_csv(csv_path, index=False)
            print(f"\n=> FILE SALVATO CORRETTAMENTE IN: {csv_path}")
    else:
        all_df = pd.DataFrame()

    print("\n" + "=" * 50)
    order_txt = ("JAS -> WB -> ext filter" if (USE_WB_FILTER and ORDER == "WB_first")
                 else "JAS -> ext filter -> WB" if USE_WB_FILTER else "JAS -> ext filter")
    print(f" Filter: {FILTER_CFG} | Order: {order_txt}")

    if not all_df.empty:
        kept_count = len(all_df[all_df["status"] == "KEPT"])
        removed_count = len(all_df[all_df["status"] == "REJECTED_FILTER"])
        wb_count = len(all_df[all_df["status"] == "REJECTED_WB"])

        if USE_WB_FILTER:
            print(f" STS rejected by WB filter:       {wb_count}")
        print(f" STS rejected by external filter: {removed_count}")
        print(f" STS KEPT:                        {kept_count}")

        elim = all_df[all_df["status"] != "KEPT"]
        if len(elim):
            print(" Rejection reasons (one event can have multiple):")
            counts = elim["reason"].str.split("+").explode().value_counts()
            for k, v in counts.items():
                print(f"   {k:20s} {v}")

        wb_rows = all_df[all_df["status"] == "REJECTED_WB"]
        if len(wb_rows):
            n_also = (wb_rows["filter_check"] != "").sum()
            print(f" Of the {len(wb_rows)} WB-rejected events, {n_also} would also fail the external filter "
                  f"and {len(wb_rows) - n_also} would pass it")
    else:
        print(" No STS transitions found (or all failed).")

    print("=" * 50)


if __name__ == "__main__":
    main()