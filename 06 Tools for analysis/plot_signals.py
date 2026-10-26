"""
plot_imu.py — Grafico di accelerometro e giroscopio (3 assi) del sensore LowerBack da data.mat.

- Finestra temporale a scelta (T_START_S / T_END_S in secondi dall'inizio, None = tutta la registrazione)
- Asse x in orario reale (da Timestamp) oppure in secondi
- Riga di riferimento a 1 g e norma |Acc| per il controllo della calibrazione
- Linee verticali opzionali su istanti di interesse (es. eventi JAS)
- Sottocampionamento automatico per finestre lunghe (24 h)

Ambiente: conda activate jas
"""
import os
import sys
import glob
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================= PARAMETRI =============================
BASE_DIR = "path/to/CVS/PFF/<site>"
PATIENT_ID = "<ID>"
TARGET_DAY = "T3_Day1"
TM_CHOICE = "TimeMeasure1"
REC_CHOICE = "Recording1"
FS = 100.0

# Finestra da plottare, in secondi dall'inizio del Recording (None = tutto)
T_START_S = None     # es. 31080 ;  None per l'intera giornata
T_END_S = None     # es. 31120 ;  None per l'intera giornata

# Istanti (s dall'inizio) da evidenziare con linee verticali, es. inizio eventi JAS
MARK_TIMES_S = []

X_AS_CLOCK = True          # True = orario reale sull'asse x, False = secondi
MAX_POINTS = 200_000       # oltre questo numero di punti per asse si sottocampiona
SHOW_PLOT = True           # apre la finestra del grafico
SAVE_PLOT = True           # salva il PNG
OUT_DIR = Path.home() / "tesi" / "risultati" / "plot_segnali"

AXIS_LABELS = ["col1 (V)", "col2 (ML)", "col3 (AP)"]
AXIS_COLORS = ["tab:red", "tab:green", "tab:blue"]


# ============================= RICERCA E CARICAMENTO =============================
def find_data_mat(base_dir, patient_id, day):
    pattern = os.path.join(base_dir, patient_id, "**", day, "*.mat")
    hits = sorted(p for p in glob.glob(pattern, recursive=True)
                  if os.path.basename(p).lower() == "data.mat")
    if not hits:
        sys.exit(f"[!] Nessun data.mat trovato con: {pattern}")
    return hits[0]


def load_lowerback(mat_path, tm, rec):
    """Restituisce acc (N,3) in g, gyr (N,3) in deg/s, timestamp (N,) Unix o None, timezone."""
    with open(mat_path, "rb") as f:
        is_v73 = f.read(8) == b"\x89HDF\r\n\x1a\n"
    ts, tz = None, None
    if is_v73:
        with h5py.File(mat_path, "r") as m:
            r = m["data"][tm][rec]
            lb = r["SU"]["LowerBack"]
            acc, gyr = np.array(lb["Acc"]), np.array(lb["Gyr"])
            if "Timestamp" in lb:
                ts = np.array(lb["Timestamp"]).ravel()
            if "TimeZone" in r:
                tz = "".join(chr(c) for c in np.array(r["TimeZone"]).ravel() if c)
    else:
        m = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
        r = getattr(getattr(m["data"], tm), rec)
        lb = r.SU.LowerBack
        acc, gyr = np.array(lb.Acc), np.array(lb.Gyr)
        if hasattr(lb, "Timestamp"):
            ts = np.array(lb.Timestamp).ravel()
        if hasattr(r, "TimeZone"):
            tz = str(r.TimeZone)
    if acc.shape[0] == 3 and acc.shape[1] != 3:
        acc = acc.T
    if gyr.shape[0] == 3 and gyr.shape[1] != 3:
        gyr = gyr.T
    return acc, gyr, ts, tz


# ============================= MAIN =============================
def main():
    mat_path = find_data_mat(BASE_DIR, PATIENT_ID, TARGET_DAY)
    print(f"File: {mat_path}")
    acc, gyr, ts, tz_name = load_lowerback(mat_path, TM_CHOICE, REC_CHOICE)
    n = len(acc)
    t_rel = np.arange(n) / FS

    # --- finestra ---
    i0 = 0 if T_START_S is None else max(0, int(T_START_S * FS))
    i1 = n if T_END_S is None else min(n, int(T_END_S * FS))
    if i1 <= i0:
        sys.exit("[!] Finestra vuota: controlla T_START_S e T_END_S")
    step = max(1, (i1 - i0) // MAX_POINTS)
    sl = slice(i0, i1, step)
    print(f"Campioni nella finestra: {i1 - i0} (plotto 1 ogni {step})")

    a, g, tr = acc[sl], gyr[sl], t_rel[sl]
    norm = np.linalg.norm(a, axis=1)

    # --- asse x ---
    use_clock = X_AS_CLOCK and ts is not None
    if use_clock:
        tz = ZoneInfo(tz_name) if tz_name else None
        to_dt = lambda sec_unix: np.array([datetime.fromtimestamp(s, tz) for s in np.atleast_1d(sec_unix)])
        x = to_dt(ts[sl])
        t0_unix = ts[0]            # tempo Unix del campione 0
        mark_x = to_dt(np.array(MARK_TIMES_S) + t0_unix) if MARK_TIMES_S else []
        xlabel = f"Orario ({tz_name or 'UTC'})"
    else:
        x, mark_x, xlabel = tr, MARK_TIMES_S, "Tempo dall'inizio (s)"

    # --- figura ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    lw = 0.8 if step == 1 else 0.5

    for k in range(3):
        ax1.plot(x, a[:, k], color=AXIS_COLORS[k], lw=lw, label=AXIS_LABELS[k])
    ax1.plot(x, norm, color="black", lw=lw, alpha=0.7, label="|Acc|")
    ax1.axhline(1.0, color="gray", ls="--", lw=1, label="1 g")
    ax1.set_ylabel("Accelerazione (g)")
    ax1.legend(loc="upper right", ncol=5, fontsize=9)
    ax1.grid(True, alpha=0.4)

    for k in range(3):
        ax2.plot(x, g[:, k], color=AXIS_COLORS[k], lw=lw, label=AXIS_LABELS[k])
    ax2.set_ylabel("Velocità angolare (°/s)")
    ax2.set_xlabel(xlabel)
    ax2.legend(loc="upper right", ncol=3, fontsize=9)
    ax2.grid(True, alpha=0.4)

    for mx in mark_x:
        for ax in (ax1, ax2):
            ax.axvline(mx, color="magenta", ls=":", lw=1.5)

    if use_clock:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=ZoneInfo(tz_name) if tz_name else None))
        fig.autofmt_xdate()

    win_txt = "intera registrazione" if (T_START_S is None and T_END_S is None) \
        else f"{t_rel[i0]:.1f}–{t_rel[i1 - 1]:.1f} s"
    fig.suptitle(f"{PATIENT_ID} {TARGET_DAY} {REC_CHOICE} — LowerBack — {win_txt}", fontweight="bold")
    plt.tight_layout()

    if SAVE_PLOT:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        tag = "all" if (T_START_S is None and T_END_S is None) else f"{int(t_rel[i0])}-{int(t_rel[i1 - 1])}s"
        out = OUT_DIR / f"IMU_{PATIENT_ID}_{TARGET_DAY}_{REC_CHOICE}_{tag}.png"
        fig.savefig(out, dpi=150)
        print(f"Salvato: {out}")
    if SHOW_PLOT:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()