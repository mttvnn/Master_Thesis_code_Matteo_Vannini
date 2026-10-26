"""
clinical_association.py — Exploratory relationship between the real-world sit-to-stand (SiSt) outcomes
and the clinical measures collected at the same CVS visit (thesis Sections 4.9 and 5.8).

Inputs
  ANALYSIS_DIR/weekly_main.csv     weekly outcomes per participant (from analysis_sensitivity_reliability.py)
  CLINICAL_XLSX                    CVS clinical dataset (visit T1)

Analyses
  1. Spearman correlation between the weekly outcomes (number of SiSt, median duration, median maximum flexion
     velocity, median maximum vertical acceleration) and the clinical measures (5CRT time per repetition, TUG,
     SPPB, 4MWT gait speed, LLFDI-FC, Short FES-I, VAS pain while walking)
  2. Pain while walking vs trunk kinematics (flexion / extension velocity at P10, P50, P90 and CoV)
  3. Capacity vs performance with the five-times chair rise test (5CRT):
       - 5CRT time per repetition (time / 5) vs median daily-life duration
       - groups: tertiles of 5CRT time among completers + participants unable to complete the test;
         Kruskal-Wallis across the four groups, Spearman correlation for the trend over the ordered groups,
         share of each group above the cohort median of SiSt per day
       - completed vs not completed, walking aid indoors yes vs no (Mann-Whitney U)
  4. Femoral neck vs extracapsular fractures: clinical capacity and weekly outcomes (Mann-Whitney U)
  5. Representative participants: the participant whose median duration is closest to the cohort median
     (among 5CRT completers), plus the participants listed in CASE_IDS

Outputs (CLINICAL_DIR)
  weekly_with_clinical.csv          weekly outcomes merged with the clinical measures (used by the figure scripts)
  correlations.csv                  rho, p-value and n for every clinical measure x outcome
  pain_vs_kinematics.csv            rho, p-value and n for pain while walking vs flexion / extension velocity
  crt_groups.csv                    median (IQR) per 5CRT group, Kruskal-Wallis and trend p-values
  group_comparisons.csv             completed vs not completed, walking aid indoors, fracture type
  representative_participants.csv   values of the representative participant and of CASE_IDS

Notes
  - a 5CRT time <= 0 s is treated as implausible: the participant is excluded from all 5CRT analyses
    (neither "completed" nor "not completed")
  - all analyses are exploratory and not corrected for multiple comparisons
Environment: conda activate jas ; run after analysis_sensitivity_reliability.py
"""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
 #============================= RUN SETTINGS =============================
VISITS = ["T1"]                    # CVS visits processed
FILTER_CFG = "default"             # JAS post-processing filter table ("default" = PFF thresholds, see ext_filter_jas.py)
ORIENT_FIX = False                 # True only if run_jas_per_ID.py writes to the _orientfix folder
FS_DEFAULT = 100.0                 # sampling frequency of the lower-back sensor (Hz)
ROOT_OUT = (Path.home() / "tesi" / "risultati" /
            f"JAS_per_ID_{'_'.join(VISITS)}_{FILTER_CFG}{'_orientfix' if ORIENT_FIX else ''}")
CLINICAL_XLSX = Path.home() / "tesi" / "Clinical Fold (Excel pff file)" / "PFF_dataset_v.7.2.xlsx"
ANALYSIS_DIR = ROOT_OUT / "analysis"            # analysis_sensitivity_reliability.py
CLINICAL_DIR = ANALYSIS_DIR / "clinical"        # clinical_association.py

# ============================= PARAMETERS =============================
VISIT = "t1"
OUTCOMES = {                                   # weekly column -> label
    "n_sts": "Number of SiSt per day",
    "duration_p50": "Duration, P50 (s)",
    "max_flexion_velocity_p50": "Max flexion velocity, P50 (deg/s)",
    "max_extension_velocity_p50": "Max extension velocity, P50 (deg/s)",
    "max_vertical_acc_p50": "Max vertical acceleration, P50 (m/s2)",
    "sparc_gyroml_p50": "SPARC, P50 (-)",
}
CORR_OUTCOMES = ["n_sts", "duration_p50", "max_flexion_velocity_p50", "max_vertical_acc_p50"]
CLINICAL = {                                   # clinical column -> label
    "crt_rep": "5CRT time per repetition (s)",
    "tugtime": "TUG time (s)",
    "total_sppb_score": "SPPB total score",
    "4mwt_max_speed": "4MWT gait speed (m/s)",
    "scaled_llfdi_func_tot": "LLFDI-FC",
    "short_fesi_score": "Short FES-I",
    "WLKPAN1N": "VAS pain while walking",
}
PAIN_COL = "WLKPAN1N"
PAIN_KINEMATICS = [f"{v}_{s}" for v in ("max_flexion_velocity", "max_extension_velocity")
                   for s in ("p10", "p50", "p90", "cov")]
CRT_REPS = 5
CRT_GROUPS = ["Fast", "Intermediate", "Slow", "Not completed"]
FRACTURE_COL = "fractype"
FRACTURES = {"Femoral neck fractures": "Femoral neck", "Extracapsular fractures": "Extracapsular"}
AID_INDOOR_COL = "aidindoor"
import json
PARTICIPANTS_FILE = "path/to/participants.json"   # local file with participant IDs, NOT in the repository (see participants_template.json)
with open(PARTICIPANTS_FILE) as _f:
    _P = json.load(_f)

CASE_IDS = _P["CASE_IDS"]                  # participants A and B described in Section 5.8


# ============================= HELPERS =============================
def med_iqr(x, dec=2):
    x = pd.Series(x).dropna()
    if not len(x):
        return ""
    q1, q3 = x.quantile(.25), x.quantile(.75)
    sep = " to " if q1 < 0 else "\u2013"           # avoid "-2.71--2.50" for negative values
    return f"{x.median():.{dec}f} ({q1:.{dec}f}{sep}{q3:.{dec}f})"


def fmt_p(p):
    if pd.isna(p):
        return ""
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def spearman(df, a, b):
    x = df[[a, b]].dropna()
    if len(x) < 3:
        return np.nan, np.nan, len(x)
    r, p = stats.spearmanr(x[a], x[b])
    return r, p, len(x)


def mwu(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if not len(a) or not len(b):
        return np.nan
    return stats.mannwhitneyu(a, b, alternative="two-sided").pvalue


# ============================= DATA =============================
def load_data(weekly_csv=ANALYSIS_DIR / "weekly_main.csv", clinical_xlsx=CLINICAL_XLSX):
    w = pd.read_csv(weekly_csv, dtype={"ID": str})
    clin = pd.read_excel(clinical_xlsx)
    clin = clin[clin["visit.number"].astype(str).str.lower() == VISIT].copy()
    clin["ID"] = clin["Local.Participant"].astype(str).str.replace(r"\.0$", "", regex=True)
    cols = ["ID", "5crtresult", "tugtime", "total_sppb_score", "4mwt_max_speed", "scaled_llfdi_func_tot",
            "short_fesi_score", PAIN_COL, AID_INDOOR_COL, FRACTURE_COL]
    cols = [c for c in cols if c in clin.columns]
    clin = clin[cols].drop_duplicates("ID")
    for c in cols:
        if c not in ("ID", FRACTURE_COL):
            clin[c] = pd.to_numeric(clin[c], errors="coerce")
    m = w.merge(clin, on="ID", how="left")
    n_bad = int((m["5crtresult"] <= 0).sum())
    if n_bad:
        print(f"[!] {n_bad} participant(s) with 5CRT time <= 0 s set to missing: "
              f"{m.loc[m['5crtresult'] <= 0, 'ID'].tolist()}")
    m["crt_invalid"] = m["5crtresult"] <= 0          # implausible value: excluded from all 5CRT analyses
    m.loc[m["crt_invalid"], "5crtresult"] = np.nan
    m["crt_rep"] = m["5crtresult"] / CRT_REPS
    m["crt_completed"] = m["5crtresult"].notna()
    return m


def assign_crt_groups(m):
    m = m.copy()
    comp = m["crt_completed"]
    m["crt_group"] = np.where(m["crt_invalid"], None, "Not completed")
    m.loc[comp, "crt_group"] = pd.qcut(m.loc[comp, "5crtresult"], 3,
                                       labels=CRT_GROUPS[:3]).astype(str)
    m["crt_group_rank"] = m["crt_group"].map({g: i for i, g in enumerate(CRT_GROUPS)})
    edges = m.loc[comp, "5crtresult"].quantile([1 / 3, 2 / 3]).round(1).tolist()
    n_not = int((~comp & ~m["crt_invalid"]).sum())
    print(f"5CRT tertile limits (s): {edges}  |  completed {int(comp.sum())}, not completed {n_not}, "
          f"excluded (implausible) {int(m['crt_invalid'].sum())}")
    return m


# ============================= ANALYSES =============================
def correlations(m):
    rows = []
    for c, cl in CLINICAL.items():
        for o in CORR_OUTCOMES:
            r, p, n = spearman(m, c, o)
            rows.append(dict(clinical=cl, outcome=OUTCOMES[o], rho=r, p_value=p, n=n,
                             cell=f"{r:+.2f}{'*' if p < 0.05 else ''}{'*' if p < 0.001 else ''} ({n})"
                             if pd.notna(r) else ""))
    return pd.DataFrame(rows)


def pain_kinematics(m):
    rows = []
    for o in PAIN_KINEMATICS:
        r, p, n = spearman(m, PAIN_COL, o)
        rows.append(dict(outcome=o, rho=r, p_value=p, n=n))
    return pd.DataFrame(rows)


def crt_analysis(m):
    rows = []
    for o in OUTCOMES:
        g = [m.loc[m["crt_group"] == k, o].dropna() for k in CRT_GROUPS]
        kw = stats.kruskal(*g).pvalue if all(len(x) for x in g) else np.nan
        r, p, _ = spearman(m, "crt_group_rank", o)
        row = dict(outcome=OUTCOMES[o], kruskal_p=kw, trend_rho=r, trend_p=p)
        for k, x in zip(CRT_GROUPS, g):
            row[k] = f"{med_iqr(x)} (n = {len(x)})" if len(x) else ""
        rows.append(row)
    tab = pd.DataFrame(rows)
    med_cohort = m["n_sts"].median()
    share = {k: round(float((m.loc[m["crt_group"] == k, "n_sts"] > med_cohort).mean() * 100), 1)
             for k in CRT_GROUPS}
    r, p, n = spearman(m, "crt_rep", "duration_p50")
    print(f"5CRT time per repetition vs median daily-life duration: rho = {r:.2f}, p = {fmt_p(p)}, n = {n}")
    print(f"5CRT time per repetition: median {m['crt_rep'].median():.2f} s | "
          f"daily-life median duration: median {m['duration_p50'].median():.2f} s")
    print(f"Share above the cohort median of SiSt per day ({med_cohort:.1f}): {share}")
    return tab


def group_comparisons(m):
    rows = []
    comp, notc = m[m["crt_completed"]], m[~m["crt_completed"] & ~m["crt_invalid"]]
    rows.append(dict(comparison="5CRT completed vs not completed", outcome=OUTCOMES["n_sts"],
                     group_1=f"{med_iqr(comp['n_sts'])} (n = {comp['n_sts'].notna().sum()})",
                     group_2=f"{med_iqr(notc['n_sts'])} (n = {notc['n_sts'].notna().sum()})",
                     p_value=mwu(comp["n_sts"], notc["n_sts"])))
    if AID_INDOOR_COL in m.columns:
        a1, a0 = m[m[AID_INDOOR_COL] == 1], m[m[AID_INDOOR_COL] == 0]
        rows.append(dict(comparison="Walking aid indoors: yes vs no", outcome=OUTCOMES["n_sts"],
                         group_1=f"{med_iqr(a1['n_sts'])} (n = {a1['n_sts'].notna().sum()})",
                         group_2=f"{med_iqr(a0['n_sts'])} (n = {a0['n_sts'].notna().sum()})",
                         p_value=mwu(a1["n_sts"], a0["n_sts"])))
    if FRACTURE_COL in m.columns:
        (k1, l1), (k2, l2) = list(FRACTURES.items())
        f1, f2 = m[m[FRACTURE_COL] == k1], m[m[FRACTURE_COL] == k2]
        for col, lab in [("5crtresult", "5CRT time (s)"), ("total_sppb_score", "SPPB total score")] + \
                        [(o, l) for o, l in OUTCOMES.items()]:
            rows.append(dict(comparison=f"{l1} vs {l2} fracture", outcome=lab,
                             group_1=f"{med_iqr(f1[col])} (n = {f1[col].notna().sum()})",
                             group_2=f"{med_iqr(f2[col])} (n = {f2[col].notna().sum()})",
                             p_value=mwu(f1[col], f2[col])))
        for ph, g in m.groupby("phase"):
            for o in ("n_sts", "duration_p50", "max_flexion_velocity_p50"):
                rows.append(dict(comparison=f"{l1} vs {l2} fracture, {ph}", outcome=OUTCOMES[o],
                                 group_1=med_iqr(g.loc[g[FRACTURE_COL] == k1, o]),
                                 group_2=med_iqr(g.loc[g[FRACTURE_COL] == k2, o]),
                                 p_value=mwu(g.loc[g[FRACTURE_COL] == k1, o], g.loc[g[FRACTURE_COL] == k2, o])))
    return pd.DataFrame(rows)


def representative(m):
    comp = m[m["crt_completed"]].dropna(subset=["duration_p50"]).copy()
    target = m["duration_p50"].median()
    comp["dist"] = (comp["duration_p50"] - target).abs()
    rep = comp.nsmallest(5, "dist").assign(role="closest to the cohort median duration")
    cases = m[m["ID"].isin(CASE_IDS)].assign(role="case described in the text")
    cols = ["role", "ID", "phase", "crt_rep", "total_sppb_score", AID_INDOOR_COL, PAIN_COL,
            "n_sts", "duration_p10", "duration_p50", "duration_p90"]
    cols = [c for c in cols if c in m.columns or c == "role"]
    return pd.concat([rep[cols], cases[cols]], ignore_index=True)


# ============================= MAIN =============================
def main():
    CLINICAL_DIR.mkdir(parents=True, exist_ok=True)
    m = assign_crt_groups(load_data())
    m.to_csv(CLINICAL_DIR / "weekly_with_clinical.csv", index=False)

    corr = correlations(m)
    corr.round(4).to_csv(CLINICAL_DIR / "correlations.csv", index=False)
    print("\nSpearman correlations (rho, * p < 0.05, ** p < 0.001, n)")
    print(corr.pivot(index="clinical", columns="outcome", values="cell").to_string())

    pk = pain_kinematics(m)
    pk.round(4).to_csv(CLINICAL_DIR / "pain_vs_kinematics.csv", index=False)
    print(f"\nPain while walking vs trunk kinematics: |rho| <= {pk['rho'].abs().max():.2f}, "
          f"p >= {pk['p_value'].min():.3f}")

    crt = crt_analysis(m)
    crt.assign(kruskal_p=crt["kruskal_p"].map(fmt_p), trend_p=crt["trend_p"].map(fmt_p),
               trend_rho=crt["trend_rho"].round(2)).to_csv(CLINICAL_DIR / "crt_groups.csv", index=False,
                                                           encoding="utf-8-sig")
    print("\n5CRT groups: median (IQR)")
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(crt.to_string(index=False))

    gc = group_comparisons(m)
    gc.assign(p_value=gc["p_value"].map(fmt_p)).to_csv(CLINICAL_DIR / "group_comparisons.csv", index=False,
                                                       encoding="utf-8-sig")
    print("\nGroup comparisons (Mann-Whitney U)")
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(gc.assign(p_value=gc["p_value"].map(fmt_p)).to_string(index=False))

    rep = representative(m)
    rep.round(2).to_csv(CLINICAL_DIR / "representative_participants.csv", index=False)
    print("\nRepresentative participants")
    print(rep.round(2).to_string(index=False))
    print(f"\n-> results saved in {CLINICAL_DIR}")


if __name__ == "__main__":
    main()