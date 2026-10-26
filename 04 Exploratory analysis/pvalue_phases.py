# =============================================================================
# P-VALUE P3 vs P4 
# Coorte = ID analyzed partecipant (P3 = 201, P4 = 87), visit T1
# Test: mean (SD) -> Welch t-test ; median (p25-p75) -> Mann-Whitney U
# =============================================================================
import pandas as pd
from scipy import stats

PATH_CLIN = "path/to/clinical_dataset.xlsx"   

import json
PARTICIPANTS_FILE = "path/to/participants.json"   # local file with participant IDs, NOT in the repository (see participants_template.json)
with open(PARTICIPANTS_FILE) as _f:
    _P = json.load(_f)

IDS_P3 = [int(x) for x in _P["P3_IDS"]]

IDS_P4 = [int(x) for x in _P["P4_IDS"]]

VARIABLES = {   # colonna nel dataset -> (etichetta in tabella, test)
    "age":                   ("Age, mean (SD)",                       "welch"),
    "height":                ("Height (cm), mean (SD)",               "welch"),
    "BMISCR1N":              ("Body mass index (BMI), mean (SD)",     "welch"),
    "total_sppb_score":      ("SPPB total score [0–12], mean (SD)",   "welch"),
    "5crtresult":            ("5CRT time (s), mean (SD)",             "welch"),
    "scaled_llfdi_func_tot": ("LLFDI-FC [0–100], median (p25–p75)",   "mannwhitney"),
    "short_fesi_score":      ("Short FES-I [7–28], median (p25–p75)", "mannwhitney"),
}

def fmt_p(p):
    """< 0.001, otherwise 3 decimali (2 decimals if p >= 0.1)."""
    if pd.isna(p):
        return "n.d."
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"

def describe(x, test):
    if test == "welch":
        return f"{x.mean():.1f} ({x.std():.1f})"
    return f"{x.median():.0f} ({x.quantile(.25):.0f}–{x.quantile(.75):.0f})"

# ---------------------------- selezione coorte ----------------------------
clin = pd.read_excel(PATH_CLIN)
t1 = clin[clin["visit.number"] == "t1"].copy()
t1["phase"] = None
t1.loc[t1["Local.Participant"].isin(IDS_P3), "phase"] = "P3"
t1.loc[t1["Local.Participant"].isin(IDS_P4), "phase"] = "P4"
t1 = t1[t1["phase"].notna()]
print(t1["phase"].value_counts())                          # P3 = 201, P4 = 87
print(pd.crosstab(t1["phase"], t1["pffsubtype"]))          # control: P3 = Post Discharge, P4 = Long Term Recovery

# ------------------------------- p-value -------------------------------
rows = []
for col, (label, test) in VARIABLES.items():
    a = pd.to_numeric(t1.loc[t1["phase"] == "P3", col], errors="coerce").dropna()
    b = pd.to_numeric(t1.loc[t1["phase"] == "P4", col], errors="coerce").dropna()
    if test == "welch":
        p = stats.ttest_ind(a, b, equal_var=False).pvalue
    else:
        p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    rows.append({"Variable": label, "Test": test,
                 "P3": describe(a, test), "n P3": len(a),
                 "P4": describe(b, test), "n P4": len(b),
                 "p": p, "p (table)": fmt_p(p)})

df_p = pd.DataFrame(rows)
print(df_p.to_string(index=False))
