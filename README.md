# Real-world sit-to-stand transitions after proximal femoral fracture

Code used in the master's thesis **"Real-World Sit-to-Stand Transitions as a Novel Digital Mobility Outcome in Hip Fracture Recovery"**
(Matteo Vannini, MSc in Biomedical Engineering, Alma Mater Studiorum – Università di Bologna, in collaboration with research team of Bethanien Hospital Heidelberg).

The code covers the whole workflow of the thesis:

1. construction of a gold standard of sit-to-stand (SiSt) transitions from pressure insoles in the Mobilise-D Technical Validation Study (TVS);
2. evaluation and optimization of three lower-back SiSt detection algorithms against the gold standard;
3. application of the selected algorithm to the seven-day real-world recordings of patients with proximal femoral fracture (PFF) of the Mobilise-D Clinical Validation Study (CVS);
4. exploratory analyses of reliability, daily pattern, and relationship with clinical measures.

> **Data are not included.** The Mobilise-D data are subject to a data-use agreement and are not publicly available. The scripts require access to the original recordings and clinical data.

---

## Detection algorithms

| ID in the thesis | Implementation | Original method |
|---|---|---|
| Algorithm A | [Sit2StandPy](https://github.com/PfizerRD/sit2standpy) | Adamowicz et al. |
| Algorithm B | [KielMAT](https://github.com/neurogeriatricskiel/KielMAT) | Pham et al. |
| Algorithm C | JAS Detector | Albites-Sanabria et al. |

Algorithm C (JAS Detector) is an in-house detector distributed as a compiled package and is **not included** in this repository; it is available from its authors upon request. The post-processing filter used to adapt its output to patients with PFF is included (`03 Real World/ext_filter_jas.py`).

---

## Repository structure

| Folder / file | Content | Thesis section | Appendix |
|---|---|---|---|
| **01 Gold Standard** | | | |
| `Validation_SiSt_StSi.py` | SiSt and StSi events from the pressure insoles (TVS) | 4.4 | B.1 |
| `manual_mark_event.m`, `manual_correction.m` | Manual verification and correction of the events | 4.4 | C |
| **02 Validation** | | | |
| `Detection_SiSt_StSi.py` | Detection with Algorithms A, B, C (baseline and optimized), walking-bout filter | 4.3, 4.5–4.7 | B.3 |
| `LOSO_CV.py` | Leave-one-subject-out parameter optimization | 4.6 | B.4 |
| `Confusion_matrix.py` | IoU matching, performance metrics, Bland–Altman analysis | 4.5, 5.4 | B.2 |
| **03 Real World** | | | |
| `run_all_subject.py` | Detection on the CVS recordings (Algorithm C + walking-bout filter + PFF filter) | 4.7 | B.5 |
| `ext_filter_jas.py` | Post-processing filter of Algorithm C with PFF thresholds | 4.7 | B.6 |
| `filter_weartime.py` | Valid-day criteria | 4.8 | |
| `aggregate_daily.py`, `aggregate_weekly.py` | Daily and weekly SiSt outcomes | 4.8 | B.8 |
| `plot_distributions.py` | Distributions and weekday pattern of the outcomes | 5.6 | |
| **04 Exploratory analysis** | | | |
| `sensitivity_analysis.py` | Sensitivity scenarios, day-to-day ICC, hourly profile | 4.9, 5.7 | A.1, A.3 |
| `clinical_association.py` | Correlations with clinical measures, 5CRT groups, fracture type | 4.9, 5.8 | A.2 |
| `pvalue_phases.py` | Comparison between recovery phases (P3 vs P4) | 5.1–5.2, 5.6 | |
| **05 Tools for analysis** | | | |
| `flip_test.py` | Detection and correction of sensor orientation | 4.7 | B.7 |
| `inspect_mat.py`, `inspection_ids.py`, `plot_signals.py` | Inspection of recordings and signals | – | |
| `run_jas_with_filter.py` | Single-recording run of Algorithm C with the filters | 4.7 | |

---

## Workflow

```
01 Gold Standard  ->  02 Validation  ->  03 Real World  ->  04 Exploratory analysis
   (TVS events)        (algorithm         (CVS pipeline:      (reliability, daily
                        evaluation and     detection, valid    pattern, clinical
                        optimization)      days, outcomes)     measures)
```

The scripts in `01` and `02` were run in Google Colab; those in `03` and `04` on a Linux workstation. Before running a script, set the data paths at the top of the file (variables `BASE_DIR`, `BASE_DRIVE`, `ROOT`, `PATH_*`).

**Participant IDs.** No participant identifiers are stored in the code. The lists of participants (TVS cohort, recovery phases, sensor-orientation corrections, manual corrections of the gold standard, cases shown in the figures) are read from a local file `participants.json`, which is excluded from the repository. To run the scripts, copy `participants_template.json` to `participants.json`, fill it in, and set `PARTICIPANTS_FILE` at the top of each script.

---

## Requirements

- Python 3.11
- numpy, pandas, scipy, matplotlib, openpyxl, pingouin
- sit2standpy, kielmat
- MATLAB (only for the manual verification scripts in `01 Gold Standard`)

```bash
pip install numpy pandas scipy matplotlib openpyxl pingouin sit2standpy kielmat
```

---

## License

The code is released under the MIT License (see `LICENSE`). Sit2StandPy and KielMAT are subject to their own licenses.

## Contact

Matteo Vannini - For any question please contact me at matteo.vannini4@studio.unibo.it.
