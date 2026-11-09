# upv-arrangement-bias

Repository: https://github.com/cnnitipong/upv-arrangement-bias

This repository is the data and code source cited in the Data availability statement of the article.

Code and data for the study "Measurement bias in ultrasonic pulse velocity assessment of reinforced concrete: consequences for quality classification and repair decisions" (Results in Engineering, under review). The study analyses 462 ultrasonic pulse velocity (UPV) test points on 26 earthquake-affected reinforced concrete (RC) columns, paired with rebound-hammer records. It shows that the transducer arrangement (direct, semi-direct, indirect) shifts the measured velocity systematically, estimates and applies a correction for this arrangement bias, converts the corrected velocities into probabilistic quality grades, and quantifies how the bias changes quality classification and the resulting repair decisions.

## Quick start
```bash
pip install -r requirements.txt
bash run_all.sh          # about 2 minutes on one CPU core
```
Python 3.13 with the pinned versions in `requirements.txt` is what the package was verified with. NumPy 2.x is required (the scripts use `np.trapezoid`). The run stops at the first failing script; console output of each script goes to `outputs/log_<script>.txt` (git-ignored). `src/revision_analysis.py` needs only `data/master_dataset.csv` and can also be run alone: `python3 src/revision_analysis.py`.

## Repository layout
```
upv-arrangement-bias/
├── README.md  LICENSE  DATA_LICENSE  CITATION.cff  .gitignore
├── requirements.txt
├── run_all.sh              # runs all scripts in order
├── data/                   # CSV inputs + reference JSON results of the submitted analysis
│   ├── master_dataset.csv
│   ├── UPV_summary_all_locations.csv
│   ├── rebound_hammer_direction.csv
│   ├── Result_Mpa.csv
│   └── *_results.json
├── src/                    # scripts 01-06, revision_analysis.py, fix_figures_rineng.py
└── outputs/                # pre-generated results (overwritten on re-run)
    ├── *.csv, *.json, SUMMARY.md
    ├── tables/             # LaTeX tables
    ├── figures/
    ├── figures_grey/
    └── figures_rineng/
```

## Data columns
`data/master_dataset.csv` has one row per UPV test point (462 rows, 26 columns of the structure):

| Column | Meaning |
|---|---|
| `col` | Column identifier (as in the inspection report) |
| `point` | Test point ID within the column |
| `vel` | Measured pulse velocity, km/s (velocities were recorded directly; transit times were not stored) |
| `rebN` | Mean rebound number at the point |
| `mpa` | Rebound-hammer strength estimate, MPa |
| `priority` | Inspection priority 1-3, assigned at column level |
| `method` | Transducer arrangement: `Direct`, `Semi-Indirect` (this means semi-direct) or `Indirect` |
| `quality` | Velocity-based quality class: Excellent, Good, Medium, Doubtful |
| `gx`, `gy` | Column position on the structural grid (`gx` includes a 0.5 offset for B sub-bays; `gy` is the grid line A=1 ... H=8) |
| `face` | `single` or `pair`, derived from the point ID (one or two letters) |
| `bad` | 1 if `quality` is Doubtful, else 0 |

The file also carries helper columns: `vel_mean`, `ksc`, `quality_th` (original Thai grade), `key`, `bay`, `line`, `sub`. The other CSVs in `data/` are the original per-point extracts (Thai headers).

## What each script produces
Scripts run in this order; all paths are relative to the repository root.

| # | Script | Writes to `outputs/` |
|---|---|---|
| 1 | `01_build_master_and_core_stats.py` | `master_dataset.csv`, `desc_overall.csv`, `by_priority.csv`, `by_method.csv`, `column_level_stats.csv`, `anomalies.csv`, `concern_index.csv`, `results.json`, intermediate `_*.pkl/_*.npz` |
| 2 | `02_deep_analysis_and_grey_figures.py` | `deep_results.json` (arrangement offsets, bias correction, reclassification, bootstrap CIs), `figures_grey/g1..g9` |
| 3 | `03_extra_analyses.py` | `extra_results.json` (variance decomposition, odds/RR, leave-one-column-out, reliability), `_extra.npz` |
| 4 | `04_extra_figures.py` | `figures_grey/gA..gE` |
| 5 | `05_probabilistic_grading_and_repair_impact.py` | `novelty_results.json`, `probabilistic_grades.csv`, `figures_grey/p1*, p2*` |
| 6 | `06_confound_bounding.py` | `confound_results.json` |
| 7 | `revision_analysis.py` | `revision_results.json`, `SUMMARY.md`, `tables/*.tex`, `figures/` |
| 8 | `fix_figures_rineng.py` | `figures_rineng/` |

The `data/*.json` files are the reference results of the submitted analysis (used by script 8); scripts 2, 3, 5 and 6 regenerate their own copies in `outputs/`.

## Seeds
All randomness is seeded: `default_rng(42)` in script 01 and `revision_analysis.py` (whose Monte-Carlo step uses seed 20 and replicates the seed-11 stream of script 03), and `default_rng(20)` in script 05. A re-run from a fresh copy reproduces the reference JSON results; `master_dataset.csv` rebuilt by script 01 matches `data/master_dataset.csv` in all values (formatting of floats may differ).

## Not included
- The original inspection report (PDF, in Thai) and field photographs. The per-point CSVs in `data/` were extracted from the report beforehand. Without the photographs, script 02 skips the photo composite `g0_field.png` and prints a note.
- Manuscript LaTeX sources and superseded exploratory scripts.

## How to cite
Please cite the article (under review) and this repository; see `CITATION.cff` (GitHub's "Cite this repository" button reads it).

## Licence
Code: MIT (`LICENSE`). Data: CC BY 4.0 (`DATA_LICENSE`).
