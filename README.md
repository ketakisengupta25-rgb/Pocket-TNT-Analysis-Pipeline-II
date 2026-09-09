# Pocket TNT Analysis — Python

A Python implementation of the analysis pipeline for the **Pocket Think/No-Think (TNT)** intrusion-control task.

This repository translates the original R analysis into a documented and reproducible Python workflow suitable for research code sharing.

## What the analysis does

The pipeline:

1. checks the distribution of participants across **Counterbalance × randomizer** cells;
2. optionally samples an equal number of participants from each cell;
3. converts repetition-level TNT variables from wide to long format;
4. calculates mean intrusion frequency and standard errors across repetitions;
5. estimates participant-specific intrusion slopes for:
   - No-Think different-target repetitions 1–5;
   - No-Think different-target repetitions 6–10;
   - No-Think same-target repetitions 1–5;
   - No-Think same-target repetitions 6–10;
6. calculates the **Index of Intrusion Control (IIC)** for first and second task halves;
7. calculates **cue independence** and **corrected cue independence**;
8. calculates Cronbach's alpha for selected intrusion blocks;
9. calculates participant-level intrusion means and Pearson correlations;
10. fits one-factor CFA models for intrusion, slope, and IIC measures;
11. z-normalizes key variables within counterbalancing groups;
12. runs Shapiro-Wilk tests, one-sample Wilcoxon signed-rank tests, repeated-measures ANOVAs, and paired t-tests;
13. exports processed data, statistical tables, and figures.

## Repository structure

```text
pocket-tnt-analysis/
│
├── pocket_tnt_analysis.py
├── original_R_analysis.R
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── README.md
│
└── outputs/
    └── .gitkeep
```

## Installation

Python 3.10 or newer is recommended.

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the analysis

Place the participant-level dataset somewhere on your computer and run:

```bash
python pocket_tnt_analysis.py path/to/TNT_data.xlsx
```

By default, the analysis samples **12 participants per Counterbalance × randomizer cell**, reproducing the sampling step in the original R workflow.

To analyze all available participants instead:

```bash
python pocket_tnt_analysis.py path/to/TNT_data.xlsx --n-per-cell 0
```

To specify a different output directory:

```bash
python pocket_tnt_analysis.py path/to/TNT_data.xlsx --output-dir results
```

To reproduce the same random sample across runs, the script uses a fixed random seed (`2026`) by default. You can change it with:

```bash
python pocket_tnt_analysis.py path/to/TNT_data.xlsx --seed 123
```

## Expected data format

The participant-level dataset must contain:

```text
ParticipantID
Counterbalance
randomizer
```

and TNT repetition columns following this naming convention:

```text
rep01_NT_sametarget
rep02_NT_sametarget
...
rep10_NT_sametarget

rep01_NT_differenttarget
...
rep10_NT_differenttarget

rep01_T_differenttarget
...
```

The script detects repetition columns from the `rep##_Condition` naming pattern.

## Main derived variables

### Intrusion slopes

For each participant, the script fits:

```text
Intrusion ~ Repetition
```

within each selected task segment. The coefficient for repetition is saved as the participant's intrusion slope.

### Index of Intrusion Control

For each five-repetition block:

```text
IIC = ((Rep1 + Rep5) / 2) + Rep2 + Rep3 + Rep4 - 4 × Rep1
```

This is calculated separately for same-target and different-target No-Think conditions and for the first and second halves of the task.

### Cue independence

```text
Cue independence =
Rep06_NT_differenttarget - Rep06_NT_sametarget
```

Corrected cue independence additionally controls for the condition difference immediately before the target switch:

```text
Corrected cue independence =
(Rep06_diff - Rep06_same) - (Rep05_diff - Rep05_same)
```

## CFA / latent-factor note

The original R analysis used **lavaan** with:

```r
estimator = "MLR"
missing = "fiml"
std.lv = TRUE
```

The Python implementation uses **semopy**.

These approaches are conceptually similar but are **not guaranteed to be numerically identical**, particularly for robust standard errors, missing-data treatment, latent-variable scaling, fit indices, and regression factor scores.

For strict confirmatory replication of previously reported lavaan results, the original R/lavaan analysis should remain the reference implementation. The Python CFA is provided as a Python-native implementation of the same factor structure.

## Output

The script creates files such as:

```text
Pocket_TNT_Analyzed.xlsx
Pocket_TNT_Long.csv
TNT_summary.csv
reliability_cronbach_alpha.csv
correlation_matrix.csv
correlation_pvalues.csv
correlation_95CI.csv
cfa_fit_statistics.csv
univariate_tests.csv
cue_independence_rm_anova.csv
passive_decay_rm_anova.csv
rows_with_missing_values.xlsx
```

and PNG figures for the intrusion trajectory and key score distributions.

## Reproducibility note

The original R script sampled participants using `slice_sample(n = 12)` without an explicit random seed. The Python implementation exposes the seed and defaults to `2026`, making the sample reproducible.

If you need to reproduce an exact historical R sample, you must use the participant IDs from that historical sample or the exact R random-number state used at the time.

## Data privacy

Raw participant-level research data are deliberately excluded from this repository through `.gitignore`.

Do **not** upload identifiable, pseudonymized, or restricted participant data to a public GitHub repository unless the study's consent, ethics approval, and data-sharing plan explicitly permit it.

The `data/` directory contains only instructions for local use.
