# Pocket TNT Analysis

Analysis pipeline for the Pocket Think/No-Think (TNT) intrusion-control task.

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



Place the participant-level dataset somewhere on your computer and run:

python pocket_tnt_analysis.py path/to/TNT_data.xlsx


By default, the analysis samples 12 participants per Counterbalance × randomizer cell

To analyze all available participants instead:

python pocket_tnt_analysis.py path/to/TNT_data.xlsx --n-per-cell 0


To specify a different output directory:

python pocket_tnt_analysis.py path/to/TNT_data.xlsx --output-dir results



## Expected data format

The participant-level dataset must contain:

ParticipantID
Counterbalance
randomizer

and TNT repetition columns following this naming convention:

rep01_NT_sametarget
rep02_NT_sametarget

rep10_NT_sametarget

rep01_NT_differenttarget

rep10_NT_differenttarget

rep01_T_differenttarget


The script detects repetition columns from the `rep##_Condition` naming pattern.

## Main derived variables

### Intrusion slopes

For each participant, the script fits:

Intrusion ~ Repetition


within each selected task segment. The coefficient for repetition is saved as the participant's intrusion slope.

### Index of Intrusion Control

For each five-repetition block:


IIC = ((Rep1 + Rep5) / 2) + Rep2 + Rep3 + Rep4 - 4 × Rep1


This is calculated separately for same-target and different-target No-Think conditions and for the first and second halves of the task.

### Cue independence

Cue independence =
Rep06_NT_differenttarget - Rep06_NT_sametarget


Corrected cue independence additionally controls for the condition difference immediately before the target switch:

Corrected cue independence =
(Rep06_diff - Rep06_same) - (Rep05_diff - Rep05_same)


