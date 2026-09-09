"""
Pocket TNT Analysis Pipeline
============================

Python translation of the Pocket Think/No-Think (TNT) analysis workflow.

The pipeline:
1. Loads participant-level TNT data.
2. Optionally balances the sample within Counterbalance x randomizer cells.
3. Reshapes repetition-level intrusion data to long format.
4. Computes participant-specific intrusion slopes.
5. Computes Index of Intrusion Control (IIC) scores.
6. Computes cue-independence and corrected cue-independence effects.
7. Calculates reliability and correlations.
8. Fits three one-factor CFA models with semopy.
9. Z-normalizes key measures within counterbalancing groups.
10. Runs distributional checks, Wilcoxon tests, repeated-measures ANOVAs,
    and paired t-tests.
11. Saves analysis tables, figures, and the final participant-level dataset.

IMPORTANT SEM NOTE
------------------
The original R analysis used lavaan with estimator="MLR", missing="fiml",
and std.lv=TRUE. semopy does not reproduce lavaan's MLR + FIML workflow
identically. This script uses semopy for a close Python-native CFA workflow.
If exact numerical replication of the lavaan models is required, retain the
R/lavaan models as the confirmatory reference analysis.

Author: Ketaki Sengupta
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
from semopy import Model, calc_stats


# =============================================================================
# CONFIGURATION
# =============================================================================

REP_SAME_FIRST = [f"rep{i:02d}_NT_sametarget" for i in range(1, 6)]
REP_SAME_SECOND = [f"rep{i:02d}_NT_sametarget" for i in range(6, 11)]
REP_DIFF_FIRST = [f"rep{i:02d}_NT_differenttarget" for i in range(1, 6)]
REP_DIFF_SECOND = [f"rep{i:02d}_NT_differenttarget" for i in range(6, 11)]

REQUIRED_METADATA = ["ParticipantID", "Counterbalance", "randomizer"]

DEFAULT_OUTPUT_DIR = Path("outputs")


@dataclass
class AnalysisResults:
    """Container for the principal outputs of the analysis pipeline."""

    data: pd.DataFrame
    long_data: pd.DataFrame
    tnt_summary: pd.DataFrame
    reliability: pd.DataFrame
    correlation_matrix: pd.DataFrame
    correlation_pvalues: pd.DataFrame
    correlation_ci: pd.DataFrame
    cfa_fit: pd.DataFrame
    statistical_tests: pd.DataFrame


# =============================================================================
# INPUT / VALIDATION
# =============================================================================

def load_dataset(path: Path) -> pd.DataFrame:
    """Load an Excel or CSV participant-level TNT dataset."""
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(
        f"Unsupported file type: {suffix}. Use .xlsx, .xls, or .csv."
    )


def validate_columns(data: pd.DataFrame) -> None:
    """Check that essential metadata and TNT repetition columns are present."""
    missing_metadata = [c for c in REQUIRED_METADATA if c not in data.columns]

    rep_columns = [
        c for c in data.columns
        if c.startswith("rep") and "_" in c
    ]

    if missing_metadata:
        raise KeyError(
            "Missing required metadata columns: "
            + ", ".join(missing_metadata)
        )

    if not rep_columns:
        raise KeyError(
            "No TNT repetition columns were found. Expected names such as "
            "'rep01_NT_sametarget'."
        )


# =============================================================================
# SAMPLE BALANCING
# =============================================================================

def counterbalance_counts(data: pd.DataFrame) -> pd.DataFrame:
    """Count participants in each Counterbalance x randomizer cell."""
    return (
        data.groupby(["Counterbalance", "randomizer"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values("n")
        .reset_index(drop=True)
    )


def sample_within_cells(
    data: pd.DataFrame,
    n_per_cell: int = 12,
    random_state: int | None = 2026,
) -> pd.DataFrame:
    """
    Sample the same number of participants from each counterbalancing cell.

    This reproduces the intent of:
        group_by(Counterbalance, randomizer) %>%
        slice_sample(n = 12)

    A random seed is exposed explicitly for reproducibility.
    """
    counts = counterbalance_counts(data)
    undersized = counts.loc[counts["n"] < n_per_cell]

    if not undersized.empty:
        cells = undersized.to_dict(orient="records")
        raise ValueError(
            f"Some cells contain fewer than {n_per_cell} rows: {cells}"
        )

    sampled = (
        data.groupby(
            ["Counterbalance", "randomizer"],
            group_keys=False,
            dropna=False,
        )
        .sample(n=n_per_cell, random_state=random_state)
        .reset_index(drop=True)
    )

    return sampled


# =============================================================================
# LONG-FORM DATA AND DESCRIPTIVE SUMMARY
# =============================================================================

def to_long_format(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns such as rep01_NT_sametarget to long format.

    Creates:
        Rep       : repetition number
        Condition : TNT condition
        Intrusion : intrusion frequency/proportion
    """
    rep_columns = [
        c for c in data.columns
        if c.startswith("rep") and "_" in c
    ]

    id_columns = [c for c in data.columns if c not in rep_columns]

    long_data = data.melt(
        id_vars=id_columns,
        value_vars=rep_columns,
        var_name="rep_condition",
        value_name="Intrusion",
    )

    extracted = long_data["rep_condition"].str.extract(
        r"^rep([0-9]+)_(.*)$"
    )

    long_data["Rep"] = pd.to_numeric(extracted[0], errors="coerce")
    long_data["Condition"] = extracted[1]

    return long_data.drop(columns="rep_condition")


def summarize_intrusions(long_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate mean intrusion and standard error by condition and repetition.

    The original R calculation divided SD by sqrt(n()), where n() includes all
    rows in the group. Here SE uses the number of non-missing observations,
    which is the conventional denominator when the mean/SD omit missing values.
    """
    def summarize_group(group: pd.DataFrame) -> pd.Series:
        values = group["Intrusion"].dropna()
        n = len(values)
        mean = values.mean() if n else np.nan
        sd = values.std(ddof=1) if n > 1 else np.nan
        se = sd / np.sqrt(n) if n > 1 else np.nan

        return pd.Series({"Mean": mean, "SE": se, "N": n})

    return (
        long_data.groupby(["Condition", "Rep"], dropna=False)
        .apply(summarize_group, include_groups=False)
        .reset_index()
    )


def plot_intrusion_trajectory(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot mean intrusion frequency across repetitions for each condition."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for condition, group in summary.groupby("Condition"):
        group = group.sort_values("Rep")
        ax.errorbar(
            group["Rep"],
            group["Mean"],
            yerr=group["SE"],
            marker="o",
            label=condition,
            capsize=3,
        )

    ax.set_xlabel("Repetition")
    ax.set_ylabel("Mean Intrusion Frequency")
    ax.set_title("Intrusion Frequency by Repetition and Condition")
    ax.set_xticks(range(1, 11))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# =============================================================================
# PARTICIPANT-SPECIFIC INTRUSION SLOPES
# =============================================================================

def compute_participant_slopes(
    long_data: pd.DataFrame,
    condition: str,
    rep_start: int,
    rep_end: int,
    output_name: str,
) -> pd.DataFrame:
    """
    Fit Intrusion ~ Rep separately for each participant.

    Returns one slope per participant. Participants with fewer than two valid
    observations receive NaN because a slope cannot be estimated reliably.
    """
    subset = long_data.loc[
        (long_data["Condition"] == condition)
        & (long_data["Rep"].between(rep_start, rep_end))
    ].copy()

    rows = []

    for participant_id, participant_data in subset.groupby("ParticipantID"):
        valid = participant_data[["Rep", "Intrusion"]].dropna()

        if len(valid) < 2 or valid["Rep"].nunique() < 2:
            slope = np.nan
        else:
            slope, _, _, _, _ = stats.linregress(
                valid["Rep"],
                valid["Intrusion"],
            )

        rows.append(
            {
                "ParticipantID": participant_id,
                output_name: slope,
            }
        )

    return pd.DataFrame(rows)


def add_intrusion_slopes(
    data: pd.DataFrame,
    long_data: pd.DataFrame,
) -> pd.DataFrame:
    """Compute and merge all four participant-specific TNT slopes."""
    slope_specs = [
        ("NT_differenttarget", 1, 5, "slope1to5_diff"),
        ("NT_differenttarget", 6, 10, "slope6to10_diff"),
        ("NT_sametarget", 1, 5, "slope1to5_same"),
        ("NT_sametarget", 6, 10, "slope6to10_same"),
    ]

    output = data.copy()

    for condition, start, end, name in slope_specs:
        slopes = compute_participant_slopes(
            long_data,
            condition,
            start,
            end,
            name,
        )
        output = output.merge(slopes, on="ParticipantID", how="left")

    return output


# =============================================================================
# INDEX OF INTRUSION CONTROL (IIC)
# =============================================================================

def calculate_iic(block: pd.DataFrame) -> pd.Series:
    """
    Calculate the Index of Intrusion Control for a five-repetition block.

    Formula from the original analysis:
        ((rep1 + rep5) / 2) + rep2 + rep3 + rep4 - (4 * rep1)
    """
    if block.shape[1] != 5:
        raise ValueError("IIC requires exactly five repetition columns.")

    rep1, rep2, rep3, rep4, rep5 = [block.iloc[:, i] for i in range(5)]

    return (
        ((rep1 + rep5) / 2)
        + rep2
        + rep3
        + rep4
        - (4 * rep1)
    )


def add_iic_scores(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate first- and second-half IIC scores for same/different targets."""
    output = data.copy()

    output["IIC_1st_diff"] = calculate_iic(output[REP_DIFF_FIRST])
    output["IIC_2nd_diff"] = calculate_iic(output[REP_DIFF_SECOND])
    output["IIC_1st_same"] = calculate_iic(output[REP_SAME_FIRST])
    output["IIC_2nd_same"] = calculate_iic(output[REP_SAME_SECOND])

    return output


# =============================================================================
# CUE-INDEPENDENCE MEASURES
# =============================================================================

def add_cue_independence(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add cue-independence and corrected cue-independence measures.

    Cue independence:
        rep06 different-target - rep06 same-target

    Corrected cue independence:
        rep06 difference - rep05 difference
    """
    output = data.copy()

    rep05_difference = (
        output["rep05_NT_differenttarget"]
        - output["rep05_NT_sametarget"]
    )
    rep06_difference = (
        output["rep06_NT_differenttarget"]
        - output["rep06_NT_sametarget"]
    )

    output["Cue_independence"] = rep06_difference
    output["Corr_cue_independence"] = (
        rep06_difference - rep05_difference
    )

    return output


# =============================================================================
# RELIABILITY AND CORRELATIONS
# =============================================================================

def cronbach_alpha_table(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate Cronbach's alpha for the repetition sets in the R workflow."""
    blocks = {
        "same_target_1sthalf": REP_SAME_FIRST,
        "diff_target_1sthalf": REP_DIFF_FIRST,
        "diff_target_2ndhalf": REP_DIFF_SECOND,
    }

    rows = []

    for name, columns in blocks.items():
        alpha, ci = pg.cronbach_alpha(data=data[columns])
        rows.append(
            {
                "Scale": name,
                "Cronbach_alpha": alpha,
                "CI95_low": ci[0],
                "CI95_high": ci[1],
            }
        )

    return pd.DataFrame(rows)


def add_intrusion_means(data: pd.DataFrame) -> pd.DataFrame:
    """Compute participant-level mean intrusion scores for each block."""
    output = data.copy()

    output["intrusion_same_1st"] = output[REP_SAME_FIRST].mean(
        axis=1, skipna=True
    )
    output["intrusion_diff_1st"] = output[REP_DIFF_FIRST].mean(
        axis=1, skipna=True
    )
    output["intrusion_diff_2nd"] = output[REP_DIFF_SECOND].mean(
        axis=1, skipna=True
    )
    output["intrusion_same_2nd"] = output[REP_SAME_SECOND].mean(
        axis=1, skipna=True
    )

    return output


def correlation_with_ci(
    data: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pairwise Pearson correlations with p-values and Fisher-z 95% CIs.

    This approximates the information produced by psych::corr.test(
    ..., use="pairwise", method="pearson", adjust="none").
    """
    r_matrix = pd.DataFrame(
        np.eye(len(columns)),
        index=columns,
        columns=columns,
        dtype=float,
    )
    p_matrix = pd.DataFrame(
        np.zeros((len(columns), len(columns))),
        index=columns,
        columns=columns,
        dtype=float,
    )
    ci_matrix = pd.DataFrame(
        "",
        index=columns,
        columns=columns,
        dtype=object,
    )

    for i, col_a in enumerate(columns):
        for j, col_b in enumerate(columns):
            if j < i:
                continue

            pair = data[[col_a, col_b]].dropna()
            n = len(pair)

            if col_a == col_b:
                r, p = 1.0, 0.0
                ci = (1.0, 1.0)
            elif n < 3:
                r, p = np.nan, np.nan
                ci = (np.nan, np.nan)
            else:
                r, p = stats.pearsonr(pair[col_a], pair[col_b])

                if n > 3 and np.isfinite(r) and abs(r) < 1:
                    z = np.arctanh(r)
                    se = 1 / np.sqrt(n - 3)
                    z_low = z - 1.96 * se
                    z_high = z + 1.96 * se
                    ci = (np.tanh(z_low), np.tanh(z_high))
                else:
                    ci = (np.nan, np.nan)

            r_matrix.loc[col_a, col_b] = r
            r_matrix.loc[col_b, col_a] = r
            p_matrix.loc[col_a, col_b] = p
            p_matrix.loc[col_b, col_a] = p

            ci_text = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if np.all(
                np.isfinite(ci)
            ) else ""
            ci_matrix.loc[col_a, col_b] = ci_text
            ci_matrix.loc[col_b, col_a] = ci_text

    return r_matrix, p_matrix, ci_matrix


# =============================================================================
# CONFIRMATORY FACTOR ANALYSIS
# =============================================================================

def fit_one_factor_cfa(
    data: pd.DataFrame,
    factor_name: str,
    indicators: list[str],
) -> tuple[pd.Series, pd.DataFrame, object]:
    """
    Fit a one-factor CFA with semopy and return participant factor scores.

    Notes
    -----
    The original R workflow used lavaan MLR + FIML + std.lv=TRUE.
    semopy is used here as a Python-native approximation and may not produce
    identical estimates, standard errors, fit measures, or factor scores.

    Missing-data handling also differs from the lavaan implementation.
    """
    model_description = (
        f"{factor_name} =~ " + " + ".join(indicators)
    )

    model_data = data[indicators].copy()

    # semopy requires enough usable information to fit the model. For the
    # Python-native implementation, factor scoring is performed on complete
    # indicator rows and missing scores are restored afterward.
    complete_mask = model_data.notna().all(axis=1)
    complete_data = model_data.loc[complete_mask].copy()

    if len(complete_data) < 5:
        warnings.warn(
            f"Too few complete rows to fit {factor_name}. "
            "Returning missing factor scores."
        )
        scores = pd.Series(np.nan, index=data.index, name=factor_name)
        return scores, pd.DataFrame(), None

    model = Model(model_description)
    model.fit(complete_data)

    factor_scores_complete = model.predict_factors(complete_data)[factor_name]

    scores = pd.Series(np.nan, index=data.index, name=factor_name)
    scores.loc[complete_mask] = factor_scores_complete.to_numpy()

    fit = calc_stats(model).T.reset_index()
    fit = fit.rename(columns={"index": "Fit_measure"})

    return scores, fit, model


def add_latent_factors(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Fit and add intrusion, slope, and IIC latent factor scores."""
    output = data.copy()

    models = {
        "Latent_intrusion_factor": [
            "intrusion_same_1st",
            "intrusion_diff_1st",
            "intrusion_diff_2nd",
        ],
        "Latent_slope_factor": [
            "slope1to5_same",
            "slope1to5_diff",
            "slope6to10_diff",
        ],
        "Latent_IIC_factor": [
            "IIC_1st_same",
            "IIC_1st_diff",
            "IIC_2nd_diff",
        ],
    }

    fit_tables = []
    fitted_models = {}

    for output_column, indicators in models.items():
        factor_name = output_column.removeprefix("Latent_").removesuffix(
            "_factor"
        ).title().replace("_", "") + "_Factor"

        scores, fit, model = fit_one_factor_cfa(
            output,
            factor_name,
            indicators,
        )

        output[output_column] = scores

        if not fit.empty:
            fit.insert(0, "Model", output_column)
            fit_tables.append(fit)

        fitted_models[output_column] = model

    combined_fit = (
        pd.concat(fit_tables, ignore_index=True)
        if fit_tables
        else pd.DataFrame()
    )

    return output, combined_fit, fitted_models


# =============================================================================
# WITHIN-COUNTERBALANCE Z-NORMALIZATION
# =============================================================================

def zscore_within_group(
    data: pd.DataFrame,
    column: str,
    group_column: str = "Counterbalance",
) -> pd.Series:
    """Z-score a variable within each counterbalancing group."""
    def standardize(series: pd.Series) -> pd.Series:
        mean = series.mean(skipna=True)
        sd = series.std(skipna=True, ddof=1)

        if pd.isna(sd) or sd == 0:
            return pd.Series(np.nan, index=series.index)

        return (series - mean) / sd

    return data.groupby(group_column)[column].transform(standardize)


def add_normalized_scores(data: pd.DataFrame) -> pd.DataFrame:
    """Add all within-counterbalance standardized variables."""
    output = data.copy()

    mappings = {
        "Cue_independence": "z_cue_independence",
        "Corr_cue_independence": "z_corr_cue_independence",
        "Latent_intrusion_factor": "z_latent_intrusion",
        "Latent_slope_factor": "z_latent_slope",
        "Latent_IIC_factor": "z_latent_IIC",
    }

    for source, destination in mappings.items():
        output[destination] = zscore_within_group(output, source)

    return output


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def safe_shapiro(series: pd.Series) -> tuple[float, float]:
    """Run Shapiro-Wilk after dropping missing values."""
    values = series.dropna()

    if len(values) < 3:
        return np.nan, np.nan

    statistic, p_value = stats.shapiro(values)
    return statistic, p_value


def safe_wilcoxon(
    series: pd.Series,
    alternative: str,
) -> tuple[float, float]:
    """One-sample Wilcoxon signed-rank test against zero."""
    values = series.dropna()

    if len(values) == 0 or np.allclose(values, 0):
        return np.nan, np.nan

    result = stats.wilcoxon(
        values,
        alternative=alternative,
        zero_method="wilcox",
    )
    return result.statistic, result.pvalue


def paired_ttest(
    data: pd.DataFrame,
    column_a: str,
    column_b: str,
) -> tuple[float, float, int]:
    """Two-sided paired-samples t-test using complete pairs."""
    pairs = data[[column_a, column_b]].dropna()

    if len(pairs) < 2:
        return np.nan, np.nan, len(pairs)

    result = stats.ttest_rel(
        pairs[column_a],
        pairs[column_b],
        alternative="two-sided",
    )

    return result.statistic, result.pvalue, len(pairs)


def run_univariate_tests(data: pd.DataFrame) -> pd.DataFrame:
    """Run the Shapiro, Wilcoxon, and paired tests from the R workflow."""
    rows = []

    shapiro_variables = [
        "Cue_independence",
        "Corr_cue_independence",
        "z_latent_intrusion",
        "Latent_slope_factor",
        "z_latent_IIC",
    ]

    for variable in shapiro_variables:
        if variable in data.columns:
            statistic, p = safe_shapiro(data[variable])
            rows.append(
                {
                    "Test": "Shapiro-Wilk",
                    "Variable": variable,
                    "Statistic": statistic,
                    "p_value": p,
                    "Alternative": "",
                    "N": data[variable].notna().sum(),
                }
            )

    wilcoxon_specs = [
        ("Cue_independence", "greater"),
        ("Corr_cue_independence", "greater"),
        ("intrusion_diff_1st", "greater"),
        ("slope1to5_diff", "less"),
        ("z_latent_IIC", "less"),
    ]

    for variable, alternative in wilcoxon_specs:
        statistic, p = safe_wilcoxon(data[variable], alternative)
        rows.append(
            {
                "Test": "Wilcoxon signed-rank",
                "Variable": variable,
                "Statistic": statistic,
                "p_value": p,
                "Alternative": alternative,
                "N": data[variable].notna().sum(),
            }
        )

    paired_specs = [
        (
            "rep05_NT_differenttarget",
            "rep05_NT_sametarget",
            "rep05: different vs same",
        ),
        (
            "rep06_NT_differenttarget",
            "rep06_NT_sametarget",
            "rep06: different vs same",
        ),
        (
            "IIC_1st_diff",
            "IIC_2nd_diff",
            "IIC: first vs second half",
        ),
    ]

    for col_a, col_b, label in paired_specs:
        statistic, p, n = paired_ttest(data, col_a, col_b)
        rows.append(
            {
                "Test": "Paired t-test",
                "Variable": label,
                "Statistic": statistic,
                "p_value": p,
                "Alternative": "two-sided",
                "N": n,
            }
        )

    return pd.DataFrame(rows)


def run_repeated_measures_anovas(
    long_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the two repeated-measures analyses from the original R workflow.

    1. Cue-independent intrusion reduction:
       Condition (NT_differenttarget, NT_sametarget) x Rep (5, 6)

    2. Passive-decay check:
       T_differenttarget across Rep 1 vs 6
    """
    cue_data = long_data.loc[
        long_data["Condition"].isin(
            ["NT_differenttarget", "NT_sametarget"]
        )
        & long_data["Rep"].isin([5, 6]),
        ["ParticipantID", "Condition", "Rep", "Intrusion"],
    ].dropna()

    cue_anova = pg.rm_anova(
        data=cue_data,
        dv="Intrusion",
        within=["Condition", "Rep"],
        subject="ParticipantID",
        detailed=True,
    )

    passive_data = long_data.loc[
        (long_data["Condition"] == "T_differenttarget")
        & long_data["Rep"].isin([1, 6]),
        ["ParticipantID", "Rep", "Intrusion"],
    ].dropna()

    passive_anova = pg.rm_anova(
        data=passive_data,
        dv="Intrusion",
        within="Rep",
        subject="ParticipantID",
        detailed=True,
    )

    return cue_anova, passive_anova


# =============================================================================
# DISTRIBUTION PLOTS
# =============================================================================

def plot_distribution(
    data: pd.DataFrame,
    column: str,
    title: str,
    x_label: str,
    output_path: Path,
) -> None:
    """Save a histogram with a vertical reference line at zero."""
    values = data[column].dropna()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(values, bins=20, edgecolor="black")
    ax.axvline(0, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_distribution_plots(
    data: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save the distribution plots represented in the original R script."""
    plot_specs = [
        (
            "Cue_independence",
            "Distribution of Cue Independence Values",
            "Cue Independence",
            "cue_independence_distribution.png",
        ),
        (
            "Corr_cue_independence",
            "Distribution of Corrected Cue Independence Values",
            "Corrected Cue Independence",
            "corrected_cue_independence_distribution.png",
        ),
        (
            "Latent_intrusion_factor",
            "Distribution of Latent Intrusion Factor",
            "Latent Intrusion Factor",
            "latent_intrusion_distribution.png",
        ),
        (
            "Latent_slope_factor",
            "Distribution of Latent Slope Factor",
            "Latent Slope Factor",
            "latent_slope_distribution.png",
        ),
        (
            "z_latent_IIC",
            "Distribution of Latent IIC Factor",
            "Standardized Latent IIC Factor",
            "latent_iic_distribution.png",
        ),
    ]

    for column, title, label, filename in plot_specs:
        if column in data.columns:
            plot_distribution(
                data,
                column,
                title,
                label,
                output_dir / filename,
            )


# =============================================================================
# COMPLETE ANALYSIS PIPELINE
# =============================================================================

def run_analysis(
    input_file: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sample_n_per_cell: int | None = 12,
    random_state: int | None = 2026,
) -> AnalysisResults:
    """Run the complete Pocket TNT analysis pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(input_file)
    validate_columns(data)

    counts_before = counterbalance_counts(data)
    counts_before.to_csv(
        output_dir / "counterbalance_counts_before_sampling.csv",
        index=False,
    )

    if sample_n_per_cell is not None:
        data = sample_within_cells(
            data,
            n_per_cell=sample_n_per_cell,
            random_state=random_state,
        )

    long_data = to_long_format(data)
    summary = summarize_intrusions(long_data)

    plot_intrusion_trajectory(
        summary,
        output_dir / "intrusion_frequency_by_repetition.png",
    )

    data = add_intrusion_slopes(data, long_data)
    data = add_iic_scores(data)
    data = add_cue_independence(data)

    # Save rows containing at least one missing value, matching the diagnostic
    # purpose of NA_DATA in the R workflow.
    na_data = data.loc[data.isna().any(axis=1)]
    na_data.to_excel(output_dir / "rows_with_missing_values.xlsx", index=False)

    reliability = cronbach_alpha_table(data)

    data = add_intrusion_means(data)

    correlation_columns = [
        "intrusion_same_1st",
        "intrusion_diff_1st",
        "intrusion_diff_2nd",
    ]

    corr_r, corr_p, corr_ci = correlation_with_ci(
        data,
        correlation_columns,
    )

    data, cfa_fit, _ = add_latent_factors(data)
    data = add_normalized_scores(data)

    statistical_tests = run_univariate_tests(data)
    cue_anova, passive_anova = run_repeated_measures_anovas(long_data)

    save_distribution_plots(data, output_dir)

    # -------------------------------------------------------------------------
    # EXPORT RESULTS
    # -------------------------------------------------------------------------

    data.to_excel(
        output_dir / "Pocket_TNT_Analyzed.xlsx",
        index=False,
    )
    long_data.to_csv(
        output_dir / "Pocket_TNT_Long.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "TNT_summary.csv",
        index=False,
    )
    reliability.to_csv(
        output_dir / "reliability_cronbach_alpha.csv",
        index=False,
    )
    corr_r.to_csv(
        output_dir / "correlation_matrix.csv",
    )
    corr_p.to_csv(
        output_dir / "correlation_pvalues.csv",
    )
    corr_ci.to_csv(
        output_dir / "correlation_95CI.csv",
    )
    cfa_fit.to_csv(
        output_dir / "cfa_fit_statistics.csv",
        index=False,
    )
    statistical_tests.to_csv(
        output_dir / "univariate_tests.csv",
        index=False,
    )
    cue_anova.to_csv(
        output_dir / "cue_independence_rm_anova.csv",
        index=False,
    )
    passive_anova.to_csv(
        output_dir / "passive_decay_rm_anova.csv",
        index=False,
    )

    return AnalysisResults(
        data=data,
        long_data=long_data,
        tnt_summary=summary,
        reliability=reliability,
        correlation_matrix=corr_r,
        correlation_pvalues=corr_p,
        correlation_ci=corr_ci,
        cfa_fit=cfa_fit,
        statistical_tests=statistical_tests,
    )


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Pocket TNT analysis pipeline."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Participant-level TNT data (.xlsx, .xls, or .csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for tables and figures (default: outputs).",
    )
    parser.add_argument(
        "--n-per-cell",
        type=int,
        default=12,
        help=(
            "Participants sampled per Counterbalance x randomizer cell. "
            "Use 0 to disable sampling and analyze all participants."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed used for cell-wise participant sampling.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()

    n_per_cell = args.n_per_cell if args.n_per_cell > 0 else None

    run_analysis(
        input_file=args.input_file,
        output_dir=args.output_dir,
        sample_n_per_cell=n_per_cell,
        random_state=args.seed,
    )

    print(f"Analysis complete. Results saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
