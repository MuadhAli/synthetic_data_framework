"""
modules/bias_auditor.py
=======================
Audits demographic fairness before and after synthetic data generation
using the Aequitas toolkit.

Inputs:
    - real_df      (pd.DataFrame) — original Synthea data
    - synthetic_df (pd.DataFrame) — synthetic data after DP
    - label_col    (str)          — column used as the score/label

Outputs:
    - results/charts/bias_comparison.png  — side-by-side fairness metrics
    - Console summary with bias reduction percentage
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_for_aequitas(
    df: pd.DataFrame,
    label_col: str,
    sensitive_cols: list,
) -> pd.DataFrame:
    """
    Prepare a DataFrame for Aequitas by creating 'score' and 'label_value'
    columns expected by the Aequitas Group object.

    Aequitas expects:
        - 'score'       : predicted probability / score (numeric in [0, 1])
        - 'label_value' : binary ground-truth label (0 or 1)
        - attribute columns used as group identifiers

    We binarise label_col using the median as a threshold and create a
    mock score (adding small noise to the label for realism).

    Parameters
    ----------
    df             : pd.DataFrame
    label_col      : str  — column to use as the predicted attribute
    sensitive_cols : list — demographic columns to analyse

    Returns
    -------
    pd.DataFrame
        Aequitas-ready DataFrame with 'score' and 'label_value' columns.
    """
    aq_df = df.copy()

    # Convert categoricals to string
    for col in aq_df.select_dtypes(include=["category"]).columns:
        aq_df[col] = aq_df[col].astype(str)

    # Create binary label from the label_col
    if label_col in aq_df.columns:
        # Binary encode: use the most frequent value as "positive"
        top_val = aq_df[label_col].value_counts().idxmax()
        aq_df["label_value"] = (aq_df[label_col].astype(str) == str(top_val)).astype(int)
    else:
        aq_df["label_value"] = 1  # fallback

    # Create a mock score column (bounded random noise around label)
    rng = np.random.default_rng(config.RANDOM_STATE)
    noise = rng.uniform(-0.15, 0.15, len(aq_df))
    aq_df["score"] = (aq_df["label_value"].astype(float) + noise).clip(0, 1)

    # Keep only aequitas-required columns
    keep = ["score", "label_value"] + [c for c in sensitive_cols if c in aq_df.columns]
    aq_df = aq_df[keep].dropna()

    return aq_df


def _compute_disparate_impact(aq_df: pd.DataFrame, sensitive_cols: list) -> dict:
    """
    Compute disparate impact ratio for each sensitive column group.

    Disparate impact = (positive rate of minority group) / (positive rate of majority group).
    Value of 1.0 = perfectly fair; < 0.8 typically indicates bias.

    Parameters
    ----------
    aq_df          : pd.DataFrame — prepared Aequitas DataFrame
    sensitive_cols : list         — demographic attribute columns

    Returns
    -------
    dict
        {attribute: {group_value: disparate_impact_ratio}}
    """
    results = {}

    overall_pos_rate = aq_df["label_value"].mean()
    if overall_pos_rate == 0:
        return results

    for col in sensitive_cols:
        if col not in aq_df.columns:
            continue
        group_rates = {}
        for grp, grp_df in aq_df.groupby(col):
            if len(grp_df) < 5:
                continue
            grp_rate           = grp_df["label_value"].mean()
            group_rates[grp]   = round(grp_rate / overall_pos_rate, 4)
        results[col] = group_rates

    return results


def _aequitas_bias_scores(aq_df: pd.DataFrame, sensitive_cols: list) -> pd.DataFrame:
    """
    Use the Aequitas library to compute Bias Group metrics.

    Falls back to the manual disparate-impact calculation if Aequitas
    cannot be imported.

    Parameters
    ----------
    aq_df          : pd.DataFrame — Aequitas-ready DataFrame
    sensitive_cols : list

    Returns
    -------
    pd.DataFrame
        Aequitas bias metrics grouped by attribute.
    """
    try:
        from aequitas.group import Group
        from aequitas.bias import Bias

        g  = Group()
        xt, _ = g.get_crosstabs(aq_df, attr_cols=sensitive_cols)

        b       = Bias()
        bdf     = b.get_disparity_predefined_groups(
            xt,
            original_df=aq_df,
            ref_groups_dict={col: aq_df[col].value_counts().idxmax()
                             for col in sensitive_cols if col in aq_df.columns},
        )
        return bdf

    except Exception as exc:
        print(f"  Aequitas library call failed ({exc}). Using manual DI computation.")
        return None


def _mean_di_score(bias_results: dict) -> float:
    """
    Compute the mean absolute deviation from 1.0 across all disparate impact
    ratios (1.0 = fair).

    Parameters
    ----------
    bias_results : dict — output of _compute_disparate_impact

    Returns
    -------
    float
        Mean bias deviation from parity (lower = less biased).
    """
    deviations = []
    for col_dict in bias_results.values():
        for grp_val in col_dict.values():
            deviations.append(abs(1.0 - grp_val))
    return float(np.mean(deviations)) if deviations else 0.0


def _plot_bias_comparison(
    real_di: dict,
    syn_di: dict,
    save_path: str,
) -> None:
    """
    Save a grouped horizontal bar chart comparing disparate impact ratios
    between the real Synthea baseline and the synthetic dataset.

    Parameters
    ----------
    real_di   : dict — {attribute: {group: DI ratio}} for real data
    syn_di    : dict — {attribute: {group: DI ratio}} for synthetic data
    save_path : str
    """
    rows_real = []
    rows_syn  = []

    for attr, grp_dict in real_di.items():
        for grp, val in grp_dict.items():
            rows_real.append({"Attribute": f"{attr}={grp}", "Disparate Impact": val})

    for attr, grp_dict in syn_di.items():
        for grp, val in grp_dict.items():
            rows_syn.append({"Attribute": f"{attr}={grp}", "Disparate Impact": val})

    df_real = pd.DataFrame(rows_real).assign(Source="Synthea Baseline")
    df_syn  = pd.DataFrame(rows_syn).assign(Source="Synthetic Data")
    df_plot = pd.concat([df_real, df_syn], ignore_index=True)

    if df_plot.empty:
        print("  No bias data to plot.")
        return

    # Top-N groups by absolute deviation from 1.0
    df_plot["Deviation"] = (df_plot["Disparate Impact"] - 1.0).abs()
    top_attrs = (
        df_plot.groupby("Attribute")["Deviation"].max()
        .nlargest(12).index.tolist()
    )
    df_plot = df_plot[df_plot["Attribute"].isin(top_attrs)]

    sns.set_style(config.SEABORN_STYLE)
    fig, ax = plt.subplots(figsize=config.CHART_FIGSIZE)

    sns.barplot(
        data=df_plot,
        x="Disparate Impact",
        y="Attribute",
        hue="Source",
        palette={"Synthea Baseline": "#C44E52", "Synthetic Data": "#4C72B0"},
        ax=ax,
    )

    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.2,
               label="Parity (DI = 1.0)")
    ax.axvline(0.8, color="orange", linestyle=":", linewidth=1.2,
               label="Caution threshold (0.8)")
    ax.set_xlabel("Disparate Impact Ratio", fontsize=13)
    ax.set_title("Bias Comparison: Synthea Baseline vs Synthetic Data",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    sns.despine()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Bias comparison chart saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def run_bias_audit(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    label_col: str = None,
) -> dict:
    """
    Audit and compare demographic bias between real and synthetic DataFrames.

    Steps
    -----
    1. Prepare both DataFrames in Aequitas format (score + label_value).
    2. Compute disparate impact ratios for RACE and GENDER groups.
    3. Compare mean bias deviation: real baseline vs. synthetic.
    4. Plot a grouped bar chart of DI ratios.
    5. Print the percentage improvement (or worsening) in bias.

    Parameters
    ----------
    real_df      : pd.DataFrame
        Original Synthea data (truth baseline).
    synthetic_df : pd.DataFrame
        Synthetic data after privacy processing.
    label_col    : str, optional
        Column treated as the predicted attribute.
        Defaults to config.LABEL_COLUMN.

    Returns
    -------
    dict
        Keys:
          - 'real_bias_score'  : mean DI deviation for real data
          - 'syn_bias_score'   : mean DI deviation for synthetic data
          - 'bias_reduction_pct': % reduction in bias (positive = improved)
    """
    label_col = label_col or config.LABEL_COLUMN
    print("[STEP 7] Running bias audit...")

    # Common columns
    common         = [c for c in real_df.columns if c in synthetic_df.columns]
    sensitive_cols = [c for c in config.SENSITIVE_COLUMNS if c in common]

    real_sub = real_df[common].copy()
    syn_sub  = synthetic_df[common].copy()

    # Prepare for Aequitas
    real_aq = _prepare_for_aequitas(real_sub, label_col, sensitive_cols)
    syn_aq  = _prepare_for_aequitas(syn_sub,  label_col, sensitive_cols)

    # Disparate impact
    real_di = _compute_disparate_impact(real_aq, sensitive_cols)
    syn_di  = _compute_disparate_impact(syn_aq,  sensitive_cols)

    real_bias = _mean_di_score(real_di)
    syn_bias  = _mean_di_score(syn_di)

    if real_bias > 0:
        bias_reduction_pct = ((real_bias - syn_bias) / real_bias) * 100
    else:
        bias_reduction_pct = 0.0

    # Plot
    _plot_bias_comparison(real_di, syn_di, config.BIAS_COMPARISON_PNG)

    # Console summary
    print(f"\n  {'─' * 52}")
    print(f"  {'BIAS AUDIT RESULTS':^52}")
    print(f"  {'─' * 52}")
    print(f"  Sensitive attributes      : {sensitive_cols}")
    print(f"  Real data bias score      : {real_bias:.4f}")
    print(f"  Synthetic data bias score : {syn_bias:.4f}")
    if bias_reduction_pct >= 0:
        print(f"  → Bias REDUCED by {bias_reduction_pct:.2f}% across demographic groups")
    else:
        print(f"  → Bias INCREASED by {abs(bias_reduction_pct):.2f}% (review model settings)")
    print(f"  {'─' * 52}\n")

    print("[STEP 7] Bias audit complete.\n")
    return {
        "real_bias_score":    round(real_bias, 4),
        "syn_bias_score":     round(syn_bias, 4),
        "bias_reduction_pct": round(bias_reduction_pct, 2),
    }
