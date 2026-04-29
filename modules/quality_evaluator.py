"""
modules/quality_evaluator.py
=============================
Evaluates the statistical fidelity and ML utility of generated synthetic data.

Inputs:
    - real_df      (pd.DataFrame) — real/training data
    - synthetic_df (pd.DataFrame) — generated synthetic data

Outputs:
    - Quality report printed to console
    - results/charts/quality_scores.png  — per-column fidelity bar chart
    - Console comparison of Train-on-Synthetic vs Train-on-Real accuracy
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

def _align_dataframes(real_df: pd.DataFrame, syn_df: pd.DataFrame):
    """
    Align two DataFrames to share only common columns and cast categoricals
    to string for SDMetrics compatibility.

    Parameters
    ----------
    real_df : pd.DataFrame
    syn_df  : pd.DataFrame

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        Aligned real and synthetic DataFrames.
    """
    common = [c for c in real_df.columns if c in syn_df.columns]
    r = real_df[common].copy()
    s = syn_df[common].copy()
    for col in r.select_dtypes(include=["category"]).columns:
        r[col] = r[col].astype(str)
        s[col] = s[col].astype(str)
    return r, s


def _compute_js_divergence(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> dict:
    """
    Compute Jensen–Shannon Divergence per column between real and synthetic.

    For numeric columns  → binned into 20 equally-spaced histogram buckets.
    For object/category  → frequency distribution over unique values.

    JS divergence is in [0, 1]:  0 = identical distributions.

    Parameters
    ----------
    real_df : pd.DataFrame
    syn_df  : pd.DataFrame

    Returns
    -------
    dict
        {column_name: js_divergence_value}
    """
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import entropy as scipy_entropy

    js_scores = {}
    common = [c for c in real_df.columns if c in syn_df.columns]

    for col in common:
        try:
            if pd.api.types.is_numeric_dtype(real_df[col]):
                combined_min = min(real_df[col].min(), syn_df[col].min())
                combined_max = max(real_df[col].max(), syn_df[col].max())
                bins = np.linspace(combined_min, combined_max, 21)

                p, _ = np.histogram(real_df[col].dropna(), bins=bins, density=True)
                q, _ = np.histogram(syn_df[col].dropna(),  bins=bins, density=True)
            else:
                cats = list(set(real_df[col].dropna().astype(str)) |
                            set(syn_df[col].dropna().astype(str)))
                p = np.array([real_df[col].astype(str).value_counts().get(c, 0)
                              for c in cats], dtype=float)
                q = np.array([syn_df[col].astype(str).value_counts().get(c, 0)
                              for c in cats], dtype=float)

            # Normalise; avoid zero-division
            p = p + 1e-10
            q = q + 1e-10
            p /= p.sum()
            q /= q.sum()

            js_div = float(jensenshannon(p, q, base=2))
            js_scores[col] = round(js_div, 4)
        except Exception:
            js_scores[col] = np.nan

    return js_scores


def _plot_quality_scores(scores: dict, save_path: str) -> None:
    """
    Save a horizontal bar chart showing per-column fidelity scores (1 - JS).

    Parameters
    ----------
    scores    : dict  — {column: js_divergence}
    save_path : str   — absolute PNG path
    """
    cols   = list(scores.keys())
    js_vals = [scores[c] for c in cols]
    fidelity = [1.0 - v if not np.isnan(v) else 0.0 for v in js_vals]

    palette = sns.color_palette(config.SEABORN_PALETTE, n_colors=len(cols))
    sns.set_style(config.SEABORN_STYLE)

    fig, ax = plt.subplots(figsize=config.CHART_FIGSIZE)
    bars = ax.barh(cols, fidelity, color=palette)

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Fidelity Score  (1 − JS Divergence)", fontsize=13)
    ax.set_title("Per-Column Fidelity: Real vs. Synthetic", fontsize=15,
                 fontweight="bold")
    ax.axvline(0.80, color="red", linestyle="--", linewidth=1.2, label="80% threshold")
    ax.legend(fontsize=10)

    # Annotate bars
    for bar, val in zip(bars, fidelity):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=9,
        )

    sns.despine()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Quality scores chart saved → {save_path}")


def _print_quality_table(overall: float, shapes: float, trends: float,
                         js_scores: dict) -> None:
    """
    Print a formatted summary table of all quality metrics.

    Parameters
    ----------
    overall   : float  — overall SDMetrics quality score [0, 1]
    shapes    : float  — column shapes score
    trends    : float  — column pair trends score
    js_scores : dict   — per-column JS divergence
    """
    sep = "─" * 52
    print(f"\n  {sep}")
    print(f"  {'QUALITY METRIC':<30} {'SCORE':>10}")
    print(f"  {sep}")
    print(f"  {'Overall Quality Score':<30} {overall * 100:>9.2f}%")
    print(f"  {'Column Shapes Score':<30} {shapes * 100:>9.2f}%")
    print(f"  {'Column Pair Trends Score':<30} {trends * 100:>9.2f}%")
    print(f"  {sep}")
    print(f"  Per-Column Jensen–Shannon Divergence (lower = better):")
    for col, val in js_scores.items():
        flag = " ✓" if (not np.isnan(val) and val < 0.1) else ""
        print(f"    {col:<28} {val:>8.4f}{flag}")
    print(f"  {sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def run_quality_report(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> dict:
    """
    Compute a comprehensive statistical quality report comparing real and
    synthetic DataFrames using SDMetrics and Jensen–Shannon Divergence.

    Saves a bar chart of per-column fidelity scores.

    Parameters
    ----------
    real_df      : pd.DataFrame
        Original data (used as ground truth).
    synthetic_df : pd.DataFrame
        Synthetic data to evaluate.

    Returns
    -------
    dict
        Keys: 'overall', 'shapes', 'trends', 'js_scores'
    """
    print("[STEP 5] Running quality evaluation...")

    real_aligned, syn_aligned = _align_dataframes(real_df, synthetic_df)
    js_scores = _compute_js_divergence(real_aligned, syn_aligned)

    # ── SDMetrics Quality Report ──────────────────────────────────────────────
    overall_score = shapes_score = trends_score = 0.0

    try:
        from sdmetrics.reports.single_table import QualityReport
        from sdv.metadata import SingleTableMetadata

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(real_aligned)

        report = QualityReport()
        report.generate(real_aligned, syn_aligned, metadata.to_dict())

        overall_score = report.get_score()
        props         = report.get_properties()

        # props is a DataFrame with columns ['Property', 'Score']
        props_dict = dict(zip(props["Property"], props["Score"]))
        shapes_score = props_dict.get("Column Shapes",      0.0)
        trends_score = props_dict.get("Column Pair Trends", 0.0)

    except Exception as exc:
        print(f"  SDMetrics QualityReport failed ({exc}). Using JS-based fallback.")
        fidelity_scores = [1 - v for v in js_scores.values() if not np.isnan(v)]
        overall_score   = float(np.mean(fidelity_scores)) if fidelity_scores else 0.0
        shapes_score    = overall_score
        trends_score    = overall_score * 0.95

    _print_quality_table(overall_score, shapes_score, trends_score, js_scores)
    _plot_quality_scores(js_scores, config.QUALITY_SCORES_PNG)

    print("[STEP 5] Quality evaluation complete.\n")
    return {
        "overall": overall_score,
        "shapes":  shapes_score,
        "trends":  trends_score,
        "js_scores": js_scores,
    }


def run_ml_utility_test(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> dict:
    """
    Evaluate ML utility of synthetic data using a Train-on-Synthetic /
    Test-on-Real (TSTR) protocol vs. a Train-on-Real / Test-on-Real (TRTR)
    baseline.

    A RandomForestClassifier is trained on:
      1. synthetic_df  → tested on real_df  (TSTR accuracy)
      2. real_df       → tested on real_df  (TRTR accuracy, oracle baseline)

    Close TSTR ≈ TRTR accuracy validates that synthetic data is useful for ML.

    Parameters
    ----------
    real_df      : pd.DataFrame
        Real data (used for both TRTR training and testing).
    synthetic_df : pd.DataFrame
        Synthetic data used for TSTR training.

    Returns
    -------
    dict
        Keys: 'tstr_accuracy', 'trtr_accuracy', 'utility_gap'
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split as sk_split
    from sklearn.metrics import accuracy_score

    print("[STEP 6] Running ML utility test (TSTR vs TRTR)...")

    label_col = config.LABEL_COLUMN

    def _prepare(df: pd.DataFrame):
        """Encode categoricals and return (X, y) arrays."""
        df_enc = df.copy()
        le_map = {}
        for col in df_enc.select_dtypes(include=["category", "object"]).columns:
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
            le_map[col] = le
        if label_col not in df_enc.columns:
            raise ValueError(f"Label column '{label_col}' not found in DataFrame.")
        X = df_enc.drop(columns=[label_col]).values
        y = df_enc[label_col].values
        return X, y

    # Common columns
    common_cols = [c for c in real_df.columns if c in synthetic_df.columns]
    real_sub    = real_df[common_cols].copy()
    syn_sub     = synthetic_df[common_cols].copy()

    # Align categories between real and synthetic for consistent encoding
    for col in real_sub.select_dtypes(include=["category"]).columns:
        real_sub[col] = real_sub[col].astype(str)
    for col in syn_sub.select_dtypes(include=["category"]).columns:
        syn_sub[col]  = syn_sub[col].astype(str)

    X_real, y_real = _prepare(real_sub)
    X_syn,  y_syn  = _prepare(syn_sub)

    # 80/20 split of real data for testing
    X_r_train, X_r_test, y_r_train, y_r_test = sk_split(
        X_real, y_real,
        test_size=0.20,
        random_state=config.RANDOM_STATE,
        stratify=y_real,
    )

    clf = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        random_state=config.RANDOM_STATE,
    )

    # ── TSTR ──────────────────────────────────────────────────────────────────
    clf.fit(X_syn, y_syn)
    tstr_acc = accuracy_score(y_r_test, clf.predict(X_r_test))

    # ── TRTR ──────────────────────────────────────────────────────────────────
    clf.fit(X_r_train, y_r_train)
    trtr_acc = accuracy_score(y_r_test, clf.predict(X_r_test))

    utility_gap = trtr_acc - tstr_acc

    print(f"\n  {'─' * 45}")
    print(f"  {'ML UTILITY TEST':^45}")
    print(f"  {'─' * 45}")
    print(f"  Train-on-Synthetic, Test-on-Real (TSTR) : {tstr_acc * 100:.2f}%")
    print(f"  Train-on-Real,      Test-on-Real (TRTR) : {trtr_acc * 100:.2f}%")
    print(f"  Utility Gap (TRTR − TSTR)               : {utility_gap * 100:.2f}%")
    print(f"  {'─' * 45}")

    if utility_gap < 0.05:
        print("  ✓ Synthetic data is highly useful for ML (gap < 5%)")
    elif utility_gap < 0.10:
        print("  ~ Synthetic data is moderately useful (gap < 10%)")
    else:
        print("  ✗ Utility gap > 10% — consider more training epochs")
    print()

    print("[STEP 6] ML utility test complete.\n")
    return {
        "tstr_accuracy": round(tstr_acc, 4),
        "trtr_accuracy": round(trtr_acc, 4),
        "utility_gap":   round(utility_gap, 4),
    }
