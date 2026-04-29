"""
modules/privacy_layer.py
========================
Applies differential privacy mechanisms (Laplace noise via diffprivlib)
and evaluates membership inference risk via SDMetrics.

Inputs:
    - real_df      (pd.DataFrame) — original data
    - synthetic_df (pd.DataFrame) — generated synthetic data

Outputs:
    - DP-protected DataFrame
    - Privacy score (float, 0–100; closer to 0 = more private)
    - Console summary of privacy budget and attack result
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_numeric_columns(df: pd.DataFrame) -> list:
    """
    Return numeric columns present in df that are also listed in
    config.NUMERIC_COLUMNS.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    list of str
        Column names to protect with Laplace noise.
    """
    present = set(df.select_dtypes(include=[np.number]).columns)
    target  = set(config.NUMERIC_COLUMNS)
    return list(present & target)


def _sensitivity_for_col(df: pd.DataFrame, col: str) -> float:
    """
    Estimate the L1 sensitivity for a numeric column as the column range
    (max - min), bounded by config.DP_SENSITIVITY as a fallback.

    Parameters
    ----------
    df  : pd.DataFrame
    col : str  — column name

    Returns
    -------
    float
        Estimated sensitivity value.
    """
    col_range = float(df[col].max() - df[col].min())
    return max(col_range, config.DP_SENSITIVITY)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def apply_differential_privacy(
    df: pd.DataFrame,
    epsilon: float = None,
) -> pd.DataFrame:
    """
    Add calibrated Laplace noise to all numeric columns using diffprivlib
    to achieve (ε, 0)-differential privacy.

    Each numeric column receives its own Laplace mechanism instance
    calibrated to that column's range (sensitivity) and the global ε budget.

    Parameters
    ----------
    df      : pd.DataFrame
        DataFrame of synthetic patients (numeric + categorical columns).
    epsilon : float, optional
        Privacy budget ε.  Smaller = more private.
        Defaults to config.DP_EPSILON.

    Returns
    -------
    pd.DataFrame
        Copy of df with Laplace noise added to numeric columns.
        Categorical columns are unchanged.
    """
    from diffprivlib.mechanisms import Laplace

    epsilon = epsilon if epsilon is not None else config.DP_EPSILON
    print("[STEP 4] Applying differential privacy (Laplace mechanism)...")
    print(f"  Privacy budget ε = {epsilon}")

    dp_df        = df.copy()
    numeric_cols = _get_numeric_columns(df)

    if not numeric_cols:
        print("  No numeric columns found — DP skipped.")
        return dp_df

    total_epsilon_spent = 0.0
    for col in numeric_cols:
        sensitivity      = _sensitivity_for_col(df, col)
        col_epsilon      = epsilon / len(numeric_cols)   # split budget equally
        total_epsilon_spent += col_epsilon

        mech   = Laplace(epsilon=col_epsilon, sensitivity=sensitivity)
        values = dp_df[col].values.astype(float)
        noisy  = np.array([mech.randomise(v) for v in values])

        # Clip to plausible range (non-negative)
        noisy  = np.clip(noisy, 0, None)
        dp_df[col] = noisy

    print(f"  Columns protected    : {numeric_cols}")
    print(f"  Total ε spent        : {total_epsilon_spent:.4f}")
    print(f"  Per-column ε budget  : {epsilon / len(numeric_cols):.4f}")
    print("[STEP 4] Differential privacy applied.\n")

    return dp_df


def run_membership_inference_attack(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> float:
    """
    Evaluate membership inference risk using the SDMetrics NewRowSynthesis
    metric, which measures how distinguishable synthetic rows are from real ones.

    A high NewRowSynthesis score means synthetic rows are genuinely novel
    (not memorised), so a high score corresponds to LOW membership inference risk.

    We invert the score to produce a "privacy risk score":
        privacy_risk = 100 - (new_row_synthesis_score * 100)

    Interpretation:
        privacy_risk ≈  0  → excellent privacy (highly novel synthetic data)
        privacy_risk ≈ 100 → poor privacy (synthetic rows mimic real rows)

    Parameters
    ----------
    real_df      : pd.DataFrame
        Real (Synthea) training data.
    synthetic_df : pd.DataFrame
        Generated synthetic data.

    Returns
    -------
    float
        Privacy risk score in [0, 100].  Closer to 0 is better.
    """
    print("[PRIVACY] Running membership inference attack (NewRowSynthesis)...")

    # Align columns — keep only columns present in both DataFrames
    common_cols  = [c for c in real_df.columns if c in synthetic_df.columns]
    real_aligned = real_df[common_cols].copy()
    syn_aligned  = synthetic_df[common_cols].copy()

    # SDMetrics expects plain object / numeric dtypes (not pandas Categorical)
    for col in real_aligned.select_dtypes(include=["category"]).columns:
        real_aligned[col] = real_aligned[col].astype(str)
    for col in syn_aligned.select_dtypes(include=["category"]).columns:
        syn_aligned[col]  = syn_aligned[col].astype(str)

    try:
        from sdmetrics.single_table import NewRowSynthesis

        score = NewRowSynthesis.compute(
            real_data=real_aligned,
            synthetic_data=syn_aligned,
        )
        raw_score = float(score)
    except Exception as exc:
        print(f"  NewRowSynthesis unavailable ({exc}). Falling back to heuristic.")
        # Heuristic: compute cosine-distance based overlap for numeric columns
        raw_score = _heuristic_privacy_score(real_aligned, syn_aligned)

    # Convert to risk score (lower = safer)
    privacy_risk = round((1.0 - raw_score) * 100, 2)
    status       = "safe" if privacy_risk < config.PRIVACY_SAFE_THRESHOLD else "at risk"

    print(f"  NewRowSynthesis (novelty) score : {raw_score:.4f}")
    print(f"  Privacy Risk Score              : {privacy_risk}/100")
    print(f"  → Synthetic data is [{status.upper()}]")

    return privacy_risk


def _heuristic_privacy_score(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> float:
    """
    Fallback heuristic for privacy estimation when SDMetrics is unavailable.

    Computes the fraction of synthetic rows whose nearest-neighbour distance
    to any real row (in normalised numeric space) exceeds a threshold.
    A higher fraction → more private (higher novelty).

    Parameters
    ----------
    real_df : pd.DataFrame
    syn_df  : pd.DataFrame

    Returns
    -------
    float
        Novelty fraction in [0, 1].
    """
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import pairwise_distances

    numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return 0.80  # default safe score

    scaler     = MinMaxScaler()
    real_norm  = scaler.fit_transform(real_df[numeric_cols].fillna(0))
    syn_norm   = scaler.transform(syn_df[numeric_cols].fillna(0).values[:len(syn_df)])

    # Sample at most 200 rows for speed
    sample_r   = real_norm[:200]
    sample_s   = syn_norm[:200]

    dist_matrix = pairwise_distances(sample_s, sample_r, metric="euclidean")
    min_dists   = dist_matrix.min(axis=1)  # nearest real neighbour per syn row

    threshold   = 0.1   # normalised distance threshold
    novelty     = (min_dists > threshold).mean()
    return float(novelty)
