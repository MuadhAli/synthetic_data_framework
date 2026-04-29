"""
main.py
=======
End-to-end orchestrator for the Synthetic Data Generation Framework.

Running `python main.py` executes the full pipeline:
    Step 1  → Load Synthea data
    Step 2  → Train generative models (CTGAN, TVAE, GaussianCopula)
    Step 3  → Generate 1 000 synthetic patient records
    Step 4  → Apply differential privacy (ε = 1.0)
    Step 5  → Quality evaluation (SDMetrics + JS divergence)
    Step 6  → ML utility test (TSTR vs TRTR)
    Step 7  → Bias audit (disparate impact comparison)

All charts are saved to results/charts/.
A final summary is printed to stdout.

Inputs:  data/raw/patients.csv  (or mock data if absent)
Outputs: data/synthetic/synthetic_patients.csv
         results/charts/*.png
"""

import os
import sys
import warnings
import time
import pandas as pd

warnings.filterwarnings("ignore")

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

from modules.data_loader       import load_synthea_data
from modules.gan_trainer       import train_all_models, generate_synthetic
from modules.privacy_layer     import apply_differential_privacy, run_membership_inference_attack
from modules.quality_evaluator import run_quality_report, run_ml_utility_test
from modules.bias_auditor      import run_bias_audit


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    """Print a prominent section banner to stdout."""
    width = 60
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


def _save_synthetic(df: pd.DataFrame) -> None:
    """
    Save the final synthetic DataFrame to the data/synthetic/ directory as CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Fully processed synthetic patient records.
    """
    os.makedirs(config.DATA_SYNTHETIC_DIR, exist_ok=True)
    save_path = os.path.join(config.DATA_SYNTHETIC_DIR, "synthetic_patients.csv")
    df.to_csv(save_path, index=False)
    print(f"  Synthetic data saved → {save_path}")


def _print_final_summary(
    quality_score: float,
    privacy_score: float,
    bias_reduction: float,
    tstr_acc: float,
    trtr_acc: float,
    elapsed: float,
) -> None:
    """
    Print the final one-line (+ table) framework summary.

    Parameters
    ----------
    quality_score   : float  — overall quality [0, 1]
    privacy_score   : float  — membership inference risk [0, 100]  (lower = safer)
    bias_reduction  : float  — % bias reduction (positive = improved)
    tstr_acc        : float  — TSTR ML accuracy [0, 1]
    trtr_acc        : float  — TRTR ML accuracy baseline [0, 1]
    elapsed         : float  — total runtime in seconds
    """
    q_pct   = round(quality_score * 100, 2)
    sep     = "─" * 56

    print(f"\n{'═' * 56}")
    print(f"  {'FRAMEWORK COMPLETE — FINAL SUMMARY':^54}")
    print(f"{'═' * 56}")
    print(f"  {sep}")
    print(f"  {'Metric':<35} {'Value':>16}")
    print(f"  {sep}")
    print(f"  {'Overall Quality Score':<35} {q_pct:>14.2f}%")
    print(f"  {'Privacy Risk Score (lower=safer)':<35} {privacy_score:>15.2f}")
    print(f"  {'Bias Reduction':<35} {bias_reduction:>14.2f}%")
    print(f"  {'TSTR Accuracy (synthetic→real)':<35} {tstr_acc * 100:>14.2f}%")
    print(f"  {'TRTR Accuracy (baseline)':<35} {trtr_acc * 100:>14.2f}%")
    print(f"  {'Total Runtime':<35} {elapsed:>13.1f}s")
    print(f"  {sep}")
    print()
    print(f"  Framework complete.")
    print(f"  Quality: {q_pct}%  |  Privacy Score: {privacy_score}  "
          f"|  Bias Reduction: {bias_reduction}%")
    print(f"{'═' * 56}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Run the complete Synthetic Data Generation Framework pipeline.

    No arguments required; all configuration is read from config.py.
    """
    t_start = time.time()
    _banner("Synthetic Data Generation Framework  |  MTech Project")

    # ── STEP 1: Load data ─────────────────────────────────────────────────────
    train_df, test_df, full_df = load_synthea_data()

    # ── STEP 2: Train / load models ───────────────────────────────────────────
    models = train_all_models(train_df, force_retrain=False)

    # ── STEP 3: Generate synthetic data ───────────────────────────────────────
    print("[STEP 3] Generating synthetic data...")
    ctgan_model  = models["ctgan"]
    synthetic_df = generate_synthetic(ctgan_model, n=config.N_SYNTHETIC_SAMPLES)
    print(f"  Generated {len(synthetic_df)} synthetic patient records.\n")

    # ── STEP 4: Differential privacy ──────────────────────────────────────────
    dp_synthetic_df = apply_differential_privacy(
        synthetic_df,
        epsilon=config.DP_EPSILON,
    )
    _save_synthetic(dp_synthetic_df)

    # ── STEP 5: Quality evaluation ────────────────────────────────────────────
    quality_results = run_quality_report(full_df, dp_synthetic_df)

    # ── STEP 6: ML utility ────────────────────────────────────────────────────
    ml_results = run_ml_utility_test(full_df, dp_synthetic_df)

    # ── STEP 7: Privacy attack evaluation ────────────────────────────────────
    privacy_score = run_membership_inference_attack(full_df, dp_synthetic_df)

    # ── STEP 8: Bias audit ────────────────────────────────────────────────────
    bias_results = run_bias_audit(full_df, dp_synthetic_df)

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    _print_final_summary(
        quality_score   = quality_results["overall"],
        privacy_score   = privacy_score,
        bias_reduction  = bias_results["bias_reduction_pct"],
        tstr_acc        = ml_results["tstr_accuracy"],
        trtr_acc        = ml_results["trtr_accuracy"],
        elapsed         = elapsed,
    )


if __name__ == "__main__":
    main()
