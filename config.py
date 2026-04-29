"""
config.py
=========
Single source of truth for ALL hyperparameters and file paths used
across the Synthetic Data Generation Framework.

Inputs:  None
Outputs: Constants imported by every other module
"""

import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR        = os.path.join(BASE_DIR, "data", "raw")
DATA_SYNTHETIC_DIR  = os.path.join(BASE_DIR, "data", "synthetic")
RESULTS_CHARTS_DIR  = os.path.join(BASE_DIR, "results", "charts")
MODEL_SAVE_DIR      = os.path.join(BASE_DIR, "data")

# Input CSV (Synthea patients file)
PATIENTS_CSV        = os.path.join(DATA_RAW_DIR, "patients.csv")

# Saved model paths
CTGAN_MODEL_PATH    = os.path.join(MODEL_SAVE_DIR, "ctgan_model.pkl")
TVAE_MODEL_PATH     = os.path.join(MODEL_SAVE_DIR, "tvae_model.pkl")
GAUSSIAN_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "gaussian_model.pkl")

# Output chart paths
TRAINING_LOSS_PNG       = os.path.join(RESULTS_CHARTS_DIR, "training_loss.png")
QUALITY_SCORES_PNG      = os.path.join(RESULTS_CHARTS_DIR, "quality_scores.png")
BIAS_COMPARISON_PNG     = os.path.join(RESULTS_CHARTS_DIR, "bias_comparison.png")
DISTRIBUTION_PLOT_PNG   = os.path.join(RESULTS_CHARTS_DIR, "distribution_comparison.png")

# ─────────────────────────────────────────────
# DATA COLUMNS
# ─────────────────────────────────────────────
# Core demographic / financial columns to load from patients.csv
CORE_COLUMNS = [
    "AGE",
    "GENDER",
    "RACE",
    "ETHNICITY",
    "MARITAL",
    "HEALTHCARE_EXPENSES",
    "HEALTHCARE_COVERAGE",
]

# Top condition columns (will be filled from data if present)
CONDITION_COLUMNS = [
    "CONDITION_1",
    "CONDITION_2",
    "CONDITION_3",
    "CONDITION_4",
    "CONDITION_5",
]

# All target columns (conditions appended dynamically in loader)
TARGET_COLUMNS = CORE_COLUMNS  # conditions added at runtime

# Column to use as label for ML utility test & bias audit
LABEL_COLUMN = "GENDER"

# Numeric columns that receive Laplace noise in differential privacy
NUMERIC_COLUMNS = [
    "AGE",
    "HEALTHCARE_EXPENSES",
    "HEALTHCARE_COVERAGE",
]

# Categorical columns
CATEGORICAL_COLUMNS = [
    "GENDER",
    "RACE",
    "ETHNICITY",
    "MARITAL",
]

# Sensitive attribute columns for bias auditing
SENSITIVE_COLUMNS = ["RACE", "GENDER"]

# ─────────────────────────────────────────────
# GAN / MODEL HYPERPARAMETERS
# ─────────────────────────────────────────────
CTGAN_EPOCHS        = 300
CTGAN_BATCH_SIZE    = 500
TVAE_EPOCHS         = 300
TVAE_BATCH_SIZE     = 500

# Number of synthetic rows to generate
N_SYNTHETIC_SAMPLES = 1000

# Train / test split ratio
TRAIN_TEST_SPLIT    = 0.80
RANDOM_STATE        = 42

# ─────────────────────────────────────────────
# DIFFERENTIAL PRIVACY
# ─────────────────────────────────────────────
DP_EPSILON          = 1.0          # privacy budget (smaller = more private)
DP_SENSITIVITY      = 1.0          # L1 sensitivity for Laplace mechanism
DP_DELTA            = 1e-5         # (unused for Laplace, kept for reference)

# ─────────────────────────────────────────────
# PRIVACY EVALUATION THRESHOLDS
# ─────────────────────────────────────────────
PRIVACY_SAFE_THRESHOLD  = 70.0     # score above this = "safe"

# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────
CHART_FIGSIZE   = (10, 6)
CHART_DPI       = 150
SEABORN_STYLE   = "whitegrid"
SEABORN_PALETTE = "muted"

# ─────────────────────────────────────────────
# RANDOM FOREST (ML utility test)
# ─────────────────────────────────────────────
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH    = 8
