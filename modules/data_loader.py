"""
modules/data_loader.py
======================
Loads and preprocesses the Synthea patients CSV dataset.

Inputs:
    - data/raw/patients.csv  (Synthea-generated CSV)

Outputs:
    - train_df  (pd.DataFrame)  — 80 % split, cleaned
    - test_df   (pd.DataFrame)  — 20 % split, cleaned
    - full_df   (pd.DataFrame)  — entire cleaned dataset
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from faker import Faker

# Bring project root onto path so config is always importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _synthesise_mock_data(n: int = 2000) -> pd.DataFrame:
    """
    Generate a plausible mock Synthea-like DataFrame using Faker when no real
    patients.csv is available.  This lets the framework run end-to-end without
    any real patient data.

    Parameters
    ----------
    n : int
        Number of mock patient rows to generate.

    Returns
    -------
    pd.DataFrame
        Mock dataset with the same schema as a real Synthea patients.csv.
    """
    fake = Faker()
    Faker.seed(config.RANDOM_STATE)
    rng  = np.random.default_rng(config.RANDOM_STATE)

    genders     = ["M", "F"]
    races       = ["white", "black", "asian", "hispanic", "native", "other"]
    ethnicities = ["nonhispanic", "hispanic"]
    maritals    = ["S", "M", "D", "W"]
    conditions  = [
        "Hypertension", "Diabetes", "Asthma", "Obesity",
        "Heart Disease", "Anxiety", "Depression", "Arthritis",
        "COPD", "Hyperlipidemia",
    ]

    rows = []
    for _ in range(n):
        row = {
            "AGE":                  int(rng.integers(1, 95)),
            "GENDER":               rng.choice(genders),
            "RACE":                 rng.choice(races),
            "ETHNICITY":            rng.choice(ethnicities),
            "MARITAL":              rng.choice(maritals),
            "HEALTHCARE_EXPENSES":  round(float(rng.uniform(500, 150_000)), 2),
            "HEALTHCARE_COVERAGE":  round(float(rng.uniform(0, 120_000)), 2),
            "CONDITION_1":          rng.choice(conditions + [""]),
            "CONDITION_2":          rng.choice(conditions + [""]),
            "CONDITION_3":          rng.choice(conditions + [""]),
            "CONDITION_4":          rng.choice(conditions + [""]),
            "CONDITION_5":          rng.choice(conditions + [""]),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def _load_csv(filepath: str) -> pd.DataFrame:
    """
    Read the Synthea patients.csv and normalise column names to upper-case.

    Parameters
    ----------
    filepath : str
        Absolute path to patients.csv.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with upper-cased column names.
    """
    df = pd.read_csv(filepath, low_memory=False)
    df.columns = df.columns.str.upper().str.strip()
    return df


def _select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select CORE_COLUMNS plus up to 5 condition-like columns from the raw
    dataset.  Missing columns are silently skipped.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Synthea DataFrame (upper-cased columns).

    Returns
    -------
    pd.DataFrame
        Subset DataFrame with only the target columns.
    """
    desired = list(config.CORE_COLUMNS)

    # Try to find condition columns (Synthea doesn't always have them by
    # that exact name; we look for any column matching CONDITION_\d)
    import re
    cond_pattern = re.compile(r"CONDITION_\d+")
    found_conds  = [c for c in df.columns if cond_pattern.match(c)]

    if not found_conds:
        # Fall back to REASONCODE / DESCRIPTION style Synthea columns
        alt_conds = [c for c in df.columns if "REASON" in c or "CODE" in c]
        found_conds = alt_conds[:5]

    for i, col in enumerate(found_conds[:5]):
        desired.append(col)
        if col != f"CONDITION_{i+1}":
            df = df.rename(columns={col: f"CONDITION_{i+1}"})
            desired[-1] = f"CONDITION_{i+1}"

    existing = [c for c in desired if c in df.columns]
    return df[existing].copy()


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values for numeric and categorical columns.

    Numeric columns  → filled with column median.
    Categorical cols → filled with 'Unknown'.
    Remaining NaNs   → dropped.

    Parameters
    ----------
    df : pd.DataFrame
        Subset DataFrame after column selection.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with no NaN values.
    """
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("Unknown")

    df = df.dropna()
    return df.reset_index(drop=True)


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode object/string columns as pandas Categorical dtype so that SDV
    models understand them as discrete variables.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with categoricals cast to 'category' dtype.
    """
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("category")
    return df


def _print_summary(df: pd.DataFrame) -> None:
    """
    Print a concise summary of the loaded DataFrame to stdout.

    Parameters
    ----------
    df : pd.DataFrame
        Full cleaned DataFrame.
    """
    print(f"\n  Shape            : {df.shape}")
    print(f"  Columns          : {list(df.columns)}")
    print(f"  Numeric columns  : {list(df.select_dtypes(include=[np.number]).columns)}")
    print(f"  Category columns : {list(df.select_dtypes(include=['category']).columns)}")
    print(f"  Missing values   : {df.isnull().sum().sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def load_synthea_data(filepath: str = None):
    """
    Main entry point for data loading.

    Loads Synthea patients.csv (or generates mock data if the file is absent),
    applies preprocessing, and returns train/test/full splits.

    Parameters
    ----------
    filepath : str, optional
        Path to patients.csv.  Defaults to config.PATIENTS_CSV.

    Returns
    -------
    train_df : pd.DataFrame
        80 % training split.
    test_df  : pd.DataFrame
        20 % test split.
    full_df  : pd.DataFrame
        Full cleaned dataset (before splitting).
    """
    print("[STEP 1] Loading Synthea data...")

    filepath = filepath or config.PATIENTS_CSV

    if os.path.exists(filepath):
        print(f"  Found patients.csv at:  {filepath}")
        raw_df = _load_csv(filepath)
        df     = _select_columns(raw_df)
        print(f"  Source: real Synthea CSV   | raw rows = {len(raw_df)}")
    else:
        print(f"  patients.csv NOT found at {filepath}")
        print("  Generating mock Synthea-like data using Faker...")
        df = _synthesise_mock_data(n=2000)

    df = _clean(df)
    df = _encode_categoricals(df)

    _print_summary(df)

    train_df, test_df = train_test_split(
        df,
        train_size=config.TRAIN_TEST_SPLIT,
        random_state=config.RANDOM_STATE,
    )

    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    print(f"  Train rows: {len(train_df)} | Test rows: {len(test_df)}")
    print("[STEP 1] Data loaded successfully.\n")

    return train_df, test_df, df
