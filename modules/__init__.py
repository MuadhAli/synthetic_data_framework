"""
modules/__init__.py
===================
Package initialiser for the synthetic_data_framework modules.
Exposes the key public symbols for convenient import.

Inputs:  None
Outputs: Package namespace
"""

from .data_loader       import load_synthea_data
from .gan_trainer       import train_all_models, generate_synthetic
from .privacy_layer     import apply_differential_privacy, run_membership_inference_attack
from .quality_evaluator import run_quality_report, run_ml_utility_test
from .bias_auditor      import run_bias_audit

__all__ = [
    "load_synthea_data",
    "train_all_models",
    "generate_synthetic",
    "apply_differential_privacy",
    "run_membership_inference_attack",
    "run_quality_report",
    "run_ml_utility_test",
    "run_bias_audit",
]
