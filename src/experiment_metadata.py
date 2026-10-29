"""Runtime metadata for reproducible model and evaluation reports."""
from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any

DISTRIBUTIONS = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "imbalanced-learn",
    "xgboost",
    "optuna",
    "shap",
    "fastapi",
)


def collect_environment_metadata() -> dict[str, Any]:
    """Capture Python/platform and installed package versions in report JSON."""
    packages: dict[str, str | None] = {}
    for distribution in DISTRIBUTIONS:
        try:
            packages[distribution] = version(distribution)
        except PackageNotFoundError:
            packages[distribution] = None
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
        "random_seed_convention": (
            "The experiment runner records the configured random_state separately; "
            "this report is environment metadata, not proof of bitwise determinism."
        ),
    }
