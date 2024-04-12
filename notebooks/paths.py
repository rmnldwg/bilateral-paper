"""
Define paths for data and models.
"""
from pathlib import Path
from typing import Literal

root = Path(__file__).parent.parent
data_dir = root / "data"
data = data_dir / "reduced.csv"
model_dir = root / "models"


def get_path(
    for_model: Literal["ipsi", "contra", "bilateral", "midline"],
    of_kind: Literal[
        "params",
        "samples",
        "priors",
        "posteriors",
        "prevalences",
        "risks",
        "history",
    ],
) -> Path:
    """Get the filename of the samples of one of the four trained models."""
    return next((model_dir / for_model).glob(f"{of_kind}.*"))
