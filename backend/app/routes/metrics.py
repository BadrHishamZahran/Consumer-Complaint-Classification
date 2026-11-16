from pathlib import Path
import pandas as pd
from fastapi import APIRouter

from app.config import EXPORTED_MODELS_DIR

router = APIRouter()

@router.get("/models/metrics")
def get_metrics():
    candidates = [
        EXPORTED_MODELS_DIR / "model_comparison.csv",
        EXPORTED_MODELS_DIR / "model_comparison_results.csv",
    ]
    comparison = next((p for p in candidates if p.exists()), None)
    comparison_rows = []
    if comparison:
        comparison_rows = pd.read_csv(comparison).replace({float("nan"): None}).to_dict(orient="records")

    confusion_matrices = {}
    for path in EXPORTED_MODELS_DIR.glob("*confusion_matrix.csv"):
        try:
            confusion_matrices[path.stem] = pd.read_csv(path, index_col=0).to_dict()
        except Exception:
            continue

    return {
        "comparison": comparison_rows,
        "confusion_matrices": confusion_matrices,
        "note": None if comparison else "model_comparison.csv was not found in exported_models.",
    }
