import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    """Calculate standard regression metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_json(payload, path: Path):
    """Save a JSON-serializable payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def build_error_table(y_true, predictions, metadata):
    """Build a row-level table containing predictions and residuals."""
    result = metadata.copy().reset_index(drop=True)
    result["actual"] = np.asarray(y_true)
    result["prediction"] = np.asarray(predictions)
    result["residual"] = result["actual"] - result["prediction"]
    result["absolute_error"] = result["residual"].abs()
    return result
