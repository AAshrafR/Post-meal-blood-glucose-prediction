import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(Path(__file__).resolve().parent))

from config import ARTIFACTS_DIR
from features import add_prediction_features, clean_dataframe


def predict(input_path: str, output_path: str | None = None):
    """Generate predictions for a CSV file using the saved model."""
    model = joblib.load(ARTIFACTS_DIR / "best_model.joblib")
    raw_data = pd.read_csv(input_path)
    features = add_prediction_features(clean_dataframe(raw_data))

    drop_columns = ["glucose_120min", "meal_time", "subject_id"]
    X = features.drop(columns=drop_columns, errors="ignore")
    predictions = model.predict(X)

    result = raw_data.copy()
    result["predicted_glucose_120min"] = predictions

    if output_path:
        result.to_csv(output_path, index=False)
    else:
        print(result[["predicted_glucose_120min"]].head(20).to_string(index=False))

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python src/predict.py <input_csv> [output_csv]"
        )

    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    predict(input_csv, output_csv)
