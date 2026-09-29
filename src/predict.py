
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(Path(__file__).resolve().parent))

from config import ARTIFACTS_DIR
from features import add_prediction_features, clean_dataframe


MODEL_PATH = ARTIFACTS_DIR / "best_model_diagnostic.joblib"


def predict_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions for rows that pass the data-quality rules.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Raw input data containing the features required by the model.

    Returns
    -------
    pd.DataFrame
        Cleaned input data with an additional
        'predicted_glucose_120min' column.
    """
    model = joblib.load(MODEL_PATH)

    # Keep the original data for the final result
    raw_data = dataframe.copy()

    # Apply the same cleaning and feature engineering
    features = add_prediction_features(
        clean_dataframe(raw_data)
    )

    drop_columns = [
        "glucose_120min",
        "meal_time",
        "subject_id",
    ]

    X = features.drop(
        columns=drop_columns,
        errors="ignore",
    )

    predictions = model.predict(X)

    # The cleaned dataframe has exactly the same number
    # of rows as the predictions.
    result = features.copy()

    result["predicted_glucose_120min"] = predictions

    return result

def predict(
    input_path: str,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Generate predictions for a CSV file using the saved model.

    Parameters
    ----------
    input_path : str
        Path to the input CSV file.

    output_path : str | None, optional
        Path where the prediction CSV will be saved.
        If None, predictions are printed instead.

    Returns
    -------
    pd.DataFrame
        Input data with predicted glucose values.
    """
    raw_data = pd.read_csv(input_path)

    result = predict_dataframe(raw_data)

    if output_path:
        result.to_csv(output_path, index=False)
    else:
        print(
            result[
                ["predicted_glucose_120min"]
            ].head(20).to_string(index=False)
        )

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python src/predict.py <input_csv> [output_csv]"
        )

    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None

    predict(input_csv, output_csv)
