import numpy as np
import pandas as pd


EXCLUDED_COLUMNS = [
    "meal_id",
    "source_file",
    "post_max_glucose_2h",
    "delta_glucose_max_2h",
    "delta_glucose_120min",
]

QUALITY_RULES = {
    "fiber": 100,
    "fat": 200,
    "carbs": 300,
    "calories": 2500,
}


def clean_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic data-quality rules without fitting statistics on the full dataset."""
    data = dataframe.drop(columns=EXCLUDED_COLUMNS, errors="ignore").copy()
    data["meal_time"] = pd.to_datetime(data["meal_time"], errors="coerce")

    for column, upper_bound in QUALITY_RULES.items():
        if column in data.columns:
            data = data.loc[data[column].isna() | (data[column] <= upper_bound)].copy()

    return data


def add_prediction_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create features using information available at prediction time."""
    data = dataframe.copy()

    data["meal_day_of_week"] = data["meal_time"].dt.dayofweek
    data["is_weekend"] = data["meal_day_of_week"].isin([5, 6]).astype(int)

    data["carb_to_fiber_ratio"] = data["carbs"] / (data["fiber"] + 1)
    data["total_macros"] = data["carbs"] + data["protein"] + data["fat"]

    data["carb_pct"] = np.divide(
        data["carbs"],
        data["total_macros"],
        out=np.zeros(len(data), dtype=float),
        where=data["total_macros"].ne(0),
    )
    data["protein_pct"] = np.divide(
        data["protein"],
        data["total_macros"],
        out=np.zeros(len(data), dtype=float),
        where=data["total_macros"].ne(0),
    )
    data["fat_pct"] = np.divide(
        data["fat"],
        data["total_macros"],
        out=np.zeros(len(data), dtype=float),
        where=data["total_macros"].ne(0),
    )

    data["glucose_diff_baseline"] = (
        data["premeal_glucose"] - data["fasting_glucose_lab"]
    )

    return data


def prepare_features(dataframe: pd.DataFrame, target: str, group_column: str):
    """Return model features, target, groups, and feature-type lists."""
    data = add_prediction_features(clean_dataframe(dataframe))

    y = data[target].copy()
    groups = data[group_column].copy()

    X = data.drop(columns=[target, "meal_time", group_column], errors="ignore")

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()

    return X, y, groups, numeric_features, categorical_features, data
