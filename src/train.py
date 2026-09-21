import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.pipeline import Pipeline

sys.path.append(str(Path(__file__).resolve().parent))

from config import (
    ARTIFACTS_DIR,
    GROUP_COLUMN,
    PROCESSED_DATA_PATH,
    RANDOM_STATES,
    RAW_DATA_PATH,
    REPORTS_DIR,
    TARGET,
    TEST_SIZE,
)
from evaluation import build_error_table, save_json
from features import prepare_features
from models import get_model_candidates
from preprocessing import build_preprocessor


def regression_metrics(y_true, y_pred):
    """Calculate standard regression metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_pipeline(estimator, numeric_features, categorical_features):
    """Create a leakage-safe preprocessing and modeling pipeline."""
    return Pipeline([
        (
            "preprocessor",
            build_preprocessor(numeric_features, categorical_features),
        ),
        ("model", estimator),
    ])


def grouped_cv_results(
    candidates,
    X_train,
    y_train,
    groups_train,
    numeric_features,
    categorical_features,
):
    """Evaluate candidate models with grouped cross-validation."""
    cv_splits = min(5, groups_train.nunique())

    if cv_splits < 2:
        raise ValueError(
            "At least two unique subjects are required for grouped CV."
        )

    cv = GroupKFold(n_splits=cv_splits)
    results = []

    for name, estimator in candidates.items():
        fold_metrics = []

        for fold_train, fold_valid in cv.split(
            X_train,
            y_train,
            groups_train,
        ):
            pipeline = build_pipeline(
                estimator,
                numeric_features,
                categorical_features,
            )

            pipeline.fit(
                X_train.iloc[fold_train],
                y_train.iloc[fold_train],
            )

            predictions = pipeline.predict(
                X_train.iloc[fold_valid]
            )

            fold_metrics.append(
                regression_metrics(
                    y_train.iloc[fold_valid],
                    predictions,
                )
            )

        results.append({
            "model": name,
            "cv_mae": float(
                np.mean([m["mae"] for m in fold_metrics])
            ),
            "cv_rmse": float(
                np.mean([m["rmse"] for m in fold_metrics])
            ),
            "cv_r2": float(
                np.mean([m["r2"] for m in fold_metrics])
            ),
            "cv_rmse_std": float(
                np.std([m["rmse"] for m in fold_metrics])
            ),
        })

    return (
        pd.DataFrame(results)
        .sort_values("cv_rmse")
        .reset_index(drop=True)
    )


def tune_extra_trees(
    X_train,
    y_train,
    groups_train,
    numeric_features,
    categorical_features,
    random_state,
):
    """Tune Extra Trees using grouped cross-validation on training data."""
    candidates = get_model_candidates(random_state)

    pipeline = build_pipeline(
        candidates["extra_trees"],
        numeric_features,
        categorical_features,
    )

    param_grid = {
        "model__n_estimators": [400, 700],
        "model__max_features": [1.0, 0.7],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_depth": [None, 12],
    }

    cv_splits = min(5, groups_train.nunique())

    cv = GroupKFold(n_splits=cv_splits)

    from sklearn.model_selection import GridSearchCV

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )

    search.fit(
        X_train,
        y_train,
        groups=groups_train,
    )

    tuning_results = (
        pd.DataFrame(search.cv_results_)
        .sort_values("rank_test_score")
    )

    columns = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "params",
    ]

    return (
        search.best_estimator_,
        search.best_params_,
        tuning_results[columns],
    )


def run_single_seed(
    random_state,
    X,
    y,
    groups,
    cleaned_df,
    numeric_features,
    categorical_features,
):
    """Run the complete training and evaluation workflow for one seed."""

    print("\n" + "=" * 70)
    print(f"Random state: {random_state}")
    print("=" * 70)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=random_state,
    )

    train_idx, test_idx = next(
        splitter.split(
            X,
            y,
            groups=groups,
        )
    )

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()

    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    groups_train = groups.iloc[train_idx].copy()
    groups_test = groups.iloc[test_idx].copy()

    results_df = grouped_cv_results(
        get_model_candidates(random_state),
        X_train,
        y_train,
        groups_train,
        numeric_features,
        categorical_features,
    )

    results_df.insert(0, "random_state", random_state)

    print("\nCross-validation results:")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model"]

    tuned_pipeline = None
    best_params = None
    tuned_name = best_name

    if best_name == "extra_trees":
        tuned_pipeline, best_params, tuning_results = tune_extra_trees(
            X_train,
            y_train,
            groups_train,
            numeric_features,
            categorical_features,
            random_state,
        )

        tuned_name = "tuned_extra_trees"

        tuning_path = (
            REPORTS_DIR
            / f"extra_trees_tuning_seed_{random_state}.csv"
        )

        tuning_results.to_csv(
            tuning_path,
            index=False,
        )

        save_json(
            best_params,
            REPORTS_DIR
            / f"extra_trees_best_params_seed_{random_state}.json",
        )

    else:
        tuned_pipeline = build_pipeline(
            get_model_candidates(random_state)[best_name],
            numeric_features,
            categorical_features,
        )

        tuned_pipeline.fit(
            X_train,
            y_train,
        )

    test_predictions = tuned_pipeline.predict(X_test)

    test_metrics = regression_metrics(
        y_test,
        test_predictions,
    )

    test_metadata = cleaned_df.iloc[test_idx][
        [
            c
            for c in [
                GROUP_COLUMN,
                "meal_type",
                "meal_time",
                "premeal_glucose",
            ]
            if c in cleaned_df.columns
        ]
    ].copy()

    error_table = build_error_table(
        y_test,
        test_predictions,
        test_metadata,
    )

    error_table.to_csv(
        REPORTS_DIR
        / f"test_predictions_and_errors_seed_{random_state}.csv",
        index=False,
    )

    subject_errors = (
        error_table
        .groupby(GROUP_COLUMN)
        .agg(
            n_samples=("absolute_error", "size"),
            mae=("absolute_error", "mean"),
            rmse=(
                "residual",
                lambda x: float(
                    np.sqrt(
                        np.mean(
                            np.square(x)
                        )
                    )
                ),
            ),
            mean_residual=("residual", "mean"),
        )
        .sort_values(
            "mae",
            ascending=False,
        )
    )

    subject_errors.to_csv(
        REPORTS_DIR
        / f"test_error_by_subject_seed_{random_state}.csv"
    )

    seed_result = {
        "random_state": random_state,
        "selected_cv_model": best_name,
        "final_model": tuned_name,
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_subjects": int(groups_train.nunique()),
        "test_subjects": int(groups_test.nunique()),
        "subject_overlap": int(
            len(
                set(groups_train)
                & set(groups_test)
            )
        ),
    }

    if best_params is not None:
        seed_result["best_params"] = json.dumps(
            best_params,
            sort_keys=True,
        )

    print(f"\nCV-selected model: {best_name}")

    if best_name == "extra_trees":
        print("Tuned final model: extra_trees")
        print(f"Best parameters: {best_params}")

    print("Held-out test metrics:")
    print(json.dumps(test_metrics, indent=2))

    return (
        seed_result,
        tuned_pipeline,
        test_idx,
        results_df,
    )


def main():
    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_df = pd.read_csv(
        RAW_DATA_PATH
    )

    (
        X,
        y,
        groups,
        numeric_features,
        categorical_features,
        cleaned_df,
    ) = prepare_features(
        raw_df,
        TARGET,
        GROUP_COLUMN,
    )

    valid_target = y.notna()

    X = X.loc[
        valid_target
    ].reset_index(drop=True)

    y = y.loc[
        valid_target
    ].reset_index(drop=True)

    groups = groups.loc[
        valid_target
    ].reset_index(drop=True)

    cleaned_df = cleaned_df.loc[
        valid_target
    ].reset_index(drop=True)

    all_seed_results = []
    all_model_results = []

    best_candidate = None

    for random_state in RANDOM_STATES:

        (
            seed_result,
            tuned_pipeline,
            test_idx,
            results_df,
        ) = run_single_seed(
            random_state=random_state,
            X=X,
            y=y,
            groups=groups,
            cleaned_df=cleaned_df,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

        all_seed_results.append(
            seed_result
        )

        all_model_results.append(
            results_df
        )

        if (
            best_candidate is None
            or seed_result["test_rmse"]
            < best_candidate["test_rmse"]
        ):
            best_candidate = {
                "random_state": random_state,
                "test_rmse": seed_result["test_rmse"],
                "pipeline": tuned_pipeline,
                "test_idx": test_idx,
            }

    seed_results_df = pd.DataFrame(
        all_seed_results
    )

    seed_results_df.to_csv(
        REPORTS_DIR
        / "random_state_results.csv",
        index=False,
    )

    model_results_df = pd.concat(
        all_model_results,
        ignore_index=True,
    )

    model_results_df.to_csv(
        REPORTS_DIR
        / "model_comparison_by_seed.csv",
        index=False,
    )

    summary_df = (
        seed_results_df[
            [
                "test_mae",
                "test_rmse",
                "test_r2",
            ]
        ]
        .agg(
            [
                "mean",
                "std",
                "min",
                "max",
            ]
        )
        .T
        .reset_index()
        .rename(
            columns={
                "index": "metric",
            }
        )
    )

    summary_df.to_csv(
        REPORTS_DIR
        / "random_state_summary.csv",
        index=False,
    )

    best_seed = int(
        best_candidate["random_state"]
    )

    print("\n" + "=" * 70)
    print("Random state summary")
    print("=" * 70)

    print(
        seed_results_df[
            [
                "random_state",
                "selected_cv_model",
                "test_mae",
                "test_rmse",
                "test_r2",
            ]
        ].to_string(index=False)
    )

    print("\nAggregate test performance:")
    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\nNote: The best-performing seed is reported for diagnostic "
        "purposes only and should not automatically be selected as "
        "the final model."
    )

    print(
        f"\nLowest test RMSE observed at seed: {best_seed}"
    )

    print(
        f"\nSaved random-state results: "
        f"{REPORTS_DIR / 'random_state_results.csv'}"
    )

    print(
        f"Saved random-state summary: "
        f"{REPORTS_DIR / 'random_state_summary.csv'}"
    )

    print(
        f"Saved model comparison: "
        f"{REPORTS_DIR / 'model_comparison_by_seed.csv'}"
    )

    print(
        f"Saved best-seed diagnostic model: "
        f"{ARTIFACTS_DIR / 'best_model_diagnostic.joblib'}"
    )

    joblib.dump(
        best_candidate["pipeline"],
        ARTIFACTS_DIR
        / "best_model_diagnostic.joblib",
    )

    cleaned_df.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
    )


if __name__ == "__main__":
    main()