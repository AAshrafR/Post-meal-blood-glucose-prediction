# Post-Meal Blood Glucose Prediction

A reproducible machine-learning project for predicting `glucose_120min` from meal-level and pre-meal information.

## Project structure

```text
post_meal_glucose_project/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   ├── 01_Data_Understanding_and_Preprocessing.ipynb
│   └── original_Data_Understanding_and_Preprocessing.ipynb
├── reports/
├── src/
│   ├── config.py
│   ├── evaluation.py
│   ├── features.py
│   ├── models.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
├── requirements.txt
└── .gitignore
```

## Workflow

The project uses a subject-level split so observations from the same `subject_id` are not shared between train and test.

1. Audit the raw dataset.
2. Remove identifiers, source metadata, and post-meal leakage variables.
3. Apply deterministic quality filters.
4. Create prediction-time features.
5. Remove rows with a missing target.
6. Create a grouped held-out test set.
7. Compare baseline and tree/linear regression models using grouped cross-validation.
8. Tune Extra Trees when it is the best cross-validation candidate.
9. Evaluate the final model once on the held-out subjects.
10. Save row-level and subject-level error analysis.
11. Save the final pipeline for inference.

## Setup

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Place the raw CSV at:

```text
data/raw/meal_level_glucose_delta_dataset.csv
```

## Train

From the project root:

```bash
python src/train.py
```

The training script saves:

- `models/best_model.joblib`
- `reports/model_comparison.csv`
- `reports/extra_trees_tuning.csv`
- `reports/extra_trees_best_params.json`
- `reports/test_metrics.json`
- `reports/test_predictions_and_errors.csv`
- `reports/test_error_by_subject.csv`
- `data/processed/clean_meal_level_glucose_delta_dataset.csv`

## Prediction

After training:

```bash
python src/predict.py path/to/new_data.csv predictions.csv
```

The output contains the original rows plus `predicted_glucose_120min`.

## Modeling notes

The current candidate models are:

- Mean baseline
- Ridge Regression
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor
- Histogram-based Gradient Boosting Regressor

Model selection is based on grouped cross-validation RMSE. If Extra Trees is selected, a small grouped grid search tunes its main complexity parameters before final fitting.

The held-out test set is kept untouched during model selection and tuning.

## Error analysis

The training script produces row-level predictions and residuals, plus subject-level error summaries. This makes it possible to inspect whether errors are concentrated in particular subjects or meal contexts instead of relying only on one aggregate score.

## Leakage prevention

Post-meal measurements are excluded from the feature set because they would not be available at prediction time. Imputation, scaling, and categorical encoding are fitted inside the modeling pipeline on training folds only.

`subject_id` is used for grouping but is not provided to the model as a feature.

All source-code comments and docstrings are written in English.
