from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = "C:/Users/hp/Desktop/Post_meal_blood_glucose_prediction/Post-meal-blood-glucose-prediction/data/raw/meal_level_glucose_delta_dataset.csv"
PROCESSED_DATA_PATH = "C:/Users/hp/Desktop/Post_meal_blood_glucose_prediction/Post-meal-blood-glucose-prediction/data/processed/clean_meal_level_glucose_delta_dataset.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET = "glucose_120min"
GROUP_COLUMN = "subject_id"
RANDOM_STATE = 42
RANDOM_STATES = [10, 20, 30, 42, 50, 70, 100]
TEST_SIZE = 0.20