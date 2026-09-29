import sys
from datetime import date, time
from pathlib import Path

import pandas as pd
import streamlit as st


# Add the src directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from predict import predict_dataframe


st.set_page_config(
    page_title="Post-Meal Blood Glucose Prediction",
    page_icon="🩸",
    layout="centered",
)


st.title("Post-Meal Blood Glucose Prediction")

st.write(
    "Enter the patient and meal information to predict "
    "the blood glucose level at 120 minutes after the meal."
)


# ============================================================
# Patient Information
# ============================================================

st.header("Patient Information")

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30,
)

gender = st.selectbox(
    "Gender",
    ["M", "F"],
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=70.0,
    value=25.0,
)

weight_lb = st.number_input(
    "Weight (lb)",
    min_value=50.0,
    max_value=600.0,
    value=150.0,
)

height_in = st.number_input(
    "Height (in)",
    min_value=36.0,
    max_value=96.0,
    value=68.0,
)

a1c = st.number_input(
    "HbA1c",
    min_value=3.0,
    max_value=20.0,
    value=5.5,
)

fasting_glucose_lab = st.number_input(
    "Fasting Glucose",
    min_value=40.0,
    max_value=400.0,
    value=90.0,
)

premeal_glucose = st.number_input(
    "Pre-Meal Glucose",
    min_value=40.0,
    max_value=400.0,
    value=90.0,
)


# ============================================================
# Meal Information
# ============================================================

st.header("Meal Information")

meal_date = st.date_input(
    "Meal Date",
    value=date.today(),
)

meal_time = st.time_input(
    "Meal Time",
    value=time(12, 0),
)

meal_type = st.selectbox(
    "Meal Type",
    ["Breakfast", "Lunch", "Dinner", "Snack"],
)

meal_hour = st.number_input(
    "Meal Hour",
    min_value=0,
    max_value=23,
    value=12,
)

calories = st.number_input(
    "Calories",
    min_value=0.0,
    max_value=2500.0,
    value=500.0,
)

carbs = st.number_input(
    "Carbohydrates (g)",
    min_value=0.0,
    max_value=300.0,
    value=60.0,
)

protein = st.number_input(
    "Protein (g)",
    min_value=0.0,
    max_value=200.0,
    value=25.0,
)

fat = st.number_input(
    "Fat (g)",
    min_value=0.0,
    max_value=200.0,
    value=20.0,
)

fiber = st.number_input(
    "Fiber (g)",
    min_value=0.0,
    max_value=100.0,
    value=5.0,
)

amount_consumed = st.number_input(
    "Amount Consumed",
    min_value=0.0,
    value=1.0,
)

glucose_sensor_used = st.selectbox(
    "Glucose Sensor Used",
    ["dexcom", "libre"],
)


# ============================================================
# Prediction
# ============================================================

if st.button("Predict", type="primary"):

    meal_datetime = pd.Timestamp.combine(
        meal_date,
        meal_time,
    )

    input_data = pd.DataFrame(
        [
            {
                "age": age,
                "gender": gender,
                "bmi": bmi,
                "weight_lb": weight_lb,
                "height_in": height_in,
                "a1c": a1c,
                "fasting_glucose_lab": fasting_glucose_lab,
                "meal_hour": meal_hour,
                "meal_type": meal_type,
                "calories": calories,
                "carbs": carbs,
                "protein": protein,
                "fat": fat,
                "fiber": fiber,
                "amount_consumed": amount_consumed,
                "premeal_glucose": premeal_glucose,
                "glucose_sensor_used": glucose_sensor_used,
                "meal_time": meal_datetime,
            }
        ]
    )

    try:
        result = predict_dataframe(input_data)

        prediction = result[
            "predicted_glucose_120min"
        ].iloc[0]

        st.success(
            f"Predicted glucose at 120 minutes: "
            f"{prediction:.2f}"
        )

    except Exception as error:
        st.error(
            f"Prediction failed: {error}"
        )