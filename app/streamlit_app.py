import sys
from datetime import date, time
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from predict import predict_dataframe

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Post-Meal Glucose Prediction",
    page_icon="🩸",
    layout="centered",
)

st.markdown("""
<style>
.med-header {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 14px;
    padding: 1.25rem 1.5rem 0 1.5rem;
    margin-bottom: 1rem;
    overflow: hidden;
}
.med-top {
    display: flex;
    align-items: center;
    gap: 1.25rem;
    margin-bottom: 1rem;
}
.med-icon-row {
    display: flex;
    gap: 10px;
    flex-shrink: 0;
}
.med-icon {
    width: 48px;
    height: 48px;
    background: #ffffff;
    border: 1px solid #BFDBFE;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
}
.med-icon.blue {
    background: #1D4ED8;
    border-color: #1D4ED8;
    color: white;
    font-size: 20px;
}
.med-title { margin: 0; }
.med-title h2 {
    font-size: 17px;
    font-weight: 600;
    color: #1E3A8A;
    margin: 0 0 3px 0;
}
.med-title p {
    font-size: 12.5px;
    color: #3B82F6;
    margin: 0;
}
.ekg-bar {
    border-top: 1px solid #BFDBFE;
    padding: 6px 0 8px 0;
}
</style>
 
<div class="med-header">
  <div class="med-top">
    <div class="med-icon-row">
      <div class="med-icon">🩸</div>
      <div class="med-icon">📈</div>
      <div class="med-icon blue">✚</div>
    </div>
    <div class="med-title">
      <h2>Post-meal blood glucose prediction</h2>
      <p>Clinical decision support &nbsp;·&nbsp; 120-minute post-meal glucose forecast</p>
    </div>
  </div>
  <div class="ekg-bar">
    <svg width="100%" height="32" preserveAspectRatio="none"
         xmlns="http://www.w3.org/2000/svg">
      <polyline
        points="0,16 50,16 62,16 72,3 82,29 92,8 102,16
                150,16 162,10 172,16 220,16 232,16 242,3
                252,29 262,8 272,16 320,16 332,10 342,16
                390,16 402,16 412,3 422,29 432,8 442,16
                490,16 502,10 512,16 560,16 572,16 582,3
                592,29 602,8 612,16 660,16 680,16"
        fill="none" stroke="#3B82F6" stroke-width="1.8"
        stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
  </div>
</div>
""", unsafe_allow_html=True)
# ── Section helper ─────────────────────────────────────────────────────────────
def section_label(icon, text):
    st.markdown(
        f'<div class="section-label">{icon}&nbsp; {text}</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PATIENT DEMOGRAPHICS
# ═══════════════════════════════════════════════════════════════════════════════
with st.container(border=True):
    section_label("👤", "Patient demographics")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)
    with col2:
        gender = st.selectbox(
            "Biological sex", ["M", "F"],
            format_func=lambda x: "Male" if x == "M" else "Female",
        )
    with col3:
        bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0,
                              value=25.0, step=0.1)

    col4, col5 = st.columns(2)
    with col4:
        weight_lb = st.number_input("Weight (lb)", min_value=50.0,
                                    max_value=600.0, value=150.0, step=0.5)
    with col5:
        height_in = st.number_input("Height (in)", min_value=36.0,
                                    max_value=96.0, value=68.0, step=0.5)

# ═══════════════════════════════════════════════════════════════════════════════
# LAB VALUES & GLUCOSE
# ═══════════════════════════════════════════════════════════════════════════════
with st.container(border=True):
    section_label("🧪", "Lab values & glucose readings")

    col1, col2, col3 = st.columns(3)
    with col1:
        a1c = st.number_input(
            "HbA1c (%)", min_value=3.0, max_value=20.0, value=5.5, step=0.1,
            help="Glycated haemoglobin — reflects average glucose over ~3 months",
        )
    with col2:
        fasting_glucose_lab = st.number_input(
            "Fasting glucose (mg/dL)", min_value=40.0, max_value=400.0,
            value=90.0, step=1.0,
            help="Most recent fasting plasma glucose from lab panel",
        )
    with col3:
        premeal_glucose = st.number_input(
            "Pre-meal glucose (mg/dL)", min_value=40.0, max_value=400.0,
            value=90.0, step=1.0,
            help="CGM or fingerstick reading immediately before this meal",
        )

    glucose_sensor_used = st.selectbox(
        "CGM device",
        ["dexcom", "libre"],
        format_func=lambda x: "Dexcom" if x == "dexcom" else "Abbott FreeStyle Libre",
        help="Continuous glucose monitor worn by the patient",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# MEAL RECORD
# ═══════════════════════════════════════════════════════════════════════════════
with st.container(border=True):
    section_label("🍽️", "Meal record")

    col1, col2, col3 = st.columns(3)
    with col1:
        meal_date = st.date_input("Date", value=date.today())
    with col2:
        meal_time_input = st.time_input("Time", value=time(12, 0))
    with col3:
        meal_type = st.selectbox(
            "Meal type", ["Breakfast", "Lunch", "Dinner", "Snack"],
        )

    meal_hour = meal_time_input.hour  # derived automatically

    st.markdown('<div class="sub-label">🥗&nbsp; Nutritional composition</div>',
                unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        calories = st.number_input("Calories (kcal)", min_value=0.0,
                                   max_value=2500.0, value=500.0, step=10.0)
    with col2:
        carbs = st.number_input("Carbs (g)", min_value=0.0,
                                max_value=300.0, value=60.0, step=1.0)
    with col3:
        protein = st.number_input("Protein (g)", min_value=0.0,
                                  max_value=200.0, value=25.0, step=1.0)
    with col4:
        fat = st.number_input("Fat (g)", min_value=0.0,
                              max_value=200.0, value=20.0, step=1.0)
    with col5:
        fiber = st.number_input("Fiber (g)", min_value=0.0,
                                max_value=100.0, value=5.0, step=0.5)

    amount_consumed = st.number_input(
        "Portion multiplier",
        min_value=0.0, value=1.0, step=0.25,
        help="1.0 = full portion as logged. 0.5 = half, 2.0 = double.",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
if st.button("⚡  Run prediction", type="primary"):

    meal_datetime = pd.Timestamp.combine(meal_date, meal_time_input)

    input_data = pd.DataFrame([{
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
    }])

    try:
        result = predict_dataframe(input_data)
        prediction = result["predicted_glucose_120min"].iloc[0]
        delta = prediction - premeal_glucose
        sign = "+" if delta >= 0 else ""

        if prediction < 140:
            css_class = "result-normal"
            status_icon = "✅"
            status_text = "Within target range"
            status_color = "#15803D"
            range_note = "ADA target: &lt;140 mg/dL at 2 h post-meal"
        elif prediction < 180:
            css_class = "result-warning"
            status_icon = "⚠️"
            status_text = "Elevated — monitor closely"
            status_color = "#B45309"
            range_note = "ADA target for diabetes: &lt;180 mg/dL at 2 h post-meal"
        else:
            css_class = "result-high"
            status_icon = "🔴"
            status_text = "High — clinical review recommended"
            status_color = "#B91C1C"
            range_note = "Exceeds ADA 2-hour post-meal threshold of 180 mg/dL"

        st.markdown(f"""
        <div class="{css_class}">
          <div class="result-label" style="color:{status_color}">
            {status_icon}&nbsp; Predicted glucose at 120 minutes
          </div>
          <div class="result-value" style="color:{status_color}">
            {prediction:.1f}
            <span style="font-size:1rem;font-weight:400;color:{status_color}">mg/dL</span>
          </div>
          <div class="result-sub">
            Excursion from pre-meal baseline:
            <strong style="color:{status_color}">{sign}{delta:.1f} mg/dL</strong>
            &nbsp;·&nbsp; {status_text}<br>
            <span style="font-size:12px">{range_note}</span>
          </div>
          <div class="metric-row">
            <div class="metric-pill">Pre-meal<span>{premeal_glucose:.0f} mg/dL</span></div>
            <div class="metric-pill">Predicted 120 min<span>{prediction:.1f} mg/dL</span></div>
            <div class="metric-pill">Excursion<span>{sign}{delta:.1f} mg/dL</span></div>
            <div class="metric-pill">HbA1c<span>{a1c:.1f}%</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📋  View full input summary"):
            summary_rows = {
                "Age": f"{age} yrs",
                "Sex": "Male" if gender == "M" else "Female",
                "BMI": f"{bmi:.1f} kg/m²",
                "Weight / Height": f"{weight_lb:.0f} lb / {height_in:.0f} in",
                "HbA1c": f"{a1c:.1f}%",
                "Fasting glucose": f"{fasting_glucose_lab:.0f} mg/dL",
                "Pre-meal glucose": f"{premeal_glucose:.0f} mg/dL",
                "CGM device": "Dexcom" if glucose_sensor_used == "dexcom" else "Abbott FreeStyle Libre",
                "Meal type": meal_type,
                "Meal time": meal_datetime.strftime("%d %b %Y, %H:%M"),
                "Calories": f"{calories:.0f} kcal",
                "Carbs / Protein / Fat / Fiber": (
                    f"{carbs:.0f} g / {protein:.0f} g / {fat:.0f} g / {fiber:.0f} g"
                ),
                "Portion multiplier": f"{amount_consumed:.2f}×",
            }
            for k, v in summary_rows.items():
                ca, cb = st.columns([2, 3])
                ca.markdown(
                    f"<span style='font-size:13px;color:#6B7280'>{k}</span>",
                    unsafe_allow_html=True,
                )
                cb.markdown(
                    f"<span style='font-size:13px;font-weight:500'>{v}</span>",
                    unsafe_allow_html=True,
                )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  For clinical decision support only &nbsp;·&nbsp;
  Not a substitute for direct patient assessment
</div>
""", unsafe_allow_html=True)