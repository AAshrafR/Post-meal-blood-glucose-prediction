import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request


# Add the src directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from predict import predict_dataframe


app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    """Render the web application."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Generate a glucose prediction from JSON input."""

    data = request.get_json()

    if not data:
        return jsonify(
            {
                "error": "No JSON data was provided."
            }
        ), 400

    try:
        input_data = pd.DataFrame([data])

        result = predict_dataframe(input_data)

        prediction = result[
            "predicted_glucose_120min"
        ].iloc[0]

        return jsonify(
            {
                "predicted_glucose_120min": float(
                    prediction
                )
            }
        )

    except Exception as error:

        return jsonify(
            {
                "error": str(error)
            }
        ), 400


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )