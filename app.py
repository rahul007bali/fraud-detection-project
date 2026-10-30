
from pathlib import Path
from database import create_table
from flask import render_template
import sqlite3

import joblib
import pandas as pd
from flask import Flask, request, jsonify

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "fraud_predictions.db"

model = joblib.load(BASE_DIR / "fraud_model.joblib")
app = Flask(__name__)
create_table()


def save_prediction(prediction, probability):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO predictions (prediction, fraud_probability)
            VALUES (?, ?)
            """,
            (prediction, probability)
        )


@app.route("/")
def home():
    return "Fraud Detection API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object"}), 400

    try:
        transaction = pd.DataFrame([data])
        prediction = int(model.predict(transaction)[0])
        probability = float(model.predict_proba(transaction)[0, 1])

        save_prediction(prediction, probability)

        return jsonify({
            "prediction": prediction,
            "result": "Likely Fraud" if prediction == 1 else "Likely Legitimate",
            "fraud_probability": round(probability, 6),
            "saved_to_database": True
        })

    except (ValueError, TypeError, KeyError) as error:
        return jsonify({"error": str(error)}), 400


@app.route("/dashboard")
def dashboard():
    with sqlite3.connect(DB_PATH) as conn:

        total = conn.execute(
            "SELECT COUNT(*) FROM predictions"
        ).fetchone()[0]

        fraud_count = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE prediction = 1"
        ).fetchone()[0]

        legitimate_count = total - fraud_count

        rows = conn.execute("""
            SELECT id, prediction, fraud_probability
            FROM predictions
            ORDER BY id DESC
            LIMIT 50
        """).fetchall()

    return render_template(
        "dashboard.html",
        total=total,
        fraud_count=fraud_count,
        legitimate_count=legitimate_count,
        rows=rows
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
