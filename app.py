
from pathlib import Path
import sqlite3

import joblib
import pandas as pd
from flask import Flask, request, jsonify

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "fraud_predictions.db"

model = joblib.load(BASE_DIR / "fraud_model.joblib")
app = Flask(__name__)


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
        rows = conn.execute("""
            SELECT id, prediction, fraud_probability
            FROM predictions
            ORDER BY id DESC
            LIMIT 50
        """).fetchall()

    html = """
    <h1>Fraud Detection Dashboard</h1>
    <p>Latest saved transaction predictions</p>
    <table border="1" cellpadding="8">
        <tr>
            <th>ID</th>
            <th>Prediction</th>
            <th>Fraud Probability</th>
        </tr>
    """

    for row_id, prediction, probability in rows:
        result = "Likely Fraud" if prediction == 1 else "Likely Legitimate"
        html += (
            f"<tr><td>{row_id}</td><td>{result}</td>"
            f"<td>{probability:.2%}</td></tr>"
        )

    html += "</table>"
    return html


if __name__ == "__main__":
    app.run(debug=True)
