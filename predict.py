
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).parent

# Load the trained model
model = joblib.load(BASE_DIR / "fraud_model.joblib")

# Load one transaction from our dataset
df = pd.read_csv(BASE_DIR / "creditcard.csv")
transaction = df.drop(columns=["Class"]).iloc[[0]]

# Predict whether the transaction is fraudulent
prediction = model.predict(transaction)[0]
probability = model.predict_proba(transaction)[0, 1]

print("Prediction:", "Potential Fraud" if prediction == 1 else "Likely Legitimate")
print(f"Estimated fraud probability: {probability:.4f}")
