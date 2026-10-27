
import pandas as pd
import requests
from pathlib import Path

BASE_DIR = Path(__file__).parent
df = pd.read_csv(BASE_DIR / "creditcard.csv")

# Select one transaction's input features
transaction = df.drop(columns=["Class"]).iloc[0].to_dict()

response = requests.post(
    "http://127.0.0.1:5000/predict",
    json=transaction
)

print("Status code:", response.status_code)
print("Response:", response.text)
