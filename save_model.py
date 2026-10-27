
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent
df = pd.read_csv(BASE_DIR / "creditcard.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train)

model_path = BASE_DIR / "fraud_model.joblib"
joblib.dump(model, model_path)

print(f"Model saved to: {model_path}")
print("Training complete!")

