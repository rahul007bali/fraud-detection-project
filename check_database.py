
import sqlite3
from pathlib import Path

db_path = Path(__file__).parent / "fraud_predictions.db"

with sqlite3.connect(db_path) as conn:
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT 5"
    ).fetchall()

print("Latest saved predictions:")
for row in rows:
    print(row)
