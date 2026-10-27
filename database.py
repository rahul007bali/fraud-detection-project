
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "fraud_predictions.db"

def create_table():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction INTEGER NOT NULL,
                fraud_probability REAL NOT NULL
            )
        """)

if __name__ == "__main__":
    create_table()
    print("Database and predictions table created successfully!")
