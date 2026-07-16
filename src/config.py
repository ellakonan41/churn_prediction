# src/config.py

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"

DATA_PATH = BASE_DIR / "data" / "raw" / "Telco-Customer-Churn.csv"

THRESHOLD = 0.4