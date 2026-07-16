from src.config import MODEL_PATH, THRESHOLD
from src.feature_engineering import create_features

import joblib
import pandas as pd

model = joblib.load(MODEL_PATH)

def predict_churn(customer_data):

    df = pd.DataFrame([customer_data])

    df = create_features(df)

    probability = model.predict_proba(df)[0][1]

    prediction = int(probability > THRESHOLD)

    return {
        "churn_prediction": prediction,
        "churn_probability": round(float(probability), 3)
    }