import pandas as pd

from src.feature_engineering import create_features


def test_create_features():

    customer = pd.DataFrame([{
    "gender": "Male",
    "SeniorCitizen": 0,
    'Partner': "No",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    'OnlineSecurity': "No",
    'OnlineBackup': "No",
    'DeviceProtection': "No",
    'TechSupport': "No",
    'StreamingTV': "No",
    'StreamingMovies': "No",
    "Contract": "Month-to-month",
    'PaperlessBilling': "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 80,
    "TotalCharges": 400
    }])

    result = create_features(customer)

    assert "tenure_segment" in result.columns
    assert "avg_monthly_spend" in result.columns
    assert "service_count" in result.columns
    assert "risk_score" in result.columns