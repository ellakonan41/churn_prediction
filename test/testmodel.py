from src.predictor import predict_churn
from src.explanability import get_feature_importance


client = {
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
}


result = predict_churn(client)

print(f"Prediction: {result}")

importance = get_feature_importance(client)

print(f"features: {importance}")