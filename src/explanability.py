import joblib
import shap
import pandas as pd
import math

from src.config import MODEL_PATH
from src.feature_engineering import create_features

# Chargement du pipeline
pipeline = joblib.load(MODEL_PATH)

# Séparation preprocessing / modèle
preprocessor = pipeline.named_steps["columntransformer"]
model = pipeline.named_steps["xgbclassifier"]

# Création de l'explainer
explainer = shap.TreeExplainer(model)


""" def get_feature_importance(customer_data):

    df = pd.DataFrame([customer_data])

    # mêmes transformations que l'entraînement
    df = create_features(df)

    X = preprocessor.transform(df)

    shap_values = explainer.shap_values(X)

    feature_names = preprocessor.get_feature_names_out()

    importance = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values[0]
    })

    importance["feature"] = (
    importance["feature"]
    .str.replace("pipeline-1__", "")
    .str.replace("pipeline-2__", "")
    .str.replace("_", " ")
)
    importance["impact_direction"] = importance["shap_value"].apply(
    lambda x: "Augmente le risque" if x > 0 else "Réduit le risque"
)
    # Trier par impact absolu
    importance["impact"] = importance["shap_value"].abs()

    importance = importance.sort_values(
        by="impact",
        ascending=False
    )

    return importance.head(5) """

    

def get_feature_importance(customer_data):

    df = pd.DataFrame([customer_data])

    # Même feature engineering que pendant l'entraînement
    df = create_features(df)

    # Même preprocessing que le pipeline
    X = preprocessor.transform(df)

    # Prédiction réelle du modèle
    model_probability = float(
        model.predict_proba(X)[0, 1]
    )

    model_prediction = int(
        model.predict(X)[0]
    )

    # SHAP
    shap_values = explainer.shap_values(X)
    shap_row = shap_values[0]

    # Valeur de référence
    expected_value = explainer.expected_value

    if hasattr(expected_value, "__len__"):
        expected_value = expected_value[0]

    expected_value = float(expected_value)

    # Contributions
    positive_sum = float(
        shap_row[shap_row > 0].sum()
    )

    negative_sum = float(
        shap_row[shap_row < 0].sum()
    )

    total_shap = float(
        shap_row.sum()
    )

    # Reconstruction SHAP
    reconstructed_output = (
        expected_value + total_shap
    )

    # Log-odds -> probabilité
    reconstructed_probability = 1 / (
        1 + math.exp(-reconstructed_output)
    )

    # Noms des variables
    feature_names = preprocessor.get_feature_names_out()

    importance = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_row
    })

    importance["feature"] = (
        importance["feature"]
        .str.replace("pipeline-1__", "", regex=False)
        .str.replace("pipeline-2__", "", regex=False)
        .str.replace("_", " ", regex=False)
    )

    importance["impact_direction"] = importance[
        "shap_value"
    ].apply(
        lambda x:
        "Augmente le risque"
        if x > 0
        else "Réduit le risque"
    )

    importance["impact"] = (
        importance["shap_value"].abs()
    )

    importance = importance.sort_values(
        by="impact",
        ascending=False
    ).reset_index(drop=True)

    return {
        "all_features": importance,
        "top_features": importance.head(5),
        "positive_sum": positive_sum,
        "negative_sum": negative_sum,
        "total_shap": total_shap,
        "expected_value": expected_value,
        "reconstructed_output": reconstructed_output,
        "model_probability": model_probability,
        "reconstructed_probability": reconstructed_probability,
        "model_prediction": model_prediction
    }