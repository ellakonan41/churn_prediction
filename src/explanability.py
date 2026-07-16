import joblib
import shap
import pandas as pd

from src.config import MODEL_PATH
from src.feature_engineering import create_features

# Chargement du pipeline
pipeline = joblib.load(MODEL_PATH)

# Séparation preprocessing / modèle
preprocessor = pipeline.named_steps["columntransformer"]
model = pipeline.named_steps["xgbclassifier"]

# Création de l'explainer
explainer = shap.TreeExplainer(model)


def get_feature_importance(customer_data):

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

    return importance.head(5)