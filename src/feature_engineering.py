import pandas as pd

services = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]
def create_features(df):

    df = df.copy()
    
    # Supprimer customerID si elle existe
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce" )

    # Segmentation ancienneté
    df['tenure_segment'] = pd.cut(df['tenure'], bins=[0, 12, 24, 60, 100],
                               labels=['<1yr', '1-2yr', '2-5yr', '>5yr'])
    # charge mensuel moyen
    df["avg_monthly_spend"] = (
    df["TotalCharges"] /
    (df["tenure"] + 1))

    # Nombre de services souscrits
    df["service_count"] = (
        df[services]
        .replace({"Yes":1,"No":0,"No internet service":0})
        .sum(axis=1)
    )

    # Score de risque basé sur les caractéristiques du client
    df["risk_score"] = 0

    df.loc[df["Contract"]=="Month-to-month",
        "risk_score"] += 1

    df.loc[df["TechSupport"]=="No",
        "risk_score"] += 1

    df.loc[df["OnlineSecurity"]=="No",
        "risk_score"] += 1

    df.loc[df["tenure"]<12,
        "risk_score"] += 1

    return df

