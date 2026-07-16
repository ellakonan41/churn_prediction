import joblib
import pandas as pd
import numpy as np

from src.config import MODEL_PATH
from src.data_loader import load_data
from src.feature_engineering import create_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import make_column_selector
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import classification_report





model= XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    scale_pos_weight= 2.77 )

def splitting(df):

    X = df.drop("Churn", axis=1)
    Y = df["Churn"]

    Y = Y.map({
    "No": 0,
    "Yes": 1
    })

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)

    return X_train,X_test,Y_train,Y_test

def preprocesssing_pipeline():
    Numerical = make_column_selector(dtype_include=np.number)
    Categorical = make_column_selector(dtype_exclude=np.number)

    Numerical_pipeline= make_pipeline(SimpleImputer(),StandardScaler())
    Categorical_pipeline= make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown="ignore"))
    
    Preprocessor = make_column_transformer((Numerical_pipeline,Numerical),(Categorical_pipeline,Categorical))

    Pipeline = make_pipeline(
    Preprocessor,
    model
)

    return Pipeline

def train_model():

    df = load_data()

    df = create_features(df)

    # séparation X/y
    X_train,X_test,Y_train,Y_test = splitting(df)
    
    pipeline = preprocesssing_pipeline()

    # preprocessing + XGBoost
    pipeline.fit(X_train,Y_train)

    joblib.dump(
        pipeline,
        MODEL_PATH
    )

    return pipeline, X_test, Y_test