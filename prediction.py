import joblib
import pandas as pd

model_name = "svm_model.sav"
feature_name = "svm_model.pkl"

model = joblib.load(model_name)
feature_cols = joblib.load(feature_name)

def prepare_input(row):
    df = pd.DataFrame([row])
    df = pd.get_dummies(df)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]
    return df

def predict(df):
    return model.predict(df)