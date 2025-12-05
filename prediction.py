import pandas as pd
import joblib

model = joblib.load("knn_model.sav")
feature_list = joblib.load("modelKNN1.pkl")

def prepare_input(row_dict):
    df = pd.DataFrame([row_dict])

    df = pd.get_dummies(df)

    for col in feature_list:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_list]

    return df

def predict(df):
    return model.predict(df)