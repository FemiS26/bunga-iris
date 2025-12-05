import pandas as pd
import joblib

# Load KNN model dan daftar fitur
model = joblib.load("knn_model.sav")
feature_list = joblib.load("modelKNN1.pkl")

def prepare_input(row_dict):
    # Convert input ke DataFrame
    df = pd.DataFrame([row_dict])

    # One-hot encoding
    df = pd.get_dummies(df)

    # Tambahkan kolom yang hilang agar sesuai dengan fitur training
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0

    # Urutkan kolom sesuai fitur model
    df = df[feature_list]

    return df

def predict(df):
    return model.predict(df)