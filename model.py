import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

seed = 42

campaigns = pd.read_csv("campaigns.csv")
users = pd.read_csv("users.csv")
ads = pd.read_csv("ads.csv")
ad_events = pd.read_csv("ad_events.csv")

df = pd.merge(ad_events, ads, on="ad_id", how="left")
df = pd.merge(df, users, on="user_id", how="left")

df["gender_match"] = (df["user_gender"] == df["target_gender"]) | (df["target_gender"] == "All")
df["gender_match"] = df["gender_match"].astype(int)

df["age_match"] = (df["age_group"] == df["target_age_group"]) | (df["target_age_group"] == "All")
df["age_match"] = df["age_match"].astype(int)

df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
df["hour_bucket"] = pd.cut(
    df["hour"],
    bins=[0, 6, 12, 18, 24],
    labels=["Night", "Morning", "Afternoon", "Evening"],
    include_lowest=True
)

features = [
    "user_gender",
    "age_group",
    "ad_platform",
    "ad_type",
    "hour_bucket",
    "gender_match",
    "age_match",
    "interaction_score"
]

X = df[features]
y = df["event_type"]  


X = pd.get_dummies(X)

feature_cols = X.columns.tolist()
joblib.dump(feature_cols, "svm_model.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed, stratify=y
)

clf = SVC(kernel='rbf', random_state=seed)
print("Training SVM model...")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

joblib.dump(clf, "svm_model.sav")
print("Model saved as svm_model.sav")
print("Feature list saved as svm_model.pkl")
