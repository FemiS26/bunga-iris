import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

seed = 42

campaigns = pd.read_csv("campaigns.csv")
users = pd.read_csv("users.csv")
ads = pd.read_csv("ads.csv")
ad_events = pd.read_csv("ad_events.csv")

df = pd.merge(ad_events, ads, on='ad_id', how='left')
df = pd.merge(df, users, on='user_id', how='left')

feature_cols = ['user_age', 'user_gender', 'ad_platform', 'ad_type', 'day_of_week']
target_col = 'event_type'

data = df[feature_cols + [target_col]].dropna().sample(n=5000, random_state=seed)

le_dict = {} 
categorical_cols = ['user_gender', 'ad_platform', 'ad_type', 'day_of_week', 'event_type']

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    le_dict[col] = le

X = data[feature_cols]
y = data[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed, stratify=y)

clf = SVC(kernel='rbf', random_state=seed)

print("Training SVM model...")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

joblib.dump(clf, "svm_model.sav")
print("Model saved as svm_model.sav")