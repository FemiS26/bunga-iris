import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

seed = 42

campaigns = pd.read_csv("campaigns.csv")
users = pd.read_csv("users.csv")
ads = pd.read_csv("ads.csv")
ad_events = pd.read_csv("ad_events.csv")

df = (ad_events
      .merge(users, on='user_id', how='left')
      .merge(ads, on='ad_id', how='left')
      .merge(campaigns, on='campaign_id', how='left'))

df['target_match'] = 'Impression'
df.loc[
    ((df['user_gender'] == df['target_gender']) |
     (df['target_gender'] == 'All') |
     (df['age_group'] == df['target_age_group']) |
     (df['target_age_group'] == 'All')),
    'target_match'
] = 'Match'

df['gender_match'] = (df['user_gender'] == df['target_gender']).astype(int)
df['age_match'] = (df['age_group'] == df['target_age_group']).astype(int)
df['hour'] = df['time_of_day'].map({'Morning':1,'Afternoon':2,'Evening':3,'Night':4})
df['interaction_score'] = df['event_type'].map({'Impression':1,'View':2,'Like':3,'Click':4})

X = df[['user_gender','age_group','ad_platform','gender_match',
        'age_match','hour','ad_type','interaction_score']]
y = df[['target_match']]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed, stratify=y
)

model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train.values.ravel())

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(model, "svm_model.sav")
joblib.dump(list(X.columns), "svm_features.pkl")