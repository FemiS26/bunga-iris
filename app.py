import streamlit as st
from prediction import prepare_input, predict

st.title("Ad Match Predictor - Universal Model (KNN)")

user_gender = st.selectbox("User Gender", ["Male", "Female"])
age_group = st.selectbox("Age Group", ["18-24", "25-34", "35-44", "45-54", "55+"])
ad_platform = st.selectbox("Ad Platform", ["Facebook", "Instagram", "TikTok", "Google"])
target_gender = st.selectbox("Target Gender", ["Male", "Female", "All"])
target_age_group = st.selectbox(
    "Target Age Group", ["18-24", "25-34", "35-44", "45-54", "55+", "All"]
)
ad_type = st.selectbox("Ad Type", ["Image", "Video", "Carousel", "Story"])
time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
event_type = st.selectbox("Event Type", ["Impression", "View", "Like", "Click"])

if st.button("Predict"):
    row = {
        "user_gender": user_gender,
        "age_group": age_group,
        "ad_platform": ad_platform,
        "gender_match": 1 if user_gender == target_gender or target_gender == "All" else 0,
        "age_match": 1 if age_group == target_age_group or target_age_group == "All" else 0,
        "hour": {"Morning": 1, "Afternoon": 2, "Evening": 3, "Night": 4}[time_of_day],
        "ad_type": ad_type,
        "interaction_score": {
            "Impression": 1,
            "View": 2,
            "Like": 3,
            "Click": 4
        }[event_type]
    }

    df = prepare_input(row)
    result = predict(df)[0]

    st.success(f"Prediction Result: {result}")
