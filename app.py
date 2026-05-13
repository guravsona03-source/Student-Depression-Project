import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the model and scaler
# These files must be in the same GitHub folder!

model = joblib.load('model.joblib')
oe = joblib.load('ordinal.joblib')
scaler = joblib.load('scaler.joblib')

st.title("Student Depression Analysis")


# -----------------------------------
# Numerical Inputs
# -----------------------------------

Age = st.number_input(
    "Age",
    min_value=15,
    max_value=40,
    value=20
)

AcademicPressure = st.slider(
    "Academic Pressure",
    0,
    10,
    5
)

WorkPressure = st.slider(
    "Work Pressure",
    0,
    10,
    2
)

CGPA = st.number_input(
    "CGPA",
    min_value=0.0,
    max_value=10.0,
    value=7.0
)

StudySatisfaction = st.slider(
    "Study Satisfaction",
    0,
    10,
    5
)

JobSatisfaction = st.slider(
    "Job Satisfaction",
    0,
    10,
    5
)

SleepDuration = st.number_input(
    "Sleep Duration",
    min_value=4,
    max_value=12,
    value=7
)

WorkStudyHours = st.number_input(
    "Work/Study Hours",
    min_value=0,
    max_value=24,
    value=4
)

FinancialStress = st.slider(
    "Financial Stress",
    0,
    10,
    5
)

# -----------------------------------
# Categorical Inputs
# -----------------------------------

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

DietaryHabits = st.selectbox(
    "Dietary Habits",
    ["Healthy", "Moderate", "Unhealthy"]
)

Degree = st.selectbox(
    "Degree",
    ["BSc", "B.Tech", "B.Pharm", "BCA", "MSc", "MBA"]
)

SuicidalThoughts = st.selectbox(
    "Have you ever had suicidal thoughts ?",
    ["Yes", "No"]
)

FamilyHistory = st.selectbox(
    "Family History of Mental Illness",
    ["Yes", "No"]
)

if st.button("Click here to get the Prediction"):

    # Categorical columns
    cat_col = [
        Gender,
        DietaryHabits,
        Degree,
        SuicidalThoughts,
        FamilyHistory
    ]

    # Numerical columns
    num_col = [
        Age,
        AcademicPressure,
        WorkPressure,
        CGPA,
        StudySatisfaction,
        JobSatisfaction,
        SleepDuration,
        WorkStudyHours,
        FinancialStress
    ]

    # Scale numerical features
    scaled_num = scaler.transform([num_col])

    # Encode categorical features
    encoded_cat = oe.transform([cat_col])

    # Combine both numerical + categorical
    final_features = np.concatenate((scaled_num, encoded_cat), axis=1)

    # Prediction
    prediction = model.predict(final_features)

    # Result
    if prediction[0] == 1:
        st.error("Student may have Depression")
    else:
        st.success("Student may not have Depression")