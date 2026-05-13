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

# 2. Input fields

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)
input_df = pd.DataFrame([[Gender]], columns=['Gender'])
encoded = oe.transform(input_df)
final_Gender_value = encoded[0][0]

DietaryHabits = st.selectbox(
    "Dietary Habits",
    ["Healthy", "Moderate", "Unhealthy"]
)
input_df = pd.DataFrame([[DietaryHabits]], columns=["Dietary Habits"])
encoded = oe.transform(input_df)
final_DH_value = encoded[0][0]

Degree = st.selectbox(
    "Degree",
    ["BSc", "B.Tech", "B.Pharm", "BCA", "MSc", "MBA"]
)
input_df = pd.DataFrame([[Degree]], columns=["Degree"])
encoded = oe.transform(input_df)
final_Degree_value = encoded[0][0]

SuicidalThoughts = st.selectbox(
    "Have you ever had suicidal thoughts ?",
    ["Yes", "No"]
)
input_df = pd.DataFrame([[SuicidalThoughts]], columns=["Have you ever had suicidal thoughts ?"])
encoded = oe.transform(input_df)
final_ST_value = encoded[0][0]

FamilyHistory = st.selectbox(
    "Family History of Mental Illness",
    ["Yes", "No"]
)
input_df = pd.DataFrame([[FamilyHistory]], columns=["Family History of Mental Illness"])
encoded = oe.transform(input_df)
final_MI_value = encoded[0][0]

Age = st.number_input(
    "Age",
    min_value=15,
    max_value=40,
    value=20
)
input_array = np.array(Age).reshape(1, -1)
scaled_age = scaler.transform(input_array)
scaled_age = scaled_age[0][0]

AcademicPressure = st.slider(
    "Academic Pressure",
    0,
    10,
    5
)
input_array = np.array(AcademicPressure).reshape(1, -1)
scaled_AcademicPressure = scaler.transform(input_array)
scaled_AcademicPressure = scaled_AcademicPressure[0][0]

WorkPressure = st.slider(
    "Work Pressure",
    0,
    10,
    2
)
input_array = np.array(WorkPressure).reshape(1, -1)
scaled_WorkPressure = scaler.transform(input_array)
scaled_WorkPressure =scaled_WorkPressure[0][0]

CGPA = st.number_input(
    "CGPA",
    min_value=0.0,
    max_value=10.0,
    value=7.0
)
input_array = np.array(CGPA).reshape(1, -1)
scaled_CGPA= scaler.transform(input_array)
scaled_CGPA =scaled_CGPA[0][0]

StudySatisfaction = st.slider(
    "Study Satisfaction",
    0,
    10,
    5
)
input_array = np.array(StudySatisfaction).reshape(1, -1)
scaled_StudySatisfaction = scaler.transform(input_array)
scaled_StudySatisfaction = scaled_StudySatisfaction[0][0]

JobSatisfaction = st.slider(
    "Job Satisfaction",
    0,
    10,
    5
)
input_array = np.array(JobSatisfaction).reshape(1, -1)
scaled_JobSatisfaction = scaler.transform(input_array)
scaled_JobSatisfaction =scaled_JobSatisfaction[0][0]

SleepDuration = st.number_input(
    "Sleep Duration",
    min_value=4,
    max_value=12,
    value=7
)
input_array = np.array(SleepDuration).reshape(1, -1)
scaled_SleepDuration = scaler.transform(input_array)
scaled_SleepDuration =scaled_SleepDuration[0][0]

WorkStudyHours = st.number_input(
    "Work/Study Hours",
    min_value=0,
    max_value=24,
    value=4
)
input_array = np.array(WorkStudyHours).reshape(1, -1)
scaled_WorkStudyHours = scaler.transform(input_array)
scaled_WorkStudyHours = scaled_WorkStudyHours[0][0]

FinancialStress = st.slider(
    "Financial Stress",
    0,
    10,
    5
)
input_array = np.array(FinancialStress).reshape(1, -1)
scaled_FinancialStress = scaler.transform(input_array)
scaled_FinancialStress =scaled_FinancialStress[0][0]

if st.button("Predict"):
    # 3. Create feature list (Ensure order matches your training data!)
    features = [[final_Gender_value ,final_DH_value , final_Degree_value , final_ST_value , final_MI_value , scaled_age,scaled_AcademicPressure,scaled_WorkPressure ,scaled_CGPA,scaled_StudySatisfaction,scaled_JobSatisfaction,scaled_SleepDuration, scaled_WorkStudyHours, scaled_FinancialStress]] 
    
    # 4. Predict
    
    prediction = model.predict(scaled_features)
    
    if prediction[0] == 1:
        st.error("Student may have Depression")
    else:
        st.success("Student may not have Depression")