#type:ignore
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("BestBreastCancerModel.pkl")

# Streamlit UI
st.title("Heart Disease 10-Year CHD Prediction 🫀")

st.write("Enter patient details to predict the risk of CHD in 10 years.")

# Input fields
male = st.selectbox("Male (1 = Yes, 0 = No)", [0, 1])
age = st.number_input("Age", min_value=20, max_value=100, value=35)
currentSmoker = st.selectbox("Current Smoker (1 = Yes, 0 = No)", [0, 1])
cigsPerDay = st.number_input("Cigarettes per day", min_value=0.0, max_value=100.0, value=0.0)
BPMeds = st.selectbox("On BP Medication (1 = Yes, 0 = No)", [0, 1])
prevalentStroke = st.selectbox("History of Stroke (1 = Yes, 0 = No)", [0, 1])
prevalentHyp = st.selectbox("Hypertension (1 = Yes, 0 = No)", [0, 1])
diabetes = st.selectbox("Diabetes (1 = Yes, 0 = No)", [0, 1])
totChol = st.number_input("Total Cholesterol", min_value=100.0, max_value=500.0, value=200.0)
sysBP = st.number_input("Systolic BP", min_value=80.0, max_value=300.0, value=120.0)
diaBP = st.number_input("Diastolic BP", min_value=50.0, max_value=200.0, value=80.0)
BMI = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
heartRate = st.number_input("Heart Rate", min_value=30.0, max_value=200.0, value=70.0)
glucose = st.number_input("Glucose Level", min_value=50.0, max_value=300.0, value=80.0)

# Prepare input for prediction
input_df = pd.DataFrame([[
    male, age, currentSmoker, cigsPerDay, BPMeds,
    prevalentStroke, prevalentHyp, diabetes, totChol,
    sysBP, diaBP, BMI, heartRate, glucose
]], columns=[
    "male", "age", "currentSmoker", "cigsPerDay", "BPMeds",
    "prevalentStroke", "prevalentHyp", "diabetes", "totChol",
    "sysBP", "diaBP", "BMI", "heartRate", "glucose"
])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None
    
    if prediction == 1:
        st.error("⚠️ High Risk of CHD in 10 years!")
    else:
        st.success("✅ Low Risk of CHD in 10 years.")
    
    if probability is not None:
        st.write(f"**Probability of CHD:** {probability:.2%}")
