import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

model = joblib.load("crime_risk_model.pkl")
crime_lapd = pd.read_csv("crime_20_24_clean.csv")

st.title("LA Crime Risk Predictor")

area = st.selectbox("Area", sorted(crime_lapd["area_name"].unique()))
date = st.date_input("Date")
time = st.time_input("Time")



dt = datetime.combine(date, time)

input_df = pd.DataFrame([{
    "area_name": area,
    "hour": dt.hour,
    "day_of_week": dt.weekday(),
    "month": dt.month
}])

# ----------------------
# Prediction
# ----------------------
if st.button("Predict Risk Level"):
    prediction = model.predict(input_df)[0]

    if prediction == "High":
        st.error("🔴 High Risk")
        st.markdown("""
        **What this means:**  
        - The area and time you selected have historically had a **high number of serious crimes**.  
        """)
    elif prediction == "Medium":
        st.warning("🟡 Medium Risk")
        st.markdown("""
        **What this means:**  
        - The area and time you selected have **moderate crime levels**.   
        """)
    else:
        st.success("🟢 Low Risk")
        st.markdown("""
        **What this means:**  
        - The area and time you selected have historically had **fewer crimes**.  
        """)
