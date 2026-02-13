import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

@st.cache_resource
def load_model():
    return joblib.load("crime_risk_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("crime_20_24_clean.csv")

model = load_model()
crime_lapd = load_data()

st.title("LA Crime Risk Predictor")
st.markdown("""
This project looks at crime risk across Los Angeles using LAPD data. Crimes are 
divided into Part 1 (more serious) and Part 2 (less serious) offenses. Each area gets 
a monthly risk score that combines the total number of crimes group by hour and date with 
the number of Part 1 offenses, and these scores are labeled as Low, Medium, or High 
risk.

The model uses the location, time of day, day of the week, and month to estimate the 
risk level at a specific place and time. By training a Random Forest classifier on 
this information, it can predict whether a particular area at a given moment is likely 
to be safer or more dangerous, giving users a clear, data-driven view of crime patterns 
throughout the city.
""")
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
