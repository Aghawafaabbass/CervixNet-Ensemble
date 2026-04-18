import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="CervixNet-Ensemble AI", page_icon="🩺", layout="centered")

# Custom CSS for Professional Look
st.markdown('''
    <style>
    .stProgress > div > div > div > div { background-color: #00b894; }
    .report-card { padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; background-color: #ffffff; }
    </style>
''', unsafe_allow_html=True)

@st.cache_resource
def load_all():
    return joblib.load('CervixNet_Ensemble_Final.pkl'), joblib.load('scaler.pkl'), joblib.load('best_threshold.pkl')

model, scaler, threshold = load_all()

st.title("🩺 CervixNet-Ensemble")
st.markdown("### **AI-Powered Cervical Cancer Screening**")
st.info("Meta-Learning Pipeline: XGBoost + LightGBM + CatBoost")

# Input Section
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Patient Age", 13, 85, 30)
    partners = st.number_input("Sexual Partners", 1, 20, 2)
    first_sex = st.number_input("Age at First Sex", 10, 40, 18)
with col2:
    pregnancies = st.number_input("Total Pregnancies", 0, 15, 2)
    smokes_yrs = st.number_input("Smoking Years", 0.0, 50.0, 0.0)
    hormonal_yrs = st.number_input("Hormonal Contraceptives (yrs)", 0.0, 30.0, 0.0)

# Process Data
input_dict = {'Age': age, 'Number of sexual partners': partners, 'First sexual intercourse': first_sex, 
              'Num of pregnancies': pregnancies, 'Smokes (years)': smokes_yrs, 
              'Hormonal Contraceptives (years)': hormonal_yrs}

# Base Data Frame
df = pd.DataFrame([input_dict])
expected_cols = scaler.feature_names_in_
for col in expected_cols:
    if col not in df.columns: df[col] = 0

# Prediction
input_scaled = scaler.transform(df[expected_cols])
proba = model.predict_proba(input_scaled)[0][1]

# Display Result with Style
st.divider()
st.subheader("📊 Diagnostic Report")

risk_score = proba * 100
st.write(f"**Calculated Risk Score:** {risk_score:.2f}%")
st.progress(proba)

if proba >= threshold:
    st.error("### ⚠️ Result: HIGH RISK")
    st.warning("**Recommendation:** Immediate referral for Colposcopy or Pap Smear is highly advised.")
else:
    st.success("### ✅ Result: LOW RISK")
    st.info("Maintain regular checkups as per standard medical guidelines.")

st.caption("Disclaimer: This is an AI research tool. Always consult a certified medical professional.")
