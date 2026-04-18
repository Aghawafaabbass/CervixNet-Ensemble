import streamlit as st
import pandas as pd
import joblib
import numpy as np
from fpdf import FPDF
import base64

st.set_page_config(page_title="CervixNet-Ensemble AI", page_icon="🩺", layout="centered")

@st.cache_resource
def load_all():
    return joblib.load('CervixNet_Ensemble_Final.pkl'), joblib.load('scaler.pkl'), joblib.load('best_threshold.pkl')

model, scaler, threshold = load_all()

def create_pdf(p_data, p_risk, p_status):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="CervixNet-Ensemble Diagnostic Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    for key, value in p_data.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=f"Risk Score: {p_risk:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"Status: {p_status}", ln=True)
    return pdf.output(dest='S').encode('latin-1')

st.title("🩺 CervixNet-Ensemble")
st.markdown("### **AI-Powered Cervical Cancer Screening**")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Patient Age", 13, 85, 30)
    partners = st.number_input("Sexual Partners", 1, 20, 2)
    first_sex = st.number_input("Age at First Sex", 10, 40, 18)
with col2:
    pregnancies = st.number_input("Total Pregnancies", 0, 15, 2)
    smokes_yrs = st.number_input("Smoking Years", 0.0, 50.0, 0.0)
    hormonal_yrs = st.number_input("Hormonal Contraceptives (yrs)", 0.0, 30.0, 0.0)

input_dict = {'Age': age, 'Partners': partners, 'First Sex': first_sex, 'Pregnancies': pregnancies, 'Smoking': smokes_yrs, 'Hormonal': hormonal_yrs}

# Alignment & Prediction
df = pd.DataFrame([{'Age': age, 'Number of sexual partners': partners, 'First sexual intercourse': first_sex, 
                    'Num of pregnancies': pregnancies, 'Smokes (years)': smokes_yrs, 'Hormonal Contraceptives (years)': hormonal_yrs}])
expected_cols = scaler.feature_names_in_
for col in expected_cols:
    if col not in df.columns: df[col] = 0

input_scaled = scaler.transform(df[expected_cols])
proba = model.predict_proba(input_scaled)[0][1]

st.divider()
risk_score = proba * 100
status = "HIGH RISK" if proba >= threshold else "LOW RISK"

if proba >= threshold:
    st.error(f"### ⚠️ Result: {status} ({risk_score:.2f}%)")
else:
    st.success(f"### ✅ Result: {status} ({risk_score:.2f}%)")

# Download Button
pdf_bytes = create_pdf(input_dict, risk_score, status)
st.download_button(label="📥 Download Diagnostic Report (PDF)", 
                   data=pdf_bytes, 
                   file_name="CervixNet_Report.pdf", 
                   mime="application/pdf")
