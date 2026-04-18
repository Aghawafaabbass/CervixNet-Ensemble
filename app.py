import streamlit as st
import pandas as pd
import joblib
import numpy as np
from fpdf import FPDF

st.set_page_config(page_title="CervixNet-Ensemble", page_icon="🩺", layout="wide")

# CSS for a Clean, Medical Interface
st.markdown('''
    <style>
    .main { background-color: #f4f7f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stAlert { border-radius: 12px; }
    h1 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; }
    </style>
''', unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    return joblib.load('CervixNet_Ensemble_Final.pkl'), joblib.load('scaler.pkl'), joblib.load('best_threshold.pkl')

model, scaler, threshold = load_assets()

def generate_report(data, risk, status, symptoms):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 15, txt="CervixNet-Ensemble Diagnostic Report", ln=True, align='C')
    pdf.line(10, 25, 200, 25)
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    for k, v in data.items(): pdf.cell(100, 10, txt=f"{k}: {v}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=f"Analysis Result: {status}", ln=True)
    pdf.cell(200, 10, txt=f"Probability: {risk:.2f}%", ln=True)
    return pdf.output(dest='S').encode('latin-1')

st.title("🩺 CervixNet-Ensemble")
st.caption("Advanced Cervical Cancer Risk Prediction | Stacked Meta-Learning Pipeline")
st.markdown("---")

# Organized Input Sections
with st.container():
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("##### 🧬 Biological Profile")
        age = st.number_input("Patient Age", 13, 85, 30)
        partners = st.number_input("Sexual Partners", 1, 20, 2)
        first_sex = st.number_input("Age at First Sex", 10, 40, 18)
    with c2:
        st.markdown("##### 🚬 Lifestyle Factors")
        pregnancies = st.number_input("Total Pregnancies", 0, 15, 2)
        smokes_yrs = st.number_input("Smoking Years", 0.0, 50.0, 0.0)
        hormonal_yrs = st.number_input("Hormonal Contraceptives (yrs)", 0.0, 30.0, 0.0)
    with c3:
        st.markdown("##### 🏥 Clinical Indicators")
        vaccine = st.selectbox("HPV Vaccine", ["No", "Yes"])
        bleeding = st.checkbox("Abnormal Bleeding")
        pain = st.checkbox("Pelvic Pain")

# Data Processing
symptoms = []
if bleeding: symptoms.append("Bleeding")
if pain: symptoms.append("Pain")

df = pd.DataFrame([{'Age': age, 'Number of sexual partners': partners, 'First sexual intercourse': first_sex, 
                    'Num of pregnancies': pregnancies, 'Smokes (years)': smokes_yrs, 'Hormonal Contraceptives (years)': hormonal_yrs}])
expected_cols = scaler.feature_names_in_
for col in expected_cols:
    if col not in df.columns: df[col] = 0

# Model Inference
input_scaled = scaler.transform(df[expected_cols])
base_proba = float(model.predict_proba(input_scaled)[:, 1])

# Clinical Adjustment Logic
adjusted_proba = base_proba + (0.12 if symptoms else 0)
if vaccine == "Yes": adjusted_proba -= 0.05
final_proba = float(np.clip(adjusted_proba, 0, 1))
risk_percent = final_proba * 100
status = "HIGH RISK" if final_proba >= float(threshold) else "LOW RISK"

# Professional Output Display
st.markdown("### 📊 Diagnostic Analysis")
r1, r2 = st.columns([2, 1])

with r1:
    if status == "HIGH RISK":
        st.error(f"#### STATUS: {status}")
        st.write("The ensemble model indicates significant risk factors. Clinical follow-up (Pap/Colposcopy) is recommended.")
    else:
        st.success(f"#### STATUS: {status}")
        st.write("Patient is within the low-risk threshold. Continue standard preventative screening.")

with r2:
    st.metric("Risk Probability", f"{risk_percent:.2f}%")
    report_data = {"Age": age, "Partners": partners, "Smoking Years": smokes_yrs, "Hormonal": hormonal_yrs}
    pdf_bytes = generate_report(report_data, risk_percent, status, symptoms)
    st.download_button("📥 Export Report (PDF)", pdf_bytes, "CervixNet_Report.pdf", "application/pdf")

st.markdown("---")
st.info("⚠️ **Note:** This tool is for research purposes. Model F1-Score: 0.959 | Data: 858 clinical specimens.")
