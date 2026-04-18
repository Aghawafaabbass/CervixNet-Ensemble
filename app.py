import streamlit as st
import pandas as pd
import joblib
import numpy as np
from fpdf import FPDF

st.set_page_config(page_title="CervixNet-Ensemble Pro", page_icon="🩺", layout="wide")

@st.cache_resource
def load_all():
    return joblib.load('CervixNet_Ensemble_Final.pkl'), joblib.load('scaler.pkl'), joblib.load('best_threshold.pkl')

model, scaler, threshold = load_all()

def create_pdf(p_data, p_risk, p_status, symptoms):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="CervixNet Clinical Diagnostic Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=11)
    for key, value in p_data.items():
        pdf.cell(200, 8, txt=f"{key}: {value}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt=f"Reported Symptoms: {', '.join(symptoms) if symptoms else 'None'}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    color = (200, 0, 0) if p_status == "HIGH RISK" else (0, 128, 0)
    pdf.set_text_color(*color)
    pdf.cell(200, 10, txt=f"FINAL RISK ASSESSMENT: {p_status} ({p_risk:.2f}%)", ln=True)
    return pdf.output(dest='S').encode('latin-1')

st.title("🩺 CervixNet-Ensemble Pro")
st.markdown("### **Clinical Decision Support System**")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Patient Age", 13, 85, 30)
    partners = st.number_input("Sexual Partners", 1, 20, 2)
    first_sex = st.number_input("Age at First Sex", 10, 40, 18)
with col2:
    pregnancies = st.number_input("Total Pregnancies", 0, 15, 2)
    smokes_yrs = st.number_input("Smoking Years", 0.0, 50.0, 0.0)
    hormonal_yrs = st.number_input("Hormonal Contraceptives (yrs)", 0.0, 30.0, 0.0)
with col3:
    vaccine = st.selectbox("HPV Vaccine Taken?", ["No", "Yes"])
    symptom_list = []
    if st.checkbox("Abnormal Bleeding"): symptom_list.append("Abnormal Bleeding")
    if st.checkbox("Pelvic Pain"): symptom_list.append("Pelvic Pain")

df = pd.DataFrame([{'Age': age, 'Number of sexual partners': partners, 'First sexual intercourse': first_sex, 
                    'Num of pregnancies': pregnancies, 'Smokes (years)': smokes_yrs, 'Hormonal Contraceptives (years)': hormonal_yrs}])
expected_cols = scaler.feature_names_in_
for col in expected_cols:
    if col not in df.columns: df[col] = 0

input_scaled = scaler.transform(df[expected_cols])
# .flatten()[1] to get the specific probability of class 1
base_proba = model.predict_proba(input_scaled).flatten()[1]

# Clinical Adjustments
adjusted_proba = base_proba + (0.15 if symptom_list else 0)
if vaccine == "Yes": adjusted_proba -= 0.05
adjusted_proba = float(np.clip(adjusted_proba, 0, 1))

st.divider()
risk_score = adjusted_proba * 100
status = "HIGH RISK" if adjusted_proba >= float(threshold) else "LOW RISK"

res_col1, res_col2 = st.columns(2)
with res_col1:
    if status == "HIGH RISK":
        st.error(f"## ⚠️ Result: {status}")
        st.write(f"**AI Risk Score:** {risk_score:.2f}%")
    else:
        st.success(f"## ✅ Result: {status}")
        st.write(f"**AI Risk Score:** {risk_score:.2f}%")

with res_col2:
    st.metric("Probability", f"{risk_score:.1f}%")
    pdf_data = {'Age': age, 'Partners': partners, 'First Sex': first_sex, 'Pregnancies': pregnancies, 'Vaccine': vaccine}
    pdf_bytes = create_pdf(pdf_data, risk_score, status, symptom_list)
    st.download_button("📥 Download Clinical Report", pdf_bytes, "CervixNet_Clinical.pdf", "application/pdf")
