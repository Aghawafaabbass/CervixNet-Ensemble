import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="CervixNet-Ensemble", page_icon="🩺", layout="wide")

# CSS: Strong Visibility Fix
st.markdown('''
    <style>
    .stApp { background-color: #f8f9fa; }
    [data-testid="stSidebar"] { background-color: #1e272e; color: white; }
    h1, h2, h3, h4, h5, p, span, label { color: #2c3e50 !important; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 15px; border-left: 5px solid #3498db; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .result-card { padding: 25px; border-radius: 15px; margin-bottom: 20px; border: 1px solid #dcdde1; }
    .high-risk { background-color: #ff7675; color: white !important; }
    .low-risk { background-color: #55efc4; color: #2d3436 !important; }
    .result-card h2, .result-card p { color: inherit !important; }
    /* Fix for sidebar labels being dark */
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p { color: white !important; }
    </style>
''', unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    return joblib.load('CervixNet_Ensemble_Final.pkl'), joblib.load('scaler.pkl'), joblib.load('best_threshold.pkl')

model, scaler, threshold = load_assets()

# Sidebar for Inputs
with st.sidebar:
    st.header("📋 Patient Data")
    age = st.number_input("Age", 13, 85, 30)
    partners = st.number_input("Sexual Partners", 1, 20, 2)
    first_sex = st.number_input("Age at First Sex", 10, 40, 18)
    pregnancies = st.number_input("Total Pregnancies", 0, 15, 2)
    smokes_yrs = st.number_input("Smoking Years", 0.0, 50.0, 0.0)
    hormonal_yrs = st.number_input("Hormonal Contraceptives (yrs)", 0.0, 30.0, 0.0)
    st.markdown("---")
    vaccine = st.selectbox("HPV Vaccine", ["No", "Yes"])
    bleeding = st.checkbox("Abnormal Bleeding")
    pain = st.checkbox("Pelvic Pain")

# Data Processing
df = pd.DataFrame([{'Age': age, 'Number of sexual partners': partners, 'First sexual intercourse': first_sex, 
                    'Num of pregnancies': pregnancies, 'Smokes (years)': smokes_yrs, 'Hormonal Contraceptives (years)': hormonal_yrs}])
expected_cols = scaler.feature_names_in_
for col in expected_cols:
    if col not in df.columns: df[col] = 0

# SAFE INFERENCE LOGIC
input_scaled = scaler.transform(df[expected_cols])
proba_output = model.predict_proba(input_scaled)

# This part handles all array/list types safely
if isinstance(proba_output, (list, np.ndarray)):
    # Check if it's a 2D array like [[0.1, 0.9]] or a list of arrays
    res = np.array(proba_output).flatten()
    base_proba = float(res[1]) if len(res) > 1 else float(res[0])
else:
    base_proba = float(proba_output)

# Clinical Adjustments
adj_proba = base_proba + (0.12 if (bleeding or pain) else 0)
if vaccine == "Yes": adj_proba -= 0.05
final_proba = float(np.clip(adj_proba, 0, 1))
risk_percent = final_proba * 100
# Ensure threshold is also a simple float
status = "HIGH RISK" if final_proba >= float(np.array(threshold).flatten()[0]) else "LOW RISK"

# Main Layout
st.title("🩺 CervixNet-Ensemble")
st.markdown("##### Advanced Diagnostic Intelligence Unit")
st.markdown("---")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 📊 Analysis Output")
    if status == "HIGH RISK":
        st.markdown(f'<div class="result-card high-risk"><h2>⚠️ {status}</h2><p>Significant risk markers detected. Clinical referral advised.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-card low-risk"><h2>✅ {status}</h2><p>Patient parameters within safe threshold. Routine screening recommended.</p></div>', unsafe_allow_html=True)

with col_right:
    st.markdown("### 📈 Risk Score")
    st.metric("Probability Index", f"{risk_percent:.2f}%")
    st.progress(final_proba)
    st.info(f"Model Threshold: {float(np.array(threshold).flatten()[0]):.2f}")

st.markdown("---")
st.caption("Developed for Clinical Research | Data-driven Ensemble Architecture")
