
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="CervixNet-Ensemble",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Styling
st.markdown('''
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #00b894; color: white; border-radius: 10px; height: 3.2em; font-weight: bold;}
    .result-high {background-color: #ffebee; padding: 25px; border-radius: 15px; border-left: 8px solid #d32f2f;}
    .result-low  {background-color: #e8f5e9; padding: 25px; border-radius: 15px; border-left: 8px solid #2e7d32;}
    </style>
''', unsafe_allow_html=True)

st.title("🩺 CervixNet-Ensemble")
st.subheader("Advanced Cervical Cancer Risk Prediction")
st.caption("Stacked Meta-Learning MLOps Pipeline | Explainable AI")

# Load Model
@st.cache_resource
def load_model():
    model = joblib.load('CervixNet_Ensemble_Final.pkl')
    scaler = joblib.load('scaler.pkl')
    threshold = joblib.load('best_threshold.pkl')
    return model, scaler, threshold

model, scaler, threshold = load_model()

# Sidebar Inputs
with st.sidebar:
    st.header("📋 Patient Risk Profile")
    st.markdown("---")
    
    age = st.slider("Age", 13, 84, 35)
    partners = st.number_input("Number of Sexual Partners", 1, 20, 2)
    first_sex = st.number_input("Age at First Sexual Intercourse", 10, 40, 18)
    pregnancies = st.number_input("Number of Pregnancies", 0, 15, 2)
    
    smokes = st.selectbox("Smokes?", ["No", "Yes"])
    stds = st.selectbox("History of STDs?", ["No", "Yes"])
    
    hormonal = st.number_input("Hormonal Contraceptives (years)", 0.0, 30.0, 5.0, step=0.1)
    iud = st.number_input("IUD Usage (years)", 0.0, 20.0, 0.0, step=0.1)

# Prepare Data
input_data = pd.DataFrame([{
    'Age': age,
    'Number of sexual partners': partners,
    'First sexual intercourse': first_sex,
    'Num of pregnancies': pregnancies,
    'Smokes': 1 if smokes == "Yes" else 0,
    'Hormonal Contraceptives (years)': hormonal,
    'IUD (years)': iud,
    'STDs': 1 if stds == "Yes" else 0,
    'Smokes (years)': 0,
    'Smokes (packs/year)': 0,
    'Hormonal Contraceptives': 1 if hormonal > 0 else 0,
    'IUD': 1 if iud > 0 else 0,
    'STDs (number)': 1 if stds == "Yes" else 0,
}])

input_data['Age_at_First_Sex'] = input_data['First sexual intercourse']
input_data['Sexual_Partners_per_Pregnancy'] = input_data['Number of sexual partners'] / (input_data['Num of pregnancies'] + 1)
input_data['Smoking_Intensity'] = 0
input_data['Hormonal_Exposure'] = hormonal + iud

# Prediction
input_scaled = scaler.transform(input_data)
proba = model.predict_proba(input_scaled)[0][1]
prediction = 1 if proba >= threshold else 0

# Show Result
st.subheader("🧬 Prediction Result")
if prediction == 1:
    st.markdown(f'''
        <div class="result-high">
            <h2>⚠️ HIGH RISK DETECTED</h2>
            <h3>Probability: {proba:.1%}</h3>
            <p><strong>Recommendation:</strong> Immediate clinical consultation is advised.</p>
        </div>
    ''', unsafe_allow_html=True)
else:
    st.markdown(f'''
        <div class="result-low">
            <h2>✅ LOW RISK</h2>
            <h3>Probability: {proba:.1%}</h3>
        </div>
    ''', unsafe_allow_html=True)

st.metric("Decision Threshold", f"{threshold:.2f}")

# SHAP Visualization
st.subheader("🔍 Feature Contribution (SHAP)")
explainer = shap.TreeExplainer(model.estimators_[0])
shap_values = explainer.shap_values(input_scaled)
fig = shap.force_plot(explainer.expected_value, shap_values[0], input_data.iloc[0], matplotlib=True, show=False)
st.pyplot(fig)

st.info("Model: Stacked Ensemble (XGBoost + LightGBM + CatBoost + RF) | CV F1: 95.94%")
