import streamlit as st
import pandas as pd
import joblib
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

st.set_page_config(page_title="CervixNet-Ensemble", page_icon="🩺", layout="wide")

# Professional Styling
st.markdown('<style>.main {background-color: #f8f9fa;}</style>', unsafe_allow_html=True)

st.title("🩺 CervixNet-Ensemble")
st.subheader("Advanced Cervical Cancer Risk Prediction")

@st.cache_resource
def load_model():
    model = joblib.load('CervixNet_Ensemble_Final.pkl')
    scaler = joblib.load('scaler.pkl')
    threshold = joblib.load('best_threshold.pkl')
    return model, scaler, threshold

model, scaler, threshold = load_model()
st.sidebar.success("✅ Model Loaded Successfully!")

# Sidebar Inputs
with st.sidebar:
    st.header("📋 Patient Risk Profile")
    age = st.slider("Age", 13, 84, 35)
    partners = st.number_input("Number of Sexual Partners", 1, 20, 2)
    first_sex = st.number_input("Age at First Sexual Intercourse", 10, 40, 18)
    pregnancies = st.number_input("Number of Pregnancies", 0, 15, 2)
    smokes = st.selectbox("Smokes?", ["No", "Yes"])
    stds = st.selectbox("History of STDs?", ["No", "Yes"])
    hormonal = st.number_input("Hormonal Contraceptives (years)", 0.0, 30.0, 5.0)
    iud = st.number_input("IUD Usage (years)", 0.0, 20.0, 0.0)

# 1. Create DataFrame with EXACT columns as per Scaler requirement
# Scaler usually expects the exact same column names in the same order
input_dict = {
    'Age': age,
    'Number of sexual partners': partners,
    'First sexual intercourse': first_sex,
    'Num of pregnancies': pregnancies,
    'Smokes': 1 if smokes == "Yes" else 0,
    'Smokes (years)': 0.0,
    'Smokes (packs/year)': 0.0,
    'Hormonal Contraceptives': 1 if hormonal > 0 else 0,
    'Hormonal Contraceptives (years)': hormonal,
    'IUD': 1 if iud > 0 else 0,
    'IUD (years)': iud,
    'STDs': 1 if stds == "Yes" else 0,
    'STDs (number)': 1 if stds == "Yes" else 1 if stds == "Yes" else 0,
}

input_df = pd.DataFrame([input_dict])

# 2. Add Engineered Features (Ensure these were in your training set)
input_df['Age_at_First_Sex'] = input_df['First sexual intercourse']
input_df['Sexual_Partners_per_Pregnancy'] = input_df['Number of sexual partners'] / (input_df['Num of pregnancies'] + 1)
input_df['Smoking_Intensity'] = input_df['Smokes (years)'] * input_df['Smokes (packs/year)']
input_df['Hormonal_Exposure'] = input_df['Hormonal Contraceptives (years)'] + input_df['IUD (years)']

# IMPORTANT: Sklearn's check_feature_names is failing. 
# We must ensure the column order matches the training data.
# Re-ordering columns to match what the scaler expects:
try:
    expected_features = scaler.feature_names_in_
    input_df = input_df[expected_features]
except:
    pass

# Prediction logic
try:
    input_scaled = scaler.transform(input_df)
    # Handle the probability output (extracting the float)
    proba_array = model.predict_proba(input_scaled)
    proba = proba_array[0][1] if len(proba_array[0]) > 1 else proba_array[0][0]
    prediction = 1 if proba >= threshold else 0

    # Show Result
    st.subheader("🧬 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ HIGH RISK DETECTED (Probability: {proba:.1%})")
    else:
        st.success(f"✅ LOW RISK (Probability: {proba:.1%})")
        
except Exception as e:
    st.error(f"Prediction Error: {e}")
    st.info("Check if the model and scaler column names match.")
