import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="CervixNet-Ensemble", page_icon="🩺", layout="wide")

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

# Sidebar
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

# 1. Base Features
data = {
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
    'STDs (number)': 1 if stds == "Yes" else 0,
}

df = pd.DataFrame([data])

# 2. Add ALL missing STD columns (Setting them to 0 as default)
missing_std_cols = [
    'STDs:condylomatosis', 'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis',
    'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis', 'STDs:pelvic inflammatory disease',
    'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV',
    'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis', 'STDs: Time since first diagnosis',
    'STDs: Time since last diagnosis'
]
for col in missing_std_cols:
    df[col] = 0

# 3. Engineered Features
df['Age_at_First_Sex'] = df['First sexual intercourse']
df['Sexual_Partners_per_Pregnancy'] = df['Number of sexual partners'] / (df['Num of pregnancies'] + 1)
df['Smoking_Intensity'] = 0.0
df['Hormonal_Exposure'] = hormonal + iud

# 4. Final Alignment with Scaler
try:
    # Scaler se columns ka sahi order lein
    expected_cols = scaler.feature_names_in_
    # Jo columns hamare paas nahi hain unhe 0 se fill karein
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
            
    final_df = df[expected_cols]
    
    # Prediction
    input_scaled = scaler.transform(final_df)
    proba = model.predict_proba(input_scaled)[0][1]
    prediction = 1 if proba >= threshold else 0

    st.subheader("🧬 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ HIGH RISK DETECTED (Probability: {proba:.1%})")
    else:
        st.success(f"✅ LOW RISK (Probability: {proba:.1%})")
        
except Exception as e:
    st.error(f"Feature Error: {e}")
