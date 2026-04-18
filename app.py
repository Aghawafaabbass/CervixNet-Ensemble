import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

st.set_page_config(page_title="CervixNet-Ensemble", page_icon="🩺", layout="wide")

st.title("🩺 CervixNet-Ensemble")
st.subheader("Advanced Cervical Cancer Risk Prediction")

@st.cache_resource
def load_model():
    model = joblib.load('CervixNet_Ensemble_Final.pkl')
    scaler = joblib.load('scaler.pkl')
    threshold = joblib.load('best_threshold.pkl')
    return model, scaler, threshold

try:
    model, scaler, threshold = load_model()
    st.success("Model Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# ... (Baaki code jo aapne pehle likha tha)
