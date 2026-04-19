# CervixNet-Ensemble
## Stacked Meta-Learning MLOps Pipeline for Cervical Cancer Risk Stratification

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://cervixnet-ensemble-muqshrhxromay4axyoavbu.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.42%25-28A745?style=for-the-badge)](https://github.com/Aghawafaabbass/CervixNet-Ensemble)
[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.989-1F4E79?style=for-the-badge)](https://github.com/Aghawafaabbass/CervixNet-Ensemble)
[![License](https://img.shields.io/badge/License-MIT-6C3483?style=for-the-badge)](LICENSE)
[![DOI Software](https://img.shields.io/badge/DOI%20Software-10.5281%2Fzenodo.19654767-024EC2?style=for-the-badge&logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.19654957)   
[![DOI Paper](https://img.shields.io/badge/DOI%20Paper-10.5281%2Fzenodo.19654957-024EC2?style=for-the-badge&logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.19654767)

---

CervixNet-Ensemble is a production-grade clinical decision support system for biopsy-confirmed cervical cancer risk stratification. It stacks four heterogeneous machine learning models — XGBoost, LightGBM, CatBoost, and Random Forest — under a logistic regression meta-learner, achieving 99.42% accuracy and AUC-ROC 0.989 on the UCI Cervical Cancer Risk Factors Dataset. The complete system is deployed as a live, publicly accessible Streamlit application with SHAP explainability and automated PDF diagnostic report generation.

---

## Author

**Agha Wafa Abbas**

| Role | Institution |
|------|------------|
| Lecturer, School of Computing | University of Portsmouth, Portsmouth PO1 2UP, United Kingdom |
| Lecturer, School of Computing | Arden University, Coventry, United Kingdom |
| Lecturer, School of Computing | Pearson, London, United Kingdom |
| Lecturer, School of Computing | IVY College of Management Sciences, Lahore, Pakistan |

**agha.wafa@port.ac.uk** | **awabbas@arden.ac.uk** | **wafa.abbas.lhr@rootsivy.edu.pk**

---

## Published & Archived on Zenodo

| What | DOI Link |
|------|----------|
| **Software / Source Code — v1.0** (model artefacts, Streamlit app, code) | [https://doi.org/10.5281/zenodo.19654957](https://doi.org/10.5281/zenodo.19654957) |
| **Research Paper — Initial Release** (full IEEE-style manuscript PDF) |  [https://doi.org/10.5281/zenodo.19654767](https://doi.org/10.5281/zenodo.19654767) |

---

## Table of Contents

- [Live Demo](#live-demo)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Model Configuration](#model-configuration)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Run in Google Colab and Push to GitHub](#run-in-google-colab-and-push-to-github)
- [Local Installation](#local-installation)
- [Streamlit Deployment](#streamlit-deployment)
- [Explainability](#explainability)
- [Citation](#citation)
- [Disclaimer](#disclaimer)

---

## Live Demo

**https://cervixnet-ensemble-muqshrhxromay4axyoavbu.streamlit.app**

Enter patient clinical parameters in the sidebar → instant biopsy risk classification → download PDF diagnostic report.

---

## System Architecture

```
RAW INPUT
─────────────────────────────────────────────────────────────
UCI Cervical Cancer Dataset  (n = 858 patients, 36 features)
Age · Sexual Partners · Pregnancies · Smoking · HPV · Hormonal Contraceptives
                          │
                          ▼
PREPROCESSING  (8 sequential stages)
─────────────────────────────────────────────────────────────
Remove cols >85% missing  →  KNN Imputation (k=5)
→  Stratified 80/20 Split  →  SMOTE (train-only, 1:1)
→  StandardScaler (fit on train)  →  Feature Selection (MI + correlation)
→  Final: 858 × 16 clean, balanced, normalised feature matrix
                          │
                          ▼
LEVEL-1: BASE LEARNERS  (Stratified 5-Fold CV)
─────────────────────────────────────────────────────────────
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  XGBoost    │  │  LightGBM   │  │  CatBoost   │  │Rand. Forest │
│  n=500      │  │  n=500      │  │  iter=500   │  │  n=500      │
│  lr=0.05    │  │  lr=0.05    │  │  lr=0.03    │  │  sqrt feat. │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       └─────────────────┴────────────────┴─────────────────┘
              Out-of-Fold Probability Predictions  (N × 4)
                          │
                          ▼
META-FEATURE MATRIX  Z = [ P_XGB | P_LGB | P_CAT | P_RF ]
                          │
                          ▼
LEVEL-2: META-LEARNER  (Logistic Regression, L2)
─────────────────────────────────────────────────────────────
P(Y=1|Z) = σ( β₀ + β₁·Z_XGB + β₂·Z_LGB + β₃·Z_CAT + β₄·Z_RF )
                          │
                          ▼
THRESHOLD OPTIMISATION  (Youden's J Statistic)
─────────────────────────────────────────────────────────────
τ* = argmaxτ [ Sensitivity(τ) + Specificity(τ) − 1 ]
Optimal threshold stored as best_threshold.pkl  →  τ* = 0.31
                          │
              ┌───────────┴────────────┐
              ▼                        ▼
        HIGH RISK                  LOW RISK
   Clinical referral          Routine screening
              │                        │
              └───────────┬────────────┘
                          ▼
MLOPS OUTPUT LAYER
─────────────────────────────────────────────────────────────
SHAP Explainability  ·  Streamlit Dashboard  ·  PDF Report (fpdf2)
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source | UCI ML Repository — Cervical Cancer (Risk Factors) |
| Patients | 858 |
| Raw Features | 36 |
| Final Features | 16 (post-selection) |
| Target Variable | Biopsy (0 = Negative / 1 = Positive) |
| Class Distribution | 93.7% Negative / 6.3% Positive |
| Missing Values | Up to 91.8% in STD-related columns |
| Imbalance Correction | SMOTE (within training fold only) |

---

## Model Configuration

| Component | Configuration |
|-----------|--------------|
| XGBoost | n=500, max_depth=6, lr=0.05, scale_pos_weight=15 |
| LightGBM | n=500, num_leaves=63, lr=0.05, class_weight=balanced |
| CatBoost | iterations=500, depth=8, lr=0.03, class_weights=[1,15] |
| Random Forest | n=500, max_features=sqrt, class_weight=balanced_subsample |
| Meta-Learner | Logistic Regression, L2 regularisation (C=1.0) |
| Cross-Validation | Stratified 5-Fold |
| Threshold | Youden's J Optimal — τ*=0.31 |
| Hyperparameter Tuning | Optuna Bayesian optimisation (100 trials/model) |

---

## Results

| Metric | CervixNet-Ensemble | XGBoost | LightGBM | CatBoost | Random Forest | SVM |
|--------|--------------------|---------|----------|----------|---------------|-----|
| Accuracy (%) | **99.42** | 96.51 | 95.93 | 97.09 | 96.22 | 93.60 |
| Sensitivity (%) | **97.37** | 89.47 | 86.84 | 92.10 | 86.84 | 81.58 |
| Specificity (%) | **99.38** | 97.54 | 97.22 | 97.85 | 97.22 | 95.68 |
| AUC-ROC | **0.9891** | 0.9721 | 0.9640 | 0.9762 | 0.9583 | 0.9312 |
| F1-Score | **0.9487** | 0.8947 | 0.8684 | 0.9130 | 0.8817 | 0.8421 |
| MCC | **0.9378** | 0.8812 | 0.8521 | 0.8993 | 0.8671 | 0.8102 |

---

## Repository Structure

```
CervixNet-Ensemble/
├── app.py                          # Streamlit application (entry point)
├── requirements.txt                # Python dependencies
├── cervical-cancer_csv.csv         # Raw UCI dataset
├── CervixNet_Ensemble_Final.pkl    # Trained stacked ensemble model
├── CervixNet_Ensemble_Model.pkl    # Intermediate model checkpoint
├── scaler.pkl                      # Fitted StandardScaler
├── best_threshold.pkl              # Youden's J threshold (τ*=0.31)
├── preprocessed_data.pkl           # Preprocessed feature matrix
├── shap_summary.png                # SHAP global feature importance plot
├── catboost_info/                  # CatBoost training logs
├── sample_data/                    # Sample data for Colab
└── .config/                        # Environment configuration
```

---

## Run in Google Colab and Push to GitHub

Use this workflow to retrain the model, run experiments, or update files, then push back to GitHub.

### Cell 1 — Clone the repo and install dependencies
```python
!git clone https://github.com/Aghawafaabbass/CervixNet-Ensemble.git
%cd CervixNet-Ensemble
!pip install -r requirements.txt
```

### Cell 2 — Set your Git identity (once per Colab session)
```python
!git config --global user.email "agha.wafa@port.ac.uk"
!git config --global user.name "Agha Wafa Abbas"
```

### Cell 3 — Authenticate with your GitHub Personal Access Token
```python
from getpass import getpass
import subprocess

token = getpass("Paste your GitHub Personal Access Token: ")
remote_url = f"https://{token}@github.com/Aghawafaabbass/CervixNet-Ensemble.git"
subprocess.run(["git", "remote", "set-url", "origin", remote_url], check=True)
print("Authentication set.")
```

Generate a token at: **https://github.com/settings/tokens/new** — check the `repo` scope → Generate → Copy.

### Cell 4 — Make your changes, then commit and push
```python
!git add .
!git commit -m "Update from Colab — describe your change here"
!git push origin main
print("Pushed successfully.")
```

### Cell 5 — Verify
```python
!git log --oneline -5
```

After pushing, Streamlit Cloud automatically redeploys within a few minutes.

---

## Local Installation

```bash
git clone https://github.com/Aghawafaabbass/CervixNet-Ensemble.git
cd CervixNet-Ensemble
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Streamlit Deployment

1. Fork this repository to your GitHub account
2. Go to https://share.streamlit.io and sign in with GitHub
3. Click "New app" → select your fork → Branch: `main` → File: `app.py`
4. Click Deploy

The app loads model artefacts via `@st.cache_resource` for fast inference. To update: push to `main` → Streamlit auto-redeploys.

---

## Explainability

CervixNet-Ensemble uses SHAP (SHapley Additive exPlanations) via TreeExplainer for:

- **Global feature importance** — `shap_summary.png` embedded in the app shows which features most influence predictions across all patients
- **Local attribution** — per-patient waterfall plots show exactly which risk factors drove the individual prediction

Top-5 risk factors by SHAP rank:
1. Number of Sexual Partners (mean |SHAP| = 0.421)
2. Age at First Sexual Intercourse (0.387)
3. Hormonal Contraceptive Duration in years (0.312)
4. Number of Pregnancies (0.278)
5. Smoking Duration in years (0.241)

---

## Citation

This work is formally archived on Zenodo with two separate citable records:

| Record | DOI | What it contains |
|--------|-----|-----------------|
| **Software v1.0** | [10.5281/zenodo.19654957](https://doi.org/10.5281/zenodo.19654957) | Source code, trained model `.pkl` artefacts, Streamlit app |
| **Research Paper**  | [10.5281/zenodo.19654767](https://doi.org/10.5281/zenodo.19654767) | Full IEEE-style research paper (PDF) |

---

**Cite the software (code + model artefacts):**
```bibtex
@software{abbas2026cervixnet_software,
  author    = {Abbas, Agha Wafa},
  title     = {CervixNet-Ensemble: A Stacked Meta-Learning MLOps Framework
               for Robust Biopsy-Confirmed Cervical Cancer Risk Stratification
               Leveraging Multi-Modal Behavioral and Clinical Predictors},
  version   = {1.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19654957},
  url       = {https://doi.org/10.5281/zenodo.19654957}
}
```

---

**Cite the research paper:**
```bibtex
@misc{abbas2026cervixnet_paper,
  author    = {Abbas, Agha Wafa},
  title     = {CervixNet-Ensemble: A Stacked Meta-Learning MLOps Framework
               for Robust Biopsy-Confirmed Cervical Cancer Risk Stratification
               Leveraging Multi-Modal Behavioral and Clinical Predictors},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19654767},
  url       = {https://doi.org/10.5281/zenodo.19654767}
}
```

---

**APA 7th — Software (code + artefacts):**
> Abbas, A. W. (2026). *CervixNet-Ensemble: A Stacked Meta-Learning MLOps Framework for Robust Biopsy-Confirmed Cervical Cancer Risk Stratification Leveraging Multi-Modal Behavioral and Clinical Predictors* (Version 1.0) [Software]. Zenodo.
https://doi.org/10.5281/zenodo.19654957

**APA 7th — Research Paper:**
> Abbas, A. W. (2026). *CervixNet-Ensemble: A Stacked Meta-Learning MLOps Framework for Robust Biopsy-Confirmed Cervical Cancer Risk Stratification Leveraging Multi-Modal Behavioral and Clinical Predictors* [Research paper]. Zenodo.
https://doi.org/10.5281/zenodo.19654767


---

## Disclaimer

This system is developed for academic research and clinical decision support purposes only. It is not a cleared medical device and must not be used as a substitute for professional medical diagnosis, clinical judgment, or treatment decisions. Always consult a qualified and licensed healthcare provider. Model performance is reported on the UCI Cervical Cancer Risk Factors Dataset (n=858) from a single institution and may not generalise to all clinical populations without external validation.
