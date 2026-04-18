# 🩺 CervixNet-Ensemble: Stacked Meta-Learning MLOps Pipeline for Cervical Cancer Risk Stratification

[![Streamlit App](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-red?style=for-the-badge)](https://cervixnet-ensemble-muqshrhxromay4axyoavbu.streamlit.app/#status-low-risk)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-99%25-brightgreen?style=for-the-badge)](https://cervixnet-ensemble-muqshrhxromay4axyoavbu.streamlit.app)
[![GitHub stars](https://img.shields.io/github/stars/Aghawafaabbass/CervixNet-Ensemble?style=for-the-badge)](https://github.com/Aghawafaabbass/CervixNet-Ensemble/stargazers)

---

> **CervixNet-Ensemble** is a production-grade MLOps pipeline achieving **~99% accuracy** in biopsy-confirmed cervical cancer risk stratification using a stacked meta-learning ensemble of XGBoost, LightGBM, CatBoost, and Random Forest, deployed as a real-time Streamlit clinical decision support tool.

---

## 👨‍💻 Author

**Agha Wafa Abbas**

| Role | Institution |
|------|------------|
| 🎓 Lecturer, School of Computing | University of Portsmouth, Winston Churchill Ave, Southsea, Portsmouth PO1 2UP, United Kingdom |
| 🎓 Lecturer, School of Computing | Arden University, Coventry, United Kingdom |
| 🎓 Lecturer, School of Computing | Pearson, London, United Kingdom |
| 🎓 Lecturer, School of Computing | IVY College of Management Sciences, Lahore, Pakistan |

📧 **agha.wafa@port.ac.uk** | **awabbas@arden.ac.uk** | **wafa.abbas.lhr@rootsivy.edu.pk**

---

## 📌 Table of Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Model Pipeline](#model-pipeline)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Quick Start — Run in Google Colab](#quick-start--run-in-google-colab)
- [Local Installation](#local-installation)
- [Streamlit Deployment](#streamlit-deployment)
- [Explainability (SHAP)](#explainability-shap)
- [Clinical Disclaimer](#clinical-disclaimer)
- [Citation](#citation)

---

## 🔍 Overview

Cervical cancer remains one of the most preventable yet deadliest cancers globally, particularly in low- and middle-income countries where early detection infrastructure is limited. **CervixNet-Ensemble** addresses this by:

- Combining **behavioral, demographic, and clinical risk factors** from the UCI Cervical Cancer Risk Factors Dataset
- Applying a **two-level stacked meta-learning architecture** (base learners → meta-learner)
- Deploying via a **Streamlit MLOps interface** with real-time PDF diagnostic report generation
- Integrating **SHAP explainability** for transparent, interpretable predictions

The pipeline achieves **~99% accuracy**, **AUC-ROC > 0.98**, with near-perfect sensitivity for biopsy-confirmed high-risk cases.

---

## 🚀 Live Demo

🔗 **[https://cervixnet-ensemble-muqshrhxromay4axyoavbu.streamlit.app](https://cervixnet-ensemble-muqshrhxromay4axyoavbu.streamlit.app/#status-low-risk)**

Enter patient parameters in the sidebar → Get instant risk classification → Export PDF diagnostic report.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RAW INPUT DATA                          │
│   (Age, Partners, Pregnancies, Smoking, HPV, Hormonal...)   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  PREPROCESSING LAYER                        │
│  • Missing Value Imputation (KNN / Median Strategy)         │
│  • SMOTE Oversampling (Class Imbalance Correction)          │
│  • StandardScaler Normalization                             │
│  • Feature Selection (Mutual Information + Correlation)     │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              LEVEL-1: BASE LEARNERS (5-Fold CV)             │
│                                                             │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐  │
│  │  XGBoost  │ │ LightGBM  │ │ CatBoost  │ │   Random  │  │
│  │           │ │           │ │           │ │   Forest  │  │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘  │
│        └─────────────┴──────────────┴─────────────┘         │
│                        Meta-Features                        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│           LEVEL-2: META-LEARNER (Logistic Regression)       │
│         Learns optimal weighting of base predictions        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                 THRESHOLD OPTIMIZATION                      │
│         (Youden's J Statistic on Validation Set)            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  RISK STRATIFICATION                        │
│           🔴 HIGH RISK  |  🟢 LOW RISK                      │
│          + SHAP Explainability + PDF Export                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Source** | UCI ML Repository — Cervical Cancer (Risk Factors) |
| **Original Records** | 858 patients |
| **Features** | 36 (behavioral + clinical + biopsy labels) |
| **Target Variable** | Biopsy (0 = Negative, 1 = Positive) |
| **Class Imbalance** | ~6% positive (addressed via SMOTE) |
| **Missing Values** | Up to 91.8% in some columns — handled via KNN imputation |

**Key Features Used:**
- Age, Number of Sexual Partners, First Sexual Intercourse Age
- Number of Pregnancies
- Smoking (years, packs/year)
- Hormonal Contraceptives (years)
- IUD (years)
- HPV (diagnosis flag)
- Symptomatic flags (abnormal bleeding, pelvic pain)

---

## ⚙️ Model Pipeline

### Base Learners

| Model | Role | Key Hyperparameters |
|-------|------|---------------------|
| **XGBoost** | Gradient Boosting | `n_estimators=500, max_depth=6, learning_rate=0.05` |
| **LightGBM** | Fast GBDT | `num_leaves=63, min_child_samples=20` |
| **CatBoost** | Categorical GBDT | `iterations=500, depth=8` |
| **Random Forest** | Bagging Ensemble | `n_estimators=500, max_features='sqrt'` |

### Meta-Learner
- **Logistic Regression** with L2 regularization
- Input: out-of-fold probability predictions from base learners
- Cross-validation: **Stratified 5-Fold**

### Threshold Selection
- Optimal threshold selected via **Youden's J Index** on validation predictions
- Stored as `best_threshold.pkl` for reproducible inference

---

## 📈 Results

| Metric | Score |
|--------|-------|
| **Accuracy** | ~99% |
| **AUC-ROC** | >0.98 |
| **Sensitivity (Recall)** | >0.97 |
| **Specificity** | >0.99 |
| **F1-Score** | >0.95 |
| **Precision** | >0.94 |

> ⚠️ Results are on the UCI cervical cancer dataset with SMOTE augmentation and should be interpreted in a research context.

---

## 📁 Repository Structure

```
CervixNet-Ensemble/
│
├── app.py                          # Streamlit MLOps application
├── requirements.txt                # Python dependencies
├── cervical-cancer_csv.csv         # Raw dataset (UCI)
│
├── CervixNet_Ensemble_Final.pkl    # Trained stacked ensemble model
├── CervixNet_Ensemble_Model.pkl    # Intermediate model checkpoint
├── scaler.pkl                      # Fitted StandardScaler
├── best_threshold.pkl              # Optimized classification threshold
├── preprocessed_data.pkl           # Preprocessed feature matrix
├── shap_summary.png                # SHAP feature importance plot
│
├── catboost_info/                  # CatBoost training logs
├── sample_data/                    # Google Colab sample data
└── .config/                        # Environment configuration
```

---

## ☁️ Quick Start — Run in Google Colab

Click below to open directly in Google Colab, run all cells, and push changes back to GitHub:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aghawafaabbass/CervixNet-Ensemble/blob/main/CervixNet_Ensemble_Colab.ipynb)

### Step-by-Step: Colab → GitHub Push

**Step 1: Clone and install**
```python
# Cell 1 — Setup
!git clone https://github.com/Aghawafaabbass/CervixNet-Ensemble.git
%cd CervixNet-Ensemble
!pip install -r requirements.txt
```

**Step 2: Configure Git identity**
```python
# Cell 2 — Git identity (run once per session)
!git config --global user.email "agha.wafa@port.ac.uk"
!git config --global user.name "Agha Wafa Abbas"
```

**Step 3: Authenticate with GitHub**
```python
# Cell 3 — GitHub Token Auth (use a Personal Access Token)
import os
from getpass import getpass

token = getpass("Enter your GitHub Personal Access Token: ")
os.environ["GH_TOKEN"] = token

!git remote set-url origin https://{token}@github.com/Aghawafaabbass/CervixNet-Ensemble.git
```

**Step 4: Make changes, then push**
```python
# Cell 4 — Stage, commit, and push
!git add .
!git commit -m "Update model via Colab — $(date)"
!git push origin main
```

> 💡 **Tip:** Generate a GitHub Personal Access Token at [github.com/settings/tokens](https://github.com/settings/tokens) with `repo` scope enabled.

---

## 💻 Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/Aghawafaabbass/CervixNet-Ensemble.git
cd CervixNet-Ensemble

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

---

## 🌐 Streamlit Deployment

The app is deployed on **Streamlit Community Cloud**:

1. Fork this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub and select `app.py` as the entry point
4. Set `main` as the branch and deploy

**Streamlit runs automatically** — the `.pkl` model files are loaded at startup via `@st.cache_resource`.

---

## 🔎 Explainability (SHAP)

CervixNet-Ensemble uses **SHAP (SHapley Additive exPlanations)** to produce:
- **Global feature importance** (`shap_summary.png`) — which features most influence predictions across the dataset
- **Local explanations** — per-patient contribution of each risk factor to the final prediction

The SHAP summary plot is embedded in the Streamlit app and stored in `shap_summary.png`.

---

## ⚠️ Clinical Disclaimer

> This tool is developed for **academic research and clinical decision support** purposes only. It is **not a substitute** for professional medical diagnosis, treatment, or clinical judgment. Always consult a qualified healthcare provider. Predictions are based on a research dataset and may not generalise to all clinical populations.

---

## 📜 Citation

If you use this work in your research, please cite:

```bibtex
@article{abbas2025cervixnet,
  title     = {CervixNet-Ensemble: A Robust Stacked Meta-Learning MLOps Pipeline
               Achieving Near-Perfect 99\% Accuracy in Biopsy-Confirmed Cervical
               Cancer Risk Stratification Using Multi-Modal Behavioral and
               Clinical Risk Factors},
  author    = {Abbas, Agha Wafa},
  journal   = {IEEE Access},
  year      = {2025},
  publisher = {IEEE},
  note      = {Preprint / Under Review}
}
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ for clinical AI research by <strong>Agha Wafa Abbas</strong>
</p>
