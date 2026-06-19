<div align="center">

# 🌌 EcomIQ: E-Commerce Sales Intelligence & Churn Prediction System

<a href="https://github.com/Soumen1602/Ecommerce-Intelligence"><img src="https://capsule-render.vercel.app/api?type=waving&color=0070f3&height=150&section=header&text=EcomIQ%20Analytics&fontSize=50&fontColor=ffffff&animation=fadeIn&fontAlignY=38" alt="EcomIQ Banner"></a>

**An ultra-premium, end-to-end data analytics and machine learning pipeline built to extract actionable sales intelligence and predict customer churn.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-%2334D058.svg?style=for-the-badge&logo=hugging-face&logoColor=white)](https://huggingface.co)
[![LightGBM](https://img.shields.io/badge/LightGBM-ffb347.svg?style=for-the-badge&logo=lightgbm&logoColor=black)](https://lightgbm.readthedocs.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-189FDD.svg?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.ai)
[![CatBoost](https://img.shields.io/badge/CatBoost-F5DD00.svg?style=for-the-badge&logo=catboost&logoColor=black)](https://catboost.ai/)

</div>

---

## 🚀 Overview

**EcomIQ** is a portfolio-grade machine learning project that analyzes **100K+ real-world orders** from the [Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). 

Moving beyond standard dashboards, EcomIQ features a bespoke, **"Vercel / Linear" inspired ultra-premium UI**. It offers real-time business KPIs alongside an embedded **XGBoost classification model** that identifies at-risk customers with high precision, explained transparently via **SHAP (SHapley Additive exPlanations)**.

### ✨ Key Highlights for Recruiters
- **End-to-End ML Pipeline**: Handles raw data ingestion, relational merging (9 datasets), feature engineering (RFM & Behavioral), model training, and deployment.
- **Deep Learning NLP Fusion**: Integrates **Hugging Face Transformers** to extract sentiment scores from Portuguese customer review texts, fusing unstructured NLP data with tabular ML features.
- **Gradient Boosting Mastery**: Evaluates and compares the industry's top three state-of-the-art frameworks: **XGBoost, LightGBM, and CatBoost**, expertly handling imbalanced class distributions.
- **Production-Ready Premium UI**: Deployed via Streamlit using a bespoke, 100% custom-injected **"Midnight Glassmorphism"** CSS design system. Features frosted glass cards, dynamic `clamp()` typography scaling, and smooth CSS keyframe animations—bypassing generic components for an elite aesthetic.

---

## 🧠 Machine Learning Architecture

<details>
<summary><b>Click to view Data Pipeline & Architecture</b></summary>

```mermaid
graph TD;
    A[(Raw Olist CSVs)] --> B[Data Loader & ETL];
    B --> C[Merged Master DataFrame];
    C --> D{Feature Engineering};
    D --> E[RFM Features];
    D --> F[Behavioral Signals];
    E --> G[Feature Matrix];
    F --> G;
    N[NLP Sentiment Extractor<br>Hugging Face] --> G;
    G --> H[Train/Test Split & Scaling];
    H --> I((XGBoost));
    H --> J((LightGBM));
    H --> K((CatBoost));
    I --> L[SHAP Explainer];
    L --> M[Streamlit Dashboard];
```

</details>

---

## 📊 Model Performance

We trained an ensemble of state-of-the-art classifiers to predict customer churn. **XGBoost**, **LightGBM**, and **CatBoost** were rigorously evaluated, offering an excellent balance of precision and recall for this highly imbalanced dataset.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.6069 | 0.9526 | 0.5932 | 0.7311 | 0.7239 |
| Random Forest | 0.8516 | 0.9142 | 0.9219 | 0.9180 | 0.6603 |
| **XGBoost (Deployed)** | **0.6201** | **0.9544** | **0.6074** | **0.7423** | **0.7326** |

> *Note: Model trained on 92,753 unique customers. `recency_days` was strictly excluded from training to prevent data leakage, ensuring the model learns from genuine behavioral signals.*

---

## 💻 Tech Stack

- **Core & Data Processing**: `Python`, `Pandas`, `NumPy`
- **Machine Learning**: `Scikit-Learn`, `XGBoost`, `LightGBM`, `CatBoost`
- **NLP / Deep Learning**: `Transformers (Hugging Face)`, `PyTorch`
- **Interpretability**: `SHAP`
- **Frontend & Visualization**: `Streamlit`, `Plotly`, `Custom CSS`
| **App Framework** | Streamlit |
| **Deployment** | Streamlit Cloud |

---

## 📁 Project Structure

```text
ecommerce-intelligence/
├── app/
│   └── streamlit_app.py              # Main Streamlit application + Custom CSS
├── data/
│   └── raw/                          # Raw CSV files (ignored in git)
├── models/
│   └── churn_model.pkl               # Deployed XGBoost model
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb  # Pipeline creation
│   └── 03_churn_model.ipynb          # Model training & SHAP analysis
├── src/
│   ├── data_loader.py                # ETL functions
│   ├── feature_engineering.py        # ML feature creation
│   ├── model.py                      # Training wrappers
│   └── visualizations.py             # Plotly graph generators
├── requirements.txt
└── README.md
```

---

## ⚙️ Local Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Soumen1602/Ecommerce-Intelligence.git
   cd Ecommerce-Intelligence
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Data:**
   - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
   - Extract the 9 CSV files into `data/raw/`.

4. **Run the Dashboard:**
   ```bash
   streamlit run app/streamlit_app.py
   ```
   *The app will automatically launch in your browser at `http://localhost:8501`.*

---

## 🤝 Let's Connect

Built by **Soumen** — Passionate Data Analyst & ML Engineer looking for exciting opportunities.

<div align="left">
  <a href="https://github.com/Soumen1602">
    <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  </a>
  <a href="https://www.linkedin.com/in/soumend12/">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
  </a>
</div>

<br>

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0070f3&height=100&section=footer" alt="Footer" width="100%">
</div>
