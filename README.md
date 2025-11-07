# 🌾 PMFBY Crop Insurance: Analysis & Predictive Dashboard

An **end-to-end data science project** analyzing **India's crop insurance schemes (2018–2022)**.  
This repository contains the **complete workflow**, from **data cleaning** and **exploratory data analysis (EDA)** to **advanced clustering**, **model experimentation (OLS, ANN, XGBoost)**, and **deployment** as an **interactive Streamlit dashboard**.

---

## 🚀 Live Demo

A live, interactive version of the final dashboard is deployed on **Streamlit Community Cloud**:

👉 [**Launch Live Dashboard**](#)  

---

## 💡 Key Features & Findings

This project goes **beyond static reports**, transforming insights into a **comprehensive analytical and predictive tool**.

### 📈 The "Vulnerable Farmer Paradox"
Uncovered a core finding where **policy participation surged +80%** while the **average insured sum declined**.  
This paradox reveals the scheme’s deep success in reaching **81.1% Small & Marginal Farmers**.  


### 🧩 5 District Archetypes
Applied **K-Means Clustering** to segment **5,600+ districts** into **five distinct archetypes**, such as:
- **Inclusive Powerhouse**
- **Mainstream Heartland**
- **Emerging Transition Zone**

Proved that a **one-size-fits-all policy** is **inefficient** and regionally inconsistent.


### 🏆 Champion Predictive Model
After extensive model benchmarking:
- **XGBoost** emerged as the **most accurate** predictor of premiums.
- Achieved a **55.7% error reduction** over a simple baseline.
- Outperformed a **Deep Learning (ANN)** model by **12.3%** in accuracy.


### ✨ The "Wow" Insight
Feature importance analysis revealed that:
- **% Female Farmers** and **% Marginal Farmers** were the **#2 and #3 most influential predictors** of district-level premiums.
  
This finding demonstrates that **social equity is a measurable financial indicator** within the crop insurance domain.


### 🔮 Interactive Predictor Tool
The dashboard includes a **real-time prediction tool** powered by the **champion XGBoost model**, allowing users to:
- Select district-level parameters.
- Instantly generate **premium predictions**.
- Visualize the **impact of inclusivity metrics** on pricing outcomes.

---

## 🛠️ Tech Stack & Libraries

| Category | Tools & Libraries |
|-----------|-------------------|
| **Data Analysis** | `pandas`, `numpy` |
| **Machine Learning** | `scikit-learn` (Clustering, OLS, Scaling), `xgboost` *(Champion Model)*, `tensorflow` *(ANN Experiment)* |
| **Visualization** | `plotly` *(Interactive Charts)*, `matplotlib`, `seaborn` |
| **Dashboard** | `streamlit` |
| **File Handling** | `joblib` *(for saving models/scalers)* |
| **Large File Storage** | `git-lfs` *(for handling large .csv and .h5 files)* |

---

## 🚀 How to Run Locally

### 🧩 Prerequisites
This project uses **Git LFS** to manage large files such as datasets and trained models.  
Please ensure **Git LFS** is installed before proceeding.

```bash
# Install Git LFS (only once per system)
git lfs install
```
### Installation and Launch
1. Clone the repository:
```bash
git clone 
cd 
