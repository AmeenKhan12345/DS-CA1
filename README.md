# 🌾 PMFBY Crop Insurance: Analysis & Predictive Dashboard

An **end-to-end data science project** analyzing **India's crop insurance schemes (2018–2022)**.  
This repository contains the **complete workflow**, from **data cleaning** and **exploratory data analysis (EDA)** to **advanced clustering**, **model experimentation (OLS, ANN, XGBoost)**, and **deployment** as an **interactive Streamlit dashboard**.

---

## 🚀 Live Demo

A live, interactive version of the final dashboard is deployed on **Streamlit Community Cloud**:

👉 [**Launch Live Dashboard**](https://ds-ca1-batgrcf45nbplqykjiwouv.streamlit.app)  

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
git clone https://github.com/AmeenKhan12345/DS-CA1.git
cd DS-CA1
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Streamlit app:
```bash
streamlit run dashboard.py
```
#### The app should open automatically in your browser at http://localhost:8501.
---
## 📊 Project Workflow & Methodology

This project followed a **rigorous 5-step data science workflow** — from raw data to an interactive public dashboard.

---

### 🧹 1. Data Cleaning & Preparation
- Loaded **6,161 raw records** with **28 columns**.  
- Performed a **7-step cleaning process** (detailed in *PMFBY_Mini_Report.md*) to fix:
  - Inconsistencies (e.g., `farmer_count` mismatches)  
  - Logical errors (e.g., 0 sum insured with 100 policies)  
  - Missing or corrupted entries  
- ✅ **Result:** A **robust dataset of 5,694 valid records** ready for analysis.


### 🔍 2. Exploratory Data Analysis (EDA)
- Identified key insights such as:
  - **"Vulnerable Farmer Paradox"**
  - **"Insurance Heartland"** (Maharashtra, Rajasthan, Madhya Pradesh)
- Created **interactive visualizations** using `plotly`:
  - Choropleth Maps  
  - Treemaps  
  - Violin Plots  
- Helped uncover deep structural and regional patterns in scheme performance.


### 🧩 3. Clustering (K-Means)
- Applied **K-Means Clustering** to uncover hidden district-level groupings.  
- Used the **Elbow Method** → Optimal **K = 5**.  
- Segmented all districts into **5 “Archetypes”** based on financial and demographic traits.  
- These cluster labels became a **powerful new feature** for predictive modeling.


### 🤖 4. Predictive Modeling (The “Showdown”)
To identify the **most accurate premium predictor**, a **4-model experiment** was conducted using **Mean Absolute Error (MAE)** on a 20% test set:

| Model | Description | Type |
|--------|--------------|------|
| **Model 1** | Simple Linear Regression | Baseline |
| **Model 2** | Multiple OLS Regression | Interpretable |
| **Model 3** | Deep Learning (ANN) | Advanced |
| **Model 4** | XGBoost | 🏆 *Champion Model* |

**XGBoost** emerged as the top performer with the **lowest MAE** and highest consistency across folds.


### 💻 5. Dashboard Deployment
- Built an **interactive dashboard** using **Streamlit**, featuring:
  - Tabbed interface for each stage (EDA → Clustering → Prediction)
  - Interactive inputs for premium prediction
- Deployed to **Streamlit Community Cloud** for **public access and reproducibility**.

---
## 📈 Key Analytical Insights

### 🧩 1. The 5 District Archetypes (Clustering)

Clustering revealed **five distinct district profiles**, providing a new analytical lens into how India’s crop insurance schemes operate across regions.  
Each cluster represents a unique combination of **policy volume**, **financial scale**, and **farmer inclusivity**.

| 🏷️ **Cluster Name (Inferred)** | 📊 **Cluster Size** | 🧾 **Avg Policies** | 💰 **Avg Sum Insured (₹)** | 👨‍🌾 **Avg % Marginal** | 👩‍🌾 **Avg % Female** | 🗺️ **Key State** |
|-------------------------------|--------------------|--------------------|----------------------------|------------------------|----------------------|------------------|
| **Mainstream Heartland** | 1,751 | 131,474 | 37,343 | 22.4% | 9.9% | Madhya Pradesh |
| **Inclusive Powerhouse** | 1,107 | 111,978 | 27,011 | 11.6% | 20.9% | Maharashtra |
| **Smallholder Specialists** | 1,451 | 5,002 | 517 | 13.6% | 14.6% | Chhattisgarh |
| **Vulnerable & Low-Scale** | 822 | 7,834 | 2,799 | 51.9% | 15.7% | Uttar Pradesh |
| **Niche Outliers (WBCIS)** | 492 | 2,666 | 392 | 13.5% | 7.1% | Rajasthan |

---

### 🌍 Insight Summary
- **Mainstream Heartland** dominates volume, representing traditional high-scale agricultural zones.  
- **Inclusive Powerhouse** demonstrates strong participation from **female and marginal farmers**, indicating equitable outreach.  
- **Smallholder Specialists** operate at micro scales, pointing toward **regional or tribal agriculture**.  
- **Vulnerable & Low-Scale** clusters reveal **financial vulnerability** and **low per-policy coverage**, needing targeted interventions.  
- **Niche Outliers (WBCIS)** capture **non-mainstream scheme patterns**, with low penetration but high scheme diversity.

---

### ⚔️ 2. The Model Showdown: Proving the Champion

A **scientific model comparison** was conducted across four techniques to determine the most accurate premium predictor.  
Results show that **XGBoost** outperformed all others — establishing itself as the **state-of-the-art model** for this problem.

| 🧠 **Model** | 🎯 **R-Squared (Accuracy)** | 💸 **MAE (Error in ₹)** | 📉 **% MAE Improvement (vs. Baseline)** |
|---------------|-----------------------------|--------------------------|----------------------------------------|
| **1. Simple Linear** | 91.5% | ₹1,515.83 | — |
| **2. Multiple OLS** | 95.9% | ₹884.00 | 41.7% |
| **3. Deep Learning (ANN)** | ~96.5% | ₹766.34 | 49.4% |
| **4. XGBoost 🏆 (Champion)** | ~96.7% | ₹672.19 | **55.7%** |

---

### ✨ 3. The "Wow" Insight: Equity as a Predictor

The **Champion XGBoost model** delivered a breakthrough finding:  
beyond the obvious financial variable (`log_sum_insured`), **demographic equity factors** were among the **top predictors** of premiums.

| 🏅 **Top Predictors (after log_sum_insured)** | 💡 **Insight** |
|-----------------------------------------------|----------------|
| % Female Farmers | Indicates inclusivity and outreach impact |
| % Marginal Farmers | Reflects vulnerability and scheme depth |

This discovery **proves that social equity is not just a moral or policy goal — it is a measurable financial indicator.**  
The model *needed* this demographic data to make its **most accurate predictions**.

## Repository Structure
```bash
PMFBY-Crop-Insurance-Dashboard/
├── dashboard.py                           # The main Streamlit app script
├── requirements.txt                       # Python libraries for deployment
├── README.md                              # You are here!
│
├── data-reports/                          # 📄 Data & Reports (use Git LFS for large files)
│   ├── processed_features_..._clustered.csv  # Final, clean dataset (LFS)
│   ├── india.geojson.txt                  # Map file for choropleth
│   ├── Final_Report.pdf                   # Original PDF report (LFS)
│   └── PMFBY_Mini_Report.md               # The condensed Markdown report
│
├── models/                                # 🧠 Models & Scalers (LFS)
│   ├── xgb_premium_predictor.json         # Champion XGBoost model (LFS)
│   ├── ann_premium_predictor.h5           # ANN model for experiment (LFS)
│   └── scaler_...pkl                      # Scaler for numeric features (LFS)
│
├── experiments/                           # 🧪 Experiments
│   └── dl_experiment.py                   # Script for the DL experiment
│
└── visuals/                               # 📊 Visuals
    └── image_16da80.png                   # XGBoost feature importance plot
```
---

## 🏁 Conclusion & Future Work

This project successfully **transformed a raw dataset into an insightful, predictive, and interactive analytical tool**.  
It achieved three major milestones:
- Identified the **"Vulnerable Farmer Paradox"**, revealing deep inclusivity trends.
- Segmented all districts into **5 distinct “Archetypes”** for policy differentiation.
- Demonstrated that **social equity metrics** (like % female and % marginal farmers) are **powerful financial predictors**.

---

### ⚠️ Limitation
The **biggest limitation** of this study is the **absence of `claims_paid` data**, which prevents the computation of true **risk efficiency** metrics.

---

### 🔮 Future Work
The next logical step is to:
- **Acquire claims data** from PMFBY datasets or government sources.
- Develop a **Loss Ratio Analysis** (`Claims / Premium`) and a **Risk Prediction Model**.  
This advancement would enable the shift from **predicting cost** → to **predicting risk**, creating a **proactive decision-making tool** for government budgeting and policy planning.

---

## 📄 License
This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute this work with proper attribution.

