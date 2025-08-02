# Crop Insurance Data Analysis (2018–2022)

This repository contains a comprehensive exploratory data analysis (EDA), report, and interactive dashboard for India’s crop insurance schemes, focusing on Pradhan Mantri Fasal Bima Yojana (PMFBY) and Weather Based Crop Insurance Scheme (WBCIS) from 2018 to 2022.

### View Live Dashboard- [Live-Dashboard](https://ameenkhan12345.github.io/DS-CA1/)

## 📂 Repository Structure

```
├── data/                  # Dataset files
│   └── cleaned_crop_insurance1.csv
│   └── raw_crop_insurance.csv
│
├── notebook/              # Combined analysis notebook and PDF
│   └── Complete-Project.ipynb
│   └── Complete-Report-withCode.pdf  # Note: visuals may not render in GitHub preview
│
├── report/                # Final report documents
│   └── report.pdf          # Note: visuals may not render in GitHub preview
│
├── dashboard/             # Interactive dashboard files
│   └── dashboard.html
│
└── README.md              # This file
```

## 📝 Project Overview

* **Objective:** Analyze national crop insurance data (2018–2022) to uncover trends in policy uptake, financial structures, demographic inclusivity, seasonality, and regional variation.
* **Scope:** Single, consolidated Jupyter notebook for data cleaning, univariate/bivariate/multivariate analysis, PCA, policy insights; plus an interactive dashboard and a written report.

**Note:** GitHub’s in-browser PDF preview may not display embedded charts. To view visuals, please download the PDF files or open the corresponding `.ipynb` notebook.

* **Objective:** Analyze national crop insurance data (2018–2022) to uncover trends in policy uptake, financial structures, demographic inclusivity, seasonality, and regional variation.
* **Scope:** Data cleaning, univariate/bivariate/multivariate analysis, PCA, policy insights, and an interactive HTML dashboard.

## 🚀 Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/AmeenKhan12345/DS-CA1.git
   cd DS-CA1
   ```
   
2. **Explore Notebooks**

   * Launch Jupyter Lab or Notebook:

     ```bash
     jupyter lab
     ```
   * Open and run notebook :

     1. `Complete-Project.ipynb`

4. **View the Report**

   * The final report is available in `report/` as PDF.

5. **Open the Dashboard**

   * Launch `dashboard/dashboard.html` in your browser to interact with charts and filters.

## 📊 Key Findings & Insights

* **PMFBY vs WBCIS:** PMFBY dominates in scale and value; WBCIS serves niche high-value segments but shifts more cost to farmers.
* **Vulnerable Farmer Paradox:** High policy counts often correspond to small landholdings, indicating deep coverage among marginal farmers.
* **Demographic Gaps:** Gender and caste distributions show under-representation of female, SC, and ST farmers.
* **Regional Variability:** Significant state-level differences in premiums and scheme uptake.
* **PCA Dimensions:** Two orthogonal axes—**Financial Scale** and **Demographic Inclusivity**—capture most data variance.

## 🛠️ Tools & Technologies

* **Python**: pandas, numpy, matplotlib, seaborn, scikit-learn, plotly
* **Jupyter Notebooks** for analysis
* **HTML/CSS/JS** for the dashboard
* **MS Word / PDF** for the report

## 🔜 Next Steps

* Incorporate clustering on PCA components to segment districts.
* Enhance dashboard with geospatial maps (choropleths).
* Develop predictive models (e.g., premium forecasting).

## 📄 License

This project is licensed under the MIT License – see `LICENSE` for details.

---

*Happy analyzing!*
