import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="PMFBY Data-Driven Review",
    page_icon="🌱",
    layout="wide"
)

# --- Constants ---
# (Update this to your actual file paths if they are different)
CLUSTERED_DATA_PATH = r'C:\Users\ASUS\Downloads\CA-1 DS\snapshots\processed_features_20251008_130631_clustered.csv'
XGB_MODEL_PATH = 'xgb_premium_predictor.json'
FEATURE_IMPORTANCE_PATH = 'image_16da80.png' # Path to your saved plot

# --- Caching: Load data once and cache it ---
@st.cache_data
def load_data():
    """Loads and preprocesses data."""
    df = pd.read_csv(CLUSTERED_DATA_PATH)
    
    # For easier interpretation in the app
    df['cluster_id'] = df['cluster_id'].astype('str')
    
    # Calculate cluster profiles for display
    cluster_profile = df.groupby('cluster_id').agg(
        avg_total_policies=('total_policies', 'mean'),
        avg_sum_insured=('sum_insured', 'mean'),
        avg_area_insured=('area_insured', 'mean'),
        avg_pct_marginal=('marginal', 'mean'),
        avg_pct_small=('small', 'mean'),
        avg_pct_female=('female', 'mean'),
        cluster_size=('cluster_id', 'count'),
        most_common_state=('state_name', lambda x: x.mode().iloc[0])
    ).reset_index().sort_values(by='cluster_size', ascending=False)
    
    return df, cluster_profile

@st.cache_resource
def load_model():
    """Loads the saved XGBoost model."""
    model = xgb.XGBRegressor()
    model.load_model(XGB_MODEL_PATH)
    return model

# --- Load all artifacts ---
df, cluster_profile = load_data()
xgb_model = load_model()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    (
        "Project Overview", 
        "District Archetypes (Clustering)", 
        "Our Prediction Models", 
        "Premium Predictor Tool"
    )
)
st.sidebar.markdown("---")
st.sidebar.info("A project analyzing India's crop insurance schemes (2018-2022).")

# ==============================================================================
# PAGE 1: PROJECT OVERVIEW
# ==============================================================================
if page == "Project Overview":
    st.title("🌱 Unlocking Equity & Efficiency")
    st.header("A Data-Driven Review of India's Crop Insurance Schemes (2018-2022)")
    
    st.markdown("""
    This project presents an in-depth review of India's flagship crop insurance programs, primarily the **Pradhan Mantri Fasal Bima Yojana (PMFBY)**.
    The analysis is based on a district-level dataset from 2018-2022, cleaned and processed from over 6,100 initial records.
    
    The objective is to identify critical patterns in operational scale, demographic inclusivity, and financial dynamics to inform data-driven policy decisions.
    """)
    
    st.subheader("Key Findings from Exploratory Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("The 'Vulnerable Farmer Paradox'")
        st.markdown("""
        Our analysis uncovered a surprising trend:
        * The **number of farmer policies surged by 80%** from 2018-2022.
        * Simultaneously, the **total insured area and average sum insured declined**.
        
        **Why?** This paradox is explained by the scheme's deep penetration among **small and marginal farmers**, who make up over 81% of beneficiaries but insure smaller, less valuable plots.
        """)
        
    with col2:
        st.warning("Significant Demographic Gaps")
        st.markdown("""
        Despite success in reaching smallholders, significant equity gaps persist:
        * **Gender Gap:** Female farmers comprise only **14-20%** of total beneficiaries.
        * **Geographic Gaps:** An "Insurance Heartland" exists in Maharashtra, Rajasthan, and Madhya Pradesh, while other regions lag significantly.
        """)

# ==============================================================================
# PAGE 2: DISTRICT ARCHETYPES (CLUSTERING)
# ==============================================================================
elif page == "District Archetypes (Clustering)":
    st.title("Discovering District 'Archetypes'")
    st.markdown("""
    Instead of just looking at states, we used **K-Means Clustering** to find hidden patterns. 
    We identified 5 distinct "archetypes" or "profiles" of districts based on their financial scale, participation, and farmer demographics.
    """)
    
    st.header("The 5 Cluster Profiles")
    st.dataframe(cluster_profile.style.format(precision=2))
    
    st.markdown("""
    **How to Read This Table:**
    * **Cluster 1 (Mainstream Heartland):** The largest group, defined by high scale but very low female participation (9.98%).
    * **Cluster 3 (Inclusive Powerhouse):** High scale, dominated by small farmers (82%) and boasting the highest female participation (21%).
    * **Cluster 0 (Smallholder Specialists):** Districts with low financial value but decent participation from small farmers.
    * **Cluster 4 (Vulnerable & Low-Scale):** Defined by the highest rate of marginal farmers (51.85%).
    * **Cluster 2 (The Niche Outliers):** A small group defined by the WBCIS scheme, with the lowest participation across the board.
    """)

# ==============================================================================
# PAGE 3: OUR PREDICTION MODELS
# ==============================================================================
elif page == "Our Prediction Models":
    st.title("Predicting the Premium")
    st.markdown("We built three models to understand and predict `gross_premium`.")

    # --- Model Comparison Table ---
    st.header("Model Performance Comparison")
    model_comparison = {
        'Model': ['1. Simple Linear Regression', '2. Multiple Linear Regression (OLS)', '3. Advanced Model (XGBoost)'],
        'Features': [1, 21, 26],
        'Adj. R-Squared': [0.915, 0.959, 'N/A (R² = 0.967)'],
        'Average Error (MAE)': ['₹1,515.83', '₹884.00', '₹672.19'],
        'Improvement vs. OLS': ['-', 'Baseline', '23.96% More Accurate']
    }
    st.dataframe(pd.DataFrame(model_comparison), use_container_width=True)
    
    st.markdown(f"""
    **Key Takeaway:** Our final XGBoost model is **23.96% more accurate** than our OLS model, reducing the average prediction error to **₹672.19**.
    """)
    
    tab1, tab2 = st.tabs(["XGBoost Feature Importance", "OLS Regression Insights"])
    
    with tab1:
        st.subheader("XGBoost Model (Highest Accuracy)")
        st.image(FEATURE_IMPORTANCE_PATH, caption="XGBoost Feature Importance Plot")
        st.markdown("""
        This model gives the most accurate predictions. Its findings are fascinating:
        1.  **`log_sum_insured`** is the most important factor.
        2.  **`female`** and **`marginal`** (demographic profiles) are the 2nd and 3rd most important predictors! 
        
        This shows that **inclusivity and vulnerability are not just social metrics; they are critical drivers of a district's financial profile.**
        """)
        
    with tab2:
        st.subheader("Multiple Regression (OLS) (Best for Interpretation)")
        st.markdown("""
        This model is best for explaining *why* a premium is what it is. It showed that:
        * **Financials:** `sum_insured`, `area_insured`, and `total_policies` all significantly *increase* the premium.
        * **Geography:** Premiums in **Maharashtra** and **Tamil Nadu** are *significantly higher* than in other states.
        * **Policy:** The **PMFBY** scheme is *significantly cheaper* than the WBCIS scheme.
        """)

# ==============================================================================
# PAGE 4: PREMIUM PREDICTOR TOOL
# ==============================================================================
elif page == "Premium Predictor Tool":
    st.title("Interactive Premium Predictor")
    st.markdown("""
    Use our state-of-the-art **XGBoost model** to estimate the `gross_premium` for a district.
    
    **How it works:**
    1.  Select a **State** and **District** to load its baseline profile (demographics, cluster, etc.).
    2.  Adjust the key financial and seasonal details.
    3.  Click "Predict" to see the result.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # --- User Inputs ---
        state_list = df['state_name'].unique()
        selected_state = st.selectbox("1. Select State:", state_list)
        
        district_list = df[df['state_name'] == selected_state]['district_name'].unique()
        selected_district = st.selectbox("2. Select District:", district_list)
        
        season_list = df['season'].unique()
        selected_season = st.selectbox("3. Select Season:", season_list)
    
    with col2:
        # Get the most recent record for this district/season to use as a baseline
        baseline_record = df[
            (df['state_name'] == selected_state) &
            (df['district_name'] == selected_district) &
            (df['season'] == selected_season)
        ].sort_values('year', ascending=False).iloc[0:1]
        
        if baseline_record.empty:
            st.error("No data found for this State/District/Season combination. Using a default record.")
            baseline_record = df.iloc[0:1]
        
        # --- Sliders for key variables ---
        st.markdown("4. Adjust Key Financials:")
        val_sum_insured = st.number_input(
            "Sum Insured (₹ Lakhs):", 
            min_value=0.0, 
            value=baseline_record['sum_insured'].values[0],
            step=1000.0
        )
        val_area_insured = st.number_input(
            "Area Insured (Hectares):", 
            min_value=0.0, 
            value=baseline_record['area_insured'].values[0],
            step=100.0
        )
        val_total_policies = st.number_input(
            "Total Policies:", 
            min_value=0.0, 
            value=baseline_record['total_policies'].values[0],
            step=100.0
        )

    # --- Prediction Logic ---
    if st.button("Predict Gross Premium", type="primary"):
        with st.spinner("Running the model..."):
            
            # --- FIX 1: Get the model's feature list *first* ---
            model_features = xgb_model.get_booster().feature_names

            # 1. Create a copy of the baseline record
            pred_input = baseline_record.copy()
            
            # 2. Overwrite with user-selected values
            pred_input['sum_insured'] = val_sum_insured
            pred_input['area_insured'] = val_area_insured
            pred_input['total_policies'] = val_total_policies
            pred_input['season'] = selected_season
            
            # 3. Re-create all engineered features *exactly* as in training
            
            # --- Log transforms ---
            pred_input['log_sum_insured'] = np.log1p(pred_input['sum_insured'])
            pred_input['log_area_insured'] = np.log1p(pred_input['area_insured'])
            pred_input['log_total_policies'] = np.log1p(pred_input['total_policies'])
            
            # --- Encodings ---
            pred_input['scheme_encoded'] = pred_input['scheme'].map({'PMFBY': 1, 'WBCIS': 0}).fillna(0)
            pred_input['season_encoded'] = pred_input['season'].map({'Kharif': 1, 'Rabi': 0}).fillna(0)
            
            # --- State dummies ---
            top_states = [
                'Uttar_Pradesh', 'Madhya_Pradesh', 'Maharashtra', 'Chhattisgarh', 
                'Rajasthan', 'Tamil_Nadu', 'Odisha', 'Assam', 'Uttarakhand', 'Telangana'
            ]
            
            # --- FIX 2: Check against 'model_features' list, not 'X' ---
            for state in top_states:
                col_name = f"state_is_{state}"
                if col_name in model_features: # <-- This was the error
                    pred_input[col_name] = 1 if selected_state.replace(' ', '_') == state else 0
            
            other_col = 'state_is_Other'
            if other_col in model_features: # <-- Also fix this one
                pred_input[other_col] = 1 if selected_state.replace(' ', '_') not in top_states else 0
            
            # --- Interaction terms ---
            pred_input['scheme_x_log_sum_insured'] = pred_input['scheme_encoded'] * pred_input['log_sum_insured']
            pred_input['season_x_log_sum_insured'] = pred_input['season_encoded'] * pred_input['log_sum_insured']
            
            # --- Cluster dummies ---
            cluster_id = baseline_record['cluster_id'].values[0]
            for i in range(5): # Assuming K=5
                cluster_col = f'cluster_{i}'
                if cluster_col in model_features: # <-- And fix this one
                    pred_input[cluster_col] = 1 if str(i) == cluster_id else 0

            # 4. Ensure order is correct using the list we fetched
            pred_input_final = pred_input[model_features]
            
            # 5. Predict!
            prediction_log = xgb_model.predict(pred_input_final)[0]
            
            # 6. Inverse transform (from log1p back to Rupees)
            prediction_inr = np.expm1(prediction_log)
            
            st.metric(
                label="Predicted Gross Premium (in ₹ Lakhs)",
                value=f"₹ {prediction_inr:,.2f}"
            )
            st.caption(f"Based on the profile of {selected_district}, {selected_state}.")