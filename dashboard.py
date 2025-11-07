import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from sklearn.preprocessing import StandardScaler
import json
import requests
from pathlib import Path
import tensorflow as tf # To load the ANN model
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="PMFBY Data-Driven Review",
    page_icon="🌱",
    layout="wide"
)

# --- 1. Constants (Deployment-Ready Relative Paths) ---
# Define folder names as they appear in your GitHub repo
DATA_FOLDER = "Data&Reports"
MODELS_FOLDER = "Models&Scalers"
VISUALS_FOLDER = "Visuals"

# --- Use os.path.join to build cross-platform paths ---
CLUSTERED_DATA_PATH = os.path.join(DATA_FOLDER, "processed_features_20251008_130631_clustered.csv")
GEOJSON_PATH = os.path.join(DATA_FOLDER, "Indian_States.txt")

XGB_MODEL_PATH = os.path.join(MODELS_FOLDER, "xgb_premium_predictor.json")
SCALER_PATH = os.path.join(MODELS_FOLDER, "scaler_20251008_130631.pkl")
ANN_MODEL_PATH = os.path.join(MODELS_FOLDER, "ann_premium_predictor.h5")

FEATURE_IMPORTANCE_PATH = os.path.join(VISUALS_FOLDER, "image_16da80.png")


# --- 3. Cluster Information (from our analysis) ---
CLUSTER_NAMES = {
    1: "The Mainstream Heartland",
    3: "The Inclusive Powerhouse",
    0: "Smallholder Specialists",
    4: "Vulnerable & Low-Scale",
    2: "The Niche Outliers"
}

CLUSTER_DESCRIPTIONS = {
    1: "The largest group, defined by high financial and area scale but the *lowest* female participation (9.98%). Common in Madhya Pradesh.",
    3: "High-scale districts, but dominated by small farmers (82%) and boasting the *highest* female participation (21%). Common in Maharashtra.",
    0: "Districts with low financial value (avg. sum ₹517) but decent participation from small farmers (75%). Common in Chhattisgarh.",
    4: "Defined by the *highest* rate of marginal farmers (51.85%). These are vulnerable districts that have not yet reached large scale.",
    2: "The smallest group, defined by the WBCIS scheme. Has the lowest participation and inclusivity metrics across the board."
}

# --- 4. Data Loading ---
@st.cache_data
def load_data():
    """Loads, processes, and caches all required data."""
    try:
        df = pd.read_csv(CLUSTERED_DATA_PATH)
    except FileNotFoundError:
        st.error(f"Error: Missing data file '{CLUSTERED_DATA_PATH}'. Please ensure it's in the same folder as the app.")
        return None, None, None, None, None # <-- Updated return
    
    # For easier interpretation in the app
    df['cluster_id'] = df['cluster_id'].astype(int)
    df['cluster_name'] = df['cluster_id'].map(CLUSTER_NAMES)
    
    # Calculate cluster profiles (same as before)
    cluster_profile = df.groupby('cluster_name').agg(
        avg_total_policies=('total_policies', 'mean'),
        avg_sum_insured=('sum_insured', 'mean'),
        avg_area_insured=('area_insured', 'mean'),
        avg_pct_marginal=('marginal', 'mean'),
        avg_pct_small=('small', 'mean'),
        avg_pct_female=('female', 'mean'),
        cluster_size=('cluster_id', 'count')
    ).reset_index().sort_values(by='cluster_size', ascending=False)
    
    # Create a map-safe state name column
    state_mapping = {
        'Andaman & Nicobar': 'Andaman & Nicobar Islands',
        'Jammu & Kashmir': 'Jammu and Kashmir',
        'Arunachal Pradesh': 'Arunanchal Pradesh',
        'Telangana': 'Andhra Pradesh'
    }
    df['state_name_for_map'] = df['state_name'].replace(state_mapping)

    # State-level aggregation
    df_state = df.groupby('state_name_for_map').agg(
        original_states=('state_name', lambda x: list(x.unique())),
        total_policies=('total_policies', 'sum'),
        total_sum_insured=('sum_insured', 'sum'),
        avg_female_pct=('female', 'mean'),
        avg_marginal_pct=('marginal', 'mean'),
        avg_gross_premium=('gross_premium', 'mean')
    ).reset_index()

    # Load the scaler (same as before)
    try:
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        st.error(f"Error: Missing scaler file '{SCALER_PATH}'. Re-fitting a new one as a fallback.")
        scaler = StandardScaler()
        numeric_features = ['log_sum_insured', 'log_area_insured', 'log_total_policies']
        scaler.fit(df[numeric_features])
        
    # --- NEW: Calculate KPIs ---
    kpi_data = {}
    try:
        # 1. Policy Growth
        policies_2018 = df[df['year'] == 2018]['total_policies'].sum()
        policies_2022 = df[df['year'] == 2022]['total_policies'].sum()
        if policies_2018 > 0:
            kpi_data['policy_delta'] = (policies_2022 - policies_2018) / policies_2018
        else:
            kpi_data['policy_delta'] = 0
        kpi_data['total_policies_2022'] = policies_2022
        
        # 2. Coverage
        kpi_data['districts_covered'] = df['district_name'].nunique()
        kpi_data['states_covered'] = df['state_name'].nunique()
        
        # 3. PMFBY Premium Growth
        avg_prem_2018 = df[(df['year'] == 2018) & (df['scheme'] == 'PMFBY')]['gross_premium'].mean()
        avg_prem_2022 = df[(df['year'] == 2022) & (df['scheme'] == 'PMFBY')]['gross_premium'].mean()
        if avg_prem_2018 > 0:
            kpi_data['prem_delta'] = (avg_prem_2022 - avg_prem_2018) / avg_prem_2018
        else:
            kpi_data['prem_delta'] = 0
        kpi_data['avg_premium_pmfby'] = df[df['scheme'] == 'PMFBY']['gross_premium'].mean()

        # 4. Inclusivity
        kpi_data['avg_female_pct'] = df['female'].mean()

    except Exception as e:
        print(f"Error calculating KPIs: {e}")
        kpi_data = {} # Return empty dict on failure

    return df, cluster_profile, scaler, df_state, kpi_data # <-- Updated return

@st.cache_resource
def load_model():
    """Loads the saved XGBoost model."""
    try:
        model = xgb.XGBRegressor()
        model.load_model(XGB_MODEL_PATH)
        return model
    except (IOError, xgb.core.XGBoostError):
        st.error(f"Error: Missing model file '{XGB_MODEL_PATH}'. The predictor tool will not work.")
        return None

@st.cache_data
def get_all_model_metrics():
    """
    Loads all data and models to calculate a complete
    metrics table, including R-Squared.
    """
    try:
        # 1. Load data
        df = pd.read_csv(CLUSTERED_DATA_PATH)
        
        # 2. Define Features & Target
        TARGET = 'log_gross_premium'
        y = df[TARGET]

        # --- Base R-Squared scores from our experiments ---
        r2_simple_ols = 0.915 # From Phase 1 result
        r2_multi_ols = 0.959  # From Phase 2 result
        
        # --- Features for ANN/XGBoost ---
        numeric_features = ['log_sum_insured', 'log_area_insured', 'log_total_policies']
        other_numeric = ['female', 'marginal', 'small']
        
        # One-hot encode clusters
        cluster_dummies = pd.get_dummies(df['cluster_id'], prefix='cluster', dtype=int)
        df = pd.concat([df, cluster_dummies], axis=1)

        categorical_features = [
            'scheme_encoded', 'season_encoded', 'state_is_Uttar_Pradesh', 
            'state_is_Madhya_Pradesh', 'state_is_Maharashtra', 'state_is_Chhattisgarh', 
            'state_is_Rajasthan', 'state_is_Tamil_Nadu', 'state_is_Odisha', 
            'state_is_Assam', 'state_is_Uttarakhand', 'state_is_Telangana', 
            'state_is_Other', 'scheme_x_log_sum_insured', 'season_x_log_sum_insured'
        ] + cluster_dummies.columns.tolist()
        
        all_features = numeric_features + other_numeric + categorical_features
        X = df[all_features]

        # 3. Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 4. Calculate XGBoost R²
        xgb_model = load_model() # Use our existing cached model
        y_pred_xgb = xgb_model.predict(X_test)
        r2_xgb = r2_score(y_test, y_pred_xgb)

        # 5. Calculate ANN R²
        try:
            scaler_main = joblib.load(SCALER_PATH)
            scaler_other = StandardScaler().fit(X_train[other_numeric])

            X_test_scaled = X_test.copy()
            X_test_scaled[numeric_features] = scaler_main.transform(X_test[numeric_features])
            X_test_scaled[other_numeric] = scaler_other.transform(X_test[other_numeric])

            ann_model = tf.keras.models.load_model(ANN_MODEL_PATH)
            y_pred_ann = ann_model.predict(X_test_scaled).flatten()
            r2_ann = r2_score(y_test, y_pred_ann)
        
        except Exception as e:
            st.warning(f"Could not load ANN model to calculate R²: {e}. Using fallback values.")
            # Fallback values from our experiment script
            r2_ann = r2_score(y_test, np.expm1(0.3593)) # Approx.
        
        # 6. Return all metrics
        metrics = {
            'r2_simple_ols': r2_simple_ols,
            'r2_multi_ols': r2_multi_ols,
            'r2_ann': r2_ann,
            'r2_xgb': r2_xgb
        }
        return metrics
    
    except FileNotFoundError as e:
        st.error(f"Error: Missing file for metrics calculation: {e}. Please check filenames.")
        return None
    

# --- Load all artifacts ---
df, cluster_profile, scaler, df_state, kpi_data = load_data()
xgb_model = load_model()

if df is None or df_state is None or kpi_data is None:
    st.stop()

# --- 5. Main Application ---
st.title("Unlocking Equity & Efficiency")
st.markdown("### An Interactive Data-Driven Review of India's Crop Insurance Schemes (2018-2022)")

# --- Create Tabs for Navigation ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🚀 Key Insights",
    "🗺️ Interactive EDA",
    "Cluster Explorer (3D)",
    "🧠 The Modeling Journey",
    "🔮 Premium Predictor",
    "🏁 Conclusion & Future Work"
])

# ==============================================================================
# TAB 1: KEY INSIGHTS
# ==============================================================================
# ==============================================================================
# TAB 1: KEY INSIGHTS
# ==============================================================================
with tab1:
    st.header("Strategic Findings from the Data")
    st.markdown("Our analysis of 5,694 district-level records revealed a complex story of success, paradox, and opportunity.")
    # --- NEW: Dynamic KPI Metrics Grid ---
    st.subheader("Project Headline Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Policies (2022)",
            value=f"{kpi_data.get('total_policies_2022', 0) / 1_000_000:.0f}M",
            delta=f"{kpi_data.get('policy_delta', 0) * 100:.0f}% from 2018"
        )
        st.caption("Based on 'total_policies' sum for the year 2022.")

    with col2:
        st.metric(
            label="Districts Covered",
            value=f"{kpi_data.get('districts_covered', 0)}",
            delta=f"{kpi_data.get('states_covered', 0)} States"
        )
        st.caption("Total unique districts found in the dataset.")

    with col3:
        st.metric(
            label="Avg. Premium (PMFBY)",
            value=f"₹{kpi_data.get('avg_premium_pmfby', 0):,.0f}",
            delta=f"{kpi_data.get('prem_delta', 0) * 100:.1f}% from 2018"
        )
        st.caption("Scheme-wide average of 'gross_premium' for PMFBY.")
        
    with col4:
        st.metric(
            label="Female Participation",
            value=f"{kpi_data.get('avg_female_pct', 0):.1f}%",
            delta="Major Equity Gap"
        )
        st.caption("Average 'female' percentage across all districts.")

    st.markdown("---")
    # --- TOP ROW ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("The 'Vulnerable Farmer Paradox'")
        st.markdown(
            "The number of farmer policies **surged by 80%**[cite: 124], yet the total insured sum declined. "
            "This is explained by the scheme's deep success in reaching **small and marginal farmers (81.1% of all beneficiaries)**[cite: 128], "
            "who insure smaller plots[cite: 125]."
        )
        st.metric(label="Small + Marginal Farmers", value="81.1%")
    
    with col2:
        st.warning("The Demographic Equity Gap")
        st.markdown(
           "Despite reaching smallholders, a significant **gender gap** persists[cite: 128]. "
            "This was identified as a *critical* predictor in our most advanced model."
        )
        
        st.metric(label="Female Beneficiaries", value="14% - 20%")
        st.progress(17) # Visual for 17%
        
    with col3:
        st.success("The 'Scale vs. Inclusivity' Axis")
        st.markdown(
            "Our PCA analysis found that districts are defined by two key independent factors: "
            "**1. Financial Scale** (premiums, sum insured) and "
           "**2. Demographic Inclusivity** (female & small farmer %)."
        )
        
        st.metric(label="Key Dimensions Found", value="2")

    st.markdown("---")

    # --- BOTTOM ROW ---
    col4, col5, col6 = st.columns(3)

    with col4:
        st.info("Finding: Who *Really* Pays?")
        st.markdown(
            "Both schemes are heavily subsidized. For **PMFBY**, farmers pay only **14%** of the premium, with the Central (44%) and State (42%) governments covering the rest. "
           "WBCIS shifts more cost to the farmer (19%)." 
        )
        
        st.metric(label="Farmer Share (PMFBY)", value="14%") 

    with col5:
        st.warning("Finding: The 'Insurance Heartland'")
        st.markdown(
           "Insurance activity is not uniform. A high-volume 'heartland' exists in **Maharashtra, Rajasthan, and Madhya Pradesh**, which dominate national participation."
            "Policy hotspots include Hanumangarh and Beed." 
        )
        
        st.metric(label="Top Policy Hotspot", value="Hanumangarh") 

    with col6:
        st.success("Finding: The 'Policy vs. Profile' Paradox")
        st.markdown(
            "Surprisingly, our bivariate analysis found **no direct correlation** between a district's policy uptake (volume) and its demographic mix (like % female or % small farmer)." 
        )
        st.metric(label="Correlation (Policy vs. % Female)", value="~0.05") 
# ==============================================================================
# TAB 2: INTERACTIVE EDA
# ==============================================================================
# ==============================================================================
# TAB 2: INTERACTIVE EDA
# ==============================================================================
with tab2:
    st.header("Interactive Exploratory Data Analysis")
    
    st.subheader("1. Geographic Treemap (District-Level View)")
    st.markdown("This treemap shows state and district contributions. **Size = Total Policies**, **Color = Female Participation %**.")
    
    # Treemap of policies and inclusivity
    fig_tree = px.treemap(
        df, 
        path=[px.Constant('India'), 'state_name', 'district_name'], 
        values='total_policies', 
        color='female',
        color_continuous_scale='RdYlGn',
        title="Geographic Treemap: Participation (Size) vs. Female Inclusivity (Color)"
    )
    fig_tree.update_layout(margin = dict(t=50, l=25, r=25, b=25))
    st.plotly_chart(fig_tree, use_container_width=True)
    
    st.markdown("---")

    # 1. Define the remote URL to try
    GEOJSON_PATH = "Indian_States.txt"
    # --- Choropleth: robust loading & auto-detection ---
    st.subheader("2. India Choropleth Map (State-Level View)")

    @st.cache_data
    def load_local_geojson(file_path):
        """Loads a GeoJSON file from a local path."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            st.error(f"Map Error: GeoJSON file not found at '{file_path}'.")
            st.error(f"Please make sure the file '{file_path}' is in the same folder as the app.")
            return None
        except Exception as e:
            st.error(f"Error loading GeoJSON file: {e}")
            return None
    
    gj = load_local_geojson(GEOJSON_PATH)
    
    if gj is not None:
        # 3. If it succeeds, define settings and draw the map

        # --- IMPORTANT ---
        # This specific file ("india.geojson") uses 'state_name' as its key.
        featureidkey = "properties.NAME_1"

        # Define the metrics for the map
        map_options = {
            'total_policies': 'Total Policies (Sum)',
            'total_sum_insured': 'Total Sum Insured (Sum)',
            'avg_female_pct': 'Average Female Participation (%)',
            'avg_marginal_pct': 'Average Marginal Farmer (%)',
            'avg_gross_premium': 'Average Gross Premium (Mean)'
        }

        # Set the default metric to display
        # --- Add these lines back in its place ---
        metric_to_map = st.selectbox(
            "Select a metric to visualize on the map:",
            options=list(map_options.keys()),
            format_func=lambda x: map_options[x]
        )

        # 4. Create the choropleth
        fig_choro = px.choropleth(
            df_state,
            geojson=gj,
            locations='state_name_for_map',
            featureidkey=featureidkey,  # Uses the corrected key
            color=metric_to_map,
            color_continuous_scale="Viridis",
            title=f"State-Level Map: {map_options[metric_to_map]}",
            hover_name='state_name_for_map',
            hover_data={
                'original_states': True,
                'total_policies': ':.2s',
                'avg_female_pct': ':.2f'
            },
        )

        fig_choro.update_geos(fitbounds="locations", visible=True)
        fig_choro.update_layout(margin={"r":0,"t":50,"l":0,"b":0}, height=600)
        st.plotly_chart(fig_choro, use_container_width=True)

    
    # --- NEW: Add an explanation for the aggregation ---
    st.info("ℹ️ **Note:** The map's GeoJSON file is from before 2014. For visualization, data from **Telangana** has been aggregated with **Andhra Pradesh**.")

    st.markdown("---")
    
    

    # --- NEW: Upgraded Distribution Explorer with Tabs ---
    st.subheader("3. Deep Dive: Distribution Explorer")
    st.markdown("Explore how financial metrics are distributed across key categories.")
    
    # Create the sub-tabs
    dist_tab1, dist_tab2, dist_tab3 = st.tabs([
        "Compare by Season", 
        "Compare by Scheme", 
        "Compare by Cluster"
    ])
    
    # --- Tab 1: Compare by Season ---
    with dist_tab1:
        st.markdown("**How do metrics differ between Kharif and Rabi?**")
        
        metric_to_plot_season = st.selectbox(
            "Select a Metric:",
            ('sum_insured', 'gross_premium', 'total_policies', 'area_insured'),
            key='dist_select_season' # Unique key
        )
        
        friendly_title_season = metric_to_plot_season.replace('_', ' ').title()

        fig_violin_season = px.violin(
            df,
            x='season',
            y=metric_to_plot_season,
            color='season',
            log_y=True,
            box=True,
            points=False,
            title=f"Distribution of {friendly_title_season} by Season"
        )
        fig_violin_season.update_layout(
            xaxis_title="Season",
            yaxis_title=f"{friendly_title_season} (Log Scale)"
        )
        st.plotly_chart(fig_violin_season, use_container_width=True)
        
        # --- NEW: Conditional Insight ---
        if metric_to_plot_season == 'area_insured':
            st.info(
                f"**💡 Key Insight (Median vs. Total):**\n\n"
                f"You spotted it! The *median* {friendly_title_season} (the white dot) for Rabi appears higher than for Kharif. \n\n"
                f"This is the **'Vulnerable Farmer Paradox'** in action. As your report confirms, Kharif *dominates* in total scale [cite: Bivariate Analysis Report], "
                f"but its median is pulled *down* by a massive number of small and marginal farmers [cite: National Scale & The 'Vulnerable Farmer Paradox']."
            )

    # --- Tab 2: Compare by Scheme ---
    with dist_tab2:
        st.markdown("**How do metrics differ between PMFBY and WBCIS?**")
        
        metric_to_plot_scheme = st.selectbox(
            "Select a Metric:",
            ('sum_insured', 'gross_premium', 'total_policies', 'area_insured'),
            key='dist_select_scheme' # Unique key
        )
        
        friendly_title_scheme = metric_to_plot_scheme.replace('_', ' ').title()

        fig_violin_scheme = px.violin(
            df,
            x='scheme',
            y=metric_to_plot_scheme,
            color='scheme',
            log_y=True,
            box=True,
            points=False,
            title=f"Distribution of {friendly_title_scheme} by Scheme"
        )
        fig_violin_scheme.update_layout(
            xaxis_title="Scheme",
            yaxis_title=f"{friendly_title_scheme} (Log Scale)"
        )
        st.plotly_chart(fig_violin_scheme, use_container_width=True)
        
        # --- NEW: Conditional Insight ---
        if metric_to_plot_scheme in ('area_insured', 'sum_insured'):
            st.info(
                f"**💡 Key Insight (Median vs. Total):**\n\n"
                f"This plot clearly shows that the *typical participant* in the niche WBCIS scheme may have a higher median {friendly_title_scheme} than the typical PMFBY participant.\n\n"
                f"However, your report confirms that **PMFBY is the dominant scheme in *total scale***, with far higher total values across the board [cite: Bivariate Analysis Report, Multivariate Analysis Report]. "
                f"The PMFBY median is lower because its participation is overwhelmingly composed of small and marginal farmers (81.1%) [cite: Demographic Inclusivity & Equity]."
            )
        elif metric_to_plot_scheme == 'total_policies':
            st.info(
                f"**💡 Key Insight (Scale):**\n\n"
                f"This plot confirms your report's finding: **PMFBY is the dominant scheme** in participation, operating on a scale that is orders of magnitude larger than the niche WBCIS scheme [cite: Bivariate Analysis Report, Multivariate Analysis Report]."
            )


    # --- Tab 3: Compare by Cluster ---
    with dist_tab3:
        st.markdown("**How do metrics differ across our 5 District Archetypes?**")
        
        metric_to_plot_cluster = st.selectbox(
            "Select a Metric:",
            ('sum_insured', 'gross_premium', 'total_policies', 'area_insured'),
            key='dist_select_cluster' # Unique key
        )
        
        friendly_title_cluster = metric_to_plot_cluster.replace('_', ' ').title()

        fig_violin_cluster = px.violin(
            df,
            x='cluster_name',
            y=metric_to_plot_cluster,
            color='cluster_name',
            log_y=True,
            box=True,
            points=False,
            title=f"Distribution of {friendly_title_cluster} by Cluster Archetype"
        )
        fig_violin_cluster.update_layout(
            xaxis_title="Cluster Archetype",
            yaxis_title=f"{friendly_title_cluster} (Log Scale)",
            xaxis={'categoryorder':'total descending'} # Order by the total size
        )
        st.plotly_chart(fig_violin_cluster, use_container_width=True)
        st.caption("This view clearly visualizes the financial scale of each cluster, reinforcing the findings from the 'Cluster Explorer' tab.")


# ==============================================================================
# TAB 3: CLUSTER EXPLORER (3D)
# ==============================================================================
with tab3:
    st.header("The 5 District 'Archetypes' (3D Cluster Explorer)")
    st.markdown(
        "This **interactive 3D plot** visualizes our clustering results. It maps every district in India based on the "
        "two key dimensions from our PCA: **Scale** (Policies & Sum Insured) and **Inclusivity** (Female %)."
    )

    # 3D Scatter Plot
    fig_3d = px.scatter_3d(
        df,
        x='log_total_policies',
        y='log_sum_insured',
        z='female',
        color='cluster_name',
        title="3D Cluster Explorer: Scale vs. Inclusivity",
        labels={
            'log_total_policies': 'Participation Scale (Log)',
            'log_sum_insured': 'Financial Scale (Log)',
            'female': 'Female Participation (%)'
        },
        hover_data=['district_name', 'state_name']
    )
    fig_3d.update_traces(marker=dict(size=3, opacity=0.7))
    st.plotly_chart(fig_3d, use_container_width=True)
    
    st.subheader("Cluster Profiles")
    st.dataframe(cluster_profile.style.format(precision=2), use_container_width=True)
    
    for cluster_num, name in CLUSTER_NAMES.items():
        with st.expander(f"**Cluster {cluster_num}: {name}**"):
            st.write(CLUSTER_DESCRIPTIONS[cluster_num])

# ==============================================================================
# TAB 4: THE MODELING JOURNEY
# ==============================================================================
# ==============================================================================
# TAB 4: THE MODELING JOURNEY
# ==============================================================================
# TAB 4: THE MODELING JOURNEY
# ==============================================================================
with tab4:
    st.header("🧠 The Modeling Journey: Finding the Best Predictor")
    st.markdown(
        "To power our predictor, we needed the most accurate model. "
        "We built and tested **four different models**, from a simple baseline to advanced Deep Learning, "
        "to scientifically determine the champion."
    )

    st.subheader("Head-to-Head: Model Performance Comparison (Test Set)")
    st.caption("We compare 'Accuracy' (R-Squared) and 'Error' (MAE in Rupees).")
    
    # --- Call the function to get all metrics ---
    all_metrics = get_all_model_metrics()

    if all_metrics:
        # --- NEW: The Comprehensive Metrics Table ---
        model_data = {
            'Model': [
                '1. Simple Linear Regression', 
                '2. Multiple OLS Regression', 
                '3. Deep Learning (ANN)', 
                '4. XGBoost (Champion)'
            ],
            'R-Squared (Accuracy)': [
                all_metrics['r2_simple_ols'], # From Phase 1
                all_metrics['r2_multi_ols'],  # From Phase 2
                all_metrics['r2_ann'],        # Calculated
                all_metrics['r2_xgb']         # Calculated
            ],
            'MAE (Error in ₹)': [ 
                1515.83,  # From Phase 1
                884.00,   # From Phase 2
                766.34,   # From DL Experiment
                672.19    # From XGBoost Script
            ],
            'RMSE (Error in ₹)': [ 
                4961.99,  # From Phase 1
                2654.91,  # From Phase 2
                'N/A',    # We didn't calculate this for DL
                1979.47   # From XGBoost Script
            ],
            '% MAE Improvement (vs. Previous Model)': [
                '--',
                '41.7%',  # OLS vs. Simple
                '13.3%',  # DL vs. OLS
                '12.3%'   # XGB vs. DL
            ],
            '% MAE Improvement (vs. Baseline)': [
                '--',
                '41.7%',
                '49.4%',
                '55.7%'
            ]
        }
        
        df_metrics = pd.DataFrame(model_data)
        
        # Format the table for better presentation
        st.dataframe(
            df_metrics.set_index('Model').style.format(
                {
                    'R-Squared (Accuracy)': '{:.1%}', # Format as percentage
                    'MAE (Error in ₹)': '₹{:,.2f}',
                    'RMSE (Error in ₹)': lambda x: f'₹{x:,.2f}' if isinstance(x, (int, float)) else 'N/A'
                }
            ), 
            use_container_width=True
        )
        
        st.markdown(
            f"**Conclusion:** The **XGBoost** model is the clear champion, with the highest 'Accuracy' **({all_metrics['r2_xgb']:.1%})** and the lowest error (MAE: **₹672.19**)."
        )

    else:
        st.error("Could not load model metrics. Please check file paths and re-run.")


    st.markdown("---")
    
    st.subheader("Why Did XGBoost Win?")
    
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        st.markdown(
            "The experiment confirmed our hypothesis: **XGBoost is the state-of-the-art for this type of data.**"
            "\n\n"
            "* **Deep Learning (ANN):** This model was powerful and beat OLS. However, it's 'data-hungry' and our 5,623 rows weren't enough for it to reach its full potential."
            "\n\n"
            "* **XGBoost (The Champion):** This model is *designed* for tabular data. It was more efficient at learning from our 5,623 rows and successfully captured the non-linear patterns, leading to the highest accuracy and lowest error."
        )

    with col_b:
        st.image(
            FEATURE_IMPORTANCE_PATH, 
            caption="XGBoost's 'Wow' Insight: 'female' & 'marginal' are Top 3 predictors", 
            use_column_width=True
        )

    st.markdown("---")
    st.subheader("The Champion's Key Insights (XGBoost)")
    st.markdown(
    """
    The winning model's feature importance plot gave us our most powerful 'wow' insight:

    1. **`log_sum_insured`** is the #1 predictor, as expected.  
    2. **`female` and `marginal` farmer percentages** are the **#2 and #3 most important features!**

    **This proves that social equity is a critical financial indicator, not just a social goal.**  
    The model *needed* this demographic data to make its most accurate predictions.
    """
)

# ==============================================================================
# TAB 5: PREMIUM PREDICTOR
# ==============================================================================
with tab5:
    st.header("🔮 Interactive Premium Predictor Tool")
    st.markdown("Estimate the `gross_premium` for a district using our state-of-the-art **XGBoost model**.")

    col1, col2 = st.columns(2)
    
    with col1:
        # --- User Inputs ---
        state_list = df['state_name'].unique()
        selected_state = st.selectbox("1. Select State:", state_list, index=list(state_list).index('Maharashtra'))
        
        district_list = df[df['state_name'] == selected_state]['district_name'].unique()
        selected_district = st.selectbox("2. Select District:", district_list)
        
        season_list = df['season'].unique()
        selected_season = st.selectbox("3. Select Season:", season_list)
    
    with col2:
        # Get the most recent record for this district to use as a baseline
        baseline_record = df[
            (df['state_name'] == selected_state) &
            (df['district_name'] == selected_district)
        ].sort_values('year', ascending=False).iloc[0:1]
        
        if baseline_record.empty:
            st.error("No data for this district. Using a default record.")
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
    if st.button("Predict Gross Premium", type="primary", use_container_width=True):
        if xgb_model is None:
            st.error("Model not loaded. Prediction failed.")
        else:
            with st.spinner("Running the model..."):
                # 1. Get the model's feature list
                model_features = xgb_model.get_booster().feature_names
                
                # 2. Create an input DataFrame from the baseline record
                pred_input = baseline_record.copy()
                
                # 3. Overwrite with user-selected values
                pred_input['sum_insured'] = val_sum_insured
                pred_input['area_insured'] = val_area_insured
                pred_input['total_policies'] = val_total_policies
                pred_input['season'] = selected_season
                
                # 4. Re-create all engineered features
                
                # Log transforms (using the pre-fitted scaler)
                numeric_features = ['log_sum_insured', 'log_area_insured', 'log_total_policies']
                pred_input['log_sum_insured'] = np.log1p(pred_input['sum_insured'])
                pred_input['log_area_insured'] = np.log1p(pred_input['area_insured'])
                pred_input['log_total_policies'] = np.log1p(pred_input['total_policies'])

                # Encodings
                pred_input['scheme_encoded'] = pred_input['scheme'].map({'PMFBY': 1, 'WBCIS': 0}).fillna(0)
                pred_input['season_encoded'] = pred_input['season'].map({'Kharif': 1, 'Rabi': 0}).fillna(0)
                
                # State dummies
                top_states = [
                    'Uttar_Pradesh', 'Madhya_Pradesh', 'Maharashtra', 'Chhattisgarh', 
                    'Rajasthan', 'Tamil_Nadu', 'Odisha', 'Assam', 'Uttarakhand', 'Telangana'
                ]
                state_name_clean = selected_state.replace(' ', '_').replace('&', 'and')
                for state in top_states:
                    col_name = f"state_is_{state}"
                    if col_name in model_features:
                        pred_input[col_name] = 1 if state_name_clean == state else 0
                
                if 'state_is_Other' in model_features:
                    pred_input['state_is_Other'] = 1 if state_name_clean not in top_states else 0
                
                # Interaction terms
                pred_input['scheme_x_log_sum_insured'] = pred_input['scheme_encoded'] * pred_input['log_sum_insured']
                pred_input['season_x_log_sum_insured'] = pred_input['season_encoded'] * pred_input['log_sum_insured']
                
                # Cluster dummies
                cluster_id = baseline_record['cluster_id'].values[0]
                for i in range(5): # Assuming K=5
                    cluster_col = f'cluster_{i}'
                    if cluster_col in model_features:
                        pred_input[cluster_col] = 1 if i == cluster_id else 0

                # 5. Ensure order is correct
                pred_input_final = pred_input[model_features]
                
                # 6. Predict!
                prediction_log = xgb_model.predict(pred_input_final)[0]
                prediction_inr = np.expm1(prediction_log)
                
                st.success(f"**Predicted Gross Premium: ₹ {prediction_inr:,.2f} (Lakhs)**")
                
                # The "Wow" Insight
                cluster_name = CLUSTER_NAMES.get(cluster_id, "Unknown")
                cluster_desc = CLUSTER_DESCRIPTIONS.get(cluster_id, "No description available.")
                st.info(f"**District Profile Insight:**\n"
                        f"This district belongs to **Cluster {cluster_id}: {cluster_name}**.\n\n"
                        f"**What this means:** {cluster_desc}")


# ==============================================================================
# TAB 6: CONCLUSION & FUTURE WORK
# ==============================================================================
with tab6:
    st.header("🏁 Conclusion & Future Work")
    
    st.subheader("Summary of Major Findings")
    st.markdown(
        """
        This project successfully demonstrated an end-to-end data science workflow, moving from 6,161 raw records to a fully interactive predictive dashboard.
        
        Our analysis proved that a "one-size-fits-all" policy is inefficient and confirmed several key findings:
        
        1.  **The "Vulnerable Farmer Paradox":** We found that while policy participation surged by 80%, the average insured sum declined. This is explained by the scheme's deep success in reaching its core target: **81.1% Small & Marginal Farmers**.
            
        2.  **Data-Driven Archetypes:** We used K-Means clustering to discover **5 distinct "District Archetypes"** (like the "Inclusive Powerhouse" vs. the "Mainstream Heartland") that share common profiles, proving that districts need tailored strategies.
            
        3.  **Equity as a Financial Predictor:** Our champion **XGBoost model** (which was 55.7% more accurate than a simple baseline) identified a "wow" insight: a district's **`% female` and `% marginal` farmer** rates are the **#2 and #3 most important predictors** of its financial premium profile.
        """
    )
    
    st.markdown("---")
    
    st.subheader("Limitations & Future Work")
    
    st.warning(
        """
        **Limitation: No Claims Data**
        
        The single biggest limitation of this study is the absence of `claims_paid` data. 
        Our models can accurately predict the *premium* (the cost), but not the *risk* (the payout).
        """
    )
    
    st.success(
        """
        **Future Work: A True Risk-Prediction Model**
        
        The clear next step is to acquire claims data. This would unlock the most valuable analysis:
        
        * **Loss Ratio Analysis:** Calculating the `Loss Ratio (Claims / Premium)` for every district and cluster.
        * **Risk Prediction Model:** Building a new classification model to predict which districts are at high risk of significant claims, creating a powerful tool for proactive government budgeting and risk management.
        """
    )


import streamlit as st
import streamlit.components.v1 as components

# --- Modern Elegant Footer ---
components.html(
    """
    <style>
        .footer-container {
            width: 100%;
            background: #ffffff;
            color: #222;
            font-family: 'Inter', sans-serif;
            padding: 50px 20px 40px 20px;
            border-radius: 18px;
            box-shadow: 0 6px 25px rgba(0, 0, 0, 0.08);
            margin-top: 60px;
            text-align: center;
            transition: all 0.3s ease;
        }

        .footer-container:hover {
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
        }

        .footer-title {
            font-weight: 700;
            font-size: 22px;
            color: #111;
            margin-bottom: 8px;
        }

        .footer-subtitle {
            font-size: 15px;
            color: #444;
            max-width: 780px;
            margin: 0 auto 25px auto;
            line-height: 1.6;
        }

        .footer-grid {
            display: flex;
            justify-content: center;
            gap: 80px;
            flex-wrap: wrap;
            margin-bottom: 25px;
            text-align: left;
        }

        .footer-col {
            font-size: 14px;
            color: #333;
            line-height: 1.6;
        }

        .footer-col b {
            color: #111;
        }

        .footer-line {
            border: none;
            border-top: 1px solid #eee;
            margin: 25px auto;
            width: 85%;
        }

        .footer-note {
            font-size: 13px;
            color: #777;
        }
    </style>

    <div class="footer-container">
        
        <div class="footer-note">
        PMFBY Data-Driven Review Dashboard | By- Ameen Khan | 
        © 2025 | All rights reserved</div>
    </div>
    """,
    height=420,
)
