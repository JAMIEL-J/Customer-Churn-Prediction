"""
Explainability Page - Why the Model Flags Churn

Keep this simple and restrained.
About trust, not exploration.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb
from pathlib import Path

st.set_page_config(page_title="Explainability", page_icon="🔍", layout="wide")

st.title("🔍 Model Explainability")
st.markdown("*Why does the model flag churn?*")

# Get project root (works on both local and Streamlit Cloud)
def get_project_root():
    """Get the project root directory."""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent

@st.cache_data
def load_data():
    root = get_project_root()
    features = pd.read_csv(root / 'data' / 'processed' / 'churn_features.csv')
    return features, root

@st.cache_resource
def load_models(_root):
    xgb_model = joblib.load(_root / 'models' / 'xgboost.pkl')
    lr_model = joblib.load(_root / 'models' / 'logistic_regression.pkl')
    rf_model = joblib.load(_root / 'models' / 'random_forest.pkl')
    
    with open(_root / 'models' / 'feature_cols.txt', 'r') as f:
        feature_cols = f.read().strip().split('\n')
    
    return lr_model, rf_model, xgb_model, feature_cols

try:
    features, root = load_data()
    lr_model, rf_model, xgb_model, feature_cols = load_models(root)
    
    st.markdown("---")
    
    st.subheader("📊 Top Churn Drivers")
    st.markdown("*Features that most strongly predict churn*")
    
    st.markdown("### Select Model to Explain")
    model_imp = st.selectbox("Model", ["Logistic Regression", "Random Forest", "XGBoost"], index=2, label_visibility="collapsed")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Native Feature Importance")
        
        if model_imp == "Logistic Regression":
            coef_df = pd.DataFrame({
                'Feature': feature_cols,
                'Coefficient': lr_model.coef_[0]
            })
            coef_df['Abs_Coef'] = coef_df['Coefficient'].abs()
            coef_df = coef_df.sort_values('Abs_Coef', ascending=False).head(10)
            coef_df['Direction'] = coef_df['Coefficient'].apply(
                lambda x: '↑ Increases Churn' if x > 0 else '↓ Decreases Churn'
            )
            
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#e74c3c' if c > 0 else '#2ecc71' for c in coef_df['Coefficient']]
            ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
            ax.set_xlabel('Coefficient (positive = increases churn risk)')
            ax.set_title('Top 10 Features by Coefficient Magnitude')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            st.dataframe(coef_df[['Feature', 'Coefficient', 'Direction']], width="stretch", hide_index=True)
        elif model_imp == "Random Forest":
            imp_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(imp_df['Feature'], imp_df['Importance'], color='#3498db')
            ax.set_xlabel('Gini Importance')
            ax.set_title('Top 10 Features by RF Importance')
            plt.tight_layout()
            st.pyplot(fig)
            st.dataframe(imp_df, width="stretch", hide_index=True)
        else:
            xgb_imp_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': xgb_model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(xgb_imp_df['Feature'], xgb_imp_df['Importance'], color='#f1c40f')
            ax.set_xlabel('F Score (Importance)')
            ax.set_title('Top 10 Features by XGBoost Importance')
            plt.tight_layout()
            st.pyplot(fig)
            st.dataframe(xgb_imp_df, width="stretch", hide_index=True)
    
    with col2:
        st.markdown(f"### SHAP Summary Plot")
        
        if model_imp == "Logistic Regression":
            summary_file = 'lr_shap_summary.png'
            force_file = 'lr_shap_force.png'
        elif model_imp == "Random Forest":
            summary_file = 'rf_shap_summary.png'
            force_file = 'rf_shap_force.png'
        else:
            summary_file = 'xgb_shap_summary.png'
            force_file = 'xgb_shap_force.png'
            
        shap_summary_path = root / 'models' / summary_file
        if shap_summary_path.exists():
            st.image(str(shap_summary_path), width="stretch")
            st.caption("SHAP Summary Plot shows feature impacts on individual predictions across the dataset.")
        else:
            st.info("SHAP summary plot not found. Please run the modeling notebook.")
            
        st.markdown("### Single Customer Risk Profile")
        shap_force_path = root / 'models' / force_file
        if shap_force_path.exists():
            st.image(str(shap_force_path), width="stretch")
            st.caption("SHAP Force Plot explaining features pushing higher/lower risk for a single customer.")
        else:
            st.info("SHAP force plot not available.")
    
    st.markdown("---")
    
    st.subheader("💡 Key Insights")
    
    st.markdown("""
    **What drives churn?**
    
    Using **XGBoost** and **SHAP**, we identify the most accurate drivers by looking at non-linear effects and feature interactions:
    
    1. **Contract Type** - Month-to-month contracts are consistently the strongest risk factor.
    2. **Tenure & Charge Ratio** - Newer customers with high price-to-tenure ratios are at extreme risk.
    3. **Tech Support Tickets** - High ticket frequency often indicates customer frustration and pending churn.
    4. **Fiber Optic Service** - Customers on fiber optic show higher sensitivity to price and service issues.
    5. **Missing Add-ons** - Customers without security or backup services lack "stickiness" and churn faster.
    
    *SHAP values quantify exactly how much each feature pushes the individual churn probability.*
    """)
    
    st.info("""
    **Actionable Insights:**
    
    - **Retention Focus**: Prioritize customers in the first 6 months of a month-to-month contract.
    - **Upsell Opportunity**: Bundling protective services (Online Security, Tech Support) significantly reduces predicted churn risk.
    - **Support Intervention**: High Tech-Support volume should trigger a proactive customer success reach-out.
    """)

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please run `03_modeling.ipynb` first to generate the required models.")
