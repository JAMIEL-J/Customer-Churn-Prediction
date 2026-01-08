"""
Explainability Page - Why the Model Flags Churn

Keep this simple and restrained.
About trust, not exploration.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
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
    rf_model = joblib.load(_root / 'models' / 'random_forest.pkl')
    lr_model = joblib.load(_root / 'models' / 'logistic_regression.pkl')
    
    with open(_root / 'models' / 'feature_cols.txt', 'r') as f:
        feature_cols = f.read().strip().split('\n')
    
    return lr_model, rf_model, feature_cols

try:
    features, root = load_data()
    lr_model, rf_model, feature_cols = load_models(root)
    
    st.markdown("---")
    
    st.subheader("📊 Top Churn Drivers")
    st.markdown("*Features that most strongly predict churn*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Logistic Regression Coefficients")
        
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
        
        st.dataframe(coef_df[['Feature', 'Coefficient', 'Direction']], use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### Random Forest Feature Importance")
        
        imp_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf_model.feature_importances_
        })
        imp_df = imp_df.sort_values('Importance', ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(imp_df['Feature'], imp_df['Importance'], color='#3498db')
        ax.set_xlabel('Importance Score')
        ax.set_title('Top 10 Features by Importance')
        plt.tight_layout()
        st.pyplot(fig)
        
        imp_df['Importance'] = imp_df['Importance'].apply(lambda x: f"{x:.4f}")
        st.dataframe(imp_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    st.subheader("💡 Key Insights")
    
    st.markdown("""
    **What drives churn?**
    
    Based on both models, the top churn drivers typically include:
    
    1. **Contract Type** - Month-to-month contracts have much higher churn
    2. **Tenure** - Newer customers are more likely to churn
    3. **Monthly Charges** - Higher charges correlate with higher churn
    4. **Internet Service Type** - Fiber optic users show higher churn
    5. **Add-on Services** - Lack of services (security, backup, support) increases risk
    
    *These align with EDA hypotheses validated in earlier analysis.*
    """)
    
    st.info("""
    **Why these features matter:**
    
    - Contract features are **actionable** → Offer contract incentives
    - Tenure is **predictive** → Focus on new customer retention
    - Add-on services are **protective** → Bundle services for engagement
    """)

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please run `03_modeling.ipynb` first to generate the required models.")
