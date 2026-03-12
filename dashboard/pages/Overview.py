"""
Overview Page - Executive Snapshot

READ-ONLY. No sliders. No toggles.
Answers: "Is this worth caring about?"
"""

import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")

st.title("📊 Executive Overview")
st.markdown("*Is this worth caring about?*")

# Get project root (works on both local and Streamlit Cloud)
def get_project_root():
    """Get the project root directory."""
    # Current file is in dashboard/pages/
    current_file = Path(__file__).resolve()
    # Go up: pages -> dashboard -> project_root
    return current_file.parent.parent.parent

# Load pre-computed data
@st.cache_data
def load_data():
    root = get_project_root()
    
    # Load recommendations
    recommendations = pd.read_csv(root / 'reports' / 'business_recommendations.csv')
    
    # Load features data for customer count and churn rate
    features = pd.read_csv(root / 'data' / 'processed' / 'churn_features.csv')
    
    return recommendations, features

try:
    recommendations, features = load_data()
    
    # Extract key metrics
    total_customers = len(features)
    churn_rate = (features['Churn'] == 'Yes').mean()
    churners = (features['Churn'] == 'Yes').sum()
    
    # Get CLV from recommendations
    clv_row = recommendations[recommendations['Metric'] == 'CLV (Average)']
    clv = float(clv_row['Value'].values[0].replace('$', '').replace(',', '')) if len(clv_row) > 0 else 1554.28
    
    # Revenue at risk = churners × CLV
    revenue_at_risk = churners * clv
    
    # Get optimal ROI
    roi_single = recommendations[recommendations['Metric'] == 'Expected Net ROI (Single)']
    roi_segment = recommendations[recommendations['Metric'] == 'Expected Net ROI (Segmented)']
    
    optimal_roi_single = roi_single['Value'].values[0] if len(roi_single) > 0 else "$105,203.83"
    optimal_roi_segment = roi_segment['Value'].values[0] if len(roi_segment) > 0 else "$106,373.83"
    
    # Key Metrics Row
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Total Customers", value=f"{total_customers:,}")
    
    with col2:
        st.metric(label="Overall Churn Rate", value=f"{churn_rate:.1%}")
    
    with col3:
        st.metric(label="Revenue at Risk", value=f"${revenue_at_risk:,.0f}")
    
    with col4:
        st.metric(label="Expected ROI (Optimal)", value=optimal_roi_segment)
    
    st.markdown("---")
    
    # Summary
    st.subheader("📋 Recommendation Summary")
    st.dataframe(recommendations, width="stretch", hide_index=True)
    
    # Best Model highlight
    model_row = recommendations[recommendations['Metric'] == 'Recommended Model']
    best_model = model_row['Value'].values[0] if len(model_row) > 0 else "XGBoost"
    
    st.success(f"**Primary Recommendation**: Deploy the **{best_model}** model. It provides the highest AUC and best captures non-linear churn drivers while natively handling class imbalance.")
    
    # Key insight
    st.markdown("---")
    st.info(f"""
    **Key Executive Insights**:
    1. **Model Performance**: {best_model} identifies ~84% of potential churners (Recall) while maintaining high precision.
    2. **Segmented ROI**: Targeting High-Value and Low-Value segments with different thresholds yields the maximum financial return.
    3. **Strategy Sensitivity**: ROI remains robust across varying retention success rates (15-30%), though intervention cost management is the primary profitability driver.
    
    → Use the **Decision Tool** to adjust thresholds for specific budget constraints.
    """)

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please run `04_business_impact.ipynb` first to generate the required reports.")
