"""
Segments Page - Who Gets Contacted

Answers: "Who exactly are we targeting, and why?"
This is where business teams actually act.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Segments", page_icon="👥", layout="wide")

st.title("👥 Customer Segments")
st.markdown("*Who exactly are we targeting, and why?*")

# Get project root (works on both local and Streamlit Cloud)
def get_project_root():
    """Get the project root directory."""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent

@st.cache_data
def load_data():
    root = get_project_root()
    features = pd.read_csv(root / 'data' / 'processed' / 'churn_features.csv')
    recommendations = pd.read_csv(root / 'reports' / 'business_recommendations.csv')
    return features, recommendations, root

@st.cache_resource
def load_models(_root):
    lr_model = joblib.load(_root / 'models' / 'logistic_regression.pkl')
    rf_model = joblib.load(_root / 'models' / 'random_forest.pkl')
    scaler = joblib.load(_root / 'models' / 'scaler.pkl')
    
    with open(_root / 'models' / 'feature_cols.txt', 'r') as f:
        feature_cols = f.read().strip().split('\n')
    
    return lr_model, rf_model, scaler, feature_cols

try:
    features, recommendations, root = load_data()
    lr_model, rf_model, scaler, feature_cols = load_models(root)
    
    st.markdown("---")
    
    # Create value segments
    charge_median = features['MonthlyCharges'].median()
    features['value_segment'] = np.where(
        features['MonthlyCharges'] >= charge_median, 'High-Value', 'Low-Value'
    )
    
    # Segment summary
    st.subheader("📊 High-Value vs Low-Value Segments")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### High-Value Customers")
        high_value = features[features['value_segment'] == 'High-Value']
        hv_churn_rate = (high_value['Churn'] == 'Yes').mean()
        hv_avg_charge = high_value['MonthlyCharges'].mean()
        
        st.metric("Count", f"{len(high_value):,}")
        st.metric("Churn Rate", f"{hv_churn_rate:.1%}")
        st.metric("Avg Monthly Charge", f"${hv_avg_charge:.2f}")
        st.metric("CLV (24 months)", f"${hv_avg_charge * 24:,.2f}")
        
        hv_thresh_row = recommendations[recommendations['Metric'] == 'High-Value Threshold']
        hv_threshold = float(hv_thresh_row['Value'].values[0]) if len(hv_thresh_row) > 0 else 0.20
        st.metric("Optimal Threshold", f"{hv_threshold:.2f}")
    
    with col2:
        st.markdown("### Low-Value Customers")
        low_value = features[features['value_segment'] == 'Low-Value']
        lv_churn_rate = (low_value['Churn'] == 'Yes').mean()
        lv_avg_charge = low_value['MonthlyCharges'].mean()
        
        st.metric("Count", f"{len(low_value):,}")
        st.metric("Churn Rate", f"{lv_churn_rate:.1%}")
        st.metric("Avg Monthly Charge", f"${lv_avg_charge:.2f}")
        st.metric("CLV (24 months)", f"${lv_avg_charge * 24:,.2f}")
        
        lv_thresh_row = recommendations[recommendations['Metric'] == 'Low-Value Threshold']
        lv_threshold = float(lv_thresh_row['Value'].values[0]) if len(lv_thresh_row) > 0 else 0.20
        st.metric("Optimal Threshold", f"{lv_threshold:.2f}")
    
    st.markdown("---")
    
    # Top N customers to contact
    st.subheader("📋 Top Customers to Contact")
    
    model_choice = st.radio("Model", ["Logistic Regression", "Random Forest"], horizontal=True)
    threshold = st.slider("Threshold", 0.1, 0.9, 0.35, 0.05)
    top_n = st.slider("Show Top N", 10, 100, 20, 10)
    
    # Prepare features
    X = features[feature_cols].copy()
    
    if model_choice == "Logistic Regression":
        X_scaled = scaler.transform(X)
        proba = lr_model.predict_proba(X_scaled)[:, 1]
    else:
        proba = rf_model.predict_proba(X)[:, 1]
    
    features['churn_probability'] = proba
    features['predicted_churn'] = (proba >= threshold).astype(int)
    
    # Filter to predicted churners, sorted by probability
    to_contact = features[features['predicted_churn'] == 1].copy()
    to_contact = to_contact.sort_values('churn_probability', ascending=False).head(top_n)
    
    # Display table
    display_cols = ['customerID', 'value_segment', 'MonthlyCharges', 'Contract', 'tenure', 'churn_probability']
    
    if len(to_contact) > 0:
        display_df = to_contact[display_cols].copy()
        display_df['churn_probability'] = display_df['churn_probability'].apply(lambda x: f"{x:.1%}")
        display_df['MonthlyCharges'] = display_df['MonthlyCharges'].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.success(f"**{len(to_contact)} customers** flagged for retention outreach at threshold {threshold:.2f}")
    else:
        st.warning("No customers flagged at this threshold.")

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please run `03_modeling.ipynb` and `04_business_impact.ipynb` first.")
