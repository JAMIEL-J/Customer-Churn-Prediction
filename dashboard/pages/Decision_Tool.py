"""
Decision Tool Page - CORE of the Dashboard

Interactive threshold optimization.
This is where business decisions are made.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Decision Tool", page_icon="🎯", layout="wide")

st.title("🎯 Decision Tool")
st.markdown("*Optimize your retention strategy*")

# Get project root (works on both local and Streamlit Cloud)
def get_project_root():
    """Get the project root directory."""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent

# Load pre-computed threshold sweep data
@st.cache_data
def load_sweep_data():
    root = get_project_root()
    lr_sweep = pd.read_csv(root / 'reports' / 'lr_threshold_sweep.csv')
    rf_sweep = pd.read_csv(root / 'reports' / 'rf_threshold_sweep.csv')
    return lr_sweep, rf_sweep

try:
    lr_sweep, rf_sweep = load_sweep_data()
    
    st.markdown("---")
    
    # Controls
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("⚙️ Controls")
        
        # Model selector
        model = st.selectbox(
            "Select Model",
            options=["Logistic Regression", "Random Forest"],
            index=0
        )
        
        # Get appropriate sweep data
        sweep_df = lr_sweep if model == "Logistic Regression" else rf_sweep
        
        # Threshold slider (bounded to reasonable range)
        min_thresh = sweep_df['Threshold'].min()
        max_thresh = sweep_df['Threshold'].max()
        
        # Find optimal threshold
        optimal_idx = sweep_df['Net_ROI'].idxmax()
        optimal_threshold = sweep_df.loc[optimal_idx, 'Threshold']
        
        threshold = st.slider(
            "Classification Threshold",
            min_value=float(min_thresh),
            max_value=float(max_thresh),
            value=float(optimal_threshold),
            step=0.05,
            help="Lower = more customers contacted, higher recall. Higher = fewer contacts, higher precision."
        )
        
        st.markdown(f"*Optimal threshold: {optimal_threshold:.2f}*")
    
    with col2:
        st.subheader(f"📊 Results at Threshold = {threshold:.2f}")
        
        # Find closest threshold in data
        closest_idx = (sweep_df['Threshold'] - threshold).abs().idxmin()
        row = sweep_df.loc[closest_idx]
        
        # Big metrics
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            st.metric(label="Customers Contacted", value=f"{int(row['Customers_Contacted']):,}")
        
        with metric_cols[1]:
            st.metric(label="Retention Cost", value=f"${row['Retention_Cost']:,.0f}")
        
        with metric_cols[2]:
            st.metric(label="Revenue Saved", value=f"${row['Revenue_Saved']:,.0f}")
        
        with metric_cols[3]:
            # NET ROI - THE BIG NUMBER
            delta_text = f"vs. ${sweep_df.loc[optimal_idx, 'Net_ROI']:,.0f} optimal" if threshold != optimal_threshold else "OPTIMAL"
            st.metric(label="💰 NET ROI", value=f"${row['Net_ROI']:,.0f}", delta=delta_text)
        
        st.markdown("---")
        
        # Confusion matrix breakdown
        st.subheader("Confusion Matrix Breakdown")
        cm_cols = st.columns(4)
        
        with cm_cols[0]:
            st.metric("✅ True Positives (TP)", f"{int(row['TP']):,}")
            st.caption("Churners correctly identified")
        
        with cm_cols[1]:
            st.metric("❌ False Positives (FP)", f"{int(row['FP']):,}")
            st.caption("Non-churners wrongly flagged")
        
        with cm_cols[2]:
            st.metric("🚨 False Negatives (FN)", f"{int(row['FN']):,}")
            st.caption("Churners missed")
        
        with cm_cols[3]:
            st.metric("✅ True Negatives (TN)", f"{int(row['TN']):,}")
            st.caption("Non-churners correctly ignored")
        
        # Precision/Recall
        st.markdown("---")
        pr_cols = st.columns(2)
        
        with pr_cols[0]:
            st.metric("Precision", f"{row['Precision']:.1%}")
            st.caption("Of those contacted, what % are actual churners?")
        
        with pr_cols[1]:
            st.metric("Recall", f"{row['Recall']:.1%}")
            st.caption("Of all churners, what % did we catch?")
    
    # Full decision table
    st.markdown("---")
    st.subheader("📋 Full Decision Table")
    
    display_df = sweep_df.copy()
    display_df['Threshold'] = display_df['Threshold'].apply(lambda x: f"{x:.2f}")
    display_df['Precision'] = display_df['Precision'].apply(lambda x: f"{x:.1%}")
    display_df['Recall'] = display_df['Recall'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(
        display_df[['Threshold', 'Customers_Contacted', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'Net_ROI']],
        use_container_width=True,
        hide_index=True
    )

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please run `04_business_impact.ipynb` first to generate the required reports.")
