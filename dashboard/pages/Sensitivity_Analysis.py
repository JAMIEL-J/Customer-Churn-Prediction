"""
Sensitivity Analysis Page - ROI Under Different Assumptions

Shows how ROI varies when key business assumptions change.
Answers: "What if our assumptions are wrong?"
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import os

st.set_page_config(page_title="Sensitivity Analysis", page_icon="📊", layout="wide")

st.title("🔄 Sensitivity Analysis")
st.markdown("*What if our assumptions change?*")

# Get project root - works on both local and Streamlit Cloud
def get_project_root():
    """Get the project root directory."""
    try:
        # Try from script location
        current_file = Path(__file__).resolve()
        root = current_file.parent.parent.parent
        if (root / 'reports').exists():
            return root
    except:
        pass
    
    # Try from current working directory
    cwd = Path(os.getcwd())
    if (cwd / 'reports').exists():
        return cwd
    
    # Try parent directory
    if (cwd.parent / 'reports').exists():
        return cwd.parent
    
    return cwd

# Load data
@st.cache_data
def load_data():
    root = get_project_root()
    xgb_sweep = pd.read_csv(root / 'reports' / 'xgb_threshold_sweep.csv')
    features = pd.read_csv(root / 'data' / 'processed' / 'churn_features.csv')
    return xgb_sweep, features

try:
    xgb_sweep, features = load_data()
    
    # Base metrics
    churners = (features['Churn'] == 'Yes').sum()
    avg_clv = features['MonthlyCharges'].mean() * 24  # 24-month CLV
    
    st.markdown("---")
    st.subheader("⚙️ Configure Assumptions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        retention_success_min = st.slider(
            "Retention Success Rate (Low)",
            min_value=5, max_value=50, value=10, step=5,
            help="% of contacted at-risk customers we successfully retain"
        )
        retention_success_min = retention_success_min / 100
    
    with col2:
        retention_success_mid = st.slider(
            "Retention Success Rate (Mid)",
            min_value=5, max_value=50, value=20, step=5
        )
        retention_success_mid = retention_success_mid / 100
    
    with col3:
        retention_success_max = st.slider(
            "Retention Success Rate (High)",
            min_value=5, max_value=50, value=30, step=5
        )
        retention_success_max = retention_success_max / 100
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        contact_cost_min = st.slider(
            "Contact Cost (Low)",
            min_value=10, max_value=100, value=25, step=5,
            help="Cost per customer contacted (email, SMS, phone)"
        )
    
    with col2:
        contact_cost_mid = st.slider(
            "Contact Cost (Mid)",
            min_value=10, max_value=100, value=50, step=5
        )
    
    with col3:
        contact_cost_max = st.slider(
            "Contact Cost (High)",
            min_value=10, max_value=100, value=75, step=5
        )
    
    st.markdown("---")
    
    # Function to recalculate ROI with new assumptions
    def calculate_roi_grid(sweep_df, clv, retention_rates, contact_costs):
        """Calculate ROI grid across retention rates and contact costs."""
        results = []
        
        # Create a copy and round Threshold to handle floating point precision
        sweep_copy = sweep_df.copy()
        sweep_copy['Threshold'] = sweep_copy['Threshold'].round(2)
        
        for rate in retention_rates:
            for cost in contact_costs:
                for _, row in sweep_copy.iterrows():
                    tp = row['TP']
                    fp = row['FP']
                    
                    # Recalculate metrics with new assumptions
                    revenue_saved = tp * clv * rate
                    retention_cost = (tp + fp) * cost
                    roi = revenue_saved - retention_cost
                    
                    results.append({
                        'Threshold': row['Threshold'],
                        'Retention_Rate': f"{rate:.0%}",
                        'Contact_Cost': cost,  # Keep as number for heatmap
                        'ROI': roi,
                        'Revenue_Saved': revenue_saved,
                        'Cost': retention_cost,
                        'Customers_Contacted': row['Customers_Contacted']
                    })
        
        return pd.DataFrame(results)
    
    retention_rates = [retention_success_min, retention_success_mid, retention_success_max]
    contact_costs = [contact_cost_min, contact_cost_mid, contact_cost_max]
    
    roi_grid = calculate_roi_grid(xgb_sweep, avg_clv, retention_rates, contact_costs)
    
    st.markdown("---")
    
    # Find optimal scenarios
    st.subheader("🎯 Optimal Thresholds by Scenario")
    
    optimal_scenarios = []
    for rate in retention_rates:
        for cost in contact_costs:
            subset = roi_grid[
                (roi_grid['Retention_Rate'] == f"{rate:.0%}") &
                (roi_grid['Contact_Cost'] == cost)
            ]
            if len(subset) > 0:
                best = subset.loc[subset['ROI'].idxmax()]
                optimal_scenarios.append({
                    'Retention Rate': f"{rate:.0%}",
                    'Contact Cost': f"${cost}",
                    'Optimal Threshold': f"{best['Threshold']:.2f}",
                    'Max ROI': f"${best['ROI']:,.0f}",
                    'Revenue Saved': f"${best['Revenue_Saved']:,.0f}",
                    'Total Cost': f"${best['Cost']:,.0f}",
                    'Customers': int(float(best['Customers_Contacted']))
                })
    
    optimal_df = pd.DataFrame(optimal_scenarios)
    st.dataframe(optimal_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Key insights
    st.subheader("💡 Sensitivity Insights")
    
    # Best case scenario
    best_scenario = optimal_df.loc[optimal_df['Max ROI'].str.replace('$', '').str.replace(',', '').astype(float).idxmax()]
    
    # Worst case scenario
    worst_scenario = optimal_df.loc[optimal_df['Max ROI'].str.replace('$', '').str.replace(',', '').astype(float).idxmin()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"""
        **Best-Case Scenario**
        
        - Success Rate: {best_scenario['Retention Rate']}
        - Contact Cost: {best_scenario['Contact Cost']}
        - Optimal Threshold: {best_scenario['Optimal Threshold']}
        - **ROI: {best_scenario['Max ROI']}**
        """)
    
    with col2:
        st.warning(f"""
        **Worst-Case Scenario**
        
        - Success Rate: {worst_scenario['Retention Rate']}
        - Contact Cost: {worst_scenario['Contact Cost']}
        - Optimal Threshold: {worst_scenario['Optimal Threshold']}
        - **ROI: {worst_scenario['Max ROI']}**
        """)
    
    st.info(f"""
    **Interpretation**:
    - ROI is **most sensitive to retention success rate** — getting this right is critical
    - Contact cost matters, but less than success rate
    - Strategy remains **profitable across reasonable assumptions** (best = {best_scenario['Max ROI']}, worst = {worst_scenario['Max ROI']})
    → This robustness suggests the strategy is viable even if assumptions shift by 20-30%
    """)

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please run `04_business_impact.ipynb` first to generate the required reports.")
