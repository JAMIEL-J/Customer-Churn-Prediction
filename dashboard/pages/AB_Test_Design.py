"""
A/B Test Design Page - Pilot & Validation Plan

Shows how to validate the churn prediction strategy with a pilot test.
Answers: "How do we prove this works?"
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from scipy import stats
import os

st.set_page_config(page_title="A/B Test Design", page_icon="📊", layout="wide")

st.title("🧪 A/B Test Design & Validation")
st.markdown("*How do we prove the retention strategy works?*")

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
    xgb_scores = pd.read_csv(root / 'reports' / 'xgb_customer_scores.csv')
    features = pd.read_csv(root / 'data' / 'processed' / 'churn_features.csv')
    return xgb_scores, features

try:
    xgb_scores, features = load_data()
    
    st.markdown("---")
    
    st.subheader("📋 Recommended Pilot Test Design")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Test Objective**
        
        Validate that contacting high-risk customers with retention offers 
        increases their likelihood to stay.
        
        **Success Metric**
        - 5%+ increase in retention rate for treatment group vs control
        """)
    
    with col2:
        st.info("""
        **Test Duration**
        
        - **Timeline**: 60 days observation + 30 days follow-up
        - **Power**: 80% (detect 5% lift with confidence)
        - **Significance Level**: 95% (α = 0.05)
        """)
    
    st.markdown("---")
    
    # Test parameters (user-configurable)
    st.subheader("⚙️ Configure Test Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        baseline_retention = st.slider(
            "Baseline Retention Rate (Control)",
            min_value=50, max_value=95, value=75,
            help="% of at-risk customers who stay without intervention"
        ) / 100
    
    with col2:
        expected_lift = st.slider(
            "Expected Retention Lift (Treatment)",
            min_value=1, max_value=20, value=5,
            help="% point increase in retention from intervention"
        ) / 100
    
    with col3:
        contact_cost = st.slider(
            "Intervention Cost per Customer",
            min_value=25, max_value=100, value=50,
            help="Cost of retention offer (discount, gift, etc.)"
        )
    
    st.markdown("---")
    
    # Get number of high-risk customers (probability >= 0.20)
    high_risk_threshold = 0.20
    high_risk_customers = xgb_scores[xgb_scores['churn_probability'] >= high_risk_threshold]
    
    treatment_size = st.slider(
        "Treatment Group Size",
        min_value=100, max_value=min(1000, len(high_risk_customers)),
        value=300,
        step=50,
        help="Number of customers to contact with retention offer"
    )
    
    control_size = treatment_size  # 1:1 test
    
    st.markdown("---")
    
    # Power analysis
    st.subheader("📊 Statistical Power Analysis")
    
    # Calculate required sample size
    # Using proportions z-test
    p1 = baseline_retention  # control
    p2 = baseline_retention + expected_lift  # treatment
    
    # Pooled proportion
    p_pool = (p1 + p2) / 2
    
    # Standard error
    se = np.sqrt(2 * p_pool * (1 - p_pool) / treatment_size)
    
    # Z-value for two-tailed test at 95% confidence
    z_critical = 1.96
    
    # Detectable effect size
    detectable_effect = z_critical * se
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Treatment Group Size", f"{treatment_size}")
    
    with col2:
        st.metric("Control Group Size", f"{control_size}")
    
    with col3:
        st.metric("Total Test Size", f"{treatment_size + control_size}")
    
    with col4:
        statistical_power = 1 - stats.norm.cdf(
            (z_critical - (abs(p2 - p1) / se)) / np.sqrt(treatment_size)
        ) if se > 0 else 0
        st.metric(
            "Statistical Power",
            f"{max(0, min(statistical_power, 1)) * 100:.0f}%",
            delta="↑ Good" if statistical_power > 0.80 else "↓ Review"
        )
    
    st.markdown("---")
    
    # Expected outcomes
    st.subheader("🎯 Expected Test Outcomes")
    
    avg_clv = features['MonthlyCharges'].mean() * 24
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Control group
        control_retained = int(control_size * baseline_retention)
        control_churned = control_size - control_retained
        
        st.info(f"""
        **Control Group** ({control_size} customers)
        
        - No intervention
        - Baseline retention: {baseline_retention:.0%}
        - Expected retained: **{control_retained}** ✅
        - Expected churned: **{control_churned}** ❌
        - Revenue impact: ${control_retained * avg_clv:,.0f}
        """)
    
    with col_right:
        # Treatment group
        treatment_retention = baseline_retention + expected_lift
        treatment_retained = int(treatment_size * treatment_retention)
        treatment_churned = treatment_size - treatment_retained
        additional_saved = treatment_retained - int(treatment_size * baseline_retention)
        
        st.success(f"""
        **Treatment Group** ({treatment_size} customers)
        
        - Retention offer at ${contact_cost}/customer
        - Expected retention: {treatment_retention:.0%}
        - Expected retained: **{treatment_retained}** ✅
        - Expected churned: **{treatment_churned}** ❌
        - Additional saved: **{additional_saved}** customers
        - Revenue impact: ${treatment_retained * avg_clv:,.0f}
        """)
    
    st.markdown("---")
    
    # ROI calculation
    st.subheader("💰 Expected ROI from Pilot")
    
    # Total intervention cost
    total_cost = treatment_size * contact_cost
    
    # Additional revenue from saved customers
    additional_revenue = additional_saved * avg_clv
    
    # Net ROI
    net_roi = additional_revenue - total_cost
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Intervention Cost", f"${total_cost:,.0f}")
    
    with col2:
        st.metric("Additional Revenue Saved", f"${additional_revenue:,.0f}")
    
    with col3:
        if net_roi > 0:
            st.success(f"""
            **Net ROI**
            
            ${net_roi:,.0f}
            
            {(additional_revenue / total_cost):.1f}x return
            """)
        else:
            st.warning(f"""
            **Net ROI**
            
            ${net_roi:,.0f}
            
            Test not profitable at these assumptions
            """)
    
    st.markdown("---")
    
    # Timeline and logistics
    st.subheader("📅 Implementation Timeline")
    
    timeline = pd.DataFrame({
        'Phase': [
            'Week 1: Setup',
            'Week 2: Randomization',
            'Weeks 3-8: Active Period',
            'Weeks 9-10: Observation',
            'Week 11: Analysis',
        ],
        'Activities': [
            '- Segment high-risk customers\n- Select treatment/control groups\n- Prepare retention offer',
            '- Random assignment (50/50 split)\n- Load test/control lists into CRM\n- Brief retention team',
            '- Execute retention outreach for treatment\n- Monitor response rates\n- Track conversions daily',
            '- Observe churn outcomes\n- Collect retention data\n- Calculate lift',
            '- Statistical analysis\n- Report findings\n- Make go/no-go decision for scale',
        ]
    })
    
    st.dataframe(timeline, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Success criteria
    st.subheader("✅ Test Success Criteria")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"""
        **Go Decision** (Scale to Full Rollout)
        
        ✓ Observed lift ≥ {expected_lift * 100:.0f}% (statistically significant)
        ✓ p-value < 0.05
        ✓ Net ROI > $0
        ✓ Confidence interval does not cross 0
        """)
    
    with col2:
        st.warning(f"""
        **No-Go Decision** (Iterate & Retrain)
        
        ✗ Observed lift < {expected_lift * 100:.0f}% or not significant
        ✗ p-value ≥ 0.05
        ✗ Net ROI < $0
        ✗ Confidence interval crosses 0
        
        → Investigate why & try different approach
        """)
    
    st.markdown("---")
    
    # Scale-up plan
    st.subheader("📈 Full Rollout Plan (If Test Succeeds)")
    
    # Based on existing threshold sweep
    xgb_optimal = 0.20
    target_contacts_monthly = 300  # Scale linearly
    
    months_to_scale = 3
    monthly_contacts = []
    cumulative_contacts = 0
    
    for month in range(1, months_to_scale + 1):
        monthly_contacts.append(target_contacts_monthly)
        cumulative_contacts += target_contacts_monthly
    
    scale_df = pd.DataFrame({
        'Month': range(1, months_to_scale + 1),
        'Monthly Contacts': monthly_contacts,
        'Cumulative': np.cumsum(monthly_contacts),
        'Threshold': xgb_optimal,
        'Expected Saved': [int(c * (baseline_retention + expected_lift) - c * baseline_retention) for c in monthly_contacts],
    })
    
    st.dataframe(scale_df, use_container_width=True, hide_index=True)
    
    st.info(f"""
    **Expected Full-Year Impact** (if test succeeds and we scale)
    
    - Customers contacted: ~{cumulative_contacts * 4} (monthly scale)
    - Additional customers retained: ~{int(cumulative_contacts * 4 * expected_lift)}
    - Additional revenue: ~${int(cumulative_contacts * 4 * expected_lift * avg_clv):,.0f}
    - Intervention cost: ~${int(cumulative_contacts * 4 * contact_cost):,.0f}
    - **Expected Annual ROI: ~${int(cumulative_contacts * 4 * expected_lift * avg_clv - cumulative_contacts * 4 * contact_cost):,.0f}**
    """)

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please run `04_business_impact.ipynb` first to generate the required reports.")
