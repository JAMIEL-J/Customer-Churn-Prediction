"""
Benchmarking Page - Industry Context

Shows how this company's churn compares to industry benchmarks.
Answers: "Is 26.5% churn good or bad?"
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import os

st.set_page_config(page_title="Industry Benchmarking", page_icon="📊", layout="wide")

st.title("🌍 Industry Benchmarking")
st.markdown("*How does this company compare to the industry?*")

# Get project root - works on both local and Streamlit Cloud
def get_project_root():
    """Get the project root directory."""
    try:
        # Try from script location
        current_file = Path(__file__).resolve()
        root = current_file.parent.parent.parent
        if (root / 'data').exists():
            return root
    except:
        pass
    
    # Try from current working directory
    cwd = Path(os.getcwd())
    if (cwd / 'data').exists():
        return cwd
    
    # Try parent directory
    if (cwd.parent / 'data').exists():
        return cwd.parent
    
    return cwd

# Load data
@st.cache_data
def load_data():
    root = get_project_root()
    features = pd.read_csv(root / 'data' / 'processed' / 'churn_features.csv')
    return features

try:
    features = load_data()
    
    company_churn_rate = (features['Churn'] == 'Yes').mean()
    total_customers = len(features)
    total_revenue_at_risk = total_customers * company_churn_rate * features['MonthlyCharges'].mean()
    
    st.markdown("---")
    
    # Industry Benchmarks (researched from industry reports)
    benchmarks = {
        'Telecom (Broadband)': {'churn_rate': 0.22, 'annual': True},
        'Telecom (Average)': {'churn_rate': 0.25, 'annual': True},
        'SaaS (Monthly)': {'churn_rate': 0.05, 'monthly': True},
        'SaaS (Annual)': {'churn_rate': 0.10, 'annual': True},
        'Streaming (Netflix-like)': {'churn_rate': 0.30, 'annual': True},
        'Mobile Carriers': {'churn_rate': 0.20, 'annual': True},
    }
    
    # This Company
    company_name = "Your Company"
    your_rate_annual = company_churn_rate
    
    st.subheader("📊 Churn Rate Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Your Company Churn Rate",
            f"{your_rate_annual:.1%}",
            help="Annual churn rate"
        )
    
    with col2:
        telecom_avg = benchmarks['Telecom (Average)']['churn_rate']
        diff = (your_rate_annual - telecom_avg) * 100
        st.metric(
            "Telecom Industry Average",
            f"{telecom_avg:.1%}",
            delta=f"+{diff:.1f}% (above average)" if diff > 0 else f"{diff:.1f}% (below average)"
        )
    
    with col3:
        st.metric(
            "Highest Risk (Streaming)",
            f"{benchmarks['Streaming (Netflix-like)']['churn_rate']:.1%}",
            help="Streaming services avg"
        )
    
    st.markdown("---")
    
    # Visualization: Bar chart
    st.subheader("Churn Comparison Across Industries")
    
    # Build benchmark data with colors
    data = []
    colors = []
    
    # Add company data first
    data.append({'Industry': 'Your Company', 'Churn Rate': your_rate_annual * 100})
    colors.append('darkred')
    
    # Add industry benchmarks
    for industry, metrics in benchmarks.items():
        rate = metrics['churn_rate']
        data.append({'Industry': industry, 'Churn Rate': rate * 100})
        colors.append('lightblue')
    
    df_benchmark = pd.DataFrame(data)
    
    # Create bar chart using plotly graph_objects
    fig = go.Figure()
    
    for i, (idx, row) in enumerate(df_benchmark.iterrows()):
        fig.add_trace(go.Bar(
            x=[row['Industry']],
            y=[row['Churn Rate']],
            marker=dict(color=colors[i]),
            name=row['Industry'],
            showlegend=False,
            text=f"{row['Churn Rate']:.1f}%",
            textposition='outside'
        ))
    
    fig.update_layout(
        title="Annual Churn Rate by Industry",
        xaxis_title="Industry / Company",
        yaxis_title="Churn Rate (%)",
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Analysis
    st.subheader("📈 Analysis")
    
    telecom_avg = benchmarks['Telecom (Average)']['churn_rate']
    saas_avg = benchmarks['SaaS (Annual)']['churn_rate']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Position in Market**
        
        Your company's **{your_rate_annual:.1%} churn rate** is:
        
        - ⚠️ **{((your_rate_annual - telecom_avg) / telecom_avg * 100):.0f}% HIGHER** than telecom average ({telecom_avg:.1%})
        - ⚠️ **{(your_rate_annual / saas_avg * 100):.0f}% HIGHER** than SaaS benchmark ({saas_avg:.1%})
        - ✅ **{((1 - your_rate_annual / benchmarks['Streaming (Netflix-like)']['churn_rate']) * 100):.0f}% BETTER** than streaming average
        
        **Verdict**: Higher-than-average churn, especially vs SaaS competitors
        """)
    
    with col2:
        # Financial opportunity
        telecom_revenue_if_match = total_customers * telecom_avg * features['MonthlyCharges'].mean()
        opportunity = total_revenue_at_risk - telecom_revenue_if_match
        
        st.success(f"""
        **Financial Opportunity**
        
        If you matched telecom average ({telecom_avg:.1%}):
        
        - Current revenue at risk: **${total_revenue_at_risk:,.0f}**
        - Revenue at risk if at telecom avg: **${telecom_revenue_if_match:,.0f}**
        - **Potential annual savings: ${opportunity:,.0f}**
        
        *(Based on {total_customers:,} customers at avg ${features['MonthlyCharges'].mean():.2f}/mo)*
        """)
    
    st.markdown("---")
    
    # Segment Analysis
    st.subheader("👥 Segment Analysis")
    
    if 'InternetService' in features.columns:
        segment_churn = features.groupby('InternetService')['Churn'].apply(
            lambda x: (x == 'Yes').sum() / len(x)
        ).sort_values(ascending=False)
        
        fig = go.Figure(data=[
            go.Bar(
                x=segment_churn.index,
                y=segment_churn.values * 100,
                marker=dict(color='orange'),
                text=[f"{v:.1f}%" for v in segment_churn.values * 100],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Churn Rate by Service Type",
            xaxis_title="Service Type",
            yaxis_title="Churn Rate (%)",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Recommendations
    st.subheader("🎯 Strategic Recommendations")
    
    st.warning(f"""
    **Key Finding**: {your_rate_annual:.1%} churn vs {telecom_avg:.1%} industry average
    
    **Actions to Consider**:
    
    1. **Immediate** (Next 30 days)
       - Deploy the churn prediction model to retain high-value customers
       - Expected impact: Reduce churn by 3-5% → Save ~${opportunity * 0.05:,.0f} annually
    
    2. **Short-term** (Next 90 days)
       - Analyze root causes (contract type, service type, pricing)
       - Implement targeted retention for fiber optic + month-to-month segments
    
    3. **Medium-term** (Next 6 months)
       - Redesign onboarding for new customers (high early-churn risk)
       - Evaluate pricing competitiveness vs. industry
    
    4. **Long-term** (Next 12 months)
       - Implement proactive customer success program
       - Target: Reach industry average ({telecom_avg:.1%}) → Save ${opportunity:,.0f} annually
    """)

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please run `04_business_impact.ipynb` first to generate the required reports.")
