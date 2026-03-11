"""
Customer Churn Prediction Dashboard
Main entry point for Streamlit app
"""

import streamlit as st

# Page config
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main landing page
st.title("💰 Customer Churn Prediction")
st.subheader("Business Impact Dashboard")

st.markdown("""
---

### Navigate to:

| Page | Purpose |
|------|---------|
| **📊 Overview** | Executive snapshot - is this worth caring about? |
| **🎯 Decision Tool** | Interactive threshold optimization (CORE) |
| **👥 Segments** | Who gets contacted and why? |
| **🔍 Explainability** | Why does the model flag churn? |

---

*Select a page from the sidebar to begin.*
""")

# Sidebar info
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    This dashboard helps business teams:
    - Optimize retention spending
    - Target the right customers
    - Maximize ROI
    
    **Models**: Logistic Regression, Random Forest, XGBoost
    
    **Data**: Pre-computed from `04_business_impact.ipynb`
    """)
