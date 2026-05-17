# Frontend Dashboard (HTML/CSS/JS)

This frontend mirrors the Streamlit pages with real project data:
- Overview
- Decision Tool
- Segments
- Explainability

## Data Sources (No Mock Data)
The UI reads directly from:
- `reports/business_recommendations.csv`
- `reports/lr_threshold_sweep.csv`
- `reports/rf_threshold_sweep.csv`
- `reports/xgb_threshold_sweep.csv`
- `reports/lr_customer_scores.csv`
- `reports/rf_customer_scores.csv`
- `reports/xgb_customer_scores.csv`
- `reports/lr_feature_importance.csv`
- `reports/rf_feature_importance.csv`
- `reports/xgb_feature_importance.csv`
- `data/processed/churn_features.csv`
- `models/*_shap_summary.png` and `models/*_shap_force.png`

## Run
From project root:

```powershell
python -m http.server 8000
```

Then open:

`http://localhost:8000/frontend/index.html`

## Refresh Data Exports
If models/data change, regenerate frontend artifacts:

```powershell
python tmp/generate_frontend_data.py
```
