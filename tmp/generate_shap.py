import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

print("Loading data...")
df = pd.read_csv('d:/Customer-Churn-Prediction/data/processed/churn_features.csv')

with open('d:/Customer-Churn-Prediction/models/feature_cols.txt', 'r') as f:
    feature_cols = f.read().strip().split('\n')

df_model = df[['Churn'] + feature_cols].copy()
df_model['Churn'] = df_model['Churn'].map({'No': 0, 'Yes': 1})

# Train Test Split matching 03_modeling.ipynb
X_train, X_test, y_train, y_test = train_test_split(df_model.drop(columns=['Churn']), df_model['Churn'], test_size=0.2, random_state=42, stratify=df_model['Churn'])

print("Scaling data...")
scaler = joblib.load('d:/Customer-Churn-Prediction/models/scaler.pkl')

# Scale numeric features
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
X_test_scaled = X_test.copy()
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

print("Loading model and calculating SHAP values...")
xgb_model = joblib.load('d:/Customer-Churn-Prediction/models/xgboost.pkl')

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_scaled)

print("Saving summary plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_scaled, show=False)
plt.title('SHAP Summary Plot', fontsize=16)
plt.tight_layout()
plt.savefig('d:/Customer-Churn-Prediction/models/shap_summary.png', bbox_inches='tight')
plt.close()

print("Saving force plot...")
f = shap.force_plot(explainer.expected_value, shap_values[0], X_test_scaled.iloc[0], matplotlib=True, show=False)
plt.savefig('d:/Customer-Churn-Prediction/models/shap_force.png', bbox_inches='tight')
plt.close()

print("Completed generating SHAP plots.")
