import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

print("Loading data...")
df = pd.read_csv('d:/Customer-Churn-Prediction/data/processed/churn_features.csv')

with open('d:/Customer-Churn-Prediction/models/feature_cols.txt', 'r') as f:
    feature_cols = f.read().strip().split('\n')

df_model = df[['Churn'] + feature_cols].copy()
df_model['Churn'] = df_model['Churn'].map({'No': 0, 'Yes': 1})

X_train, X_test, y_train, y_test = train_test_split(df_model.drop(columns=['Churn']), df_model['Churn'], test_size=0.2, random_state=42, stratify=df_model['Churn'])

scaler = joblib.load('d:/Customer-Churn-Prediction/models/scaler.pkl')
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)

output_dir = 'd:/Customer-Churn-Prediction/models'

def get_shap_components(explainer, shap_vals, is_tree=False):
    exp_val = explainer.expected_value
    shaps = shap_vals
    if isinstance(shaps, list):
        shaps = shaps[1]
        exp_val = exp_val[1] if isinstance(exp_val, (list, np.ndarray)) else exp_val
    elif hasattr(shaps, 'shape') and len(shaps.shape) == 3:
        shaps = shaps[:, :, 1]
        exp_val = exp_val[1] if isinstance(exp_val, (list, np.ndarray)) else exp_val
    
    if isinstance(exp_val, (list, np.ndarray)):
        # If it's still an array, just take the first element (or class 1 if length 2)
        if len(exp_val) > 1:
            exp_val = exp_val[1]
        else:
            exp_val = exp_val[0]
            
    return exp_val, shaps

# LR
try:
    print("Generating SHAP for Logistic Regression...")
    lr_model = joblib.load(os.path.join(output_dir, 'logistic_regression.pkl'))
    lr_explainer = shap.LinearExplainer(lr_model, X_train_scaled)
    lr_shap_values_raw = lr_explainer.shap_values(X_test_scaled)
    lr_exp_val, lr_shap_values = get_shap_components(lr_explainer, lr_shap_values_raw)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(lr_shap_values, X_test_scaled, show=False)
    plt.title('SHAP Summary Plot (Logistic Regression)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lr_shap_summary.png'), bbox_inches='tight')
    plt.close()

    shap.force_plot(lr_exp_val, lr_shap_values[0], X_test_scaled.iloc[0], matplotlib=True, show=False)
    plt.savefig(os.path.join(output_dir, 'lr_shap_force.png'), bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"Error in LR SHAP: {e}")

# RF
try:
    print("Generating SHAP for Random Forest...")
    rf_model = joblib.load(os.path.join(output_dir, 'random_forest.pkl'))
    rf_explainer = shap.TreeExplainer(rf_model)
    rf_shap_values_raw = rf_explainer.shap_values(X_test)
    rf_exp_val, rf_shap_values = get_shap_components(rf_explainer, rf_shap_values_raw, True)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(rf_shap_values, X_test, show=False)
    plt.title('SHAP Summary Plot (Random Forest)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rf_shap_summary.png'), bbox_inches='tight')
    plt.close()

    shap.force_plot(rf_exp_val, rf_shap_values[0], X_test.iloc[0], matplotlib=True, show=False)
    plt.savefig(os.path.join(output_dir, 'rf_shap_force.png'), bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"Error in RF SHAP: {e}")

# XGB
try:
    print("Generating SHAP for XGBoost...")
    xgb_model = joblib.load(os.path.join(output_dir, 'xgboost.pkl'))
    xgb_explainer = shap.TreeExplainer(xgb_model)
    xgb_shap_values_raw = xgb_explainer.shap_values(X_test)
    xgb_exp_val, xgb_shap_values = get_shap_components(xgb_explainer, xgb_shap_values_raw, True)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(xgb_shap_values, X_test, show=False)
    plt.title('SHAP Summary Plot (XGBoost)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'xgb_shap_summary.png'), bbox_inches='tight')
    plt.close()

    shap.force_plot(xgb_exp_val, xgb_shap_values[0], X_test.iloc[0], matplotlib=True, show=False)
    plt.savefig(os.path.join(output_dir, 'xgb_shap_force.png'), bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"Error in XGB SHAP: {e}")

print("All SHAP plots generated and saved!")
