"""
Threshold Optimization for Churn Prediction

This module contains functions for sweeping thresholds and
calculating business metrics at each threshold level.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def calculate_business_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    cost_matrix: dict
) -> dict:
    """
    Calculate business metrics for a given threshold.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        threshold: Classification threshold
        cost_matrix: Dictionary with cost values
        
    Returns:
        Dictionary with threshold, TP, FP, FN, TN, costs, revenue, ROI
    """
    # Make predictions at threshold
    y_pred = (y_proba >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Customers contacted = predicted positive (TP + FP)
    customers_contacted = tp + fp
    
    # Total cost of retention campaign
    total_retention_cost = customers_contacted * cost_matrix['RETENTION_COST']
    
    # Revenue saved from True Positives
    revenue_saved = tp * cost_matrix['RETENTION_SUCCESS_RATE'] * cost_matrix['CLV']
    
    # Opportunity cost from False Negatives
    opportunity_cost = fn * cost_matrix['RETENTION_SUCCESS_RATE'] * cost_matrix['CLV']
    
    # Net ROI = Revenue saved - Total cost
    net_roi = revenue_saved - total_retention_cost
    
    # Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        'Threshold': threshold,
        'Customers_Contacted': customers_contacted,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
        'Precision': precision,
        'Recall': recall,
        'Retention_Cost': total_retention_cost,
        'Revenue_Saved': revenue_saved,
        'Opportunity_Cost': opportunity_cost,
        'Net_ROI': net_roi
    }


def threshold_sweep(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_matrix: dict,
    start: float = 0.1,
    end: float = 0.95,
    step: float = 0.05
) -> pd.DataFrame:
    """
    Sweep through thresholds and calculate metrics at each level.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        cost_matrix: Dictionary with cost values
        start: Starting threshold
        end: Ending threshold
        step: Step size
        
    Returns:
        DataFrame with metrics at each threshold
    """
    thresholds = np.arange(start, end, step)
    
    results = []
    for t in thresholds:
        metrics = calculate_business_metrics(y_true, y_proba, t, cost_matrix)
        results.append(metrics)
    
    return pd.DataFrame(results)


def find_optimal_threshold(sweep_df: pd.DataFrame, metric: str = 'Net_ROI') -> pd.Series:
    """
    Find the threshold that maximizes the given metric.
    
    Args:
        sweep_df: DataFrame from threshold_sweep
        metric: Metric to optimize (default: 'Net_ROI')
        
    Returns:
        Series with optimal threshold metrics
    """
    optimal_idx = sweep_df[metric].idxmax()
    return sweep_df.loc[optimal_idx]


def print_decision_table(sweep_df: pd.DataFrame, model_name: str) -> None:
    """Print formatted decision table."""
    print(f"📊 {model_name} - DECISION TABLE")
    print("=" * 100)
    
    display_cols = ['Threshold', 'Customers_Contacted', 'TP', 'FP', 'FN', 
                    'Precision', 'Recall', 'Retention_Cost', 'Revenue_Saved', 'Net_ROI']
    
    display_df = sweep_df[display_cols].copy()
    display_df['Threshold'] = display_df['Threshold'].apply(lambda x: f"{x:.2f}")
    display_df['Precision'] = display_df['Precision'].apply(lambda x: f"{x:.2%}")
    display_df['Recall'] = display_df['Recall'].apply(lambda x: f"{x:.2%}")
    
    print(display_df.to_string(index=False))
