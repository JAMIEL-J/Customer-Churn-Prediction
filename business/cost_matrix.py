"""
Cost Matrix Definitions for Churn Analysis

This module contains all cost-related assumptions and calculations
for the business impact analysis.
"""

import pandas as pd


def calculate_clv(df: pd.DataFrame, avg_remaining_months: int = 24) -> float:
    """
    Calculate Customer Lifetime Value based on average monthly charges.
    
    Args:
        df: DataFrame with 'MonthlyCharges' column
        avg_remaining_months: Assumed average customer lifetime in months
        
    Returns:
        Average CLV value
    """
    avg_monthly_charge = df['MonthlyCharges'].mean()
    return avg_monthly_charge * avg_remaining_months


def get_cost_matrix(
    clv: float,
    retention_cost: float = 50,
    retention_success_rate: float = 0.25
) -> dict:
    """
    Build the cost matrix for churn analysis.
    
    Args:
        clv: Customer Lifetime Value
        retention_cost: Cost per customer contacted with retention offer
        retention_success_rate: Probability of saving an at-risk customer
        
    Returns:
        Dictionary with cost values for TP, FP, FN, TN scenarios
    """
    # True Positive: Correctly identify churner
    # Benefit: CLV × success_rate, Cost: Retention cost
    tp_net_value = (clv * retention_success_rate) - retention_cost
    
    # False Positive: Flag non-churner as churner
    # Cost: Wasted retention cost
    fp_net_value = -retention_cost
    
    # False Negative: Miss a churner
    # Cost: Opportunity cost (could have saved)
    fn_net_value = -(clv * retention_success_rate)
    
    # True Negative: Correctly ignore non-churner
    # No action, no cost
    tn_net_value = 0
    
    return {
        'CLV': clv,
        'RETENTION_COST': retention_cost,
        'RETENTION_SUCCESS_RATE': retention_success_rate,
        'TP': tp_net_value,
        'FP': fp_net_value,
        'FN': fn_net_value,
        'TN': tn_net_value
    }


def print_cost_breakdown(cost_matrix: dict) -> None:
    """Print formatted cost breakdown."""
    print("\n💰 COST-BENEFIT BREAKDOWN")
    print("=" * 50)
    
    clv = cost_matrix['CLV']
    rc = cost_matrix['RETENTION_COST']
    rate = cost_matrix['RETENTION_SUCCESS_RATE']
    
    print(f"\n✅ True Positive (correctly flagged churner):")
    print(f"   Revenue saved: ${clv * rate:,.2f} (CLV × {rate:.0%} success)")
    print(f"   Minus cost:    ${rc:,.2f}")
    print(f"   Net value:     ${cost_matrix['TP']:,.2f}")
    
    print(f"\n❌ False Positive (wrongly flagged non-churner):")
    print(f"   Revenue saved: $0.00")
    print(f"   Minus cost:    ${rc:,.2f}")
    print(f"   Net value:     ${cost_matrix['FP']:,.2f}")
    
    print(f"\n🚨 False Negative (missed churner):")
    print(f"   Revenue saved: $0.00")
    print(f"   Opportunity cost: ${clv * rate:,.2f} (could have saved)")
    print(f"   Net value:     ${cost_matrix['FN']:,.2f}")
    
    print(f"\n✅ True Negative (correctly ignored):")
    print(f"   Net value: $0.00 (no action, no cost)")
