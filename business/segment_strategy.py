"""
Segment Strategy for Churn Prediction

This module contains functions for segment-level ROI analysis,
including high-value vs low-value customer strategies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .threshold_optimization import threshold_sweep


def create_value_segments(
    df: pd.DataFrame,
    charge_column: str = 'MonthlyCharges'
) -> pd.DataFrame:
    """
    Create value segments based on monthly charges.
    
    Args:
        df: DataFrame with customer data
        charge_column: Column name for charges
        
    Returns:
        DataFrame with 'value_segment' column added
    """
    df = df.copy()
    charge_median = df[charge_column].median()
    
    df['value_segment'] = np.where(
        df[charge_column] >= charge_median,
        'High-Value',
        'Low-Value'
    )
    
    return df, charge_median


def calculate_segment_clv(
    df: pd.DataFrame,
    segment: str,
    avg_remaining_months: int = 24
) -> float:
    """
    Calculate CLV for a specific segment.
    
    Args:
        df: DataFrame with 'value_segment' and 'MonthlyCharges' columns
        segment: Segment name ('High-Value' or 'Low-Value')
        avg_remaining_months: Assumed customer lifetime
        
    Returns:
        Segment-specific CLV
    """
    segment_df = df[df['value_segment'] == segment]
    return segment_df['MonthlyCharges'].mean() * avg_remaining_months


def segment_threshold_analysis(
    df_segment: pd.DataFrame,
    proba_col: str,
    segment_clv: float,
    base_cost_matrix: dict,
    thresholds: np.ndarray = None
) -> pd.DataFrame:
    """
    Analyze threshold performance for a specific segment.
    
    Args:
        df_segment: DataFrame for segment
        proba_col: Column with probabilities
        segment_clv: Segment-specific CLV
        base_cost_matrix: Base cost matrix to copy
        thresholds: Array of thresholds to test
        
    Returns:
        DataFrame with segment metrics
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)
    
    # Create segment-specific cost matrix
    segment_cost_matrix = base_cost_matrix.copy()
    segment_cost_matrix['CLV'] = segment_clv
    segment_cost_matrix['TP'] = (segment_clv * base_cost_matrix['RETENTION_SUCCESS_RATE']) - base_cost_matrix['RETENTION_COST']
    segment_cost_matrix['FN'] = -(segment_clv * base_cost_matrix['RETENTION_SUCCESS_RATE'])
    
    return threshold_sweep(
        df_segment['y_true'].values,
        df_segment[proba_col].values,
        segment_cost_matrix
    )


def compare_strategies(
    high_value_sweep: pd.DataFrame,
    low_value_sweep: pd.DataFrame,
    best_single_threshold: float
) -> dict:
    """
    Compare single threshold vs segment-specific strategies.
    
    Args:
        high_value_sweep: High-value segment sweep results
        low_value_sweep: Low-value segment sweep results
        best_single_threshold: Best single threshold for all customers
        
    Returns:
        Dictionary with comparison results
    """
    # Segment-specific optimal thresholds
    high_value_optimal = high_value_sweep.loc[high_value_sweep['Net_ROI'].idxmax()]
    low_value_optimal = low_value_sweep.loc[low_value_sweep['Net_ROI'].idxmax()]
    
    # Single threshold ROI for each segment
    single_high = high_value_sweep[
        high_value_sweep['Threshold'] == best_single_threshold
    ]['Net_ROI'].values[0]
    single_low = low_value_sweep[
        low_value_sweep['Threshold'] == best_single_threshold
    ]['Net_ROI'].values[0]
    
    single_total = single_high + single_low
    segment_total = high_value_optimal['Net_ROI'] + low_value_optimal['Net_ROI']
    improvement = segment_total - single_total
    
    return {
        'single_threshold': best_single_threshold,
        'single_high_roi': single_high,
        'single_low_roi': single_low,
        'single_total_roi': single_total,
        'high_value_optimal': high_value_optimal,
        'low_value_optimal': low_value_optimal,
        'segment_total_roi': segment_total,
        'improvement': improvement
    }


def plot_segment_roi(
    high_value_sweep: pd.DataFrame,
    low_value_sweep: pd.DataFrame
) -> None:
    """Plot segment-level ROI comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    high_optimal = high_value_sweep.loc[high_value_sweep['Net_ROI'].idxmax()]
    low_optimal = low_value_sweep.loc[low_value_sweep['Net_ROI'].idxmax()]
    
    ax.plot(high_value_sweep['Threshold'], high_value_sweep['Net_ROI'], 
            'b-o', label='High-Value Customers', linewidth=2, markersize=6)
    ax.plot(low_value_sweep['Threshold'], low_value_sweep['Net_ROI'], 
            'g-s', label='Low-Value Customers', linewidth=2, markersize=6)
    
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Break-even')
    ax.axvline(x=high_optimal['Threshold'], color='b', linestyle=':', alpha=0.7)
    ax.axvline(x=low_optimal['Threshold'], color='g', linestyle=':', alpha=0.7)
    
    ax.set_xlabel('Threshold', fontsize=11)
    ax.set_ylabel('Net ROI ($)', fontsize=11)
    ax.set_title('Segment-Level ROI vs Threshold', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def print_segment_analysis(comparison: dict, charge_median: float) -> None:
    """Print segment analysis results."""
    print("\n💡 KEY INSIGHT: Single vs Segment-Specific Thresholds")
    print("=" * 60)
    
    print(f"\n📊 Single Threshold ({comparison['single_threshold']:.2f}) for all:")
    print(f"   High-Value ROI: ${comparison['single_high_roi']:,.2f}")
    print(f"   Low-Value ROI:  ${comparison['single_low_roi']:,.2f}")
    print(f"   Total ROI:      ${comparison['single_total_roi']:,.2f}")
    
    hv = comparison['high_value_optimal']
    lv = comparison['low_value_optimal']
    
    print(f"\n📊 Segment-Specific Thresholds:")
    print(f"   High-Value ({hv['Threshold']:.2f}): ${hv['Net_ROI']:,.2f}")
    print(f"   Low-Value ({lv['Threshold']:.2f}):  ${lv['Net_ROI']:,.2f}")
    print(f"   Total ROI:      ${comparison['segment_total_roi']:,.2f}")
    
    print(f"\n🎯 IMPROVEMENT FROM SEGMENTATION: ${comparison['improvement']:,.2f}")
