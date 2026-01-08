"""
ROI Analysis for Churn Prediction

This module contains functions for calculating and visualizing
ROI across different thresholds and models.
"""

import matplotlib.pyplot as plt
import pandas as pd


def compare_models(lr_sweep: pd.DataFrame, rf_sweep: pd.DataFrame) -> dict:
    """
    Compare two models and select the best one based on max ROI.
    
    Args:
        lr_sweep: Logistic Regression threshold sweep results
        rf_sweep: Random Forest threshold sweep results
        
    Returns:
        Dictionary with best model info
    """
    lr_optimal_idx = lr_sweep['Net_ROI'].idxmax()
    rf_optimal_idx = rf_sweep['Net_ROI'].idxmax()
    
    lr_optimal = lr_sweep.loc[lr_optimal_idx]
    rf_optimal = rf_sweep.loc[rf_optimal_idx]
    
    if lr_optimal['Net_ROI'] > rf_optimal['Net_ROI']:
        return {
            'model': 'Logistic Regression',
            'threshold': lr_optimal['Threshold'],
            'roi': lr_optimal['Net_ROI'],
            'sweep': lr_sweep,
            'optimal': lr_optimal
        }
    else:
        return {
            'model': 'Random Forest',
            'threshold': rf_optimal['Threshold'],
            'roi': rf_optimal['Net_ROI'],
            'sweep': rf_sweep,
            'optimal': rf_optimal
        }


def plot_roi_comparison(lr_sweep: pd.DataFrame, rf_sweep: pd.DataFrame) -> None:
    """
    Plot ROI comparison across thresholds for both models.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    lr_optimal = lr_sweep.loc[lr_sweep['Net_ROI'].idxmax()]
    rf_optimal = rf_sweep.loc[rf_sweep['Net_ROI'].idxmax()]
    
    # ROI Comparison
    axes[0].plot(lr_sweep['Threshold'], lr_sweep['Net_ROI'], 
                 'b-o', label='Logistic Regression', linewidth=2, markersize=6)
    axes[0].plot(rf_sweep['Threshold'], rf_sweep['Net_ROI'], 
                 'g-s', label='Random Forest', linewidth=2, markersize=6)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Break-even')
    axes[0].axvline(x=lr_optimal['Threshold'], color='b', linestyle=':', alpha=0.7)
    axes[0].axvline(x=rf_optimal['Threshold'], color='g', linestyle=':', alpha=0.7)
    axes[0].set_xlabel('Threshold', fontsize=11)
    axes[0].set_ylabel('Net ROI ($)', fontsize=11)
    axes[0].set_title('Net ROI vs Threshold', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Recall vs Precision trade-off
    axes[1].plot(lr_sweep['Threshold'], lr_sweep['Recall'], 
                 'b-o', label='LR Recall', linewidth=2)
    axes[1].plot(lr_sweep['Threshold'], lr_sweep['Precision'], 
                 'b--s', label='LR Precision', alpha=0.7)
    axes[1].plot(rf_sweep['Threshold'], rf_sweep['Recall'], 
                 'g-o', label='RF Recall', linewidth=2)
    axes[1].plot(rf_sweep['Threshold'], rf_sweep['Precision'], 
                 'g--s', label='RF Precision', alpha=0.7)
    axes[1].set_xlabel('Threshold', fontsize=11)
    axes[1].set_ylabel('Score', fontsize=11)
    axes[1].set_title('Recall vs Precision Trade-off', fontsize=14, fontweight='bold')
    axes[1].legend(loc='center left')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def print_model_selection(best: dict, lr_optimal: pd.Series, rf_optimal: pd.Series) -> None:
    """Print model selection decision."""
    print("\n🏆 MODEL SELECTION DECISION")
    print("=" * 60)
    
    print(f"\n📊 Logistic Regression:")
    print(f"   Optimal Threshold: {lr_optimal['Threshold']:.2f}")
    print(f"   Net ROI: ${lr_optimal['Net_ROI']:,.2f}")
    print(f"   Recall: {lr_optimal['Recall']:.2%}")
    print(f"   Precision: {lr_optimal['Precision']:.2%}")
    print(f"   Customers to Contact: {lr_optimal['Customers_Contacted']:.0f}")
    
    print(f"\n📊 Random Forest:")
    print(f"   Optimal Threshold: {rf_optimal['Threshold']:.2f}")
    print(f"   Net ROI: ${rf_optimal['Net_ROI']:,.2f}")
    print(f"   Recall: {rf_optimal['Recall']:.2%}")
    print(f"   Precision: {rf_optimal['Precision']:.2%}")
    print(f"   Customers to Contact: {rf_optimal['Customers_Contacted']:.0f}")
    
    print(f"\n✅ RECOMMENDED MODEL: {best['model']}")
    print(f"✅ OPTIMAL THRESHOLD: {best['threshold']:.2f}")
    print(f"✅ EXPECTED NET ROI: ${best['roi']:,.2f}")
