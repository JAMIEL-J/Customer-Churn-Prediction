"""
Business Impact Analysis Module

This package provides tools for converting churn prediction models
into actionable business decisions through cost-benefit analysis.

Modules:
    - cost_matrix: Cost assumptions and CLV calculations
    - threshold_optimization: Threshold sweep and business metrics
    - roi_analysis: ROI calculations and model comparison
    - segment_strategy: Value-based customer segmentation
"""
from .cost_matrix import calculate_clv, get_cost_matrix, print_cost_breakdown
from .threshold_optimization import (
    calculate_business_metrics,
    threshold_sweep,
    find_optimal_threshold,
    print_decision_table
)
from .roi_analysis import compare_models, plot_roi_comparison, print_model_selection
from .segment_strategy import (
    create_value_segments,
    calculate_segment_clv,
    segment_threshold_analysis,
    compare_strategies,
    plot_segment_roi,
    print_segment_analysis
)

__all__ = [
    # cost_matrix
    'calculate_clv',
    'get_cost_matrix',
    'print_cost_breakdown',
    # threshold_optimization
    'calculate_business_metrics',
    'threshold_sweep',
    'find_optimal_threshold',
    'print_decision_table',
    # roi_analysis
    'compare_models',
    'plot_roi_comparison',
    'print_model_selection',
    # segment_strategy
    'create_value_segments',
    'calculate_segment_clv',
    'segment_threshold_analysis',
    'compare_strategies',
    'plot_segment_roi',
    'print_segment_analysis',
]
