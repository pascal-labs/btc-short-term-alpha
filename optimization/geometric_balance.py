"""
Geometric Balance Objective: Preventing Degenerate Optimization Solutions

The core problem in trading strategy optimization:
How do you balance FREQUENCY (taking many trades) with QUALITY (each trade being good)?

Naive objectives produce degenerate solutions:
- Maximize total return -> trade everything at any quality -> low win rate, high drawdown
- Maximize per-trade return -> trade rarely at extreme quality -> too few trades, high variance
- Maximize Sharpe ratio -> dominated by trade frequency in denominator

The geometric balance objective solves this by requiring BOTH frequency AND quality:

    G_window = total_log_return / n_windows    (rewards frequency: more trades = more G)
    G_trade  = total_log_return / n_trades     (rewards quality: better trades = more G)

    G = sqrt(G_window * G_trade)               (geometric mean balances both)

The geometric mean has a key property: if either component is zero, G is zero.
You can't game it by maximizing one at the expense of the other.

Robust scoring adds fold stability:
    robust_score = min(fold_Gs) - 0.5 * std(fold_Gs)

This penalizes variance across CV folds and focuses on worst-case performance,
preventing the optimizer from finding parameters that work great on some folds
but fail catastrophically on others.

Parameter space: 30 total (15 YES + 15 NO)
Startup trials: 30 x 50 = 1,500 (ensures adequate exploration before TPE kicks in)
CV: 5-fold expanding windows, 20% holdout
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class OptimizationResult:
    """Result from a single backtest evaluation."""
    n_trades: int
    win_rate: float
    G_window: float      # Total log return / n_windows
    G_trade: float       # Total log return / n_trades
    avg_size: float
    yes_trades: int
    no_trades: int


def compute_geometric_balance(G_window: float, G_trade: float) -> float:
    """
    Compute geometric mean of frequency and quality metrics.

    G = sqrt(G_window * G_trade)

    Special cases:
    - Both positive: standard geometric mean
    - Both negative: -sqrt(|G_window| * |G_trade|) (preserve sign)
    - Mixed signs: min(G_window, G_trade) (conservative: use the worse one)

    Args:
        G_window: Per-window growth rate (rewards trade frequency)
        G_trade: Per-trade growth rate (rewards trade quality)

    Returns:
        Geometric balance score
    """
    if G_window > 0 and G_trade > 0:
        return np.sqrt(G_window * G_trade)
    elif G_window < 0 and G_trade < 0:
        return -np.sqrt(abs(G_window) * abs(G_trade))
    else:
        return min(G_window, G_trade)


def compute_robust_score(fold_scores: List[float]) -> float:
    """
    Compute robust objective from cross-validation fold scores.

    robust_score = min(fold_scores) - 0.5 * std(fold_scores)

    This objective:
    - Focuses on worst-case fold (min) -> prevents catastrophic failure
    - Penalizes variance (std) -> prefers stable parameters
    - The 0.5 coefficient balances robustness vs. average performance

    A parameter set with [0.05, 0.04, 0.05, 0.04, 0.05] (mean=0.046)
    scores HIGHER than [0.10, 0.08, 0.02, -0.01, 0.07] (mean=0.052)
    because the first set is stable while the second has a failing fold.

    Args:
        fold_scores: Geometric balance scores from each CV fold

    Returns:
        Robust score (higher is better)
    """
    return min(fold_scores) - 0.5 * np.std(fold_scores)


def create_cv_folds(n_windows: int, n_folds: int = 5) -> List[Dict]:
    """
    Create expanding-window cross-validation folds.

    Fold structure (5-fold example with 600 windows):
        Fold 1: validate on windows [100:200]
        Fold 2: validate on windows [200:300]
        Fold 3: validate on windows [300:400]
        Fold 4: validate on windows [400:500]
        Fold 5: validate on windows [500:600]

    Each fold's training set is all windows before the validation set.
    This prevents look-ahead bias: we never train on future data.

    The holdout set (last 20% of all windows) is reserved for final
    evaluation and never seen during optimization.

    Args:
        n_windows: Total number of market windows in tuning set
        n_folds: Number of CV folds (default 5)

    Returns:
        List of fold dicts with 'val_start' and 'val_end' indices
    """
    fold_size = n_windows // (n_folds + 1)
    folds = []

    for i in range(n_folds):
        val_start = (i + 1) * fold_size
        val_end = val_start + fold_size
        folds.append({
            'val_start': val_start,
            'val_end': min(val_end, n_windows),
        })

    return folds


def run_optimization_trial(backtest_fn,
                           params,
                           tuning_windows: List,
                           n_folds: int = 5) -> Dict:
    """
    Run a single Optuna trial with geometric balance objective.

    Steps:
    1. Create CV folds from tuning windows
    2. Run backtest on each fold
    3. Compute geometric balance per fold
    4. Compute robust score across folds

    Args:
        backtest_fn: Function(params, window_list) -> OptimizationResult
        params: Strategy parameters to evaluate
        tuning_windows: List of market window IDs (chronological)
        n_folds: Number of CV folds

    Returns:
        Dict with robust_score, fold details, and summary stats
    """
    folds = create_cv_folds(len(tuning_windows), n_folds)

    fold_Gs = []
    fold_details = []

    for fold in folds:
        val_windows = tuning_windows[fold['val_start']:fold['val_end']]

        if len(val_windows) < 20:
            fold_Gs.append(-0.001)
            continue

        result = backtest_fn(params, val_windows)

        geo_balance = compute_geometric_balance(result.G_window, result.G_trade)
        fold_Gs.append(geo_balance)
        fold_details.append({
            'n_trades': result.n_trades,
            'win_rate': result.win_rate,
            'G_window': result.G_window,
            'G_trade': result.G_trade,
            'geometric': geo_balance,
        })

    # Pad if some folds were too small
    while len(fold_Gs) < n_folds:
        fold_Gs.append(-0.001)

    robust = compute_robust_score(fold_Gs)

    return {
        'robust_score': robust,
        'fold_scores': fold_Gs,
        'fold_details': fold_details,
        'mean_geometric': np.mean(fold_Gs),
        'std_geometric': np.std(fold_Gs),
        'min_geometric': min(fold_Gs),
    }
