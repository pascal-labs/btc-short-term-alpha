"""
Walk-Forward Backtest Engine

Runs the lock-in strategy across a set of market windows and computes
growth metrics (G_window, G_trade) for the geometric balance objective.

Key design decisions:
- Single entry per market (no re-entry after exit)
- Hold to resolution (binary markets resolve automatically)
- Log returns for Kelly-consistent growth accounting
- Separate YES/NO evaluation with score-based tiebreaking

The backtest processes each market window independently:
1. Iterate through ticks chronologically
2. At each tick, evaluate both YES and NO entry conditions
3. Take the first entry that triggers (single-side constraint)
4. Compute log return based on market resolution
"""

# Data Pipeline Integration:
# In production, BTC spot prices are captured via pulsefeed
# (github.com/pascal-labs/pulsefeed) which aggregates 8 exchanges
# with USDT premium normalization. Binary market prices are fetched
# via polymarket-sdk (github.com/pascal-labs/polymarket-sdk) using
# WebSocket feeds for sub-second updates. This backtest engine
# accepts pre-formatted price arrays for reproducibility.

import numpy as np
from typing import Dict, List, Optional


# Position sizing — PARAM_PLACEHOLDER
BASE_KELLY = None     # Base Kelly fraction
SLIPPAGE = None       # Fill-or-kill slippage assumption


def run_backtest(cache: Dict,
                 outcomes: Dict,
                 yes_params,
                 no_params,
                 window_ids: List[str]) -> Dict:
    """
    Run walk-forward backtest across multiple market windows.

    Args:
        cache: Dict of precomputed features per market (from features.py)
        outcomes: Dict mapping market_id -> bool (True if YES won)
        yes_params: YES side entry parameters
        no_params: Side B entry parameters
        window_ids: List of market window IDs to evaluate

    Returns:
        Dict with n_trades, win_rate, G_window, G_trade, etc.
    """
    from strategy.lock_in import run_single_market

    results = []
    yes_trades = 0
    no_trades = 0

    for market_id in window_ids:
        if market_id not in cache or market_id not in outcomes:
            continue

        result = run_single_market(
            cache[market_id],
            outcomes[market_id],
            yes_params,
            no_params,
        )

        if result is not None:
            results.append(result)
            if result['side'] == 'yes':
                yes_trades += 1
            else:
                no_trades += 1

    n_trades = len(results)
    if n_trades == 0:
        return {
            'n_trades': 0,
            'win_rate': 0,
            'G_window': -0.001,
            'G_trade': -0.001,
            'avg_size': 0,
            'yes_trades': 0,
            'no_trades': 0,
        }

    wins = [r for r in results if r['won']]
    win_rate = len(wins) / n_trades

    total_log_return = sum(r['log_return'] for r in results)
    G_window = total_log_return / len(window_ids)
    G_trade = total_log_return / n_trades

    avg_size = np.mean([r['size_mult'] for r in results])

    return {
        'n_trades': n_trades,
        'win_rate': win_rate,
        'G_window': G_window,
        'G_trade': G_trade,
        'avg_size': avg_size,
        'yes_trades': yes_trades,
        'no_trades': no_trades,
    }
