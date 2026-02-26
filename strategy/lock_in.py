"""
Lock-In Entry Logic: Late-Window Binary Market Strategy

Core thesis: In 15-minute BTC binary markets, price evolves stochastically.
Once BTC has moved far enough in one direction, the binary outcome is
near-certain — but the market contract still offers a discount because
resolution hasn't occurred yet.

Entry constraints:
1. Time window: Only enter during a specific late-window phase
2. Volatility gate: Reject entries when realized vol exceeds threshold
3. Price cap: Reject entries when contract price is too expensive
4. Score threshold: Unified score must exceed dynamic threshold
5. Single-side constraint: One entry per market (either YES or NO)

The optimizer selected a single dominant side — the market microstructure
signal was asymmetric. This is an important empirical result: the
strategy is not symmetric, and attempting to trade both sides would
degrade performance. Which side dominates is redacted.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# Entry parameters — all PARAM_PLACEHOLDER (optimized via Optuna)
@dataclass
class EntryParams:
    """Strategy parameters for one side (YES or NO)."""
    entry_start: int = 0        # PARAM_PLACEHOLDER — tick count for entry window start
    entry_end: int = 0          # PARAM_PLACEHOLDER — tick count for entry window end
    cap: float = 0.0            # PARAM_PLACEHOLDER — max entry price
    vol_max: float = 0.0        # PARAM_PLACEHOLDER — max realized volatility
    w_slope_short: float = 0.0  # PARAM_PLACEHOLDER
    w_slope_long: float = 0.0   # PARAM_PLACEHOLDER
    w_spread: float = 0.0       # PARAM_PLACEHOLDER
    w_compression: float = 0.0  # PARAM_PLACEHOLDER
    w_dist: float = 0.0         # PARAM_PLACEHOLDER
    w_R: float = 0.0            # PARAM_PLACEHOLDER
    w_time: float = 0.0         # PARAM_PLACEHOLDER
    thresh_start: float = 0.0   # PARAM_PLACEHOLDER
    thresh_end: float = 0.0     # PARAM_PLACEHOLDER
    score_scale: float = 0.0    # PARAM_PLACEHOLDER


# Fixed constants — PARAM_PLACEHOLDER
SLIPPAGE = None       # Fill-or-kill slippage assumption
BASE_KELLY = None     # Base Kelly fraction for position sizing
MIN_TICKS = None      # PARAM_PLACEHOLDER — minimum ticks before entry (cold start avoidance)


def evaluate_entry(prices: np.ndarray,
                   features: Optional[Dict],
                   vol: float,
                   R: float,
                   progress: float,
                   time_remaining: int,
                   tick: int,
                   params: EntryParams) -> Optional[Tuple[float, float, float]]:
    """
    Evaluate whether to enter a position at the current tick.

    Applies the full entry filter cascade:
    1. Time window check
    2. Minimum tick check (cold start)
    3. Volatility gate
    4. Score computation
    5. Dynamic threshold comparison
    6. Price cap check

    Args:
        prices: Price array for this side
        features: MA features dict (or None if insufficient data)
        vol: Current realized volatility
        R: Reward-to-risk ratio
        progress: Window progress [0, 1]
        time_remaining: Ticks until resolution
        tick: Current tick index
        params: Entry parameters for this side

    Returns:
        (entry_price, size_multiplier, score) if entry triggered, else None
    """
    # Filter 1: Time window
    if not (params.entry_start >= time_remaining > params.entry_end):
        return None

    # Filter 2: Cold start
    if MIN_TICKS is None:
        raise ValueError(
            "MIN_TICKS must be set via config — no hardcoded fallback."
        )
    if tick < MIN_TICKS:
        return None

    # Filter 3: Volatility gate
    if vol > params.vol_max:
        return None

    # Filter 4: Unified score
    from strategy.unified_score import calc_unified_score, get_threshold_linear, calc_score_sizing

    score = calc_unified_score(
        features, R, progress,
        params.w_slope_short, params.w_slope_long,
        params.w_spread, params.w_compression, params.w_dist,
        params.w_R, params.w_time
    )

    if score is None:
        return None

    # Filter 5: Dynamic threshold
    window_len = params.entry_start - params.entry_end
    time_in_phase = params.entry_start - time_remaining
    threshold = get_threshold_linear(time_in_phase, window_len,
                                      params.thresh_start, params.thresh_end)

    if score <= threshold:
        return None

    # Filter 6: Price cap
    current_price = prices[tick]
    if current_price > params.cap:
        return None

    # Entry triggered
    entry_price = current_price + (SLIPPAGE or 0)
    score_excess = score - threshold
    size_mult = calc_score_sizing(score_excess, params.score_scale)

    return (entry_price, size_mult, score)


def run_single_market(cache: Dict,
                      outcome: bool,
                      yes_params: EntryParams,
                      no_params: EntryParams) -> Optional[Dict]:
    """
    Run strategy on a single market window.

    Enforces single-side constraint: scans all ticks, takes the first
    entry that triggers (either YES or NO). If both trigger on the
    same tick, picks the higher score.

    The optimizer converged on a single-side dominant strategy. The non-dominant
    side's parameters typically have very tight entry windows or high thresholds
    that effectively disable its entries.

    Args:
        cache: Precomputed features for this market
        outcome: True if YES won, False if NO won
        yes_params: YES side entry parameters
        no_params: NO side entry parameters

    Returns:
        Trade result dict, or None if no entry
    """
    yes_prices = cache['yes_prices']
    no_prices = cache['no_prices']
    features_list = cache['features']
    vol_arr = cache['vol']
    yes_R_arr = cache['yes_R']
    no_R_arr = cache['no_R']
    progress_arr = cache['progress']
    total_ticks = len(yes_prices)

    for tick in range(total_ticks):
        features = features_list[tick] if tick < len(features_list) else None
        vol = vol_arr[tick] if tick < len(vol_arr) else 0
        progress = progress_arr[tick] if tick < len(progress_arr) else 0
        time_remaining = total_ticks - tick

        # Evaluate both sides
        yes_result = evaluate_entry(
            yes_prices, features, vol, yes_R_arr[tick], progress,
            time_remaining, tick, yes_params
        )
        no_result = evaluate_entry(
            no_prices, features, vol, no_R_arr[tick], progress,
            time_remaining, tick, no_params
        )

        # Resolve: pick higher score if both trigger
        chosen = None
        chosen_side = None

        if yes_result and no_result:
            if yes_result[2] >= no_result[2]:
                chosen, chosen_side = yes_result, 'yes'
            else:
                chosen, chosen_side = no_result, 'no'
        elif yes_result:
            chosen, chosen_side = yes_result, 'yes'
        elif no_result:
            chosen, chosen_side = no_result, 'no'

        if chosen is not None:
            entry_price, size_mult, score = chosen
            won = outcome if chosen_side == 'yes' else not outcome
            trade_R = (1 - entry_price) / entry_price if entry_price > 0 else 0
            if BASE_KELLY is None:
                raise ValueError(
                    "BASE_KELLY must be set via config — no hardcoded fallback."
                )
            f_trade = BASE_KELLY * size_mult

            if won:
                log_return = np.log(1 + f_trade * trade_R)
            else:
                log_return = np.log(max(1e-10, 1 - f_trade))

            return {
                'won': won,
                'side': chosen_side,
                'entry_price': entry_price,
                'size_mult': size_mult,
                'score': score,
                'log_return': log_return,
            }

    return None  # No entry triggered
