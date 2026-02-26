"""
Unified Scoring Function: 7-Feature Linear Factor Model

Computes a single entry score from 5 MA features + payoff geometry (R) + time progress.

Score = sum_i w_i * feature_i

    = w_slope_short * slope_short
    + w_slope_long  * slope_long
    + w_spread      * spread
    + w_compression * compression
    + w_dist        * dist
    + w_R           * R              (payoff geometry)
    + w_time        * progress       (window progress)

The linear form is intentional:
- Interpretable: each weight has a direct meaning
- Fast: O(1) per evaluation
- Regularizable: weight magnitudes are directly comparable
- No interaction terms: prevents overfitting on 30-param space

The score is compared against a time-varying threshold to generate entry signals.
"""

from typing import Dict, Optional


# All weights are PARAM_PLACEHOLDER — optimized via Optuna cross-validation
# Side A weights
YES_WEIGHTS = {
    'w_slope_short': None,   # PARAM_PLACEHOLDER
    'w_slope_long': None,    # PARAM_PLACEHOLDER
    'w_spread': None,        # PARAM_PLACEHOLDER
    'w_compression': None,   # PARAM_PLACEHOLDER
    'w_dist': None,          # PARAM_PLACEHOLDER
    'w_R': None,             # PARAM_PLACEHOLDER
    'w_time': None,          # PARAM_PLACEHOLDER
}

# Side B weights
NO_WEIGHTS = {
    'w_slope_short': None,   # PARAM_PLACEHOLDER
    'w_slope_long': None,    # PARAM_PLACEHOLDER
    'w_spread': None,        # PARAM_PLACEHOLDER
    'w_compression': None,   # PARAM_PLACEHOLDER
    'w_dist': None,          # PARAM_PLACEHOLDER
    'w_R': None,             # PARAM_PLACEHOLDER
    'w_time': None,          # PARAM_PLACEHOLDER
}


def calc_unified_score(features: Optional[Dict[str, float]],
                       R: float,
                       progress: float,
                       w_slope_short: float,
                       w_slope_long: float,
                       w_spread: float,
                       w_compression: float,
                       w_dist: float,
                       w_R: float,
                       w_time: float) -> Optional[float]:
    """
    Compute unified entry score from features + R + time.

    Volatility is handled upstream by a hard gate (vol <= vol_max),
    not by the scoring function. This keeps the score space clean
    and prevents vol from masking other signals.

    Args:
        features: Dict with 5 MA features (slope_short, slope_long, spread,
                  compression, dist). None if insufficient price history.
        R: Reward-to-risk ratio = (1 - entry_price) / entry_price
        progress: Window progress in [0, 1] (0 = start, 1 = resolution)
        w_*: Feature weights (optimized via cross-validation)

    Returns:
        Unified score (float), or None if features unavailable
    """
    if features is None:
        return None

    return (w_slope_short * features['slope_short']
          + w_slope_long * features['slope_long']
          + w_spread * features['spread']
          + w_compression * features['compression']
          + w_dist * features['dist']
          + w_R * R
          + w_time * progress)


def get_threshold_linear(time_in_phase: float,
                         window_len: float,
                         thresh_start: float,
                         thresh_end: float) -> float:
    """
    Time-varying entry threshold via linear interpolation.

    The threshold changes linearly from thresh_start (beginning of entry
    window) to thresh_end (end of entry window). This allows the strategy
    to become more or less selective as the window progresses.

    Args:
        time_in_phase: Time elapsed within the entry window
        window_len: Total length of the entry window
        thresh_start: Threshold at start of entry window
        thresh_end: Threshold at end of entry window

    Returns:
        Current threshold value
    """
    if window_len <= 0:
        return thresh_start
    progress = min(1.0, time_in_phase / window_len)
    return thresh_start + progress * (thresh_end - thresh_start)


def calc_score_sizing(score_excess: float, score_scale: float) -> float:
    """
    Score-based position size multiplier.

    When the score exceeds the threshold by a large margin, increase
    position size. When it barely exceeds, reduce size.

    multiplier = clamp(1.0 + score_scale * score_excess, [0.5, 1.5])

    Args:
        score_excess: score - threshold (how much score exceeds threshold)
        score_scale: Sensitivity parameter (PARAM_PLACEHOLDER)

    Returns:
        Size multiplier in [0.5, 1.5]
    """
    return max(0.5, min(1.5, 1.0 + score_scale * score_excess))
