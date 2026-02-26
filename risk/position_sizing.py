"""
Position Sizing: Score-Based Dynamic Kelly Fraction

Three-layer multiplicative sizing system:

Layer 1: Z-Score Sizing
    When price is overextended (high z-score relative to recent window),
    reduce position size. Prevents entering at statistical extremes.

    mult = max(z_min, 1.0 - z_decay * (z - z_threshold))

Layer 2: Slope Sizing
    Scale position with trend strength. Steeper trends in the entry
    direction get slightly larger positions.

    mult = clamp(1.0 + slope_k * normalized_slope, [0.7, 1.3])

Layer 3: R-Ratio Sizing
    Adjust for payoff geometry. Cheaper entries (higher R = (1-p)/p)
    get larger positions because the risk/reward is more favorable.

    mult = clamp((R / R_ref)^alpha, [r_min, 1.0])

Combined: product of all three layers, clamped to [0.25, 1.5]

Score-Based Alternative (used in production):
    mult = clamp(1.0 + score_scale * score_excess, [0.5, 1.5])

    This simpler approach was found to perform comparably to the
    three-layer system in cross-validation, with fewer parameters.
"""

import numpy as np
from typing import Dict, Optional


# All parameters are PARAM_PLACEHOLDER — optimized via cross-validation

def compute_z_score_sizing(prices: list,
                           z_lookback: int = None,
                           z_threshold: float = None,
                           z_decay_k: float = None,
                           z_min_size: float = None) -> float:
    """
    Reduce position size when price is statistically overextended.

    High z-score = price far from recent mean = likely to revert.
    This prevents entering at the tail of a move where the
    risk/reward has deteriorated.

    Args:
        prices: Recent price history
        z_lookback: Window for z-score computation (PARAM_PLACEHOLDER)
        z_threshold: Z-score threshold before reduction starts (PARAM_PLACEHOLDER)
        z_decay_k: Rate of size reduction per unit z above threshold (PARAM_PLACEHOLDER)
        z_min_size: Floor on size multiplier (PARAM_PLACEHOLDER)

    Returns:
        Size multiplier in [z_min_size, 1.0]
    """
    if any(p is None for p in [z_lookback, z_threshold, z_decay_k, z_min_size]):
        return 1.0  # No adjustment when params not set

    if len(prices) < z_lookback:
        return 1.0

    window = prices[-z_lookback:]
    mean_p = np.mean(window)
    std_p = np.std(window)

    if std_p < 1e-9:
        return 1.0

    z = abs(prices[-1] - mean_p) / std_p

    if z <= z_threshold:
        return 1.0

    return max(z_min_size, 1.0 - z_decay_k * (z - z_threshold))


def compute_slope_sizing(prices: list,
                         slope_lookback: int = None,
                         slope_k: float = None) -> float:
    """
    Scale position with trend strength.

    Positive slope (trend in entry direction) -> slight boost.
    Negative slope (trend against entry) -> slight reduction.

    Args:
        prices: Recent price history
        slope_lookback: Lookback for slope estimation (PARAM_PLACEHOLDER)
        slope_k: Sensitivity coefficient (PARAM_PLACEHOLDER)

    Returns:
        Size multiplier in [0.7, 1.3]
    """
    if slope_lookback is None or slope_k is None:
        return 1.0

    if len(prices) < slope_lookback:
        return 1.0

    window = prices[-slope_lookback:]
    x = np.arange(len(window))
    slope = np.polyfit(x, window, 1)[0]

    mean_p = np.mean(window)
    if mean_p < 1e-9:
        return 1.0

    norm_slope = slope / mean_p
    return max(0.7, min(1.3, 1.0 + slope_k * norm_slope))


def compute_r_sizing(R: float,
                     r_ref: float = None,
                     r_alpha: float = None,
                     r_min: float = None) -> float:
    """
    Scale position based on payoff geometry (reward-to-risk ratio).

    Higher R (cheaper entry) -> larger position.
    Lower R (expensive entry) -> smaller position.

    R = (1 - entry_price) / entry_price

    Args:
        R: Reward-to-risk ratio
        r_ref: Reference R for full sizing (PARAM_PLACEHOLDER)
        r_alpha: Power law exponent (PARAM_PLACEHOLDER)
        r_min: Floor multiplier (PARAM_PLACEHOLDER)

    Returns:
        Size multiplier in [r_min, 1.0]
    """
    if any(p is None for p in [r_ref, r_alpha, r_min]):
        return 1.0

    if R <= 0:
        return r_min

    return max(r_min, min(1.0, (R / r_ref) ** r_alpha))


def compute_full_sizing(prices: list, R: float, params: Dict) -> float:
    """
    Compute combined position size from all three layers.

    Combined = z_mult * slope_mult * r_mult, clamped to [0.25, 1.5]

    The multiplicative combination means each layer independently
    scales the position. A high z-score reduces size even if slope
    and R are favorable.

    Args:
        prices: Recent price history
        R: Reward-to-risk ratio
        params: Dict with all sizing parameters (all PARAM_PLACEHOLDER)

    Returns:
        Combined size multiplier in [0.25, 1.5]
    """
    z_mult = compute_z_score_sizing(
        prices,
        params.get('z_lookback'),
        params.get('z_threshold'),
        params.get('z_decay_k'),
        params.get('z_min_size'),
    )

    slope_mult = compute_slope_sizing(
        prices,
        params.get('slope_lookback'),
        params.get('slope_k'),
    )

    r_mult = compute_r_sizing(
        R,
        params.get('r_ref'),
        params.get('r_alpha'),
        params.get('r_min'),
    )

    return max(0.25, min(1.5, z_mult * slope_mult * r_mult))


def compute_score_sizing(score: float, threshold: float,
                         score_scale: float = None) -> float:
    """
    Simplified score-based sizing (production version).

    mult = clamp(1.0 + score_scale * (score - threshold), [0.5, 1.5])

    This single-layer approach was found to perform comparably to
    the three-layer system with fewer parameters. Used in the final
    optimized strategy.

    Args:
        score: Unified score from scoring function
        threshold: Current dynamic threshold
        score_scale: Sensitivity parameter (PARAM_PLACEHOLDER)

    Returns:
        Size multiplier in [0.5, 1.5]
    """
    if score_scale is None:
        return 1.0

    score_excess = score - threshold
    return max(0.5, min(1.5, 1.0 + score_scale * score_excess))
