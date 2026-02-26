"""
Feature Engineering: 5 Continuous Moving Average Features

Extracts microstructure features from binary market price series.
These 5 features form the input to the linear scoring function.

Features:
1. slope_short — Short-term momentum (normalized price change over short MA period)
2. slope_long — Long-term momentum (normalized price change over long MA period)
3. spread — MA divergence: (MA_short - MA_long) / price x 100
4. compression — MA convergence: (MA_max - MA_min) / price x 100
5. dist — Distance from mid MA: (price - MA_mid) / price x 100

All features are continuous, bounded, and interpretable as percentage deviations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


# MA periods — PARAM_PLACEHOLDER (optimized via cross-validation)
MA_SHORT = None   # Short-term lookback
MA_MID = None     # Mid-term lookback
MA_LONG = None    # Long-term lookback


def compute_ma_features(prices: np.ndarray, ma_short: int = None,
                        ma_mid: int = None, ma_long: int = None) -> Optional[Dict[str, float]]:
    """
    Compute 5 continuous MA features from a price series.

    Args:
        prices: Array of YES prices (binary market, 0-1 range)
        ma_short: Short MA period (default: MA_SHORT)
        ma_mid: Mid MA period (default: MA_MID)
        ma_long: Long MA period (default: MA_LONG)

    Returns:
        Dict with 5 features, or None if insufficient data
    """
    ma_short = ma_short or MA_SHORT
    ma_mid = ma_mid or MA_MID
    ma_long = ma_long or MA_LONG

    if ma_short is None or ma_mid is None or ma_long is None:
        raise ValueError("MA periods must be set (currently PARAM_PLACEHOLDER)")

    max_period = max(ma_short, ma_mid, ma_long)
    if len(prices) < max_period:
        return None

    price = prices[-1]
    if price < 0.01:
        return None

    # Simple moving averages
    ma_s = np.mean(prices[-ma_short:])
    ma_m = np.mean(prices[-ma_mid:])
    ma_l = np.mean(prices[-ma_long:])

    # Feature 1: Short-term slope (normalized momentum)
    slope_short = _calc_slope(prices, ma_short, ma_s)

    # Feature 2: Long-term slope (normalized momentum)
    slope_long = _calc_slope(prices, ma_long, ma_l)

    # Feature 3: MA spread (short vs long divergence)
    spread = (ma_s - ma_l) / price * 100

    # Feature 4: MA compression (convergence of all three MAs)
    ma_max = max(ma_s, ma_m, ma_l)
    ma_min = min(ma_s, ma_m, ma_l)
    compression = (ma_max - ma_min) / price * 100

    # Feature 5: Distance from mid MA
    dist = (price - ma_m) / price * 100

    return {
        'slope_short': slope_short,
        'slope_long': slope_long,
        'spread': spread,
        'compression': compression,
        'dist': dist,
    }


def _calc_slope(prices: np.ndarray, lookback: int, ma_value: float) -> float:
    """Normalized slope: (price[t] - price[t-lookback]) / lookback / MA x 100."""
    if len(prices) <= lookback or ma_value < 0.01:
        return 0.0
    return (prices[-1] - prices[-1 - lookback]) / lookback / ma_value * 100


def precompute_features_vectorized(price_series: pd.DataFrame,
                                    ma_short: int = None,
                                    ma_mid: int = None,
                                    ma_long: int = None) -> Dict:
    """
    Vectorized feature precomputation for backtest efficiency.

    Computes all 5 MA features for every tick in a market window
    using cumulative sum tricks for O(N) moving averages.

    Args:
        price_series: DataFrame with 'yes_price' and 'no_price' columns
        ma_short: Short MA period
        ma_mid: Mid MA period
        ma_long: Long MA period

    Returns:
        Dict with vectorized feature arrays and metadata
    """
    ma_short = ma_short or MA_SHORT
    ma_mid = ma_mid or MA_MID
    ma_long = ma_long or MA_LONG

    if ma_short is None:
        raise ValueError("MA periods must be set (currently PARAM_PLACEHOLDER)")

    yes_prices = price_series['yes_price'].values
    no_prices = price_series['no_price'].values
    n_ticks = len(yes_prices)

    # Progress through window [0, 1]
    progress = np.arange(n_ticks) / max(1, n_ticks - 1)

    # Rolling volatility
    vol = pd.Series(yes_prices).rolling(window=50, min_periods=2).std().fillna(0).values

    # Reward-to-risk ratio (R)
    slippage = None  # PARAM_PLACEHOLDER
    if slippage is not None:
        yes_entry = yes_prices + slippage
        no_entry = no_prices + slippage
        yes_R = np.where(yes_entry > 0.01, (1 - yes_entry) / yes_entry, 0)
        no_R = np.where(no_entry > 0.01, (1 - no_entry) / no_entry, 0)
    else:
        yes_R = np.zeros(n_ticks)
        no_R = np.zeros(n_ticks)

    # Vectorized MA computation via cumulative sums
    features_list = [None] * n_ticks

    if n_ticks >= ma_long:
        cumsum = np.cumsum(np.insert(yes_prices, 0, 0))

        ma_s = np.zeros(n_ticks)
        ma_m = np.zeros(n_ticks)
        ma_l = np.zeros(n_ticks)

        ma_s[ma_short-1:] = (cumsum[ma_short:] - cumsum[:-ma_short]) / ma_short
        ma_m[ma_mid-1:] = (cumsum[ma_mid:] - cumsum[:-ma_mid]) / ma_mid
        ma_l[ma_long-1:] = (cumsum[ma_long:] - cumsum[:-ma_long]) / ma_long

        # Vectorized slopes
        slope_short = np.zeros(n_ticks)
        slope_long = np.zeros(n_ticks)
        slope_short[ma_short:] = (yes_prices[ma_short:] - yes_prices[:-ma_short]) / ma_short
        slope_short = np.where(ma_s > 0.01, slope_short / ma_s * 100, 0)
        slope_long[ma_long:] = (yes_prices[ma_long:] - yes_prices[:-ma_long]) / ma_long
        slope_long = np.where(ma_l > 0.01, slope_long / ma_l * 100, 0)

        # Vectorized spread, compression, dist
        spread = np.where(yes_prices > 0.01, (ma_s - ma_l) / yes_prices * 100, 0)
        ma_max = np.maximum(np.maximum(ma_s, ma_m), ma_l)
        ma_min = np.minimum(np.minimum(ma_s, ma_m), ma_l)
        compression = np.where(yes_prices > 0.01, (ma_max - ma_min) / yes_prices * 100, 0)
        dist = np.where(yes_prices > 0.01, (yes_prices - ma_m) / yes_prices * 100, 0)

        for tick in range(ma_long - 1, n_ticks):
            if yes_prices[tick] < 0.01:
                continue
            features_list[tick] = {
                'slope_short': slope_short[tick],
                'slope_long': slope_long[tick],
                'spread': spread[tick],
                'compression': compression[tick],
                'dist': dist[tick],
            }

    return {
        'yes_prices': yes_prices,
        'no_prices': no_prices,
        'features': features_list,
        'vol': vol,
        'yes_R': yes_R,
        'no_R': no_R,
        'progress': progress,
    }
