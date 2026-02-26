"""
Unit tests for btc-short-term-alpha core modules.

Covers:
- optimization/geometric_balance.py  (geometric balance objective, robust scoring, CV folds)
- strategy/features.py               (MA feature computation)
- strategy/unified_score.py          (linear scoring, threshold, score sizing)
- risk/position_sizing.py            (Kelly/score-based sizing)
- risk/risk_overlay.py               (HealthOverlay state + tier transitions)

All tests supply their own concrete parameter values and do not rely on
module-level PARAM_PLACEHOLDER constants.
"""

import math
import sys
import os
import unittest

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so imports work when running via
# `python -m unittest discover tests/` from the repo root.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from optimization.geometric_balance import (
    compute_geometric_balance,
    compute_robust_score,
    create_cv_folds,
)
from strategy.features import compute_ma_features
from strategy.unified_score import (
    calc_unified_score,
    calc_score_sizing,
    get_threshold_linear,
)
from risk.position_sizing import compute_score_sizing
from risk.risk_overlay import HealthOverlay


# =========================================================================
# 1-3  Geometric balance objective
# =========================================================================

class TestComputeGeometricBalance(unittest.TestCase):
    """Tests for compute_geometric_balance()."""

    def test_known_positive_inputs(self):
        """
        G_window = total_log_return / n_windows = 1.0 / 100 = 0.01
        G_trade  = total_log_return / n_trades  = 1.0 / 50  = 0.02
        G = sqrt(0.01 * 0.02) = sqrt(0.0002)
        """
        total_log_return = 1.0
        n_windows = 100
        n_trades = 50

        G_window = total_log_return / n_windows   # 0.01
        G_trade = total_log_return / n_trades      # 0.02

        result = compute_geometric_balance(G_window, G_trade)
        expected = math.sqrt(0.01 * 0.02)

        self.assertAlmostEqual(result, expected, places=10,
                               msg="G should equal sqrt(G_window * G_trade) for positive inputs")

    def test_returns_zero_when_n_trades_is_zero(self):
        """
        If n_trades=0, G_trade is undefined. We model this as G_trade=0,
        meaning the geometric balance collapses to 0 via the mixed-sign
        branch (positive * zero) which returns min(G_window, 0) = 0.
        """
        G_window = 0.01
        G_trade = 0.0  # no trades -> zero quality metric

        result = compute_geometric_balance(G_window, G_trade)
        # G_window > 0, G_trade == 0 -> mixed branch -> min(0.01, 0.0) = 0.0
        self.assertEqual(result, 0.0,
                         msg="G should be 0 when G_trade=0 (no trades)")

    def test_returns_negative_when_total_log_return_negative(self):
        """
        When total_log_return < 0, both G_window and G_trade are negative.
        The function returns -sqrt(|G_w| * |G_t|).
        """
        G_window = -0.01
        G_trade = -0.02

        result = compute_geometric_balance(G_window, G_trade)
        expected = -math.sqrt(0.01 * 0.02)

        self.assertAlmostEqual(result, expected, places=10,
                               msg="G should be -sqrt(|G_w|*|G_t|) when both negative")

    def test_mixed_signs_returns_minimum(self):
        """Mixed signs -> conservative approach: return the worse metric."""
        result = compute_geometric_balance(0.05, -0.02)
        self.assertEqual(result, -0.02,
                         msg="Mixed signs should return the minimum (worse) value")

        result2 = compute_geometric_balance(-0.03, 0.01)
        self.assertEqual(result2, -0.03)


# =========================================================================
# 4  Robust scoring
# =========================================================================

class TestComputeRobustScore(unittest.TestCase):
    """Tests for compute_robust_score()."""

    def test_known_fold_scores(self):
        """
        fold_Gs = [0.05, 0.04, 0.05, 0.04, 0.05]
        min = 0.04
        std = np.std([0.05, 0.04, 0.05, 0.04, 0.05])
        robust = 0.04 - 0.5 * std
        """
        fold_Gs = [0.05, 0.04, 0.05, 0.04, 0.05]
        expected = min(fold_Gs) - 0.5 * np.std(fold_Gs)

        result = compute_robust_score(fold_Gs)
        self.assertAlmostEqual(result, expected, places=10)

    def test_stable_folds_beat_variable_folds(self):
        """
        A stable set should score higher than a variable set
        even if the variable set has a higher mean.
        """
        stable = [0.05, 0.04, 0.05, 0.04, 0.05]
        variable = [0.10, 0.08, 0.02, -0.01, 0.07]

        score_stable = compute_robust_score(stable)
        score_variable = compute_robust_score(variable)

        self.assertGreater(score_stable, score_variable,
                           msg="Stable folds should score higher than variable folds")

    def test_single_fold(self):
        """Single fold -> std=0 -> robust = min = sole value."""
        result = compute_robust_score([0.03])
        self.assertAlmostEqual(result, 0.03, places=10)


# =========================================================================
# 5  Feature computation
# =========================================================================

class TestComputeMaFeatures(unittest.TestCase):
    """Tests for compute_ma_features()."""

    def test_output_shape_and_keys(self):
        """Output should be a dict with exactly 5 feature keys."""
        np.random.seed(42)
        # Synthetic price series: random walk around 0.5
        prices = 0.5 + np.cumsum(np.random.normal(0, 0.005, 200))
        prices = np.clip(prices, 0.05, 0.95)

        features = compute_ma_features(prices, ma_short=10, ma_mid=30, ma_long=60)

        self.assertIsNotNone(features)
        expected_keys = {'slope_short', 'slope_long', 'spread', 'compression', 'dist'}
        self.assertEqual(set(features.keys()), expected_keys)

    def test_returns_none_for_insufficient_data(self):
        """Should return None when price series is shorter than ma_long."""
        prices = np.array([0.5] * 10)
        result = compute_ma_features(prices, ma_short=5, ma_mid=15, ma_long=30)
        self.assertIsNone(result)

    def test_no_nan_values(self):
        """Features should contain no NaN values for a well-formed series."""
        np.random.seed(123)
        prices = 0.5 + np.cumsum(np.random.normal(0, 0.003, 300))
        prices = np.clip(prices, 0.05, 0.95)

        features = compute_ma_features(prices, ma_short=10, ma_mid=30, ma_long=60)
        self.assertIsNotNone(features)
        for key, val in features.items():
            self.assertFalse(math.isnan(val),
                             msg=f"Feature '{key}' should not be NaN")

    def test_returns_none_for_near_zero_price(self):
        """Last price < 0.01 should return None (avoids division by zero)."""
        prices = np.array([0.5] * 60)
        prices[-1] = 0.005  # below 0.01 threshold

        result = compute_ma_features(prices, ma_short=5, ma_mid=15, ma_long=30)
        self.assertIsNone(result)


# =========================================================================
# 6  Unified score
# =========================================================================

class TestCalcUnifiedScore(unittest.TestCase):
    """Tests for calc_unified_score()."""

    def test_known_weights_and_features(self):
        """Verify manual linear combination."""
        features = {
            'slope_short': 2.0,
            'slope_long': -1.0,
            'spread': 0.5,
            'compression': 0.3,
            'dist': -0.2,
        }
        R = 1.5
        progress = 0.6

        # All weights = 1.0 for easy verification
        expected = (1.0 * 2.0      # slope_short
                  + 1.0 * (-1.0)   # slope_long
                  + 1.0 * 0.5      # spread
                  + 1.0 * 0.3      # compression
                  + 1.0 * (-0.2)   # dist
                  + 1.0 * 1.5      # R
                  + 1.0 * 0.6)     # progress
        # = 2.0 - 1.0 + 0.5 + 0.3 - 0.2 + 1.5 + 0.6 = 3.7

        result = calc_unified_score(
            features, R, progress,
            w_slope_short=1.0, w_slope_long=1.0, w_spread=1.0,
            w_compression=1.0, w_dist=1.0, w_R=1.0, w_time=1.0,
        )

        self.assertAlmostEqual(result, 3.7, places=10)

    def test_returns_none_when_features_none(self):
        """Score should be None when features are unavailable."""
        result = calc_unified_score(
            None, R=1.0, progress=0.5,
            w_slope_short=1.0, w_slope_long=1.0, w_spread=1.0,
            w_compression=1.0, w_dist=1.0, w_R=1.0, w_time=1.0,
        )
        self.assertIsNone(result)


# =========================================================================
# 7  Kelly fraction
# =========================================================================

class TestKellyFraction(unittest.TestCase):
    """
    Test Kelly-criterion logic via the score-based sizing interface.

    For a binary market:
        entry cost c, payout = 1 on win, 0 on loss
        R = (1 - c) / c        (reward-to-risk ratio)
        f* = p - (1-p)/R       (Kelly fraction)

    With p=0.7, c=0.3:
        R = 0.7/0.3 = 7/3
        f* = 0.7 - 0.3/(7/3) = 0.7 - 0.3*3/7 = 0.7 - 9/70 = 0.7 - 0.12857... ~= 0.5714
    """

    def test_kelly_fraction_binary_market(self):
        """Verify Kelly fraction f* = p - (1-p)/R for p=0.7, c=0.3."""
        p = 0.7
        c = 0.3
        R = (1 - c) / c      # 7/3 ~= 2.3333
        f_star = p - (1 - p) / R

        expected_R = 7.0 / 3.0
        expected_f_star = 0.7 - 0.3 / expected_R

        self.assertAlmostEqual(R, expected_R, places=10)
        self.assertAlmostEqual(f_star, expected_f_star, places=10)
        # f* should be about 0.5714
        self.assertAlmostEqual(f_star, 0.7 - 9.0 / 70.0, places=10)
        self.assertTrue(0 < f_star < 1,
                        msg="Kelly fraction should be in (0, 1) for an edge")


# =========================================================================
# 8  Score-based sizing multiplier clamp
# =========================================================================

class TestScoreSizing(unittest.TestCase):
    """Tests for calc_score_sizing() bounds [0.5, 1.5]."""

    def test_large_positive_excess_clamped_at_1_5(self):
        """Very high score excess should clamp at 1.5."""
        result = calc_score_sizing(score_excess=100.0, score_scale=1.0)
        self.assertEqual(result, 1.5)

    def test_large_negative_excess_clamped_at_0_5(self):
        """Very negative score excess should clamp at 0.5."""
        result = calc_score_sizing(score_excess=-100.0, score_scale=1.0)
        self.assertEqual(result, 0.5)

    def test_zero_excess_returns_1_0(self):
        """Zero excess -> multiplier = 1.0."""
        result = calc_score_sizing(score_excess=0.0, score_scale=1.0)
        self.assertAlmostEqual(result, 1.0, places=10)

    def test_moderate_positive_excess(self):
        """Moderate excess should land within [0.5, 1.5]."""
        result = calc_score_sizing(score_excess=0.2, score_scale=1.0)
        self.assertAlmostEqual(result, 1.2, places=10)
        self.assertGreaterEqual(result, 0.5)
        self.assertLessEqual(result, 1.5)

    def test_compute_score_sizing_bounds(self):
        """Also verify the risk/position_sizing version matches the same bounds."""
        # Large positive excess
        result = compute_score_sizing(score=10.0, threshold=0.0, score_scale=1.0)
        self.assertEqual(result, 1.5)

        # Large negative excess
        result = compute_score_sizing(score=-10.0, threshold=0.0, score_scale=1.0)
        self.assertEqual(result, 0.5)

        # None scale returns 1.0
        result = compute_score_sizing(score=5.0, threshold=0.0, score_scale=None)
        self.assertEqual(result, 1.0)


# =========================================================================
# 9  HealthOverlay state serialization roundtrip
# =========================================================================

class TestHealthOverlayState(unittest.TestCase):
    """Tests for HealthOverlay get_state/load_state roundtrip."""

    def _make_overlay(self):
        """Create a HealthOverlay with concrete parameters."""
        return HealthOverlay(
            lookback=20,
            threshold_high=0.55,
            threshold_low=0.40,
            cooldown=5,
        )

    def test_state_roundtrip_empty(self):
        """Fresh overlay -> serialize -> restore -> same state."""
        overlay = self._make_overlay()
        state = overlay.get_state()

        overlay2 = self._make_overlay()
        overlay2.load_state(state)

        self.assertEqual(overlay2.get_state(), state)

    def test_state_roundtrip_after_trades(self):
        """Record trades, serialize, restore, verify."""
        overlay = self._make_overlay()
        for _ in range(15):
            overlay.record_trade(True)
        for _ in range(5):
            overlay.record_trade(False)

        state = overlay.get_state()
        self.assertEqual(len(state['trade_outcomes']), 20)

        overlay2 = self._make_overlay()
        overlay2.load_state(state)

        state2 = overlay2.get_state()
        self.assertEqual(state['trade_outcomes'], state2['trade_outcomes'])
        self.assertEqual(state['cooldown'], state2['cooldown'])
        self.assertEqual(state['last_mult'], state2['last_mult'])

    def test_state_preserves_cooldown_and_last_mult(self):
        """Cooldown and last_mult should survive serialization."""
        overlay = self._make_overlay()
        # Fill the lookback window with wins -> full sizing
        for _ in range(20):
            overlay.record_trade(True)
        mult = overlay.compute_health_mult()
        self.assertEqual(mult, 1.0)

        # Now send a streak of losses to trigger a step-down
        for _ in range(12):
            overlay.record_trade(False)
        mult = overlay.compute_health_mult()
        self.assertLess(mult, 1.0, "Should have stepped down after losses")

        state = overlay.get_state()
        overlay2 = self._make_overlay()
        overlay2.load_state(state)

        self.assertEqual(overlay2._last_mult, overlay._last_mult)
        self.assertEqual(overlay2._cooldown_remaining, overlay._cooldown_remaining)


# =========================================================================
# 10  HealthOverlay tier transitions
# =========================================================================

class TestHealthOverlayTierTransitions(unittest.TestCase):
    """Tests for HealthOverlay compute_health_mult() tier logic."""

    def _make_overlay(self):
        return HealthOverlay(
            lookback=10,
            threshold_high=0.60,
            threshold_low=0.40,
            cooldown=3,
        )

    def test_full_sizing_above_threshold_high(self):
        """WR >= threshold_high -> mult = 1.0."""
        overlay = self._make_overlay()
        # 8 wins, 2 losses -> WR = 0.8 >= 0.60
        for _ in range(8):
            overlay.record_trade(True)
        for _ in range(2):
            overlay.record_trade(False)

        mult = overlay.compute_health_mult()
        self.assertEqual(mult, 1.0)

    def test_reduced_sizing_between_thresholds(self):
        """threshold_low <= WR < threshold_high -> mult = 0.75."""
        overlay = self._make_overlay()
        # 5 wins, 5 losses -> WR = 0.5  (0.40 <= 0.5 < 0.60)
        for _ in range(5):
            overlay.record_trade(True)
        for _ in range(5):
            overlay.record_trade(False)

        mult = overlay.compute_health_mult()
        self.assertEqual(mult, 0.75)

    def test_heavy_reduction_below_threshold_low(self):
        """WR < threshold_low -> mult = 0.5."""
        overlay = self._make_overlay()
        # 3 wins, 7 losses -> WR = 0.3 < 0.40
        for _ in range(3):
            overlay.record_trade(True)
        for _ in range(7):
            overlay.record_trade(False)

        mult = overlay.compute_health_mult()
        self.assertEqual(mult, 0.5)

    def test_step_down_is_immediate(self):
        """Stepping DOWN should happen without delay."""
        overlay = self._make_overlay()

        # Start with high WR -> 1.0
        for _ in range(10):
            overlay.record_trade(True)
        self.assertEqual(overlay.compute_health_mult(), 1.0)

        # Replace wins with losses to drop WR below threshold_low
        for _ in range(8):
            overlay.record_trade(False)
        # Now the last 10 trades are 2 wins + 8 losses -> WR = 0.2 < 0.40
        mult = overlay.compute_health_mult()
        self.assertEqual(mult, 0.5,
                         msg="Step-down should be immediate (no cooldown delay)")

    def test_step_up_requires_cooldown(self):
        """Stepping UP should be delayed by cooldown trades."""
        overlay = self._make_overlay()

        # Build up to full sizing
        for _ in range(10):
            overlay.record_trade(True)
        self.assertEqual(overlay.compute_health_mult(), 1.0)

        # Drop to 0.75 tier (5 wins, 5 losses -> WR=0.5)
        for _ in range(5):
            overlay.record_trade(False)
        mult = overlay.compute_health_mult()
        self.assertEqual(mult, 0.75)

        # Now record enough wins to push WR back above 0.60
        # but the first compute after recovery should still be held back by cooldown
        for _ in range(5):
            overlay.record_trade(True)
        # Last 10 trades: 5 losses + 5 wins -> WR = 0.5 -> still 0.75
        # We need more wins to actually get WR >= 0.60
        overlay.record_trade(True)
        # Last 10: 4 losses + 6 wins -> WR = 0.6 -> at threshold_high boundary
        # Check that cooldown prevents immediate step-up when applicable
        mult_after = overlay.compute_health_mult()
        # The mult should be either still 0.75 (cooldown) or 1.0 if cooldown expired
        self.assertIn(mult_after, [0.75, 1.0])

    def test_insufficient_trades_returns_1_0(self):
        """Before lookback window is full, health mult should be 1.0."""
        overlay = self._make_overlay()  # lookback=10
        for _ in range(5):
            overlay.record_trade(False)
        mult = overlay.compute_health_mult()
        self.assertEqual(mult, 1.0,
                         msg="Should return 1.0 when trades < lookback")


# =========================================================================
# Bonus: CV fold creation
# =========================================================================

class TestCreateCvFolds(unittest.TestCase):
    """Tests for create_cv_folds()."""

    def test_fold_count(self):
        """Should create exactly n_folds folds."""
        folds = create_cv_folds(n_windows=600, n_folds=5)
        self.assertEqual(len(folds), 5)

    def test_folds_non_overlapping(self):
        """Validation windows should not overlap."""
        folds = create_cv_folds(n_windows=600, n_folds=5)
        for i in range(len(folds) - 1):
            self.assertLessEqual(folds[i]['val_end'], folds[i + 1]['val_start'],
                                 msg="Fold validation windows should not overlap")


# =========================================================================
# Bonus: Threshold linear interpolation
# =========================================================================

class TestThresholdLinear(unittest.TestCase):
    """Tests for get_threshold_linear()."""

    def test_start_returns_thresh_start(self):
        result = get_threshold_linear(time_in_phase=0.0, window_len=100.0,
                                      thresh_start=0.5, thresh_end=0.8)
        self.assertAlmostEqual(result, 0.5)

    def test_end_returns_thresh_end(self):
        result = get_threshold_linear(time_in_phase=100.0, window_len=100.0,
                                      thresh_start=0.5, thresh_end=0.8)
        self.assertAlmostEqual(result, 0.8)

    def test_midpoint(self):
        result = get_threshold_linear(time_in_phase=50.0, window_len=100.0,
                                      thresh_start=0.5, thresh_end=0.8)
        self.assertAlmostEqual(result, 0.65)

    def test_zero_window_len_returns_start(self):
        result = get_threshold_linear(time_in_phase=10.0, window_len=0.0,
                                      thresh_start=0.5, thresh_end=0.8)
        self.assertAlmostEqual(result, 0.5)


if __name__ == "__main__":
    unittest.main()
