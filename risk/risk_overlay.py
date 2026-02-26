"""
Risk Overlay: Health-Based Position Size Management

Monitors rolling win rate and adjusts position sizing when performance
deteriorates. Acts as a circuit breaker to limit drawdown damage.

Three-tier system:
- WR >= threshold_high: Full sizing (1.0x)
- threshold_low <= WR < threshold_high: Reduced sizing (PARAM_PLACEHOLDER)
- WR < threshold_low: Heavy reduction (PARAM_PLACEHOLDER)

Hysteresis prevents whipsawing:
- Step DOWN is immediate (protect capital quickly)
- Step UP requires N consecutive trades at the new level (cooldown)

This asymmetry is intentional: the cost of being too slow to reduce size
(drawdown) is much higher than the cost of being too slow to increase size
(missed opportunity).
"""

import numpy as np
from typing import List, Optional, Tuple


# All thresholds are PARAM_PLACEHOLDER
ROLLING_LOOKBACK = None     # Number of recent trades to track
THRESHOLD_HIGH = None       # WR above this -> full sizing
THRESHOLD_LOW = None        # WR below this -> heavy reduction
COOLDOWN_TRADES = None      # Trades before allowing recovery


class HealthOverlay:
    """
    Rolling win-rate based position size manager.

    Tracks recent trade outcomes and computes a multiplicative
    health factor that reduces position size during drawdowns.

    State is designed to be serializable for persistence across
    bot restarts.
    """

    def __init__(self,
                 lookback: int = None,
                 threshold_high: float = None,
                 threshold_low: float = None,
                 cooldown: int = None):
        """
        Initialize health overlay.

        All parameters are PARAM_PLACEHOLDER — optimized via
        backtesting on historical drawdown episodes.

        Args:
            lookback: Rolling window size for win rate
            threshold_high: WR threshold for full sizing
            threshold_low: WR threshold for heavy reduction
            cooldown: Trades before allowing size recovery
        """
        self.lookback = lookback or ROLLING_LOOKBACK or 20
        self.threshold_high = threshold_high or THRESHOLD_HIGH or 0.6
        self.threshold_low = threshold_low or THRESHOLD_LOW or 0.5
        self.cooldown = cooldown or COOLDOWN_TRADES or 2

        self._trade_outcomes: List[bool] = []
        self._cooldown_remaining: int = 0
        self._last_mult: float = 1.0

    def record_trade(self, won: bool):
        """Record a trade outcome."""
        self._trade_outcomes.append(won)
        # Keep only lookback window
        if len(self._trade_outcomes) > self.lookback:
            self._trade_outcomes = self._trade_outcomes[-self.lookback:]

    def compute_health_mult(self) -> float:
        """
        Compute health multiplier with hysteresis.

        Returns:
            Multiplier in {0.5, 0.75, 1.0} based on rolling WR
        """
        if len(self._trade_outcomes) < self.lookback:
            return 1.0

        rolling_wr = sum(self._trade_outcomes[-self.lookback:]) / self.lookback

        # Compute raw target multiplier
        if rolling_wr < self.threshold_low:
            raw_mult = 0.5
        elif rolling_wr < self.threshold_high:
            raw_mult = 0.75
        else:
            raw_mult = 1.0

        # Apply hysteresis
        if raw_mult < self._last_mult:
            # Step DOWN: immediate
            health_mult = raw_mult
            self._cooldown_remaining = 0
        elif raw_mult > self._last_mult:
            if self._cooldown_remaining > 0:
                # Want to step up but in cooldown
                health_mult = self._last_mult
                self._cooldown_remaining -= 1
            else:
                health_mult = raw_mult
        else:
            health_mult = raw_mult

        # Set cooldown when stepping down from 1.0
        if health_mult < 1.0 and self._last_mult == 1.0:
            self._cooldown_remaining = self.cooldown

        self._last_mult = health_mult
        return health_mult

    def get_state(self) -> dict:
        """Serialize state for persistence."""
        return {
            'trade_outcomes': self._trade_outcomes,
            'cooldown': self._cooldown_remaining,
            'last_mult': self._last_mult,
        }

    def load_state(self, state: dict):
        """Restore state from persistence."""
        self._trade_outcomes = state.get('trade_outcomes', [])
        self._cooldown_remaining = state.get('cooldown', 0)
        self._last_mult = state.get('last_mult', 1.0)
