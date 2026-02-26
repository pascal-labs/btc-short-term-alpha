# Position Sizing: From Kelly Criterion to Score-Based Dynamic Sizing

## Kelly Criterion Foundation

The Kelly criterion says to size each bet to maximize the expected log return (equivalently, the geometric growth rate). For a binary bet with win probability p and payout odds b (you risk 1 to win b):

```
f* = (p * b - 1) / b = p - (1-p)/b
```

In the binary market context:
- Entry price = c (cost of the contract)
- Win payout = 1 (binary markets pay $1 on correct outcome)
- R = (1 - c) / c (reward-to-risk ratio)
- f* = p - (1-p) * c / (1-c) = p - (1-p) / R

With a 77% win rate and typical entry prices around PARAM_PLACEHOLDER, the theoretical Kelly fraction is substantial. In practice, we use a fraction of Kelly (BASE_KELLY = PARAM_PLACEHOLDER) for several reasons:

1. **Estimation error**: The true win probability p is unknown. Overbetting due to overestimated p causes faster bankroll depletion than underbetting.
2. **Non-stationarity**: The edge changes over time. Fractional Kelly provides a buffer against edge deterioration.
3. **Drawdown tolerance**: Full Kelly produces drawdowns of 50%+. Fractional Kelly (e.g., 1/4 Kelly) reduces max drawdown roughly proportionally.

## Score-Based Dynamic Sizing (Production)

The production strategy uses a simplified single-layer sizing system:

```
size_multiplier = clamp(1.0 + score_scale * (score - threshold), [0.5, 1.5])
```

Where:
- `score` is the unified 7-feature score at the entry tick
- `threshold` is the current dynamic threshold
- `score_scale` (PARAM_PLACEHOLDER) controls sensitivity
- The result multiplies the base Kelly fraction

**Intuition**: When the score barely exceeds the threshold (score_excess ~ 0), the entry is marginal -- size conservatively (multiplier near 1.0). When the score exceeds the threshold by a large margin (high score_excess), the entry is high-conviction -- size up (multiplier up to 1.5). This creates a natural confidence-weighted sizing system.

**Why [0.5, 1.5] bounds**: The floor of 0.5 prevents Kelly from being reduced below half (still worth trading). The ceiling of 1.5 prevents overconcentration in any single trade. These bounds also serve as implicit regularization -- the optimizer can't push position sizes to extreme values.

## Three-Layer Legacy System

Before settling on score-based sizing, the strategy used a three-layer multiplicative system. It's included here because the design principles are instructive, even though the simpler system was ultimately preferred.

### Layer 1: Z-Score Sizing

```
z = |price - mean(prices[-lookback:])| / std(prices[-lookback:])
mult = max(z_min, 1.0 - z_decay * (z - z_threshold))   if z > z_threshold
mult = 1.0                                              if z <= z_threshold
```

When the current price is statistically overextended (high z-score relative to recent history), reduce position size. The thesis: if the price has already moved far, the remaining discount is smaller and the risk of reversal is higher.

### Layer 2: Slope Sizing

```
slope = linear_regression_slope(prices[-lookback:])
norm_slope = slope / mean(prices[-lookback:])
mult = clamp(1.0 + slope_k * norm_slope, [0.7, 1.3])
```

Scale position with trend strength. A strong trend in the entry direction (for NO: downward YES price = upward NO price) gets a slight boost. A trend against the entry direction gets a slight reduction.

### Layer 3: R-Ratio Sizing

```
mult = clamp((R / R_ref) ^ alpha, [r_min, 1.0])
```

Adjust for payoff geometry. Cheaper entries (higher R = more profit per unit risk) get larger positions. Expensive entries (lower R) get smaller positions. This is a direct application of Kelly -- the optimal fraction scales with the odds.

### Combined

```
total_mult = z_mult * slope_mult * r_mult
total_mult = clamp(total_mult, [0.25, 1.5])
```

The multiplicative combination means each layer independently scales the position. A high z-score reduces size even if slope and R are favorable. This creates a conservative consensus: all three layers must be at least neutral for full sizing.

### Why Score-Based Won

In cross-validation, the three-layer system performed comparably to the score-based system but with 6 additional parameters (z_lookback, z_threshold, z_decay, slope_lookback, slope_k, r_ref, r_alpha, r_min). The score-based system achieved similar performance with a single parameter (score_scale), reducing overfitting risk.

The likely reason: the unified score already captures the information that the three layers were separately extracting. Slope sizing overlaps with slope_short/slope_long features. R-ratio sizing overlaps with the w_R weight. Z-score sizing overlaps with the dist feature. The score-based approach avoids double-counting these signals.

## Health Overlay

On top of the per-trade sizing, a health overlay monitors rolling win rate and adjusts position size during drawdowns.

### Three-Tier System

| Rolling WR | Health Multiplier | Interpretation |
|-----------|------------------|----------------|
| >= threshold_high | 1.0x | Full sizing, strategy is healthy |
| threshold_low to threshold_high | 0.75x | Caution, reducing exposure |
| < threshold_low | 0.5x | Defensive, significant reduction |

All thresholds are PARAM_PLACEHOLDER.

### Hysteresis

The step-down (reducing size) is immediate: as soon as rolling WR drops below a threshold, the multiplier changes. The step-up (increasing size) requires a cooldown period: N consecutive trades at the current level before allowing size recovery.

This asymmetry is intentional:
- **Cost of slow step-down**: Continued large positions during a drawdown, compounding losses.
- **Cost of slow step-up**: Missed opportunity from smaller positions during recovery, lower gains.

The cost of the first is much higher than the second, so the asymmetry is warranted.

### State Persistence

The health overlay maintains state (recent trade outcomes, cooldown counter, last multiplier) that is designed to be serializable. In a production bot, this state would be persisted across restarts to maintain continuity.

## Final Position Size Formula

```
position_size = BASE_KELLY * score_multiplier * health_multiplier
```

Where:
- `BASE_KELLY` (PARAM_PLACEHOLDER) is the fractional Kelly base
- `score_multiplier` in [0.5, 1.5] from score-based sizing
- `health_multiplier` in {0.5, 0.75, 1.0} from health overlay

The resulting position size is always a small fraction of the bankroll, consistent with the fractional Kelly approach. Even in the worst case (all multipliers at minimum), the position is 0.25x the base Kelly fraction -- still positive, still trading, but at greatly reduced risk.
