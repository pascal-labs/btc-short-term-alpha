# Strategy: Late-Window Lock-In for Binary BTC Markets

## The Lock-In Thesis

Binary BTC markets on Polymarket resolve every 15 minutes with a simple question: did BTC close above or below a specific price? Within each 15-minute window, the BTC spot price evolves stochastically -- it moves up and down with some volatility. The binary contract price reflects the market's real-time estimate of the probability that the resolution condition will be met.

The key insight is temporal: as the window progresses and BTC has already moved significantly in one direction, the outcome becomes increasingly certain. If BTC needs to close above $100,000 and it's currently at $100,150 with 2 minutes left, the YES probability is very high. But the contract might still trade at $0.92 rather than $1.00 because resolution hasn't happened yet -- there's still some time remaining, and the market prices in the possibility of a reversal.

This creates a systematic discount: the contract lags the spot reality. The "lock-in" strategy enters during this late-window phase, buying contracts at a discount to their expected resolution value.

## Why Late-Window Works

Three factors create the late-window edge:

1. **Information asymmetry is temporal, not informational.** Everyone can see the BTC spot price. The discount isn't from having better information -- it's from acting on the implication of existing information before the market fully prices it in. The contract price is a lagging indicator of the spot price implication.

2. **Volatility compression.** With less time remaining, the range of possible BTC movements narrows. A move that could easily happen in 10 minutes becomes unlikely in 2 minutes. This reduces the true uncertainty faster than the market adjusts.

3. **Liquidity dynamics.** Late in the window, active traders are less willing to take the other side at fair value because the risk/reward of being wrong is worse. This creates brief dislocations that can be captured.

## Entry Logic: The Filter Cascade

Every tick (price update) during the entry window is evaluated through a cascade of filters. All filters must pass for an entry to trigger:

### Filter 1: Time Window Gate

Only evaluate entries during a specific phase of the 15-minute window. This is defined by `entry_start` and `entry_end` parameters (in ticks remaining before resolution). Entries outside this window are immediately rejected.

The window parameters are optimized by the optimizer. Too early = too much uncertainty. Too late = not enough discount.

### Filter 2: Cold Start Avoidance

Reject entries in the first N ticks of a market window. This ensures the moving average features have enough data to be meaningful. Without this, the first entries would be based on noisy, unstable features.

### Filter 3: Volatility Gate

Reject entries when the realized volatility (rolling standard deviation of recent prices) exceeds a threshold. High volatility means the outcome is still uncertain even in the late window, negating the lock-in thesis.

This is a hard gate (pass/fail), not a continuous feature in the scoring function. The design choice is intentional: volatility is a regime indicator, not a directional signal. Keeping it out of the score prevents it from masking other features.

### Filter 4: Unified Score

Compute the 7-feature linear score:

```
score = w_slope_short * slope_short
      + w_slope_long  * slope_long
      + w_spread      * spread
      + w_compression * compression
      + w_dist        * dist
      + w_R           * R
      + w_time        * progress
```

If the score is `None` (insufficient data for features), reject the entry.

### Filter 5: Dynamic Threshold

The score must exceed a time-varying threshold. The threshold interpolates linearly between `thresh_start` (at the beginning of the entry window) and `thresh_end` (at the end). This allows the strategy to become more or less selective as the window progresses.

### Filter 6: Price Cap

Reject entries when the contract price exceeds a maximum. This prevents overpaying for contracts where the discount has already been captured by other market participants.

## Single-Side Constraint

Each market window gets at most one entry. If both YES and NO trigger on the same tick, the higher-scoring entry is taken. Once an entry is made, no further entries are evaluated for that window.

The position is held to resolution -- binary markets resolve automatically, so there's no exit decision.

## The Optimizer Found Asymmetry

The optimizer searched a symmetric 30-parameter space: 15 parameters for YES entries and 15 for NO entries. Both sides had identical feature engineering, scoring functions, and filter cascades. The optimizer was free to find any combination.

It converged on a single-side dominant strategy. One side's parameters were optimized to effectively never trigger -- through some combination of very tight entry windows, high thresholds, or restrictive price caps. Which side dominates is redacted.

This is not a design choice. Attempts to force symmetry (by adding balance penalties or constraining the optimizer to trade both sides) degraded walk-forward performance. The microstructure signal is one-directional: one side offers consistent alpha through a specific pattern of timing, volatility regime, and price discount that the other side does not replicate.

Possible explanations for the asymmetry:
- **Bid-ask dynamics**: One side's order book may have systematically different liquidity profiles
- **Behavioral bias**: Market participants may systematically misprice one side
- **Structural**: The way resolution interacts with BTC spot price dynamics may create asymmetric opportunities

The strategy embraces this finding rather than forcing a preconceived notion of symmetry.
