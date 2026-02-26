# Feature Engineering: 5 Continuous Moving Average Features

## Design Philosophy

The feature engineering serves a specific purpose: transform a raw binary market price series into a compact, continuous representation that a linear scoring function can use to identify entry points.

Key design constraints:
- **Continuous**: No binary/categorical features. The linear model needs smooth gradients for the optimizer to work efficiently.
- **Bounded**: Features are normalized as percentage deviations, preventing any single feature from dominating due to scale differences.
- **Interpretable**: Each feature has a clear physical meaning in terms of price dynamics.
- **Low-dimensional**: Only 5 features. Combined with R and time progress, the scoring function has 7 inputs total. This keeps the parameter count manageable (7 weights per side = 14 weight parameters).

## The 5 Features

### 1. slope_short -- Short-Term Momentum

```
slope_short = (price[t] - price[t - MA_SHORT]) / MA_SHORT / MA_short_value * 100
```

Measures the recent rate of price change, normalized by the short-term moving average. Positive values mean the price has been rising over the short lookback. The normalization by MA value makes the feature comparable across different price levels.

**What it captures**: Fast momentum. A sharp recent move in one direction suggests the outcome is becoming more certain.

### 2. slope_long -- Long-Term Momentum

```
slope_long = (price[t] - price[t - MA_LONG]) / MA_LONG / MA_long_value * 100
```

Same construction as slope_short but over the long lookback period. This captures the broader trend within the 15-minute window.

**What it captures**: Sustained directional pressure. A consistently trending window is more predictable than a choppy one.

### 3. spread -- MA Divergence

```
spread = (MA_short - MA_long) / price * 100
```

The difference between the short and long moving averages, expressed as a percentage of the current price. Positive spread means the short MA is above the long MA (bullish trend). Negative spread means the reverse.

**What it captures**: Trend confirmation. When short and long MAs agree on direction (large spread), the trend is strong. When they're close (small spread), the market is indecisive.

### 4. compression -- MA Convergence

```
compression = (max(MA_short, MA_mid, MA_long) - min(MA_short, MA_mid, MA_long)) / price * 100
```

The range of all three MAs, expressed as a percentage of price. Low compression means all three MAs are close together (price is consolidating). High compression means the MAs are spread out (strong trend or recent reversal).

**What it captures**: Market regime. Compression acts as a regime indicator -- low compression suggests range-bound behavior, high compression suggests directional movement. The optimizer uses this to filter for specific regimes.

### 5. dist -- Distance From Mid MA

```
dist = (price - MA_mid) / price * 100
```

How far the current price is from the mid-term moving average, as a percentage. Positive dist means the price is above the mid MA. Negative dist means below.

**What it captures**: Overextension. When price is far from its mid-term average (high |dist|), it may be overextended. The optimizer learned a negative weight for dist on the dominant side -- indicating that overextension in one direction is a signal to enter the opposite side.

## Why Three MA Periods

Three periods (short, mid, long) provide the minimum structure needed to capture:
- **Momentum** (short slope)
- **Trend** (long slope, spread)
- **Regime** (compression)
- **Mean reversion** (dist from mid)

Two periods would lose the compression feature (need three points to measure convergence). Four or more periods would add parameters without proportional information gain.

The specific periods (PARAM_PLACEHOLDER) were optimized via Optuna. The optimizer searches over integer ranges for each period, subject to the constraint that short < mid < long.

## Why Linear Model

The scoring function is a simple weighted sum:

```
score = sum(w_i * feature_i) for i in 1..7
```

No interaction terms, no nonlinearities, no feature crosses.

This is a deliberate constraint, not a limitation:

1. **Parameter efficiency**: 7 weights per side vs. 28+ for a model with pairwise interactions. With only ~600 market windows for training, the simpler model is less likely to overfit.

2. **Interpretability**: Each weight has a direct meaning. `w_slope_long = 0.28` means slope_long contributes 0.28 units of score per unit of feature value. You can reason about why the optimizer chose each weight.

3. **Optimizer efficiency**: The linear score creates a smooth objective landscape for Optuna's TPE sampler. Nonlinear models create ridges and plateaus that slow convergence.

4. **Feature interactions through the optimizer**: Although the model is linear, the features themselves interact through the optimizer. The optimizer jointly tunes all 7 weights plus the threshold trajectory and entry window, effectively learning feature interactions through the entry filter cascade.

## Vectorized Computation

For backtest efficiency, all 5 features are precomputed for every tick in a market window using cumulative sum tricks:

```python
cumsum = np.cumsum(np.insert(prices, 0, 0))
ma_s = (cumsum[ma_short:] - cumsum[:-ma_short]) / ma_short
```

This makes the moving average computation O(N) instead of O(N * max_period), which is critical when the backtest evaluates thousands of Optuna trials, each requiring a full pass through all market windows.

## Feature Statistics (Illustrative)

| Feature | Mean | Std | Min | Max | Unit |
|---------|------|-----|-----|-----|------|
| slope_short | 0.02 | 0.8 | -4.2 | 3.9 | %/tick/MA |
| slope_long | 0.01 | 0.3 | -1.8 | 1.6 | %/tick/MA |
| spread | 0.03 | 1.2 | -5.1 | 4.8 | % of price |
| compression | 0.9 | 0.7 | 0.0 | 4.5 | % of price |
| dist | 0.1 | 1.5 | -6.2 | 5.8 | % of price |

These are illustrative ranges based on the general characteristics of 15-minute binary market price dynamics. Actual values depend on the specific MA periods used.
