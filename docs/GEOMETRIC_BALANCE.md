# Geometric Balance Objective: Solving the Degenerate Optimization Problem

## The Core Problem

When optimizing a trading strategy, you need an objective function that tells the optimizer what "good" looks like. This seems straightforward -- just maximize profit, right? In practice, naive objectives produce degenerate solutions that look great on paper but fail in production.

### Degenerate Solution 1: Maximize Total Return

```
objective = sum(log_returns)
```

The optimizer discovers that taking more trades increases total return (as long as the average trade is slightly positive). It loosens all filters: wide entry windows, low thresholds, no volatility gates. The resulting strategy trades on 80% of windows with a 55% win rate. Total return is high because of sheer volume, but any deterioration in edge causes catastrophic drawdown. The strategy has no selectivity.

### Degenerate Solution 2: Maximize Per-Trade Return

```
objective = sum(log_returns) / n_trades
```

Now the optimizer discovers that being extremely selective maximizes per-trade return. It tightens all filters: narrow entry windows, high thresholds, strict volatility gates. The strategy trades on 3% of windows with a 95% win rate. Each trade is excellent, but you take 15 trades per month. The variance is enormous -- a single losing trade wipes out weeks of gains. The strategy is too rare to be practical.

### Degenerate Solution 3: Sharpe Ratio

```
objective = mean(returns) / std(returns)
```

Sharpe ratio seems like it should balance frequency and quality, but in practice it's dominated by the denominator. The optimizer can inflate Sharpe by taking many similar, small, highly correlated trades. It can also game it by avoiding trades during volatile periods (when it should be trading with reduced size, not abstaining).

## The Geometric Balance Solution

The geometric balance objective requires BOTH frequency AND quality by construction:

```
G_window = total_log_return / n_windows    (per-window growth)
G_trade  = total_log_return / n_trades     (per-trade growth)

G = sqrt(G_window * G_trade)               (geometric mean)
```

### Why These Two Metrics

**G_window** is the total log return divided by the number of available market windows (not just the ones traded). This metric increases when the strategy trades more frequently -- more trades contribute more log return to the numerator while the denominator stays fixed. A strategy that trades 50% of windows has twice the opportunity to accumulate G_window compared to one that trades 25% of windows.

**G_trade** is the total log return divided by the number of trades taken. This metric increases when each trade is better quality. A strategy with 80% win rate on high-R entries has higher G_trade than one with 60% win rate on mediocre entries. Taking more trades with low quality dilutes G_trade.

### Why Geometric Mean

The geometric mean has a critical property: **if either component is zero, the result is zero.** This makes it impossible to game:

- All-in on frequency (G_window high, G_trade low): G = sqrt(high * low) = moderate
- All-in on quality (G_window low, G_trade high): G = sqrt(low * high) = moderate
- Balanced (both moderate-high): G = sqrt(mod * mod) = moderate-high

The optimizer cannot make G large without making BOTH components large simultaneously. This is exactly the tradeoff we want: enough trades to reduce variance, each trade good enough to sustain growth.

### Special Cases

When both metrics are positive (the normal case), the standard geometric mean applies. When both are negative (a losing strategy), we preserve the sign: `G = -sqrt(|G_window| * |G_trade|)`. When the signs are mixed (one positive, one negative -- possible but rare), we take the minimum (conservative estimate).

### Connection to Kelly Criterion

The use of log returns is not arbitrary. Under Kelly criterion, the growth rate of a bankroll is maximized by sizing positions to maximize expected log return. By using log returns in both G_window and G_trade, the geometric balance objective is directly optimizing for long-term growth rate.

Specifically:
- `G_window` is the compounding rate: how fast does the bankroll grow per available opportunity?
- `G_trade` is the edge quality: how much growth does each trade contribute?
- `G = sqrt(G_window * G_trade)` is the growth-optimal balance between trading more (capturing more opportunities) and trading better (maintaining edge per trade).

## Robust Scoring: Worst-Case Stability

The geometric balance G is computed for each cross-validation fold independently. This gives us a vector of fold scores: `[G_1, G_2, G_3, G_4, G_5]`. The robust score combines these:

```
robust_score = min(fold_Gs) - 0.5 * std(fold_Gs)
```

### Why min() Instead of mean()

The `min()` focuses the optimizer on worst-case performance. A parameter set that scores well on 4 folds but fails on 1 fold gets a low robust score, even if its average is high. This prevents the optimizer from finding parameters that are tuned to specific market conditions -- it must find parameters that work consistently.

Example:

```
Set A: [0.05, 0.04, 0.05, 0.04, 0.05]  -> min=0.04, std=0.005, robust=0.038
Set B: [0.10, 0.08, 0.02, -0.01, 0.07] -> min=-0.01, std=0.042, robust=-0.031
```

Set A has a lower average (0.046 vs 0.052) but a much higher robust score because it's consistent. Set B has a failing fold that the `min()` captures, and the standard deviation penalty further reduces its score.

### Why 0.5 * std()

The coefficient 0.5 controls the tradeoff between worst-case focus and variance penalty. At 0.0, the objective is purely min(folds) -- too conservative, ignoring that high variance is bad even when the minimum is OK. At 1.0, the variance penalty dominates -- too aggressive, penalizing strategies with one great fold and four good folds. The 0.5 value was chosen to balance these considerations.

## Walk-Forward Cross-Validation

### 5-Fold Expanding Windows

The market windows are ordered chronologically. The CV scheme creates 5 validation folds with expanding training sets:

```
|----train----|--val 1--|
|--------train---------|--val 2--|
|------------train-------------|--val 3--|
|----------------train----------------|--val 4--|
|--------------------train--------------------|--val 5--|
```

Each fold's training set is all windows before the validation set. This is critical for time series: we never train on future data. The expanding (not sliding) window means later folds have more training data, which models the realistic scenario of accumulating more history over time.

### 20% Holdout

The last 20% of all windows is reserved as a holdout set. It is never seen during optimization. After the optimizer converges, the best parameters are evaluated once on the holdout set to get an unbiased performance estimate. This is the number reported in the backtest results.

## Parameter Space Design

### 30 Parameters (15 per side)

Per side (YES and NO):
- 7 score weights (slope_short, slope_long, spread, compression, dist, R, time)
- 2 entry window bounds (entry_start, entry_end)
- 1 price cap
- 1 volatility threshold
- 2 threshold trajectory (thresh_start, thresh_end)
- 1 score scale (position sizing sensitivity)
- 1 reserved

Total: 15 x 2 = 30 parameters.

### 1,500 Startup Trials

Optuna's TPE (Tree-structured Parzen Estimator) sampler begins with random exploration before building its probabilistic model. The rule of thumb is 50x the parameter count for startup trials: 30 x 50 = 1,500 random trials before TPE kicks in.

This ensures adequate coverage of the 30-dimensional parameter space. Without sufficient startup, TPE can lock onto a local optimum in a poorly-explored region.

### Why Optuna Over Grid/Random/Bayesian

- **vs. Grid search**: 30 parameters with even 5 values each = 5^30 ~ 10^21 combinations. Not feasible.
- **vs. Pure random**: Works for startup but doesn't exploit structure. TPE is more efficient once it has enough data.
- **vs. Gaussian process Bayesian**: GP-based methods (like scikit-optimize) scale as O(n^3) with the number of trials. At 1,500+ trials, this becomes computationally prohibitive. TPE scales linearly.

## Why This Beats Sharpe as an Objective

| Aspect | Sharpe Ratio | Geometric Balance |
|--------|-------------|-------------------|
| Frequency incentive | Indirect (through mean) | Direct (G_window) |
| Quality incentive | Indirect (through std) | Direct (G_trade) |
| Can be gamed by | Many correlated trades | Neither component alone |
| Log return consistency | No (uses arithmetic returns) | Yes (native log returns) |
| Kelly-consistent | No | Yes |
| Worst-case focus | No | Yes (min + std penalty) |
| Handles rare strategies | Poorly (high variance) | Well (G_trade stays high) |

The fundamental advantage is that geometric balance explicitly decomposes the optimization target into two orthogonal components (frequency and quality) and requires both to be high. Sharpe ratio conflates these into a single ratio that can be satisfied by degenerate solutions.
