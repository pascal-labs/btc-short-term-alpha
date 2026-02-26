# Optimization Objective: Solving the Degenerate Optimization Problem

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

## Development: From Geometric Balance to Per-Window Growth

The initial design computed two complementary metrics and took their geometric mean:

```
G_window = total_log_return / n_windows    (per-window growth -- rewards frequency)
G_trade  = total_log_return / n_trades     (per-trade growth -- rewards quality)

G = sqrt(G_window * G_trade)               (geometric mean)
```

The geometric mean has an appealing property: if either component is zero, the result is zero. You can't game it by maximizing one at the expense of the other. In theory, this forces the optimizer to find parameter sets that trade frequently enough AND with sufficient quality.

**In practice, G_trade dominated.** Because n_trades is always <= n_windows, G_trade >= G_window for any positive strategy. The geometric mean was pulled toward G_trade's behavior, and the optimizer would sacrifice frequency to inflate per-trade quality -- a subtler version of Degenerate Solution 2. The geometric balance shifted the degeneracy threshold rather than eliminating it.

## The Production Objective: G_window + Robust Scoring

The production objective uses G_window alone:

```
G_window = total_log_return / n_windows
```

G_window rewards both frequency and quality through a single metric. More trades accumulate more log return in the numerator (frequency). Bad trades reduce the numerator (quality). The denominator is fixed -- total available windows -- so the optimizer can't game it by restricting the sample.

The degenerate solutions are prevented by the robust scoring layer (below) rather than by balancing two G components. This is simpler and empirically more effective.

### Connection to Kelly Criterion

The use of log returns is not arbitrary. Under Kelly criterion, the growth rate of a bankroll is maximized by sizing positions to maximize expected log return. G_window is directly the compounding rate: how fast does the bankroll grow per available opportunity? This makes the objective Kelly-consistent by construction.

## Robust Scoring: Worst-Case Stability

G_window is computed for each cross-validation fold independently. This gives us a vector of fold scores: `[G_1, G_2, G_3, G_4, G_5]`. The robust score combines these:

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

| Aspect | Sharpe Ratio | G_window + Robust Scoring |
|--------|-------------|---------------------------|
| Frequency incentive | Indirect (through mean) | Direct (G_window numerator) |
| Quality incentive | Indirect (through std) | Direct (bad trades reduce G_window) |
| Can be gamed by | Many correlated trades | Neither -- robust scoring penalizes instability |
| Log return consistency | No (uses arithmetic returns) | Yes (native log returns) |
| Kelly-consistent | No | Yes |
| Worst-case focus | No | Yes (min + std penalty across folds) |
| Overfitting resistance | Low | High (worst-fold focus + variance penalty) |

The fundamental advantage is that G_window with robust scoring optimizes for consistent, growth-rate-maximizing behavior across all market regimes in the training set. Sharpe ratio conflates frequency and quality into a single ratio that can be satisfied by degenerate solutions.
