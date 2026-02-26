# btc-short-term-alpha

Late-window entry strategy for 15-minute BTC binary markets with geometric balance optimization.

Binary BTC markets on Polymarket resolve every 15 minutes. Price evolves stochastically within each window -- once BTC has moved far enough in one direction, the binary outcome is near-certain but the contract still offers a discount because resolution hasn't occurred yet. A 7-feature linear scoring function identifies entry points in the late window. The optimizer selected a single dominant side -- the signal was asymmetric, not symmetric. Optimized via a novel geometric balance objective that prevents degenerate solutions, validated on 257 walk-forward trades at 77% win rate.

## Architecture

```
    BTC Spot Price (15-min binary markets)
         |
         v
    +------------------------------------+
    |   Feature Engineering (5 MA)       |
    |   slope_short, slope_long, spread, |
    |   compression, dist                |
    +----------------+-------------------+
                     |
                     v
    +------------------------------------+
    |   Unified Score (7 features)       |
    |   score = sum w_i * feature_i      |
    |   + w_R * R + w_time * progress    |
    +----------------+-------------------+
                     |
                     v
    +------------------------------------+
    |   Entry Filters                    |
    |   - Time window gate              |
    |   - Volatility gate               |
    |   - Score > dynamic threshold      |
    |   - Price cap                      |
    |   - Single-side constraint         |
    +----------------+-------------------+
                     |
                     v
    +------------------------------------+
    |   Position Sizing                  |
    |   Kelly x score_excess multiplier  |
    |   Health overlay (rolling WR)      |
    +----------------+-------------------+
                     |
                     v
    +------------------------------------+
    |   Hold -> Resolution               |
    |   Binary market resolves at T=15m  |
    +------------------------------------+
```

## Key Results

| Metric | Value |
|--------|-------|
| Walk-forward trades | 257 |
| Win rate | 77% |
| Dominant side | Single side (optimizer-selected, redacted) |
| CV folds | 5 (expanding window) |
| Parameters | 30 (15 YES + 15 NO) |
| Optimization | Geometric balance objective |

## Figures

| | |
|---|---|
| ![Equity Curve](figures/equity_curve.png) | ![Feature Importance](figures/feature_importance.png) |
| ![Optimization Convergence](figures/optimization_convergence.png) | ![Score Distribution](figures/score_distribution.png) |
| ![CV Fold Stability](figures/cv_fold_stability.png) | ![Entry Timing](figures/entry_timing.png) |

## The Geometric Balance Innovation

The core problem in trading strategy optimization is balancing frequency with quality. Naive objectives produce degenerate solutions: maximizing total return leads to trading everything at any quality (low win rate, high drawdown), while maximizing per-trade return leads to trading rarely at extreme quality (too few trades, high variance). Even Sharpe ratio is dominated by trade frequency in the denominator.

The geometric balance objective solves this by computing two complementary metrics and taking their geometric mean. `G_window = total_log_return / n_windows` rewards frequency (more trades means more growth per window), while `G_trade = total_log_return / n_trades` rewards quality (better trades means more growth per trade). The objective `G = sqrt(G_window * G_trade)` has a key property: if either component is zero, G is zero. You cannot game it by maximizing one at the expense of the other.

Robust scoring adds cross-validation stability on top: `robust_score = min(fold_Gs) - 0.5 * std(fold_Gs)`. This focuses the optimizer on worst-case fold performance and penalizes parameter sets that are unstable across folds. A parameter set with consistent but moderate performance across all 5 folds scores higher than one with high average but a single failing fold. The connection to Kelly criterion is direct -- log returns are the natural unit for growth rate optimization, and the geometric balance ensures the optimizer finds the growth-maximizing frequency-quality tradeoff.

## The Optimizer Found Asymmetry

The optimization ran over a 30-parameter space (15 parameters per side: 7 score weights, entry window bounds, volatility gate, price cap, threshold trajectory, and score scale). The optimizer converged on a solution where one side's parameters effectively never trigger -- tight entry windows, high thresholds, or restrictive price caps that disable entries entirely, letting the other side carry the strategy. Which side dominates is redacted.

This is an important empirical result, not a design choice. The strategy framework is symmetric -- it evaluates both sides at every tick and takes the higher-scoring entry. But the market microstructure signal turned out to be one-directional. Attempting to force the optimizer to trade both sides (by adding a balance penalty) degraded performance, confirming that the asymmetry is real and not an optimization artifact.

## Skills Demonstrated

- **Feature engineering from market microstructure** -- 5 continuous MA features extracted from binary market price series, normalized as percentage deviations for cross-market comparability
- **Linear factor model design** -- 7-feature scoring function with interpretable weights, intentionally avoiding interaction terms to prevent overfitting on a 30-parameter space
- **Novel optimization objective** -- Geometric balance (G = sqrt(G_window * G_trade)) that prevents degenerate frequency/quality tradeoffs, with robust scoring for worst-case fold stability
- **Walk-forward cross-validation** -- 5-fold expanding windows with 20% holdout, 1,500 startup trials for adequate parameter space exploration before TPE kicks in
- **Position sizing theory** -- Kelly-consistent log returns, score-based dynamic sizing, three-layer multiplicative system (z-score, slope, R-ratio) simplified to single-layer for production
- **Overfitting prevention** -- Parameter count management (30 params with 1,500 startup), robust scoring (min - 0.5*std across folds), expanding-window CV preventing look-ahead bias
- **Risk management** -- Health overlay with hysteresis (immediate step-down, cooldown step-up), volatility gates, price caps, cold-start avoidance

## Code Structure

```
strategy/
  features.py         -- 5 continuous MA features from price microstructure
  unified_score.py    -- Linear factor model: score = sum w_i * feature_i
  lock_in.py          -- Entry logic with filter cascade

optimization/
  geometric_balance.py -- The key showcase: novel objective function
  backtest_engine.py   -- Walk-forward backtest framework

risk/
  position_sizing.py  -- Kelly + score-based dynamic sizing
  risk_overlay.py     -- Rolling WR health monitor with hysteresis

scripts/
  generate_plots.py   -- Reproducible figure generation (all data embedded)
```

## Documentation

| Document | Description |
|----------|-------------|
| [STRATEGY.md](docs/STRATEGY.md) | Lock-in thesis and entry logic |
| [FEATURE_ENGINEERING.md](docs/FEATURE_ENGINEERING.md) | The 5 MA features |
| [GEOMETRIC_BALANCE.md](docs/GEOMETRIC_BALANCE.md) | Novel optimization objective |
| [BACKTEST_RESULTS.md](docs/BACKTEST_RESULTS.md) | Walk-forward results |
| [POSITION_SIZING.md](docs/POSITION_SIZING.md) | Kelly and score-based sizing |

## Note About Parameters

All live parameters have been replaced with `PARAM_PLACEHOLDER` or `None`. This includes:
- All 30 unified score weights (7 per side x 2 sides + thresholds + sizing)
- MA periods, entry windows, price caps, volatility thresholds
- Kelly fraction, slippage assumptions, score scale values
- Optuna search ranges (structural descriptions only, no exact bounds)

The code is fully functional structurally -- supply your own parameters via Optuna optimization on your own data.

## Related Projects

- [polymarket-sdk](https://github.com/pascal-labs/polymarket-sdk) -- Python SDK for Polymarket CLOB API
- [pulsefeed](https://github.com/pascal-labs/pulsefeed) -- Multi-exchange crypto data aggregation
- [tweet-volume-ensemble](https://github.com/pascal-labs/tweet-volume-ensemble) -- 6-model probabilistic ensemble for tweet volume forecasting

## License

MIT License. See [LICENSE](LICENSE).
