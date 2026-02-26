# Backtest Results: Walk-Forward Validation

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total trades | 257 |
| Win rate | 77% |
| Dominant side | Single side, >95% of entries (side redacted) |
| CV folds | 5 |
| Holdout size | 20% |
| Parameters optimized | 30 |
| Optimization trials | 500 |
| Startup trials (random) | 1,500 (pre-TPE exploration) |

## Cross-Validation Fold Performance

| Fold | Trades | Win Rate | G_window | G_trade | Geometric Balance |
|------|--------|----------|----------|---------|-------------------|
| 1 | 52 | 78% | 0.0032 | 0.0053 | 0.0041 |
| 2 | 48 | 75% | 0.0028 | 0.0044 | 0.0035 |
| 3 | 54 | 79% | 0.0035 | 0.0056 | 0.0044 |
| 4 | 50 | 76% | 0.0030 | 0.0046 | 0.0037 |
| 5 | 53 | 77% | 0.0031 | 0.0049 | 0.0039 |

**Robust score**: min(fold_Gs) - 0.5 * std(fold_Gs) = 0.0035 - 0.5 * 0.0003 = 0.0034

The fold stability is the key result. Win rates range from 75% to 79% across folds, and geometric balance scores range from 0.0035 to 0.0044. No fold is dramatically worse than the others, confirming that the parameters generalize across the time period.

## Side Distribution

The optimizer converged on a single-side dominant strategy:

| Side | Trades | Win Rate |
|------|--------|----------|
| Dominant side | 248 | 77.4% |
| Non-dominant side | 9 | 66.7% |

The 9 non-dominant trades that slipped through represent edge cases where the score slightly exceeded the effectively-disabled threshold. In a production implementation, the non-dominant side could be explicitly disabled to simplify the strategy.

## Entry Characteristics

### Timing

Entries cluster in a specific late-window phase. The exact tick range is PARAM_PLACEHOLDER, but the pattern is consistent: entries occur when there is enough time for the contract to offer a discount but little enough time that the outcome is near-certain.

### Price

Dominant-side entry prices cluster in a specific range. The price cap (PARAM_PLACEHOLDER) prevents overpaying for contracts where the discount has already been captured.

### Volatility

Entries are filtered to low-volatility regimes where the lock-in thesis holds. The volatility gate (PARAM_PLACEHOLDER) rejects windows where BTC is still oscillating too much for the outcome to be predictable.

## Growth Accounting

Using Kelly-consistent log returns:

```
Total log return: sum of 257 individual trade log returns
Per-window growth (G_window): total_log_return / n_total_windows
Per-trade growth (G_trade): total_log_return / 257

Terminal growth multiple: exp(total_log_return) ~ 3.5x
```

The terminal growth multiple represents the hypothetical compounded growth if the strategy had been running with consistent position sizing from the start of the walk-forward period to the end.

## Honest Limitations

### In-Sample Optimization Caveat

The 77% win rate and fold stability are from the optimization period (tuning set + holdout). While the walk-forward CV methodology prevents look-ahead bias within the dataset, the overall result is still conditioned on the specific market regime during the backtest period.

Out-of-sample performance (on truly future data) may differ because:
- Market microstructure can change (new market makers, different liquidity)
- BTC volatility regimes shift over time
- Polymarket's matching engine and fee structure may change
- Competition from other algorithmic traders can erode edge

### Execution Assumptions

The backtest assumes:
- **Fill-or-kill at quoted price + slippage**: Real fills may be worse, especially for larger sizes
- **No market impact**: The strategy's orders don't move the market. This is reasonable for small position sizes but may not hold at scale
- **Constant slippage**: In practice, slippage varies with order book depth and market conditions
- **No latency**: Real execution has network latency that can cause missed entries or worse fills

### Parameter Count Concern

30 parameters on ~600 market windows is a parameter-to-sample ratio of 1:20. While the robust scoring and expanding-window CV mitigate overfitting risk, this ratio is on the aggressive side. The linear model (no interaction terms) and the geometric balance objective (which penalizes degenerate solutions) provide additional regularization, but the risk of some overfitting remains.

### Market Regime Dependence

The strategy's edge depends on specific market microstructure conditions:
- Sufficient liquidity in the dominant side's order book
- Predictable volatility patterns within 15-minute windows
- Consistent relationship between BTC spot moves and binary contract prices

A regime change in any of these could degrade performance. The health overlay provides some protection by reducing position size during drawdowns, but it cannot prevent losses from a sustained regime shift.

### Single-Market Concentration

All trades are on a single side of BTC binary markets. This is concentrated risk -- if the dominant-side edge disappears (due to market microstructure changes, increased competition, or platform changes), the entire strategy stops working. Diversification across market types or assets would reduce this concentration risk but is outside the scope of this project.
