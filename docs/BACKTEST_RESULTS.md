# Backtest Results: Walk-Forward Validation

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total trades (full set) | 544 |
| Win rate (full set) | 72.4% |
| R (win/loss ratio) | 0.90 |
| Holdout trades | 129 |
| Holdout win rate | 71.3% |
| Holdout R | 0.80 |
| Dominant side | Single side, >95% of entries (side redacted) |
| CV folds | 5 |
| Holdout size | 20% |
| Parameters optimized | 30 |
| Execution | Fill-and-kill (FAK) orders |
| Slippage assumption | $0.10 per contract |

## Holdout Validation

The holdout set (last 20% of market windows, never seen during optimization) provides the most honest performance estimate:

| Metric | CV (in-sample) | Holdout (out-of-sample) |
|--------|---------------|------------------------|
| Trades | 544 | 129 |
| Win rate | 72.4% | 71.3% |
| R (W/L ratio) | 0.90 | 0.80 |
| G (per-window growth) | -- | 0.000374 |

The ~1% WR degradation from CV to holdout is modest and expected -- it represents the gap between optimized and truly out-of-sample performance. The R degradation (0.90 to 0.80) is more significant and reflects that later market windows offered slightly less favorable payoff geometry, likely due to increased competition or changing microstructure.

## Side Distribution

The optimizer converged on a single-side dominant strategy. The non-dominant side's parameters were effectively disabled (>95% of trades on the dominant side). In production, the non-dominant side could be explicitly disabled to simplify execution.

## Execution Design

### Fill-and-Kill (FAK) Orders

The strategy uses FAK orders rather than GTC (Good-Til-Cancelled). FAK orders either fill immediately at the specified price or are cancelled -- there's no resting order on the book. This is deliberate:

- **No information leakage**: A resting GTC order signals intent to the market. In thin late-window order books, this would move the price against the strategy before the fill.
- **Price certainty**: FAK guarantees that if filled, the fill price equals the specified price. No partial fills at worse prices.
- **Latency tolerance**: The strategy evaluates entry conditions on each tick. If the FAK doesn't fill, the next tick re-evaluates -- the opportunity may still be there at a slightly different price.

### Slippage Assumption

The backtest uses a $0.10 slippage assumption per contract. This is added to the entry price before computing returns:

```
effective_entry = quoted_price + $0.10
```

The $0.10 assumption is conservative for typical late-window order book depth. It accounts for:
- Bid-ask spread crossing
- Partial depth at the top of book
- Execution latency between signal and fill

The slippage was chosen based on empirical order book analysis, not optimized -- it's a fixed assumption that the optimizer treats as given.

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
Total log return: sum of 544 individual trade log returns
Per-window growth (G_window): total_log_return / n_total_windows
Holdout G_window: 0.000374 per window
```

The holdout G_window of 0.000374 represents the per-window growth rate on truly out-of-sample data. This is the compounding rate that drives long-term bankroll growth under the Kelly framework. G_window is the production objective -- it rewards both trade frequency (more trades contribute more log return to the numerator) and trade quality (bad trades reduce the numerator).

## Honest Limitations

### In-Sample Optimization Caveat

The 72.4% win rate is from the full optimization set. The holdout (71.3% WR, 129 trades) provides a more honest estimate. While the walk-forward CV methodology prevents look-ahead bias within the dataset, the overall result is still conditioned on the specific market regime during the backtest period.

Out-of-sample performance (on truly future data) may differ because:
- Market microstructure can change (new market makers, different liquidity)
- BTC volatility regimes shift over time
- Polymarket's matching engine and fee structure may change
- Competition from other algorithmic traders can erode edge

### Execution Assumptions

The backtest assumes:
- **FAK at quoted price + $0.10 slippage**: The $0.10 assumption is conservative for typical late-window depth. Real FAK fills match exactly or don't fill.
- **No market impact**: The strategy's orders don't move the market. This is reasonable for small position sizes but may not hold at scale.
- **Constant slippage**: In practice, slippage varies with order book depth and market conditions. The fixed $0.10 is a simplification.
- **No latency**: Real execution has network latency that can cause missed entries or worse fills.

### Parameter Count Concern

30 parameters on ~5,000 market windows is a parameter-to-sample ratio of ~1:167. The robust scoring (min - 0.5*std across folds) and expanding-window CV mitigate overfitting risk. The linear model (no interaction terms) provides additional regularization. The parameter-to-sample ratio is comfortable, but the risk of regime-specific overfitting remains -- the strategy may be tuned to microstructure conditions that evolve over time.

### Market Regime Dependence

The strategy's edge depends on specific market microstructure conditions:
- Sufficient liquidity in the dominant side's order book
- Predictable volatility patterns within 15-minute windows
- Consistent relationship between BTC spot moves and binary contract prices

A regime change in any of these could degrade performance. The health overlay provides some protection by reducing position size during drawdowns, but it cannot prevent losses from a sustained regime shift.

### Single-Market Concentration

All trades are on a single side of BTC binary markets. This is concentrated risk -- if the dominant-side edge disappears (due to market microstructure changes, increased competition, or platform changes), the entire strategy stops working. Diversification across market types or assets would reduce this concentration risk but is outside the scope of this project.
