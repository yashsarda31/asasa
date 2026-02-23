# asasa

LSTM-based swing/position strategy code for **top NSE stocks** is included in:

- `nse_lstm_strategy.py`

## Run

```bash
python nse_lstm_strategy.py --frequency weekly --start 2016-01-01 --top-k 10
```

You can switch to monthly bars:

```bash
python nse_lstm_strategy.py --frequency monthly --start 2012-01-01 --top-k 10
```

## Outputs

The script writes:

- `backtest_returns.csv`
- `backtest_picks.csv`
- `today_recommendations.csv`

Model uses **log-return features** plus momentum/volatility/volume and technical factors, then performs a walk-forward backtest with periodic re-training.
