"""LSTM-based ranking strategy for top NSE stocks.

What this script does:
1. Downloads OHLCV data for a top-100 NSE universe.
2. Builds alpha features (including log returns) on weekly or monthly bars.
3. Trains an LSTM to predict next-period log returns.
4. Runs a walk-forward backtest with periodic re-training.
5. Produces today's buy-and-hold candidates from model predictions.

Example:
    python nse_lstm_strategy.py \
        --frequency weekly \
        --start 2016-01-01 \
        --train-years 4 \
        --top-k 10
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam

# Approximate Nifty 100 universe (NSE symbols without suffix).
NSE_TOP_100 = [
    "RELIANCE", "TCS", "HDFCBANK", "BHARTIARTL", "ICICIBANK", "SBIN", "INFY", "ITC",
    "LT", "HINDUNILVR", "KOTAKBANK", "BAJFINANCE", "AXISBANK", "ASIANPAINT", "MARUTI",
    "SUNPHARMA", "TITAN", "ULTRACEMCO", "NESTLEIND", "WIPRO", "HCLTECH", "ONGC",
    "NTPC", "POWERGRID", "TATAMOTORS", "M&M", "TATASTEEL", "TECHM", "ADANIENT",
    "ADANIPORTS", "COALINDIA", "INDUSINDBK", "BAJAJFINSV", "HINDALCO", "JSWSTEEL",
    "CIPLA", "DRREDDY", "EICHERMOT", "DIVISLAB", "GRASIM", "BRITANNIA", "HEROMOTOCO",
    "APOLLOHOSP", "SBILIFE", "BAJAJ-AUTO", "SHREECEM", "UPL", "BAJAJHLDNG", "TRENT",
    "HDFCLIFE", "ADANIGREEN", "ADANIPOWER", "BPCL", "IOC", "SIEMENS", "PIDILITIND",
    "DABUR", "HAVELLS", "GODREJCP", "AMBUJACEM", "INDIGO", "NAUKRI", "ZOMATO",
    "TATACONSUM", "DLF", "CHOLAFIN", "ICICIPRULI", "SBICARD", "MUTHOOTFIN", "POLYCAB",
    "PFC", "RECLTD", "TVSMOTOR", "BANKBARODA", "VEDL", "SAIL", "JINDALSTEL", "CANBK",
    "AUROPHARMA", "LUPIN", "TORNTPHARM", "BIOCON", "COLPAL", "MARICO", "BERGEPAINT",
    "MCDOWELL-N", "PAGEIND", "MOTHERSON", "CONCOR", "GAIL", "IDFCFIRSTB", "PNB",
    "ABB", "BOSCHLTD", "SRF", "HAL", "BEL", "IRCTC", "DMART", "HINDPETRO",
]

FEATURE_COLUMNS = [
    "log_return_1",
    "log_return_4",
    "log_return_12",
    "volatility_4",
    "volatility_12",
    "momentum_4",
    "momentum_12",
    "volume_zscore",
    "rsi_14",
    "macd",
    "macd_signal",
    "bb_pos",
    "rel_strength_vs_nifty",
]


@dataclass
class BacktestResult:
    returns: pd.Series
    picks: pd.DataFrame

    @property
    def equity_curve(self) -> pd.Series:
        return (1.0 + self.returns).cumprod()

    @property
    def total_return(self) -> float:
        return float(self.equity_curve.iloc[-1] - 1.0) if len(self.returns) else 0.0

    @property
    def annualized_return(self) -> float:
        if len(self.returns) < 2:
            return 0.0
        periods_per_year = 52 if self.returns.index.freqstr and "W" in self.returns.index.freqstr else 12
        n = len(self.returns)
        return float((1 + self.total_return) ** (periods_per_year / n) - 1)

    @property
    def sharpe(self) -> float:
        if self.returns.std() == 0 or len(self.returns) < 2:
            return 0.0
        periods_per_year = 52 if self.returns.index.freqstr and "W" in self.returns.index.freqstr else 12
        return float((self.returns.mean() / self.returns.std()) * math.sqrt(periods_per_year))


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, macd_signal


def download_data(symbols: List[str], start: str, end: str | None = None) -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        ticker = f"{sym}.NS"
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            continue
        data[sym] = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return data


def to_frequency(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    rule = "W-FRI" if frequency == "weekly" else "M"
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    out = df.resample(rule).agg(agg).dropna()
    return out


def build_features(price_df: pd.DataFrame, nifty_close: pd.Series) -> pd.DataFrame:
    df = price_df.copy()

    df["log_return_1"] = np.log(df["Close"] / df["Close"].shift(1))
    df["log_return_4"] = np.log(df["Close"] / df["Close"].shift(4))
    df["log_return_12"] = np.log(df["Close"] / df["Close"].shift(12))

    df["volatility_4"] = df["log_return_1"].rolling(4).std()
    df["volatility_12"] = df["log_return_1"].rolling(12).std()

    df["momentum_4"] = df["Close"].pct_change(4)
    df["momentum_12"] = df["Close"].pct_change(12)

    vol_mean = df["Volume"].rolling(12).mean()
    vol_std = df["Volume"].rolling(12).std().replace(0, np.nan)
    df["volume_zscore"] = (df["Volume"] - vol_mean) / vol_std

    df["rsi_14"] = _rsi(df["Close"], 14)
    df["macd"], df["macd_signal"] = _macd(df["Close"])

    mid = df["Close"].rolling(20).mean()
    sigma = df["Close"].rolling(20).std()
    upper = mid + 2 * sigma
    lower = mid - 2 * sigma
    df["bb_pos"] = (df["Close"] - lower) / (upper - lower)

    nifty_ret = np.log(nifty_close / nifty_close.shift(1)).reindex(df.index)
    df["rel_strength_vs_nifty"] = df["log_return_1"] - nifty_ret

    # Prediction target = next period log return.
    df["target"] = df["log_return_1"].shift(-1)

    return df.dropna()


def make_sequences(df: pd.DataFrame, lookback: int) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    x_list, y_list, t_list = [], [], []
    values = df[FEATURE_COLUMNS].values
    target = df["target"].values
    dates = df.index

    for i in range(lookback, len(df)):
        x_list.append(values[i - lookback : i])
        y_list.append(target[i])
        t_list.append(dates[i])

    return np.array(x_list), np.array(y_list), t_list


def build_model(input_shape: Tuple[int, int]) -> Sequential:
    model = Sequential(
        [
            LSTM(64, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    return model


def prepare_panel(
    raw_data: Dict[str, pd.DataFrame], frequency: str, start: str, end: str | None
) -> Dict[str, pd.DataFrame]:
    nifty = yf.download("^NSEI", start=start, end=end, auto_adjust=True, progress=False)
    nifty = to_frequency(nifty[["Open", "High", "Low", "Close", "Volume"]], frequency)

    panel: Dict[str, pd.DataFrame] = {}
    for sym, df in raw_data.items():
        fr = to_frequency(df, frequency)
        feat = build_features(fr, nifty["Close"])
        if len(feat) > 80:
            panel[sym] = feat
    return panel


def train_and_predict(
    panel: Dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    lookback: int,
    epochs: int,
    min_train_points: int,
) -> pd.Series:
    x_train, y_train = [], []
    latest_windows = {}

    for sym, df in panel.items():
        df_cut = df[df.index <= as_of]
        if len(df_cut) <= lookback + min_train_points:
            continue

        scaler = StandardScaler()
        df_scaled = df_cut.copy()
        df_scaled[FEATURE_COLUMNS] = scaler.fit_transform(df_cut[FEATURE_COLUMNS])

        x, y, _ = make_sequences(df_scaled, lookback)
        if len(x) < min_train_points:
            continue

        x_train.append(x)
        y_train.append(y)
        latest_windows[sym] = x[-1]

    if not x_train:
        return pd.Series(dtype=float)

    X = np.concatenate(x_train, axis=0)
    y = np.concatenate(y_train, axis=0)

    model = build_model((lookback, len(FEATURE_COLUMNS)))
    cb = [EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)]
    model.fit(X, y, epochs=epochs, batch_size=64, verbose=0, callbacks=cb)

    preds = {sym: float(model.predict(win[None, :, :], verbose=0).squeeze()) for sym, win in latest_windows.items()}
    return pd.Series(preds).sort_values(ascending=False)


def run_backtest(
    panel: Dict[str, pd.DataFrame],
    lookback: int,
    epochs: int,
    top_k: int,
    train_years: int,
    min_train_points: int,
) -> BacktestResult:
    all_dates = sorted(set().union(*[set(df.index) for df in panel.values()]))
    if not all_dates:
        return BacktestResult(pd.Series(dtype=float), pd.DataFrame())

    start_date = min(all_dates) + pd.DateOffset(years=train_years)
    rebalance_dates = [d for d in all_dates if d >= start_date]

    rets = []
    picks_log = []

    for dt in rebalance_dates[:-1]:
        preds = train_and_predict(panel, dt, lookback, epochs, min_train_points)
        if preds.empty:
            continue

        selected = preds.head(top_k)
        next_dt = next((d for d in rebalance_dates if d > dt), None)
        if next_dt is None:
            break

        stock_rets = []
        for sym in selected.index:
            df = panel[sym]
            row = df[df.index == dt]
            nxt = df[df.index == next_dt]
            if row.empty or nxt.empty:
                continue
            period_return = float(nxt["Close"].iloc[0] / row["Close"].iloc[0] - 1)
            stock_rets.append(period_return)

        if stock_rets:
            rets.append((next_dt, np.mean(stock_rets)))
            for sym, score in selected.items():
                picks_log.append({"rebalance_date": dt, "symbol": sym, "predicted_log_return": score})

    ret_series = pd.Series({d: r for d, r in rets}).sort_index()
    if not ret_series.empty:
        inferred = pd.infer_freq(ret_series.index)
        if inferred:
            ret_series.index = pd.DatetimeIndex(ret_series.index, freq=inferred)

    return BacktestResult(returns=ret_series, picks=pd.DataFrame(picks_log))


def print_report(result: BacktestResult) -> None:
    print("\n=== Backtest Summary ===")
    print(f"Periods: {len(result.returns)}")
    print(f"Total Return: {result.total_return:.2%}")
    print(f"Annualized Return: {result.annualized_return:.2%}")
    print(f"Sharpe (naive): {result.sharpe:.2f}")
    if not result.returns.empty:
        mdd = (result.equity_curve / result.equity_curve.cummax() - 1).min()
        print(f"Max Drawdown: {mdd:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NSE LSTM alpha ranking strategy")
    parser.add_argument("--frequency", choices=["weekly", "monthly"], default="weekly")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--lookback", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--train-years", type=int, default=4)
    parser.add_argument("--min-train-points", type=int, default=80)
    args = parser.parse_args()

    print("Downloading NSE universe...")
    raw = download_data(NSE_TOP_100, start=args.start, end=args.end)
    if not raw:
        raise SystemExit("No price data downloaded. Check internet or symbols.")

    print("Engineering features and building panel...")
    panel = prepare_panel(raw, args.frequency, args.start, args.end)
    if not panel:
        raise SystemExit("No usable panel after feature engineering.")

    print("Running walk-forward backtest...")
    result = run_backtest(
        panel=panel,
        lookback=args.lookback,
        epochs=args.epochs,
        top_k=args.top_k,
        train_years=args.train_years,
        min_train_points=args.min_train_points,
    )
    print_report(result)

    latest_date = max(max(df.index) for df in panel.values())
    print(f"\nGenerating recommendations as of {latest_date.date()} ...")
    recs = train_and_predict(
        panel=panel,
        as_of=latest_date,
        lookback=args.lookback,
        epochs=args.epochs,
        min_train_points=args.min_train_points,
    )

    if recs.empty:
        print("No recommendations generated.")
        return

    picks = recs.head(args.top_k)
    print("\n=== BUY & HOLD Candidates (Top predicted next-period log-return) ===")
    for i, (sym, score) in enumerate(picks.items(), start=1):
        print(f"{i:>2}. {sym:12s} predicted_log_return={score:+.4f}")

    result.returns.to_csv("backtest_returns.csv", index_label="date")
    result.picks.to_csv("backtest_picks.csv", index=False)
    picks.to_csv("today_recommendations.csv", header=["predicted_log_return"])
    print("\nSaved: backtest_returns.csv, backtest_picks.csv, today_recommendations.csv")


if __name__ == "__main__":
    main()
