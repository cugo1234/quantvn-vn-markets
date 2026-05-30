import numpy as np
import pandas as pd


def rolling_rank(series: pd.Series, window: int) -> pd.Series:
    def rank_last(x):
        s = pd.Series(x)
        return s.rank(pct=True).iloc[-1]
    return series.rolling(window, min_periods=window).apply(rank_last, raw=False)


def rolling_neutralize(series: pd.Series, window: int) -> pd.Series:
    return series - series.rolling(window, min_periods=window).mean()


def ampnorm(series: pd.Series, window: int = 12) -> pd.Series:
    denom = series.abs().rolling(window, min_periods=window).mean()
    denom = denom.replace(0, np.nan)
    return series / denom


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def rolling_decay(series: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()

    def decay_fn(x):
        return np.dot(x, weights)

    return series.rolling(window, min_periods=window).apply(decay_fn, raw=True)


def gen_position(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    close_col = None
    for c in ["close", "Close", "c"]:
        if c in df.columns:
            close_col = c
            break
    if close_col is None:
        raise ValueError(f"Cannot find close column. Available columns: {list(df.columns)}")

    df["close"] = pd.to_numeric(df[close_col], errors="coerce")

    # x1: base signal, ở đây dùng return 1-bar
    df["x1"] = df["close"].pct_change()

    # giống pipeline trong ảnh
    df["rank"] = rolling_rank(df["x1"], window=90)
    df["x2"] = rolling_neutralize(df["rank"], window=10)
    df["score"] = ampnorm(df["x2"], window=20)
    df["smooth"] = rolling_mean(df["score"], window=13)
    df["decayed"] = rolling_decay(df["smooth"], window=1)

    # chuyển alpha liên tục sang tín hiệu giao dịch đơn giản
    df["signal"] = 0
    df.loc[df["decayed"] > 0, "signal"] = 1
    df.loc[df["decayed"] < 0, "signal"] = -1

# flip signal
    df["signal"] = -df["signal"]

    df["position"] = df["signal"].astype(int)
    signal_counts = df["signal"].value_counts().to_dict()

    df["count_1"] = signal_counts.get(1, 0)
    df["count_0"] = signal_counts.get(0, 0)
    df["count_-1"] = signal_counts.get(-1, 0)
    return df