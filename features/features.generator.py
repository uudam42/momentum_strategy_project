import os
import pandas as pd

# where our raw CSVs live:
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw_data")
# where we’ll save our features CSVs:
OUT_DIR = os.path.join(os.path.dirname(__file__), "features_data")

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute the classic 14-period RSI using Pandas methods,
    which preserves the original index so assignment works.
    """
    # daily change
    delta = close.diff()

    # separate gains and losses
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # rolling average of gains and losses
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # relative strength
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_ema(close: pd.Series, span: int = 20) -> pd.Series:
    """Compute exponential moving average."""
    return close.ewm(span=span, adjust=False).mean()

def compute_macd(close: pd.Series) -> pd.Series:
    """Compute MACD line (12-EMA minus 26-EMA)."""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    return ema12 - ema26

def generate_all_features():
    # sanity checks and setup
    if not os.path.isdir(RAW_DIR):
        raise FileNotFoundError(f"No raw_data folder at {RAW_DIR}")
    os.makedirs(OUT_DIR, exist_ok=True)

    for fn in os.listdir(RAW_DIR):
        if not fn.endswith(".csv"):
            continue

        symbol  = fn.replace(".csv", "")
        raw_path = os.path.join(RAW_DIR, fn)
        out_path = os.path.join(OUT_DIR, f"{symbol}_features.csv")

        # load + sort
        df = pd.read_csv(raw_path, parse_dates=True, index_col=0)
        df.sort_index(inplace=True)

        # compute indicators
        df["RSI"]   = compute_rsi(df["Close"])
        df["EMA20"] = compute_ema(df["Close"], span=20)
        df["MACD"]  = compute_macd(df["Close"])

        # drop the NaN head (first ~26 rows)
        df.dropna(inplace=True)

        # write out
        df.to_csv(out_path, columns=[
            "Open","High","Low","Close","Volume",
            "RSI","EMA20","MACD"
        ])
        print(f"✔ {symbol}_features.csv generated ({len(df)} rows)")

if __name__ == "__main__":
    generate_all_features()