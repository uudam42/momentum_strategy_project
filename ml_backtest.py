# ml_backtest.py

import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

def load_model():
    """Load the saved RandomForest model from disk."""
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "features", "model", "random_forest_model.pkl")
    return joblib.load(model_path)

def load_all_features():
    """
    Read every *_features.csv file under features/features_data
    and return a dict mapping stock symbol to its DataFrame.
    """
    base_dir  = os.path.dirname(os.path.abspath(__file__))
    feat_dir  = os.path.join(base_dir, "features", "features_data")
    data_dict = {}
    for fname in os.listdir(feat_dir):
        if not fname.endswith("_features.csv"):
            continue
        symbol = fname.replace("_features.csv", "")
        path   = os.path.join(feat_dir, fname)
        df     = pd.read_csv(path, index_col=0, parse_dates=True)
        df.sort_index(inplace=True)
        data_dict[symbol] = df
    return data_dict

def backtest():
    # load the trained model and all feature data
    clf       = load_model()
    data_dict = load_all_features()

    # dictionaries to hold daily returns per stock
    strat_ret_dict = {}
    bench_ret_dict = {}

    for sym, df in data_dict.items():
        # get the feature columns for prediction
        X = df[["RSI", "EMA20", "MACD"]]

        # model predicts 1 = go long next day; 0 = stay out
        signals = clf.predict(X)

        # compute next-day returns
        next_ret = df["Close"].pct_change().shift(-1).fillna(0)

        # strategy return = signal * next-day return
        strat_ret_dict[sym] = signals * next_ret

        # benchmark return is just the next-day return
        bench_ret_dict[sym] = next_ret

    # turn per-stock returns into DataFrames
    strat_df = pd.DataFrame(strat_ret_dict)
    bench_df = pd.DataFrame(bench_ret_dict)

    # average across all stocks each day
    strat_port_ret = strat_df.mean(axis=1)
    bench_port_ret = bench_df.mean(axis=1)

    # build cumulative return curves
    strat_cum = (1 + strat_port_ret).cumprod()
    bench_cum = (1 + bench_port_ret).cumprod()

    # plot results on a log scale
    plt.figure(figsize=(12, 6))
    plt.plot(strat_cum, label="ML Strategy")
    plt.plot(bench_cum, label="Equal-Weighted Benchmark")
    plt.yscale("log")
    plt.title("ML Strategy vs Benchmark (log scale)")
    plt.legend()
    plt.grid(True, which="both")
    plt.savefig("images/returns_comparison_log.png")
    plt.show()

if __name__ == "__main__":
    backtest()
