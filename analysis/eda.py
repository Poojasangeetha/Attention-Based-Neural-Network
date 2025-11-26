
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import os

def run_eda(df, freq=24, out_dir="reports"):
    os.makedirs(out_dir, exist_ok=True)
    series = df['value'].values
    # plot
    plt.figure(figsize=(10,4))
    plt.plot(series)
    plt.title("Time series")
    plt.savefig(f"{out_dir}/ts_plot.png")
    plt.close()

    # stationarity test
    adf_res = adfuller(series[:len(series)//10])  # quick test on subset
    adf_summary = {
        'adf_stat': adf_res[0],
        'p_value': adf_res[1],
        'used_lag': adf_res[2],
        'n_obs': adf_res[3]
    }

    # seasonal decompose (may be slow for large series)
    try:
        decomp = seasonal_decompose(series, period=freq, two_sided=False, extrapolate_trend='freq')
        decomp.trend[:1]  # touch to ensure success
        decomp.trend
        decomp.plot().savefig(f"{out_dir}/decompose.png")
    except Exception as e:
        print("Seasonal decomposition skipped:", e)

    # basic stats saved
    desc = df.describe()
    desc.to_csv(f"{out_dir}/eda_stats.csv")
    # return summary
    return adf_summary
