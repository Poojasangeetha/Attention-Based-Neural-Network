
import pandas as pd
import numpy as np
import os

def load_or_generate(path="data/dataset.csv", min_length=2000):
    if os.path.exists(path):
        df = pd.read_csv(path)
        num = df.select_dtypes(include=[np.number]).columns
        if len(num)==0:
            raise ValueError("No numeric column found.")
        return df[num[0]].to_frame('value')
    # generate complex synthetic series
    np.random.seed(42)
    t = np.arange(0, min_length)
    trend = 0.02 * t
    daily = 5.0 * np.sin(2 * np.pi * t / 24)
    weekly = 2.0 * np.sin(2 * np.pi * t / (24*7))
    noise = np.random.normal(scale=1.5, size=min_length)
    values = 20 + trend + daily + weekly + noise
    df = pd.DataFrame({'value': values})
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    return df
