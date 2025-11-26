
import pandas as pd
import numpy as np

def load_complex_dataset():
    np.random.seed(42)
    t = np.arange(0, 2000)
    series = 0.05*t + 10*np.sin(2*np.pi*t/50) + np.random.normal(scale=2, size=len(t))
    df = pd.DataFrame({'value': series})
    return df
