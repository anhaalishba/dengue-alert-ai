"""Feature engineering for a HORIZON-week-ahead forecast.

At week t we only use information known at week t (weather lags 0..8, recent cases)
and predict cases at week t+HORIZON  ->  column "target".
"""
import math
import pandas as pd
from src.data import WEATHER_COLS

HORIZON = 4  # weeks ahead
LAGS = [0, 1, 2, 3, 4, 6, 8]


def make_features(df, horizon=HORIZON):
    d = df.copy().reset_index(drop=True)
    new = {}
    for c in WEATHER_COLS:
        for k in LAGS:
            new[f"{c}_lag{k}"] = d[c].shift(k)
        new[f"{c}_roll4"] = d[c].rolling(4).mean()
    for k in [0, 1, 2, 3]:
        new[f"cases_lag{k}"] = d["total_cases"].shift(k)
    new["cases_roll4"] = d["total_cases"].rolling(4).mean()
    new["week_sin"] = (2 * math.pi * d["weekofyear"] / 52).apply(math.sin)
    new["week_cos"] = (2 * math.pi * d["weekofyear"] / 52).apply(math.cos)
    feature_cols = list(new.keys())
    new["target"] = d["total_cases"].shift(-horizon)
    d = pd.concat([d, pd.DataFrame(new)], axis=1)
    # drop only rows where OUR features/target are missing (other CSV columns like ndvi are ignored)
    d = d.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
    return d, feature_cols