"""Data loading for DengueAlert AI.

Priority:
 1. data/custom_dengue.csv  -> ANY dengue dataset (see CONFIG below / README). Weekly or daily.
 2. data/dengue_features_train.csv + dengue_labels_train.csv -> DengAI (DrivenData).
 3. Otherwise a SYNTHETIC dataset is generated, only to test that the pipeline runs.

Whatever the source, the rest of the project sees the same standard columns:
  week_start_date, year, weekofyear, total_cases, and some of the weather columns below.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

R_RAIN = "station_precip_mm"
R_HUM = "reanalysis_relative_humidity_percent"
R_TEMP = "station_avg_temp_c"
DENGAI_WEATHER = [R_RAIN, R_HUM, "reanalysis_air_temp_k", R_TEMP, "reanalysis_dew_point_temp_k"]

WEATHER_COLS = list(DENGAI_WEATHER)  # updated IN PLACE by load_data to match the dataset
SOURCE = "not loaded yet"            # human-readable description of the data in use

# ---------------------------------------------------------------- custom dataset settings
# Leave a value as None and the loader guesses the column from its name. Or write the exact column name.
CONFIG = {
    "file": "custom_dengue.csv",   # looked up inside data/
    "date": None,                  # e.g. "data_iniSE" (InfoDengue) or "date"
    "cases": None,                 # e.g. "casos"
    "rain": None,                  # optional, e.g. "precipitation_sum"
    "humidity": None,              # optional, e.g. "umidmed"
    "temp": None,                  # optional, e.g. "tempmed"
    "temp_unit": "c",              # "c" or "k"
    "filter_column": None,         # if the CSV has many cities: e.g. "city"
    "filter_value": None,          # ... and the one to keep: e.g. "Karachi"
    "name": None,                  # label shown in the app, e.g. "Rio de Janeiro (InfoDengue)"
}

CANDIDATES = {
    "date": ["week_start_date", "data_inise", "date", "week", "ds", "time", "datetime", "week_start"],
    "cases": ["total_cases", "casos", "cases", "dengue_cases", "case_count", "confirmed_cases", "count", "casos_est"],
    "rain": ["station_precip_mm", "precipitation_sum", "precipitation", "precip", "rainfall", "rain_sum", "rain", "prcp"],
    "humidity": ["reanalysis_relative_humidity_percent", "umidmed", "relative_humidity_2m_mean",
                 "relative_humidity", "humidity", "rh"],
    "temp": ["station_avg_temp_c", "tempmed", "temperature_2m_mean", "temperature", "avg_temp", "temp_c", "temp"],
}


def _find(raw, explicit, key, required):
    lower = {c.lower(): c for c in raw.columns}
    if explicit:
        if explicit.lower() not in lower:
            raise ValueError(f"Column '{explicit}' (for {key}) not found. Columns in file: {list(raw.columns)}")
        return lower[explicit.lower()]
    for cand in CANDIDATES[key]:
        if cand in lower:
            return lower[cand]
    if required:
        raise ValueError(f"Could not guess the {key} column. Set CONFIG['{key}'] in src/data.py. "
                         f"Columns in file: {list(raw.columns)}")
    return None


def _load_custom(path):
    raw = pd.read_csv(path)
    if CONFIG["filter_column"]:
        raw = raw[raw[CONFIG["filter_column"]].astype(str) == str(CONFIG["filter_value"])]
    date_c = _find(raw, CONFIG["date"], "date", True)
    cases_c = _find(raw, CONFIG["cases"], "cases", True)
    found = {"rain": _find(raw, CONFIG["rain"], "rain", False),
             "humidity": _find(raw, CONFIG["humidity"], "humidity", False),
             "temp": _find(raw, CONFIG["temp"], "temp", False)}
    if not any(found.values()):
        raise ValueError("No weather column found (rain / humidity / temperature). "
                         f"Set them in CONFIG. Columns in file: {list(raw.columns)}")

    dates = raw[date_c]
    if pd.api.types.is_numeric_dtype(dates):
        med = float(dates.median())
        dates = pd.to_datetime(dates, unit="ms" if med > 1e11 else "s")
    else:
        dates = pd.to_datetime(dates, errors="coerce")

    d = pd.DataFrame({"week_start_date": dates,
                      "total_cases": pd.to_numeric(raw[cases_c], errors="coerce")})
    names = {"rain": R_RAIN, "humidity": R_HUM, "temp": R_TEMP}
    for k, col in found.items():
        if col:
            v = pd.to_numeric(raw[col], errors="coerce")
            if k == "temp" and CONFIG["temp_unit"].lower() == "k":
                v = v - 273.15
            d[names[k]] = v
    d = d.dropna(subset=["week_start_date"]).sort_values("week_start_date")
    d = d.drop_duplicates("week_start_date")

    # daily (or finer) data -> weekly
    gaps = d["week_start_date"].diff().dt.days.dropna()
    if len(gaps) and gaps.median() < 6:
        agg = {"total_cases": "sum"}
        for c in d.columns:
            if c not in ("week_start_date", "total_cases"):
                agg[c] = "sum" if c == R_RAIN else "mean"
        d = d.set_index("week_start_date").resample("W").agg(agg).reset_index()

    d = d.dropna(subset=["total_cases"]).reset_index(drop=True)
    wcols = [c for c in (R_RAIN, R_HUM, R_TEMP) if c in d.columns]
    d[wcols] = d[wcols].interpolate(limit_direction="both")
    d = d.dropna(subset=wcols).reset_index(drop=True)
    if len(d) < 150:
        raise ValueError(f"Only {len(d)} weekly rows found; need at least ~150 (3 years) for a meaningful model.")
    iso = d["week_start_date"].dt.isocalendar()
    d["year"] = iso.year.astype(int)
    d["weekofyear"] = iso.week.astype(int)
    d["city"] = "custom"
    return d, wcols


def _synthetic(n_weeks=936, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n_weeks)
    season = np.sin(2 * np.pi * t / 52 - 1.0)
    temp = 27 + 3 * season + rng.normal(0, 0.7, n_weeks)
    hum = 75 + 10 * season + rng.normal(0, 3, n_weeks)
    rain = np.clip(40 + 35 * season + rng.normal(0, 15, n_weeks), 0, None)
    dew = 290 + 2 * season + rng.normal(0, 0.5, n_weeks)
    # cases respond to rain/temp with a lag of ~4-6 weeks
    drive = pd.Series(0.04 * rain + 0.8 * temp).shift(5).bfill().values
    base = np.exp((drive - drive.mean()) / drive.std() * 0.9 + 2.5)
    cases = rng.poisson(base).astype(int)
    return pd.DataFrame({
        "city": "synthetic",
        "year": 1990 + t // 52,
        "weekofyear": (t % 52) + 1,
        "week_start_date": pd.date_range("1990-04-30", periods=n_weeks, freq="7D"),
        R_RAIN: rain,
        R_HUM: hum,
        "reanalysis_air_temp_k": temp + 273.15,
        R_TEMP: temp,
        "reanalysis_dew_point_temp_k": dew,
        "total_cases": cases,
    })


def load_data(city="sj"):
    """Returns (dataframe, is_synthetic)."""
    global SOURCE
    custom = os.path.join(DATA_DIR, CONFIG["file"])
    f = os.path.join(DATA_DIR, "dengue_features_train.csv")
    l = os.path.join(DATA_DIR, "dengue_labels_train.csv")
    if os.path.exists(custom):
        df, wcols = _load_custom(custom)
        WEATHER_COLS[:] = wcols
        SOURCE = CONFIG["name"] or f"Custom dataset ({CONFIG['file']})"
        return df, False
    if os.path.exists(f) and os.path.exists(l):
        feats = pd.read_csv(f, parse_dates=["week_start_date"])
        labels = pd.read_csv(l)
        df = feats.merge(labels, on=["city", "year", "weekofyear"])
        df = df[df["city"] == city].sort_values("week_start_date").reset_index(drop=True)
        df[DENGAI_WEATHER] = df[DENGAI_WEATHER].interpolate(limit_direction="both")
        WEATHER_COLS[:] = DENGAI_WEATHER
        SOURCE = "DengAI benchmark: " + ("San Juan" if city == "sj" else "Iquitos") + \
                 f" ({df['week_start_date'].min().year}-{df['week_start_date'].max().year}, past data)"
        return df, False
    WEATHER_COLS[:] = DENGAI_WEATHER
    SOURCE = "SYNTHETIC demo data"
    return _synthetic(), True


def time_split(df, test_frac=0.2):
    """Time-based split: old data = train, newest = test (no random split!)."""
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()