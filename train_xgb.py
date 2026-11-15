"""Naive vs XGBoost variants (HORIZON weeks ahead) + ablation + SHAP. Run: python train_xgb.py

Models compared (all on the same unseen test weeks):
  naive            : cases stay the same as this week
  xgboost          : cases history + weather + season
  xgboost_residual : same inputs, but learns the CHANGE from this week's cases
  weather_only     : weather + season only, NO case history  (ablation: how much can weather alone say?)
The app uses whichever of xgboost / xgboost_residual is better on a validation slice of the training data.
"""
import json, os
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.data import load_data, time_split
import src.data as D
from src.features import make_features, HORIZON

os.makedirs("outputs", exist_ok=True)
df, synthetic = load_data()
feat, cols = make_features(df)
train, test = time_split(feat)
train = train.iloc[:-HORIZON]  # their targets would fall inside the test period (leakage)
print("Data:", D.SOURCE, "| weekly rows:", len(df), "| weather columns:", D.WEATHER_COLS)

weather_cols = [c for c in cols if not c.startswith("cases_")]  # weather + season, no case history
thr = float(np.percentile(train["target"], 75))  # outbreak = above 75th percentile of TRAIN targets
yte = test["target"].values


def make_model():
    return XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.8, random_state=42)


def fit(frame, use_cols, residual):
    y = frame["target"] - (frame["cases_lag0"] if residual else 0)
    return make_model().fit(frame[use_cols], y)


def predict(model, frame, use_cols, residual):
    p = model.predict(frame[use_cols])
    if residual:
        p = p + frame["cases_lag0"].values
    return np.clip(p, 0, None)


def score(pred):
    true_out = yte > thr
    return {"MAE": float(mean_absolute_error(yte, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(yte, pred))),
            "outbreak_recall": float(((pred > thr) & true_out).sum() / max(true_out.sum(), 1))}


# choose direct vs residual on a validation slice of TRAIN (never on test)
nval = max(int(len(train) * 0.15), 20)
tr_in, va = train.iloc[:-(nval + HORIZON)], train.iloc[-nval:]
val_mae = {}
for name, res in (("xgboost", False), ("xgboost_residual", True)):
    m = fit(tr_in, cols, res)
    val_mae[name] = float(mean_absolute_error(va["target"], predict(m, va, cols, res)))
main_name = min(val_mae, key=val_mae.get)
print("validation MAE:", {k: round(v, 2) for k, v in val_mae.items()}, "->", main_name, "drives the app")

preds, models = {}, {}
preds["naive"] = test["cases_lag0"].values.astype(float)
for name, use_cols, res in (("xgboost", cols, False), ("xgboost_residual", cols, True),
                            ("weather_only", weather_cols, False)):
    models[name] = fit(train, use_cols, res)
    preds[name] = predict(models[name], test, use_cols, res)

res_json = {"synthetic_data": synthetic, "horizon_weeks": HORIZON, "data_source": D.SOURCE,
            "outbreak_threshold": thr, "main_model": main_name, "validation_MAE": val_mae}
for k, p in preds.items():
    res_json[k] = score(p)

main_res = main_name == "xgboost_residual"
joblib.dump({"model": models[main_name], "cols": cols, "thr": thr, "horizon": HORIZON,
             "residual": main_res, "weather_model": models["weather_only"],
             "weather_cols": weather_cols, "main_name": main_name}, "outputs/xgb.joblib")
json.dump(res_json, open("outputs/xgb_metrics.json", "w"), indent=2)

dates = test["week_start_date"].values
plt.figure(figsize=(11, 4.5))
plt.plot(dates, yte, label=f"Actual (t+{HORIZON}w)", color="#555")
plt.plot(dates, preds[main_name], label=f"{main_name} (used in app)", color="#d93025")
plt.plot(dates, preds["weather_only"], label="weather + season only", color="#1a73e8", alpha=0.8)
plt.plot(dates, preds["naive"], label="naive", alpha=0.4)
plt.axhline(thr, ls="--", color="#e8710a", lw=1)
plt.legend(); plt.title(f"Dengue cases {HORIZON} weeks ahead: test period")
plt.tight_layout(); plt.savefig("outputs/xgb_forecast.png", dpi=120); plt.close()

for tag, mdl, use_cols in (("main", models[main_name], cols), ("weather", models["weather_only"], weather_cols)):
    sv = shap.TreeExplainer(mdl).shap_values(test[use_cols])
    shap.summary_plot(sv, test[use_cols], show=False, max_display=12)
    plt.tight_layout(); plt.savefig(f"outputs/shap_{tag}.png", dpi=120); plt.close()

print(f"\nRESULTS - {HORIZON} weeks ahead (test set)")
for k in ("naive", "xgboost", "xgboost_residual", "weather_only"):
    r = res_json[k]
    print(f"{k:18s} MAE={r['MAE']:.2f}  RMSE={r['RMSE']:.2f}  outbreak_recall={r['outbreak_recall']:.2f}")