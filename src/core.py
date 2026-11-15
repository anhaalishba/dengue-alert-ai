"""Core engine used by the website backend (server.py): forecast, explanations, what-if, advice, Q&A."""
import json, os, re, threading
import joblib
import numpy as np
import pandas as pd
import shap
import src.data as D
from src.data import load_data, time_split
from src.features import make_features
from src.rag import Retriever
from src.advisory import generate_advisory, answer_question, LANGS

NAMES = {
    "station_precip_mm": "rainfall",
    "reanalysis_relative_humidity_percent": "humidity",
    "reanalysis_air_temp_k": "air temperature",
    "station_avg_temp_c": "temperature",
    "reanalysis_dew_point_temp_k": "dew point",
    "cases": "recent dengue cases",
}


def plain(name):
    if name in ("week_sin", "week_cos"):
        return "Time of year (season)"
    m = re.match(r"(.*)_lag(\d+)$", name)
    if m:
        k = int(m.group(2))
        when = "this week" if k == 0 else f"{k} week{'s' if k > 1 else ''} ago"
        return f"{NAMES.get(m.group(1), m.group(1))} ({when})".capitalize()
    m = re.match(r"(.*)_roll4$", name)
    if m:
        return f"{NAMES.get(m.group(1), m.group(1))} (4-week average)".capitalize()
    return name.replace("_", " ").capitalize()


class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        art = joblib.load("outputs/xgb.joblib")
        df, self.synthetic = load_data()
        feat, self.cols = make_features(df)
        _, test = time_split(feat)
        self.test = test.reset_index(drop=True)
        self.H = art.get("horizon", 4)
        self.model, self.thr = art["model"], art["thr"]
        self.residual = art.get("residual", False)
        self.wmodel, self.wcols = art["weather_model"], art["weather_cols"]
        self.source = D.SOURCE
        self.bases = {re.sub(r"_(lag\d+|roll4)$", "", c) for c in self.cols}
        self.exp_main = shap.TreeExplainer(self.model)
        self.exp_w = shap.TreeExplainer(self.wmodel)
        self.retriever = Retriever()
        self._advice, self._ans = {}, {}
        self.fc_all = self._fc(self.test)

    # ------------------------------------------------------------ helpers
    def _fc(self, frame):
        p = self.model.predict(frame[self.cols])
        if self.residual:
            p = p + frame["cases_lag0"].values
        return np.clip(p, 0, None)

    def risk_of(self, p):
        return "HIGH" if p > self.thr else ("MEDIUM" if p > self.thr * 0.6 else "LOW")

    def _idx(self, idx):
        return int(np.clip(int(idx), 0, len(self.test) - 1))

    def _date(self, ts):
        return pd.Timestamp(ts).strftime("%Y-%m-%d")

    # ------------------------------------------------------------ API pieces
    def meta(self):
        t = self.test
        return {
            "source": self.source, "synthetic": bool(self.synthetic), "horizon": self.H,
            "threshold": float(self.thr),
            "langs": list(LANGS.keys()),
            "has_key": bool(os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY")),
            "sliders": {"rain": D.R_RAIN in self.bases, "hum": D.R_HUM in self.bases, "temp": D.R_TEMP in self.bases},
            "weeks": [self._date(d) for d in t["week_start_date"]],
            "series": {
                "dates": [self._date(d + pd.Timedelta(weeks=self.H)) for d in t["week_start_date"]],
                "actual": [float(x) for x in t["target"]],
                "forecast": [round(float(x), 2) for x in self.fc_all],
            },
        }

    def week(self, idx):
        idx = self._idx(idx)
        row = self.test.iloc[[idx]]
        with self.lock:
            pred = float(self._fc(row)[0])
            w_pred = float(np.clip(self.wmodel.predict(row[self.wcols])[0], 0, None))
            sv = self.exp_main.shap_values(row[self.cols])[0]
            wsv = self.exp_w.shap_values(row[self.wcols])[0]

        def top(values, names):
            order = np.argsort(-np.abs(values))[:3]
            return [{"label": plain(names[i]), "dir": "up" if values[i] > 0 else "down",
                     "impact": float(abs(values[i]))} for i in order]

        weather = {}
        for key, col in (("rain", D.R_RAIN), ("humidity", D.R_HUM), ("temp", D.R_TEMP)):
            if col in row:
                weather[key] = float(row[col].iloc[0])
        made_on = row["week_start_date"].iloc[0]
        return {
            "idx": idx, "made_on": self._date(made_on),
            "target_date": self._date(made_on + pd.Timedelta(weeks=self.H)),
            "pred": pred, "w_pred": w_pred, "risk": self.risk_of(pred), "threshold": float(self.thr),
            "actual": float(row["target"].iloc[0]), "now_cases": float(row["total_cases"].iloc[0]),
            "what": "the change from this week" if self.residual else "the forecast",
            "main": top(sv, self.cols), "weather_factors": top(wsv, self.wcols), "weather": weather,
        }

    def whatif(self, idx, rain_pct, hum_d, temp_d):
        idx = self._idx(idx)
        row = self.test.iloc[[idx]].copy()
        for c in self.cols:
            base = re.sub(r"_(lag\d+|roll4)$", "", c)
            if base == D.R_RAIN:
                row[c] = row[c] * (1 + rain_pct / 100)
            elif base == D.R_HUM:
                row[c] = (row[c] + hum_d).clip(upper=100)
            elif base in (D.R_TEMP, "reanalysis_air_temp_k", "reanalysis_dew_point_temp_k"):
                row[c] = row[c] + temp_d
        orig = self.test.iloc[[idx]]
        with self.lock:
            w_new = float(np.clip(self.wmodel.predict(row[self.wcols])[0], 0, None))
            m_new = float(self._fc(row)[0])
            w_old = float(np.clip(self.wmodel.predict(orig[self.wcols])[0], 0, None))
            m_old = float(self._fc(orig)[0])
        return {"weather_new": w_new, "weather_old": w_old, "full_new": m_new, "full_old": m_old,
                "risk": self.risk_of(max(w_new, m_new)), "risk_old": self.risk_of(max(w_old, m_old))}

    def advice(self, idx, lang):
        w = self.week(idx)
        factors = tuple(f["label"].lower() for f in w["weather_factors"])
        key = (w["risk"], int(round(w["pred"])), factors, lang)
        if key not in self._advice:
            passages = self.retriever.search(
                f"dengue {w['risk']} risk precautions rain humidity mosquito prevention", k=3)
            text, engine = generate_advisory(w["risk"], int(round(w["pred"])), list(factors), passages, lang)
            res = {"text": text, "engine": engine, "language": lang,
                   "sources": [{"source": p["source"], "text": p["text"]} for p in passages]}
            if engine.startswith("template"):
                return res  # do not cache failures, so the next try can succeed
            self._advice[key] = res
        return self._advice[key]

    def ask(self, question, lang):
        key = (question.strip().lower(), lang)
        if key not in self._ans:
            passages = self.retriever.search(question, k=3)
            text, engine = answer_question(question, passages, lang)
            res = {"text": text, "engine": engine, "language": lang,
                   "sources": [{"source": p["source"], "text": p["text"]} for p in passages]}
            if engine.startswith(("template", "no matching")):
                return res
            self._ans[key] = res
        return self._ans[key]

    def results(self):
        if not os.path.exists("outputs/xgb_metrics.json"):
            return {"rows": []}
        m = json.load(open("outputs/xgb_metrics.json"))
        names = {"naive": "Simple guess (cases stay the same)",
                 "xgboost": "XGBoost: cases + weather",
                 "xgboost_residual": "XGBoost: learns the change",
                 "weather_only": "XGBoost: weather + season only",
                 "lstm": "LSTM (deep learning)"}
        rows = []
        for k, nm in names.items():
            if k in m:
                rows.append({"key": k, "name": nm, "used": k == m.get("main_model"),
                             "mae": round(m[k]["MAE"], 2), "rmse": round(m[k]["RMSE"], 2),
                             "recall": round(m[k].get("outbreak_recall", 0) * 100)})
        return {"rows": rows, "horizon": m.get("horizon_weeks", self.H)}