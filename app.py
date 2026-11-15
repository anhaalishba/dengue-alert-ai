"""DengueAlert AI - Streamlit demo. Run: streamlit run app.py"""
import json, os, re
import altair as alt
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import src.data as D
from src.data import load_data, time_split
from src.features import make_features
from src.rag import Retriever
from src.advisory import generate_advisory, answer_question, LANGS

st.set_page_config(page_title="DengueAlert AI", page_icon="🦟", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 2rem;}
.risk-box {border-radius: 14px; padding: 22px 26px; margin: 6px 0 18px 0;}
.risk-box h2 {margin: 0 0 4px 0; font-size: 1.8rem;}
.risk-box p {margin: 0; font-size: 1.05rem;}
.stMarkdown p, .stMarkdown li {unicode-bidi: plaintext;}
</style>
""", unsafe_allow_html=True)

if not os.path.exists("outputs/xgb.joblib"):
    st.error("Pehle `python train_xgb.py` aur `python train_lstm.py` chalayein.")
    st.stop()


@st.cache_resource
def setup():
    art = joblib.load("outputs/xgb.joblib")
    df, synthetic = load_data()
    feat, cols = make_features(df)
    _, test = time_split(feat)
    return art, test.reset_index(drop=True), synthetic, Retriever(), D.SOURCE


art, test, synthetic, retriever, source = setup()
H = art.get("horizon", 4)
model, cols, thr = art["model"], art["cols"], art["thr"]
residual = art.get("residual", False)
wmodel, wcols = art["weather_model"], art["weather_cols"]

NAMES = {
    "station_precip_mm": "rainfall",
    "reanalysis_relative_humidity_percent": "humidity",
    "reanalysis_air_temp_k": "air temperature",
    "station_avg_temp_c": "temperature",
    "reanalysis_dew_point_temp_k": "dew point",
    "cases": "recent dengue cases",
}
BASES = {re.sub(r"_(lag\d+|roll4)$", "", c) for c in cols}


def plain(name):
    if name in ("week_sin", "week_cos"):
        return "time of year (season)"
    m = re.match(r"(.*)_lag(\d+)$", name)
    if m:
        k = int(m.group(2))
        when = "this week" if k == 0 else f"{k} week{'s' if k > 1 else ''} ago"
        return f"{NAMES.get(m.group(1), m.group(1))} ({when})"
    m = re.match(r"(.*)_roll4$", name)
    if m:
        return f"{NAMES.get(m.group(1), m.group(1))} (4-week average)"
    return name.replace("_", " ")


def forecast(frame):
    p = model.predict(frame[cols])
    if residual:
        p = p + frame["cases_lag0"].values
    return np.clip(p, 0, None)


def cases_txt(n):
    n = int(round(n))
    return f"{n} case" if n == 1 else f"{n} cases"


def risk_of(p):
    return "HIGH" if p > thr else ("MEDIUM" if p > thr * 0.6 else "LOW")


def apply_whatif(row, rain_pct, hum_d, temp_d):
    r = row.copy()
    for c in cols:
        base = re.sub(r"_(lag\d+|roll4)$", "", c)
        if base == D.R_RAIN:
            r[c] = r[c] * (1 + rain_pct / 100)
        elif base == D.R_HUM:
            r[c] = (r[c] + hum_d).clip(upper=100)
        elif base in (D.R_TEMP, "reanalysis_air_temp_k", "reanalysis_dew_point_temp_k"):
            r[c] = r[c] + temp_d
    return r


RISK = {
    "LOW": ("#e6f4ea", "#137333", "🟢 LOW RISK",
            "Cases are expected to stay near normal levels. Keep basic precautions."),
    "MEDIUM": ("#fef7e0", "#b06000", "🟠 MEDIUM RISK",
               "Cases may be rising. A good time to clear standing water and use repellent."),
    "HIGH": ("#fce8e6", "#c5221f", "🔴 HIGH RISK",
             "A dengue outbreak is likely. Take strong precautions and watch for symptoms."),
}


@st.cache_data(show_spinner="Preparing advisory...")
def get_advisory(risk, pred_rounded, factors, language):
    passages = retriever.search(f"dengue {risk} risk precautions rain humidity mosquito prevention", k=3)
    text, engine = generate_advisory(risk, pred_rounded, list(factors), passages, language)
    return text, engine, passages


@st.cache_data(show_spinner="Searching the guidelines...")
def get_answer(question, language):
    passages = retriever.search(question, k=3)
    text, engine = answer_question(question, passages, language)
    return text, engine, passages


# ------------------------------------------------------------------ header
st.title("🦟 DengueAlert AI")
st.markdown("**Know about a dengue outbreak weeks before it happens**, and what to do about it.")
st.caption(f"Data: {source}")
if synthetic:
    st.warning("⚠️ Demo data (synthetic) is being used. Put a real dataset in the `data/` folder.")

with st.expander("What does this app do?", expanded=False):
    st.markdown(
        f"1. Pick a week.\n"
        f"2. The AI looks at the weather (rain, humidity, temperature) and recent cases, and forecasts "
        f"dengue cases **{H} weeks ahead**.\n"
        f"3. It shows a simple risk level and explains what drove the forecast.\n"
        f"4. You can change the weather and see what happens (what-if).\n"
        f"5. It writes easy precautions in English, Urdu or Roman Urdu, based on WHO guidelines, and you can ask it questions."
    )

# ------------------------------------------------------------------ controls
last = len(test) - 1
idx = st.select_slider(
    "Pick a week (the forecast is made on this week)",
    options=list(range(len(test))), value=last,
    format_func=lambda i: test["week_start_date"].iloc[i].strftime("%d %b %Y"),
)
lang = st.radio("🌐 Advice language", list(LANGS.keys()), horizontal=True)

row = test.iloc[[idx]]
made_on = row["week_start_date"].iloc[0]
target_date = made_on + pd.Timedelta(weeks=H)
pred = float(forecast(row)[0])
w_pred = float(np.clip(wmodel.predict(row[wcols])[0], 0, None))
actual = float(row["target"].iloc[0])
now_cases = float(row["total_cases"].iloc[0])
risk = risk_of(pred)
bg, fg, title, msg = RISK[risk]

tab_fc, tab_bot, tab_res, tab_how = st.tabs(["📈 Forecast", "💬 Ask DengueBot", "📊 Model results", "ℹ️ How it works"])

with tab_fc:
    st.markdown(
        f"<div class='risk-box' style='background:{bg};color:{fg}'>"
        f"<h2>{title}</h2><p>Forecast for the week of <b>{target_date.strftime('%d %b %Y')}</b>: "
        f"about <b>{cases_txt(pred)}</b>. {msg}</p></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Cases this week", f"{now_cases:.0f}")
    c2.metric(f"Forecast in {H} weeks", f"{pred:.0f}", delta=f"{int(round(pred - now_cases)):+d} vs this week",
              delta_color="inverse")
    c3.metric(f"What actually happened in {H} weeks", f"{actual:.0f}",
              help="Shown because this demo uses past data, so we can check the forecast.")

    # chart
    dfc = pd.DataFrame({
        "Week": test["week_start_date"] + pd.Timedelta(weeks=H),
        "Actual cases": test["target"].values,
        "AI forecast": forecast(test),
    }).melt("Week", var_name="Series", value_name="Cases")
    line = alt.Chart(dfc).mark_line(strokeWidth=2).encode(
        x=alt.X("Week:T", title=None),
        y=alt.Y("Cases:Q", title="Dengue cases per week", scale=alt.Scale(zero=True)),
        color=alt.Color("Series:N", legend=alt.Legend(orient="top", title=None),
                        scale=alt.Scale(domain=["Actual cases", "AI forecast"], range=["#9aa0a6", "#d93025"])),
    )
    thr_rule = alt.Chart(pd.DataFrame({"y": [thr]})).mark_rule(strokeDash=[5, 4], color="#e8710a").encode(y="y:Q")
    thr_lbl = alt.Chart(pd.DataFrame({"y": [thr], "t": ["outbreak level"]})).mark_text(
        align="left", dx=4, dy=-6, color="#e8710a").encode(y="y:Q", text="t:N", x=alt.value(4))
    sel = alt.Chart(pd.DataFrame({"Week": [target_date]})).mark_rule(color="#1a73e8", strokeWidth=2).encode(x="Week:T")
    st.altair_chart((line + thr_rule + thr_lbl + sel).properties(height=320), width="stretch")
    st.caption("Blue line = the week you picked. Orange dashed line = level above which we call it an outbreak.")

    # why
    st.subheader("Why this forecast?")
    sv = shap.TreeExplainer(model).shap_values(row[cols])[0]
    top = np.argsort(-np.abs(sv))[:3]
    what = "the change from this week" if residual else "the forecast"
    st.markdown(f"Biggest influences on {what}:")
    for i in top:
        arrow = "pushes risk **up** ⬆️" if sv[i] > 0 else "pushes risk **down** ⬇️"
        st.markdown(f"- **{plain(cols[i]).capitalize()}**: {arrow}")

    st.markdown("##### What does the weather alone say?")
    wsv = shap.TreeExplainer(wmodel).shap_values(row[wcols])[0]
    wtop = np.argsort(-np.abs(wsv))[:3]
    factors = tuple(plain(wcols[i]) for i in wtop)
    st.markdown(f"Using **only weather and time of year** (no case history), the model expects about "
                f"**{cases_txt(w_pred)}**. Weather factors that mattered most:")
    for i in wtop:
        arrow = "⬆️ raises" if wsv[i] > 0 else "⬇️ lowers"
        st.markdown(f"- **{plain(wcols[i]).capitalize()}** {arrow} the forecast")
    wcs = st.columns(3)
    k = 0
    for col, label, fmt in (("station_precip_mm", "🌧️ Rainfall this week", "{:.0f} mm"),
                            ("reanalysis_relative_humidity_percent", "💧 Humidity", "{:.0f} %"),
                            ("station_avg_temp_c", "🌡️ Temperature", "{:.1f} °C")):
        if col in row:
            wcs[k].metric(label, fmt.format(row[col].iloc[0])); k += 1

    # what-if simulator
    st.subheader("🧪 What if the weather changes?")
    st.caption("Change the weather of the past weeks and see how the forecast responds.")
    sc = st.columns(3)
    rain_pct = sc[0].slider("🌧️ Rainfall change (%)", -100, 200, 0, 10) if D.R_RAIN in BASES else 0
    hum_d = sc[1].slider("💧 Humidity change (points)", -20, 20, 0, 1) if D.R_HUM in BASES else 0
    temp_d = sc[2].slider("🌡️ Temperature change (°C)", -5.0, 5.0, 0.0, 0.5) if D.R_TEMP in BASES else 0.0
    row_w = apply_whatif(row, rain_pct, hum_d, temp_d)
    w_new = float(np.clip(wmodel.predict(row_w[wcols])[0], 0, None))
    m_new = float(forecast(row_w)[0])
    r_new = risk_of(max(w_new, m_new))
    q1, q2, q3 = st.columns(3)
    q1.metric("Weather-based forecast", f"{w_new:.0f}", delta=f"{w_new - w_pred:+.1f} vs original", delta_color="inverse")
    q2.metric("Full model forecast", f"{m_new:.0f}", delta=f"{m_new - pred:+.1f} vs original", delta_color="inverse")
    q3.metric("Risk with these changes", {"HIGH": "🔴 HIGH", "MEDIUM": "🟠 MEDIUM", "LOW": "🟢 LOW"}[r_new])
    st.caption("This shows how the model reacts to weather. It is an exploration of the model, not proof of cause and effect.")

    # advisory
    st.subheader("What should people do?")
    text, engine, passages = get_advisory(risk, int(round(pred)), factors, lang)
    with st.container(border=True):
        st.markdown(text)
    if engine.startswith("template"):
        st.caption("Advice assembled from the guideline text (English only). "
                   "Add a Gemini or Groq API key to get AI-written advice in Urdu.")
    else:
        st.caption(f"Written by {engine} using only the guideline passages below.")
    with st.expander("Sources used"):
        for p in passages:
            st.markdown(f"**{p['source']}**\n\n{p['text']}")

with tab_bot:
    st.subheader("💬 Ask DengueBot")
    st.markdown("Ask a question about dengue. Answers come **only from the guideline documents**, "
                "with the sources shown. (Type your question in English; the answer uses the language you selected.)")
    examples = ["What are the warning signs of severe dengue?",
                "Which painkillers should be avoided?",
                "How is dengue treated?",
                "How long do symptoms last?"]
    ec = st.columns(len(examples))
    picked = None
    for i, ex in enumerate(examples):
        if ec[i].button(ex, key=f"ex{i}", width="stretch"):
            picked = ex
    with st.form("ask"):
        typed = st.text_input("Your question", placeholder="e.g. When should I go to the hospital?")
        go = st.form_submit_button("Ask")
    question = picked or (typed.strip() if go and typed.strip() else None)
    if question:
        st.markdown(f"**Q: {question}**")
        ans, eng, psg = get_answer(question, lang)
        with st.container(border=True):
            st.markdown(ans)
        st.caption(f"Answered by {eng} using only the guideline passages below."
                   if not eng.startswith(("template", "no matching")) else f"Source text only ({eng[:60]}).")
        with st.expander("Sources used"):
            for p in psg:
                st.markdown(f"**{p['source']}**\n\n{p['text']}")
    st.caption("Informational only. For symptoms or emergencies, see a doctor.")

with tab_res:
    st.subheader("How good is the forecast?")
    st.markdown(f"We tested several approaches on weeks the models had **never seen**, forecasting {H} weeks ahead.")
    if os.path.exists("outputs/xgb_metrics.json"):
        m = json.load(open("outputs/xgb_metrics.json"))
        names = {"naive": "Simple guess (cases stay the same)",
                 "xgboost": "XGBoost: cases + weather",
                 "xgboost_residual": "XGBoost: learns the change from this week",
                 "weather_only": "XGBoost: weather + season only (no case history)",
                 "lstm": "LSTM (deep learning)"}
        rows = []
        for k, nm in names.items():
            if k in m:
                used = " ✅ used in app" if k == m.get("main_model") else ""
                rows.append({"Approach": nm + used,
                             "Average error (cases)": round(m[k]["MAE"], 1),
                             "Outbreaks caught": f"{m[k].get('outbreak_recall', float('nan')) * 100:.0f}%"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.markdown(
            "- **Average error**: how many cases the forecast is off by, on average. Lower is better.\n"
            "- **Outbreaks caught**: out of all real outbreak weeks, how many the model warned about. "
            "Higher is better, and this matters most for an early warning system.\n"
            "- **Weather + season only** shows how much the weather *alone* can tell us, without looking at recent cases."
        )

with tab_how:
    st.subheader("How it works")
    st.markdown(
        "**1. Machine learning (XGBoost)** learns the link between past weather, recent cases and future cases.\n\n"
        "**2. Deep learning (LSTM)** reads the last 12 weeks as a sequence and does the same, for comparison.\n\n"
        "**3. Explainability (SHAP)** shows which inputs pushed the forecast up or down.\n\n"
        "**4. Generative AI (RAG)** searches dengue guidelines and writes precautions and answers from those "
        "passages only, in English, Urdu or Roman Urdu, so it does not make things up."
    )

st.divider()
st.caption("Informational tool only. Not medical advice. If you have fever with severe symptoms, see a doctor.")