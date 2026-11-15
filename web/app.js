(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];

  const RISK = {
    LOW:    { badge: "LOW RISK",    title: "Cases should stay near normal", msg: "Keep basic precautions: avoid mosquito bites and clear standing water." },
    MEDIUM: { badge: "MEDIUM RISK", title: "Cases may be rising",           msg: "A good time to clear standing water, use repellent and spread awareness." },
    HIGH:   { badge: "HIGH RISK",   title: "A dengue outbreak is likely",   msg: "Take strong precautions and watch for symptoms. See a doctor early if fever appears." },
  };
  const RISK_LABEL = { LOW: "Low", MEDIUM: "Medium", HIGH: "High" };

  const state = { meta: null, idx: 0, lang: DA.getLang(), chart: null, playing: false, timer: null,
                  weekTok: 0, adviceTok: 0, wiTok: 0, week: null };

  /* ---------- helpers */
  const fmt = (iso) => new Date(iso + "T00:00:00").toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
  const fmtMonth = (iso) => new Date(iso + "T00:00:00").toLocaleDateString("en-GB", { month: "short", year: "numeric" });
  const n0 = (x) => Math.round(x).toString();
  const casesTxt = (x) => { const n = Math.round(x); return n === 1 ? "1 case" : `${n} cases`; };
  const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  const { api, toast, renderText } = DA;

  /* ---------- theme */
  function applyChartTheme() {
    if (!window.Chart) return;
    Chart.defaults.color = cssVar("--muted");
    Chart.defaults.borderColor = cssVar("--line");
    Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
    [state.chart, state.errChart, state.recChart].forEach((c) => c && c.update());
  }
  document.addEventListener("themechange", applyChartTheme);

  /* ---------- main chart */
  const selPlugin = {
    id: "sel",
    afterDraw(chart) {
      const pt = chart.getDatasetMeta(1).data[state.idx];
      if (!pt) return;
      const { ctx, chartArea: { top, bottom } } = chart;
      ctx.save();
      ctx.strokeStyle = "#475569"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(pt.x, top); ctx.lineTo(pt.x, bottom); ctx.stroke();
      ctx.fillStyle = "#1e40af"; ctx.beginPath(); ctx.arc(pt.x, pt.y, 5, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
    },
  };
  function buildChart() {
    const s = state.meta.series, thr = state.meta.threshold;
    state.chart = new Chart($("#mainChart"), {
      type: "line",
      data: {
        labels: s.dates,
        datasets: [
          { label: "Actual cases", data: s.actual, borderColor: "#8a94a6", borderWidth: 2, pointRadius: 0, tension: 0.25 },
          { label: "AI forecast", data: s.forecast, borderColor: "#1e40af", backgroundColor: "rgba(30,64,175,.08)", fill: true, borderWidth: 2.5, pointRadius: 0, tension: 0.25 },
          { label: "Outbreak level", data: s.dates.map(() => thr), borderColor: "#b45309", borderDash: [6, 5], borderWidth: 1.5, pointRadius: 0 },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: { duration: 500 },
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { position: "top", align: "start", labels: { usePointStyle: true, boxWidth: 8 } },
          tooltip: { filter: (i) => i.datasetIndex < 2, callbacks: { title: (it) => fmt(it[0].label) } },
        },
        scales: {
          x: { grid: { display: false }, ticks: { maxTicksLimit: 8, maxRotation: 0, callback(v) { return fmtMonth(this.getLabelForValue(v)); } } },
          y: { beginAtZero: true, title: { display: true, text: "Dengue cases per week" } },
        },
        onClick(e, _els, chart) {
          const pts = chart.getElementsAtEventForMode(e, "index", { intersect: false }, true);
          if (pts.length) setWeek(pts[0].index);
        },
      },
      plugins: [selPlugin],
    });
  }

  /* ---------- render forecast */
  function setDelta(el, diff) {
    const r = Math.round(diff);
    el.className = "delta " + (r > 0 ? "up" : r < 0 ? "down" : "flat");
    el.textContent = (r === 0 ? "No change" : (r > 0 ? "+" : "") + r) + " vs " + (el.dataset.vs || "this week");
  }
  function factorList(ul, items) {
    ul.replaceChildren();
    const max = Math.max(...items.map((f) => f.impact), 1e-9);
    items.forEach((f) => {
      const li = document.createElement("li");
      const name = document.createElement("span"); name.textContent = f.label;
      const tag = document.createElement("span");
      tag.className = "tag " + f.dir; tag.textContent = f.dir === "up" ? "Raises risk" : "Lowers risk";
      const bar = document.createElement("div"); bar.className = "bar";
      const i = document.createElement("i"); i.style.width = "0%"; bar.appendChild(i);
      li.append(name, tag, bar); ul.appendChild(li);
      requestAnimationFrame(() => { i.style.width = Math.max(6, (f.impact / max) * 100) + "%"; });
    });
  }
  function renderWeek(w) {
    state.week = w;
    $("#weekLabel").textContent = `${fmt(w.made_on)}  to forecast for ${fmt(w.target_date)}`;
    const card = $("#riskCard"); card.dataset.risk = w.risk; card.classList.remove("loading");
    const r = RISK[w.risk];
    $("#riskBadge").textContent = r.badge;
    $("#riskTitle").textContent = `About ${casesTxt(w.pred)} expected`;
    $("#riskMsg").textContent = r.title;
    $("#gNum").textContent = n0(w.pred);
    $("#thrTxt").textContent = n0(w.threshold);
    $("#gFill").setAttribute("stroke-dasharray", `${Math.min(w.pred / (w.threshold * 1.5), 1) * 100} 100`);
    $("#mNow").textContent = n0(w.now_cases);
    $("#mPred").textContent = n0(w.pred);
    setDelta($("#mDelta"), w.pred - w.now_cases);
    $("#mActual").textContent = n0(w.actual);
    factorList($("#whyMain"), w.main);
    const wx = w.weather;
    const set = (id, key, f) => { const el = $(id); el.hidden = !(key in wx); if (key in wx) $(".m-val", el).textContent = f(wx[key]); };
    set("#wxRain", "rain", (v) => `${v.toFixed(0)} mm`);
    set("#wxHum", "humidity", (v) => `${v.toFixed(0)} %`);
    set("#wxTemp", "temp", (v) => `${v.toFixed(1)} °C`);
    if (state.chart) state.chart.draw();
  }

  async function loadWeek() {
    const tok = ++state.weekTok;
    $("#riskCard").classList.add("loading");
    try {
      const w = await api(`/api/week?idx=${state.idx}`);
      if (tok !== state.weekTok) return;
      renderWeek(w);
    } catch (e) { toast("Could not reach the server. Is server.py running?"); }
    if (!state.playing) { loadAdviceSoon(); loadWhatIfSoon(); }
  }
  const loadWeekSoon = debounce(loadWeek, 120);

  function setWeek(i, immediate) {
    const max = state.meta.weeks.length - 1;
    state.idx = Math.max(0, Math.min(max, i));
    $("#weekRange").value = state.idx;
    $("#weekLabel").textContent = fmt(state.meta.weeks[state.idx]);
    if (state.chart) state.chart.draw();
    immediate ? loadWeek() : loadWeekSoon();
  }

  /* ---------- advice */
  function engineNote(engine) {
    if (!/^(template|no matching)/.test(engine)) return "";
    if (engine.includes("no API key")) return "AI is off: add GEMINI_API_KEY or GROQ_API_KEY to the .env file and restart the server. Showing guideline text.";
    if (engine.includes("LLM errors")) return "AI service problem: " + engine.replace(/^template \(LLM errors: /, "").replace(/\)$/, "").slice(0, 480);
    return "Guideline text only (no matching passage found)";
  }
  async function loadAdvice() {
    const tok = ++state.adviceTok;
    const box = $("#adviceBox");
    box.replaceChildren(...["", "s2", "s3"].map((c) => Object.assign(document.createElement("div"), { className: "skel " + c })));
    try {
      const a = await api(`/api/advice?idx=${state.idx}&lang=${encodeURIComponent(state.lang)}`);
      if (tok !== state.adviceTok) return;
      renderText(box, a.text);
      const meta = $("#adviceMeta"); meta.replaceChildren();
      const note = engineNote(a.engine);
      if (note) { const e = document.createElement("span"); e.className = "engine"; e.textContent = note; meta.appendChild(e); }
    } catch (e) { box.textContent = "Could not load advice."; }
  }
  const loadAdviceSoon = debounce(loadAdvice, 450);

  /* ---------- what-if */
  const wi = { rain: $("#rain"), hum: $("#hum"), temp: $("#temp") };
  function wiLabels() {
    $("#rainOut").textContent = `${wi.rain.value > 0 ? "+" : ""}${wi.rain.value}%`;
    $("#humOut").textContent = `${wi.hum.value > 0 ? "+" : ""}${wi.hum.value} pts`;
    $("#tempOut").textContent = `${wi.temp.value > 0 ? "+" : ""}${wi.temp.value} °C`;
  }
  async function loadWhatIf() {
    const tok = ++state.wiTok;
    try {
      const r = await api(`/api/whatif?idx=${state.idx}&rain=${wi.rain.value}&hum=${wi.hum.value}&temp=${wi.temp.value}`);
      if (tok !== state.wiTok) return;
      $("#wiW").textContent = n0(r.weather_new); $("#wiF").textContent = n0(r.full_new);
      const wd = $("#wiWd"), fd = $("#wiFd");
      wd.dataset.vs = fd.dataset.vs = "original";
      setDelta(wd, r.weather_new - r.weather_old); setDelta(fd, r.full_new - r.full_old);
      $("#wiRisk").textContent = RISK_LABEL[r.risk]; $("#wiRisk").dataset.risk = r.risk;
    } catch (e) { /* ignore */ }
  }
  const loadWhatIfSoon = debounce(loadWhatIf, 200);
  Object.values(wi).forEach((el) => el.addEventListener("input", () => { wiLabels(); loadWhatIfSoon(); }));
  $("#wiReset").addEventListener("click", () => { wi.rain.value = 0; wi.hum.value = 0; wi.temp.value = 0; wiLabels(); loadWhatIf(); });

  /* ---------- controls */
  $("#weekRange").addEventListener("input", (e) => setWeek(+e.target.value));
  $("#prevBtn").addEventListener("click", () => setWeek(state.idx - 1));
  $("#nextBtn").addEventListener("click", () => setWeek(state.idx + 1));
  $("#playBtn").addEventListener("click", () => {
    if (state.playing) return stopPlay();
    state.playing = true; $("#playBtn").textContent = "Pause";
    if (state.idx >= state.meta.weeks.length - 1) state.idx = 0;
    state.timer = setInterval(() => {
      if (state.idx >= state.meta.weeks.length - 1) return stopPlay();
      setWeek(state.idx + 1, true);
    }, 700);
  });
  function stopPlay() {
    clearInterval(state.timer); state.playing = false; $("#playBtn").textContent = "Play";
    loadAdviceSoon(); loadWhatIfSoon();
  }
  $$("#langSeg button").forEach((b) => b.classList.toggle("on", b.dataset.lang === state.lang));
  $$("#langSeg button").forEach((b) => b.addEventListener("click", () => {
    $$("#langSeg button").forEach((x) => x.classList.toggle("on", x === b));
    state.lang = b.dataset.lang; DA.setLang(state.lang); loadAdvice();
  }));

  /* active nav link while scrolling */
  const links = $$("#navLinks a").filter((a) => a.getAttribute("href").startsWith("#"));
  const secs = links.map((a) => $(a.getAttribute("href")));
  const io = new IntersectionObserver((entries) => {
    entries.forEach((en) => { if (en.isIntersecting) links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + en.target.id)); });
  }, { rootMargin: "-40% 0px -55% 0px" });
  secs.forEach((s) => s && io.observe(s));

  /* ---------- boot */
  async function boot() {
    try {
      const m = await api("/api/meta");
      state.meta = m;
      $$(".h-weeks").forEach((e) => (e.textContent = m.horizon));
      $("#chipSource").textContent = m.source;
      $("#slRain").classList.toggle("off", !m.sliders.rain);
      $("#slHum").classList.toggle("off", !m.sliders.hum);
      $("#slTemp").classList.toggle("off", !m.sliders.temp);
      const rng = $("#weekRange"); rng.max = m.weeks.length - 1;
      applyChartTheme(); buildChart(); wiLabels();
      state.idx = m.weeks.length - 1; rng.value = state.idx;
      setWeek(state.idx, true);
    } catch (e) { toast("Could not reach the server. Is server.py running?"); }
  }
  boot();
})();