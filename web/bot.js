(() => {
  "use strict";
  const { $, api, renderText, getLang, setLang } = DA;
  const QUESTIONS = ["What are the warning signs of severe dengue?", "Which painkillers should be avoided?",
                     "How is dengue treated?", "How long do symptoms last?", "How can I prevent mosquito bites?"];
  let lang = getLang();
  const chat = $("#chat");

  function engineNote(engine) {
    if (!/^(template|no matching)/.test(engine)) return "";
    if (engine.includes("no API key")) return "AI is off: add GEMINI_API_KEY or GROQ_API_KEY to the .env file and restart the server.";
    if (engine.includes("LLM errors")) return "AI service problem: " + engine.replace(/^template \(LLM errors: /, "").replace(/\)$/, "").slice(0, 480);
    return "";
  }
  function addBubble(kind, text) {
    const w = $("#welcome"); if (w) w.remove();
    const b = document.createElement("div"); b.className = "bubble " + kind;
    const body = document.createElement("div"); body.className = "b-body"; body.dir = "auto";
    if (kind === "user") body.textContent = text; else renderText(body, text);
    b.appendChild(body); chat.appendChild(b); chat.scrollTop = chat.scrollHeight;
    return b;
  }
  async function ask(q) {
    q = q.trim(); if (!q) return;
    addBubble("user", q);
    if (/^(hi+|hey+|hello+|salam|salaam|assalam[\w\s-]*|aoa|hola|yo)\W*$/i.test(q)) {
      addBubble("bot", "Salam! Ask me about dengue symptoms, treatment or precautions."); return;
    }
    const t = document.createElement("div"); t.className = "bubble bot";
    t.innerHTML = '<span class="typing"><i></i><i></i><i></i></span>'; chat.appendChild(t); chat.scrollTop = chat.scrollHeight;
    try {
      const a = await api("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" },
                                        body: JSON.stringify({ question: q, lang }) });
      t.remove();
      const b = addBubble("bot", a.text);
      const note = engineNote(a.engine);
      if (note) { const m = document.createElement("div"); m.className = "b-meta"; m.textContent = note; b.appendChild(m); }
      chat.scrollTop = chat.scrollHeight;
    } catch (e) { t.remove(); addBubble("bot", e.message || "Sorry, something went wrong."); }
  }
  $("#askForm").addEventListener("submit", (e) => { e.preventDefault(); const i = $("#askInput"); const q = i.value; i.value = ""; ask(q); });
  QUESTIONS.forEach((q) => {
    const b = document.createElement("button"); b.type = "button"; b.className = "q-chip"; b.textContent = q;
    b.addEventListener("click", () => ask(q)); $("#askChips").appendChild(b);
  });
  document.querySelectorAll("#langSeg button").forEach((b) => {
    b.classList.toggle("on", b.dataset.lang === lang);
    b.addEventListener("click", () => {
      document.querySelectorAll("#langSeg button").forEach((x) => x.classList.toggle("on", x === b));
      lang = b.dataset.lang; setLang(lang);
    });
  });
  $("#clearBtn").addEventListener("click", () => location.reload());
  $("#askInput").focus();
})();