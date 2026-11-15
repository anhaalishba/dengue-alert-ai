(() => {
  "use strict";
  const $ = (s) => document.querySelector(s);

  /* theme toggle */
  $("#themeBtn").addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme")
      || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("theme", next); } catch (e) { /* ignore */ }
  });

  /* tabs */
  function show(which) {
    const login = which === "login";
    $("#loginForm").hidden = !login; $("#signupForm").hidden = login;
    $("#tabLogin").classList.toggle("on", login); $("#tabSignup").classList.toggle("on", !login);
    $("#tabLogin").setAttribute("aria-selected", login); $("#tabSignup").setAttribute("aria-selected", !login);
    (login ? $("#liEmail") : $("#suName")).focus({ preventScroll: true });
  }
  $("#tabLogin").addEventListener("click", () => show("login"));
  $("#tabSignup").addEventListener("click", () => show("signup"));
  $("#toSignup").addEventListener("click", (e) => { e.preventDefault(); show("signup"); });
  $("#toLogin").addEventListener("click", (e) => { e.preventDefault(); show("login"); });
  if (location.hash === "#signup") show("signup");

  /* show / hide password */
  document.querySelectorAll(".eye").forEach((b) => b.addEventListener("click", () => {
    const i = document.getElementById(b.dataset.for);
    const hidden = i.type === "password";
    i.type = hidden ? "text" : "password";
    b.setAttribute("aria-label", hidden ? "Hide password" : "Show password");
  }));

  /* password strength */
  $("#suPass").addEventListener("input", (e) => {
    const v = e.target.value;
    let score = 0;
    if (v.length >= 8) score++;
    if (v.length >= 12) score++;
    if (/[a-z]/.test(v) && /[A-Z]/.test(v)) score++;
    if (/\d/.test(v) && /[^A-Za-z0-9]/.test(v)) score++;
    const pct = v.length ? Math.max(15, score * 25) : 0;
    const bar = $("#meterBar");
    bar.style.width = pct + "%";
    bar.style.background = score <= 1 ? "var(--high)" : score === 2 ? "var(--med)" : "var(--low)";
    $("#meterTxt").textContent = !v.length ? "Use 8+ characters" : v.length < 8 ? "Too short" : score <= 1 ? "Okay" : score === 2 ? "Good" : "Strong";
  });

  /* submit helpers */
  async function post(url, body) {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    let data = {};
    try { data = await r.json(); } catch (e) { /* ignore */ }
    if (!r.ok) throw new Error(data.error || "Something went wrong. Please try again.");
    return data;
  }
  function wire(formSel, msgSel, btnSel, getBody, url, busyText) {
    const form = $(formSel), msg = $(msgSel), btn = $(btnSel), label = btn.textContent;
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      msg.className = "form-msg"; msg.textContent = "";
      const body = getBody();
      btn.disabled = true; btn.textContent = busyText;
      try {
        await post(url, body);
        msg.className = "form-msg ok"; msg.textContent = "Success! Opening your dashboard…";
        location.href = "/dashboard";
      } catch (err) {
        msg.textContent = err.message; btn.disabled = false; btn.textContent = label;
      }
    });
  }
  wire("#loginForm", "#liMsg", "#liBtn",
    () => ({ email: $("#liEmail").value, password: $("#liPass").value, remember: $("#liRemember").checked }),
    "/api/auth/login", "Logging in…");
  wire("#signupForm", "#suMsg", "#suBtn",
    () => ({ name: $("#suName").value, email: $("#suEmail").value, password: $("#suPass").value, remember: true }),
    "/api/auth/signup", "Creating account…");
})();