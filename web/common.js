/* Shared by dashboard + DengueBot pages: API helper, safe text rendering, theme, user chip, logout, mobile menu */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);

  async function api(path, opts) {
    const r = await fetch(path, opts);
    if (r.status === 401) { location.href = "/"; throw new Error("Please log in."); }
    if (!r.ok) {
      let msg = "Request failed";
      try { msg = (await r.json()).error || msg; } catch (e) { /* ignore */ }
      throw new Error(msg);
    }
    return r.json();
  }

  let toastT;
  function toast(msg) {
    const t = $("#toast"); if (!t) return;
    t.textContent = msg; t.classList.add("show");
    clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove("show"), 3500);
  }

  /* safe text -> DOM (never injects HTML from the AI) */
  function inline(el, s) {
    s.split(/\*\*(.+?)\*\*/g).forEach((part, i) => {
      if (i % 2) { const b = document.createElement("strong"); b.textContent = part; el.appendChild(b); }
      else el.appendChild(document.createTextNode(part));
    });
  }
  function renderText(el, text) {
    el.replaceChildren();
    let ul = null;
    for (const raw of String(text || "").split("\n")) {
      const line = raw.trim();
      if (!line) { ul = null; continue; }
      const m = line.match(/^[-*•]\s+(.*)$/);
      if (m) {
        if (!ul) { ul = document.createElement("ul"); el.appendChild(ul); }
        const li = document.createElement("li"); inline(li, m[1]); ul.appendChild(li);
      } else {
        ul = null;
        const p = document.createElement("p"); p.dir = "auto"; inline(p, line); el.appendChild(p);
      }
    }
  }

  /* theme */
  const themeBtn = $("#themeBtn");
  if (themeBtn) themeBtn.addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme")
      || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("theme", next); } catch (e) { /* ignore */ }
    document.dispatchEvent(new Event("themechange"));
  });

  /* mobile sidebar */
  const body = document.body;
  const closeMenu = () => body.classList.remove("side-open");
  const menuBtn = $("#menuBtn");
  if (menuBtn) menuBtn.addEventListener("click", () => body.classList.toggle("side-open"));
  const scrim = $("#scrim"); if (scrim) scrim.addEventListener("click", closeMenu);
  document.querySelectorAll(".side-nav a").forEach((a) => a.addEventListener("click", closeMenu));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeMenu(); });

  /* user + logout */
  (async function loadMe() {
    try {
      const u = await api("/api/me");
      $("#userName").textContent = u.name;
      $("#avatar").textContent = u.name.trim().charAt(0).toUpperCase() || "U";
      $("#userChip").hidden = false;
    } catch (e) { /* redirected to login if not authorised */ }
  })();
  const out = $("#logoutBtn");
  if (out) out.addEventListener("click", async () => {
    try { await fetch("/api/auth/logout", { method: "POST" }); } catch (e) { /* ignore */ }
    location.href = "/";
  });

  /* remembered advice / chat language (shared by both pages) */
  const getLang = () => { try { return localStorage.getItem("lang") || "English"; } catch (e) { return "English"; } };
  const setLang = (l) => { try { localStorage.setItem("lang", l); } catch (e) { /* ignore */ } };

  window.DA = { $, api, toast, renderText, getLang, setLang };
})();