/* SonicSentinel AI - shared browser code (every page). */
(function () {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const config = JSON.parse(($("#app-config") || { textContent: "{}" }).textContent || "{}");
  const csrf = ($('meta[name="csrf-token"]') || {}).content || "";
  window.SS_CLASS_PARENTS = config.classParents || {};

  /* ---------- fetch with CSRF + JSON errors ---------- */
  async function api(url, options = {}) {
    const opts = Object.assign({ credentials: "same-origin", headers: {} }, options);
    opts.headers = Object.assign({ "X-CSRFToken": csrf, Accept: "application/json" }, opts.headers || {});
    if (opts.json !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(opts.json);
      delete opts.json;
    }
    const res = await fetch(url, opts);
    let data = null;
    try { data = await res.json(); } catch (e) { data = null; }
    if (!res.ok) {
      const err = new Error((data && data.error) || `Request failed (${res.status})`);
      err.status = res.status;
      err.data = data;
      throw err;
    }
    return data;
  }

  /* ---------- formatting ---------- */
  const fmt = {
    pct(x, d = 0) { return x === null || x === undefined ? "-" : `${(x * 100).toFixed(d)}%`; },
    time(iso) {
      const d = new Date(iso);
      return isNaN(d) ? iso : d.toLocaleString(undefined, { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", second: "2-digit" });
    },
    clock(d = new Date()) { return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" }); },
    rel(iso) {
      const s = (Date.now() - new Date(iso).getTime()) / 1000;
      if (s < 45) return "just now";
      if (s < 3600) return `${Math.round(s / 60)} min ago`;
      if (s < 86400) return `${Math.round(s / 3600)} h ago`;
      return null;
    },
    tc(sec) { sec = Math.max(0, sec || 0); const m = Math.floor(sec / 60); return `${m}:${(sec - m * 60).toFixed(1).padStart(4, "0")}`; },
  };

  function localiseTimes(root = document) {
    $$("time.js-time", root).forEach((el) => {
      const iso = el.getAttribute("datetime");
      if (!iso) return;
      const d = new Date(iso);
      if (isNaN(d)) return;
      const rel = el.classList.contains("rel") ? fmt.rel(iso) : null;
      const old = Date.now() - d.getTime() > 180 * 86400e3;
      el.textContent = rel || d.toLocaleString(undefined, { day: "2-digit", month: "short", year: old ? "numeric" : undefined, hour: "2-digit", minute: "2-digit" });
      el.title = d.toLocaleString();
    });
  }

  /* ---------- element helper (text only, never innerHTML with data) ---------- */
  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k === "style") node.setAttribute("style", v);
      else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat()) {
      if (c === null || c === undefined || c === false) continue;
      node.appendChild(typeof c === "string" || typeof c === "number" ? document.createTextNode(String(c)) : c);
    }
    return node;
  }

  /** node.append() that skips null / undefined / false (DOM append would print "null"). */
  function add(node, ...items) {
    items.flat().forEach((c) => {
      if (c === null || c === undefined || c === false) return;
      node.append(typeof c === "number" ? String(c) : c);
    });
    return node;
  }

  const AGREE_SYM = { "Acceptable Match": "=", "Weak Match": "≈", "Model Disagreement": "≠", "Uncertain Result": "?", "Single-Model Result": "1" };
  const chips = {
    sev(s) { return s ? el("span", { class: `chip sev-${s}`, title: `Severity: ${s}` }, el("span", { class: "glyph" }), s) : el("span", { class: "muted", text: "-" }); },
    agree(s) { return s ? el("span", { class: `chip agree agree-${s.replace(/ /g, "-")}` }, el("span", { class: "sym", text: AGREE_SYM[s] || "?" }), s) : el("span", { class: "muted", text: "-" }); },
    quality(q) { return q ? el("span", { class: `chip q-${q}` }, el("span", { class: "qbars" }, el("i"), el("i"), el("i"), el("i")), q) : el("span", { class: "muted", text: "-" }); },
    status(s) { return s ? el("span", { class: `chip status status-${s.replace(/ /g, "-")}` }, el("span", { class: "glyph" }), s) : null; },
    cat(name, codes, large) {
      const code = (codes || window.SS_CLASS_CODES || {})[name];
      // A species is shown under its category: "Animal Sound / Dog Bark".
      const parent = (window.SS_CLASS_PARENTS || {})[name];
      return el("span", { class: large ? "cat cat-lg" : "cat" },
        parent ? el("span", { class: "parent", text: `${parent} / ` }) : null,
        name || "-", code ? el("span", { class: "code", text: code }) : null);
    },
  };

  /* ---------- toasts & alert feed ---------- */
  const toastBox = $("#toasts");
  function toast({ title, body, severity = "Critical", href, timeout = 12000 }) {
    if (!toastBox) return;
    const t = el("div", { class: `toast sev-${severity}`, role: "alert" },
      el("span", { class: "stripe" }),
      el("div", { class: "body" }, el("b", { text: title }), body ? el("div", { text: body }) : null,
        href ? el("a", { href, text: "Open event" }) : null),
      el("button", { class: "close", "aria-label": "Dismiss", text: "×", onclick: () => t.remove() }));
    toastBox.prepend(t);
    while (toastBox.children.length > 4) toastBox.lastChild.remove();
    if (timeout) setTimeout(() => t.remove(), timeout);
  }

  let audioCtx = null;
  function chime(severity) {
    if (!config.alertSound) return;
    try {
      audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
      const now = audioCtx.currentTime;
      const tones = severity === "Critical" ? [880, 660, 880] : [660, 880];
      tones.forEach((f, i) => {
        const o = audioCtx.createOscillator();
        const g = audioCtx.createGain();
        o.type = "sine";
        o.frequency.value = f;
        g.gain.setValueAtTime(0.0001, now + i * 0.18);
        g.gain.exponentialRampToValueAtTime(0.18, now + i * 0.18 + 0.02);
        g.gain.exponentialRampToValueAtTime(0.0001, now + i * 0.18 + 0.16);
        o.connect(g).connect(audioCtx.destination);
        o.start(now + i * 0.18);
        o.stop(now + i * 0.18 + 0.18);
      });
    } catch (e) { /* audio not allowed */ }
  }

  function setBadge(name, value) {
    $$(`[data-badge="${name}"]`).forEach((b) => {
      b.textContent = value;
      b.classList.toggle("hidden", !value);
    });
  }

  let lastAlertId = null;
  const seen = new Set();
  async function pollAlerts() {
    if (!config.feedUrl || document.hidden) return;
    try {
      const data = await api(`${config.feedUrl}?since=${lastAlertId === null ? 0 : lastAlertId}`);
      setBadge("alerts", data.active);
      if (lastAlertId === null) {
        lastAlertId = data.latest_id;  // first poll: remember where we are, do not replay old alerts
        return;
      }
      for (const a of data.alerts) {
        if (seen.has(a.id)) continue;
        seen.add(a.id);
        toast({ title: `${a.severity}: ${a.category}`, body: a.message, severity: a.severity, href: a.url });
        chime(a.severity);
        document.dispatchEvent(new CustomEvent("ss:alert", { detail: a }));
      }
      lastAlertId = Math.max(lastAlertId, data.latest_id);
    } catch (e) { /* offline or signed out: try again later */ }
  }

  /* ---------- theme ---------- */
  function currentTheme() {
    const t = document.documentElement.getAttribute("data-theme");
    if (t) return t;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  function applyThemeIcon() {
    const use = $("[data-theme-toggle] use");
    if (use) use.setAttribute("href", currentTheme() === "dark" ? "#i-sun" : "#i-moon");
  }
  function toggleTheme(e) {
    const next = currentTheme() === "dark" ? "light" : "dark";
    const apply = () => {
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("ss-theme", next); } catch (err) { /* ignore */ }
      applyThemeIcon();
      document.dispatchEvent(new CustomEvent("ss:theme", { detail: next }));
    };
    const full = !window.SSMotion || window.SSMotion.full;
    if (!full || !document.startViewTransition) { apply(); return; }
    // New theme spreads out in a circle from the button.
    const x = e ? e.clientX : window.innerWidth - 60, y = e ? e.clientY : 28;
    const r = Math.hypot(Math.max(x, window.innerWidth - x), Math.max(y, window.innerHeight - y));
    const root = document.documentElement;
    root.classList.add("vt-theme");
    const vt = document.startViewTransition(apply);
    vt.ready.then(() => root.animate({ clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${r}px at ${x}px ${y}px)`] },
      { duration: 700, easing: "cubic-bezier(.2, .7, .1, 1)", pseudoElement: "::view-transition-new(root)" })).catch(() => {});
    vt.finished.finally(() => root.classList.remove("vt-theme"));
  }

  /* ---------- chart colours from CSS tokens ---------- */
  function token(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function chartTheme() {
    return {
      ink: token("--ink"), ink2: token("--ink-2"), ink3: token("--ink-3"), surface: token("--surface"),
      grid: token("--grid"), axis: token("--axis"), data: token("--data"), dataMuted: token("--data-muted"),
      python: token("--python"), gtm: token("--gtm"), good: token("--good"), warning: token("--warning"),
      serious: token("--serious"), critical: token("--critical"), neutral: token("--neutral"),
    };
  }

  /* ---------- wiring ---------- */
  document.addEventListener("click", (e) => {
    const navBtn = e.target.closest("[data-toggle-nav]");
    if (navBtn) { document.body.classList.toggle("nav-open"); return; }
    if (document.body.classList.contains("nav-open") && !e.target.closest("#sidebar")) document.body.classList.remove("nav-open");
    const menuBtn = e.target.closest("[data-menu-button]");
    $$("[data-menu].open").forEach((m) => { if (!menuBtn || !m.contains(menuBtn)) m.classList.remove("open"); });
    if (menuBtn) {
      const m = menuBtn.closest("[data-menu]");
      m.classList.toggle("open");
      menuBtn.setAttribute("aria-expanded", m.classList.contains("open"));
    }
    if (e.target.closest("[data-theme-toggle]")) toggleTheme(e);
    if (e.target.closest("[data-motion-toggle]") && window.SSMotion) window.SSMotion.toggle();
    const tv = e.target.closest("[data-table-toggle]");
    if (tv) {
      const card = tv.closest(".chart-card");
      card.classList.toggle("show-table");
      tv.setAttribute("aria-pressed", card.classList.contains("show-table"));
      tv.textContent = card.classList.contains("show-table") ? "Chart" : "Table";
    }
    const opener = e.target.closest("[data-open-dialog]");
    if (opener) {
      const dlg = document.getElementById(opener.getAttribute("data-open-dialog"));
      if (dlg && dlg.showModal) dlg.showModal();
    }
    const closer = e.target.closest("[data-close-dialog]");
    if (closer) {
      const dlg = closer.closest("dialog");
      if (dlg) dlg.close();
    }
    const confirmEl = e.target.closest("[data-confirm]");
    if (confirmEl && !window.confirm(confirmEl.getAttribute("data-confirm"))) e.preventDefault();
  });
  document.addEventListener("change", (e) => {
    const auto = e.target.closest("[data-autosubmit]");
    if (auto && auto.form) auto.form.submit();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      $$("[data-menu].open").forEach((m) => m.classList.remove("open"));
      document.body.classList.remove("nav-open");
    }
  });

  try {
    if (localStorage.getItem("ss-live") === "on") $("#live-pill") && $("#live-pill").classList.remove("hidden");
  } catch (e) { /* ignore */ }

  applyThemeIcon();
  localiseTimes();
  setInterval(() => localiseTimes(), 60000);
  if (config.feedUrl) {
    pollAlerts();
    setInterval(pollAlerts, 5000);
  }

  window.SS = { api, fmt, el, add, chips, toast, chime, setBadge, localiseTimes, chartTheme, token, csrf, config, $, $$ };
})();
