/* Console motion layer (GSAP, vendored).
 * Page entrance, scroll reveals, count-ups, growing bars, cascading rows,
 * a gliding highlight in the sidebar and a navigation progress line.
 * Works on every page without per-page code; switched off when the visitor
 * chose reduced motion (static/js/theme.js). */
(function () {
  "use strict";
  const root = document.documentElement;
  const gsap = window.gsap;
  const full = !window.SSMotion || window.SSMotion.full;
  const content = document.getElementById("content");
  const showContent = () => root.classList.remove("m-pending");

  // Menu label of the motion switch.
  document.querySelectorAll("[data-motion-toggle] .switch-state").forEach((s) => { s.textContent = full ? "On" : "Off"; });
  if (window.SSMotion) window.SSMotion.prompt();
  if (!gsap || !full || !content) { showContent(); return; }

  const $$ = (sel, r) => Array.from((r || document).querySelectorAll(sel));
  const ease = "power3.out";

  /* ---------- helpers ---------- */
  function countUp(el) {
    if (el.dataset.counted) return;
    const node = Array.from(el.childNodes).find((n) => n.nodeType === 3 && /\d/.test(n.textContent));
    if (!node) return;
    const m = node.textContent.match(/^(\s*)(-?[\d,]*\.?\d+)(.*)$/s);
    if (!m) return;
    const target = parseFloat(m[2].replace(/,/g, ""));
    if (!isFinite(target) || Math.abs(target) > 1e7) return;
    const decimals = (m[2].split(".")[1] || "").length;
    const commas = m[2].includes(",");
    el.dataset.counted = "1";
    const n = { v: 0 };
    gsap.to(n, {
      v: target, duration: 1.4, ease: "power2.out",
      onUpdate: () => {
        let s = decimals ? n.v.toFixed(decimals) : String(Math.round(n.v));
        if (commas) s = Number(s).toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
        node.textContent = m[1] + s + m[3];
      },
    });
  }
  function animateInside(block, delay) {
    const d = delay || 0;
    gsap.from($$(".kpi", block), { opacity: 0, y: 16, duration: 0.7, ease, stagger: 0.06, delay: d });
    $$(".kpi .value, .stat-strip .v, .hero-figure, .confidence-dial .v, [data-count-up]", block).forEach(countUp);
    $$(".bar > span, .progress > span, .stackbar > span", block).forEach((b, i) =>
      gsap.from(b, { width: 0, duration: 1.1, ease: "power3.inOut", delay: d + 0.15 + Math.min(i, 20) * 0.02 }));
    $$(".minibar", block).forEach((b) => gsap.from(b, { width: 0, duration: 1, ease: "power3.inOut", delay: d + 0.2 }));
    $$("table.table", block).forEach((t) => gsap.from($$("tbody tr", t).slice(0, 16), { opacity: 0, x: -10, duration: 0.5, ease, stagger: 0.03, delay: d + 0.1 }));
    gsap.from($$(".alert-card", block).slice(0, 12), { opacity: 0, x: 18, duration: 0.6, ease, stagger: 0.05, delay: d + 0.1 });
    gsap.from($$(".tl li", block).slice(0, 20), { opacity: 0, x: -12, duration: 0.5, ease, stagger: 0.03, delay: d + 0.1 });
    gsap.from($$(".strip24 .mark", block), { opacity: 0, y: -16, duration: 0.6, ease: "back.out(2.2)", stagger: 0.04, delay: d + 0.2 });
    gsap.from($$(".file-item, .conditions li, .kv dt, .kv dd, .top3 li, .conf-table tbody tr", block).slice(0, 30),
      { opacity: 0, y: 8, duration: 0.45, ease, stagger: 0.015, delay: d + 0.1 });
    gsap.from($$(".cat-lg", block), { opacity: 0, y: 14, duration: 0.7, ease, delay: d + 0.05 });
  }

  /* ---------- page entrance ---------- */
  const blocks = Array.from(content.children).filter((b) => b.nodeType === 1 && b.tagName !== "SCRIPT" && b.tagName !== "DIALOG");
  gsap.set(blocks, { opacity: 0 });
  showContent();
  const vh = window.innerHeight;
  const head = content.querySelector(":scope > .page-head");
  const tl = gsap.timeline({ defaults: { ease } });
  if (head) {
    tl.set(head, { opacity: 1 });
    const h1 = head.querySelector("h1");
    if (h1 && window.SplitText) {
      const split = window.SplitText.create(h1, { type: "lines,chars", mask: "lines" });
      tl.from(split.chars, { yPercent: 110, duration: 0.8, stagger: 0.018, ease: "power4.out" }, 0);
    }
    tl.from(head.querySelectorAll(".eyebrow, .lede, .actions > *"), { opacity: 0, y: 12, duration: 0.6, stagger: 0.06 }, 0.1);
  }
  let i = 0;
  const later = [];
  blocks.forEach((b) => {
    if (b === head) return;
    if (b.getBoundingClientRect().top < vh * 0.92) {
      const at = 0.18 + i * 0.07;
      tl.fromTo(b, { opacity: 0, y: 26 }, { opacity: 1, y: 0, duration: 0.8 }, at);
      animateInside(b, at);
      i++;
    } else {
      later.push(b);
    }
  });

  /* ---------- scroll reveals for everything below the fold ---------- */
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (!e.isIntersecting) return;
      io.unobserve(e.target);
      gsap.fromTo(e.target, { opacity: 0, y: 34 }, { opacity: 1, y: 0, duration: 0.9, ease });
      animateInside(e.target, 0.05);
    });
  }, { rootMargin: "0px 0px -8% 0px" });
  later.forEach((b) => io.observe(b));

  /* ---------- sidebar: first visit entrance + gliding highlight ---------- */
  const nav = document.querySelector(".sidebar nav");
  if (nav) {
    let first = true;
    try { first = !sessionStorage.getItem("ss-shell"); sessionStorage.setItem("ss-shell", "1"); } catch (e) { /* ignore */ }
    if (first) {
      gsap.from(".sidebar .brand, .sidebar .nav-section-title, .sidebar .nav-item, .sidebar-foot", { opacity: 0, x: -14, duration: 0.6, ease, stagger: 0.025 });
      gsap.from(".topbar > *", { opacity: 0, y: -10, duration: 0.6, ease, stagger: 0.04, delay: 0.1 });
    }
    const glide = document.createElement("div");
    glide.className = "nav-glide";
    nav.prepend(glide);
    const move = (el) => gsap.to(glide, { y: el.offsetTop, height: el.offsetHeight, opacity: 1, duration: 0.35, ease });
    $$(".nav-item", nav).forEach((it) => it.addEventListener("pointerenter", () => { if (!it.classList.contains("active")) move(it); else gsap.to(glide, { opacity: 0, duration: 0.2 }); }));
    nav.addEventListener("pointerleave", () => gsap.to(glide, { opacity: 0, duration: 0.25 }));
  }

  /* ---------- navigation progress line on the top bar ---------- */
  const bar = document.querySelector(".topbar");
  function loading() {
    if (!bar) return;
    bar.style.setProperty("--load-o", "1");
    requestAnimationFrame(() => bar.style.setProperty("--load", "0.72"));
  }
  document.addEventListener("click", (e) => {
    const a = e.target.closest("a[href]");
    if (!a || e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || a.target || a.hasAttribute("download")) return;
    const url = new URL(a.href, window.location.href);
    if (url.origin !== window.location.origin || url.pathname.startsWith("/static/")) return;
    if (url.pathname === window.location.pathname && url.search === window.location.search) return;
    loading();
  });
  // Checked a tick later: page scripts may cancel the submit and send it with fetch instead.
  document.addEventListener("submit", (e) => { setTimeout(() => { if (!e.defaultPrevented && e.target.method !== "dialog") loading(); }); });
  window.addEventListener("pageshow", () => { if (bar) { bar.style.setProperty("--load", "0"); bar.style.setProperty("--load-o", "0"); } });
})();
