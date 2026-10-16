/* SonicSentinel public page: motion and interaction.
 * GSAP + ScrollTrigger + SplitText + Lenis (vendored, loaded before this module)
 * and the Three.js sound field. Everything degrades to a static, readable page
 * when motion is reduced or a library is missing. */
import { mountSoundField, SIGNATURES } from "./soundfield.js";

const { gsap, ScrollTrigger, SplitText } = window;
// Full motion unless the visitor chose "reduced" (see static/js/theme.js).
const reduce = !(window.SSMotion ? window.SSMotion.full : true);
const motion = !!(gsap && ScrollTrigger && SplitText && !reduce);
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const CLASSES = JSON.parse(($("#lp-classes") || {}).textContent || "[]");
const BY_NAME = Object.fromEntries(CLASSES.map((c) => [c.name, c]));
const SEV = { Critical: "crit", High: "high", Medium: "med", Low: "low", Informational: "info" };

if (gsap) gsap.registerPlugin(ScrollTrigger, SplitText);

/* ---------------------------------------------------------------- smooth scroll */
let lenis = null;
if (motion && window.Lenis) {
  lenis = new window.Lenis({ lerp: 0.095, wheelMultiplier: 0.95 });
  lenis.on("scroll", ScrollTrigger.update);
  gsap.ticker.add((t) => lenis.raf(t * 1000));
  gsap.ticker.lagSmoothing(0);
}
function scrollToTarget(el) {
  if (lenis) lenis.scrollTo(el, { offset: -40, duration: 1.5 });
  else el.scrollIntoView({ behavior: reduce ? "auto" : "smooth" });
}
$$('a[data-link][href^="#"]').forEach((a) => a.addEventListener("click", (e) => {
  const target = $(a.getAttribute("href"));
  if (target) { e.preventDefault(); scrollToTarget(target); }
}));

/* ---------------------------------------------------------------- hero sound field + HUD */
const hud = { name: $("[data-hud-event]"), code: $("[data-hud-code]"), sev: $("[data-hud-sev]"), box: $(".lp-hud-now") };
const meter = $$(".lp-hud-meter i");
function showEvent(name, on) {
  const c = on ? BY_NAME[name] : BY_NAME["Background Noise"];
  if (!c || !hud.name) return;
  hud.name.textContent = on ? (c.display_name || c.name) : "Background noise";
  hud.code.textContent = c.code;
  hud.sev.textContent = c.severity;
  hud.sev.className = `lp-sev lp-sev-${SEV[c.severity] || "low"}`;
  hud.box.classList.toggle("is-event", on);
  if (motion) gsap.fromTo(hud.name, { yPercent: 70, opacity: 0 }, { yPercent: 0, opacity: 1, duration: 0.5, ease: "power3.out" });
}
let levelTick = 0;
function showLevel(level) {
  if (++levelTick % 2) return;
  const n = Math.round(level * meter.length);
  meter.forEach((m, i) => { m.classList.toggle("on", i < n); m.classList.toggle("hot", i >= meter.length - 5); });
}
const fieldEl = $("#soundfield");
const field = fieldEl ? mountSoundField(fieldEl, { onEvent: showEvent, onLevel: showLevel, animate: !reduce, interactive: motion }) : null;

/* ---------------------------------------------------------------- motion switch + nav state */
const motionBtn = $("[data-motion-toggle]");
if (motionBtn && window.SSMotion) {
  motionBtn.setAttribute("aria-pressed", String(window.SSMotion.full));
  motionBtn.addEventListener("click", () => window.SSMotion.toggle());
}
const nav = $("[data-nav]");
let lastY = 0;
function onScroll(y) {
  nav.classList.toggle("is-solid", y > 40);
  nav.classList.toggle("is-hidden", motion && y > lastY && y > 500);
  lastY = y;
}
if (lenis) lenis.on("scroll", ({ scroll }) => onScroll(scroll));
else window.addEventListener("scroll", () => onScroll(window.scrollY), { passive: true });
onScroll(window.scrollY);

const clock = $("[data-clock]");
function tickClock() {
  if (clock) clock.textContent = new Date().toISOString().slice(11, 19) + " UTC";
}
tickClock();
setInterval(tickClock, 1000);

/* ---------------------------------------------------------------- cursor + magnetic buttons */
if (motion && window.matchMedia("(hover: hover) and (pointer: fine)").matches) {
  document.documentElement.classList.add("has-cursor");
  const cursor = $(".lp-cursor");
  const cx = gsap.quickTo(cursor, "x", { duration: 0.35, ease: "power3.out" });
  const cy = gsap.quickTo(cursor, "y", { duration: 0.35, ease: "power3.out" });
  window.addEventListener("pointermove", (e) => { cx(e.clientX); cy(e.clientY); cursor.classList.add("is-live"); }, { passive: true });
  document.addEventListener("pointerleave", () => cursor.classList.remove("is-live"));
  document.addEventListener("pointerover", (e) => {
    cursor.classList.toggle("is-hover", !!e.target.closest("a, button, summary, .lp-sound, [data-magnetic]"));
  });
  $$("[data-magnetic]").forEach((el) => {
    const mx = gsap.quickTo(el, "x", { duration: 0.5, ease: "power3.out" });
    const my = gsap.quickTo(el, "y", { duration: 0.5, ease: "power3.out" });
    el.addEventListener("pointermove", (e) => {
      const r = el.getBoundingClientRect();
      mx((e.clientX - r.left - r.width / 2) * 0.25);
      my((e.clientY - r.top - r.height / 2) * 0.35);
    });
    el.addEventListener("pointerleave", () => gsap.to(el, { x: 0, y: 0, duration: 0.9, ease: "elastic.out(1, 0.45)" }));
  });
}

/* ---------------------------------------------------------------- intro + hero entrance */
function heroIn() {
  if (!motion) return;
  const split = SplitText.create("[data-hero-title]", { type: "lines,chars", mask: "lines" });
  gsap.from(split.chars, { yPercent: 115, duration: 1.15, ease: "power4.out", stagger: 0.016 });
  gsap.from("[data-hero-fade]", { opacity: 0, y: 26, duration: 1.1, ease: "power3.out", stagger: 0.09, delay: 0.3 });
  // the nav element itself keeps its own transform for hide / show on scroll
  gsap.from(".lp-nav > :not(.lp-progress)", { y: -24, opacity: 0, duration: 0.9, ease: "power3.out", stagger: 0.08, delay: 0.15 });
}
function playIntro(done) {
  let seen = true;
  try { seen = !!sessionStorage.getItem("ss-intro"); sessionStorage.setItem("ss-intro", "1"); } catch (e) { /* private mode */ }
  const intro = $(".lp-intro");
  if (!motion || seen || !intro) { document.documentElement.classList.add("no-intro"); done(); return; }
  const count = $("[data-intro-count]");
  const n = { v: 0 };
  gsap.timeline({ onComplete: () => intro.remove() })
    .to(".lp-intro-trace", { scaleX: 1, duration: 0.95, ease: "power2.inOut" })
    .to(n, { v: 100, duration: 0.95, ease: "power2.inOut", onUpdate: () => { count.textContent = String(Math.round(n.v)).padStart(3, "0"); } }, 0)
    .to(".lp-intro-read", { opacity: 0, duration: 0.2 })
    .to(intro, { clipPath: "inset(50% 0% 50% 0%)", duration: 0.75, ease: "power4.inOut" }, "<")
    .add(done, "-=0.45");
}

/* ---------------------------------------------------------------- sound signatures (2D canvas) */
function hexToRgb(h) {
  const m = h.replace("#", "").match(/.{2}/g) || ["0", "0", "0"];
  return m.map((x) => parseInt(x, 16));
}
const ramp = [
  [0, hexToRgb(cssVar("--carbon-2") || "#121310")],
  [0.3, hexToRgb(cssVar("--bone-4") || "#5a564d")],
  [0.6, hexToRgb(cssVar("--bone") || "#ece6d9")],
  [1, hexToRgb(cssVar("--signal") || "#ff5a1f")],
];
function colour(v) {
  v = Math.max(0, Math.min(1, v));
  for (let i = 1; i < ramp.length; i++) {
    if (v <= ramp[i][0]) {
      const [a, ca] = ramp[i - 1], [b, cb] = ramp[i], t = (v - a) / (b - a);
      return ca.map((c, k) => Math.round(c + (cb[k] - c) * t));
    }
  }
  return ramp[ramp.length - 1][1];
}
function buildSignature(slug) {
  const sig = SIGNATURES[slug];
  const w = 180, h = 72;
  const img = document.createElement("canvas");
  img.width = w; img.height = h;
  const ctx = img.getContext("2d");
  const data = ctx.createImageData(w, h);
  for (let x = 0; x < w; x++) {
    const t = (x / w) * sig.dur;
    for (let y = 0; y < h; y++) {
      const f = 0.6 * (1 - y / h);
      const [r, g, b] = colour(Math.min(1, sig.fn(t, f) * 0.95));
      const o = (y * w + x) * 4;
      data.data[o] = r; data.data[o + 1] = g; data.data[o + 2] = b; data.data[o + 3] = 255;
    }
  }
  ctx.putImageData(data, 0, 0);
  return img;
}
function mountSignature(canvas) {
  const slug = canvas.dataset.sig;
  if (!SIGNATURES[slug]) return;
  const src = buildSignature(slug);
  const ctx = canvas.getContext("2d");
  const state = { scan: -1 };
  function draw() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.globalAlpha = state.scan >= 0 ? 0.55 : 1;
    ctx.drawImage(src, 0, 0, w, h);
    ctx.globalAlpha = 1;
    if (state.scan >= 0) {
      const x = state.scan * w;
      ctx.save();
      ctx.beginPath(); ctx.rect(0, 0, x, h); ctx.clip();
      ctx.drawImage(src, 0, 0, w, h);
      ctx.restore();
      ctx.fillStyle = cssVar("--signal") || "#ff5a1f";
      ctx.fillRect(Math.min(w - 1, x), 0, 1.5, h);
    }
  }
  draw();
  new ResizeObserver(draw).observe(canvas);
  if (!motion) return;
  let tween = null;
  const card = canvas.closest(".lp-sound");
  const sweep = (repeat) => {
    if (tween) tween.kill();
    tween = gsap.fromTo(state, { scan: 0 }, { scan: 1, duration: 1.6, ease: "none", repeat, onUpdate: draw,
      onComplete: () => { state.scan = -1; draw(); } });
  };
  card.addEventListener("pointerenter", () => sweep(-1));
  card.addEventListener("pointerleave", () => { if (tween) tween.kill(); state.scan = -1; draw(); });
  ScrollTrigger.create({ trigger: card, start: "top 85%", once: true, onEnter: () => sweep(0) });
  // gentle 3D tilt
  card.addEventListener("pointermove", (e) => {
    const r = card.getBoundingClientRect();
    gsap.to(card, { rotateY: ((e.clientX - r.left) / r.width - 0.5) * 6, rotateX: -((e.clientY - r.top) / r.height - 0.5) * 6,
      transformPerspective: 800, duration: 0.5, ease: "power2.out" });
  });
  card.addEventListener("pointerleave", () => gsap.to(card, { rotateX: 0, rotateY: 0, duration: 0.7, ease: "power3.out" }));
}
$$("canvas[data-sig]").forEach(mountSignature);

/* ---------------------------------------------------------------- verdict cycling */
const VERDICTS = [
  ["=", "Acceptable match", "Same class, similar confidence. The alert rule decides what happens next."],
  ["≈", "Weak match", "Same class, but one model is much less sure. Flagged for a second look."],
  ["≠", "Disagreement", "Different classes. The segment goes to the review queue."],
  ["?", "Uncertain", "Neither model is sure. Logged as an unknown sound and reviewed."],
];
function cycleVerdict() {
  const sym = $("[data-verdict-sym]"), name = $("[data-verdict-name]"), text = $("[data-verdict-text]");
  if (!sym || !motion) return;
  let i = 0;
  const next = () => {
    i = (i + 1) % VERDICTS.length;
    gsap.timeline()
      .to([sym, name, text], { opacity: 0, y: -10, duration: 0.3, ease: "power2.in", stagger: 0.04 })
      .add(() => { sym.textContent = VERDICTS[i][0]; name.textContent = VERDICTS[i][1]; text.textContent = VERDICTS[i][2]; })
      .fromTo(sym, { rotateY: -90 }, { rotateY: 0, duration: 0.6, ease: "back.out(1.6)" })
      .to([sym, name, text], { opacity: 1, y: 0, duration: 0.45, ease: "power3.out", stagger: 0.05 }, "<");
  };
  let timer = null;
  ScrollTrigger.create({ trigger: ".lp-duel", start: "top 80%", end: "bottom 20%",
    onToggle: (st) => { clearInterval(timer); if (st.isActive) timer = setInterval(next, 3200); } });
}

/* ---------------------------------------------------------------- console preview (oscilloscope) */
function mountConsole() {
  const canvas = $("[data-scope]");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const out = { win: $("[data-console-win]"), db: $("[data-console-db]"), py: $("[data-console-py]"), gtm: $("[data-console-gtm]"), fin: $("[data-console-final]") };
  const scene = ["Glass Breaking", "Vehicle Horn", "Animal Sound", "Alarm or Siren"];
  let t = 0, win = 1, ev = null, evT = 0, next = 2.2, raf = 0, visible = false, last = 0;
  const bone = cssVar("--bone") || "#ece6d9", dim = cssVar("--rule") || "#2b2c26", sig = cssVar("--signal") || "#ff5a1f";
  function setReadouts() {
    const c = ev ? BY_NAME[ev] : null;
    const label = c ? (c.display_name || c.name) : "Background noise";
    [out.py, out.gtm].forEach((el) => { el.textContent = label; el.classList.toggle("is-hot", !!c); });
    out.fin.textContent = c ? (c.alerts ? `${c.severity} alert` : "Logged") : "No event";
    out.fin.classList.toggle("is-hot", !!(c && c.alerts));
  }
  function frame(now) {
    raf = 0;
    const dt = Math.min(0.05, (now - (last || now)) / 1000);
    last = now; t += dt;
    if (!ev && t > next) { ev = scene[Math.floor(t) % scene.length]; evT = 0; setReadouts(); }
    if (ev) { evT += dt; if (evT > 1.6) { ev = null; next = t + 1.6; setReadouts(); } }
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = dim; ctx.lineWidth = 1;
    for (let x = 0; x <= w; x += w / 12) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
    for (let y = 0; y <= h; y += h / 6) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
    const amp = ev ? Math.max(0.15, 1 - evT / 1.6) : 0.12;
    ctx.beginPath();
    for (let x = 0; x <= w; x += 2) {
      const u = x / w;
      const y = h / 2 + h * 0.42 * amp * (Math.sin(u * 60 + t * 9) * 0.5 + Math.sin(u * 173 + t * 23) * 0.3 + (Math.random() - 0.5) * 0.5);
      x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.strokeStyle = ev ? sig : bone; ctx.lineWidth = 1.4; ctx.stroke();
    if (Math.floor(t) !== win - 1) { win = Math.floor(t) + 1; out.win.textContent = String(win).padStart(4, "0"); }
    out.db.textContent = String(Math.round(-40 + amp * 34));
    if (visible && !document.hidden) raf = requestAnimationFrame(frame);
  }
  setReadouts();
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; if (visible && !raf && !reduce) { last = 0; raf = requestAnimationFrame(frame); } }).observe(canvas);
  if (reduce) frame(performance.now());
}

/* ---------------------------------------------------------------- FAQ accordion */
$$(".lp-q").forEach((q) => {
  const summary = q.querySelector("summary"), body = q.querySelector(".lp-a");
  summary.addEventListener("click", (e) => {
    if (!motion) return;
    e.preventDefault();
    if (q.open) {
      gsap.to(body, { height: 0, opacity: 0, duration: 0.45, ease: "power3.inOut", onComplete: () => { q.open = false; gsap.set(body, { clearProps: "all" }); } });
    } else {
      q.open = true;
      gsap.from(body, { height: 0, opacity: 0, duration: 0.55, ease: "power3.out", clearProps: "all" });
    }
  });
});

/* ---------------------------------------------------------------- scroll choreography */
function scrollScenes() {
  if (!motion) return;
  // hero: camera flies into the terrain, copy lifts away
  ScrollTrigger.create({ trigger: ".lp-hero", start: "top top", end: "bottom top", scrub: true,
    onUpdate: (st) => field && field.setProgress(st.progress) });
  gsap.to(".lp-hero-inner", { yPercent: -35, opacity: 0, ease: "none",
    scrollTrigger: { trigger: ".lp-hero", start: "top top", end: "70% top", scrub: true } });
  gsap.to(".lp-hud", { y: -60, opacity: 0, ease: "none",
    scrollTrigger: { trigger: ".lp-hero", start: "10% top", end: "60% top", scrub: true } });

  // scroll progress + active section link
  gsap.to(".lp-progress span", { scaleX: 1, ease: "none", scrollTrigger: { start: 0, end: "max", scrub: 0.3 } });
  $$("[data-link]").forEach((a) => {
    const id = a.getAttribute("href");
    if (!id || id.length < 2 || !$(id) || !a.closest(".lp-links")) return;
    ScrollTrigger.create({ trigger: id, start: "top 55%", end: "bottom 55%",
      onToggle: (st) => a.classList.toggle("is-active", st.isActive) });
  });

  // section titles: lines rise out of a mask
  $$("[data-reveal-title]").forEach((el) => {
    const s = SplitText.create(el, { type: "lines", mask: "lines" });
    gsap.from(s.lines, { yPercent: 110, duration: 1.2, ease: "power4.out", stagger: 0.09,
      scrollTrigger: { trigger: el, start: "top 86%" } });
  });
  $$(".lp-index, .lp-sec-lede").forEach((el) => gsap.from(el, { opacity: 0, y: 18, duration: 0.9, ease: "power3.out",
    scrollTrigger: { trigger: el, start: "top 90%" } }));

  // manifesto: words light up as they are read
  const statement = $("[data-words]");
  if (statement) {
    const words = SplitText.create(statement, { type: "words", wordsClass: "w" }).words;
    gsap.to(words, { color: cssVar("--bone"), stagger: 0.12, ease: "none",
      scrollTrigger: { trigger: statement, start: "top 78%", end: "bottom 42%", scrub: true } });
  }

  // cards: rise in batches
  gsap.set("[data-reveal-card]", { opacity: 0, y: 48 });
  ScrollTrigger.batch("[data-reveal-card]", { start: "top 90%", once: true,
    onEnter: (batch) => gsap.to(batch, { opacity: 1, y: 0, duration: 1, ease: "power3.out", stagger: 0.07 }) });

  // model bars fill, stats count up
  $$(".lp-bars b").forEach((b) => gsap.from(b, { scaleX: 0, duration: 1.4, ease: "power3.out",
    scrollTrigger: { trigger: b.closest(".lp-bars"), start: "top 85%" } }));
  $$("[data-count]").forEach((el) => {
    const target = parseFloat(el.dataset.count);
    const decimals = (el.dataset.count.split(".")[1] || "").length;
    const n = { v: 0 };
    gsap.to(n, { v: target, duration: 1.8, ease: "power2.out",
      scrollTrigger: { trigger: el, start: "top 90%", once: true },
      onUpdate: () => { el.textContent = decimals ? n.v.toFixed(decimals) : Math.round(n.v).toLocaleString("en-US"); } });
  });
  gsap.from(".lp-num", { opacity: 0, y: 40, duration: 1, ease: "power3.out", stagger: 0.08,
    scrollTrigger: { trigger: ".lp-numbers", start: "top 85%" } });

  // alert flow: the line fills and each step lights up
  const flow = $("[data-flow]");
  if (flow) {
    const items = $$("li", flow);
    ScrollTrigger.create({ trigger: flow, start: "top 70%", end: "bottom 60%", scrub: true,
      onUpdate: (st) => {
        flow.style.setProperty("--flow", st.progress.toFixed(3));
        items.forEach((li, i) => li.classList.toggle("is-on", st.progress >= i / (items.length - 1) - 0.02));
      } });
    gsap.from(items, { opacity: 0, x: 30, duration: 0.9, ease: "power3.out", stagger: 0.12,
      scrollTrigger: { trigger: flow, start: "top 80%" } });
  }
  gsap.from(".lp-console", { opacity: 0, y: 60, rotateX: 8, transformPerspective: 1200, duration: 1.3, ease: "power3.out",
    scrollTrigger: { trigger: ".lp-console", start: "top 85%" } });
  gsap.from(".lp-ethics-grid .icon", { scale: 0, rotate: -90, duration: 0.8, ease: "back.out(2)", stagger: 0.08,
    scrollTrigger: { trigger: ".lp-ethics-grid", start: "top 85%" } });
  gsap.from(".lp-q", { opacity: 0, y: 24, duration: 0.8, ease: "power3.out", stagger: 0.07,
    scrollTrigger: { trigger: ".lp-faq-list", start: "top 85%" } });

  // pipeline: pinned horizontal travel on wide screens, drawn illustrations
  const mm = gsap.matchMedia();
  mm.add("(min-width: 901px)", () => {
    const track = $("[data-track]");
    const now = $("[data-step-now]");
    const bar = $("[data-pipe-bar]");
    const steps = $$(".lp-step");
    const distance = () => Math.max(0, track.scrollWidth - window.innerWidth);
    const travel = gsap.to(track, { x: () => -distance(), ease: "none",
      scrollTrigger: { trigger: ".lp-pipeline", start: "top top", end: () => `+=${distance()}`, pin: true, scrub: 0.8,
        invalidateOnRefresh: true,
        onUpdate: (st) => {
          bar.style.transform = `scaleX(${st.progress})`;
          now.textContent = String(Math.min(steps.length, 1 + Math.floor(st.progress * steps.length * 0.999))).padStart(2, "0");
        } } });
    steps.forEach((step) => drawArt(step, { containerAnimation: travel, trigger: step, start: "left 80%" }));
  });
  mm.add("(max-width: 900px)", () => {
    $$(".lp-step").forEach((step) => drawArt(step, { trigger: step, start: "top 80%" }));
  });

  // marquee: endless, faster while scrolling
  const track = $("[data-marquee]");
  if (track) {
    const loop = gsap.to(track, { xPercent: -50, duration: 28, ease: "none", repeat: -1 });
    ScrollTrigger.create({ trigger: ".lp-final", start: "top bottom", end: "bottom top",
      onUpdate: (st) => {
        const boost = 1 + Math.min(5, Math.abs(st.getVelocity()) / 300);
        gsap.to(loop, { timeScale: boost * (st.direction < 0 ? -1 : 1), duration: 0.2, overwrite: true,
          onComplete: () => gsap.to(loop, { timeScale: st.direction < 0 ? -1 : 1, duration: 1.2 }) });
      } });
  }
  gsap.from(".lp-final-inner .lp-btn, .lp-demo", { opacity: 0, y: 30, duration: 1, ease: "power3.out", stagger: 0.1,
    scrollTrigger: { trigger: ".lp-final-inner", start: "top 80%" } });
}
function drawArt(step, trigger) {
  const paths = $$(".lp-art .d", step);
  paths.forEach((p) => {
    const len = Math.ceil(p.getTotalLength());
    p.style.strokeDasharray = `${len} ${len}`;
    p.style.strokeDashoffset = len;
  });
  gsap.to(paths, { strokeDashoffset: 0, duration: 1.6, ease: "power2.inOut", stagger: 0.12, scrollTrigger: trigger });
  gsap.from(step.querySelector(".lp-step-no"), { opacity: 0, x: -20, duration: 0.8, ease: "power3.out", scrollTrigger: trigger });
}

/* ---------------------------------------------------------------- start */
mountConsole();
const ready = document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve();
ready.then(() => {
  scrollScenes();
  cycleVerdict();
  playIntro(heroIn);
  if (motion) ScrollTrigger.refresh();
  if (window.SSMotion) window.SSMotion.prompt();
});
