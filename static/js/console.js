/* Acoustic console: the drawings on the administration dashboard.
 *
 * Everything here is made from the numbers the page already carries, so the
 * shapes are the data and not decoration:
 *   - header spectrum: one bar per category, height = its share of detections,
 *     breathing slowly so the panel reads as a live instrument;
 *   - recorder trace: events per hour for the last 24 h, drawn as a strip chart;
 *   - category pattern: a polar plot, the way a microphone's pickup pattern is drawn;
 *   - audio traces: the real stored waveform of the last recordings analysed;
 *   - agreement bar and confidence histogram.
 */
(function () {
  "use strict";
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  // The role desks show the instrument header without the analytics payload, so a missing
  // #console-data only means "draw what is on the page", not "do nothing".
  const cfgEl = $("#console-data");
  let D = {};
  if (cfgEl) { try { D = JSON.parse(cfgEl.textContent || "{}"); } catch (e) { D = {}; } }
  if (!$("#console") && !$("#gauges")) return;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const NS = "http://www.w3.org/2000/svg";
  const css = (name, fallback) =>
    (getComputedStyle(document.documentElement).getPropertyValue(name) || "").trim() || fallback;

  const svg = (tag, attrs) => {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) if (attrs[k] !== null && attrs[k] !== undefined) n.setAttribute(k, attrs[k]);
    return n;
  };

  /* ---------- header spectrum ---------- */
  function spectrum() {
    const c = $("#console-spectrum");
    if (!c) return;
    const cats = (D.categories || []).filter((x) => x.count > 0);
    const bars = cats.length ? cats : Array.from({ length: 24 }, () => ({ count: 1, category: "" }));
    const max = Math.max(...bars.map((b) => b.count), 1);
    // Each category becomes a group of three bars so the strip looks like a spectrum, not a bar chart.
    const cols = [];
    bars.forEach((b) => {
      const base = Math.max(0.08, b.count / max);
      const colour = (D.colors || {})[b.category] || "#ff5a1f";
      for (let k = 0; k < 3; k++) cols.push({ v: base * (0.55 + 0.45 * Math.sin((k + 1) * 1.1)), colour });
    });
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    let t = 0;
    function draw() {
      const w = c.clientWidth;
      const h = c.clientHeight;
      if (!w || !h) return;
      if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
      const g = c.getContext("2d");
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, w, h);
      const n = cols.length;
      const bw = Math.max(2, (w / n) * 0.55);
      const gap = w / n;
      for (let i = 0; i < n; i++) {
        const wob = reduced ? 1 : 0.82 + 0.18 * Math.sin(t / 34 + i * 0.7);
        const bh = Math.max(2, cols[i].v * h * 0.9 * wob);
        const x = i * gap + (gap - bw) / 2;
        const grad = g.createLinearGradient(0, h, 0, h - bh);
        grad.addColorStop(0, cols[i].colour + "00");
        grad.addColorStop(1, cols[i].colour + "cc");
        g.fillStyle = grad;
        g.fillRect(x, h - bh, bw, bh);
      }
      t += 1;
      if (!reduced) requestAnimationFrame(draw);
    }
    draw();
    window.addEventListener("resize", () => { if (reduced) draw(); });
  }

  /* ---------- 24 h recorder trace ---------- */
  function recorderTrace() {
    const el = $("#hour-trace");
    const h = D.hourly;
    if (!el || !h) return;
    const W = 1000;
    const H = 190;
    const pad = 12;
    const n = h.events.length;
    const max = Math.max(...h.events, ...h.background, 1);
    const x = (i) => (n === 1 ? W / 2 : (i / (n - 1)) * (W - pad * 2) + pad);
    const y = (v) => H - pad - (v / max) * (H - pad * 2);

    // baseline grid
    for (let k = 1; k <= 3; k++) {
      const gy = pad + ((H - pad * 2) / 4) * k;
      el.appendChild(svg("line", { x1: 0, x2: W, y1: gy, y2: gy, stroke: css("--rule", "#e3ded2"), "stroke-width": 1 }));
    }
    const line = (vals) => vals.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");

    // background ambience as a soft filled area
    const bgArea = svg("path", {
      d: `${line(h.background)} L${x(n - 1)},${H - pad} L${x(0)},${H - pad} Z`,
      fill: css("--rule-strong", "#cfc8b8"), opacity: .28, stroke: "none",
    });
    el.appendChild(bgArea);

    const evPath = svg("path", { d: line(h.events), fill: "none", stroke: css("--signal", "#ff5a1f"), "stroke-width": 2.2, "stroke-linejoin": "round", "stroke-linecap": "round" });
    el.appendChild(evPath);

    // critical ticks, like event marks on a recorder chart
    h.critical.forEach((v, i) => {
      if (!v) return;
      el.appendChild(svg("line", { x1: x(i), x2: x(i), y1: y(h.events[i]), y2: H - pad,
        stroke: css("--critical", "#c0392b"), "stroke-width": 2, opacity: .75 }));
      el.appendChild(svg("circle", { cx: x(i), cy: y(h.events[i]), r: 3.4, fill: css("--critical", "#c0392b") }));
    });

    if (!reduced) {
      const len = evPath.getTotalLength ? evPath.getTotalLength() : 0;
      if (len) {
        evPath.style.strokeDasharray = len;
        evPath.style.strokeDashoffset = len;
        evPath.style.transition = "stroke-dashoffset 1.5s cubic-bezier(.2,.7,.1,1)";
        requestAnimationFrame(() => { evPath.style.strokeDashoffset = 0; });
      }
    }
    const axis = $("#hour-axis");
    if (axis) {
      [0, Math.floor(n / 3), Math.floor((2 * n) / 3), n - 1].forEach((i) => {
        const s = document.createElement("span");
        s.textContent = h.labels[i] || "";
        axis.appendChild(s);
      });
    }
  }

  /* ---------- polar category pattern ---------- */
  function polar() {
    const el = $("#cat-polar");
    const legend = $("#cat-legend");
    if (!el) return;
    const cats = (D.categories || []).filter((c) => c.count > 0).sort((a, b) => b.count - a.count).slice(0, 16);
    if (!cats.length) return;
    const cx = 210;
    const cy = 210;
    const rMin = 46;
    const rMax = 190;
    const max = Math.max(...cats.map((c) => c.count), 1);

    for (const f of [0.25, 0.5, 0.75, 1]) {
      el.appendChild(svg("circle", { cx, cy, r: rMin + (rMax - rMin) * f, fill: "none",
        stroke: css("--rule", "#e3ded2"), "stroke-width": 1, "stroke-dasharray": f === 1 ? "" : "2 4" }));
    }
    const step = (Math.PI * 2) / cats.length;
    cats.forEach((c, i) => {
      const a0 = i * step - Math.PI / 2 + step * 0.08;
      const a1 = (i + 1) * step - Math.PI / 2 - step * 0.08;
      const r = rMin + (rMax - rMin) * (c.count / max);
      const colour = (D.colors || {})[c.category] || css("--signal", "#ff5a1f");
      const pt = (ang, rad) => `${(cx + Math.cos(ang) * rad).toFixed(2)},${(cy + Math.sin(ang) * rad).toFixed(2)}`;
      const arc = (rad, sweep) => `A${rad},${rad} 0 0 ${sweep} `;
      const wedge = svg("path", {
        d: `M${pt(a0, rMin)} L${pt(a0, r)} ${arc(r, 1)}${pt(a1, r)} L${pt(a1, rMin)} ${arc(rMin, 0)}${pt(a0, rMin)} Z`,
        fill: colour, opacity: .82,
      });
      wedge.appendChild(svg("title", {})).textContent = `${c.category}: ${c.count}`;
      el.appendChild(wedge);
      if (!reduced) {
        wedge.style.transformOrigin = `${cx}px ${cy}px`;
        wedge.style.transform = "scale(.25)";
        wedge.style.opacity = "0";
        wedge.style.transition = `transform .7s cubic-bezier(.2,.7,.1,1) ${i * 35}ms, opacity .5s ${i * 35}ms`;
        requestAnimationFrame(() => { wedge.style.transform = "scale(1)"; wedge.style.opacity = ".82"; });
      }
      if (legend) {
        const li = document.createElement("li");
        const sw = document.createElement("i");
        sw.style.background = colour;
        const nm = document.createElement("span");
        nm.textContent = c.category;
        const b = document.createElement("b");
        b.textContent = c.count;
        li.append(sw, nm, b);
        legend.appendChild(li);
      }
    });
    el.appendChild(svg("circle", { cx, cy, r: rMin - 6, fill: "none", stroke: css("--rule-strong", "#cfc8b8"), "stroke-width": 1 }));
  }

  /* ---------- the real waveform of each recent recording ---------- */
  function miniTraces() {
    $$(".trace-item").forEach((li, idx) => {
      const c = $(".trace-mini", li);
      let peaks;
      try { peaks = JSON.parse(li.dataset.peaks || "[]"); } catch (e) { peaks = []; }
      if (!c || !peaks.length) return;
      const colour = li.dataset.color || css("--signal", "#ff5a1f");
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const paint = (progress) => {
        const w = c.clientWidth;
        const h = c.clientHeight;
        if (!w || !h) return;
        if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
        const g = c.getContext("2d");
        g.setTransform(dpr, 0, 0, dpr, 0, 0);
        g.clearRect(0, 0, w, h);
        const mid = h / 2;
        const n = Math.max(1, Math.floor(peaks.length * progress));
        const bw = w / peaks.length;
        g.fillStyle = colour;
        for (let i = 0; i < n; i++) {
          const v = Math.max(0.02, peaks[i]);
          const bh = Math.max(1, v * (h - 2));
          g.globalAlpha = 0.55 + 0.45 * v;
          g.fillRect(i * bw, mid - bh / 2, Math.max(0.7, bw * 0.75), bh);
        }
        g.globalAlpha = 1;
      };
      if (reduced) { paint(1); return; }
      const start = performance.now() + idx * 70;
      const tick = (now) => {
        const p = Math.min(1, Math.max(0, (now - start) / 550));
        paint(p);
        if (p < 1) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
      window.addEventListener("resize", () => paint(1));
    });
  }

  /* ---------- agreement bar + confidence histogram ---------- */
  const AGREE_COLOURS = {
    "Acceptable Match": "--good", "Weak Match": "--warning",
    "Model Disagreement": "--critical", "Uncertain Result": "--serious", "Single-Model Result": "--ink-3",
  };
  function agreement() {
    const bar = $("#agree-bar");
    const key = $("#agree-key");
    const counts = D.consistency || {};
    const total = Object.values(counts).reduce((a, b) => a + b, 0);
    if (!bar || !total) return;
    Object.entries(counts).filter(([, v]) => v > 0).forEach(([name, v]) => {
      const colour = css(AGREE_COLOURS[name] || "--ink-3", "#8a8578");
      const s = document.createElement("span");
      s.style.background = colour;
      s.title = `${name}: ${v}`;
      bar.appendChild(s);
      requestAnimationFrame(() => { s.style.width = `${(v / total) * 100}%`; });
      if (key) {
        const li = document.createElement("li");
        const i = document.createElement("i");
        i.style.background = colour;
        li.append(i, document.createTextNode(`${name} ${Math.round((v / total) * 100)}%`));
        key.appendChild(li);
      }
    });
  }

  function histogram() {
    const box = $("#conf-hist");
    const conf = D.confidence;
    if (!box || !conf) return;
    const values = conf.fused || conf.final || conf.counts || conf.python || [];
    const max = Math.max(...values, 1);
    values.forEach((v, i) => {
      const s = document.createElement("span");
      s.title = `${v} window(s)`;
      box.appendChild(s);
      setTimeout(() => { s.style.height = `${Math.max(2, (v / max) * 100)}%`; }, 40 * i);
    });
  }

  /* ---------- entrance ---------- */
  function reveal() {
    if (reduced) return;
    document.body.classList.add("js-motion");
    const items = [...$$(".console-body > *"), ...$$(".console-readout > div"), ...$$(".gauge"), ...$$(".console-panel")];
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        const i = items.indexOf(e.target);
        e.target.style.transitionDelay = `${Math.min(8, Math.max(0, i)) * 45}ms`;
        e.target.classList.add("is-in");
        io.unobserve(e.target);
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.05 });
    items.forEach((el) => io.observe(el));
  }

  function gaugeRules() {
    $$(".gauge").forEach((g, i) => {
      const bar = $(".gauge-rule i", g);
      if (bar) setTimeout(() => { bar.style.width = "100%"; }, 220 + i * 70);
    });
    $$(".ibar").forEach((b, i) => setTimeout(() => b.classList.add("is-in"), 320 + i * 120));
  }

  function run() {
    reveal();
    spectrum();
    recorderTrace();
    polar();
    miniTraces();
    agreement();
    histogram();
    gaugeRules();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", run);
  else run();
})();
