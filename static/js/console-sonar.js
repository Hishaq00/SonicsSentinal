/* Acoustic input monitor for the dashboards.
 *
 * The dashboards are about sound, so they show sound. This draws the waves:
 *
 *   - resting: a slow ambient trace, so the panel reads as an instrument that
 *     is switched on rather than a placeholder image;
 *   - listening: the real microphone, as a mirrored waveform over a spectrum,
 *     with the input level, peak and dominant band as readouts;
 *   - signal: when the level crosses the detection threshold the panel turns
 *     to the critical colour, the hero flares and the count goes up.
 *
 * Nothing is uploaded and nothing is stored: the audio never leaves the
 * browser and the microphone only opens after the visitor asks for it. The
 * full pipeline (classification, alert rules, sessions) stays on the live
 * monitor page; this is the dashboard's ear.
 */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  const sonar = $("sonar");
  const hero = $("console");
  const heroWave = $("console-wave");
  if (!sonar && !heroWave) return;

  const reduced = () =>
    (window.SSMotion && !window.SSMotion.full) ||
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const ui = {
    stage: $("sonar-stage"),
    wave: $("sonar-wave"),
    spectrum: $("sonar-spectrum"),
    level: $("sonar-level"),
    db: $("sonar-db"),
    peak: $("sonar-peak"),
    hits: $("sonar-hits"),
    band: $("sonar-band"),
    tag: $("sonar-tag-state"),
    note: $("sonar-note"),
  };

  /* ---------- state ---------- */
  const S = {
    ctx: null, stream: null, analyser: null, source: null,
    running: false, raf: null, idleRaf: null,
    time: null, freq: null,
    db: -99, peakDb: -99, smooth: 0, hits: 0, hot: false, hotUntil: 0,
    t: 0, sampleRate: 48000,
  };

  const THRESHOLD_DB = -42;   // a clear sound in a normal room
  const HOLD_MS = 900;        // how long the panel stays in the signal state
  const SEGMENTS = 22;

  /* ---------- level segments ---------- */
  if (ui.level) {
    ui.level.innerHTML = "";
    for (let i = 0; i < SEGMENTS; i++) ui.level.appendChild(document.createElement("i"));
  }
  const segs = ui.level ? Array.from(ui.level.children) : [];

  function paintLevel(pct) {
    const on = Math.round((pct / 100) * segs.length);
    for (let i = 0; i < segs.length; i++) {
      const lit = i < on;
      segs[i].classList.toggle("on", lit);
      segs[i].classList.toggle("hot", lit && i >= segs.length - 4);
    }
  }

  /* ---------- theme colours ---------- */
  function palette() {
    const t = (window.SS && window.SS.chartTheme) ? window.SS.chartTheme() : {};
    return {
      bone: "#ece6d9",
      dim: "#5a564d",
      signal: "#ff5a1f",
      critical: t.critical || "#ff4d4d",
      python: t.python || "#5b9cf0",
    };
  }
  let P = palette();

  /* ---------- canvas sizing ---------- */
  function fit(c) {
    if (!c) return null;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = c.clientWidth;
    const h = c.clientHeight;
    if (!w || !h) return null;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
    }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    return { g, w, h };
  }

  /* ---------- the resting trace ---------- */
  /* Three slow sine components, so the line never repeats visibly. Drawn dim:
     it is the instrument idling, not data. */
  function idleFrame() {
    S.t += 1;
    const draw = (canvas, amp, colour) => {
      const f = fit(canvas);
      if (!f) return;
      const { g, w, h } = f;
      const mid = h / 2;
      g.strokeStyle = colour;
      g.lineWidth = 1.4;
      g.globalAlpha = 0.65;
      g.beginPath();
      for (let x = 0; x <= w; x += 2) {
        const p = x / w;
        const v =
          Math.sin(p * 9 + S.t / 42) * 0.55 +
          Math.sin(p * 21 - S.t / 29) * 0.28 +
          Math.sin(p * 41 + S.t / 17) * 0.12;
        const env = Math.sin(p * Math.PI) * 0.85 + 0.15;
        const y = mid - v * env * amp * mid;
        if (x === 0) g.moveTo(x, y); else g.lineTo(x, y);
      }
      g.stroke();
      g.globalAlpha = 1;
    };
    draw(ui.wave, 0.3, P.dim);
    S.idleRaf = requestAnimationFrame(idleFrame);
  }

  function startIdle() {
    if (S.idleRaf || S.running) return;
    if (reduced()) {
      // A flat baseline: the instrument is on, it just is not animating.
      const f = fit(ui.wave);
      if (f) {
        f.g.strokeStyle = P.dim;
        f.g.lineWidth = 1.4;
        f.g.beginPath();
        f.g.moveTo(0, f.h / 2);
        f.g.lineTo(f.w, f.h / 2);
        f.g.stroke();
      }
      return;
    }
    S.idleRaf = requestAnimationFrame(idleFrame);
  }
  function stopIdle() {
    if (S.idleRaf) cancelAnimationFrame(S.idleRaf);
    S.idleRaf = null;
  }

  /* ---------- the live frame ---------- */
  function liveFrame() {
    if (!S.running || !S.analyser) return;
    S.analyser.getFloatTimeDomainData(S.time);
    S.analyser.getByteFrequencyData(S.freq);

    /* level in dBFS, from the RMS of the window */
    let sum = 0;
    let peak = 0;
    for (let i = 0; i < S.time.length; i++) {
      const v = S.time[i];
      sum += v * v;
      const a = Math.abs(v);
      if (a > peak) peak = a;
    }
    const db = 20 * Math.log10(Math.sqrt(sum / S.time.length) + 1e-9);
    S.db = db;
    if (db > S.peakDb) S.peakDb = db;
    else S.peakDb = Math.max(db, S.peakDb - 0.25);   // slow peak fall-back

    const pct = Math.max(0, Math.min(100, ((db + 70) / 70) * 100));
    S.smooth = S.smooth * 0.75 + pct * 0.25;
    paintLevel(S.smooth);

    /* the detection state, with a hold so a short clap is still visible */
    const now = performance.now();
    const over = db > THRESHOLD_DB;
    if (over) {
      if (!S.hot && now > S.hotUntil) {
        S.hits += 1;
        if (ui.hits) ui.hits.textContent = String(S.hits);
      }
      S.hot = true;
      S.hotUntil = now + HOLD_MS;
    } else if (S.hot && now > S.hotUntil) {
      S.hot = false;
    }
    setState(S.hot ? "signal" : "listening");

    /* readouts */
    if (ui.db) {
      ui.db.textContent = `${Math.max(-99, db).toFixed(0)} dB`;
      ui.db.classList.toggle("hot", S.hot);
    }
    if (ui.peak) ui.peak.textContent = `peak ${Math.max(-99, S.peakDb).toFixed(0)} dBFS`;
    if (ui.band) ui.band.textContent = dominantBand();

    drawSpectrum();
    drawWave(peak);
    drawHero();

    S.raf = requestAnimationFrame(liveFrame);
  }

  /* The loudest bin, named as a band - more readable than a raw frequency. */
  function dominantBand() {
    let best = 0;
    let bestI = 0;
    for (let i = 1; i < S.freq.length; i++) {
      if (S.freq[i] > best) { best = S.freq[i]; bestI = i; }
    }
    if (best < 24) return "quiet";
    const hz = (bestI / S.freq.length) * (S.sampleRate / 2);
    if (hz < 250) return "low / rumble";
    if (hz < 800) return "voice / low-mid";
    if (hz < 2500) return "voice / speech";
    if (hz < 5000) return "high-mid / alarm";
    return "high / transient";
  }

  /* Frequency bars, log-spaced so speech is not squeezed into the left edge. */
  function drawSpectrum() {
    const f = fit(ui.spectrum);
    if (!f) return;
    const { g, w, h } = f;
    const bars = Math.max(28, Math.min(72, Math.floor(w / 13)));
    const bw = w / bars;
    const maxBin = Math.floor((8000 / (S.sampleRate / 2)) * S.freq.length);
    for (let i = 0; i < bars; i++) {
      const frac = i / bars;
      const bin = Math.min(maxBin - 1, Math.max(1, Math.floor(Math.pow(frac, 1.7) * maxBin)));
      const nextBin = Math.min(maxBin, Math.max(bin + 1, Math.floor(Math.pow((i + 1) / bars, 1.7) * maxBin)));
      let v = 0;
      for (let b = bin; b < nextBin; b++) v = Math.max(v, S.freq[b]);
      v /= 255;
      const bh = Math.max(1, v * h * 0.92);
      const x = i * bw;
      const grad = g.createLinearGradient(0, h, 0, h - bh);
      grad.addColorStop(0, (S.hot ? P.critical : P.signal) + "00");
      grad.addColorStop(1, (S.hot ? P.critical : P.signal) + "b0");
      g.fillStyle = grad;
      g.fillRect(x + bw * 0.14, h - bh, bw * 0.72, bh);
    }
  }

  /* The waveform, as mirrored bars: an envelope, the way a recorder prints it. */
  function drawWave(peakAbs) {
    const f = fit(ui.wave);
    if (!f) return;
    const { g, w, h } = f;
    const mid = h / 2;
    // Display gain only: quiet rooms still show a wave, the audio is untouched.
    const gain = Math.min(9, Math.max(1, 0.8 / Math.max(peakAbs, 1e-4)));
    const bars = Math.max(48, Math.min(200, Math.floor(w / 5)));
    const step = Math.floor(S.time.length / bars) || 1;
    const bw = w / bars;
    g.fillStyle = S.hot ? P.critical : P.bone;
    for (let i = 0; i < bars; i++) {
      let m = 0;
      const start = i * step;
      for (let k = 0; k < step; k++) {
        const a = Math.abs(S.time[start + k] || 0);
        if (a > m) m = a;
      }
      const amp = Math.min(1, m * gain);
      const bh = Math.max(1.2, amp * (mid - 6));
      g.globalAlpha = 0.45 + 0.55 * amp;
      g.fillRect(i * bw + bw * 0.2, mid - bh, Math.max(1, bw * 0.6), bh * 2);
    }
    g.globalAlpha = 1;
  }

  /* The hero ribbon: one continuous line with a filled skirt. */
  function drawHero() {
    const f = fit(heroWave);
    if (!f) return;
    const { g, w, h } = f;
    const base = h;
    const pts = [];
    const n = Math.max(40, Math.min(160, Math.floor(w / 8)));
    const step = Math.floor(S.time.length / n) || 1;
    let peak = 0;
    for (let i = 0; i < S.time.length; i++) peak = Math.max(peak, Math.abs(S.time[i]));
    const gain = Math.min(9, Math.max(1, 0.8 / Math.max(peak, 1e-4)));
    for (let i = 0; i <= n; i++) {
      let m = 0;
      for (let k = 0; k < step; k++) {
        const a = Math.abs(S.time[i * step + k] || 0);
        if (a > m) m = a;
      }
      pts.push({ x: (i / n) * w, y: base - Math.min(1, m * gain) * (h - 8) });
    }
    const colour = S.hot ? P.critical : P.signal;
    g.beginPath();
    g.moveTo(0, base);
    pts.forEach((p) => g.lineTo(p.x, p.y));
    g.lineTo(w, base);
    g.closePath();
    const grad = g.createLinearGradient(0, base - h, 0, base);
    grad.addColorStop(0, colour + "55");
    grad.addColorStop(1, colour + "00");
    g.fillStyle = grad;
    g.fill();
    g.beginPath();
    pts.forEach((p, i) => (i ? g.lineTo(p.x, p.y) : g.moveTo(p.x, p.y)));
    g.strokeStyle = colour;
    g.lineWidth = 1.6;
    g.stroke();
  }

  /* ---------- state plumbing ---------- */
  const STATE_TEXT = {
    idle: "Idle",
    listening: "Listening",
    signal: "Sound detected",
    denied: "Microphone blocked",
    error: "Unavailable",
  };
  let current = "";
  function setState(name) {
    if (current === name) return;
    current = name;
    if (sonar) sonar.dataset.state = name;
    if (hero) {
      hero.dataset.live = (name === "listening" || name === "signal") ? "on" : "off";
      hero.dataset.signal = name === "signal" ? "hot" : "";
    }
    $$("[data-sonar-label]").forEach((el) => { el.textContent = STATE_TEXT[name] || name; });
    if (ui.tag) ui.tag.textContent = name === "signal" ? "signal" : (name === "listening" ? "live input" : "standby");
    const live = name === "listening" || name === "signal";
    $$("[data-sonar-start]").forEach((b) => b.classList.toggle("hidden", live));
    $$("[data-sonar-stop]").forEach((b) => b.classList.toggle("hidden", !live));
  }

  /* ---------- start / stop ---------- */
  async function start() {
    if (S.running) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.AudioContext) {
      setState("error");
      if (ui.note) ui.note.textContent = "This browser cannot open a microphone.";
      return;
    }
    $$("[data-sonar-start]").forEach((b) => { b.disabled = true; });
    try {
      S.stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
    } catch (err) {
      setState(err && err.name === "NotAllowedError" ? "denied" : "error");
      if (ui.note) {
        ui.note.textContent = err && err.name === "NotAllowedError"
          ? "Microphone permission was declined. Allow it in the browser address bar to see the waves."
          : "No microphone was found on this device.";
      }
      $$("[data-sonar-start]").forEach((b) => { b.disabled = false; });
      startIdle();
      return;
    }
    S.ctx = new (window.AudioContext || window.webkitAudioContext)();
    S.sampleRate = S.ctx.sampleRate;
    S.source = S.ctx.createMediaStreamSource(S.stream);
    S.analyser = S.ctx.createAnalyser();
    S.analyser.fftSize = 2048;
    S.analyser.smoothingTimeConstant = 0.72;
    S.source.connect(S.analyser);      // not connected to the output: no feedback loop
    S.time = new Float32Array(S.analyser.fftSize);
    S.freq = new Uint8Array(S.analyser.frequencyBinCount);
    S.running = true;
    S.hits = 0;
    S.peakDb = -99;
    if (ui.hits) ui.hits.textContent = "0";
    if (ui.note) ui.note.textContent = "Audio stays in this browser. Nothing is uploaded or stored.";
    $$("[data-sonar-start]").forEach((b) => { b.disabled = false; });
    stopIdle();
    P = palette();
    setState("listening");
    S.raf = requestAnimationFrame(liveFrame);
  }

  function stop() {
    if (S.raf) cancelAnimationFrame(S.raf);
    S.raf = null;
    S.running = false;
    S.hot = false;
    if (S.stream) S.stream.getTracks().forEach((t) => t.stop());
    if (S.ctx && S.ctx.state !== "closed") S.ctx.close().catch(() => {});
    S.stream = null; S.ctx = null; S.analyser = null; S.source = null;
    paintLevel(0);
    if (ui.db) { ui.db.textContent = "-"; ui.db.classList.remove("hot"); }
    if (ui.peak) ui.peak.textContent = "peak -- dBFS";
    if (ui.band) ui.band.textContent = "-";
    if (ui.note) ui.note.textContent = "Microphone off.";
    setState("idle");
    startIdle();
  }

  /* ---------- wiring ---------- */
  $$("[data-sonar-start]").forEach((b) => b.addEventListener("click", start));
  $$("[data-sonar-stop]").forEach((b) => b.addEventListener("click", stop));
  window.addEventListener("pagehide", stop);
  document.addEventListener("visibilitychange", () => { if (document.hidden && !S.running) stopIdle(); else if (!document.hidden && !S.running) startIdle(); });
  window.addEventListener("resize", () => { P = palette(); });
  // The theme switch changes the tokens the wave is drawn from.
  document.addEventListener("click", (e) => {
    if (e.target.closest("[data-theme-toggle]")) setTimeout(() => { P = palette(); }, 60);
  });

  setState("idle");
  startIdle();
})();
