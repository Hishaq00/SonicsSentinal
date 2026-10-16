/* SonicSentinel "sound field": a live 3D spectrogram waterfall (Three.js).
 *
 * Frequency runs left to right, time runs into the distance, height is energy.
 * A small synthetic sound scene (siren sweeps, gunshot impulses, breaking
 * glass, barks, horns, machinery, speech) scrolls through the terrain so the
 * page shows what the product actually looks at: spectrograms of events.
 * Ridge lines are drawn over dark "curtains" that hide the rows behind them.
 *
 *   import { mountSoundField } from "./soundfield.js";
 *   const field = mountSoundField(element, { onEvent: (name) => ... });
 *   field.setProgress(0..1);   // scroll-driven camera move
 */
import * as THREE from "../vendor/three.module.min.js";

const EVENTS = [
  { name: "Alarm or Siren", dur: 3.0, fn: siren },
  { name: "Gunshot", dur: 1.1, fn: gunshot },
  { name: "Glass Breaking", dur: 1.4, fn: glass },
  { name: "Animal Sound", dur: 1.4, fn: bark },
  { name: "Vehicle Horn", dur: 1.6, fn: horn },
  { name: "Machinery Fault", dur: 2.2, fn: machinery },
  { name: "Panic Scream", dur: 1.5, fn: scream },
  { name: "Person Asking for Help", dur: 1.8, fn: speech },
];

/* ---------- synthetic spectral signatures: energy(t, f), f in [0, 1] ---------- */
const g = (x, mu, w) => Math.exp(-((x - mu) * (x - mu)) / (2 * w * w));
function harmonics(f, f0, n, width, decay) {
  let e = 0;
  for (let k = 1; k <= n; k++) e += g(f, f0 * k, width) / Math.pow(k, decay);
  return e;
}
function siren(t, f) {
  const f0 = 0.13 + 0.06 * Math.sin((2 * Math.PI * t) / 1.35);
  return 0.95 * harmonics(f, f0, 4, 0.006, 0.7) * Math.min(1, t * 4, (3 - t) * 4);
}
function gunshot(t, f) {
  const hit = t < 0.05 ? 1.1 * (1.05 - 0.45 * f) : 0;
  const tail = Math.exp(-t / 0.13) * (0.85 - 0.6 * f);
  const echo = t > 0.32 && t < 0.4 ? 0.35 * (0.9 - 0.6 * f) : 0;
  return Math.max(0, hit + tail + echo);
}
function glass(t, f) {
  let e = t < 0.06 ? 0.9 * (0.5 + 0.5 * f) : Math.exp(-t / 0.25) * 0.45 * f;
  const s = Math.floor(t * 18);
  for (let k = 0; k < 3; k++) {
    const r = Math.sin((s + 1) * 12.9898 * (k + 1)) * 43758.5453;
    const fr = 0.45 + 0.5 * (r - Math.floor(r));
    e += 0.75 * g(f, fr, 0.01) * Math.exp(-t / 0.7);
  }
  return e;
}
function bark(t, f) {
  const burst = [0, 0.42, 0.84].some((s) => t >= s && t < s + 0.16);
  if (!burst) return 0;
  return 0.95 * harmonics(f, 0.055, 7, 0.009, 0.5) * g(f, 0.22, 0.16);
}
function horn(t, f) {
  const env = Math.min(1, t * 8) * Math.min(1, (1.6 - t) * 6);
  return 0.85 * harmonics(f, 0.085, 6, 0.005, 0.35) * env;
}
function machinery(t, f) {
  const knock = (t % 0.2) < 0.035 ? 0.7 * g(f, 0.18, 0.12) : 0;
  return knock + 0.45 * (g(f, 0.035, 0.004) + g(f, 0.07, 0.004) + 0.6 * g(f, 0.105, 0.004));
}
function scream(t, f) {
  const f0 = 0.24 + 0.06 * (t / 1.5) + 0.01 * Math.sin(t * 38);
  return 0.95 * harmonics(f, f0, 3, 0.007, 0.6) * Math.min(1, t * 6, (1.5 - t) * 5);
}
function speech(t, f) {
  const syl = [0.05, 0.45, 1.05, 1.45].some((s) => t >= s && t < s + 0.28);
  if (!syl) return 0;
  return 0.8 * (g(f, 0.07, 0.02) + 0.7 * g(f, 0.16, 0.025) + 0.45 * g(f, 0.3, 0.03)) *
    (0.6 + 0.4 * harmonics(f, 0.028, 12, 0.004, 0.2));
}
function aggression(t, f) {
  let e = 0;
  [[0.0, 0.07], [0.25, 0.09], [0.55, 0.065], [0.8, 0.1], [1.05, 0.075]].forEach(([s, f0], k) => {
    if (t >= s && t < s + 0.34) e += 0.55 * harmonics(f, f0 + 0.004 * Math.sin(t * 30 + k), 6, 0.007, 0.45);
  });
  return e;
}
function background(t, f) {
  const r = Math.sin((Math.floor(t * 40) + 1) * 78.233 + f * 311.7) * 43758.5453;
  return 0.18 * (1 - 0.6 * f) * (0.4 + 0.6 * (r - Math.floor(r)));
}

/** Spectral signature per class slug: energy(t, f) over `dur` seconds (used for the sound cards). */
export const SIGNATURES = {
  machinery_fault: { fn: machinery, dur: 1.6 },
  glass_breaking: { fn: glass, dur: 1.2 },
  alarm_or_siren: { fn: siren, dur: 2.7 },
  vehicle_horn: { fn: horn, dur: 1.6 },
  animal_sound: { fn: bark, dur: 1.2 },
  gunshot: { fn: gunshot, dur: 0.9 },
  panic_scream: { fn: scream, dur: 1.5 },
  aggression: { fn: aggression, dur: 1.4 },
  person_asking_for_help: { fn: speech, dur: 1.8 },
  background_noise: { fn: background, dur: 1.5 },
};

function css(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

export function mountSoundField(container, opts = {}) {
  const reduce = opts.animate === false;
  const COLS = opts.cols || 150;
  const ROWS = opts.rows || 96;
  const W = 19, DZ = 0.17, H = opts.height || 2.3, ZFRONT = 4.2;
  const ROW_RATE = 30; // rows per second

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
  } catch (e) {
    container.classList.add("no-webgl");
    return { setProgress() {}, destroy() {}, pause() {}, resume() {} };
  }
  const bg = new THREE.Color(opts.background || css("--carbon", "#0c0d0b"));
  const lo = new THREE.Color(opts.low || "#3a3a33");
  const mid = new THREE.Color(opts.mid || css("--bone", "#ece6d9"));
  const hot = new THREE.Color(opts.hot || css("--signal", "#ff5a1f"));

  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
  renderer.setClearColor(bg, opts.transparent ? 0 : 1);
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(bg, 7, 21);
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 60);

  /* ---------- geometry: ridge lines + occluding curtains ---------- */
  const heights = new Float32Array(ROWS * COLS);
  const xs = new Float32Array(COLS);
  for (let i = 0; i < COLS; i++) xs[i] = (i / (COLS - 1) - 0.5) * W;

  const linePos = new Float32Array(ROWS * COLS * 3);
  const lineCol = new Float32Array(ROWS * COLS * 3);
  const lineIdx = [];
  for (let r = 0; r < ROWS; r++) {
    for (let i = 0; i < COLS; i++) {
      const v = r * COLS + i;
      linePos[v * 3] = xs[i];
      linePos[v * 3 + 2] = ZFRONT - r * DZ;
      if (i < COLS - 1) lineIdx.push(v, v + 1);
    }
  }
  const lineGeo = new THREE.BufferGeometry();
  lineGeo.setAttribute("position", new THREE.BufferAttribute(linePos, 3).setUsage(THREE.DynamicDrawUsage));
  lineGeo.setAttribute("color", new THREE.BufferAttribute(lineCol, 3).setUsage(THREE.DynamicDrawUsage));
  lineGeo.setIndex(lineIdx);
  const lines = new THREE.LineSegments(lineGeo, new THREE.LineBasicMaterial({ vertexColors: true, fog: true }));

  const curPos = new Float32Array(ROWS * COLS * 2 * 3);
  const curIdx = [];
  for (let r = 0; r < ROWS; r++) {
    for (let i = 0; i < COLS; i++) {
      const v = (r * COLS + i) * 2;
      const z = ZFRONT - r * DZ - 0.002;
      curPos[v * 3] = xs[i]; curPos[v * 3 + 1] = 0; curPos[v * 3 + 2] = z;
      curPos[(v + 1) * 3] = xs[i]; curPos[(v + 1) * 3 + 1] = -0.4; curPos[(v + 1) * 3 + 2] = z;
      if (i < COLS - 1) curIdx.push(v, v + 1, v + 2, v + 1, v + 3, v + 2);
    }
  }
  const curGeo = new THREE.BufferGeometry();
  curGeo.setAttribute("position", new THREE.BufferAttribute(curPos, 3).setUsage(THREE.DynamicDrawUsage));
  curGeo.setIndex(curIdx);
  // Slightly translucent, so a loud broadband row (a gunshot) dims the rows behind it instead of hiding them.
  const curtains = new THREE.Mesh(curGeo, new THREE.MeshBasicMaterial({
    color: bg, side: THREE.DoubleSide, transparent: true, opacity: 0.86,
    polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
  }));

  const field = new THREE.Group();
  field.add(curtains, lines);
  scene.add(field);

  /* ---------- the synthetic sound scene ---------- */
  let clock = 0, rowClock = 0, evIndex = opts.startIndex || 0, evStart = 0.6, evActive = false;
  let seed = 1;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  function newRow(t) {
    if (opts.source) { // real data (e.g. the live microphone): one row of 0..1 energies per call
      const data = opts.source();
      heights.copyWithin(COLS, 0, (ROWS - 1) * COLS);
      let peak = 0;
      for (let i = 0; i < COLS; i++) {
        const e = data ? data[i] : 0.03 * rand();
        heights[i] = Math.min(1.15, e);
        if (e > peak) peak = e;
      }
      if (opts.onLevel) opts.onLevel(Math.min(1, peak));
      return;
    }
    const ev = EVENTS[evIndex % EVENTS.length];
    const local = t - evStart;
    const on = local >= 0 && local < ev.dur;
    if (on && !evActive) { evActive = true; opts.onEvent && opts.onEvent(ev.name, true); }
    if (!on && evActive) { evActive = false; opts.onEvent && opts.onEvent(ev.name, false); }
    if (local >= ev.dur + 0.9) { evIndex++; evStart = t + 0.2; }
    heights.copyWithin(COLS, 0, (ROWS - 1) * COLS);
    const hum = 0.05 + 0.02 * Math.sin(t * 0.7);
    let peak = 0;
    for (let i = 0; i < COLS; i++) {
      const f = i / (COLS - 1);
      let e = (0.045 + 0.05 * rand()) * (1 - 0.55 * f) + hum * g(f, 0.02, 0.01);
      if (on) e += ev.fn(local, f) * (0.72 + 0.38 * rand());
      heights[i] = Math.min(1.15, e);
      if (e > peak) peak = e;
    }
    if (opts.onLevel) opts.onLevel(Math.min(1, peak));
  }
  const L = [lo.r, lo.g, lo.b], M = [mid.r, mid.g, mid.b], K = [hot.r, hot.g, hot.b];
  function writeGeometry() {
    for (let v = 0; v < ROWS * COLS; v++) {
      const h = heights[v];
      const y = h * H;
      linePos[v * 3 + 1] = y;
      curPos[v * 2 * 3 + 1] = y;
      const a = h < 0.42 ? L : M, b = h < 0.42 ? M : K;
      const t = h < 0.42 ? h / 0.42 : Math.min(1, (h - 0.42) / 0.5);
      lineCol[v * 3] = a[0] + (b[0] - a[0]) * t;
      lineCol[v * 3 + 1] = a[1] + (b[1] - a[1]) * t;
      lineCol[v * 3 + 2] = a[2] + (b[2] - a[2]) * t;
    }
    lineGeo.attributes.position.needsUpdate = true;
    lineGeo.attributes.color.needsUpdate = true;
    curGeo.attributes.position.needsUpdate = true;
  }
  // Fill the history so the first frame already shows a landscape.
  for (let k = 0; k < ROWS; k++) { clock += 1 / ROW_RATE; newRow(clock); }
  writeGeometry();

  /* ---------- camera, pointer, scroll ---------- */
  const pointer = { x: 0, y: 0, tx: 0, ty: 0 };
  let progress = 0;
  const onPointer = (e) => {
    const r = container.getBoundingClientRect();
    pointer.tx = ((e.clientX - r.left) / r.width - 0.5) * 2;
    pointer.ty = ((e.clientY - r.top) / r.height - 0.5) * 2;
  };
  if (opts.interactive !== false) window.addEventListener("pointermove", onPointer, { passive: true });

  function placeCamera() {
    pointer.x += (pointer.tx - pointer.x) * 0.04;
    pointer.y += (pointer.ty - pointer.y) * 0.04;
    const p = progress;
    camera.position.set(pointer.x * 0.9, 3.4 - 1.9 * p - pointer.y * 0.35, 9.6 - 4.2 * p);
    camera.lookAt(pointer.x * 0.4, 0.35 + 0.3 * p, -3 - 2 * p);
    field.rotation.y = -0.05 * pointer.x;
  }

  function resize() {
    const w = container.clientWidth || 1, h = container.clientHeight || 1;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.fov = w < 700 ? 52 : 38;
    camera.updateProjectionMatrix();
  }
  const ro = new ResizeObserver(resize);
  ro.observe(container);
  resize();

  /* ---------- loop (paused when off screen or hidden) ---------- */
  let raf = 0, last = performance.now(), visible = true, running = !reduce;
  function frame(now) {
    raf = 0;
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    rowClock += dt;
    let added = false;
    while (rowClock >= 1 / ROW_RATE) {
      rowClock -= 1 / ROW_RATE;
      clock += 1 / ROW_RATE;
      newRow(clock);
      added = true;
    }
    if (added) writeGeometry();
    placeCamera();
    renderer.render(scene, camera);
    if (running && visible && !document.hidden) raf = requestAnimationFrame(frame);
  }
  function kick() {
    if (!raf && running && visible && !document.hidden) { last = performance.now(); raf = requestAnimationFrame(frame); }
  }
  const io = new IntersectionObserver(([e]) => { visible = e.isIntersecting; kick(); }, { threshold: 0 });
  io.observe(container);
  const onVis = () => kick();
  document.addEventListener("visibilitychange", onVis);
  placeCamera();
  renderer.render(scene, camera);
  kick();

  return {
    setProgress(p) {
      progress = Math.max(0, Math.min(1, p));
      if (!running) { placeCamera(); renderer.render(scene, camera); }
    },
    pause() { running = false; },
    resume() { if (!reduce) { running = true; kick(); } },
    destroy() {
      running = false;
      cancelAnimationFrame(raf);
      io.disconnect(); ro.disconnect();
      window.removeEventListener("pointermove", onPointer);
      document.removeEventListener("visibilitychange", onVis);
      lineGeo.dispose(); curGeo.dispose(); renderer.dispose();
      renderer.domElement.remove();
    },
  };
}
