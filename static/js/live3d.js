/* Live monitor: 3D spectrum waterfall of the real microphone (Three.js).
 * live.js announces its AnalyserNode with an "ss:analyser" event; every row of
 * the waterfall is one FFT frame (0-8 kHz, like the 2D spectrogram panel). */
import { mountSoundField } from "./soundfield.js";

const el = document.getElementById("spectrum3d");
const status = document.getElementById("spectrum3d-state");
if (el) {
  const COLS = 128;
  const out = new Float32Array(COLS);
  let analyser = null, buf = null, rate = 48000;
  document.addEventListener("ss:analyser", (e) => {
    analyser = e.detail ? e.detail.analyser : null;
    rate = (e.detail && e.detail.sampleRate) || 48000;
    buf = analyser ? new Float32Array(analyser.frequencyBinCount) : null;
    if (status) status.textContent = analyser ? "microphone, 0–8 kHz" : "waiting for the microphone";
  });
  function source() {
    if (!analyser) return null;
    analyser.getFloatFrequencyData(buf);
    const maxBin = Math.floor((8000 / (rate / 2)) * buf.length);
    for (let i = 0; i < COLS; i++) {
      const a = Math.floor((i / COLS) * maxBin), b = Math.max(a + 1, Math.floor(((i + 1) / COLS) * maxBin));
      let m = -160;
      for (let k = a; k < b; k++) if (buf[k] > m) m = buf[k];
      out[i] = Math.max(0, Math.min(1.1, (m + 100) / 70)); // -100 dBFS -> 0, -30 dBFS -> 1
    }
    return out;
  }
  const token = (n, f) => getComputedStyle(document.documentElement).getPropertyValue(n).trim() || f;
  let field = null;
  function mount() {
    if (field) field.destroy();
    field = mountSoundField(el, {
      source, cols: COLS, rows: 84, height: 2.1,
      animate: !(window.SSMotion && !window.SSMotion.full),
      background: token("--surface", "#161713"), low: token("--rule-strong", "#3b3c35"),
      mid: token("--ink", "#ece6d9"), hot: token("--signal", "#ff5a1f"),
    });
  }
  mount();
  document.addEventListener("ss:theme", () => setTimeout(mount, 30));
}
