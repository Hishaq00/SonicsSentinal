/* Google Teachable Machine (GTM) audio model runner.
 *
 * GTM audio models are TensorFlow.js "speech-commands" (BROWSER_FFT) models.
 * In GTM's own page the input comes from a Web Audio AnalyserNode. To classify
 * uploaded files and our own microphone windows we compute exactly the same
 * representation here:
 *   44 100 Hz mono, AnalyserNode fftSize 2048 (Blackman window, no smoothing),
 *   one frame every 1024 samples, 20*log10(|X[k]| / 2048) for bins 0..231,
 *   43 frames = one ~1 s example [1, 43, 232, 1], z-score normalised.
 * The same maths is implemented in Python in feature_extraction/gtm_spectrogram.py
 * (tests/test_gtm.py checks both give the same numbers).
 *
 * Independence: this code only ever sees audio. The Python model's prediction
 * is never passed to GTM.
 */
(function () {
  "use strict";
  const P = { SR: 44100, FFT: 2048, HOP: 1024, FRAMES: 43, BINS: 232 };
  const INPUT = P.FFT + (P.FRAMES - 1) * P.HOP; // 45 056 samples, about 1.02 s

  /* ---------- Blackman window and FFT tables ---------- */
  const WIN = new Float32Array(P.FFT);
  for (let n = 0; n < P.FFT; n++) {
    WIN[n] = 0.42 - 0.5 * Math.cos((2 * Math.PI * n) / P.FFT) + 0.08 * Math.cos((4 * Math.PI * n) / P.FFT);
  }
  const LOG2N = Math.log2(P.FFT);
  const REV = new Uint32Array(P.FFT);
  for (let i = 0; i < P.FFT; i++) {
    let r = 0;
    for (let b = 0; b < LOG2N; b++) r |= ((i >> b) & 1) << (LOG2N - 1 - b);
    REV[i] = r;
  }
  const COS = new Float64Array(P.FFT / 2);
  const SIN = new Float64Array(P.FFT / 2);
  for (let i = 0; i < P.FFT / 2; i++) {
    COS[i] = Math.cos((2 * Math.PI * i) / P.FFT);
    SIN[i] = -Math.sin((2 * Math.PI * i) / P.FFT);
  }

  /** In-place iterative radix-2 FFT of length 2048. */
  function fft(re, im) {
    const N = P.FFT;
    for (let i = 0; i < N; i++) {
      const j = REV[i];
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let size = 2; size <= N; size <<= 1) {
      const half = size >> 1;
      const step = N / size;
      for (let start = 0; start < N; start += size) {
        for (let k = 0; k < half; k++) {
          const wr = COS[k * step];
          const wi = SIN[k * step];
          const a = start + k;
          const b = a + half;
          const tr = wr * re[b] - wi * im[b];
          const ti = wr * im[b] + wi * re[b];
          re[b] = re[a] - tr; im[b] = im[a] - ti;
          re[a] += tr; im[a] += ti;
        }
      }
    }
  }

  /* Deterministic tiny dither (a real microphone never outputs exact zeros). */
  function makeDither(seed) {
    let s = seed >>> 0;
    return () => {
      s = (1664525 * s + 1013904223) >>> 0;
      return ((s / 4294967296) - 0.5) * 3.4e-5;
    };
  }

  /** (43 x 232) dB spectrogram, row-major, from >= INPUT samples at 44.1 kHz. */
  function spectrogram(x, dither = true) {
    const out = new Float32Array(P.FRAMES * P.BINS);
    const re = new Float64Array(P.FFT);
    const im = new Float64Array(P.FFT);
    const noise = makeDither(7);
    for (let f = 0; f < P.FRAMES; f++) {
      const start = f * P.HOP;
      for (let n = 0; n < P.FFT; n++) {
        const v = start + n < x.length ? x[start + n] : 0;
        re[n] = (v + (dither ? noise() : 0)) * WIN[n];
        im[n] = 0;
      }
      fft(re, im);
      for (let k = 0; k < P.BINS; k++) {
        const mag = Math.sqrt(re[k] * re[k] + im[k] * im[k]) / P.FFT;
        out[f * P.BINS + k] = 20 * Math.log10(Math.max(mag, 1e-12));
      }
    }
    return out;
  }

  /** speech-commands normalize(): (x - mean) / (std + epsilon) over the whole example. */
  function normalize(spec) {
    let mean = 0;
    for (let i = 0; i < spec.length; i++) mean += spec[i];
    mean /= spec.length;
    let v = 0;
    for (let i = 0; i < spec.length; i++) v += (spec[i] - mean) ** 2;
    const std = Math.sqrt(v / spec.length);
    const out = new Float32Array(spec.length);
    for (let i = 0; i < spec.length; i++) out[i] = (spec[i] - mean) / (std + 1e-7);
    return out;
  }

  /** Start of the ~1 s window centred on the loudest 50 ms frame (same rule as Python). */
  function bestWindowStart(y, sr = P.SR, win = INPUT) {
    if (y.length <= win) return 0;
    const frame = Math.max(1, Math.floor(0.05 * sr));
    const n = Math.floor(y.length / frame);
    let best = 0;
    let bestE = -1;
    for (let i = 0; i < n; i++) {
      let e = 0;
      for (let j = i * frame; j < (i + 1) * frame; j++) e += y[j] * y[j];
      if (e > bestE) { bestE = e; best = i; }
    }
    const centre = (best + 0.5) * frame;
    return Math.min(Math.max(0, Math.round(centre - win / 2)), y.length - win);
  }

  /** Exactly INPUT samples: centre shorter clips with zero padding. */
  function fitInput(y) {
    if (y.length === INPUT) return y;
    const out = new Float32Array(INPUT);
    if (y.length > INPUT) { out.set(y.subarray(0, INPUT)); return out; }
    out.set(y, Math.floor((INPUT - y.length) / 2));
    return out;
  }

  /** Resample any Float32Array using linear interpolation (fast, deterministic, no browser context limits). */
  async function resampleTo(data, fromRate, toRate = P.SR) {
    if (!data || data.length === 0 || fromRate === toRate) return data;
    const ratio = fromRate / toRate;
    const outLength = Math.max(1, Math.round(data.length / ratio));
    const out = new Float32Array(outLength);
    for (let i = 0; i < outLength; i++) {
      const pos = i * ratio;
      const idx = Math.floor(pos);
      const frac = pos - idx;
      const s0 = data[idx] || 0;
      const s1 = idx + 1 < data.length ? data[idx + 1] : s0;
      out[i] = s0 + frac * (s1 - s0);
    }
    return out;
  }

  /** Decode a file (ArrayBuffer) to 44.1 kHz mono. */
  async function decodeTo44k(arrayBuffer) {
    const ctx = new OfflineAudioContext(1, 1, P.SR);
    const audio = await ctx.decodeAudioData(arrayBuffer.slice(0));
    const n = audio.length;
    const mono = new Float32Array(n);
    for (let c = 0; c < audio.numberOfChannels; c++) {
      const ch = audio.getChannelData(c);
      for (let i = 0; i < n; i++) mono[i] += ch[i] / audio.numberOfChannels;
    }
    return audio.sampleRate === P.SR ? mono : resampleTo(mono, audio.sampleRate, P.SR);
  }

  /* ---------- the model ---------- */
  class GTMModel {
    constructor(cfg) {
      this.cfg = cfg || {};
      this.model = null;
      this.labels = [];
      this.version = null;
      this.loading = null;
    }

    get ready() { return !!this.model; }

    async load(force = false) {
      if (force) {
        this.model = null;
        this.loading = null;
      }
      if (this.model) return this;
      if (this.loading) return this.loading;
      this.loading = (async () => {
        if (!this.cfg.enabled || !this.cfg.model_url) throw new Error("GTM model is not configured.");
        if (!window.tf) throw new Error("TensorFlow.js failed to load.");
        const tParam = (url) => url + (url.includes("?") ? "&" : "?") + "t=" + Date.now();
        const res = await fetch(tParam(this.cfg.metadata_url), { credentials: "same-origin", cache: "no-store" });
        if (!res.ok) throw new Error(`GTM metadata.json not found (${res.status}).`);
        const meta = await res.json();
        this.labels = meta.wordLabels || meta.words || meta.labels || [];
        const model = await window.tf.loadLayersModel(tParam(this.cfg.model_url), {
          fetchFunc: (url, init) => fetch(url, { ...init, cache: "no-store" })
        });
        const shape = model.inputs[0].shape;
        if (shape.length !== 4 || shape[1] !== P.FRAMES || shape[2] !== P.BINS) {
          throw new Error(`Unexpected GTM input shape [${shape}]. Export the model from a Teachable Machine *audio* project.`);
        }
        const units = model.outputs[0].shape[1];
        if (units !== this.labels.length) throw new Error(`GTM model has ${units} outputs but ${this.labels.length} labels.`);
        this.model = model;
        this.version = this.cfg.version_label || `${meta.modelName || "tm-audio-model"}@${meta.timeStamp || "unknown"}`;
        await this.reportStatus({ ok: true, labels: this.labels, version: this.version, source: this.cfg.model_url });
        return this;
      })();
      try {
        return await this.loading;
      } catch (err) {
        this.loading = null;
        await this.reportStatus({ ok: false, error: String(err.message || err) });
        throw err;
      }
    }

    async reportStatus(payload) {
      if (!this.cfg.status_url || GTMModel.reported) return;
      GTMModel.reported = true;
      try {
        await fetch(this.cfg.status_url, {
          method: "POST", credentials: "same-origin",
          headers: { "Content-Type": "application/json", "X-CSRFToken": (window.SS && window.SS.csrf) || "" },
          body: JSON.stringify(payload),
        });
      } catch (e) { /* non-fatal */ }
    }

    /** Classify one ~1 s window (Float32Array at 44.1 kHz). Returns {label: probability}. */
    async classifyWindow(x44k) {
      const spec = normalize(spectrogram(fitInput(x44k)));
      const tf = window.tf;
      const probs = tf.tidy(() => this.model.predict(tf.tensor4d(spec, [1, P.FRAMES, P.BINS, 1])).dataSync());
      const scores = {};
      this.labels.forEach((label, i) => { scores[label] = probs[i]; });
      return scores;
    }

    /** Classify the loudest second of a segment. */
    async classifySegment(y44k, startSec, endSec) {
      const s0 = Math.max(0, Math.floor(startSec * P.SR));
      const s1 = Math.min(y44k.length, Math.max(s0 + 1, Math.ceil(endSec * P.SR)));
      const seg = y44k.subarray(s0, s1);
      const ws = bestWindowStart(seg);
      const scores = await this.classifyWindow(seg.subarray(ws, ws + INPUT));
      return { scores, window_start: (s0 + ws) / P.SR };
    }
  }
  GTMModel.reported = false;

  /** Run a server GTM job: fetch the pre-processed audio, classify each segment, post the scores. */
  async function runJob(job, model, onProgress) {
    const t0 = performance.now();
    try {
      if (job.gtm && job.gtm.model_url) {
        if (model.version && job.gtm.version_label && model.version !== job.gtm.version_label) {
          model.cfg = { ...model.cfg, ...job.gtm };
          await model.load(true);
        } else {
          model.cfg = { ...model.cfg, ...job.gtm };
          await model.load();
        }
      } else {
        await model.load();
      }
      const res = await fetch(job.audio_url, { credentials: "same-origin" });
      if (!res.ok) throw new Error(`Could not fetch the processed audio (${res.status}).`);
      const y = await decodeTo44k(await res.arrayBuffer());
      const segments = [];
      for (let i = 0; i < job.segments.length; i++) {
        const s = job.segments[i];
        const r = await model.classifySegment(y, s.start, s.end);
        segments.push({ index: s.index, scores: r.scores, window_start: r.window_start });
        if (onProgress) onProgress(i + 1, job.segments.length);
      }
      return await window.SS.api(job.submit_url, {
        method: "POST",
        json: { model_version: model.version, labels: model.labels, segments, elapsed_ms: Math.round(performance.now() - t0) },
      });
    } catch (err) {
      // Tell the server so the record is finalised as a single-model result instead of waiting.
      try {
        return await window.SS.api(job.submit_url, { method: "POST", json: { error: String(err.message || err) } });
      } catch (e2) {
        throw err;
      }
    }
  }

  window.SSGTM = { P, INPUT, GTMModel, spectrogram, normalize, bestWindowStart, fitInput, resampleTo, decodeTo44k, runJob };
})();
