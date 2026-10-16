/* Live microphone monitoring.
 *
 * Capture: getUserMedia (echo cancellation and noise suppression off, gain control
 * on - the rendering feature_extraction/live_render.py trains the Python model on)
 * -> AudioWorklet -> ring buffer at the device sample rate. Nothing is ever routed
 * to the speakers: that feedback path made the browser's echo canceller cancel the
 * microphone against itself. Every `hop` seconds the newest `window` seconds are
 * taken and, unless the window is silent:
 *   1. GTM runs here on the loudest ~1 s of the window (44.1 kHz spectrogram),
 *   2. the window is encoded as 16-bit WAV (22.05 kHz) and posted to the server
 *      together with GTM's scores; the server runs the Python model, compares
 *      both, applies the alert rules and answers with the decision.
 * GTM finishes before the request is sent, so it can never see the Python result.
 */
(function () {
  "use strict";
  const { el, chips, fmt, api, toast, chime } = window.SS;
  const cfg = JSON.parse(document.getElementById("live-config").textContent);
  window.SS_CLASS_CODES = cfg.classCodes;
  const $ = (id) => document.getElementById(id);
  const ui = {
    state: $("mic-state"), label: $("mic-label"), start: $("start-btn"), pause: $("pause-btn"), stop: $("stop-btn"),
    win: $("win"), hop: $("hop"), location: $("location"), privacy: $("privacy-strip"), banner: $("alert-banner"),
    device: $("mic-device"),
    nowCat: $("now-cat"), nowFacts: $("now-facts"), nowAction: $("now-action"), nowMeta: $("now-meta"),
    pyTop: $("py-top"), gtmTop: $("gtm-top"), gtmState: $("gtm-state"), level: $("level"), levelVal: $("level-val"),
    windows: $("windows"), scope: $("scope"), spec: $("livespec"),
    st: { windows: $("st-windows"), detections: $("st-detections"), alerts: $("st-alerts"), silent: $("st-silent"), latency: $("st-latency") },
  };
  const TARGET_SR = 22050;
  const S = {
    stream: null, ctx: null, source: null, node: null, analyser: null, sink: null, silentOut: null,
    session: null, windowSec: 2, hopSec: 1, running: false, paused: false, inflight: false,
    ring: null, write: 0, filled: 0, index: 0, timer: null, raf: null,
    gtm: null, stats: { windows: 0, detections: 0, alerts: 0, silent: 0, latency: [] },
  };

  /* ---------- microphone status ---------- */
  const LABELS = { available: "Microphone available", active: "Monitoring - microphone on", paused: "Paused",
                   disconnected: "Microphone disconnected", denied: "Permission denied", unknown: "Checking microphone..." };
  function setState(state, detail) {
    ui.state.dataset.state = state;
    ui.label.textContent = detail ? `${LABELS[state]} - ${detail}` : LABELS[state];
  }

  async function checkAvailability() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setState("disconnected", "this browser cannot access microphones (use HTTPS or localhost)");
      ui.start.disabled = true;
      return;
    }
    let permission = "prompt";
    try {
      const p = await navigator.permissions.query({ name: "microphone" });
      permission = p.state;
      p.onchange = () => { if (!S.running) checkAvailability(); };
    } catch (e) { /* Permissions API not supported for microphone (Firefox/Safari) */ }
    if (permission === "denied") { setState("denied", "allow the microphone in the browser's site settings"); return; }
    try {
      const inputs = await listInputs();
      if (!inputs.length) { setState("disconnected", "no microphone found"); return; }
    } catch (e) { /* ignore */ }
    setState("available", permission === "granted" ? "permission granted" : "permission will be requested");
  }

  /** Fill the microphone picker. Labels are only readable once permission has been granted. */
  async function listInputs() {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const inputs = devices.filter((d) => d.kind === "audioinput" && d.deviceId !== "communications");
    if (!ui.device) return inputs;
    const chosen = ui.device.value || (() => { try { return localStorage.getItem("ss-mic") || ""; } catch (e) { return ""; } })();
    ui.device.textContent = "";
    ui.device.append(el("option", { value: "", text: "Default input" }));
    inputs.forEach((d, i) => {
      const opt = el("option", { value: d.deviceId, text: d.label || `Microphone ${i + 1}` });
      if (d.deviceId === chosen) opt.selected = true;
      ui.device.append(opt);
    });
    return inputs;
  }

  if (ui.device) {
    ui.device.addEventListener("change", () => {
      try { localStorage.setItem("ss-mic", ui.device.value); } catch (e) { /* ignore */ }
    });
  }

  navigator.mediaDevices && navigator.mediaDevices.addEventListener && navigator.mediaDevices.addEventListener("devicechange", async () => {
    if (!S.running) { checkAvailability(); return; }
    const devices = await navigator.mediaDevices.enumerateDevices();
    if (!devices.some((d) => d.kind === "audioinput")) handleDisconnect();
  });

  /* ---------- consent ---------- */
  function askConsent() {
    return new Promise((resolve) => {
      const dlg = $("consent-dialog");
      const done = async (ok) => {
        dlg.close();
        if (ok) {
          try { await api(cfg.consentUrl, { method: "POST", json: {} }); cfg.consent = true; } catch (e) { ok = false; toast({ title: "Could not save consent", body: e.message, severity: "Medium" }); }
        }
        resolve(ok);
      };
      $("consent-ok").onclick = () => done(true);
      $("consent-cancel").onclick = () => done(false);
      dlg.showModal();
    });
  }

  /* ---------- start / pause / stop ---------- */
  async function start() {
    if (S.running) return;
    if (!cfg.consent && !(await askConsent())) return;
    ui.start.disabled = true;
    S.windowSec = Number(ui.win.value);
    S.hopSec = Math.min(S.windowSec, Number(ui.hop.value));
    try {
      const wanted = ui.device ? ui.device.value : "";
      S.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          // Echo cancellation is for voice calls: it removes whatever the page is playing from the
          // capture, and it also gates non-speech sounds. This page is a sound monitor, so it stays off.
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: true,
          ...(wanted ? { deviceId: { exact: wanted } } : {}),
        }
      });
    } catch (err) {
      try {
        S.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      } catch (err2) {
        ui.start.disabled = false;
        if (err2.name === "NotAllowedError" || err2.name === "SecurityError") setState("denied", "microphone access was blocked");
        else if (err2.name === "NotFoundError" || err2.name === "OverconstrainedError") setState("disconnected", "no microphone found");
        else setState("disconnected", err2.message);
        return;
      }
    }
    const track = S.stream.getAudioTracks()[0];
    track.addEventListener("ended", handleDisconnect);
    // "muted" on a track means the source cannot currently deliver data - a microphone muted in
    // Windows, or held exclusively by another application. That is silence the page cannot fix.
    const sayMuted = () => {
      setState("disconnected", `${track.label || "the microphone"} is muted`);
      toast({ title: "The microphone is muted", severity: "Medium", timeout: 9000,
              body: `Windows reports ${track.label || "this input"} as muted. Unmute it in Settings > System > `
                    + "Sound > Input (or on the device itself), or pick another microphone above." });
    };
    track.addEventListener("mute", sayMuted);
    track.addEventListener("unmute", () => { if (S.running) setState("active", track.label || "default input"); });
    if (track.muted) sayMuted();
    await listInputs();  // labels become readable once permission is granted

    S.ctx = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: "interactive" });
    if (S.ctx.state === "suspended") {
      await S.ctx.resume();
    }
    S.source = S.ctx.createMediaStreamSource(S.stream);
    S.analyser = S.ctx.createAnalyser();
    S.analyser.fftSize = 2048;
    S.analyser.smoothingTimeConstant = 0.15;
    S.source.connect(S.analyser);
    S.ring = new Float32Array(Math.ceil(S.ctx.sampleRate * 4));
    S.write = 0;
    S.filled = 0;
    // The graph needs a destination so the analyser and the capture node keep being pulled, but the
    // microphone must never reach the speakers: that is a feedback path, and with echo cancellation on
    // it makes the browser cancel the microphone against itself and deliver pure silence.
    S.sink = S.ctx.createGain();
    S.sink.gain.value = 0.00001;
    S.silentOut = S.ctx.createMediaStreamDestination();
    S.sink.connect(S.silentOut);
    S.analyser.connect(S.sink);

    let workletReady = false;
    if (S.ctx.audioWorklet && cfg.workletUrl) {
      try {
        await S.ctx.audioWorklet.addModule(cfg.workletUrl);
        const workletNode = new AudioWorkletNode(S.ctx, "capture-processor");
        workletNode.port.onmessage = (ev) => {
          if (!S.running || S.paused) return;
          push(ev.data);
        };
        S.source.connect(workletNode);
        workletNode.connect(S.sink);
        S.workletNode = workletNode;
        workletReady = true;
      } catch (e) {
        console.warn("AudioWorklet not available, falling back to ScriptProcessor:", e);
      }
    }
    if (!workletReady) {
      S.node = S.ctx.createScriptProcessor(2048, 1, 1);
      window._liveScriptNode = S.node; // Prevent garbage collection
      S.node.onaudioprocess = (ev) => {
        if (!S.running || S.paused) return;
        const ch = ev.inputBuffer.getChannelData(0);
        push(ch);
      };
      S.source.connect(S.node);
      S.node.connect(S.sink);
    }

    if (cfg.gtm && cfg.gtm.enabled) {
      ui.gtmState.textContent = "(loading...)";
      S.gtm = new window.SSGTM.GTMModel(cfg.gtm);
      try { await S.gtm.load(); ui.gtmState.textContent = ""; }
      catch (err) { S.gtm = null; ui.gtmState.textContent = `(unavailable: ${err.message})`; }
    }
    try {
      const res = await api(cfg.startUrl, { method: "POST", json: { window: S.windowSec, hop: S.hopSec, device: track.label || "", location: ui.location.value } });
      S.session = res.session;
    } catch (err) {
      stopAudio();
      ui.start.disabled = false;
      setState("available");
      toast({ title: "Monitoring could not start", body: err.message, severity: "Medium" });
      return;
    }
    S.running = true;
    S.paused = false;
    S.index = 0;
    S.silentRun = 0;
    S.silentWarned = false;
    setState("active", track.label || "default input");
    ui.nowMeta.textContent = `session ${S.session} · ${S.windowSec} s windows every ${S.hopSec} s`;
    ui.privacy.classList.remove("hidden");
    ui.start.classList.add("hidden");
    ui.pause.classList.remove("hidden");
    ui.stop.classList.remove("hidden");
    ui.win.disabled = ui.hop.disabled = ui.location.disabled = true;
    if (ui.device) ui.device.disabled = true;
    try { localStorage.setItem("ss-live", "on"); } catch (e) { /* ignore */ }
    const pill = document.getElementById("live-pill");
    if (pill) pill.classList.remove("hidden");
    S.timer = setInterval(tick, S.hopSec * 1000);
    draw();
    checkSignal(track);
  }

  /** Is the chosen input delivering anything at all? Report within two seconds, not after 30 windows. */
  function checkSignal(track) {
    const buf = new Float32Array(2048);
    let peak = 0;
    let checks = 0;
    const timer = setInterval(() => {
      if (!S.running || !S.analyser) { clearInterval(timer); return; }
      S.analyser.getFloatTimeDomainData(buf);
      for (let i = 0; i < buf.length; i++) peak = Math.max(peak, Math.abs(buf[i]));
      if (peak > 1e-6 || ++checks >= 8) {
        clearInterval(timer);
        if (peak > 1e-6) return;
        const name = track.label || "the selected microphone";
        const others = ui.device && ui.device.options.length > 2;
        setState("disconnected", `${name} is muted or delivering no sound`);
        toast({
          title: "This microphone is muted", severity: "Medium", timeout: 14000,
          body: `${name} is returning digital silence, which is what Windows sends while an input is muted. `
                + "Open Settings > System > Sound > Input, select this device and click the mute (speaker) "
                + "button so it is off, then start monitoring again"
                + (others ? ", or pick a different microphone in the list above." : "."),
        });
      }
    }, 250);
  }

  function push(block) {
    if (!block || !block.length) return;
    const n = S.ring.length;
    for (let i = 0; i < block.length; i++) {
      S.ring[S.write] = block[i];
      S.write = (S.write + 1) % n;
    }
    S.filled = Math.min(n, S.filled + block.length);
  }

  function latestWindow() {
    const targetLen = Math.round(S.windowSec * S.ctx.sampleRate);
    const minLen = Math.round(Math.min(S.windowSec, 1.0) * S.ctx.sampleRate);
    if (S.filled < minLen) return null;
    const len = Math.min(S.filled, targetLen);
    const out = new Float32Array(targetLen);
    const n = S.ring.length;
    let start = ((S.write - len) % n + n) % n;
    const offset = targetLen - len;
    for (let i = 0; i < len; i++) {
      out[offset + i] = S.ring[start];
      start = (start + 1) % n;
    }
    return out;
  }

  async function pauseResume() {
    if (!S.running) return;
    S.paused = !S.paused;
    setState(S.paused ? "paused" : "active");
    ui.pause.textContent = "";
    ui.pause.append(svgIcon(S.paused ? "play" : "pause"), S.paused ? "Resume" : "Pause");
    if (S.paused) S.ctx.suspend(); else S.ctx.resume();
    try { await api(`${cfg.sessionBase}${S.session}/status`, { method: "POST", json: { status: S.paused ? "paused" : "active" } }); } catch (e) { /* ignore */ }
  }

  function svgIcon(name) {
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("class", "icon");
    const use = document.createElementNS(ns, "use");
    use.setAttribute("href", `#i-${name}`);
    svg.appendChild(use);
    return svg;
  }

  function stopAudio() {
    clearInterval(S.timer);
    cancelAnimationFrame(S.raf);
    if (S.workletNode) {
      try { S.workletNode.disconnect(); } catch (e) {}
      S.workletNode = null;
    }
    if (S.node) {
      try { S.node.disconnect(); } catch (e) {}
      S.node.onaudioprocess = null;
      window._liveScriptNode = null;
    }
    if (S.silentOut) S.silentOut.stream.getTracks().forEach((t) => t.stop());
    if (S.stream) S.stream.getTracks().forEach((t) => t.stop());
    if (S.ctx) S.ctx.close();
    S.stream = S.ctx = S.node = S.analyser = S.workletNode = S.silentOut = null;
  }

  async function stop(reason = "stopped") {
    if (!S.running) return;
    S.running = false;
    stopAudio();
    try { await api(`${cfg.sessionBase}${S.session}/stop`, { method: "POST", json: { reason } }); } catch (e) { /* ignore */ }
    try { localStorage.removeItem("ss-live"); } catch (e) { /* ignore */ }
    const pill = document.getElementById("live-pill");
    if (pill) pill.classList.add("hidden");
    ui.privacy.classList.add("hidden");
    ui.start.classList.remove("hidden");
    ui.start.disabled = false;
    ui.pause.classList.add("hidden");
    ui.stop.classList.add("hidden");
    ui.win.disabled = ui.hop.disabled = ui.location.disabled = false;
    if (ui.device) ui.device.disabled = false;
    ui.level.firstElementChild.style.width = "0%";
    ui.nowMeta.textContent = `session ${S.session} ended`;
    if (reason === "disconnected") setState("disconnected", "the microphone was unplugged or turned off");
    else checkAvailability();
  }

  function handleDisconnect() {
    if (S.running) {
      toast({ title: "Microphone disconnected", body: "Monitoring stopped.", severity: "Medium" });
      stop("disconnected");
    }
  }

  /* ---------- per-window processing ---------- */
  function encodeWav(samples, rate) {
    const buf = new ArrayBuffer(44 + samples.length * 2);
    const v = new DataView(buf);
    const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    w(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); w(8, "WAVE"); w(12, "fmt ");
    v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    w(36, "data"); v.setUint32(40, samples.length * 2, true);
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Blob([buf], { type: "audio/wav" });
  }

  function renderLiveEdge(topCat, topConf, ranked) {
    if (!S.running || S.paused) return;
    ui.gtmTop.textContent = "";
    ui.gtmTop.append(...top3(ranked.slice(0, 3), "gtm"));
  }

  /** The server's silence rule (src/services/live_service.py), applied here before anything is classified. */
  function isSilent(x) {
    let peak = 0;
    let sum = 0;
    for (let i = 0; i < x.length; i++) {
      const a = Math.abs(x[i]);
      if (a > peak) peak = a;
      sum += x[i] * x[i];
    }
    const rmsDbfs = 20 * Math.log10(Math.sqrt(sum / x.length) + 1e-12);
    return peak < 1e-4 || rmsDbfs < -60;
  }

  async function tick() {
    if (!S.running || S.paused) return;
    const raw = latestWindow();
    if (!raw) return;
    const sr = S.ctx.sampleRate;
    const t0 = performance.now();
    // A silent window carries no sound to classify. Running the models on it anyway makes them
    // report a confident class for what is really just the dither floor.
    const silent = isSilent(raw);
    let gtm = null;
    try {
      if (S.gtm && !silent) {
        let y44 = await window.SSGTM.resampleTo(raw, sr, 44100);
        const ws = window.SSGTM.bestWindowStart(y44);
        const scores = await S.gtm.classifyWindow(y44.subarray(ws, ws + window.SSGTM.INPUT));
        gtm = { labels: S.gtm.labels, scores, model_version: S.gtm.version, window_start: ws / 44100 };

        // Real-Time Edge Visualization in the GTM card
        const ranked = Object.entries(scores).sort((a, b) => b[1] - a[1]);
        if (ranked.length > 0) {
          renderLiveEdge(ranked[0][0], ranked[0][1], ranked);
        }
      } else if (cfg.gtm && cfg.gtm.enabled && !silent && !S.gtm) {
        gtm = { error: "GTM model failed to load in the browser" };
      }
    } catch (err) {
      gtm = { error: String(err.message || err) };
    }

    if (S.inflight) return; // Keep server communication non-overlapping
    S.inflight = true;
    const index = ++S.index;
    try {
      let pcm = await window.SSGTM.resampleTo(raw, sr, TARGET_SR);
      const form = new FormData();
      form.append("audio", encodeWav(pcm, TARGET_SR), `window-${index}.wav`);
      form.append("meta", JSON.stringify({ window_index: index, client_time: Date.now() / 1000, gtm }));
      const res = await api(`${cfg.sessionBase}${S.session}/windows`, { method: "POST", body: form });
      res.round_trip_ms = Math.round(performance.now() - t0);
      render(res);
    } catch (err) {
      if (err.status === 409) { stop("stopped"); }
      else toast({ title: "Window could not be analysed", body: err.message, severity: "Medium", timeout: 5000 });
    } finally {
      S.inflight = false;
    }
  }

  function top3(list, kind) {
    return (list || []).map(([c, v], i) => el("li", { style: i ? "" : "" },
      el("span", { text: c }), el("span", { class: "v", text: fmt.pct(v, 1) }),
      el("div", { class: `bar ${kind}` }, el("span", { style: `width:${(v * 100).toFixed(1)}%` }))));
  }

  function render(r) {
    S.stats.windows += 1;
    if (r.status === "silence") S.stats.silent += 1;
    if (r.latency_ms !== undefined) S.stats.latency.push(r.round_trip_ms || r.latency_ms);
    if (S.stats.latency.length > 30) S.stats.latency.shift();
    ui.st.windows.textContent = S.stats.windows;
    ui.st.silent.textContent = S.stats.silent;
    const lat = S.stats.latency;
    ui.st.latency.textContent = lat.length ? `${Math.round(lat.reduce((a, b) => a + b, 0) / lat.length)} ms` : "-";

    const placeholder = ui.windows.querySelector(".placeholder");
    if (placeholder) placeholder.remove();
    let row;
    if (r.status === "silence") {
      row = el("tr", { class: "new" }, el("td", { class: "mono small", text: fmt.clock() }), el("td", { class: "num", text: r.window_index }),
        el("td", { colspan: 8, class: "muted", text: `Silence (${r.level_dbfs} dBFS) - not classified` }));
      // Nothing was heard, so no model result belongs on screen any more.
      S.silentRun = (S.silentRun || 0) + 1;
      ui.nowCat.textContent = "";
      ui.nowCat.append(el("span", { class: "muted", text: "Silence" }));
      ui.nowAction.textContent = "";
      ui.pyTop.textContent = "";
      ui.gtmTop.textContent = "";
      ui.nowFacts.textContent = "";
      ui.nowFacts.append(el("span", { class: "muted small", text: `Silence (${r.level_dbfs} dBFS) - nothing to classify.` }));
      // Every window silent while the user believes they are making noise means the browser is not
      // getting the microphone, not that the room is quiet. Say so instead of counting silent windows.
      if (S.silentRun >= 6 && S.stats.silent === S.stats.windows) {
        ui.nowAction.textContent = "No audio is arriving from the microphone. It is almost certainly muted: "
          + "open Settings > System > Sound > Input, select this device and turn its mute button off. "
          + "Also check that no other application has taken the microphone.";
        if (!S.silentWarned) {
          S.silentWarned = true;
          toast({ title: "The microphone is delivering silence", severity: "Medium", timeout: 9000,
                  body: "Every window so far is empty, which is what a muted input sends. Unmute the "
                        + "microphone in Settings > System > Sound > Input, then start monitoring again." });
        }
      }
    } else {
      S.silentRun = 0;
      if (!["Background Noise", "Unknown Sound"].includes(r.category)) {
        S.stats.detections += 1;
        ui.st.detections.textContent = S.stats.detections;
      }
      ui.nowCat.textContent = "";
      ui.nowCat.append(chips.cat(r.category, null, true));
      ui.nowFacts.textContent = "";
      window.SS.add(ui.nowFacts, chips.sev(r.severity), chips.agree(r.consistency), chips.quality(r.quality),
        el("span", { class: "chip chip-plain", text: `${fmt.pct(r.confidence)} ${r.confidence_level}` }),
        r.alert_status && r.alert_status !== "No Alert" ? el("span", { class: `chip ${r.alert_status === "Alert Generated" ? "alert-Active" : "chip-plain"}` }, el("span", { class: "glyph" }), r.alert_status === "Pending Confirmation" ? `Confirming (${r.consecutive})` : r.alert_status) : null);
      ui.nowAction.textContent = r.needs_review ? `Manual review: ${r.review_reasons.join("; ")}` : (r.category !== "Background Noise" ? r.recommended_action : "");
      ui.pyTop.textContent = "";
      ui.pyTop.append(...top3(r.python && r.python.top3, "python"));
      ui.gtmTop.textContent = "";
      if (r.gtm) ui.gtmTop.append(...top3(r.gtm.top3, "gtm"));
      else ui.gtmTop.append(el("li", { class: "muted", style: "font-weight:400", text: r.gtm_note || "No GTM result for this window." }));
      const link = el("a", { class: "id", href: r.record_url, text: r.audio_id });
      row = el("tr", { class: "new" },
        el("td", { class: "mono small", text: fmt.clock() }),
        el("td", { class: "num", text: r.window_index }),
        el("td", {}, chips.cat(r.category), el("div", { class: "small" }, link)),
        el("td", { class: "num", text: fmt.pct(r.confidence) }),
        el("td", { class: "small", text: r.python ? `${r.python.top} ${fmt.pct(r.python.confidence)}` : "-" }),
        el("td", { class: "small", text: r.gtm ? `${r.gtm.top} ${fmt.pct(r.gtm.confidence)}` : "-" }),
        el("td", {}, chips.agree(r.consistency)),
        el("td", {}, chips.sev(r.severity)),
        el("td", { class: "small", text: r.alert_status }),
        el("td", {}, chips.quality(r.quality)));
      if (r.alert) showAlert(r);
    }
    ui.windows.prepend(row);
    while (ui.windows.children.length > 30) ui.windows.lastChild.remove();
  }

  function showAlert(r) {
    S.stats.alerts += 1;
    ui.st.alerts.textContent = S.stats.alerts;
    const b = ui.banner;
    b.className = `banner mb-3${r.alert.severity === "Critical" ? "" : " high"}`;
    b.textContent = "";
    b.append(el("span", { class: "glyph-lg" }),
      el("div", { style: "flex:1;min-width:0" }, el("div", { class: "title", text: `${r.alert.severity}: ${r.category} detected` }),
        el("div", { class: "sub", text: `${r.alert.message} ${r.recommended_action}` })),
      el("a", { class: "btn btn-sm", href: r.record_url, text: "Open event" }),
      el("a", { class: "btn btn-sm", href: "/alerts", text: "Alerts" }));
    chime(r.alert.severity);
  }

  /* ---------- drawing: scope, spectrogram, level ---------- */
  function fitCanvas(c) {
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth;
    const h = c.clientHeight;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
      return true;
    }
    return false;
  }

  const timeBuf = new Float32Array(2048);
  let freqBuf = null;
  function draw() {
    if (!S.running || !S.analyser) return;
    const t = window.SS.chartTheme();
    // level
    S.analyser.getFloatTimeDomainData(timeBuf);
    let sum = 0;
    for (let i = 0; i < timeBuf.length; i++) sum += timeBuf[i] * timeBuf[i];
    const db = 20 * Math.log10(Math.sqrt(sum / timeBuf.length) + 1e-9);
    const pct = Math.max(0, Math.min(100, ((db + 70) / 70) * 100));
    ui.level.firstElementChild.style.width = `${pct}%`;
    ui.level.classList.toggle("hot", db > -3);
    ui.levelVal.textContent = `${Math.max(-99, db).toFixed(0)} dBFS`;
    // scope
    const c = ui.scope;
    fitCanvas(c);
    const g = c.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = c.clientWidth;
    const h = c.clientHeight;
    g.clearRect(0, 0, w, h);
    g.fillStyle = t.grid;
    g.fillRect(0, h / 2, w, 1);
    // Display auto-gain so quiet inputs stay visible (the audio itself is not changed).
    let peak = 0;
    for (let i = 0; i < timeBuf.length; i++) peak = Math.max(peak, Math.abs(timeBuf[i]));
    const target = Math.min(12, Math.max(1, 0.85 / Math.max(peak, 1e-4)));
    S.scopeGain = S.scopeGain ? S.scopeGain * 0.9 + target * 0.1 : target;
    g.strokeStyle = S.paused ? t.dataMuted : t.ink;
    g.lineWidth = 1.5;
    g.beginPath();
    for (let x = 0; x < w; x++) {
      const v = timeBuf[Math.floor((x / w) * timeBuf.length)];
      const y = h / 2 - v * (h / 2 - 6) * S.scopeGain;
      if (x === 0) g.moveTo(x, y); else g.lineTo(x, y);
    }
    g.stroke();
    g.fillStyle = t.ink3;
    g.font = "11px 'Plex Mono', monospace";
    g.fillText(`display gain x${S.scopeGain.toFixed(1)}`, w - 128, 16);
    // scrolling spectrogram (ink on paper, like a printed sonagram)
    const sc = ui.spec;
    const resized = fitCanvas(sc);
    const sg = sc.getContext("2d");
    const W = sc.width;
    const H = sc.height;
    if (resized) { sg.fillStyle = t.surface; sg.fillRect(0, 0, W, H); }
    if (!S.paused) {
      freqBuf = freqBuf || new Uint8Array(S.analyser.frequencyBinCount);
      S.analyser.getByteFrequencyData(freqBuf);
      sg.drawImage(sc, -2, 0);
      const maxBin = Math.floor((8000 / (S.ctx.sampleRate / 2)) * freqBuf.length);
      const col = sg.createImageData(2, H);
      const dark = document.documentElement.getAttribute("data-theme") === "dark" ||
        (!document.documentElement.getAttribute("data-theme") && window.matchMedia("(prefers-color-scheme: dark)").matches);
      for (let y = 0; y < H; y++) {
        const frac = 1 - y / H;
        const bin = Math.min(freqBuf.length - 1, Math.floor(Math.pow(frac, 1.6) * maxBin));
        const v = freqBuf[bin] / 255;
        const shade = dark ? 26 + v * 214 : 252 - v * 225;
        for (let x = 0; x < 2; x++) {
          const i = (y * 2 + x) * 4;
          col.data[i] = shade; col.data[i + 1] = shade * 0.99; col.data[i + 2] = shade * 0.96; col.data[i + 3] = 255;
        }
      }
      sg.putImageData(col, W - 2, 0);
    }
    S.raf = requestAnimationFrame(draw);
  }

  /* ---------- wiring ---------- */
  ui.start.addEventListener("click", start);
  ui.pause.addEventListener("click", pauseResume);
  ui.stop.addEventListener("click", () => stop("stopped"));
  window.addEventListener("pagehide", () => {
    if (S.running && S.session) {
      const form = new FormData();
      form.append("csrf_token", window.SS.csrf);
      navigator.sendBeacon(`${cfg.sessionBase}${S.session}/stop`, form);
      try { localStorage.removeItem("ss-live"); } catch (e) { /* ignore */ }
    }
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && S.running) toast({ title: "Monitoring continues in the background", body: "Keep this tab open.", severity: "Medium", timeout: 4000 });
  });
  checkAvailability();
})();
