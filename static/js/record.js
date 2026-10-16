/* Record detail page: finish a pending GTM job, audio player, event track,
 * spectrogram axes, segment playback and the reviewer form. */
(function () {
  "use strict";
  const { el, fmt, api } = window.SS;
  const data = JSON.parse(document.getElementById("record-data").textContent);

  /* 1. GTM still pending: run it in this browser, then reload. */
  if (data.pending && window.SSGTM) {
    const progress = document.getElementById("gtm-progress");
    (async () => {
      try {
        const job = await api(data.jobUrl);
        if (job.status !== "pending") { window.location.reload(); return; }
        const model = new window.SSGTM.GTMModel(job.gtm);
        await window.SSGTM.runJob(job, model, (i, n) => { if (progress) progress.textContent = `${i}/${n}`; });
      } catch (err) {
        if (progress) progress.textContent = `(${err.message})`;
      }
      window.location.reload();
    })();
  }

  /* 2. Player */
  let player = null;
  const root = document.getElementById("player");
  if (root && data.src) {
    player = new window.WavePlayer(root, { src: data.src, waveform: data.waveform, duration: data.duration });
  }
  const duration = (data.waveform && data.waveform.duration) || data.duration || 0;

  /* 3. Event track under the waveform */
  const track = document.getElementById("seg-track");
  if (track && duration) {
    (data.events || []).forEach((ev) => {
      const left = (ev.start / duration) * 100;
      const width = Math.max(0.8, ((Math.min(ev.end, duration) - ev.start) / duration) * 100);
      const isBg = ev.category === "Background Noise";
      const node = el("button", {
        class: `ev sev-${ev.severity}${isBg ? " bg" : ""}`, type: "button",
        style: `left:${left}%;width:${width}%`,
        title: `${ev.category} ${fmt.tc(ev.start)}–${fmt.tc(ev.end)} (${fmt.pct(ev.confidence)})`,
        text: isBg ? "" : ev.category,
      });
      node.style.border = "0";
      node.style.cursor = "pointer";
      node.addEventListener("click", () => player && player.playSegment(ev.start, ev.end));
      track.appendChild(node);
    });
  }

  /* 4. Spectrogram frequency axis and time axis */
  const axis = document.getElementById("spec-axis");
  if (axis && data.waveform && data.waveform.spectrogram_ticks) {
    data.waveform.spectrogram_ticks.forEach((t) => {
      const label = t.hz >= 1000 ? `${t.hz / 1000}k` : `${t.hz}`;
      axis.appendChild(el("span", { style: `bottom:${(t.pos * 100).toFixed(1)}%`, text: label }));
    });
  }
  const timeAxis = document.getElementById("time-axis");
  if (timeAxis && duration) {
    for (let i = 0; i <= 4; i++) timeAxis.appendChild(el("span", { text: fmt.tc((duration * i) / 4) }));
  }

  /* 5. Segment playback */
  document.querySelectorAll("[data-play-seg]").forEach((b) => {
    b.addEventListener("click", () => {
      if (!player) return;
      player.playSegment(Number(b.dataset.playSeg), Number(b.dataset.segEnd));
      root.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  });

  /* 6. Reviewer form */
  const form = document.getElementById("review-form");
  if (form) {
    const select = form.querySelector("select[name=category]");
    const sync = () => {
      const v = form.querySelector("input[name=decision]:checked").value;
      select.disabled = v === "confirmed";
    };
    form.querySelectorAll("input[name=decision]").forEach((r) => r.addEventListener("change", sync));
    sync();
  }
})();
