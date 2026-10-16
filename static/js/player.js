/* Audio preview: play, pause, replay, seek, volume, waveform with play-head,
 * detected-event track and "play segment" for reviewers. */
(function () {
  "use strict";

  class WavePlayer {
    constructor(root, opts) {
      this.root = root;
      this.opts = opts;
      this.peaks = (opts.waveform && opts.waveform.peaks) || [];
      this.duration = (opts.waveform && opts.waveform.duration) || opts.duration || 0;
      this.audio = new Audio();
      this.audio.preload = "metadata";
      this.audio.src = opts.src;
      this.segmentEnd = null;
      this.hoverX = null;
      this.canvas = root.querySelector("canvas.wave");
      this.wrap = root.querySelector(".wave-wrap");
      this.timeEl = root.querySelector("[data-time]");
      this.playBtn = root.querySelector("[data-play]");
      this.specHead = root.querySelector(".spec-playhead");
      this.bind();
      this.draw();
      new ResizeObserver(() => this.draw()).observe(this.wrap);
      document.addEventListener("ss:theme", () => this.draw());
    }

    bind() {
      const a = this.audio;
      this.playBtn.addEventListener("click", () => this.toggle());
      const replay = this.root.querySelector("[data-replay]");
      if (replay) replay.addEventListener("click", () => { this.segmentEnd = null; a.currentTime = 0; a.play(); });
      const vol = this.root.querySelector("[data-volume]");
      if (vol) {
        vol.addEventListener("input", () => { a.volume = Number(vol.value); });
        a.volume = Number(vol.value);
      }
      const src = this.root.querySelector("[data-source]");
      if (src) {
        src.addEventListener("change", () => {
          const t = a.currentTime;
          const playing = !a.paused;
          a.src = src.value;
          a.addEventListener("loadedmetadata", () => { a.currentTime = Math.min(t, a.duration || t); if (playing) a.play(); }, { once: true });
        });
      }
      a.addEventListener("play", () => this.setIcon(true));
      a.addEventListener("pause", () => this.setIcon(false));
      a.addEventListener("ended", () => { this.setIcon(false); this.segmentEnd = null; });
      a.addEventListener("loadedmetadata", () => { if (!this.duration) this.duration = a.duration; this.updateTime(); });
      a.addEventListener("error", () => {
        this.timeEl.textContent = "Audio unavailable";
        this.playBtn.disabled = true;
      });
      const tick = () => {
        if (this.segmentEnd !== null && a.currentTime >= this.segmentEnd) { a.pause(); this.segmentEnd = null; }
        this.updateTime();
        this.draw();
        if (!a.paused) requestAnimationFrame(tick);
      };
      a.addEventListener("play", () => requestAnimationFrame(tick));
      a.addEventListener("seeked", () => { this.updateTime(); this.draw(); });

      const seekFromEvent = (e) => {
        const r = this.wrap.getBoundingClientRect();
        const x = Math.min(Math.max(0, e.clientX - r.left), r.width);
        return (x / r.width) * (this.duration || a.duration || 0);
      };
      let dragging = false;
      this.wrap.addEventListener("pointerdown", (e) => { dragging = true; this.wrap.setPointerCapture(e.pointerId); this.segmentEnd = null; a.currentTime = seekFromEvent(e); });
      this.wrap.addEventListener("pointermove", (e) => {
        const r = this.wrap.getBoundingClientRect();
        this.hoverX = e.clientX - r.left;
        if (dragging) a.currentTime = seekFromEvent(e);
        this.draw();
      });
      this.wrap.addEventListener("pointerup", () => { dragging = false; });
      this.wrap.addEventListener("pointerleave", () => { this.hoverX = null; this.draw(); });
      this.wrap.tabIndex = 0;
      this.wrap.setAttribute("role", "slider");
      this.wrap.setAttribute("aria-label", "Seek audio");
      this.wrap.addEventListener("keydown", (e) => {
        if (e.key === "ArrowRight") { a.currentTime = Math.min((a.duration || 0), a.currentTime + 1); e.preventDefault(); }
        if (e.key === "ArrowLeft") { a.currentTime = Math.max(0, a.currentTime - 1); e.preventDefault(); }
        if (e.key === " " || e.key === "Enter") { this.toggle(); e.preventDefault(); }
      });
    }

    setIcon(playing) {
      const use = this.playBtn.querySelector("use");
      if (use) use.setAttribute("href", playing ? "#i-pause" : "#i-play");
      this.playBtn.setAttribute("aria-label", playing ? "Pause" : "Play");
    }

    toggle() {
      if (this.audio.paused) { this.audio.play().catch(() => {}); } else { this.audio.pause(); }
    }

    playSegment(start, end) {
      this.segmentEnd = end;
      this.audio.currentTime = start;
      this.audio.play().catch(() => {});
    }

    updateTime() {
      const f = window.SS.fmt.tc;
      this.timeEl.textContent = `${f(this.audio.currentTime)} / ${f(this.duration || this.audio.duration || 0)}`;
      if (this.specHead && this.duration) {
        this.specHead.style.left = `${(this.audio.currentTime / this.duration) * 100}%`;
      }
      this.wrap.setAttribute("aria-valuenow", this.audio.currentTime.toFixed(1));
    }

    draw() {
      const c = this.canvas;
      const dpr = window.devicePixelRatio || 1;
      const w = this.wrap.clientWidth;
      const h = this.wrap.clientHeight;
      if (!w || !h) return;
      if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
        c.width = Math.round(w * dpr);
        c.height = Math.round(h * dpr);
      }
      const ctx = c.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const t = window.SS.chartTheme();
      const mid = h / 2;
      ctx.fillStyle = t.grid;
      ctx.fillRect(0, mid, w, 1);
      const n = this.peaks.length;
      if (!n) return;
      const progress = this.duration ? this.audio.currentTime / this.duration : 0;
      const cols = Math.max(1, Math.floor(w / 2));
      for (let i = 0; i < cols; i++) {
        const a = Math.floor((i / cols) * n);
        const b = Math.max(a + 1, Math.floor(((i + 1) / cols) * n));
        let lo = 0;
        let hi = 0;
        for (let j = a; j < b && j < n; j++) { lo = Math.min(lo, this.peaks[j][0]); hi = Math.max(hi, this.peaks[j][1]); }
        const peak = (this.opts.waveform && this.opts.waveform.peak) || 1;
        const scale = (mid - 6) / Math.max(peak, 1e-3);
        const x = i * 2;
        ctx.fillStyle = x / w <= progress ? t.ink : t.dataMuted;
        const y0 = mid - hi * scale;
        const y1 = mid - lo * scale;
        ctx.fillRect(x, y0, 1.2, Math.max(1, y1 - y0));
      }
      ctx.fillStyle = t.ink;
      ctx.fillRect(Math.round(progress * w), 0, 1.5, h);
      if (this.hoverX !== null) {
        ctx.fillStyle = t.ink3;
        ctx.fillRect(Math.round(this.hoverX), 0, 1, h);
        const sec = (this.hoverX / w) * (this.duration || 0);
        ctx.font = "11px 'Plex Mono', monospace";
        const label = window.SS.fmt.tc(sec);
        const tw = ctx.measureText(label).width + 8;
        const lx = Math.min(Math.max(0, this.hoverX + 6), w - tw);
        ctx.fillStyle = t.surface;
        ctx.fillRect(lx, 4, tw, 16);
        ctx.fillStyle = t.ink2;
        ctx.fillText(label, lx + 4, 16);
      }
    }
  }

  window.WavePlayer = WavePlayer;
})();
