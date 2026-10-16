/* Model-comparison runner: one test recording at a time (server: Python model,
 * this browser: GTM on the same segments), then the summary is computed. */
(function () {
  "use strict";
  const { api } = window.SS;
  const cfg = JSON.parse(document.getElementById("eval-config").textContent);
  const runBtn = document.getElementById("run-btn");
  const stopBtn = document.getElementById("stop-btn");
  const bar = document.getElementById("run-bar");
  const status = document.getElementById("run-status");
  const log = document.getElementById("run-log");
  const gtm = cfg.gtm && cfg.gtm.enabled ? new window.SSGTM.GTMModel(cfg.gtm) : null;
  let running = false;
  let done = cfg.completed;
  let failures = 0;

  function progress() {
    bar.style.width = `${((done / (cfg.total || 1)) * 100).toFixed(1)}%`;
    status.textContent = `${done} of ${cfg.total} analysed${failures ? ` · ${failures} failed` : ""}`;
  }

  async function loop() {
    while (running) {
      let step;
      try {
        step = await api(cfg.nextUrl, { method: "POST", json: {} });
      } catch (err) {
        log.textContent = `Error: ${err.message}`;
        running = false;
        break;
      }
      if (step.done) {
        log.textContent = "All recordings analysed. Building the report...";
        await api(cfg.finishUrl, { method: "POST", json: {} });
        window.location.reload();
        return;
      }
      if (step.failed) {
        failures += 1;
        log.textContent = `${step.failed}: ${step.error}`;
        continue;
      }
      log.textContent = `${step.dataset_audio_id} (${step.actual})`;
      if (step.gtm_job && gtm) {
        try { await window.SSGTM.runJob(step.gtm_job, gtm); }
        catch (err) { log.textContent = `GTM error: ${err.message}`; }
      }
      done = step.completed;
      progress();
    }
    runBtn.classList.remove("hidden");
    stopBtn.classList.add("hidden");
  }

  runBtn.addEventListener("click", async () => {
    if (gtm) {
      log.textContent = "Loading the Teachable Machine model...";
      try { await gtm.load(); } catch (err) { log.textContent = `GTM model could not load (${err.message}); the run continues with the Python model only.`; }
    }
    running = true;
    runBtn.classList.add("hidden");
    stopBtn.classList.remove("hidden");
    loop();
  });
  stopBtn.addEventListener("click", () => { running = false; log.textContent = "Paused - press Resume to continue."; });
  window.addEventListener("beforeunload", (e) => { if (running) { e.preventDefault(); e.returnValue = ""; } });
  progress();
})();
