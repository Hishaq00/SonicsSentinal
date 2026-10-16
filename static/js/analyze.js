/* Analyze audio: file selection, preview, upload with progress, GTM hand-off and results. */
(function () {
  "use strict";
  const { el, chips, fmt, api } = window.SS;
  const cfg = JSON.parse(document.getElementById("analyze-config").textContent);
  window.SS_CLASS_CODES = cfg.classCodes;
  const input = document.getElementById("file-input");
  const drop = document.getElementById("dropzone");
  const list = document.getElementById("file-list");
  const analyzeBtn = document.getElementById("analyze-btn");
  const clearBtn = document.getElementById("clear-btn");
  const summary = document.getElementById("queue-summary");
  const batchLink = document.getElementById("batch-link");
  const gtm = cfg.gtm && cfg.gtm.enabled ? new window.SSGTM.GTMModel(cfg.gtm) : null;
  const STEPS = [["upload", "Upload"], ["validate", "Validate"], ["pre", "Pre-process"], ["python", "Python model"], ["gtm", "GTM model"], ["decision", "Decision"]];
  let items = [];
  let running = false;

  function bytes(n) { return n > 1048576 ? `${(n / 1048576).toFixed(1)} MB` : `${Math.round(n / 1024)} KB`; }

  function precheck(file) {
    const ext = (file.name.split(".").pop() || "").toLowerCase();
    if (!cfg.formats.includes(ext)) return `Unsupported format .${ext}. Allowed: ${cfg.formats.join(", ").toUpperCase()}.`;
    if (file.size > cfg.maxMb * 1048576) return `File is ${bytes(file.size)}; the limit is ${cfg.maxMb} MB.`;
    if (file.size === 0) return "The file is empty.";
    return null;
  }

  function stepper() {
    return el("div", { class: "pipeline" }, STEPS.map(([key, label]) => el("span", { class: `step ${key}`, "data-step": key, text: label })));
  }

  function setStep(item, key, state) {
    const idx = STEPS.findIndex((s) => s[0] === key);
    item.node.querySelectorAll(".pipeline .step").forEach((s, i) => {
      if (state === "fail" && i === idx) { s.className = `step ${s.dataset.step} fail`; return; }
      if (i < idx || (i === idx && state === "done")) s.className = `step ${s.dataset.step} done`;
      else if (i === idx) s.className = `step ${s.dataset.step} active`;
    });
  }

  function addFiles(files) {
    if (running) return;
    const incoming = Array.from(files);
    if (!cfg.canBatch) { items.forEach((i) => URL.revokeObjectURL(i.url)); items = []; list.textContent = ""; }
    for (const file of incoming) {
      if (items.length >= (cfg.canBatch ? cfg.maxFiles : 1)) break;
      const url = URL.createObjectURL(file);
      const problem = precheck(file);
      const node = el("li", { class: "file-item" },
        el("div", { class: "name", title: file.name, text: file.name }),
        el("button", { class: "btn btn-sm btn-ghost", type: "button", "aria-label": `Remove ${file.name}`, text: "Remove",
                       onclick: () => remove(item) }),
        el("div", { class: "sub", text: `${bytes(file.size)} · ${file.type || "audio"}${problem ? "" : " · ready"}` }),
        el("audio", { controls: true, preload: "metadata", src: url }),
        stepper(),
        el("div", { class: "result-inline" }));
      const item = { file, url, node, problem, done: false };
      if (problem) {
        node.querySelector(".sub").textContent = problem;
        node.querySelector(".sub").classList.add("errors");
        setStep(item, "upload", "fail");
      }
      items.push(item);
      list.appendChild(node);
    }
    refresh();
  }

  function remove(item) {
    if (running) return;
    URL.revokeObjectURL(item.url);
    item.node.remove();
    items = items.filter((i) => i !== item);
    refresh();
  }

  function refresh() {
    const ready = items.filter((i) => !i.problem && !i.done);
    analyzeBtn.disabled = running || ready.length === 0;
    clearBtn.disabled = running || items.length === 0;
    summary.textContent = items.length ? `${items.length} file${items.length > 1 ? "s" : ""} selected, ${ready.length} ready to analyse.` : "No files selected.";
  }

  function upload(file, batch, onProgress) {
    return new Promise((resolve, reject) => {
      const form = new FormData();
      form.append("file", file);
      if (batch) form.append("batch", batch);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", cfg.uploadUrl);
      xhr.setRequestHeader("X-CSRFToken", window.SS.csrf);
      xhr.setRequestHeader("Accept", "application/json");
      xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total); };
      xhr.onload = () => {
        let data = null;
        try { data = JSON.parse(xhr.responseText); } catch (e) { /* not JSON */ }
        if (xhr.status >= 200 && xhr.status < 300 && data) resolve(data);
        else reject(Object.assign(new Error((data && data.error) || `Upload failed (${xhr.status})`), { data, status: xhr.status }));
      };
      xhr.onerror = () => reject(new Error("Network error during upload."));
      xhr.send(form);
    });
  }

  function renderChecks(item, checks) {
    if (!checks || !checks.length) return;
    const ul = el("ul", { class: "checks" }, checks.map((c) =>
      el("li", {}, el("span", { class: c.status, text: c.status === "pass" ? "✓" : c.status === "warn" ? "!" : "✗" }),
        el("span", { class: "muted", text: c.name }), el("span", { text: c.detail }))));
    item.node.appendChild(ul);
  }

  function renderResult(item, rec) {
    const box = item.node.querySelector(".result-inline");
    box.textContent = "";
    window.SS.add(box,
      chips.cat(rec.category),
      el("span", { class: "muted", text: rec.confidence !== null && rec.confidence !== undefined ? fmt.pct(rec.confidence) : "" }),
      chips.sev(rec.severity), chips.agree(rec.consistency), chips.quality(rec.quality),
      rec.alert_status === "Alert Generated" ? el("span", { class: "chip alert-Active" }, el("span", { class: "glyph" }), "Alert raised") : null,
      rec.needs_review ? el("span", { class: "chip status status-Manual-Review" }, el("span", { class: "glyph" }), "Manual review") : null,
      rec.duplicate_of ? el("span", { class: "tag", text: `Duplicate of ${rec.duplicate_of}` }) : null,
      rec.near_duplicate_of ? el("span", { class: "tag", text: `Near-duplicate of ${rec.near_duplicate_of}` }) : null,
      el("a", { class: "btn btn-sm", href: rec.url, text: "Open analysis" }));
  }

  async function processItem(item, batch) {
    const sub = item.node.querySelector(".sub");
    setStep(item, "upload", "active");
    try {
      const res = await upload(item.file, batch, (p) => { sub.textContent = `Uploading ${Math.round(p * 100)}%`; });
      setStep(item, "python", "done");
      const rec = res.record;
      sub.textContent = `${rec.audio_id} · ${rec.duration ? rec.duration.toFixed(1) : "?"} s · quality ${rec.quality}${rec.processing_ms ? ` · server ${rec.processing_ms} ms` : ""}`;
      let final = rec;
      if (res.gtm_job) {
        setStep(item, "gtm", "active");
        const out = await window.SSGTM.runJob(res.gtm_job, gtm, (i, n) => { sub.textContent = `${rec.audio_id} · GTM segment ${i}/${n}`; });
        final = out.record || rec;
        if (final.gtm_status === "failed") sub.textContent += " · GTM failed, Python-only result";
      }
      setStep(item, "decision", "done");
      if (!res.gtm_job) {
        const gtmStep = item.node.querySelector('[data-step="gtm"]');
        gtmStep.textContent = "GTM n/a";
      }
      renderResult(item, final.url ? final : Object.assign({}, rec, final));
      if (!final.category) {
        const fresh = await api(`/api/records/${rec.audio_id}`);
        renderResult(item, fresh);
      }
      item.done = true;
    } catch (err) {
      const step = err.status === 400 && err.data && err.data.checks ? "validate" : err.status === 503 ? "python" : "upload";
      setStep(item, step, "fail");
      sub.textContent = err.message;
      sub.classList.add("errors");
      if (err.data && err.data.checks) renderChecks(item, err.data.checks);
      item.done = true;
    }
  }

  analyzeBtn.addEventListener("click", async () => {
    const ready = items.filter((i) => !i.problem && !i.done);
    if (!ready.length) return;
    running = true;
    refresh();
    let batch = null;
    try {
      if (cfg.canBatch && ready.length > 1) {
        const b = await api(cfg.batchUrl, { method: "POST", json: { files: ready.length } });
        batch = b.batch;
        batchLink.classList.remove("hidden");
        batchLink.textContent = "";
        batchLink.append(`Batch ${b.batch}: `, el("a", { href: b.url, text: "open the batch summary" }), " when processing has finished.");
      }
    } catch (err) {
      window.SS.toast({ title: "Batch could not be created", body: err.message, severity: "Medium" });
      running = false;
      refresh();
      return;
    }
    for (const item of ready) {
      item.node.querySelectorAll("button").forEach((b) => { b.disabled = true; });
      await processItem(item, batch);
    }
    running = false;
    refresh();
  });

  clearBtn.addEventListener("click", () => {
    items.forEach((i) => URL.revokeObjectURL(i.url));
    items = [];
    list.textContent = "";
    batchLink.classList.add("hidden");
    refresh();
  });

  input.addEventListener("change", () => { addFiles(input.files); input.value = ""; });
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));
})();
