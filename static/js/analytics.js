/* Analytics page charts (each with a legend where >= 2 series and a table view). */
(function () {
  "use strict";
  const { api, el, fmt } = window.SS;
  const C = window.SSCharts;
  const cfg = JSON.parse(document.getElementById("analytics-config").textContent);

  function card(id) {
    const root = document.getElementById(id);
    return { root, canvas: root.querySelector("canvas"), table: root.querySelector(".table-view"), legend: root.querySelector(".legend") };
  }

  function stackBar(container, parts, total) {
    container.textContent = "";
    if (!total) { container.append(el("p", { class: "muted mb-0", text: "No data in this period." })); return; }
    const bar = el("div", { class: "stackbar", role: "img", "aria-label": parts.map((p) => `${p.label} ${p.value}`).join(", ") });
    parts.filter((p) => p.value > 0).forEach((p) => bar.append(el("span", { style: `flex:${p.value};background:${p.color}`, title: `${p.label}: ${p.value}` })));
    const table = el("table", { class: "table compact mt-2" }, el("tbody", {}, parts.map((p) =>
      el("tr", {}, el("td", {}, el("span", { class: "legend", style: "padding:0;display:inline-flex" }, el("span", {}, el("i", { style: `background:${p.color}` }), p.label))),
        el("td", { class: "num", text: p.value }), el("td", { class: "num muted", text: fmt.pct(p.value / total) })))));
    container.append(bar, table);
  }

  api(cfg.url).then((d) => {
    /* category frequency: one series, one colour */
    const cats = d.categories.slice().sort((a, b) => b.count - a.count);
    const c1 = card("c-cat");
    C.bar(c1.canvas, { labels: cats.map((c) => c.category), series: [{ label: "Events", data: cats.map((c) => c.count), color: (t) => t.data }], horizontal: true,
      valueFormat: (v) => `${v} event${v === 1 ? "" : "s"}` });
    C.tableView(c1.table, ["Category", "Events"], cats.map((c) => [c.category, c.count]));

    /* detections per day: emphasis on critical + high */
    const tr = d.trend;
    const hot = tr.labels.map((_, i) => (tr.series.Critical[i] || 0) + (tr.series.High[i] || 0));
    const rest = tr.labels.map((_, i) => ["Medium", "Low", "Informational"].reduce((s, k) => s + (tr.series[k][i] || 0), 0));
    const short = tr.labels.map((l) => l.slice(5));
    const c2 = card("c-trend");
    C.bar(c2.canvas, { labels: short, stacked: true, series: [
      { label: "Critical & high", data: hot, color: (t) => t.critical },
      { label: "Other events", data: rest, color: (t) => t.dataMuted },
    ] });
    C.legend(c2.legend, [{ label: "Critical & high", color: "var(--critical)" }, { label: "Other events", color: "var(--data-muted)" }]);
    C.tableView(c2.table, ["Day", "Critical & high", "Other"], tr.labels.map((l, i) => [l, hot[i], rest[i]]));

    /* confidence histogram: Python vs GTM */
    const cf = d.confidence;
    const c3 = card("c-conf");
    C.bar(c3.canvas, { labels: cf.labels, series: [
      { label: "Python", data: cf.python, color: (t) => t.python },
      { label: "GTM", data: cf.gtm, color: (t) => t.gtm },
    ], valueTitle: "Recordings" });
    C.legend(c3.legend, [{ label: "Python model", color: "var(--python)" }, { label: "Teachable Machine", color: "var(--gtm)" }]);
    C.tableView(c3.table, ["Top-class confidence", "Python", "GTM", "Final (fused)"], cf.labels.map((l, i) => [l, cf.python[i], cf.gtm[i], cf.final[i]]));

    /* scatter: agreement is the story */
    const agree = d.scatter.filter((p) => p[2]).map((p) => ({ x: p[0], y: p[1] }));
    const disagree = d.scatter.filter((p) => !p[2]).map((p) => ({ x: p[0], y: p[1] }));
    const c4 = card("c-scatter");
    C.scatter(c4.canvas, { xTitle: "Python confidence", yTitle: "GTM confidence", series: [
      { label: "Models agree", data: agree, color: (t) => t.data },
      { label: "Models disagree", data: disagree, color: (t) => t.serious },
    ] });
    C.legend(c4.legend, [{ label: "Models agree", color: "var(--data)" }, { label: "Models disagree", color: "var(--serious)" }]);
    const mean = (arr, k) => (arr.length ? arr.reduce((s, p) => s + p[k], 0) / arr.length : null);
    C.tableView(c4.table, ["Group", "Recordings", "Mean Python conf.", "Mean GTM conf."], [
      ["Agree", agree.length, fmt.pct(mean(agree, "x")), fmt.pct(mean(agree, "y"))],
      ["Disagree", disagree.length, fmt.pct(mean(disagree, "x")), fmt.pct(mean(disagree, "y"))],
    ]);

    /* quality and agreement: part-to-whole bars with numbers */
    const q = d.quality;
    const qTotal = Object.values(q).reduce((a, b) => a + b, 0);
    stackBar(document.getElementById("quality-block"), [
      { label: "Good", value: q.Good, color: "var(--good)" },
      { label: "Acceptable", value: q.Acceptable, color: "var(--data)" },
      { label: "Poor", value: q.Poor, color: "var(--warning)" },
      { label: "Unusable", value: q.Unusable, color: "var(--critical)" },
    ], qTotal);
    const a = d.consistency;
    const order = [["Acceptable Match", "var(--good)"], ["Weak Match", "var(--warning)"], ["Model Disagreement", "var(--serious)"], ["Uncertain Result", "var(--neutral)"], ["Single-Model Result", "var(--data-muted)"]];
    const aParts = order.map(([k, color]) => ({ label: k, value: a[k] || 0, color }));
    stackBar(document.getElementById("agree-block"), aParts, aParts.reduce((s, p) => s + p.value, 0));
  }).catch((err) => {
    window.SS.toast({ title: "Analytics could not load", body: err.message, severity: "Medium" });
  });
})();
