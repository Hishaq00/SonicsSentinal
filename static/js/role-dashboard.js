/* Security / maintenance console: detections per day in the alerting categories. */
(function () {
  "use strict";
  const el = document.getElementById("role-dash-data");
  const C = window.SSCharts;
  if (!el || !C) return;
  const data = JSON.parse(el.textContent || "{}");
  const tr = data.trend || { days: [], series: {}, colors: {} };
  const card = document.getElementById("trend-card");
  const canvas = card && card.querySelector("canvas");
  if (!canvas) return;
  // Categories that were never heard would only add empty legend entries.
  const cats = (data.priority || []).filter((c) => (tr.series[c] || []).some((n) => n > 0));
  if (!cats.length) {
    canvas.parentElement.innerHTML = '<div class="empty"><div>No detections in your alerting categories in this period.</div></div>';
    return;
  }
  C.bar(canvas, {
    labels: tr.days.map((d) => d.slice(5)),
    stacked: true,
    series: cats.map((c) => ({ label: c, data: tr.series[c], color: tr.colors[c] })),
    valueFormat: (v) => `${v} detection${v === 1 ? "" : "s"}`,
  });
  C.legend(card.querySelector(".legend"), cats.map((c) => ({ label: c, color: tr.colors[c] })));
  C.tableView(card.querySelector(".table-view"), ["Day"].concat(cats),
    tr.days.map((d, i) => [d].concat(cats.map((c) => tr.series[c][i]))));
})();
