/* Overview page: 24-hour critical strip positions and the category chart. */
(function () {
  "use strict";
  const now = Date.now();
  document.querySelectorAll(".strip24 .mark").forEach((m) => {
    const t = new Date(m.dataset.ts).getTime();
    const pos = Math.min(100, Math.max(0, 100 - ((now - t) / 864e5) * 100));
    m.style.left = `${pos}%`;
  });

  const data = JSON.parse((document.getElementById("dash-data") || { textContent: "{}" }).textContent || "{}");
  const canvas = document.getElementById("cat-chart");
  if (!canvas || !data.categories || !window.SSCharts) return;
  const rows = data.categories.slice().sort((a, b) => b.count - a.count);
  window.SSCharts.bar(canvas, {
    labels: rows.map((r) => r.category),
    series: [{ label: "Events", data: rows.map((r) => r.count), color: (t) => t.data }],
    horizontal: true,
    valueFormat: (v) => `${v} event${v === 1 ? "" : "s"}`,
  });
  window.SSCharts.tableView(document.querySelector("#cat-card .table-view"), ["Category", "Events"],
    rows.map((r) => [r.category, r.count]));
})();
