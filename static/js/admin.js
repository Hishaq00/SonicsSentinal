/* Admin overview charts. */
(function () {
  "use strict";
  const C = window.SSCharts;
  const data = JSON.parse(document.getElementById("admin-data").textContent);
  const q = (id, sel) => document.querySelector(`#${id} ${sel}`);

  const cats = data.categories.slice().sort((a, b) => b.count - a.count);
  C.bar(q("adm-cat", "canvas"), { labels: cats.map((c) => c.category), horizontal: true,
    series: [{ label: "Events", data: cats.map((c) => c.count), color: (t) => t.data }] });
  C.tableView(q("adm-cat", ".table-view"), ["Category", "Events"], cats.map((c) => [c.category, c.count]));

  window.SS.api(data.trendUrl).then((d) => {
    const tr = d.trend;
    const hot = tr.labels.map((_, i) => (tr.series.Critical[i] || 0) + (tr.series.High[i] || 0));
    const rest = tr.labels.map((_, i) => ["Medium", "Low", "Informational"].reduce((s, k) => s + (tr.series[k][i] || 0), 0));
    C.bar(q("adm-trend", "canvas"), { labels: tr.labels.map((l) => l.slice(5)), stacked: true, series: [
      { label: "Critical & high", data: hot, color: (t) => t.critical },
      { label: "Other events", data: rest, color: (t) => t.dataMuted },
    ] });
    C.legend(q("adm-trend", ".legend"), [{ label: "Critical & high", color: "var(--critical)" }, { label: "Other events", color: "var(--data-muted)" }]);
    C.tableView(q("adm-trend", ".table-view"), ["Day", "Critical & high", "Other"], tr.labels.map((l, i) => [l, hot[i], rest[i]]));
  }).catch(() => {});
})();
