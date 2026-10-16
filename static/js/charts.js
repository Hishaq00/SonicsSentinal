/* Chart helpers on top of Chart.js, following the project's chart rules:
 * thin bars (<= 24 px) with 4 px rounded data-ends and a square baseline,
 * 2 px lines, >= 8 px markers with a 2 px surface ring, 2 px surface gaps
 * between stacked segments, hairline solid grid, one axis, legend for >= 2
 * series, an HTML tooltip where the value leads, and a table view twin.
 */
(function () {
  "use strict";
  const charts = [];

  function theme() { return window.SS.chartTheme(); }

  function tooltipEl(chart) {
    let tip = chart.canvas.parentNode.querySelector(".ss-tip");
    if (!tip) {
      tip = document.createElement("div");
      tip.className = "ss-tip";
      tip.setAttribute("role", "status");
      Object.assign(tip.style, {
        position: "absolute", pointerEvents: "none", zIndex: 5, minWidth: "120px",
        background: "var(--surface)", border: "1px solid var(--rule-strong)", borderRadius: "6px",
        boxShadow: "var(--shadow-pop)", padding: "7px 9px", fontSize: "12px", color: "var(--ink)",
        transition: "opacity .08s", opacity: 0, whiteSpace: "nowrap",
      });
      chart.canvas.parentNode.appendChild(tip);
    }
    return tip;
  }

  function externalTooltip(valueFormat) {
    return (context) => {
      const { chart, tooltip } = context;
      const tip = tooltipEl(chart);
      if (tooltip.opacity === 0) { tip.style.opacity = 0; return; }
      tip.textContent = "";
      const head = document.createElement("div");
      head.style.cssText = "color:var(--ink-3);margin-bottom:4px";
      head.textContent = (tooltip.title || []).join(" ");
      tip.appendChild(head);
      (tooltip.dataPoints || []).forEach((dp) => {
        const row = document.createElement("div");
        row.style.cssText = "display:flex;align-items:center;gap:7px;line-height:1.5";
        const key = document.createElement("span");
        key.style.cssText = `display:inline-block;width:12px;height:3px;border-radius:2px;background:${dp.dataset.keyColor || dp.dataset.backgroundColor}`;
        const val = document.createElement("b");
        val.textContent = valueFormat ? valueFormat(dp.raw, dp) : String(dp.formattedValue);
        const name = document.createElement("span");
        name.style.color = "var(--ink-3)";
        name.textContent = dp.dataset.label || "";
        row.append(key, val, name);
        tip.appendChild(row);
      });
      const { offsetLeft: left, offsetTop: top } = chart.canvas;
      const x = Math.min(left + tooltip.caretX + 12, chart.canvas.parentNode.clientWidth - tip.offsetWidth - 4);
      tip.style.left = `${Math.max(4, x)}px`;
      tip.style.top = `${Math.max(4, top + tooltip.caretY - tip.offsetHeight - 8)}px`;
      tip.style.opacity = 1;
    };
  }

  function baseOptions(t, opts = {}) {
    const horizontal = !!opts.horizontal;
    const valueAxis = {
      beginAtZero: true, stacked: !!opts.stacked,
      grid: { color: t.grid, lineWidth: 1, drawTicks: false },
      border: { display: false },
      ticks: { color: t.ink3, padding: 6, precision: opts.precision ?? 0, font: { size: 11 },
               callback: opts.valueTick || undefined },
      max: opts.max,
      title: opts.valueTitle ? { display: true, text: opts.valueTitle, color: t.ink3, font: { size: 11 } } : undefined,
    };
    const catAxis = {
      stacked: !!opts.stacked,
      grid: { display: false },
      border: { color: t.axis },
      ticks: { color: t.ink2, font: { size: 11 }, autoSkip: true, maxRotation: 0 },
      title: opts.catTitle ? { display: true, text: opts.catTitle, color: t.ink3, font: { size: 11 } } : undefined,
    };
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 250 },
      indexAxis: horizontal ? "y" : "x",
      interaction: { mode: opts.stacked || opts.grouped ? "index" : "nearest", intersect: !(opts.stacked || opts.grouped || opts.line) , axis: horizontal ? "y" : "x" },
      layout: { padding: { top: 4, right: 8 } },
      plugins: {
        legend: { display: false },
        tooltip: { enabled: false, external: externalTooltip(opts.valueFormat) },
      },
      scales: horizontal ? { x: valueAxis, y: catAxis } : { x: catAxis, y: valueAxis },
    };
  }

  function barDataset(label, data, color, t, opts = {}) {
    const radius = opts.stacked ? 0 : 4;
    return {
      label, data, backgroundColor: color, keyColor: color,
      hoverBackgroundColor: color,
      maxBarThickness: 24, categoryPercentage: opts.grouped ? 0.7 : 0.72, barPercentage: opts.grouped ? 0.92 : 0.9,
      borderRadius: opts.horizontal ? { topRight: radius, bottomRight: radius } : { topLeft: radius, topRight: radius },
      borderSkipped: "start",
      borderWidth: opts.stacked ? (opts.horizontal ? { right: 2 } : { top: 2 }) : 0,
      borderColor: t.surface,
    };
  }

  function register(canvas, build) {
    const entry = { canvas, build, chart: null };
    entry.chart = build(theme());
    charts.push(entry);
    return entry.chart;
  }

  document.addEventListener("ss:theme", () => {
    setTimeout(() => {
      charts.forEach((c) => {
        c.chart.destroy();
        c.chart = c.build(theme());
      });
    }, 30);
  });

  function setupDefaults() {
    if (!window.Chart) return false;
    window.Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
    window.Chart.defaults.font.size = 12;
    return true;
  }

  /* ---- public helpers ---- */
  function bar(canvas, { labels, series, horizontal = false, stacked = false, valueFormat, precision, max, valueTick, valueTitle, catTitle }) {
    if (!setupDefaults()) return null;
    const grouped = series.length > 1 && !stacked;
    return register(canvas, (t) => new window.Chart(canvas, {
      type: "bar",
      data: {
        labels,
        datasets: series.map((s) => barDataset(s.label, s.data, typeof s.color === "function" ? s.color(t) : s.color, t, { horizontal, stacked, grouped })),
      },
      options: baseOptions(t, { horizontal, stacked, grouped, valueFormat, precision, max, valueTick, valueTitle, catTitle }),
    }));
  }

  function line(canvas, { labels, series, valueFormat, precision, max }) {
    if (!setupDefaults()) return null;
    return register(canvas, (t) => new window.Chart(canvas, {
      type: "line",
      data: {
        labels,
        datasets: series.map((s) => {
          const color = typeof s.color === "function" ? s.color(t) : s.color;
          return {
            label: s.label, data: s.data, borderColor: color, keyColor: color, backgroundColor: color,
            borderWidth: 2, pointRadius: 0, pointHoverRadius: 5, pointHoverBorderWidth: 2, pointHoverBorderColor: t.surface,
            tension: 0.25, fill: false,
          };
        }),
      },
      options: baseOptions(t, { line: true, grouped: true, valueFormat, precision, max }),
    }));
  }

  function scatter(canvas, { series, xTitle, yTitle }) {
    if (!setupDefaults()) return null;
    return register(canvas, (t) => {
      const opts = baseOptions(t, { valueFormat: (raw) => `${(raw.x * 100).toFixed(0)}% / ${(raw.y * 100).toFixed(0)}%` });
      opts.interaction = { mode: "nearest", intersect: false };
      opts.scales = {
        x: { min: 0, max: 1, grid: { color: t.grid }, border: { color: t.axis }, ticks: { color: t.ink3, callback: (v) => `${Math.round(v * 100)}%` },
             title: { display: true, text: xTitle, color: t.ink3, font: { size: 11 } } },
        y: { min: 0, max: 1, grid: { color: t.grid }, border: { display: false }, ticks: { color: t.ink3, callback: (v) => `${Math.round(v * 100)}%` },
             title: { display: true, text: yTitle, color: t.ink3, font: { size: 11 } } },
      };
      return new window.Chart(canvas, {
        type: "scatter",
        data: {
          datasets: series.map((s) => {
            const color = typeof s.color === "function" ? s.color(t) : s.color;
            return { label: s.label, data: s.data, backgroundColor: color, keyColor: color, pointRadius: 4,
                     pointBorderColor: t.surface, pointBorderWidth: 2, pointHoverRadius: 6, pointHitRadius: 12 };
          }),
        },
        options: opts,
      });
    });
  }

  /* Table view twin (built with textContent only). */
  function tableView(container, headers, rows) {
    container.textContent = "";
    const table = document.createElement("table");
    table.className = "table compact";
    const thead = table.createTHead().insertRow();
    headers.forEach((h, i) => { const th = document.createElement("th"); th.textContent = h; if (i) th.className = "num"; thead.appendChild(th); });
    const body = table.createTBody();
    rows.forEach((r) => {
      const tr = body.insertRow();
      r.forEach((v, i) => { const td = tr.insertCell(); td.textContent = v; if (i) td.className = "num"; });
    });
    container.appendChild(table);
  }

  function legend(container, items) {
    container.textContent = "";
    items.forEach(({ label, color, line: isLine }) => {
      const s = document.createElement("span");
      const i = document.createElement("i");
      if (isLine) i.className = "line";
      i.style.background = color;
      s.append(i, document.createTextNode(label));
      container.appendChild(s);
    });
  }

  window.SSCharts = { bar, line, scatter, tableView, legend, theme };
})();
