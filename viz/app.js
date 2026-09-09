/* Interactive viewer for a simulation run trace.
   Zero dependencies: open index.html directly and load out/run.json. */

let trace = null;
let traceB = null;
let idx = 0;
let playing = false;
let timer = null;

const $ = (id) => document.getElementById(id);

function fmtClock(startISO, minutes) {
  const d = new Date(new Date(startISO).getTime() + minutes * 60000);
  return d.toLocaleString(undefined, {
    day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", hour12: false,
  });
}

function loadTrace(file, slot) {
  const reader = new FileReader();
  reader.onload = () => {
    const data = JSON.parse(reader.result);
    if (slot === "A") {
      trace = data;
      idx = 0;
      $("scrub").max = Math.max(0, data.snapshots.length - 1);
      $("runmeta").textContent =
        `${data.meta.domain} — scenario "${data.meta.scenario}", policy "${data.meta.policy}", seed ${data.meta.seed}`;
      render();
    } else {
      traceB = data;
      renderCompare();
    }
  };
  reader.readAsText(file);
}

$("file").addEventListener("change", (e) => e.target.files[0] && loadTrace(e.target.files[0], "A"));
$("fileB").addEventListener("change", (e) => e.target.files[0] && loadTrace(e.target.files[0], "B"));

$("scrub").addEventListener("input", (e) => { idx = +e.target.value; render(); });

$("play").addEventListener("click", () => {
  if (!trace) return;
  playing = !playing;
  $("play").textContent = playing ? "❚❚ Pause" : "▶ Play";
  if (playing) tick(); else clearTimeout(timer);
});

function tick() {
  if (!playing || !trace) return;
  idx = Math.min(idx + 1, trace.snapshots.length - 1);
  $("scrub").value = idx;
  render();
  if (idx >= trace.snapshots.length - 1) {
    playing = false;
    $("play").textContent = "▶ Play";
    return;
  }
  timer = setTimeout(tick, 400 / +$("speed").value);
}

function render() {
  if (!trace || !trace.snapshots.length) return;
  const s = trace.snapshots[idx];
  const meta = trace.meta;

  $("clock").textContent = fmtClock(meta.start, s.t);
  $("regime").textContent = s.regime && s.regime !== "normal" ? `regime: ${s.regime}` : "";

  const kpis = [
    ["queue", s.queue], ["overdue", s.overdue], ["completed", s.completed],
    ["breached", s.breached], ["utilisation", (s.utilisation * 100).toFixed(0) + "%"],
    ["labour cost", Math.round(s.labour_cost)], ["overtime min", Math.round(s.overtime_minutes)],
  ];
  $("kpigrid").innerHTML = kpis
    .map(([k, v]) => `<div class="kpi"><div class="v">${v}</div><div class="k">${k}</div></div>`)
    .join("");

  // Departments
  $("depts").innerHTML = Object.entries(s.departments).map(([id, d]) => {
    const total = d.busy + d.available;
    const load = total ? d.busy / total : 0;
    const cls = load > 0.85 ? "crit" : load > 0.6 ? "hot" : "";
    return `<tr><td><strong>${id}</strong></td><td>${d.queue}</td><td>${d.overdue}</td>
      <td>${d.busy}</td><td>${d.available}</td><td>${d.absent}</td>
      <td><div class="bar"><i class="${cls}" style="width:${(load * 100).toFixed(0)}%"></i></div></td></tr>`;
  }).join("");

  // Stations
  $("stations").innerHTML = Object.entries(s.locations).map(([id, l]) => {
    const name = meta.locations[id] ? meta.locations[id].name : id;
    const cls = l.queue > 12 ? "hot" : l.queue > 5 ? "busy" : "";
    return `<div class="station ${cls}"><div class="n">${name}</div>
      <div class="d">queue ${l.queue} · busy ${l.busy}/${l.staff}</div></div>`;
  }).join("");

  const fb = [
    ["CSAT", s.csat == null ? "—" : s.csat.toFixed(2)],
    ["NPS", s.nps == null ? "—" : s.nps.toFixed(0)],
    ["responses", s.surveys == null ? 0 : s.surveys],
    ["perf index", s.performance == null ? "—" : s.performance.toFixed(2)],
    ["op cost", s.cost_total == null ? "—" : Math.round(s.cost_total)],
  ];
  $("fbgrid").innerHTML = fb
    .map(([k, v]) => `<div class="kpi"><div class="v">${v}</div><div class="k">${k}</div></div>`)
    .join("");

  renderEvents(s.t);
  renderDecisions(s.t);
  drawChart();
}

function renderEvents(now) {
  const evs = trace.events.filter((e) => e.time <= now).slice(-70).reverse();
  if (!evs.length) { $("events").innerHTML = '<div class="empty">no events yet</div>'; return; }
  $("events").innerHTML = evs.map((e) => {
    const cascade = e.caused_by ? " cascade" : "";
    const badge = e.caused_by ? `<span class="badge">cascade of ${e.caused_by}</span>` : "";
    return `<div class="ev ${e.kind}${cascade}">
      <span class="t">${fmtClock(trace.meta.start, e.time).slice(7)}</span>${e.message}${badge}</div>`;
  }).join("");
}

function renderDecisions(now) {
  const ds = (trace.decisions || []).filter((d) => d.t <= now).slice(-60).reverse();
  if (!ds.length) { $("decisions").innerHTML = '<div class="empty">no decisions yet</div>'; return; }
  $("decisions").innerHTML = ds.map((d) => {
    if (d.kind === "dispatch") {
      const cls = d.chosen === "defer" ? "defer" : "";
      return `<div class="dec"><span>${fmtClock(trace.meta.start, d.t).slice(7)}
        · ${d.activity} @ ${d.location} (${d.n_candidates} eligible)</span>
        <span class="who ${cls}">${d.chosen}</span></div>`;
    }
    return `<div class="dec"><span>${fmtClock(trace.meta.start, d.t).slice(7)}
      · staffing ${d.department}</span><span class="who">${d.chosen} (${d.changed})</span></div>`;
  }).join("");
}

function drawChart() {
  const snaps = trace.snapshots;
  const W = 720, H = 190, pad = 8;
  const n = snaps.length;
  const maxQ = Math.max(1, ...snaps.map((s) => s.queue));
  const maxB = Math.max(1, ...snaps.map((s) => s.breached));
  const x = (i) => pad + (i / Math.max(1, n - 1)) * (W - 2 * pad);
  const y = (v, max) => H - pad - (v / max) * (H - 2 * pad);

  const line = (fn, max, color) => {
    const pts = snaps.map((s, i) => `${x(i).toFixed(1)},${y(fn(s), max).toFixed(1)}`).join(" ");
    return `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1.6"/>`;
  };

  const cursor = `<line x1="${x(idx)}" y1="0" x2="${x(idx)}" y2="${H}"
    stroke="#e6ebf0" stroke-width="1" opacity="0.45"/>`;

  $("chart").innerHTML =
    line((s) => s.queue, maxQ, "#4da3ff") +
    line((s) => s.utilisation, 1, "#3ecf8e") +
    line((s) => s.overdue, maxQ, "#ffb020") +
    line((s) => s.breached, maxB, "#ff5f56") +
    cursor;
}

function renderCompare() {
  if (!trace || !traceB) return;
  $("comparepanel").hidden = false;
  const keys = ["completion_rate", "sla_breach_rate", "mean_queue_time_min",
                "mean_utilisation", "labour_cost", "overtime_minutes", "total_reward"];
  const a = trace.kpis, b = traceB.kpis;
  const rows = keys.map((k) => {
    const av = a[k], bv = b[k];
    const better = typeof av === "number" && typeof bv === "number"
      ? (k === "sla_breach_rate" || k === "mean_queue_time_min" || k === "labour_cost"
          ? (av < bv ? "A" : av > bv ? "B" : "=")
          : (av > bv ? "A" : av < bv ? "B" : "="))
      : "";
    return `<tr><td>${k}</td><td>${av}</td><td>${bv}</td><td>${better}</td></tr>`;
  }).join("");
  $("compare").innerHTML = `<table>
    <thead><tr><th>metric</th><th>A: ${a.policy}</th><th>B: ${b.policy}</th><th>better</th></tr></thead>
    <tbody>${rows}</tbody></table>
    <p class="empty">Same scenario and seed means the exogenous world is identical,
    so differences are attributable to the policy.</p>`;
}
