"use strict";

// memtune demo: reads precomputed data (site/data, made by scripts/export_demo_data.py) and shows the four stages.

const STEP_DELAY_MS = 750;
const state = { index: null, meta: null, byId: new Map(), cache: new Map(), current: null, model: "rf", running: false };
const $ = (id) => document.getElementById(id);
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const MODEL_NAME = { rf: "random forest", gb: "XGBoost" };
const ROLE_TEXT = {
  test: ["Test problem", "The models never saw this problem during training."],
  train: ["Training problem", "This problem was part of the training data, so the models have seen its runs."],
  incomplete: ["Not used for training", "Some of this problem's 100 runs are missing, so it was used neither for training nor for testing."],
};

// ---------- formatting ----------
const fmtInt = (x) => Math.round(x).toLocaleString("en-US");
function fmtNum(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "–";
  const a = Math.abs(x);
  if (a === 0) return "0";
  if (Number.isInteger(x) && a < 1e9) return fmtInt(x);
  if (a >= 1e5 || a < 1e-3) return x.toExponential(2);
  return x.toPrecision(3);
}
function fmtSeconds(s) {
  if (s === null || s === undefined) return "not measured";
  if (s >= 100) return `${fmtInt(s)} s`;
  if (s >= 1) return `${s.toPrecision(3)} s`;
  if (s >= 1e-3) return `${(s * 1e3).toPrecision(3)} ms`;
  return `${(s * 1e6).toPrecision(3)} µs`;
}
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const sleep = (ms) => new Promise((r) => setTimeout(r, reduceMotion ? 0 : ms));

// ---------- data ----------
async function loadIndex() {
  const res = await fetch("data/index.json");
  if (!res.ok) throw new Error(`data/index.json: HTTP ${res.status}. Run scripts/export_demo_data.py first.`);
  const data = await res.json();
  state.meta = data.meta;
  state.index = data.instances;
  for (const inst of state.index) state.byId.set(inst.id, inst);
}

async function loadInstance(id) {
  if (!state.cache.has(id)) {
    const res = await fetch(`data/instances/${encodeURIComponent(id)}.json`);
    if (!res.ok) throw new Error(`instance ${id}: HTTP ${res.status}`);
    state.cache.set(id, await res.json());
  }
  return state.cache.get(id);
}

// ---------- controls ----------
function fillProblems() {
  const names = [...new Set(state.index.map((i) => i.name))].sort((a, b) => a.localeCompare(b));
  $("problem").innerHTML = names.map((n) => `<option value="${n}">${n}</option>`).join("");
}

function fillSizes(name, preferId) {
  const insts = state.index.filter((i) => i.name === name).sort((a, b) => a.nvar - b.nvar);
  const roleShort = { test: "test", train: "training", incomplete: "not used" };
  $("size").innerHTML = insts
    .map((i) => `<option value="${i.id}">${fmtInt(i.nvar)} (${roleShort[i.role]})</option>`)
    .join("");
  const pick = insts.find((i) => i.id === preferId) || insts.find((i) => i.role === "test") || insts[0];
  $("size").value = pick.id;
}

function selectInstance(id) {
  const inst = state.byId.get(id);
  if (!inst) return false;
  $("problem").value = inst.name;
  fillSizes(inst.name, id);
  return true;
}

function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  return { id: p.get("id"), model: p.get("model") };
}
function writeHash() {
  const p = new URLSearchParams({ id: $("size").value, model: state.model });
  history.replaceState(null, "", `#${p}`);
}

function setRunning(on) {
  state.running = on;
  for (const id of ["run", "random", "problem", "size"]) $(id).disabled = on;
  for (const r of document.querySelectorAll('input[name="model"]')) r.disabled = on;
}

function markStep(n) {
  for (const li of $("steps").children) {
    const s = Number(li.dataset.step);
    li.classList.toggle("active", s === n);
    li.classList.toggle("done", s < n || n === 5);
  }
}
const panel = (n) => document.querySelector(`[data-panel="${n}"]`);

// ---------- rendering ----------
function plotLayout(title, ytitle, extra = {}) {
  return {
    title: { text: title, font: { size: 14, color: css("--text") }, x: 0.02, xanchor: "left" },
    margin: { l: 56, r: 12, t: 36, b: 40 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: css("--muted"), size: 12 },
    xaxis: { title: { text: "mem" }, range: [0, 101], gridcolor: css("--line"), zeroline: false },
    yaxis: { title: { text: ytitle }, type: "log", gridcolor: css("--line"), zeroline: false, exponentformat: "power" },
    showlegend: false,
    hovermode: "x unified",
    ...extra,
  };
}
const PLOT_CONFIG = { displayModeBar: false, responsive: true };

function lineTrace(x, y, name) {
  return { x, y, name, type: "scatter", mode: "lines", line: { color: css("--accent"), width: 2.5 },
           hovertemplate: "%{y:.3s}" };
}
function dotTrace(x, y, name, status) {
  const failed = (status || []).map((s) => s && s !== "first_order");
  return {
    x, y, name, type: "scatter", mode: "markers",
    marker: { color: css("--observed"), size: 5, opacity: 0.8, symbol: failed.map((f) => (f ? "x" : "circle")) },
    hovertemplate: "%{y:.3s}",
  };
}

function renderFeatures(d) {
  const [label, text] = ROLE_TEXT[d.role];
  const badge = $("role");
  badge.textContent = label;
  badge.className = `badge ${d.role}`;
  badge.title = text;
  const rows = Object.entries(d.features).map(([k, v]) =>
    `<tr><td><div class="fname">${k}</div><div class="help">${state.meta.features[k] || ""}</div></td><td class="num-cell">${fmtNum(v)}</td></tr>`);
  $("features").innerHTML = `<thead><tr><th>Feature</th><th class="num-cell">Value</th></tr></thead><tbody>${rows.join("")}</tbody>`;
}

function renderStage1(d) {
  const mb = (arr) => arr.map((v) => (v === null ? null : v / 1e6));
  const specs = [
    ["chart-obj", "Objective evaluations", "count", d.stage1.neval_obj, d.observed.neval_obj],
    ["chart-grad", "Gradient evaluations", "count", d.stage1.neval_grad, d.observed.neval_grad],
    ["chart-bytes", "Memory allocated", "MB", mb(d.stage1.bytes), mb(d.observed.bytes)],
  ];
  for (const [id, title, unit, pred, obs] of specs) {
    Plotly.react(id, [lineTrace(d.mem, pred, "predicted"), dotTrace(d.mem, obs, "measured", d.observed.status)],
                 plotLayout(title, unit), PLOT_CONFIG);
  }
}

function renderStage2(d, showRecommendation) {
  const rec = d.rec[state.model];
  const marks = [[state.meta.default_mem, "default 5", css("--muted"), "dash"],
                 [state.meta.fixed_mem, "fixed 87", css("--muted"), "dot"]];
  if (showRecommendation) marks.push([rec, `recommended ${rec}`, css("--accent"), "solid"]);
  const shapes = marks.map(([x, , color, dash]) => ({
    type: "line", xref: "x", yref: "paper", x0: x, x1: x, y0: 0, y1: 1, line: { color, width: dash === "solid" ? 2.5 : 1.5, dash },
  }));
  const annotations = marks.map(([x, text, color], i) => ({
    x, y: 1, xref: "x", yref: "paper", yanchor: "bottom", text, showarrow: false,
    font: { size: 11, color }, xshift: i === 2 ? 0 : 0,
  }));
  const layout = plotLayout(`Run time (${MODEL_NAME[state.model]} Stage 2)`, "seconds",
                            { shapes, annotations, margin: { l: 60, r: 14, t: 52, b: 40 }, showlegend: true,
                              legend: { orientation: "h", x: 1, xanchor: "right", y: 1.18, font: { size: 11 } } });
  Plotly.react("chart-time", [lineTrace(d.mem, d.stage2[state.model], "predicted"),
                              dotTrace(d.mem, d.observed.time, "measured", d.observed.status)], layout, PLOT_CONFIG);
}

function renderRecommendation(d) {
  const rec = d.rec[state.model];
  $("rec-mem").textContent = rec;
  const t = d.observed_time_at;
  const best = t.best;
  const ratio = (x) => (x === null || best === null || best === 0 ? "–" : `${(x / best).toFixed(2)}×`);
  const rows = [
    ["Default of JSOSolvers.jl", state.meta.default_mem, t.default, ""],
    ["Fixed value, best on the whole benchmark", state.meta.fixed_mem, t.fixed, ""],
    [`Framework (${MODEL_NAME[state.model]})`, rec, t[state.model], "framework"],
    ["Best measured for this problem (hindsight)", d.best_observed_mem, best, "best"],
  ];
  $("compare").innerHTML =
    `<thead><tr><th>Choice</th><th class="num-cell">mem</th><th class="num-cell">Measured time</th><th class="num-cell">Ratio to best</th></tr></thead><tbody>` +
    rows.map(([label, mem, time, cls]) => /* one row per policy */
      `<tr class="${cls}"><td>${label}</td><td class="num-cell">${mem ?? "–"}</td><td class="num-cell">${fmtSeconds(time)}</td><td class="num-cell">${ratio(time)}</td></tr>`).join("") +
    "</tbody>";
  const notes = [ROLE_TEXT[d.role][1]];
  if (d.n_observed < 100) notes.push(`Only ${d.n_observed} of the 100 runs were measured; "not measured" marks the missing ones.`);
  if (d.observed.status.some((s) => s && s !== "first_order")) notes.push("Crosses in the charts are runs that hit the time limit; their time is a penalty value, not a real run time.");
  notes.push("The paper counts differences below 0.1 s as ties.");
  $("compare-note").textContent = notes.join(" ");
}

function renderAll(d) {
  renderFeatures(d);
  for (const n of [1, 2, 3, 4]) panel(n).hidden = false;
  renderStage1(d);
  renderStage2(d, true);
  renderRecommendation(d);
  markStep(5);
}

async function run(animate) {
  if (state.running) return;
  const id = $("size").value;
  writeHash();
  setRunning(true);
  $("status").textContent = "";
  $("status").classList.remove("error");
  try {
    const d = await loadInstance(id);
    state.current = d;
    if (!animate) { renderAll(d); return; }
    for (const n of [1, 2, 3, 4]) panel(n).hidden = true;
    markStep(1);
    renderFeatures(d); panel(1).hidden = false;
    await sleep(STEP_DELAY_MS);
    markStep(2); panel(2).hidden = false; renderStage1(d);
    await sleep(STEP_DELAY_MS);
    markStep(3); panel(3).hidden = false; renderStage2(d, false);
    await sleep(STEP_DELAY_MS);
    markStep(4); renderStage2(d, true); renderRecommendation(d); panel(4).hidden = false;
    await sleep(STEP_DELAY_MS / 2);
    markStep(5);
  } catch (err) {
    $("status").textContent = `Could not load this problem: ${err.message}`;
    $("status").classList.add("error");
  } finally {
    setRunning(false);
  }
}

function rerenderCurrent() {
  if (!state.current || state.running) return;
  renderAll(state.current);
}

// ---------- start ----------
async function main() {
  try {
    await loadIndex();
  } catch (err) {
    $("status").textContent = `Could not load the data: ${err.message}`;
    $("status").classList.add("error");
    return;
  }
  const m = state.meta;
  $("range").textContent = `${fmtInt(m.training_nvar_range[0])} – ${fmtInt(m.training_nvar_range[1])}`;
  $("meta").textContent = `${m.n_instances} instances of ${m.n_problems} problems · data generated ${m.generated}` +
    ` · memtune ${m.memtune_version}${m.git ? ` (${m.git})` : ""} · scikit-learn ${m.libraries["scikit-learn"]}, XGBoost ${m.libraries.xgboost}`;
  fillProblems();

  const h = readHash();
  if (h.model === "gb" || h.model === "rf") state.model = h.model;
  document.querySelector(`input[name="model"][value="${state.model}"]`).checked = true;
  if (!h.id || !selectInstance(h.id)) {
    const first = state.index.find((i) => i.role === "test") || state.index[0];
    selectInstance(first.id);
  }
  $("status").textContent = "";

  $("problem").addEventListener("change", () => { fillSizes($("problem").value); run(false); });
  $("size").addEventListener("change", () => run(false));
  for (const r of document.querySelectorAll('input[name="model"]')) {
    r.addEventListener("change", () => { state.model = r.value; writeHash(); rerenderCurrent(); });
  }
  $("run").addEventListener("click", () => run(true));
  $("random").addEventListener("click", () => {
    const tests = state.index.filter((i) => i.role === "test");
    selectInstance(tests[Math.floor(Math.random() * tests.length)].id);
    run(true);
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", rerenderCurrent);
  window.addEventListener("hashchange", () => {
    const { id, model } = readHash();
    if (model && model !== state.model && MODEL_NAME[model]) {
      state.model = model;
      document.querySelector(`input[name="model"][value="${model}"]`).checked = true;
    }
    if (id && id !== $("size").value && selectInstance(id)) run(false); else rerenderCurrent();
  });
  await run(false);
}

main();
