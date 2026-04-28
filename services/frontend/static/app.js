const $ = (id) => document.getElementById(id);

const modeRadios = () => document.querySelectorAll('input[name="mode"]');
const mode = () => [...modeRadios()].find((r) => r.checked)?.value || "text";

/** @type {Record<string, string>} */
let configIdToLabel = {};
let backendStatusTimer = null;

const LOG_MAX = 500;

function ts() {
  const t = new Date();
  return t.toLocaleTimeString() + "." + String(t.getMilliseconds()).padStart(3, "0");
}

function appendLog(pre, line) {
  if (!pre) return;
  pre.textContent += `${ts()} ${line}\n`;
  const lines = pre.textContent.split("\n");
  if (lines.length > LOG_MAX) {
    pre.textContent = lines.slice(-LOG_MAX).join("\n");
  }
  pre.scrollTop = pre.scrollHeight;
}

function fmtF(n, d) {
  if (n == null || Number.isNaN(Number(n))) return "-";
  return Number(n).toFixed(d);
}

function backendLabel(id) {
  return configIdToLabel[id] || id;
}

async function loadConfigs() {
  const r = await fetch("/api/configs");
  if (!r.ok) throw new Error("configs");
  const c = await r.json();
  const el = $("checkboxes");
  if (!el) return;
  el.replaceChildren();
  for (const [k, v] of Object.entries(c)) {
    configIdToLabel[k] = v.label || k;
    const d = document.createElement("div");
    const isDis = v.available === false;
    d.className = isDis ? "disabled" : "";
    const ch = document.createElement("input");
    ch.type = "checkbox";
    ch.value = k;
    ch.checked = !isDis;
    ch.disabled = isDis;
    const sp = document.createElement("span");
    sp.appendChild(ch);
    sp.append(` ${v.label} (${k}) `);
    if (v.has_prometheus) sp.append(" [Prom] ");
    if (isDis && v.unavailable_reason) sp.append(`- ${v.unavailable_reason} `);
    d.appendChild(sp);
    el.appendChild(d);
  }
}

function selectedConfigIds() {
  const cbs = document.querySelectorAll("#checkboxes input[type=checkbox]:checked");
  return Array.from(cbs, (c) => c.value);
}

function onMode() {
  const m = mode();
  const article = $("article");
  const limitRow = $("jsonlLimitRow");
  const file = $("file");
  const fileHint = $("fileHint");
  if (m === "jsonl") {
    if (article) article.placeholder = "Optional note box. Upload one JSONL file below.";
    if (limitRow) limitRow.hidden = false;
    if (file) {
      file.multiple = false;
      file.accept = ".jsonl,application/jsonl,text/plain";
    }
    if (fileHint) fileHint.textContent = "JSONL mode accepts exactly one .jsonl file. Each line should be a JSON object with a text field.";
  } else {
    if (article) article.placeholder = "Paste one article here, or upload one or more .txt files below.";
    if (limitRow) limitRow.hidden = true;
    if (file) {
      file.multiple = true;
      file.accept = ".txt,text/plain";
    }
    if (fileHint) fileHint.textContent = "Text mode accepts one or more .txt files. Each file is treated as one full input.";
  }
}

function fmtSummaryLine(s) {
  if (!s || s.error) return s?.message || s?.error || "-";
  if (Array.isArray(s.items)) {
    return `${s.n_inputs || s.items.length} inputs | tput ${fmtF(s.throughput_req_per_s, 2)} r/s | p50 ${fmtF(s.latency_p50_s, 3)}s | TTFT avg ${fmtF(s.ttft_avg_s, 3)}s`;
  }
  if (s.throughput_req_per_s != null) {
    const acc = s.accuracy_valid_only == null ? "-" : `${((s.accuracy_valid_only || 0) * 100).toFixed(1)}%`;
    return `tput ${fmtF(s.throughput_req_per_s, 2)} r/s | p50 ${fmtF(s.latency_p50_s, 3)}s | acc ${acc}`;
  }
  return "-";
}

function renderBackendStatus(status) {
  const badge = $("backendStatus");
  const active = $("backendActive");
  const details = $("backendDetails");
  if (!badge || !active || !details) return;
  const state = status?.status || "idle";
  badge.textContent = state;
  badge.dataset.state = state;
  const activeLabel = status?.active_label || backendLabel(status?.active_config_name) || "None";
  active.textContent = activeLabel;
  const parts = [];
  if (status?.message) parts.push(status.message);
  if (status?.pid) parts.push(`pid ${status.pid}`);
  if (status?.desired_label && status?.desired_label !== activeLabel) parts.push(`target ${status.desired_label}`);
  details.textContent = parts.join(" | ") || "No backend service is running.";
}

async function refreshBackendStatus() {
  try {
    const r = await fetch("/api/backend-status");
    if (!r.ok) throw new Error(String(r.status));
    renderBackendStatus(await r.json());
  } catch (err) {
    renderBackendStatus({ status: "unknown", message: err?.message || String(err), active_label: "Unavailable" });
  }
}

function renderResultTable(results, errors, order) {
  const host = $("resultTableHost");
  const panel = $("resultsPanel");
  if (!host) return;
  host.replaceChildren();

  const r = results && typeof results === "object" ? results : {};
  const e = errors && typeof errors === "object" ? errors : {};
  const all = new Set([...Object.keys(r), ...Object.keys(e)]);
  const ids = order && order.length
    ? [...order.filter((k) => all.has(k)), ...[...all].filter((k) => !order.includes(k)).sort()]
    : [...all].sort();

  if (!ids.length) {
    if (panel) panel.hidden = true;
    return;
  }
  if (panel) panel.hidden = false;

  const isJsonl = mode() === "jsonl";
  const table = document.createElement("table");
  table.className = "data-table";
  const thead = document.createElement("thead");
  if (isJsonl) {
    thead.innerHTML = `<tr>
      <th>Backend</th>
      <th>Throughput (req/s)</th>
      <th>Latency p50 (s)</th>
      <th>Latency p95 (s)</th>
      <th>TTFT avg (s)</th>
      <th>Accuracy (valid) %</th>
      <th>Measured n</th>
      <th>Error / note</th>
    </tr>`;
  } else {
    thead.innerHTML = `<tr>
      <th>Backend</th>
      <th>Input</th>
      <th>Predicted label</th>
      <th>Total latency (s)</th>
      <th>Queue wait (s)</th>
      <th>TTFT (s)</th>
      <th>TPS</th>
      <th>Error / note</th>
    </tr>`;
  }
  table.appendChild(thead);
  const tbody = document.createElement("tbody");

  for (const id of ids) {
    const v = r[id];
    const note = e[id] || (v && typeof v === "object" && v.error != null ? String(v.error) : "");
    if (isJsonl) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="backend-name">${backendLabel(id)}</td>
        <td class="num">${note || !v ? "-" : fmtF(v.throughput_req_per_s, 2)}</td>
        <td class="num">${note || !v ? "-" : fmtF(v.latency_p50_s, 3)}</td>
        <td class="num">${note || !v ? "-" : fmtF(v.latency_p95_s, 3)}</td>
        <td class="num">${note || !v ? "-" : fmtF(v.ttft_avg_s, 3)}</td>
        <td class="num">${note || !v || v.accuracy_valid_only == null ? "-" : `${(v.accuracy_valid_only * 100).toFixed(1)}%`}</td>
        <td class="num">${note || !v || v.n_requests_measured == null ? "-" : String(v.n_requests_measured)}</td>
        <td class="err-cell">${note || v?.label_mode || "-"}</td>
      `;
      tbody.appendChild(tr);
      continue;
    }

    const items = Array.isArray(v?.items) ? v.items : v ? [v] : [];
    if (!items.length) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="backend-name">${backendLabel(id)}</td>
        <td class="num">-</td>
        <td class="num">-</td>
        <td class="num">-</td>
        <td class="num">-</td>
        <td class="num">-</td>
        <td class="num">-</td>
        <td class="err-cell">${note || "-"}</td>
      `;
      tbody.appendChild(tr);
      continue;
    }
    for (const item of items) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="backend-name">${backendLabel(id)}</td>
        <td>${item.input_name || "-"}</td>
        <td class="num">${item.pred_label != null ? String(item.pred_label) : "-"}</td>
        <td class="num">${fmtF(item.latency_s, 3)}</td>
        <td class="num">${fmtF(item.queue_wait_s, 3)}</td>
        <td class="num">${fmtF(item.ttft_s, 3)}</td>
        <td class="num">${item.tps == null ? "-" : fmtF(item.tps, 1)}</td>
        <td class="err-cell">${note || "-"}</td>
      `;
      tbody.appendChild(tr);
    }
  }
  table.appendChild(tbody);
  host.appendChild(table);
}

function setCard(cards, id, update) {
  let c = cards.querySelector(`[data-id="${id}"]`);
  if (!c) {
    c = document.createElement("div");
    c.className = "mcard";
    c.dataset.id = id;
    c.innerHTML = `<h3 class="h"></h3><div class="phase">-</div><div class="metrics">-</div><pre>-</pre>`;
    cards.appendChild(c);
  }
  const h3 = c.querySelector(".h");
  const ph = c.querySelector(".phase");
  const metrics = c.querySelector(".metrics");
  const pr = c.querySelector("pre");
  if (update?.label) h3.textContent = update.label;
  if (update?.phase) {
    ph.textContent = update.phase;
    if (update.phase === "error") c.style.outline = "1px solid var(--err)";
    if (update.phase === "done") c.style.outline = "1px solid var(--ok)";
  }
  if (update?.summary) metrics.textContent = fmtSummaryLine(update.summary);
  if (update?.log_tail) pr.textContent = String(update.log_tail || "").split("\n").slice(-12).join("\n");
  return c;
}

function handleSseData(log, data, cards) {
  if (data.type === "log" && data.payload) {
    if (data.payload.message) appendLog(log, data.payload.message);
    if (data.payload.log_tail) appendLog(log, "-- server log tail --\n" + data.payload.log_tail);
  }
  if (data.type === "config" && data.payload?.id) {
    setCard(cards, data.payload.id, data.payload);
    if (data.payload.phase) appendLog(log, `config[${data.payload.id}] phase=${data.payload.phase}`);
  }
  if (data.type === "error" && data.payload) {
    appendLog(log, "ERR: " + String(data.payload.message != null ? data.payload.message : data.payload.error));
  }
  if (data.type === "job" && data.payload) {
    if (data.payload.id && $("jobid")) $("jobid").textContent = data.payload.id;
    if (data.payload.state === "finished") {
      appendLog(log, "==== run finished");
      renderResultTable(data.payload.results, data.payload.errors, data.payload.config_ids);
    } else {
      appendLog(log, "==== " + data.payload.state);
    }
  }
}

async function go() {
  const btn = $("go");
  if (btn) btn.disabled = true;
  const log = $("log");
  const cards = $("cards");
  if (log) {
    log.textContent = "";
    appendLog(log, "Submitting...");
  }
  if (cards) cards.replaceChildren();
  const panel = $("resultsPanel");
  const host = $("resultTableHost");
  if (host) host.replaceChildren();
  if (panel) panel.hidden = true;

  const m = mode();
  const form = new FormData();
  form.set("mode", m);
  form.set("concurrency", String(+$("concurrency")?.value || 4));
  const lim = +$("limit")?.value;
  if (m === "text") {
    form.set("text", $("article")?.value || "");
  }
  if (lim) form.set("limit", String(lim));
  const ids = selectedConfigIds();
  if (ids.length) form.set("configs", JSON.stringify(ids));

  const file = $("file");
  const selectedFiles = file?.files ? Array.from(file.files) : [];
  if (m === "jsonl") {
    if (!selectedFiles.length) throw new Error("jsonl mode requires one .jsonl file");
    form.append("files", selectedFiles[0], selectedFiles[0].name);
  } else {
    for (const f of selectedFiles) {
      form.append("files", f, f.name);
    }
  }

  try {
    const st = await fetch("/api/jobs", { method: "POST", body: form });
    if (!st.ok) {
      const t = await st.text();
      throw new Error(t || String(st.status));
    }
    const job = await st.json();
    if ($("jobid")) $("jobid").textContent = job.id;
    if (log) appendLog(log, "job id=" + job.id + " - listening on SSE...");

    await new Promise((resolve) => {
      const es = new EventSource("/api/jobs/" + job.id + "/events");
      const done = () => {
        es.close();
        if (btn) btn.disabled = false;
        refreshBackendStatus().catch(() => undefined);
        resolve();
      };
      es.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          handleSseData(log, data, cards);
          if (data.type === "job" && data.payload?.state === "finished") done();
        } catch (err) {
          appendLog(log, (err && err.message) || String(err));
        }
      };
      es.onerror = () => {
        appendLog(log, "EventSource closed or errored.");
        done();
      };
    });
  } catch (err) {
    appendLog(log, (err && err.message) || String(err));
    if (btn) btn.disabled = false;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  $("go")?.addEventListener("click", () => go().catch((err) => console.error(err)));
  modeRadios().forEach((r) => r.addEventListener("change", onMode));
  onMode();
  loadConfigs().catch((err) => console.error(err));
  refreshBackendStatus().catch((err) => console.error(err));
  backendStatusTimer = window.setInterval(() => {
    refreshBackendStatus().catch(() => undefined);
  }, 3000);
});
