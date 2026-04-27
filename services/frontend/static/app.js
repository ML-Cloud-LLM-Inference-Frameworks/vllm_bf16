const $ = (id) => document.getElementById(id);

const modeRadios = () => document.querySelectorAll('input[name="mode"]');
const mode = () => [...modeRadios()].find((r) => r.checked)?.value || "text";

/** @type {Record<string, string>} */
let configIdToLabel = {};

const LOG_MAX = 500;

function ts() {
  const t = new Date();
  return t.toLocaleTimeString() + "." + String(t.getMilliseconds()).padStart(3, "0");
}

function appendLog(pre, line) {
  if (!pre) return;
  pre.textContent += `${ts()} ${line}\n`;
  const L = pre.textContent.split("\n");
  if (L.length > LOG_MAX) {
    pre.textContent = L.slice(-LOG_MAX).join("\n");
  }
  pre.scrollTop = pre.scrollHeight;
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
    if (v.has_prometheus) sp.append(" [Prom]", " ");
    if (isDis && v.unavailable_reason) sp.append(`— ${v.unavailable_reason} `);
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
  const ja = $("article");
  const jo = $("jsonlOpts");
  if (m === "jsonl") {
    if (ja) ja.placeholder = "Optional if you upload a JSONL file";
    if (jo) jo.hidden = false;
  } else {
    if (ja) ja.placeholder = "Paste a news article here…";
    if (jo) jo.hidden = true;
  }
}

function fmtF(n, d) {
  if (n == null || Number.isNaN(Number(n))) return "—";
  return Number(n).toFixed(d);
}

function fmtSummaryLine(s) {
  if (!s || s.error) return s?.message || s?.error || "—";
  const b = s.throughput_req_per_s;
  if (b != null) {
    return `tput ${b.toFixed(2)} r/s p50 ${(s.latency_p50_s ?? 0).toFixed(3)}s acc ${((s.accuracy_valid_only || 0) * 100).toFixed(1)}% (valid only)`;
  }
  // Single-article run (pred_label is often null if the model did not return a clean label)
  if (s.mode === "single_text" || s.latency_s != null || s.ttft_s != null || s.tps != null) {
    const lab = s.pred_label != null ? String(s.pred_label) : "—";
    return `label ${lab} | latency ${fmtF(s.latency_s, 3)}s | TTFT ${fmtF(s.ttft_s, 3)}s | TPS ${fmtF(s.tps, 1)}`;
  }
  return "—";
}

function backendLabel(id) {
  return configIdToLabel[id] || id;
}

/**
 * @param {Record<string, any>} results
 * @param {Record<string, string>} errors
 * @param {string[]|undefined} order
 */
function renderResultTable(results, errors, order) {
  const host = $("resultTableHost");
  const panel = $("resultsPanel");
  if (!host) return;
  host.replaceChildren();

  const r = results && typeof results === "object" ? results : {};
  const e = errors && typeof errors === "object" ? errors : {};
  const all = new Set([...Object.keys(r), ...Object.keys(e)]);
  let ids;
  if (order && order.length) {
    const rest = [...all].filter((k) => !order.includes(k)).sort();
    ids = [...order.filter((k) => all.has(k)), ...rest];
  } else {
    ids = [...all].sort();
  }
  if (!ids.length) {
    if (panel) panel.hidden = true;
    return;
  }
  if (panel) panel.hidden = false;

  const jsonl = mode() === "jsonl";
  const table = document.createElement("table");
  table.className = "data-table";
  const thead = document.createElement("thead");
  if (jsonl) {
    thead.innerHTML = `<tr>
      <th>Backend</th>
      <th>Throughput (req/s)</th>
      <th>Latency p50 (s)</th>
      <th>Latency p95 (s)</th>
      <th>Accuracy (valid) %</th>
      <th>Measured n</th>
      <th>Error / note</th>
    </tr>`;
  } else {
    thead.innerHTML = `<tr>
      <th>Backend</th>
      <th>Predicted label</th>
      <th>Total latency (s)</th>
      <th>TTFT (s)</th>
      <th>TPS</th>
      <th>Error / note</th>
    </tr>`;
  }
  table.appendChild(thead);
  const tbody = document.createElement("tbody");

  for (const id of ids) {
    const tr = document.createElement("tr");
    const nameCell = document.createElement("td");
    nameCell.className = "backend-name";
    nameCell.textContent = backendLabel(id);
    tr.appendChild(nameCell);

    const errFromJob = e[id];
    const v = r[id];
    const resErr = v && typeof v === "object" && v.error != null ? String(v.error) : "";
    const note = errFromJob || resErr;

    if (jsonl) {
      if (note) {
        for (let i = 0; i < 5; i += 1) {
          const td = document.createElement("td");
          td.className = "num";
          td.textContent = "—";
          tr.appendChild(td);
        }
        const ncell = document.createElement("td");
        ncell.className = "err-cell";
        ncell.textContent = note;
        tr.appendChild(ncell);
      } else if (!v) {
        for (let i = 0; i < 5; i += 1) {
          const td = document.createElement("td");
          td.className = "num";
          td.textContent = "—";
          tr.appendChild(td);
        }
        const ncell = document.createElement("td");
        ncell.className = "err-cell";
        ncell.textContent = "—";
        tr.appendChild(ncell);
      } else {
        const t1 = document.createElement("td");
        t1.className = "num";
        t1.textContent = fmtF(v.throughput_req_per_s, 2);
        const t2 = document.createElement("td");
        t2.className = "num";
        t2.textContent = fmtF(v.latency_p50_s, 3);
        const t3 = document.createElement("td");
        t3.className = "num";
        t3.textContent = fmtF(v.latency_p95_s, 3);
        const t4 = document.createElement("td");
        t4.className = "num";
        t4.textContent = v.accuracy_valid_only == null ? "—" : `${(v.accuracy_valid_only * 100).toFixed(1)}%`;
        const t5 = document.createElement("td");
        t5.className = "num";
        t5.textContent = v.n_requests_measured == null ? "—" : String(v.n_requests_measured);
        const t6 = document.createElement("td");
        t6.className = "err-cell";
        t6.textContent = "—";
        tr.append(t1, t2, t3, t4, t5, t6);
      }
    } else {
      if (note) {
        for (let i = 0; i < 4; i += 1) {
          const td = document.createElement("td");
          td.className = "num";
          td.textContent = "—";
          tr.appendChild(td);
        }
        const ncell = document.createElement("td");
        ncell.className = "err-cell";
        ncell.textContent = note;
        tr.appendChild(ncell);
      } else if (!v) {
        for (let i = 0; i < 4; i += 1) {
          const td = document.createElement("td");
          td.className = "num";
          td.textContent = "—";
          tr.appendChild(td);
        }
        const ncell = document.createElement("td");
        ncell.className = "err-cell";
        ncell.textContent = "—";
        tr.appendChild(ncell);
      } else {
        const t1 = document.createElement("td");
        t1.className = "num";
        t1.textContent = v.pred_label != null ? String(v.pred_label) : "—";
        const t2 = document.createElement("td");
        t2.className = "num";
        t2.textContent = fmtF(v.latency_s, 3);
        const t3 = document.createElement("td");
        t3.className = "num";
        t3.textContent = fmtF(v.ttft_s, 3);
        const t4 = document.createElement("td");
        t4.className = "num";
        t4.textContent = v.tps == null ? "—" : fmtF(v.tps, 1);
        const t5 = document.createElement("td");
        t5.className = "err-cell";
        t5.textContent = "—";
        tr.append(t1, t2, t3, t4, t5);
      }
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  host.appendChild(table);
}

function setCard(cards, id, u) {
  let c = cards.querySelector(`[data-id="${id}"]`);
  if (!c) {
    c = document.createElement("div");
    c.className = "mcard";
    c.dataset.id = id;
    c.innerHTML = `<h3 class="h"></h3><div class="phase">—</div><div class="metrics">—</div><pre>—</pre>`;
    cards.appendChild(c);
  }
  const h3 = c.querySelector(".h");
  const ph = c.querySelector(".phase");
  const m = c.querySelector(".metrics");
  const pr = c.querySelector("pre");
  if (u?.label) h3.textContent = u.label;
  if (u?.phase) {
    ph.textContent = u.phase;
    if (u.phase === "error") c.style.outline = "1px solid var(--err)";
    if (u.phase === "done") c.style.outline = "1px solid var(--ok)";
  }
  if (u?.summary) m.textContent = fmtSummaryLine(u.summary) || m.textContent;
  if (u?.log_tail) pr.textContent = (u.log_tail || "").split("\n").slice(-12).join("\n");
  return c;
}

function handleSseData(log, d, cards) {
  if (d.type === "log" && d.payload) {
    const p = d.payload;
    if (p.message) appendLog(log, p.message);
    if (p.log_tail) appendLog(log, "— server log (tail) —\n" + p.log_tail);
  }
  if (d.type === "config" && d.payload) {
    const p = d.payload;
    if (p.id) {
      setCard(cards, p.id, p);
      if (p.phase) appendLog(log, `config[${p.id}] phase=${p.phase}`);
    }
  }
  if (d.type === "error" && d.payload) {
    const ep = d.payload;
    if (ep && (ep.message != null || ep.error != null)) {
      appendLog(log, "ERR: " + String(ep.message != null ? ep.message : ep.error));
    } else {
      appendLog(log, "ERR: " + String(d.payload));
    }
  }
  if (d.type === "job" && d.payload) {
    const p = d.payload;
    if (p.id) {
      if ($("jobid")) $("jobid").textContent = p.id;
    }
    if (p.state === "finished") {
      appendLog(log, "==== run finished: " + p.state);
      renderResultTable(p.results, p.errors, p.config_ids);
      appendLog(log, "Results table updated below.");
    } else {
      appendLog(log, "==== " + p.state);
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
    appendLog(log, "Submitting…");
  }
  if (cards) cards.replaceChildren();
  const rpanel = $("resultsPanel");
  const rhost = $("resultTableHost");
  if (rhost) rhost.replaceChildren();
  if (rpanel) rpanel.hidden = true;
  const m = mode();
  const form = new FormData();
  form.set("mode", m);
  form.set("concurrency", String(+$("concurrency")?.value || 4));
  const lim = +$("limit")?.value;
  if (m === "text") {
    const t = $("article")?.value || "";
    form.set("text", t);
  }
  if (lim) form.set("limit", String(lim));
  const ids = selectedConfigIds();
  if (ids.length) form.set("configs", JSON.stringify(ids));
  const f = $("file");
  if (f && f.files && f.files[0]) form.set("file", f.files[0], f.files[0].name);

  try {
    if (m === "jsonl" && (!f || !f.files[0])) {
      throw new Error("jsonl: choose a .jsonl file first");
    }
    const st = await fetch("/api/jobs", { method: "POST", body: form });
    if (!st.ok) {
      const t = await st.text();
      throw new Error(t || String(st.status));
    }
    const j = await st.json();
    if ($("jobid")) $("jobid").textContent = j.id;
    if (log) appendLog(log, "job id=" + j.id + " — listening on SSE…");

    return await new Promise((resolve, reject) => {
      const es = new EventSource("/api/jobs/" + j.id + "/events");
      const done = () => {
        es.close();
        if (btn) btn.disabled = false;
        resolve();
      };
      es.onmessage = (e) => {
        try {
          const d = JSON.parse(e.data);
          handleSseData(log, d, cards);
          if (d.type === "job" && d.payload && d.payload.state === "finished") {
            if (log) appendLog(log, "(You may close this page; results were not written to the server.)");
            done();
          }
        } catch (x) {
          if (log) appendLog(log, (x && x.message) || String(x) + " raw=" + (e.data || ""));
        }
      };
      es.onerror = () => {
        if (log) appendLog(log, "EventSource closed or error (re-open the page to start again)");
        done();
      };
    });
  } catch (e) {
    if (log) appendLog(log, (e && e.message) || String(e));
    if (btn) btn.disabled = false;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const goBtn = $("go");
  if (goBtn) goBtn.addEventListener("click", () => go().catch((e) => (console.error(e), undefined)));
  modeRadios().forEach((r) => r.addEventListener("change", onMode));
  onMode();
  loadConfigs().catch((e) => (console.error(e), undefined));
});
