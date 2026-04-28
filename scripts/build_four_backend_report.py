import argparse
import csv
import html
import io
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "branch_comparison"
CONCURRENCIES = [1, 2, 4, 8, 16]
BACKENDS = [
    {
        "slug": "hf_baseline_bf16",
        "label": "HF baseline bf16",
        "short_label": "HF baseline",
        "ref": "origin/haotian/hf-baseline",
        "bench_template": "outputs/hf_baseline_bf16/bench_1000_c{c}.json",
        "nvidia_template": "outputs/hf_baseline_bf16/nvidia_smi/smi_1000_c{c}.csv",
        "prom_template": None,
        "family": "hf",
        "color": "#1f4b99",
    },
    {
        "slug": "vllm_bf16",
        "label": "vLLM bf16",
        "short_label": "vLLM bf16",
        "ref": "origin/main",
        "bench_template": "outputs/vllm_bf16/bench_1000_c{c}.json",
        "nvidia_template": "outputs/vllm_bf16/nvidia_smi/smi_1000_c{c}.csv",
        "prom_template": "outputs/vllm_bf16/proms/prom_1000_c{c}.json",
        "family": "vllm",
        "color": "#d55c00",
    },
    {
        "slug": "vllm_bf16_prefixcaching",
        "label": "vLLM bf16 + prefix caching",
        "short_label": "vLLM + APC",
        "ref": "origin/main",
        "bench_template": "outputs/vllm_bf16_prefixcaching/bench_1000_c{c}.json",
        "nvidia_template": "outputs/vllm_bf16_prefixcaching/nvidia_smi/smi_1000_c{c}.csv",
        "prom_template": "outputs/vllm_bf16_prefixcaching/proms/prom_1000_c{c}.json",
        "family": "vllm",
        "color": "#16803c",
    },
    {
        "slug": "vllm_int4",
        "label": "vLLM int4",
        "short_label": "vLLM int4",
        "ref": "origin/main",
        "bench_template": "outputs/vllm_int4/bench_1000_c{c}.json",
        "nvidia_template": "outputs/vllm_int4/nvidia_smi/smi_1000_c{c}.csv",
        "prom_template": "outputs/vllm_int4/proms/prom_1000_c{c}.json",
        "family": "vllm",
        "color": "#7a2ea8",
    },
]
PROM_LINE_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)


def git_show_text(ref: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def load_json_from_git(ref: str, path: str) -> dict:
    return json.loads(git_show_text(ref, path))


def parse_percent_or_mib(raw: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", raw or "")
    return float(match.group(0)) if match else None


def parse_nvidia_csv(csv_text: str) -> dict:
    reader = csv.DictReader(io.StringIO(csv_text), skipinitialspace=True)
    rows = list(reader)
    gpu_utils = [parse_percent_or_mib(row.get("utilization.gpu [%]")) for row in rows]
    mem_utils = [parse_percent_or_mib(row.get("utilization.memory [%]")) for row in rows]
    mem_used = [parse_percent_or_mib(row.get("memory.used [MiB]")) for row in rows]
    mem_total = [parse_percent_or_mib(row.get("memory.total [MiB]")) for row in rows]

    def clean(values: list[float | None]) -> list[float]:
        return [value for value in values if value is not None]

    gpu_utils = clean(gpu_utils)
    mem_utils = clean(mem_utils)
    mem_used = clean(mem_used)
    mem_total = clean(mem_total)

    summary = {"samples": len(rows)}
    if gpu_utils:
        summary["gpu_util_avg_pct"] = sum(gpu_utils) / len(gpu_utils)
        summary["gpu_util_max_pct"] = max(gpu_utils)
    if mem_utils:
        summary["mem_util_avg_pct"] = sum(mem_utils) / len(mem_utils)
        summary["mem_util_max_pct"] = max(mem_utils)
    if mem_used:
        summary["mem_used_avg_mib"] = sum(mem_used) / len(mem_used)
        summary["mem_used_max_mib"] = max(mem_used)
    if mem_total:
        summary["mem_total_mib"] = max(mem_total)
    return summary


def parse_prometheus_text(raw_text: str) -> dict[str, list[float]]:
    metrics: dict[str, list[float]] = {}
    for line in raw_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = PROM_LINE_RE.match(line)
        if not match:
            continue
        name = match.group("name")
        value = float(match.group("value"))
        metrics.setdefault(name, []).append(value)
    return metrics


def sum_metric(metrics: dict[str, list[float]], name: str) -> float | None:
    values = metrics.get(name)
    if not values:
        return None
    return sum(values)


def max_metric(metrics: dict[str, list[float]], name: str) -> float | None:
    values = metrics.get(name)
    if not values:
        return None
    return max(values)


def mean_from_sum_and_count(metrics: dict[str, list[float]], base_name: str) -> float | None:
    total = sum_metric(metrics, f"{base_name}_sum")
    count = sum_metric(metrics, f"{base_name}_count")
    if total is None or count in (None, 0):
        return None
    return total / count


def derive_vllm_metrics(prometheus_doc: dict, bench_doc: dict) -> dict:
    raw_text = prometheus_doc.get("raw_prometheus", "")
    metrics = parse_prometheus_text(raw_text) if raw_text else {}
    derived: dict[str, float | None] = {}

    derived["ttft_mean_s"] = mean_from_sum_and_count(metrics, "vllm:time_to_first_token_seconds")

    bench_derived = (bench_doc.get("prometheus") or {}).get("derived") or {}
    if derived["ttft_mean_s"] is None and bench_derived.get("ttft_mean_s") is not None:
        derived["ttft_mean_s"] = bench_derived["ttft_mean_s"]

    return derived


def build_rows() -> list[dict]:
    rows: list[dict] = []
    for backend in BACKENDS:
        for concurrency in CONCURRENCIES:
            bench_path = backend["bench_template"].format(c=concurrency)
            bench = load_json_from_git(backend["ref"], bench_path)

            row = {
                "backend": backend["slug"],
                "backend_label": backend["label"],
                "backend_short_label": backend["short_label"],
                "family": backend["family"],
                "source_ref": backend["ref"],
                "source_bench_path": bench_path,
                "concurrency": concurrency,
                "model_id": bench.get("model_id"),
                "n_requests_measured": bench.get("n_requests_measured"),
                "warmup_requests": bench.get("warmup_requests"),
                "throughput_req_per_s": bench.get("throughput_req_per_s"),
                "latency_avg_s": bench.get("latency_avg_s"),
                "latency_p50_s": bench.get("latency_p50_s"),
                "latency_p95_s": bench.get("latency_p95_s"),
                "latency_p99_s": bench.get("latency_p99_s"),
                "ttft_avg_s": bench.get("ttft_avg_s"),
                "ttft_p50_s": bench.get("ttft_p50_s"),
                "ttft_p95_s": bench.get("ttft_p95_s"),
                "n_valid_predictions": bench.get("n_valid_predictions"),
                "n_invalid_predictions": bench.get("n_invalid_predictions"),
                "accuracy_valid_only": bench.get("accuracy_valid_only"),
                "accuracy_overall_invalid_as_wrong": bench.get("accuracy_overall_invalid_as_wrong"),
            }

            if row["throughput_req_per_s"] and row["n_requests_measured"]:
                row["measured_window_s"] = row["n_requests_measured"] / row["throughput_req_per_s"]

            nvidia_path = backend["nvidia_template"].format(c=concurrency)
            nvidia_text = git_show_text(backend["ref"], nvidia_path)
            row["source_nvidia_path"] = nvidia_path
            row.update(parse_nvidia_csv(nvidia_text))

            if backend["prom_template"]:
                prom_path = backend["prom_template"].format(c=concurrency)
                prom_doc = load_json_from_git(backend["ref"], prom_path)
                row["source_prometheus_path"] = prom_path
                derived = derive_vllm_metrics(prom_doc, bench)
                row.update(derived)
                if row.get("ttft_avg_s") is None and row.get("ttft_mean_s") is not None:
                    row["ttft_avg_s"] = row.get("ttft_mean_s")
            rows.append(row)
    return rows


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sort_rows(rows: list[dict]) -> list[dict]:
    order = {backend["slug"]: index for index, backend in enumerate(BACKENDS)}
    return sorted(rows, key=lambda row: (order[row["backend"]], row["concurrency"]))


def format_number(value, digits: int = 3, pct: bool = False) -> str:
    if value is None:
        return ""
    if pct:
        return f"{value * 100:.1f}%"
    return f"{value:.{digits}f}"


def build_csv(rows: list[dict], output_path: Path) -> None:
    all_keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                all_keys.append(key)
                seen.add(key)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)


def card_html(title: str, value: str, note: str) -> str:
    return f"""
    <div class="card">
      <div class="card-title">{html.escape(title)}</div>
      <div class="card-value">{html.escape(value)}</div>
      <div class="card-note">{html.escape(note)}</div>
    </div>
    """


def line_chart_svg(rows: list[dict], metric: str, title: str, formatter, y_axis_label: str) -> str:
    series = {}
    for backend in BACKENDS:
        points = [
            (row["concurrency"], row.get(metric))
            for row in rows
            if row["backend"] == backend["slug"] and row.get(metric) is not None
        ]
        if points:
            series[backend["slug"]] = points

    all_values = [value for points in series.values() for _, value in points]
    if not all_values:
        return f"<section><h3>{html.escape(title)}</h3><p>No data.</p></section>"

    width = 860
    height = 320
    margin_left = 72
    margin_right = 24
    margin_top = 26
    margin_bottom = 56
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    y_min = 0.0
    y_max = max(all_values)
    if y_max <= y_min:
        y_max = y_min + 1.0
    y_max *= 1.08
    x_values = CONCURRENCIES

    def x_pos(concurrency: int) -> float:
        index = x_values.index(concurrency)
        if len(x_values) == 1:
            return margin_left + plot_width / 2
        return margin_left + (plot_width * index / (len(x_values) - 1))

    def y_pos(value: float) -> float:
        return margin_top + plot_height - ((value - y_min) / (y_max - y_min) * plot_height)

    grid_lines = []
    for tick in range(5):
        ratio = tick / 4
        y_value = y_min + (y_max - y_min) * ratio
        y = y_pos(y_value)
        grid_lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" class="grid" />'
        )
        grid_lines.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" class="axis-label">{html.escape(formatter(y_value))}</text>'
        )

    x_ticks = []
    for concurrency in x_values:
        x = x_pos(concurrency)
        x_ticks.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}" class="grid xgrid" />')
        x_ticks.append(
            f'<text x="{x:.2f}" y="{height - margin_bottom + 22}" text-anchor="middle" class="axis-label">c{concurrency}</text>'
        )

    paths = []
    points_markup = []
    legend_items = []
    for backend in BACKENDS:
        points = series.get(backend["slug"])
        if not points:
            continue
        path_cmd = []
        for index, (concurrency, value) in enumerate(points):
            command = "M" if index == 0 else "L"
            path_cmd.append(f"{command} {x_pos(concurrency):.2f} {y_pos(value):.2f}")
            points_markup.append(
                f'<circle cx="{x_pos(concurrency):.2f}" cy="{y_pos(value):.2f}" r="4" fill="{backend["color"]}">'
                f"<title>{html.escape(backend['short_label'])} c{concurrency}: {html.escape(formatter(value))}</title>"
                f"</circle>"
            )
        paths.append(
            f'<path d="{" ".join(path_cmd)}" fill="none" stroke="{backend["color"]}" stroke-width="3" stroke-linecap="round" />'
        )
        legend_items.append(
            f'<div class="legend-item"><span class="legend-swatch" style="background:{backend["color"]}"></span>{html.escape(backend["short_label"])}</div>'
        )

    return f"""
    <section class="chart-section">
      <div class="chart-header">
        <h3>{html.escape(title)}</h3>
        <div class="legend">{''.join(legend_items)}</div>
      </div>
      <svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{html.escape(title)}">
        <text x="{width / 2:.2f}" y="18" text-anchor="middle" class="chart-title">{html.escape(title)}</text>
        <text x="18" y="{height / 2:.2f}" transform="rotate(-90 18 {height / 2:.2f})" class="axis-label">{html.escape(y_axis_label)}</text>
        {''.join(grid_lines)}
        {''.join(x_ticks)}
        <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" class="axis" />
        <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" class="axis" />
        {''.join(paths)}
        {''.join(points_markup)}
      </svg>
    </section>
    """


def metrics_table(rows: list[dict]) -> str:
    headers = [
        "Backend",
        "Concurrency",
        "Throughput",
        "Latency avg",
        "Latency p50",
        "Latency p95",
        "Latency p99",
        "TTFT avg",
        "Accuracy valid-only",
        "Accuracy overall",
        "Invalid preds",
        "GPU util avg",
        "GPU util max",
        "GPU mem util avg",
        "GPU mem avg",
        "GPU mem max",
    ]
    body_rows = []
    for row in rows:
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(row['backend_short_label'])}</td>"
            f"<td>c{row['concurrency']}</td>"
            f"<td>{html.escape(format_number(row.get('throughput_req_per_s'), 2))}</td>"
            f"<td>{html.escape(format_number(row.get('latency_avg_s'), 3))}</td>"
            f"<td>{html.escape(format_number(row.get('latency_p50_s'), 3))}</td>"
            f"<td>{html.escape(format_number(row.get('latency_p95_s'), 3))}</td>"
            f"<td>{html.escape(format_number(row.get('latency_p99_s'), 3))}</td>"
            f"<td>{html.escape(format_number(row.get('ttft_avg_s'), 3))}</td>"
            f"<td>{html.escape(format_number(row.get('accuracy_valid_only'), 3, pct=True))}</td>"
            f"<td>{html.escape(format_number(row.get('accuracy_overall_invalid_as_wrong'), 3, pct=True))}</td>"
            f"<td>{html.escape(str(row.get('n_invalid_predictions', '')))}</td>"
            f"<td>{html.escape(format_number(row.get('gpu_util_avg_pct'), 1))}</td>"
            f"<td>{html.escape(format_number(row.get('gpu_util_max_pct'), 1))}</td>"
            f"<td>{html.escape(format_number(row.get('mem_util_avg_pct'), 1))}</td>"
            f"<td>{html.escape(format_number(row.get('mem_used_avg_mib'), 0))}</td>"
            f"<td>{html.escape(format_number(row.get('mem_used_max_mib'), 0))}</td>"
            "</tr>"
        )
    header_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    return f"""
    <section>
      <h3>Normalized Results Table</h3>
      <div class="table-wrap">
        <table>
          <thead><tr>{header_html}</tr></thead>
          <tbody>{''.join(body_rows)}</tbody>
        </table>
      </div>
    </section>
    """


def best_row(rows: list[dict], metric: str, reverse: bool = True) -> dict | None:
    candidates = [row for row in rows if row.get(metric) is not None]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: row[metric], reverse=reverse)[0]


def build_html(rows: list[dict], output_path: Path) -> None:
    best_throughput = best_row(rows, "throughput_req_per_s", reverse=True)
    best_latency = best_row(rows, "latency_p95_s", reverse=False)
    best_accuracy = best_row(rows, "accuracy_overall_invalid_as_wrong", reverse=True)
    best_ttft = best_row(rows, "ttft_avg_s", reverse=False)

    cards = []
    if best_throughput:
        cards.append(
            card_html(
                "Best throughput",
                f"{best_throughput['backend_short_label']} c{best_throughput['concurrency']}",
                f"{format_number(best_throughput['throughput_req_per_s'], 2)} req/s",
            )
        )
    if best_latency:
        cards.append(
            card_html(
                "Lowest p95 latency",
                f"{best_latency['backend_short_label']} c{best_latency['concurrency']}",
                f"{format_number(best_latency['latency_p95_s'], 3)} s",
            )
        )
    if best_accuracy:
        cards.append(
            card_html(
                "Best overall accuracy",
                f"{best_accuracy['backend_short_label']} c{best_accuracy['concurrency']}",
                format_number(best_accuracy["accuracy_overall_invalid_as_wrong"], 3, pct=True),
            )
        )
    if best_ttft:
        cards.append(
            card_html(
                "Lowest TTFT",
                f"{best_ttft['backend_short_label']} c{best_ttft['concurrency']}",
                f"{format_number(best_ttft['ttft_avg_s'], 3)} s",
            )
        )

    charts = [
        line_chart_svg(rows, "throughput_req_per_s", "Throughput vs concurrency", lambda value: f"{value:.1f}", "req/s"),
        line_chart_svg(rows, "latency_avg_s", "Average latency vs concurrency", lambda value: f"{value:.2f}", "seconds"),
        line_chart_svg(rows, "latency_p50_s", "p50 latency vs concurrency", lambda value: f"{value:.2f}", "seconds"),
        line_chart_svg(rows, "latency_p95_s", "p95 latency vs concurrency", lambda value: f"{value:.2f}", "seconds"),
        line_chart_svg(rows, "latency_p99_s", "p99 latency vs concurrency", lambda value: f"{value:.2f}", "seconds"),
        line_chart_svg(rows, "ttft_avg_s", "Average TTFT vs concurrency", lambda value: f"{value:.2f}", "seconds"),
        line_chart_svg(
            rows,
            "accuracy_valid_only",
            "Valid-only accuracy vs concurrency",
            lambda value: f"{value * 100:.1f}%",
            "accuracy",
        ),
        line_chart_svg(
            rows,
            "accuracy_overall_invalid_as_wrong",
            "Overall accuracy vs concurrency",
            lambda value: f"{value * 100:.1f}%",
            "accuracy",
        ),
        line_chart_svg(rows, "n_invalid_predictions", "Invalid predictions vs concurrency", lambda value: f"{value:.0f}", "count"),
        line_chart_svg(rows, "gpu_util_avg_pct", "Average GPU utilization vs concurrency", lambda value: f"{value:.0f}%", "gpu %"),
        line_chart_svg(rows, "gpu_util_max_pct", "Peak GPU utilization vs concurrency", lambda value: f"{value:.0f}%", "gpu %"),
        line_chart_svg(rows, "mem_util_avg_pct", "Average GPU memory utilization vs concurrency", lambda value: f"{value:.0f}%", "mem %"),
        line_chart_svg(rows, "mem_util_max_pct", "Peak GPU memory utilization vs concurrency", lambda value: f"{value:.0f}%", "mem %"),
        line_chart_svg(rows, "mem_used_avg_mib", "Average GPU memory used vs concurrency", lambda value: f"{value:.0f}", "MiB"),
        line_chart_svg(rows, "mem_used_max_mib", "Peak GPU memory used vs concurrency", lambda value: f"{value:.0f}", "MiB"),
    ]

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Four-backend Benchmark Comparison</title>
  <style>
    :root {{
      --bg: #f7f3ea;
      --surface: #fffdf8;
      --ink: #1f1f1f;
      --muted: #6a665f;
      --grid: #d9d1c4;
      --border: #cfc5b5;
      --accent: #a04b00;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(160, 75, 0, 0.08), transparent 24rem),
        linear-gradient(180deg, #fbf8f1, var(--bg));
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 24px 56px;
    }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    p {{ line-height: 1.55; }}
    .lede {{
      max-width: 78ch;
      color: var(--muted);
      margin-bottom: 22px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 0.95rem;
      margin-bottom: 24px;
    }}
    .card-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 16px 18px;
      box-shadow: 0 8px 30px rgba(31, 31, 31, 0.06);
    }}
    .card-title {{
      color: var(--muted);
      font-size: 0.92rem;
      margin-bottom: 8px;
    }}
    .card-value {{
      font-size: 1.25rem;
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .card-note {{
      color: var(--muted);
      font-size: 0.95rem;
    }}
    .chart-section {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 16px;
      margin-bottom: 20px;
      box-shadow: 0 8px 30px rgba(31, 31, 31, 0.05);
    }}
    .chart-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      margin-bottom: 8px;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .legend-swatch {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      display: inline-block;
    }}
    .chart {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .axis, .grid {{
      stroke: var(--grid);
      stroke-width: 1;
    }}
    .xgrid {{
      opacity: 0.55;
    }}
    .axis-label {{
      fill: var(--muted);
      font-size: 12px;
    }}
    .chart-title {{
      fill: var(--ink);
      font-size: 15px;
      font-weight: 600;
    }}
    section {{
      margin-bottom: 28px;
    }}
    .table-wrap {{
      overflow-x: auto;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 8px 30px rgba(31, 31, 31, 0.05);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 900px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid #ece4d8;
      text-align: left;
      font-size: 0.95rem;
    }}
    th {{
      background: #f3ede1;
      position: sticky;
      top: 0;
    }}
    .note {{
      color: var(--muted);
      font-size: 0.94rem;
      max-width: 80ch;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Four-backend benchmark comparison</h1>
    <p class="lede">
      This report compares the three vLLM result folders from <code>origin/main</code> with the
      HF baseline result folder from <code>origin/haotian/hf-baseline</code>. It was generated
      directly from branch artifacts with <code>git show</code>, so no branch checkout or merge was required.
    </p>
    <div class="meta">
      Generated at {html.escape(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))}.
      Concurrency sweep: c1, c2, c4, c8, c16.
    </div>
    <div class="card-grid">{''.join(cards)}</div>
    <section>
      <h2>Common Metrics</h2>
      <p class="note">
        TTFT is compared across all four backends, but it comes from different instrumentation:
        HF baseline uses per-response timing, while vLLM uses the Prometheus time-to-first-token mean when the benchmark JSON omits TTFT.
      </p>
    </section>
    {''.join(charts)}
    {metrics_table(rows)}
  </main>
</body>
</html>
"""
    output_path.write_text(html_doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    ensure_output_dir(args.output_dir)
    rows = sort_rows(build_rows())

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [{key: backend[key] for key in ("slug", "label", "ref", "bench_template", "prom_template", "nvidia_template")} for backend in BACKENDS],
        "concurrencies": CONCURRENCIES,
        "rows": rows,
    }

    json_path = args.output_dir / "four_backend_summary.json"
    csv_path = args.output_dir / "four_backend_summary.csv"
    html_path = args.output_dir / "four_backend_comparison.html"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    build_csv(rows, csv_path)
    build_html(rows, html_path)

    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "html": str(html_path)}, indent=2))


if __name__ == "__main__":
    main()
