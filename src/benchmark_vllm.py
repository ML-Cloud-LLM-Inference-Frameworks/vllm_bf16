import asyncio
import json
import time
from pathlib import Path
from statistics import mean
from openai import AsyncOpenAI

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
INPUT_PATH = Path("data/agnews_bench_1000.jsonl")
PROMPT_TEMPLATE = Path("prompt_template.txt").read_text(encoding="utf-8")

CONCURRENCY = 4
LIMIT = 100

client = AsyncOpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy",
)

def load_rows(path: Path, limit=None):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            rows.append(json.loads(line))
    return rows

async def one_request(row, sem):
    prompt = PROMPT_TEMPLATE.format(article=row["text"])
    async with sem:
        t0 = time.perf_counter()
        resp = await client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=1,
            max_tokens=8,
        )
        dt = time.perf_counter() - t0
        return {
            "id": row["id"],
            "latency_s": dt,
            "raw_output": resp.choices[0].message.content or "",
        }

async def main():
    rows = load_rows(INPUT_PATH, LIMIT)
    sem = asyncio.Semaphore(CONCURRENCY)

    t_start = time.perf_counter()
    results = await asyncio.gather(*[one_request(r, sem) for r in rows])
    total_time = time.perf_counter() - t_start

    latencies = [r["latency_s"] for r in results]
    throughput = len(results) / total_time if total_time > 0 else 0.0

    xs = sorted(latencies)
    def pct(p):
        if not xs:
            return None
        k = int(round((p / 100) * (len(xs) - 1)))
        return xs[k]

    summary = {
        "num_requests": len(results),
        "concurrency": CONCURRENCY,
        "total_time_s": total_time,
        "throughput_req_per_s": throughput,
        "latency_mean_s": mean(latencies) if latencies else None,
        "latency_p50_s": pct(50),
        "latency_p95_s": pct(95),
        "latency_p99_s": pct(99),
    }

    Path("outputs").mkdir(exist_ok=True)
    out_path = Path("outputs") / f"bench_c{CONCURRENCY}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    asyncio.run(main())