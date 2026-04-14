import json
import time
from pathlib import Path
from openai import OpenAI

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
INPUT_PATH = Path("data/agnews_dev_100.jsonl")
OUTPUT_PATH = Path("outputs/dev_predictions_raw.jsonl")
PROMPT_TEMPLATE = Path("prompt_template.txt").read_text(encoding="utf-8")

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy",
)

def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        for row in load_jsonl(INPUT_PATH):
            prompt = PROMPT_TEMPLATE.format(article=row["text"])

            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                top_p=1,
                max_tokens=8,
            )
            latency_s = time.perf_counter() - t0

            raw_output = resp.choices[0].message.content or ""

            out = {
                "id": row["id"],
                "gold_label": row["label_name"],
                "raw_output": raw_output,
                "latency_s": latency_s,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Saved predictions to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()