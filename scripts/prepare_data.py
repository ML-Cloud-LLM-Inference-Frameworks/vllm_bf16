import json
from pathlib import Path
from datasets import load_dataset

OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ID = "pietrolesci/agnews"

LABEL_MAP = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}

DEV_PER_LABEL = 25
BENCH_PER_LABEL = 250

def get_text(example: dict) -> str:
    for key in ["text", "description", "content", "article"]:
        if key in example and isinstance(example[key], str):
            return example[key]
    title = example.get("title", "")
    desc = example.get("description", "")
    merged = (title + "\n" + desc).strip()
    if merged:
        return merged
    raise KeyError(f"No text field found. Keys: {list(example.keys())}")

def get_label(example: dict) -> int:
    for key in ["label", "labels"]:
        if key in example:
            return int(example[key])
    raise KeyError(f"No label field found. Keys: {list(example.keys())}")

def build_subset(split, per_label: int):
    buckets = {0: [], 1: [], 2: [], 3: []}
    for ex in split:
        y = get_label(ex)
        if y in buckets and len(buckets[y]) < per_label:
            buckets[y].append({
                "id": None,
                "text": get_text(ex),
                "label_id": y,
                "label_name": LABEL_MAP[y],
            })
        if all(len(v) >= per_label for v in buckets.values()):
            break

    rows = []
    idx = 0
    for y in range(4):
        for ex in buckets[y]:
            ex["id"] = idx
            rows.append(ex)
            idx += 1
    return rows

def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def main():
    ds = load_dataset(DATASET_ID)
    split_name = "test" if "test" in ds else list(ds.keys())[0]
    split = ds[split_name]

    dev_rows = build_subset(split, DEV_PER_LABEL)
    bench_rows = build_subset(split, BENCH_PER_LABEL)

    write_jsonl(OUT_DIR / "agnews_dev_100.jsonl", dev_rows)
    write_jsonl(OUT_DIR / "agnews_bench_1000.jsonl", bench_rows)

    print(f"Wrote {len(dev_rows)} rows to data/agnews_dev_100.jsonl")
    print(f"Wrote {len(bench_rows)} rows to data/agnews_bench_1000.jsonl")

if __name__ == "__main__":
    main()