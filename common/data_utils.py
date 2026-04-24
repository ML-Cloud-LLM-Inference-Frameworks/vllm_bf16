import json
from pathlib import Path
from typing import Iterable, List, Dict, Optional
from datasets import load_dataset

from common.config import DATASET_ID, LABEL_MAP


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


def load_agnews_rows(split_name: str = "test") -> List[Dict]:
    ds = load_dataset(DATASET_ID)

    if split_name == "all":
        splits = []
        for s in ds.keys():
            splits.extend(ds[s])
        split = splits
    else:
        if split_name not in ds:
            raise ValueError(f"Split '{split_name}' not found. Available: {list(ds.keys())}")
        split = ds[split_name]

    rows = []
    for idx, ex in enumerate(split):
        y = get_label(ex)
        rows.append({
            "id": idx,
            "text": get_text(ex),
            "label_id": y,
            "label_name": LABEL_MAP[y],
        })

    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            rows.append(json.loads(line))
    return rows