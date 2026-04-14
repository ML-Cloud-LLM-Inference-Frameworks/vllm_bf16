import json
from pathlib import Path

INPUT_PATH = Path("outputs/dev_predictions_raw.jsonl")
OUTPUT_PATH = Path("outputs/dev_predictions_parsed.jsonl")

LABELS = ["World", "Sports", "Business", "Sci/Tech"]

def parse_label(text: str):
    if not text:
        return None

    x = text.strip().lower()

    for label in LABELS:
        if x == label.lower():
            return label

    candidates = []
    if "world" in x:
        candidates.append("World")
    if "sports" in x or "sport" in x:
        candidates.append("Sports")
    if "business" in x:
        candidates.append("Business")
    if (
        "sci/tech" in x
        or "science/technology" in x
        or "science and technology" in x
        or "technology" in x
        or "tech" in x
    ):
        candidates.append("Sci/Tech")

    if len(candidates) == 1:
        return candidates[0]
    return None

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with INPUT_PATH.open("r", encoding="utf-8") as fin, OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            pred = parse_label(row["raw_output"])
            row["pred_label"] = pred
            row["is_valid"] = pred is not None
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved parsed predictions to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()