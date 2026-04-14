import json
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

INPUT_PATH = Path("outputs/dev_predictions_parsed.jsonl")

def main():
    gold = []
    pred = []
    invalid = 0
    total = 0

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            total += 1
            if row["pred_label"] is None:
                invalid += 1
                continue
            gold.append(row["gold_label"])
            pred.append(row["pred_label"])

    if pred:
        acc = accuracy_score(gold, pred)
        f1 = f1_score(gold, pred, average="macro")
    else:
        acc = 0.0
        f1 = 0.0

    print({
        "total": total,
        "valid_predictions": len(pred),
        "invalid_predictions": invalid,
        "accuracy_on_valid": acc,
        "macro_f1_on_valid": f1,
    })

if __name__ == "__main__":
    main()