import argparse
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from common.backend_client import classify_article
from common.data_utils import load_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input JSONL")
    parser.add_argument("--output", required=True, help="Path to save predictions JSONL")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model-id", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--config-name", default="vllm_bf16")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input), limit=args.limit)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gold = []
    pred = []
    invalid = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for row in tqdm(rows, desc="Running evaluation"):
            result = classify_article(
                article=row["text"],
                base_url=args.base_url,
                model_id=args.model_id,
            )

            pred_label = result["prediction"]
            if pred_label is None:
                invalid += 1
            else:
                gold.append(row["label_name"])
                pred.append(pred_label)

            record = {
                "id": row["id"],
                "gold_label": row["label_name"],
                "pred_label": pred_label,
                "raw_output": result["raw_output"],
                "latency_s": result["latency_s"],
                "config_name": args.config_name,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    acc_valid = accuracy_score(gold, pred) if pred else 0.0
    f1_valid = f1_score(gold, pred, average="macro") if pred else 0.0
    acc_overall = len(pred) / len(rows) * acc_valid if rows else 0.0

    summary = {
        "config_name": args.config_name,
        "n_total": len(rows),
        "n_valid": len(pred),
        "n_invalid": invalid,
        "accuracy_valid_only": acc_valid,
        "macro_f1_valid_only": f1_valid,
        "accuracy_overall_invalid_as_wrong": acc_overall,
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()