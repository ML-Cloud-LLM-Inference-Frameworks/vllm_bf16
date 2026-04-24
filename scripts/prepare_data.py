from pathlib import Path
import argparse
from common.config import DATA_DIR
from common.data_utils import load_agnews_rows, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output filename. Default uses agnews_<split>_full.jsonl",
    )
    args = parser.parse_args()

    rows = load_agnews_rows(split_name=args.split)

    if args.output is None:
        out_path = DATA_DIR / f"agnews_{args.split}_full.jsonl"
    else:
        out_path = Path(args.output)

    write_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()