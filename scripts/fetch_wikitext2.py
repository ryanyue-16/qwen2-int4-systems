"""Fetch a pinned slice of the WikiText-2 raw test split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


DATASET = "Salesforce/wikitext"
CONFIG = "wikitext-2-raw-v1"
SPLIT = "test"
ENDPOINT = "https://datasets-server.huggingface.co/rows"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--output", default=str(root / "data/wikitext2_test_100rows.txt"))
    args = parser.parse_args()
    if not 1 <= args.rows <= 100:
        raise ValueError("the dataset rows endpoint accepts 1..100 rows per request")

    query = urlencode(
        {
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "offset": 0,
            "length": args.rows,
        }
    )
    source_url = f"{ENDPOINT}?{query}"
    with urlopen(source_url, timeout=60) as response:
        payload = json.load(response)
    texts = [item["row"]["text"] for item in payload["rows"]]
    corpus = "\n".join(texts).strip() + "\n"

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(corpus, encoding="utf-8")
    digest = hashlib.sha256(corpus.encode("utf-8")).hexdigest()
    metadata = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "offset": 0,
        "rows": len(texts),
        "source_url": source_url,
        "sha256": digest,
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(texts)} rows to {output}")
    print(f"sha256: {digest}")


if __name__ == "__main__":
    main()
