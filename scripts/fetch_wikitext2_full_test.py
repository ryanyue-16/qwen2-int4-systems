"""Fetch the complete pinned WikiText-2 raw test split for Phase C."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from urllib.request import urlopen


DATASET = "Salesforce/wikitext"
CONFIG = "wikitext-2-raw-v1"
SPLIT = "test"
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
PARQUET_NAME = "test-00000-of-00001.parquet"
SOURCE_URL = (
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/"
    f"{REVISION}/{CONFIG}/{PARQUET_NAME}"
)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=str(root / "data/phase_c/wikitext2_test_full.txt")
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    metadata_path = output.with_suffix(".metadata.json")
    existing = [path for path in (output, metadata_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing outputs: {[str(path) for path in existing]}"
        )

    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("pyarrow is required to prepare the full split") from error

    with urlopen(SOURCE_URL, timeout=120) as response:
        parquet_bytes = response.read()
    with tempfile.NamedTemporaryFile(suffix=".parquet") as handle:
        handle.write(parquet_bytes)
        handle.flush()
        table = parquet.read_table(handle.name, columns=["text"])
    texts = [text.rstrip() for text in table.column("text").to_pylist()]
    if len(texts) != 4358:
        raise RuntimeError(f"expected 4358 rows, received {len(texts)}")
    corpus = "\n".join(texts).strip() + "\n"
    corpus_bytes = corpus.encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(corpus_bytes)
    metadata = {
        "schema_version": 1,
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "revision": REVISION,
        "rows": len(texts),
        "source_url": SOURCE_URL,
        "source_parquet": PARQUET_NAME,
        "source_parquet_sha256": digest_bytes(parquet_bytes),
        "normalization": "Remove trailing whitespace from each source row.",
        "corpus_sha256": digest_bytes(corpus_bytes),
        "corpus_size_bytes": len(corpus_bytes),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(texts)} rows to {output}")
    print(f"corpus sha256: {metadata['corpus_sha256']}")


if __name__ == "__main__":
    main()
