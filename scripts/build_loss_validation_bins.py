from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import sentencepiece as spm


DATAFILE_MAGIC = 20240520
DATAFILE_VERSION = 1


def iter_local_rows(path: Path, text_field: str) -> Iterable[str]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                value = row.get(text_field)
                if value:
                    yield str(value)
    else:
        yield path.read_text(encoding="utf-8")


def iter_hf_rows(dataset: str, name: str | None, split: str, text_fields: list[str], max_rows: int) -> Iterable[str]:
    from datasets import load_dataset

    ds = load_dataset(dataset, name, split=split) if name else load_dataset(dataset, split=split)
    count = 0
    for row in ds:
        parts = [str(row[field]) for field in text_fields if field in row and row[field] is not None]
        if parts:
            yield "\n".join(parts)
            count += 1
            if max_rows > 0 and count >= max_rows:
                break


def write_bin(path: Path, tokens: np.ndarray) -> None:
    if tokens.dtype != np.uint16:
        tokens = tokens.astype(np.uint16)
    header = np.zeros(256, dtype="<i4")
    header[0] = DATAFILE_MAGIC
    header[1] = DATAFILE_VERSION
    header[2] = int(tokens.size)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        header.tofile(f)
        tokens.tofile(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build small CE validation .bin shards with a SentencePiece tokenizer.")
    parser.add_argument("--tokenizer", required=True, help="SentencePiece .model path.")
    parser.add_argument("--output", required=True, help="Output .bin path.")
    parser.add_argument("--hf-dataset", default="", help="Hugging Face dataset path.")
    parser.add_argument("--hf-name", default="", help="Optional Hugging Face dataset config/name.")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--text-fields", default="text", help="Comma-separated row fields to join.")
    parser.add_argument("--local", default="", help="Local .txt or .jsonl path.")
    parser.add_argument("--local-text-field", default="text")
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--max-tokens", type=int, default=2_000_000)
    parser.add_argument("--eos", action="store_true", help="Append tokenizer EOS between rows if available.")
    args = parser.parse_args()

    if not args.hf_dataset and not args.local:
        raise ValueError("Provide --hf-dataset or --local")
    if args.hf_dataset and args.local:
        raise ValueError("Use only one of --hf-dataset or --local")

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    if sp.vocab_size() > np.iinfo(np.uint16).max + 1:
        raise ValueError(f"vocab_size={sp.vocab_size()} does not fit uint16 shard format")
    eos_id = sp.eos_id()

    if args.local:
        rows = iter_local_rows(Path(args.local), args.local_text_field)
    else:
        fields = [field.strip() for field in args.text_fields.split(",") if field.strip()]
        rows = iter_hf_rows(args.hf_dataset, args.hf_name or None, args.split, fields, args.max_rows)

    tokens: list[int] = []
    docs = 0
    for text in rows:
        ids = sp.encode(text, out_type=int)
        if not ids:
            continue
        tokens.extend(ids)
        if args.eos and eos_id >= 0:
            tokens.append(eos_id)
        docs += 1
        if args.max_tokens > 0 and len(tokens) >= args.max_tokens:
            tokens = tokens[: args.max_tokens]
            break

    if len(tokens) < 2:
        raise ValueError("Need at least 2 tokens to build a validation shard")

    write_bin(Path(args.output), np.asarray(tokens, dtype=np.uint16))
    print(f"wrote {len(tokens):,} tokens from {docs:,} docs to {args.output}")


if __name__ == "__main__":
    main()
