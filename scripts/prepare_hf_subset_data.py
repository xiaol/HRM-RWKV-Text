"""Prepare HRM-Text-style data into the V1Dataset layout.

It accepts rows with `instruction`, `response`, and optional `condition`,
tokenizes them, and writes the same dataset structure used by `dataset_new.py`.
Rows can come from Hugging Face streaming, a single JSONL file, or a local raw
HRM-Text snapshot containing JSONL/parquet files. Use `--compact-uint16` with
vocabularies up to 65536 to halve token storage versus `tokens.npy`.
"""

from __future__ import annotations

import argparse
from array import array
import json
import shutil
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from tokenizers import Tokenizer


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def iter_jsonl_batches(path: Path, batch_size: int) -> Iterable[list[dict]]:
    batch = []
    for row in iter_jsonl(path):
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def iter_parquet_batches(path: Path, columns: Sequence[str], batch_size: int) -> Iterable[list[dict]]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    available_columns = set(parquet_file.schema_arrow.names)
    requested_columns = [column for column in columns if column in available_columns]
    if not requested_columns:
        return

    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=requested_columns):
        rows = batch.to_pydict()
        if not rows:
            continue
        row_count = len(next(iter(rows.values())))
        yield [{key: values[idx] for key, values in rows.items()} for idx in range(row_count)]


def iter_local_snapshot(
    root: Path,
    patterns: Sequence[str],
    columns: Sequence[str],
    parquet_batch_size: int,
) -> Iterable[tuple[Path, list[dict]]]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    files = sorted({path for path in files if path.is_file()})
    if not files:
        raise FileNotFoundError(f"No local files matched {patterns} under {root}")

    for path in files:
        if path.suffix == ".jsonl":
            for batch in iter_jsonl_batches(path, parquet_batch_size):
                yield path, batch
        elif path.suffix == ".parquet":
            for batch in iter_parquet_batches(path, columns=columns, batch_size=parquet_batch_size):
                yield path, batch


def iter_hf_rows(dataset: str, name: str, split: str, streaming: bool) -> Iterable[dict]:
    from datasets import load_dataset

    ds = load_dataset(dataset, name or None, split=split, streaming=streaming)
    yield from ds


def first_present(row: dict, names: list[str]):
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def parse_condition_tokens(raw: str) -> tuple[dict[str, str], dict[str, int]]:
    token_by_condition = {}
    for pair in raw.split(","):
        key, value = pair.split("=", 1)
        token_by_condition[key] = value
    return token_by_condition, {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a streamed subset in HRM-Text V1Dataset format.")
    parser.add_argument("--hf-dataset", default="sapientinc/HRM-Text-data-io-cleaned-20260515")
    parser.add_argument("--hf-name", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-jsonl", default="")
    parser.add_argument("--local-root", default="", help="Local raw HRM-Text snapshot root containing data/**/*.jsonl and data_clustered/**/*.parquet")
    parser.add_argument("--local-patterns", default="data/**/*.jsonl,data_clustered/**/*.parquet")
    parser.add_argument("--parquet-batch-size", type=int, default=8192)
    parser.add_argument("--tokenizer", required=True, help="Tokenizer JSON path.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context-size", type=int, default=4097)
    parser.add_argument("--target-tokens", type=int, default=1_000_000_000)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--compact-uint16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--instruction-field", default="instruction")
    parser.add_argument("--response-field", default="response")
    parser.add_argument("--condition-field", default="condition")
    parser.add_argument("--default-condition", default="direct")
    parser.add_argument("--retry-wait", type=float, default=15.0)
    parser.add_argument("--log-every", type=int, default=10000)
    parser.add_argument("--boq", default="<|im_start|>")
    parser.add_argument("--eoq", default="<|im_end|>")
    parser.add_argument("--eoa", default="<|box_end|>")
    parser.add_argument(
        "--conditions",
        default="direct=<|object_ref_start|>,cot=<|object_ref_end|>,noisy=<|quad_start|>,synth=<|quad_end|>",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_file(args.tokenizer)

    def token_id(token: str) -> int:
        idx = tokenizer.token_to_id(token)
        if idx is None:
            raise ValueError(f"special token {token!r} not in tokenizer vocab")
        return idx

    condition_tokens, _ = parse_condition_tokens(args.conditions)
    condition_ids = {name: token_id(token) for name, token in condition_tokens.items()}
    boq_id = token_id(args.boq)
    eoq_id = token_id(args.eoq)
    eoa_id = token_id(args.eoa)
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if args.compact_uint16 and vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError(f"vocab_size={vocab_size} does not fit compact uint16 tokens")

    token_dtype = np.uint16 if args.compact_uint16 else np.int32
    tokens_path = out_dir / ("tokens.bin" if args.compact_uint16 else "tokens.npy")
    token_writer = tokens_path.open("wb") if args.compact_uint16 else None
    tokens_array: list[int] = []

    inst_start = array("q")
    inst_len = array("q")
    resp_start = array("q")
    resp_len = array("q")
    total_tokens = 0
    kept_rows = 0
    skipped_rows = 0
    max_sample_len = 0

    instruction_fields = [field.strip() for field in args.instruction_field.split(",") if field.strip()]
    response_fields = [field.strip() for field in args.response_field.split(",") if field.strip()]
    condition_fields = [field.strip() for field in args.condition_field.split(",") if field.strip()]
    local_patterns = [pattern.strip() for pattern in args.local_patterns.split(",") if pattern.strip()]
    requested_columns = list(dict.fromkeys(instruction_fields + response_fields + condition_fields))

    sources = sum(bool(x) for x in (args.local_jsonl, args.local_root))
    if sources > 1:
        raise ValueError("Use only one local source: --local-jsonl or --local-root")

    if args.local_jsonl:
        rows = ((Path(args.local_jsonl), batch) for batch in iter_jsonl_batches(Path(args.local_jsonl), args.parquet_batch_size))
    elif args.local_root:
        rows = iter_local_snapshot(
            Path(args.local_root),
            patterns=local_patterns,
            columns=requested_columns,
            parquet_batch_size=args.parquet_batch_size,
        )
    else:
        def iter_hf_batches() -> Iterable[tuple[Path, list[dict]]]:
            batch = []
            for row in iter_hf_rows(args.hf_dataset, args.hf_name, args.split, args.streaming):
                batch.append(row)
                if len(batch) >= args.parquet_batch_size:
                    yield Path(args.hf_dataset), batch
                    batch = []
            if batch:
                yield Path(args.hf_dataset), batch

        rows = iter_hf_batches()

    def write_ids(ids: list[int]) -> None:
        nonlocal total_tokens
        if args.compact_uint16:
            np.asarray(ids, dtype=token_dtype).tofile(token_writer)
        else:
            tokens_array.extend(ids)
        total_tokens += len(ids)

    def condition_to_ids(raw_condition) -> list[int]:
        parts = [part.strip() for part in str(raw_condition or "").split(",") if part.strip()]
        ids = [condition_ids[part] for part in parts if part in condition_ids]
        if not ids:
            ids = [condition_ids[args.default_condition]]
        return ids

    row_iter = iter(rows)
    current_source: Path | None = None
    while True:
        try:
            source_path, row_batch = next(row_iter)
            if source_path != current_source:
                current_source = source_path
                print(f"source={source_path}", flush=True)
        except StopIteration:
            break
        except Exception as exc:
            if args.local_jsonl or args.local_root:
                raise
            print(f"stream read failed: {type(exc).__name__}: {exc}; retrying in {args.retry_wait}s", flush=True)
            time.sleep(args.retry_wait)
            def retry_hf_batches() -> Iterable[tuple[Path, list[dict]]]:
                batch = []
                for row in iter_hf_rows(args.hf_dataset, args.hf_name, args.split, args.streaming):
                    batch.append(row)
                    if len(batch) >= args.parquet_batch_size:
                        yield Path(args.hf_dataset), batch
                        batch = []
                if batch:
                    yield Path(args.hf_dataset), batch

            rows = retry_hf_batches()
            row_iter = iter(rows)
            continue

        instructions: list[str] = []
        responses: list[str] = []
        batch_condition_ids: list[list[int]] = []
        for row in row_batch:
            instruction = first_present(row, instruction_fields)
            response = first_present(row, response_fields)
            if not instruction or not response:
                skipped_rows += 1
                continue
            condition = first_present(row, condition_fields) or args.default_condition
            instructions.append(str(instruction))
            responses.append(str(response))
            batch_condition_ids.append(condition_to_ids(condition))

        if not instructions:
            continue

        encoded_inst = tokenizer.encode_batch(instructions, add_special_tokens=False)
        encoded_resp = tokenizer.encode_batch(responses, add_special_tokens=False)

        for inst_encoding, resp_encoding, condition_token_ids in zip(encoded_inst, encoded_resp, batch_condition_ids):
            inst_ids = inst_encoding.ids
            resp_ids = resp_encoding.ids
            sample = [boq_id, *condition_token_ids, *inst_ids, eoq_id, *resp_ids, eoa_id]
            if len(sample) >= args.context_size:
                skipped_rows += 1
                continue

            i_start = total_tokens
            prefix_len = 1 + len(condition_token_ids) + len(inst_ids) + 1
            r_start = i_start + prefix_len
            write_ids(sample)
            inst_start.append(i_start)
            inst_len.append(r_start - i_start)
            resp_start.append(r_start)
            resp_len.append(len(resp_ids) + 1)
            kept_rows += 1
            max_sample_len = max(max_sample_len, len(sample))

            if args.log_every > 0 and kept_rows % args.log_every == 0:
                print(f"rows={kept_rows:,} tokens={total_tokens:,} skipped={skipped_rows:,}", flush=True)
            if args.max_rows > 0 and kept_rows >= args.max_rows:
                break
            if args.target_tokens > 0 and total_tokens >= args.target_tokens:
                break

        if args.max_rows > 0 and kept_rows >= args.max_rows:
            break
        if args.target_tokens > 0 and total_tokens >= args.target_tokens:
            break

    if token_writer is not None:
        token_writer.close()
    else:
        np.save(tokens_path, np.asarray(tokens_array, dtype=token_dtype))

    if kept_rows == 0:
        raise ValueError("No rows were written")

    shutil.copyfile(args.tokenizer, out_dir / "tokenizer.json")

    tokenizer_info = {
        "tokenizer_path": str(out_dir),
        "boq": args.boq,
        "eoq": args.eoq,
        "eoa": args.eoa,
        "condition_mapping": condition_tokens,
        "vocab_size": vocab_size,
    }
    (out_dir / "tokenizer_info.json").write_text(json.dumps(tokenizer_info) + "\n")
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "tokenizer_info": tokenizer_info,
                "vocab_size": None,
                "max_seq_len": args.context_size,
                "total_length": int(total_tokens),
                "token_dtype": np.dtype(token_dtype).name,
            }
        )
        + "\n"
    )

    inst_start_np = np.frombuffer(inst_start, dtype=np.int64)
    inst_len_np = np.frombuffer(inst_len, dtype=np.int64)
    resp_start_np = np.frombuffer(resp_start, dtype=np.int64)
    resp_len_np = np.frombuffer(resp_len, dtype=np.int64)
    rng = np.random.Generator(np.random.Philox(seed=args.seed))
    for epoch in range(args.epochs):
        perm = rng.permutation(len(inst_start_np))
        ep_dir = out_dir / f"epoch_{epoch}"
        ep_dir.mkdir(exist_ok=True)
        np.save(ep_dir / "inst_start.npy", inst_start_np[perm])
        np.save(ep_dir / "inst_len.npy", inst_len_np[perm])
        np.save(ep_dir / "resp_start.npy", resp_start_np[perm])
        np.save(ep_dir / "resp_len.npy", resp_len_np[perm])

    print(
        f"wrote rows={kept_rows:,} skipped={skipped_rows:,} tokens={total_tokens:,} "
        f"max_sample_len={max_sample_len} to {out_dir}"
    )


if __name__ == "__main__":
    main()
