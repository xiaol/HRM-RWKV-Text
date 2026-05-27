"""Prepare HRM-Text-style data into the V1Dataset layout.

It accepts rows with `instruction`, `response`, and optional `condition`,
tokenizes them, and writes the same dataset structure used by `dataset_new.py`.
Rows can come from Hugging Face streaming, a single JSONL file, or a local raw
HRM-Text snapshot containing JSONL/parquet files. Use `--compact-uint16` with
vocabularies up to 65536 to halve token storage versus int32 tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from tokenizers import Tokenizer


INDEX_FIELDS = ("inst_start", "inst_len", "resp_start", "resp_len")


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


def list_local_snapshot_files(root: Path, patterns: Sequence[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    files = sorted({path for path in files if path.is_file()})
    if not files:
        raise FileNotFoundError(f"No local files matched {patterns} under {root}")
    return files


def iter_local_snapshot(
    root: Path,
    patterns: Sequence[str],
    columns: Sequence[str],
    parquet_batch_size: int,
) -> Iterable[tuple[Path, list[dict]]]:
    for path in list_local_snapshot_files(root, patterns):
        for batch in iter_file_batches(path, columns, parquet_batch_size):
            yield path, batch


def iter_file_batches(path: Path, columns: Sequence[str], batch_size: int) -> Iterable[list[dict]]:
    if path.suffix == ".jsonl":
        yield from iter_jsonl_batches(path, batch_size)
    elif path.suffix == ".parquet":
        yield from iter_parquet_batches(path, columns=columns, batch_size=batch_size)


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


def source_fingerprint(files: Sequence[Path], root: Path | None) -> str:
    digest = hashlib.sha256()
    for path in files:
        rel = path.relative_to(root) if root is not None else path
        stat = path.stat()
        digest.update(str(rel).encode())
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def truncate_file(path: Path, size: int) -> None:
    if path.exists():
        with path.open("ab") as f:
            f.truncate(size)


def copy_raw_index_to_npy(src_path: Path, dst_path: Path, rows: int, chunk_rows: int) -> None:
    src = np.memmap(src_path, mode="r", dtype=np.int64, shape=(rows,))
    dst = np.lib.format.open_memmap(dst_path, mode="w+", dtype=np.int64, shape=(rows,))
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        dst[start:end] = src[start:end]
    dst.flush()
    del dst
    del src


def copy_raw_index_to_shuffled_npy(src_path: Path, dst_path: Path, perm: np.ndarray, chunk_rows: int) -> None:
    rows = int(perm.shape[0])
    src = np.memmap(src_path, mode="r", dtype=np.int64, shape=(rows,))
    dst = np.lib.format.open_memmap(dst_path, mode="w+", dtype=np.int64, shape=(rows,))
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        dst[start:end] = src[perm[start:end]]
    dst.flush()
    del dst
    del src


def write_length_npy(
    inst_len_path: Path,
    resp_len_path: Path,
    dst_path: Path,
    rows: int,
    chunk_rows: int,
    perm: np.ndarray | None = None,
) -> None:
    inst_len = np.memmap(inst_len_path, mode="r", dtype=np.int64, shape=(rows,))
    resp_len = np.memmap(resp_len_path, mode="r", dtype=np.int64, shape=(rows,))
    dst = np.lib.format.open_memmap(dst_path, mode="w+", dtype=np.int64, shape=(rows,))
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        if perm is None:
            dst[start:end] = inst_len[start:end] + resp_len[start:end] - 1
        else:
            idx = perm[start:end]
            dst[start:end] = inst_len[idx] + resp_len[idx] - 1
    dst.flush()
    del dst
    del resp_len
    del inst_len


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a streamed subset in HRM-Text V1Dataset format.")
    parser.add_argument("--hf-dataset", default="sapientinc/HRM-Text-data-io-cleaned-20260515")
    parser.add_argument("--hf-name", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-jsonl", default="")
    parser.add_argument(
        "--local-root",
        default="",
        help="Local raw HRM-Text snapshot root containing data/**/*.jsonl and data_clustered/**/*.parquet",
    )
    parser.add_argument("--local-patterns", default="data/**/*.jsonl,data_clustered/**/*.parquet")
    parser.add_argument("--parquet-batch-size", type=int, default=8192)
    parser.add_argument("--tokenizer-batch-size", type=int, default=4096)
    parser.add_argument("--tokenizer", required=True, help="Tokenizer JSON path.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing output directory before writing.")
    parser.add_argument("--keep-build-files", action="store_true", help="Keep resumable raw index files after finalization.")
    parser.add_argument("--shuffle-index", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--index-copy-chunk-rows", type=int, default=1_000_000)
    parser.add_argument("--max-in-memory-shuffle-rows", type=int, default=20_000_000)
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

    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.tokenizer_batch_size <= 0:
        raise ValueError("--tokenizer-batch-size must be positive")

    out_dir = Path(args.output)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_dir = out_dir / "_prepare"
    build_dir.mkdir(exist_ok=True)

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
    token_itemsize = np.dtype(token_dtype).itemsize
    tokens_path = out_dir / "tokens.bin"
    index_paths = {field: build_dir / f"{field}.bin" for field in INDEX_FIELDS}
    checkpoint_path = out_dir / "prepare_checkpoint.json"

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

    source_files: list[Path] = []
    source_root: Path | None = None
    if args.local_jsonl:
        source_files = [Path(args.local_jsonl)]
    elif args.local_root:
        source_root = Path(args.local_root)
        source_files = list_local_snapshot_files(source_root, local_patterns)

    signature = {
        "version": 2,
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "context_size": args.context_size,
        "compact_uint16": args.compact_uint16,
        "token_dtype": np.dtype(token_dtype).name,
        "instruction_fields": instruction_fields,
        "response_fields": response_fields,
        "condition_fields": condition_fields,
        "default_condition": args.default_condition,
        "boq": args.boq,
        "eoq": args.eoq,
        "eoa": args.eoa,
        "conditions": condition_tokens,
        "vocab_size": vocab_size,
        "source_kind": "local_jsonl" if args.local_jsonl else "local_root" if args.local_root else "hf",
        "local_root": str(source_root.resolve()) if source_root is not None else "",
        "local_patterns": local_patterns,
        "source_count": len(source_files),
        "source_fingerprint": source_fingerprint(source_files, source_root) if source_files else "",
        "hf_dataset": args.hf_dataset,
        "hf_name": args.hf_name,
        "split": args.split,
    }

    resume_next_file_index = 0
    loaded_checkpoint = False
    partial_paths = [tokens_path, checkpoint_path, *index_paths.values(), out_dir / "metadata.json", out_dir / "epoch_0"]
    if args.resume:
        if checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text())
            if checkpoint.get("signature") != signature:
                raise ValueError("Checkpoint signature does not match this preparation command")
            resume_next_file_index = int(checkpoint["next_file_index"])
            total_tokens = int(checkpoint["total_tokens"])
            kept_rows = int(checkpoint["kept_rows"])
            skipped_rows = int(checkpoint["skipped_rows"])
            max_sample_len = int(checkpoint["max_sample_len"])
            truncate_file(tokens_path, total_tokens * token_itemsize)
            for path in index_paths.values():
                truncate_file(path, kept_rows * np.dtype(np.int64).itemsize)
            loaded_checkpoint = True
            print(
                f"resuming next_file_index={resume_next_file_index} rows={kept_rows:,} "
                f"tokens={total_tokens:,} skipped={skipped_rows:,}",
                flush=True,
            )
        elif any(path.exists() for path in partial_paths):
            raise ValueError(f"{out_dir} has partial data but no checkpoint; pass --overwrite to restart")
    elif any(path.exists() for path in partial_paths) and not args.overwrite:
        raise ValueError(f"{out_dir} already contains prepared data; pass --overwrite or --resume")

    token_writer = tokens_path.open("ab" if loaded_checkpoint else "wb")
    index_writers = {field: index_paths[field].open("ab" if loaded_checkpoint else "wb") for field in INDEX_FIELDS}

    def write_ids(ids: list[int]) -> None:
        nonlocal total_tokens
        np.asarray(ids, dtype=token_dtype).tofile(token_writer)
        total_tokens += len(ids)

    def write_index_chunk(values: dict[str, list[int]]) -> None:
        for field in INDEX_FIELDS:
            np.asarray(values[field], dtype=np.int64).tofile(index_writers[field])

    def condition_to_ids(raw_condition) -> list[int]:
        parts = [part.strip() for part in str(raw_condition or "").split(",") if part.strip()]
        ids = [condition_ids[part] for part in parts if part in condition_ids]
        if not ids:
            ids = [condition_ids[args.default_condition]]
        return ids

    def checkpoint(next_file_index: int) -> None:
        token_writer.flush()
        os.fsync(token_writer.fileno())
        for writer in index_writers.values():
            writer.flush()
            os.fsync(writer.fileno())
        atomic_write_json(
            checkpoint_path,
            {
                "signature": signature,
                "next_file_index": next_file_index,
                "total_tokens": total_tokens,
                "kept_rows": kept_rows,
                "skipped_rows": skipped_rows,
                "max_sample_len": max_sample_len,
                "updated_at": time.time(),
            },
        )

    def should_stop() -> bool:
        return (args.max_rows > 0 and kept_rows >= args.max_rows) or (
            args.target_tokens > 0 and total_tokens >= args.target_tokens
        )

    def process_row_batch(row_batch: list[dict]) -> bool:
        nonlocal kept_rows, skipped_rows, max_sample_len
        for start in range(0, len(row_batch), args.tokenizer_batch_size):
            row_chunk = row_batch[start : start + args.tokenizer_batch_size]
            instructions: list[str] = []
            responses: list[str] = []
            batch_condition_ids: list[list[int]] = []
            for row in row_chunk:
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
            index_values = {field: [] for field in INDEX_FIELDS}

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
                index_values["inst_start"].append(i_start)
                index_values["inst_len"].append(r_start - i_start)
                index_values["resp_start"].append(r_start)
                index_values["resp_len"].append(len(resp_ids) + 1)
                kept_rows += 1
                max_sample_len = max(max_sample_len, len(sample))

                if args.log_every > 0 and kept_rows % args.log_every == 0:
                    print(f"rows={kept_rows:,} tokens={total_tokens:,} skipped={skipped_rows:,}", flush=True)
                if should_stop():
                    break

            if index_values["inst_start"]:
                write_index_chunk(index_values)
            if should_stop():
                return True
        return False

    stopped = False
    if source_files:
        for file_index, source_path in enumerate(source_files[resume_next_file_index:], start=resume_next_file_index):
            print(f"source={source_path}", flush=True)
            for row_batch in iter_file_batches(source_path, requested_columns, args.parquet_batch_size):
                stopped = process_row_batch(row_batch)
                if stopped:
                    break
            if stopped:
                break
            checkpoint(file_index + 1)
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

        row_iter = iter(iter_hf_batches())
        while True:
            try:
                source_path, row_batch = next(row_iter)
                print(f"source={source_path}", flush=True)
            except StopIteration:
                break
            except Exception as exc:
                print(f"stream read failed: {type(exc).__name__}: {exc}; retrying in {args.retry_wait}s", flush=True)
                time.sleep(args.retry_wait)
                row_iter = iter(iter_hf_batches())
                continue
            stopped = process_row_batch(row_batch)
            if stopped:
                break

    token_writer.close()
    for writer in index_writers.values():
        writer.close()

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

    rng = np.random.Generator(np.random.Philox(seed=args.seed))
    for epoch in range(args.epochs):
        ep_dir = out_dir / f"epoch_{epoch}"
        if ep_dir.exists():
            shutil.rmtree(ep_dir)
        ep_dir.mkdir(exist_ok=True)
        if args.shuffle_index:
            if kept_rows > args.max_in_memory_shuffle_rows:
                raise ValueError(
                    f"--shuffle-index needs a {kept_rows:,}-row permutation in RAM; "
                    "use --no-shuffle-index for large local snapshots"
                )
            perm = rng.permutation(kept_rows)
            for field, src_path in index_paths.items():
                copy_raw_index_to_shuffled_npy(src_path, ep_dir / f"{field}.npy", perm, args.index_copy_chunk_rows)
            write_length_npy(
                index_paths["inst_len"],
                index_paths["resp_len"],
                ep_dir / "length.npy",
                kept_rows,
                args.index_copy_chunk_rows,
                perm=perm,
            )
        else:
            for field, src_path in index_paths.items():
                copy_raw_index_to_npy(src_path, ep_dir / f"{field}.npy", kept_rows, args.index_copy_chunk_rows)
            write_length_npy(
                index_paths["inst_len"],
                index_paths["resp_len"],
                ep_dir / "length.npy",
                kept_rows,
                args.index_copy_chunk_rows,
            )

    atomic_write_json(
        out_dir / "metadata.json",
        {
            "tokenizer_info": tokenizer_info,
            "vocab_size": None,
            "max_seq_len": args.context_size,
            "total_length": int(total_tokens),
            "token_dtype": np.dtype(token_dtype).name,
        },
    )
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    if not args.keep_build_files:
        shutil.rmtree(build_dir)

    print(
        f"wrote rows={kept_rows:,} skipped={skipped_rows:,} tokens={total_tokens:,} "
        f"max_sample_len={max_sample_len} to {out_dir}"
    )


if __name__ == "__main__":
    main()
