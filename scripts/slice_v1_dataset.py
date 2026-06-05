#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


INDEX_NAMES = ("inst_start", "inst_len", "resp_start", "resp_len")


def parse_args():
    parser = argparse.ArgumentParser(description="Create a row-prefix subset of a prepared HRM V1 dataset.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-tokens", required=True, type=int, help="Target shifted training tokens from epoch_0/length.npy.")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--scan-rows", type=int, default=10_000_000)
    parser.add_argument("--copy-tokenizer", action="store_true", help="Copy tokenizer files instead of symlinking/copying small metadata only.")
    return parser.parse_args()


def replace_symlink(link_path: Path, target_path: Path):
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(target_path)


def main():
    args = parse_args()
    source = args.source.resolve()
    output = args.output
    source_epoch = source / f"epoch_{args.epoch}"
    output_epoch = output / "epoch_0"

    if not source_epoch.is_dir():
        raise FileNotFoundError(source_epoch)
    if args.target_tokens <= 0:
        raise ValueError("--target-tokens must be positive")
    if args.scan_rows <= 0:
        raise ValueError("--scan-rows must be positive")

    output.mkdir(parents=True, exist_ok=True)
    output_epoch.mkdir(parents=True, exist_ok=True)

    length = np.load(source_epoch / "length.npy", mmap_mode="r")
    rows = 0
    selected_tokens = 0
    running_tokens = 0
    for start in range(0, length.shape[0], args.scan_rows):
        end = min(start + args.scan_rows, length.shape[0])
        chunk_cumsum = np.cumsum(np.asarray(length[start:end]), dtype=np.int64)
        chunk_total = int(chunk_cumsum[-1])
        if running_tokens + chunk_total >= args.target_tokens:
            offset = int(np.searchsorted(chunk_cumsum, args.target_tokens - running_tokens, side="left"))
            rows = start + offset + 1
            selected_tokens = running_tokens + int(chunk_cumsum[offset])
            break
        running_tokens += chunk_total

    if rows == 0:
        rows = int(length.shape[0])
        selected_tokens = running_tokens

    for name in INDEX_NAMES + ("length",):
        src = np.load(source_epoch / f"{name}.npy", mmap_mode="r")
        np.save(output_epoch / f"{name}.npy", src[:rows])

    with (source / "metadata.json").open("r") as f:
        metadata = json.load(f)
    metadata["total_length"] = selected_tokens
    with (output / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    for name in ("tokenizer.json", "tokenizer_info.json"):
        src = source / name
        dst = output / name
        if args.copy_tokenizer:
            shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)

    token_file = "tokens.bin" if (source / "tokens.bin").exists() else "tokens.npy"
    replace_symlink(output / token_file, source / token_file)

    print(f"source={source}")
    print(f"output={output.resolve()}")
    print(f"rows={rows:,}")
    print(f"tokens={selected_tokens:,}")
    print(f"token_file={output / token_file} -> {os.readlink(output / token_file)}")


if __name__ == "__main__":
    main()
