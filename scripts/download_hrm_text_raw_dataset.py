from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the raw HRM-Text pretraining dataset snapshot.")
    parser.add_argument("--repo-id", default="sapientinc/HRM-Text-data-io-cleaned-20260515")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--token-file", default=str(Path.home() / ".cache" / "huggingface" / "token"))
    parser.add_argument("--token", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    output = Path(args.output)
    cache_dir = Path(args.cache_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    token: bool | str
    if args.token:
        token = os.environ.get("HF_TOKEN", "").strip()
        if not token and args.token_file:
            token_path = Path(args.token_file)
            if token_path.exists():
                token = token_path.read_text().strip()
        if token:
            print("Using Hugging Face token from environment or token file.", flush=True)
        else:
            print("No explicit Hugging Face token found; falling back to huggingface_hub defaults.", flush=True)
            token = True
    else:
        token = False

    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=output,
        cache_dir=cache_dir,
        max_workers=args.max_workers,
        token=token,
    )
    print(path)


if __name__ == "__main__":
    main()
