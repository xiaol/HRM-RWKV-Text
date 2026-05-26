from __future__ import annotations

import argparse
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests
from huggingface_hub import HfApi, snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the raw HRM-Text pretraining dataset snapshot.")
    parser.add_argument("--repo-id", default="sapientinc/HRM-Text-data-io-cleaned-20260515")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--token-file", default=str(Path.home() / ".cache" / "huggingface" / "token"))
    parser.add_argument("--token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT"))
    parser.add_argument(
        "--method",
        choices=("snapshot", "direct"),
        default="snapshot",
        help="Use huggingface_hub snapshot_download, or direct resumable GET downloads.",
    )
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

    if args.endpoint:
        print(f"Using Hugging Face endpoint: {args.endpoint}", flush=True)

    # snapshot_download returns an existing non-empty local_dir when the remote
    # cannot be reached. Preflight metadata first so transient network failures
    # do not look like a completed download.
    api = HfApi(endpoint=args.endpoint)
    info = api.repo_info(
        repo_id=args.repo_id,
        repo_type="dataset",
        token=token,
    )
    print(
        f"Remote snapshot {info.sha} has {len(info.siblings or [])} files.",
        flush=True,
    )

    if args.method == "direct":
        files = sorted(
            sibling.rfilename
            for sibling in (info.siblings or [])
            if getattr(sibling, "rfilename", None)
        )
        completed = _download_direct(
            files=files,
            repo_id=args.repo_id,
            revision=info.sha,
            output=output,
            endpoint=args.endpoint or "https://huggingface.co",
            token=token if isinstance(token, str) else None,
            max_workers=args.max_workers,
        )
        print(f"Direct download complete: {completed}/{len(files)} files present.", flush=True)
        print(output.resolve())
        return

    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=output,
        cache_dir=cache_dir,
        max_workers=args.max_workers,
        token=token,
        endpoint=args.endpoint,
    )
    print(path)


def _download_direct(
    *,
    files: list[str],
    repo_id: str,
    revision: str,
    output: Path,
    endpoint: str,
    token: str | None,
    max_workers: int,
) -> int:
    existing = sum(1 for name in files if (output / name).is_file())
    missing = [name for name in files if not (output / name).is_file()]
    print(
        f"Direct downloader: {existing} existing files, {len(missing)} files to fetch.",
        flush=True,
    )
    if not missing:
        return len(files)

    done = existing
    failures: list[tuple[str, str]] = []
    workers = max(1, max_workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_one_file, repo_id, revision, endpoint, token, output, name): name
            for name in missing
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                final_size = future.result()
            except Exception as exc:  # noqa: BLE001 - keep background log actionable.
                failures.append((name, repr(exc)))
                print(f"FAILED {name}: {exc!r}", flush=True)
            else:
                done += 1
                if done % 25 == 0 or final_size > 512 * 1024 * 1024:
                    print(f"present {done}/{len(files)}: {name} ({final_size} bytes)", flush=True)

    if failures:
        preview = "\n".join(f"{name}: {error}" for name, error in failures[:10])
        raise RuntimeError(f"{len(failures)} files failed during direct download.\n{preview}")
    return done


def _download_one_file(
    repo_id: str,
    revision: str,
    endpoint: str,
    token: str | None,
    output: Path,
    name: str,
) -> int:
    final_path = output / name
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.is_file():
        return final_path.stat().st_size

    part_path = final_path.with_name(f"{final_path.name}.part")
    url = f"{endpoint.rstrip('/')}/datasets/{repo_id}/resolve/{revision}/{quote(name, safe='/')}"
    base_headers = {}
    if token:
        base_headers["Authorization"] = f"Bearer {token}"

    last_error: Exception | None = None
    for attempt in range(1, 9):
        resume_at = part_path.stat().st_size if part_path.exists() else 0
        headers = dict(base_headers)
        mode = "wb"
        if resume_at:
            headers["Range"] = f"bytes={resume_at}-"
            mode = "ab"
        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(30, 180),
                allow_redirects=True,
            ) as response:
                if response.status_code == 416 and resume_at:
                    total = _parse_total_size(response.headers.get("Content-Range"))
                    if total is not None and resume_at >= total:
                        shutil.move(part_path, final_path)
                        return final_path.stat().st_size
                    part_path.unlink(missing_ok=True)
                    continue
                if response.status_code == 200 and resume_at:
                    part_path.unlink(missing_ok=True)
                    resume_at = 0
                    mode = "wb"
                response.raise_for_status()
                if response.status_code not in (200, 206):
                    raise RuntimeError(f"unexpected HTTP status {response.status_code}")

                expected_total = _parse_total_size(response.headers.get("Content-Range"))
                if expected_total is None:
                    content_length = response.headers.get("Content-Length")
                    expected_total = resume_at + int(content_length) if content_length else None

                with part_path.open(mode + "") as handle:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)

                current_size = part_path.stat().st_size
                if expected_total is not None and current_size < expected_total:
                    raise IOError(f"incomplete download: {current_size}/{expected_total} bytes")

                shutil.move(part_path, final_path)
                return final_path.stat().st_size
        except Exception as exc:  # noqa: BLE001 - retry network/download failures.
            last_error = exc
            time.sleep(min(60, 2**attempt))

    raise RuntimeError(f"exhausted retries for {name}: {last_error!r}")


def _parse_total_size(content_range: str | None) -> int | None:
    if not content_range:
        return None
    match = re.search(r"/(\d+)$", content_range)
    return int(match.group(1)) if match else None


if __name__ == "__main__":
    main()
