from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_summary(text: str) -> dict[str, float | int]:
    summary_pos = text.rfind("EVALUATION SUMMARY")
    if summary_pos >= 0:
        text = text[summary_pos:]

    metrics: dict[str, float | int] = {}
    for line in text.splitlines():
        match = re.match(r"(?P<key>[A-Za-z0-9_]+)\.+:\s+(?P<value>-?\d+(?:\.\d+)?)\s*$", line.strip())
        if not match:
            continue
        key = match.group("key")
        value_raw = match.group("value")
        metrics[key] = int(value_raw) if "." not in value_raw else float(value_raw)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract final MMLU metrics from an evaluation.main log.")
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    metrics = parse_summary(args.log_path.read_text(errors="replace"))
    if not metrics:
        raise SystemExit(f"No metrics found in {args.log_path}")

    payload = {
        "log_path": str(args.log_path),
        "n": metrics.get("n"),
        "acc": metrics.get("acc"),
        "invalid": metrics.get("invalid"),
        "metrics": metrics,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
