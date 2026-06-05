from __future__ import annotations

import argparse
import json
import sys
from math import prod
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.baselines.hrm_hybrid_rwkv7_nocarry_bp_warmup import HierarchicalHybridRWKV7Model
from models.lm_head import LMHead


def map_hf_key_to_hybrid_h(key: str) -> str | None:
    if key == "lm_head.weight":
        return "lm_head.weight"
    if key == "model.embed_tokens.weight":
        return "embed_tokens.embedding_weight"
    if key == "model.z_L_init":
        return "model.zL_init"
    if key.startswith("model.L_module."):
        return "model.L_level.core." + key.removeprefix("model.L_module.")
    return None


def target_config_from_hf(hf_config: dict[str, Any], rwkv7_expansion: float) -> dict[str, Any]:
    hidden_size = int(hf_config["hidden_size"])
    intermediate_size = int(hf_config["intermediate_size"])
    transformer_expansion = intermediate_size * 3.0 / (hidden_size * 2.0)
    return {
        "max_seq_len": int(hf_config["max_position_embeddings"]),
        "n_layers": int(hf_config["num_hidden_layers"]) * 2,
        "hidden_size": hidden_size,
        "num_heads": int(hf_config["num_attention_heads"]),
        "expansion": transformer_expansion,
        "transformer_expansion": transformer_expansion,
        "rwkv7_expansion": rwkv7_expansion,
        "norm_type": "pre",
        "norm_eps": float(hf_config["rms_norm_eps"]),
        "rope_theta": float(hf_config["rope_theta"]),
        "pos_emb_type": "rope",
        "init_type": "lecun_normal",
        "half_layers": True,
        "H_cycles": int(hf_config["H_cycles"]),
        "L_cycles": int(hf_config["L_cycles"]),
        "H_arch": "rwkv7",
        "L_arch": "transformer",
        "H_override": {},
        "L_override": {},
        "bp_warmup_ratio": 0.2,
        "bp_min_steps": 2,
        "bp_max_steps": 5,
        "rwkv7_head_size": 64,
        "rwkv7_backend": "cuda",
        "rwkv7_chunk_len": 16,
        "rwkv7_enable_v_first_mix": True,
        "vocab_size": int(hf_config["vocab_size"]),
        "target_only": True,
    }


def build_target_state_shapes(target_config: dict[str, Any]) -> tuple[dict[str, tuple[int, ...]], int]:
    with torch.device("meta"):
        model = LMHead(HierarchicalHybridRWKV7Model(target_config), target_config)
    state = model.state_dict()
    return {key: tuple(value.shape) for key, value in state.items()}, sum(value.numel() for value in state.values())


def numel(shape: tuple[int, ...] | list[int]) -> int:
    return int(prod(shape))


def convert(args: argparse.Namespace) -> dict[str, Any]:
    source_dir = Path(args.source_dir)
    source_path = source_dir / "model.safetensors"
    config_path = source_dir / "config.json"
    output_path = Path(args.output)
    report_path = Path(args.report) if args.report else output_path.with_suffix(".report.json")

    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if output_path.exists() and not args.force:
        raise FileExistsError(f"{output_path} already exists; pass --force to overwrite")

    hf_config = json.loads(config_path.read_text())
    target_config = target_config_from_hf(hf_config, args.rwkv7_expansion)
    target_shapes, target_total_params = build_target_state_shapes(target_config)

    copied: dict[str, torch.Tensor] = {}
    copied_records = []
    skipped_records = []
    mismatch_records = []
    source_total_params = 0

    with safe_open(source_path, framework="pt", device="cpu") as source:
        for source_key in source.keys():
            source_slice = source.get_slice(source_key)
            source_shape = tuple(source_slice.get_shape())
            source_params = numel(source_shape)
            source_total_params += source_params
            target_key = map_hf_key_to_hybrid_h(source_key)
            if target_key is None:
                skipped_records.append(
                    {
                        "source_key": source_key,
                        "shape": list(source_shape),
                        "params": source_params,
                        "reason": "no direct mapping to H-RWKV target",
                    }
                )
                continue

            target_shape = target_shapes.get(target_key)
            if target_shape != source_shape:
                mismatch_records.append(
                    {
                        "source_key": source_key,
                        "target_key": target_key,
                        "source_shape": list(source_shape),
                        "target_shape": None if target_shape is None else list(target_shape),
                    }
                )
                continue

            copied[target_key] = source.get_tensor(source_key)
            copied_records.append(
                {
                    "source_key": source_key,
                    "target_key": target_key,
                    "shape": list(source_shape),
                    "params": source_params,
                }
            )

    if mismatch_records:
        raise RuntimeError(f"Found {len(mismatch_records)} shape/key mismatches; first={mismatch_records[0]}")

    missing_target_keys = sorted(set(target_shapes) - set(copied))
    report = {
        "source_dir": str(source_dir),
        "source_path": str(source_path),
        "output_path": str(output_path),
        "target": "hrm_h_rwkv7",
        "hf_config": hf_config,
        "target_config": target_config,
        "source_total_params": source_total_params,
        "target_total_params": target_total_params,
        "copied_tensor_count": len(copied_records),
        "copied_param_count": sum(record["params"] for record in copied_records),
        "skipped_tensor_count": len(skipped_records),
        "skipped_param_count": sum(record["params"] for record in skipped_records),
        "missing_target_tensor_count": len(missing_target_keys),
        "missing_target_param_count": sum(numel(target_shapes[key]) for key in missing_target_keys),
        "copied": copied_records,
        "skipped": skipped_records,
        "missing_target_keys": missing_target_keys,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(copied, str(output_path), metadata={"format": "pt", "source": "sapientinc/HRM-Text-1B"})
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HF HRM-Text weights into a partial hrm_h_rwkv7 initializer.")
    parser.add_argument("--source-dir", default="/run/media/xiaol/B214449214445C0B/hf_models/sapientinc/HRM-Text-1B")
    parser.add_argument("--output", default="/run/media/xiaol/B214449214445C0B/hrm_text_migrations/hrm_text_1b_to_hrm_h_rwkv7_init.safetensors")
    parser.add_argument("--report", default="")
    parser.add_argument("--rwkv7-expansion", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    report = convert(args)
    print(
        "converted "
        f"copied_tensors={report['copied_tensor_count']} "
        f"copied_params={report['copied_param_count']} "
        f"skipped_tensors={report['skipped_tensor_count']} "
        f"missing_target_tensors={report['missing_target_tensor_count']} "
        f"output={report['output_path']}"
    )


if __name__ == "__main__":
    main()
