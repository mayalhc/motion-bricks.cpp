#!/usr/bin/env python3
"""Generate deterministic PyTorch fixtures from the pinned upstream modules.

Run only in the reference container.  This script imports NVIDIA's actual
network classes, loads the verified safetensors intermediates, and records
inputs, selected layer outputs, and final outputs without opening a pickle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from safetensors import safe_open


UPSTREAM_REVISION = "a0732b642c0333077e127a2f56ab0014c196bca4"
SAFE_HASHES = {
    "pose": "01327768be7413111dc927947a95cfb3e9ee7c52acfa35a054e8c2f3b838b888",
    "root": "d1529a8c9da915cb7dd0499272baba61db480660c0aae92b67fbe69828e83c5a",
    "vqvae": "544782e605ed96d60bf999243ef8f44640ea021a5655ef20af2d2853c3e18b5b",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual}, expected {expected}")


class Rep:
    pass


def make_motion_rep() -> Rep:
    """Minimal shape-compatible rep used only by upstream constructors."""
    global_rep, local_rep, combined = Rep(), Rep(), Rep()
    global_rep.root_mode = "global"
    global_rep.indices = {
        "all": np.arange(418),
        "root": np.arange(5),
        "global_root_pos": np.arange(3),
        "global_root_pos_2d": np.arange(2),
        "global_root_heading": np.arange(2, 4),
        "ric_data": np.arange(99),
        "global_rot_data": np.arange(99, 303),
    }
    local_rep.root_mode = "local"
    local_rep.indices = {
        "all": np.arange(413),
        "root": np.arange(4),
        "local_root_rot_vel": np.arange(2),
        "local_root_vel": np.arange(2, 4),
        "global_root_y": np.arange(1),
        "ric_data": np.arange(99),
        "global_rot_data": np.arange(99, 303),
    }
    dual = SimpleNamespace(global_motion_rep=global_rep, local_motion_rep=local_rep)
    combined.dual_rep = dual
    combined.root_mode = "global"
    combined.indices = global_rep.indices
    return combined


def state(path: Path, prefix: str, select=lambda _name: True) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as source:
        for name in source.keys():
            if name.startswith(prefix) and select(name):
                result[name.removeprefix(prefix)] = source.get_tensor(name)
    return result


def ramp(shape: tuple[int, ...], scale: float, phase: float = 0.0) -> torch.Tensor:
    count = int(np.prod(shape))
    values = torch.arange(count, dtype=torch.float32)
    return (torch.sin(values * 0.017 + phase) * scale).reshape(shape)


def trace_modules(model: torch.nn.Module, output: dict[str, torch.Tensor]) -> list[Any]:
    handles = []
    traced_types = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.TransformerEncoderLayer)
    for name, module in model.named_modules():
        if not name or not isinstance(module, traced_types):
            continue

        def capture(_module, _inputs, value, trace_name=name):
            if isinstance(value, torch.Tensor):
                if value.is_nested:
                    value = value.to_padded_tensor(0.0)
                output[f"trace.{trace_name}"] = value.detach().cpu().contiguous()

        handles.append(module.register_forward_hook(capture))
    return handles


def pose_fixture(pose_type, safe: Path) -> dict[str, torch.Tensor]:
    args = {
        "cond_root_feature": "root_without_hip_height",
        "cond_root_feature_is_from_motion_rep": "global",
        "down_t": 2,
        "local_pose_feature": "joint_positions_and_rotations_and_hip_height",
        "max_tokens": 16,
        "min_tokens": 6,
        "n_embd": 1024,
        "n_head": 16,
        "n_layers": 16,
        "pose_feat_width": 640,
        "pose_root_mode": "pose",
        "pose_token_mlp_num_layers": 2,
        "pose_vqvae": {"code_dim": 256, "has_codebook": True, "nb_code": 100_000_000, "num_heads": 8},
        "root_feat_width": 256,
        "root_vqvae": {"code_dim": 60, "nb_code": 1024, "num_heads": 1},
        "text_emb_dim": 4096,
        "text_embeddings": None,
        "token_length_feat_width": 128,
    }
    model = pose_type(make_motion_rep(), args).eval()
    missing, unexpected = model.load_state_dict(state(safe / "pose.safetensors", "backbone_net."), strict=True)
    if missing or unexpected:
        raise ValueError(f"pose state mismatch: missing={missing}, unexpected={unexpected}")

    result: dict[str, torch.Tensor] = {}
    pose_tokens = (torch.arange(48).reshape(1, 6, 8) * 7 % 11).to(torch.int64)
    local_root = ramp((1, 24, 4), 0.35, 0.1)
    pose_cond = ramp((1, 24, 304), 0.2, 0.7)
    has_pose = torch.zeros((1, 24), dtype=torch.bool)
    has_pose[:, [0, 1, 22, 23]] = True
    num_tokens = torch.tensor([[6]], dtype=torch.int64)
    result.update({
        "input.pose_tokens": pose_tokens,
        "input.local_root_values": local_root,
        "input.pose_cond": pose_cond,
        "input.has_pose_cond": has_pose,
        "input.num_tokens": num_tokens,
    })
    handles = trace_modules(model, result)
    with torch.inference_mode():
        output = model(pose_tokens, local_root, pose_cond, has_pose, num_tokens)
    result["output.pose_logits"] = output["pose_logits"].contiguous()
    for handle in handles:
        handle.remove()
    return result


def root_fixture(root_type, safe: Path) -> dict[str, torch.Tensor]:
    args = {
        "activation": "relu", "depth": 4, "dilation_growth_rate": 3, "down_t": 2,
        "global_root_feat_dim": 64, "global_root_feature": "root",
        "input_feat_mlp_num_layers": 2,
        "local_pose_feature": "joint_positions_and_rotations_and_hip_height",
        "local_root_feat_dim": 64, "local_root_feature": "root",
        "max_tokens": 16, "min_tokens": 6, "n_embd": 512, "n_head": 16,
        "n_layers_root_token": 3, "n_layers_shared": 3, "norm": "None",
        "pose_feat_dim": 256,
        "pose_vqvae": {"code_dim": 60, "nb_code": 1024, "num_heads": 1},
        "root_vqvae": {"code_dim": 60, "nb_code": 1024, "num_heads": 1},
        "text_emb_dim": 4096, "text_embeddings": None,
        "use_hard_num_token_emb_for_root_prediction": True, "width": 512,
    }
    model = root_type(args, make_motion_rep()).eval()
    missing, unexpected = model.load_state_dict(state(safe / "root.safetensors", "backbone_net."), strict=True)
    if missing or unexpected:
        raise ValueError(f"root state mismatch: missing={missing}, unexpected={unexpected}")

    result: dict[str, torch.Tensor] = {}
    global_root = ramp((1, 8, 5), 0.3, 0.2)
    local_root = ramp((1, 8, 4), 0.25, 0.9)
    poses = ramp((1, 8, 304), 0.15, 1.7)
    has_global = torch.tensor([[1, 1, 1, 1, 0, 0, 1, 1]], dtype=torch.bool)
    has_local = torch.tensor([[1, 1, 1, 1, 0, 0, 1, 1]], dtype=torch.bool)
    has_poses = torch.tensor([[1, 1, 1, 1, 0, 0, 1, 1]], dtype=torch.bool)
    num_tokens = torch.tensor([[6]], dtype=torch.int64)
    result.update({
        "input.global_root_values": global_root, "input.has_global_root_values": has_global,
        "input.local_root_values": local_root, "input.has_local_root_values": has_local,
        "input.poses": poses, "input.has_poses": has_poses, "input.num_tokens": num_tokens,
    })
    handles = trace_modules(model, result)
    with torch.inference_mode():
        output = model(global_root, has_global, local_root, has_local, poses, has_poses, num_tokens)
    result["output.num_token_logits"] = output["num_token_logits"].contiguous()
    result["output.pred_num_tokens"] = output["pred_num_tokens"].contiguous()
    result["output.pred_global_root_values"] = output["pred_global_root_values"].contiguous()
    for handle in handles:
        handle.remove()
    return result


def decoder_fixture(decoder_type, safe: Path) -> dict[str, torch.Tensor]:
    model = decoder_type(
        input_emb_width=413, output_emb_width=256, down_t=2, width=512, depth=4,
        dilation_growth_rate=3, activation="relu", norm="None",
        target_cond_dim=304, external_cond_dim=2, cond_fusion_last_layer=False,
    ).eval()
    decoder_state = state(safe / "vqvae.safetensors", "pose_net.decoder.")
    missing, unexpected = model.load_state_dict(decoder_state, strict=True)
    if missing or unexpected:
        raise ValueError(f"decoder state mismatch: missing={missing}, unexpected={unexpected}")
    codebook = state(
        safe / "vqvae.safetensors", "pose_net.",
        lambda name: name == "pose_net.quantizer.vq._codebook.embed",
    )["quantizer.vq._codebook.embed"]

    result: dict[str, torch.Tensor] = {}
    indices = (torch.arange(48).reshape(1, 6, 8) * 3 + 1).remainder(10).to(torch.int64)
    batch_index = torch.arange(indices.shape[0])[:, None, None]
    head_index = torch.arange(8)[None, None, :]
    quantized = codebook[head_index, indices].reshape(1, 6, 256).transpose(1, 2).contiguous()
    target = ramp((1, 24, 304), 0.18, 0.4)
    has_target = torch.zeros((1, 24), dtype=torch.bool)
    has_target[:, [0, 1, 22, 23]] = True
    external = ramp((1, 24, 2), 0.22, 1.1)
    token_mask = torch.ones((1, 6), dtype=torch.bool)
    result.update({
        "input.indices": indices, "input.quantized": quantized,
        "input.target_cond": target, "input.has_target_cond": has_target,
        "input.external_cond": external, "input.token_mask": token_mask,
    })
    handles = trace_modules(model, result)
    with torch.inference_mode():
        result["output.motion"] = model(
            quantized, external_cond=external, target_cond=target,
            has_target_cond=has_target, token_mask=token_mask,
        ).transpose(1, 2).contiguous()
    for handle in handles:
        handle.remove()
    return result


def tensor_record(name: str, value: torch.Tensor) -> dict[str, Any]:
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "name": name, "dtype": str(value.dtype).removeprefix("torch."),
        "shape": list(value.shape), "numel": value.numel(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def write_fixture(path: Path, tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    ordered = {name: value.detach().cpu().contiguous().clone() for name, value in sorted(tensors.items())}
    dtype_names = {
        torch.float32: "F32", torch.int64: "I64", torch.int32: "I32", torch.bool: "BOOL",
    }
    header: dict[str, Any] = {
        "__metadata__": {
            "format": "motionbricks-reference-fixture-v1",
            "torch": torch.__version__,
            "upstream_revision": UPSTREAM_REVISION,
        }
    }
    payloads: list[bytes] = []
    offset = 0
    for name, value in ordered.items():
        if value.dtype not in dtype_names:
            raise ValueError(f"unsupported fixture dtype for {name}: {value.dtype}")
        payload = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        header[name] = {
            "dtype": dtype_names[value.dtype], "shape": list(value.shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        payloads.append(payload)
        offset += len(payload)
    encoded_header = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded_header += b" " * (-len(encoded_header) % 8)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded_header)))
        stream.write(encoded_header)
        for payload in payloads:
            stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return {
        "path": path.name, "sha256": sha256(path), "tensor_count": len(ordered),
        "tensors": [tensor_record(name, value) for name, value in ordered.items()],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--safe-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not Path("/.dockerenv").exists():
        raise RuntimeError("reference fixtures must be generated inside the container")
    for name, expected in SAFE_HASHES.items():
        verify(args.safe_directory / f"{name}.safetensors", expected)

    sys.path.insert(0, str(args.upstream_root / "motionbricks"))
    from motionbricks.motion_backbone.neural_modules.pose_backbone import pose_backbone_network
    from motionbricks.motion_backbone.neural_modules.root_backbone import root_backbone_network
    from motionbricks.vqvae.neural_modules.encdec_double_cond import DoubleCondDecoder

    torch.manual_seed(7)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    generated = {
        "pose": write_fixture(args.output / "pose.safetensors", pose_fixture(pose_backbone_network, args.safe_directory)),
        "root": write_fixture(args.output / "root.safetensors", root_fixture(root_backbone_network, args.safe_directory)),
        "vq-decoder": write_fixture(
            args.output / "vq-decoder.safetensors", decoder_fixture(DoubleCondDecoder, args.safe_directory)
        ),
    }
    manifest = {
        "format": "motionbricks-reference-fixtures-v1",
        "upstream_revision": UPSTREAM_REVISION,
        "torch": torch.__version__, "numpy": np.__version__, "fixtures": generated,
    }
    temporary = args.output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output / "manifest.json")
    for name, record in generated.items():
        print(f"{name}: {record['tensor_count']} tensors, {record['sha256']}")


if __name__ == "__main__":
    main()
