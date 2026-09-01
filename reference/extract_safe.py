#!/usr/bin/env python3
"""One-time extraction of pinned MotionBricks pickle checkpoints.

Run this only inside the disposable reference container. Normal conversion and
native inference consume the resulting safetensors and JSON and never open a
PyTorch pickle container.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickletools
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import torch
import yaml
from safetensors.torch import save_file


UPSTREAM_REVISION = "a0732b642c0333077e127a2f56ab0014c196bca4"

INPUTS = {
    "clips": (
        "motionbricks/out/G1-clip.ckpt",
        "84afc7c229473351a24b0a7d79fc47be9dbb81bd12774285f2f60a3c0e9028df",
    ),
    "vqvae": (
        "motionbricks/out/motionbricks_vqvae/version_1/checkpoints/model-step=2000000.ckpt",
        "f12a09d46ad390a8e2eecbe7219b2472fcab6b59df0a13f6a40c35cb6da4d99a",
    ),
    "pose": (
        "motionbricks/out/motionbricks_pose/version_1/checkpoints/model-step=2000000.ckpt",
        "0223c352b308ba638a499cc5c92104da36cb1d04cea2f8ce61d54a2489f853f1",
    ),
    "root": (
        "motionbricks/out/motionbricks_root/version_1/checkpoints/model-step=2000000.ckpt",
        "d7299a9b1f5aca35730c36dfe7ea28075708ac8266bf384fe8e2ef9c9aee69c7",
    ),
    "joints": (
        "motionbricks/out/motionbricks_pose/version_1/skeleton/joints.p",
        "8a582b7020d1609a34a9ea5ddfa597c8727b3e726e38cc80ecefe68171f43ecd",
    ),
    "parents": (
        "motionbricks/out/motionbricks_pose/version_1/skeleton/parents.p",
        "4ba0237379480ef33e64b7b8564f1d38dcb3e1ad5bfe01652ea1b7c97ccacb71",
    ),
    "mean": (
        "motionbricks/out/motionbricks_pose/version_1/stats/motion/mean.npy",
        "ca390f0081e2373ab71e860a3546cb70cc11fdfac6ce525155403e047b66fdea",
    ),
    "std": (
        "motionbricks/out/motionbricks_pose/version_1/stats/motion/std.npy",
        "fca7dd6135cbe96504b1307a7db97e9004556937a2fc72d139077b962cee6bd7",
    ),
}

CONFIGS = {
    "vqvae": (
        "motionbricks/out/motionbricks_vqvae/version_1/config.yaml",
        "027a2d7ba5f49cadeadbcc9a6b0c6784d657d5a842e41ff820c5233f1cd6c1f3",
    ),
    "pose": (
        "motionbricks/out/motionbricks_pose/version_1/config.yaml",
        "273af770d328b458510ee6049fac8e38fa486c8792cc27b3515dddc094a7cd1f",
    ),
    "root": (
        "motionbricks/out/motionbricks_root/version_1/config.yaml",
        "b174f03a333f7f7857c3e2b8a32da517caddd198f228efeea2e203d046ce212a",
    ),
}

ALLOWED_PICKLE_GLOBALS = {
    "collections OrderedDict",
    "torch BoolStorage",
    "torch DoubleStorage",
    "torch FloatStorage",
    "torch IntStorage",
    "torch LongStorage",
    "torch._utils _rebuild_tensor_v2",
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


def inspect_pickle_globals(path: Path) -> None:
    with ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if len(names) != 1:
            raise ValueError(f"{path} has {len(names)} data.pkl entries")
        data = archive.read(names[0])
    globals_found = {
        str(argument)
        for opcode, argument, _ in pickletools.genops(data)
        if opcode.name == "GLOBAL"
    }
    unexpected = globals_found - ALLOWED_PICKLE_GLOBALS
    if unexpected:
        raise ValueError(f"unexpected pickle globals in {path}: {sorted(unexpected)}")


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_record(name: str, value: torch.Tensor) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": str(value.dtype).removeprefix("torch."),
        "shape": [int(item) for item in value.shape],
        "numel": int(value.numel()),
        "sha256": hashlib.sha256(tensor_bytes(value)).hexdigest(),
    }


def write_safetensors(path: Path, tensors: dict[str, torch.Tensor], source_hash: str) -> dict[str, Any]:
    if not tensors:
        raise ValueError(f"refusing to write empty tensor set to {path}")
    ordered = {
        name: value.detach().cpu().contiguous().clone()
        for name, value in sorted(tensors.items())
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(
        ordered,
        temporary,
        metadata={
            "format": "motionbricks-safe-v1",
            "upstream_revision": UPSTREAM_REVISION,
            "source_sha256": source_hash,
        },
    )
    os.replace(temporary, path)
    inventory = {
        "path": path.name,
        "sha256": sha256(path),
        "tensor_count": len(ordered),
        "parameter_count": sum(int(value.numel()) for value in ordered.values()),
        "tensors": [tensor_record(name, value) for name, value in ordered.items()],
    }
    del ordered
    return inventory


def load_lightning(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any], str]:
    inspect_pickle_globals(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"{path} is not a Lightning state dictionary")
    state = checkpoint["state_dict"]
    if not isinstance(state, dict) or any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError(f"{path} contains an invalid state_dict")
    metadata = {
        "global_step": plain(checkpoint.get("global_step")),
        "epoch": plain(checkpoint.get("epoch")),
        "pytorch_lightning_version": plain(checkpoint.get("pytorch-lightning_version")),
        "hyper_parameters": plain(checkpoint.get("hyper_parameters", {})),
    }
    return state, metadata, sha256(path)


def load_tensor_mapping(path: Path) -> tuple[dict[str, torch.Tensor], str]:
    inspect_pickle_globals(path)
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or any(not isinstance(item, torch.Tensor) for item in value.values()):
        raise ValueError(f"{path} is not a tensor mapping")
    return value, sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, tuple[Path, str]] = {}
    for key, (relative, expected) in INPUTS.items():
        path = args.upstream_root / relative
        verify(path, expected)
        resolved[key] = (path, expected)
    for relative, expected in CONFIGS.values():
        verify(args.upstream_root / relative, expected)

    manifest: dict[str, Any] = {
        "format": "motionbricks-safe-manifest-v1",
        "upstream_revision": UPSTREAM_REVISION,
        "extractor": {
            "python": os.sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "inputs": {
            key: {"path": relative, "sha256": expected}
            for key, (relative, expected) in INPUTS.items()
        },
        "configs": {},
        "components": {},
        "checkpoint_metadata": {},
    }

    for component in ("vqvae", "pose", "root"):
        path, _ = resolved[component]
        state, checkpoint_metadata, source_hash = load_lightning(path)
        manifest["components"][component] = write_safetensors(
            args.output / f"{component}.safetensors", state, source_hash
        )
        manifest["checkpoint_metadata"][component] = checkpoint_metadata
        del state
        gc.collect()

    clip_path, clip_hash = resolved["clips"]
    clips, _ = load_tensor_mapping(clip_path)
    manifest["components"]["clips"] = write_safetensors(
        args.output / "clips.safetensors", clips, clip_hash
    )
    del clips
    gc.collect()

    joints_path, joints_hash = resolved["joints"]
    parents_path, _ = resolved["parents"]
    inspect_pickle_globals(joints_path)
    inspect_pickle_globals(parents_path)
    joints = torch.load(joints_path, map_location="cpu", weights_only=False)
    parents = torch.load(parents_path, map_location="cpu", weights_only=False)
    mean = torch.from_numpy(np.load(resolved["mean"][0], allow_pickle=False))
    std = torch.from_numpy(np.load(resolved["std"][0], allow_pickle=False))
    support = {
        "neutral_joints": joints,
        "joint_parents": parents,
        "motion_mean": mean,
        "motion_std": std,
    }
    if any(not isinstance(value, torch.Tensor) for value in support.values()):
        raise ValueError("support inputs did not decode to tensors")
    manifest["components"]["support"] = write_safetensors(
        args.output / "support.safetensors", support, joints_hash
    )

    for name, (relative, expected) in CONFIGS.items():
        path = args.upstream_root / relative
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        manifest["configs"][name] = {
            "path": relative,
            "sha256": expected,
            "value": value,
        }

    manifest_path = args.output / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)

    for name, component in manifest["components"].items():
        print(
            f"{name}: {component['tensor_count']} tensors, "
            f"{component['parameter_count']:,} values, {component['sha256']}"
        )


if __name__ == "__main__":
    main()
