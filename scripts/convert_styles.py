#!/usr/bin/env python3
"""Convert the verified upstream G1 exemplar clips into checked style GGUFs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open


CLIP_SHA256 = "7ea40c20130c8a9799d83c0438750b51a5be9b1e09357c4928887961fc992d33"
STYLES = (
    ("idle", 0.0, 6), ("slow_walk", 0.3, 6), ("walk", 1.0, 6),
    ("hand_crawling", 0.5, 6), ("walk_boxing", 1.0, 11),
    ("elbow_crawling", 0.8, 11), ("stealth_walk", 1.0, 6),
    ("injured_walk", 0.5, 6), ("walk_stealth", 0.7, 6),
    ("walk_happy_dance", 1.0, 6), ("walk_zombie", 0.6, 6),
    ("walk_gun", 0.6, 6), ("walk_scared", 0.6, 6),
    ("walk_left", 0.2, 4), ("walk_right", 0.2, 4),
)


def converter_module(path: Path):
    spec = importlib.util.spec_from_file_location("motionbricks_converter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    converter = converter_module(Path(__file__).with_name("convert_to_gguf.py"))
    source = args.safe_directory / "clips.safetensors"
    if converter.sha256(source) != CLIP_SHA256:
        raise ValueError("upstream clip safetensors hash mismatch")
    with safe_open(source, framework="numpy") as clips:
        positions = clips.get_tensor("global_joint_positions")
        rotations = clips.get_tensor("global_joint_rotations")
        roots = clips.get_tensor("global_root_positions")
        headings = clips.get_tensor("global_headings")
        lengths = clips.get_tensor("num_frames_per_clip")
    headings = headings.copy()
    headings[3, 0] = 0.0
    headings[5] -= 0.95
    args.output.mkdir(parents=True, exist_ok=True)
    records = {}
    for index, (name, speed, allowed_count) in enumerate(STYLES):
        frames = int(lengths[index])
        allowed = np.zeros(11, dtype=np.int32)
        allowed[:allowed_count] = 1
        values = [
            ("global_joint_positions", positions[index, :frames]),
            ("global_joint_rotations", rotations[index, :frames].reshape(frames, 34, 9)),
            ("global_root_positions", roots[index, :frames]),
            ("global_headings", headings[index, :frames]),
            ("allowed_tokens", allowed),
        ]
        records[name] = converter.write_component(
            args.output / f"{name}.mbstyle", "style", values, CLIP_SHA256,
            general_name=name,
            extra_metadata=[
                converter.kv_string("motionbricks.style_name", name),
                converter.kv_f32("motionbricks.style_speed", speed),
                converter.kv_u32("motionbricks.style_frames", frames),
                converter.kv_u32("motionbricks.style_index", index),
            ],
        )
    (args.output / "manifest.json").write_text(
        json.dumps({"format": "motionbricks-style-bundle-v1", "styles": records},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
