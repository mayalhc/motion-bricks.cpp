#!/usr/bin/env python3
"""Package deterministic safetensors reference fixtures as GGUF v3."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from safetensors import safe_open


def load_converter(path: Path):
    spec = importlib.util.spec_from_file_location("motionbricks_gguf_converter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    converter = load_converter(Path(__file__).with_name("convert_to_gguf.py"))
    args.output.mkdir(parents=True, exist_ok=True)
    for fixture in ("pose", "root", "vq-decoder"):
        source_path = args.fixtures / f"{fixture}.safetensors"
        values = []
        with safe_open(source_path, framework="numpy") as source:
            for name in source.keys():
                values.append((name, source.get_tensor(name)))
        converter.write_component(
            args.output / f"{fixture}.gguf", f"fixture-{fixture}", values,
            converter.sha256(source_path),
        )


if __name__ == "__main__":
    main()
