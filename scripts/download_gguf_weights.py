#!/usr/bin/env python3
"""Download and verify the published MotionBricks GGUF/style distribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "LocalAI-io/MotionBricks-G1-GGML"
DEFAULT_REVISION = "cc2a47603dbc203a4f18f35dd06ed3611833f506"
DEFAULT_MANIFEST = ROOT / "scripts/hf/MotionBricks-G1-GGML/MANIFEST.json"
FORMAT = "motionbricks-gguf-distribution-v1"
UPSTREAM_REVISION = "a0732b642c0333077e127a2f56ab0014c196bca4"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def valid(path: Path, entry: dict[str, object]) -> bool:
    return (path.is_file() and path.stat().st_size == entry["bytes"]
            and digest(path) == entry["sha256"])


def safe_relative(value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe distribution path: {value!r}")
    if path.parts[0] not in {"g1-f32", "styles"}:
        raise ValueError(f"unexpected distribution path: {value!r}")
    if path.suffix not in {".gguf", ".mbstyle", ".json"}:
        raise ValueError(f"unexpected distribution file type: {value!r}")
    return path


def load_manifest(path: Path, repo: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format") != FORMAT:
        raise ValueError(f"unsupported manifest format in {path}")
    if value.get("repository") != repo:
        raise ValueError(f"manifest is for {value.get('repository')}, not {repo}")
    if value.get("upstream_revision") != UPSTREAM_REVISION:
        raise ValueError("manifest refers to an unexpected upstream revision")
    files = value.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("manifest has no files")
    for entry in files:
        if (not isinstance(entry, dict) or not isinstance(entry.get("bytes"), int)
                or entry["bytes"] <= 0 or not isinstance(entry.get("sha256"), str)
                or len(entry["sha256"]) != 64):
            raise ValueError("manifest contains a malformed file record")
        safe_relative(entry.get("path"))
    return value


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "motion-bricks.cpp/0.1"})
    try:
        with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=8 << 20)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "generated",
                        help="destination containing g1-f32/ and styles/")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face model repository")
    parser.add_argument("--revision", default=DEFAULT_REVISION,
                        help="pinned Hugging Face commit or revision")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="version-controlled distribution manifest")
    args = parser.parse_args()

    try:
        manifest = load_manifest(args.manifest, args.repo)
        quoted_repo = urllib.parse.quote(args.repo, safe="/")
        quoted_revision = urllib.parse.quote(args.revision, safe="")
        base = f"https://huggingface.co/{quoted_repo}/resolve/{quoted_revision}"
        for entry in manifest["files"]:
            relative = safe_relative(entry["path"])
            destination = args.output / relative
            if valid(destination, entry):
                print(f"verified {destination}")
                continue
            print(f"downloading {relative} from {args.repo}@{args.revision}")
            download(f"{base}/{urllib.parse.quote(relative.as_posix(), safe='/')}", destination)
            if not valid(destination, entry):
                raise ValueError(f"downloaded file failed size/SHA-256 verification: {destination}")
    except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("MotionBricks GGUF and style bundles are present and verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
