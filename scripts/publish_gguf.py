#!/usr/bin/env python3
"""Validate and optionally publish the MotionBricks GGUF distribution to HF."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
HF_REPO = "LocalAI-io/MotionBricks-G1-GGML"
UPSTREAM_REPO = "NVlabs/GR00T-WholeBodyControl"
UPSTREAM_REVISION = "a0732b642c0333077e127a2f56ab0014c196bca4"
FORMAT = "motionbricks-gguf-distribution-v1"
CARD_DIR = ROOT / "scripts/hf/MotionBricks-G1-GGML"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def checked_json(path: Path, expected_format: str) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"missing manifest: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format") != expected_format:
        raise ValueError(f"unexpected format in {path}")
    return value


def artifacts(model: Path, styles: Path) -> list[tuple[Path, str]]:
    model_manifest = checked_json(model / "manifest.json", "motionbricks-gguf-bundle-v1")
    style_manifest = checked_json(styles / "manifest.json", "motionbricks-style-bundle-v1")
    if model_manifest.get("upstream_revision") != UPSTREAM_REVISION:
        raise ValueError("model bundle came from an unexpected upstream revision")
    result = [(model / "manifest.json", "g1-f32/manifest.json")]
    for record in model_manifest["components"].values():
        source = model / record["path"]
        if digest(source) != record["sha256"]:
            raise ValueError(f"model manifest SHA-256 mismatch: {source}")
        result.append((source, f"g1-f32/{source.name}"))
    result.append((styles / "manifest.json", "styles/manifest.json"))
    for record in style_manifest["styles"].values():
        source = styles / record["path"]
        if digest(source) != record["sha256"]:
            raise ValueError(f"style manifest SHA-256 mismatch: {source}")
        result.append((source, f"styles/{source.name}"))
    for source, destination in result:
        if not source.is_file() or source.stat().st_size == 0:
            raise ValueError(f"missing or empty artifact: {source}")
        if source.suffix in {".gguf", ".mbstyle"}:
            with source.open("rb") as stream:
                if stream.read(4) != b"GGUF":
                    raise ValueError(f"invalid GGUF magic: {source}")
        path = Path(destination)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe destination: {destination}")
    return sorted(result, key=lambda item: item[1])


def distribution_manifest(repo: str, files: list[tuple[Path, str]]) -> tuple[dict[str, object], bytes]:
    records = [
        {"path": destination, "bytes": source.stat().st_size, "sha256": digest(source)}
        for source, destination in files
    ]
    value = {
        "format": FORMAT,
        "repository": repo,
        "upstream_repository": UPSTREAM_REPO,
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_license": "NVIDIA Open Model License",
        "files": records,
    }
    payload = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    return value, payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "generated/g1-f32")
    parser.add_argument("--styles", type=Path, default=ROOT / "generated/styles")
    parser.add_argument("--repo", default=HF_REPO)
    parser.add_argument("--write-manifest", type=Path,
                        help="write the checked distribution manifest to this path")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--confirm-upstream-licence", action="store_true",
                        help="required with --upload; confirms redistribution conditions were reviewed")
    args = parser.parse_args()

    try:
        for required in (CARD_DIR / "README.md", CARD_DIR / "NOTICE", CARD_DIR / "UPSTREAM_LICENSE"):
            if not required.is_file():
                raise ValueError(f"missing publication asset: {required}")
        files = artifacts(args.model, args.styles)
        manifest, manifest_payload = distribution_manifest(args.repo, files)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    sums = "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in manifest["files"])
    total = sum(entry["bytes"] for entry in manifest["files"])
    print(f"repo:  https://huggingface.co/{args.repo}")
    print(f"files: {len(files)} model/style artifacts, {total / 1e9:.3f} GB")
    for entry in manifest["files"]:
        print(f"  {entry['sha256']}  {entry['bytes']:>12}  {entry['path']}")
    if args.write_manifest:
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_bytes(manifest_payload)
        print(f"wrote {args.write_manifest}")
    if not args.upload:
        print("[dry-run] nothing uploaded")
        return 0
    if not args.confirm_upstream_licence:
        print("error: --upload requires --confirm-upstream-licence", file=sys.stderr)
        return 2

    try:
        from huggingface_hub import CommitOperationAdd, HfApi

        api = HfApi()
        api.create_repo(args.repo, repo_type="model", exist_ok=True)
        operations = [
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=CARD_DIR / "README.md"),
            CommitOperationAdd(path_in_repo="NOTICE", path_or_fileobj=CARD_DIR / "NOTICE"),
            CommitOperationAdd(path_in_repo="UPSTREAM_LICENSE", path_or_fileobj=CARD_DIR / "UPSTREAM_LICENSE"),
            *(CommitOperationAdd(path_in_repo=destination, path_or_fileobj=source)
              for source, destination in files),
            CommitOperationAdd(path_in_repo="MANIFEST.json", path_or_fileobj=io.BytesIO(manifest_payload)),
            CommitOperationAdd(path_in_repo="SHA256SUMS", path_or_fileobj=io.BytesIO(sums.encode())),
        ]
        commit = api.create_commit(
            repo_id=args.repo, repo_type="model", operations=operations,
            commit_message=f"Publish MotionBricks G1 GGUF bundle from {UPSTREAM_REVISION[:12]}",
        )
    except Exception as error:  # huggingface_hub supplies the useful API detail
        print(f"error: Hugging Face upload failed: {error}", file=sys.stderr)
        return 1
    print(f"published: {commit.commit_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
