# Versioned format registry

A reader must reject an unknown major version rather than guessing.

| Format | Version | Status |
|---|---:|---|
| Installed C ABI | 1 | Model, style, command, agent, and motion contract implemented |
| GGUF model-bundle manifest | 1 | F32 inference components implemented |
| MotionBricks style asset | 1 | Upstream G1 exemplar schema implemented |
| Reference fixture manifest | 1 | Root, pose, and decoder fixtures implemented |
| WebSocket animation protocol | 1 | Reserved; schema not implemented |

Minor, backwards-compatible additions are represented by optional keys or by
new API functions. Existing function signatures and field meanings do not
change within one C ABI version.

## GGUF model bundle v1

A bundle is a directory containing `manifest.json` and four GGUF v3 files:

| Component | Runtime contents | Tensors | Learned parameters |
|---|---|---:|---:|
| `pose.gguf` | Pose-token planner | 209 | 136,588,272 |
| `root.gguf` | Root-trajectory and duration planner | 150 | 34,122,833 |
| `vq-decoder.gguf` | Pose codebook and convolutional decoder | 51 | 12,437,277 |
| `support.gguf` | G1 skeleton, parents, mean, and standard deviation | 4 | 0 |

The learned inference total is 183,148,382 F32 parameters. The support file
contains 972 non-learned scalar values. Training-only VQ encoder tensors,
codebook EMA state, and initialization flags are deliberately omitted.

Every component carries `general.architecture=motionbricks`, format version,
component role, `g1skel34` skeleton identity, pinned upstream revision, source
safetensors hash, and exact scalar count. PyTorch tensor dimensions are stored
in reversed GGML order. Names at or above GGML's 64-byte limit are compacted
deterministically and collision-checked by the converter.

## MotionBricks style asset v1

A `.mbstyle` is a GGUF v3 file with `component=style`, the `g1skel34`
skeleton identity, a source SHA-256, name, configured speed, and frame count.
It contains checked F32 tensors for global joint positions
`[frames,34,3]`, flattened global rotation matrices `[frames,34,9]`, root
positions `[frames,3]`, and headings `[frames]`, plus an I32 allowed-duration
mask `[11]` corresponding to 6--16 tokens. The runtime validates all metadata,
shapes, finite values, frame limits, and the binary mask before accepting it.

The source identity is not restricted to NVIDIA's clip archive, so the same
schema can hold a future checked Kimodo/G1 conversion. Skeleton and coordinate
compatibility must still be established by the converter.
