# Upstream reference

The native implementation is validated against a pinned, unmodified NVIDIA
MotionBricks checkout.

| Component | Repository | Revision |
|---|---|---|
| MotionBricks | `https://github.com/NVlabs/GR00T-WholeBodyControl.git` | `a0732b642c0333077e127a2f56ab0014c196bca4` |
| GGML | `https://github.com/ggml-org/ggml.git` | `8c63e70982c95ceb862e3a1073a2c1beef75d60a` (`v0.20.2`) |

The `ggml/` submodule is pinned to the same revision used by the local
`kimodo.cpp` and `skin-tokens.cpp` reference ports. Compatibility patches, if
required, belong under `patches/ggml/` and will be applied to a build-directory
copy during CMake configuration; the submodule worktree will remain untouched.

## Checkpoint identities

The required MotionBricks checkpoints and G1 meshes have been fetched through
Git LFS. These identities are the initial trusted reference inputs:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `G1-clip.ckpt` | 7,761,041 | `84afc7c229473351a24b0a7d79fc47be9dbb81bd12774285f2f60a3c0e9028df` |
| VQ-VAE `model-step=2000000.ckpt` | 285,607,244 | `f12a09d46ad390a8e2eecbe7219b2472fcab6b59df0a13f6a40c35cb6da4d99a` |
| pose `model-step=2000000.ckpt` | 1,639,126,476 | `0223c352b308ba638a499cc5c92104da36cb1d04cea2f8ce61d54a2489f853f1` |
| root `model-step=2000000.ckpt` | 409,551,754 | `d7299a9b1f5aca35730c36dfe7ea28075708ac8266bf384fe8e2ef9c9aee69c7` |

Saved configuration identities:

| Configuration | SHA-256 |
|---|---|
| VQ-VAE `config.yaml` | `027a2d7ba5f49cadeadbcc9a6b0c6784d657d5a842e41ff820c5233f1cd6c1f3` |
| pose `config.yaml` | `273af770d328b458510ee6049fac8e38fa486c8792cc27b3515dddc094a7cd1f` |
| root `config.yaml` | `b174f03a333f7f7857c3e2b8a32da517caddd198f228efeea2e203d046ce212a` |

The 29-DOF G1 XML identities used by the initial interactive reference are:

- `g1_29dof.xml`: `58660a6f1d0d33ffd8ee967ab3860def53e3327d956cb009dd1385ddaf430f56`
- `scene_29dof.xml`: `e254f11acce2ec6f6efa5bf9b15e288bbd0ca29aeeabb1e1d0fea92f65436bbf`

Legacy Lightning `.ckpt` and demo-clip files are deserialized only inside the
reference container. Normal conversion accepts safetensors and JSON, never
pickle-bearing checkpoints.

Build and run the trusted extractor from the project root:

```sh
docker build -t motionbricks-reference:torch2.4 reference
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" \
  -v /path/to/GR00T-WholeBodyControl:/upstream:ro \
  motionbricks-reference:torch2.4 \
  --upstream-root /upstream \
  --output /work/generated/safe
```

The extractor verifies every pinned input hash and the small allowlist of
pickle globals before loading. It writes independent safetensors for the three
model checkpoints, original demo clips, and shared skeleton/statistics, plus a
manifest containing every tensor name, shape, dtype, value count, and hash.

## Native inference bundle

The normal converter reads only those verified safetensors and JSON files:

```sh
nix develop --command python scripts/convert_to_gguf.py \
  --safe-directory generated/safe \
  --output generated/g1-f32
```

It emits three inference-weight components plus one small skeleton/statistics
component. The VQ-VAE encoder and optimizer/training state are excluded. The
resulting learned parameter inventory is:

- pose planner: 136,588,272
- root planner: 34,122,833
- VQ pose decoder and codebook: 12,437,277
- total: 183,148,382

Validate the result through the same public model-loading path applications
use:

```sh
./build/debug/bin/motionbricks-cli inspect generated/g1-f32
```

## Reference fixtures and styles

Generate deterministic PyTorch layer fixtures inside the same pinned
container, then package them as small GGUF test inputs:

```sh
docker run --rm --user "$(id -u):$(id -g)" \
  --entrypoint python \
  -v "$PWD:/work" \
  -v /path/to/GR00T-WholeBodyControl:/upstream:ro \
  motionbricks-reference:torch2.4 \
  /work/reference/generate_fixtures.py \
  --upstream-root /upstream \
  --safe-directory /work/generated/safe \
  --output /work/generated/fixtures

python scripts/convert_fixtures_to_gguf.py \
  --fixtures generated/fixtures \
  --output generated/fixtures-gguf

python scripts/convert_styles.py \
  --safe-directory generated/safe \
  --output generated/styles
```

The native suite checks every released neural component against these actual
upstream PyTorch forwards. On the current reference machine, complete
style-to-animation output also matches between CPU and strict-F32 Vulkan with
the same 44-frame duration; observed maximum absolute differences were
`2.19e-5` for root translations and `1.02e-4` for local quaternion components.
