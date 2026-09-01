# motion-bricks.cpp implementation sketch and plan

This document proposes an implementation derived from `motions-bricks.md` and
the checked-out NVIDIA MotionBricks source. It does not replace the human-led
design document.

## Goal and first supported slice

`motion-bricks.cpp` will be a C++23/GGML inference port of the released G1
MotionBricks planner. It will accept a recent G1 motion context plus a
movement/facing command and a short reference-style clip, then generate the
next continuous section of 30 FPS skeletal animation. The initial product
slice is:

- batch size one;
- the released 34-joint G1 model only;
- the same 4-frame context and 4-frame target-pose convention as upstream;
- 24--64 generated frames (6--16 tokens, 4 frames per token);
- F32 CPU inference with reference parity;
- root translations and 34 local joint quaternions as the public animation
  result;
- a Go/PureGo server and a plain HTML/CSS/Three.js interactive viewer.

Vulkan follows CPU parity. F16 and quantization are optimizations after both
backends pass the F32 reference suite.

Kimodo is an authoring input, not a runtime dependency of the MotionBricks
neural network. A Kimodo G1 animation supplies characteristic poses for a
named style. The controller positions and rotates those poses in the world,
and MotionBricks generates the transition from the character's actual current
motion. The original upstream demo clips use the same style-asset boundary.

## Current implementation status

The native inference slice described above is operational. The strict bundle
loader, all three released neural graphs, normalization and motion
representation conversion, original G1 style assets, spring-based target
construction, stateful agent, and public skeletal outputs are implemented.
The deterministic component fixtures pass against upstream PyTorch on CPU;
CPU and Vulkan preserve duration and pose-token choices and agree end to end
within small F32 tolerances.

Still outstanding from the wider plan are direct Kimodo GLB style import,
recorded upstream controller/session fixtures, context-output blending,
Gumbel sampling, graph/buffer caching and performance work, the Go/Three.js
demo, fuzzing, and optimized weight formats. The human-led
`motions-bricks.md` remains unchanged.

## System shape

```text
                      offline / authoring

 upstream .ckpt + config + stats        Kimodo G1 GLB / original demo clips
          |                                           |
 trusted reference container                         |
          |                                           |
    safetensors + manifest                 checked style conversion
          |                                           |
          +---------------> GGUF model bundle <-------+

                         runtime

 keyboard/game input
          |
 Go session controller -- movement, facing, speed, selected style
          |
 PureGo -> stable C ABI -> stateful C++ agent
                              |
          recent 4 frames -> target construction and normalization
                              |
                    root transformer
                              |
          predicted root path and predicted duration
                              |
                    pose transformer
                              |
                    VQ pose decoder
                              |
                 representation inverse / FK
                              |
             root translations + local joint rotations
                              |
              binary WebSocket frames -> Three.js
```

There are three deliberately separate layers:

1. **Neural inference** reproduces the root transformer, pose-token
   transformer, and VQ decoder.
2. **Planner/controller** reproduces context management, the critically
   damped spring target, style-pose sampling/alignment, replanning, and output
   blending.
3. **Application** maps keyboard input to commands, owns user sessions, and
   renders results. It does not implement model math.

This separation lets network parity be established without the interactive
demo and lets controller behavior be tested with recorded neural outputs.

## Proposed repository layout

```text
CMakeLists.txt
CMakePresets.json
flake.nix
include/motionbricks/
  motionbricks.h                 stable C ABI; opaque handles only
  motionbricks.hpp               optional RAII C++ wrapper
src/
  backend.{hpp,cpp}              GGML CPU/Vulkan selection and buffers
  error.{hpp,cpp}                status mapping and exception firewall
  gguf.{hpp,cpp}                 checked metadata/tensor loading
  model.{hpp,cpp}                immutable model bundle
  transformer.{hpp,cpp}          MHA, FFN, norms, masks, embeddings
  root_model.{hpp,cpp}           duration and root-trajectory graph
  pose_model.{hpp,cpp}           masked pose-token prediction graph
  vq_decoder.{hpp,cpp}           multi-head code lookup and conv decoder
  sampling.{hpp,cpp}             argmax/Gumbel and local deterministic PRNG
  skeleton_g1.{hpp,cpp}          G1 hierarchy, rest pose, FK and mappings
  motion_rep.{hpp,cpp}           414/413/418 feature conversion and stats
  clip.{hpp,cpp}                 checked skeletal clip/style assets
  controller.{hpp,cpp}           spring target and exemplar alignment
  agent.{hpp,cpp}                stateful replanning and frame buffer
  capi.cpp                       PureGo-facing ownership API
  cli.cpp                        inspect, plan, convert-style, benchmark
scripts/
  convert_to_gguf.py             safetensors/config -> GGUF
  inspect_bundle.py              independent metadata/tensor inspection
reference/
  Dockerfile                     pinned trusted PyTorch environment
  extract_safe.py                legacy Lightning checkpoint -> safetensors
  capture_fixture.py             layer/stage/end-to-end reference capture
  capture_demo_session.py        recorded upstream controller interaction
tests/
  unit/                          deterministic C++ math and validation
  parity/                        F32 CPU and Vulkan fixture runners
  integration/                   C ABI, agent, GLB/style and demo protocol
fuzz/
  capi_fuzz.cpp                  public non-GGUF inputs and call sequences
  clip_fuzz.cpp                  GLB/style parsing and validation
demo/
  main.go                        PureGo server and per-client sessions
  web/index.html
  web/app.js
  web/style.css
  web/vendor/                    pinned Three.js distribution or import map
```

GGML should be a pinned submodule as in the two reference projects. CMake
builds shared and static library variants, a CLI, tests, and optional fuzzers.
No source path, model path, GPU index, or backend path is compiled in.

## Runtime data contracts

### Canonical public animation

The public animation boundary should be independent of MuJoCo:

- `root_translation`: row-major F32 `[frames, 3]`, Y-up motion space;
- `local_rotation_xyzw`: row-major F32 `[frames, 34, 4]`;
- frame rate: 30 FPS for the released model;
- joint names, parents, and rest offsets: queried from model metadata;
- optional contacts: row-major F32 `[frames, 4]`.

MuJoCo qpos is an adapter/debug output, not the primary animation format. The
29-DOF G1 qpos form is useful for exact upstream demo parity but loses the
generic skeletal-animation boundary needed by Three.js and Kimodo.

### Neural input

Internally the planner builds the same unnormalized sparse constraints as
upstream:

- 4 recent context frames and up to 4 target frames;
- global root features `[constraint_frames, 5]`;
- local root features `[constraint_frames, 4]`;
- local-pose constraint features;
- masks saying which sparse constraints are present;
- a requested duration or an allowed-duration mask;
- optional text embeddings kept absent for the released demo path.

The released inference path recenters root X/Z, normalizes features, predicts
the root trajectory and duration, predicts 8-head pose codes, decodes four
frames per token, converts local-root features back to global motion, and
restores the original world offset.

### Style assets

A style is application/controller data rather than a learned model selector.
It contains:

- a name and optional command binding;
- one or more G1 reference clips;
- an allowed-duration mask;
- average travel speed or an explicit override;
- optional sampling range and heading correction;
- provenance, skeleton identity, FPS, and source hash.

Original demo styles are extracted from `G1-clip.ckpt` in the trusted reference
container. User styles are imported from G1 skeletal GLB, including GLB output
from `kimodo.cpp`. Both are converted to the same checked style format. The
controller samples four adjacent pose frames, applies the requested world
heading and spring-produced root target, and supplies them as ending
constraints. A style can therefore be changed or added without changing
MotionBricks weights.

Before claiming direct Kimodo compatibility, a fixture must verify the G1
joint order, rest pose, local-rotation convention, coordinate system, and FPS.
The adapter will reject rather than guess when metadata is incompatible.

## PureGo-compatible C API sketch

All configurable records and returned objects are opaque heap handles. Public
integer types have explicit widths; booleans and enum-like values are
`uint32_t`; no C enum or by-value struct is part of the ABI. Every constructor
has a matching free function. Functions return a status and use a caller-owned
error buffer.

```c
typedef uint32_t mb_status;
typedef uint32_t mb_device;

typedef struct mb_runtime_options mb_runtime_options;
typedef struct mb_model           mb_model;
typedef struct mb_style           mb_style;
typedef struct mb_command         mb_command;
typedef struct mb_agent           mb_agent;
typedef struct mb_motion          mb_motion;

uint32_t mb_abi_version(void);

mb_status mb_runtime_options_create(mb_runtime_options **out,
                                    char *err, uint64_t err_cap);
void mb_runtime_options_free(mb_runtime_options *value);
mb_status mb_runtime_options_set_device(mb_runtime_options *, mb_device,
                                        char *, uint64_t);
mb_status mb_runtime_options_set_threads(mb_runtime_options *, uint32_t,
                                         char *, uint64_t);
mb_status mb_runtime_options_set_backend_directory(mb_runtime_options *,
                                                   const char *, char *, uint64_t);
mb_status mb_runtime_options_get_device(const mb_runtime_options *, mb_device *,
                                        char *, uint64_t);

mb_status mb_model_load(const char *bundle_directory,
                        const mb_runtime_options *, mb_model **out,
                        char *err, uint64_t err_cap);
void mb_model_free(mb_model *);
mb_status mb_model_get_joint_count(const mb_model *, uint32_t *out,
                                   char *, uint64_t);
mb_status mb_model_get_joint_name(const mb_model *, uint32_t joint,
                                  const char **borrowed, char *, uint64_t);
mb_status mb_model_get_joint_parent(const mb_model *, uint32_t joint,
                                    int32_t *out, char *, uint64_t);

mb_status mb_style_load(const mb_model *, const char *style_path,
                        mb_style **out, char *err, uint64_t err_cap);
void mb_style_free(mb_style *);
mb_status mb_style_get_name(const mb_style *, const char **borrowed,
                            char *, uint64_t);
mb_status mb_style_set_speed(mb_style *, float metres_per_second,
                             char *, uint64_t);

mb_status mb_command_create(mb_command **out, char *err, uint64_t err_cap);
void mb_command_free(mb_command *);
mb_status mb_command_set_style(mb_command *, const mb_style *, char *, uint64_t);
mb_status mb_command_set_movement_direction(mb_command *, float x, float y, float z,
                                            char *, uint64_t);
mb_status mb_command_set_facing_direction(mb_command *, float x, float y, float z,
                                          char *, uint64_t);
mb_status mb_command_set_target_speed(mb_command *, float, char *, uint64_t);
mb_status mb_command_set_world_target(mb_command *, float x, float y, float z,
                                      float heading_radians, uint32_t enabled,
                                      char *, uint64_t);
mb_status mb_command_set_seed(mb_command *, uint64_t, char *, uint64_t);

mb_status mb_agent_create(const mb_model *, mb_agent **out,
                          char *err, uint64_t err_cap);
void mb_agent_free(mb_agent *);
mb_status mb_agent_reset(mb_agent *, const mb_style *initial_style,
                         char *err, uint64_t err_cap);
mb_status mb_agent_set_context(mb_agent *, const float *root_xyz,
                               const float *local_xyzw, uint64_t frames,
                               uint64_t joints, char *err, uint64_t err_cap);
mb_status mb_agent_plan(mb_agent *, const mb_command *, mb_motion **out,
                        char *err, uint64_t err_cap);
mb_status mb_agent_advance(mb_agent *, uint32_t frames, char *, uint64_t);

void mb_motion_free(mb_motion *);
mb_status mb_motion_get_frame_count(const mb_motion *, uint64_t *out,
                                    char *, uint64_t);
mb_status mb_motion_get_joint_count(const mb_motion *, uint64_t *out,
                                    char *, uint64_t);
mb_status mb_motion_get_root_translations(const mb_motion *,
                                          const float **borrowed, uint64_t *values,
                                          char *, uint64_t);
mb_status mb_motion_get_local_rotations_xyzw(const mb_motion *,
                                             const float **borrowed, uint64_t *values,
                                             char *, uint64_t);
```

The final header will add getters for every option setter and explicit lifetime
documentation. Borrowed strings and arrays remain valid until their owning
handle is freed or mutated. `mb_model` is immutable and may be shared;
`mb_agent`, `mb_command`, and `mb_motion` are not concurrently mutable. No C++
exception crosses `capi.cpp`.

The CLI uses this same C API for end-to-end operations so the ABI is exercised
outside unit tests. Internal parity tools may access raw tensors through a
non-installed test header; raw GGML objects and 413/414-dimensional model
features are not exposed as stable public API.

## Model conversion and bundle

Use a directory manifest with separate memory-mappable component files:

```text
motionbricks-g1-f32/
  manifest.json
  root-f32.gguf
  pose-f32.gguf
  vq-decoder-f32.gguf
  styles/
    upstream-defaults.mbstyle
```

The manifest records format version, model identity, source revision, source
and intermediate hashes, skeleton key, FPS, component dtype, GGML revision,
and mutually compatible component identities. Loading rejects mixed bundles.

GGUF metadata records all architecture values rather than relying on C++
defaults, including the released 512-wide root transformer, 1024-wide pose
transformer, 16 attention heads, layer counts, token range, eight pose-code
heads, four frames per token, normalization statistics, feature-index tables,
skeleton data, and sampling defaults. Every tensor lookup is named, typed,
shape checked, and bounds checked.

The ordinary converter accepts safetensors plus JSON only. NVIDIA's Lightning
checkpoints and the serialized demo-clip checkpoint are Python pickle formats;
only the pinned, network-disabled reference container loads those files. It
then writes safetensors/JSON with hashes for the normal converter. The VQ
encoder is omitted from runtime bundles because inference only needs the code
embeddings and decoder; a reference/debug bundle may retain it for tokenizer
parity.

## Neural implementation order

### Shared operators

Implement the smallest operation set used by the released checkpoints:

- checked embedding lookup, linear layers, ReLU, LayerNorm, residual add;
- PyTorch-compatible multi-head self-attention and key-padding masks;
- fixed/learned position embeddings;
- 1D convolution, transpose convolution, and dilated residual blocks;
- reshape, permute, concatenate, gather/scatter, softmax and argmax.

An operation audit against the actual `state_dict` and forward hooks precedes
graph work. PyTorch defaults such as post-norm transformer order, bias,
activation, epsilon, and attention scaling must be captured explicitly, not
reconstructed from memory.

### Root model

The root graph embeds sparse start/end pose and root constraints, duration,
and position; runs shared and output transformer stacks; predicts duration;
and expands transformer features through the conditioned convolutional decoder
to the global root trajectory. Duration filtering uses the caller/style's
allowed-token mask exactly as upstream.

### Pose model

The pose graph combines masked multi-head VQ code embeddings, predicted root
features, sparse pose constraints, duration embedding, and position embedding;
runs the 16-layer transformer; and produces per-head code logits. One sampling
iteration is the released demo default. Argmax is supported for deterministic
diagnostics; production Gumbel sampling uses an agent-local PRNG and seed.
Reference fixtures store the random uniforms/Gumbel values explicitly instead
of assuming PyTorch and C++ generators produce identical streams.

### VQ decoder and representation inverse

Selected codes are expanded through the eight code heads and conditioned
decoder. Sparse start/end constraints and the predicted root path are applied
as upstream. The resulting normalized local representation is converted to
global features, denormalized, and transformed into skeletal animation. These
ordinary math stages are implemented in C++ and tested independently of GGML.

### Stateful controller

An agent owns its current generated buffer and frame cursor. At a replan it:

1. gathers the previous four actual frames;
2. derives current root velocity and heading velocity;
3. applies the upstream critically damped spring to the requested movement,
   facing, speed, or explicit world target;
4. samples and aligns four frames from the selected style clip;
5. constructs sparse model constraints and runs inference;
6. restores the world transform, converts the output to local rotations, and
   applies the upstream context blend;
7. exposes the valid predicted prefix and retains it as the next context.

The controller uses seconds internally and derives frame counts from model FPS.
It must not silently enable canonicalization, target skipping, alternate root
selection, extra pose iterations, or any other non-default upstream option.

## Web demo

The Go server dynamically loads the shared library with PureGo. It owns one
`mb_agent` per WebSocket session and shares one immutable `mb_model`. Initially
planning requests are serialized through a worker so Vulkan ownership and peak
memory are predictable; parallel sessions can be added after profiling.

The browser sends compact command state only: pressed controls, movement and
camera-facing vectors, selected style, and monotonically increasing sequence
number. The server returns a binary animation chunk containing a small
versioned header, frame count, root translations, and local quaternions.
Three.js keeps a short playback buffer and samples at render time. The server,
not the browser, owns authoritative agent state and drops stale commands.

HTTP endpoints provide model/style metadata and health information. A
WebSocket carries interactive commands and generated chunks. The frontend
includes:

- WASD movement relative to the camera;
- independently controllable facing direction;
- style/action buttons populated from the style manifest;
- a skeleton/mesh viewer, pause/reset, latency and buffered-frame display;
- an optional debug view for target position, facing vector, contacts, and
  replan boundaries.

The initial viewer can render a simple G1 skeleton. A skinned character and
retargeting are later integrations, potentially using `skin-tokens.cpp`.

## Reference and parity strategy

Every fixture has a JSON manifest containing the upstream commit, checkpoint
and safetensors hashes, config hash, PyTorch version, dtype/device, tensor
shapes, seed/random inputs, selected style, command stream, and tolerances.
Binary tensors are little-endian and versioned. Tests refuse mismatched
fixtures.

Capture and compare in this order:

| Boundary | Reference values |
|---|---|
| checkpoint extraction | every tensor name, dtype, shape, hash |
| skeletal math | qpos/local rotations, FK positions, global rotations |
| motion representation | raw, global 414, local 413, dual conversion, normalization |
| style/controller | sampled frames, average speed, spring roots/headings, aligned targets |
| shared neural ops | linear, LayerNorm, attention, masks, conv/residual blocks |
| root network | every projection, transformer layer, duration logits, decoder layer, root path |
| pose network | every embedding/projection, transformer layer, logits, sampled codes |
| VQ decoder | code embeddings and every decoder layer |
| inference composition | recenter, constraints, root, pose, decode, restore world transform |
| agent session | recorded command changes, replans, valid lengths, final skeletal frames |

The first exemplar set should be small but discriminating:

- idle continuation with a fixed duration;
- walk forward, then a 90-degree turn during playback;
- switch from walk to one strongly recognizable upstream style;
- explicit world target behind the character;
- a fixed Kimodo G1 reference clip used as a style;
- minimum and maximum allowed duration;
- both argmax and stored-Gumbel-token inference.

F32 CPU tolerances are set per stage from observed error, with maximum absolute
and relative L2 limits. End-to-end animation also checks root error, joint
position error after FK, quaternion angular error, token equality where
applicable, duration equality, and contact equality. Vulkan has its own
documented tolerances but must preserve duration and discrete code decisions
on the exemplar suite unless an explicitly reviewed near-tie is present.

## Safety, fuzzing, and validation

GGUF parsing remains GGML's responsibility, but motion-bricks validates bundle
metadata and every expected tensor. Fuzz targets cover:

- the installed C ABI with generated valid/invalid call sequences;
- opaque-handle nulls, ownership order, repeated free attempts in the harness,
  lengths, finite values, dimensions, and invalid enums;
- GLB/style assets, skeleton metadata, malformed animation tracks, NaN/Inf,
  huge frame counts, and incompatible joint orders;
- command normalization, degenerate direction vectors, duration masks, spring
  math, representation conversions, quaternion/matrix conversions, and FK;
- the versioned WebSocket binary decoder independently in Go.

Run C++ fuzzers with ASan and UBSan. Unit and integration tests run under the
same sanitizers. File parsers impose explicit byte, frame, joint, accessor, and
allocation limits before allocating. No user-provided path is interpreted by
the browser; the server uses configured style/model roots.

## Build and configuration

The normal build is distro-agnostic CMake. Nix supplies a reproducible
developer shell containing Clang, CMake, Ninja, Go, Python fixture tools,
Vulkan headers/loader/tools, shaderc, and sanitizers. Docker contains upstream
Python/CUDA extraction and fixture capture. Neither is required by installed
runtime consumers.

Suggested configuration inputs:

- CMake cache: GGML source directory, tests/fuzzers, Vulkan enablement;
- environment/CLI: model bundle, style directory, backend directory, device,
  CPU thread count, server address, fixture directory;
- Vulkan device selection: documented loader variables or a CLI-selected
  GGML Vulkan device after confirming backend support. The Nix shell supplies
  loader paths but never assumes an NVIDIA device index.

CI lanes are formatting/static checks, CPU unit tests, ASan/UBSan tests, CPU
parity with optional fixture artifacts, and a hardware-tagged Vulkan parity
lane. Large weights and fixtures are downloaded by hash and never committed.

## Implementation milestones

### 0. Lock the reference

- Record the upstream MotionBricks commit and GGML revision.
- Fetch Git LFS checkpoints/assets and record their hashes/licences.
- Build a pinned Python reference environment and reproduce the interactive
  demo unchanged.
- Capture one deterministic end-to-end upstream exemplar.

**Exit:** the reference script reproduces a stored animation and manifest on a
fresh environment.

### 1. Scaffold and define contracts

- Add CMake, presets, Nix flake, pinned GGML, library/CLI/test targets.
- Land the opaque C header, C++ wrapper, status/error rules, and stub handles.
- Define fixture, GGUF bundle, style, and WebSocket format versions.
- Add ABI lifecycle tests from C and PureGo.

**Exit:** CPU-only stubs build, install, and are called successfully through
PureGo without any mirrored struct.

### 2. Safe extraction and exemplar capture

- Extract root, pose, VQ, stats, skeleton, and original style tensors to
  safetensors/JSON in the trusted container.
- Add forward hooks/capture for every required layer and preprocessing stage.
- Capture the exemplar matrix above, including random sampling inputs.
- Build the safetensors-to-GGUF converter and bundle inspector.

**Exit:** source and converted tensor inventories match exactly and all bundle
identities are validated.

### 3. Deterministic motion/controller math

- Implement G1 metadata, quaternion/matrix conversions, FK, feature extraction,
  414/413 representation conversion, stats, and skeletal output conversion.
- Implement style import, clip alignment, spring trajectory, duration masks,
  context selection, and blending without neural inference.
- Add unit, fixture parity, and sanitizer tests for each stage.

**Exit:** C++ matches all non-neural upstream fixture boundaries.

### 4. VQ decoder on CPU

- Audit and implement required GGML convolution/residual operations.
- Load codebooks and decoder tensors from GGUF.
- Match each decoder layer and decoded local-motion features in F32.

**Exit:** CPU VQ decoder fixture passes end to end.

### 5. Root model on CPU

- Implement shared transformer operators and root embeddings/masks.
- Match each transformer layer, duration selection, conditioned decoder, and
  predicted global/local root trajectories.

**Exit:** CPU root fixtures pass for fixed, predicted, minimum, and maximum
durations.

### 6. Pose model and composed inference on CPU

- Implement pose embeddings, transformer, logits, argmax/Gumbel sampling.
- Compose root -> pose -> VQ -> representation inverse.
- Implement the stateful agent and verify recorded replanning sessions.

**Exit:** F32 CPU passes every layer and end-to-end skeletal exemplar.

### 7. C API, Kimodo styles, and web demo

- Complete opaque getters/setters, ownership, error paths, and installed C API.
- Verify Kimodo G1 GLB import and add a style conversion workflow.
- Implement Go/PureGo server, binary WebSocket protocol, and Three.js viewer.
- Reproduce upstream styles and interactive turn/style-switch scenarios.

**Exit:** a fresh build runs the interactive demo using both an original and a
Kimodo-authored style, with no Python process at runtime.

### 8. Vulkan parity and performance

- Enable GGML Vulkan and select the intended GPU through runtime configuration.
- Match every component and end-to-end fixture using Vulkan-specific
  tolerances.
- Profile allocation, upload, graph construction, inference, and WebSocket
  latency; cache graphs/buffers where safe.
- Establish a replan cadence the machine sustains while playback remains
  buffered.

**Exit:** Vulkan preserves exemplar behavior and meets the documented
interactive latency/buffer target on the test machine.

### 9. Hardening and optimized weights

- Run long ASan/UBSan fuzz campaigns and multi-session soak tests.
- Add F16 conversion/parity, then evaluate quantization component by component.
- Package install targets, model downloader, licences, deployment docs, and
  reproducible release manifests.

**Exit:** release artifacts are reproducible, validated, and contain no hidden
machine-specific paths or upstream Python dependency.

## Decisions to revisit after the first fixtures

- Whether the public style format should be a small GGUF, GLB plus JSON, or a
  custom binary. GLB plus JSON is easiest for user-authored Kimodo assets;
  GGUF is likely best for exact preprocessed upstream tensors.
- Whether `mb_agent_plan` should retain output internally or return an owning
  immutable copy. The owning result above is simpler and safer for PureGo;
  profiling may justify an explicit caller-buffer API later.
- Whether to expose a generic sparse-keyframe API in ABI version 1. It is more
  capable than the interactive controller but substantially enlarges the
  validation surface. The first slice can support movement/facing/style/world
  target while keeping the generic constraint builder internal.
- Whether to link a small shared GLB module from `kimodo.cpp`/`skin-tokens.cpp`
  or keep style conversion in the CLI. Reuse should happen only through a
  clean library boundary, not by copying diverging parser code.
- Exact interactive latency, memory, and browser buffer targets. Measure the
  F32 CPU baseline and F32 Vulkan implementation before promising numbers.

## Explicitly deferred

- training or fine-tuning MotionBricks;
- arbitrary skeleton inference or automatic retargeting;
- text input to MotionBricks itself;
- robot torque/control-policy output;
- WebAssembly inference in the browser;
- F16/quantized defaults before F32 parity;
- multi-batch inference and concurrent mutation of one agent.
