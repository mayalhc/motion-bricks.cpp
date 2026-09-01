# motion-bricks.cpp

A C++23/GGML port of NVIDIA MotionBricks for CPU and Vulkan inference, with a
stable C ABI suitable for PureGo.

The released batch-one G1 inference path is implemented end to end: strict
GGUF loading, root/duration planning, pose-token prediction, VQ decoding,
418/414/413 feature conversion, style alignment, and skeletal animation
output. CPU and Vulkan use the same public API and preserve the same duration
and pose-token decisions in the reference suite.

## Design

- [Human-led design](docs/motions-bricks.md)
- [Implementation sketch and plan](docs/IMPLEMENTATION.md)
- [Versioned formats](docs/FORMATS.md)
- [Pinned upstream reference](reference/README.md)

## Build

The normal build uses CMake and does not depend on Nix:

```sh
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

On NixOS, enter the reproducible development shell first:

```sh
nix develop
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

When the pinned `ggml/` submodule is present, it is included automatically.
The non-neural ABI and validation subset can also be built without GGML:

```sh
cmake -S . -B build/debug -G Ninja -DMOTIONBRICKS_ENABLE_GGML=OFF
```

The sanitizer lane is:

```sh
cmake --preset asan-ubsan
cmake --build --preset asan-ubsan
ctest --preset asan-ubsan
```

## Current ABI

The installed C API uses only fixed-width scalars, pointers, and opaque heap
handles. Callers never reproduce a C or C++ structure layout. All constructors
have matching free functions, and no C++ exception crosses the ABI boundary.

The current CLI can report ABI information:

```sh
./build/debug/bin/motionbricks-cli abi
```

After producing the trusted safetensors intermediates described in
`reference/README.md`, build and inspect an F32 runtime bundle with:

```sh
python scripts/convert_to_gguf.py \
  --safe-directory generated/safe \
  --output generated/g1-f32
./build/debug/bin/motionbricks-cli inspect generated/g1-f32
```

The released G1 inference path contains exactly **183,148,382 learned F32
parameters**. The bundle loader validates the upstream revision, source
checkpoint identities, component roles, tensor counts, parameter counts,
anchor shapes, and the 34-joint parent topology before accepting a model.

Convert the 15 original demo styles with:

```sh
python scripts/convert_styles.py \
  --safe-directory generated/safe \
  --output generated/styles
```

At runtime the high-level flow is: load one immutable model, load one or more
`.mbstyle` assets, create an agent, reset it from an initial style (or supply
at least four frames of G1 context), set movement/facing/style on a command,
then call `mb_agent_plan`. The returned motion owns row-major F32 root
translations `[frames,3]` and local XYZW rotations `[frames,34,4]`. Call
`mb_agent_advance` as playback progresses so replanning uses the generated
motion as its next context.

The current implementation covers original preprocessed G1 styles. Direct
Kimodo GLB-to-`.mbstyle` conversion and the Go/browser demo remain subsequent
integration work.
