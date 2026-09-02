---
license: other
library_name: ggml
tags: [gguf, ggml, motion-generation, unitree-g1, motionbricks]
---

# MotionBricks-G1-GGML

Native F32 GGML/GGUF conversion of NVIDIA MotionBricks' released G1 pose,
root, and VQ-decoder checkpoints, plus the 15 released style primitives. It is
produced by
[`motion-bricks.cpp`](https://github.com/localai-org/motion-bricks.cpp) for CPU and
Vulkan inference.

NVIDIA currently publishes MotionBricks through Git LFS rather than a separate
Hugging Face model repository. This conversion is linked to the pinned
[`NVlabs/GR00T-WholeBodyControl` revision](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/a0732b642c0333077e127a2f56ab0014c196bca4/motionbricks).

Install the complete checked bundle at `generated/g1-f32` and
`generated/styles` with:

```sh
python scripts/download_gguf_weights.py
```

`MANIFEST.json` and `SHA256SUMS` record the source revision, exact paths,
sizes, and SHA-256 hashes. The bundle contains 183,148,382 learned F32
parameters and targets MotionBricks' released 34-joint Unitree G1 skeleton.

## License and provenance

The original checkpoints and this converted representation remain subject to
the NVIDIA Open Model License in `UPSTREAM_LICENSE`. Redistribution requires
retaining that agreement and NVIDIA's attribution in `NOTICE`. This conversion
grants no additional rights. Use is also subject to NVIDIA's Trustworthy AI
and trade-compliance terms described by that agreement.
