# GGML compatibility patches

The initial GGML revision is
`8c63e70982c95ceb862e3a1073a2c1beef75d60a` (`v0.20.2`). No compatibility
patch is currently required.

If a patch becomes necessary, add numbered patch files here and update CMake
to copy GGML into the build directory and apply them in lexical order. Never
modify the pinned submodule worktree during configuration.
