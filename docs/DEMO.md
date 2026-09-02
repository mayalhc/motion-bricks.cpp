# Go and Three.js demo

The initial demo is a local Go application that calls `libmotionbricks`
through the PureGo binding and serves an embedded Three.js viewer. It renders
the released 34-joint G1 hierarchy directly from MotionBricks root
translations and local XYZW joint rotations; it does not require MuJoCo or a
skinned mesh. Solid cyan cylinders and round joints identify the generated
character. Orange diamond-jointed ghost skeletons identify the four placed
style-pose constraints supplied to the planner.

## Build and run

First build the native project and create the model/style assets described in
the main README. Then build the Go application:

```sh
cmake --build --preset debug
cd demo
go build -o ../build/debug/bin/motionbricks-demo .
cd ..
```

Run it from the repository root:

```sh
./build/debug/bin/motionbricks-demo \
  -listen 127.0.0.1:8080 \
  -library ./build/debug/libmotionbricks.so \
  -model ./generated/g1-f32 \
  -styles ./generated/styles \
  -device cpu
```

Open `http://127.0.0.1:8080/`. Hold physical W/A/S/D keys to move, or tap an
on-screen direction to latch it; tap the active pad direction again to stop.
Space or Escape also stops movement. Left/right arrow keys rotate the facing
direction without changing the current travel vector. The selector switches
among all `.mbstyle` files found in the style directory, including the 15
converted upstream styles. Drag over the viewport to orbit, use the wheel to
zoom, and use **Reset camera** to restore the automatically framed view. The
target toggle switches between all four ghosts and the final target only.

`-device` accepts `cpu`, `vulkan`, or `auto`. The server deliberately binds to
localhost by default. Model inference is serialized while sessions keep
independent agent/context state.

## Runtime shape

The browser creates a session, receives a 30 FPS animation chunk, and asks for
a replacement chunk when controls change or playback approaches the end.
Each request contains movement, facing, style, seed, and the number of frames
already consumed. The Go server advances that session's native agent and
returns owned animation data and target constraints as JSON:

- root translations: `[frames, 3]`;
- local joint rotations: `[frames, 34, 4]`, XYZW;
- placed target roots: `[4, 3]`;
- placed target local rotations: `[4, 34, 4]`, XYZW;
- G1 joint names, parent indices, and neutral positions from the loaded model.

The native target data is captured after style-frame sampling, spring-based
world placement, and heading correction. The ghosts therefore visualize the
actual planner inputs. The viewer does not force them in front: forward travel
normally places them ahead, while stops and turns can make them overlap the
character or move sideways.

The browser builds `THREE.Bone` objects from the returned hierarchy and draws
solid cylinders/spheres for the generated skeleton and translucent
cylinders/diamonds for target poses. It also draws generated and target root
paths on the floor. Animation chunks are immutable in JavaScript; a later
version can replace JSON with a binary streaming protocol without changing
the native API.

Three.js r180 is vendored under `demo/web/vendor` so the demo has no runtime
CDN dependency.

## Tests

With the generated assets present, CTest registers `motionbricks-go-demo` when
Go and Chromium are available. The test starts an in-process HTTP server,
loads the real native model, plans an initial `walk` chunk, then uses headless
Chromium to select `walk_zombie`, turn right, plan another chunk, render the
34-joint generated hierarchy plus all four target ghosts, and capture initial,
forward-motion, and style-and-turn screenshots. It uses real Chrome click and
keyboard events and asserts forward-pad movement, pad stop, keyboard movement,
and keyboard stop commands before the visual self-test.

The Go tests can also be run directly:

```sh
cd demo
MOTIONBRICKS_LIB=../build/debug/libmotionbricks.so \
MOTIONBRICKS_MODEL=../generated/g1-f32 \
MOTIONBRICKS_STYLES=../generated/styles \
MOTIONBRICKS_CHROME="$(command -v chromium)" \
go test -v ./...
```

Without the native asset environment variables, the parser test still runs
and the native/browser integration cases are skipped.

## Initial limitations

- G1 is the only skeleton supported by the released model.
- The viewer intentionally shows a bone skeleton, not a skinned avatar.
- HTTP JSON carries whole planned chunks; binary streaming and client-side
  overlap blending are future work.
- Sessions are in-memory and intended for a trusted local demo, not an
  internet-facing multi-user service.
