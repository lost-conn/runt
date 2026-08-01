# runt

A custom Rust 3D engine for the browser (primary) and native (secondary), tuned
to a procedural-first design flow. See `DESIGN.md` for the doctrine; this file is
how to run it.

v0 ships two programs on the same engine:

- **the engine demo** — the generator/renderer showcase (`crates/runt-app`)
- **runt ball** — the v0 demo *game*: a rolling-ball collector on procedural
  terrain, seeded and replayable (`demo/ball`)

## Layout

A cargo workspace (DESIGN §2). The engine is windowless; hosts own the window,
the surface and presentation, and contain no game logic.

```
crates/runt-mesh/   pure, GPU-free mesh generation (glam only)
crates/runt-core/   engine: ECS world, fixed tick, generators + cache, physics,
                    renderer. Draws into a caller-provided wgpu::TextureView.
crates/runt-app/    player host: winit loop, surface + present, wasm entry.
                    `RunConfig` + `run_with` is the seam every program uses.
demo/ball/          the v0 game. All rules in src/game.rs, all content in
                    assets/level1.ron. Zero engine changes.
```

`cargo run` from the root builds `runt-app` (the workspace's default member).

## Run

Native:

```
cargo run                          # the engine demo
cargo run -p runt-ball             # the game
cargo run -p runt-ball -- --record run.trace    # …and save the input trace
cargo run -p runt-ball -- --replay run.trace    # watch a run play itself back
```

Web (two separate trunk targets):

```
trunk serve --release --open                 # engine demo, :8080
trunk build --release                        # → ./dist

cd demo/ball && trunk serve --release --open # the game, :8080
cd demo/ball && trunk build --release        # → ./demo/ball/dist
```

Tests:

```
cargo test --workspace
```

## runt ball

WASD or the arrow keys roll the ball, camera-relative. Collect all twelve rings.
Falling off the edge puts you back at the start; it costs time and nothing else.

Score, elapsed time and the win are shown **without a text renderer** — DESIGN
§13 leaves HUD text open, so the game writes a `StatusLine` resource and the host
paints it where the platform already has cheap text: the window title natively,
`document.title` plus the `#runt-status` DOM overlay on web. Removing that `div`
from `demo/ball/index.html` degrades to the tab title and breaks nothing.

### Replays

A trace is `(tick index, input event)` pairs in postcard — DESIGN §4's *"replays
are just recorded input traces + seeds"*, with the seed already in `level1.ron`.
The trace is keyed on **ticks, not wall time**, which is what makes it survive a
different frame rate: `demo/ball/tests/replay.rs` runs the whole game under a
clean 60 fps host and a 5/30/7 ms stuttering one and compares a per-tick hash of
every transform plus the entire `GameState`, bit for bit.

`--record` writes on a clean window close (the browser never gives a page the
last word, so it is native-only). Passing both flags re-records a replay, which
must reproduce the input file byte for byte — a cheap end-to-end self-check.

### Tuning, and why

Measured, not guessed: every number below is asserted in `demo/ball/tests/`
(`feel.rs` prints the measurements with `-- --nocapture`).

| Knob | Value | Why |
|---|---|---|
| terrain seed | 514 | Of 600 seeds surveyed at this amplitude/frequency, the widest height range whose *steepest point anywhere on the patch* is still under 30° — hills worth cresting, no slope the ball cannot climb. |
| size / amplitude / frequency | 48×48 m / 2.5 / 0.07 | 3.5 m of height range over 48 m: mean slope ~10°, steepest 28.0°. At the engine default frequency (0.045) it reads as a putting green. |
| `accel` 16, `rolling_friction` 2.2 | terminal roll ≈ 7.3 m/s | 20 m from a standing start in **3.37 s**, so half the map is ~4 s. The engine defaults (22 / 1.2 → 18 m/s) cross the whole map in under 3 s and make the follow camera seasick. |
| `max_speed` 14 | | A downhill run can outpace the roll terminal velocity without moving far enough in one tick (0.23 m) to tunnel a 0.8 m post. |
| climb headroom | 16 vs 8.29 m/s² | Input acceleration against gravity's pull on the steepest slope on the map — 1.9× margin, so there are no dead ends in a game with no jump. |
| camera offset (0, 7, 10.5), stiffness 3.5 | | Loose enough to lag on a hard turn (which is what sells the speed), tight enough not to swing. Measured minimum clearance over 94 m of driving: **6.5 m** — it cannot clip a hill even with the ball in the deepest hollow. |
| pickup clearance 0.9 m, trigger 0.7 | | Every ring is takeable by a ball *resting on the ground beneath it*, with >0.8 m of horizontal slack, at both extremes of the bob. |

## Stack

- `wgpu` 30 — graphics abstraction (WebGPU + WebGL2 backends)
- `winit` 0.30 — windowing / input (native + web canvas)
- `bevy_ecs` 0.19 — the ECS only, à la carte (DESIGN §3)
- `glam` — math
- `ron` / `serde` / `postcard` — scenes, params, caches, traces
- `trunk` — wasm bundler + dev server

## Compatibility notes (from the spike)

- **WebGPU presence ≠ WebGPU usable.** A browser can expose `navigator.gpu`
  yet return `null` from `requestAdapter()` (headless, no GPU, blocklisted
  driver). We use `wgpu::util::new_instance_with_webgpu_detection`, which
  probes for a real adapter and drops the WebGPU backend if none exists.
- **WebGL2 fallback** is enabled via the `webgl` feature on the wasm target.
  Verified rendering in headless Chrome (WebGPU adapter unavailable) through
  ANGLE / OpenGL ES 3.2. Cost: ~2 MB more wasm (naga GLSL codegen + GL glue).
- **One `#[wasm_bindgen(start)]` per module**, and wasm-bindgen collects them
  from dependency rlibs. `runt-app`'s own entry point is therefore behind its
  default `wasm-entry` feature, which `demo/ball` switches off so it can declare
  its own. A new web program does the same.
- **Limits:** we request `downlevel_webgl2_defaults()` so the same code path
  is valid on either backend. Drop this to full limits once WebGL2 is no
  longer a target — it unlocks compute shaders, storage buffers, etc.
- **Canvas sizing gotcha:** winit does *not* resize the web canvas backing
  store to its CSS box. Without an explicit `request_inner_size`, the drawing
  buffer stays 1×1 and gets stretched (uniform-color screen). We sync it to
  the browser viewport each frame.
- **No blocking on web:** device init is async; we build graphics in
  `spawn_local` and hand it back to the winit loop via a user event.

## Binary size

Release wasm, with the WebGL2 fallback compiled in:

| target | raw | gzipped |
|---|---|---|
| engine demo | 3.0 MB | 1.15 MB |
| runt ball | 3.0 MB | 1.16 MB |

The game costs ~43 KB over the engine demo — the whole of `game.rs` plus the
level. Most of the bundle is `wgpu` + naga's GLSL backend; dropping the `webgl`
feature is the single biggest lever if WebGL2 stops being a target.
