# runt

Compatibility spike for a custom Rust 3D engine that runs in the browser.
A spinning, lit, depth-tested cube rendered through `wgpu`, running natively
(Vulkan/Metal/DX12) and on the web (WebGPU, falling back to WebGL2).

## Layout

A cargo workspace (see `DESIGN.md` §2). The engine is windowless; hosts own the
window, the surface and presentation.

```
crates/runt-mesh/   pure, GPU-free mesh generation (glam only)
crates/runt-core/   engine: vertex layout, shader, pipeline, depth, Renderer.
                    Renders into a caller-provided wgpu::TextureView.
crates/runt-app/    player host: winit loop, surface + present, wasm entry
```

`cargo run` from the root builds `runt-app` (the workspace's default member).

## Stack

- `wgpu` 30 — graphics abstraction (WebGPU + WebGL2 backends)
- `winit` 0.30 — windowing / input (native + web canvas)
- `glam` — math
- `trunk` — wasm bundler + dev server

## Run

Native:

```
cargo run
```

Web:

```
trunk serve --release --open      # dev server at http://localhost:8080
trunk build --release             # emits ./dist
```

Tests:

```
cargo test --workspace
```

`runt-core`'s `headless_screenshot` test builds a device with no surface at
all, renders the demo scene to an offscreen 512² texture, reads it back and
asserts geometry actually covered the frame. Run it with `-- --nocapture` to
print the frame's pixel hash.

## Compatibility notes (from the spike)

- **WebGPU presence ≠ WebGPU usable.** A browser can expose `navigator.gpu`
  yet return `null` from `requestAdapter()` (headless, no GPU, blocklisted
  driver). We use `wgpu::util::new_instance_with_webgpu_detection`, which
  probes for a real adapter and drops the WebGPU backend if none exists.
- **WebGL2 fallback** is enabled via the `webgl` feature on the wasm target.
  Verified rendering in headless Chrome (WebGPU adapter unavailable) through
  ANGLE / OpenGL ES 3.2. Cost: ~2 MB more wasm (naga GLSL codegen + GL glue).
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

Release wasm ≈ 2.3 MB raw / ~900 KB gzipped with the WebGL2 fallback compiled
in. Dropping to WebGPU-only (remove the `webgl` feature) returns it to
~370 KB raw / ~150 KB gzipped.
