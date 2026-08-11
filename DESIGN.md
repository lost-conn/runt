# runt — Engine Design

Status: **locked** 2026-07-31. Changes to a *Decision* section require a deliberate
revisit, not drift. Open questions are collected at the bottom.

runt is a personal 3D engine for the browser (primary) and native (secondary),
tuned to a procedural-first design flow. It is not a general-purpose engine and
does not compete with Bevy/Godot; it optimizes for one person's iteration loop:
*write a generator → tweak params → see it instantly → place it in a scene*.

---

## 1. Pillars (unchanged from 2026-07-17)

1. **Procedural-first, not procedural-only.** Generators are the emphasis;
   authored scenes and placed objects are first-class.
2. **Triangles, generated.** Classic rasterized meshes are THE render path.
   SDFs may appear as a *generation* input (meshed to triangles at bake time),
   never as a runtime raymarcher.
3. **Low-end is a hard constraint.** Average integrated GPU and mid-range
   Android must run everything, scaling *down* (quality tiers) rather than
   failing. Baseline fits WebGL2 downlevel limits.
4. **Determinism.** Same params + same build → identical output. Enables
   content-addressed caching, LOD regeneration, and save-as-params.
5. **One baseline + capability-gated enhancements.** A single code path runs
   everywhere; compute particles, live texture eval, higher LOD tiers switch on
   where supported (desktop / real WebGPU).

Non-goals: photorealism, skeletal animation import pipelines, glTF authoring
workflows, a general asset store, mobile-native builds (browser covers mobile).

---

## 2. Crate architecture

**Decision:** split into a windowless core and thin hosts.

```
runt/                      # workspace root
├─ crates/
│  ├─ runt-core/           # engine: ECS world, sim, renderer, generators.
│  │                       #   NO winit, NO window/surface ownership.
│  │                       #   Renders into a caller-provided wgpu::TextureView.
│  ├─ runt-mesh/           # current src/mesh/ moved here. Pure, GPU-free.
│  ├─ runt-app/            # native + web player host: winit loop, surface,
│  │                       #   input translation, canvas sizing. Thin.
│  └─ runt-editor/         # native rinch app: RenderSurface viewport +
│                          #   param panels. Depends on runt-core only.
```

Rules:

- `runt-core` owns a `wgpu::Device`/`Queue` it is *given* (or creates headless)
  and renders to any `TextureView`. Presenting is the host's job. This is what
  makes the rinch editor, headless screenshot tests, and the web player all the
  same engine.
- `runt-mesh` stays free of wgpu types (already true). Everything in it is
  unit-testable without a GPU.
- Hosts contain no engine logic. If a feature needs host code beyond event
  translation and surface management, it's designed wrong.

## 3. World model — bevy_ecs à la carte

**Decision:** adopt `bevy_ecs` (+ `bevy_reflect` for the editor) as standalone
crates. No other Bevy crates — renderer, assets, audio, app loop are ours.

- **Core components** (initial set): `Transform` (TRS, glam), `GlobalTransform`,
  `Parent`/`Children` (optional hierarchy — flat by default, hierarchy only
  where attachment is needed), `MeshRef` (content hash + generator params ref),
  `MaterialRef`, `Visibility`, `Interpolated` (previous-tick transform for
  render interpolation).
- **Schedules:** `Startup`, `FixedSim` (the deterministic tick), `PostSim`
  (transform propagation, interpolation bookkeeping), `Extract` (world →
  renderer snapshot). Rendering itself is not a system; the host calls
  `Engine::render(alpha, view)` after zero or more ticks.
- **Determinism rules for systems:** system order in `FixedSim` is explicitly
  chained (no ambiguous parallel ordering); no `HashMap` iteration feeding
  simulation state (use sorted keys or `Vec`s); RNG is always an explicitly
  seeded PRNG component/resource, never thread RNG. Query iteration order in
  bevy_ecs is stable for a fixed spawn order, and spawn order is itself
  deterministic under these rules — but we do not rely on it where ordering
  matters; sort by `Entity` when it does.
- **Reflection:** generator param structs and editor-visible components derive
  `Reflect`. This is the contract that lets the editor build panels without
  hand-written UI per generator.

Why ECS over the flat table: gameplay logic was always going to need *some*
composition story, and bolting one on later is worse than pinning bevy_ecs's
ordering down now. Cost accepted: wasm size (+~200–400 KB) and compile time.

## 4. Time — fixed-tick simulation

**Decision:** fixed 60 Hz sim tick, render interpolation, accumulator loop.

- `TICK_DT = 1/60` exactly. Accumulator clamped at 0.25 s (spiral-of-death
  guard: below-floor devices slow down rather than freeze).
- Render pass receives `alpha ∈ [0,1)` and draws transforms lerped/slerped
  between previous and current tick (`Interpolated` component).
- All gameplay/sim mutation happens in `FixedSim`. Per-frame code (camera
  smoothing may live render-side) must not feed back into sim state.
- Determinism scope: same build + same platform + same input trace → identical
  run. Cross-platform float identity is **not** promised for sim (it is
  *checked* for content generation via `content_hash()`, which will surface any
  platform divergence in generators loudly rather than silently).
- Input is captured by the host, translated to engine input events, and
  consumed at tick boundaries — replays are just recorded input traces + seeds.

**Actions (2026-08-04, for the 3dimenshift port).** The engine owns the
action-map *mechanism* and nothing else: `runt-core`'s `action.rs` is a
`Bindings` table — plain serializable data mapping keys, mouse buttons, pad
buttons, analog triggers and stick directions onto action slots, plus the
move/look stick assignments — and a per-tick `resolve_actions::<A>` pass at
the head of `FixedSim`. Each game owns its *vocabulary*: an enum implementing
`ActionId`, whose dense index is the bit position in `Actions<A>` and the row
in the table. The engine has no opinion about whether a game has a "jump";
gameplay systems read `Actions<A>` and never a `Key`, so remapping is editing
data (a RON blob in the KV store — `localStorage` on web, XDG config natively)
rather than editing systems, and a rebind UI is a later view onto the same
bytes. Replays stay independent of the bindings because a trace records **raw
`InputEvent`s**, not resolved actions: playback re-derives the actions from
the events, so a run recorded under one binding set and played back under
another is a different *reproducible* run rather than a corrupt trace. The one
subtlety worth naming: action edges are computed by diffing this tick's held
bits against last tick's rather than forwarding `Input::just_pressed`, which
is the only way an analog trigger crossing its threshold — a level, with no
edge in `Input` at all — can be a press, and the only way two sources bound to
one action produce one press instead of two.

## 5. Renderer — stylized forward

**Decision:** single forward opaque pass, stylized (non-PBR) lighting.

- **Look:** directional key light + hemisphere ambient (sky/ground colors),
  vertex color × procedural albedo, optional ramp/toon step as a material
  variant. No metallic-roughness, no image-based lighting. We control every
  surface procedurally, so "plausible PBR response" is a burden with no payoff.
- **Material** = small uniform block (base color, ramp params, texture slots)
  + a *shader variant key* (bitflags: vertex-color on/off, texture on/off,
  ramp on/off…). Variants are compiled from one WGSL source with a tiny
  preprocessor (feature `const`s), cached by key. Adding a new *look* means a
  new variant flag, not a new pipeline architecture.
- **Mesh registry:** GPU buffers keyed by `MeshData::content_hash()`. Two
  entities with identical generated meshes share buffers automatically —
  determinism paying rent. Uploads interleave SoA → the renderer `Vertex`
  (unchanged from today).
- **Draw path v1:** per-instance uniform (model matrix + material) with dynamic
  uniform-buffer offsets, sorted by (pipeline variant, mesh) to minimize state
  changes. True GPU instancing (per-instance vertex buffer) is the first
  optimization once the same (mesh, material) repeats a lot — the sort order
  already groups for it.
- **Passes:** clear → opaque forward → (when render scale < 1, see §11, *or*
  when the phase circle is up) a fullscreen nearest blit from the internal
  target to the host's view, which is also where the phase screen effect runs →
  (later, in order of likely need) transparent-sorted pass, simple shadow map
  for the key light (capability-gated resolution), post tonemap/vignette. No
  render-graph framework; a hand-rolled ordered pass list is enough at this
  scale, forever.
- **Camera:** a `Camera` component (projection params) + host-fed viewport
  size. Engine renders exactly one camera per `render()` call — and, since
  2026-08-06, one more per `render_to_texture()` call (see below).
- **Limits:** everything above fits `downlevel_webgl2_defaults()`. Compute,
  storage buffers, and texture arrays live behind the capability gate.

**The transparent pass and instancing landed (2026-08-04, for the 3dimenshift
port).** Both were written above as "later" and both are now the only path.

- **Blend and depth state come out of the variant key** (`render_state`), which
  is what "a new look is a new flag, never a new pipeline architecture" was
  always supposed to mean. Five bits appended permanently from 5: `TRANSPARENT`,
  `ADDITIVE`, `DEPTH_GREATER`, `PHASE_CIRCLE`, `BILLBOARD_UNLIT`. The first two
  are also a *scheduling* mask: `DrawItem::sort_key` leads with a `pass` field,
  extraction partitions blended draws to the end, and the renderer re-orders
  that suffix back-to-front once it holds the camera. Both halves still ride one
  loop, one render pass, one submit — the "transparent-sorted pass" is a
  partition of the pass list, not a second one.
- **Draw path v2 is per-instance vertex data, wholesale.** `InstanceRaw` (96 B:
  model, colour, params) on buffer slot 1, `step_mode: Instance`, shader
  locations 4–9 alongside the mesh's 0–3 — ten attributes of the sixteen WebGL2
  guarantees. The dynamic-uniform-offset path and its 256-byte striding are
  gone, `@group(1)` is now an unused hole so `@group(2)` keeps its number, and
  the fragment stage reads colour and params as `flat` varyings. Adjacent items
  in the sorted list agreeing on (variant, mesh, texture) coalesce into one
  `draw_indexed` over a range of the frame's instance buffer; a singleton is a
  run of one, so there is no second code path and the golden opaque frame is
  byte-identical across the change.
- **`Visibility` and CPU frustum culling** filter either side of the camera.
  `Visibility { visible }` is absent-means-visible and is applied in extraction
  (it is world state, so a replay hides the same things at the same ticks); the
  frustum is applied by the renderer, against a per-mesh object-space AABB
  measured at upload, centre-and-extent against six unnormalized planes pulled
  from `view_proj`. Conservative in the only direction that matters: it may cost
  a draw call, never a pixel. A mesh with no measured bounds is kept.
- **What it bought.** `demo/ball`'s level: 21 draw items → 13 culled → **4 draw
  calls** where there were 21. The 5 000-instance submit microbench costs
  ~0.24 ms of CPU (~48 ns/instance) against a ~1 ms budget. Both filters and the
  coalescer are pure functions of (list, camera, transforms), so the command
  stream is as reproducible as it was — there is simply less of it.

**The shadow map landed (2026-08-10, for the port's ground-contact cue).** The
pass list's "(later)" slot, filled as written: one depth-only pass before the
forward pass, rendering the opaque lit draws through a fixed-size orthographic
box aimed along the key light and centred down the camera's central ray, its
light-space translation snapped to whole texels so a panning camera cannot
shimmer the edges (`shadow.rs`). The main pass samples it through a comparison
sampler (hardware PCF — core WebGL2, §11's floor) and attenuates the **key term
only**; the hemisphere ambient is skylight and stays whole. Receiving is a
*variant bit* (`SHADOW`), renderer-resolved exactly as `LIVE_TEX` is §7's gate:
with the gate closed every pipeline keeps its old key and therefore its old
bytes — which is not a nicety but a scar, because a runtime uniform branch
around the key term moved lit pixels by an LSB with shadows *off*. The gate is
§11's row (`ShadowQuality`: off / 512² / 2048², resource-flipped, default
**off**), the tuning is a reflected `ShadowSettings` resource (box extent,
constant + slope bias), and `render_to_texture` skips the pass outright — a
viewport card is lit pre-shadow rather than paying a per-target map. Off costs
nothing: no texture, no pipeline, no pass, byte-identical frames
(`tests/shadows.rs`).

**Render-to-texture: a second camera, a second world (2026-08-06, for the
port's tutorial cards).** "Exactly one camera per `render()` call" is still
true; there is now a second call.

- **`Renderer::render_to_texture(RenderTarget, w, h, frame, draws, libs)`**
  draws a scene into an offscreen colour+depth target through the *same*
  `encode_scene` the host frame uses — same cull, same sort, same instancing,
  same sky, same pipeline cache (the target carries `target_format`, which is
  what lets one cache serve both). What it deliberately leaves out is
  everything belonging to the *screen* rather than to a scene: no UI pass, no
  render scale, no fullscreen pass, and the phase circle written at rest.
- **`Engine::render_to_texture(target, &mut Sim, w, h)`** is that call for a
  whole second `Sim` — its own `World`, its own clock, its own libraries, like
  Godot's `SubViewport` with `own_world_3d`. Two worlds are safe to share one
  renderer *because* §5's registries are content-addressed: identical geometry
  in two worlds is one upload by construction, and different geometry cannot
  collide. The only non-content identity — the target's own — lives in a
  reserved half of the `TextureHandle` space (bit 63, which `content_key`
  masks off), so "impossible" rather than "improbable".
- **The frame block is per target.** `queue.write_buffer` is ordered against
  *submissions*, not against passes inside one, so a second camera writing the
  one frame uniform would be seen by whichever pass ran last. Each target owns
  a uniform buffer and each render is its own submission; the instance buffer
  is shared and safe on the second property alone.
- **The UI batch can name a texture per run.** A viewport is sampled by an
  ordinary `UiQuad`, so `UiBatch` gained texture *runs*
  (`set_texture`) — consecutive spans, never sorted, one instanced draw each.
  A batch that never switches is one run and encodes the commands it always
  did. That is the whole of §13's "one atlas per batch" being relaxed: the
  atlas is still the default, and runs are the exception a demo card needs.
- **Cost when unused:** nothing. No target, no allocation; a HUD with no runs
  is the same single draw; a frame with no offscreen pass is byte-identical
  (`tests/render_to_texture.rs`, and E2's golden hash in `tests/ui.rs`).

## 6. Content pipeline — params → cache → GPU

**Decision:** generation is always *ahead of* the frame, never in it.

- A generator is a pure function `fn(&Params, Quality) -> MeshData` where
  `Params: Reflect + Serialize + Hash`. The registry maps a stable generator
  name + params hash + quality → content hash → mesh.
- **Content-addressed cache**, two layers: in-memory (hash → GPU buffers, the
  mesh registry above) and on-disk/IndexedDB (hash → serialized `MeshData`)
  so revisits skip regeneration. Cache is *purely* an optimization: deleting
  it must never change output (determinism makes this checkable in CI).
- **Quality tiers:** `Quality(f32)` multiplier selects segment counts etc.
  A device tier (Low/Med/High, probed once at startup from adapter info +
  a micro-benchmark) picks the default quality; per-generator overrides allowed.
  Different quality → different content hash → coexisting LODs for free.
- Generation runs at load/scene-build time on a worker (web worker / thread),
  never per frame. "Live param tweaking" in the editor regenerates on change —
  that is still load-time semantics, just a fast loop.
- **Scene file** = save-as-params: RON listing generator invocations (name +
  params + quality policy) and entity placements (transform, material,
  generator ref). No baked binary scene format until proven necessary.

## 7. Textures — procedural, bake-vs-live

Same philosophy as meshes, one level down the pipeline.

- A procedural texture is a WGSL fragment snippet with a params uniform,
  evaluated in UV space. Two modes, same source:
  - **Baked** (baseline): rendered once to an RGBA8 texture at tier-scaled
    resolution at load time. This is a fragment pass, not compute — it works
    fully within WebGL2 limits (bake resolution capped at 2048² there).
  - **Live** (perf-gated): evaluated per-pixel in the material shader — for
    surfaces where animation/param-morphing matters. The gate is fragment-shader
    *cost* (noise/fBm per pixel per frame vs. one texture sample), **not** the
    backend: WebGL2 runs live eval fine API-wise. Gated on perf tier,
    so a WebGL2 desktop qualifies while a WebGPU phone does not (see §11).
- Noise/hash functions must be precision-robust (integer-style hashing, no
  large-argument `sin` hashes) — cheap mobile GPUs run "highp" as fp24
  internally, on either backend.
- Bake outputs are content-addressed like meshes (params hash → texture).

**Live path landed (2026-08-04).** `MaterialVariant::LIVE_TEX` is implemented,
and the reserved bit paid off exactly as intended: no new flag, no new pipeline
shape, 32 variants before and after.

- **One source, both modes, enforced.** `noise.wgsl` is now prepended to every
  material variant as well as to the bake, and the contrast/brightness curve and
  the gradient ramp moved into it as *parameterized* functions so the two modes
  read the same code from different uniforms. A live fragment and a baked texel
  of one spec are the same number — `tests/live_texture.rs` holds the shader to
  the CPU twin at **sub-LSB** (median 0.36/255, better than the bake, which has
  a texel and a filter in the way).
- **Live is the original's semantics, not a per-pixel bake.** Unbounded 3D
  world-space noise at the shading point: no tile, no wrap, and **no triplanar
  projection** — a 3D field is defined everywhere, so projecting it would throw
  away the third dimension and then pay nine anti-tiling taps per plane to
  disguise the loss. One octave loop, not three.
- **Octave LOD replaces the mip chain.** Live has nothing pre-filtered to fall
  back to, so an octave whose cells are finer than `LIVE_LOD_CELL_PIXELS` across
  fades out and is *skipped*. Driven by `dpdx`/`dpdy` of the world position —
  the footprint, which is what a mip selector uses — rather than by camera
  distance, which is only its cause and cannot see a grazing angle.
- **Normals** accumulate the boundary gradient against the pixel footprint in
  the same octave loop, then perturb the geometric normal in a screen-derived
  tangent frame. The tangent offset is built by normalizing against the bake's
  own packed `z = 0.5`, which bounds it exactly as the bake's packing does — an
  unbounded analytic gradient turns rock's authored `strength: 29.6` inside out.
- **The gate is a per-frame flag** (`Engine::set_live_textures`), and the draw
  list applies it (`draw::resolve_variant`); the material stays declarative.
  `TEXTURE` and `LIVE_TEX` are mutually exclusive on a draw, resolved in live's
  favour in both the draw list and the shader.
- **Cost, measured** (3dimenshift `test_level.ron`, RADV RAPHAEL_MENDOCINO
  iGPU, 300 frames/mode, wall clock with a device drain per frame):
  8.4 ms → 41.6 ms at 1080p (**4.9×**), 17.9 ms → 74.9 ms at 1440p (**4.2×**).
  The octave window is worth ~4% of that on a close third-person framing and
  would be worth more on a vista. All 32 variants emit valid GLSL ES 300, so
  WebGL2 can run it — which is why §11 gates it on tier and not on backend.

## 8. Audio — synthesized, worklet-resident

**Decision (2026-08-01, spike complete — see `spikes/audio/FINDINGS.md`):**
`fundsp` for synthesis, running *inside* an `AudioWorklet` on web and on the
cpal callback natively. The same patch code serves both hosts; the host is a
dumb pump.

- **No `SharedArrayBuffer`, no COOP/COEP, no service worker.** The demo ships
  on GitHub Pages, which cannot set response headers, so cross-origin
  isolation is unavailable and the engine does not ask for it. A SAB
  ring-buffer path was built and measured; it is strictly worse here (160 ms
  of buffering to stay glitch-free, vs zero) and is **not** adopted.
- **The worklet wasm is a separate raw `cdylib`, not a wasm-bindgen module.**
  `AudioWorkletGlobalScope` lacks `TextEncoder`/`TextDecoder`/`fetch`, which
  wasm-bindgen's glue requires. A C-ABI module over linear memory needs none
  of them and has zero imports. The main thread compiles the module and
  passes the structured-cloneable `WebAssembly.Module` through
  `processorOptions`; the processor instantiates it synchronously and is live
  on its first `process()`. Measured: 200 KB raw / 84 KB gzip at two patch
  models (2026-08-01); 343 KB / 125 KB gzip at six (with drums + additive
  bass, 2026-08-04) — still fetched lazily on first audio start, never
  blocking first frame.
- **Cost:** 12 µs per 128-frame stereo quantum in wasm (1.19× native) against
  a 2 666 µs budget — 0.45% of one core. Voice count is bounded by design
  taste, not CPU.
- **Control path:** params and note triggers cross as `postMessage` records,
  drained at the top of `process()`. Measured round trip median 6.6 ms, p95
  10.8 ms. Event→sound is dominated by the OS audio stack; ~25–55 ms
  end-to-end on real hardware — acceptable for game SFX as-is.
- **Audio params are content**, like generators (§6): a patch is a
  `Reflect + Serialize + Hash` param struct plus an explicit seed, serialized
  in scene RON, edited through the same reflected panels as mesh generators
  (§10), content-addressed by params hash.
- **Determinism scope — same as sim (§4):** same params + seed + build +
  platform → bit-identical samples (verified across processes). Cross-platform
  bit-identity is **not** promised and does not hold (~1e-10 libm divergence
  through IIR state; perceptually identical).
- **Low-end story (§3):** one code path everywhere; the tier knob, if ever
  needed, is voice count / oversampling — data, not divergent code. No
  capability gate: AudioWorklet exists in every browser runt targets.
- Sim-side seam: an `AudioOut` resource queues `AudioEvent`s during `FixedSim`
  and flushes once per tick — never mid-tick, so replays stay deterministic.
  Hosts implement one `AudioBackend::submit(&[AudioEvent])` trait; web
  serializes to `postMessage`, native pushes to an SPSC queue read by the
  cpal callback.
- Phase-3 order: `runt-audio` crate (Patch trait + pluck/drone + offline hash
  test) → worklet cdylib + trunk wiring (recipe in FINDINGS) → `AudioOut` +
  pickup sound in the v0 demo → voice pool with stealing + master limiter +
  camera-relative pan → editor patch panels with audition button.
- **Not doing:** sample playback pipelines, music sequencing, convolution
  reverb, HRTF/ambisonics, SAB in any form.

## 9. Physics — hand-rolled kinematic

**Decision (2026-07-31):** no physics crate. Collision and motion are ordinary
`FixedSim` systems in `runt-core`.

- **Terrain is analytic, not mesh.** Terrain generators are built around a pure
  height field `h(x, z)` (+ gradient). The rendered mesh is a *view* of that
  field; collision samples the field directly. Nothing ever collides with
  triangles. This makes terrain collision deterministic, cheap, and exact at
  every quality tier — visual LOD cannot change physics.
- **Ball/character motion:** semi-implicit Euler point integrator at tick rate.
  Gravity, slope response from the field gradient, rolling friction + air
  damping as params, clamped restitution on steep contacts. Visual spin is
  derived from velocity (cosmetic, never simulated state).
- **Discrete shapes:** sphere-vs-sphere and sphere-vs-AABB overlap only
  (pickups, obstacles, triggers), kinematic push-out, no impulse exchange.
- **Not doing:** dynamic-dynamic response, stacking, joints, arbitrary mesh
  colliders. A game that needs these forces a doctrine revisit (likely
  rapier3d behind a feature flag), not an incremental slide into one.

### 9a. Collision v2 (2026-08-04, for the 3dimenshift port)

The revisit the clause above asked for, taken deliberately. Porting a Godot
platformer (`3dimenshift-runt/docs/PORT_SPEC.md`) needs a moveset the point
integrator cannot express — 14 land states built on `move_and_slide`,
per-contact normals, runtime `floor_max_angle` mutation, shape queries with
layer masks. The answer is **not** rapier3d: everything the moveset needs is a
few hundred lines of closest-point math, and the §9 properties (analytic
terrain, tick-rate determinism, no mesh collision) survive intact. It is a
sibling module, `runt-core/src/collide.rs`; `physics.rs` is untouched and the
demo's pinned 240-tick fingerprint is unchanged.

**Added:**

- **Capsule character solver.** `move_and_slide(&CollisionWorld, &mut
  CharacterBody, position, velocity, dt) -> MoveResult`. A **library function**,
  not a system: game code calls it from its own `FixedSim` system and owns the
  position and velocity it passes in — there is no second copy of the simulation
  state in a component. Discrete, iterative: translate, collect contacts, push
  the deepest one out, project velocity onto every contact plane, repeat up to
  five times. The moving shape is a swept sphere along a **vertical** segment —
  capsule or, degenerately, sphere — so the port's runtime capsule↔sphere roll
  swap is one enum field. Contacts classify as floor / wall / ceiling against a
  per-body `max_floor_angle` (45° standing, 89° slam, 180° rolling), and Godot's
  floor snap is reproduced: probe down `snap_length`, land straight along `up`,
  keep horizontal velocity.
- **`CharacterBody::floor_stop_on_slope`, default `true`** — Godot's flag, and
  its condition: on floor, and the velocity is gravity and nothing else
  (`(v.normalized() + up).length() < 0.01`, their literal). Projecting velocity
  onto the floor plane is right for a body going somewhere and wrong for one
  standing still — the tick's gravity comes back as downhill tangential velocity
  and the caller's friction is left fighting a force the solver invented, which
  walked the port's Idle 0.73 m down its 15° slope in five seconds. The stop
  cancels that motion and zeroes the velocity, so a standing body is bit-stable.
  It cancels *only* the sub-step the floor absorbed whole; a body that genuinely
  fell, or one still coming out of a penetration, keeps the ordinary push-out, so
  the stop can never freeze an overlap in place. Steep faces are walls, walls
  never stop a body, and `false` restores the old behaviour exactly.
- **`ObbCollider { half_extents, rotation: Quat }`** — full rotation, not the
  yaw-only box originally planned. The contact solve happens in the box's own
  frame, where orientation has already been divided out, so a pitched ramp costs
  what a yawed wall costs and restricting to yaw bought nothing. `AabbCollider`
  stays as the zero-rotation fast path (a vertical segment against an
  axis-aligned box has a closed-form contact point; an OBB needs a search).
  Authored in a scene as `obb_collider`, rotation in Euler degrees exactly like a
  transform.
- **`CollisionLayers { memberships: u16, mask: u16 }`**, one-way and query-side:
  *a collider is visible to a query iff `query_mask & collider.memberships != 0`*.
  That is Godot's rule (`A.collision_mask & B.collision_layer`, evaluated from
  the mover's side), not a symmetric both-must-agree variant — the symmetric
  form breaks the port's phase mechanic, which mutates only the player's mask and
  expects static world geometry to become passable. Absent component =
  `{ memberships: layer 0, mask: all }`, so every scene written before layers
  behaves identically. Mutable from any system; a `CollisionWorld` is a value
  taken once per solve, so a mask write can never land mid-tick.
- **Queries:** `overlap_sphere`, `overlap_capsule`, `overlap_body` and `raycast`,
  all mask-filtered, all reporting triggers with a flag rather than hiding them.
  The raycast is exact against boxes and spheres (slab test in the box frame) and
  a fixed-step march plus fixed-count bisection against the analytic height
  field — never its mesh, so §9's tessellation-independence claim extends to
  rays. The scan is linear over an `Entity`-sorted `Vec`; the one method it goes
  through is the seam a spatial index would replace.

- **Trimesh colliders (static only), 2026-08-04 — the full-port revisit.**
  Previously refused here; green-lit for the playground port. CSG-baked
  geometry has no analytic form, and the N64 port's verdict (triangle soup +
  BVH for the static world, `dimenshift64/src/world.c`) was already
  acknowledged as the endgame. Constraints that keep §9's properties intact:
  a `Trimesh` is immutable after `build()` (welded verts, degenerates
  dropped); the BVH is built once at load, deterministically — median split
  on the longest centroid axis, stable sort keyed `(axis value, tri index)`,
  fixed leaf size; traversal uses a fixed-size explicit stack and visits
  children in a fixed order; contact and raycast ties resolve to the lowest
  triangle index. Contacts feed the same `Contact` pipeline —
  classification, snap, and `floor_stop_on_slope` are untouched downstream.
  Dynamic trimeshes, convex decomposition, and mesh-vs-mesh stay refused.

**Still refused:**

- Dynamic-dynamic response, impulse exchange, stacking, joints. A solve moves
  the character and only the character; the other body never learns it was hit.
- Swept CCD. Motion is capped per sub-step at the moving shape's radius, which is
  half the no-gap bound (`2·radius`) — the port's fastest motion is a 30 m/s slam
  (0.5 m/tick, two sub-steps) against 0.5 m-thick walls that would need 1.2 m of
  travel to tunnel.
- Rotated or scaled terrain, and non-vertical capsules. Both are the same refusal
  as v1's: a rotated height field is not a height field.

**Determinism, extended to the new solver** (DESIGN §3, §4 still govern):

- `CollisionWorld` sorts colliders and terrain patches by `Entity` at
  construction and every scan walks that order. No hash container is iterated.
- Contact selection takes the greatest depth, ties to the lowest `Entity`;
  velocity is projected against contacts in `Entity` order; the "which floor is
  *the* floor" rule is most-upright-wins, ties to lowest `Entity`.
- Every loop bound is a compile-time constant — slide iterations, sub-steps, the
  segment/box ternary search, the ray march and its bisections. Nothing iterates
  to an error threshold that a different machine could reach on a different step.
- Sub-step count is a function of the *entry* velocity and `dt` alone, so what a
  tick collides with cannot change how it was integrated.
- The whole module is pure: same snapshot + same position/velocity ⇒ the same
  `MoveResult`, bit for bit, under any host frame cadence.

## 10. Editor — native rinch app

**Decision:** the editor is a native rinch application (`runt-editor`) using
rinch's `RenderSurface`; the engine renders offscreen and submits frames.

- **Why rinch:** it's ours; `RenderSurface` is purpose-built for editor-with-
  viewport apps (thread-safe frame submission, input events routed back with
  surface-local coordinates, real component library for panels).
- **Version reality:** rinch pins a forked wgpu 27 / winit 0.31-beta; runt is
  on wgpu 30 / winit 0.30. Therefore v1 uses the **CPU bridge**: runt-core
  renders on its *own* wgpu device to an offscreen texture, reads back RGBA8,
  and submits via `SurfaceWriter`. Version-independent, fast enough for an
  editor. When rinch and runt converge on a wgpu major, switch to
  `GpuTextureRegistrar` zero-copy. The bridge is one small module; nothing
  else in the editor knows which path is active.
- **Input:** `SurfaceEvent`s from rinch translate to the same engine input
  events the player hosts produce — the engine cannot tell editor from game.
- **Panels** are generated from `Reflect` param structs: a `Reflect`-walking
  widget mapper (f32 → slider with range attributes, enum → select, Vec3 →
  triple, seed → reroll button). Hand-written panels only where reflection
  genuinely can't express the interaction.
- **Scope guard:** param panels + scene arrangement (place/transform/duplicate
  instances, pick via ray from surface mouse events) + save/load scene RON.
  Not a DCC tool: no mesh editing, no timeline, no node canvas v1 (the fluent
  op API maps to a node graph *later* if ever).
- Web editor: explicitly out. Browser is the *player*. (rinch's DOM backend
  can't host `RenderSurface`, and a web editor serves no pillar.)

### 10a. Tooling is in-game, not in rinch (2026-08-06)

**Decision:** the tooling direction above is superseded. There is no
`runt-editor` rinch application; tools live **inside the running game**, drawn
with the engine's own `UiBatch` and driven by the engine's own `Input`.

Three things pushed it over:

- **The CPU bridge was the whole cost of rinch.** A readback per frame to
  serve a viewport is a price paid forever for a version skew that may never
  close, and everything §10 wanted out of rinch — panels over `Reflect` params,
  a viewport, input routed back — the game already has natively. The game *is*
  the viewport.
- **The tuning loop is the point.** A slider is worth having because you drag
  it while standing under the sky it changes. An editor in another window that
  re-renders the same scene through a bridge is a worse version of the thing
  the game does for free at 60 fps, and it cannot be opened on a Steam Deck.
- **It costs one feature, not one crate.** The player must not carry any of
  it, and `reflect` is already the flag that says so.

**Where it landed:** `runt_core::tweak` is the registry (opt-in roots →
`Reflect`-walked dotted paths → clamped edits → a serializable overrides map),
`runt_core::tweak_panel` is the debug overlay that drives it, and both are
behind `reflect` so a wasm build compiles them away entirely. The panel takes a
`PanelFont` from the host, because `UiBatch` is quads and glyph pixels are game
content.

**Amended 2026-08-07 (§10a-font):** this section used to say "the one thing the
engine cannot supply is a font". That was two claims wearing one coat, and only
the second survives:

- The engine **can** supply text *layout*, and now does. `runt_core::font` is a
  `BitmapFont` — glyph rects, bearings, advances, a sorted codepoint map,
  optional kerning — plus `width` / `ink_width` / `line_height` / `text` /
  `text_centered` / `text_right` / `wrap` / `shape` over it, and a blanket
  `impl PanelFont for BitmapFont`. `PanelFont` stays a trait so a game with some
  other idea of text can still be one, but no game has to write the impl any
  more; 3dimenshift's eight-line `AtlasFont` shim is deleted.
- The engine **must not** supply a *typeface*, and still does not. Letterforms
  are game content the way an albedo texture is. The one exception is
  `font::micro`, 102 glyphs of 8 × 8 behind the off-by-default `default-font`
  feature — 816 bytes so a new game has *some* text before it picks a face, and
  so the baker has something to substitute for a codepoint a real face lacks.

The decoder and the rasterizer live in **`tools/font-bake`**, a workspace member
that is deliberately not a *default* member (the convention 3dimenshift's
manifest already documents for its `tools/*`). It takes a TTF/OTF, a charset and
a list of `SCALE:PX` sizes and writes a postcard `FontAsset`: one atlas of 8-bit
coverage, one glyph table per size. It is deterministic — same bytes in, same
bytes out — because headless screenshot tests depend on stable pixels. `ab_glyph`
is a dependency of that binary and of nothing else in the tree, which is the
whole point: `cargo tree --target wasm32-unknown-unknown` for a player finds no
rasterizer and no font file, only rectangles.

Three constraints shaped it. Sizes are baked **per size**, never scaled, because
the UI sampler is `Nearest` with `filterable: false` (§11's downlevel WebGL2
floor) and a resampled glyph looks like it. All sizes share **one atlas image**,
because `TextureRegistry::insert_image` is idempotent by handle. And
`tweak_panel::row_pitch()` stays `font::UNIT · SCALE + 2` — the same 18 pixels it
always was — because a game switching typefaces must not move anybody's finger.

What §10 still holds for: **panels are generated from `Reflect`, with bounds
declared at the param** (`FieldRange`). That was the load-bearing claim and it
survives the venue change intact. What is dropped is rinch, the CPU bridge, and
`runt-editor`. Scene arrangement — place/transform/pick, save scene RON — is
still wanted and is still unbuilt; it lands in-game too when it lands.

**Scene arrangement landed** (`runt_core::editor` + `editor_gizmo`, behind an
`editor` feature that implies `reflect`). The doctrine it is built on, which is
the part worth restating here rather than in the module:

> **Every edit is a data operation on the scene description, never a direct
> world poke.**

An edit does not write a `Transform`; it applies an *op* to the thing the level
file is made of, and the game then re-realizes the affected world. Undo/redo,
saving, and one day shipping level editing to a player all fall out of that one
decision instead of each needing machinery: the op log **is** the undo stack (a
`Vec` and a cursor; undo is `apply(invert(op))`), and the thing being edited is
already the thing that gets written out, so there is no export step that could
disagree with the screen.

The engine half is game-agnostic and almost entirely pure functions — screen
rays, a gizmo hit test, drag maths, snapping, a bounded op log. What a level
*is* sits behind one trait, `EditableScene`, whose associated `Id` must be
stable across a reload and whose associated `Op` must carry enough to invert
exactly. The engine deliberately owns **no** editor resource and **no** editor
system: every piece of the state is parameterized by the game's id type, so the
resource is the game's and the engine hands it the pieces. `EditorState::dirty`
is the whole hot-reload contract — the engine says *that* the description
changed and never how much of the world to rebuild, because a full respawn and a
surgical one are both correct and only the game knows which it can afford.

The pointer is the editor's own, moved by `Input::mouse_delta` and clamped to
the viewport, because runt has no cursor seam at all and a delta-driven cursor
is the one shape that is identical on native and web *and* survives a trace
replay.

## 11. Capability gates

Two **independent** axes, probed once at startup, stored as a resource,
immutable for the session:

- **Backend capability** — what the API can do at all. Boolean per feature,
  derived from adapter limits/features (compute, storage buffers, texture
  size). WebGL2 fallback fails these; real WebGPU/native passes them.
- **Perf tier** (Low/Med/High) — what the device can afford per frame.
  Probed from adapter info + a startup micro-benchmark. Independent of
  backend: a desktop on WebGL2 fallback can be High; a phone with real
  WebGPU is still Low.

| Feature | Gate axis | Low / fail | High / pass |
|---|---|---|---|
| Limits requested | backend | `downlevel_webgl2_defaults` | full adapter limits |
| Mesh quality default | perf | tier-scaled `Quality` | higher default quality |
| Render scale (below) | perf | 0.5 or lower | 1.0 |
| Live texture eval (§7) | perf | baked only | live variants allowed |
| *(v1: an explicit `set_live_textures` flag, default off — see below)* | | | |
| Bake resolution | both | ≤2048², tier-scaled | up to 4096² |
| Particles | backend (compute) | CPU, modest counts | GPU compute |
| Shadows | perf | off or 512² single cascade | 2048² |

Gates select *data and variants*, never divergent engine code paths. A gated
feature must have a baseline degradation story before it merges.

**There is no perf probe yet.** The tier column above describes where the
decisions belong, not machinery that exists: nothing measures the device at
startup. Live texture eval (§7) is the first feature to actually need it, and
what shipped instead is the API the probe will eventually drive —
`Engine::set_live_textures(bool)`, default **off**, flipped by hand (the port
binds it to **T**). When the probe lands it sets that flag at startup and no
call site moves.

**Recommended policy for when it does**, from §7's measured 4–5× fragment cost:

- **Off by default at every tier.** Baked already carries a mip chain and
  anti-tiling and looks the same; live's payoff is *animation and param
  morphing*, and no content asks for either yet. A gate that costs 4× and buys
  nothing visible should be closed however fast the part is.
- **A tier is a necessary condition, not a sufficient one.** Live belongs on
  High as an opt-in the *content* asks for (`MaterialDesc::live_texture`), with
  the tier able to refuse. That is what the "gate can only say yes to more"
  rule in `draw::resolve_variant` is holding a place for.
- **The High bar, if anyone wants a number**: this iGPU spends 42 ms of a 1080p
  frame on live and would need roughly 5× that throughput to spend live's cost
  where baked spends its own. So the honest threshold is "a discrete GPU, at
  1080p, for a scene where live is not covering most of the screen" — which is
  another way of saying the gate wants to be per-material rather than global.

### Render scale — the blunt lever (2026-08-04)

Every gate above chooses *what* to draw. This one chooses **how many pixels to
draw it into**, which is the only lever whose payoff is quadratic and whose
content cost is zero:

- `Engine::set_render_scale(f32)`, clamped to `[0.1, 1.0]`, default 1.0. The
  scene is drawn into an internal color+depth target of `round(scale × view)`
  (half up, floor of one pixel) and blitted to the host's view with a
  **nearest** sampler. Nearest, not linear: a half-resolution frame smeared
  bilinearly is a blur that costs the same, and chonky pixels are a look.
- **At 1.0 nothing happens** — no internal target, no blit pipeline compiled,
  the same single submit into the host's view, and pixels bit-identical to
  what the renderer produced before the feature existed. That is a test
  (`tests/render_scale.rs`), not an intention.
- **Unless the phase circle is up** (2026-08-06). The screen effect —
  luminance inverted and 40% desaturated inside the circle, the port's
  signature look — has to read the finished frame, and the fullscreen pass is
  the only place that holds one, so it rides this one rather than adding a
  second. The consequence is that a *native-resolution* frame with a circle in
  it now takes the offscreen detour too: one full-screen read and write it did
  not use to pay, charged only while the circle is actually up, and gone again
  at radius zero — where the old path is still bit-identical
  (`tests/phase_screen.rs`). The HUD escapes it for free, because the UI pass
  is encoded after this one and straight onto the host's view.
- The value is a `RenderScale` **resource**, like §7's live-texture switch, so
  a game can bind it to a key from one `FixedSim` system and have it work on
  both hosts. Nothing in a tick reads it, so no fingerprint can move when it
  changes. `RenderScale::STEPS` is the Godot-shaped ladder a UI walks:
  0.25, 1/3, 0.5, 0.75, 1.0.
- **On web it multiplies with device pixel ratio**, which is where the win is:
  the host configures its surface at CSS pixels × DPR, so the fragments drawn
  are `css_w × css_h × DPR² × scale²`. A DPR-3 phone at 0.5 is still rendering
  at 1.5× CSS density — visibly sharper than a desktop at 1.0 — for a quarter
  of the phone's native fragment cost.
- **Why it is the first thing the probe should reach for:** §7's live path
  costs 4–5× in the fragment shader, and 0.5 render scale gives 4× back on
  *any* fragment work, live or baked, with no content decision attached to it.

## 12. Build order

Each step is shippable and exercises the previous one. v0 target: a playable
**rolling-ball collector** demo (procedural terrain, roll to collect pickups,
seeded deterministic runs) in browser + native.

1. **Workspace split** (§2) + headless render-to-texture + screenshot test.
2. **ECS world + fixed tick** (§3, §4): port demo scene to entities; spinning
   handled by a sim system; interpolation proven with a tick-rate toggle;
   host→engine input events (keyboard + mouse).
3. **Renderer v1** (§5): mesh registry, material variants, per-entity draws
   replacing the merged-buffer demo; follow camera.
4. **Generator registry + cache** (§6): `MeshRef` resolves through content
   hash; disk/IndexedDB cache; scene RON load/save; **heightfield terrain
   generator** (pure `h(x,z)` field → mesh, per §9).
5. **Physics** (§9): field-sampling ball integrator + overlap shapes, as
   `FixedSim` systems.
6. **v0 demo game** (`demo/ball/`): terrain + procedural props + pickups,
   follow camera, scoring, seeded runs replayable from an input trace.
   Ships web + native.
7. **Phase 2:** editor v1 (§10), audio spike (§8), procedural textures baked
   path (§7), shadows, transparent pass, instancing — order per content needs.

## 13. Open questions (tracked, not blocking)

- **Picking:** CPU ray vs. ID-buffer readback for editor selection. Decide
  when editor work starts (phase 2); CPU ray against generated tri data is
  likely enough.
- **Worker story on web** for generation (§6): plain web worker with message
  passing vs. SharedArrayBuffer. Audio no longer shares this question (§8
  needs no isolation); if generation-on-a-worker later wants SAB it must
  justify coi-serviceworker's costs (first-load reload, COEP embed breakage)
  on its own. Default assumption: message passing suffices.
- ~~**Text/HUD rendering** in the player (not editor): none designed. Cheapest
  candidate: DOM overlay on web, nothing native, until a real need appears.~~
  **Closed 2026-08-07.** Not a DOM overlay: it is in the same `UiBatch` as
  everything else, so it works native, headless and in a screenshot test, which
  a DOM overlay does not. `runt_core::font` is the layout, `tools/font-bake` is
  the build-time rasterizer, and the pixels belong to the game — see §10a's
  amendment. What is still *not* designed, and is deliberately out of scope:
  runtime rasterization, growable atlases, an R8 atlas format
  (`bake::BAKE_FORMAT` stays `Rgba8Unorm`), a linear-sampled pipeline variant,
  and any of shaping, bidi or ligatures. Greedy whitespace `wrap()` is the whole
  paragraph engine.
- **rinch wgpu convergence:** revisit the GPU bridge when rinch leaves the
  wgpu-27 fork.
