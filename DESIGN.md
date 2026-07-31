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
- **Passes:** clear → opaque forward → (later, in order of likely need)
  transparent-sorted pass, simple shadow map for the key light (capability-
  gated resolution), post tonemap/vignette. No render-graph framework; a
  hand-rolled ordered pass list is enough at this scale, forever.
- **Camera:** a `Camera` component (projection params) + host-fed viewport
  size. Engine renders exactly one camera per `render()` call v1.
- **Limits:** everything above fits `downlevel_webgl2_defaults()`. Compute,
  storage buffers, and texture arrays live behind the capability gate.

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
- v1 ships baked-only; the material variant key already reserves a bit for
  live eval so shaders don't need restructuring later.

## 8. Audio — synthesized, de-risk first

- `fundsp` for synthesis, rendered in an `AudioWorklet` on web / cpal native.
- **Known risk, unresolved:** AudioWorklet + wasm module loading + COOP/COEP
  headers (if `SharedArrayBuffer` is used for the audio ring buffer). This is
  the next *spike* before any audio feature work: a page that plays a fundsp
  patch through AudioWorklet with the same trunk build we ship. Outcome
  updates this section.
- Audio params are content too: patches are param structs, seeded, serialized
  in scene files like generators.

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
| Live texture eval (§7) | perf | baked only | live variants allowed |
| Bake resolution | both | ≤2048², tier-scaled | up to 4096² |
| Particles | backend (compute) | CPU, modest counts | GPU compute |
| Shadows | perf | off or 512² single cascade | 2048² |

Gates select *data and variants*, never divergent engine code paths. A gated
feature must have a baseline degradation story before it merges.

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
  passing vs. SharedArrayBuffer. Interacts with the audio COOP/COEP question —
  resolve both in the audio spike.
- **Text/HUD rendering** in the player (not editor): none designed. Cheapest
  candidate: DOM overlay on web, nothing native, until a real need appears.
- **rinch wgpu convergence:** revisit the GPU bridge when rinch leaves the
  wgpu-27 fork.
