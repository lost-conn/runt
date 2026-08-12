// Stylized forward opaque shading (DESIGN §5): directional key light +
// hemisphere ambient, no PBR anywhere.
//
// This file is NOT standalone WGSL. `runt_core::material::variant_source`
// prepends two things: the `F_*` feature consts it branches on (`const bool`,
// so every disabled branch folds away at compile time), and `noise.wgsl` — the
// procedural-noise library the live-texture variant evaluates per pixel, and
// the identical library the bake pass runs offline.

// Field order must match `runt_core::FrameUniform`, and `sky.wgsl` restates the
// same block for the background pass.
struct Frame {
    view_proj: mat4x4<f32>,
    // Only the sky pass reads this one; it lives in the shared block because
    // there is exactly one frame uniform and splitting it to save 64 bytes
    // would cost a second bind group.
    inv_view_proj: mat4x4<f32>,
    // xyz: direction *towards* the key light, world space. w: unused.
    light_dir: vec4<f32>,
    light_color: vec4<f32>,
    sky_color: vec4<f32>,
    ground_color: vec4<f32>,
    horizon_color: vec4<f32>,
    // The screen-space phase circle. xy: centre in NDC (+Y up). z: radius in
    // NDC-Y units, the X offset aspect-corrected so the disc is round on
    // screen. w: effect strength, 0..1, which drives the edge fringe only.
    // A zero radius is a circle nothing is inside — the resting state.
    phase: vec4<f32>,
    // x: the render clock in seconds (never a simulation input). y: this
    // frame's interpolation alpha. zw: reserved.
    time: vec4<f32>,
    // xy: the render target's size in pixels. zw: its reciprocal. The *render*
    // target's, so `position.xy * viewport.zw` is a fragment's place in the
    // frame at any render scale.
    viewport: vec4<f32>,
    // x: cloud cover. y: sun-disk size. zw: reserved. The sky pass's, and read
    // by nothing here — restated because the block is one buffer with one
    // layout and all three files move together or it is silently misaligned.
    sky_params: vec4<f32>,
    // World → the key light's clip space, for the shadow-map lookup. Identity
    // while shadows are off, and never read then: shadow_params.x gates it.
    light_view_proj: mat4x4<f32>,
    // x: 1.0 while a shadow map is bound, 0.0 otherwise. y: constant depth
    // bias. z: slope-scaled depth bias. w: the rim fade band's width in map-uv
    // units (floored above zero at upload; see `shadow_factor`).
    shadow_params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> frame: Frame;

// The key light's shadow map, riding in the frame group (bindings 1–2; group 1
// stays the documented hole below). Bound for every variant — the 1×1 dummy
// while shadows are off — and sampled only on the lit path, only while
// `shadow_params.x` says a real map is there. The sampler is a *comparison*
// sampler: one tap answers "is this fragment nearer the light than what the
// map saw?", averaged 2×2 by the hardware (PCF) because the sampler filters
// linearly. Both halves are core WebGL2 (DESIGN §11).
@group(0) @binding(1) var t_shadow: texture_depth_2d;
@group(0) @binding(2) var s_shadow: sampler_comparison;

// Per-entity data arrives as **vertex attributes** on buffer slot 1, stepped
// per instance (DESIGN §5's first sanctioned optimization; `runt_core::
// InstanceRaw`). It used to be a uniform at `@group(1)` addressed with a
// dynamic offset — one bind-group set and 256 bytes of stride per entity — and
// `@group(1)` is now deliberately empty so `@group(2)` below keeps its number.
//
// Locations 4–7 are the model matrix's four columns: a vertex attribute is at
// most a `vec4`, so a `mat4x4` costs four of them, and the vertex shader
// reassembles it. 8 is the base colour, 9 the material params.

// The baked procedural texture (DESIGN §7). Present in the pipeline layout for
// every variant — an untextured draw binds a 1×1 white albedo and a 1×1 flat
// normal, so the bit-unset path is provably the pre-texture look rather than
// merely intended to be.
@group(2) @binding(0) var t_albedo: texture_2d<f32>;
@group(2) @binding(1) var t_normal: texture_2d<f32>;
@group(2) @binding(2) var t_sampler: sampler;

// Params of the *texture*, not the instance: they describe how this material
// maps onto the world, which two entities wearing it must agree on.
//
// Field order must match `runt_core::bake::TextureUniform`. Everything from
// `mode` down is `runt_core::BakeUniform` restated verbatim — the same block
// the bake pass reads out of its own `@group(0)` — because the live path
// (`F_LIVE_TEX`) evaluates the very same `TextureSpec` here in the fragment
// shader. One spec, one packing, two consumers; that is DESIGN §7's "same
// source, two modes" made concrete on the CPU side as well as in WGSL.
//
// A baked-only draw reads exactly one `vec4` of this and the rest is inert.
// Binding it anyway (rather than a second, smaller uniform for the baked case)
// keeps `@group(2)`'s layout independent of the variant key, which is what lets
// the toggle in §7's gate swap a pipeline without rebuilding a bind group.
struct TexParams {
    // x: world units → tile units, y: triplanar sharpness, z: anti-tiling
    // on/off, w: octave-0 lattice cells per world metre (live only).
    config: vec4<f32>,
    // x: log2 of the quantized lacunarity, y: the pixel width at which an
    // octave starts to fade (0 = live octave-LOD off), zw: unused. Live only.
    live: vec4<f32>,
    // The seed offset, folded into the tile so that the live field and the
    // baked one are the same picture (`TextureSpec::live_seed_offset`). Not
    // `seed` below, which is the bake's unfolded one.
    live_seed: vec4<f32>,
    // --- `BakeUniform` from here down -------------------------------------
    mode: vec4<u32>,
    counts: vec4<u32>,
    shape: vec4<f32>,
    normal: vec4<f32>,
    seed: vec4<f32>,
    octaves: array<vec4<f32>, 8>,
    ramp: array<vec4<f32>, 8>,
};
@group(2) @binding(3) var<uniform> tex: TexParams;

// `F_VERTEX_WAVE`'s two crossed-sine ratios. Constants in
// `fx/water.gdshader:29-30` rather than uniforms, and constants here for the
// same reason: they are what makes the pair read as a *crossed* swell instead
// of one diagonal wave, and an author who could retune them would only ever
// retune them wrong.
const WAVE_CROSS_FREQ: f32 = 1.3;
const WAVE_CROSS_SPEED: f32 = 0.85;

struct VSOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    // World position, for the triplanar sample. There is no UV in the texture
    // path at all: the original is a world-space shader and CSG-ish generated
    // geometry has no UV worth trusting, so the world *is* the parameterization.
    @location(2) world_pos: vec3<f32>,
    // The two per-instance values the *fragment* stage needs, carried across as
    // `flat` varyings now that they are attributes rather than a uniform.
    //
    // Flat, not interpolated, and that is not a micro-optimization: the value is
    // constant over the primitive, and a barycentric blend of three identical
    // floats is only *approximately* that float. `flat` passes the bits through
    // unchanged, which is what makes moving off the uniform a visual no-op
    // rather than a one-LSB drift across every surface in the frame. WebGL2
    // (GLSL ES 3.00) has `flat` as core; the provoking-vertex difference between
    // GLES and WebGPU cannot be observed on a value all three vertices share.
    @location(3) @interpolate(flat) base_color: vec4<f32>,
    @location(4) @interpolate(flat) params: vec4<f32>,
    // The mesh's own parameterization. Only `F_EMISSIVE_SWEEP` reads it today —
    // everything else in here is world-space by doctrine (see `world_pos`) —
    // and it is carried unconditionally rather than behind the flag because a
    // varying cannot be conditional in WGSL and one more `vec2` interpolant is
    // below the noise floor of a pass that already passes two flat `vec4`s.
    @location(5) uv: vec2<f32>,
    // The same point as `world_pos`, before the model matrix — the sampling
    // basis `F_LOCAL_SPACE` swaps in so that a procedural pattern belongs to the
    // object rather than to the world (`MaterialVariant::LOCAL_SPACE`).
    //
    // **Unconditional, and it cannot be otherwise.** The preprocessor's `F_*`
    // are `const bool`s — values, not `#ifdef`s — and a struct member is not
    // inside any control flow for a value to gate. WGSL has no conditional
    // compilation, no `@location` attribute that takes a condition, and no
    // `@must_use`-style pruning of an unread varying that the *interface*
    // declares; so a varying is declared for every variant or for none. `uv`
    // above is the same concession made for the same reason, and this one is the
    // more expensive half of it: three interpolated floats on every draw in the
    // engine, read by two variants.
    //
    // The two ways out are worse in kind, not merely in degree:
    //
    //   * **Reuse `world_pos`** — write the local point into it when the bit is
    //     set and carry nothing new. It cannot be done: `shadow_factor` needs
    //     the *world* point in the same fragment, and `F_SHADOW` is ORed onto
    //     every lit draw by the renderer (`resolve_shadow_variant`), so a
    //     local-space lit surface would look its shadow up in the wrong space
    //     and acquire a self-shadow pattern that travels with it. One varying is
    //     cheaper than that bug, and the bug would only appear once the shadow
    //     gate was open.
    //   * **Undo the transform in the fragment stage** — carry the model
    //     matrix's four columns as `flat` varyings (sixteen floats, not three)
    //     and invert per fragment, or upload a precomputed inverse and grow
    //     `InstanceRaw` by 64 bytes per entity per frame. Both cost more than
    //     the thing they avoid, and the vertex stage already has the
    //     pre-transform position in a register.
    //
    // The varying budget is not close: this is location 6 of the sixteen `vec4`
    // slots GLSL ES 3.00 guarantees, which is the downlevel target DESIGN §11
    // holds the renderer to.
    @location(6) local_pos: vec3<f32>,
};

@vertex
fn vs_main(
    // Slot 0 — the mesh, one step per vertex.
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
    // Slot 1 — the instance, one step per entity.
    @location(4) m0: vec4<f32>,
    @location(5) m1: vec4<f32>,
    @location(6) m2: vec4<f32>,
    @location(7) m3: vec4<f32>,
    @location(8) base_color: vec4<f32>,
    @location(9) params: vec4<f32>,
) -> VSOut {
    var out: VSOut;
    // Column-major, the order `glam` stores a `Mat4` and the order
    // `InstanceRaw` writes one — so this is the same matrix the uniform held,
    // and the multiply below is the same arithmetic on the same bits.
    let model = mat4x4<f32>(m0, m1, m2, m3);
    var local = pos;
    if (F_VERTEX_WAVE) {
        // `fx/water.gdshader:27-33`, verbatim: the two sines are sampled in
        // *world* space (so a ribbon and the pond it falls into share one wave
        // field rather than each swaying in its own frame) and the displacement
        // is applied to the **local** vertex, which is what Godot's `VERTEX.y +=`
        // does after it has read `MODEL_MATRIX * VERTEX` for the phase.
        //
        // `frame.time.x` is the render clock and only ever that: this moves
        // pixels, never state (see `MaterialVariant::VERTEX_WAVE`).
        let wp = (model * vec4<f32>(pos, 1.0)).xyz;
        let amplitude = params.x;
        let frequency = params.y;
        let speed = params.z;
        let w = sin(wp.x * frequency + frame.time.x * speed)
            + sin(wp.z * frequency * WAVE_CROSS_FREQ + frame.time.x * speed * WAVE_CROSS_SPEED);
        local.y = local.y + w * amplitude * 0.5;
    }
    let world = model * vec4<f32>(local, 1.0);
    out.clip = frame.view_proj * world;
    // The *displaced* position, where Godot's water shader carried the
    // undisplaced one across to its fragment stage. The difference is confined
    // to the Y plane of a triplanar blend on a surface whose normal is mostly
    // Y — i.e. weighted almost to nothing — and "where this fragment actually
    // is" is the only thing a shading input called `world_pos` can mean.
    out.world_pos = world.xyz;
    // `local`, not `pos`: the *displaced* vertex, so that `local_pos` and
    // `world_pos` are one point in two bases and never two points. A wave
    // surface asking for `F_LOCAL_SPACE` would otherwise get a pattern that
    // slides along the swell by exactly the displacement — the very artifact the
    // bit exists to remove, reintroduced in the one variant that moves geometry.
    out.local_pos = local;
    // Rotating the normal by the model matrix is exact for the uniform-scale
    // transforms the engine places entities with; non-uniform scale would want
    // the cofactor matrix (`cross` of the column pairs — no inverse needed).
    // Deliberately unchanged by instancing: the instance block never carried a
    // normal matrix, so adding one here would be a *look* change riding along
    // with a plumbing change, and it would move the golden frame for a reason
    // that has nothing to do with instancing. It stays a separate decision.
    out.normal = (model * vec4<f32>(normal, 0.0)).xyz;
    out.base_color = base_color;
    out.params = params;
    out.uv = uv;
    if (F_VERTEX_COLOR) {
        out.color = color;
    } else {
        out.color = vec3<f32>(1.0, 1.0, 1.0);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Baked-texture sampling (DESIGN §7)
// ---------------------------------------------------------------------------
//
// Everything below samples with **explicit gradients**, taken once per fragment
// from the un-offset plane UV and handed to every tap.
//
// Three things have to be true at once and only `textureSampleGrad` makes them
// so. (1) The bake carries a full mip chain now, so a tap has to pick a level
// or it shimmers. (2) The anti-tiling offsets are a per-virtual-cell hash: two
// neighbouring pixels of a quad can land in different cells, so the *implicit*
// derivative of `uv + o` jumps by a whole tile at every cell boundary, and an
// implicit-LOD `textureSample` would pick the bottom of the chain there —
// a grid of blurred seams, which is worse than the repetition the trick exists
// to hide. (3) `dpdx`/`dpdy` are implicit-derivative built-ins and WGSL only
// allows them in uniform control flow, so they are taken in `fs_main` and
// passed down rather than computed inside the loop.
//
// Downlevel: WGSL `textureSampleGrad` lowers to GLSL `textureGrad`, which is
// core in GLSL ES 3.00 (so, all of WebGL2) and carries no uniformity
// requirement of its own — unlike `texture()`, it is well-defined inside the
// loop. naga's GLSL backend emits it for `SampleLevel::Gradient` and even uses
// it as its *workaround* for backends where `textureLod` misbehaves, so it is
// the better-supported of the two paths, not the riskier one.

// One plane's texture-space derivatives, computed before any offset is applied.
struct Grad {
    dx: vec2<f32>,
    dy: vec2<f32>,
};

fn tap(t: texture_2d<f32>, uv: vec2<f32>, g: Grad) -> vec4<f32> {
    return textureSampleGrad(t, t_sampler, uv, g.dx, g.dy);
}

// A 2D integer-style hash for the anti-tiling offsets. The original used
// `fract(sin(p) * 43758.5453); DESIGN §7 forbids sin-hashes (fp24 "highp" on
// cheap mobile parts destroys them), so this is the same Hoskins family the
// noise library uses.
fn notile_hash2(p_in: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3<f32>(p_in.x, p_in.y, p_in.x) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 = p3 + vec3<f32>(dot(p3, p3.yzx + vec3<f32>(33.33)));
    return fract((p3.xx + p3.yz) * p3.zy);
}

// Inigo Quilez's texture-repetition trick: give each virtual grid cell a random
// offset into the tile and blend the 3×3 neighbourhood with smooth weights, so
// a single bake covering a hillside never reads as wallpaper. ~9 taps.
//
// The tile itself is exactly seamless (see `runt_core::texture`), so this is
// solving *repetition*, not *seams* — two different problems that a blended
// "seamless" texture conflates.
fn notile(t: texture_2d<f32>, uv: vec2<f32>, g: Grad) -> vec4<f32> {
    let cell = floor(uv);
    let f = uv - cell;
    var acc = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var j = -1; j <= 1; j = j + 1) {
        for (var i = -1; i <= 1; i = i + 1) {
            let n = vec2<f32>(f32(i), f32(j));
            let o = notile_hash2(cell + n);
            let r = n - f + o;
            let w = exp(-5.0 * dot(r, r));
            // The *same* gradient for every tap. The offset is a translation in
            // tile space, so it does not change how fast the texture moves
            // under the pixel — only where it is read from.
            acc = acc + w * tap(t, uv + o, g);
            wsum = wsum + w;
        }
    }
    return acc / wsum;
}

fn plane_tap(t: texture_2d<f32>, uv: vec2<f32>, g: Grad, anti_tiling: bool) -> vec4<f32> {
    if (anti_tiling) {
        return notile(t, uv, g);
    }
    return tap(t, uv, g);
}

// Triplanar weights: `pow(abs(N), sharpness)` normalized to sum to 1. The
// degenerate case (a zero normal on broken geometry) falls back to an even
// third rather than dividing by zero.
fn triplanar_blend(n: vec3<f32>, sharpness: f32) -> vec3<f32> {
    let b = pow(abs(n), vec3<f32>(sharpness));
    let s = b.x + b.y + b.z;
    if (s > 1.0e-6) {
        return b / s;
    }
    return vec3<f32>(1.0 / 3.0);
}

// ---------------------------------------------------------------------------
// Live procedural evaluation (DESIGN §7's live path)
// ---------------------------------------------------------------------------
//
// > *Live (perf-gated): evaluated per-pixel in the material shader … The gate
// > is fragment-shader cost … not the backend.* — DESIGN §7
//
// This is the **original's** semantics, not a per-pixel re-derivation of the
// bake. The Godot material sampled unbounded 3D noise at the shading point:
//
//   * no tile, so no wrap — `cellular` gets a zero period and the field is
//     aperiodic in every direction. There is nothing to repeat, so there is
//     nothing for the anti-tiling sampler to hide and it is not run.
//   * no triplanar projection *for the albedo*. Triplanar exists to give a 2D
//     image a place to live on 3D geometry; a 3D field is already defined
//     everywhere, so projecting it would be throwing away the third dimension
//     and then paying nine taps to disguise the loss.
//   * one octave loop, not three.
//
// The world → noise map is the one the bake uses, restated without the tile:
// the bake evaluates octave *i* at `uv · span_i + offset · freq_i`, and
// `uv = world · world_scale`, so `world · (span_0 · world_scale) + offset`
// scaled by `freq_i` is the same point. `config.w` carries that product. A
// live fragment and a bake texel therefore sample *the same field* — which is
// what `tests/live_texture.rs` holds them to, and what makes the A/B toggle a
// comparison rather than two different materials.

struct LiveSample {
    value: f32,
    // Height derivatives along screen x and y — the live twin of the bake's
    // analytic texel delta, differentiated against the real pixel footprint.
    dndx: f32,
    dndy: f32,
};

// `q` is the world position in octave-0 cell units (seed offset already added);
// `dqx`/`dqy` are its screen derivatives in the same units. The caller takes
// those with `dpdx`/`dpdy` in uniform control flow — same discipline as the
// baked path's `Grad`, and for the same WGSL rule.
fn live_sample(q: vec3<f32>, dqx: vec3<f32>, dqy: vec3<f32>, lod: vec2<f32>) -> LiveSample {
    let lattice = tex.mode.x;
    let ret = tex.mode.y;
    let fractal = tex.mode.z;
    let count = tex.mode.w;
    let normal_mode = tex.counts.x;

    let jitter = tex.shape.x;
    let weighted_strength = tex.shape.w;
    let edge_width = max(tex.normal.x, 1.0e-6);
    let strength = tex.normal.y;

    var accum = fbm_new();
    var dndx = 0.0;
    var dndy = 0.0;

    for (var i = 0u; i < count; i = i + 1u) {
        let o = tex.octaves[i];
        let freq = o.y;
        let amplitude = o.z;
        // The plan's own weight (1 everywhere — the CPU has no camera) times
        // this fragment's distance/footprint fade. Multiplying rather than
        // replacing keeps the one knob the plan owns intact.
        let w = o.w * octave_weight(i, lod.x, lod.y);

        // The payoff of the octave window: a faded-out octave is *skipped*, not
        // evaluated and multiplied by zero, so a distant fragment really does
        // pay for two nineteen-cell Voronoi loops instead of five. A zero
        // weight contributes nothing to either half of the fBm normalization,
        // so skipping is arithmetically identical — except under `RIDGED`,
        // where every octave feeds the next one's suppression whatever its
        // weight, and there the loop runs in full.
        if (w <= 0.0 && fractal != FRACTAL_RIDGED) {
            continue;
        }

        let p = q * freq;
        // `vec3(0.0)` is `wrap_cell`'s "do not wrap this axis" sentinel on all
        // three axes: unbounded noise, which is the whole point of live.
        let cs = cellular(p, lattice, ret, jitter, vec3<f32>(0.0));
        accum = fbm_push(accum, cs.value, amplitude, w, fractal, weighted_strength);

        if (normal_mode != NORMAL_NONE) {
            let edge_mag = 1.0 - smoothstep(0.0, edge_width, cs.d2 - cs.d1);
            var d = p - cs.f1;                       // to-point: radial
            if (normal_mode == NORMAL_TO_EDGE) {
                d = cs.f2 - cs.f1;                   // to-edge: across the boundary
            }
            let dlen = length(d);
            var dir = vec3<f32>(0.0);
            if (dlen > 1.0e-4) {
                // Negated: `d` points away from the nearest feature point, but
                // dndx/dndy are a height gradient and cell centres are the
                // relief's high points, so uphill is *toward* the feature
                // point. See `TextureSpec::sample_at`'s matching comment in
                // texture.rs for the full argument — the two are held to the
                // same sign by `tests/live_texture.rs`.
                dir = -d / dlen;
            }
            // Byte for byte the bake's weight — `amplitude · freq · edge · w ·
            // strength`. Only the *step* differs: the bake walks a fixed
            // fraction of the tile, this walks one pixel. `dqx` is in octave-0
            // cells and `dir` is a unit vector in octave-*i* space, which is
            // exactly the `freq` already inside `nw`.
            let nw = amplitude * freq * edge_mag * w * strength;
            dndx = dndx + nw * dot(dir, dqx);
            dndy = dndy + nw * dot(dir, dqy);
        }
    }

    var r: LiveSample;
    r.value = fbm_finish(accum);
    r.dndx = dndx;
    r.dndy = dndy;
    return r;
}

// The reference height the bake's packed normal measures its two derivatives
// against: `normalize(-dndx, -dndy, 0.5)`. The live path has no packing step,
// so it has to restate the constant or the same authored
// `NormalSpec::strength` would mean two different amounts of relief.
const LIVE_NORMAL_Z: f32 = 0.5;

// Perturb a normal from screen-space height derivatives on a surface with no
// UVs and therefore no tangent frame.
//
// Two halves, and the first is the one that is easy to get wrong.
//
// **The tangent normal is built exactly as the bake packs one.** Normalizing
// `(-dhdx, -dhdy, 0.5)` is not a formatting step, it is a *saturation*: it
// bounds the tangent offset at 1 however large the gradient gets, so a material
// authored with a hot `strength` (rock runs 29.6) leans a surface hard and
// never turns it inside out. A raw analytic gradient — mathematically the
// better answer — has no such bound, and on this content it drives normals past
// the tangent plane and blacks the floor out. Matching the bake's saturation is
// also what makes the two modes the same *look* and not merely the same colour.
//
// **The frame the offset is applied in comes from the pixel.** `dpx`/`dpy` span
// the pixel in world space and lie in the surface, so crossing each with the
// normal gives the two surface directions the derivatives run along
// (Mikkelsen's construction); `det` carries the winding, and normalizing the
// pair drops the pixel's size, which is what keeps the relief the same depth at
// every distance. Then `normalize(n + tangent)` is the baked path's own
// whiteout blend, one plane instead of three.
fn perturb_normal(n: vec3<f32>, dpx: vec3<f32>, dpy: vec3<f32>, dhdx: f32, dhdy: f32) -> vec3<f32> {
    let t = normalize(vec3<f32>(-dhdx, -dhdy, LIVE_NORMAL_Z));

    let r1 = cross(dpy, n);
    let r2 = cross(n, dpx);
    let det = dot(dpx, r1);
    // A degenerate pixel — a silhouette-thin triangle, or a fragment whose quad
    // straddles nothing — has no frame to lean on. Leave the normal alone
    // rather than normalizing a zero vector into a NaN that would spread
    // through the lighting.
    if (abs(det) < 1.0e-12 || length(r1) < 1.0e-12 || length(r2) < 1.0e-12) {
        return n;
    }
    let s = sign(det);
    return normalize(n + t.x * normalize(s * r1) + t.y * normalize(s * r2));
}

// ---------------------------------------------------------------------------
// The phase circle (`F_PHASE_CIRCLE`)
// ---------------------------------------------------------------------------
//
// A disc in **screen** space, centred on wherever the game put it. Screen space
// is the original's choice and it is the right one: a world-space sphere hugs
// the geometry it cuts and its edge becomes an unreadable silhouette, where a
// screen circle stays a circle you can track while the camera moves. See
// `3dimenshift/shaders/phase_common.gdshaderinc`, which this restates.
//
// Three things are worth knowing about the coordinates:
//
//   * The fragment's place in the frame comes from `@builtin(position)` — in
//     the fragment stage that is the framebuffer pixel, not the clip vector the
//     vertex shader wrote — divided by the render target's size. So it needs no
//     varying and costs nothing in the variants that do not use it.
//   * **Render scale is free.** `viewport` is the size of the target actually
//     being drawn into, so the division lands in the same 0..1 whether the
//     frame is native or a quarter of it, and the circle is in the same place
//     on screen after the blit. The one thing that changes is how many pixels
//     the edge fringe is smeared over, which is the point of drawing small.
//   * The X offset is multiplied by the aspect ratio, so the disc is round on
//     screen rather than round in NDC.

/// Below this the circle is "off": nothing is inside it, so world geometry is
/// solid and phase-only geometry is gone. Godot's `phase_common` uses the same
/// 0.001, and the resting state has to be exact or every phase object flickers
/// at radius zero.
const PHASE_MIN_RADIUS: f32 = 0.001;
/// Width of the fringe at the circle's edge, in NDC-Y units.
const PHASE_EDGE: f32 = 0.03;

// The fragment's normalized device coordinates, +Y up, from the framebuffer
// pixel `@builtin(position)` carries in the fragment stage.
fn frag_ndc(frag_pos: vec2<f32>) -> vec2<f32> {
    let uv = frag_pos * frame.viewport.zw;
    return vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
}

// Aspect-corrected distance from the circle's centre, in NDC-Y units.
fn phase_distance(frag_pos: vec2<f32>) -> f32 {
    var d = frag_ndc(frag_pos) - frame.phase.xy;
    // viewport.x / viewport.y, written as a multiply by the reciprocal we are
    // already carrying.
    d.x = d.x * frame.viewport.x * frame.viewport.w;
    return length(d);
}

// ---------------------------------------------------------------------------
// The key light's shadow (DESIGN §5, §11; `src/shadow.rs`)
// ---------------------------------------------------------------------------
//
// How lit by the key light this fragment is: 1 in the open, 0 hard in shadow,
// fractional on the PCF-softened edge and across the light box's rim band
// (below). The term scales the **key term only**
// — hemisphere ambient is skylight, not sunlight, and leaving it untouched is
// what keeps a shadowed floor a dimmer version of itself (the two-colour
// ambient still separating its silhouettes) rather than a black hole.
//
// `textureSampleCompareLevel`, not `textureSampleCompare`: the map has one mip
// so the level is a formality, and the Level form carries no implicit-derivative
// uniformity requirement — this runs after the phase circle's discard, which is
// exactly the control flow the implicit form is not allowed in.
fn shadow_factor(world_pos: vec3<f32>, n: vec3<f32>) -> f32 {
    let pos = frame.light_view_proj * vec4<f32>(world_pos, 1.0);
    // Orthographic, so w is 1 and this is not a divide in disguise.
    let ndc = pos.xyz;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    // Above the light's near plane is *lit*: casters up there are pancaked
    // into the map (`shadow.wgsl`), receivers up there are above the light.
    if (ndc.z <= 0.0) {
        return 1.0;
    }
    // The rim: the box tracks the camera, so the world past its edge must
    // fade to daylight rather than to a wall of shadow — and "fade" is
    // literal, because a hard rim is a cutoff that crawls across long shadows
    // as the camera moves. `rim` is the distance to the nearest way out of
    // the map (uv edges and far depth; the near wall is handled above), and
    // the shadow dissolves over the outermost `shadow_params.w` of it.
    let rim = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    let fade = smoothstep(0.0, frame.shadow_params.w, min(rim, 1.0 - ndc.z));
    if (fade <= 0.0) {
        return 1.0;
    }
    // Constant + slope bias, in light-depth units (see `ShadowSettings` for
    // the acne/peter-panning trade). Applied to the reference depth rather
    // than baked into the map, so the knobs work without a pipeline rebuild.
    let ndotl = max(dot(n, normalize(frame.light_dir.xyz)), 0.0);
    let bias = frame.shadow_params.y + frame.shadow_params.z * (1.0 - ndotl);
    let shadow = textureSampleCompareLevel(t_shadow, s_shadow, uv, ndc.z - bias);
    // `1 − fade·(1 − shadow)` rather than `mix(1.0, shadow, fade)`:
    // algebraically the same, but this form is *exactly* 1.0 whenever the tap
    // is — an unoccluded receiver inside the band stays byte-identical to
    // shadows-off, which the no-acne test pins.
    return 1.0 - fade * (1.0 - shadow);
}

// ---------------------------------------------------------------------------
// The rim term (`F_FRESNEL`)
// ---------------------------------------------------------------------------
//
// The world-space direction the camera looks along through this fragment,
// unprojected from the fragment's own NDC column — `sky.wgsl`'s construction,
// which is the only one available here: the frame block carries `view_proj` and
// its inverse and no camera position, and a rim term is the first thing that has
// ever asked for one.
//
// Sign is not load-bearing. The rim is `1 − |N·V|`, so a view vector pointing
// away from the eye instead of towards it gives the same band — which is also
// what makes it agree with the original's `VIEW` (view space, towards the eye)
// without reproducing that basis.
fn view_direction(frag_pos: vec2<f32>) -> vec3<f32> {
    let ndc = frag_ndc(frag_pos);
    let near = frame.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let far = frame.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let delta = far.xyz / far.w - near.xyz / near.w;
    let len = length(delta);
    // A degenerate view-projection (the no-camera path hands us the identity)
    // must produce a direction, not a NaN — same guard `fs_sky` carries.
    if (len > 0.0) {
        return delta / len;
    }
    return vec3<f32>(0.0, 0.0, 1.0);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    // Before anything expensive: a discarded fragment should not pay for a
    // triplanar sample it is about to throw away.
    var phase_dist = 0.0;
    if (F_PHASE_CIRCLE) {
        phase_dist = phase_distance(in.clip.xy);
        let inside = frame.phase.z > PHASE_MIN_RADIUS && phase_dist < frame.phase.z;
        // `in.params.x`: 0 world-only, 1 phase-only, 2 effect-only. Compared
        // with slack rather than `==` because it arrives as a float in a
        // uniform, and an authored 1.0 that survived a round trip through RON
        // must not become mode 0.
        let mode = in.params.x;
        if (mode < 0.5) {
            // World-only: the circle removes it.
            if (inside) {
                discard;
            }
        } else if (mode < 1.5) {
            // Phase-only: it exists nowhere else.
            if (!inside) {
                discard;
            }
        }
    }

    var n = normalize(in.normal);

    var albedo = in.base_color.rgb;
    if (F_VERTEX_COLOR) {
        albedo = albedo * in.color;
    }

    // ---- The sampling basis (`F_LOCAL_SPACE`) -----------------------------
    //
    // One decision, both texture paths. "Which point is this fragment sampling
    // at" is a single question whether the answer goes on to index a triplanar
    // tile (`F_TEXTURE`) or to be evaluated as a 3D field (`F_LIVE_TEX`), so it
    // is asked once here rather than twice below — a second copy is a second
    // thing to forget when a third texture path lands.
    //
    // World is the default and the original's (`MaterialVariant::TEXTURE`), and
    // it is right for terrain: a hillside's grass should not restart at every
    // brush boundary. It is wrong for anything that *moves*, because the pattern
    // then belongs to the world and the surface travels through it — the
    // "sliding marble" that kept the port's player untextured (see
    // `MaterialVariant::LOCAL_SPACE`).
    //
    // A `const if` and not `select(in.world_pos, in.local_pos, F_LOCAL_SPACE)`.
    // `select` is an expression and both of its arms survive into the emitted
    // module for the backend to fold; a `const bool` branch is folded by naga
    // before a backend sees it, so a draw without the bit generates the
    // instructions it generated before this existed. That distinction is a scar,
    // not a preference — `MaterialVariant::SHADOW` records a runtime branch
    // around the key term moving lit pixels by one LSB *with its feature off*,
    // purely by perturbing the driver's scheduling.
    //
    // # Feature size follows the object's scale, and that is the answer
    //
    // Local units are the model matrix's scale away from metres, so the same
    // material on a 0.5× entity draws half-size features. Nothing here divides
    // that out, and the reason is not cost — a `length(m0.xyz)` and a divide in
    // the vertex stage would do it, per vertex, with no new instance data.
    //
    // It is that normalizing would make the picture on a mesh a function of the
    // *instance* rather than of (mesh, material). Two entities sharing both would
    // then wear two different pictures, and — the part that actually decides it —
    // an entity **animating** its scale would have its pattern crawl across its
    // own surface by exactly the scale ratio, frame over frame. That is the
    // sliding artifact this bit exists to remove, reintroduced along the one axis
    // world-space sampling never had a problem with. Raw local units are also the
    // only definition that stays coherent under a *non-uniform* model matrix,
    // where there is no single scale to divide by and picking one axis's would
    // silently distort along the other two.
    //
    // So the contract is the strong, simple one: the pattern is painted on the
    // mesh, and paint scales with the thing it is painted on. An author who wants
    // metre-fixed feature size on a scaled object already has that — it is this
    // bit turned off — and one who wants a different feature size for a
    // particular material has `TextureSpec::world_scale`, which costs a re-bake
    // and nothing else.
    var p_source = in.world_pos;
    if (F_LOCAL_SPACE) {
        p_source = in.local_pos;
    }

    // `F_LIVE_TEX` supersedes `F_TEXTURE`: live evaluates the spec and never
    // reads the bake, so running both would be one of them for nothing. The
    // draw list makes the two mutually exclusive (`draw::resolve_variant`);
    // this `else` is what makes the combination *defined* rather than merely
    // unreachable.
    if (F_LIVE_TEX) {
        // Every derivative this path needs, taken here — at the top of the
        // fragment, in uniform control flow — because the octave loop may not
        // call `dpdx`/`dpdy` and would want the un-scaled world step anyway.
        // Exactly the hoist the baked path does for its `Grad`s.
        //
        // **Two pairs, and they are not interchangeable.** `dpx`/`dpy` span the
        // pixel in *world* space and are what `perturb_normal` builds its
        // tangent frame from — that frame is crossed with `n`, which is a world
        // normal, so a local-space step there would mix two bases and lean the
        // relief in a direction that rotates with the object. `dsx`/`dsy` span
        // the pixel in the *sampling* basis and are what the height derivative
        // needs, because `dndx` is `d(height)/d(screen x)` and the height is a
        // function of `q`. Under `F_LOCAL_SPACE` the two differ by the model
        // matrix's rotation and scale; without it they are the same expression
        // and the const branch folds the second pair away entirely.
        let dpx = dpdx(in.world_pos);
        let dpy = dpdy(in.world_pos);
        var dsx = dpx;
        var dsy = dpy;
        if (F_LOCAL_SPACE) {
            dsx = dpdx(in.local_pos);
            dsy = dpdy(in.local_pos);
        }

        let cells_per_metre = tex.config.w;
        let q = p_source * cells_per_metre + tex.live_seed.xyz;
        let dqx = dsx * cells_per_metre;
        let dqy = dsy * cells_per_metre;

        // The mip substitute: how many octave-0 cells this pixel covers decides
        // how many octaves are worth evaluating (see `live_octave_window`).
        // Combining the two screen axes into one scalar has three candidates:
        //  - `max(len_x, len_y)` (what this was): never under-resolves either
        //    axis, so it never aliases — but a surface at a grazing angle has
        //    one axis's world-space derivative explode while the other stays
        //    modest (the ground is nearly edge-on to the camera; a wall beside
        //    it at the same distance is not), and `max` picks that worst axis
        //    for *both*. The ground then loses every octave while the wall
        //    keeps two, which is a fade that visibly disagrees between two
        //    surfaces the player has no reason to think are different.
        //  - `min(len_x, len_y)`: the opposite failure — sharp on the good axis
        //    but keeps octaves the bad axis cannot resolve, which is exactly
        //    the point-sampled shimmer a mip chain exists to prevent and this
        //    window is the live path's only defence against.
        //  - `sqrt(len_x * len_y)` (geometric mean): the standard anisotropic-
        //    filtering compromise between the two — it under-blurs relative to
        //    `max` and under-sharpens relative to `min`, on both axes at once,
        //    which is why a mip selector reaches for it when it cannot afford a
        //    true anisotropic footprint. Chosen here for the same reason.
        // `max(len_x * len_y, 0.0)` guards a `sqrt` of a product that should
        // never be negative (`length()` cannot return one) but would produce a
        // NaN that propagates through the whole fBm sum if it ever were.
        let len_x = length(dqx);
        let len_y = length(dqy);
        let footprint = sqrt(max(len_x * len_y, 0.0));
        let lod = live_octave_window(footprint, tex.live.x, tex.live.y);

        let r = live_sample(q, dqx, dqy, lod);
        albedo = albedo * ramp_lookup(
            tex.ramp,
            tex.counts.y,
            tex_postprocess(r.value, tex.shape.y, tex.shape.z),
        );

        if (F_NORMAL_MAP) {
            n = perturb_normal(n, dpx, dpy, r.dndx, r.dndy);
        }
    } else if (F_TEXTURE) {
        let world_scale = tex.config.x;
        let sharpness = tex.config.y;
        let anti_tiling = tex.config.z > 0.5;
        let p = p_source * world_scale;
        // The three planes' derivatives, taken here — at the top of the
        // fragment, in uniform control flow — because `dpdx`/`dpdy` may not be
        // called from inside the anti-tiling loop, and because the loop wants
        // the derivative of the *un-offset* UV anyway. `world_scale` is a
        // uniform, so this is the sampling basis's derivative scaled once — and
        // it needs no `F_LOCAL_SPACE` case of its own, because it is taken from
        // `p` and `p` is already in whichever basis the bit chose.
        let g_xy = Grad(dpdx(p.xy), dpdy(p.xy));
        let g_xz = Grad(dpdx(p.xz), dpdy(p.xz));
        let g_yz = Grad(dpdx(p.yz), dpdy(p.yz));
        // The weight-to-plane mapping is the original's: the Z-normal weight
        // drives the XY plane, Y drives XZ, X drives YZ.
        //
        // **This stays the world normal under `F_LOCAL_SPACE`, and that is a
        // known and bounded incompleteness.** The plane *coordinates* above have
        // moved into object space, so the pattern no longer slides — which is the
        // artifact the bit exists to kill, and the dominant one. The plane
        // *weights* have not, so a surface whose world normal turns will
        // cross-fade between the three projections as it turns, which reads as a
        // slow dissolve rather than as a slide. Translation is therefore exact
        // (a moving platform, a carried boulder: the normal does not change), and
        // rotation is exact for any turn that leaves `abs(n)` alone — a Z-spin of
        // a +Z-facing sheet, which `tests/local_space.rs` measures.
        //
        // Fixing it wants a *second* interpolated `vec3` (the pre-transform
        // normal) on every draw in the engine, and then the whiteout blend below
        // would have to carry its per-plane offsets back into world space, which
        // needs the model matrix's rotation in the fragment stage — the very
        // thing `VSOut::local_pos`'s comment rejects. `F_LIVE_TEX` has no
        // triplanar projection at all (a 3D field is defined everywhere), so the
        // live path is exact under any rigid transform and is what a rotating
        // object should ask for. That is a recommendation with a cost attached
        // rather than a second varying spent on every draw for one case.
        let blend = triplanar_blend(n, sharpness);

        let c_xy = plane_tap(t_albedo, p.xy, g_xy, anti_tiling);
        let c_xz = plane_tap(t_albedo, p.xz, g_xz, anti_tiling);
        let c_yz = plane_tap(t_albedo, p.yz, g_yz, anti_tiling);
        albedo = albedo * (c_xy.rgb * blend.z + c_xz.rgb * blend.y + c_yz.rgb * blend.x);

        if (F_NORMAL_MAP) {
            // Plain taps, no anti-tiling: the crinkle is high-frequency and
            // repetition does not read on it, so it is not worth 3× the taps.
            // Mip-correct all the same — an unmipped normal map is the loudest
            // shimmer on screen, because the lighting amplifies it.
            let n_xy = tap(t_normal, p.xy, g_xy).xy * 2.0 - vec2<f32>(1.0);
            let n_xz = tap(t_normal, p.xz, g_xz).xy * 2.0 - vec2<f32>(1.0);
            let n_yz = tap(t_normal, p.yz, g_yz).xy * 2.0 - vec2<f32>(1.0);
            // "Whiteout" triplanar normal blend: each plane's tangent offsets
            // are applied along that plane's world axes and summed onto the
            // geometric normal. Cheap, stable, and it cannot flip a normal
            // inside out the way a per-plane TBN reconstruction can.
            let perturb = blend.z * vec3<f32>(n_xy.x, n_xy.y, 0.0)
                        + blend.y * vec3<f32>(n_xz.x, 0.0, n_xz.y)
                        + blend.x * vec3<f32>(0.0, n_yz.x, n_yz.y);
            n = normalize(n + perturb);
        }
    }

    var color: vec3<f32>;
    // The alpha the fragment leaves with. `base_color.a` unless one of the
    // unlit looks below owns it (only `F_FRESNEL` does), because a rim that
    // faded its colour and not its coverage would be a solid shell with a
    // gradient painted on it.
    var alpha = in.base_color.a;
    if (F_FRESNEL) {
        // `phase_outline.gdshader`, style 0: a silhouette rim, in the colour
        // *and* in the coverage, unshaded. `params.y` is the exponent — higher
        // is thinner and sharper — floored at the original's own hint range so
        // an unset param cannot turn the rim into a solid shell (`pow(x, 0)`
        // is 1 everywhere).
        let rim = pow(
            1.0 - abs(dot(n, view_direction(in.clip.xy))),
            max(in.params.y, 0.5),
        );
        color = albedo * rim;
        alpha = alpha * rim;
    } else if (F_EMISSIVE_SWEEP) {
        // `logic_wire.gdshader`: a wipe running along the mesh's own `u`, from
        // the end the signal entered. `t` is 1 on the swept side.
        //
        // Two tones and two gains, and no third colour slot: the mesh's vertex
        // colour is the un-swept side, `base_color` the swept one. `albedo` is
        // deliberately not used — it is the *product* of the two, which is not
        // either tone.
        let t = 1.0 - smoothstep(
            in.params.y - in.params.z,
            in.params.y + in.params.z,
            in.uv.x,
        );
        color = mix(in.color, in.base_color.rgb, t) * mix(in.params.x, in.params.w, t);
    } else if (F_BILLBOARD_UNLIT) {
        // No lighting at all. A camera-facing quad's normal is whatever its
        // CPU-built basis happens to point at, so shading it would make the
        // surface swim as the camera turns — and these are emissive things
        // (glyphs, motes, prompts) that were never lit in the original either.
        color = albedo;
    } else {
        // Hemisphere ambient: sky above, ground below, blended on the normal's
        // Y. Cheap, and it separates silhouettes far better than a constant
        // term.
        let hemi = mix(frame.ground_color.rgb, frame.sky_color.rgb, 0.5 + 0.5 * n.y);
        // The shadow attenuates the key light alone; `hemi` stays whole (see
        // `shadow_factor`), so a shadowed floor keeps the two-colour ambient.
        //
        // Behind `F_SHADOW` — a **const**, not the runtime uniform, and that
        // is a scar rather than a style: a live `if` around the key term
        // perturbed the driver's instruction scheduling enough to move a few
        // lit pixels by one LSB with shadows *off*, which is a moved golden
        // hash. A const branch folds away entirely, so every pre-shadow
        // variant compiles to the pre-shadow instructions — same key, same
        // cached pipeline, same bytes (`MaterialVariant::SHADOW`). The
        // runtime check stays inside the folded arm as insurance against a
        // shadowed pipeline ever meeting a shadowless frame block.
        if (F_SHADOW) {
            var shade = 1.0;
            if (frame.shadow_params.x > 0.5) {
                shade = shadow_factor(in.world_pos, n);
            }
            let key = frame.light_color.rgb
                * max(dot(n, normalize(frame.light_dir.xyz)), 0.0)
                * shade;
            color = (hemi + key) * albedo;
        } else {
            let key = frame.light_color.rgb * max(dot(n, normalize(frame.light_dir.xyz)), 0.0);
            color = (hemi + key) * albedo;
        }
    }

    if (F_PHASE_CIRCLE) {
        // The fringe: a band at the circle's edge, whitened by `phase.w`. It is
        // what makes the boundary read as an *event* rather than a clipping
        // plane, and it is the only thing mode 2 (effect-only) is for. Two
        // instructions on a fragment that already survived the discard above.
        if (frame.phase.z > PHASE_MIN_RADIUS) {
            let edge = 1.0 - smoothstep(0.0, PHASE_EDGE, abs(phase_dist - frame.phase.z));
            color = mix(color, vec3<f32>(1.0), edge * frame.phase.w);
        }
    }

    return vec4<f32>(color, alpha);
}
