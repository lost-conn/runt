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
};
@group(0) @binding(0) var<uniform> frame: Frame;

// Per-entity slot, addressed with a dynamic uniform-buffer offset.
struct Instance {
    model: mat4x4<f32>,
    base_color: vec4<f32>,
    params: vec4<f32>,
};
@group(1) @binding(0) var<uniform> inst: Instance;

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

struct VSOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    // World position, for the triplanar sample. There is no UV in the texture
    // path at all: the original is a world-space shader and CSG-ish generated
    // geometry has no UV worth trusting, so the world *is* the parameterization.
    @location(2) world_pos: vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
) -> VSOut {
    var out: VSOut;
    let world = inst.model * vec4<f32>(pos, 1.0);
    out.clip = frame.view_proj * world;
    out.world_pos = world.xyz;
    // Rotating the normal by the model matrix is exact for the uniform-scale
    // transforms the engine places entities with; non-uniform scale would want
    // an inverse-transpose in the instance block.
    out.normal = (inst.model * vec4<f32>(normal, 0.0)).xyz;
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
                dir = d / dlen;
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

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    var n = normalize(in.normal);

    var albedo = inst.base_color.rgb;
    if (F_VERTEX_COLOR) {
        albedo = albedo * in.color;
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
        let dpx = dpdx(in.world_pos);
        let dpy = dpdy(in.world_pos);

        let cells_per_metre = tex.config.w;
        let q = in.world_pos * cells_per_metre + tex.live_seed.xyz;
        let dqx = dpx * cells_per_metre;
        let dqy = dpy * cells_per_metre;

        // The mip substitute: how many octave-0 cells this pixel covers decides
        // how many octaves are worth evaluating (see `live_octave_window`).
        let footprint = max(length(dqx), length(dqy));
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
        let p = in.world_pos * world_scale;
        // The three planes' derivatives, taken here — at the top of the
        // fragment, in uniform control flow — because `dpdx`/`dpdy` may not be
        // called from inside the anti-tiling loop, and because the loop wants
        // the derivative of the *un-offset* UV anyway. `world_scale` is a
        // uniform, so this is the world-space derivative scaled once.
        let g_xy = Grad(dpdx(p.xy), dpdy(p.xy));
        let g_xz = Grad(dpdx(p.xz), dpdy(p.xz));
        let g_yz = Grad(dpdx(p.yz), dpdy(p.yz));
        // The weight-to-plane mapping is the original's: the Z-normal weight
        // drives the XY plane, Y drives XZ, X drives YZ.
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

    // Hemisphere ambient: sky above, ground below, blended on the normal's Y.
    // Cheap, and it separates silhouettes far better than a constant term.
    let hemi = mix(frame.ground_color.rgb, frame.sky_color.rgb, 0.5 + 0.5 * n.y);
    let key = frame.light_color.rgb * max(dot(n, normalize(frame.light_dir.xyz)), 0.0);

    return vec4<f32>((hemi + key) * albedo, inst.base_color.a);
}
