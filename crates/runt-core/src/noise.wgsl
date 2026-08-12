// The procedural-noise library (DESIGN §7).
//
// A *library*, not a shader: no bindings, no entry points, nothing that assumes
// a pass. It is prepended to whichever shader needs it — `bake.wgsl` by
// `runt_core::bake::bake_shader_source`, and *every* material variant by
// `runt_core::material::variant_source`, because §7's live path evaluates the
// same field in the fragment shader.
//
// That "every variant" is deliberate. The alternative — prepend it only when
// `F_LIVE_TEX` is set — would make the generated source depend on the variant
// key in a second, invisible way, and a variant system's whole value is that
// one source covers every key. Nothing here is reachable from a baked-only
// variant, so the backend strips it; what it costs is compile time, not
// instructions.
//
// `src/noise.rs` is the line-for-line CPU twin, the same relationship `sky.rs`
// has to `sky.wgsl`. `tests/noise_bake.rs` samples real baked texels and holds
// them against the Rust side, so the two cannot drift.
//
// PRECISION DOCTRINE (DESIGN §7): integer-style hashing only. Every hash here
// is the Dave Hoskins large-coefficient `fract` family — no `sin(dot(p, big))`
// anywhere, because cheap mobile GPUs run "highp" as fp24 internally and a
// sin-hash disintegrates at that precision. The magic numbers (0.1031, 0.1030,
// 0.0973, 31.32, 33.33) and the asymmetric `p3.yxx` in `hash33` are load-bearing
// and match the 3dimenshift originals byte for byte.

// Noise-kind codes, mirroring `runt_core::noise::NoiseKind::code`.
const KIND_CELLULAR: u32 = 0u;
const KIND_GRID: u32 = 1u;
const KIND_RADIAL_GRID: u32 = 2u;

// Lattice codes, mirroring `runt_core::noise::Lattice`.
const LATTICE_CUBIC: u32 = 0u;
const LATTICE_FCC: u32 = 1u;

// Return-type codes, mirroring `runt_core::noise::CellReturn::code`.
const RET_F1: u32 = 0u;
const RET_F2: u32 = 1u;
const RET_F2_MINUS_F1: u32 = 2u;
const RET_F1_PLUS_F2: u32 = 3u;
const RET_CELL_VALUE: u32 = 4u;

// Fractal codes, mirroring `runt_core::noise::Fractal::code`.
const FRACTAL_FBM: u32 = 1u;
const FRACTAL_RIDGED: u32 = 2u;
const FRACTAL_RIDGED_FNL: u32 = 3u;

// Normal modes, mirroring `runt_core::texture::NormalMode::code`.
const NORMAL_NONE: u32 = 0u;
const NORMAL_TO_POINT: u32 = 1u;
const NORMAL_TO_EDGE: u32 = 2u;

// ---------------------------------------------------------------------------
// Hashes
// ---------------------------------------------------------------------------

fn n_fract(x: f32) -> f32 {
    return x - floor(x);
}

fn n_fract2(v: vec2<f32>) -> vec2<f32> {
    return v - floor(v);
}

fn n_fract3(v: vec3<f32>) -> vec3<f32> {
    return v - floor(v);
}

fn hash11(p_in: f32) -> f32 {
    var p = n_fract(p_in * 0.1031);
    p = p * (p + 33.33);
    p = p * (p + p);
    return n_fract(p);
}

// One scalar per lattice cell. Note `p3.zyx` and the 31.32 — its siblings use
// 33.33, this one does not, and that is deliberate.
fn hash13(p3_in: vec3<f32>) -> f32 {
    var p3 = n_fract3(p3_in * 0.1031);
    p3 = p3 + vec3<f32>(dot(p3, p3.zyx + vec3<f32>(31.32)));
    return n_fract((p3.x + p3.y) * p3.z);
}

// The feature-point jitter. The `p3.yxx` in the final swizzle (where symmetry
// would want `p3.yzz`) is the quirk the port spec calls out; keep it.
fn hash33(p3_in: vec3<f32>) -> vec3<f32> {
    var p3 = n_fract3(p3_in * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 = p3 + vec3<f32>(dot(p3, p3.yxz + vec3<f32>(33.33)));
    return n_fract3((p3.xxy + p3.yxx) * p3.zyx);
}

// Quintic fade 6t^5 - 15t^4 + 10t^3, C2 continuous. Unused by the cellular
// path; carried over so a value/Perlin spec variant does not have to re-port it.
fn quintic3(t: vec3<f32>) -> vec3<f32> {
    return t * t * t * (t * (t * 6.0 - vec3<f32>(15.0)) + vec3<f32>(10.0));
}

fn seed_offset_3d(s: f32) -> vec3<f32> {
    return vec3<f32>(
        hash11(s) * 1000.0,
        hash11(s + 47.32) * 1000.0,
        hash11(s + 93.17) * 1000.0,
    );
}

// A 2D integer-style hash for the anti-tiling sampler. The original used
// `fract(sin(p) * 43758.5453)`, which DESIGN §7 forbids; this is the same
// family as the rest of the file and behaves identically at fp24.
fn hash22(p_in: vec2<f32>) -> vec2<f32> {
    var p3 = n_fract3(vec3<f32>(p_in.x, p_in.y, p_in.x) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 = p3 + vec3<f32>(dot(p3, p3.yzx + vec3<f32>(33.33)));
    return n_fract2((p3.xx + p3.yz) * p3.zy);
}

// ---------------------------------------------------------------------------
// The FCC lattice
// ---------------------------------------------------------------------------

// Home, the 12 face neighbours at distance sqrt(2), and the 6 axis-aligned +-2
// cells (which share only a vertex with home and matter only at large jitter).
// The count stays flat at 19 rather than branching: a GPU wants a uniform loop.
var<private> FCC_OFFSETS: array<vec3<f32>, 19> = array<vec3<f32>, 19>(
    vec3<f32>( 0.0,  0.0,  0.0),
    vec3<f32>( 1.0,  1.0,  0.0), vec3<f32>( 1.0, -1.0,  0.0),
    vec3<f32>(-1.0,  1.0,  0.0), vec3<f32>(-1.0, -1.0,  0.0),
    vec3<f32>( 1.0,  0.0,  1.0), vec3<f32>( 1.0,  0.0, -1.0),
    vec3<f32>(-1.0,  0.0,  1.0), vec3<f32>(-1.0,  0.0, -1.0),
    vec3<f32>( 0.0,  1.0,  1.0), vec3<f32>( 0.0,  1.0, -1.0),
    vec3<f32>( 0.0, -1.0,  1.0), vec3<f32>( 0.0, -1.0, -1.0),
    vec3<f32>( 2.0,  0.0,  0.0), vec3<f32>(-2.0,  0.0,  0.0),
    vec3<f32>( 0.0,  2.0,  0.0), vec3<f32>( 0.0, -2.0,  0.0),
    vec3<f32>( 0.0,  0.0,  2.0), vec3<f32>( 0.0,  0.0, -2.0),
);

// Round to the nearest FCC lattice point: an integer triple whose components
// sum to an even number. Plain rounding lands on odd parity half the time; when
// it does, the axis whose rounding cost the most is nudged one further step
// towards p.
//
// The parity test is `fract(sum/2) > 0.25` rather than `mod(sum, 2.0) > 0.5`:
// same answer, but it stays exact at every magnitude a float still represents
// integers at, which is what fp24 "highp" needs.
fn fcc_round(p: vec3<f32>) -> vec3<f32> {
    var c = floor(p + vec3<f32>(0.5));
    if (n_fract((c.x + c.y + c.z) * 0.5) > 0.25) {
        let err = abs(p - c);
        var sgn = sign(p - c);
        // sign(0) == 0 would leave the parity unfixed; pick +1.
        if (sgn.x == 0.0) { sgn.x = 1.0; }
        if (sgn.y == 0.0) { sgn.y = 1.0; }
        if (sgn.z == 0.0) { sgn.z = 1.0; }
        if (err.x >= err.y && err.x >= err.z) {
            c.x = c.x + sgn.x;
        } else if (err.y >= err.z) {
            c.y = c.y + sgn.y;
        } else {
            c.z = c.z + sgn.z;
        }
    }
    return c;
}

// ---------------------------------------------------------------------------
// Seamless wrapping
// ---------------------------------------------------------------------------

// Wrap one axis of a lattice index into [0, period); period 0 means "do not
// wrap this axis". `c` is a lattice index and is therefore an integer.
//
// The snap-and-fix after the modulo is not paranoia, it is a bug fix. GPUs
// routinely lower `a / b` to `a * (1/b)`, and the reciprocal is rounded: for
// an exact multiple like 78/6 that lands on 12.9999995 instead of 13.0, so
// `floor` drops a whole period and the wrap returns 6 where it should return 0.
// Since cell indices only ever hash — a different index is a completely
// different random value, not a slightly different one — that showed up as ~20%
// of texels disagreeing with the CPU twin by up to a full ramp. Rounding to the
// integer the value is *known* to be and then folding the ±1-period error away
// makes the wrap exact on any lowering.
fn wrap_axis(c: f32, period: f32) -> f32 {
    if (period <= 0.0) {
        return c;
    }
    var m = floor(c - floor(c / period) * period + 0.5);
    if (m < 0.0) {
        m = m + period;
    }
    if (m >= period) {
        m = m - period;
    }
    return m;
}

// Wrapping the *hash input* — and only that; the feature-point positions stay
// unwrapped, or the field would tear — makes cellular noise exactly periodic.
// That is what lets a bake tile with no blend and no seam. Periods must be even
// on FCC or the wrap would flip cell parity; `texture.rs` guarantees it.
fn wrap_cell(cell: vec3<f32>, period: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        wrap_axis(cell.x, period.x),
        wrap_axis(cell.y, period.y),
        wrap_axis(cell.z, period.z),
    );
}

// ---------------------------------------------------------------------------
// Cellular noise
// ---------------------------------------------------------------------------

struct CellSample {
    value: f32,
    // Absolute sample-space positions of the nearest and second-nearest
    // feature points, and their true (post-sqrt) distances.
    f1: vec3<f32>,
    f2: vec3<f32>,
    d1: f32,
    d2: f32,
};

fn cell_return(ret: u32, d1: f32, d2: f32, cell: vec3<f32>) -> f32 {
    if (ret == RET_CELL_VALUE) { return hash13(cell); }
    if (ret == RET_F2) { return d2; }
    if (ret == RET_F2_MINUS_F1) { return d2 - d1; }
    if (ret == RET_F1_PLUS_F2) { return (d1 + d2) * 0.5; }
    return d1;
}

fn cellular(p: vec3<f32>, lattice: u32, ret: u32, jitter: f32, period: vec3<f32>) -> CellSample {
    // Squared distances throughout; one sqrt each at the end.
    var f1 = 1.0e20;
    var f2 = 1.0e20;
    var nearest_cell = vec3<f32>(0.0);
    var nearest_pos = vec3<f32>(0.0);
    var second_pos = vec3<f32>(0.0);

    if (lattice == LATTICE_FCC) {
        let home = fcc_round(p);
        for (var i = 0; i < 19; i = i + 1) {
            let cell = home + FCC_OFFSETS[i];
            let key = wrap_cell(cell, period);
            // FCC feature points sit *on* the lattice point plus jitter.
            let point = cell + (hash33(key) - vec3<f32>(0.5)) * jitter;
            let delta = point - p;
            let d = dot(delta, delta);
            if (d < f1) {
                f2 = f1;
                second_pos = nearest_pos;
                f1 = d;
                nearest_cell = key;
                nearest_pos = point;
            } else if (d < f2) {
                f2 = d;
                second_pos = point;
            }
        }
    } else {
        let base = floor(p);
        for (var z = -1; z <= 1; z = z + 1) {
            for (var y = -1; y <= 1; y = y + 1) {
                for (var x = -1; x <= 1; x = x + 1) {
                    let cell = base + vec3<f32>(f32(x), f32(y), f32(z));
                    let key = wrap_cell(cell, period);
                    // Cubic feature points live *inside* the cell: a half-unit
                    // centre plus jitter, unlike FCC's.
                    let local = vec3<f32>(0.5) + (hash33(key) - vec3<f32>(0.5)) * jitter;
                    let point = cell + local;
                    let delta = point - p;
                    let d = dot(delta, delta);
                    if (d < f1) {
                        f2 = f1;
                        second_pos = nearest_pos;
                        f1 = d;
                        nearest_cell = key;
                        nearest_pos = point;
                    } else if (d < f2) {
                        f2 = d;
                        second_pos = point;
                    }
                }
            }
        }
    }

    var out: CellSample;
    out.d1 = sqrt(f1);
    out.d2 = sqrt(f2);
    out.f1 = nearest_pos;
    out.f2 = second_pos;
    out.value = cell_return(ret, out.d1, out.d2, nearest_cell);
    return out;
}

// ---------------------------------------------------------------------------
// The jitter-free grid
// ---------------------------------------------------------------------------

// The cubic lattice with jitter 0, in closed form: no hash, no 27-cell loop.
//
// Not a lookalike for `cellular` — the same field. Lock every feature point to
// its cell centre and the search is computing algebra: the nearest point *is*
// the centre, and the second-nearest is one unit step along whichever axis the
// sample has drifted furthest down (a second axis costs a whole unit and buys
// back less). `runt_core::noise::grid` is the CPU twin and carries the full
// argument; `tests/noise_cpu.rs` holds the two together.
//
// `value` is `1 - d1²/d2²`: 1 at a cell centre, 0 on the boundary. Positive on
// purpose — FastNoiseLite's own `RETURN_DISTANCE2_DIV` is `ratio - 1` and would
// clamp to black under FBM, while under RIDGED (`1 - |n|`, which is `ratio`
// either way) the two are identical.
//
// No `period` argument: with no hash there is nothing to wrap, and the field is
// already exactly 1-periodic, so any whole-cell tile is seamless for free.
fn noise_grid(p: vec3<f32>) -> CellSample {
    // `cell + 0.5`, matching `cellular`'s cubic branch — the two fields sit on
    // the same lattice rather than half a cell apart.
    let centre = floor(p) + vec3<f32>(0.5);
    let q = p - centre;
    let a = abs(q);
    let m = max(max(a.x, a.y), a.z);

    let d1_sq = dot(q, q);
    // `d1² >= m²` and `m <= 0.5`, so this is at least `(1 - m)² >= 0.25`: it
    // cannot reach zero and needs no epsilon.
    let d2_sq = d1_sq + 1.0 - 2.0 * m;

    // The dominant axis, stepped towards the sample. `sign(0) == 0` in WGSL
    // would leave `f2` sitting on top of `f1`, so a zero picks +1 — the same
    // tie-break `fcc_round` makes.
    var axis = vec3<f32>(0.0, 0.0, 1.0);
    if (a.x >= a.y && a.x >= a.z) {
        axis = vec3<f32>(1.0, 0.0, 0.0);
    } else if (a.y >= a.z) {
        axis = vec3<f32>(0.0, 1.0, 0.0);
    }
    var s = sign(dot(q, axis));
    if (s == 0.0) { s = 1.0; }

    var out: CellSample;
    out.value = 1.0 - d1_sq / d2_sq;
    out.f1 = centre;
    out.f2 = centre + axis * s;
    out.d1 = sqrt(d1_sq);
    out.d2 = sqrt(d2_sq);
    return out;
}

// The same grid in cylindrical coordinates about +Y: wedges around, bands up,
// rings out — the shape a UV-mapped texture makes on a sphere, which is what a
// ball wants and what an axis-aligned box lattice does not give it.
//
//   theta = atan2(p.z, p.x)/tau + 0.5    the turn, in [0,1)
//   u     = theta * sectors              wedges, a whole number of them
//   v     = p.y                          bands
//   w     = length(p.xz)                 rings
//
// `sectors` arrives already multiplied by the octave's frequency (theta is
// invariant under a uniform scale of p, so the angular density cannot come from
// the scale the way v and w do) and is rounded *here*, with floor(x + 0.5)
// rather than `round`: WGSL rounds half to even and Rust rounds half away from
// zero, and a sector count that disagreed would be a different field per side.
// A whole count is what makes theta = 1 wrap onto theta = 0 with no seam.
//
// The Y axis is a singularity where every wedge meets — exactly what a UV
// sphere does at its poles. `runt_core::noise::radial_grid` is the CPU twin and
// carries the full argument, including why f1/f2 are mapped back out of the
// warp and d1/d2 are not.
fn noise_radial_grid(p: vec3<f32>, sectors: f32) -> CellSample {
    let s = max(floor(sectors + 0.5), 1.0);
    let turn = atan2(p.z, p.x) / 6.283185307179586 + 0.5;
    let radius = length(p.xz);

    var cell = noise_grid(vec3<f32>(turn * s, p.y, 0.5));

    // Back to the caller's basis, on the sample's own cylinder, so the boundary
    // normal points somewhere real.
    let a1 = (cell.f1.x / s) * 6.283185307179586;
    let a2 = (cell.f2.x / s) * 6.283185307179586;
    cell.f1 = vec3<f32>(radius * cos(a1), cell.f1.y, radius * sin(a1));
    cell.f2 = vec3<f32>(radius * cos(a2), cell.f2.y, radius * sin(a2));
    return cell;
}

// One noise evaluation, whichever kind the spec asked for — the single seam
// both `bake.wgsl` and `shader.wgsl`'s live path go through, and the twin of
// `runt_core::noise::field`. `lattice`/`ret`/`jitter`/`period` are `cellular`'s
// arguments and mean nothing to `KIND_GRID`, which has no parameters at all.
fn noise_field(
    p: vec3<f32>,
    kind: u32,
    lattice: u32,
    ret: u32,
    jitter: f32,
    sectors: f32,
    period: vec3<f32>,
) -> CellSample {
    if (kind == KIND_GRID) {
        return noise_grid(p);
    }
    if (kind == KIND_RADIAL_GRID) {
        return noise_radial_grid(p, sectors);
    }
    return cellular(p, lattice, ret, jitter, period);
}

// ---------------------------------------------------------------------------
// Fractal layering
// ---------------------------------------------------------------------------

// Per-octave distance-LOD weight. `max < min` is the "LOD off" sentinel, which
// is what the bake passes (there is no camera at bake time). The weight
// multiplies into the normalizing amplitude as well as the sum, so fading an
// octave out costs detail and never brightness.
fn octave_weight(i: u32, ofmin: f32, ofmax: f32) -> f32 {
    if (ofmax < ofmin) {
        return 1.0;
    }
    let t = (f32(i) - ofmin) / max(ofmax - ofmin, 1.0e-4);
    return 1.0 - clamp(t, 0.0, 1.0);
}

// Running amplitude-normalized accumulation. Dividing by the amplitude actually
// used (rather than the geometric-series limit) is what keeps the output range
// independent of the octave count, so adding detail never slides the whole
// texture along its colour ramp.
struct FbmAccum {
    sum: f32,
    max_amplitude: f32,
    // Ridged only: the previous octave's feedback weight.
    weight: f32,
};

fn fbm_new() -> FbmAccum {
    var a: FbmAccum;
    a.sum = 0.0;
    a.max_amplitude = 0.0;
    a.weight = 1.0;
    return a;
}

fn fbm_push(
    accum: FbmAccum,
    n_in: f32,
    amplitude: f32,
    w: f32,
    fractal: u32,
    weighted_strength: f32,
) -> FbmAccum {
    var a = accum;
    var n = n_in;
    if (fractal == FRACTAL_RIDGED) {
        n = 1.0 - abs(n);
        n = n * n;
        n = n * a.weight;
        a.weight = clamp(n * weighted_strength, 0.0, 1.0);
    } else if (fractal == FRACTAL_RIDGED_FNL) {
        // FastNoiseLite's own ridge. `(1 - 2|n|)` is its fold; the `+1, *0.5` is
        // `NoiseTexture2D`'s remap to [0,1], done per octave so it shares
        // `fbm_finish`'s denominator and cancels — see
        // `runt_core::noise::Fractal::RidgedFnl`, which argues why that is not
        // the same as doing it once at the end.
        let m = abs(n);
        n = ((1.0 - 2.0 * m) * a.weight + 1.0) * 0.5;
        // `lerp(1, 1 - m, weighted_strength)`. The clamp is the one guard FNL
        // lacks; it cannot bite on a generator whose range is [-1, 1].
        a.weight = clamp(a.weight * (1.0 - weighted_strength * m), 0.0, 1.0);
    }
    a.sum = a.sum + n * amplitude * w;
    a.max_amplitude = a.max_amplitude + amplitude * w;
    return a;
}

// Whether an octave feeds the *next* one's amplitude, and so whether a
// faded-out octave may be skipped. The twin of
// `runt_core::noise::Fractal::feeds_forward`, and the reason `shader.wgsl` asks
// rather than testing one variant: skipping a ridge drops the suppression it
// owes the octaves above it, which is a different field, and there are two
// ridges to forget.
fn fractal_feeds_forward(fractal: u32) -> bool {
    return fractal == FRACTAL_RIDGED || fractal == FRACTAL_RIDGED_FNL;
}

fn fbm_finish(a: FbmAccum) -> f32 {
    if (a.max_amplitude > 0.0) {
        return a.sum / a.max_amplitude;
    }
    return 0.0;
}

// The octave window a *live* fragment should evaluate, as an `octave_weight`
// pair (DESIGN §7: live eval has no mip chain, so this is its substitute).
//
// `footprint` is how many octave-0 lattice cells one pixel covers — the same
// quantity a mip selector computes, taken from `dpdx`/`dpdy` of the world
// position rather than of a texture coordinate. An octave whose cells are
// finer than `cell_pixels` pixels across is under-sampled: evaluating it adds
// noise the screen cannot resolve, which is exactly the shimmer mipmaps exist
// to remove, so it fades out instead.
//
// Solving `freq_i · footprint · cell_pixels = 1` for `i` gives the top of the
// window; `min` sits one octave below it, so the fade is a smooth octave wide
// rather than a popping step. `cell_pixels <= 0` returns the "LOD off"
// sentinel — every octave at full weight, which is what the bake passes.
//
// Distance from the camera is the *cause* of a large footprint and is what the
// Godot original ramped on; the footprint is the effect, and using it directly
// also handles a floor seen at a grazing angle, which no distance ramp can.
//
// `top` is floored at `1.0` — the CPU twin's doc comment
// (`TextureSpec::live_octave_window`) works the arithmetic in full, but the
// short version: unfloored, a large enough footprint pushes `top` to `0` or
// below, `octave_weight(0, top - 1, top)` is `0` there, and `fbm_finish`
// divides `0 / 0` into its zero fallback — the surface goes flat instead of
// merely blurry. Flooring pins the window at `(0.0, 1.0)`, which gives octave 0
// full weight (`octave_weight(0, 0, 1) == 1.0`, unconditionally) while octave 1
// still fades to `0` exactly as before. A live material may blur; it must
// never disappear.
fn live_octave_window(footprint: f32, log2_lacunarity: f32, cell_pixels: f32) -> vec2<f32> {
    if (cell_pixels <= 0.0) {
        return vec2<f32>(1.0, 0.0); // max < min: octave LOD off.
    }
    let top = max(-log2(max(footprint * cell_pixels, 1.0e-8)) / max(log2_lacunarity, 1.0e-3), 1.0);
    return vec2<f32>(top - 1.0, top);
}

// ---------------------------------------------------------------------------
// Post-processing and the colour ramp
// ---------------------------------------------------------------------------
//
// DESIGN §7 wants *one* WGSL source behind both texture modes. These two are
// where that bites: the bake reads its numbers out of `@group(0)`'s bake block
// and the live material variant reads them out of `@group(2)`'s texture block,
// so the values differ and the arithmetic must not. Passing them in as
// arguments is the whole trick — a bake and a live fragment that disagreed
// about contrast would be a look that changes when the perf gate flips, which
// is precisely what §11 forbids of a gated feature.
//
// `runt_core::texture::TextureSpec::{postprocess, ramp_at}` are the CPU twins.

// Contrast, brightness, clamp — in the original's order.
fn tex_postprocess(v: f32, contrast: f32, brightness: f32) -> f32 {
    let n = clamp((v - 0.5) * contrast + 0.5, 0.0, 1.0);
    return clamp(n * brightness, 0.0, 1.0);
}

// The gradient ramp. Linear between stops, held flat outside the ends —
// Godot's `GradientTexture1D` semantics, which is what the authored ramps were
// drawn against. `count == 0` is greyscale.
//
// `stops` is taken **by value**. A fixed-size array parameter is a copy, which
// is what lets one function serve two different uniform blocks; at eight
// `vec4`s it is 128 bytes of registers on a path that already carries a
// nineteen-cell Voronoi loop, and it is the price of not having the ramp
// written twice. `runt_core::texture::MAX_RAMP_STOPS` fixes the 8.
fn ramp_lookup(stops: array<vec4<f32>, 8>, count: u32, t_in: f32) -> vec3<f32> {
    let t = clamp(t_in, 0.0, 1.0);
    if (count == 0u) {
        return vec3<f32>(t, t, t);
    }
    var s = stops;
    if (t <= s[0].w) {
        return s[0].xyz;
    }
    for (var i = 1u; i < count; i = i + 1u) {
        let a = s[i - 1u];
        let b = s[i];
        if (t <= b.w) {
            let span = b.w - a.w;
            var f = 0.0;
            if (span > 1.0e-6) {
                f = (t - a.w) / span;
            }
            return mix(a.xyz, b.xyz, f);
        }
    }
    return s[count - 1u].xyz;
}
