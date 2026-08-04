// The procedural-noise library (DESIGN §7).
//
// A *library*, not a shader: no bindings, no entry points, nothing that assumes
// a pass. It is prepended to whichever shader needs it (`bake.wgsl` today, the
// live-eval material variant if §7's perf gate ever opens) by
// `runt_core::texture_shader_source`.
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
    }
    a.sum = a.sum + n * amplitude * w;
    a.max_amplitude = a.max_amplitude + amplitude * w;
    return a;
}

fn fbm_finish(a: FbmAccum) -> f32 {
    if (a.max_amplitude > 0.0) {
        return a.sum / a.max_amplitude;
    }
    return 0.0;
}
