//! The procedural-noise library, in Rust (DESIGN §7).
//!
//! This is the **CPU twin** of `noise.wgsl`, in the same sense `sky.rs` is the
//! twin of `sky.wgsl`: a line-for-line port of the shader, kept next to it so a
//! test can hold real baked texels against a model that never went near a GPU.
//! `tests/noise_bake.rs` samples a baked texture and demands it match
//! [`crate::texture::TextureSpec::albedo_at`] — which is built out of this
//! module — so the two copies cannot drift apart without a red test.
//!
//! It is also not *only* a test fixture. DESIGN §9 makes terrain an analytic
//! field that physics samples directly; unifying terrain colour with the baked
//! texture look later needs exactly these functions on the CPU.
//!
//! ## Provenance
//!
//! Ported from the 3dimenshift Godot shaders (`shaders/noise/*.gdshaderinc`,
//! recoverable at `d619383^`). The hashes are the well-known Dave Hoskins
//! large-coefficient `fract` family, chosen there and kept here because DESIGN
//! §7 requires *integer-style* hashing: cheap mobile GPUs run "highp" as fp24
//! internally, and a `sin(dot(p, big))` hash disintegrates at that precision.
//! Every constant below is load-bearing, including the `p3.yxx` in
//! [`hash33`] — it is asymmetric on purpose (it is what the original does) and
//! "fixing" it changes every texture the engine has ever baked.
//!
//! ## Seamless tiling
//!
//! Everything here takes an optional lattice **period**. Cellular noise depends
//! on its inputs only through integer cell indices, so wrapping the index (and
//! *only* the index — the feature-point positions stay unwrapped, or the
//! geometry would tear) makes the field exactly periodic. That is what lets a
//! bake tile with no blend, no mirror and no seam; see
//! [`crate::texture`] for how the per-octave periods are chosen.

use glam::Vec3;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

/// GLSL `fract`: `x - floor(x)`, always in `[0, 1)`.
///
/// **Not** `f32::fract`, which truncates towards zero and therefore returns
/// negative values for negative inputs. Cellular noise indexes cells with
/// `floor` and hashes them; getting this wrong mirrors the whole field about
/// the origin.
#[inline]
pub fn fract(x: f32) -> f32 {
    x - x.floor()
}

/// Component-wise [`fract`].
#[inline]
pub fn fract3(v: Vec3) -> Vec3 {
    Vec3::new(fract(v.x), fract(v.y), fract(v.z))
}

/// `hash11` from `noise_common.gdshaderinc`. Only used to derive seed offsets.
#[inline]
pub fn hash11(p: f32) -> f32 {
    let mut p = fract(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    fract(p)
}

/// `float hash13(vec3)` — one scalar from a lattice cell.
///
/// Note `p3.zyx` and the `31.32` (not the `33.33` its siblings use): both are
/// what the original says.
#[inline]
pub fn hash13(p3: Vec3) -> f32 {
    let mut p = fract3(p3 * 0.1031);
    p += Vec3::splat(p.dot(Vec3::new(p.z, p.y, p.x) + Vec3::splat(31.32)));
    fract((p.x + p.y) * p.z)
}

/// `vec3 hash33(vec3)` — the feature-point jitter.
///
/// The `p3.yxx` in the final swizzle (where symmetry would want `p3.yzz`) is
/// the quirk the port spec calls out. It is preserved deliberately.
#[inline]
pub fn hash33(p3: Vec3) -> Vec3 {
    let mut p = fract3(p3 * Vec3::new(0.1031, 0.1030, 0.0973));
    p += Vec3::splat(p.dot(Vec3::new(p.y, p.x, p.z) + Vec3::splat(33.33)));
    fract3((Vec3::new(p.x, p.x, p.y) + Vec3::new(p.y, p.x, p.x)) * Vec3::new(p.z, p.y, p.x))
}

/// Quintic fade `6t⁵ − 15t⁴ + 10t³`, C² continuous.
///
/// Carried over from the original library for completeness: the cellular path
/// does not interpolate, but the value/Perlin paths a later spec variant would
/// add all need exactly this curve, and porting it once is cheaper than porting
/// it under deadline.
#[inline]
pub fn quintic(t: Vec3) -> Vec3 {
    t * t * t * (t * (t * 6.0 - Vec3::splat(15.0)) + Vec3::splat(10.0))
}

/// A large pseudo-random displacement from a scalar seed, as
/// `seed_offset_3d`. Multiplied into the sample point so two specs that differ
/// only in `seed_offset` land on unrelated parts of the lattice.
#[inline]
pub fn seed_offset_3d(seed: f32) -> Vec3 {
    Vec3::new(
        hash11(seed) * 1000.0,
        hash11(seed + 47.32) * 1000.0,
        hash11(seed + 93.17) * 1000.0,
    )
}

// ---------------------------------------------------------------------------
// Cellular noise
// ---------------------------------------------------------------------------

/// Which lattice the Voronoi feature points sit on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect))]
pub enum Lattice {
    /// One point per unit cube. 27-cell search.
    Cubic,
    /// Face-centred cubic: integer coordinates whose components sum to an even
    /// number. The Voronoi cell is a rhombic dodecahedron rather than a
    /// distorted box, which is what stops the classic cubic-Worley "everything
    /// is secretly a grid" read at low octave counts. 19-cell search.
    ///
    /// Density is 0.5 points per unit volume against cubic's 1.0, so the same
    /// `frequency` reads coarser; the original compensates by ×2^(1/3) ≈ 1.26
    /// in the material, and the authored params already have that baked in.
    #[default]
    Fcc,
}

/// What a cellular sample returns.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect))]
pub enum CellReturn {
    /// Distance to the nearest feature point.
    F1,
    /// Distance to the second nearest.
    F2,
    /// `F2 − F1`: bright everywhere except on cell boundaries. Cracks, bark.
    F2MinusF1,
    /// The mean of the two.
    F1PlusF2,
    /// A flat random value per cell — the "shattered plates" look the terrain
    /// materials are built on.
    #[default]
    CellValue,
}

impl CellReturn {
    /// The integer the WGSL side switches on. Matches the original's
    /// `cell_return_type` uniform so the two dispatch tables cannot drift.
    pub fn code(self) -> u32 {
        match self {
            CellReturn::F1 => 0,
            CellReturn::F2 => 1,
            CellReturn::F2MinusF1 => 2,
            CellReturn::F1PlusF2 => 3,
            CellReturn::CellValue => 4,
        }
    }
}

/// How octaves are combined.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect))]
pub enum Fractal {
    /// Plain amplitude-normalized fBm.
    #[default]
    Fbm,
    /// `1 − |n|`, squared, with a feedback weight — ridges.
    Ridged,
}

impl Fractal {
    pub fn code(self) -> u32 {
        match self {
            Fractal::Fbm => 1,
            Fractal::Ridged => 2,
        }
    }
}

/// One cellular evaluation: the scalar plus the feature-point data the
/// boundary-normal accumulation needs.
///
/// `f1`/`f2` are *absolute* sample-space positions and `d1`/`d2` are true
/// distances (the Euclidean squares are rooted before they leave the search),
/// so a caller can reason about them in the units it handed in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CellSample {
    pub value: f32,
    pub f1: Vec3,
    pub f2: Vec3,
    pub d1: f32,
    pub d2: f32,
}

/// The 19 FCC cells searched: home, the 12 face neighbours at distance √2, and
/// the 6 axis-aligned ±2 cells.
///
/// The last six share only a vertex with home and matter solely when jitter is
/// large enough that a point two axis-steps away can beat everything in the
/// 13-cell set. Keeping the count flat at 19 rather than branching is what the
/// original does and what a GPU wants.
pub const FCC_OFFSETS: [Vec3; 19] = [
    Vec3::new(0.0, 0.0, 0.0),
    Vec3::new(1.0, 1.0, 0.0),
    Vec3::new(1.0, -1.0, 0.0),
    Vec3::new(-1.0, 1.0, 0.0),
    Vec3::new(-1.0, -1.0, 0.0),
    Vec3::new(1.0, 0.0, 1.0),
    Vec3::new(1.0, 0.0, -1.0),
    Vec3::new(-1.0, 0.0, 1.0),
    Vec3::new(-1.0, 0.0, -1.0),
    Vec3::new(0.0, 1.0, 1.0),
    Vec3::new(0.0, 1.0, -1.0),
    Vec3::new(0.0, -1.0, 1.0),
    Vec3::new(0.0, -1.0, -1.0),
    Vec3::new(2.0, 0.0, 0.0),
    Vec3::new(-2.0, 0.0, 0.0),
    Vec3::new(0.0, 2.0, 0.0),
    Vec3::new(0.0, -2.0, 0.0),
    Vec3::new(0.0, 0.0, 2.0),
    Vec3::new(0.0, 0.0, -2.0),
];

/// Round to the nearest FCC lattice point — an integer triple of even sum.
///
/// Plain rounding lands on an odd-parity cell half the time. When it does, the
/// axis whose rounding *cost the most* is nudged one step further towards `p`,
/// which is the cheapest fix that stays a nearest-ish neighbour and is
/// branch-uniform on a GPU. `sign(0) == 0` would leave the parity unfixed, so a
/// zero error picks `+1`.
pub fn fcc_round(p: Vec3) -> Vec3 {
    let mut c = (p + Vec3::splat(0.5)).floor();
    if fract((c.x + c.y + c.z) * 0.5) > 0.25 {
        // (x+y+z) odd. `fract(s/2) > 0.25` is the GPU-friendly parity test:
        // it is 0 for even sums and 0.5 for odd ones, at every magnitude a
        // float can still represent exactly.
        let err = (p - c).abs();
        let mut sgn = (p - c).signum();
        if p.x - c.x == 0.0 {
            sgn.x = 1.0;
        }
        if p.y - c.y == 0.0 {
            sgn.y = 1.0;
        }
        if p.z - c.z == 0.0 {
            sgn.z = 1.0;
        }
        if err.x >= err.y && err.x >= err.z {
            c.x += sgn.x;
        } else if err.y >= err.z {
            c.y += sgn.y;
        } else {
            c.z += sgn.z;
        }
    }
    c
}

/// Wrap a lattice index into `[0, period)` per axis, leaving axes whose period
/// is `0` alone. `cell` is a lattice index and is therefore integral.
///
/// Only the *hash input* is wrapped, never the position a distance is measured
/// from — that is the whole trick behind a seamless bake (see the module docs).
/// FCC periods must be even or the wrap would flip cell parity and the lattice
/// would fall apart at the seam; [`crate::texture`] enforces that when it plans
/// the octaves.
///
/// The snap-and-fix after the modulo mirrors `noise.wgsl` exactly, and exists
/// because of the GPU: a driver that lowers `a / b` to `a * (1/b)` computes
/// `78 / 6` as `12.9999995`, and `floor` then drops a whole period. On a value
/// that only ever feeds a hash, one-off is not a small error — it is a
/// different random number. Rounding to the integer the index is known to be
/// and folding the ±1-period case away makes the two sides agree exactly.
#[inline]
pub fn wrap_cell(cell: Vec3, period: Vec3) -> Vec3 {
    let one = |c: f32, p: f32| {
        if p <= 0.0 {
            return c;
        }
        let mut m = (c - (c / p).floor() * p + 0.5).floor();
        if m < 0.0 {
            m += p;
        }
        if m >= p {
            m -= p;
        }
        m
    };
    Vec3::new(
        one(cell.x, period.x),
        one(cell.y, period.y),
        one(cell.z, period.z),
    )
}

/// Euclidean squared distance. The only metric the port needs — every authored
/// material uses it — so Manhattan/Chebyshev are deliberately not carried over.
#[inline]
fn dist_sq(a: Vec3, b: Vec3) -> f32 {
    let d = a - b;
    d.dot(d)
}

/// Cellular noise at `p`, tracking both nearest and second-nearest points.
///
/// `period` wraps the *cell index* used for hashing (see [`wrap_cell`]); pass
/// [`Vec3::ZERO`] for the unbounded field.
pub fn cellular(
    p: Vec3,
    lattice: Lattice,
    ret: CellReturn,
    jitter: f32,
    period: Vec3,
) -> CellSample {
    let mut f1 = f32::INFINITY;
    let mut f2 = f32::INFINITY;
    let mut nearest_cell = Vec3::ZERO;
    let mut nearest_pos = Vec3::ZERO;
    let mut second_pos = Vec3::ZERO;

    let mut consider = |cell: Vec3, point: Vec3| {
        let d = dist_sq(point, p);
        if d < f1 {
            f2 = f1;
            second_pos = nearest_pos;
            f1 = d;
            nearest_cell = cell;
            nearest_pos = point;
        } else if d < f2 {
            f2 = d;
            second_pos = point;
        }
    };

    match lattice {
        Lattice::Cubic => {
            let base = p.floor();
            for z in -1..=1 {
                for y in -1..=1 {
                    for x in -1..=1 {
                        let cell = base + Vec3::new(x as f32, y as f32, z as f32);
                        let key = wrap_cell(cell, period);
                        // Cubic feature points live *inside* the cell: a
                        // half-unit centre plus jitter, unlike FCC's, which sit
                        // on the lattice point itself.
                        let local = Vec3::splat(0.5) + (hash33(key) - Vec3::splat(0.5)) * jitter;
                        consider(key, cell + local);
                    }
                }
            }
        }
        Lattice::Fcc => {
            let home = fcc_round(p);
            for offset in FCC_OFFSETS {
                let cell = home + offset;
                let key = wrap_cell(cell, period);
                let point = cell + (hash33(key) - Vec3::splat(0.5)) * jitter;
                consider(key, point);
            }
        }
    }

    let d1 = f1.sqrt();
    let d2 = f2.sqrt();
    let value = match ret {
        CellReturn::F1 => d1,
        CellReturn::F2 => d2,
        CellReturn::F2MinusF1 => d2 - d1,
        CellReturn::F1PlusF2 => (d1 + d2) * 0.5,
        CellReturn::CellValue => hash13(nearest_cell),
    };
    CellSample {
        value,
        f1: nearest_pos,
        f2: second_pos,
        d1,
        d2,
    }
}

// ---------------------------------------------------------------------------
// Fractal layering
// ---------------------------------------------------------------------------

/// The "octave LOD off" sentinel: any `(min, max)` with `max < min`.
pub const OCTAVE_LOD_OFF: (f32, f32) = (1.0, 0.0);

/// Per-octave fade weight, `1 − clamp((i − min) / (max − min), 0, 1)`.
///
/// The distance-LOD knob the original drives from view depth: octaves past the
/// window contribute nothing, so a distant surface pays for two octaves instead
/// of five. The bake has no camera, so it passes [`OCTAVE_LOD_OFF`] and every
/// weight is 1 — but the weight still multiplies into *both* the sum and the
/// normalizing amplitude, which is exactly why fading an octave out darkens
/// nothing (see [`normalized_fbm`]).
#[inline]
pub fn octave_weight(i: u32, min: f32, max: f32) -> f32 {
    if max < min {
        return 1.0;
    }
    let t = (i as f32 - min) / (max - min).max(1e-4);
    1.0 - t.clamp(0.0, 1.0)
}

/// Fold one octave's raw noise into a running amplitude-normalized sum.
///
/// Split out of the loop so the ordinary fBm path, the ridged path and the
/// normal-accumulating path in [`crate::texture`] cannot disagree about the
/// arithmetic.
#[derive(Clone, Copy, Debug, Default)]
pub struct FbmAccum {
    pub sum: f32,
    pub max_amplitude: f32,
    /// Ridged only: the previous octave's feedback weight.
    pub weight: f32,
}

impl FbmAccum {
    pub fn new() -> FbmAccum {
        FbmAccum {
            sum: 0.0,
            max_amplitude: 0.0,
            weight: 1.0,
        }
    }

    /// Add octave `n` with amplitude `amplitude` and LOD weight `w`.
    ///
    /// `weighted_strength` only matters for [`Fractal::Ridged`], where it is the
    /// feedback gain that makes a ridge suppress the octave above it.
    pub fn push(&mut self, n: f32, amplitude: f32, w: f32, fractal: Fractal, weighted_strength: f32) {
        let n = match fractal {
            Fractal::Fbm => n,
            Fractal::Ridged => {
                let mut r = 1.0 - n.abs();
                r *= r;
                r *= self.weight;
                self.weight = (r * weighted_strength).clamp(0.0, 1.0);
                r
            }
        };
        self.sum += n * amplitude * w;
        self.max_amplitude += amplitude * w;
    }

    /// The normalized result. Dividing by the amplitude actually used — rather
    /// than by the geometric-series limit — is what makes the output range
    /// independent of the octave count, so raising `octaves` adds detail
    /// without shifting the whole texture's brightness (and therefore without
    /// sliding it along the colour ramp).
    pub fn finish(&self) -> f32 {
        if self.max_amplitude > 0.0 {
            self.sum / self.max_amplitude
        } else {
            0.0
        }
    }
}

/// Amplitude-normalized fractal cellular noise at `p`, unbounded (no tiling).
///
/// The plain entry point, for callers that just want a number. The bake path
/// goes through [`crate::texture`] instead, because it needs the feature points
/// for boundary normals and the per-octave periods for seamlessness.
#[allow(clippy::too_many_arguments)]
pub fn normalized_fbm(
    p: Vec3,
    lattice: Lattice,
    ret: CellReturn,
    jitter: f32,
    octaves: u32,
    lacunarity: f32,
    gain: f32,
    fractal: Fractal,
    weighted_strength: f32,
) -> f32 {
    let mut accum = FbmAccum::new();
    let mut freq = 1.0f32;
    let mut amplitude = 1.0f32;
    for i in 0..octaves {
        let w = octave_weight(i, OCTAVE_LOD_OFF.0, OCTAVE_LOD_OFF.1);
        let n = cellular(p * freq, lattice, ret, jitter, Vec3::ZERO).value;
        accum.push(n, amplitude, w, fractal, weighted_strength);
        freq *= lacunarity;
        amplitude *= gain;
    }
    accum.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fract_is_the_glsl_one_not_the_rust_one() {
        assert_eq!(fract(1.25), 0.25);
        assert_eq!(fract(-0.25), 0.75);
        assert!(fract(-3.5) >= 0.0);
    }

    #[test]
    fn the_hashes_stay_inside_the_unit_range() {
        for i in 0..2000 {
            let p = Vec3::new(i as f32 * 0.37 - 300.0, i as f32 * -1.1, i as f32 * 7.0 - 91.0);
            let h1 = hash13(p);
            assert!((0.0..1.0).contains(&h1), "hash13({p:?}) = {h1}");
            let h3 = hash33(p);
            for c in h3.to_array() {
                assert!((0.0..1.0).contains(&c), "hash33({p:?}) = {h3:?}");
            }
        }
    }

    #[test]
    fn fcc_rounding_always_lands_on_an_even_parity_integer() {
        for i in 0..3000 {
            let p = Vec3::new(
                (i as f32 * 0.131).sin() * 40.0,
                (i as f32 * 0.717).cos() * 40.0,
                (i as f32 * 0.313).sin() * 40.0,
            );
            let c = fcc_round(p);
            for v in c.to_array() {
                assert_eq!(v, v.round(), "{c:?} is not integral");
            }
            let sum = c.x + c.y + c.z;
            assert_eq!(sum % 2.0, 0.0, "{c:?} has odd parity");
            // And it stays a near neighbour: never further than the FCC
            // circumradius plus the one-step parity fix.
            assert!((p - c).length() <= 2.0, "{p:?} rounded to distant {c:?}");
        }
    }

    #[test]
    fn wrapping_a_cell_index_is_a_true_modulus() {
        assert_eq!(wrap_cell(Vec3::new(7.0, -1.0, 3.0), Vec3::new(6.0, 6.0, 0.0)),
                   Vec3::new(1.0, 5.0, 3.0));
        // Period 0 means "do not wrap this axis".
        assert_eq!(wrap_cell(Vec3::new(9.0, 9.0, 9.0), Vec3::ZERO), Vec3::splat(9.0));
        // Even periods preserve FCC parity, which is why texture.rs rounds to
        // even; an odd one would not, and this documents the difference.
        let cell = Vec3::new(5.0, 3.0, 0.0); // sum 8, even
        let wrapped = wrap_cell(cell, Vec3::new(4.0, 4.0, 0.0));
        assert_eq!((wrapped.x + wrapped.y + wrapped.z) % 2.0, 0.0);
    }

    #[test]
    // The 2.718 below is grass's authored lacunarity, not an approximation of
    // `e`; clippy cannot tell them apart.
    #[allow(clippy::approx_constant)]
    fn the_fbm_range_does_not_move_with_the_octave_count() {
        // The point of amplitude normalization: 1 octave and 6 octaves of the
        // same field occupy the same range, so the colour ramp does not have to
        // be re-authored every time detail is added.
        let mut bounds = Vec::new();
        for octaves in [1u32, 2, 3, 5, 8] {
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            for i in 0..300 {
                let p = Vec3::new(i as f32 * 0.13, i as f32 * -0.07, i as f32 * 0.31);
                let n = normalized_fbm(
                    p,
                    Lattice::Fcc,
                    CellReturn::CellValue,
                    1.0,
                    octaves,
                    2.718,
                    0.562,
                    Fractal::Fbm,
                    0.5,
                );
                assert!((0.0..=1.0).contains(&n), "{octaves} octaves gave {n}");
                lo = lo.min(n);
                hi = hi.max(n);
            }
            bounds.push((lo, hi));
        }
        // Every octave count covers a comparable slice of [0,1]; without the
        // normalization the 8-octave mean would sit near 1/(1-gain) times the
        // 1-octave one.
        let means: Vec<f32> = bounds.iter().map(|(lo, hi)| (lo + hi) * 0.5).collect();
        let first = means[0];
        for m in &means {
            assert!(
                (m - first).abs() < 0.25,
                "octave count moved the mid-range: {means:?}"
            );
        }
    }

    #[test]
    fn cellular_is_a_pure_function() {
        let p = Vec3::new(3.7, -2.1, 0.5);
        let a = cellular(p, Lattice::Fcc, CellReturn::CellValue, 1.0, Vec3::ZERO);
        let b = cellular(p, Lattice::Fcc, CellReturn::CellValue, 1.0, Vec3::ZERO);
        assert_eq!(a, b);
    }

    #[test]
    fn the_two_lattices_are_actually_different_fields() {
        let p = Vec3::new(1.3, 4.9, -2.2);
        let cubic = cellular(p, Lattice::Cubic, CellReturn::F1, 1.0, Vec3::ZERO);
        let fcc = cellular(p, Lattice::Fcc, CellReturn::F1, 1.0, Vec3::ZERO);
        assert_ne!(cubic.value, fcc.value);
    }
}
