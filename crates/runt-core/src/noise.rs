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

use glam::{Vec2, Vec3};
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

/// Which field a texture is made of.
///
/// The dispatch [`field`] switches on, and the value
/// [`NoiseSpec`](crate::texture::NoiseSpec) carries. Two entries rather than
/// one because [`Grid`](NoiseKind::Grid) is not a *parameterisation* of
/// [`Cellular`](NoiseKind::Cellular) — it is a closed form that replaces the
/// search entirely, and giving it its own code is what lets the shader skip the
/// loop rather than run it with degenerate arguments.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect))]
pub enum NoiseKind {
    /// Voronoi over hashed feature points — [`cellular`].
    #[default]
    Cellular,
    /// The jitter-free cubic lattice in closed form — [`grid`].
    Grid,
    /// The same grid in cylindrical coordinates about `+Y`: wedges and bands
    /// about an axis — [`radial_grid`].
    RadialGrid,
}

impl NoiseKind {
    /// The integer the WGSL side switches on, mirroring `noise.wgsl`'s `KIND_*`.
    pub fn code(self) -> u32 {
        match self {
            NoiseKind::Cellular => 0,
            NoiseKind::Grid => 1,
            NoiseKind::RadialGrid => 2,
        }
    }

    /// Whether the field is defined **about an axis** through the sample-space
    /// origin — and therefore whether translating that origin destroys it.
    ///
    /// [`seed_offset_3d`] hands out displacements in the *thousands*, which is
    /// exactly right for a hashed lattice: the field is translation-invariant in
    /// distribution, so a large shift lands two specs on unrelated parts of it
    /// and correlates nothing. [`RadialGrid`](NoiseKind::RadialGrid) is not
    /// translation-invariant at all. Its wedges radiate from `x = z = 0`, so a
    /// shift of a few hundred units does not decorrelate it — it moves the axis
    /// off the object entirely and leaves the surface sampling a sliver of one
    /// enormous wedge, which reads as *no wedges at all*.
    ///
    /// That is not hypothetical: it is what shipped for one render of the
    /// player's ball. `seed_offset: 0.0` looks like "no offset", and it is not —
    /// `hash11(0)` is zero but `hash11(47.32)` and `hash11(93.17)` are not, so
    /// the default seed still displaces `y` and `z` by −3.8 and **365.8**.
    /// [`TextureSpec::seed_displacement`](crate::texture::TextureSpec::seed_displacement)
    /// is what reads this and keeps only the component *along* the axis, which
    /// slides the bands without moving the centre.
    pub fn has_axis(self) -> bool {
        match self {
            NoiseKind::Cellular | NoiseKind::Grid => false,
            NoiseKind::RadialGrid => true,
        }
    }
}

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
    /// `1 − |n|`, **squared**, with a *replacing* feedback weight — ridges.
    ///
    /// This engine's own ridge, and the one every material authored before
    /// [`RidgedFnl`](Fractal::RidgedFnl) existed is tuned against. The square
    /// makes the ridge lines thin and the cells fat; the feedback replaces the
    /// running weight with `n · weighted_strength` each octave, so a
    /// `weighted_strength` below ~0.5 suppresses everything above octave 0
    /// almost completely.
    Ridged,
    /// FastNoiseLite's `FRACTAL_RIDGED`, bit for bit: `1 − 2|n|`, **not**
    /// squared, with an amplitude the previous octave *lerps* rather than
    /// replaces.
    ///
    /// # Why this exists beside [`Ridged`](Fractal::Ridged) rather than fixing it
    ///
    /// The two are the same idea and different arithmetic, and the number that
    /// tells them apart is `weighted_strength` — it is a *replacing gain* to one
    /// and a *lerp factor* to the other, so the same authored value produces
    /// visibly different textures. A material transcribed from a Godot
    /// `FastNoiseLite` therefore cannot use [`Ridged`](Fractal::Ridged) and be
    /// faithful, and a material tuned against [`Ridged`](Fractal::Ridged) cannot
    /// be moved to this one without being re-tuned. Changing the existing
    /// variant in place would silently re-tone every ridged material in every
    /// scene; adding one lets a transcription say which arithmetic it means.
    ///
    /// # The fold, and where the halving comes from
    ///
    /// FastNoiseLite accumulates `(1 − 2|n|) · amp` into a sum it divides by
    /// `Σ gain^i`, which lands in `[−1, 1]`; Godot's `NoiseTexture2D` then maps
    /// that to `[0, 1]` with `(x + 1) / 2` before the colour ramp sees it.
    /// [`FbmAccum::push`] does the `+1, ×0.5` **per octave** instead of once at
    /// the end, which is not a rearrangement for tidiness: `finish` divides by
    /// the amplitude actually used, so folding the constant in per octave is
    /// what makes it share that same denominator and cancel. Doing it once in
    /// `finish` would divide the `+1` by `Σ gain^i` and the rest by
    /// `Σ gain^i · w_i`, and the two only agree when `weighted_strength` is 0.
    ///
    /// `the_fnl_ridge_is_fastnoiselites_own` measures the whole chain against a
    /// transcription of the original loop: they agree to float epsilon.
    RidgedFnl,
}

impl Fractal {
    pub fn code(self) -> u32 {
        match self {
            Fractal::Fbm => 1,
            Fractal::Ridged => 2,
            Fractal::RidgedFnl => 3,
        }
    }

    /// Whether an octave feeds the *next* one's amplitude, and therefore
    /// whether a faded-out octave may be skipped.
    ///
    /// Both ridges do. `shader.wgsl`'s live path skips octaves whose LOD weight
    /// has reached zero — arithmetically identical under [`Fbm`](Fractal::Fbm),
    /// where an octave contributes to nothing but its own term — and skipping a
    /// ridge would drop the suppression it owes the octaves above it, which
    /// *is* a different field. Asking here rather than testing one variant at
    /// the call site is what stopped the second ridge from being forgotten
    /// there.
    pub fn feeds_forward(self) -> bool {
        match self {
            Fractal::Fbm => false,
            Fractal::Ridged | Fractal::RidgedFnl => true,
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

/// The jitter-free cubic lattice, in closed form: a **grid of round cells**,
/// with no hash and no neighbourhood search.
///
/// # It is not an imitation of [`cellular`] — it is the same field
///
/// Set [`Lattice::Cubic`] and `jitter = 0` and every feature point snaps to its
/// cell's centre. The 27-cell search is then computing something a line of
/// algebra already knows: with `q = fract(p) − 0.5` the offset from the nearest
/// centre, the nearest point is that centre, and the second-nearest is one unit
/// step along whichever axis the sample has drifted furthest down — no other
/// neighbour can beat it, because stepping along a second axis adds a whole
/// unit and buys back less than one.
///
/// ```text
/// q    = fract(p) − 0.5          offset from the nearest cell centre
/// d1²  = q · q
/// m    = max(|q.x|, |q.y|, |q.z|)
/// d2²  = d1² + 1 − 2m            one step along the dominant axis
/// value = 1 − d1²/d2²
/// ```
///
/// `the_grid_is_jitter_free_cellular` holds that against a real 27-cell search
/// at 200k points; they agree to float epsilon. So this is a *fast path* that
/// happens to be spelled as its own [`NoiseKind`], not a lookalike — the ratio
/// `d1²/d2²` is exactly FastNoiseLite's `RETURN_DISTANCE2_DIV` under
/// `DISTANCE_EUCLIDEAN_SQUARED`, which is what `3dimenshift`'s player material
/// is authored with (`player.tscn:91-101`).
///
/// # The value's polarity, and why it is `1 − ratio` rather than `ratio − 1`
///
/// FastNoiseLite returns `d1²/d2² − 1`, which lands in `[−1, 0]`: negative
/// everywhere, and under [`Fractal::Fbm`] the ramp would clamp the whole
/// texture to black. This returns `1 − d1²/d2²` instead — `[0, 1]`, **1 at a
/// cell centre and 0 on the boundary** — so fBm reads as round blobs and the
/// ramp gets the range it expects.
///
/// Under **either** ridge the two are *identical*, which is the case that
/// matters, because both take `|n|` before they do anything else and
/// `|1 − ratio| = |ratio − 1|`. So the polarity is free to be chosen for fBm's
/// sake: [`Fractal::Ridged`] arrives at `ratio²` and [`Fractal::RidgedFnl`] at
/// `2·ratio − 1` whichever spelling it is handed — both bright on the lattice
/// lines and dark in the cells, which is the read the Godot material's
/// *decreasing* ramp was drawn against.
/// `the_grid_ridges_the_same_either_way` pins the first and
/// `the_fnl_ridge_is_fastnoiselites_own` the second.
///
/// # No `period`, because it does not need one
///
/// [`cellular`] tiles by wrapping the cell index it hashes ([`wrap_cell`]).
/// There is no hash here, so there is nothing to wrap: the field is *already*
/// exactly periodic with period 1, and any tile spanning a whole number of
/// cells wraps with no seam and no blend. [`crate::texture::TextureSpec`]'s
/// span quantization rounds to a whole number on [`Lattice::Cubic`], so a grid
/// spec gets that for free.
///
/// `f1`/`f2`/`d1`/`d2` are filled in the same units and with the same meaning
/// [`cellular`] gives them, so the boundary-normal accumulation in
/// [`crate::texture`] needs no special case.
pub fn grid(p: Vec3) -> CellSample {
    // Cubic feature points sit at cell centre — `cell + 0.5`, matching
    // `cellular`'s cubic branch, so the two fields coincide rather than being
    // half a cell out of step.
    let base = p.floor();
    let centre = base + Vec3::splat(0.5);
    let q = p - centre;
    let a = q.abs();
    let m = a.max_element();

    let d1_sq = q.dot(q);
    // `d1² ≥ m²` and `m ≤ 0.5`, so this is at least `(1 − m)² ≥ 0.25`. It
    // cannot reach zero and needs no epsilon guard.
    let d2_sq = d1_sq + 1.0 - 2.0 * m;

    // The dominant axis, stepped *towards* the sample. `f32::signum` answers
    // `+1` for `+0.0`, which is the same "a zero error picks +1" tie-break
    // `fcc_round` makes.
    let axis = if a.x >= a.y && a.x >= a.z {
        Vec3::X
    } else if a.y >= a.z {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let step = axis * q.dot(axis).signum();

    CellSample {
        value: 1.0 - d1_sq / d2_sq,
        f1: centre,
        f2: centre + step,
        d1: d1_sq.sqrt(),
        d2: d2_sq.sqrt(),
    }
}

/// The same grid in **cylindrical** coordinates about the `+Y` axis: wedges
/// around and bands up.
///
/// # What it is for
///
/// [`grid`] is a lattice of axis-aligned boxes, so a sphere cutting through it
/// shows the *planes* it happens to intersect. A UV-mapped texture on a sphere
/// does something different and more familiar: `u` runs around the equator and
/// `v` from pole to pole, so the pattern is a set of longitude wedges and
/// latitude bands that meet at the poles. This is that, as a 3D field —
/// which is what a ball wants, and what `3dimenshift`'s player material gets
/// from its UV mapping for free.
///
/// # The warp
///
/// ```text
/// θ = atan2(p.z, p.x) / τ + 0.5      the turn, in [0, 1)
/// u = θ · sectors                    wedges — an integer count, see below
/// v = p.y                            bands
/// w = 0.5                            pinned: no rings
/// ```
///
/// and then [`grid`] on `(u, v, w)`. `θ` is invariant under a positive uniform
/// scale of `p`, which is why the angular density cannot come from the octave's
/// frequency the way `v` does and has to be passed in.
///
/// # Why there is no third dimension
///
/// The obvious third coordinate is `length(p.xz)` — rings out from the axis —
/// and it is deliberately *not* used. A UV wrap has exactly two axes, and rings
/// are a third: constant-radius cylinders cut a sphere in latitude circles,
/// which lands a second set of horizontal bands on top of the ones `v` already
/// draws and leaves the wedges the harder thing to see. Both were rendered and
/// compared; the two-axis form is the one that reads as the original's UV grid.
///
/// Pinning `w` at `0.5` puts it exactly on a cell centre, so `q_w` is zero, the
/// third axis never wins the `max` and [`grid`] degenerates cleanly to a plane
/// lattice. The field is then constant along any ray from the axis — which for
/// a surface wrapped *around* that axis is not a loss of anything visible.
///
/// # `sectors` must be a whole number, per octave
///
/// `u` wraps from `sectors` back to `0` at `θ = 1`, and [`grid`] is 1-periodic,
/// so the two sides of that wrap are the same cell **only if `sectors` is an
/// integer**. A fractional count puts a seam down the `+X` half-plane. The
/// rounding happens here, with `floor(x + 0.5)` rather than `round` — WGSL's
/// `round` is round-half-to-even and Rust's is round-half-away-from-zero, and a
/// sector count that disagreed between the two would be a different field on
/// each side.
///
/// Callers pass `sectors × octave_freq` so the wedges refine with the octave
/// like everything else; `the_radial_grid_has_no_seam_at_the_wrap` measures the
/// result across `θ = 0`.
///
/// # The axis is a singularity, and so are a UV sphere's poles
///
/// Every wedge meets on the `Y` axis, where `θ` is undefined and `atan2(0, 0)`
/// answers `0`. That is not a defect to be smoothed away: it is exactly what a
/// UV sphere does at its poles, where every column of `u` converges, and
/// reproducing it is the point.
///
/// # `f1`/`f2` come back in the *caller's* space
///
/// The cell centres [`grid`] finds are points in `(u, v, w)`, and the
/// boundary-normal accumulation in [`crate::texture`] subtracts them from `p`
/// and treats the difference as a direction. So they are mapped back before
/// they leave — a centre at `(u_c, v_c, w_c)` is
/// `(w_c·cos θ_c, v_c, w_c·sin θ_c)` with `θ_c = (u_c / sectors)·τ`. `d1`/`d2`
/// stay as measured, in warped units, because they are only ever compared with
/// each other (`d2 − d1` is the edge term). Mixing the two is deliberate and is
/// the combination that makes the normal point somewhere real.
pub fn radial_grid(p: Vec3, sectors: f32) -> CellSample {
    // At least one wedge, and a whole number of them. See the docs above for
    // why this is `floor(x + 0.5)` and not `round`.
    let s = (sectors + 0.5).floor().max(1.0);
    let turn = p.z.atan2(p.x) / std::f32::consts::TAU + 0.5;
    let radius = Vec2::new(p.x, p.z).length();

    let cell = grid(Vec3::new(turn * s, p.y, 0.5));

    // Back to the caller's basis, on the sample's own cylinder. `u / s` is the
    // turn the centre sits at.
    let unwarp = |q: Vec3| {
        let angle = (q.x / s) * std::f32::consts::TAU;
        Vec3::new(radius * angle.cos(), q.y, radius * angle.sin())
    };
    CellSample {
        f1: unwarp(cell.f1),
        f2: unwarp(cell.f2),
        ..cell
    }
}

/// One noise evaluation, whichever kind the spec asked for.
///
/// The single seam both texture paths go through — [`crate::texture`]'s bake
/// twin and its live twin — and the exact counterpart of `noise.wgsl`'s
/// `noise_field`. Adding a kind means adding an arm here and there, and
/// nowhere else.
///
/// `lattice`, `ret`, `jitter` and `period` are [`cellular`]'s arguments and mean
/// nothing to the two grids; `sectors` is [`radial_grid`]'s and means nothing to
/// the other two. Callers pass `sectors × octave_freq` — the wedge count refines
/// with the octave, and [`radial_grid`] rounds it to a whole number.
pub fn field(
    p: Vec3,
    kind: NoiseKind,
    lattice: Lattice,
    ret: CellReturn,
    jitter: f32,
    sectors: f32,
    period: Vec3,
) -> CellSample {
    match kind {
        NoiseKind::Cellular => cellular(p, lattice, ret, jitter, period),
        NoiseKind::Grid => grid(p),
        NoiseKind::RadialGrid => radial_grid(p, sectors),
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
    /// `weighted_strength` is read by both ridges and means something different
    /// to each — a replacing gain to [`Fractal::Ridged`], a lerp factor to
    /// [`Fractal::RidgedFnl`]. [`Fractal::Fbm`] ignores it.
    ///
    /// `self.weight` is likewise per-variant state: `Ridged` **replaces** it
    /// every octave, `RidgedFnl` **multiplies** into it. One field, because no
    /// accumulator is ever asked for two fractals, and the arms are the only
    /// places that read it.
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
            Fractal::RidgedFnl => {
                let a = n.abs();
                // `(1 − 2a)` is FNL's fold; the `+1, ×0.5` is `NoiseTexture2D`'s
                // remap to `[0, 1]`, applied here rather than in `finish` so it
                // shares that function's denominator. The variant's own docs
                // work through why the two are not the same place.
                let r = ((1.0 - 2.0 * a) * self.weight + 1.0) * 0.5;
                // `lerp(1, 1 − a, weighted_strength)`, i.e. `1 − ws·a`.
                //
                // The clamp is the one thing FNL does not do. It cannot bite on
                // any generator whose range is `[−1, 1]` — `a ≤ 1` and `ws ≤ 1`
                // keep the factor in `[0, 1]` on their own — but a return type
                // that can exceed 1 (`F2` on a sparse lattice) would drive it
                // negative and *invert* every octave above, which is never what
                // an author meant by "suppress".
                self.weight = (self.weight * (1.0 - weighted_strength * a)).clamp(0.0, 1.0);
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

    /// A deterministic sample walk over a few thousand points, avoiding
    /// `rand` — the same trick the fBm test above uses.
    fn walk(i: u32) -> Vec3 {
        let f = i as f32;
        Vec3::new(f * 0.137 - 9.0, f * -0.0791 + 4.0, f * 0.2113 - 2.0)
    }

    /// The load-bearing claim of [`grid`]: it is not *like* a jitter-free
    /// cubic Voronoi, it **is** one, and a real 27-cell search says so.
    ///
    /// Both the value and the feature-point data are checked, because the
    /// boundary-normal accumulation in `crate::texture` reads `f1`/`f2`/`d1`/`d2`
    /// and a closed form that got the scalar right and the second-nearest point
    /// wrong would ship a correct albedo on a wrong normal map.
    #[test]
    fn the_grid_is_jitter_free_cellular() {
        for i in 0..20_000 {
            let p = walk(i);
            let want = cellular(p, Lattice::Cubic, CellReturn::F1, 0.0, Vec3::ZERO);
            let got = grid(p);

            assert!(
                (got.d1 - want.d1).abs() < 1.0e-5,
                "{p:?}: d1 {} vs {}",
                got.d1,
                want.d1
            );
            assert!(
                (got.d2 - want.d2).abs() < 1.0e-5,
                "{p:?}: d2 {} vs {}",
                got.d2,
                want.d2
            );
            // Only where there is a winner to agree on. A sample sitting on a
            // cell *corner* is equidistant from two centres, and which one a
            // search calls "nearest" is then an artifact of iteration order —
            // `d1`, `d2` and the value are identical either way, which is why
            // they are asserted unconditionally above.
            if want.d2 - want.d1 > 1.0e-4 {
                assert!(
                    (got.f1 - want.f1).length() < 1.0e-5,
                    "{p:?}: f1 {:?} vs {:?}",
                    got.f1,
                    want.f1
                );
            }
            // The scalar the search would have produced with FastNoiseLite's
            // `RETURN_DISTANCE2_DIV`, in this function's own polarity.
            let ratio = (want.d1 * want.d1) / (want.d2 * want.d2);
            assert!(
                (got.value - (1.0 - ratio)).abs() < 1.0e-5,
                "{p:?}: value {} vs {}",
                got.value,
                1.0 - ratio
            );
        }
    }

    /// `f2` is the *second* nearest centre and not merely some neighbour: it is
    /// exactly one unit step from `f1`, along one axis.
    ///
    /// `the_grid_is_jitter_free_cellular` compares `f2`'s distance against the
    /// search but not its position, because two centres can tie — this pins the
    /// shape of the answer instead, which is what `NormalMode::ToEdge` reads.
    #[test]
    fn the_grids_second_point_is_one_axis_step_away() {
        for i in 0..2_000 {
            let step = grid(walk(i)).f2 - grid(walk(i)).f1;
            let a = step.abs();
            assert!(
                (a.max_element() - 1.0).abs() < 1.0e-5 && a.min_element() < 1.0e-5,
                "{step:?} is not a unit step along one axis"
            );
        }
    }

    /// Under [`Fractal::Ridged`] this function's `1 − ratio` and
    /// FastNoiseLite's own `ratio − 1` are the *same* fold, which is what lets
    /// the polarity be chosen for fBm's sake without changing the look of the
    /// material that motivated the variant.
    ///
    /// `FbmAccum::push` folds `1 − |n|`, and `|1 − r| = |r − 1|`, so the two
    /// differ by nothing at all once the absolute value has run.
    #[test]
    fn the_grid_ridges_the_same_either_way() {
        for i in 0..2_000 {
            let ours = grid(walk(i)).value; // `1 − ratio`
            let fnl = -ours; // `ratio − 1`, FastNoiseLite's spelling.
            for fractal in [Fractal::Ridged, Fractal::RidgedFnl] {
                let mut a = FbmAccum::new();
                let mut b = FbmAccum::new();
                a.push(ours, 1.0, 1.0, fractal, 0.22);
                b.push(fnl, 1.0, 1.0, fractal, 0.22);
                assert_eq!(a.finish(), b.finish(), "{fractal:?}");
            }
        }
    }

    /// [`Fractal::RidgedFnl`] is FastNoiseLite's ridge and not merely something
    /// like it — held against a direct transcription of the original loop.
    ///
    /// The reference below is `GenFractalRidged` followed by Godot's
    /// `NoiseTexture2D` remap, written the way the C# reads it: a
    /// `fractalBounding` computed once from `gain` and the octave count, a
    /// running `amp` that carries *both* the gain and the per-octave lerp, and a
    /// single `(x + 1) / 2` at the end. [`FbmAccum`] does none of those three
    /// things in the same place — it takes `gain^i` from the octave plan, keeps
    /// the lerp product separately, and folds the remap in per octave — so an
    /// agreement here is a real result about the rearrangement rather than a
    /// restatement of it.
    ///
    /// Driven by [`grid`] because that is the field the variant was added for,
    /// and because a closed form contributes no noise of its own to hide a
    /// discrepancy in.
    #[test]
    fn the_fnl_ridge_is_fastnoiselites_own() {
        const OCTAVES: usize = 3;
        const LACUNARITY: f32 = 2.49;
        const GAIN: f32 = 0.255;
        const WS: f32 = 0.22;

        // `GenFractalRidged`, transcribed.
        let reference = |p: Vec3| {
            let mut bounding = 0.0;
            let mut a = 1.0;
            for _ in 0..OCTAVES {
                bounding += a;
                a *= GAIN;
            }
            let (mut sum, mut amp, mut freq) = (0.0f32, 1.0f32, 1.0f32);
            for _ in 0..OCTAVES {
                // FNL's polarity is `ratio − 1`; `grid` returns `1 − ratio`.
                let n = (-grid(p * freq).value).abs();
                sum += (n * -2.0 + 1.0) * amp;
                amp *= 1.0 + WS * ((1.0 - n) - 1.0); // lerp(1, 1 − n, ws)
                amp *= GAIN;
                freq *= LACUNARITY;
            }
            // `NoiseTexture2D`: [−1, 1] → [0, 1].
            (sum / bounding + 1.0) * 0.5
        };

        for i in 0..20_000 {
            let p = walk(i);
            let mut accum = FbmAccum::new();
            let (mut amp, mut freq) = (1.0f32, 1.0f32);
            for _ in 0..OCTAVES {
                accum.push(grid(p * freq).value, amp, 1.0, Fractal::RidgedFnl, WS);
                amp *= GAIN;
                freq *= LACUNARITY;
            }
            let got = accum.finish();
            let want = reference(p);
            assert!(
                (got - want).abs() < 1.0e-5,
                "{p:?}: {got} vs FastNoiseLite's {want}"
            );
            assert!((0.0..=1.0).contains(&got), "{got} is outside [0, 1]");
        }
    }

    /// The two ridges are genuinely different folds, so a material cannot be
    /// moved between them without being re-tuned — the claim
    /// [`Fractal::RidgedFnl`]'s docs make, and the whole reason it is a second
    /// variant rather than a correction to the first.
    #[test]
    fn the_two_ridges_are_not_the_same_curve() {
        let mut worst = 0.0f32;
        for i in 0..2_000 {
            let v = grid(walk(i)).value;
            let mut a = FbmAccum::new();
            let mut b = FbmAccum::new();
            a.push(v, 1.0, 1.0, Fractal::Ridged, 0.22);
            b.push(v, 1.0, 1.0, Fractal::RidgedFnl, 0.22);
            worst = worst.max((a.finish() - b.finish()).abs());
        }
        assert!(
            worst > 0.2,
            "the two ridges agree to {worst}; one of them has stopped being itself"
        );
    }

    /// Both ridges feed the octave above them, and `Fbm` does not — the
    /// property `shader.wgsl` skips octaves on.
    #[test]
    fn only_the_ridges_feed_the_next_octave() {
        assert!(!Fractal::Fbm.feeds_forward());
        assert!(Fractal::Ridged.feeds_forward());
        assert!(Fractal::RidgedFnl.feeds_forward());
    }

    /// [`radial_grid`] is continuous across `θ = 0`, which is the seam an
    /// integer sector count exists to close.
    ///
    /// The `+X` half-plane is where `atan2` wraps, and it is where a fractional
    /// count would tear. Sampled either side of it at a spread of radii and
    /// heights — the value must not jump, though `f1` may, because `θ = 0` lands
    /// exactly on a cell boundary and the nearest centre genuinely differs
    /// across one.
    #[test]
    fn the_radial_grid_has_no_seam_at_the_wrap() {
        const EPS: f32 = 1.0e-4;
        for sectors in [1.0f32, 2.0, 3.0, 4.0, 7.0, 16.0] {
            for i in 0..200 {
                let r = 0.05 + i as f32 * 0.037;
                let y = -3.0 + i as f32 * 0.031;
                // Just below and just above the wrap in `θ`.
                let below = Vec3::new(r * (-EPS).cos(), y, r * (-EPS).sin());
                let above = Vec3::new(r * EPS.cos(), y, r * EPS.sin());
                let a = radial_grid(below, sectors).value;
                let b = radial_grid(above, sectors).value;
                assert!(
                    (a - b).abs() < 1.0e-3,
                    "sectors {sectors}: {a} vs {b} across θ = 0 at r {r}, y {y}"
                );
            }
        }
    }

    /// A fractional sector count is *rounded*, not honoured — which is what
    /// keeps the wrap closed when an octave scales the count off an integer.
    #[test]
    fn the_radial_grid_rounds_its_sectors() {
        for i in 0..500 {
            let p = walk(i);
            for (fractional, whole) in [(3.7f32, 4.0f32), (4.2, 4.0), (6.5, 7.0), (0.1, 1.0)] {
                assert_eq!(
                    radial_grid(p, fractional).value,
                    radial_grid(p, whole).value,
                    "{p:?}: {fractional} sectors did not round to {whole}"
                );
            }
        }
    }

    /// The wedges really are wedges: the field is **constant** along a ray from
    /// the axis, so two points at the same turn and height agree whatever their
    /// radius.
    ///
    /// A stronger claim than it first was, and the strengthening is the point:
    /// while a ring coordinate was in play only the *sector index* could be
    /// compared, because the value was allowed to differ along it. With the
    /// third axis pinned there is nothing left to differ, which is what makes
    /// this a two-axis field and therefore a UV wrap.
    #[test]
    fn the_radial_grid_cuts_wedges_around_the_axis() {
        let sectors = 4.0;
        let sector_of = |turn: f32, r: f32| {
            let angle = turn * std::f32::consts::TAU;
            let p = Vec3::new(r * angle.cos(), 0.25, r * angle.sin());
            // `f1.xz` is the wedge's centre direction; its turn is the index.
            let f1 = radial_grid(p, sectors).f1;
            (f1.z.atan2(f1.x) / std::f32::consts::TAU + 0.5 + 1.0) % 1.0
        };
        for step in 0..40 {
            let turn = step as f32 / 40.0;
            let near = sector_of(turn, 0.4);
            let far = sector_of(turn, 3.9);
            assert!(
                (near - far).abs() < 1.0e-3,
                "turn {turn}: wedge {near} near the axis, {far} far from it"
            );
            // …and the value itself, not merely which wedge it fell in.
            let angle = turn * std::f32::consts::TAU;
            let at = |r: f32| {
                radial_grid(Vec3::new(r * angle.cos(), 0.25, r * angle.sin()), sectors).value
            };
            assert!(
                (at(0.4) - at(3.9)).abs() < 1.0e-5,
                "turn {turn}: {} at r 0.4 against {} at r 3.9",
                at(0.4),
                at(3.9)
            );
        }
    }

    /// It is a *different field* from the Cartesian [`grid`], which is the whole
    /// reason it is its own kind rather than a coordinate the caller could have
    /// warped itself.
    #[test]
    fn the_radial_grid_is_not_the_cartesian_one() {
        let mut differ = 0;
        for i in 0..2_000 {
            let p = walk(i);
            if (grid(p).value - radial_grid(p, 4.0).value).abs() > 0.05 {
                differ += 1;
            }
        }
        assert!(differ > 1_500, "only {differ}/2000 samples differ");
    }

    /// The field is exactly 1-periodic, which is the whole of why a grid tile
    /// is seamless with no `period` argument and no wrap.
    ///
    /// The tolerance is `1e-5` rather than exact, and the slack is the *test's*
    /// and not the field's: `p + 3.0` is a float addition, and it rounds the
    /// fractional part by an amount that grows with `|p|` — which the ratio
    /// then carries through to the value. Shifting by an integer is the only
    /// way to state the claim, and it cannot be done exactly at any magnitude
    /// where the exponent changes. So this walk is folded into `[−4, 4)`
    /// (the shift takes it to 15) where the worst residual measures 3.2e-6,
    /// rather than [`walk`]'s open-ended one, which reaches `|p| ≈ 265` and a
    /// residual an order of magnitude larger.
    #[test]
    fn the_grid_repeats_every_cell() {
        for i in 0..2_000 {
            let w = walk(i);
            let p = Vec3::new(
                w.x.rem_euclid(8.0) - 4.0,
                w.y.rem_euclid(8.0) - 4.0,
                w.z.rem_euclid(8.0) - 4.0,
            );
            let shifted = p + Vec3::new(3.0, -7.0, 12.0);
            assert!(
                (grid(p).value - grid(shifted).value).abs() < 1.0e-5,
                "{p:?} does not repeat"
            );
        }
    }

    /// The two ends of the range, named: a cell centre is 1 and a face centre
    /// is 0. Everything the ramp is authored against hangs off this.
    #[test]
    fn the_grid_runs_from_one_at_the_centre_to_zero_on_the_boundary() {
        assert!((grid(Vec3::splat(0.5)).value - 1.0).abs() < 1.0e-6);
        // The middle of a cell face: d1 and d2 are equal there.
        assert!(grid(Vec3::new(1.0, 0.5, 0.5)).value.abs() < 1.0e-6);
        for i in 0..2_000 {
            let v = grid(walk(i)).value;
            assert!((0.0..=1.0).contains(&v), "{v} is outside [0, 1]");
        }
    }
}
