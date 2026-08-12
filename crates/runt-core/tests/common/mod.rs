//! Test fixtures the texture suite owns.
//!
//! # Why these are here and not in `runt_core::texture`
//!
//! They were `texture::grass()` and `texture::rock()` — two `pub fn`s the engine
//! shipped, transcribed from a game's Godot materials, on the argument that a
//! scene file naming them made them reference data. That argument does not hold
//! up. The engine defines texture *types*: [`TextureSpec`] is the uniform block,
//! its `noise` field is the seam a new kind of noise lands on, and
//! `runt_core::bake` is what turns either into pixels. Which *values* a surface
//! uses is content — no more the engine's business than a brush's size is — so
//! the values live in the game that authors them now, and their whole derivation
//! record went with them.
//!
//! What the *tests* needed them for was never "is grass right". It was: a spec
//! whose octave chain quantizes cleanly, and a spec whose chain does not. So
//! that is what these are named for.
//!
//! # The numbers are unchanged, deliberately
//!
//! Every field below is what the two functions returned on the day they were
//! deleted. Nothing here claims to *be* any material, but the invariants this
//! suite pins — the seamless wrap, the live/baked agreement window, the
//! resolution ladder, the cache round trip — were all measured against these
//! exact chains, and re-basing them on prettier numbers at the same time as
//! moving them would have thrown away the regression history for nothing.
//!
//! [`TextureSpec`]: runt_core::texture::TextureSpec

#![allow(dead_code)]

use glam::Vec3;
use runt_core::noise::{CellReturn, Fractal, Lattice};
use runt_core::texture::{NoiseSpec, NormalMode, NormalSpec, TextureSpec};

/// A **cleanly quantized** tile: 6 cells across a 27.8 m tile, so the whole
/// lacunarity chain rounds almost for free (2.9% on the worst octave), at 36.9
/// texels per metre. To-point normals — each cell a rounded pebble.
///
/// The default fixture. Anything that just needs "a real spec with a normal map"
/// wants this one.
// `lacunarity: 2.718` is a number an artist typed, not an approximation of `e`;
// swapping in the constant would change every bake this suite measures.
#[allow(clippy::approx_constant)]
pub fn fine() -> TextureSpec {
    TextureSpec {
        noise: NoiseSpec::Cellular {
            lattice: Lattice::Fcc,
            return_type: CellReturn::CellValue,
            jitter: 1.0,
        },
        frequency: 0.21,
        octaves: 5,
        lacunarity: 2.718,
        gain: 0.562,
        ramp: vec![
            (0.12, Vec3::new(0.0, 0.31, 0.175_666_65)),
            (0.457_142_86, Vec3::new(0.0, 0.444_816_65, 0.310_580_22)),
            (1.0, Vec3::new(0.0, 0.534_645_26, 0.378_443_15)),
        ],
        normal: Some(NormalSpec {
            mode: NormalMode::ToPoint,
            edge_width: 0.52,
            strength: 5.106,
        }),
        world_scale: 0.036,
        triplanar_sharpness: 4.0,
        base_resolution: 1024,
        ..TextureSpec::default()
    }
}

/// A **coarsely quantized** tile: a 40 m tile holding *two* cells of a 21.7 m
/// base feature, so the base octave alone rounds 8.7% — the far end of what
/// [`TextureSpec::lacunarity_error`] is allowed to report. Hard to-edge normals
/// at a very high strength, and `triplanar_sharpness: 1.0`.
///
/// The awkward fixture, and the one that matters: a base octave holding four
/// cells or fewer has almost no live/baked agreement window
/// ([`TextureSpec::live_agreement_window`] is `2 / span` of the tile per side),
/// which is exactly the edge case worth a test.
///
/// [`TextureSpec::lacunarity_error`]: runt_core::texture::TextureSpec::lacunarity_error
/// [`TextureSpec::live_agreement_window`]: runt_core::texture::TextureSpec::live_agreement_window
pub fn coarse() -> TextureSpec {
    TextureSpec {
        noise: NoiseSpec::Cellular {
            lattice: Lattice::Fcc,
            return_type: CellReturn::CellValue,
            jitter: 1.0,
        },
        frequency: 0.046,
        octaves: 5,
        lacunarity: 3.512,
        gain: 0.543,
        ramp: vec![
            (0.12, Vec3::new(0.23, 0.1909, 0.207_191_66)),
            (0.457_142_86, Vec3::new(0.27, 0.2187, 0.240_075)),
            (1.0, Vec3::new(0.41, 0.3362, 0.366_949_98)),
        ],
        normal: Some(NormalSpec {
            mode: NormalMode::ToEdge,
            edge_width: 0.351,
            strength: 29.605,
        }),
        world_scale: 0.025,
        triplanar_sharpness: 1.0,
        base_resolution: 1024,
        ..TextureSpec::default()
    }
}

/// The two of them, labelled, for tests that sweep both.
pub fn both() -> [(&'static str, TextureSpec); 2] {
    [("fine", fine()), ("coarse", coarse())]
}

/// A ridged fixture — the one [`Fractal`] variant neither of the above uses, so
/// a test that wants the `weighted_strength` path has something to reach for.
pub fn ridged() -> TextureSpec {
    TextureSpec {
        fractal: Fractal::Ridged,
        weighted_strength: 0.5,
        ..fine()
    }
}
