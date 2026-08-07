//! The generator registry (DESIGN §6).
//!
//! §6 asks for "a stable generator name + params hash + quality → content hash →
//! mesh". [`GeneratorSpec`] *is* that registry in v1: one serde-tagged enum whose
//! variants name the generators and carry their params. A `HashMap<String, Box<dyn
//! Fn>>` would buy dynamic registration we have no caller for, and would cost the
//! two things that actually matter here — a hand-editable scene file and an
//! exhaustive `match` the compiler checks.
//!
//! Three properties everything downstream leans on:
//!
//! - **Pure.** [`GeneratorSpec::generate`] is `fn(&self, Quality) -> MeshData`
//!   with no world, no GPU and no globals, so it can move to a worker later
//!   without a rewrite (§6: generation is never in the frame).
//! - **Placement-free.** A spec describes *shape*, never where a thing sits. Two
//!   entities differing only in transform must generate the same `MeshData`, or
//!   content-addressed dedup never fires.
//! - **Stably hashed.** [`GeneratorSpec::param_key`] is the layer-A cache key;
//!   see its docs for why it goes through `postcard` rather than `derive(Hash)`.
//!
//! ## Serde field names are the interface
//!
//! Variant and field names here are what a hand-written `assets/*.ron` says.
//! Renaming one is a breaking scene-file change; adding a `#[serde(default)]`
//! field is not.
//!
//! ## Reflection (§6's `Params: Reflect + Serialize + Hash`)
//!
//! `Reflect` arrives with the editor, behind the **`reflect` feature**, which is
//! off by default: the editor is its only consumer and `bevy_reflect` is dead
//! weight in the wasm player's bundle. `Serialize` plus the serialized-bytes
//! hash below covers everything determinism and caching need with the feature
//! off, so nothing in the engine may depend on reflection being available.
//!
//! Because glam is a version behind in `bevy_reflect`, the vector params carry
//! `#[reflect(remote = …)]` pointers into [`crate::reflect`]; see that module
//! for why. The `#[reflect(@FieldRange…)]` attributes are editor slider bounds.

use glam::{Vec2, Vec3};
use runt_mesh::{cone, cube, cylinder, plane, terrain, torus, uv_sphere, MeshData, Quality};
use serde::{Deserialize, Serialize};

pub use runt_mesh::{HeightField, TerrainParams};

/// How a generator's normals are computed once its base geometry exists.
///
/// Separate from the shape params because it is orthogonal to them: the same
/// sphere is a different *mesh* faceted vs. smoothed, but it is the same shape
/// request with a different finish.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect))]
pub enum Shading {
    /// Keep whatever normals the primitive generated (smooth for the curved
    /// ones, per-face for the box).
    #[default]
    Generated,
    /// Faceted: one normal per triangle, no shared vertices.
    Flat,
    /// Crease-angle smoothing, threshold in degrees. `180` is fully smooth.
    Smooth(f32),
}

/// Every generator runt knows how to run, with its params.
///
/// Variants map onto `runt-mesh` primitives and ops. New generators are appended
/// at the end — see [`param_key`](GeneratorSpec::param_key) for why order is
/// (mildly) load-bearing.
/// ## Editor ranges
///
/// The `#[reflect(@FieldRange…)]` attributes below are what the editor's
/// reflection-driven panels use for slider bounds (see
/// [`crate::reflect`]). They are **advisory** — every generator still has to
/// behave outside them, because a scene file can say anything — but they are
/// declared here rather than in a table in the editor so that adding a param
/// cannot leave its bound behind in another crate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect))]
pub enum GeneratorSpec {
    /// Flat XZ grid, normal +Y.
    Plane {
        #[cfg_attr(
            feature = "reflect",
            reflect(remote = crate::reflect::Vec2Def, @crate::reflect::FieldRange::new(0.1, 128.0))
        )]
        size: Vec2,
        /// Quads per side at `Quality::FULL`.
        #[cfg_attr(feature = "reflect", reflect(@crate::reflect::FieldRange::new(1.0, 256.0)))]
        subdivisions: u32,
        #[serde(default)]
        shading: Shading,
        #[serde(default)]
        #[cfg_attr(
            feature = "reflect",
            reflect(remote = crate::reflect::OptVec3Def, @crate::reflect::FieldRange::new(0.0, 1.0))
        )]
        color: Option<Vec3>,
    },
    /// Axis-aligned box of `size` on every edge.
    Cube {
        #[cfg_attr(feature = "reflect", reflect(@crate::reflect::FieldRange::new(0.01, 20.0)))]
        size: f32,
        #[serde(default)]
        shading: Shading,
        #[serde(default)]
        #[cfg_attr(
            feature = "reflect",
            reflect(remote = crate::reflect::OptVec3Def, @crate::reflect::FieldRange::new(0.0, 1.0))
        )]
        color: Option<Vec3>,
    },
    UvSphere {
        #[cfg_attr(feature = "reflect", reflect(@crate::reflect::FieldRange::new(0.01, 20.0)))]
        radius: f32,
        #[cfg_attr(feature = "reflect", reflect(@crate::reflect::FieldRange::new(2.0, 128.0)))]
        rings: u32,
        #[cfg_attr(feature = "reflect", reflect(@crate::reflect::FieldRange::new(3.0, 256.0)))]
        sectors: u32,
        #[serde(default)]
        shading: Shading,
        #[serde(default)]
        #[cfg_attr(
            feature = "reflect",
            reflect(remote = crate::reflect::OptVec3Def, @crate::reflect::FieldRange::new(0.0, 1.0))
        )]
        color: Option<Vec3>,
    },
    /// Y-axis cylinder, centered, flat caps.
    Cylinder {
        #[cfg_attr(feature = "reflect", reflect(@crate::reflect::FieldRange::new(0.01, 20.0)))]
        radius: f32,
        #[cfg_attr(feature = "reflect", reflect(@crate::reflect::FieldRange::new(0.01, 50.0)))]
        height: f32,
        #[cfg_attr(feature = "reflect", reflect(@crate::reflect::FieldRange::new(3.0, 256.0)))]
        segments: u32,
        #[serde(default)]
        shading: Shading,
        #[serde(default)]
        #[cfg_attr(
            feature = "reflect",
            reflect(remote = crate::reflect::OptVec3Def, @crate::reflect::FieldRange::new(0.0, 1.0))
        )]
        color: Option<Vec3>,
    },
    /// Y-axis cone, base at `-height/2`, apex at `+height/2`.
    Cone {
        #[cfg_attr(feature = "reflect", reflect(@crate::reflect::FieldRange::new(0.01, 20.0)))]
        radius: f32,
        #[cfg_attr(feature = "reflect", reflect(@crate::reflect::FieldRange::new(0.01, 50.0)))]
        height: f32,
        #[cfg_attr(feature = "reflect", reflect(@crate::reflect::FieldRange::new(3.0, 256.0)))]
        segments: u32,
        #[serde(default)]
        shading: Shading,
        #[serde(default)]
        #[cfg_attr(
            feature = "reflect",
            reflect(remote = crate::reflect::OptVec3Def, @crate::reflect::FieldRange::new(0.0, 1.0))
        )]
        color: Option<Vec3>,
    },
    /// Torus in the XZ plane.
    Torus {
        #[cfg_attr(feature = "reflect", reflect(@crate::reflect::FieldRange::new(0.01, 20.0)))]
        major_radius: f32,
        #[cfg_attr(feature = "reflect", reflect(@crate::reflect::FieldRange::new(0.001, 10.0)))]
        minor_radius: f32,
        #[cfg_attr(feature = "reflect", reflect(@crate::reflect::FieldRange::new(3.0, 256.0)))]
        major_segments: u32,
        #[cfg_attr(feature = "reflect", reflect(@crate::reflect::FieldRange::new(3.0, 128.0)))]
        minor_segments: u32,
        #[serde(default)]
        shading: Shading,
        #[serde(default)]
        #[cfg_attr(
            feature = "reflect",
            reflect(remote = crate::reflect::OptVec3Def, @crate::reflect::FieldRange::new(0.0, 1.0))
        )]
        color: Option<Vec3>,
    },
    /// A box stretched to `dims`, twisted about Y, then tapered along Y.
    ///
    /// The deformation is *shape*, not placement, so it belongs in the generator
    /// and legitimately changes the content hash — unlike the `scale` a
    /// placement would use.
    TwistedBox {
        #[cfg_attr(
            feature = "reflect",
            reflect(remote = crate::reflect::Vec3Def, @crate::reflect::FieldRange::new(0.01, 20.0))
        )]
        dims: Vec3,
        /// Radians of twist per world unit along Y.
        // `TAU` by name rather than as 6.2832: clippy's `approx_constant` is
        // deny-by-default and fires on the literal, which made
        // `cargo clippy --features reflect` fail outright.
        #[cfg_attr(feature = "reflect", reflect(@crate::reflect::FieldRange::new(-std::f32::consts::TAU, std::f32::consts::TAU)))]
        twist: f32,
        /// Cross-section scale at the top; `1.0` is no taper.
        #[cfg_attr(feature = "reflect", reflect(@crate::reflect::FieldRange::new(0.0, 4.0)))]
        taper: f32,
        #[serde(default)]
        shading: Shading,
        #[serde(default)]
        #[cfg_attr(
            feature = "reflect",
            reflect(remote = crate::reflect::OptVec3Def, @crate::reflect::FieldRange::new(0.0, 1.0))
        )]
        color: Option<Vec3>,
    },
    /// Heightfield terrain — a *view* of the analytic surface physics samples
    /// (DESIGN §9). See [`TerrainParams`].
    Terrain(
        #[cfg_attr(feature = "reflect", reflect(remote = crate::reflect::TerrainParamsDef))]
        TerrainParams,
    ),
}

impl GeneratorSpec {
    /// Every generator's [`kind`](GeneratorSpec::kind), in declaration order.
    ///
    /// The editor's variant dropdown is built from this. Kept next to the enum
    /// (rather than derived by reflection) so it exists with the `reflect`
    /// feature off and so the compiler's exhaustiveness check on
    /// [`default_of_kind`](GeneratorSpec::default_of_kind) is the thing that
    /// keeps it honest.
    pub const KINDS: &'static [&'static str] = &[
        "Plane",
        "Cube",
        "UvSphere",
        "Cylinder",
        "Cone",
        "Torus",
        "TwistedBox",
        "Terrain",
    ];

    /// A sensible starting spec for a named variant — what "switch this
    /// generator to a Torus" should produce.
    ///
    /// Not `Default`: there is no meaningful default *generator*, only a default
    /// per shape. The numbers are the smallest set that renders as a recognizable
    /// version of the thing, so a switch never lands on a degenerate mesh.
    pub fn default_of_kind(kind: &str) -> Option<GeneratorSpec> {
        Some(match kind {
            "Plane" => GeneratorSpec::Plane {
                size: Vec2::splat(4.0),
                subdivisions: 8,
                shading: Shading::default(),
                color: None,
            },
            "Cube" => GeneratorSpec::Cube {
                size: 1.0,
                shading: Shading::default(),
                color: None,
            },
            "UvSphere" => GeneratorSpec::UvSphere {
                radius: 1.0,
                rings: 16,
                sectors: 24,
                shading: Shading::default(),
                color: None,
            },
            "Cylinder" => GeneratorSpec::Cylinder {
                radius: 0.5,
                height: 2.0,
                segments: 24,
                shading: Shading::default(),
                color: None,
            },
            "Cone" => GeneratorSpec::Cone {
                radius: 0.5,
                height: 1.5,
                segments: 20,
                shading: Shading::default(),
                color: None,
            },
            "Torus" => GeneratorSpec::Torus {
                major_radius: 1.0,
                minor_radius: 0.3,
                major_segments: 32,
                minor_segments: 16,
                shading: Shading::default(),
                color: None,
            },
            "TwistedBox" => GeneratorSpec::TwistedBox {
                dims: Vec3::new(1.0, 1.6, 1.0),
                twist: 0.9,
                taper: 0.6,
                shading: Shading::Flat,
                color: None,
            },
            "Terrain" => GeneratorSpec::Terrain(TerrainParams::default()),
            _ => return None,
        })
    }

    /// The variant's stable name. Used in logs and mixed into
    /// [`param_key`](GeneratorSpec::param_key); it is not the scene file's
    /// generator *entry* name (that one is chosen per scene).
    pub fn kind(&self) -> &'static str {
        match self {
            GeneratorSpec::Plane { .. } => "Plane",
            GeneratorSpec::Cube { .. } => "Cube",
            GeneratorSpec::UvSphere { .. } => "UvSphere",
            GeneratorSpec::Cylinder { .. } => "Cylinder",
            GeneratorSpec::Cone { .. } => "Cone",
            GeneratorSpec::Torus { .. } => "Torus",
            GeneratorSpec::TwistedBox { .. } => "TwistedBox",
            GeneratorSpec::Terrain(_) => "Terrain",
        }
    }

    /// Run the generator. Pure: same spec + same quality → same `MeshData`,
    /// forever, on any thread, with or without a cache.
    ///
    /// `quality` scales tessellation only. Every segment count has a floor, so a
    /// very low tier degrades to a coarse mesh rather than to a broken one
    /// (DESIGN §11: scale down, never fail).
    pub fn generate(&self, quality: Quality) -> MeshData {
        match *self {
            GeneratorSpec::Plane {
                size,
                subdivisions,
                shading,
                color,
            } => finish(plane(size, quality.segs(subdivisions, 1)), shading, color),

            GeneratorSpec::Cube {
                size,
                shading,
                color,
            } => finish(cube(size), shading, color),

            GeneratorSpec::UvSphere {
                radius,
                rings,
                sectors,
                shading,
                color,
            } => finish(
                uv_sphere(radius, quality.segs(rings, 2), quality.segs(sectors, 3)),
                shading,
                color,
            ),

            GeneratorSpec::Cylinder {
                radius,
                height,
                segments,
                shading,
                color,
            } => finish(
                cylinder(radius, height, quality.segs(segments, 3)),
                shading,
                color,
            ),

            GeneratorSpec::Cone {
                radius,
                height,
                segments,
                shading,
                color,
            } => finish(
                cone(radius, height, quality.segs(segments, 3)),
                shading,
                color,
            ),

            GeneratorSpec::Torus {
                major_radius,
                minor_radius,
                major_segments,
                minor_segments,
                shading,
                color,
            } => finish(
                torus(
                    major_radius,
                    minor_radius,
                    quality.segs(major_segments, 3),
                    quality.segs(minor_segments, 3),
                ),
                shading,
                color,
            ),

            GeneratorSpec::TwistedBox {
                dims,
                twist,
                taper,
                shading,
                color,
            } => finish(
                cube(1.0).scale(dims).twist(twist, Vec3::Y).taper(taper, Vec3::Y),
                shading,
                color,
            ),

            GeneratorSpec::Terrain(params) => terrain(&params, quality),
        }
    }

    /// The layer-A cache key: a stable hash of *(variant, params, quality)*.
    ///
    /// Not `derive(Hash)`. Params are full of `f32`s, and `f32` has no `Hash`
    /// precisely because bit-pattern hashing is a trap (`-0.0` ≠ `0.0`, NaN
    /// payloads); hashing raw bits by hand would additionally bake in struct
    /// padding and field order. Instead the spec is serialized to `postcard` —
    /// a canonical, little-endian, no-padding byte form — and those bytes are
    /// FNV-1a'd together with the variant name and the quality multiplier.
    ///
    /// FNV rather than `DefaultHasher`: `DefaultHasher` is SipHash with keys
    /// that std explicitly does not promise to keep, so a toolchain bump could
    /// silently invalidate every on-disk key. This one is a documented constant
    /// and will produce the same `u64` in ten years.
    ///
    /// **Not** a content hash. Two different param keys may well resolve to the
    /// same [`MeshData::content_hash`] (two qualities that round to the same
    /// segment count, say); that is fine and expected — layer A maps many keys
    /// onto one piece of geometry.
    ///
    /// Reordering the enum's variants changes `postcard`'s variant index and so
    /// changes existing keys. Nothing breaks — stale entries simply stop being
    /// found — but append rather than insert if you would like the cache to
    /// survive.
    pub fn param_key(&self, quality: Quality) -> u64 {
        let bytes = postcard::to_stdvec(self)
            .expect("GeneratorSpec serializes to postcard infallibly (no maps, no non-UTF8)");
        let mut h = FNV_OFFSET;
        h = fnv(h, self.kind().as_bytes());
        h = fnv(h, &bytes);
        h = fnv(h, &quality.0.to_bits().to_le_bytes());
        h
    }
}

/// Apply the shading and color finish shared by every primitive variant.
fn finish(mesh: MeshData, shading: Shading, color: Option<Vec3>) -> MeshData {
    let mesh = match shading {
        Shading::Generated => mesh,
        Shading::Flat => mesh.flat_normals(),
        Shading::Smooth(degrees) => mesh.smooth_normals(degrees),
    };
    match color {
        Some(c) => mesh.with_color(c),
        None => mesh,
    }
}

pub(crate) const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a over a byte slice, continuing from `h`.
///
/// `pub(crate)` because [`crate::texture::TextureSpec`] hashes itself the same
/// way: §6's scheme is the engine's one content-key scheme, not the mesh
/// pipeline's private habit.
pub(crate) fn fnv(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}
