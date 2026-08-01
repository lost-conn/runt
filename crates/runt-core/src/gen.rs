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
//! ## Known deviation from §6
//!
//! §6 specifies `Params: Reflect + Serialize + Hash`. `bevy_reflect` is
//! **deliberately deferred to phase 2** (the editor is the only consumer, and it
//! is dead weight in the wasm bundle until then). `Serialize` plus the
//! serialized-bytes hash below covers everything determinism and caching need
//! today; the editor's reflection-driven param panels are the only thing waiting
//! on it.

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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum GeneratorSpec {
    /// Flat XZ grid, normal +Y.
    Plane {
        size: Vec2,
        /// Quads per side at `Quality::FULL`.
        subdivisions: u32,
        #[serde(default)]
        shading: Shading,
        #[serde(default)]
        color: Option<Vec3>,
    },
    /// Axis-aligned box of `size` on every edge.
    Cube {
        size: f32,
        #[serde(default)]
        shading: Shading,
        #[serde(default)]
        color: Option<Vec3>,
    },
    UvSphere {
        radius: f32,
        rings: u32,
        sectors: u32,
        #[serde(default)]
        shading: Shading,
        #[serde(default)]
        color: Option<Vec3>,
    },
    /// Y-axis cylinder, centered, flat caps.
    Cylinder {
        radius: f32,
        height: f32,
        segments: u32,
        #[serde(default)]
        shading: Shading,
        #[serde(default)]
        color: Option<Vec3>,
    },
    /// Y-axis cone, base at `-height/2`, apex at `+height/2`.
    Cone {
        radius: f32,
        height: f32,
        segments: u32,
        #[serde(default)]
        shading: Shading,
        #[serde(default)]
        color: Option<Vec3>,
    },
    /// Torus in the XZ plane.
    Torus {
        major_radius: f32,
        minor_radius: f32,
        major_segments: u32,
        minor_segments: u32,
        #[serde(default)]
        shading: Shading,
        #[serde(default)]
        color: Option<Vec3>,
    },
    /// A box stretched to `dims`, twisted about Y, then tapered along Y.
    ///
    /// The deformation is *shape*, not placement, so it belongs in the generator
    /// and legitimately changes the content hash — unlike the `scale` a
    /// placement would use.
    TwistedBox {
        dims: Vec3,
        /// Radians of twist per world unit along Y.
        twist: f32,
        /// Cross-section scale at the top; `1.0` is no taper.
        taper: f32,
        #[serde(default)]
        shading: Shading,
        #[serde(default)]
        color: Option<Vec3>,
    },
    /// Heightfield terrain — a *view* of the analytic surface physics samples
    /// (DESIGN §9). See [`TerrainParams`].
    Terrain(TerrainParams),
}

impl GeneratorSpec {
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

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a over a byte slice, continuing from `h`.
fn fnv(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}
