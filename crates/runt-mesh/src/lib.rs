//! GPU-agnostic mesh generation.
//!
//! Generation is pure: primitives and ops are `fn(params) -> MeshData` with no
//! GPU or global state, so meshes can be built and unit-tested headless, cached
//! by param hash, and later moved to a worker thread without a rewrite.
//!
//! Layout is struct-of-arrays: ops touch one attribute at a time (transform
//! positions, recompute normals, remap UVs), and the renderer interleaves into
//! its vertex format only at upload.

use glam::{Mat3, Mat4, Quat, Vec2, Vec3};

pub mod ops;
pub mod primitives;

pub use primitives::*;

#[cfg(test)]
mod tests;

/// A triangle mesh in struct-of-arrays form. `positions` and `indices` are
/// authoritative; `normals`, `uvs`, and `colors` are either empty or exactly
/// `positions.len()` long (primitives always fill all four).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeshData {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub colors: Vec<Vec3>,
    pub indices: Vec<u32>,
}

impl MeshData {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Axis-aligned bounds, or `None` if the mesh has no vertices.
    pub fn bounds(&self) -> Option<(Vec3, Vec3)> {
        let mut it = self.positions.iter();
        let first = *it.next()?;
        let (mut min, mut max) = (first, first);
        for &p in it {
            min = min.min(p);
            max = max.max(p);
        }
        Some((min, max))
    }

    /// Iterate triangles as position triples.
    pub fn triangles(&self) -> impl Iterator<Item = [Vec3; 3]> + '_ {
        self.indices.chunks_exact(3).map(move |t| {
            [
                self.positions[t[0] as usize],
                self.positions[t[1] as usize],
                self.positions[t[2] as usize],
            ]
        })
    }

    /// Debug-only invariant check: attribute lengths and index bounds.
    pub fn validate(&self) {
        debug_assert!(self.indices.len() % 3 == 0, "index count not a multiple of 3");
        let n = self.positions.len();
        for (name, len) in [
            ("normals", self.normals.len()),
            ("uvs", self.uvs.len()),
            ("colors", self.colors.len()),
        ] {
            debug_assert!(len == 0 || len == n, "{name} length {len} != vertex count {n}");
        }
        for &i in &self.indices {
            debug_assert!((i as usize) < n, "index {i} out of range for {n} vertices");
        }
    }

    /// Content hash over quantized attributes — a stable key for deduping or
    /// caching identical generated meshes. Not a param-level cache key (that
    /// lives with the generator inputs), just a cheap dedup of the output.
    pub fn content_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        let q = |f: f32| (f * 4096.0).round() as i32; // ~0.00024 grid
        for p in &self.positions {
            q(p.x).hash(&mut h);
            q(p.y).hash(&mut h);
            q(p.z).hash(&mut h);
        }
        for n in &self.normals {
            q(n.x).hash(&mut h);
            q(n.y).hash(&mut h);
            q(n.z).hash(&mut h);
        }
        for uv in &self.uvs {
            q(uv.x).hash(&mut h);
            q(uv.y).hash(&mut h);
        }
        for c in &self.colors {
            q(c.x).hash(&mut h);
            q(c.y).hash(&mut h);
            q(c.z).hash(&mut h);
        }
        self.indices.hash(&mut h);
        h.finish()
    }

    // --- fluent sugar over the free ops in `ops` ---------------------------

    pub fn transform(self, m: Mat4) -> Self {
        ops::transform(self, m)
    }

    pub fn translate(self, offset: Vec3) -> Self {
        ops::transform(self, Mat4::from_translation(offset))
    }

    pub fn scale(self, factor: Vec3) -> Self {
        ops::transform(self, Mat4::from_scale(factor))
    }

    pub fn rotate(self, q: Quat) -> Self {
        ops::transform(self, Mat4::from_quat(q))
    }

    pub fn merge(self, other: MeshData) -> Self {
        ops::merge(self, other)
    }

    pub fn with_color(self, color: Vec3) -> Self {
        ops::set_color(self, color)
    }

    /// Faceted shading: one normal per face, no shared vertices.
    pub fn flat_normals(self) -> Self {
        ops::flat_normals(self)
    }

    /// Crease-angle shading: smooth across edges whose adjacent faces differ by
    /// less than `crease_degrees`, hard past it. `0` ≈ flat, `180` ≈ fully smooth.
    pub fn smooth_normals(self, crease_degrees: f32) -> Self {
        ops::creased_normals(self, crease_degrees)
    }

    /// Twist positions about `axis` (through the origin), `radians_per_unit`
    /// along that axis. Does not update normals — recompute after deforming.
    pub fn twist(self, radians_per_unit: f32, axis: Vec3) -> Self {
        ops::twist(self, radians_per_unit, axis)
    }

    /// Linearly scale cross-sections along `axis`: `factor` at the max extent,
    /// `1.0` at the min. Does not update normals — recompute after deforming.
    pub fn taper(self, factor: f32, axis: Vec3) -> Self {
        ops::taper(self, factor, axis)
    }
}

/// A resolution multiplier for device/LOD tiers. A generator writes explicit
/// base segment counts; the tier system scales them. A different quality is a
/// legitimately different mesh (and cache key), so determinism is preserved.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quality(pub f32);

impl Quality {
    pub const FULL: Quality = Quality(1.0);

    /// Scale a base segment count, clamped to at least `min`.
    pub fn segs(self, base: u32, min: u32) -> u32 {
        ((base as f32 * self.0).round() as u32).max(min)
    }
}

impl Default for Quality {
    fn default() -> Self {
        Quality::FULL
    }
}

/// Squared length of a triangle's raw cross product below which we treat it as
/// degenerate. A real triangle on a unit-scale mesh has a cross ~1e-2+ long
/// (squared ~1e-4); floating-point pole slivers (`sin(PI) != 0`) land near
/// 1e-17, so this cleanly separates them.
pub(crate) const DEGENERATE_AREA_SQ: f32 = 1.0e-12;

/// Shared helper: the outward-pointing geometric normal of a triangle, scaled
/// by twice its area (i.e. the raw cross product). Zero for degenerate tris.
pub(crate) fn face_cross(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    (b - a).cross(c - a)
}

pub(crate) fn normal_matrix(m: Mat4) -> Mat3 {
    Mat3::from_mat4(m).inverse().transpose()
}
