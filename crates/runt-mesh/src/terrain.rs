//! Heightfield terrain: a pure analytic field first, a mesh second
//! (DESIGN §6, §9).
//!
//! The important object here is [`HeightField`], not the mesh. Per DESIGN §9
//! terrain collision samples `h(x, z)` **directly** — nothing ever collides with
//! triangles — so the mesh is only a *view* of the field at some tessellation.
//! That is what makes visual LOD unable to change physics: `h` does not take a
//! quality, a segment count, or a vertex index.
//!
//! Two consequences the tests pin down:
//!
//! - Vertex normals come from the field's **gradient**, not from averaging face
//!   normals. A face-averaged normal is a normal of the *approximation*; the
//!   gradient is the normal of the surface the ball will actually roll on, so
//!   shading and physics agree at every quality tier.
//! - `h` is identical for a given `(x, z)` whichever quality generated the mesh.
//!
//! ## Noise
//!
//! Seeded **value noise** on an integer lattice, summed as fBm. The lattice hash
//! is integer-only (splitmix64) — no `sin`-based hashing, per DESIGN §7's
//! precision doctrine, because cheap mobile GPUs run "highp" as fp24 internally
//! and large-argument trig hashes fall apart there. The same function has to be
//! portable to WGSL later, so it stays integer arithmetic all the way down.
//!
//! Interpolation is the classic cubic smoothstep `t²(3−2t)`. Its derivative is
//! `6t(1−t)`, and the bilinear patch between four lattice values is a closed
//! form, so the gradient is **analytic** — not a finite difference. There is no
//! epsilon to tune and no disagreement between the height a mesh vertex uses and
//! the slope physics reads.

use glam::{Vec2, Vec3};

use super::{MeshData, Quality};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Upper bound on fBm octaves. Past this the added frequencies are far below a
/// vertex spacing anyone will ever tessellate to, and the clamp keeps a bad
/// param from turning generation into a hang.
pub const MAX_OCTAVES: u32 = 12;

/// splitmix64 — a full-avalanche integer mixer. Deterministic on every platform
/// (pure `u64` wrapping arithmetic), which is the whole requirement.
#[inline]
pub fn hash_u64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Pseudo-random value in `[-1, 1)` at integer lattice point `(ix, iz)`.
///
/// The two coordinates are mixed with different odd constants before hashing so
/// that `(a, b)` and `(b, a)` do not collide — a symmetric combine shows up as a
/// visible diagonal ridge in the finished terrain.
#[inline]
fn lattice(ix: i32, iz: i32, seed: u64) -> f32 {
    let x = (ix as i64 as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
    let z = (iz as i64 as u64).wrapping_mul(0xA076_1D64_78BD_642F);
    let h = hash_u64(seed ^ x ^ z.rotate_left(32));
    // Top 24 bits → [0, 2) → [-1, 1). 24 bits is exactly f32's mantissa, so the
    // conversion is exact and the same integer always gives the same float.
    ((h >> 40) as f32) * (1.0 / 8_388_608.0) - 1.0
}

/// One octave of value noise at `(x, z)`: `(value, ∂/∂x, ∂/∂z)`.
///
/// Bilinear blend of four lattice values with smoothstep weights. Writing the
/// blend as `a + b·u + c·v + d·u·v` makes both partials fall out directly, which
/// is why the gradient below needs no finite differencing.
#[inline]
fn value_noise_d(x: f32, z: f32, seed: u64) -> (f32, f32, f32) {
    let xf = x.floor();
    let zf = z.floor();
    let ix = xf as i32;
    let iz = zf as i32;
    let tx = x - xf;
    let tz = z - zf;

    // smoothstep and its derivative
    let ux = tx * tx * (3.0 - 2.0 * tx);
    let uz = tz * tz * (3.0 - 2.0 * tz);
    let dux = 6.0 * tx * (1.0 - tx);
    let duz = 6.0 * tz * (1.0 - tz);

    let n00 = lattice(ix, iz, seed);
    let n10 = lattice(ix + 1, iz, seed);
    let n01 = lattice(ix, iz + 1, seed);
    let n11 = lattice(ix + 1, iz + 1, seed);

    let a = n00;
    let b = n10 - n00;
    let c = n01 - n00;
    let d = n00 - n10 - n01 + n11;

    let value = a + b * ux + c * uz + d * ux * uz;
    let ddx = (b + d * uz) * dux;
    let ddz = (c + d * ux) * duz;
    (value, ddx, ddz)
}

/// A pure, deterministic terrain height field: `h(x, z)` plus its gradient.
///
/// **This is the physics surface** (DESIGN §9). Step 5's ball integrator samples
/// exactly this, in the same world coordinates the mesh was built in — see
/// [`TerrainParams::field`] and `runt_core::ecs::TerrainSurface`.
///
/// `Copy` and 40 bytes: cheap to hand to a system, cheap to store per entity, and
/// with no interior state there is nothing that could make two samples of the
/// same point disagree.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HeightField {
    /// Master seed. Each octave derives its own lattice seed from this.
    pub seed: u64,
    /// Peak-to-mean height, in world units. `h` is normalized by the octave
    /// weights first, so this stays meaningful when `octaves` changes.
    pub amplitude: f32,
    pub octaves: u32,
    /// Lattice cells per world unit at octave 0. Small values → broad hills.
    pub frequency: f32,
    /// Frequency multiplier per octave.
    pub lacunarity: f32,
    /// Amplitude multiplier per octave.
    pub gain: f32,
}

impl Default for HeightField {
    fn default() -> HeightField {
        HeightField {
            seed: 0,
            amplitude: 1.0,
            octaves: 4,
            frequency: 0.08,
            lacunarity: 2.0,
            gain: 0.5,
        }
    }
}

impl HeightField {
    /// Height and gradient in one pass — the form every other accessor is built
    /// from, and the one physics should call (it wants both anyway).
    ///
    /// Returns `(h, (∂h/∂x, ∂h/∂z))`.
    pub fn sample(&self, x: f32, z: f32) -> (f32, Vec2) {
        let octaves = self.octaves.clamp(1, MAX_OCTAVES);
        let mut freq = self.frequency;
        let mut amp = 1.0f32;
        let mut height = 0.0f32;
        let mut grad = Vec2::ZERO;
        let mut weight = 0.0f32;

        for octave in 0..octaves {
            // A derived seed per octave rather than a coordinate offset: an
            // offset leaves octaves correlated near the origin, a fresh lattice
            // does not.
            let seed = hash_u64(self.seed ^ (octave as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let (v, ddx, ddz) = value_noise_d(x * freq, z * freq, seed);
            height += amp * v;
            // Chain rule: the noise was evaluated in lattice space.
            grad += Vec2::new(amp * ddx * freq, amp * ddz * freq);
            weight += amp;
            freq *= self.lacunarity;
            amp *= self.gain;
        }

        if weight <= 0.0 {
            return (0.0, Vec2::ZERO);
        }
        let k = self.amplitude / weight;
        (height * k, grad * k)
    }

    /// Surface height at `(x, z)`. Independent of tessellation, quality tier and
    /// mesh — DESIGN §9's whole point.
    pub fn height(&self, x: f32, z: f32) -> f32 {
        self.sample(x, z).0
    }

    /// `(∂h/∂x, ∂h/∂z)`, analytic. Slope response in the ball integrator reads
    /// this; so does the mesh, for its vertex normals.
    pub fn gradient(&self, x: f32, z: f32) -> Vec2 {
        self.sample(x, z).1
    }

    /// Unit surface normal at `(x, z)`.
    ///
    /// For `y = h(x, z)` the surface is `F = y − h = 0`, so `∇F = (−h_x, 1, −h_z)`.
    pub fn normal(&self, x: f32, z: f32) -> Vec3 {
        normal_from_gradient(self.gradient(x, z))
    }
}

/// The unit normal implied by a height-field gradient. Shared so the mesh and
/// any physics contact code cannot drift apart on the sign convention.
#[inline]
pub fn normal_from_gradient(grad: Vec2) -> Vec3 {
    Vec3::new(-grad.x, 1.0, -grad.y).normalize()
}

/// Generator params for [`terrain`] — the serialized, hashed, scene-file form
/// (DESIGN §6). The field-relevant subset is extracted by [`TerrainParams::field`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TerrainParams {
    pub seed: u64,
    /// World extent on X and Z. The patch is centered on the generator's origin.
    pub size: Vec2,
    pub amplitude: f32,
    pub octaves: u32,
    pub frequency: f32,
    pub lacunarity: f32,
    pub gain: f32,
    /// Grid quads per side at `Quality::FULL`; scaled by the quality tier.
    pub base_segments: u32,
    /// Flat vertex color for the whole patch. `None` leaves it white.
    #[cfg_attr(feature = "serde", serde(default))]
    pub color: Option<Vec3>,
}

impl Default for TerrainParams {
    fn default() -> TerrainParams {
        TerrainParams {
            seed: 0,
            size: Vec2::splat(32.0),
            amplitude: 1.0,
            octaves: 4,
            frequency: 0.08,
            lacunarity: 2.0,
            gain: 0.5,
            base_segments: 64,
            color: None,
        }
    }
}

impl TerrainParams {
    /// The analytic surface these params describe. Cheap and pure — call it
    /// wherever you need `h`, rather than caching a copy that could go stale.
    pub fn field(&self) -> HeightField {
        HeightField {
            seed: self.seed,
            amplitude: self.amplitude,
            octaves: self.octaves,
            frequency: self.frequency,
            lacunarity: self.lacunarity,
            gain: self.gain,
        }
    }

    /// Segment count this patch meshes at under `quality`.
    pub fn segments(&self, quality: Quality) -> u32 {
        quality.segs(self.base_segments.max(1), 1)
    }
}

/// Mesh a terrain patch: a grid over `params.size`, centered on the origin,
/// displaced by `h` and shaded by the field gradient.
///
/// Winding and UV convention match [`plane`](super::plane) exactly, so swapping a
/// flat ground plane for terrain changes nothing downstream. Segment count is the
/// only thing `quality` touches — the surface itself is quality-invariant.
pub fn terrain(params: &TerrainParams, quality: Quality) -> MeshData {
    let n = params.segments(quality);
    let field = params.field();
    let half = params.size * 0.5;
    let color = params.color.unwrap_or(Vec3::ONE);

    let verts = ((n + 1) * (n + 1)) as usize;
    let mut m = MeshData {
        positions: Vec::with_capacity(verts),
        normals: Vec::with_capacity(verts),
        uvs: Vec::with_capacity(verts),
        colors: Vec::with_capacity(verts),
        indices: Vec::with_capacity((n * n * 6) as usize),
    };

    for j in 0..=n {
        for i in 0..=n {
            let fx = i as f32 / n as f32;
            let fz = j as f32 / n as f32;
            let x = -half.x + fx * params.size.x;
            let z = -half.y + fz * params.size.y;
            let (h, grad) = field.sample(x, z);
            m.positions.push(Vec3::new(x, h, z));
            m.normals.push(normal_from_gradient(grad));
            m.uvs.push(Vec2::new(fx, fz));
            m.colors.push(color);
        }
    }

    let stride = n + 1;
    for j in 0..n {
        for i in 0..n {
            let a = j * stride + i;
            let b = j * stride + i + 1;
            let c = (j + 1) * stride + i + 1;
            let d = (j + 1) * stride + i;
            // CCW seen from +Y (above) — same as `plane`.
            m.indices.extend_from_slice(&[a, d, c, a, c, b]);
        }
    }
    m
}
