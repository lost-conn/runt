//! Procedural textures: the spec, its identity, and the CPU evaluator
//! (DESIGN §7).
//!
//! > *A procedural texture is a WGSL fragment snippet with a params uniform,
//! > evaluated in UV space … **Baked** (baseline): rendered once to an RGBA8
//! > texture at tier-scaled resolution at load time … Bake outputs are
//! > content-addressed like meshes.* — DESIGN §7
//!
//! [`TextureSpec`] is that params block. It is to textures what
//! [`GeneratorSpec`](crate::gen::GeneratorSpec) is to meshes, down to the
//! hashing scheme: `postcard` bytes through FNV-1a, for the same reasons (§6's
//! `param_key` docs carry the full argument).
//!
//! ```text
//!   TextureSpec ──param_key()──► spec identity        (resolution-free)
//!        │
//!        └──content_key(res)───► bake identity        (what the cache files)
//!                    │
//!                    ▼
//!            bake.rs: fragment pass ──► RGBA8 albedo + RGBA8 normal
//! ```
//!
//! The split matters. A texture's *content* is scale-free in a way a mesh's is
//! not: baking grass at 512² and at 2048² produces the same field sampled more
//! finely, not two different materials. So `param_key` deliberately excludes
//! resolution — it is the answer to "is this the same texture?" — and
//! `content_key` folds it back in for the cache, which does have to tell two
//! resolutions apart.
//!
//! ## Seamless tiling: the decision
//!
//! The original is a *world-space* shader: it evaluates 3D noise at the shading
//! point and never has a UV, so it has no seams to speak of and no tile either.
//! A bake must pick a domain. Three options were on the table:
//!
//! 1. **Three bakes (or a 2D array), one per triplanar plane.** Exact, and 3×
//!    the memory and bake time for a look whose whole point is that the three
//!    planes are interchangeable.
//! 2. **One tile, world-space UV, seams accepted** and hidden by the anti-tiling
//!    sampler. Cheapest, but a visible discontinuity is a visible discontinuity.
//! 3. **One tile baked in a wrapped lattice domain, plus anti-tiling on top.**
//!
//! runt does (3), which is also the shape the Godot fallback shader settled on
//! (`shaders/terrain/terrain_common.gdshaderinc`: a seamless `NoiseTexture2D`,
//! sampled triplanar, with Quilez anti-tiling). Cellular noise reads its input
//! only through integer cell indices, so wrapping the index makes the field
//! *exactly* periodic — no blend, no mirror, no ghosting, and the seam is
//! bit-identical rather than merely inconspicuous. That is strictly better than
//! Godot's blended seamless mode, and it costs one `floor` per cell.
//!
//! The price is [`OctavePlan`]: each octave's cell span across the tile has to
//! be a whole number (an *even* one on the FCC lattice, or the wrap would flip
//! cell parity), so the authored lacunarity chain is quantized. The deviation is
//! sub-percent at every octave for the authored materials and is reported by
//! [`TextureSpec::lacunarity_error`] so it can never be silently large.
//!
//! Tiling is still tiling: one texture repeated across a hillside would read as
//! wallpaper. That is what the anti-tiling in `shader.wgsl` is for, and it is
//! the same Quilez trick the original used — with the `sin`-based hash replaced,
//! because DESIGN §7 forbids those.

use bevy_ecs::prelude::Resource;
use glam::{Vec2, Vec3};
use runt_mesh::Quality;
use serde::{Deserialize, Serialize};

use crate::noise::{
    self, CellReturn, FbmAccum, Fractal, Lattice, OCTAVE_LOD_OFF,
};

// ---------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------

/// Octaves the bake uniform has room for. The authored materials use 4–5.
pub const MAX_OCTAVES: usize = 8;

/// Gradient stops the bake uniform has room for. The authored ramps use 3.
pub const MAX_RAMP_STOPS: usize = 8;

/// Bake resolution floor. Below this the anti-tiling sampler has nothing to
/// work with and the normal map is mush; DESIGN §11 says scale *down*, not off.
pub const MIN_RESOLUTION: u32 = 64;

/// Bake resolution ceiling (DESIGN §11: "≤2048², tier-scaled" — and 2048 is
/// exactly `downlevel_webgl2_defaults().max_texture_dimension_2d`, so the
/// baseline path cannot ask for a texture WebGL2 would refuse).
pub const MAX_RESOLUTION: u32 = 2048;

/// The texel step the boundary-normal accumulation differentiates over,
/// expressed as a fraction of the tile.
///
/// **Not** the real texel size. The packed normal is
/// `normalize(-dndx, -dndy, 0.5)`, so the magnitude of the derivative decides
/// how steep the normal reads; tying it to the actual resolution would make a
/// Low-tier device's terrain visibly *flatter* rather than merely coarser, and
/// DESIGN §11's contract is that a gate picks data, never a different look. A
/// fixed reference step makes the normal map resolution-independent.
pub const NORMAL_REFERENCE_TEXELS: f32 = 1024.0;

// ---------------------------------------------------------------------------
// The spec
// ---------------------------------------------------------------------------

/// Which noise the texture is made of.
///
/// An enum with one variant today, on purpose: it is the seam where value /
/// Perlin / simplex land if a material ever wants them, and having it now keeps
/// the serialized form (and therefore every cached bake) stable when they do.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect))]
pub enum NoiseSpec {
    Cellular {
        #[serde(default)]
        lattice: Lattice,
        #[serde(default)]
        return_type: CellReturn,
        /// `0` locks feature points to cell centres (a regular grid); `1` lets
        /// them anywhere in the cell. Every authored material uses `1`.
        #[serde(default = "one")]
        jitter: f32,
    },
}

impl Default for NoiseSpec {
    fn default() -> NoiseSpec {
        NoiseSpec::Cellular {
            lattice: Lattice::default(),
            return_type: CellReturn::default(),
            jitter: 1.0,
        }
    }
}

impl NoiseSpec {
    pub fn lattice(self) -> Lattice {
        match self {
            NoiseSpec::Cellular { lattice, .. } => lattice,
        }
    }

    pub fn return_type(self) -> CellReturn {
        match self {
            NoiseSpec::Cellular { return_type, .. } => return_type,
        }
    }

    pub fn jitter(self) -> f32 {
        match self {
            NoiseSpec::Cellular { jitter, .. } => jitter,
        }
    }
}

/// Where a Voronoi-boundary normal points.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect))]
pub enum NormalMode {
    /// Radially outward from the nearest feature point through the sample —
    /// each cell reads as a rounded pebble. The grass material's choice.
    #[default]
    ToPoint,
    /// Perpendicular to the F1→F2 boundary — each cell reads as a facet with a
    /// crisp bevelled edge. The rock material's choice.
    ToEdge,
}

impl NormalMode {
    pub fn code(self) -> u32 {
        match self {
            NormalMode::ToPoint => 1,
            NormalMode::ToEdge => 2,
        }
    }
}

/// The boundary-normal pass's params.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect))]
pub struct NormalSpec {
    #[serde(default)]
    pub mode: NormalMode,
    /// How far from a cell boundary the crease is still felt, in cell units.
    /// Small values give thin crisp creases, large ones wide soft bevels.
    #[serde(default = "quarter")]
    pub edge_width: f32,
    /// Multiplies the accumulated gradient before packing. The authored
    /// materials run hot (≈5–6) because the boundary term is small.
    #[serde(default = "one")]
    pub strength: f32,
}

impl Default for NormalSpec {
    fn default() -> NormalSpec {
        NormalSpec {
            mode: NormalMode::default(),
            edge_width: 0.25,
            strength: 1.0,
        }
    }
}

/// A procedural texture, as a scene file writes it.
///
/// Field names are the interface (same rule as `GeneratorSpec`): renaming one
/// breaks scene files, adding a `#[serde(default)]` one does not.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect))]
pub struct TextureSpec {
    #[serde(default)]
    pub noise: NoiseSpec,
    /// Cells per world unit at octave 0, before [`world_scale`] maps world
    /// units onto the tile. The authored number from the Godot material.
    ///
    /// [`world_scale`]: TextureSpec::world_scale
    #[serde(default = "one")]
    pub frequency: f32,
    #[serde(default = "five")]
    pub octaves: u32,
    #[serde(default = "two")]
    pub lacunarity: f32,
    #[serde(default = "half")]
    pub gain: f32,
    #[serde(default)]
    pub fractal: Fractal,
    /// [`Fractal::Ridged`] only: how hard a ridge suppresses the octave above.
    #[serde(default = "half")]
    pub weighted_strength: f32,
    /// `(n − 0.5) · contrast + 0.5`, clamped.
    #[serde(default = "one")]
    pub contrast: f32,
    /// Applied after contrast, before the ramp lookup.
    #[serde(default = "one")]
    pub brightness: f32,
    /// Gradient stops as `(offset, rgb)`, offsets in `[0, 1]`. Empty means
    /// greyscale. Values are used as authored — runt has no colour management,
    /// so these are the same numbers `base_color` uses.
    ///
    /// Invisible to the editor's reflected panels: `bevy_reflect` targets a
    /// glam major behind runt's, and the `#[reflect(remote = …)]` escape in
    /// [`crate::reflect`] works on a *field*, not on a `Vec3` buried inside a
    /// `Vec<(f32, Vec3)>`. A gradient wants a purpose-built widget anyway —
    /// `MaterialDesc`'s colours are ignored for the same reason.
    #[serde(default)]
    #[cfg_attr(feature = "reflect", reflect(ignore))]
    pub ramp: Vec<(f32, Vec3)>,
    /// `None` leaves the normal map flat (and the bake still writes it, so the
    /// material path has no special case).
    #[serde(default)]
    pub normal: Option<NormalSpec>,
    /// Displaces the sampled region. Two specs differing only here are two
    /// unrelated textures.
    #[serde(default)]
    pub seed_offset: f32,
    /// World units → tile units: one tile spans `1 / world_scale` world units.
    /// The `noise_scale` of the Godot fallback material, same meaning.
    ///
    /// **Authoring note.** This is the knob that decides how cleanly the tile
    /// quantizes: `frequency / world_scale` is the cell span across the tile
    /// and wants to land near an even integer (see
    /// [`lacunarity_error`](TextureSpec::lacunarity_error)). It does *not*
    /// change feature size — a cell is `1 / frequency` world units across
    /// whatever this says — so it is free to tune.
    #[serde(default = "default_world_scale")]
    pub world_scale: f32,
    /// Triplanar blend exponent. Higher is a harder switch between planes.
    #[serde(default = "four")]
    pub triplanar_sharpness: f32,
    /// Quilez anti-tiling in the *sampler* (not the bake). ~4 taps per plane
    /// instead of 1, which is the difference between "one texture, invisible"
    /// and "one texture, obviously". Off is the low-tier fallback.
    #[serde(default = "yes")]
    pub anti_tiling: bool,
    /// Bake resolution at `Quality::FULL`, before the tier scales it and before
    /// [`MIN_RESOLUTION`]/[`MAX_RESOLUTION`] clamp it.
    #[serde(default = "default_resolution")]
    pub base_resolution: u32,
}

fn one() -> f32 {
    1.0
}
fn two() -> f32 {
    2.0
}
fn four() -> f32 {
    4.0
}
fn half() -> f32 {
    0.5
}
fn quarter() -> f32 {
    0.25
}
fn five() -> u32 {
    5
}
fn yes() -> bool {
    true
}
fn default_world_scale() -> f32 {
    0.05
}
fn default_resolution() -> u32 {
    512
}

impl Default for TextureSpec {
    fn default() -> TextureSpec {
        TextureSpec {
            noise: NoiseSpec::default(),
            frequency: 1.0,
            octaves: 5,
            lacunarity: 2.0,
            gain: 0.5,
            fractal: Fractal::default(),
            weighted_strength: 0.5,
            contrast: 1.0,
            brightness: 1.0,
            ramp: Vec::new(),
            normal: None,
            seed_offset: 0.0,
            world_scale: default_world_scale(),
            triplanar_sharpness: 4.0,
            anti_tiling: true,
            base_resolution: default_resolution(),
        }
    }
}

/// One octave, resolved against the tile.
///
/// Computed on the CPU and uploaded verbatim, so the shader has no `pow`, no
/// rounding rule and no chance of disagreeing with the CPU twin about what it
/// is sampling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OctavePlan {
    /// Lattice cells this octave spans across one tile. A whole number (even on
    /// FCC) — that is what makes the wrap seamless.
    pub span: f32,
    /// This octave's frequency relative to octave 0, i.e. the *quantized*
    /// lacunarity chain. Used to scale the boundary-normal contribution, as
    /// `amplitude · freq` in the original.
    pub freq: f32,
    /// `gain^i`.
    pub amplitude: f32,
    /// Distance-LOD weight. Always 1 at bake time (there is no camera), but it
    /// multiplies into the normalizing sum as well as the value, so the code
    /// path is the one a live-eval variant will need unchanged.
    pub weight: f32,
}

impl TextureSpec {
    // -- identity -----------------------------------------------------------

    /// The spec's stable identity: FNV-1a over its `postcard` bytes.
    ///
    /// Resolution-free on purpose — see the module docs. Same scheme as
    /// [`GeneratorSpec::param_key`](crate::gen::GeneratorSpec::param_key), and
    /// for the same reasons: `f32` has no `Hash`, `DefaultHasher` is not stable
    /// across toolchains, and a canonical byte form sidesteps both.
    pub fn param_key(&self) -> u64 {
        let bytes = postcard::to_stdvec(self)
            .expect("TextureSpec serializes to postcard infallibly (no maps, no non-UTF8)");
        let h = crate::gen::fnv(crate::gen::FNV_OFFSET, b"TextureSpec");
        crate::gen::fnv(h, &bytes)
    }

    /// The bake's cache key: the spec plus the resolution it was baked at.
    pub fn content_key(&self, resolution: u32) -> u64 {
        crate::gen::fnv(self.param_key(), &resolution.to_le_bytes())
    }

    // -- resolution ---------------------------------------------------------

    /// Bake resolution for a quality tier: `base × quality`, clamped to
    /// `[MIN_RESOLUTION, MAX_RESOLUTION]` and rounded to a power of two.
    ///
    /// Powers of two because the bake is a repeat-sampled tile and every
    /// downlevel backend is happier with one, and because it keeps the tier
    /// ladder short enough that two nearby qualities share a cache entry rather
    /// than each minting their own.
    pub fn resolution(&self, quality: Quality) -> u32 {
        let want = (self.base_resolution as f32 * quality.0.max(0.0)).round().max(1.0);
        let pow2 = 1u32 << (want.log2().round().max(0.0) as u32).min(16);
        pow2.clamp(MIN_RESOLUTION, MAX_RESOLUTION)
    }

    // -- octave planning ----------------------------------------------------

    /// Cells across one tile at octave 0, before quantization.
    ///
    /// One tile spans `1 / world_scale` world units and the noise runs at
    /// `frequency` cells per world unit, so this is just their product.
    pub fn base_span(&self) -> f32 {
        (self.frequency / self.world_scale.max(1e-6)).max(0.0)
    }

    /// Round a cell span to something the wrap can use: a whole number, and an
    /// even one on FCC (an odd period would flip `x+y+z` parity at the seam and
    /// the lattice would not line up with itself).
    fn quantize_span(&self, ideal: f32) -> f32 {
        match self.noise.lattice() {
            Lattice::Cubic => ideal.round().max(1.0),
            Lattice::Fcc => (ideal * 0.5).round().max(1.0) * 2.0,
        }
    }

    /// The octaves, resolved. `octaves` is clamped to [`MAX_OCTAVES`] — a spec
    /// asking for more gets fewer rather than an overflowing uniform.
    pub fn octave_plan(&self) -> Vec<OctavePlan> {
        let count = (self.octaves.max(1) as usize).min(MAX_OCTAVES);
        let base = self.quantize_span(self.base_span());
        let mut plan = Vec::with_capacity(count);
        let mut ideal = self.base_span();
        let mut amplitude = 1.0f32;
        for i in 0..count {
            let span = self.quantize_span(ideal);
            plan.push(OctavePlan {
                span,
                freq: span / base,
                amplitude,
                weight: noise::octave_weight(i as u32, OCTAVE_LOD_OFF.0, OCTAVE_LOD_OFF.1),
            });
            ideal *= self.lacunarity;
            amplitude *= self.gain;
        }
        plan
    }

    /// The largest relative distance between an octave's authored frequency and
    /// the quantized one it is actually baked at.
    ///
    /// The honest cost of seamlessness. `0.03` means "the worst octave is 3%
    /// off the lacunarity chain you typed"; anything much above that means the
    /// tile is too small for the frequency and `world_scale` wants lowering.
    pub fn lacunarity_error(&self) -> f32 {
        let mut worst = 0.0f32;
        let mut ideal = self.base_span();
        for plan in self.octave_plan() {
            if ideal > 0.0 {
                worst = worst.max(((plan.span - ideal) / ideal).abs());
            }
            ideal *= self.lacunarity;
        }
        worst
    }

    // -- CPU evaluation -----------------------------------------------------

    /// The raw field at a tile coordinate, plus the boundary-normal gradient.
    ///
    /// `uv` is in tile units: `(0,0)` and `(1,0)` are the same texel by
    /// construction. Returns `(value, dndx, dndy)` with `value` **before**
    /// contrast/brightness — [`postprocess`](TextureSpec::postprocess) applies
    /// those.
    ///
    /// This is the exact arithmetic `noise.wgsl`'s `bake_sample` performs; the
    /// two are held together by `tests/noise_bake.rs`.
    pub fn sample_at(&self, uv: Vec2) -> (f32, f32, f32) {
        let plan = self.octave_plan();
        let offset = noise::seed_offset_3d(self.seed_offset);
        let normal = self.normal.unwrap_or_default();
        let has_normal = self.normal.is_some();
        let lattice = self.noise.lattice();
        let ret = self.noise.return_type();
        let jitter = self.noise.jitter();

        // The step the normal differentiates over, in *base* sample-space units
        // — resolution-independent by design (see NORMAL_REFERENCE_TEXELS).
        let base_span = plan.first().map(|p| p.span).unwrap_or(1.0);
        let step = base_span / NORMAL_REFERENCE_TEXELS;

        let mut accum = FbmAccum::new();
        let (mut dndx, mut dndy) = (0.0f32, 0.0f32);

        for octave in &plan {
            let p = Vec3::new(
                uv.x * octave.span + offset.x * octave.freq,
                uv.y * octave.span + offset.y * octave.freq,
                offset.z * octave.freq,
            );
            let period = Vec3::new(octave.span, octave.span, 0.0);
            let cell = noise::cellular(p, lattice, ret, jitter, period);

            accum.push(
                cell.value,
                octave.amplitude,
                octave.weight,
                self.fractal,
                self.weighted_strength,
            );

            if has_normal {
                let edge_mag =
                    1.0 - smoothstep(0.0, normal.edge_width.max(1e-6), cell.d2 - cell.d1);
                let delta = match normal.mode {
                    NormalMode::ToEdge => cell.f2 - cell.f1,
                    NormalMode::ToPoint => p - cell.f1,
                };
                let len = delta.length();
                let dir = if len > 1e-4 { delta / len } else { Vec3::ZERO };
                let w = octave.amplitude * octave.freq * edge_mag * octave.weight * normal.strength;
                dndx += w * dir.x * step;
                dndy += w * dir.y * step;
            }
        }

        (accum.finish(), dndx, dndy)
    }

    /// Contrast, brightness and the `[0,1]` clamp, in the original's order.
    pub fn postprocess(&self, n: f32) -> f32 {
        let n = ((n - 0.5) * self.contrast + 0.5).clamp(0.0, 1.0);
        (n * self.brightness).clamp(0.0, 1.0)
    }

    /// The gradient ramp at `t ∈ [0,1]`. Linear between stops, held flat
    /// outside the first and last — Godot's `GradientTexture1D` semantics.
    pub fn ramp_at(&self, t: f32) -> Vec3 {
        let t = t.clamp(0.0, 1.0);
        if self.ramp.is_empty() {
            return Vec3::splat(t);
        }
        let stops = &self.ramp;
        if t <= stops[0].0 {
            return stops[0].1;
        }
        for pair in stops.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            if t <= b.0 {
                let span = b.0 - a.0;
                let f = if span > 1e-6 { (t - a.0) / span } else { 0.0 };
                return a.1.lerp(b.1, f);
            }
        }
        stops[stops.len() - 1].1
    }

    /// The baked albedo at a tile coordinate — the CPU model of one texel of
    /// the albedo target.
    pub fn albedo_at(&self, uv: Vec2) -> Vec3 {
        let (n, _, _) = self.sample_at(uv);
        self.ramp_at(self.postprocess(n))
    }

    /// The baked normal at a tile coordinate, **packed** to `[0,1]` exactly as
    /// the texture stores it.
    pub fn packed_normal_at(&self, uv: Vec2) -> Vec3 {
        let (_, dndx, dndy) = self.sample_at(uv);
        let n = Vec3::new(-dndx, -dndy, 0.5).normalize_or(Vec3::Z);
        n * 0.5 + Vec3::splat(0.5)
    }
}

// ---------------------------------------------------------------------------
// Triplanar
// ---------------------------------------------------------------------------

/// Triplanar blend weights for a world-space normal — the CPU twin of
/// `shader.wgsl`'s `triplanar_blend`.
///
/// `pow(abs(n), sharpness)` normalized to sum to 1. **The weight-to-plane
/// mapping is the original's and is not the obvious one**: the `z` weight drives
/// the *XY* plane's sample, `y` drives *XZ*, `x` drives *YZ*. A ground plane
/// (normal `+Y`) therefore samples the XZ plane, which is what anyone would
/// want and not what a naive reading of "blend.x → plane x" would give.
///
/// A zero-length normal (broken geometry) falls back to an even third rather
/// than dividing by zero.
pub fn triplanar_blend(normal: Vec3, sharpness: f32) -> Vec3 {
    let b = Vec3::new(
        normal.x.abs().powf(sharpness),
        normal.y.abs().powf(sharpness),
        normal.z.abs().powf(sharpness),
    );
    let sum = b.x + b.y + b.z;
    if sum > 1e-6 {
        b / sum
    } else {
        Vec3::splat(1.0 / 3.0)
    }
}

/// `smoothstep`, GLSL semantics.
#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0).max(1e-9)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------
// Handles and the library
// ---------------------------------------------------------------------------

/// A baked texture's key: [`TextureSpec::content_key`].
///
/// The texture-side twin of [`MeshHandle`](crate::registry::MeshHandle) — equal
/// handles mean "the same pixels", so two materials naming the same spec at the
/// same resolution share one GPU texture without anyone arranging it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextureHandle(pub u64);

/// Handle → the spec (and resolution) that produced it, as a world resource.
///
/// GPU-free, exactly like [`MeshLibrary`](crate::registry::MeshLibrary): the
/// scene loader fills it, and the renderer bakes from it lazily the first time
/// a draw actually asks for the texture. Scene load therefore stays a
/// device-free operation and the sim's tests never need an adapter.
#[derive(Resource, Default, Clone, Debug)]
pub struct TextureLibrary {
    entries: Vec<(TextureHandle, TextureSpec, u32)>,
}

impl TextureLibrary {
    pub fn new() -> TextureLibrary {
        TextureLibrary::default()
    }

    /// Register `spec` at `resolution`, returning its handle. Idempotent.
    ///
    /// A `Vec` rather than a `HashMap` because scenes have a handful of
    /// textures, insertion order is the scene file's order, and DESIGN §3 would
    /// rather nothing near content resolution iterate a hash container.
    pub fn insert(&mut self, spec: TextureSpec, resolution: u32) -> TextureHandle {
        let handle = TextureHandle(spec.content_key(resolution));
        if !self.entries.iter().any(|(h, _, _)| *h == handle) {
            self.entries.push((handle, spec, resolution));
        }
        handle
    }

    pub fn get(&self, handle: TextureHandle) -> Option<(&TextureSpec, u32)> {
        self.entries
            .iter()
            .find(|(h, _, _)| *h == handle)
            .map(|(_, spec, res)| (spec, *res))
    }

    pub fn contains(&self, handle: TextureHandle) -> bool {
        self.entries.iter().any(|(h, _, _)| *h == handle)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (TextureHandle, &TextureSpec, u32)> {
        self.entries.iter().map(|(h, s, r)| (*h, s, *r))
    }
}

// ---------------------------------------------------------------------------
// Authored materials (DESIGN §7, ported from 3dimenshift)
// ---------------------------------------------------------------------------

/// The `grass_cell` material's params, verbatim from the port spec.
///
/// Here rather than only in a scene file because they are *reference data*: the
/// port spec's table is the source of truth and a test holds this against it,
/// so a scene file that drifts is a scene file that is wrong.
///
/// The one number that is **not** from the Godot material is `world_scale`,
/// which the original does not have an equivalent of in the live-noise shader
/// (it sampled world space unbounded). It is chosen so the tile quantizes
/// cleanly — see [`TextureSpec::world_scale`].
// `lacunarity: 2.718` is what the artist typed into the Godot material. It is
// not an approximation of `e`, and swapping in `std::f32::consts::E` would
// change every texture the engine has ever baked — hence the allow.
#[allow(clippy::approx_constant)]
pub fn grass() -> TextureSpec {
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
            (0.12, Vec3::new(0.0, 0.31, 0.17566665)),
            (0.45714286, Vec3::new(0.0, 0.44481665, 0.31058022)),
            (1.0, Vec3::new(0.0, 0.53464526, 0.37844315)),
        ],
        normal: Some(NormalSpec {
            mode: NormalMode::ToPoint,
            edge_width: 0.52,
            strength: 5.106,
        }),
        // 27.8 m tile, 5.83 → 6 cells across it: a 2.9% frequency nudge.
        world_scale: 0.036,
        triplanar_sharpness: 4.0,
        base_resolution: 1024,
        ..TextureSpec::default()
    }
}

/// The `rock_cell` material's params, verbatim from the port spec.
pub fn rock() -> TextureSpec {
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
            (0.12, Vec3::new(0.23, 0.1909, 0.20719166)),
            (0.45714286, Vec3::new(0.27, 0.2187, 0.240075)),
            (1.0, Vec3::new(0.41, 0.3362, 0.36694998)),
        ],
        normal: Some(NormalSpec {
            mode: NormalMode::ToEdge,
            edge_width: 0.351,
            strength: 5.921,
        }),
        // The Godot fallback said 0.01 (a 100 m tile), which puts 4.6 cells
        // across the tile and would cost a 13% frequency nudge to quantize.
        // 0.0046 is the nearest value that lands on exactly 10 cells — same
        // feature size (21.7 m per cell), a bigger tile, no rounding.
        world_scale: 0.0046,
        triplanar_sharpness: 1.0,
        base_resolution: 1024,
        ..TextureSpec::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tile_is_seamless_in_both_axes() {
        // The whole point of the wrapped-lattice bake: the field at u=0 and u=1
        // is not merely close, it is the same number.
        let spec = grass();
        for i in 0..16 {
            let t = i as f32 / 16.0;
            let (a, _, _) = spec.sample_at(Vec2::new(0.0, t));
            let (b, _, _) = spec.sample_at(Vec2::new(1.0, t));
            assert!((a - b).abs() < 1e-5, "u seam at v={t}: {a} vs {b}");

            let (c, _, _) = spec.sample_at(Vec2::new(t, 0.0));
            let (d, _, _) = spec.sample_at(Vec2::new(t, 1.0));
            assert!((c - d).abs() < 1e-5, "v seam at u={t}: {c} vs {d}");
        }
    }

    #[test]
    fn octave_spans_are_whole_and_even_on_fcc() {
        for spec in [grass(), rock()] {
            for octave in spec.octave_plan() {
                assert_eq!(octave.span, octave.span.round(), "span must be integral");
                assert_eq!(octave.span % 2.0, 0.0, "FCC needs an even period");
                assert!(octave.span >= 2.0);
            }
        }
    }

    #[test]
    fn quantizing_the_lacunarity_chain_stays_cheap() {
        // The documented price of seamlessness. If this ever fails, the tile is
        // too small for the frequency, not the rounding rule being wrong.
        for (name, spec) in [("grass", grass()), ("rock", rock())] {
            let error = spec.lacunarity_error();
            assert!(error < 0.05, "{name}: lacunarity quantized by {error}");
        }
    }

    #[test]
    fn param_key_ignores_resolution_but_content_key_does_not() {
        let spec = grass();
        assert_eq!(spec.param_key(), grass().param_key());
        assert_ne!(spec.content_key(512), spec.content_key(1024));
        assert_eq!(spec.content_key(512), grass().content_key(512));

        let mut other = grass();
        other.seed_offset = 4.0;
        assert_ne!(spec.param_key(), other.param_key());
    }

    #[test]
    fn resolution_scales_with_the_tier_and_respects_the_caps() {
        let spec = TextureSpec {
            base_resolution: 512,
            ..TextureSpec::default()
        };
        assert_eq!(spec.resolution(Quality::FULL), 512);
        assert_eq!(spec.resolution(Quality(0.5)), 256);
        assert_eq!(spec.resolution(Quality(2.0)), 1024);
        // DESIGN §11: never past 2048 on the baseline path, never below usable.
        assert_eq!(spec.resolution(Quality(64.0)), MAX_RESOLUTION);
        assert_eq!(spec.resolution(Quality(0.0001)), MIN_RESOLUTION);
    }

    #[test]
    fn the_ramp_holds_its_ends_and_interpolates_between() {
        let spec = grass();
        let first = spec.ramp[0].1;
        let last = spec.ramp[2].1;
        assert_eq!(spec.ramp_at(0.0), first);
        assert_eq!(spec.ramp_at(0.05), first, "held flat below the first stop");
        assert_eq!(spec.ramp_at(1.0), last);
        let mid = spec.ramp_at(0.3);
        assert!(mid.y > first.y && mid.y < spec.ramp[1].1.y, "{mid:?}");

        // A spec with no ramp is greyscale, not black.
        let plain = TextureSpec::default();
        assert_eq!(plain.ramp_at(0.7), Vec3::splat(0.7));
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn the_authored_params_match_the_port_spec_table() {
        // freq / oct / lacunarity / gain / return / normal mode / edge / strength
        let g = grass();
        assert_eq!(g.frequency, 0.21);
        assert_eq!(g.octaves, 5);
        assert_eq!(g.lacunarity, 2.718);
        assert_eq!(g.gain, 0.562);
        assert_eq!(g.noise.return_type(), CellReturn::CellValue);
        assert_eq!(g.noise.jitter(), 1.0);
        let gn = g.normal.expect("grass has normals");
        assert_eq!(gn.mode, NormalMode::ToPoint);
        assert_eq!(gn.edge_width, 0.52);
        assert_eq!(gn.strength, 5.106);

        let r = rock();
        assert_eq!(r.frequency, 0.046);
        assert_eq!(r.octaves, 5);
        assert_eq!(r.lacunarity, 3.512);
        assert_eq!(r.gain, 0.543);
        let rn = r.normal.expect("rock has normals");
        assert_eq!(rn.mode, NormalMode::ToEdge);
        assert_eq!(rn.edge_width, 0.351);
        assert_eq!(rn.strength, 5.921);
    }

    #[test]
    fn the_library_dedups_by_content_key() {
        let mut library = TextureLibrary::new();
        let a = library.insert(grass(), 512);
        let b = library.insert(grass(), 512);
        let c = library.insert(grass(), 1024);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(library.len(), 2);
        assert!(library.contains(a));
        assert_eq!(library.get(a).expect("registered").1, 512);
    }

    #[test]
    fn triplanar_weights_always_sum_to_one() {
        // Every direction on a coarse sphere, at every sharpness the materials
        // use — a weight set that does not sum to 1 darkens or blows out the
        // surface it lands on, and it would do so quietly.
        for sharpness in [1.0f32, 2.0, 4.0, 8.0, 16.0] {
            for i in 0..64 {
                for j in 0..32 {
                    let theta = i as f32 / 64.0 * std::f32::consts::TAU;
                    let phi = j as f32 / 31.0 * std::f32::consts::PI;
                    let n = Vec3::new(
                        phi.sin() * theta.cos(),
                        phi.cos(),
                        phi.sin() * theta.sin(),
                    );
                    let b = triplanar_blend(n, sharpness);
                    let sum = b.x + b.y + b.z;
                    assert!(
                        (sum - 1.0).abs() < 1e-4,
                        "n {n:?} sharpness {sharpness}: weights {b:?} sum to {sum}"
                    );
                    assert!(b.min_element() >= 0.0, "negative weight in {b:?}");
                }
            }
        }
    }

    #[test]
    fn triplanar_picks_the_plane_the_surface_faces() {
        // Ground (+Y) reads the XZ plane, which is the `y` weight — the
        // original's mapping, and the one that is easy to get backwards.
        let ground = triplanar_blend(Vec3::Y, 4.0);
        assert!(ground.y > 0.999, "{ground:?}");
        let wall = triplanar_blend(Vec3::X, 4.0);
        assert!(wall.x > 0.999, "{wall:?}");
        // A 45° normal splits evenly between the two planes it faces.
        let slope = triplanar_blend(Vec3::new(1.0, 1.0, 0.0).normalize(), 4.0);
        assert!((slope.x - slope.y).abs() < 1e-5 && slope.z < 1e-5, "{slope:?}");
        // Degenerate input must produce weights, not NaN.
        let broken = triplanar_blend(Vec3::ZERO, 4.0);
        assert!(broken.abs_diff_eq(Vec3::splat(1.0 / 3.0), 1e-6), "{broken:?}");
    }

    #[test]
    fn a_normal_free_spec_bakes_a_flat_normal() {
        let spec = TextureSpec {
            normal: None,
            ..grass()
        };
        let packed = spec.packed_normal_at(Vec2::new(0.3, 0.7));
        assert!(packed.abs_diff_eq(Vec3::new(0.5, 0.5, 1.0), 1e-5), "{packed:?}");
    }
}
