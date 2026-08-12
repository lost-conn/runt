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
    self, CellReturn, FbmAccum, Fractal, Lattice, NoiseKind, OCTAVE_LOD_OFF,
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
///
/// # It is *not* tile-independent
///
/// The step the bake actually uses is `base_span / NORMAL_REFERENCE_TEXELS`, in
/// units of octave-0 cells — which is one texel of a 1024² bake *of this tile*,
/// so in world metres it is `tile_meters / 1024`. Shrink the tile and the step
/// shrinks with it, the field rises less across it, and the whole normal map
/// flattens in proportion.
///
/// So [`world_scale`] is free to tune for the albedo and **not** free for the
/// normal: halving the tile halves the relief unless
/// [`NormalSpec::strength`] doubles. What is invariant — and what a material
/// re-tiling itself has to hold fixed — is the product
/// `strength × octave_plan()[0].span`; `rock`'s comment works an example, and
/// `the_normal_amplitude_survived_the_density_retune` pins it.
///
/// Making the step a fixed fraction of a *cell* instead would drop the coupling
/// entirely and is the better formulation, but it re-scales `strength` for every
/// material ever authored against this one — a decision worth taking on its own
/// rather than as a side effect of a tiling change.
///
/// [`world_scale`]: TextureSpec::world_scale
pub const NORMAL_REFERENCE_TEXELS: f32 = 1024.0;

/// How many pixels wide a lattice cell must stay before §7's **live** path
/// stops evaluating its octave.
///
/// The live path has no mip chain — there is nothing pre-filtered to fall back
/// to — so this is its substitute, and it is the same rule a mip selector uses:
/// detail finer than the sampling rate is not detail, it is noise. Two pixels
/// per cell is one octave below the Nyquist limit of a point-sampled field,
/// which leaves the fade a whole octave to happen in before aliasing would
/// start.
///
/// Raising it blurs distant surfaces and makes them cheaper; lowering it is how
/// you buy shimmer. Zero turns the window off entirely (every octave at full
/// weight, the bake's behaviour), which is what the CPU twin compares against.
///
/// This used to be the whole knob — one value, engine-wide. It is now only the
/// *default* for [`TextureSpec::live_cell_pixels`]: a material made of fine
/// detail (grass) and one made of large slow-changing blobs (a cliff face) do
/// not want to give up their finest octave at the same footprint, and a single
/// `const` could not tell them apart. Kept rather than deleted because it is
/// still the number every existing scene file and cached bake was authored
/// against, and it is what [`TextureUniform::inert`](crate::bake::TextureUniform::inert)
/// uses for a draw that has no spec to read a per-material value from.
pub const LIVE_LOD_CELL_PIXELS: f32 = 2.0;

// ---------------------------------------------------------------------------
// The spec
// ---------------------------------------------------------------------------

/// Which noise the texture is made of.
///
/// The seam where value / Perlin / simplex land if a material ever wants them.
/// Two variants today, and the second is worth an explanation because it looks
/// like a special case of the first.
///
/// # Why `Grid` is a variant and not `Cellular { jitter: 0.0 }`
///
/// It *is* that field — [`noise::grid`]'s docs work the algebra and
/// `the_grid_is_jitter_free_cellular` measures it — but the spelling decides
/// what the shader does. Written as a cellular spec, a jitter-free lattice
/// still pays for a 27-cell search and 27 hashes to rediscover a number two
/// lines of arithmetic already know. Written as its own kind, the fragment
/// shader takes a branch and skips the loop, which is what makes it cheap
/// enough for §7's **live** path on a moving object.
///
/// The second reason is that it is parameterised by *nothing*. `lattice`,
/// `return_type` and `jitter` are all meaningless to it — a grid has one
/// lattice, one return and no jitter by definition — and a variant that
/// carried three fields it ignores would advertise combinations (FCC, a grid
/// with `CellValue`) that the closed form does not cover.
///
/// [`noise::grid`]: crate::noise::grid
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect))]
pub enum NoiseSpec {
    Cellular {
        #[serde(default)]
        lattice: Lattice,
        #[serde(default)]
        return_type: CellReturn,
        /// `0` locks feature points to cell centres (a regular grid); `1` lets
        /// them anywhere in the cell. Every authored material uses `1` — a
        /// material that wants `0` wants [`Grid`](NoiseSpec::Grid) instead.
        #[serde(default = "one")]
        jitter: f32,
    },
    /// The jitter-free cubic lattice in closed form: round cells on a regular
    /// grid, no hash, no search. See [`noise::grid`](crate::noise::grid).
    Grid,
    /// [`Grid`](NoiseSpec::Grid) in **cylindrical** coordinates about `+Y`:
    /// wedges around and bands up — the two axes a UV-mapped texture has, and
    /// the shape it makes on a sphere. See
    /// [`noise::radial_grid`](crate::noise::radial_grid).
    ///
    /// # It is about an axis, so it is only meaningful in object space
    ///
    /// The axis is the `+Y` through the **sample space's origin**. Under
    /// `MaterialVariant::LOCAL_SPACE` that is the entity's own axis, which is
    /// what this is for; in the world basis it is the world origin, and a
    /// surface anywhere else gets a slice of somebody else's wedges.
    ///
    /// [`TextureSpec::seed_offset`] therefore means something different here
    /// than elsewhere, and [`TextureSpec::seed_displacement`] is what makes it
    /// safe: the seed slides the bands **along** the axis and never moves it.
    /// Left unfiltered it displaces the origin by hundreds of units even at
    /// `seed_offset: 0.0`, which does not decorrelate this field — it deletes
    /// it. [`NoiseKind::has_axis`](crate::noise::NoiseKind::has_axis) tells that
    /// story in full.
    ///
    /// # It does not tile
    ///
    /// The angular coordinate wraps by construction and `y` is periodic, but
    /// neither is a *translation*: a tile is a plane, and this field is not
    /// translation-invariant in `x` or `z`. Worse, a bake's plane is
    /// `z = 0` — so `atan2(0, x)` is constant across it and a baked tile of this
    /// kind is **bands only**, with the wedges collapsed out entirely.
    ///
    /// A material wanting this should therefore ask for the **live** path
    /// (`MaterialVariant::LIVE_TEX`). The baked path is left defined rather than
    /// forbidden because it is what a live/baked A/B toggle shows, and a mode
    /// that refused to render would be worse than one that renders honestly
    /// badly — but the two modes are *not* the same picture here, which is the
    /// one place in §7 where that is true and is why it is said this loudly.
    RadialGrid {
        /// Wedges around one full turn at octave 0. Rounded to a whole number
        /// per octave — a fractional count puts a seam down the `+X` half-plane,
        /// which [`noise::radial_grid`](crate::noise::radial_grid) explains.
        #[serde(default = "four_sectors")]
        sectors: u32,
    },
}

fn four_sectors() -> u32 {
    4
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
    pub fn kind(self) -> NoiseKind {
        match self {
            NoiseSpec::Cellular { .. } => NoiseKind::Cellular,
            NoiseSpec::Grid => NoiseKind::Grid,
            NoiseSpec::RadialGrid { .. } => NoiseKind::RadialGrid,
        }
    }

    /// The lattice the field sits on. `Grid` answers [`Lattice::Cubic`] because
    /// it *is* the cubic lattice — which is not cosmetic: it is what
    /// `TextureSpec::quantize_span` reads to decide that a grid tile rounds to
    /// a whole number of cells rather than an even one.
    pub fn lattice(self) -> Lattice {
        match self {
            NoiseSpec::Cellular { lattice, .. } => lattice,
            NoiseSpec::Grid | NoiseSpec::RadialGrid { .. } => Lattice::Cubic,
        }
    }

    /// Ignored by `Grid`, whose return is `1 − d1²/d2²` and nothing else.
    pub fn return_type(self) -> CellReturn {
        match self {
            NoiseSpec::Cellular { return_type, .. } => return_type,
            NoiseSpec::Grid | NoiseSpec::RadialGrid { .. } => CellReturn::default(),
        }
    }

    /// `Grid` answers `0` — the value that *defines* it, and the one a panel
    /// reading this field should show.
    pub fn jitter(self) -> f32 {
        match self {
            NoiseSpec::Cellular { jitter, .. } => jitter,
            NoiseSpec::Grid | NoiseSpec::RadialGrid { .. } => 0.0,
        }
    }

    /// Wedges around one turn at octave 0. `0` for every kind that has no axis
    /// — the value the shader is handed and ignores.
    pub fn sectors(self) -> f32 {
        match self {
            NoiseSpec::RadialGrid { sectors } => sectors as f32,
            NoiseSpec::Cellular { .. } | NoiseSpec::Grid => 0.0,
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
    ///
    /// A negative value inverts the relief — cell centres read as basins
    /// instead of domes — because `strength` is a factor of `w` in
    /// [`sample_at`](TextureSpec::sample_at)'s per-octave weight, and that
    /// weight scales `dndx`/`dndy` linearly. No separate "invert" flag is
    /// needed for the rare material that wants the sunk look; the sign of
    /// this field already is that flag.
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
    /// How hard a ridge suppresses the octave above it. Read by both ridges and
    /// ignored by [`Fractal::Fbm`] — and it means a **different thing** to each
    /// of the two, so it does not survive a change of `fractal`: a replacing
    /// gain to [`Fractal::Ridged`], a lerp factor to [`Fractal::RidgedFnl`].
    /// Those variants' own docs work the difference out.
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
    /// **Authoring note.** This one knob is pulled by two forces:
    ///
    /// * **Quantization.** `frequency / world_scale` is the cell span across
    ///   the tile and wants to land near an even integer, or the whole
    ///   lacunarity chain rounds (see
    ///   [`lacunarity_error`](TextureSpec::lacunarity_error)). Bigger tiles
    ///   quantize more cheaply, because rounding to an even number is a smaller
    ///   *relative* step the more cells there are.
    /// * **Texel density.** A tile is one bake, so a bigger tile spreads the
    ///   same texels over more metres (see
    ///   [`texel_density`](TextureSpec::texel_density)). Past a point the
    ///   surface is simply blurred up close and no sampler can undo it.
    ///
    /// They pull in opposite directions and density wins, because quantization
    /// error is a few percent on cell sizes nobody measures while density is
    /// the sharpness of every pixel. Repetition — the third cost of a small
    /// tile — is the one with a real fix: [`anti_tiling`].
    ///
    /// It does *not* change feature size — a cell is `1 / frequency` world
    /// units across whatever this says, up to the quantization above.
    ///
    /// It *does* change the normal map's amplitude, which is the one
    /// non-obvious coupling in this struct: see
    /// [`NORMAL_REFERENCE_TEXELS`]. Retiling a material means scaling
    /// [`NormalSpec::strength`] inversely with the base span, or the surface
    /// comes out flatter.
    ///
    /// [`anti_tiling`]: TextureSpec::anti_tiling
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
    /// The per-material override of [`LIVE_LOD_CELL_PIXELS`]: how many pixels
    /// wide a lattice cell must stay before §7's **live** path stops
    /// evaluating its octave. Only the live path reads this — the bake has no
    /// camera and always passes [`OCTAVE_LOD_OFF`].
    ///
    /// `#[serde(default)]` at the constant's own value, so every scene file and
    /// cached bake written before this field existed keeps today's fade
    /// exactly. Raise it for a material whose detail is coarse and slow to
    /// change (a cliff face can give up its finest octave at a smaller
    /// footprint than grass can without anyone noticing); lower it to hold
    /// detail closer to the camera at the cost of the shimmer that is the
    /// whole reason the window exists.
    ///
    /// # It is folded into `content_key` like everything else here
    ///
    /// It cannot change one baked pixel — [`bake::BakeUniform`](crate::bake::BakeUniform)
    /// never reads it — but [`content_key`](TextureSpec::content_key) still
    /// hashes it, on purpose. The handle is not only the pixel cache's key; it
    /// is also [`TextureRegistry::resolve`](crate::bake::TextureRegistry::resolve)'s
    /// key for the *uniform* two materials share, and `world_scale` and
    /// `triplanar_sharpness` already sit in that same boat — neither reaches
    /// [`BakeUniform`](crate::bake::BakeUniform) either, both still mint a new
    /// handle when they change. Two specs that differed only in
    /// `live_cell_pixels` but shared a handle would share one resident
    /// [`TextureUniform`](crate::bake::TextureUniform): whichever spec resolved
    /// the handle first would silently pick the fade the *other* material
    /// renders with. Folding it into identity costs a rebake — never a wrong
    /// pixel — the first time an old cache is read with this field in the
    /// struct, because postcard encodes every field positionally regardless of
    /// its default; that rebake produces byte-identical pixels under a new key,
    /// which is what makes a stale on-disk entry merely wasted space rather
    /// than a correctness bug.
    #[serde(default = "default_live_cell_pixels")]
    pub live_cell_pixels: f32,
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
fn default_live_cell_pixels() -> f32 {
    LIVE_LOD_CELL_PIXELS
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
            live_cell_pixels: default_live_cell_pixels(),
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
    ///
    /// The top bit is cleared on the way out, which is the one place the
    /// content-addressed half of the handle space is *defined* to be 63 bits
    /// wide. Everything above it belongs to the engine (see
    /// [`TextureHandle::RESERVED_BIT`]) — an offscreen scene target has to be
    /// nameable by a `TextureHandle` too, and "improbable" is not the same
    /// claim as "impossible". Losing one bit of a 64-bit hash costs nothing
    /// measurable: the birthday bound over a scene's few hundred textures is
    /// still astronomically far from 2⁶³.
    pub fn content_key(&self, resolution: u32) -> u64 {
        let key = crate::gen::fnv(self.param_key(), &resolution.to_le_bytes());
        key & !TextureHandle::RESERVED_BIT
    }

    // -- resolution ---------------------------------------------------------

    /// How many world metres one tile spans.
    pub fn tile_meters(&self) -> f32 {
        1.0 / self.world_scale.max(1e-6)
    }

    /// Baked texels per world metre at `resolution`.
    ///
    /// The number that decides whether a surface reads sharp or smeared when
    /// the camera is close to it: a fragment covering less than one texel is
    /// magnifying the bake, and no amount of resolution *ceiling* helps if the
    /// tile is authored large enough to spend that resolution on empty metres.
    /// Roughly 25 px/m is where a surface stops looking soft at arm's length in
    /// this engine's framing; below ~10 it is visibly blurred.
    ///
    /// Mipmaps fix the *other* end (minification shimmer) and are free of this
    /// — but they also make it safe to author a small tile, because the
    /// repetition a small tile brings is what the anti-tiling sampler is for.
    pub fn texel_density(&self, resolution: u32) -> f32 {
        resolution as f32 * self.world_scale
    }

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

    /// Octave-0 lattice cells per world metre — §7's live path's entire
    /// world → noise map, and the `config.w` of
    /// [`TextureUniform`](crate::bake::TextureUniform).
    ///
    /// It is the bake's map with the tile divided out. The bake evaluates
    /// octave *i* at `uv · span_i`, and `uv = world · world_scale`, so a live
    /// fragment at `world · (span_0 · world_scale) · freq_i` is sampling the
    /// *same field* — which is why the two modes can be A/B'd at all, and why
    /// `tests/live_texture.rs` can hold the live shader against
    /// [`sample_at`](TextureSpec::sample_at) directly.
    pub fn live_cells_per_metre(&self) -> f32 {
        self.octave_plan().first().map(|o| o.span).unwrap_or(1.0) * self.world_scale
    }

    /// The seed offset §7's **live** path samples at: the bake's, folded into
    /// the tile.
    ///
    /// # Why it is not just the bake's offset
    ///
    /// [`seed_offset_3d`](crate::noise::seed_offset_3d) hands out offsets in
    /// the *thousands*, and the finest octave multiplies them by its frequency
    /// — `grass`'s octave 4 lands around 25 000 cells out. The bake survives
    /// that because it wraps every cell index back into the tile before
    /// hashing; live has no wrap by definition, so it would be taking a `floor`
    /// at a magnitude where `f32` resolves about 0.002 of a cell. That is
    /// precisely the failure `wrap_axis` was hardened against (see
    /// `noise.wgsl`), one step earlier in the pipeline and with no wrap to
    /// absorb it. Folding puts the live evaluation back where floats are dense.
    ///
    /// The second benefit is that it makes the two modes the *same picture*
    /// rather than two neighbourhoods of one field — see
    /// [`live_agreement_window`](TextureSpec::live_agreement_window), which
    /// bounds where that holds and is exact inside it.
    ///
    /// # Why folding is allowed
    ///
    /// Octave *i* sees the offset as `offset · freq_i` and wraps modulo
    /// `span_i = span_0 · freq_i`, so shifting the offset by a whole `span_0`
    /// moves every octave by a whole period at once: wrapped cell keys
    /// unchanged, feature-point *offsets* unchanged, baked image unchanged. On
    /// FCC the shift is a multiple of an even number, so lattice parity
    /// survives it too. Which window of an unbounded field the world maps onto
    /// is free; this picks the one the bake was already showing.
    ///
    /// Only x and y fold. The bake gives z no period at all (a 2D tile through
    /// a 3D field), so there is no wrap to be equivalent to and folding z would
    /// genuinely move the slice.
    ///
    /// The result is centred on zero rather than left in `[0, span_0)`, which
    /// keeps the magnitudes smallest and the agreement window widest.
    pub fn live_seed_offset(&self) -> Vec3 {
        let offset = self.seed_displacement();
        let span = self.octave_plan().first().map(|o| o.span).unwrap_or(1.0);
        let fold = |v: f32| {
            let r = v.rem_euclid(span);
            if r >= span * 0.5 {
                r - span
            } else {
                r
            }
        };
        Vec3::new(fold(offset.x), fold(offset.y), offset.z)
    }

    /// The seed displacement this spec's noise may actually be given.
    ///
    /// [`seed_offset_3d`](crate::noise::seed_offset_3d) for every kind that is
    /// translation-invariant, and **axis-only** for a kind that is not — see
    /// [`NoiseKind::has_axis`], which carries the argument and the bug that
    /// prompted it. For [`NoiseSpec::RadialGrid`] that means the seed slides the
    /// bands along `Y` and leaves the wedges' centre where the object is.
    ///
    /// Every place the offset enters — the bake uniform, [`sample_at`], and
    /// [`live_seed_offset`] — goes through here, so a kind cannot pick up a
    /// displacement on one path and not another.
    ///
    /// [`sample_at`]: TextureSpec::sample_at
    /// [`live_seed_offset`]: TextureSpec::live_seed_offset
    /// [`NoiseKind::has_axis`]: crate::noise::NoiseKind::has_axis
    pub fn seed_displacement(&self) -> Vec3 {
        let offset = noise::seed_offset_3d(self.seed_offset);
        if self.noise.kind().has_axis() {
            Vec3::new(0.0, offset.y, 0.0)
        } else {
            offset
        }
    }

    /// The tile-space window over which the live field and the baked tile are
    /// the *same* pixels, as `(min, max)` in uv.
    ///
    /// The fold above lines the two up; this says where the alignment holds.
    /// The baked wrap is the identity only while a cell index stays inside
    /// `[0, span_i)`, which — because the fold is by `span_0` and every octave
    /// scales together — is one condition for all octaves:
    /// `uv + offset/span_0 ∈ [0, 1)`. Outside it the bake repeats and live does
    /// not, which is the whole difference between the two modes and not a bug
    /// in either.
    ///
    /// The window is a full tile wide before margin, shifted by at most half a
    /// tile, so at least half of any tile is always inside it.
    pub fn live_agreement_window(&self) -> (Vec2, Vec2) {
        let span = self.octave_plan().first().map(|o| o.span).unwrap_or(1.0);
        let offset = self.live_seed_offset();
        // The lattice neighbourhood reaches two cells out (`FCC_OFFSETS`), so a
        // sample within two cells of the tile edge has neighbours the bake
        // wraps and live does not. Two cells is `2/span_0` of the tile at the
        // *coarsest* octave, which is the binding one — every finer octave's
        // two cells are a smaller fraction. A base octave holding four cells or
        // fewer therefore has no window at all, which is a real statement about
        // that material and not a limitation here.
        let margin = 2.0 / span;
        // Deliberately not clamped to `[0, 1]`: the baked tile is exactly
        // periodic, so a uv outside the unit square is a legitimate way to name
        // a point in it, and clamping would throw away most of the window
        // whenever the folded offset is not near zero.
        let lo = |o: f32| margin - o / span;
        let hi = |o: f32| 1.0 - margin - o / span;
        (
            Vec2::new(lo(offset.x), lo(offset.y)),
            Vec2::new(hi(offset.x), hi(offset.y)),
        )
    }

    /// `log2` of the *quantized* lacunarity — the average octave-to-octave
    /// frequency step the plan actually uses, not the authored number.
    ///
    /// Quantized, because that is what the shader is sampling: rounding each
    /// span to a whole (even, on FCC) number bends the chain, and the live
    /// octave window has to place its cutoff on the real one or it fades the
    /// wrong octave. Single-octave specs report the authored value, since a
    /// chain of one has no step to measure.
    pub fn log2_lacunarity(&self) -> f32 {
        let plan = self.octave_plan();
        match plan.len() {
            0 | 1 => self.lacunarity.max(1.0 + 1e-3).log2(),
            n => (plan[n - 1].freq.max(1e-6).log2() / (n - 1) as f32).max(1e-3),
        }
    }

    /// The octave window a live fragment covering `footprint` octave-0 cells
    /// should evaluate — the CPU twin of `noise.wgsl`'s `live_octave_window`.
    ///
    /// Returns `(min, max)` for [`noise::octave_weight`]; `cell_pixels <= 0`
    /// gives [`OCTAVE_LOD_OFF`](crate::noise::OCTAVE_LOD_OFF).
    ///
    /// # `top` is floored at `1.0`
    ///
    /// Unfloored, `top` keeps falling as the footprint grows, and once it
    /// reaches `0` [`octave_weight(0, top - 1, top)`](noise::octave_weight)
    /// reads `1 - clamp(1 - top, 0, 1)`, which is `0` for every `top <= 0` —
    /// octave 0 fades along with everything above it. `FbmAccum`'s normalizing
    /// sum is then zero over zero octaves, `finish` returns its `0.0` fallback,
    /// and the surface does not go blurry, it goes to a single flat value: the
    /// live path was deleting the material rather than reducing its detail.
    /// Flooring `top` at `1.0` pins the window to `(0.0, 1.0)` for any
    /// footprint that would otherwise have pushed it lower, which makes
    /// `octave_weight(0, 0, 1) = 1 - clamp(1, 0, 1) = 1 - 1 + 1 = 1` — full
    /// weight, unconditionally. (`1 - clamp((0-0)/1, 0, 1) = 1 - 0 = 1`, to
    /// spell out the arithmetic the doc-test below pins.) Octave 1 at that same
    /// window is `1 - clamp((1-0)/1, 0, 1) = 1 - 1 = 0`, so the floor costs
    /// nothing above octave 0 — every octave past it still fades to zero
    /// exactly as before, just never quite to a blank surface.
    pub fn live_octave_window(&self, footprint: f32, cell_pixels: f32) -> (f32, f32) {
        if cell_pixels <= 0.0 {
            return OCTAVE_LOD_OFF;
        }
        let top = (-(footprint * cell_pixels).max(1e-8).log2() / self.log2_lacunarity()).max(1.0);
        (top - 1.0, top)
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
        let offset = self.seed_displacement();
        let normal = self.normal.unwrap_or_default();
        let has_normal = self.normal.is_some();
        let kind = self.noise.kind();
        let lattice = self.noise.lattice();
        let ret = self.noise.return_type();
        let jitter = self.noise.jitter();
        let sectors = self.noise.sectors();

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
            let cell = noise::field(p, kind, lattice, ret, jitter, sectors * octave.freq, period);

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
                // `delta` points *away* from the nearest feature point — ToPoint
                // is literally `sample - f1`, and ToEdge's `f2 - f1` is the far
                // side of the boundary. But `dndx`/`dndy` are consumed below as
                // a height gradient (`packed_normal_at`'s
                // `normalize(-dndx, -dndy, 0.5)`), under the convention that a
                // cell *centre* is the relief's high point — a tile stands
                // proud, its grout recedes, DESIGN's rock and grass materials
                // both read that way. Uphill is therefore toward the feature
                // point, the opposite of `delta`, so the direction is negated
                // once here rather than at the two `+=` sites below (which
                // would need the same flip twice and be one place easier to get
                // half-right).
                let dir = if len > 1e-4 { -delta / len } else { Vec3::ZERO };
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

    /// The **live** field at a world-space point: unbounded 3D noise, no tile,
    /// no wrap, every octave at full weight (DESIGN §7's live path).
    ///
    /// The CPU twin of `shader.wgsl`'s `live_sample`, minus the boundary
    /// gradient — that one differentiates against a pixel footprint and there
    /// is no pixel here. Returns the value **before** contrast/brightness, like
    /// [`sample_at`](TextureSpec::sample_at).
    ///
    /// Held against the shader by `tests/live_texture.rs`, and against the
    /// *bake* by `the_live_field_and_the_baked_tile_are_one_field` below — the
    /// second is the load-bearing one, because a live path that quietly sampled
    /// a different field would still look fine on its own.
    pub fn live_value_at(&self, world: Vec3) -> f32 {
        let q = world * self.live_cells_per_metre() + self.live_seed_offset();
        let kind = self.noise.kind();
        let lattice = self.noise.lattice();
        let ret = self.noise.return_type();
        let jitter = self.noise.jitter();
        let sectors = self.noise.sectors();

        let mut accum = FbmAccum::new();
        for octave in &self.octave_plan() {
            let cell = noise::field(
                q * octave.freq,
                kind,
                lattice,
                ret,
                jitter,
                sectors * octave.freq,
                Vec3::ZERO,
            );
            accum.push(
                cell.value,
                octave.amplitude,
                octave.weight,
                self.fractal,
                self.weighted_strength,
            );
        }
        accum.finish()
    }

    /// The live albedo at a world-space point: [`live_value_at`] through
    /// contrast/brightness and the ramp.
    ///
    /// [`live_value_at`]: TextureSpec::live_value_at
    pub fn live_albedo_at(&self, world: Vec3) -> Vec3 {
        self.ramp_at(self.postprocess(self.live_value_at(world)))
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
///
/// # Two halves of the handle space
///
/// Content addressing answers "is this the same pixels?" for anything a
/// *generator* produced. It cannot answer it for pixels that are produced by
/// rendering — an offscreen scene target's contents change every frame, so its
/// identity has to be a name rather than a hash. Those live in the top half:
///
/// - **bit 63 clear** — content-addressed. `content_key` masks the bit off, so
///   this is a definition rather than a hope.
/// - **bit 63 set** — engine-allocated ([`RESERVED_BIT`]). Today the only
///   inhabitant is [`render_target`], the colour texture of a
///   [`RenderTarget`](crate::RenderTarget); bits 32–62 are free for whatever
///   kind comes next.
///
/// A game choosing its own handle for a
/// [`UiAtlasImage`](crate::ui::UiAtlasImage) must therefore leave bit 63 alone.
/// That is the one rule this split imposes on the outside world, and it is the
/// price of two worlds' textures being safe to hold in one registry.
///
/// [`RESERVED_BIT`]: TextureHandle::RESERVED_BIT
/// [`render_target`]: TextureHandle::render_target
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextureHandle(pub u64);

impl TextureHandle {
    /// The bit that separates engine-allocated handles from content hashes —
    /// see the type's docs.
    pub const RESERVED_BIT: u64 = 1 << 63;

    /// The handle an offscreen scene target with this id is registered under.
    ///
    /// Stable and pure: the same id always names the same handle, on every
    /// run and in both worlds, so a game can write it down at build time.
    pub const fn render_target(id: u32) -> TextureHandle {
        TextureHandle(TextureHandle::RESERVED_BIT | id as u64)
    }

    /// Whether this handle is the engine's rather than some content's. Nothing
    /// a bake or a `TextureLibrary` produces can answer `true`.
    pub const fn is_reserved(self) -> bool {
        self.0 & TextureHandle::RESERVED_BIT != 0
    }
}

/// Handle → the spec (and resolution) that produced it, as a world resource.
///
/// GPU-free, exactly like [`MeshLibrary`](crate::registry::MeshLibrary): the
/// scene loader fills it, and the renderer bakes from it lazily the first time
/// a draw actually asks for the texture. Scene load therefore stays a
/// device-free operation and the sim's tests never need an adapter.
/// It also carries §7's live/baked switch, because the draw-list builder needs
/// it and this is the one texture-shaped resource already in the world. That is
/// a render-side flag living on a content resource, which is a small
/// impurity — but the alternative is a second resource whose entire content is
/// one bool that only ever changes with this one.
#[derive(Resource, Default, Clone, Debug)]
pub struct TextureLibrary {
    entries: Vec<(TextureHandle, TextureSpec, u32)>,
    live: bool,
}

impl TextureLibrary {
    pub fn new() -> TextureLibrary {
        TextureLibrary::default()
    }

    /// Whether textured draws evaluate their spec per pixel (§7's live path)
    /// instead of sampling its bake. Default `false` — baked is the baseline.
    pub fn live_textures(&self) -> bool {
        self.live
    }

    /// Switch every textured draw between §7's two modes.
    ///
    /// **This is v1's gate.** DESIGN §11 files live eval under the *perf* tier
    /// and there is no perf probe yet, so what ships is the manual override the
    /// probe would eventually drive: off by default, on when a host says so.
    /// When the probe lands it sets this at startup and the API does not move.
    ///
    /// It is a per-frame decision and costs nothing to flip: the bind group is
    /// the same either way ([`TextureUniform`](crate::bake::TextureUniform)
    /// carries both modes' data), so only the pipeline changes — and both
    /// pipelines are already in the variant cache after one frame of each.
    ///
    /// It changes **no simulation state**. Live evaluation happens in the
    /// fragment shader; nothing in `FixedSim` can observe it, so a determinism
    /// fingerprint cannot move when it flips.
    ///
    /// # It does not skip the bake
    ///
    /// A live draw still resolves its [`TextureHandle`] through the renderer's
    /// texture registry, because that is what owns the `@group(2)` uniform the
    /// live path reads its spec out of — so the bake still runs at load, and its
    /// pixels go unread. That is deliberate rather than pending: the flag is a
    /// per-frame decision, and a mode you can turn off has to have something to
    /// turn off *to*. An app that knew it would never bake could skip the two
    /// render passes and keep the uniform, but it would give up the toggle, and
    /// the toggle is the whole of v1's gate.
    pub fn set_live_textures(&mut self, live: bool) {
        self.live = live;
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

    /// Forget `handle`, so nothing can resolve it any more. `true` if it was
    /// there.
    ///
    /// # What this is for, and what it deliberately is not
    ///
    /// A content-addressed entry is never *stale* — the pixels a handle names
    /// are the pixels that spec bakes, this session and every other. So the
    /// reason to remove one is never that it went wrong; it is that **nothing
    /// references it any more**, which is a question only the world can answer.
    ///
    /// The case that needs it is live authoring. Editing a
    /// [`TextureSpec`](crate::texture::TextureSpec) cannot mutate an entry in
    /// place — the handle *is* the params, so an edit is a new handle and the
    /// old one is left behind. A designer dragging one slider through fifty
    /// values would therefore mint fifty entries and fifty bakes, and both this
    /// library and the renderer's registry would hold every one of them for the
    /// rest of the session. Removing the superseded entry is how that stops
    /// being a leak.
    ///
    /// It does **not** free any GPU memory on its own. The registry is a cache
    /// of this resource, so the pixels go when the two are reconciled —
    /// [`Engine::sweep_baked_textures`], which is the call that makes this one
    /// worth making.
    ///
    /// [`Engine::sweep_baked_textures`]: crate::Engine::sweep_baked_textures
    pub fn remove(&mut self, handle: TextureHandle) -> bool {
        let before = self.entries.len();
        self.entries.retain(|(h, _, _)| *h != handle);
        self.entries.len() != before
    }

    /// Keep only the entries `keep` answers `true` for; returns how many went.
    ///
    /// The bulk form of [`remove`](TextureLibrary::remove), and the shape a
    /// reconcile against the world wants: gather the handles the live
    /// [`Material`](crate::Material)s name, then retain the ones in that set.
    /// Order is preserved among the survivors, which matters for the same
    /// reason [`insert`](TextureLibrary::insert) keeps a `Vec` — nothing near
    /// content resolution may depend on a hash container's iteration order.
    pub fn retain(
        &mut self,
        mut keep: impl FnMut(TextureHandle, &TextureSpec, u32) -> bool,
    ) -> usize {
        let before = self.entries.len();
        self.entries
            .retain(|(handle, spec, resolution)| keep(*handle, spec, *resolution));
        before - self.entries.len()
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The unit tests' own spec: a cleanly quantized tile with a normal map.
    ///
    /// The same numbers `tests/common/mod.rs` calls `fine`, duplicated rather
    /// than shared because a `#[cfg(test)]` module inside the crate and an
    /// integration test cannot see one another. Named for its character, not for
    /// a material — the engine has no opinion about what grass looks like.
    // A number an artist typed, not an approximation of `e`.
    #[allow(clippy::approx_constant)]
    fn fine() -> TextureSpec {
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

    /// The awkward one: a 40 m tile holding two cells of its base feature, so
    /// the base octave rounds 8.7%. `tests/common/mod.rs`'s `coarse`.
    fn coarse() -> TextureSpec {
        TextureSpec {
            frequency: 0.046,
            octaves: 5,
            lacunarity: 3.512,
            gain: 0.543,
            normal: Some(NormalSpec {
                mode: NormalMode::ToEdge,
                edge_width: 0.351,
                strength: 29.605,
            }),
            world_scale: 0.025,
            triplanar_sharpness: 1.0,
            ..fine()
        }
    }

    #[test]
    fn the_tile_is_seamless_in_both_axes() {
        // The whole point of the wrapped-lattice bake: the field at u=0 and u=1
        // is not merely close, it is the same number.
        let spec = fine();
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
        for spec in [fine(), coarse()] {
            for octave in spec.octave_plan() {
                assert_eq!(octave.span, octave.span.round(), "span must be integral");
                assert_eq!(octave.span % 2.0, 0.0, "FCC needs an even period");
                assert!(octave.span >= 2.0);
            }
        }
    }

    #[test]
    fn lacunarity_error_reports_the_price_of_seamlessness() {
        // Whether a *particular* material's quantization is acceptable is a
        // question for whoever authored it — the game holds its own budgets
        // (`shift/src/textures.rs`). What the engine owes is that the number is
        // honest, because a budget can only be enforced against a measurement
        // that moves.
        //
        // The two fixtures bracket the range: `fine` has 6 cells across its tile
        // and rounds cheaply, `coarse` has fewer than 2 and rounds dearly.
        let cheap = fine().lacunarity_error();
        let dear = coarse().lacunarity_error();
        assert!(cheap < 0.035, "fine quantized by {:.2}%", cheap * 100.0);
        assert!(
            (0.08..0.09).contains(&dear),
            "coarse quantized by {:.2}%, expected ~8.7%",
            dear * 100.0
        );
        assert!(dear > cheap, "fewer cells across the tile must cost more");

        // A spec whose spans are already whole and even pays nothing, which is
        // the floor the measurement has to be able to report.
        let exact = TextureSpec {
            frequency: 2.0,
            world_scale: 1.0,
            lacunarity: 2.0,
            octaves: 4,
            ..fine()
        };
        assert_eq!(exact.lacunarity_error(), 0.0);
    }

    #[test]
    fn texel_density_is_resolution_over_the_tile() {
        // The arithmetic a material author tunes `world_scale` against, and the
        // regression it exists for: a 217 m tile at 1024² is 4.7 texels per
        // metre — one texel every 21 cm — which is the blur that got a whole
        // material re-tiled. The *threshold* is the game's to set; that the
        // number is this ratio is the engine's to guarantee.
        let spec = TextureSpec {
            world_scale: 0.025,
            ..fine()
        };
        assert_eq!(spec.tile_meters(), 40.0);
        assert_eq!(spec.texel_density(1024), 25.6);
        // Ten times the tile is a tenth the density, which is what makes the
        // trade legible when it is written down.
        let wide = TextureSpec {
            world_scale: 0.0025,
            ..spec
        };
        assert_eq!(wide.tile_meters(), 400.0);
        assert_eq!(wide.texel_density(1024), 2.56);
    }

    #[test]
    fn param_key_ignores_resolution_but_content_key_does_not() {
        let spec = fine();
        assert_eq!(spec.param_key(), fine().param_key());
        assert_ne!(spec.content_key(512), spec.content_key(1024));
        assert_eq!(spec.content_key(512), fine().content_key(512));

        let mut other = fine();
        other.seed_offset = 4.0;
        assert_ne!(spec.param_key(), other.param_key());
    }

    #[test]
    fn the_reserved_half_of_the_handle_space_is_unreachable_by_content() {
        // Not "unlikely": the mask is what makes it a proof. Swept over enough
        // specs and resolutions that a mask applied to only one branch of
        // `content_key` would show up here.
        for seed in 0..64 {
            let mut spec = fine();
            spec.seed_offset = seed as f32;
            for resolution in [16, 64, 512, 1024] {
                let handle = TextureHandle(spec.content_key(resolution));
                assert!(
                    !handle.is_reserved(),
                    "content key {:#018x} landed in the engine's half",
                    handle.0
                );
            }
        }

        // …and the engine's own names are all on the other side of the line,
        // whatever id they carry.
        assert!(TextureHandle::render_target(0).is_reserved());
        assert!(TextureHandle::render_target(u32::MAX).is_reserved());
        assert_ne!(
            TextureHandle::render_target(0),
            TextureHandle::render_target(1)
        );
    }

    #[test]
    fn octave_zero_never_fades_no_matter_how_large_the_footprint() {
        // The bug this pins: before the `top.max(1.0)` floor, a footprint large
        // enough pushed `top` to `0` or below, and `octave_weight(0, top - 1,
        // top)` is `0` there — every octave, including the first, dropped out
        // of `FbmAccum`'s normalizing sum, and the material collapsed to a flat
        // colour instead of merely losing detail. The floor exists so octave 0
        // is the one thing the live path never takes away.
        let spec = fine();
        for footprint in [
            1e-6, 0.01, 0.5, 1.0, 5.0, 100.0, 1.0e6, 1.0e12, f32::MAX / 4.0,
        ] {
            let (min, max) = spec.live_octave_window(footprint, LIVE_LOD_CELL_PIXELS);
            let w0 = noise::octave_weight(0, min, max);
            assert_eq!(
                w0, 1.0,
                "footprint {footprint}: window {min:?}..{max:?} gave octave 0 weight {w0}, not 1.0"
            );
        }

        // The floor is a *pin*, not a general amnesty: with the window still
        // exactly one octave wide, octave 1 fades to zero at the same absurd
        // footprints that would have zeroed octave 0 pre-fix.
        for footprint in [100.0, 1.0e6, 1.0e12] {
            let (min, max) = spec.live_octave_window(footprint, LIVE_LOD_CELL_PIXELS);
            let w1 = noise::octave_weight(1, min, max);
            assert_eq!(
                w1, 0.0,
                "footprint {footprint}: octave 1 should have fully faded, got weight {w1}"
            );
        }

        // And a sub-cell footprint is unaffected by the floor at all — the
        // window sits well above 1.0, exactly as before this fix.
        let (min, _) = spec.live_octave_window(1e-6, LIVE_LOD_CELL_PIXELS);
        assert!(min > 5.0, "a tiny footprint should leave every octave on, got min {min}");
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
        let spec = fine();
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

    /// Root-mean-square tilt of the packed normal over a uv grid — how steep a
    /// surface reads, as one number. Per-uv comparison is meaningless across a
    /// re-tile: `world_scale` changes which part of the field a uv lands on, so
    /// the two are the same *relief* over different neighbourhoods.
    fn rms_tilt(spec: &TextureSpec) -> f32 {
        let n = 24;
        let mut sum = 0.0;
        for y in 0..n {
            for x in 0..n {
                let uv = Vec2::new(x as f32 / n as f32, y as f32 / n as f32);
                let (_, dndx, dndy) = spec.sample_at(uv);
                sum += dndx * dndx + dndy * dndy;
            }
        }
        (sum / (n * n) as f32).sqrt()
    }

    #[test]
    fn normal_amplitude_is_strength_times_the_base_span_and_nothing_else() {
        // [`NORMAL_REFERENCE_TEXELS`]'s one non-obvious consequence, as an
        // executable statement: the bake differentiates the field over
        // `base_span / 1024`, so the packed tilt is proportional to
        // `strength × base_span` and to no other field in the struct. Re-tile a
        // material without scaling `strength` inversely and it goes flat.
        //
        // This used to be a claim about a specific ported material's number
        // (5.921 against a ten-cell tile, rescaled to 29.605 when the tile
        // shrank to two). The material went to the game and its arithmetic went
        // with it; the *coupling* is the engine's, so what is left here is the
        // coupling.
        let base = TextureSpec {
            frequency: 1.0,
            world_scale: 0.25,
            normal: Some(NormalSpec {
                mode: NormalMode::ToEdge,
                edge_width: 0.35,
                strength: 8.0,
            }),
            ..fine()
        };
        assert_eq!(base.octave_plan()[0].span, 4.0);

        // Halve the tile, double the strength: the same product, and — the
        // claim — the same packed normal, to floating-point noise.
        let retiled = TextureSpec {
            world_scale: 0.5,
            normal: Some(NormalSpec {
                strength: 16.0,
                ..base.normal.expect("a normal")
            }),
            ..base.clone()
        };
        assert_eq!(retiled.octave_plan()[0].span, 2.0);

        // Same product, same relief: 0.108 against 0.106, which is two
        // neighbourhoods of one field rather than two amplitudes.
        let (a, b) = (rms_tilt(&base), rms_tilt(&retiled));
        assert!(
            (a - b).abs() / a < 0.05,
            "compensated retiling changed the relief: {a:.5} -> {b:.5}"
        );

        // …and forgetting to compensate does not. Half the span at the same
        // strength is half the relief — exactly, because `step` is the only
        // place the span reaches the gradient — which is the failure the
        // coupling exists to make findable: the albedo gets sharper, the surface
        // goes flat, and the two changes look unrelated.
        let flattened = TextureSpec {
            world_scale: 0.5,
            ..base.clone()
        };
        let f = rms_tilt(&flattened);
        assert!(
            (f - b * 0.5).abs() / f < 0.05,
            "half the tile at the same strength should be half the relief: \
             {f:.5} against {:.5}",
            b * 0.5
        );
    }

    #[test]
    fn the_library_dedups_by_content_key() {
        let mut library = TextureLibrary::new();
        let a = library.insert(fine(), 512);
        let b = library.insert(fine(), 512);
        let c = library.insert(fine(), 1024);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(library.len(), 2);
        assert!(library.contains(a));
        assert_eq!(library.get(a).expect("registered").1, 512);
    }

    #[test]
    fn the_library_forgets_what_a_spec_edit_superseded() {
        // The live-authoring lifecycle: a spec is edited, so the old handle is
        // not stale — it is unreferenced, and nothing but the world knows that.
        let mut library = TextureLibrary::new();
        let before = library.insert(fine(), 512);
        let edited = TextureSpec {
            contrast: 1.4,
            ..fine()
        };
        let after = library.insert(edited, 512);
        assert_ne!(before, after, "an edited spec is a different handle");
        assert_eq!(library.len(), 2, "the edit left the old entry behind");

        assert!(library.remove(before));
        assert!(!library.contains(before));
        assert!(library.contains(after), "the edit's own entry survived");
        assert_eq!(library.len(), 1);
        assert!(!library.remove(before), "removing twice is not an error");
    }

    #[test]
    fn retain_keeps_the_survivors_in_order() {
        // A reconcile is a bulk remove, and the order it leaves behind is the
        // insertion order of whatever it kept — same reason `insert` is a `Vec`.
        let mut library = TextureLibrary::new();
        let handles: Vec<TextureHandle> = (0..5)
            .map(|i| {
                library.insert(
                    TextureSpec {
                        seed_offset: i as f32,
                        ..fine()
                    },
                    512,
                )
            })
            .collect();

        let keep: Vec<TextureHandle> = vec![handles[0], handles[3], handles[4]];
        let dropped = library.retain(|handle, _, _| keep.contains(&handle));
        assert_eq!(dropped, 2);
        assert_eq!(
            library.iter().map(|(h, _, _)| h).collect::<Vec<_>>(),
            keep,
            "retain reordered the survivors"
        );

        // A retain that keeps everything reports nothing and moves nothing.
        assert_eq!(library.retain(|_, _, _| true), 0);
        assert_eq!(library.len(), 3);
        // …and one that keeps nothing empties it rather than half-emptying it.
        assert_eq!(library.retain(|_, _, _| false), 3);
        assert!(library.is_empty());
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
            ..fine()
        };
        let packed = spec.packed_normal_at(Vec2::new(0.3, 0.7));
        assert!(packed.abs_diff_eq(Vec3::new(0.5, 0.5, 1.0), 1e-5), "{packed:?}");
    }

    #[test]
    fn a_cell_reads_as_a_dome_and_not_a_bowl() {
        // The regression the negation in `sample_at`'s `dir` guards against:
        // before it, `dir` pointed *away* from the nearest feature point, so
        // `packed_normal_at`'s `normalize(-dndx, -dndy, 0.5)` tilted toward
        // cell centres — every cell a basin, its boundary a ridge, which read
        // as grout standing proud of the tile it grouts. The fix makes a
        // cell's own feature point its high point, so the unpacked normal's
        // (x, y) has to lean *away* from that point everywhere off it, and
        // lean harder the closer the sample gets to the boundary — a field
        // of domes, not bowls.
        //
        // One octave, so there is exactly one feature point in play near any
        // sample and nothing else's contribution to argue the sign with.
        let spec = TextureSpec {
            octaves: 1,
            ..fine()
        };
        let plan = spec.octave_plan();
        assert_eq!(plan.len(), 1, "the claim wants exactly one feature point in play");
        let span = plan[0].span;
        assert_eq!(plan[0].freq, 1.0, "octave 0 is always its own reference frequency");
        let offset = spec.seed_displacement();
        let lattice = spec.noise.lattice();
        let ret = spec.noise.return_type();
        let jitter = spec.noise.jitter();
        let period = Vec3::new(span, span, 0.0);

        // `sample_at`'s own `p(uv)` for a single octave (freq 1, so
        // `offset * freq` is just `offset`) — see its `let p = ...`.
        let sample_p =
            |uv: Vec2| Vec3::new(uv.x * span + offset.x, uv.y * span + offset.y, offset.z);

        // Any tile point names a cell; walk from its centre toward the
        // second-nearest point until `d2 - d1` bottoms out, which is the
        // Voronoi boundary by definition. Scanning rather than assuming the
        // midpoint, because a jittered feature point is not symmetric around
        // it — `fine()` runs jitter at 1.0.
        let home = noise::cellular(sample_p(Vec2::new(0.5, 0.5)), lattice, ret, jitter, period);
        let centre_uv = Vec2::new((home.f1.x - offset.x) / span, (home.f1.y - offset.y) / span);
        let dir_xy = Vec2::new(home.f2.x - home.f1.x, home.f2.y - home.f1.y);
        let dir_n = dir_xy.normalize();
        let reach = dir_xy.length() * 1.5;

        let steps = 300;
        let (mut boundary_t, mut narrowest) = (0.0f32, f32::INFINITY);
        for i in 1..steps {
            let t = reach * i as f32 / steps as f32;
            let uv = centre_uv + dir_n * (t / span);
            let cs = noise::cellular(sample_p(uv), lattice, ret, jitter, period);
            let gap = cs.d2 - cs.d1;
            if gap < narrowest {
                narrowest = gap;
                boundary_t = t;
            }
        }
        assert!(
            narrowest < 0.05,
            "no boundary found within {reach} units of the centre: closest gap was {narrowest}"
        );

        // 5% and 90% of the way from centre to boundary: solidly interior,
        // and hugging the edge without crossing it.
        let near_centre = centre_uv + dir_n * (0.05 * boundary_t / span);
        let near_boundary = centre_uv + dir_n * (0.9 * boundary_t / span);

        // Both probes have to still belong to `home` — the same nearest
        // point — or the comparison below is two different cells' relief
        // rather than one cell's centre against its own edge.
        let still_home = |uv: Vec2| {
            noise::cellular(sample_p(uv), lattice, ret, jitter, period)
                .f1
                .abs_diff_eq(home.f1, 1e-4)
        };
        assert!(still_home(near_centre), "{near_centre:?} crossed into the next cell");
        assert!(still_home(near_boundary), "{near_boundary:?} crossed into the next cell");

        let tilt = |uv: Vec2| {
            let packed = spec.packed_normal_at(uv);
            Vec2::new(packed.x * 2.0 - 1.0, packed.y * 2.0 - 1.0)
        };
        let (centre_lean, boundary_lean) = (tilt(near_centre).dot(dir_n), tilt(near_boundary).dot(dir_n));

        // The claim: the normal leans *outward* — toward `dir_n`, the
        // direction from the feature point to the sample — both deep inside
        // the cell and hugging its boundary, and leans harder the closer it
        // gets. The pre-fix bug would have failed the sign check outright,
        // at either point.
        assert!(centre_lean > 0.0, "near the centre the normal should already lean outward: {centre_lean:.4}");
        assert!(boundary_lean > 0.0, "near the boundary the normal should lean outward too: {boundary_lean:.4}");
        assert!(
            boundary_lean > centre_lean,
            "the outward lean should grow approaching the boundary: {centre_lean:.4} at centre vs {boundary_lean:.4} near the edge"
        );
    }
}
