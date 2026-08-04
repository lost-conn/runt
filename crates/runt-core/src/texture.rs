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
pub const LIVE_LOD_CELL_PIXELS: f32 = 2.0;

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
        let offset = noise::seed_offset_3d(self.seed_offset);
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
    pub fn live_octave_window(&self, footprint: f32, cell_pixels: f32) -> (f32, f32) {
        if cell_pixels <= 0.0 {
            return OCTAVE_LOD_OFF;
        }
        let top = -(footprint * cell_pixels).max(1e-8).log2() / self.log2_lacunarity();
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
        let lattice = self.noise.lattice();
        let ret = self.noise.return_type();
        let jitter = self.noise.jitter();

        let mut accum = FbmAccum::new();
        for octave in &self.octave_plan() {
            let cell = noise::cellular(q * octave.freq, lattice, ret, jitter, Vec3::ZERO);
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextureHandle(pub u64);

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
        // 27.8 m tile, 5.83 → 6 cells across it: a 2.9% frequency nudge, and
        // 36.9 texels per metre at 1024² — comfortably sharp, so this one never
        // needed the retune `rock` did.
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
            // The Godot material says 5.921. This is `5.921 × 10 / 2` — the
            // exact compensation for the tile shrinking below, not a restyling.
            //
            // The bake differentiates the field over `base_span / 1024` cells
            // (see NORMAL_REFERENCE_TEXELS), so the packed normal's tilt is
            // proportional to `strength × base_span` and to nothing else in
            // this struct. The base span went 10 → 2 with the retune, so
            // leaving 5.921 alone would have shipped a rock five times flatter
            // — which is precisely what the first screenshot of the retune
            // showed: a sharp texture on a surface with no relief left.
            // 29.605 × 2 = 59.21 = 5.921 × 10, so the normal map the shader
            // samples is the one Godot's number asked for.
            strength: 29.605,
        }),
        // A 40 m tile: 1024² of bake over 40 m is **25.6 texels per metre**.
        //
        // This used to be `0.0046`, chosen because it puts *exactly* ten cells
        // across the tile and quantizes the whole lacunarity chain for free.
        // That was optimizing the wrong number. Ten cells of a 21.7 m feature
        // is a 217 m tile, and 1024² spread over 217 m is 4.7 px/m — a metre of
        // rock covered by fewer than five texels, which is exactly the blur
        // that was reported. The clean quantization was bought with an eightyfold
        // deficit in the thing anyone can actually see.
        //
        // Shrinking the tile 5.4× costs quantization instead, and the trade is
        // lopsided in the other direction (see `lacunarity_error`): the ideal
        // spans become 1.84 / 6.46 / 22.7 / 79.7 / 280, which round to
        // 2 / 6 / 22 / 80 / 280 — worst octave 8.7% off, against 2.5% before.
        // What that 8.7% *means* is that the coarsest cell is 20.0 m instead of
        // 21.7 m and the second is 6.7 m instead of 6.0 m; the three octaves
        // that carry the grain anyone reads as "stone" (1.82 m, 0.50 m,
        // 0.143 m) move by 3.7%, 0.2% and 0.06%. Nobody can see the first pair.
        // Everybody could see 4.7 px/m.
        //
        // The alternative was nudging `lacunarity` off the authored 3.512 to an
        // integer, which quantizes exactly from a base span of 2 — and moves
        // the *fine* octaves instead: 4.0 makes the finest grain 0.085 m
        // (under-resolved again at 25 px/m), 3.0 makes it 0.27 m (visibly
        // chunkier stone). The authored chain is kept and the tile absorbs the
        // error; the repetition a 40 m tile brings is hidden by the anti-tiling
        // sampler, which is what it is for, and the level's rock is on props
        // 2–7 m across that never show a 40 m period anyway.
        world_scale: 0.025,
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
    fn quantizing_the_lacunarity_chain_stays_within_its_budget() {
        // The documented price of seamlessness, per material — one budget for
        // both would either let `grass` drift or forbid `rock`'s retune.
        //
        // `rock` is deliberately the expensive one: its 40 m tile holds under
        // two cells of a 21.7 m base feature, so the base octave alone rounds
        // 8.7% (see `rock`'s comment for why that is the right trade against
        // 4.7 texels per metre). If *this* ever fails the tile has shrunk
        // further still, and the next lever is `frequency`, not the rounding
        // rule.
        for (name, spec, budget) in [("grass", grass(), 0.035), ("rock", rock(), 0.09)] {
            let error = spec.lacunarity_error();
            assert!(
                error < budget,
                "{name}: lacunarity quantized by {:.2}%, budget {:.1}%",
                error * 100.0,
                budget * 100.0
            );
        }
    }

    #[test]
    fn the_authored_materials_are_dense_enough_to_read_as_sharp() {
        // The regression this exists for: `rock` shipped a 217 m tile at 1024²
        // — 4.7 texels per metre, one texel every 21 cm, and a stepping stone
        // the player stands on covered by nine of them.
        for (name, spec) in [("grass", grass()), ("rock", rock())] {
            let density = spec.texel_density(spec.base_resolution);
            println!(
                "{name}: {:.1} m tile, {:.1} px/m at {}²",
                spec.tile_meters(),
                density,
                spec.base_resolution
            );
            assert!(
                density >= 25.0,
                "{name}: {density:.1} texels/m is blurred up close"
            );
            // …and the other end: a tile so small the anti-tiling sampler has
            // to work over a period the eye can take in at a glance.
            assert!(spec.tile_meters() >= 20.0, "{name}: {} m tile", spec.tile_meters());
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
        // `strength` is the one authored number that is *rescaled* rather than
        // verbatim, because the bake ties normal amplitude to the tile — see
        // `the_normal_amplitude_survived_the_density_retune`, which holds the
        // quantity Godot's 5.921 actually meant.
        assert_eq!(rn.strength, 29.605);
    }

    #[test]
    fn the_normal_amplitude_survived_the_density_retune() {
        // What the bake multiplies the boundary gradient by is
        // `strength × (base_span / NORMAL_REFERENCE_TEXELS)`, so `strength`
        // alone says nothing about how steep a material reads — the product
        // does. Godot authored 5.921 against a tile holding ten cells; any
        // retiling that keeps `strength × span` at 59.21 keeps the relief.
        //
        // Without this the coupling is invisible: the albedo gets sharper, the
        // surface goes flat, and the two changes look unrelated.
        let r = rock();
        let amplitude = r.normal.expect("rock has normals").strength * r.octave_plan()[0].span;
        assert!(
            (amplitude - 59.21).abs() < 1e-3,
            "rock's normal amplitude is {amplitude}, not the authored 59.21"
        );

        // grass never re-tiled, so its authored number stands unscaled.
        let g = grass();
        let g_amp = g.normal.expect("grass has normals").strength * g.octave_plan()[0].span;
        assert!((g_amp - 5.106 * 6.0).abs() < 1e-3, "{g_amp}");
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
