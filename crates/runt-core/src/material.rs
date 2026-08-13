//! Materials and shader variants (DESIGN §5).
//!
//! A material is two things: a small uniform block (base color now, ramp params
//! later) and a *variant key* — a bitflag set naming the features its shader
//! needs. One WGSL source covers every variant; [`variant_source`] prepends the
//! feature `const`s and the compiler folds the dead branches away.
//!
//! Adding a new look is therefore a new flag plus a branch in the shader, never
//! a new pipeline architecture. The renderer caches one pipeline per key and
//! the draw list sorts by key, so a variant costs one state change per frame at
//! worst.

use bevy_ecs::prelude::Component;
use glam::Vec4;

use crate::texture::TextureHandle;

/// The un-preprocessed shader. Not valid WGSL on its own — the feature `const`s
/// it branches on are prepended by [`variant_source`].
pub const BASE_SHADER: &str = include_str!("shader.wgsl");

/// Shader feature bits.
///
/// Everything but `RAMP` is implemented; `RAMP` is reserved so that the key
/// space (and every cache built on it) is stable when it lands.
///
/// **Bit positions are permanent.** `NORMAL_MAP` is appended at bit 4 rather
/// than slotted in beside `TEXTURE` for exactly that reason: renumbering would
/// silently re-key every pipeline cache and every scene that spells a variant
/// out. New looks append — bits 5..9 (the render-state set) went on the end for
/// the same reason, even though `TRANSPARENT` beside `RAMP` would have read
/// better.
///
/// # Render state is part of the key (DESIGN §5)
///
/// Bits 5..7 name no shader branch at all: they select the pipeline's blend
/// mode and depth state, which [`Renderer::ensure_pipeline`] derives from the
/// key instead of hardcoding. That is the same doctrine one level down — a new
/// *look* is a new key, never a new pipeline architecture — and it is why the
/// preprocessor still emits a `const` for them (see [`FLAGS`]): the flag list
/// and the bit set must agree, whether or not the WGSL happens to read one.
///
/// [`Renderer::ensure_pipeline`]: crate::Renderer::ensure_pipeline
/// [`FLAGS`]: MaterialVariant::FLAGS
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaterialVariant(u32);

impl MaterialVariant {
    /// No features: flat `base_color`, hemisphere + key light only.
    pub const NONE: MaterialVariant = MaterialVariant(0);
    /// Multiply albedo by the mesh's per-vertex color.
    pub const VERTEX_COLOR: MaterialVariant = MaterialVariant(1 << 0);
    /// Sample a baked procedural texture (§7), triplanar in world space.
    pub const TEXTURE: MaterialVariant = MaterialVariant(1 << 1);
    /// Reserved (§5): toon/ramp remap of the key-light term.
    pub const RAMP: MaterialVariant = MaterialVariant(1 << 2);
    /// Evaluate the procedural texture per pixel instead of sampling a bake
    /// (§7's live path): unbounded 3D world-space noise, no tile, no triplanar.
    ///
    /// **Mutually exclusive with [`TEXTURE`](MaterialVariant::TEXTURE)** on any
    /// draw the engine emits — live never reads the bake, so carrying both bits
    /// would compile a pipeline half of which is dead. [`draw::resolve_variant`]
    /// is where that is enforced, and it resolves in live's favour. The shader
    /// defines the combination anyway (live wins there too) so that a key
    /// arriving from a scene file or a test is a look, not undefined behaviour.
    ///
    /// Which bit a textured draw gets is a **render-side** decision, not a
    /// material one (DESIGN §11: "gates select data and variants"). See
    /// [`TextureLibrary::set_live_textures`].
    ///
    /// [`draw::resolve_variant`]: crate::draw::resolve_variant
    /// [`TextureLibrary::set_live_textures`]: crate::texture::TextureLibrary::set_live_textures
    pub const LIVE_TEX: MaterialVariant = MaterialVariant(1 << 3);
    /// Perturb the shading normal with the texture's crinkle (§7).
    ///
    /// Separate from [`TEXTURE`](MaterialVariant::TEXTURE) rather than implied
    /// by it: the crinkle is the expensive half (three more taps and a
    /// re-normalize) and plenty of surfaces want the colour without it. It is
    /// inert on its own — with no texture bound there is nothing to perturb by.
    ///
    /// It means the same thing on both paths and is spelled differently by
    /// each: baked reads the normal map and blends it triplanar; live has no
    /// map and takes the field's own gradient against the pixel footprint.
    pub const NORMAL_MAP: MaterialVariant = MaterialVariant(1 << 4);
    /// Alpha-blend over what is already there: `src·α + dst·(1−α)`, depth
    /// tested (`Less`) but **not** depth written.
    ///
    /// Also a *scheduling* bit. A blended draw cannot be state-sorted with the
    /// opaque ones — the order it needs is the camera's, not the pipeline's —
    /// so [`draw`] partitions it out and the renderer draws it after the whole
    /// opaque loop, back to front, in the same pass. See
    /// [`sort_draw_list_for_view`].
    ///
    /// Depth writes are off because a blended surface is not an occluder: two
    /// overlapping ghosts must both be visible, and a written depth would make
    /// whichever drew first delete the other. The cost is that blending is only
    /// as correct as the per-object sort — the usual bargain, and the reason
    /// the sort is a total order rather than a heuristic.
    ///
    /// [`draw`]: crate::draw
    /// [`sort_draw_list_for_view`]: crate::draw::sort_draw_list_for_view
    pub const TRANSPARENT: MaterialVariant = MaterialVariant(1 << 5);
    /// Additive blend: `src·α + dst`, no depth write. Light, not paint — glows,
    /// motes, silhouettes. Scheduled with [`TRANSPARENT`] (both are in
    /// [`BLENDED`]) and **wins** over it when a key carries both, because
    /// additive is the mode that cannot be expressed as a special case of the
    /// other.
    ///
    /// [`TRANSPARENT`]: MaterialVariant::TRANSPARENT
    /// [`BLENDED`]: MaterialVariant::BLENDED
    pub const ADDITIVE: MaterialVariant = MaterialVariant(1 << 6);
    /// Invert the depth test: `Greater` instead of `Less` — the fragment is
    /// drawn exactly where something else is already *in front of* it.
    ///
    /// This is the see-through-walls silhouette, and it composes rather than
    /// stands alone: the object draws normally once, then a second draw of the
    /// same geometry with `ADDITIVE | DEPTH_GREATER` paints the occluded part.
    /// Orthogonal to the blend bits on purpose — it changes when a fragment
    /// passes, not what happens to it after.
    pub const DEPTH_GREATER: MaterialVariant = MaterialVariant(1 << 7);
    /// The phase circle (the port's signature effect): a screen-space disc,
    /// centred and sized by [`FrameUniform::phase`], that decides per fragment
    /// whether this surface exists.
    ///
    /// This bit turns the effect **on**. Which way round it cuts is two more
    /// bits — [`PHASE_INVERT`] and [`PHASE_NO_DISCARD`] — so the whole mode
    /// lives in the key:
    ///
    /// | key | meaning |
    /// |---|---|
    /// | `PHASE_CIRCLE` | **world-only** — discard *inside* the circle |
    /// | `PHASE_CIRCLE \| PHASE_INVERT` | **phase-only** — discard *outside* it |
    /// | `PHASE_CIRCLE \| PHASE_NO_DISCARD` | **effect only** — never discard, just take the edge fringe |
    ///
    /// World-only is the bare bit, so the default reading of "phase geometry"
    /// is the commonest one and a key that says nothing extra says the right
    /// thing.
    ///
    /// The edge fringe is drawn for every fragment that survives, whatever the
    /// mode — it is a property of the *circle*, not of the geometry, and
    /// `PHASE_NO_DISCARD` is simply the mode with nothing else left.
    ///
    /// A radius of ~0 means "no circle": nothing is inside it, so world
    /// geometry is solid and phase geometry is gone — which is the resting
    /// state, and matches the original's `phase_common.gdshaderinc` exactly.
    ///
    /// # Why the mode is bits and not a number in [`Material::params`]
    ///
    /// It was `params.x`, and `params.x` is not the material's to spend. Three
    /// of the four slots already belong to [`VERTEX_WAVE`], whose amplitude
    /// *is* `x`, and the two are not alternatives — the port's waterfall is one
    /// draw that sways **and** answers to the circle
    /// (`3dimenshift/scenes/objects/water_flow.tscn`'s `Phaseable` over
    /// `fx/water.gdshader`'s vertex stage). One `f32` cannot be an amplitude
    /// and a mode at once, and the reading that lost was the wave: a mode of
    /// `0` is an amplitude of `0`, so the waterfall went flat the moment it
    /// learned to phase.
    ///
    /// Godot never had to choose — its shader carries `phase_mode` and
    /// `wave_amplitude` as two uniforms — and neither does this, because a mode
    /// is exactly what the variant key is for: three states, known at pipeline
    /// build time, folded to straight-line code by the `const` branch rather
    /// than compared against a float every fragment. `params` is for numbers an
    /// author *tunes*; the key is for choices that pick a shader. The
    /// waterfall is the draw that proved the difference, and it is why
    /// `params.x` is now [`VERTEX_WAVE`]'s alone.
    ///
    /// [`FrameUniform::phase`]: crate::FrameUniform::phase
    /// [`PHASE_INVERT`]: MaterialVariant::PHASE_INVERT
    /// [`PHASE_NO_DISCARD`]: MaterialVariant::PHASE_NO_DISCARD
    /// [`VERTEX_WAVE`]: MaterialVariant::VERTEX_WAVE
    pub const PHASE_CIRCLE: MaterialVariant = MaterialVariant(1 << 8);
    /// No lighting at all: the fragment is `base_color` × vertex color ×
    /// texture, written flat.
    ///
    /// Named for what wants it — camera-facing quads, whose normals are a lie
    /// and whose hemisphere term would therefore swim as the camera turns. The
    /// billboard *basis* is built on the CPU into the model matrix, so this is
    /// a fragment-side bit only and the vertex shader is untouched.
    pub const BILLBOARD_UNLIT: MaterialVariant = MaterialVariant(1 << 9);
    /// A silhouette rim: unlit, `pow(1 − |N·V|, power)` of `base_color`, in both
    /// the colour and the alpha. `power` is [`Material::params`]`.y`.
    ///
    /// The original's `phase_outline.gdshader` in its `SHELL` style — a
    /// holographic ghost of a phase object drawn exactly where the phase circle
    /// removed it, so a step you cannot stand on yet still tells you it is
    /// there. It composes: the ghost is `PHASE_CIRCLE | FRESNEL | ADDITIVE` on
    /// the same mesh, with the circle's mode *flipped* relative to the solid
    /// draw, which is how "render what the main pass discarded" is spelled with
    /// no second discard rule.
    ///
    /// The view vector is rebuilt per fragment from
    /// [`FrameUniform::inv_view_proj`] — the construction `sky.wgsl` already
    /// uses — because the frame block carries no camera position and a rim term
    /// is the only thing that has ever wanted one. Two matrix multiplies on a
    /// population of fragments that is, by construction, a thin rim.
    ///
    /// [`FrameUniform::inv_view_proj`]: crate::FrameUniform::inv_view_proj
    pub const FRESNEL: MaterialVariant = MaterialVariant(1 << 10);
    /// An unlit two-tone wipe along `uv.x`, driven entirely by
    /// [`Material::params`]: `(inactive_gain, progress, fade_width,
    /// active_gain)`.
    ///
    /// The original's `logic_wire.gdshader`: light runs down a wire from the
    /// switch that fired it. The *tones* come from the two colour channels a
    /// draw already has — the mesh's own vertex colour is the un-swept side and
    /// `base_color` is the swept one — so the effect needs no third colour slot
    /// and no new uniform. It therefore wants [`VERTEX_COLOR`] alongside it;
    /// without it the un-swept side is white.
    ///
    /// Emissive in the only sense this renderer has one (DESIGN §5: no emission
    /// channel): the gain multiplies the tone and the target clamps, so an
    /// "energy" above 1 reads as a colour blowing out to white — which is what
    /// the original's `EMISSION` did on the way through its tonemap.
    ///
    /// Generic on purpose: it is a parameterised wipe, not a wire. A charge bar,
    /// a fuse, a filling gauge is the same three numbers.
    ///
    /// [`VERTEX_COLOR`]: MaterialVariant::VERTEX_COLOR
    pub const EMISSIVE_SWEEP: MaterialVariant = MaterialVariant(1 << 11);
    /// Two crossed sines displacing the vertex along its **local +Y**, on the
    /// render clock — a water surface swaying, and the only *vertex*-stage look
    /// in the set so far.
    ///
    /// The original is `3dimenshift/shaders/fx/water.gdshader:27-33`, verbatim:
    ///
    /// ```text
    /// wp = (MODEL_MATRIX * vec4(VERTEX, 1)).xyz
    /// w  = sin(wp.x·frequency + TIME·speed)
    ///    + sin(wp.z·frequency·1.3 + TIME·speed·0.85)
    /// VERTEX.y += w · amplitude · 0.5
    /// ```
    ///
    /// The two ratios (`1.3` across Z, `0.85` on the second sine's clock) are
    /// what stop the pair reading as one diagonal wave, and they are constants
    /// in the original rather than uniforms — so they are constants here too
    /// (`WAVE_CROSS_FREQ` / `WAVE_CROSS_SPEED` in `shader.wgsl`). The three
    /// numbers that *are* authored ride in [`Material::params`]`.xyz`.
    ///
    /// # It is a render-clock effect and nothing else
    ///
    /// The phase comes from [`FrameUniform::time`]`.x`, which is host wall
    /// seconds and is **never** a simulation input (DESIGN §4). Nothing on the
    /// CPU knows where a displaced vertex ended up: a swimmer's membership test,
    /// a collider, a raycast all see the undisplaced surface. That is the
    /// original's arrangement too — Godot's water body does its swim maths
    /// against the ribbon's analytic frames and the shader sways the mesh
    /// alone — and it is why this can be a pure GPU bit with no CPU half.
    ///
    /// Because the displacement moves geometry *outside* the mesh's own bounds,
    /// a culler that trusts the source AABB can pop a swaying surface at a
    /// grazing angle. Godot's own answer is `custom_aabb`, grown by a wave
    /// margin; this engine's culling is per-draw against the same bounds, so a
    /// caller that cares wants its mesh built with the margin already in it.
    ///
    /// [`FrameUniform::time`]: crate::FrameUniform::time
    pub const VERTEX_WAVE: MaterialVariant = MaterialVariant(1 << 12);
    /// Draw both faces: `cull_mode: None` instead of the back-face cull every
    /// other key carries.
    ///
    /// The original is `3dimenshift/shaders/fx/water.gdshader:2`'s
    /// `cull_disabled` — a pond surface is one sheet of triangles and the
    /// swimmer is meant to see it from underneath, where a culled sheet is a
    /// hole in the world. Waterfalls are the same argument in the vertical.
    ///
    /// # Pipeline-only, and the first bit that is
    ///
    /// [`TRANSPARENT`], [`ADDITIVE`] and [`DEPTH_GREATER`] also select
    /// fixed-function state rather than a shader branch, but they select the
    /// *blend* and *depth* halves that live in the pipeline descriptor beside
    /// the module. This one is the same kind of thing one field over
    /// ([`PipelineState::cull`]), and it gets a `const` in the preprocessor for
    /// the same reason they do: the flag list is the declaration of the key
    /// space, not a list of things the WGSL reads.
    ///
    /// It does **not** join `tests/material_variants.rs`'s `SHADER_BITS` sweep,
    /// because crossing a bit that cannot change a module's bytes into a
    /// combinatorial *compile* sweep doubles its runtime to compile the same
    /// modules twice. What it does instead is keyed like any other bit, which
    /// `every_state_combination_is_its_own_pipeline` covers.
    ///
    /// Culling is a real saving on the population that can least afford fill,
    /// so this is opt-in per material rather than a global loosening — a
    /// two-sided draw pays double fragment cost on every triangle it has.
    ///
    /// [`TRANSPARENT`]: MaterialVariant::TRANSPARENT
    /// [`ADDITIVE`]: MaterialVariant::ADDITIVE
    /// [`DEPTH_GREATER`]: MaterialVariant::DEPTH_GREATER
    /// [`PipelineState::cull`]: crate::PipelineState::cull
    pub const TWO_SIDED: MaterialVariant = MaterialVariant(1 << 13);
    /// Receive the key light's shadow map (DESIGN §5, §11; see
    /// [`shadow`](crate::shadow)): the lit branch's key term is scaled by a
    /// comparison-sampled visibility factor.
    ///
    /// # A renderer-resolved bit, never a material's
    ///
    /// No `Material` sets this and no scene file should spell it: the renderer
    /// ORs it onto every lit draw while its shadow gate is open
    /// ([`resolve_shadow_variant`](crate::resolve_shadow_variant)), exactly as
    /// [`LIVE_TEX`] is §7's gate applied to [`TEXTURE`] — §11's "gates select
    /// data and variants", third instance. It is a *variant* rather than a
    /// runtime uniform branch for a reason bought with a moved golden hash:
    /// a live branch around the key term perturbed the driver's instruction
    /// scheduling enough to shift a handful of lit pixels by one LSB *with the
    /// gate closed*. A `const` branch folds to the old straight-line code, so
    /// shadows-off draws through the byte-identical pipeline it always had —
    /// the same key, the same cached object.
    ///
    /// Unlit looks ([`UNLIT`]) never receive the bit — their fragment path
    /// replaces the lighting term the shadow would scale — so the gate opening
    /// compiles a shadowed twin per *lit* variant in use, and nothing else.
    ///
    /// [`LIVE_TEX`]: MaterialVariant::LIVE_TEX
    /// [`TEXTURE`]: MaterialVariant::TEXTURE
    /// [`UNLIT`]: MaterialVariant::UNLIT
    pub const SHADOW: MaterialVariant = MaterialVariant(1 << 14);
    /// Sample the procedural texture in the entity's **own** space instead of
    /// the world's: the pattern is painted on the object and travels with it.
    ///
    /// Both texture paths honour it and there is exactly one decision in the
    /// shader — `shader.wgsl`'s `p_source` — because "which point are we
    /// sampling at" is one question whether the answer feeds a triplanar tile
    /// lookup ([`TEXTURE`]) or a per-pixel field evaluation ([`LIVE_TEX`]).
    ///
    /// # The read it exists to fix
    ///
    /// Without it a moving textured object is a hole cut in a pattern nailed to
    /// the world, and the pattern slides across the surface as the object goes
    /// past — the "sliding marble". The port's player is the standing case:
    /// `3dimenshift-runt/shift/src/model.rs`'s `spawn` doc comment records that
    /// the ball was left untextured for exactly this reason and that
    /// `model.rs`'s `mottled()` — a hashed patch brightness baked into vertex
    /// colours — is the stand-in it settled for. Carried boulders and moving
    /// platforms are the same problem with fewer axes.
    ///
    /// # What it costs
    ///
    /// One interpolated `vec3` on **every** draw, including the ones that never
    /// read it, because a WGSL varying cannot be conditional — see `VSOut`'s
    /// `local_pos` in `shader.wgsl` for the whole argument and for why reusing
    /// `world_pos` was rejected. The fragment side is free: the basis choice is
    /// a `const` branch that folds, so a draw without the bit compiles to the
    /// instructions it always had.
    ///
    /// # Feature size scales with the object, on purpose
    ///
    /// Object-local units are the model matrix's scale away from metres, so the
    /// same material on a half-scale entity gets half-size features. That is
    /// kept rather than divided out; `shader.wgsl`'s `p_source` comment carries
    /// the argument, the short form of which is that normalizing would make the
    /// picture on a mesh a function of the *instance*, and an entity animating
    /// its scale would then slide along the axis this bit exists to nail down.
    ///
    /// [`TEXTURE`]: MaterialVariant::TEXTURE
    /// [`LIVE_TEX`]: MaterialVariant::LIVE_TEX
    pub const LOCAL_SPACE: MaterialVariant = MaterialVariant(1 << 15);
    /// Flip which side of the phase circle [`PHASE_CIRCLE`] discards: *outside*
    /// instead of inside. Godot's `phase_mode = 1`, "phase-only".
    ///
    /// Geometry that exists nowhere but inside the circle — the port's ten
    /// materialising steps and floats, which are intangible until you shift and
    /// then are the only thing you can stand on. Inert without
    /// [`PHASE_CIRCLE`]: it names which way that bit cuts and has no meaning of
    /// its own.
    ///
    /// It is also how the **ghost** is spelled. `phase_outline.gdshader` writes
    /// its complement as arithmetic (`1 - phase_mode`) so that a second,
    /// additive draw of the same mesh fills in exactly what the solid draw threw
    /// away; here the same statement is this bit toggled, which is the same
    /// swap without the float.
    ///
    /// # Bit 16, and not the 12 the mode moved out of
    ///
    /// Bit positions are permanent (see the type docs) and 0..=15 were all
    /// spoken for when the mode left `params` — 12 is [`VERTEX_WAVE`] and 13 is
    /// [`TWO_SIDED`]. So the pair appends, like every look before it.
    ///
    /// [`PHASE_CIRCLE`]: MaterialVariant::PHASE_CIRCLE
    /// [`VERTEX_WAVE`]: MaterialVariant::VERTEX_WAVE
    /// [`TWO_SIDED`]: MaterialVariant::TWO_SIDED
    pub const PHASE_INVERT: MaterialVariant = MaterialVariant(1 << 16);
    /// [`PHASE_CIRCLE`] with **no discard at all**: the surface is there on both
    /// sides and takes only the edge fringe. Godot's `phase_mode = 2`,
    /// "effect only".
    ///
    /// For geometry that must never vanish but should still register that the
    /// circle swept over it — and, in the port, for `Phaseable` bodies tagged
    /// into the world dimension *and* a phase tier at once, which are solid
    /// either way and so have no side to be removed from.
    ///
    /// # With [`PHASE_INVERT`]
    ///
    /// This bit **wins**, and the shader says so with an `if / else if` in that
    /// order rather than leaving the pair undefined — the same resolution rule
    /// [`LIVE_TEX`] beats [`TEXTURE`] by and [`UNLIT`]'s chain uses. Nothing
    /// should author the combination: "never discard" and "discard on the other
    /// side" are not two halves of a look, they are two answers to one question,
    /// and a key carrying both is a caller that has not decided. Defined so that
    /// it cannot be a *surprise*, not so that it can be used.
    ///
    /// [`PHASE_CIRCLE`]: MaterialVariant::PHASE_CIRCLE
    /// [`PHASE_INVERT`]: MaterialVariant::PHASE_INVERT
    /// [`LIVE_TEX`]: MaterialVariant::LIVE_TEX
    /// [`TEXTURE`]: MaterialVariant::TEXTURE
    /// [`UNLIT`]: MaterialVariant::UNLIT
    pub const PHASE_NO_DISCARD: MaterialVariant = MaterialVariant(1 << 17);
    /// Slide the point a procedural texture is sampled at along the render
    /// clock, so the pattern **travels** across a surface that is standing still.
    ///
    /// One authored scalar, [`Material::params`]`.w`, in **world units per
    /// second**: the sample point walks `speed` along the basis's X and
    /// `speed · SCROLL_CROSS_SPEED` along its Z. The original is
    /// `3dimenshift/shaders/fx/water.gdshader:35`, verbatim —
    ///
    /// ```text
    /// uv = world_pos.xz · noise_scale + vec2(TIME·scroll_speed, TIME·scroll_speed·0.7)
    /// ```
    ///
    /// — and the `0.7` is a constant in the original rather than a uniform, so
    /// it is a constant here too (`SCROLL_CROSS_SPEED` in `shader.wgsl`, beside
    /// [`VERTEX_WAVE`]'s own pair). What it buys is the same thing the crossed
    /// sines buy: a single-axis drift reads as a conveyor belt, and two axes at
    /// an irrational-looking ratio read as a current.
    ///
    /// # A vector was the obvious shape and is the wrong one
    ///
    /// Three of `params`' four slots are [`VERTEX_WAVE`]'s and the fourth is
    /// this, which is exactly enough for the one number an author has ever
    /// wanted — the surfaces that scroll are water, and water scrolls *one way
    /// at one rate*. A free `vec2` (let alone a `vec3`) would need slots the
    /// uniform does not have, and would buy the ability to author a ratio whose
    /// only correct value is the one the original fixed. The rule
    /// [`WAVE_CROSS_FREQ`] set is that a shape constant stays a constant and
    /// only the *rate* is authored; this follows it.
    ///
    /// # Both texture paths, one decision
    ///
    /// The offset is added to `shader.wgsl`'s `p_source` — after
    /// [`LOCAL_SPACE`] has chosen the basis and before either texture branch
    /// reads it — so [`TEXTURE`]'s three plane taps and [`LIVE_TEX`]'s field
    /// evaluation move together and by the same amount. That is why the number
    /// is in world units rather than in tile units: the two paths scale
    /// `p_source` by different factors (`world_scale` against
    /// `live_cells_per_metre`), so a tile-space rate would make one authored
    /// number mean two different speeds either side of §7's live gate, and the
    /// A/B toggle would stop being a comparison.
    ///
    /// Under [`LOCAL_SPACE`] the walk is along the *object's* X and Z, which is
    /// the only reading that keeps the pattern the object's own — a world-axis
    /// drift on an object-space pattern would slide it across the surface,
    /// which is the artifact that bit exists to remove.
    ///
    /// Inert with no texture bound, like [`LOCAL_SPACE`] and [`NORMAL_MAP`]:
    /// it moves a sampling point, and a draw that samples nothing has no point
    /// to move.
    ///
    /// # It is a render-clock effect and nothing else
    ///
    /// [`FrameUniform::time`]`.x` is host wall seconds and is never a
    /// simulation input (DESIGN §4). This moves *which texel is under a
    /// fragment* and touches no vertex, no normal and no bound — so unlike
    /// [`VERTEX_WAVE`] it cannot even push geometry past a culler's AABB. There
    /// is no CPU half of this bit and there is nothing for one to do.
    ///
    /// # The slot is shared with [`EMISSIVE_SWEEP`], and that is the rule
    ///
    /// `params.w` is the sweep's "swept gain". Slot sharing across variants is
    /// already how this block works — "whichever variant reads the slot gets the
    /// number the author put there" — and the two are not a collision but a
    /// caller describing one surface as two looks: a wire that wipes is not a
    /// pond that drifts, and nothing in the port authors both on one draw. Said
    /// out loud here rather than left to the table below, because a shared slot
    /// discovered later reads as a bug.
    ///
    /// [`FrameUniform::time`]: crate::FrameUniform::time
    /// [`VERTEX_WAVE`]: MaterialVariant::VERTEX_WAVE
    /// [`WAVE_CROSS_FREQ`]: MaterialVariant::VERTEX_WAVE
    /// [`LOCAL_SPACE`]: MaterialVariant::LOCAL_SPACE
    /// [`NORMAL_MAP`]: MaterialVariant::NORMAL_MAP
    /// [`TEXTURE`]: MaterialVariant::TEXTURE
    /// [`LIVE_TEX`]: MaterialVariant::LIVE_TEX
    /// [`EMISSIVE_SWEEP`]: MaterialVariant::EMISSIVE_SWEEP
    pub const TEXTURE_SCROLL: MaterialVariant = MaterialVariant(1 << 18);

    /// The two bits that name *which way* [`PHASE_CIRCLE`] cuts, as one mask.
    ///
    /// What a caller setting a mode wants to clear first: the three states are
    /// exclusive, so "world-only" is the absence of both and cannot be OR'd on.
    /// [`Material::set_phase_mode`] is that operation with the rule already in it.
    ///
    /// Neither bit means anything without [`PHASE_CIRCLE`], which is why they
    /// are a mask over it rather than looks of their own.
    ///
    /// [`PHASE_CIRCLE`]: MaterialVariant::PHASE_CIRCLE
    /// [`Material::set_phase_mode`]: Material::set_phase_mode
    pub const PHASE_MODE: MaterialVariant =
        MaterialVariant(MaterialVariant::PHASE_INVERT.0 | MaterialVariant::PHASE_NO_DISCARD.0);

    /// Every declared flag, with the WGSL `const` it maps to. The order here is
    /// the order the preprocessor emits, so generated sources are stable.
    ///
    /// The render-state bits are in here with the rest even though nothing in
    /// the WGSL reads `F_TRANSPARENT`, `F_ADDITIVE`, `F_DEPTH_GREATER` or
    /// `F_TWO_SIDED`: this list is the *declaration* of the key space, and a bit
    /// missing from it would be a key the preprocessor silently could not
    /// describe.
    pub const FLAGS: [(&'static str, MaterialVariant); 19] = [
        ("F_VERTEX_COLOR", MaterialVariant::VERTEX_COLOR),
        ("F_TEXTURE", MaterialVariant::TEXTURE),
        ("F_RAMP", MaterialVariant::RAMP),
        ("F_LIVE_TEX", MaterialVariant::LIVE_TEX),
        ("F_NORMAL_MAP", MaterialVariant::NORMAL_MAP),
        ("F_TRANSPARENT", MaterialVariant::TRANSPARENT),
        ("F_ADDITIVE", MaterialVariant::ADDITIVE),
        ("F_DEPTH_GREATER", MaterialVariant::DEPTH_GREATER),
        ("F_PHASE_CIRCLE", MaterialVariant::PHASE_CIRCLE),
        ("F_BILLBOARD_UNLIT", MaterialVariant::BILLBOARD_UNLIT),
        ("F_FRESNEL", MaterialVariant::FRESNEL),
        ("F_EMISSIVE_SWEEP", MaterialVariant::EMISSIVE_SWEEP),
        ("F_VERTEX_WAVE", MaterialVariant::VERTEX_WAVE),
        ("F_TWO_SIDED", MaterialVariant::TWO_SIDED),
        ("F_SHADOW", MaterialVariant::SHADOW),
        ("F_LOCAL_SPACE", MaterialVariant::LOCAL_SPACE),
        ("F_PHASE_INVERT", MaterialVariant::PHASE_INVERT),
        ("F_PHASE_NO_DISCARD", MaterialVariant::PHASE_NO_DISCARD),
        ("F_TEXTURE_SCROLL", MaterialVariant::TEXTURE_SCROLL),
    ];

    /// The two blend bits, as one mask: the draws that leave the opaque
    /// state-sort and are drawn back-to-front after it.
    pub const BLENDED: MaterialVariant =
        MaterialVariant(MaterialVariant::TRANSPARENT.0 | MaterialVariant::ADDITIVE.0);

    /// The flags that actually do something. Anything outside this is declared,
    /// hashed and cached correctly — it just does not do anything yet.
    pub const IMPLEMENTED: MaterialVariant = MaterialVariant(
        MaterialVariant::VERTEX_COLOR.0
            | MaterialVariant::TEXTURE.0
            | MaterialVariant::LIVE_TEX.0
            | MaterialVariant::NORMAL_MAP.0
            | MaterialVariant::TRANSPARENT.0
            | MaterialVariant::ADDITIVE.0
            | MaterialVariant::DEPTH_GREATER.0
            | MaterialVariant::PHASE_CIRCLE.0
            | MaterialVariant::BILLBOARD_UNLIT.0
            | MaterialVariant::FRESNEL.0
            | MaterialVariant::EMISSIVE_SWEEP.0
            | MaterialVariant::VERTEX_WAVE.0
            | MaterialVariant::TWO_SIDED.0
            | MaterialVariant::SHADOW.0
            | MaterialVariant::LOCAL_SPACE.0
            | MaterialVariant::PHASE_INVERT.0
            | MaterialVariant::PHASE_NO_DISCARD.0
            | MaterialVariant::TEXTURE_SCROLL.0,
    );

    /// The four bits that each *replace* the lighting term rather than feeding
    /// it. Exactly one of them wins per fragment (the shader's `else if` chain,
    /// in this order), so a key carrying two is defined rather than undefined —
    /// the same resolution rule [`LIVE_TEX`] beats [`TEXTURE`] by.
    ///
    /// [`LIVE_TEX`]: MaterialVariant::LIVE_TEX
    /// [`TEXTURE`]: MaterialVariant::TEXTURE
    pub const UNLIT: MaterialVariant = MaterialVariant(
        MaterialVariant::FRESNEL.0
            | MaterialVariant::EMISSIVE_SWEEP.0
            | MaterialVariant::BILLBOARD_UNLIT.0,
    );

    pub const fn from_bits(bits: u32) -> MaterialVariant {
        MaterialVariant(bits)
    }

    pub const fn bits(self) -> u32 {
        self.0
    }

    pub const fn contains(self, other: MaterialVariant) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether *any* bit of `other` is set — [`contains`] asks for all of them.
    /// The distinction matters for masks like [`BLENDED`], where either bit
    /// alone is the whole answer.
    ///
    /// [`contains`]: MaterialVariant::contains
    /// [`BLENDED`]: MaterialVariant::BLENDED
    pub const fn intersects(self, other: MaterialVariant) -> bool {
        self.0 & other.0 != 0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Flags set here but not in [`IMPLEMENTED`](MaterialVariant::IMPLEMENTED).
    pub const fn unimplemented(self) -> MaterialVariant {
        MaterialVariant(self.0 & !MaterialVariant::IMPLEMENTED.0)
    }
}

impl std::ops::BitOr for MaterialVariant {
    type Output = MaterialVariant;
    fn bitor(self, rhs: MaterialVariant) -> MaterialVariant {
        MaterialVariant(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for MaterialVariant {
    fn bitor_assign(&mut self, rhs: MaterialVariant) {
        self.0 |= rhs.0;
    }
}

/// Build the WGSL for one variant: the feature `const`s, the shared noise
/// library, then the base source.
///
/// Deliberately the dumbest preprocessor that works — no `#ifdef` grammar, no
/// includes, no token scanning. WGSL const-folds `if (F_X)` on a `const bool`
/// away entirely, so a "disabled" feature costs nothing at runtime, and the
/// base source stays readable as ordinary WGSL.
///
/// The noise library ([`bake::NOISE_SHADER`](crate::bake::NOISE_SHADER)) is
/// prepended for **every** key, not only for `LIVE_TEX`. Making the *source*
/// depend on the variant in a second, invisible way would undo the property
/// that makes a variant system worth having — one source, one key, and the
/// generated bytes a pure function of the two. Nothing in the library is
/// reachable from a baked-only variant, so what it costs is compile time.
pub fn variant_source(base: &str, variant: MaterialVariant) -> String {
    let noise = crate::bake::NOISE_SHADER;
    let mut out = String::with_capacity(base.len() + noise.len() + 200);
    out.push_str("// generated by runt_core::material::variant_source\n");
    for (name, flag) in MaterialVariant::FLAGS {
        out.push_str("const ");
        out.push_str(name);
        out.push_str(": bool = ");
        out.push_str(if variant.contains(flag) { "true" } else { "false" });
        out.push_str(";\n");
    }
    out.push_str(noise);
    out.push('\n');
    out.push_str(base);
    out
}

/// A material: uniform payload + variant key (DESIGN §5).
///
/// Small and `Copy` on purpose — it is written straight into the per-instance
/// uniform slot with no indirection, which is what keeps the draw path a plain
/// loop over a sorted list.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Material {
    /// Albedo tint, multiplied into vertex color when `VERTEX_COLOR` is set.
    ///
    /// Alpha is written to the target, and *blended* only for a draw carrying
    /// [`TRANSPARENT`] or [`ADDITIVE`] — on the opaque path it still goes to
    /// the attachment and still does nothing (§5).
    ///
    /// [`TRANSPARENT`]: MaterialVariant::TRANSPARENT
    /// [`ADDITIVE`]: MaterialVariant::ADDITIVE
    pub base_color: Vec4,
    /// The per-material scalar slot, read by whichever variants a key names:
    ///
    /// | bits | `x` | `y` | `z` | `w` |
    /// |---|---|---|---|---|
    /// | [`FRESNEL`] | — | rim power | — | — |
    /// | [`EMISSIVE_SWEEP`] | un-swept gain | progress | fade width | swept gain |
    /// | [`VERTEX_WAVE`] | amplitude | frequency | speed | — |
    /// | [`TEXTURE_SCROLL`] | — | — | — | scroll speed, m/s |
    ///
    /// These are numbers an author **tunes**. A choice between a fixed set of
    /// behaviours is not one of those and does not belong here: it goes in the
    /// variant key, where it costs a `const` branch the compiler folds instead
    /// of a float comparison every fragment.
    ///
    /// [`PHASE_CIRCLE`] used to be the exception — its mode was `x`, a
    /// three-valued float compared with slack — and the exception is what broke.
    /// [`VERTEX_WAVE`] owns three of these four slots, `x` among them, and the
    /// port has a draw that is both: a waterfall that sways *and* answers to the
    /// phase circle. One slot cannot be an amplitude and a mode at once. The
    /// mode moved to [`PHASE_INVERT`] / [`PHASE_NO_DISCARD`] and `x` is the
    /// wave's alone; the whole argument is under [`PHASE_CIRCLE`].
    ///
    /// `x` is still shared between the sweep and the wave, and that stays
    /// defined rather than forbidden — a wire is not a water surface, and the
    /// rule is "whichever variant reads the slot gets the number the author put
    /// there". What changed is that the set of readers is now only ever
    /// *authored* numbers, so two of them on one draw is a caller describing one
    /// surface as two looks, not a mode colliding with a tuning value. `w` is the
    /// second such pair — [`EMISSIVE_SWEEP`]'s swept gain and
    /// [`TEXTURE_SCROLL`]'s speed — and it is deliberately *not* the exception
    /// the phase mode was: both are tuned numbers, a wire that wipes is not a
    /// pond that drifts, and no draw in the port is both. The rest is reserved
    /// for ramp threshold/softness/… as those variants land. Uploaded whole from
    /// the start so a new variant never changes the uniform layout.
    ///
    /// [`FRESNEL`]: MaterialVariant::FRESNEL
    /// [`EMISSIVE_SWEEP`]: MaterialVariant::EMISSIVE_SWEEP
    /// [`VERTEX_WAVE`]: MaterialVariant::VERTEX_WAVE
    /// [`TEXTURE_SCROLL`]: MaterialVariant::TEXTURE_SCROLL
    /// [`PHASE_CIRCLE`]: MaterialVariant::PHASE_CIRCLE
    /// [`PHASE_INVERT`]: MaterialVariant::PHASE_INVERT
    /// [`PHASE_NO_DISCARD`]: MaterialVariant::PHASE_NO_DISCARD
    pub params: Vec4,
    /// Which baked texture (§7) this material samples, if any.
    ///
    /// A content key, not a pointer — the same handle discipline `MeshRef`
    /// uses, so two materials naming one spec share one bake. It rides on the
    /// material rather than in `params` because it is a *binding*, resolved by
    /// the renderer's texture registry, not a number the shader reads.
    pub texture: Option<TextureHandle>,
    pub variant: MaterialVariant,
}

impl Default for Material {
    fn default() -> Material {
        Material::vertex_colored()
    }
}

impl Material {
    /// Turn the phase circle on in `mode`, replacing whatever mode was there.
    ///
    /// `mode` is one of [`NONE`] (world-only — discard inside the circle),
    /// [`PHASE_INVERT`] (phase-only) or [`PHASE_NO_DISCARD`] (effect only);
    /// [`PHASE_CIRCLE`] itself is added for you, because a mode with the effect
    /// switched off is not a state, it is a caller who forgot.
    ///
    /// The three modes are **exclusive**, which is the whole reason this exists
    /// rather than a bare `|=` at each call site: world-only is the *absence* of
    /// the two mode bits, so it cannot be OR'd on and re-tagging a material that
    /// already had a mode has to clear first. One copy of that rule, here.
    ///
    /// This replaced three `f32` constants (`PHASE_WORLD_ONLY`, `PHASE_ONLY`,
    /// `PHASE_EFFECT_ONLY`) that named values for `params.x`. They are gone
    /// rather than redefined as masks: a mask is a [`MaterialVariant`] and every
    /// other named combination of bits lives on that type ([`BLENDED`],
    /// [`UNLIT`], [`PHASE_MODE`]), so keeping phase aliases on `Material` would
    /// have been the one set of variant constants filed under the uniform
    /// struct. And world-only has no mask to be — it is the empty one — so two
    /// of the three would have read as "nothing" at the call site. A verb that
    /// takes the mode says it once and says it correctly.
    ///
    /// [`NONE`]: MaterialVariant::NONE
    /// [`PHASE_CIRCLE`]: MaterialVariant::PHASE_CIRCLE
    /// [`PHASE_INVERT`]: MaterialVariant::PHASE_INVERT
    /// [`PHASE_NO_DISCARD`]: MaterialVariant::PHASE_NO_DISCARD
    /// [`PHASE_MODE`]: MaterialVariant::PHASE_MODE
    /// [`BLENDED`]: MaterialVariant::BLENDED
    /// [`UNLIT`]: MaterialVariant::UNLIT
    pub fn set_phase_mode(&mut self, mode: MaterialVariant) {
        let kept = self.variant.bits() & !MaterialVariant::PHASE_MODE.bits();
        let mode = mode.bits() & MaterialVariant::PHASE_MODE.bits();
        self.variant =
            MaterialVariant::from_bits(kept | mode) | MaterialVariant::PHASE_CIRCLE;
    }

    /// Untinted, taking its color from the mesh's vertex colors.
    pub fn vertex_colored() -> Material {
        Material {
            base_color: Vec4::ONE,
            params: Vec4::ZERO,
            texture: None,
            variant: MaterialVariant::VERTEX_COLOR,
        }
    }

    /// A flat color, ignoring whatever the mesh's vertex colors say.
    pub fn flat(base_color: Vec4) -> Material {
        Material {
            base_color,
            params: Vec4::ZERO,
            texture: None,
            variant: MaterialVariant::NONE,
        }
    }

    /// Vertex colors tinted by `base_color`.
    pub fn tinted(base_color: Vec4) -> Material {
        Material {
            base_color,
            ..Material::vertex_colored()
        }
    }

    /// A baked procedural texture (§7), sampled triplanar and crinkled by its
    /// normal map. `base_color` tints it; vertex colors are off, because a
    /// generator's flat per-vertex tint fighting a procedural albedo is never
    /// what anyone wanted.
    pub fn textured(handle: TextureHandle) -> Material {
        Material {
            base_color: Vec4::ONE,
            params: Vec4::ZERO,
            texture: Some(handle),
            variant: MaterialVariant::TEXTURE | MaterialVariant::NORMAL_MAP,
        }
    }
}
