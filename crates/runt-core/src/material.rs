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
    /// The mode is per material, in [`Material::params`]`.x`:
    ///
    /// | `params.x` | meaning |
    /// |---|---|
    /// | `0` | **world-only** — discard *inside* the circle |
    /// | `1` | **phase-only** — discard *outside* it |
    /// | `2` | **effect only** — never discard, just take the edge fringe |
    ///
    /// A radius of ~0 means "no circle": nothing is inside it, so world
    /// geometry is solid and phase geometry is gone — which is the resting
    /// state, and matches the original's `phase_common.gdshaderinc` exactly.
    ///
    /// [`FrameUniform::phase`]: crate::FrameUniform::phase
    pub const PHASE_CIRCLE: MaterialVariant = MaterialVariant(1 << 8);
    /// No lighting at all: the fragment is `base_color` × vertex color ×
    /// texture, written flat.
    ///
    /// Named for what wants it — camera-facing quads, whose normals are a lie
    /// and whose hemisphere term would therefore swim as the camera turns. The
    /// billboard *basis* is built on the CPU into the model matrix, so this is
    /// a fragment-side bit only and the vertex shader is untouched.
    pub const BILLBOARD_UNLIT: MaterialVariant = MaterialVariant(1 << 9);

    /// Every declared flag, with the WGSL `const` it maps to. The order here is
    /// the order the preprocessor emits, so generated sources are stable.
    ///
    /// The render-state bits are in here with the rest even though nothing in
    /// the WGSL reads `F_TRANSPARENT`, `F_ADDITIVE` or `F_DEPTH_GREATER`: this
    /// list is the *declaration* of the key space, and a bit missing from it
    /// would be a key the preprocessor silently could not describe.
    pub const FLAGS: [(&'static str, MaterialVariant); 10] = [
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
    /// The per-material scalar slot. `x` is [`PHASE_CIRCLE`]'s mode
    /// ([`PHASE_WORLD_ONLY`] / [`PHASE_ONLY`] / [`PHASE_EFFECT_ONLY`]); the
    /// rest is still reserved for ramp threshold/softness/… as those variants
    /// land. Uploaded whole from the start so a new variant never changes the
    /// uniform layout.
    ///
    /// [`PHASE_CIRCLE`]: MaterialVariant::PHASE_CIRCLE
    /// [`PHASE_WORLD_ONLY`]: Material::PHASE_WORLD_ONLY
    /// [`PHASE_ONLY`]: Material::PHASE_ONLY
    /// [`PHASE_EFFECT_ONLY`]: Material::PHASE_EFFECT_ONLY
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
    /// [`PHASE_CIRCLE`](MaterialVariant::PHASE_CIRCLE) mode for `params.x`:
    /// ordinary world geometry, which the circle *removes*.
    pub const PHASE_WORLD_ONLY: f32 = 0.0;
    /// …geometry that only exists inside the circle.
    pub const PHASE_ONLY: f32 = 1.0;
    /// …geometry the circle never removes: it takes the edge fringe and
    /// nothing else.
    pub const PHASE_EFFECT_ONLY: f32 = 2.0;

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
