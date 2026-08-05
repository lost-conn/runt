//! World → draw list: the `Extract` half of the frame (DESIGN §3, §5).
//!
//! The renderer never queries the world. It is handed a flat, sorted `Vec` of
//! [`DrawItem`]s plus one [`FrameParams`], which is what keeps the GPU side
//! testable, the ECS side GPU-free, and the sort order an ordinary unit test.
//!
//! The one decision this layer makes rather than copies is §7's live/baked
//! variant bit — see [`resolve_variant`]. It belongs here because it is a
//! *render* policy (DESIGN §11's gate) applied to a *declarative* material:
//! putting it on `Material` would mean every entity carrying a copy of a
//! host-wide switch, and flipping it would mean a world mutation.
//!
//! Sorting is by `(pass, variant, texture, mesh, entity)`: variant first
//! because a pipeline swap is the expensive state change, texture second
//! because a bind-group swap is the next one, mesh third because vertex/index
//! buffer binds are after that, entity last purely as a deterministic
//! tie-break — two frames from the same world state must produce byte-identical
//! command streams.
//!
//! # The blended pass
//!
//! `pass` is ahead of all of it, and it is the one place where correctness
//! outranks state changes. A draw carrying [`TRANSPARENT`] or [`ADDITIVE`]
//! blends with whatever is already in the attachment and writes no depth, so
//! the order it needs is **the camera's**, not the pipeline's: farthest first,
//! after every opaque draw. Those items are partitioned to the end of the list
//! here and re-ordered by [`sort_draw_list_for_view`] once the view matrix is
//! known.
//!
//! Why the second step is separate: extraction is a pure function of the world
//! and an alpha (that is what makes `Sim::draw_list` cheap and testable), and
//! the camera lives in [`FrameParams`], one layer up. So this layer decides
//! *which pass* an item belongs to — a property of the material alone — and the
//! renderer, which is holding the view-projection anyway, decides the order
//! within the blended one. A frame with no blended item never runs the second
//! step at all, which is what keeps the opaque path byte-identical.
//!
//! [`TRANSPARENT`]: MaterialVariant::TRANSPARENT
//! [`ADDITIVE`]: MaterialVariant::ADDITIVE

use bevy_ecs::prelude::*;
use glam::{Mat4, Vec4};

use crate::ecs::{Interpolated, Lighting, MeshRef, Transform};
use crate::material::{Material, MaterialVariant};
use crate::registry::MeshHandle;
use crate::texture::{TextureHandle, TextureLibrary};

/// One indexed draw: which pipeline, which buffers, and the instance uniform to
/// write for it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrawItem {
    pub entity: Entity,
    pub variant: MaterialVariant,
    pub mesh: MeshHandle,
    /// Render-time model matrix: interpolated where the entity has an
    /// [`Interpolated`], the plain transform where it does not.
    pub model: Mat4,
    pub base_color: Vec4,
    pub params: Vec4,
    /// The baked texture (DESIGN §7) this draw binds, if any. `None` binds the
    /// renderer's 1×1 white/flat default, so the render loop has no branch.
    pub texture: Option<TextureHandle>,
}

impl DrawItem {
    /// The sort key. Public so the ordering can be asserted directly.
    ///
    /// `pass` leads: `0` for opaque, `1` for blended (see the module docs).
    /// It cannot be folded into the variant bits — [`PHASE_CIRCLE`] and
    /// [`BILLBOARD_UNLIT`] are numerically above the blend bits and are
    /// perfectly ordinary opaque looks — so it is its own field.
    ///
    /// The tie-break is the entity's *index*, not `Entity`'s own `Ord` — that
    /// one compares opaque bits, which today happens to run backwards from
    /// spawn order and could change between bevy_ecs releases. Index-then-bits
    /// is just as total, and it reads the way a person expects.
    ///
    /// Untextured draws key as `0`, which sorts them ahead of every textured
    /// one within a variant — so the two populations never interleave and the
    /// default bind group is set at most once per variant.
    ///
    /// [`PHASE_CIRCLE`]: MaterialVariant::PHASE_CIRCLE
    /// [`BILLBOARD_UNLIT`]: MaterialVariant::BILLBOARD_UNLIT
    pub fn sort_key(&self) -> (u32, u32, u64, u64, u32, u64) {
        (
            self.pass(),
            self.variant.bits(),
            self.texture.map(|t| t.0).unwrap_or(0),
            self.mesh.0,
            self.entity.index_u32(),
            self.entity.to_bits(),
        )
    }

    /// Whether this draw blends rather than replaces — [`BLENDED`] in the key.
    ///
    /// [`BLENDED`]: MaterialVariant::BLENDED
    pub fn is_blended(&self) -> bool {
        self.variant.intersects(MaterialVariant::BLENDED)
    }

    /// `0` opaque, `1` blended: which half of the frame this item is drawn in.
    pub fn pass(&self) -> u32 {
        self.is_blended() as u32
    }

    /// The blended pass's key: farthest fragment first, then entity.
    ///
    /// **Every component is an integer**, which is the point. Depth is a `f32`
    /// that came out of a matrix multiply, and DESIGN's determinism rule is not
    /// "the same inputs give the same floats" (they do) but "the *order* must
    /// never depend on how a comparison treats a float". So the depth is mapped
    /// to a `u32` that carries IEEE-754 total order — sign-magnitude flipped to
    /// two's-complement-ish, the standard radix trick — and then complemented,
    /// because back-to-front is descending depth and `sort_by_key` ascends.
    /// NaN and ±0 land in fixed places instead of making a comparator
    /// non-transitive; nothing is ever *equal-but-unordered*.
    ///
    /// Depth is `clip.w` of the model's origin: for the engine's perspective
    /// projection that is exactly the view-space distance along the camera's
    /// forward axis, with no view matrix needed on this side (we are only ever
    /// handed the product). A projection whose `w` is constant — an
    /// orthographic one, or [`FrameParams::default`]'s identity — makes every
    /// depth equal, and the entity tie-break carries the whole order. That is a
    /// degradation to "arbitrary but stable", never to "unstable".
    ///
    /// Per *object*, not per triangle: a blended object that intersects another
    /// is still drawn in one piece and can still sort wrong. Splitting geometry
    /// to fix that is a cost the content does not need us to pay.
    pub fn blended_key(&self, view_proj: &Mat4) -> (u32, u32, u64) {
        let clip = *view_proj * self.model.w_axis;
        (
            !ordered_f32(clip.w),
            self.entity.index_u32(),
            self.entity.to_bits(),
        )
    }
}

/// `f32` → `u32` preserving IEEE-754 total order: positives keep their order
/// above the sign flip, negatives reverse below it. Bijective, so no two
/// distinct bit patterns collide and no comparison is ever ambiguous.
fn ordered_f32(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits ^ 0x8000_0000
    }
}

/// Per-frame constants: the camera's view-projection and the light rig.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameParams {
    pub view_proj: Mat4,
    pub lighting: Lighting,
}

impl Default for FrameParams {
    fn default() -> FrameParams {
        FrameParams {
            view_proj: Mat4::IDENTITY,
            lighting: Lighting::default(),
        }
    }
}

/// Components a drawable entity must have.
pub type DrawQuery = (
    Entity,
    &'static MeshRef,
    &'static Material,
    &'static Transform,
    Option<&'static Interpolated>,
);

/// Collect every drawable entity at interpolation `alpha`, sorted.
///
/// Takes `&mut World` because that is what building a fresh `QueryState` costs;
/// [`Sim`](crate::Sim) caches one instead and calls [`extract_draw_list`].
pub fn build_draw_list(world: &mut World, alpha: f32) -> Vec<DrawItem> {
    let mut query = world.query::<DrawQuery>();
    extract_draw_list(&mut query, world, alpha)
}

/// Which variant a textured draw actually gets, once §7's live/baked gate has
/// had its say (DESIGN §11: *gates select data and variants*).
///
/// [`TEXTURE`] and [`LIVE_TEX`] are **mutually exclusive on a draw**. Live
/// evaluates the spec per pixel and never reads the bake, so a key carrying
/// both would compile a pipeline half of which is unreachable — and would leave
/// the sort order with two populations that are really one. Live wins when both
/// are asked for, here and in the shader, so the resolution is the same wherever
/// you look.
///
/// An **untextured** material is returned untouched. Its `TEXTURE` /
/// `LIVE_TEX` bits (if a hand-built `Material` set them) are already inert —
/// there is no handle to sample and no spec to evaluate, so the bind group is
/// the 1×1 default either way — and rewriting them would make this function
/// change the look of a draw it has no business having an opinion about.
///
/// [`TEXTURE`]: MaterialVariant::TEXTURE
/// [`LIVE_TEX`]: MaterialVariant::LIVE_TEX
pub fn resolve_variant(
    variant: MaterialVariant,
    textured: bool,
    live_textures: bool,
) -> MaterialVariant {
    if !textured {
        return variant;
    }
    let exclusive = MaterialVariant::TEXTURE.bits() | MaterialVariant::LIVE_TEX.bits();
    // The gate is a promotion, not a mask: a material that asked for live in
    // its scene file keeps it whether or not the host flipped the global
    // switch. v1 has no perf probe, so there is no tier to demote against —
    // when §11's probe lands, *it* becomes the thing that can say no, and the
    // per-material request goes back to being a request.
    let wants_live = live_textures || variant.contains(MaterialVariant::LIVE_TEX);
    let mode = if wants_live {
        MaterialVariant::LIVE_TEX
    } else {
        MaterialVariant::TEXTURE
    };
    MaterialVariant::from_bits((variant.bits() & !exclusive) | mode.bits())
}

/// As [`build_draw_list`], reusing a cached query state.
pub fn extract_draw_list(
    query: &mut QueryState<DrawQuery>,
    world: &World,
    alpha: f32,
) -> Vec<DrawItem> {
    // A world with no texture library (a bare `World` in a unit test) has no
    // textures either, so the flag it would have carried cannot matter.
    let live = world
        .get_resource::<TextureLibrary>()
        .is_some_and(TextureLibrary::live_textures);
    let mut items: Vec<DrawItem> = query
        .iter(world)
        .map(|(entity, mesh, material, transform, interpolated)| DrawItem {
            entity,
            variant: resolve_variant(material.variant, material.texture.is_some(), live),
            mesh: mesh.0,
            model: match interpolated {
                Some(prev) => prev.blend(transform, alpha),
                None => transform.matrix(),
            },
            base_color: material.base_color,
            params: material.params,
            texture: material.texture,
        })
        .collect();
    sort_draw_list(&mut items);
    items
}

/// Sort in place by `(pass, variant, texture, mesh, entity)` — see the module
/// docs for why that order and why the tie-break is not optional.
///
/// This leaves the blended items at the end, grouped by state rather than
/// ordered by depth; [`sort_draw_list_for_view`] is what finishes them.
pub fn sort_draw_list(items: &mut [DrawItem]) {
    items.sort_unstable_by_key(|item| item.sort_key());
}

/// Whether any item needs the camera-ordered second pass.
///
/// The renderer's guard: a frame of nothing but opaque geometry must take the
/// exact path it took before blending existed — same slice, same slots, same
/// commands — and that is a claim this predicate is the whole of.
pub fn has_blended(items: &[DrawItem]) -> bool {
    items.iter().any(DrawItem::is_blended)
}

/// The full frame order: opaque by state, then blended back-to-front from
/// `view_proj`.
///
/// Called by the renderer, which is the first thing to hold both the list and
/// the camera. Idempotent and total: sorting an already-sorted list is a no-op,
/// and two lists with the same contents in any spawn order come out identical.
pub fn sort_draw_list_for_view(items: &mut [DrawItem], view_proj: &Mat4) {
    sort_draw_list(items);
    // `sort_draw_list` put every blended item last, so the blended half is one
    // contiguous suffix and `partition_point` finds its start in log n.
    let start = items.partition_point(|item| !item.is_blended());
    items[start..].sort_unstable_by_key(|item| item.blended_key(view_proj));
}
