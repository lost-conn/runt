//! World → draw list: the `Extract` half of the frame (DESIGN §3, §5).
//!
//! The renderer never queries the world. It is handed a flat, sorted `Vec` of
//! [`DrawItem`]s plus one [`FrameParams`], which is what keeps the GPU side
//! testable, the ECS side GPU-free, and the sort order an ordinary unit test.
//!
//! Sorting is by `(variant, texture, mesh, entity)`: variant first because a
//! pipeline swap is the expensive state change, texture second because a
//! bind-group swap is the next one, mesh third because vertex/index buffer
//! binds are after that, entity last purely as a deterministic tie-break — two
//! frames from the same world state must produce byte-identical command
//! streams.

use bevy_ecs::prelude::*;
use glam::{Mat4, Vec4};

use crate::ecs::{Interpolated, Lighting, MeshRef, Transform};
use crate::material::{Material, MaterialVariant};
use crate::registry::MeshHandle;
use crate::texture::TextureHandle;

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
    /// The tie-break is the entity's *index*, not `Entity`'s own `Ord` — that
    /// one compares opaque bits, which today happens to run backwards from
    /// spawn order and could change between bevy_ecs releases. Index-then-bits
    /// is just as total, and it reads the way a person expects.
    ///
    /// Untextured draws key as `0`, which sorts them ahead of every textured
    /// one within a variant — so the two populations never interleave and the
    /// default bind group is set at most once per variant.
    pub fn sort_key(&self) -> (u32, u64, u64, u32, u64) {
        (
            self.variant.bits(),
            self.texture.map(|t| t.0).unwrap_or(0),
            self.mesh.0,
            self.entity.index_u32(),
            self.entity.to_bits(),
        )
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

/// As [`build_draw_list`], reusing a cached query state.
pub fn extract_draw_list(
    query: &mut QueryState<DrawQuery>,
    world: &World,
    alpha: f32,
) -> Vec<DrawItem> {
    let mut items: Vec<DrawItem> = query
        .iter(world)
        .map(|(entity, mesh, material, transform, interpolated)| DrawItem {
            entity,
            variant: material.variant,
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

/// Sort in place by `(variant, texture, mesh, entity)` — see the module docs
/// for why that order and why the tie-break is not optional.
pub fn sort_draw_list(items: &mut [DrawItem]) {
    items.sort_unstable_by_key(|item| item.sort_key());
}
