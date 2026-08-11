//! The key light's shadow map (DESIGN §5's "simple shadow map", §11's gate).
//!
//! One directional light, one cascade, one orthographic box: the engine's
//! whole shadow story is a depth-only render of the opaque draw list from the
//! key light's point of view, and a comparison-sampled lookup while the main
//! pass shades. No cascades, no perspective warps, no blob fallback — the
//! Godot original this serves is a stock `DirectionalLight3D` with
//! `shadow_enabled`, and at the port's near-top-down camera zoom the shadow's
//! whole job is the ground-contact cue under the player.
//!
//! This module is the CPU half: the two resources a game flips and tunes, and
//! the pure matrix arithmetic the renderer calls per frame. The GPU half —
//! the depth pass, the map, the comparison sampler — lives in the renderer
//! (`ShadowPass` in `lib.rs`), and the sampling itself in `shader.wgsl`.
//!
//! # The box, and why it is snapped
//!
//! The light's frustum is a fixed-size orthographic box aimed along
//! [`Lighting::key_dir`](crate::Lighting::key_dir) and centred a
//! half-box-length down the camera's central view ray — which is where the
//! player is, for any follow camera worth the name, without the renderer
//! having to know what a player is. A box that big moves every frame, and a
//! naïvely moving box makes every shadow edge shimmer: the map re-rasterizes
//! the same world into a grid that slid by a fraction of a texel. So the
//! light-space translation is **snapped to whole texels**
//! ([`light_view_proj`]): the sampling grid is pinned to the world, the camera
//! pans underneath it, and an edge only moves when the world does.
//!
//! # Off is the default, and off is nothing
//!
//! [`ShadowQuality::Off`] — the engine default — allocates no map, compiles no
//! pipeline, encodes no pass, and leaves the frame **byte-identical** to an
//! engine built before this module existed (`tests/shadows.rs` pins it, the
//! same way `tests/render_scale.rs` pins native scale). The gate follows §11's
//! table: off / 512² / 2048², hand-flipped in the style of
//! [`Engine::set_live_textures`](crate::Engine::set_live_textures) until the
//! perf probe exists to flip it at startup.

use bevy_ecs::prelude::Resource;
use glam::{Mat4, Vec3, Vec4};

/// The shadow gate (DESIGN §11: *off or 512² single cascade / 2048²*).
///
/// A resource, exactly like [`RenderScale`](crate::ecs::RenderScale) and for
/// the same reason: it lives where a `FixedSim` system or an install hook can
/// write it, both hosts pick the write up, and nothing in a tick reads it — so
/// no determinism fingerprint can move when it changes. The renderer mirrors
/// it once per frame ([`Engine::render`](crate::Engine::render)).
///
/// **`Off` is the default.** A gated feature must cost nothing while the gate
/// is closed (§11), and here that is literal: no texture, no pipeline, no
/// pass, and pixels bit-for-bit what they were before shadows existed.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ShadowQuality {
    /// No shadow map at all. The engine default.
    #[default]
    Off,
    /// A 512² map — §11's low tier. Soft-edged and a little chunky, which at
    /// the port's camera distance reads fine as a contact cue.
    Low,
    /// A 2048² map — §11's high tier.
    High,
}

impl ShadowQuality {
    /// The map's edge in texels, or `None` for [`Off`](ShadowQuality::Off).
    pub fn resolution(self) -> Option<u32> {
        match self {
            ShadowQuality::Off => None,
            ShadowQuality::Low => Some(512),
            ShadowQuality::High => Some(2048),
        }
    }
}

/// The shadow's tuning: how big the box is and how hard the acne is pushed
/// back. Reflected, so a game can hang it in its tweak panel beside
/// [`Lighting`](crate::Lighting) — it is lighting tuning in every sense but
/// one, and that one is why it is not *in* `Lighting`: the rig is scene data
/// with a RON twin ([`scene::LightingDesc`](crate::scene)), while these are
/// render policy like [`RenderScale`](crate::ecs::RenderScale) — no scene
/// should serialize a depth bias, and the scene loader must not reset one.
///
/// # Tuning notes
///
/// The two biases trade **acne** (too little: a surface shadows itself in
/// stripes) against **peter-panning** (too much: the shadow detaches from the
/// caster's feet). Both are in normalized light-depth units, where 1.0 is the
/// whole `4 × extent` depth range of the box — so their useful magnitudes are
/// small, and they scale with `extent` on purpose: a bigger box has coarser
/// texels and needs proportionally more bias. The defaults are tuned against
/// the 512² map (the coarser, acne-prone tier) on the demo scene;
/// `tests/shadows.rs` holds a bare lit plane to "no acne" at both tiers.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect))]
pub struct ShadowSettings {
    /// Half-size of the orthographic box, world units, on every axis. The map
    /// covers a `2·extent` square of world; texel size is `2·extent ÷
    /// resolution`, so smaller is sharper and larger reaches further.
    #[cfg_attr(feature = "reflect", reflect(@crate::reflect::FieldRange::new(4.0, 96.0)))]
    pub extent: f32,
    /// Constant depth bias, in normalized light-depth units. The floor every
    /// receiver gets whatever its slope.
    #[cfg_attr(feature = "reflect", reflect(@crate::reflect::FieldRange::new(0.0, 0.02)))]
    pub bias: f32,
    /// Extra bias scaled by `1 − N·L`: a surface at a grazing angle to the
    /// light spans many depth values per texel and needs more room.
    #[cfg_attr(feature = "reflect", reflect(@crate::reflect::FieldRange::new(0.0, 0.05)))]
    pub slope_bias: f32,
}

impl Default for ShadowSettings {
    fn default() -> ShadowSettings {
        ShadowSettings {
            extent: 20.0,
            bias: 0.0015,
            slope_bias: 0.004,
        }
    }
}

/// World space → the key light's clip space: the matrix the depth pass renders
/// through and the main pass samples through. Pure, so the snap is a unit test
/// rather than a screenshot.
///
/// Three steps:
///
/// 1. **Find the focus.** The camera's central view ray is rebuilt from
///    `view_proj`'s inverse (the same unprojection `sky.wgsl` does per pixel,
///    done once here), and the box is centred `extent` down it — half a
///    box-length, so the near half of the box covers the ground at the
///    camera's feet and the far half reaches ahead of it. A degenerate
///    view-projection (the no-camera identity, a NaN from a broken transform)
///    resolves to the origin rather than poisoning the matrix.
///
/// 2. **Aim the light.** `look_at` from `center + dir·2·extent` back at the
///    center, so the depth range `[0, 4·extent]` holds casters up to a whole
///    box-length above the box as well as everything in it. The up vector
///    ducks to `+Z` when the light is near-vertical — which the port's key
///    light almost is — because a `look_at` whose up parallels its view has no
///    basis at all.
///
/// 3. **Snap.** The view's light-space XY translation is rounded to whole
///    texels (`2·extent ÷ resolution`). The rotation never changes (the light
///    direction is authored, not animated per frame), so snapping translation
///    alone pins the sampling grid to the world: a camera pan re-renders the
///    same world into the same texels, and shadow edges hold still. The
///    Z translation is deliberately *not* snapped — depth is continuous and
///    compared with a bias, not rasterized into a grid.
pub fn light_view_proj(
    view_proj: &Mat4,
    key_dir: Vec3,
    extent: f32,
    resolution: u32,
) -> Mat4 {
    let extent = if extent.is_finite() { extent.max(1.0) } else { ShadowSettings::default().extent };

    // 1. The focus, off the camera's central ray.
    let inv = view_proj.inverse();
    let unproject = |z: f32| {
        let p = inv * Vec4::new(0.0, 0.0, z, 1.0);
        if p.w.abs() > 1.0e-9 {
            p.truncate() / p.w
        } else {
            Vec3::ZERO
        }
    };
    let near = unproject(0.0);
    let far = unproject(1.0);
    let forward = (far - near).normalize_or(Vec3::NEG_Z);
    let mut center = near + forward * extent;
    if !center.is_finite() {
        center = Vec3::ZERO;
    }

    // 2. The light's view.
    let dir = key_dir.normalize_or(Vec3::Y);
    let up = if dir.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
    let mut view = glam::camera::rh::view::look_at_mat4(center + dir * (2.0 * extent), center, up);

    // 3. The texel snap.
    let texel = (2.0 * extent) / resolution.max(1) as f32;
    view.w_axis.x = (view.w_axis.x / texel).round() * texel;
    view.w_axis.y = (view.w_axis.y / texel).round() * texel;

    let proj = glam::camera::rh::proj::directx::orthographic(
        -extent,
        extent,
        -extent,
        extent,
        0.0,
        4.0 * extent,
    );
    proj * view
}
