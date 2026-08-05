//! World model — components, schedule labels and the `FixedSim` systems
//! (DESIGN §3).
//!
//! `bevy_ecs` à la carte: no `bevy_app`, no `bevy_time`, no plugin machinery.
//! The tick loop that drives these schedules lives in [`crate::sim::Sim`].
//!
//! Determinism rules that this module exists to enforce (DESIGN §3):
//! every schedule is explicitly `.chain()`ed, every schedule runs on the
//! single-threaded executor, and nothing here iterates a hash container.

use bevy_ecs::prelude::*;
use bevy_ecs::schedule::ScheduleLabel;
use glam::{Mat4, Quat, Vec2, Vec3};
use runt_mesh::{HeightField, Quality, TerrainParams};

use crate::registry::MeshHandle;

// ---------------------------------------------------------------------------
// Schedule labels
// ---------------------------------------------------------------------------

/// Runs exactly once, when the [`Sim`](crate::sim::Sim) is constructed.
#[derive(ScheduleLabel, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Startup;

/// Interpolation bookkeeping. Runs at the **start** of every tick, before
/// [`FixedSim`], so that `Interpolated` captures the *previous* tick's
/// transform while `FixedSim` goes on to produce the current one.
#[derive(ScheduleLabel, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PostSim;

/// The deterministic tick. All sim mutation happens here and nowhere else.
#[derive(ScheduleLabel, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FixedSim;

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// The tick length, in seconds. **Constant for the life of the sim** — this is
/// not a frame delta, and treating it as one would break the whole point of a
/// fixed tick. It is a resource only so systems can read the configured rate
/// (DESIGN §12 step 2 wants a tick-rate toggle to prove interpolation).
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct FixedTick {
    pub dt_secs: f32,
}

/// Number of ticks executed since the sim started. Monotonic, never reset —
/// the x-axis of a replay trace.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TickCount(pub u64);

/// The demo's spinning entity (the twisted box). Only a convenience handle for
/// tests and the demo's follow camera — nothing in the render path needs it.
///
/// Set by the scene loader from the camera's follow target, falling back to the
/// first entity that spins.
#[derive(Resource, Clone, Copy, Debug)]
pub struct DemoEntity(pub Entity);

/// One line of text the host is asked to show somewhere outside the 3D frame.
///
/// **The engine has no text renderer** and DESIGN §13 leaves HUD text open
/// ("cheapest candidate: DOM overlay on web, nothing native, until a real need
/// appears"). A game still needs to say *3/12 · 12.4 s* somewhere on tick one of
/// its existence, so this is the seam: a game system writes a string, and the
/// host paints it wherever its platform has cheap text — the window title
/// natively, `document.title` plus a `#runt-status` element on web.
///
/// Deliberately a plain `String` and deliberately *not* read by anything in the
/// engine: it is an output channel, never sim state. Nothing branches on it, so
/// a host that ignores it entirely still runs the same simulation, and it cannot
/// enter a determinism fingerprint (which is over transforms).
///
/// Written from a `FixedSim` system like any other gameplay output; the host
/// reads [`Sim::status_line`](crate::Sim::status_line) after each frame and only
/// touches the platform when the string actually changed.
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct StatusLine(pub String);

impl StatusLine {
    /// Replace the line, reporting whether it actually changed. Cheap enough to
    /// call every tick, which is what a game system wants to do.
    pub fn set(&mut self, text: impl Into<String>) -> bool {
        let text = text.into();
        if self.0 == text {
            return false;
        }
        self.0 = text;
        true
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// The device/LOD quality multiplier for this session (DESIGN §6, §11).
///
/// Read once, at scene load, and turned into a [`Quality`] per generator via the
/// scene's quality policy. It is not consulted per frame: a different quality is
/// a different *mesh*, not a different way of drawing one, so changing it means
/// reloading the scene.
///
/// The device-tier probe of §11 will write this at startup; until then it is
/// 1.0 unless a caller says otherwise.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect))]
pub struct QualityTier(pub f32);

impl Default for QualityTier {
    fn default() -> QualityTier {
        QualityTier(1.0)
    }
}

impl QualityTier {
    pub fn quality(self) -> Quality {
        Quality(self.0)
    }
}

/// The fraction of the host's target resolution the 3D scene is drawn at
/// (DESIGN §11's resolution lever) — "pixel chonkiness".
///
/// At 1.0 (the default) the renderer draws straight into the view the host
/// handed it and this resource costs nothing. Below 1.0 the whole pass sequence
/// goes into an internal color+depth target of [`size`](RenderScale::size) and
/// is then blitted up with a **nearest** sampler, so a 0.5 frame is honest 2×2
/// blocks rather than a blur. Fragment cost falls with the *area*: 0.5 is a
/// quarter of the pixels, which is why this is the first thing to reach for on a
/// device that cannot afford the shading it is being asked for.
///
/// # Why it is a resource and not a `Renderer` field
///
/// Exactly the arrangement §7's live-texture switch already uses (see
/// [`TextureLibrary::set_live_textures`](crate::texture::TextureLibrary::set_live_textures)):
/// the value lives where a `FixedSim` system can write it — so a game binds it
/// to a key once and both hosts, the window and the canvas, get the binding —
/// while the *effect* is confined to the render path. Nothing in the engine
/// reads it inside a tick, so no simulation state can depend on it and no
/// determinism fingerprint can move when it changes. It is an output knob that
/// happens to be reachable from input.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect))]
pub struct RenderScale(f32);

impl Default for RenderScale {
    fn default() -> RenderScale {
        RenderScale(RenderScale::MAX)
    }
}

impl RenderScale {
    /// The floor. A tenth of each axis is a hundredth of the pixels; below that
    /// the frame stops being a picture of anything, and a scale of zero would
    /// ask for a zero-sized attachment.
    pub const MIN: f32 = 0.1;

    /// The ceiling. Supersampling is a different feature with different costs
    /// (and a different filter); this knob only ever goes down.
    pub const MAX: f32 = 1.0;

    /// The steps a host's "make it chunkier" key walks through, ascending.
    ///
    /// Godot's own resolution-scale presets, near enough: quarter, third, half,
    /// three-quarter, native. Any `f32` in `[MIN, MAX]` is legal — this is the
    /// list a UI offers, not a restriction on the value.
    pub const STEPS: [f32; 5] = [0.25, 1.0 / 3.0, 0.5, 0.75, 1.0];

    /// Clamp `scale` into `[MIN, MAX]`. A NaN becomes [`MAX`](RenderScale::MAX):
    /// the renderer allocates from this number, so "no opinion" has to resolve
    /// to the safe end rather than to a zero-sized texture.
    pub fn new(scale: f32) -> RenderScale {
        RenderScale(if scale.is_nan() {
            RenderScale::MAX
        } else {
            scale.clamp(RenderScale::MIN, RenderScale::MAX)
        })
    }

    pub fn get(self) -> f32 {
        self.0
    }

    /// Replace the value, clamped as [`new`](RenderScale::new).
    pub fn set(&mut self, scale: f32) {
        *self = RenderScale::new(scale);
    }

    /// Whether the scene is drawn at the host's own resolution — the one case
    /// with no internal target and no blit in the frame at all.
    pub fn is_native(self) -> bool {
        self.0 >= RenderScale::MAX
    }

    /// The internal target's size for a `width` × `height` host view.
    ///
    /// Round half up (`f32::round` is half-away-from-zero, and these are all
    /// positive), floor of 1 on each axis: a 3-pixel-wide viewport at 0.25 is
    /// one pixel wide, never zero. Exactly `(width, height)` at scale 1.0, which
    /// is what lets the renderer take the old path unchanged there.
    pub fn size(self, width: u32, height: u32) -> (u32, u32) {
        let scaled = |n: u32| ((n.max(1) as f32) * self.0).round().max(1.0) as u32;
        (scaled(width), scaled(height))
    }

    /// Move `delta` places along [`STEPS`](RenderScale::STEPS) — what a host's
    /// `[` / `]` pair calls.
    ///
    /// The first place moved is always to the next step *in the direction of
    /// travel*, so a value that is not on the ladder (`0.42` from a URL query
    /// or a config file) steps up to 0.5 and down to 1/3 rather than skipping
    /// one. Saturates at both ends rather than wrapping: mashing `[` on a phone
    /// should bottom out, not jump back to native.
    pub fn stepped(self, delta: i32) -> RenderScale {
        if delta == 0 {
            return self;
        }
        // Wide enough to absorb the f32 error in 1/3, narrow enough that no two
        // steps can be confused for each other.
        const EPS: f32 = 1e-4;
        let last = RenderScale::STEPS.len() as i32 - 1;
        let index = if delta > 0 {
            let first_above = RenderScale::STEPS
                .iter()
                .position(|s| *s > self.0 + EPS)
                .map_or(last, |i| i as i32);
            first_above + (delta - 1)
        } else {
            let last_below = RenderScale::STEPS
                .iter()
                .rposition(|s| *s < self.0 - EPS)
                .map_or(0, |i| i as i32);
            last_below + (delta + 1)
        };
        RenderScale::new(RenderScale::STEPS[index.clamp(0, last) as usize])
    }
}

/// Where the screen-space phase circle is, as the *world* sees it (D1, DESIGN
/// §5's [`PHASE_CIRCLE`] variant).
///
/// [`Renderer::set_phase_fx`] takes NDC and a radius, which is the right thing
/// for a host holding a viewport and the wrong thing for a game holding a
/// player: the projection needs the frame's own view-projection and aspect, and
/// neither exists until the frame is being drawn. So a game says *what* the
/// circle is centred on and *how big*, in units that survive a resize, and
/// [`Engine::render`] resolves it against the frame it is about to draw. That
/// is the same seam [`UiBatch`](crate::ui::UiBatch) uses, and it exists for the
/// same reason: the sim owns the intent, the renderer owns the pixels.
///
/// **A render value living in the world.** Nothing in the engine reads it
/// inside a tick, exactly like [`RenderScale`], so a `FixedSim` system may write
/// it every tick without any fingerprint being able to see it. Absent means
/// "no circle", which is the resting state and costs nothing.
///
/// [`PHASE_CIRCLE`]: crate::MaterialVariant::PHASE_CIRCLE
/// [`Renderer::set_phase_fx`]: crate::Renderer::set_phase_fx
/// [`Engine::render`]: crate::Engine::render
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct PhaseFx {
    /// The world point the disc is centred on.
    pub center: Vec3,
    /// How big, in the units [`cover`](PhaseFx::cover) selects. Zero is off.
    pub radius: f32,
    /// The edge fringe's strength, `0..1`.
    pub strength: f32,
    /// What [`radius`](PhaseFx::radius) is measured in.
    ///
    /// `false` — NDC-Y units, the same thing [`Renderer::set_phase_fx`] takes:
    /// an absolute size on screen, which is what a small fixed flourish wants.
    ///
    /// `true` — a fraction of the distance from the centre to the *farthest
    /// corner of this frame*, plus a hair of padding. So `1.0` covers the whole
    /// viewport from wherever the centre happens to be, at whatever aspect the
    /// window happens to have, and a game can animate `0 → 1` without ever
    /// computing a corner. The original does the corner search itself, once,
    /// when the phase begins; doing it per frame instead is both simpler and
    /// correct across a resize mid-transition.
    ///
    /// [`Renderer::set_phase_fx`]: crate::Renderer::set_phase_fx
    pub cover: bool,
}

impl Default for PhaseFx {
    fn default() -> PhaseFx {
        PhaseFx {
            center: Vec3::ZERO,
            radius: 0.0,
            strength: 0.0,
            cover: false,
        }
    }
}

/// Padding added to the farthest-corner distance under [`PhaseFx::cover`], in
/// NDC-Y units.
///
/// The original's `_max_radius += 0.1`, in its own units — fractions of the
/// viewport *height*, where NDC-Y spans two of those. Without it a circle that
/// exactly reaches the corner leaves the corner pixel itself on the boundary,
/// and the fringe (which is drawn *at* the boundary) frames the screen.
pub const PHASE_COVER_PADDING: f32 = 0.2;

/// Below this much clip `w`, the centre is pinned to the middle of the screen.
///
/// The original's `cam_dist < 1.0` first-person guard: a projection is unusable
/// when the point is all but inside the lens, and the answer that reads is
/// "centred", not "somewhere off the edge". `w` is the view-space depth for the
/// engine's perspective projection, so this is that distance, in metres, with no
/// camera pose needed on this side.
pub const PHASE_PIN_DISTANCE: f32 = 1.0;

/// Resolve a [`PhaseFx`] against the frame it will be drawn in: `(centre in
/// NDC, radius in NDC-Y units)`.
///
/// Three cases, and they are the original's
/// (`scripts/phase/phase_visual_effect.gd`) restated in clip space:
///
/// - **all but on the lens** (`|w| < `[`PHASE_PIN_DISTANCE`]) — pin to the
///   centre of the screen;
/// - **behind the camera** (`w < 0`) — mirror through the centre and clamp,
///   so a circle whose anchor is behind you slides off the correct edge instead
///   of jumping to the opposite one;
/// - otherwise the plain perspective divide.
///
/// Pure, and public, so the whole of it is an ordinary unit test.
pub fn project_phase_fx(view_proj: &Mat4, aspect: f32, fx: &PhaseFx) -> (Vec2, f32) {
    let clip = *view_proj * fx.center.extend(1.0);
    let w = clip.w;
    let center = if w.abs() < PHASE_PIN_DISTANCE {
        Vec2::ZERO
    } else if w < 0.0 {
        // The divide by a negative `w` already mirrors the point through the
        // origin, so this is the original's `1 - uv` written once. The clamp is
        // its `[-0.5, 1.5]` in UV, which is `[-2, 2]` in NDC.
        Vec2::new(clip.x / w, clip.y / w).clamp(Vec2::splat(-2.0), Vec2::splat(2.0))
    } else {
        Vec2::new(clip.x / w, clip.y / w)
    };
    let aspect = if aspect.is_finite() && aspect > 0.0 {
        aspect
    } else {
        1.0
    };
    let radius = if fx.cover {
        fx.radius * (farthest_corner(center, aspect) + PHASE_COVER_PADDING)
    } else {
        fx.radius
    };
    (center, radius.max(0.0))
}

/// Aspect-corrected distance from `center` to the farthest corner of the NDC
/// square — the radius at which a disc centred there covers the frame.
fn farthest_corner(center: Vec2, aspect: f32) -> f32 {
    let mut worst = 0.0f32;
    for corner in [
        Vec2::new(-1.0, -1.0),
        Vec2::new(1.0, -1.0),
        Vec2::new(-1.0, 1.0),
        Vec2::new(1.0, 1.0),
    ] {
        let mut d = corner - center;
        d.x *= aspect;
        worst = worst.max(d.length());
    }
    worst
}

/// Which scene generator an entity's geometry came from.
///
/// [`MeshRef`] is a content hash and stays one — it is the renderer's key and it
/// must not grow a provenance field the render path would have to skip past.
/// This is the other half: the *inputs* that produced that hash, so a later
/// quality change or editor param tweak can regenerate an entity without
/// reloading the scene, and so `save_scene` knows which generator entry an
/// entity belongs to.
#[derive(Component, Clone, Debug, PartialEq, Eq, Hash)]
pub struct GeneratorRef {
    /// The scene file's generator entry name.
    pub name: String,
    /// `GeneratorSpec::param_key(quality)` for the spec that ran — the layer-A
    /// cache key, so regeneration is a cache lookup away.
    pub param_key: u64,
}

/// The analytic terrain surface an entity renders (DESIGN §9).
///
/// **This is the seam physics uses.** Step 5's ball integrator queries
/// `(&TerrainSurface, &Transform)` and calls the `*_world` methods below; it
/// never looks at the mesh, the `MeshRef`, or the tessellation. The mesh on the
/// same entity is a *view* of this field, so the two cannot disagree at any
/// quality tier.
///
/// v1 assumes a terrain entity is translated only — no rotation, no scale — so
/// world↔field is a subtraction. A rotated heightfield is not a heightfield in
/// world space, and pretending otherwise is how "the ball fell through the
/// terrain" bugs start; the loader asserts it in debug builds instead.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct TerrainSurface {
    /// The pure field. Sample it directly for anything that is not tied to this
    /// entity's placement.
    pub field: HeightField,
    /// World extent of the meshed patch on X and Z, centered on the entity.
    /// Outside it the field is still defined; there is simply nothing drawn.
    pub size: Vec2,
}

impl TerrainSurface {
    pub fn new(params: &TerrainParams) -> TerrainSurface {
        TerrainSurface {
            field: params.field(),
            size: params.size,
        }
    }

    /// Field-space coordinates for a world point, given the entity's origin.
    #[inline]
    pub fn to_local(origin: Vec3, x: f32, z: f32) -> Vec2 {
        Vec2::new(x - origin.x, z - origin.z)
    }

    /// World-space surface height under `(x, z)`.
    #[inline]
    pub fn height_world(&self, origin: Vec3, x: f32, z: f32) -> f32 {
        let p = TerrainSurface::to_local(origin, x, z);
        origin.y + self.field.height(p.x, p.y)
    }

    /// World-space slope `(∂h/∂x, ∂h/∂z)`. Translation does not affect it.
    #[inline]
    pub fn gradient_world(&self, origin: Vec3, x: f32, z: f32) -> Vec2 {
        let p = TerrainSurface::to_local(origin, x, z);
        self.field.gradient(p.x, p.y)
    }

    /// World-space unit surface normal.
    #[inline]
    pub fn normal_world(&self, origin: Vec3, x: f32, z: f32) -> Vec3 {
        runt_mesh::terrain::normal_from_gradient(self.gradient_world(origin, x, z))
    }

    /// Height and gradient together — one field evaluation, which is what a
    /// contact solve wants.
    #[inline]
    pub fn sample_world(&self, origin: Vec3, x: f32, z: f32) -> (f32, Vec2) {
        let p = TerrainSurface::to_local(origin, x, z);
        let (h, g) = self.field.sample(p.x, p.y);
        (origin.y + h, g)
    }

    /// Whether `(x, z)` falls inside the meshed patch.
    pub fn contains_world(&self, origin: Vec3, x: f32, z: f32) -> bool {
        let p = TerrainSurface::to_local(origin, x, z);
        p.x.abs() <= self.size.x * 0.5 && p.y.abs() <= self.size.y * 0.5
    }
}

/// The scene's light rig, uploaded verbatim into the per-frame uniform
/// (DESIGN §5): one directional key light plus a sky/ground hemisphere ambient.
///
/// The same three ambient colors are also what the background gradient is drawn
/// from (see [`crate::sky`]), so a scene has one set of numbers describing its
/// environment rather than two that can drift apart: brighten the sky ambient
/// and the sky itself brightens with it.
///
/// A resource, not a component, because v1 has exactly one rig; when a scene
/// wants more it becomes a component on a light entity and this stays as the
/// fallback.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct Lighting {
    /// Direction *towards* the key light. Normalized in the shader.
    pub key_dir: Vec3,
    pub key_color: Vec3,
    /// Ambient seen by upward-facing normals, and the background at the zenith.
    pub sky_color: Vec3,
    /// Ambient seen by downward-facing normals, and the background at the nadir.
    pub ground_color: Vec3,
    /// The background color where the view ray is horizontal. `None` — the
    /// default, and what every scene file written before the sky existed parses
    /// to — is the midpoint of sky and ground, so an old rig gains a background
    /// without gaining a decision. See [`Lighting::horizon`].
    pub horizon: Option<Vec3>,
    /// How much of the sky the drifting cloud layer covers, `0..1`. **`0` — the
    /// default — is no cloud pass at all**, which is what keeps the three-stop
    /// gradient exactly the picture it was before clouds existed.
    ///
    /// A single number rather than the original's eight (`cloud_scale`,
    /// `_density`, `_softness`, `_brightness`, `_height`, `_flatten`, `_speed`,
    /// `_direction`): those are one authored *look*, and a scene that wants a
    /// different one wants a different shader, not seven more RON fields. The
    /// look is `sky.wgsl`'s constants, which are `simple_sky.gdshader`'s
    /// defaults.
    pub clouds: f32,
    /// The angular radius of the sun disk, as `1 − cos θ` — the original's
    /// `sun_disk_size`, in its units. **`0` is no disk.**
    ///
    /// Drawn in [`key_color`](Lighting::key_color) at
    /// [`key_dir`](Lighting::key_dir), so a scene cannot end up with a sun in
    /// one place and its shadows coming from another.
    pub sun: f32,
}

impl Default for Lighting {
    /// The pre-material look, restated as a rig: the same key direction, and an
    /// ambient that averages to the flat 0.25 term the old shader used, split
    /// into a cool sky and a warm-dark ground.
    fn default() -> Lighting {
        Lighting {
            key_dir: Vec3::new(0.4, 1.0, 0.6),
            key_color: Vec3::new(0.74, 0.72, 0.68),
            sky_color: Vec3::new(0.30, 0.33, 0.40),
            ground_color: Vec3::new(0.14, 0.13, 0.12),
            horizon: None,
            // Off: the sky stays the three-stop gradient it has always been
            // unless a scene asks for weather.
            clouds: 0.0,
            sun: 0.0,
        }
    }
}

impl Lighting {
    /// The resolved horizon color: whatever the rig says, or the sky/ground
    /// midpoint when it says nothing.
    #[inline]
    pub fn horizon(&self) -> Vec3 {
        self.horizon.unwrap_or_else(|| default_horizon(self.sky_color, self.ground_color))
    }
}

/// The horizon color an unspecified [`Lighting::horizon`] resolves to.
///
/// A plain midpoint: it cannot be brighter than the brightest ambient (so no rig
/// acquires a glow it did not ask for) and it is a pure function of two numbers
/// the scene already had, which is what makes adding the field a non-event for
/// existing files.
#[inline]
pub fn default_horizon(sky: Vec3, ground: Vec3) -> Vec3 {
    (sky + ground) * 0.5
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/// Local transform, TRS. Applied scale → rotation → translation.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Transform {
        Transform::IDENTITY
    }
}

impl Transform {
    pub const IDENTITY: Transform = Transform {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    pub fn from_translation(translation: Vec3) -> Transform {
        Transform {
            translation,
            ..Transform::IDENTITY
        }
    }

    pub fn from_rotation(rotation: Quat) -> Transform {
        Transform {
            rotation,
            ..Transform::IDENTITY
        }
    }

    pub fn from_scale(scale: Vec3) -> Transform {
        Transform {
            scale,
            ..Transform::IDENTITY
        }
    }

    /// A transform at `eye` oriented so that local −Z points at `target` — the
    /// camera convention, and the exact inverse of `Mat4::look_at_rh`.
    pub fn looking_at(eye: Vec3, target: Vec3, up: Vec3) -> Transform {
        Transform {
            translation: eye,
            rotation: crate::camera::look_rotation(eye, target, up),
            scale: Vec3::ONE,
        }
    }

    /// The 4×4 model matrix for this transform.
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}

/// World-space transform, produced from [`Transform`] by `propagate_transforms`.
///
/// There is no hierarchy yet (DESIGN §3: flat by default), so propagation is the
/// identity — but the component exists now so that adding `ChildOf` later is a
/// change to one system, not to every consumer.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct GlobalTransform(pub Mat4);

impl Default for GlobalTransform {
    fn default() -> GlobalTransform {
        GlobalTransform(Mat4::IDENTITY)
    }
}

/// The previous tick's transform, for render interpolation (DESIGN §4).
///
/// Written by `snapshot_interpolation` at the top of each tick. At render time
/// `Interpolated` is tick *N-1* and [`Transform`] is tick *N*; the renderer
/// blends between them with `alpha ∈ [0,1)`.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Interpolated {
    pub prev_translation: Vec3,
    pub prev_rotation: Quat,
    pub prev_scale: Vec3,
}

impl Default for Interpolated {
    fn default() -> Interpolated {
        Interpolated::from(&Transform::IDENTITY)
    }
}

impl From<&Transform> for Interpolated {
    fn from(t: &Transform) -> Interpolated {
        Interpolated {
            prev_translation: t.translation,
            prev_rotation: t.rotation,
            prev_scale: t.scale,
        }
    }
}

impl Interpolated {
    /// Blend towards `current` and build the model matrix the renderer draws
    /// with. `alpha` is the fraction of a tick elapsed since the last tick.
    pub fn blend(&self, current: &Transform, alpha: f32) -> Mat4 {
        let alpha = alpha.clamp(0.0, 1.0);
        Mat4::from_scale_rotation_translation(
            self.prev_scale.lerp(current.scale, alpha),
            self.prev_rotation.slerp(current.rotation, alpha),
            self.prev_translation.lerp(current.translation, alpha),
        )
    }
}

/// Which geometry an entity draws (DESIGN §3, §5).
///
/// A content hash, not a pointer: identical generated meshes collapse onto one
/// handle and therefore one pair of GPU buffers. Step 4 (§6) grows this into
/// "hash + generator params ref" without the render path noticing.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MeshRef(pub MeshHandle);

/// Whether an entity is drawn at all (DESIGN §3's core component set).
///
/// **Absent means visible.** That is the whole design: nothing that exists
/// today has to acquire a component to keep being rendered, a game only pays
/// for the entities it actually hides, and there is no "default visibility"
/// question to get wrong. [`draw::extract_draw_list`] drops a hidden entity
/// before it computes anything for it — no interpolated matrix, no variant
/// resolution, no instance slot, no frustum test.
///
/// It is *sim* state, not render state, and deliberately so: a game toggles it
/// from a `FixedSim` system like any other component, and two runs of the same
/// replay hide the same things at the same ticks. That is the difference
/// between this and [`RenderScale`] — hiding an object is a decision the world
/// makes, and the world is allowed to remember making it.
///
/// It replaces the parking trick (translating something to `y = -1000` to get
/// it off screen), which cost a draw, a matrix and a frustum test to achieve
/// nothing, and which lied to every system that read a transform.
///
/// [`draw::extract_draw_list`]: crate::draw::extract_draw_list
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Visibility {
    pub visible: bool,
}

impl Default for Visibility {
    /// Visible — the same answer as having no `Visibility` at all, so adding
    /// the component with its default can never change what a frame looks like.
    fn default() -> Visibility {
        Visibility::VISIBLE
    }
}

impl Visibility {
    pub const VISIBLE: Visibility = Visibility { visible: true };
    pub const HIDDEN: Visibility = Visibility { visible: false };

    pub fn new(visible: bool) -> Visibility {
        Visibility { visible }
    }

    /// Flip it, returning the new state — what a toggle system wants.
    pub fn toggle(&mut self) -> bool {
        self.visible = !self.visible;
        self.visible
    }
}

/// Demo-only: constant-rate rotation about a local axis.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Spin {
    pub axis: Vec3,
    pub rad_per_sec: f32,
}

/// Marks entities spawned by a scene file.
///
/// Named for the demo it was introduced with; it means "came from the loaded
/// scene" now. Entities spawned by gameplay code carry no such marker and are
/// (deliberately) not saved — see [`crate::scene::save_scene`].
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct DemoScene;

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// `PostSim`: copy the current transform into `Interpolated`.
///
/// Ordering is the whole trick. This runs *before* `FixedSim` mutates anything,
/// so what it captures is the value tick *N-1* left behind. Doing it after the
/// sim instead would make `prev == current` and interpolation a no-op.
pub fn snapshot_interpolation(mut q: Query<(&Transform, &mut Interpolated)>) {
    for (transform, mut interp) in &mut q {
        *interp = Interpolated::from(transform);
    }
}

/// `FixedSim`: advance every [`Spin`] by exactly one tick's worth of rotation.
///
/// No delta-time parameter — ticks are uniform by definition, so the step is
/// `rad_per_sec * dt` with `dt` the *configured* tick length, never a measured
/// one. Renormalising each tick keeps repeated quaternion products from drifting
/// off the unit sphere; it is a pure function of the previous value, so it costs
/// nothing in determinism.
pub fn spin(tick: Res<FixedTick>, mut q: Query<(&Spin, &mut Transform)>) {
    for (spin, mut transform) in &mut q {
        let step = Quat::from_axis_angle(spin.axis.normalize(), spin.rad_per_sec * tick.dt_secs);
        transform.rotation = (step * transform.rotation).normalize();
    }
}

/// `FixedSim` (tail): local transform → world transform.
///
/// Identity propagation for now; see [`GlobalTransform`].
pub fn propagate_transforms(mut q: Query<(&Transform, &mut GlobalTransform)>) {
    for (transform, mut global) in &mut q {
        *global = GlobalTransform(transform.matrix());
    }
}

/// `FixedSim` (tail): bump the tick counter. Last system in the chain, so
/// `TickCount` reads as "ticks fully completed".
pub fn advance_tick_count(mut count: ResMut<TickCount>) {
    count.0 += 1;
}

// ---------------------------------------------------------------------------
// Schedule construction
// ---------------------------------------------------------------------------

/// Build a schedule that is single-threaded and explicitly ordered. Every runt
/// schedule goes through here so no schedule can accidentally acquire ambiguous
/// parallel ordering (DESIGN §3).
fn deterministic_schedule(label: impl ScheduleLabel) -> Schedule {
    let mut schedule = Schedule::new(label);
    schedule.set_executor(bevy_ecs::schedule::SingleThreadedExecutor::new());
    schedule
}

/// `Startup`: load whatever scene the [`Sim`](crate::sim::Sim) was configured
/// with (DESIGN §6 — the scene file *is* the content).
///
/// Exclusive, because resolving generators needs `GenCache` and `MeshLibrary`
/// mutably at the same time and spawning needs the world itself. There is
/// exactly one startup system and it runs once, so nothing is lost by not
/// parallelizing it.
pub fn startup_schedule() -> Schedule {
    let mut s = deterministic_schedule(Startup);
    s.add_systems(crate::scene::load_pending_scene);
    s
}

/// A `Startup` that does nothing: the code-path fallback for tests that want a
/// world with no content in it.
pub fn empty_startup_schedule() -> Schedule {
    deterministic_schedule(Startup)
}

pub fn post_sim_schedule() -> Schedule {
    let mut s = deterministic_schedule(PostSim);
    s.add_systems(snapshot_interpolation);
    s
}

/// The tick, in order (DESIGN §3: explicitly chained, never ambiguous).
///
/// ```text
/// update_overlap_messages   advance the OverlapEvent double buffer (§9)
/// spin                      demo-only constant rotation
/// integrate_balls           input + gravity + terrain contact (§9)
/// resolve_overlaps          discrete shapes → events + push-out (§9)
/// roll_spin                 cosmetic ball rotation, reads velocity only (§9)
/// follow_camera             cameras chase where things ended up
/// propagate_transforms      local → world
/// flush_audio               this tick's sound leaves, as one batch (§8)
/// advance_tick_count        the tick is now complete
/// ```
///
/// Why that order: the message swap goes first so an event written this tick
/// survives into the next one intact (see [`OverlapEvent`](crate::OverlapEvent));
/// the integrator produces a position before the overlap pass corrects it;
/// `roll_spin` reads the *final* velocity; cameras follow *after* the things they
/// follow have moved, so a follow is never a tick behind its target; audio
/// flushes after the camera has settled (so a pan is computed against this
/// tick's pose) and before the tick counter turns over (so an event is stamped
/// with the index of the tick that produced it) — see
/// [`crate::audio`] for the full argument.
///
/// Every physics system is a no-op on a world with no `Ball` and no collider —
/// `assets/demo.ron` has neither, and `tests/physics.rs` pins its tick output to
/// the value it had before any of this existed.
pub fn fixed_sim_schedule() -> Schedule {
    let mut s = deterministic_schedule(FixedSim);
    s.add_systems(
        (
            crate::physics::update_overlap_messages,
            spin,
            crate::physics::integrate_balls,
            crate::physics::resolve_overlaps,
            crate::physics::roll_spin,
            crate::camera::follow_camera,
            propagate_transforms,
            crate::audio::flush_audio,
            advance_tick_count,
        )
            .chain(),
    );
    s
}

#[cfg(test)]
mod phase_fx_tests {
    use super::*;

    /// A perspective camera at `eye` looking at the origin, as the engine's own
    /// `Camera` builds one — so this tests the projection the frame is really
    /// drawn with rather than a hand-rolled matrix that resembles it.
    fn view_proj(eye: Vec3, aspect: f32) -> Mat4 {
        let camera = crate::camera::Camera::default();
        camera.view_proj(Transform::looking_at(eye, Vec3::ZERO, Vec3::Y).matrix(), aspect)
    }

    fn fx(center: Vec3, radius: f32, cover: bool) -> PhaseFx {
        PhaseFx {
            center,
            radius,
            strength: 1.0,
            cover,
        }
    }

    #[test]
    fn a_point_in_front_projects_where_the_camera_sees_it() {
        let vp = view_proj(Vec3::new(0.0, 0.0, 8.0), 16.0 / 9.0);
        // Dead centre of the frame.
        let (c, r) = project_phase_fx(&vp, 16.0 / 9.0, &fx(Vec3::ZERO, 0.3, false));
        assert!(c.abs_diff_eq(Vec2::ZERO, 1e-5), "{c:?}");
        assert_eq!(r, 0.3, "an absolute radius passes through untouched");

        // Up and to the right of it.
        let (c, _) = project_phase_fx(&vp, 16.0 / 9.0, &fx(Vec3::new(1.0, 1.0, 0.0), 0.0, false));
        assert!(c.x > 0.0 && c.y > 0.0, "{c:?}");
    }

    #[test]
    fn a_point_behind_the_camera_is_mirrored_and_clamped() {
        let vp = view_proj(Vec3::new(0.0, 0.0, 8.0), 16.0 / 9.0);
        // Well behind the eye, and off to the +X side of the world. The point
        // has no honest place on screen; what it must not do is land in the
        // middle of the frame as if it were in front.
        let behind = Vec3::new(4.0, 0.0, 40.0);
        let (c, _) = project_phase_fx(&vp, 16.0 / 9.0, &fx(behind, 0.0, false));
        assert!(c.x <= 2.0 && c.x >= -2.0 && c.y <= 2.0 && c.y >= -2.0, "{c:?}");
        assert!(c.x < 0.0, "a point to the right and behind mirrors left: {c:?}");
    }

    #[test]
    fn a_point_all_but_on_the_lens_pins_to_the_centre() {
        let vp = view_proj(Vec3::new(0.0, 0.0, 8.0), 16.0 / 9.0);
        // Half a metre in front of the eye and a long way off-axis: the divide
        // would fling it far outside the frame, and "centred" is the answer
        // that reads.
        let (c, _) = project_phase_fx(&vp, 16.0 / 9.0, &fx(Vec3::new(3.0, 0.0, 7.5), 0.0, false));
        assert_eq!(c, Vec2::ZERO);
    }

    #[test]
    fn a_cover_radius_of_one_reaches_every_corner_at_any_aspect() {
        for aspect in [1.0, 16.0 / 9.0, 0.5] {
            let vp = view_proj(Vec3::new(0.0, 0.0, 8.0), aspect);
            for center in [Vec3::ZERO, Vec3::new(2.0, 1.5, 0.0)] {
                let full = fx(center, 1.0, true);
                let (c, r) = project_phase_fx(&vp, aspect, &full);
                // Every corner of the NDC square, aspect-corrected exactly as
                // `shader.wgsl`'s `phase_distance` does it, is inside.
                for corner in [
                    Vec2::new(-1.0, -1.0),
                    Vec2::new(1.0, -1.0),
                    Vec2::new(-1.0, 1.0),
                    Vec2::new(1.0, 1.0),
                ] {
                    let mut d = corner - c;
                    d.x *= aspect;
                    assert!(
                        d.length() < r,
                        "corner {corner:?} outside a full-cover circle \
                         (aspect {aspect}, centre {c:?}, radius {r})"
                    );
                }
                // …and it is not wildly bigger than it needs to be: the padding
                // is the whole of the slack.
                assert!(r < 4.0, "{r}");
            }
        }
    }

    #[test]
    fn a_cover_radius_of_zero_is_off_and_a_negative_one_cannot_happen() {
        let vp = view_proj(Vec3::new(0.0, 0.0, 8.0), 1.0);
        assert_eq!(project_phase_fx(&vp, 1.0, &fx(Vec3::ZERO, 0.0, true)).1, 0.0);
        assert_eq!(project_phase_fx(&vp, 1.0, &fx(Vec3::ZERO, -1.0, false)).1, 0.0);
        // A host that has not sized its window yet must not produce a NaN
        // radius that would make every comparison in the shader false.
        assert!(project_phase_fx(&vp, 0.0, &fx(Vec3::ZERO, 1.0, true)).1.is_finite());
        assert!(project_phase_fx(&vp, f32::NAN, &fx(Vec3::ZERO, 1.0, true))
            .1
            .is_finite());
    }

    #[test]
    fn the_default_is_the_resting_state() {
        let fx = PhaseFx::default();
        assert_eq!(fx.radius, 0.0);
        assert_eq!(
            project_phase_fx(&Mat4::IDENTITY, 1.0, &fx),
            (Vec2::ZERO, 0.0)
        );
    }
}
