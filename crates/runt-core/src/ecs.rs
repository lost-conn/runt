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
