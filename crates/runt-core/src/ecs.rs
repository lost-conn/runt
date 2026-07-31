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
use glam::{Mat4, Quat, Vec3};

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

/// The one entity carrying the merged demo-scene buffer. Placeholder until
/// per-entity draws land (DESIGN §12 step 3), at which point this goes away in
/// favour of `MeshRef` on many entities.
#[derive(Resource, Clone, Copy, Debug)]
pub struct DemoEntity(pub Entity);

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

/// Demo-only: constant-rate rotation about a local axis.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Spin {
    pub axis: Vec3,
    pub rad_per_sec: f32,
}

/// Marks the single entity that owns the merged demo-scene mesh.
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

/// `Startup`: spawn the merged demo scene as a single spinning entity.
///
/// 0.4 rad/s about +Y — the rate the pre-ECS renderer hardcoded, kept so the
/// screenshot test compares like with like.
pub fn spawn_demo_scene(mut commands: Commands) {
    let entity = commands
        .spawn((
            DemoScene,
            Transform::IDENTITY,
            GlobalTransform::default(),
            Interpolated::default(),
            Spin {
                axis: Vec3::Y,
                rad_per_sec: 0.4,
            },
        ))
        .id();
    commands.insert_resource(DemoEntity(entity));
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

pub fn startup_schedule() -> Schedule {
    let mut s = deterministic_schedule(Startup);
    s.add_systems(spawn_demo_scene);
    s
}

pub fn post_sim_schedule() -> Schedule {
    let mut s = deterministic_schedule(PostSim);
    s.add_systems(snapshot_interpolation);
    s
}

pub fn fixed_sim_schedule() -> Schedule {
    let mut s = deterministic_schedule(FixedSim);
    s.add_systems((spin, propagate_transforms, advance_tick_count).chain());
    s
}
