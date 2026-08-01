//! The fixed-tick simulation (DESIGN §4).
//!
//! [`Sim`] is the whole engine minus the GPU: world, schedules, tick
//! accumulator and input buffer. [`Engine`](crate::Engine) is `Sim` + a
//! [`Renderer`](crate::Renderer). Keeping them separable is what lets the
//! determinism tests run with no adapter at all.
//!
//! **The engine never reads a clock.** The host passes monotonically increasing
//! wall time to [`Sim::update`]; everything downstream is a function of that
//! number and the input trace. That is the property replays are built on.

use bevy_ecs::prelude::*;
use bevy_ecs::query::QueryState;
use glam::Mat4;

use crate::cache::{CacheStore, GenCache, NoopCache};
use crate::camera::Camera;
use crate::draw::{self, DrawItem, DrawQuery, FrameParams};
use crate::ecs::{
    self, DemoEntity, FixedTick, Interpolated, Lighting, QualityTier, TickCount, Transform,
};
use crate::input::{Input, InputEvent};
use crate::registry::MeshLibrary;
use crate::scene::{PendingScene, DEMO_SCENE_RON};

/// The camera components the render path reads.
type CameraQuery = (
    Entity,
    &'static Camera,
    &'static Transform,
    Option<&'static Interpolated>,
);

/// The tick length DESIGN §4 fixes: 1/60 s exactly.
pub const TICK_DT: f64 = 1.0 / 60.0;

/// Spiral-of-death guard (DESIGN §4). If a host stalls for longer than this,
/// the surplus is *discarded* rather than simulated: a device below the floor
/// runs in slow motion instead of freezing while it tries to catch up.
pub const MAX_ACCUMULATED: f64 = 0.25;

/// Largest `f32` strictly below 1.0 — `alpha` is documented as `[0,1)` and
/// callers are entitled to rely on it.
const ALPHA_MAX: f32 = 1.0 - f32::EPSILON;

/// Everything a [`Sim`] needs before its first tick.
///
/// A struct rather than five constructors because the axes are independent:
/// tick rate is a §4 concern, quality and cache are §6 concerns, and the scene
/// is content. Tests reach for exactly one of them at a time.
pub struct SimConfig {
    /// Seconds per tick. See [`Sim::with_tick_rate`].
    pub tick_dt: f64,
    /// Device/LOD multiplier applied to every generator at load (DESIGN §6).
    pub quality: QualityTier,
    /// Persistence for the generation cache. The default stores nothing, so
    /// building a `Sim` never touches a filesystem — a host opts into
    /// [`cache::platform_default`](crate::cache::platform_default).
    pub cache: Box<dyn CacheStore>,
    /// Scene RON to load during `Startup`, or `None` for an empty world.
    pub scene: Option<String>,
}

impl Default for SimConfig {
    fn default() -> SimConfig {
        SimConfig {
            tick_dt: TICK_DT,
            quality: QualityTier::default(),
            cache: Box::new(NoopCache),
            scene: Some(DEMO_SCENE_RON.to_string()),
        }
    }
}

impl SimConfig {
    pub fn with_tick_dt(mut self, tick_dt: f64) -> SimConfig {
        self.tick_dt = tick_dt;
        self
    }

    pub fn with_tick_rate(mut self, hz: f64) -> SimConfig {
        assert!(hz > 0.0, "tick rate must be positive, got {hz}");
        self.tick_dt = 1.0 / hz;
        self
    }

    pub fn with_quality(mut self, quality: f32) -> SimConfig {
        self.quality = QualityTier(quality);
        self
    }

    pub fn with_cache(mut self, cache: Box<dyn CacheStore>) -> SimConfig {
        self.cache = cache;
        self
    }

    pub fn with_scene(mut self, scene: impl Into<String>) -> SimConfig {
        self.scene = Some(scene.into());
        self
    }

    /// No scene at all: the code-path fallback for tests that want a world they
    /// populate themselves.
    pub fn without_scene(mut self) -> SimConfig {
        self.scene = None;
        self
    }
}

pub struct Sim {
    world: World,
    startup: Schedule,
    post_sim: Schedule,
    fixed_sim: Schedule,

    tick_dt: f64,
    /// Wall time of the first `update` call — the origin all sim time is
    /// measured from, so the host's clock epoch never matters.
    origin: Option<f64>,
    /// Wall time already accounted for: ticks run, plus any backlog dropped by
    /// the clamp. Recomputing the outstanding time from `elapsed - origin -
    /// consumed` on every call (rather than accumulating host deltas) is what
    /// makes the tick count a pure function of the elapsed value the host
    /// passes, independent of how it chopped the interval up.
    consumed: f64,
    alpha: f32,
    pending: Vec<InputEvent>,

    /// Cached query states for the render-side reads. Built once, updated for
    /// new archetypes on use — rebuilding them per frame would be pure waste in
    /// the one place that runs every frame.
    draw_query: QueryState<DrawQuery>,
    camera_query: QueryState<CameraQuery>,
}

impl Sim {
    /// A sim at the standard 60 Hz tick, with `Startup` already run.
    pub fn new() -> Sim {
        Sim::with_tick_dt(TICK_DT)
    }

    /// A sim at `hz` ticks per second. Exists for DESIGN §12 step 2's
    /// tick-rate toggle: dropping to 10 Hz makes render interpolation visible
    /// (and testable) instead of a 16 ms detail.
    pub fn with_tick_rate(hz: f64) -> Sim {
        Sim::from_config(SimConfig::default().with_tick_rate(hz))
    }

    /// A sim with an explicit tick length in seconds.
    pub fn with_tick_dt(tick_dt: f64) -> Sim {
        Sim::from_config(SimConfig::default().with_tick_dt(tick_dt))
    }

    /// A sim with no scene loaded: an empty world with the standard resources
    /// and schedules. The code-path fallback for tests.
    pub fn without_scene() -> Sim {
        Sim::from_config(SimConfig::default().without_scene())
    }

    /// The general constructor. Resources go in, `Startup` runs (which is where
    /// the scene is generated and spawned), and the sim is ready to tick.
    pub fn from_config(config: SimConfig) -> Sim {
        let SimConfig {
            tick_dt,
            quality,
            cache,
            scene,
        } = config;
        assert!(
            tick_dt > 0.0 && tick_dt.is_finite(),
            "tick_dt must be positive and finite, got {tick_dt}"
        );

        let mut world = World::new();
        world.insert_resource(FixedTick {
            dt_secs: tick_dt as f32,
        });
        world.insert_resource(TickCount::default());
        world.insert_resource(Input::new());
        world.insert_resource(MeshLibrary::new());
        world.insert_resource(Lighting::default());
        world.insert_resource(quality);
        world.insert_resource(GenCache::new(cache));
        world.insert_resource(PendingScene(scene));

        let draw_query = world.query::<DrawQuery>();
        let camera_query = world.query::<CameraQuery>();

        let mut sim = Sim {
            world,
            startup: ecs::startup_schedule(),
            post_sim: ecs::post_sim_schedule(),
            fixed_sim: ecs::fixed_sim_schedule(),
            tick_dt,
            origin: None,
            consumed: 0.0,
            alpha: 0.0,
            pending: Vec::new(),
            draw_query,
            camera_query,
        };
        sim.startup.run(&mut sim.world);
        sim
    }

    // -- accessors ----------------------------------------------------------

    pub fn world(&self) -> &World {
        &self.world
    }

    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    /// Seconds per tick.
    pub fn tick_dt(&self) -> f64 {
        self.tick_dt
    }

    /// Ticks completed since construction.
    pub fn tick_count(&self) -> u64 {
        self.world.resource::<TickCount>().0
    }

    /// Fraction of a tick elapsed since the last one, in `[0, 1)`. Valid after
    /// the most recent [`update`](Sim::update).
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Sim time outstanding, not yet turned into a tick.
    pub fn pending_time(&self) -> f64 {
        self.alpha as f64 * self.tick_dt
    }

    /// The demo's spinning entity (the twisted box).
    ///
    /// Panics if the loaded scene has nothing to focus on; use
    /// [`try_demo_entity`](Sim::try_demo_entity) on a world built without one.
    pub fn demo_entity(&self) -> Entity {
        self.world.resource::<DemoEntity>().0
    }

    /// As [`demo_entity`](Sim::demo_entity), but `None` on a scene-less world.
    pub fn try_demo_entity(&self) -> Option<Entity> {
        self.world.get_resource::<DemoEntity>().map(|d| d.0)
    }

    /// A scene entity by its `name` in the scene file.
    pub fn scene_entity(&self, name: &str) -> Option<Entity> {
        self.world
            .get_resource::<crate::scene::LoadedScene>()?
            .entity(name)
    }

    /// The generation cache's counters (DESIGN §6). Diagnostics only — nothing
    /// in the engine may branch on them.
    pub fn cache_stats(&self) -> crate::cache::CacheStats {
        self.world
            .get_resource::<GenCache>()
            .map(|c| c.stats())
            .unwrap_or_default()
    }

    /// The session's quality multiplier.
    pub fn quality_tier(&self) -> QualityTier {
        self.world
            .get_resource::<QualityTier>()
            .copied()
            .unwrap_or_default()
    }

    // -- input --------------------------------------------------------------

    /// Buffer a host input event. Buffered events are applied at the next tick
    /// boundary and nowhere else (DESIGN §4).
    pub fn push_input(&mut self, event: InputEvent) {
        self.pending.push(event);
    }

    /// Number of events waiting for the next tick boundary.
    pub fn pending_input_len(&self) -> usize {
        self.pending.len()
    }

    /// Read the per-tick input state (as `FixedSim` systems see it).
    pub fn input(&self) -> &Input {
        self.world.resource::<Input>()
    }

    // -- the loop -----------------------------------------------------------

    /// Advance the sim to `elapsed_seconds` of host wall time and return how
    /// many ticks ran.
    ///
    /// `elapsed_seconds` must be monotonically increasing; the first call
    /// establishes the origin and never ticks. Time that goes backwards is
    /// ignored rather than trusted (a host with a jumpy clock must not be able
    /// to rewind the sim).
    pub fn update(&mut self, elapsed_seconds: f64) -> u32 {
        // A NaN or infinite time would poison the origin permanently, freezing
        // the sim for good. Refuse it outright rather than latch it.
        if !elapsed_seconds.is_finite() {
            self.alpha = 0.0;
            return 0;
        }

        let origin = *self.origin.get_or_insert(elapsed_seconds);

        let mut outstanding = (elapsed_seconds - origin) - self.consumed;
        if !outstanding.is_finite() || outstanding < 0.0 {
            outstanding = 0.0;
        }
        // Never clamp below one tick, or a sim slower than 4 Hz could never
        // tick at all.
        let backlog_cap = MAX_ACCUMULATED.max(self.tick_dt);
        if outstanding > backlog_cap {
            // Drop the backlog: charge it to `consumed` without simulating it.
            self.consumed += outstanding - backlog_cap;
            outstanding = backlog_cap;
        }

        let mut ticks = 0u32;
        while outstanding >= self.tick_dt {
            outstanding -= self.tick_dt;
            self.consumed += self.tick_dt;
            self.tick();
            ticks += 1;
        }

        let raw = (outstanding / self.tick_dt) as f32;
        self.alpha = if raw.is_finite() {
            raw.clamp(0.0, ALPHA_MAX)
        } else {
            0.0
        };
        ticks
    }

    /// One tick. Input is drained first (so every system in the tick sees the
    /// same input snapshot), then `PostSim` captures the outgoing transforms
    /// for interpolation, then `FixedSim` produces the new ones.
    pub fn tick(&mut self) {
        let drained: Vec<InputEvent> = self.pending.drain(..).collect();
        self.world.resource_mut::<Input>().begin_tick(drained);

        self.post_sim.run(&mut self.world);
        self.fixed_sim.run(&mut self.world);
    }

    // -- render-side reads --------------------------------------------------

    /// The interpolated model matrix for `entity` at the current `alpha`.
    ///
    /// Returns the un-interpolated transform if the entity has no
    /// [`Interpolated`], and `None` if it has no [`Transform`] at all.
    pub fn model_matrix(&self, entity: Entity) -> Option<Mat4> {
        self.model_matrix_at(entity, self.alpha)
    }

    /// As [`model_matrix`](Sim::model_matrix) but at an explicit alpha — the
    /// interpolation tests drive this directly.
    pub fn model_matrix_at(&self, entity: Entity, alpha: f32) -> Option<Mat4> {
        let transform = self.world.get::<Transform>(entity)?;
        Some(match self.world.get::<Interpolated>(entity) {
            Some(interp) => interp.blend(transform, alpha),
            None => transform.matrix(),
        })
    }

    /// The demo spinner's interpolated model matrix. Convenience for tests; the
    /// render path goes through [`draw_list`](Sim::draw_list) like everything
    /// else.
    pub fn demo_model_matrix(&self) -> Mat4 {
        self.model_matrix(self.demo_entity())
            .unwrap_or(Mat4::IDENTITY)
    }

    /// Generated geometry, keyed by content hash. The renderer uploads from
    /// here on demand.
    pub fn mesh_library(&self) -> &MeshLibrary {
        self.world.resource::<MeshLibrary>()
    }

    /// The scene's light rig.
    pub fn lighting(&self) -> Lighting {
        *self.world.resource::<Lighting>()
    }

    /// The single camera entity (DESIGN §5: exactly one per `render()`).
    ///
    /// If a scene somehow spawns several, the lowest `Entity` wins — an
    /// arbitrary rule, but a *stable* one, which beats "whichever archetype the
    /// query reached first".
    pub fn camera_entity(&mut self) -> Option<Entity> {
        let mut found: Option<Entity> = None;
        let mut count = 0usize;
        for (entity, _, _, _) in self.camera_query.iter(&self.world) {
            count += 1;
            found = Some(match found {
                Some(best) => best.min(entity),
                None => entity,
            });
        }
        debug_assert!(count <= 1, "DESIGN §5: exactly one camera, found {count}");
        found
    }

    /// The per-frame constants for a viewport of the given aspect ratio:
    /// view-projection from the camera entity's interpolated pose, plus the
    /// light rig. `None` when the world has no camera at all.
    pub fn frame_params(&mut self, aspect: f32) -> Option<FrameParams> {
        self.frame_params_at(aspect, self.alpha)
    }

    /// As [`frame_params`](Sim::frame_params) but at an explicit alpha.
    pub fn frame_params_at(&mut self, aspect: f32, alpha: f32) -> Option<FrameParams> {
        let entity = self.camera_entity()?;
        let (_, camera, transform, interpolated) = self.camera_query.get(&self.world, entity).ok()?;
        let pose = match interpolated {
            Some(prev) => prev.blend(transform, alpha),
            None => transform.matrix(),
        };
        Some(FrameParams {
            view_proj: camera.view_proj(pose, aspect),
            lighting: *self.world.resource::<Lighting>(),
        })
    }

    /// Every drawable entity at the current alpha, sorted for the render pass
    /// (DESIGN §5). Entities without an [`Interpolated`] are drawn at their
    /// current transform.
    pub fn draw_list(&mut self) -> Vec<DrawItem> {
        self.draw_list_at(self.alpha)
    }

    /// As [`draw_list`](Sim::draw_list) but at an explicit alpha.
    pub fn draw_list_at(&mut self, alpha: f32) -> Vec<DrawItem> {
        draw::extract_draw_list(&mut self.draw_query, &self.world, alpha)
    }
}

impl Default for Sim {
    fn default() -> Sim {
        Sim::new()
    }
}
