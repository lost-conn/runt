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
    self, DemoEntity, FixedTick, Interpolated, Lighting, QualityTier, StatusLine, TickCount,
    Transform, WindowMode,
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

/// How fast sim time runs against wall time (DESIGN §4; port decision D9).
///
/// `1.0` is real time, `0.5` is half speed, `0.0` is frozen. [`Sim::update`]
/// integrates **scaled** wall-clock deltas into its accumulator, so a tick is
/// still exactly [`TICK_DT`] of *sim* time — what stretches is the wall-clock
/// spacing between ticks. Nothing inside a tick can tell what the speed is
/// unless it reads this resource: `FixedTick::dt_secs` does not move, physics
/// does not move, and an [`InputTrace`](crate::InputTrace) is indexed by tick
/// number, so **a replay is unaffected by the speed history of the run it was
/// recorded from** (`tests/sim_speed.rs` proves it).
///
/// ## The contract: written from `FixedSim`, read by the host loop
///
/// Slowmo is *gameplay* — a collect freeze, a hit stop, a pause — so the value
/// is sim state and belongs to sim systems: write it as `ResMut<SimSpeed>` from
/// a `FixedSim` system, deterministically, from tick numbers and events. It is
/// then read exactly once per [`Sim::update`] call, **before** any tick in that
/// call runs, which is what makes "the speed a tick sees" a meaningless
/// question: the value in force for an update is the one the previous update's
/// last tick left behind, and a tick that changes it changes the *next* update.
///
/// A host that writes it (a debug hotkey, the editor, a test) is not lying to
/// anything — no fingerprint can move — but it is writing over whatever the
/// game's own systems decided, so it goes through the deliberately clunky
/// [`Sim::set_sim_speed`] rather than being part of the per-frame host API.
///
/// ## Exactly zero is a latch
///
/// A frozen sim runs no ticks, and a `FixedSim` system that does not run cannot
/// write the resource that would un-freeze it. **A tick can stop time but
/// cannot start it again** — only a host can, through
/// [`set_sim_speed`](Sim::set_sim_speed). This is not a bug to be fixed with an
/// escape hatch inside the tick (a "speed-setting systems still run while
/// frozen" rule would be a second, invisible schedule); it is the reason a
/// *pause menu* is a `Paused` resource that gates gameplay system sets while the
/// tick keeps running, and `SimSpeed` is for **slowmo**, where the interesting
/// values are `0.1..1.0` and something in the tick is always counting down.
///
/// ## Range
///
/// Reads are sanitised into `[MIN, MAX]` by [`SimSpeed::get`] — the field is
/// public so a system can write it like any other tuple resource, and the read
/// side, not the write side, is where the guard has to be. A non-finite speed
/// reads as `1.0`: a NaN is a bug in the writer, and freezing the game for it
/// hides the bug behind a symptom that looks like a hang.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct SimSpeed(pub f32);

impl SimSpeed {
    /// Frozen. Not negative: time does not run backwards, and a sim that could
    /// rewind would need a state history it does not keep.
    pub const MIN: f32 = 0.0;
    /// Four times real time. A ceiling rather than a policy — it exists so a
    /// stray large value cannot turn one host frame into a spiral of ticks the
    /// clamp then has to throw away.
    pub const MAX: f32 = 4.0;
    /// Real time.
    pub const NORMAL: SimSpeed = SimSpeed(1.0);

    /// A clamped, finite speed.
    pub fn new(speed: f32) -> SimSpeed {
        SimSpeed(Self::sanitise(speed))
    }

    /// The value [`Sim::update`] actually uses: clamped into `[MIN, MAX]`, with
    /// a non-finite value read as [`NORMAL`](SimSpeed::NORMAL).
    pub fn get(self) -> f32 {
        Self::sanitise(self.0)
    }

    /// Write a clamped speed.
    pub fn set(&mut self, speed: f32) {
        self.0 = Self::sanitise(speed);
    }

    /// Whether the sim is stopped. Exactly zero, because that is the only value
    /// that produces no ticks at all.
    pub fn is_frozen(self) -> bool {
        self.get() == 0.0
    }

    fn sanitise(speed: f32) -> f32 {
        if speed.is_finite() {
            speed.clamp(SimSpeed::MIN, SimSpeed::MAX)
        } else {
            1.0
        }
    }
}

impl Default for SimSpeed {
    fn default() -> SimSpeed {
        SimSpeed::NORMAL
    }
}

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
    /// Sim time already accounted for: ticks run, plus any backlog dropped by
    /// the clamp. Recomputing the outstanding time from `elapsed - origin -
    /// consumed + warp` on every call (rather than accumulating host deltas) is
    /// what makes the tick count a pure function of the elapsed value the host
    /// passes, independent of how it chopped the interval up — at
    /// [`SimSpeed::NORMAL`], where `warp` is exactly zero.
    consumed: f64,
    /// The largest `elapsed_seconds` seen so far — the base the next update's
    /// delta is measured from. Monotone by construction so a host clock that
    /// jumps backwards cannot manufacture a negative delta.
    last: f64,
    /// Accumulated `Σ delta × (speed − 1)`: how far sim time has been dragged
    /// away from wall time by [`SimSpeed`].
    ///
    /// Stored as the *difference* rather than as a scaled clock so that at speed
    /// `1.0` every term is `delta × 0.0 == 0.0` and `warp` stays **bit-exactly**
    /// zero forever. That is deliberate: it means introducing slowmo did not
    /// perturb a single float in the un-slowed path, and the accumulator tests
    /// that predate it still describe the same arithmetic.
    warp: f64,
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
        world.insert_resource(crate::texture::TextureLibrary::new());
        world.insert_resource(Lighting::default());
        world.insert_resource(StatusLine::default());
        world.insert_resource(crate::ecs::RenderScale::default());
        world.insert_resource(SimSpeed::default());
        world.insert_resource(crate::audio::AudioOut::new());
        // The frame's HUD (plan D11). A standard output seam like `StatusLine`
        // and `AudioOut`: present from tick zero so a game's HUD system can take
        // `ResMut<UiBatch>` without inserting it first, empty until something
        // fills it, and free while it stays that way.
        world.insert_resource(crate::ui::UiBatch::default());
        world.insert_resource(quality);
        world.insert_resource(GenCache::new(cache));
        world.insert_resource(PendingScene(scene));
        // `MessageWriter` is `ResMut<Messages<_>>` underneath, so the buffer has
        // to exist before the first tick even in a world with no physics in it.
        crate::physics::register_messages(&mut world);

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
            last: 0.0,
            warp: 0.0,
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

    /// The overlaps generated by the most recent tick (DESIGN §9).
    ///
    /// A host-side read: it looks at the current write buffer without touching
    /// any system's cursor, so calling it never hides an event from a system in
    /// the schedule (and vice versa). Empty between ticks only in the sense that
    /// it reports the *last* tick's batch until the next one runs.
    ///
    /// Game code that wants to *react* — despawn a pickup, add a point — belongs
    /// in a `FixedSim` system reading `MessageReader<OverlapEvent>`; see
    /// [`fixed_sim_mut`](Sim::fixed_sim_mut). This accessor is for HUDs, audio
    /// and tests, which live outside the tick anyway.
    pub fn overlaps(&self) -> impl ExactSizeIterator<Item = &crate::physics::OverlapEvent> {
        self.world
            .resource::<bevy_ecs::message::Messages<crate::physics::OverlapEvent>>()
            .iter_current_update_messages()
    }

    /// The `FixedSim` schedule, so game code can chain its own systems onto the
    /// engine's (DESIGN §4: *all* gameplay mutation happens in this schedule).
    ///
    /// Systems added here run after everything
    /// [`fixed_sim_schedule`](crate::ecs::fixed_sim_schedule) installed unless
    /// they say otherwise, which is what a reader of `OverlapEvent` wants: it
    /// sees this tick's overlaps on this tick.
    pub fn fixed_sim_mut(&mut self) -> &mut Schedule {
        &mut self.fixed_sim
    }

    /// The line of text a game asked the host to display (DESIGN §13's open
    /// HUD question, answered with the cheapest thing that works). Empty until
    /// something writes [`StatusLine`].
    pub fn status_line(&self) -> &str {
        self.world
            .get_resource::<StatusLine>()
            .map(|s| s.as_str())
            .unwrap_or("")
    }

    /// The window the game is asking for (DESIGN §2's host seam, the same shape
    /// [`status_line`](Sim::status_line) has). Default until something writes
    /// [`WindowMode`], and default is "whatever the host opened with".
    pub fn window_mode(&self) -> WindowMode {
        self.world
            .get_resource::<WindowMode>()
            .copied()
            .unwrap_or_default()
    }

    // -- audio (DESIGN §8) ---------------------------------------------------

    /// The audio queue, for a host or a test.
    ///
    /// Game code reaches it as an ordinary `ResMut<AudioOut>` inside a `FixedSim`
    /// system; this accessor is for the two places that live *outside* the tick.
    pub fn audio_out(&self) -> &crate::audio::AudioOut {
        self.world.resource::<crate::audio::AudioOut>()
    }

    pub fn audio_out_mut(&mut self) -> &mut crate::audio::AudioOut {
        self.world.resource_mut::<crate::audio::AudioOut>().into_inner()
    }

    /// The `(tick, event)` pairs flushed and not yet drained.
    ///
    /// Tick-indexed on purpose: it lines up with an
    /// [`InputTrace`](crate::InputTrace) without an off-by-one, which is what
    /// makes "replay the trace, compare the audio log" a one-line assertion.
    pub fn audio_events(&self) -> &[(u64, crate::audio::AudioEvent)] {
        self.audio_out().outbox()
    }

    /// Hand everything the sim has produced to `backend` and clear the queue.
    ///
    /// The whole host-side audio API. Called once per frame, after
    /// [`update`](Sim::update) — so one call carries however many ticks that
    /// frame ran, in tick order, as a single batch.
    pub fn drain_audio(&mut self, backend: &mut dyn crate::audio::AudioBackend) {
        let events: Vec<crate::audio::AudioEvent> = self
            .world
            .resource_mut::<crate::audio::AudioOut>()
            .drain()
            .map(|(_, event)| event)
            .collect();
        if !events.is_empty() {
            backend.submit(&events);
        }
    }

    // -- input traces (DESIGN §4) -------------------------------------------

    /// Start recording every tick's input into an [`InputTrace`].
    ///
    /// Installed at the head of `FixedSim`, after [`trace::apply`] so that
    /// recording *a replay* records what the replay actually fed the tick.
    pub fn record_input_trace(&mut self) {
        self.world.init_resource::<crate::trace::InputTrace>();
        self.fixed_sim.add_systems(
            crate::trace::record
                .after(crate::physics::update_overlap_messages)
                .after(crate::trace::apply)
                .before(crate::physics::integrate_balls),
        );
    }

    /// Replay `trace`: from now on every tick's [`Input`] comes from it and
    /// from nothing else, host events included (see [`trace::apply`]).
    pub fn play_input_trace(&mut self, trace: crate::trace::InputTrace) {
        self.world
            .insert_resource(crate::trace::Playback::new(trace));
        self.fixed_sim.add_systems(
            crate::trace::apply
                .after(crate::physics::update_overlap_messages)
                .before(crate::physics::integrate_balls),
        );
    }

    /// The trace recorded so far, if [`record_input_trace`](Sim::record_input_trace)
    /// was called.
    pub fn input_trace(&self) -> Option<&crate::trace::InputTrace> {
        self.world.get_resource::<crate::trace::InputTrace>()
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

    /// The speed sim time runs at (DESIGN §4; D9). See [`SimSpeed`].
    pub fn sim_speed(&self) -> f32 {
        self.world
            .get_resource::<SimSpeed>()
            .copied()
            .unwrap_or_default()
            .get()
    }

    /// Overwrite the sim speed from outside the tick.
    ///
    /// **Not the normal path** — [`SimSpeed`] is sim state and slowmo is
    /// gameplay, so a game writes `ResMut<SimSpeed>` in a `FixedSim` system.
    /// This exists for the two callers that have no tick to write from: a host
    /// or editor with a debug control, and the one case a tick provably cannot
    /// handle itself — resuming from a frozen (`0.0`) sim, where no system runs
    /// to raise the speed it lowered.
    ///
    /// Safe with respect to replays either way: the value is not recorded, not
    /// read by a trace, and cannot move a fingerprint.
    pub fn set_sim_speed(&mut self, speed: f32) {
        self.world.insert_resource(SimSpeed::new(speed));
    }

    /// Advance the sim to `elapsed_seconds` of host wall time and return how
    /// many ticks ran.
    ///
    /// `elapsed_seconds` must be monotonically increasing; the first call
    /// establishes the origin and never ticks. Time that goes backwards is
    /// ignored rather than trusted (a host with a jumpy clock must not be able
    /// to rewind the sim).
    ///
    /// ## Speed
    ///
    /// The wall-clock delta since the previous call is multiplied by
    /// [`SimSpeed`] before it reaches the accumulator, so ticks stay
    /// [`TICK_DT`] apart *in sim time* and stretch or compress in wall time.
    /// The resource is read **once**, here, before any tick runs: a tick that
    /// writes it is writing the speed of the next update, and no update can be
    /// half at one speed and half at another.
    ///
    /// At speed `0.0` the accumulator does not advance at all: this returns `0`
    /// and leaves [`alpha`](Sim::alpha) where it was (to the last ulp of the
    /// host's clock — the two terms that cancel are added, not skipped), so a host that
    /// keeps calling `update` and rendering shows a frozen — not a stuttering,
    /// and not a fast-forwarding-on-resume — picture. Nothing about the host
    /// loop stalls; it simply has no new ticks to draw between.
    ///
    /// ## The spiral-of-death clamp is in sim time
    ///
    /// [`MAX_ACCUMULATED`] caps the **scaled** backlog, which is the only
    /// consistent place for it: the guard exists to bound how many ticks one
    /// call can run, and a tick costs the same whatever the clock is doing.
    /// The wall-clock stall it forgives therefore scales with the speed —
    /// 0.25 s at `1.0`, a whole second at `0.25`, 62 ms at `4.0` — and the cap
    /// is never below one tick, so no speed above zero can starve the sim.
    pub fn update(&mut self, elapsed_seconds: f64) -> u32 {
        // A NaN or infinite time would poison the origin permanently, freezing
        // the sim for good. Refuse it outright rather than latch it.
        if !elapsed_seconds.is_finite() {
            self.alpha = 0.0;
            return 0;
        }

        let first = self.origin.is_none();
        let origin = *self.origin.get_or_insert(elapsed_seconds);
        if first {
            self.last = elapsed_seconds;
        }

        // Read the speed once, before any tick can touch it (see the docs
        // above), and charge this call's delta at it. `last` only ever moves
        // forwards, so a backwards clock contributes nothing here and is caught
        // by the `outstanding < 0` guard below exactly as it always was.
        let speed = self.sim_speed() as f64;
        let delta = (elapsed_seconds - self.last).max(0.0);
        self.last = self.last.max(elapsed_seconds);
        let warp = self.warp + delta * (speed - 1.0);
        // Only a delta big enough to overflow could do this, but a poisoned
        // `warp` would wedge the sim permanently and the guard is one compare.
        self.warp = if warp.is_finite() { warp } else { 0.0 };

        let mut outstanding = (elapsed_seconds - origin) - self.consumed + self.warp;
        if !outstanding.is_finite() || outstanding < 0.0 {
            outstanding = 0.0;
        }
        // Never clamp below one tick, or a sim slower than 4 Hz could never
        // tick at all. The cap is in *sim* seconds — `outstanding` is already
        // scaled — which is what makes it a bound on ticks-per-call rather than
        // a bound that slowmo could quietly tighten or loosen.
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

    /// Procedural texture specs, keyed by content key (DESIGN §7). The renderer
    /// bakes from here — eagerly at load if a host asked, lazily on first draw
    /// otherwise.
    pub fn texture_library(&self) -> &crate::texture::TextureLibrary {
        self.world.resource::<crate::texture::TextureLibrary>()
    }

    /// Whether textured draws use §7's live path.
    pub fn live_textures(&self) -> bool {
        self.world
            .get_resource::<crate::texture::TextureLibrary>()
            .is_some_and(crate::texture::TextureLibrary::live_textures)
    }

    /// Switch every textured draw between §7's baked and live paths.
    ///
    /// Render-side only: it changes which variant bit
    /// [`draw::resolve_variant`](crate::draw::resolve_variant) hands the
    /// renderer and nothing a `FixedSim` system can read, so a determinism
    /// fingerprint cannot move when it flips. See
    /// [`TextureLibrary::set_live_textures`](crate::texture::TextureLibrary::set_live_textures)
    /// for why it is v1's whole gate.
    pub fn set_live_textures(&mut self, live: bool) {
        if let Some(mut library) = self
            .world
            .get_resource_mut::<crate::texture::TextureLibrary>()
        {
            library.set_live_textures(live);
        }
    }

    /// The fraction of the host's resolution the scene is drawn at
    /// (DESIGN §11). See [`RenderScale`](crate::ecs::RenderScale).
    pub fn render_scale(&self) -> crate::ecs::RenderScale {
        self.world
            .get_resource::<crate::ecs::RenderScale>()
            .copied()
            .unwrap_or_default()
    }

    /// Draw the scene at `scale` × the host's resolution, clamped into
    /// `[RenderScale::MIN, RenderScale::MAX]`.
    ///
    /// Render-side only, exactly like
    /// [`set_live_textures`](Sim::set_live_textures): the renderer reads it
    /// while sizing its internal target and no system reads it at all, so a
    /// determinism fingerprint cannot move when it changes. Game code that wants
    /// it on a key writes `ResMut<RenderScale>` from a `FixedSim` system.
    pub fn set_render_scale(&mut self, scale: f32) {
        let scale = crate::ecs::RenderScale::new(scale);
        self.world.insert_resource(scale);
    }

    /// The persistent content store, so the bake can consult it (DESIGN §7).
    pub fn cache_store(&self) -> &dyn CacheStore {
        self.world.resource::<GenCache>().store()
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
