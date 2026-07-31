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
use glam::Mat4;

use crate::ecs::{
    self, DemoEntity, FixedTick, Interpolated, TickCount, Transform,
};
use crate::input::{Input, InputEvent};

/// The tick length DESIGN §4 fixes: 1/60 s exactly.
pub const TICK_DT: f64 = 1.0 / 60.0;

/// Spiral-of-death guard (DESIGN §4). If a host stalls for longer than this,
/// the surplus is *discarded* rather than simulated: a device below the floor
/// runs in slow motion instead of freezing while it tries to catch up.
pub const MAX_ACCUMULATED: f64 = 0.25;

/// Largest `f32` strictly below 1.0 — `alpha` is documented as `[0,1)` and
/// callers are entitled to rely on it.
const ALPHA_MAX: f32 = 1.0 - f32::EPSILON;

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
        assert!(hz > 0.0, "tick rate must be positive, got {hz}");
        Sim::with_tick_dt(1.0 / hz)
    }

    /// A sim with an explicit tick length in seconds.
    pub fn with_tick_dt(tick_dt: f64) -> Sim {
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

    /// The single entity holding the merged demo scene.
    pub fn demo_entity(&self) -> Entity {
        self.world.resource::<DemoEntity>().0
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

    /// The demo scene's interpolated model matrix — what the renderer draws the
    /// merged buffer with until per-entity draws land (DESIGN §12 step 3).
    pub fn demo_model_matrix(&self) -> Mat4 {
        self.model_matrix(self.demo_entity())
            .unwrap_or(Mat4::IDENTITY)
    }
}

impl Default for Sim {
    fn default() -> Sim {
        Sim::new()
    }
}
