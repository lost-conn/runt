//! Slowmo: [`SimSpeed`] scales the *clock*, never the tick (DESIGN §4; D9).
//!
//! Two claims, and the second is the one that matters:
//!
//! 1. A tick is still `TICK_DT` of sim time at any speed; what moves is how much
//!    wall time passes between two of them.
//! 2. **A replay does not know what the speed was doing.** An
//!    [`InputTrace`](runt_core::InputTrace) is indexed by tick number, and a tick
//!    is a tick, so a run recorded while the speed oscillated (and froze, and
//!    ran at 4×) replays bit-for-bit at a flat 1.0 — on a host with a completely
//!    different frame pattern, with no input pushed at all.
//!
//! No GPU here, like `sim_determinism.rs`: this is arithmetic and a schedule.

use bevy_ecs::prelude::*;

use runt_core::ecs::advance_tick_count;
use runt_core::input::{Input, InputEvent, Key};
use runt_core::sim::TICK_DT;
use runt_core::{Sim, SimConfig, SimSpeed, TickCount, Transform};

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/// A running FNV-1a over everything a `FixedSim` system could read from
/// [`Input`], mixed with the tick it was read on.
///
/// This is the fingerprint the replay claim is made about. It is taken from
/// `Input` rather than from world state on purpose: the demo scene's spinner is
/// a pure function of the tick *count*, so it would match even if every tick had
/// seen the wrong input. What has to be reproduced is the per-tick input stream,
/// and that is exactly what this hashes.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Fingerprint {
    hash: u64,
    ticks: u64,
}

impl Fingerprint {
    fn mix(&mut self, value: u64) {
        self.hash ^= value;
        self.hash = self.hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

/// `FixedSim` (tail): fold this tick's input into the fingerprint.
fn fingerprint(mut fp: ResMut<Fingerprint>, tick: Res<TickCount>, input: Res<Input>) {
    let now = tick.0;
    fp.mix(now);
    for (index, key) in Key::ALL.into_iter().enumerate() {
        let bits = (input.held(key) as u64)
            | ((input.just_pressed(key) as u64) << 1)
            | ((input.just_released(key) as u64) << 2);
        if bits != 0 {
            fp.mix((index as u64) << 8 | bits);
        }
    }
    for button in 0..3u8 {
        let bits = (input.button_held(button) as u64)
            | ((input.button_just_pressed(button) as u64) << 1)
            | ((input.button_just_released(button) as u64) << 2);
        if bits != 0 {
            fp.mix((button as u64) << 16 | bits);
        }
    }
    let mouse = input.mouse_delta();
    fp.mix(mouse.x.to_bits() as u64);
    fp.mix(mouse.y.to_bits() as u64);
    fp.mix(input.wheel().to_bits() as u64);
    let drive = input.drive();
    fp.mix(drive.x.to_bits() as u64);
    fp.mix(drive.y.to_bits() as u64);
    fp.ticks += 1;
}

/// The wall-clock times a host delivers events at, and what it delivers.
///
/// Deliberately spread across the whole run so that *when* they land depends on
/// how fast the sim was running — which is the thing the replay must not care
/// about.
fn script() -> Vec<(f64, InputEvent)> {
    vec![
        (0.05, InputEvent::KeyDown(Key::W)),
        (0.21, InputEvent::MouseMove { dx: 2.5, dy: -0.75 }),
        (0.40, InputEvent::KeyDown(Key::Space)),
        (0.63, InputEvent::KeyUp(Key::W)),
        (0.90, InputEvent::MouseButton { button: 0, pressed: true }),
        (1.15, InputEvent::Wheel { dy: -1.0 }),
        (1.42, InputEvent::KeyDown(Key::A)),
        (1.70, InputEvent::KeyUp(Key::Space)),
        (2.05, InputEvent::MouseButton { button: 0, pressed: false }),
        (2.33, InputEvent::MouseMove { dx: -4.0, dy: 3.25 }),
        (2.60, InputEvent::KeyUp(Key::A)),
        (2.95, InputEvent::KeyDown(Key::W)),
    ]
}

/// A sim with no scene (the plumbing is the subject here) and the fingerprint
/// system installed at the tail of `FixedSim` — after
/// [`trace::apply`](runt_core::trace::apply), which sits ahead of
/// `advance_tick_count`, so on a replay it hashes what the *trace* fed the tick.
fn sim() -> Sim {
    let mut sim = Sim::from_config(SimConfig::default().without_scene());
    sim.world_mut().init_resource::<Fingerprint>();
    sim.fixed_sim_mut()
        .add_systems(fingerprint.after(advance_tick_count));
    sim
}

fn fingerprint_of(sim: &Sim) -> Fingerprint {
    *sim.world().resource::<Fingerprint>()
}

fn speed_of(sim: &Sim) -> f32 {
    sim.sim_speed()
}

/// Drive `sim` over a 60 Hz wall clock for `frames` frames, delivering the
/// script's events at their wall-clock times. Returns the wall time it stopped
/// at.
fn drive(sim: &mut Sim, frames: usize, mut before_update: impl FnMut(&mut Sim, usize)) -> f64 {
    let events = script();
    let mut next = 0usize;
    sim.update(0.0);
    let mut t = 0.0;
    for frame in 1..=frames {
        t = frame as f64 / 60.0;
        while next < events.len() && events[next].0 <= t {
            sim.push_input(events[next].1);
            next += 1;
        }
        before_update(sim, frame);
        sim.update(t);
    }
    t
}

// ---------------------------------------------------------------------------
// The scalar itself
// ---------------------------------------------------------------------------

#[test]
fn the_default_is_real_time_and_the_range_is_clamped_on_read() {
    assert_eq!(SimSpeed::default(), SimSpeed::NORMAL);
    assert_eq!(SimSpeed::default().get(), 1.0);
    assert_eq!(Sim::new().sim_speed(), 1.0);

    assert_eq!(SimSpeed::new(0.25).get(), 0.25);
    assert_eq!(SimSpeed::new(-3.0).get(), SimSpeed::MIN);
    assert_eq!(SimSpeed::new(1e9).get(), SimSpeed::MAX);

    // The field is public, so the guard has to live on the read side.
    assert_eq!(SimSpeed(-1.0).get(), SimSpeed::MIN);
    assert_eq!(SimSpeed(100.0).get(), SimSpeed::MAX);
    assert_eq!(
        SimSpeed(f32::NAN).get(),
        1.0,
        "a NaN speed must not freeze the game and hide the bug"
    );
    assert_eq!(SimSpeed(f32::INFINITY).get(), 1.0);

    assert!(SimSpeed(0.0).is_frozen());
    assert!(SimSpeed(-0.5).is_frozen(), "clamped to zero, so frozen");
    assert!(!SimSpeed(0.001).is_frozen());
}

// ---------------------------------------------------------------------------
// Scaled time
// ---------------------------------------------------------------------------

#[test]
fn half_speed_runs_half_as_many_ticks_over_the_same_wall_time() {
    let mut slow = Sim::new();
    slow.set_sim_speed(0.5);
    let mut normal = Sim::new();

    for sim in [&mut slow, &mut normal] {
        sim.update(0.0);
        for i in 1..=120 {
            sim.update(i as f64 / 60.0);
        }
    }

    assert!(
        (119..=120).contains(&normal.tick_count()),
        "2 s at 60 Hz, got {}",
        normal.tick_count()
    );
    assert!(
        (59..=60).contains(&slow.tick_count()),
        "2 s of wall time at half speed is 1 s of sim time, got {}",
        slow.tick_count()
    );
}

#[test]
fn double_speed_runs_twice_as_many() {
    let mut fast = Sim::new();
    fast.set_sim_speed(2.0);
    fast.update(0.0);
    for i in 1..=60 {
        fast.update(i as f64 / 60.0);
    }
    assert!(
        (119..=120).contains(&fast.tick_count()),
        "1 s at 2x is 2 s of sim time, got {}",
        fast.tick_count()
    );
}

#[test]
fn a_tick_is_still_tick_dt_whatever_the_speed_is() {
    // The whole point of scaling the clock instead of the step: nothing inside
    // the tick moves, so physics, tweens and the fixed timestep are untouched.
    let mut sim = Sim::new();
    let dt_before = sim.world().resource::<runt_core::ecs::FixedTick>().dt_secs;
    sim.set_sim_speed(0.25);
    sim.update(0.0);
    for i in 1..=240 {
        sim.update(i as f64 / 60.0);
    }
    let dt_after = sim.world().resource::<runt_core::ecs::FixedTick>().dt_secs;
    assert_eq!(dt_before.to_bits(), dt_after.to_bits());
    assert_eq!(dt_after, TICK_DT as f32);

    // 4 s of wall time at quarter speed is 1 s of sim time: 60 ticks, each one
    // a sixtieth of a second of *sim* time.
    assert!(
        (59..=60).contains(&sim.tick_count()),
        "got {}",
        sim.tick_count()
    );
}

#[test]
fn normal_speed_is_bit_identical_to_never_touching_the_resource() {
    // `warp` is stored as a difference precisely so that speed 1.0 adds
    // `delta * 0.0` and stays exactly zero. If it were a scaled clock instead,
    // this would drift in the last bits and every pre-existing accumulator test
    // would be describing slightly different arithmetic.
    let mut untouched = Sim::new();
    let mut pinned = Sim::new();
    pinned.set_sim_speed(1.0);

    let mut times = Vec::new();
    let mut t = 0.0f64;
    for step in [0.005f64, 0.031, 0.0071, 0.019] {
        for _ in 0..40 {
            t += step;
            times.push(t);
        }
    }

    for sim in [&mut untouched, &mut pinned] {
        sim.update(0.0);
        for &t in &times {
            sim.update(t);
        }
    }

    assert_eq!(untouched.tick_count(), pinned.tick_count());
    assert_eq!(untouched.alpha().to_bits(), pinned.alpha().to_bits());
    let (a, b) = (
        untouched
            .world()
            .get::<Transform>(untouched.demo_entity())
            .copied()
            .expect("demo transform"),
        pinned
            .world()
            .get::<Transform>(pinned.demo_entity())
            .copied()
            .expect("demo transform"),
    );
    assert_eq!(
        a.rotation.to_array().map(f32::to_bits),
        b.rotation.to_array().map(f32::to_bits)
    );
}

// ---------------------------------------------------------------------------
// Frozen
// ---------------------------------------------------------------------------

#[test]
fn zero_speed_freezes_without_stalling_the_host_loop() {
    let mut sim = Sim::with_tick_rate(10.0);
    sim.update(0.0);
    sim.update(0.15); // one tick in, alpha halfway to the next
    let ticks = sim.tick_count();
    let alpha = sim.alpha();
    assert_eq!(ticks, 1);
    assert!(alpha > 0.4 && alpha < 0.6, "alpha {alpha}");

    sim.set_sim_speed(0.0);
    let e = sim.demo_entity();
    let frozen_pose = sim.model_matrix(e).expect("model").to_cols_array();

    // Ten seconds of host frames — three hundred update calls, no ticks, and a
    // render pose that never moves. The host keeps running; the world does not.
    for i in 1..=300 {
        let t = 0.15 + i as f64 / 30.0;
        assert_eq!(sim.update(t), 0, "a frozen sim must not tick");
        assert!(
            (sim.alpha() - alpha).abs() < 1e-6,
            "alpha must hold at {alpha}, got {} at t={t}",
            sim.alpha()
        );
        assert_eq!(
            sim.model_matrix(e).expect("model").to_cols_array(),
            frozen_pose,
            "the render pose must be frozen, not stuttering"
        );
    }
    assert_eq!(sim.tick_count(), ticks);

    // Resuming does **not** replay the ten seconds that went by: the clamp never
    // saw them, because they were never sim time in the first place.
    sim.set_sim_speed(1.0);
    let resumed = sim.update(0.15 + 300.0 / 30.0 + 0.1);
    assert_eq!(resumed, 1, "one 100 ms tick, not a ten-second catch-up");
    assert_eq!(sim.tick_count(), ticks + 1);
}

#[test]
fn a_tick_can_freeze_the_sim_but_only_a_host_can_thaw_it() {
    // The latch documented on `SimSpeed`: the system that would raise the speed
    // lives in a schedule that a zero speed stops running. This test exists so
    // that the day someone "fixes" it, they have to come here and say so.
    fn freeze(tick: Res<TickCount>, mut speed: ResMut<SimSpeed>) {
        if tick.0 >= 5 {
            *speed = SimSpeed(0.0);
        } else {
            *speed = SimSpeed::NORMAL;
        }
    }

    let mut sim = Sim::from_config(SimConfig::default().without_scene());
    sim.fixed_sim_mut()
        .add_systems(freeze.after(advance_tick_count));

    sim.update(0.0);
    for i in 1..=600 {
        sim.update(i as f64 / 60.0);
    }
    assert_eq!(sim.tick_count(), 6, "froze on the tick that set the speed");
    assert_eq!(speed_of(&sim), 0.0);

    sim.set_sim_speed(1.0);
    assert!(sim.update(600.0 / 60.0 + 0.1) > 0, "the host can thaw it");
}

// ---------------------------------------------------------------------------
// The spiral-of-death clamp, in sim time
// ---------------------------------------------------------------------------

#[test]
fn the_backlog_clamp_bounds_ticks_per_call_not_wall_time() {
    let max_ticks = (runt_core::MAX_ACCUMULATED / TICK_DT) as u32;

    // Whatever the speed, one call can never run more than the cap's worth of
    // ticks — that is what the guard is for.
    for speed in [0.25f32, 1.0, 4.0] {
        let mut sim = Sim::new();
        sim.set_sim_speed(speed);
        sim.update(0.0);
        assert_eq!(
            sim.update(10.0),
            max_ticks,
            "a ten-second stall at {speed}x must still clamp to {max_ticks} ticks"
        );
    }

    // The wall-clock stall the clamp *forgives* scales with the speed, because
    // the cap is measured in sim seconds. 0.4 s of wall time is 0.4 s of sim
    // time at 1.0 — over the 0.25 s cap, so ticks are dropped …
    let mut normal = Sim::new();
    normal.update(0.0);
    assert_eq!(normal.update(0.4), max_ticks);

    // … but only 0.2 s of sim time at half speed, which is under the cap and so
    // survives intact: 12 ticks, none dropped.
    let mut slow = Sim::new();
    slow.set_sim_speed(0.5);
    slow.update(0.0);
    assert_eq!(slow.update(0.4), 12);
}

// ---------------------------------------------------------------------------
// The claim: replays are tick-indexed, so the speed history is not in them
// ---------------------------------------------------------------------------

#[test]
fn a_replay_is_unaffected_by_the_speed_history_of_the_run_that_recorded_it() {
    /// `FixedSim`: the game's own slowmo logic — a pure function of the tick
    /// number, which is what "written from FixedSim, deterministically" means.
    /// Zero is left out on purpose: a frozen sim would never run this system
    /// again (see `a_tick_can_freeze_the_sim_but_only_a_host_can_thaw_it`); the
    /// host imposes the freeze window below instead.
    fn oscillate(tick: Res<TickCount>, mut speed: ResMut<SimSpeed>) {
        const STEPS: [f32; 6] = [1.0, 0.25, 2.5, 0.5, 4.0, 0.1];
        *speed = SimSpeed::new(STEPS[(tick.0 as usize / 7) % STEPS.len()]);
    }

    // -- record, while the speed thrashes ----------------------------------
    let mut rec = sim();
    rec.record_input_trace();
    rec.fixed_sim_mut()
        .add_systems(oscillate.after(advance_tick_count));

    let mut speeds_seen: Vec<u32> = Vec::new();
    drive(&mut rec, 300, |sim, frame| {
        // A host-side pause in the middle of the recording, on top of the
        // sim-side oscillation: the run is frozen for a second of wall time.
        match frame {
            120 => sim.set_sim_speed(0.0),
            180 => sim.set_sim_speed(1.0),
            _ => {}
        }
        speeds_seen.push(sim.sim_speed().to_bits());
    });

    speeds_seen.sort_unstable();
    speeds_seen.dedup();
    assert!(
        speeds_seen.len() >= 5,
        "the recording must actually have run at several speeds, saw {}",
        speeds_seen.len()
    );

    let recorded = fingerprint_of(&rec);
    let trace = rec.input_trace().expect("recording").clone();
    assert!(recorded.ticks > 30, "too few ticks to prove anything");
    assert!(!trace.is_empty(), "the run must have recorded input");

    // The speed history really did change the shape of the run: a flat-1.0 host
    // over the same five seconds would have ticked ~300 times.
    assert!(
        rec.tick_count() < 250,
        "the oscillation should have cost ticks, got {}",
        rec.tick_count()
    );

    // -- replay, at a flat 1.0, on a different frame pattern ----------------
    let mut play = sim();
    play.play_input_trace(trace.clone());
    assert_eq!(play.sim_speed(), 1.0);

    // A ragged host, and *no input pushed at all*: everything the replayed ticks
    // see comes from the trace.
    play.update(0.0);
    let mut t = 0.0f64;
    let steps = [0.004f64, 0.021, 0.009];
    let mut i = 0usize;
    while play.tick_count() < rec.tick_count() {
        t += steps[i % steps.len()];
        i += 1;
        play.update(t);
        assert!(i < 100_000, "replay is not converging");
    }

    let replayed = fingerprint_of(&play);
    assert_eq!(
        play.tick_count(),
        rec.tick_count(),
        "the replay must land on the same tick count"
    );
    assert_eq!(
        replayed, recorded,
        "identical fingerprints: the trace is tick-indexed, so the speed history \
         of the recording is not in it"
    );

    // And the replay re-records to the same trace — the fixed point
    // `tests/trace.rs` asserts, held under a speed-warped recording.
    let mut again = sim();
    again.play_input_trace(trace.clone());
    again.record_input_trace();
    again.set_sim_speed(4.0);
    again.update(0.0);
    let mut t = 0.0f64;
    while again.tick_count() < rec.tick_count() {
        t += 1.0 / 240.0;
        again.update(t);
    }
    assert_eq!(
        again.input_trace().expect("recording").events,
        trace.events,
        "re-recording a replay reproduces the trace, at any speed"
    );
    assert_eq!(fingerprint_of(&again), recorded);
}
