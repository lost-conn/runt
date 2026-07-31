//! Fixed-tick determinism and render interpolation (DESIGN §4, §12 step 2).
//!
//! No GPU here: `Sim` is the engine minus the renderer precisely so these can
//! run anywhere, including CI boxes with no adapter.

use runt_core::input::{InputEvent, Key};
use runt_core::sim::{MAX_ACCUMULATED, TICK_DT};
use runt_core::{Interpolated, Sim, Transform};

use glam::{Mat4, Quat, Vec3};

/// A scripted input trace: `(wall time to deliver at, event)`.
///
/// A replay is exactly this plus the seeds (DESIGN §4), so driving the two
/// worlds from one trace is the same thing a replay would do.
fn trace() -> Vec<(f64, InputEvent)> {
    vec![
        (0.05, InputEvent::KeyDown(Key::W)),
        (0.12, InputEvent::MouseMove { dx: 3.5, dy: -1.25 }),
        (0.20, InputEvent::KeyDown(Key::Space)),
        (0.31, InputEvent::KeyUp(Key::W)),
        (0.44, InputEvent::MouseButton { button: 0, pressed: true }),
        (0.55, InputEvent::Wheel { dy: 2.0 }),
        (0.70, InputEvent::KeyUp(Key::Space)),
        (0.88, InputEvent::MouseButton { button: 0, pressed: false }),
    ]
}

/// Drive a fresh `Sim` over `schedule` (a list of wall-clock times to call
/// `update` at), delivering the trace's events before the first update at or
/// after each event's timestamp.
fn drive(schedule: &[f64]) -> Sim {
    let mut sim = Sim::new();
    let events = trace();
    let mut next = 0usize;

    // Establish the time origin without ticking.
    sim.update(0.0);

    for &t in schedule {
        while next < events.len() && events[next].0 <= t {
            sim.push_input(events[next].1);
            next += 1;
        }
        sim.update(t);
    }
    sim
}

fn demo_transform(sim: &Sim) -> Transform {
    *sim.world()
        .get::<Transform>(sim.demo_entity())
        .expect("demo entity has a Transform")
}

fn demo_interpolated(sim: &Sim) -> Interpolated {
    *sim.world()
        .get::<Interpolated>(sim.demo_entity())
        .expect("demo entity has an Interpolated")
}

/// Bit-exact comparison. `assert_eq!` on `f32` is already bitwise for normal
/// values, but going through the bit patterns says so out loud and catches
/// `-0.0`/`NaN` differences that `==` would hide.
fn quat_bits(q: Quat) -> [u32; 4] {
    [
        q.x.to_bits(),
        q.y.to_bits(),
        q.z.to_bits(),
        q.w.to_bits(),
    ]
}

fn vec3_bits(v: Vec3) -> [u32; 3] {
    [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()]
}

fn transform_bits(t: &Transform) -> ([u32; 3], [u32; 4], [u32; 3]) {
    (
        vec3_bits(t.translation),
        quat_bits(t.rotation),
        vec3_bits(t.scale),
    )
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

#[test]
fn ragged_and_uniform_update_patterns_agree_bit_for_bit() {
    // One host runs a clean 60 fps; the other stutters through 5/30/7 ms
    // chunks. Both are handed the same input trace and both finish at exactly
    // t = 1.0, so both must have run the same ticks with the same inputs.
    let uniform: Vec<f64> = (1..=60).map(|i| i as f64 / 60.0).collect();

    let ragged: Vec<f64> = {
        let steps = [0.005, 0.030, 0.007];
        let mut times = Vec::new();
        let mut t: f64 = 0.0;
        let mut i = 0usize;
        while t < 1.0 {
            t += steps[i % steps.len()];
            i += 1;
            times.push(t.min(1.0));
        }
        // Land exactly on 1.0 so the two hosts agree on the end instant.
        if *times.last().expect("non-empty") != 1.0 {
            times.push(1.0);
        }
        times
    };
    assert!(ragged.len() != uniform.len(), "the two patterns must differ");

    let a = drive(&uniform);
    let b = drive(&ragged);

    assert_eq!(
        a.tick_count(),
        b.tick_count(),
        "same elapsed time must mean the same tick count regardless of call pattern"
    );
    // A time that lands *exactly* on a tick boundary can round either side of
    // it in f64, so 59 or 60 are both correct — what matters is that both hosts
    // made the same call. (The lag is transient and never accumulates: see
    // `tick_count_tracks_wall_time_without_drift`.)
    assert!(
        (59..=60).contains(&a.tick_count()),
        "1.0 s at 60 Hz, got {}",
        a.tick_count()
    );

    let (ta, tb) = (demo_transform(&a), demo_transform(&b));
    assert_eq!(
        transform_bits(&ta),
        transform_bits(&tb),
        "transforms must be bit-identical, got {ta:?} vs {tb:?}"
    );

    let (ia, ib) = (demo_interpolated(&a), demo_interpolated(&b));
    assert_eq!(
        (vec3_bits(ia.prev_translation), quat_bits(ia.prev_rotation)),
        (vec3_bits(ib.prev_translation), quat_bits(ib.prev_rotation)),
        "the interpolation snapshot must match too"
    );

    // The input trace ended with everything released, on both hosts.
    assert!(!a.input().held(Key::W) && !a.input().held(Key::Space));
    assert_eq!(a.input().held_count(), b.input().held_count());
}

#[test]
fn same_pattern_twice_is_reproducible() {
    let schedule: Vec<f64> = (1..=90).map(|i| i as f64 / 60.0).collect();
    let a = drive(&schedule);
    let b = drive(&schedule);
    assert_eq!(a.tick_count(), b.tick_count());
    assert_eq!(
        transform_bits(&demo_transform(&a)),
        transform_bits(&demo_transform(&b))
    );
}

#[test]
fn input_is_consumed_only_at_tick_boundaries() {
    let mut sim = Sim::new();
    sim.update(0.0);

    sim.push_input(InputEvent::KeyDown(Key::W));
    // Not a full tick: the event stays buffered and the world has not seen it.
    assert_eq!(sim.update(TICK_DT * 0.5), 0);
    assert_eq!(sim.pending_input_len(), 1);
    assert!(!sim.input().held(Key::W));

    assert_eq!(sim.update(TICK_DT * 1.5), 1);
    assert_eq!(sim.pending_input_len(), 0);
    assert!(sim.input().held(Key::W));
    assert!(sim.input().just_pressed(Key::W));

    // The press edge is a one-tick affair.
    assert_eq!(sim.update(TICK_DT * 2.5), 1);
    assert!(sim.input().held(Key::W));
    assert!(!sim.input().just_pressed(Key::W));
}

// ---------------------------------------------------------------------------
// Accumulator behaviour
// ---------------------------------------------------------------------------

#[test]
fn first_update_only_sets_the_origin() {
    let mut sim = Sim::new();
    // A host whose clock epoch is far from zero must not trigger a huge catch-up.
    assert_eq!(sim.update(12345.0), 0, "the first call establishes the origin");
    assert_eq!(sim.tick_count(), 0);
    assert_eq!(sim.update(12345.0 + TICK_DT * 1.5), 1);
}

#[test]
fn tick_count_tracks_wall_time_without_drift() {
    // The accumulator must not fall behind wall time over a long run — an
    // exact-boundary time may round a tick late, but that lag is transient and
    // never compounds.
    let mut sim = Sim::new();
    sim.update(0.0);
    for i in 1..=6000u64 {
        let t = i as f64 / 60.0;
        sim.update(t);
        let lag = i - sim.tick_count();
        assert!(lag <= 1, "fell {lag} ticks behind wall time at t={t}");
    }
    assert!(
        sim.tick_count() >= 5999,
        "100 s at 60 Hz, got {}",
        sim.tick_count()
    );
}

#[test]
fn accumulator_is_clamped_against_the_spiral_of_death() {
    let mut sim = Sim::new();
    sim.update(0.0);

    // A ten-second stall (tab backgrounded, breakpoint, …). Un-clamped this
    // would be 600 ticks; DESIGN §4 caps the backlog at 0.25 s.
    let ticks = sim.update(10.0);
    let max_ticks = (MAX_ACCUMULATED / TICK_DT) as u32;
    assert_eq!(ticks, max_ticks, "0.25 s of backlog is 15 ticks at 60 Hz");
    assert_eq!(sim.tick_count(), max_ticks as u64);

    // The dropped time is gone for good: the sim does not try to make it up
    // later, it just runs behind wall time.
    assert_eq!(sim.update(10.0 + TICK_DT * 1.5), 1);
    assert_eq!(sim.tick_count(), max_ticks as u64 + 1);
}

#[test]
fn a_slow_tick_rate_still_ticks() {
    // The 0.25 s backlog cap must not be able to starve a sim whose tick is
    // longer than the cap.
    let mut sim = Sim::with_tick_rate(2.0);
    sim.update(0.0);
    assert_eq!(sim.update(0.5), 1, "a 2 Hz sim ticks every 0.5 s");
    assert_eq!(sim.update(1.0), 1);
}

#[test]
fn non_finite_time_is_refused_without_wedging_the_sim() {
    let mut sim = Sim::new();
    assert_eq!(sim.update(f64::NAN), 0);
    assert_eq!(sim.update(f64::INFINITY), 0);

    // The origin was never poisoned, so a sane clock still works.
    sim.update(0.0);
    assert_eq!(sim.update(TICK_DT * 1.5), 1);
}

#[test]
fn time_going_backwards_never_rewinds_the_sim() {
    let mut sim = Sim::new();
    sim.update(0.0);
    sim.update(1.0);
    let before = sim.tick_count();

    assert_eq!(sim.update(0.5), 0, "a backwards clock produces no ticks");
    assert_eq!(sim.tick_count(), before);
    assert_eq!(sim.alpha(), 0.0);

    // And the sim picks up again from where it was, not from the bogus value.
    assert_eq!(sim.update(1.0 + TICK_DT * 1.5), 1);
}

#[test]
fn alpha_stays_in_the_half_open_unit_interval() {
    let mut sim = Sim::new();
    sim.update(0.0);

    // Walk in a step that is deliberately not a tick divisor, so alpha lands
    // all over the interval.
    let mut t = 0.0;
    for _ in 0..500 {
        t += 0.0071;
        sim.update(t);
        let a = sim.alpha();
        assert!(
            (0.0..1.0).contains(&a),
            "alpha must be in [0,1), got {a} at t={t}"
        );
    }

    // Including at the extremes: exactly on a boundary, and just short of one.
    let mut sim = Sim::new();
    sim.update(0.0);
    sim.update(TICK_DT);
    assert_eq!(sim.alpha(), 0.0, "on a tick boundary alpha is 0");

    let mut sim = Sim::new();
    sim.update(0.0);
    sim.update(TICK_DT * 0.999_999_9);
    let a = sim.alpha();
    assert!(a < 1.0 && a > 0.99, "just short of a tick, got {a}");
}

// ---------------------------------------------------------------------------
// Interpolation
// ---------------------------------------------------------------------------

/// The demo spin rate baked into `spawn_demo_scene`.
const DEMO_RAD_PER_SEC: f32 = 0.4;

fn angle_of(model: Mat4) -> f32 {
    let (_, rotation, _) = model.to_scale_rotation_translation();
    // The demo spins about +Y only, so the Y euler angle is the whole story.
    let (_, y, _) = rotation.to_euler(glam::EulerRot::XYZ);
    y
}

#[test]
fn snapshot_holds_the_previous_tick_not_the_current_one() {
    let mut sim = Sim::new();
    sim.update(0.0);
    sim.update(TICK_DT * 4.0);

    let current = demo_transform(&sim);
    let prev = demo_interpolated(&sim);
    assert_ne!(
        quat_bits(prev.prev_rotation),
        quat_bits(current.rotation),
        "prev must lag current by exactly one tick, not equal it"
    );

    // One tick of the demo's spin separates them.
    let step = Quat::from_axis_angle(Vec3::Y, DEMO_RAD_PER_SEC * TICK_DT as f32);
    let expected = (step * prev.prev_rotation).normalize();
    assert!(
        expected.abs_diff_eq(current.rotation, 1e-6),
        "prev * one tick should be current: {expected:?} vs {:?}",
        current.rotation
    );
}

#[test]
fn alpha_zero_and_one_hit_the_tick_poses_exactly() {
    let mut sim = Sim::new();
    sim.update(0.0);
    sim.update(TICK_DT * 3.0);

    let e = sim.demo_entity();
    let prev = demo_interpolated(&sim);
    let current = demo_transform(&sim);

    let at_zero = angle_of(sim.model_matrix_at(e, 0.0).expect("model matrix"));
    let at_one = angle_of(sim.model_matrix_at(e, 1.0).expect("model matrix"));

    let prev_angle = angle_of_quat(prev.prev_rotation);
    let current_angle = angle_of_quat(current.rotation);
    assert!(
        (at_zero - prev_angle).abs() < 1e-6,
        "alpha 0 must be the previous tick exactly: {at_zero} vs {prev_angle}"
    );
    assert!(
        (at_one - current_angle).abs() < 1e-6,
        "alpha 1 must be the current tick exactly: {at_one} vs {current_angle}"
    );
}

fn angle_of_quat(q: Quat) -> f32 {
    let (_, y, _) = q.to_euler(glam::EulerRot::XYZ);
    y
}

#[test]
fn ten_hz_sim_renders_smoothly_between_ticks() {
    // DESIGN §12 step 2: the tick-rate toggle is how interpolation is *proven*.
    // At 10 Hz a tick is 100 ms, so between two ticks there is a wide, easily
    // observable gap that only interpolation can fill.
    let mut sim = Sim::with_tick_rate(10.0);
    assert!((sim.tick_dt() - 0.1).abs() < 1e-12);

    sim.update(0.0);
    sim.update(0.1);
    assert_eq!(sim.tick_count(), 1);

    let e = sim.demo_entity();

    // Sample the render pose across one tick's worth of wall time, ticking
    // nowhere in between (the sim state is frozen; only alpha moves).
    let mut samples = Vec::new();
    for i in 0..=9 {
        let t = 0.1 + i as f64 * 0.01;
        let ticks = sim.update(t);
        assert_eq!(ticks, 0, "no tick should fire inside the 100 ms window");
        samples.push((sim.alpha(), angle_of(sim.model_matrix(e).expect("model"))));
    }

    assert_eq!(sim.tick_count(), 1, "still the same tick throughout");

    // Alpha sweeps the interval and the pose follows it, monotonically.
    assert!(samples[0].0 < 0.01, "starts at the tick boundary");
    assert!(samples[9].0 > 0.85, "ends near the next tick, got {}", samples[9].0);
    for w in samples.windows(2) {
        assert!(w[1].0 > w[0].0, "alpha must advance: {:?}", (w[0].0, w[1].0));
        assert!(
            w[1].1 > w[0].1,
            "render angle must advance with alpha: {:?} -> {:?}",
            w[0],
            w[1]
        );
    }

    // The full sweep covers one tick of rotation: 0.4 rad/s × 0.1 s.
    let swept = samples[9].1 - samples[0].1;
    let one_tick = DEMO_RAD_PER_SEC * 0.1;
    assert!(
        swept > one_tick * 0.8 && swept < one_tick,
        "sweep {swept} should approach one tick of spin ({one_tick})"
    );
}

#[test]
fn render_pose_is_exactly_one_tick_behind_wall_time() {
    // DESIGN §4 blends between the *previous* and *current* tick, so the frame
    // on screen is up to one tick in the past. That is the deliberate trade:
    // interpolating between two known states never overshoots, where
    // extrapolating past the newest one does. Asserting the exact formula here
    // pins the ordering down — if `snapshot_interpolation` ever moved to the
    // end of the tick, `prev` would equal `current` and this would collapse.
    //
    // Sampled at 137 Hz, coprime with both tick rates, so most samples land
    // mid-tick where only interpolation can be right.
    for hz in [10.0f64, 60.0] {
        let mut sim = Sim::with_tick_rate(hz);
        sim.update(0.0);
        let e = sim.demo_entity();
        let dt = sim.tick_dt() as f32;

        // Stepped, not jumped: a one-second leap would hit the 0.25 s clamp.
        for i in 1..=137 {
            let t = i as f64 / 137.0;
            sim.update(t);

            let ticks = sim.tick_count();
            if ticks == 0 {
                continue; // Nothing to interpolate between yet.
            }
            let rendered = angle_of(sim.model_matrix(e).expect("model"));
            let expected = DEMO_RAD_PER_SEC * dt * (ticks as f32 - 1.0 + sim.alpha());
            assert!(
                (rendered - expected).abs() < 1e-4,
                "{hz} Hz at t={t}: rendered {rendered}, expected {expected} \
                 (tick {ticks}, alpha {})",
                sim.alpha()
            );

            // Never ahead of wall time, never more than one tick behind it.
            let wall = DEMO_RAD_PER_SEC * t as f32;
            let lag = wall - rendered;
            assert!(
                lag >= -1e-4 && lag <= DEMO_RAD_PER_SEC * dt + 1e-4,
                "{hz} Hz at t={t}: lag {lag} outside [0, one tick]"
            );
        }
    }
}

#[test]
fn render_pose_is_a_pure_function_of_alpha() {
    let mut sim = Sim::with_tick_rate(10.0);
    sim.update(0.0);
    sim.update(0.1);
    sim.update(0.2);
    sim.update(0.25);

    let e = sim.demo_entity();
    let a = sim.model_matrix_at(e, 0.5).expect("model");
    let b = sim.model_matrix_at(e, 0.5).expect("model");
    assert_eq!(a.to_cols_array(), b.to_cols_array(), "same alpha, same matrix");

    // Different alpha, different matrix — nothing is being silently snapped to
    // the tick pose.
    let c = sim.model_matrix_at(e, 0.9).expect("model");
    assert_ne!(a.to_cols_array(), c.to_cols_array());

    // Out-of-range alphas clamp rather than extrapolate.
    assert_eq!(
        sim.model_matrix_at(e, 1.5).expect("model").to_cols_array(),
        sim.model_matrix_at(e, 1.0).expect("model").to_cols_array()
    );
    assert_eq!(
        sim.model_matrix_at(e, -0.5).expect("model").to_cols_array(),
        sim.model_matrix_at(e, 0.0).expect("model").to_cols_array()
    );
}
