//! Seeded runs, replayable from an input trace (DESIGN §12 step 6, §4).
//!
//! > *replays are just recorded input traces + seeds* — DESIGN §4
//!
//! The seed is in `level1.ron`; the trace is [`script`]. What has to hold is
//! that those two things determine the run *completely* — not the host's frame
//! rate, not when it happened to poll, not how the wall clock was chopped up.
//!
//! So the comparison is not "the ball ends up in about the same place". It is a
//! per-tick fingerprint of **every transform in the world plus the whole
//! [`GameState`]**, hashed inside the tick by [`fingerprint`], collected over
//! hundreds of ticks, and compared as a `Vec<u64>`. Two runs agree on the bit or
//! they do not agree.
//!
//! The script is chosen so the run actually exercises the game rather than
//! rolling to a stop on flat ground: it takes a pickup around tick 76 and then
//! drives off the edge of the patch, which the kill plane catches twice.

use bevy_ecs::prelude::*;

use runt_ball::game::{GameState, Phase};
use runt_core::ecs::advance_tick_count;
use runt_core::trace::InputTrace;
use runt_core::{InputEvent, Key, Sim, Transform};

/// Hold "forward and left" off the spawn, straighten up, and keep going until
/// the ground runs out. Verified below to score and to fall.
fn script() -> InputTrace {
    InputTrace::from_pairs([
        (2, InputEvent::KeyDown(Key::W)),
        (2, InputEvent::KeyDown(Key::A)),
        (18, InputEvent::KeyUp(Key::A)),
    ])
}

const TICKS: u64 = 900;

// ---------------------------------------------------------------------------
// Per-tick fingerprint
// ---------------------------------------------------------------------------

#[derive(Resource, Default)]
struct Fingerprints(Vec<u64>);

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn mix(h: &mut u64, word: u64) {
    *h ^= word;
    *h = h.wrapping_mul(FNV_PRIME);
}

/// `FixedSim` (tail): hash the whole world's pose plus the game state.
///
/// Rows are sorted before hashing, so the fingerprint does not depend on
/// archetype iteration order — a difference in *that* would be a real
/// divergence the test should not be able to manufacture on its own.
fn fingerprint(
    mut out: ResMut<Fingerprints>,
    state: Res<GameState>,
    transforms: Query<&Transform>,
) {
    let mut rows: Vec<[u32; 10]> = transforms
        .iter()
        .map(|t| {
            let (p, r, s) = (t.translation, t.rotation, t.scale);
            [
                p.x.to_bits(), p.y.to_bits(), p.z.to_bits(),
                r.x.to_bits(), r.y.to_bits(), r.z.to_bits(), r.w.to_bits(),
                s.x.to_bits(), s.y.to_bits(), s.z.to_bits(),
            ]
        })
        .collect();
    rows.sort_unstable();

    let mut h = FNV_OFFSET;
    mix(&mut h, rows.len() as u64);
    for row in rows {
        for word in row {
            mix(&mut h, word as u64);
        }
    }
    mix(&mut h, state.score as u64);
    mix(&mut h, state.total as u64);
    mix(&mut h, state.elapsed_ticks);
    mix(&mut h, state.resets as u64);
    mix(&mut h, (state.phase == Phase::Won) as u64);
    mix(&mut h, state.kill_y.to_bits() as u64);
    out.0.push(h);
}

/// A game replaying `trace`, with the fingerprint recorder installed at the very
/// end of the tick.
fn replaying(trace: InputTrace) -> Sim {
    let mut sim = runt_ball::headless_sim();
    sim.play_input_trace(trace);
    sim.world_mut().init_resource::<Fingerprints>();
    sim.fixed_sim_mut()
        .add_systems(fingerprint.after(advance_tick_count));
    sim
}

fn stream(sim: &Sim) -> &[u64] {
    &sim.world().resource::<Fingerprints>().0
}

/// Drive a sim through the host's `update(elapsed)` entry point on a given
/// cadence of wall-clock instants — the path a real window takes.
fn drive(trace: InputTrace, schedule: &[f64]) -> Sim {
    let mut sim = replaying(trace);
    sim.update(0.0); // Establish the origin without ticking.
    for &t in schedule {
        sim.update(t);
    }
    sim
}

fn uniform(seconds: f64) -> Vec<f64> {
    let ticks = (seconds * 60.0).round() as u64;
    (1..=ticks).map(|i| i as f64 / 60.0).collect()
}

/// A host that stutters: 5 ms, then 30 ms, then 7 ms, forever, landing exactly
/// on the same final instant.
fn ragged(seconds: f64) -> Vec<f64> {
    let steps = [0.005, 0.030, 0.007];
    let mut times = Vec::new();
    let (mut t, mut i) = (0.0f64, 0usize);
    while t < seconds {
        t += steps[i % steps.len()];
        i += 1;
        times.push(t.min(seconds));
    }
    if *times.last().expect("non-empty") != seconds {
        times.push(seconds);
    }
    times
}

// ---------------------------------------------------------------------------

#[test]
fn the_script_actually_plays_the_game() {
    // Without this, every determinism test below could pass on a ball that
    // never moved. The script has to *do* something: score, and fall.
    let mut sim = replaying(script());
    for _ in 0..TICKS {
        sim.tick();
    }
    let state = sim.world().resource::<GameState>();
    assert!(
        state.score >= 1,
        "the script collected nothing; determinism over an idle ball proves nothing"
    );
    assert!(
        state.resets >= 1,
        "the script never left the map, so the kill plane is untested here"
    );
    assert_eq!(stream(&sim).len(), TICKS as usize);
}

#[test]
fn a_ragged_host_and_a_smooth_one_replay_the_same_run() {
    // DESIGN §4's promise, end to end and through the *whole game*: same trace,
    // same seed, same final instant → the same ticks with the same inputs, and
    // therefore the same floats. One host runs a clean 60 fps; the other
    // stutters through 5/30/7 ms chunks and calls `update` a different number of
    // times.
    let seconds = TICKS as f64 / 60.0;
    let smooth = uniform(seconds);
    let jerky = ragged(seconds);
    assert_ne!(smooth.len(), jerky.len(), "the two cadences must differ");

    let a = drive(script(), &smooth);
    let b = drive(script(), &jerky);

    assert_eq!(a.tick_count(), b.tick_count(), "same wall time, same ticks");
    assert!(a.tick_count() >= TICKS - 1, "got {} ticks", a.tick_count());
    assert_eq!(
        stream(&a),
        stream(&b),
        "the per-tick transform + GameState stream diverged"
    );
    assert_eq!(
        a.world().resource::<GameState>(),
        b.world().resource::<GameState>()
    );
}

#[test]
fn the_same_trace_twice_is_the_same_run_twice() {
    let a = drive(script(), &uniform(6.0));
    let b = drive(script(), &uniform(6.0));
    assert_eq!(stream(&a), stream(&b));
}

#[test]
fn a_recorded_run_replays_through_a_file() {
    // The round trip the `--record` / `--replay` flags take: play, record what
    // the tick saw, serialize to postcard, read it back, replay. The replay must
    // reproduce the original run bit for bit — and the trace it re-records must
    // come back identical, which is what makes `--record --replay` a self-check.
    let mut original = runt_ball::headless_sim();
    original.play_input_trace(script());
    original.record_input_trace();
    original.world_mut().init_resource::<Fingerprints>();
    original
        .fixed_sim_mut()
        .add_systems(fingerprint.after(advance_tick_count));
    for _ in 0..TICKS {
        original.tick();
    }

    let recorded = original.input_trace().expect("recording").clone();
    assert!(!recorded.is_empty(), "the recorder captured nothing");

    let bytes = recorded.to_bytes().expect("postcard");
    let restored = InputTrace::from_bytes(&bytes).expect("postcard round trip");
    assert_eq!(restored, recorded, "the file is not what was recorded");

    let mut replay = replaying(restored);
    replay.record_input_trace();
    for _ in 0..TICKS {
        replay.tick();
    }

    assert_eq!(
        stream(&replay),
        stream(&original),
        "the replay diverged from the run it was recorded from"
    );
    assert_eq!(
        replay.world().resource::<GameState>(),
        original.world().resource::<GameState>()
    );
    assert_eq!(
        replay.input_trace().expect("recording"),
        &recorded,
        "re-recording a replay must reproduce the trace it was given"
    );
}

#[test]
fn a_replay_ignores_whatever_the_host_is_doing() {
    // A trace replaces the tick's input outright, so a player mashing keys over
    // a replaying window cannot change it. (Which is also what makes
    // `--record --replay` meaningful: the recorder sees the trace, not the
    // keyboard.)
    let clean = {
        let mut sim = replaying(script());
        for _ in 0..300 {
            sim.tick();
        }
        stream(&sim).to_vec()
    };

    let heckled = {
        let mut sim = replaying(script());
        for tick in 0..300 {
            // Every key the game uses, pressed and released at odd moments.
            if tick % 7 == 0 {
                sim.push_input(InputEvent::KeyDown(Key::D));
                sim.push_input(InputEvent::KeyDown(Key::S));
            }
            if tick % 11 == 0 {
                sim.push_input(InputEvent::KeyUp(Key::D));
                sim.push_input(InputEvent::KeyUp(Key::S));
            }
            sim.tick();
        }
        stream(&sim).to_vec()
    };

    assert_eq!(clean, heckled, "host input leaked into a replay");
}

#[test]
fn quality_cannot_change_the_run() {
    // DESIGN §9's claim, restated at the level of a whole game: the terrain mesh
    // is a *view* of the analytic field, so a low-end device draws fewer
    // triangles and plays exactly the same level.
    let run = |quality: f32| {
        let mut sim = Sim::from_config(
            runt_core::SimConfig::default()
                .with_scene(runt_ball::LEVEL1_RON)
                .with_quality(quality),
        );
        runt_ball::game::setup(&mut sim);
        sim.play_input_trace(script());
        sim.world_mut().init_resource::<Fingerprints>();
        sim.fixed_sim_mut()
            .add_systems(fingerprint.after(advance_tick_count));
        for _ in 0..TICKS {
            sim.tick();
        }
        (
            sim.world()
                .get::<runt_core::MeshRef>(sim.scene_entity("ground").expect("ground"))
                .expect("MeshRef")
                .0,
            stream(&sim).to_vec(),
            sim.world().resource::<GameState>().clone(),
        )
    };

    let (coarse_mesh, coarse, coarse_state) = run(0.3);
    let (full_mesh, full, full_state) = run(1.0);
    assert_ne!(
        coarse_mesh, full_mesh,
        "the tiers must really produce different geometry, or this proves nothing"
    );
    assert_eq!(coarse, full, "visual LOD changed the game");
    assert_eq!(coarse_state, full_state);
}
