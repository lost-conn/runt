//! The sim-side audio seam (DESIGN §8).
//!
//! Three claims are under test and they are the three the design leans on:
//!
//! 1. **One flush per tick, at a fixed point.** Not zero, not two, not "whenever
//!    a system happened to ask".
//! 2. **Audio is output, never input.** A world with nothing to play emits
//!    nothing; a replayed input trace emits the *same* events on the *same*
//!    ticks; the presence of a backend changes no simulation state.
//! 3. **The vocabulary matches `runt-audio`'s**, which is not checkable by the
//!    compiler because the two crates deliberately do not depend on each other.

use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};

use runt_core::audio::{
    flush_audio, AudioEvent, AudioOut, Listener, ParamId, PatchId, RecordingBackend, Rolloff,
    OUTBOX_CAP,
};
use runt_core::ecs::TickCount;
use runt_core::trace::InputTrace;
use runt_core::{Input, InputEvent, Key, Sim, SimConfig, Transform};

const BEEP: PatchId = PatchId::new("beep");

// ---------------------------------------------------------------------------
// Vocabulary — mirrored in runt-audio/tests/wire.rs
// ---------------------------------------------------------------------------

#[test]
fn the_patch_id_hash_matches_the_synths() {
    // `runt_audio::PatchId::new` must agree, or a `Play` names a preset the
    // synth has never heard of. Same three constants on both sides.
    assert_eq!(PatchId::new(""), PatchId(0xcbf2_9ce4_8422_2325));
    assert_eq!(PatchId::new("pluck"), PatchId(0x980a_104d_ddba_6b6a));
    assert_eq!(PatchId::new("drone"), PatchId(0x6d09_40c9_3eca_e8d1));
}

#[test]
fn the_shared_param_vocabulary_matches_the_synths() {
    assert_eq!(ParamId::GAIN.0, 0);
    assert_eq!(ParamId::PAN.0, 1);
    assert_eq!(ParamId::PITCH.0, 2);
    assert_eq!(ParamId::CUTOFF.0, 3);
}

// ---------------------------------------------------------------------------
// Positioning (DESIGN §8 phase-3 item 4)
// ---------------------------------------------------------------------------

/// A listener at the origin looking down −Z, the convention `Transform::looking_at`
/// builds.
fn origin_listener() -> Listener {
    Listener::from_pose(&Transform::looking_at(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y))
}

#[test]
fn pan_follows_the_side_of_the_screen_a_source_is_on() {
    let listener = origin_listener();
    let rolloff = Rolloff::default();

    let (_, ahead) = listener.spatialize(Vec3::new(0.0, 0.0, -10.0), rolloff);
    let (_, right) = listener.spatialize(Vec3::new(10.0, 0.0, -10.0), rolloff);
    let (_, left) = listener.spatialize(Vec3::new(-10.0, 0.0, -10.0), rolloff);

    assert!(ahead.abs() < 1e-5, "straight ahead is centre, got {ahead}");
    assert!(right > 0.0, "camera-space +X is the right ear, got {right}");
    assert!(left < 0.0, "and −X the left, got {left}");
    assert!((right + left).abs() < 1e-5, "and the law is symmetric");
}

#[test]
fn pan_is_clamped_and_saturates_off_axis() {
    let listener = origin_listener();
    let rolloff = Rolloff::default();
    // Hard right of the camera: 90° off axis, and `pan_width` above 1 means it
    // saturates before that.
    let (_, hard) = listener.spatialize(Vec3::new(10.0, 0.0, 0.0), rolloff);
    assert_eq!(hard, 1.0);
    let (_, hard_left) = listener.spatialize(Vec3::new(-10.0, 0.0, 0.0), rolloff);
    assert_eq!(hard_left, -1.0);
}

#[test]
fn a_source_behind_the_camera_pans_to_its_own_side() {
    // No HRTF and no front/back cue (DESIGN §8): behind-right is still right.
    let listener = origin_listener();
    let (_, pan) = listener.spatialize(Vec3::new(5.0, 0.0, 10.0), Rolloff::default());
    assert!(pan > 0.0, "got {pan}");
}

#[test]
fn a_rotated_listener_pans_relative_to_where_it_looks() {
    // The whole point of "camera-relative": turning the camera moves the sound,
    // without the sound moving.
    let listener = Listener {
        position: Vec3::ZERO,
        // Yaw 90° left, so world +X is now straight ahead.
        rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
    };
    let (_, pan) = listener.spatialize(Vec3::new(10.0, 0.0, 0.0), Rolloff::default());
    assert!(pan.abs() < 1e-5, "now dead ahead, got {pan}");
    // …and world +Z, which was behind, is now hard *left*: a +90° yaw swings the
    // camera's −Z view axis onto world −X, putting +Z on the far side.
    let (_, pan) = listener.spatialize(Vec3::new(0.0, 0.0, 10.0), Rolloff::default());
    assert!(pan < -0.9, "got {pan}");
}

#[test]
fn gain_is_flat_inside_the_reference_radius_and_one_over_d_outside() {
    let listener = origin_listener();
    let rolloff = Rolloff {
        reference: 4.0,
        ..Rolloff::default()
    };
    let gain_at = |d: f32| listener.spatialize(Vec3::new(0.0, 0.0, -d), rolloff).0;

    assert_eq!(gain_at(0.0), 1.0);
    assert_eq!(gain_at(2.0), 1.0);
    assert_eq!(gain_at(4.0), 1.0);
    assert!((gain_at(8.0) - 0.5).abs() < 1e-5);
    assert!((gain_at(40.0) - 0.1).abs() < 1e-5);
    assert!(gain_at(4000.0) > 0.0, "distant is quiet, not culled");
}

#[test]
fn a_source_on_top_of_the_listener_is_centred_rather_than_infinite() {
    let listener = origin_listener();
    let (gain, pan) = listener.spatialize(Vec3::ZERO, Rolloff::default());
    assert_eq!((gain, pan), (1.0, 0.0));
}

#[test]
fn a_non_finite_position_cannot_reach_a_filter() {
    let listener = origin_listener();
    let (gain, pan) = listener.spatialize(Vec3::new(f32::NAN, 0.0, -1.0), Rolloff::default());
    assert!(gain.is_finite() && pan.is_finite(), "got {gain}, {pan}");
}

// ---------------------------------------------------------------------------
// Flush semantics
// ---------------------------------------------------------------------------

/// A `FixedSim` system that plays one note per tick, so the flush is being
/// observed against a known number of events.
fn beep_every_tick(mut out: ResMut<AudioOut>, tick: Res<TickCount>) {
    out.play(BEEP, tick.0, 1.0, 0.0);
}

#[test]
fn the_queue_is_flushed_exactly_once_per_tick() {
    let mut sim = Sim::from_config(SimConfig::default().without_scene());
    sim.fixed_sim_mut()
        .add_systems(beep_every_tick.before(flush_audio));

    for expected in 1..=10u64 {
        sim.tick();
        assert_eq!(sim.audio_out().flushes(), expected);
        assert_eq!(sim.tick_count(), expected);
    }
    assert_eq!(sim.audio_events().len(), 10);
}

#[test]
fn the_tick_queue_is_empty_again_after_the_flush() {
    let mut sim = Sim::from_config(SimConfig::default().without_scene());
    sim.fixed_sim_mut()
        .add_systems(beep_every_tick.before(flush_audio));
    sim.tick();
    assert!(
        sim.audio_out().queued().is_empty(),
        "a flushed batch must not be able to go out twice"
    );
    sim.tick();
    assert_eq!(sim.audio_events().len(), 2, "and the next tick adds one more");
}

#[test]
fn events_are_stamped_with_the_tick_that_produced_them() {
    // Zero-based, matching `InputTrace`'s indexing — the flush runs before the
    // tick counter turns over precisely so these two agree.
    let mut sim = Sim::from_config(SimConfig::default().without_scene());
    sim.fixed_sim_mut()
        .add_systems(beep_every_tick.before(flush_audio));
    for _ in 0..5 {
        sim.tick();
    }
    let ticks: Vec<u64> = sim.audio_events().iter().map(|(t, _)| *t).collect();
    assert_eq!(ticks, vec![0, 1, 2, 3, 4]);

    // And the seed each system read was the same tick index.
    for (tick, event) in sim.audio_events() {
        let AudioEvent::Play { seed, .. } = event else {
            panic!("expected a Play, got {event:?}");
        };
        assert_eq!(seed, tick);
    }
}

#[test]
fn a_world_with_nothing_to_play_emits_nothing() {
    // Physics-free, game-free: the engine itself never makes a sound.
    let mut sim = Sim::from_config(SimConfig::default().without_scene());
    for _ in 0..120 {
        sim.tick();
    }
    assert!(sim.audio_events().is_empty());
    assert_eq!(sim.audio_out().flushes(), 120, "but the flush still ran");
}

#[test]
fn the_engine_demo_scene_is_silent() {
    // `assets/demo.ron` has no game logic attached, so a full second of it must
    // produce no audio at all.
    let mut sim = Sim::new();
    for _ in 0..60 {
        sim.tick();
    }
    assert!(sim.audio_events().is_empty());
}

#[test]
fn a_voice_id_is_minted_once_and_addresses_the_same_note_later() {
    let mut out = AudioOut::new();
    let a = out.play(BEEP, 1, 1.0, 0.0);
    let b = out.play(BEEP, 2, 1.0, 0.0);
    assert_ne!(a, b);
    out.set_param(a, ParamId::PITCH, 2.0);
    out.stop(b);

    assert_eq!(
        out.queued(),
        &[
            AudioEvent::Play {
                voice: a,
                patch: BEEP,
                seed: 1,
                gain: 1.0,
                pan: 0.0
            },
            AudioEvent::Play {
                voice: b,
                patch: BEEP,
                seed: 2,
                gain: 1.0,
                pan: 0.0
            },
            AudioEvent::SetParam {
                voice: a,
                id: ParamId::PITCH,
                value: 2.0
            },
            AudioEvent::Stop { voice: b },
        ]
    );
}

#[test]
fn a_non_finite_gain_or_pan_never_leaves_the_queue() {
    let mut out = AudioOut::new();
    out.play(BEEP, 0, f32::NAN, f32::INFINITY);
    out.set_param(runt_core::VoiceId(0), ParamId::GAIN, f32::NAN);
    for event in out.queued() {
        match *event {
            AudioEvent::Play { gain, pan, .. } => {
                assert!(gain.is_finite() && pan.is_finite());
                assert!((-1.0..=1.0).contains(&pan));
            }
            AudioEvent::SetParam { value, .. } => assert!(value.is_finite()),
            AudioEvent::Stop { .. } => {}
        }
    }
}

#[test]
fn an_undrained_outbox_is_capped_rather_than_unbounded() {
    let mut out = AudioOut::new();
    for i in 0..(OUTBOX_CAP as u64 + 100) {
        out.play(BEEP, i, 1.0, 0.0);
    }
    // One flush, more events than the cap.
    let mut sim = Sim::from_config(SimConfig::default().without_scene());
    *sim.audio_out_mut() = out;
    sim.tick();

    assert_eq!(sim.audio_events().len(), OUTBOX_CAP);
    assert_eq!(sim.audio_out().dropped(), 100);
    // The *newest* survived: audio drops history, not the present.
    let AudioEvent::Play { seed, .. } = sim.audio_events().last().unwrap().1 else {
        panic!("expected a Play");
    };
    assert_eq!(seed, OUTBOX_CAP as u64 + 99);
}

// ---------------------------------------------------------------------------
// The host seam
// ---------------------------------------------------------------------------

#[test]
fn draining_hands_the_batch_to_the_backend_and_clears_it() {
    let mut sim = Sim::from_config(SimConfig::default().without_scene());
    sim.fixed_sim_mut()
        .add_systems(beep_every_tick.before(flush_audio));

    let mut backend = RecordingBackend::default();
    for _ in 0..3 {
        sim.tick();
    }
    sim.drain_audio(&mut backend);

    assert_eq!(backend.events.len(), 3);
    assert_eq!(backend.batches, 1, "three ticks in one frame is one submit");
    assert!(sim.audio_events().is_empty());

    // A frame with no ticks submits nothing at all rather than an empty batch.
    sim.drain_audio(&mut backend);
    assert_eq!(backend.batches, 1);
}

#[test]
fn the_backend_cannot_influence_the_simulation() {
    // The `StatusLine` rule, restated for audio: with and without a backend the
    // tick stream is identical. If audio could ever feed back, this is where it
    // would show up.
    let run = |wired: bool| {
        let mut sim = Sim::from_config(SimConfig::default().without_scene());
        sim.fixed_sim_mut()
            .add_systems(beep_every_tick.before(flush_audio));
        let mut backend = RecordingBackend::default();
        let mut ticks = Vec::new();
        for _ in 0..20 {
            sim.tick();
            if wired {
                sim.drain_audio(&mut backend);
            }
            ticks.push(sim.tick_count());
        }
        ticks
    };
    assert_eq!(run(true), run(false));
}

// ---------------------------------------------------------------------------
// Replay determinism (DESIGN §4 extended to §8)
// ---------------------------------------------------------------------------

/// Plays a note on every tick a key goes down, seeded by the tick — so the event
/// stream is a direct function of the input trace and nothing else.
fn beep_on_keypress(input: Res<Input>, tick: Res<TickCount>, mut out: ResMut<AudioOut>) {
    if input.just_pressed(Key::Space) {
        out.play(BEEP, tick.0, 0.8, 0.25);
    }
    if input.just_pressed(Key::R) {
        out.stop(runt_core::VoiceId(0));
    }
}

fn keypress_sim() -> Sim {
    let mut sim = Sim::from_config(SimConfig::default().without_scene());
    // Where game code goes: after the physics solve, before the flush. That is
    // also, load-bearingly, *after* `trace::apply` — which is installed between
    // `update_overlap_messages` and `integrate_balls` — so a replayed trace has
    // already replaced the tick's `Input` by the time this reads it. Without the
    // constraint the system runs in insertion order, lands ahead of `apply`, and
    // a replay silently hears nothing.
    sim.fixed_sim_mut().add_systems(
        beep_on_keypress
            .after(runt_core::physics::resolve_overlaps)
            .before(flush_audio),
    );
    sim
}

#[test]
fn a_replayed_trace_produces_the_same_events_on_the_same_ticks() {
    let script = InputTrace::from_pairs([
        (3, InputEvent::KeyDown(Key::Space)),
        (4, InputEvent::KeyUp(Key::Space)),
        (11, InputEvent::KeyDown(Key::Space)),
        (12, InputEvent::KeyUp(Key::Space)),
        (25, InputEvent::KeyDown(Key::R)),
        (26, InputEvent::KeyUp(Key::R)),
    ]);

    let log = |chunk: usize| {
        let mut sim = keypress_sim();
        sim.play_input_trace(script.clone());
        // Drive the sim in different-sized bites so the *host cadence* is not
        // what the two runs have in common.
        let mut ran = 0;
        while ran < 40 {
            let n = chunk.min(40 - ran);
            for _ in 0..n {
                sim.tick();
            }
            ran += n;
        }
        sim.audio_events().to_vec()
    };

    let a = log(1);
    let b = log(7);
    assert!(!a.is_empty(), "the script must actually make a sound");
    assert_eq!(a, b, "same trace → same tick-indexed audio, bit for bit");

    // And the log says what the script said, on the ticks the script said it.
    let ticks: Vec<u64> = a.iter().map(|(t, _)| *t).collect();
    assert_eq!(ticks, vec![3, 11, 25]);
    assert!(matches!(a[0].1, AudioEvent::Play { seed: 3, .. }));
    assert!(matches!(a[2].1, runt_core::AudioEvent::Stop { .. }));
}

#[test]
fn a_recorded_run_replays_to_the_same_audio() {
    // The end-to-end version: play a run through the host's own input path,
    // record the trace, replay it, compare the audio logs.
    let mut live = keypress_sim();
    live.record_input_trace();
    for tick in 0..30u64 {
        if tick == 5 || tick == 17 {
            live.push_input(InputEvent::KeyDown(Key::Space));
        }
        if tick == 6 || tick == 18 {
            live.push_input(InputEvent::KeyUp(Key::Space));
        }
        live.tick();
    }
    let recorded = live.audio_events().to_vec();
    let trace = live.input_trace().expect("recording").clone();

    let mut replay = keypress_sim();
    replay.play_input_trace(trace);
    for _ in 0..30 {
        replay.tick();
    }

    assert_eq!(recorded.len(), 2);
    assert_eq!(recorded, replay.audio_events());
}
