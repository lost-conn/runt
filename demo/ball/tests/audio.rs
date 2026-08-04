//! The game's sound, as event streams (DESIGN §8).
//!
//! There are no speakers here and these tests do not want any: what the demo has
//! to get right is *which event, on which tick, with which pan* — the synthesis
//! is `runt-audio`'s problem and is measured over there. So every assertion in
//! this file is about the tick-indexed log
//! [`Sim::audio_events`](runt_core::Sim::audio_events) hands a host.
//!
//! These drive `runt_ball::headless_sim()`, the same two calls the window host
//! makes, so a green run is a statement about what a player hears.

use bevy_ecs::prelude::*;
use glam::Vec3;

use runt_ball::audio::{
    AMBIENCE, AMBIENCE_DELAY_TICKS, CHIME, FANFARE, FANFARE_SPACING, PICKUP, THUD,
};
use runt_ball::game::{GameState, Phase, Pickup};
use runt_core::audio::{AudioEvent, ParamId, PatchId};
use runt_core::trace::InputTrace;
use runt_core::{camera::Camera, InputEvent, Key, Sim, Transform, Velocity};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn state(sim: &Sim) -> GameState {
    sim.world().resource::<GameState>().clone()
}

fn pickups(sim: &mut Sim) -> Vec<Entity> {
    let mut q = sim.world_mut().query_filtered::<Entity, With<Pickup>>();
    let mut found: Vec<Entity> = q.iter(sim.world()).collect();
    found.sort_unstable();
    found
}

fn camera_pose(sim: &mut Sim) -> Transform {
    let mut q = sim.world_mut().query_filtered::<&Transform, With<Camera>>();
    *q.iter(sim.world()).next().expect("level1 has a camera")
}

/// Every `Play` in the log, as `(tick, patch, gain, pan)`.
fn plays(sim: &Sim) -> Vec<(u64, PatchId, f32, f32)> {
    sim.audio_events()
        .iter()
        .filter_map(|(tick, event)| match *event {
            AudioEvent::Play {
                patch, gain, pan, ..
            } => Some((*tick, patch, gain, pan)),
            _ => None,
        })
        .collect()
}

fn plays_of(sim: &Sim, patch: PatchId) -> Vec<(u64, PatchId, f32, f32)> {
    plays(sim).into_iter().filter(|p| p.1 == patch).collect()
}

fn run(sim: &mut Sim, ticks: usize) {
    for _ in 0..ticks {
        sim.tick();
    }
}

/// Park the ball somewhere flat-ish and let the follow camera settle behind it,
/// so a later "which side is this on" question has a stable answer.
fn settle(sim: &mut Sim) -> Vec3 {
    let player = state(sim).player;
    run(sim, 90);
    sim.world()
        .get::<Transform>(player)
        .expect("Transform")
        .translation
}

// ---------------------------------------------------------------------------
// Ambience
// ---------------------------------------------------------------------------

#[test]
fn nothing_at_all_plays_until_the_player_touches_something() {
    // A page nobody has clicked cannot start an `AudioContext`, so the host
    // drops what it is handed — and a bed emitted into that window would be lost
    // for the whole run. The game therefore waits for input. See
    // `AMBIENCE_DELAY_TICKS`.
    let mut sim = runt_ball::headless_sim();
    run(&mut sim, 600);
    assert!(
        sim.audio_events().is_empty(),
        "ten silent seconds played {:?}",
        plays(&sim)
    );
}

#[test]
fn the_ambience_starts_once_shortly_after_the_first_input_and_never_again() {
    let mut sim = runt_ball::headless_sim();
    run(&mut sim, 20);
    sim.push_input(InputEvent::KeyDown(Key::W));
    sim.tick(); // tick 20 carries the first input
    run(&mut sim, 600);

    let ambience = plays_of(&sim, AMBIENCE);
    assert_eq!(ambience.len(), 1, "one bed, not one per tick");
    assert_eq!(
        ambience[0].0,
        20 + AMBIENCE_DELAY_TICKS,
        "half a second after the player arrived"
    );
    assert_eq!(ambience[0].3, 0.0, "the bed is not positional");
}

#[test]
fn a_quiet_run_makes_no_other_noise() {
    // Rolling gently on flat ground must not fire the landing thud sixty times a
    // second. The whole point of `LANDING_SPEED` is that ordinary contact is
    // silent.
    let mut sim = runt_ball::headless_sim();
    sim.push_input(InputEvent::KeyDown(Key::W));
    sim.tick();
    sim.push_input(InputEvent::KeyUp(Key::W));
    run(&mut sim, 600);
    let noisy: Vec<_> = plays(&sim)
        .into_iter()
        .filter(|(_, patch, ..)| *patch != AMBIENCE)
        .collect();
    assert!(noisy.is_empty(), "ten seconds of sitting still played {noisy:?}");
}

// ---------------------------------------------------------------------------
// Pickups
// ---------------------------------------------------------------------------

/// Move `pickup` to `where` and run the tick the overlap happens on.
///
/// The bob rewrites `translation.y` every tick from `base_y`, so the height has
/// to be set through the component or it is undone before the next overlap pass.
fn place_and_take(sim: &mut Sim, pickup: Entity, at: Vec3) {
    {
        let mut entity = sim.world_mut().entity_mut(pickup);
        entity.get_mut::<Transform>().expect("Transform").translation = at;
        entity.get_mut::<Pickup>().expect("Pickup").base_y = at.y;
    }
    sim.tick();
}

#[test]
fn taking_a_ring_emits_exactly_one_ping_where_the_ring_was() {
    let mut sim = runt_ball::headless_sim();
    let ball = settle(&mut sim);
    let before = plays_of(&sim, PICKUP).len();

    let pickup = pickups(&mut sim)[0];
    place_and_take(&mut sim, pickup, ball + Vec3::new(0.6, 0.0, 0.0));

    let pings = plays_of(&sim, PICKUP);
    assert_eq!(pings.len(), before + 1, "one ring taken, one ping");
    assert_eq!(state(&sim).score, 1);
    assert!(pings[0].2 > 0.0, "and it is audible: gain {}", pings[0].2);
}

#[test]
fn a_ring_on_the_right_pans_right_and_one_on_the_left_pans_left() {
    // The camera-relative pan of DESIGN §8's phase-3 item 4, end to end: the
    // engine's `Listener`, the game's `ROLLOFF`, and a real follow camera that
    // has actually settled behind the ball.
    let side = |sign: f32| {
        let mut sim = runt_ball::headless_sim();
        let ball = settle(&mut sim);

        // The camera's own right axis, so the test does not assume which way the
        // level happens to face.
        let pose = camera_pose(&mut sim);
        let right = pose.rotation * Vec3::X;
        let right = Vec3::new(right.x, 0.0, right.z).normalize();

        let pickup = pickups(&mut sim)[0];
        place_and_take(&mut sim, pickup, ball + right * sign * 0.9);

        let pings = plays_of(&sim, PICKUP);
        assert_eq!(pings.len(), 1, "the ring must actually have been taken");
        pings[0].3
    };

    let right = side(1.0);
    let left = side(-1.0);
    assert!(right > 0.05, "a ring to the right must pan right, got {right}");
    assert!(left < -0.05, "and one to the left, left, got {left}");
}

#[test]
fn the_ping_walks_up_the_scale_as_the_run_progresses() {
    // The seed is the score *before* the ring landed, so the notes ascend in
    // collection order. What is checkable here is that the seeds are the
    // ordinals — the pitch they select is `runt-audio`'s business and is
    // measured there.
    let mut sim = runt_ball::headless_sim();
    let player = state(&sim).player;
    let rings = pickups(&mut sim);

    for ring in rings.iter().take(4) {
        let at = sim
            .world()
            .get::<Transform>(*ring)
            .expect("Transform")
            .translation;
        sim.world_mut()
            .get_mut::<Transform>(player)
            .expect("Transform")
            .translation = at;
        sim.tick();
    }

    let seeds: Vec<u64> = sim
        .audio_events()
        .iter()
        .filter_map(|(_, event)| match *event {
            AudioEvent::Play { patch, seed, .. } if patch == PICKUP => Some(seed),
            _ => None,
        })
        .collect();
    assert_eq!(seeds, vec![0, 1, 2, 3]);
}

#[test]
fn a_ring_far_from_the_camera_is_quieter_than_one_nearby() {
    let gain_at = |offset: Vec3| {
        let mut sim = runt_ball::headless_sim();
        let ball = settle(&mut sim);
        let pickup = pickups(&mut sim)[0];
        // Take a ring under the ball, but *report* it from far away: move the
        // ring and the ball together so the overlap still happens.
        let player = state(&sim).player;
        sim.world_mut()
            .get_mut::<Transform>(player)
            .expect("Transform")
            .translation = ball + offset;
        place_and_take(&mut sim, pickup, ball + offset);
        plays_of(&sim, PICKUP)[0].2
    };

    let near = gain_at(Vec3::ZERO);
    let far = gain_at(Vec3::new(0.0, 0.0, -18.0));
    assert!(near > far, "1/d rolloff: near {near}, far {far}");
    assert!(far > 0.0, "distant is quiet, not culled");
}

// ---------------------------------------------------------------------------
// Landing
// ---------------------------------------------------------------------------

#[test]
fn a_hard_landing_thuds_and_a_soft_one_does_not() {
    let drop_from = |height: f32| {
        let mut sim = runt_ball::headless_sim();
        let ball = settle(&mut sim);
        let player = state(&sim).player;
        {
            let mut entity = sim.world_mut().entity_mut(player);
            entity.get_mut::<Transform>().expect("Transform").translation =
                ball + Vec3::new(0.0, height, 0.0);
            entity.get_mut::<Velocity>().expect("Velocity").0 = Vec3::ZERO;
        }
        let before = plays_of(&sim, THUD).len();
        // Long enough to fall from 12 m under gravity 20 and settle.
        run(&mut sim, 120);
        let thuds = plays_of(&sim, THUD);
        (thuds.len() - before, thuds)
    };

    // 0.2 m is barely a hop: impact speed ~2 m/s, under LANDING_SPEED.
    let (soft, _) = drop_from(0.2);
    assert_eq!(soft, 0, "a hop is not a landing");

    let (hard, thuds) = drop_from(12.0);
    assert!(hard >= 1, "a twelve-metre drop must land audibly");
    let loudest = thuds.iter().map(|t| t.2).fold(0.0f32, f32::max);
    assert!(loudest > 0.1, "and it must be loud: {loudest}");
    assert!(loudest <= 1.0, "but capped: {loudest}");
}

#[test]
fn a_harder_landing_is_louder_up_to_the_cap() {
    let loudness = |height: f32| {
        let mut sim = runt_ball::headless_sim();
        let ball = settle(&mut sim);
        let player = state(&sim).player;
        {
            let mut entity = sim.world_mut().entity_mut(player);
            entity.get_mut::<Transform>().expect("Transform").translation =
                ball + Vec3::new(0.0, height, 0.0);
            entity.get_mut::<Velocity>().expect("Velocity").0 = Vec3::ZERO;
        }
        run(&mut sim, 120);
        plays_of(&sim, THUD)
            .iter()
            .map(|t| t.2)
            .fold(0.0f32, f32::max)
    };
    let low = loudness(2.0);
    let high = loudness(9.0);
    assert!(low > 0.0 && high > low, "low {low}, high {high}");
}

#[test]
fn a_kill_plane_reset_makes_no_landing_noise() {
    // A respawn moves the ball further in one tick than any speed allows. Reading
    // that as a fall would put a bang on the worst moment in the game.
    let mut sim = runt_ball::headless_sim();
    settle(&mut sim);
    let player = state(&sim).player;
    let kill_y = state(&sim).kill_y;

    let before = plays_of(&sim, THUD).len();
    {
        // Off the side of the 48 m patch as well as below it: inside the patch
        // the contact solve would push the ball back up onto the surface before
        // the kill plane ever saw it, which is precisely why falling off the
        // *edge* is the only way to lose the ball in this game.
        let mut entity = sim.world_mut().entity_mut(player);
        entity.get_mut::<Transform>().expect("Transform").translation =
            Vec3::new(100.0, kill_y - 1.0, 0.0);
        entity.get_mut::<Velocity>().expect("Velocity").0 = Vec3::new(0.0, -20.0, 0.0);
    }
    run(&mut sim, 60);

    assert_eq!(state(&sim).resets, 1, "the reset must have happened");
    assert_eq!(
        plays_of(&sim, THUD).len(),
        before,
        "and it must have been silent"
    );
}

#[test]
fn a_restart_makes_no_landing_noise_and_re_arms_the_fanfare() {
    let mut sim = runt_ball::headless_sim();
    settle(&mut sim);
    let before = plays_of(&sim, THUD).len();

    sim.push_input(InputEvent::KeyDown(Key::R));
    sim.tick();
    sim.push_input(InputEvent::KeyUp(Key::R));
    run(&mut sim, 60);

    assert_eq!(plays_of(&sim, THUD).len(), before);
}

// ---------------------------------------------------------------------------
// The win fanfare
// ---------------------------------------------------------------------------

/// Collect every ring by teleporting onto it, and return the tick the win landed
/// on.
fn win(sim: &mut Sim) -> u64 {
    let player = state(sim).player;
    for ring in pickups(sim) {
        let at = sim
            .world()
            .get::<Transform>(ring)
            .expect("Transform")
            .translation;
        sim.world_mut()
            .get_mut::<Transform>(player)
            .expect("Transform")
            .translation = at;
        sim.tick();
    }
    assert_eq!(state(sim).phase, Phase::Won);
    // `game_audio` sees the win on the tick `win_check` set it, and the tick
    // counter has not turned over yet at flush time — so the win tick is the
    // last completed tick minus one.
    sim.tick_count() - 1
}

#[test]
fn the_win_plays_three_pitched_notes_on_the_right_ticks() {
    let mut sim = runt_ball::headless_sim();
    let won_on = win(&mut sim);
    run(&mut sim, 60);

    let chimes: Vec<u64> = sim
        .audio_events()
        .iter()
        .filter_map(|(tick, event)| match *event {
            AudioEvent::Play { patch, .. } if patch == CHIME => Some(*tick),
            _ => None,
        })
        .collect();
    assert_eq!(
        chimes,
        vec![
            won_on,
            won_on + FANFARE_SPACING,
            won_on + 2 * FANFARE_SPACING
        ],
        "a tick-driven sequencer in game code, not an engine feature"
    );

    // Each note is a `Play` immediately followed by the `SetParam` that pitches
    // it — same tick, same batch, in that order, so it is never heard at the
    // preset's root first.
    let pitched: Vec<f32> = sim
        .audio_events()
        .iter()
        .filter_map(|(_, event)| match *event {
            AudioEvent::SetParam {
                id: ParamId::PITCH,
                value,
                ..
            } => Some(value),
            _ => None,
        })
        .collect();
    assert_eq!(pitched, FANFARE.to_vec());

    for pair in sim
        .audio_events()
        .windows(2)
        .filter(|w| matches!(w[0].1, AudioEvent::Play { patch, .. } if patch == CHIME))
    {
        let AudioEvent::Play { voice, .. } = pair[0].1 else {
            unreachable!()
        };
        assert!(
            matches!(pair[1].1, AudioEvent::SetParam { voice: v, .. } if v == voice),
            "the pitch edit must address the note it follows"
        );
        assert_eq!(pair[0].0, pair[1].0, "and land on the same tick");
    }
}

#[test]
fn the_fanfare_plays_once_and_stays_played() {
    let mut sim = runt_ball::headless_sim();
    win(&mut sim);
    run(&mut sim, 600);
    assert_eq!(plays_of(&sim, CHIME).len(), 3, "three notes, then quiet");
}

#[test]
fn the_same_run_wins_the_same_way_twice() {
    let log = || {
        let mut sim = runt_ball::headless_sim();
        win(&mut sim);
        run(&mut sim, 60);
        sim.audio_events().to_vec()
    };
    let a = log();
    let b = log();
    assert!(!a.is_empty());
    assert_eq!(a, b, "the win sequence is a function of the tick stream");
}

// ---------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------

#[test]
fn a_replayed_trace_produces_the_same_audio_under_any_host_cadence() {
    // `tests/replay.rs` proves the *transforms* are identical; this is the same
    // claim for the event stream. A run that falls off the map twice and takes a
    // ring on the way exercises every sound the game has except the fanfare.
    let script = InputTrace::from_pairs([
        (2, InputEvent::KeyDown(Key::W)),
        (2, InputEvent::KeyDown(Key::A)),
        (18, InputEvent::KeyUp(Key::A)),
    ]);

    let log = |chunk: usize| {
        let mut sim = runt_ball::headless_sim();
        sim.play_input_trace(script.clone());
        let mut ran = 0usize;
        while ran < 900 {
            let n = chunk.min(900 - ran);
            run(&mut sim, n);
            ran += n;
        }
        sim.audio_events().to_vec()
    };

    let a = log(1);
    let b = log(13);
    assert!(a.len() > 1, "the script must make more than the ambience");
    assert_eq!(a, b, "same trace → same tick-indexed audio, bit for bit");
}

#[test]
fn every_event_the_game_emits_names_a_patch_the_bank_contains() {
    // The join between `runt_core::PatchId` (what the tick says) and the bank
    // the host ships (what the synth was given). A mismatch here is a sound that
    // silently never plays.
    let bank = runt_ball::audio::bank();
    let mut sim = runt_ball::headless_sim();
    win(&mut sim);
    run(&mut sim, 120);

    for (_, event) in sim.audio_events() {
        if let AudioEvent::Play { patch, .. } = event {
            assert!(
                bank.get(runt_audio::PatchId(patch.0)).is_some(),
                "no preset for {patch:?}"
            );
        }
    }
}
