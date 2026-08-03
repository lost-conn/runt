//! The rules (DESIGN §12 step 6).
//!
//! These drive the *shipped* game — `runt_ball::headless_sim()` is the same
//! scene and the same [`setup`](runt_ball::game::setup) the window host uses,
//! minus the window. So a green run here is a statement about what a player gets
//! rather than about a rig that resembles it.
//!
//! Collecting twelve rings by steering is not what is under test, so these
//! teleport the ball onto each pickup instead. Everything downstream of that —
//! the overlap pass, the trigger flag, the message reader, the despawn, the
//! score, the win — is the real thing.

use bevy_ecs::prelude::*;
use glam::Vec3;

use runt_ball::game::{GameState, Phase, Pickup};
use runt_core::{InputEvent, Key, Sim, StatusLine, Transform, Velocity};

fn state(sim: &Sim) -> GameState {
    sim.world().resource::<GameState>().clone()
}

fn position(sim: &Sim, entity: Entity) -> Vec3 {
    sim.world().get::<Transform>(entity).expect("Transform").translation
}

/// Every live pickup, in `Entity` order so the walk is deterministic.
fn pickups(sim: &mut Sim) -> Vec<Entity> {
    let mut q = sim.world_mut().query_filtered::<Entity, With<Pickup>>();
    let mut found: Vec<Entity> = q.iter(sim.world()).collect();
    found.sort_unstable();
    found
}

/// Tap R and run the tick it lands on. Goes through the host's own input path
/// (`push_input` → tick boundary), not through the resource, so what is under
/// test is what a player's keyboard reaches.
fn press_restart(sim: &mut Sim) {
    sim.push_input(InputEvent::KeyDown(Key::R));
    sim.tick();
    sim.push_input(InputEvent::KeyUp(Key::R));
}

/// Put the ball on `target` and run one tick — the tick the overlap happens on,
/// and therefore (the chain sits after `resolve_overlaps`) the tick it scores.
fn take(sim: &mut Sim, player: Entity, target: Vec3) {
    sim.world_mut()
        .get_mut::<Transform>(player)
        .expect("Transform")
        .translation = target;
    sim.world_mut().get_mut::<Velocity>(player).expect("Velocity").0 = Vec3::ZERO;
    sim.tick();
}

// ---------------------------------------------------------------------------

#[test]
fn collecting_every_pickup_wins_and_stops_the_clock() {
    let mut sim = runt_ball::headless_sim();
    let player = state(&sim).player;
    let total = state(&sim).total;
    let all = pickups(&mut sim);
    assert_eq!(all.len() as u32, total);

    let mut previous_score = 0;
    let mut win_tick = None;
    for (i, pickup) in all.iter().enumerate() {
        let target = position(&sim, *pickup);
        take(&mut sim, player, target);

        let now = state(&sim);
        assert_eq!(
            now.score,
            i as u32 + 1,
            "pickup {i} did not score on the tick it was touched"
        );
        assert!(now.score >= previous_score, "score must never go backwards");
        previous_score = now.score;

        // Out of play, so it cannot be collected twice. It is *hidden*, not
        // despawned, so that `restart_run` has something to bring back — see
        // `Pickup::collected` for the trade.
        let taken = sim.world().get::<Pickup>(*pickup).expect("still an entity");
        assert!(taken.collected, "a collected pickup must be flagged");
        assert_eq!(
            position(&sim, *pickup).y,
            runt_ball::game::HIDDEN_Y,
            "a collected pickup must be parked out of sight"
        );

        if now.won() && win_tick.is_none() {
            win_tick = Some(now.elapsed_ticks);
        }
    }

    let won = state(&sim);
    assert_eq!(won.phase, Phase::Won);
    assert_eq!(won.score, total);
    assert_eq!(
        win_tick,
        Some(won.elapsed_ticks),
        "the win must land on the tick the last pickup was taken"
    );

    // The clock is frozen: the final time is the time it took, not the time the
    // window stayed open afterwards.
    let frozen = won.elapsed_ticks;
    for _ in 0..120 {
        sim.tick();
    }
    let after = state(&sim);
    assert_eq!(after.elapsed_ticks, frozen, "the clock kept running after the win");
    assert_eq!(after.score, total, "and the score did not drift");
    assert!(after.status().contains("WON"));
}

#[test]
fn the_clock_counts_ticks_of_play() {
    let mut sim = runt_ball::headless_sim();
    assert_eq!(state(&sim).elapsed_ticks, 0);
    for i in 1..=90u64 {
        sim.tick();
        assert_eq!(state(&sim).elapsed_ticks, i);
    }
    // …and reports them in seconds off the tick length, never off a clock.
    let s = state(&sim);
    assert!((s.elapsed_secs() - 1.5).abs() < 1e-6, "{}", s.elapsed_secs());
}

#[test]
fn falling_off_the_world_puts_the_ball_back_and_costs_nothing() {
    let mut sim = runt_ball::headless_sim();
    let GameState {
        player,
        spawn_point,
        kill_y,
        ..
    } = state(&sim);

    // Score one first, so the test can watch the score *survive* the fall.
    let pickup = pickups(&mut sim)[0];
    let target = position(&sim, pickup);
    take(&mut sim, player, target);
    assert_eq!(state(&sim).score, 1);

    // Off the edge of the patch and well under the plane, falling fast.
    sim.world_mut()
        .get_mut::<Transform>(player)
        .expect("Transform")
        .translation = Vec3::new(40.0, kill_y - 1.0, 40.0);
    sim.world_mut().get_mut::<Velocity>(player).expect("Velocity").0 =
        Vec3::new(3.0, -22.0, -1.0);
    sim.tick();

    let after = state(&sim);
    assert_eq!(after.resets, 1);
    assert_eq!(
        position(&sim, player),
        spawn_point,
        "a reset puts the ball exactly back on the spawn point"
    );
    assert_eq!(
        sim.world().get::<Velocity>(player).expect("Velocity").0,
        Vec3::ZERO,
        "…and drops the fall's velocity, or it would be fired straight back off"
    );
    assert_eq!(after.score, 1, "falling is not a penalty in v0");
    assert!(after.status().contains("1 fall"));

    // And it stays put: one tick later it is resting, not still being reset.
    sim.tick();
    assert_eq!(state(&sim).resets, 1);
    assert!(position(&sim, player).y > kill_y);
}

#[test]
fn the_status_line_reports_the_score_and_the_host_sees_it() {
    let mut sim = runt_ball::headless_sim();
    sim.tick();
    assert_eq!(
        sim.status_line(),
        sim.world().resource::<StatusLine>().0,
        "the accessor is the resource"
    );
    assert!(sim.status_line().contains("0/12"), "{}", sim.status_line());

    let player = state(&sim).player;
    let pickup = pickups(&mut sim)[0];
    let target = position(&sim, pickup);
    take(&mut sim, player, target);

    let line = sim.status_line().to_string();
    assert!(line.contains("1/12"), "expected a score in {line:?}");
    assert!(line.starts_with("runt ball"));
    // The seam is one-way: nothing in the engine reads it back.
    assert_eq!(line, state(&sim).status());
}

#[test]
fn the_bob_floats_the_pickups_without_moving_them() {
    // Cosmetic in the same sense §9 means for roll spin: it changes what you
    // see, never where anything is in plan view — and it is a function of the
    // tick, so it cannot drift over a long session.
    let mut sim = runt_ball::headless_sim();
    let all = pickups(&mut sim);
    let bases: Vec<Vec3> = all.iter().map(|e| position(&sim, *e)).collect();

    let mut extremes: Vec<(f32, f32)> = vec![(f32::MAX, f32::MIN); all.len()];
    for _ in 0..600 {
        sim.tick();
        for (i, entity) in all.iter().enumerate() {
            let p = position(&sim, *entity);
            assert_eq!(
                (p.x, p.z),
                (bases[i].x, bases[i].z),
                "the bob moved a pickup horizontally"
            );
            let offset = p.y - sim.world().get::<Pickup>(*entity).expect("Pickup").base_y;
            extremes[i].0 = extremes[i].0.min(offset);
            extremes[i].1 = extremes[i].1.max(offset);
        }
    }

    let amp = runt_ball::game::BOB_AMPLITUDE;
    for (i, (low, high)) in extremes.iter().enumerate() {
        assert!(
            *low >= -amp - 1e-6 && *high <= amp + 1e-6,
            "pickup {i} left its envelope: [{low}, {high}] vs ±{amp}"
        );
        assert!(
            *low < -amp * 0.9 && *high > amp * 0.9,
            "pickup {i} barely moved: [{low}, {high}]"
        );
    }

    // Not in lockstep — the golden-angle phases are doing their job.
    let heights: Vec<f32> = all.iter().map(|e| position(&sim, *e).y).collect();
    let offsets: Vec<f32> = heights
        .iter()
        .zip(&all)
        .map(|(y, e)| y - sim.world().get::<Pickup>(*e).expect("Pickup").base_y)
        .collect();
    let spread = offsets.iter().cloned().fold(f32::MIN, f32::max)
        - offsets.iter().cloned().fold(f32::MAX, f32::min);
    assert!(spread > amp, "all twelve pickups bob in unison (spread {spread})");
}

#[test]
fn r_puts_the_run_back_to_tick_zero() {
    // The restart has to undo *everything* a run accumulates, and the only two
    // places a run accumulates anything are `GameState` and the pickups. So the
    // test scores, falls, wins — and then demands the world it gets back is
    // indistinguishable from a fresh one.
    let mut sim = runt_ball::headless_sim();
    let GameState {
        player,
        spawn_point,
        kill_y,
        total,
        ..
    } = state(&sim);
    let all = pickups(&mut sim);
    let bases: Vec<Vec3> = all.iter().map(|e| position(&sim, *e)).collect();

    // A run with something to undo: three rings and a fall.
    for pickup in all.iter().take(3) {
        let target = position(&sim, *pickup);
        take(&mut sim, player, target);
    }
    sim.world_mut()
        .get_mut::<Transform>(player)
        .expect("Transform")
        .translation = Vec3::new(40.0, kill_y - 1.0, 40.0);
    sim.tick();
    for _ in 0..30 {
        sim.tick();
    }
    let before = state(&sim);
    assert_eq!(before.score, 3);
    assert_eq!(before.resets, 1);
    assert!(before.elapsed_ticks > 30);

    press_restart(&mut sim);

    let after = state(&sim);
    assert_eq!(after.score, 0, "the score survived a restart");
    assert_eq!(after.resets, 0, "the falls survived a restart");
    assert_eq!(after.phase, Phase::Playing);
    // The clock is reset by `restart_run` and then ticked once by `game_clock`,
    // which is chained after it — one tick of play, exactly as at the start.
    assert_eq!(after.elapsed_ticks, 1);
    assert_eq!(after.total, total, "the ring count is level data, not run state");
    assert_eq!(after.spawn_point, spawn_point);

    assert_eq!(position(&sim, player), spawn_point, "the ball went home");
    assert_eq!(
        sim.world().get::<Velocity>(player).expect("Velocity").0,
        Vec3::ZERO,
        "a restart that kept the fall's velocity would fire the ball off the map"
    );

    // Every ring is back, at its own (x, z) and within a bob of its base.
    for (i, entity) in all.iter().enumerate() {
        let pickup = *sim.world().get::<Pickup>(*entity).expect("Pickup");
        assert!(!pickup.collected, "ring {i} stayed collected");
        let p = position(&sim, *entity);
        assert_eq!((p.x, p.z), (bases[i].x, bases[i].z));
        assert!(
            (p.y - pickup.base_y).abs() <= runt_ball::game::BOB_AMPLITUDE + 1e-6,
            "ring {i} came back at {p:?}, not near {}",
            pickup.base_y
        );
    }

    // And the restarted run is playable: the same twelve rings win it again.
    for pickup in &all {
        let target = position(&sim, *pickup);
        take(&mut sim, player, target);
    }
    assert_eq!(state(&sim).phase, Phase::Won);
    assert_eq!(state(&sim).score, total);

    // Even from the win screen.
    press_restart(&mut sim);
    assert_eq!(state(&sim).phase, Phase::Playing);
    assert_eq!(state(&sim).score, 0);
}

#[test]
fn holding_r_restarts_once_rather_than_every_tick() {
    // `just_pressed`, not `held`: an auto-repeating key must not pin the clock
    // at zero for as long as a finger rests on it.
    let mut sim = runt_ball::headless_sim();
    sim.push_input(InputEvent::KeyDown(Key::R));
    sim.tick();
    assert_eq!(state(&sim).elapsed_ticks, 1);
    for expected in 2..=5 {
        sim.tick(); // R is still held, and no fresh press arrives.
        assert_eq!(state(&sim).elapsed_ticks, expected, "the clock stalled");
    }
}

#[test]
fn a_solid_post_is_not_a_pickup() {
    // Overlap events are emitted for solids too (§9 gives the game impact
    // sounds), so `collect_pickups` has to be the thing that distinguishes
    // them — not the physics.
    let mut sim = runt_ball::headless_sim();
    let player = state(&sim).player;
    let post = sim.scene_entity("post_6").expect("post_6");
    let target = position(&sim, post);

    take(&mut sim, player, target + Vec3::new(0.0, 0.2, 0.0));
    let after = state(&sim);
    assert_eq!(after.score, 0, "a post scored");
    assert!(
        sim.world().get::<Transform>(post).is_some(),
        "a post was despawned"
    );
    // And it did push the ball out, so the overlap really happened.
    assert!(
        position(&sim, player).distance(target) > 0.4,
        "the post did not resolve the overlap, so this test proved nothing"
    );
}
