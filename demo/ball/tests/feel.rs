//! The tuning, as assertions (DESIGN §12 step 6's "feel pass").
//!
//! "Does it feel good" is not testable, but the three things that make it feel
//! *bad* are, and all three are numbers in `level1.ron`:
//!
//! - **Sluggish.** A 48 m map with a ball that tops out at walking pace is a
//!   chore. So: cross half the map in well under ten seconds.
//! - **Camera in the dirt.** A follow camera slung too low clips through the
//!   hill behind you on every crest. So: sample the terrain under the camera
//!   every tick of a long run and demand clearance.
//! - **Out of reach.** There is no jump in v0, so a collectible the resting ball
//!   cannot touch is a collectible you cannot get. So: check the geometry, not
//!   the intention.
//!
//! These are the reason the level does not use the engine's default ball
//! parameters; the derivations are in `level1.ron`'s comments and README.md.

use bevy_ecs::prelude::*;
use glam::{Vec2, Vec3};

use runt_ball::game::GameState;
use runt_core::ecs::TerrainSurface;
use runt_core::gen::GeneratorSpec;
use runt_core::scene::parse_scene;
use runt_core::{Camera, InputEvent, InputTrace, Key, Sim, Transform};

fn hold(key: Key) -> InputTrace {
    InputTrace::from_pairs([(0, InputEvent::KeyDown(key))])
}

fn position(sim: &Sim, entity: Entity) -> Vec3 {
    sim.world().get::<Transform>(entity).expect("Transform").translation
}

fn plan_distance(a: Vec3, b: Vec3) -> f32 {
    Vec2::new(a.x - b.x, a.z - b.z).length()
}

/// The one terrain patch, as the physics sees it.
fn terrain(sim: &mut Sim) -> (TerrainSurface, Vec3) {
    let mut q = sim.world_mut().query::<(&TerrainSurface, &Transform)>();
    let found: Vec<(TerrainSurface, Vec3)> = q
        .iter(sim.world())
        .map(|(s, t)| (*s, t.translation))
        .collect();
    assert_eq!(found.len(), 1, "level 1 has exactly one terrain patch");
    found[0]
}

// ---------------------------------------------------------------------------

#[test]
fn the_ball_covers_ground_fast_enough_to_be_worth_playing() {
    // Terminal roll is `accel / rolling_friction` = 16 / 2.2 ≈ 7.3 m/s, so 20 m
    // of headroom-free travel should take on the order of three seconds plus the
    // spin-up. Ten seconds is the failure line — comfortably slack, because the
    // point is to catch "the ball is a boulder", not to pin a stopwatch.
    let mut sim = runt_ball::headless_sim();
    sim.play_input_trace(hold(Key::W));
    let player = sim.world().resource::<GameState>().player;
    let spawn = position(&sim, player);

    let mut reached_20 = None;
    let mut peak = 0.0f32;
    for tick in 1..=600u64 {
        sim.tick();
        // Stop at the first reset: past that, "distance from spawn" restarts.
        if sim.world().resource::<GameState>().resets > 0 {
            break;
        }
        let d = plan_distance(position(&sim, player), spawn);
        peak = peak.max(d);
        if d >= 20.0 && reached_20.is_none() {
            reached_20 = Some(tick);
        }
    }

    let ticks = reached_20.unwrap_or_else(|| {
        panic!("holding W for 10 s only got {peak:.1} m from the spawn point")
    });
    let seconds = ticks as f32 / 60.0;
    assert!(
        seconds < 10.0,
        "20 m took {seconds:.1} s — the ball is sluggish for a 48 m map"
    );
    // And not *absurdly* fast either: under a second and a half would mean the
    // whole level is four seconds wide and the camera can never keep up.
    assert!(
        seconds > 1.5,
        "20 m in {seconds:.1} s — that is a rocket, not a marble"
    );
    println!("feel: 20 m from a standing start in {seconds:.2} s");
}

#[test]
fn the_ball_can_climb_the_steepest_slope_on_the_map() {
    // A jumpless game where a hill can trap you is a game with dead ends. The
    // input acceleration has to beat gravity's pull along the worst slope the
    // patch has, which for `y = h` is `g·|∇h|/(1 + |∇h|²)`.
    let scene = parse_scene(runt_ball::LEVEL1_RON).expect("parse");
    let GeneratorSpec::Terrain(params) = scene
        .generators
        .iter()
        .find(|g| g.name == "ground")
        .expect("ground")
        .spec
        .clone()
    else {
        panic!("ground is not Terrain");
    };
    let field = params.field();
    let half = params.size * 0.5;

    let mut worst = 0.0f32;
    for j in 0..=256 {
        for i in 0..=256 {
            let x = -half.x + params.size.x * i as f32 / 256.0;
            let z = -half.y + params.size.y * j as f32 / 256.0;
            worst = worst.max(field.gradient(x, z).length());
        }
    }

    let player = scene
        .entities
        .iter()
        .find(|e| e.name.as_deref() == Some("player"))
        .expect("player");
    let ball = player.ball.expect("the player is a ball");
    let accel = player.ball_controller.expect("controller").accel;

    let downhill_pull = ball.gravity * worst / (1.0 + worst * worst);
    assert!(
        accel > downhill_pull * 1.8,
        "input accel {accel} against a worst-slope pull of {downhill_pull} \
         (|∇h| = {worst}): the ball would crawl up the steepest hill"
    );
    println!(
        "feel: steepest slope |∇h| = {worst:.3} ({:.1}°), pull {downhill_pull:.2} m/s² \
         vs accel {accel}",
        worst.atan().to_degrees()
    );
}

#[test]
fn the_camera_never_gets_close_to_the_ground() {
    // The follow camera is sim state, so this is checkable without a renderer:
    // sample `h(x, z)` under the camera every tick of a run that crosses the map
    // and crests real hills. `offset.y = 8` against a field that spans about
    // 3.5 m is deliberate overkill — a camera that clips terrain once per minute
    // is a camera nobody trusts.
    let mut sim = runt_ball::headless_sim();
    sim.play_input_trace(InputTrace::from_pairs([
        (0, InputEvent::KeyDown(Key::W)),
        (150, InputEvent::KeyDown(Key::D)),
        (150, InputEvent::KeyUp(Key::W)),
        (330, InputEvent::KeyDown(Key::S)),
        (330, InputEvent::KeyUp(Key::D)),
        (520, InputEvent::KeyDown(Key::A)),
        (520, InputEvent::KeyUp(Key::S)),
        (700, InputEvent::KeyUp(Key::A)),
        (700, InputEvent::KeyDown(Key::W)),
    ]));
    let (surface, origin) = terrain(&mut sim);
    let camera = sim.camera_entity().expect("the level has a camera");
    assert!(
        sim.world().get::<Camera>(camera).is_some(),
        "that entity is the camera"
    );

    let mut lowest = f32::MAX;
    let mut travelled = 0.0f32;
    let player = sim.world().resource::<GameState>().player;
    let mut previous = position(&sim, player);
    for _ in 0..900 {
        sim.tick();
        let here = position(&sim, player);
        travelled += plan_distance(here, previous);
        previous = here;

        let eye = position(&sim, camera);
        if !surface.contains_world(origin, eye.x, eye.z) {
            continue; // Off the patch: nothing under it to clip through.
        }
        lowest = lowest.min(eye.y - surface.height_world(origin, eye.x, eye.z));
    }

    assert!(
        travelled > 40.0,
        "the ball only covered {travelled:.1} m, so the camera was never tested"
    );
    assert!(
        lowest > 3.0,
        "the camera came within {lowest:.2} m of the ground"
    );
    println!("feel: camera clearance ≥ {lowest:.2} m over {travelled:.0} m of driving");
}

#[test]
fn every_pickup_is_takeable_by_a_ball_resting_on_the_ground_under_it() {
    // No jump in v0. So the question is not "is the slope gentle" (level.rs
    // asks that) but the blunter one: with the ball sitting on the surface
    // directly beneath a ring, do the two shapes actually touch?
    let mut sim = runt_ball::headless_sim();
    let (surface, origin) = terrain(&mut sim);
    let scene = parse_scene(runt_ball::LEVEL1_RON).expect("parse");

    let player = scene
        .entities
        .iter()
        .find(|e| e.name.as_deref() == Some("player"))
        .expect("player");
    let radius = player.ball.expect("ball").radius;

    for pickup in scene.entities.iter().filter(|e| e.generator == "pickup") {
        let p = pickup.transform.translation;
        let trigger = pickup.sphere_collider.expect("pickups are triggers");
        let reach = radius + trigger;

        // The ball's centre when it is resting directly under the ring, at both
        // extremes of the bob.
        let resting = surface.height_world(origin, p.x, p.z) + radius;
        for bob in [-runt_ball::game::BOB_AMPLITUDE, 0.0, runt_ball::game::BOB_AMPLITUDE] {
            let gap = (p.y + bob - resting).abs();
            assert!(
                gap < reach,
                "{:?} is {gap} above the resting ball with a reach of {reach}",
                pickup.name
            );
            // And with margin to spare, so the take does not need the ball to be
            // within a centimetre in plan view: this is the horizontal slack.
            let slack = (reach * reach - gap * gap).sqrt();
            assert!(
                slack > 0.8,
                "{:?} allows only {slack} m of horizontal error",
                pickup.name
            );
        }
    }
}

#[test]
fn a_ball_left_alone_settles_instead_of_creeping() {
    // The spawn is level to within 5°, and `REST_SPEED` snaps a slow tangential
    // velocity to exactly zero — so an untouched game must be *still*, not
    // slowly drifting off a hill while the player reads the controls.
    let mut sim = runt_ball::headless_sim();
    let player = sim.world().resource::<GameState>().player;
    for _ in 0..120 {
        sim.tick();
    }
    let settled = position(&sim, player);
    for _ in 0..600 {
        sim.tick();
    }
    let drift = plan_distance(position(&sim, player), settled);
    assert!(
        drift < 0.01,
        "the ball drifted {drift} m in ten idle seconds"
    );
    assert_eq!(sim.world().resource::<GameState>().score, 0);
}
