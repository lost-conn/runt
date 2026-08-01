//! Hand-rolled kinematic physics (DESIGN §9).
//!
//! No GPU anywhere: the ball integrator, the overlap pass and the contact solve
//! are `FixedSim` systems, and `Sim` is the engine minus the renderer, so all of
//! this runs on a CI box with no adapter.
//!
//! The load-bearing tests, in the order the doctrine states them:
//!
//! - [`terrain_tessellation_cannot_change_the_trajectory`] — §9's headline
//!   claim, end to end: two quality tiers, two different meshes, one
//!   bit-identical trajectory.
//! - [`ragged_and_uniform_hosts_produce_identical_physics`] — a replay is an
//!   input trace, so two hosts with different frame cadences must agree to the
//!   bit (§4).
//! - [`visual_spin_cannot_change_the_trajectory`] — §9 says spin is cosmetic;
//!   this is what "cosmetic" has to mean.
//! - [`the_demo_scene_ticks_exactly_as_it_did_before_physics_existed`] — the
//!   physics systems are a no-op on a world with no ball in it.

use bevy_ecs::prelude::*;
use glam::{Quat, Vec2, Vec3};

use runt_core::ecs::{advance_tick_count, GlobalTransform, TerrainSurface};
use runt_core::input::{Input, InputEvent, Key};
use runt_core::physics::{
    self, integrate_balls, resolve_overlaps, update_overlap_messages, AabbCollider, Ball,
    BallController, Grounded, OverlapEvent, RollSpin, SphereCollider, Trigger, Velocity,
};
use runt_core::scene::{self, save_scene};
use runt_core::{MeshRef, Sim, SimConfig, TickCount, Transform};

/// The scene file the RON-level tests drive.
const PHYSICS_SCENE_RON: &str = include_str!("physics_scene.ron");

// ---------------------------------------------------------------------------
// Scene helpers
// ---------------------------------------------------------------------------

/// `{:?}` rather than `{}`: `Display` prints `0` for `0.0`, and a scene field
/// typed `f32` wants a float literal.
fn f(v: f32) -> String {
    format!("{v:?}")
}

fn v3(v: Vec3) -> String {
    format!("({}, {}, {})", f(v.x), f(v.y), f(v.z))
}

/// A terrain generator entry. `amplitude: 0.0` is exactly flat — the field
/// normalizes by the octave weights, so a zero amplitude is a zero field and a
/// zero gradient, with no noise left in the answer.
fn ground_entry(amplitude: f32) -> String {
    format!(
        r#"(name: "ground", spec: Terrain((
            seed: 5, size: (80.0, 80.0), amplitude: {}, octaves: 4,
            frequency: 0.05, lacunarity: 2.0, gain: 0.5, base_segments: 32,
        )))"#,
        f(amplitude)
    )
}

/// A scene: one terrain patch at the origin, a camera looking down −Z, and
/// whatever entities the test appends.
fn scene_ron(amplitude: f32, entities: &str) -> String {
    format!(
        r#"(
            generators: [
                {ground},
                (name: "marble", spec: UvSphere(radius: 0.5, rings: 12, sectors: 16)),
                (name: "block", spec: Cube(size: 1.0)),
            ],
            entities: [
                (name: Some("ground"), generator: "ground"),
                {entities}
            ],
            camera: (eye: (0.0, 6.0, 10.0), target: (0.0, 0.0, 0.0)),
        )"#,
        ground = ground_entry(amplitude),
        entities = entities,
    )
}

/// A ball placement with the tuned defaults and the given start state.
fn ball_entry(position: Vec3, velocity: Vec3, extra: &str) -> String {
    format!(
        r#"(name: Some("ball"), generator: "marble",
            transform: (translation: {}),
            ball: Some((radius: 0.5)),
            velocity: Some({}),
            {})"#,
        v3(position),
        v3(velocity),
        extra,
    )
}

/// A ball with friction and damping switched off, so that a test about
/// *collision* is not quietly also a test about how far a ball coasts.
fn coasting_ball(position: Vec3, velocity: Vec3) -> String {
    format!(
        r#"(name: Some("ball"), generator: "marble",
            transform: (translation: {}),
            ball: Some((radius: 0.5, rolling_friction: 0.0, air_damping: 0.0)),
            velocity: Some({}))"#,
        v3(position),
        v3(velocity),
    )
}

fn sim_with(ron: &str) -> Sim {
    Sim::from_config(SimConfig::default().with_scene(ron))
}

fn ball_of(sim: &Sim) -> Entity {
    sim.scene_entity("ball").expect("the scene names a ball")
}

fn position(sim: &Sim, entity: Entity) -> Vec3 {
    sim.world()
        .get::<Transform>(entity)
        .expect("Transform")
        .translation
}

fn velocity(sim: &Sim, entity: Entity) -> Vec3 {
    sim.world().get::<Velocity>(entity).expect("Velocity").0
}

fn rotation(sim: &Sim, entity: Entity) -> Quat {
    sim.world().get::<Transform>(entity).expect("Transform").rotation
}

fn run(sim: &mut Sim, ticks: u32) {
    for _ in 0..ticks {
        sim.tick();
    }
}

fn vec3_bits(v: Vec3) -> [u32; 3] {
    [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()]
}

fn quat_bits(q: Quat) -> [u32; 4] {
    [q.x.to_bits(), q.y.to_bits(), q.z.to_bits(), q.w.to_bits()]
}

/// Position + velocity, bit for bit, every tick — the trajectory as a value that
/// two runs can be compared on without a tolerance to argue about.
fn trajectory(sim: &mut Sim, ticks: u32) -> Vec<([u32; 3], [u32; 3])> {
    let ball = ball_of(sim);
    (0..ticks)
        .map(|_| {
            sim.tick();
            (vec3_bits(position(sim, ball)), vec3_bits(velocity(sim, ball)))
        })
        .collect()
}

/// The terrain patch's surface and origin.
fn terrain_of(sim: &mut Sim) -> (TerrainSurface, Vec3) {
    let mut q = sim.world_mut().query::<(&TerrainSurface, &Transform)>();
    let found: Vec<(TerrainSurface, Vec3)> = q
        .iter(sim.world())
        .map(|(s, t)| (*s, t.translation))
        .collect();
    assert_eq!(found.len(), 1, "these scenes have exactly one terrain");
    found[0]
}

// ---------------------------------------------------------------------------
// The no-op property
// ---------------------------------------------------------------------------

/// FNV-1a over every `Transform` in the world, for 240 ticks of the demo scene.
fn demo_tick_fingerprint() -> u64 {
    let mut sim = Sim::new();
    sim.update(0.0);
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for i in 1..=240u64 {
        sim.update(i as f64 / 60.0);
        let mut rows: Vec<[u32; 10]> = sim
            .world_mut()
            .query::<&Transform>()
            .iter(sim.world())
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
        for row in rows {
            for word in row {
                h ^= word as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    h
}

#[test]
fn the_demo_scene_ticks_exactly_as_it_did_before_physics_existed() {
    // `assets/demo.ron` has no ball and no collider, so every system this step
    // added must fall straight through. The constant below was measured on the
    // commit *before* `physics.rs` existed; if it moves, the physics chain has
    // acquired a side effect on worlds that asked for no physics.
    assert_eq!(
        demo_tick_fingerprint(),
        0xdff6_c185_a3fe_c4b3,
        "the demo scene's 240-tick transform stream changed"
    );
}

// ---------------------------------------------------------------------------
// Resting, slopes, bouncing
// ---------------------------------------------------------------------------

#[test]
fn a_ball_at_rest_on_flat_ground_does_not_jitter() {
    // Spawned exactly on the surface: y = h + radius, with h = 0.
    let mut sim = sim_with(&scene_ron(
        0.0,
        &ball_entry(Vec3::new(0.0, 0.5, 0.0), Vec3::ZERO, ""),
    ));
    let ball = ball_of(&sim);

    sim.tick();
    let settled = position(&sim, ball).y;
    for tick in 0..100 {
        sim.tick();
        let p = position(&sim, ball);
        assert!(
            (p.y - settled).abs() < 1e-5,
            "tick {tick}: y drifted from {settled} to {}",
            p.y
        );
        assert!(
            velocity(&sim, ball).length() < 1e-5,
            "tick {tick}: a resting ball still has velocity {:?}",
            velocity(&sim, ball)
        );
    }
    // Not merely small: the rest snap makes it exact, which is what stops a
    // hundred ticks of "nearly zero" from adding up to a creep.
    assert_eq!(velocity(&sim, ball), Vec3::ZERO);
}

#[test]
fn a_ball_on_a_slope_accelerates_down_the_gradient() {
    // Slope response is not a special case in the integrator: gravity is applied
    // in world space and the contact solve keeps whatever part of it lies along
    // the surface. So the check is that the resulting horizontal velocity points
    // exactly along −∇h.
    let mut probe = sim_with(&scene_ron(3.0, ""));
    let (surface, origin) = terrain_of(&mut probe);

    // Find a genuinely sloped spot rather than trusting the noise to provide one.
    let mut steep: Option<(Vec3, Vec2)> = None;
    'search: for i in -12..=12 {
        for j in -12..=12 {
            let (x, z) = (i as f32 * 1.5, j as f32 * 1.5);
            let (h, grad) = surface.sample_world(origin, x, z);
            if grad.length() > 0.2 {
                steep = Some((Vec3::new(x, h + 0.5, z), grad));
                break 'search;
            }
        }
    }
    let (start, grad) = steep.expect("a 3.0-amplitude field has a slope somewhere");

    let mut sim = sim_with(&scene_ron(3.0, &ball_entry(start, Vec3::ZERO, "")));
    let ball = ball_of(&sim);
    sim.tick();

    let v = velocity(&sim, ball);
    let downhill = Vec3::new(-grad.x, 0.0, -grad.y).normalize();
    let horizontal = Vec3::new(v.x, 0.0, v.z);
    assert!(
        horizontal.length() > 1e-4,
        "the ball did not start moving on a slope of |∇h| = {}",
        grad.length()
    );
    assert!(
        horizontal.normalize().dot(downhill) > 0.999,
        "horizontal velocity {horizontal:?} is not along −∇h {downhill:?}"
    );

    // And it keeps gaining speed rather than reaching a one-tick blip.
    let first = horizontal.length();
    run(&mut sim, 20);
    let later = velocity(&sim, ball);
    assert!(
        Vec3::new(later.x, 0.0, later.z).length() > first * 2.0,
        "downhill speed should build: {first} → {:?}",
        later
    );
}

#[test]
fn a_dropped_ball_bounces_with_shrinking_apexes_then_settles() {
    let mut sim = sim_with(&scene_ron(
        0.0,
        &ball_entry(Vec3::new(0.0, 6.0, 0.0), Vec3::ZERO, ""),
    ));
    let ball = ball_of(&sim);

    // Sample the height stream and pull the local maxima out of it. The first
    // "apex" is the drop itself, so it is skipped.
    let mut heights = Vec::new();
    for _ in 0..900 {
        sim.tick();
        heights.push(position(&sim, ball).y);
    }

    let mut apexes = Vec::new();
    for w in heights.windows(3) {
        if w[1] > w[0] && w[1] >= w[2] && w[1] > 0.51 {
            apexes.push(w[1]);
        }
    }
    assert!(
        apexes.len() >= 2,
        "restitution {} should give at least two bounces, got {apexes:?}",
        physics::BALL_RESTITUTION
    );
    for pair in apexes.windows(2) {
        assert!(
            pair[1] < pair[0],
            "bounce apexes must shrink, got {apexes:?}"
        );
    }

    // And it ends up asleep on the surface, not buzzing against it.
    assert!((position(&sim, ball).y - 0.5).abs() < 1e-5, "did not settle");
    assert_eq!(velocity(&sim, ball), Vec3::ZERO);
}

#[test]
fn a_ball_with_nothing_under_it_free_falls() {
    // Off the edge of the 80×80 patch: `contains_world` finds no ground, so the
    // integrator leaves it in free fall. A kill plane is game logic (step 6).
    let mut sim = sim_with(&scene_ron(
        0.0,
        &ball_entry(Vec3::new(60.0, 2.0, 0.0), Vec3::ZERO, ""),
    ));
    let ball = ball_of(&sim);
    run(&mut sim, 60);

    let v = velocity(&sim, ball);
    assert!(v.y < -15.0, "one second of free fall, got vy {}", v.y);
    assert!(position(&sim, ball).y < -5.0, "it should be well below the patch");
    assert!(
        !sim.world().get::<Grounded>(ball).expect("Grounded").grounded,
        "nothing to be grounded on"
    );
}

// ---------------------------------------------------------------------------
// Friction and damping
// ---------------------------------------------------------------------------

/// Ticks until the ball's horizontal speed snaps to zero, at a given tick rate.
fn time_to_rest(hz: f64) -> f64 {
    let mut sim = Sim::from_config(
        SimConfig::default()
            .with_tick_rate(hz)
            .with_scene(scene_ron(
                0.0,
                &ball_entry(Vec3::new(0.0, 0.5, 0.0), Vec3::new(5.0, 0.0, 0.0), ""),
            )),
    );
    let ball = ball_of(&sim);
    for tick in 1..=(hz as u32 * 30) {
        sim.tick();
        if velocity(&sim, ball) == Vec3::ZERO {
            return tick as f64 / hz;
        }
    }
    panic!("a rolling ball never came to rest at {hz} Hz");
}

#[test]
fn rolling_friction_brings_a_ball_to_rest_at_a_tick_rate_independent_time() {
    // The whole reason friction is `exp(-rate · dt)` and not a per-tick factor:
    // `exp` composes over wall time, so halving the tick rate doubles the step
    // and changes nothing about *when* the ball stops. A `powf(k, dt·60)` would
    // agree here too; a bare `v *= 0.98` would not, and would silently make the
    // game feel different on a slower sim.
    let sixty = time_to_rest(60.0);
    let thirty = time_to_rest(30.0);
    assert!(
        (sixty - thirty).abs() < 0.05,
        "time to rest must not depend on the tick rate: {sixty} s at 60 Hz vs {thirty} s at 30 Hz"
    );
    // Sanity: it is a real coast, not an instant stop.
    assert!(sixty > 2.0 && sixty < 10.0, "implausible coast time {sixty} s");
}

#[test]
fn speed_is_clamped_at_max_speed() {
    // A frictionless ball under constant input would run away; `max_speed` is
    // what bounds how far one tick can move it, and therefore what keeps the
    // discrete contact test from being tunnelled through.
    let entity = r#"(name: Some("ball"), generator: "marble",
            transform: (translation: (0.0, 0.5, 0.0)),
            ball: Some((radius: 0.5, rolling_friction: 0.0, air_damping: 0.0, max_speed: 5.0)),
            ball_controller: Some((accel: 40.0)))"#;
    let mut sim = sim_with(&scene_ron(0.0, entity));
    let ball = ball_of(&sim);

    for _ in 0..400 {
        sim.push_input(InputEvent::KeyDown(Key::W));
        sim.tick();
        let speed = velocity(&sim, ball).length();
        assert!(speed <= 5.0 + 1e-4, "speed {speed} exceeded max_speed");
    }
    assert!(
        velocity(&sim, ball).length() > 4.9,
        "it should be pinned at the clamp"
    );
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

#[test]
fn input_accelerates_the_ball_relative_to_the_camera() {
    // The scene camera sits at (0, 6, 10) looking at the origin, so its forward
    // axis flattens onto −Z and its right axis onto +X.
    let start = Vec3::new(0.0, 0.5, 0.0);
    for (key, expected) in [
        (Key::W, Vec3::NEG_Z),
        (Key::S, Vec3::Z),
        (Key::D, Vec3::X),
        (Key::A, Vec3::NEG_X),
        (Key::Up, Vec3::NEG_Z),
        (Key::Right, Vec3::X),
    ] {
        let mut sim = sim_with(&scene_ron(
            0.0,
            &format!(
                r#"(name: Some("ball"), generator: "marble",
                    transform: (translation: {}),
                    ball: Some((radius: 0.5)),
                    ball_controller: Some((accel: 22.0)))"#,
                v3(start)
            ),
        ));
        let ball = ball_of(&sim);
        for _ in 0..15 {
            sim.push_input(InputEvent::KeyDown(key));
            sim.tick();
        }
        let v = velocity(&sim, ball);
        let horizontal = Vec3::new(v.x, 0.0, v.z);
        assert!(
            horizontal.length() > 0.5,
            "{key:?} produced no motion ({horizontal:?})"
        );
        assert!(
            horizontal.normalize().dot(expected) > 0.999,
            "{key:?} should push along {expected:?}, got {horizontal:?}"
        );
    }
}

#[test]
fn a_diagonal_is_not_faster_than_a_straight_line() {
    let entity = r#"(name: Some("ball"), generator: "marble",
            transform: (translation: (0.0, 0.5, 0.0)),
            ball: Some((radius: 0.5)),
            ball_controller: Some((accel: 22.0)))"#;
    let speed_after = |keys: &[Key]| {
        let mut sim = sim_with(&scene_ron(0.0, entity));
        let ball = ball_of(&sim);
        for _ in 0..30 {
            for k in keys {
                sim.push_input(InputEvent::KeyDown(*k));
            }
            sim.tick();
        }
        velocity(&sim, ball).length()
    };
    let straight = speed_after(&[Key::W]);
    let diagonal = speed_after(&[Key::W, Key::D]);
    assert!(
        (straight - diagonal).abs() < 1e-4,
        "diagonal {diagonal} should match straight {straight}"
    );
}

#[test]
fn input_does_nothing_without_a_camera() {
    // Documented behaviour: the control basis is camera-relative, so a world
    // with no camera has no basis and the input term is exactly zero. The ball
    // still falls — physics does not depend on there being a view.
    let mut sim = Sim::without_scene();
    let ball = sim
        .world_mut()
        .spawn((
            Transform::from_translation(Vec3::new(0.0, 10.0, 0.0)),
            GlobalTransform::default(),
            Ball::default(),
            Velocity::default(),
            Grounded::default(),
            BallController::default(),
        ))
        .id();

    for _ in 0..30 {
        sim.push_input(InputEvent::KeyDown(Key::W));
        sim.push_input(InputEvent::KeyDown(Key::D));
        sim.tick();
    }
    let v = sim.world().get::<Velocity>(ball).expect("Velocity").0;
    assert_eq!((v.x, v.z), (0.0, 0.0), "no camera, no horizontal input");
    assert!(v.y < -4.0, "but gravity still applies, got {v:?}");
}

// ---------------------------------------------------------------------------
// Discrete overlaps
// ---------------------------------------------------------------------------

/// A collider placement.
fn collider_entry(name: &str, position: Vec3, shape: &str) -> String {
    format!(
        r#"(name: Some("{name}"), generator: "block",
            transform: (translation: {}), {shape})"#,
        v3(position)
    )
}

#[test]
fn a_trigger_reports_the_overlap_without_touching_the_ball() {
    let ball = coasting_ball(Vec3::new(-4.0, 0.5, 0.0), Vec3::new(6.0, 0.0, 0.0));
    let near = collider_entry(
        "gate",
        Vec3::new(0.0, 0.5, 0.0),
        "sphere_collider: Some(0.6), trigger: true",
    );
    // The control differs only in where the trigger sits: same entity count,
    // same spawn order, same everything the sim could notice.
    let far = collider_entry(
        "gate",
        Vec3::new(0.0, 60.0, 0.0),
        "sphere_collider: Some(0.6), trigger: true",
    );

    let mut sim = sim_with(&scene_ron(0.0, &format!("{ball}, {near}")));
    let ball_entity = ball_of(&sim);
    let gate = sim.scene_entity("gate").expect("gate");
    let gate_position = position(&sim, gate);

    let mut event_ticks = Vec::new();
    let mut overlap_ticks = Vec::new();
    for tick in 0..120 {
        sim.tick();
        let p = position(&sim, ball_entity);
        // 0.5 (ball) + 0.6 (trigger) is the contact distance.
        if (p - gate_position).length() < 1.1 {
            overlap_ticks.push(tick);
        }
        let events: Vec<OverlapEvent> = sim.overlaps().copied().collect();
        if !events.is_empty() {
            assert_eq!(events.len(), 1, "one ball, one trigger, one event");
            assert!(events[0].trigger, "the gate is a trigger");
            assert_eq!(events[0].ball, ball_entity);
            assert_eq!(events[0].other, gate);
            assert!(events[0].depth > 0.0, "depth must be a real penetration");
            event_ticks.push(tick);
        }
    }
    assert!(!overlap_ticks.is_empty(), "the ball never reached the trigger");
    assert_eq!(
        event_ticks, overlap_ticks,
        "an event on exactly the overlapping ticks, no more and no fewer"
    );

    // The trigger stayed where it was put…
    assert_eq!(position(&sim, gate), gate_position);
    // …and the ball flew through it as though it were not there.
    let mut control = sim_with(&scene_ron(0.0, &format!("{ball}, {far}")));
    let mut deflected = sim_with(&scene_ron(0.0, &format!("{ball}, {near}")));
    assert_eq!(
        trajectory(&mut control, 120),
        trajectory(&mut deflected, 120),
        "a trigger must not deflect the ball by a single bit"
    );
}

#[test]
fn a_solid_aabb_pushes_the_ball_out_and_it_slides_along_the_face() {
    let ball = coasting_ball(Vec3::new(-4.0, 0.5, 0.0), Vec3::new(6.0, 0.0, 2.0));
    let wall = collider_entry(
        "wall",
        Vec3::new(0.0, 0.5, 0.0),
        "aabb_collider: Some((0.5, 0.5, 4.0))",
    );
    let mut sim = sim_with(&scene_ron(0.0, &format!("{ball}, {wall}")));
    let ball_entity = ball_of(&sim);
    let wall_entity = sim.scene_entity("wall").expect("wall");
    let wall_center = position(&sim, wall_entity);
    let half = Vec3::new(0.5, 0.5, 4.0);

    let mut hit = false;
    let mut post_hit_velocity = Vec3::ZERO;
    for tick in 0..90 {
        sim.tick();

        // The invariant, asserted every single tick: after resolution the ball
        // is never inside the box.
        let p = position(&sim, ball_entity);
        let closest = (p - wall_center).clamp(-half, half) + wall_center;
        let gap = (p - closest).length();
        assert!(
            gap >= 0.5 - 1e-4,
            "tick {tick}: ball penetrated the box, gap {gap} < radius 0.5"
        );

        for event in sim.overlaps() {
            assert!(!event.trigger, "the wall is solid");
            assert_eq!(event.other, wall_entity);
            // Approaching from −X, so the push-out is along −X.
            assert!(
                event.normal.dot(Vec3::NEG_X) > 0.99,
                "unexpected contact normal {:?}",
                event.normal
            );
            if !hit {
                hit = true;
                post_hit_velocity = velocity(&sim, ball_entity);
            }
        }
    }

    assert!(hit, "the ball never reached the wall");
    assert!(
        post_hit_velocity.x.abs() < 1e-5,
        "velocity into the face must be killed, got x = {}",
        post_hit_velocity.x
    );
    assert!(
        (post_hit_velocity.z - 2.0).abs() < 1e-4,
        "…and the tangential component must survive untouched, got z = {}",
        post_hit_velocity.z
    );
    // It slid: the ball ends up well along +Z from where it started.
    assert!(position(&sim, ball_entity).z > 1.0);
}

#[test]
fn a_solid_sphere_emits_an_event_too_and_stops_the_ball() {
    // §9 gives the game bounce sounds as well as pickups, so a *solid* overlap
    // is reported exactly like a trigger — the flag is the only difference.
    let ball = coasting_ball(Vec3::new(-4.0, 0.5, 0.0), Vec3::new(6.0, 0.0, 0.0));
    let rock = collider_entry("rock", Vec3::new(0.0, 0.5, 0.0), "sphere_collider: Some(0.75)");
    let mut sim = sim_with(&scene_ron(0.0, &format!("{ball}, {rock}")));
    let ball_entity = ball_of(&sim);
    let rock_entity = sim.scene_entity("rock").expect("rock");
    let rock_position = position(&sim, rock_entity);

    let mut solid_events = 0;
    for _ in 0..90 {
        sim.tick();
        for event in sim.overlaps() {
            assert_eq!((event.ball, event.other), (ball_entity, rock_entity));
            assert!(!event.trigger);
            solid_events += 1;
        }
        let gap = (position(&sim, ball_entity) - rock_position).length();
        assert!(gap >= 1.25 - 1e-4, "penetration: gap {gap}");
    }
    assert!(solid_events > 0, "a solid overlap must still be reported");
    assert!(
        velocity(&sim, ball_entity).x.abs() < 1e-4,
        "the approach velocity should be gone"
    );
    // The obstacle never moved: §9 rules impulse exchange out entirely.
    assert_eq!(position(&sim, rock_entity), rock_position);
}

// ---------------------------------------------------------------------------
// Message semantics
// ---------------------------------------------------------------------------

/// `(tick index, overlapped entity)` pairs, recorded from inside the schedule.
#[derive(Resource, Default)]
struct Heard(Vec<(u64, Entity)>);

#[derive(Resource, Default)]
struct HeardEarly(Vec<(u64, Entity)>);

fn listen_late(mut heard: ResMut<Heard>, tick: Res<TickCount>, mut reader: MessageReader<OverlapEvent>) {
    for event in reader.read() {
        heard.0.push((tick.0, event.other));
    }
}

fn listen_early(
    mut heard: ResMut<HeardEarly>,
    tick: Res<TickCount>,
    mut reader: MessageReader<OverlapEvent>,
) {
    for event in reader.read() {
        heard.0.push((tick.0, event.other));
    }
}

#[test]
fn where_a_reader_sits_in_the_chain_decides_which_tick_it_hears_on() {
    // The lifetime `OverlapEvent`'s docs claim, checked against the real 0.19
    // double buffer: a reader chained *after* the resolver hears on the tick the
    // overlap happened; one chained *before* it hears on the next tick. Neither
    // misses an event, and neither hears one twice — `MessageReader` carries a
    // per-system cursor.
    let ball = coasting_ball(Vec3::new(-4.0, 0.5, 0.0), Vec3::new(6.0, 0.0, 0.0));
    let gate = collider_entry(
        "gate",
        Vec3::new(0.0, 0.5, 0.0),
        "sphere_collider: Some(0.6), trigger: true",
    );
    let mut sim = sim_with(&scene_ron(0.0, &format!("{ball}, {gate}")));
    sim.world_mut().init_resource::<Heard>();
    sim.world_mut().init_resource::<HeardEarly>();
    sim.fixed_sim_mut().add_systems((
        listen_late.after(resolve_overlaps).before(advance_tick_count),
        listen_early
            .after(update_overlap_messages)
            .before(integrate_balls),
    ));

    run(&mut sim, 120);

    let late = &sim.world().resource::<Heard>().0;
    let early = &sim.world().resource::<HeardEarly>().0;
    assert!(!late.is_empty(), "the ball never reached the trigger");
    assert_eq!(
        late.len(),
        early.len(),
        "every event reaches every reader exactly once"
    );
    for (l, e) in late.iter().zip(early) {
        assert_eq!(l.1, e.1, "the same events, in the same order");
        assert_eq!(
            e.0,
            l.0 + 1,
            "a reader before the resolver is exactly one tick behind it"
        );
    }
}

#[test]
fn overlaps_are_visible_to_the_host_for_the_tick_they_happened_on() {
    // `Sim::overlaps` reads the current write buffer, so it reports the last
    // tick's batch and nothing older — no cursor, no accumulation.
    let ball = coasting_ball(Vec3::new(-4.0, 0.5, 0.0), Vec3::new(6.0, 0.0, 0.0));
    let gate = collider_entry(
        "gate",
        Vec3::new(0.0, 0.5, 0.0),
        "sphere_collider: Some(0.6), trigger: true",
    );
    let mut sim = sim_with(&scene_ron(0.0, &format!("{ball}, {gate}")));

    let mut batches = Vec::new();
    for _ in 0..120 {
        sim.tick();
        batches.push(sim.overlaps().len());
    }
    assert!(batches.contains(&1), "some tick must overlap");
    assert!(
        batches.iter().all(|&n| n <= 1),
        "one ball and one trigger can never make two events in a tick: {batches:?}"
    );
    // Well past the trigger, the buffer is empty again rather than holding on.
    assert_eq!(*batches.last().expect("ticked"), 0);
}

// ---------------------------------------------------------------------------
// The §9 properties
// ---------------------------------------------------------------------------

#[test]
fn terrain_tessellation_cannot_change_the_trajectory() {
    // DESIGN §9's headline claim, end to end. Collision samples the analytic
    // field; the mesh is only a view of it. So a coarse tier and a full tier
    // draw different triangles and simulate the same ball — not "close", the
    // same bits.
    let entity = ball_entry(Vec3::new(-3.0, 6.0, 2.5), Vec3::new(2.0, 0.0, -1.5), "");
    let ron = scene_ron(3.0, &entity);

    let mut coarse = Sim::from_config(SimConfig::default().with_scene(&ron).with_quality(0.3));
    let mut full = Sim::from_config(SimConfig::default().with_scene(&ron).with_quality(1.0));

    let mesh_of = |sim: &Sim| {
        sim.world()
            .get::<MeshRef>(sim.scene_entity("ground").expect("ground"))
            .expect("MeshRef")
            .0
    };
    assert_ne!(
        mesh_of(&coarse),
        mesh_of(&full),
        "the tiers must actually produce different geometry, or this proves nothing"
    );

    assert_eq!(
        trajectory(&mut coarse, 400),
        trajectory(&mut full, 400),
        "visual LOD changed the physics"
    );
}

#[test]
fn visual_spin_cannot_change_the_trajectory() {
    // §9: "visual spin is derived from velocity (cosmetic, never simulated
    // state)". The test of that is a full trajectory stream with the roll on and
    // off — if any solver ever reads `Transform.rotation`, this diverges.
    let rolling = r#"(name: Some("ball"), generator: "marble",
            transform: (translation: (-3.0, 6.0, 2.5)),
            ball: Some((radius: 0.5, roll_spin: true)),
            velocity: Some((2.0, 0.0, -1.5)))"#;
    let sliding = rolling.replace("roll_spin: true", "roll_spin: false");

    let mut with_spin = sim_with(&scene_ron(3.0, rolling));
    let mut without_spin = sim_with(&scene_ron(3.0, &sliding));

    let spinning_ball = ball_of(&with_spin);
    let sliding_ball = ball_of(&without_spin);
    assert!(with_spin.world().get::<RollSpin>(spinning_ball).is_some());
    assert!(without_spin.world().get::<RollSpin>(sliding_ball).is_none());

    assert_eq!(
        trajectory(&mut with_spin, 400),
        trajectory(&mut without_spin, 400),
        "the cosmetic roll fed back into the simulation"
    );

    // And the roll really did happen — otherwise the comparison above is
    // vacuous.
    assert_ne!(
        quat_bits(rotation(&with_spin, spinning_ball)),
        quat_bits(rotation(&without_spin, sliding_ball)),
    );
    assert_eq!(quat_bits(rotation(&without_spin, sliding_ball)), quat_bits(Quat::IDENTITY));
}

// ---------------------------------------------------------------------------
// Replay determinism
// ---------------------------------------------------------------------------

/// A scripted input trace, keyed by **tick index** — which is what a replay
/// actually is (DESIGN §4: "input is … consumed at tick boundaries — replays are
/// just recorded input traces + seeds").
///
/// Keying on tick rather than on wall time is the whole point of the exercise.
/// The *arrival* time of a key press is a property of the host's polling, not of
/// the run: two hosts with different frame cadences will genuinely hand the same
/// press to different ticks, and the sim will genuinely diverge — correctly so,
/// because they were given different input. What has to be cadence-proof is the
/// tick sequence, and that is what this trace pins down.
#[derive(Resource, Clone, Default)]
struct Script(Vec<(u64, InputEvent)>);

fn input_trace() -> Vec<(u64, InputEvent)> {
    vec![
        (3, InputEvent::KeyDown(Key::W)),
        (13, InputEvent::KeyDown(Key::D)),
        (31, InputEvent::KeyUp(Key::W)),
        (44, InputEvent::KeyDown(Key::S)),
        (59, InputEvent::KeyUp(Key::D)),
        (73, InputEvent::KeyDown(Key::A)),
        (93, InputEvent::KeyUp(Key::S)),
        (110, InputEvent::KeyUp(Key::A)),
    ]
}

/// Feed the current tick's slice of the trace into the [`Input`] resource,
/// exactly as the tick loop would have from the host's buffer.
fn apply_script(script: Res<Script>, tick: Res<TickCount>, mut input: ResMut<Input>) {
    let now = tick.0;
    let events: Vec<InputEvent> = script
        .0
        .iter()
        .filter(|(t, _)| *t == now)
        .map(|(_, e)| *e)
        .collect();
    input.begin_tick(events);
}

/// Drive a physics scene over a list of wall-clock times to call `update` at,
/// with the trace replayed from inside the tick.
fn drive(schedule: &[f64]) -> Sim {
    let entity = r#"(name: Some("ball"), generator: "marble",
            transform: (translation: (0.0, 3.0, 0.0)),
            ball: Some((radius: 0.5)),
            ball_controller: Some((accel: 22.0)))"#;
    let mut sim = sim_with(&scene_ron(3.0, entity));
    sim.world_mut().insert_resource(Script(input_trace()));
    sim.fixed_sim_mut().add_systems(
        apply_script
            .after(update_overlap_messages)
            .before(integrate_balls),
    );

    sim.update(0.0); // Establish the origin without ticking.
    for &t in schedule {
        sim.update(t);
    }
    sim
}

#[test]
fn ragged_and_uniform_hosts_produce_identical_physics() {
    // One host runs a clean 60 fps; the other stutters through 5/30/7 ms chunks.
    // Same trace, same end instant, therefore the same ticks with the same
    // inputs — and, because nothing in the integrator reads a wall clock, the
    // same floats.
    let uniform: Vec<f64> = (1..=120).map(|i| i as f64 / 60.0).collect();
    let ragged: Vec<f64> = {
        let steps = [0.005, 0.030, 0.007];
        let mut times = Vec::new();
        let (mut t, mut i) = (0.0f64, 0usize);
        while t < 2.0 {
            t += steps[i % steps.len()];
            i += 1;
            times.push(t.min(2.0));
        }
        if *times.last().expect("non-empty") != 2.0 {
            times.push(2.0);
        }
        times
    };
    assert!(ragged.len() != uniform.len(), "the two patterns must differ");

    let a = drive(&uniform);
    let b = drive(&ragged);
    assert_eq!(a.tick_count(), b.tick_count());
    assert!(a.tick_count() >= 119, "2 s at 60 Hz, got {}", a.tick_count());

    let (ea, eb) = (ball_of(&a), ball_of(&b));
    let ta = a.world().get::<Transform>(ea).expect("Transform");
    let tb = b.world().get::<Transform>(eb).expect("Transform");
    assert_eq!(
        (vec3_bits(ta.translation), quat_bits(ta.rotation)),
        (vec3_bits(tb.translation), quat_bits(tb.rotation)),
        "ball transforms must be bit-identical, got {ta:?} vs {tb:?}"
    );
    assert_eq!(
        vec3_bits(velocity(&a, ea)),
        vec3_bits(velocity(&b, eb)),
        "and so must the velocities"
    );
    // The ball actually went somewhere, so the comparison means something.
    let travelled = Vec3::new(ta.translation.x, 0.0, ta.translation.z).length();
    assert!(travelled > 0.5, "the trace should move it, got {travelled}");
}

#[test]
fn the_same_run_twice_is_reproducible() {
    let schedule: Vec<f64> = (1..=180).map(|i| i as f64 / 60.0).collect();
    let a = drive(&schedule);
    let b = drive(&schedule);
    let (ea, eb) = (ball_of(&a), ball_of(&b));
    assert_eq!(
        vec3_bits(position(&a, ea)),
        vec3_bits(position(&b, eb))
    );
    assert_eq!(vec3_bits(velocity(&a, ea)), vec3_bits(velocity(&b, eb)));
}

// ---------------------------------------------------------------------------
// The scene file
// ---------------------------------------------------------------------------

#[test]
fn the_physics_scene_spawns_what_it_describes() {
    let mut sim = sim_with(PHYSICS_SCENE_RON);

    let player = sim.scene_entity("player").expect("player");
    let world = sim.world();
    // The loader supplies every component the systems need, not just the ones
    // the file names.
    assert_eq!(
        world.get::<Ball>(player).copied(),
        Some(Ball::with_radius(0.5))
    );
    assert_eq!(world.get::<Velocity>(player).copied(), Some(Velocity(Vec3::ZERO)));
    assert_eq!(world.get::<Grounded>(player).copied(), Some(Grounded::default()));
    assert!(world.get::<RollSpin>(player).is_some(), "roll_spin defaults on");
    assert!(world.get::<runt_core::Interpolated>(player).is_some(), "a ball moves");
    assert_eq!(
        world.get::<SphereCollider>(player).map(|c| c.radius),
        Some(0.5),
        "the ball gets a collider matching its physics radius"
    );
    assert_eq!(
        world.get::<BallController>(player).map(|c| c.accel),
        Some(22.0)
    );

    let pickup = sim.scene_entity("pickup").expect("pickup");
    assert_eq!(
        sim.world().get::<SphereCollider>(pickup).map(|c| c.radius),
        Some(0.6)
    );
    assert!(sim.world().get::<Trigger>(pickup).is_some());
    assert!(sim.world().get::<Ball>(pickup).is_none(), "a pickup is not a ball");

    let wall = sim.scene_entity("wall").expect("wall");
    assert_eq!(
        sim.world().get::<AabbCollider>(wall).map(|c| c.half_extents),
        Some(Vec3::new(0.5, 0.5, 2.0))
    );
    assert!(sim.world().get::<Trigger>(wall).is_none(), "a wall is solid");

    // And it runs: the ball drops onto the flat field and settles at h + r.
    run(&mut sim, 300);
    assert!(
        (position(&sim, player).y - 0.5).abs() < 1e-5,
        "the ball should be resting on the field, at {:?}",
        position(&sim, player)
    );
}

#[test]
fn a_physics_scene_survives_a_save_and_load_round_trip() {
    // The same fixed-point property `tests/scene.rs` holds the demo to, extended
    // over the physics fields: load → save → load → save must be idempotent, and
    // the two worlds must agree.
    let a = sim_with(PHYSICS_SCENE_RON);
    let saved = save_scene(a.world()).expect("save");
    let b = sim_with(&saved);
    let saved_again = save_scene(b.world()).expect("save again");
    assert_eq!(saved, saved_again, "saving is idempotent");

    let da = scene::scene_desc(a.world()).expect("desc");
    let db = scene::scene_desc(b.world()).expect("desc");
    assert_eq!(da, db);

    // An authored file and a saved one describe the same scene.
    let authored = scene::parse_scene(PHYSICS_SCENE_RON).expect("authored parses");
    assert_eq!(authored.entities, da.entities);
    assert_eq!(authored.generators, da.generators);
}

#[test]
fn a_moved_balls_velocity_is_written_back_on_save() {
    // Velocity is authored state the sim rewrites, so `save_scene` refreshes it
    // on the same terms as the transform: only once it differs from the file.
    let mut sim = sim_with(PHYSICS_SCENE_RON);
    let before = scene::scene_desc(sim.world()).expect("desc");
    let player_index = before
        .entities
        .iter()
        .position(|e| e.name.as_deref() == Some("player"))
        .expect("player");
    assert_eq!(before.entities[player_index].velocity, None, "authored as None");

    run(&mut sim, 5); // Falling: velocity is now decidedly non-zero.
    let after = scene::scene_desc(sim.world()).expect("desc");
    let v = after.entities[player_index]
        .velocity
        .expect("a falling ball has a velocity to record");
    assert!(v.y < -1.0, "expected downward motion, got {v:?}");
}
