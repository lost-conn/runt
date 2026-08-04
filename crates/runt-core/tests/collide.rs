//! Collision v2 (DESIGN §9's 2026-08-04 amendment) — capsule character solver,
//! OBB colliders, layers, queries.
//!
//! Headless, no GPU: every one of these builds a bare `World` (or a `Sim` with
//! no renderer) and calls the solver directly, which is exactly how a game's
//! state machine will.
//!
//! The geometry is not invented. The angles, thicknesses and speeds come from
//! `3dimenshift-runt/docs/PORT_SPEC.md`'s PoC level inventory — 15°/16.7°/30°/40°
//! pitched ramps, a 0.5 m-thick phase wall at −98° yaw, a 3 m shaft — so a test
//! passing here says something about the port rather than about a shape someone
//! chose because it was easy.
//!
//! The load-bearing claims, in the order the amendment states them:
//!
//! - [`the_solver_is_a_pure_function_of_the_snapshot`] and
//!   [`a_scripted_capsule_run_is_identical_under_ragged_and_uniform_hosts`] —
//!   determinism (DESIGN §3, §4).
//! - [`a_terrain_raycast_is_independent_of_tessellation`] — §9's headline claim,
//!   extended to the new query: the field is what is sampled, never the mesh.
//! - [`a_default_layers_component_collides_exactly_as_no_component_does`] — the
//!   additive property, at the layer level.

use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};

use runt_core::collide::{
    self, move_and_slide, CharacterBody, CharacterShape, CollisionLayers, CollisionWorld,
    ContactKind, MoveResult, ObbCollider, ALL_LAYERS,
};
use runt_core::physics::{AabbCollider, SphereCollider, Trigger, Velocity};
use runt_core::{Sim, SimConfig, Transform};

const DT: f32 = 1.0 / 60.0;
const GRAVITY: f32 = 32.65; // PORT_SPEC's g_down.
/// PORT_SPEC's `max_speed / deceleration_time` — 8 m/s shed in 0.305 s, which is
/// 0.44 m/s of braking a tick. Comfortably more than the downhill velocity one
/// tick of gravity projects onto a 30° face, which is why the creep this module
/// now stops was never a friction problem.
const DECELERATION: f32 = 8.0 / 0.305;

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/// A world with colliders in it, plus the snapshot the solver reads.
struct Level {
    world: World,
}

impl Level {
    fn new() -> Level {
        Level {
            world: World::new(),
        }
    }

    fn aabb(&mut self, center: Vec3, half: Vec3) -> Entity {
        self.world
            .spawn((
                Transform::from_translation(center),
                AabbCollider { half_extents: half },
            ))
            .id()
    }

    fn obb(&mut self, center: Vec3, half: Vec3, rotation: Quat) -> Entity {
        self.world
            .spawn((
                Transform::from_translation(center),
                ObbCollider {
                    half_extents: half,
                    rotation,
                },
            ))
            .id()
    }

    fn sphere(&mut self, center: Vec3, radius: f32) -> Entity {
        self.world
            .spawn((Transform::from_translation(center), SphereCollider { radius }))
            .id()
    }

    fn layers(&mut self, entity: Entity, layers: CollisionLayers) -> Entity {
        self.world.entity_mut(entity).insert(layers);
        entity
    }

    fn trigger(&mut self, entity: Entity) -> Entity {
        self.world.entity_mut(entity).insert(Trigger);
        entity
    }

    fn snapshot(&mut self) -> CollisionWorld {
        CollisionWorld::from_world(&mut self.world)
    }
}

/// PORT_SPEC's floor: 50 × 1 × 50 centered at y = −0.5, so its top is y = 0.
fn with_floor(level: &mut Level) -> Entity {
    level.aabb(Vec3::new(0.0, -0.5, 0.0), Vec3::new(25.0, 0.5, 25.0))
}

/// A ramp pitched about world X by `degrees`, PORT_SPEC's form. Positive
/// degrees tilt the +Z edge down, so the surface descends towards +Z.
fn ramp(level: &mut Level, center: Vec3, half: Vec3, degrees: f32) -> Entity {
    level.obb(center, half, Quat::from_rotation_x(degrees.to_radians()))
}

/// PORT_SPEC's standing body: capsule r = 0.35, total height 2.0, feet 1.0 below
/// the centre.
fn standing() -> CharacterBody {
    CharacterBody::default()
        .with_shape(CharacterShape::Capsule {
            radius: 0.35,
            height: 2.0,
        })
        .with_max_floor_degrees(45.0)
        .with_snap_length(0.0)
}

/// The centre of a standing capsule whose feet rest exactly on `y`.
fn standing_on(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y + 1.0, z)
}

fn degrees(radians: f32) -> f32 {
    radians.to_degrees()
}

/// One tick of "gravity while airborne, then move" — the shape of the port's
/// loop, reduced to the two lines these tests care about.
fn step(
    geometry: &CollisionWorld,
    body: &mut CharacterBody,
    position: &mut Vec3,
    velocity: &mut Vec3,
) -> MoveResult {
    if !body.on_floor {
        velocity.y -= GRAVITY * DT;
    }
    let result = move_and_slide(geometry, body, *position, *velocity, DT);
    *position = result.position;
    *velocity = result.velocity;
    result
}

// ---------------------------------------------------------------------------
// Resting and classification
// ---------------------------------------------------------------------------

#[test]
fn a_capsule_falls_onto_a_box_floor_and_stays_there() {
    let mut level = Level::new();
    with_floor(&mut level);
    let geometry = level.snapshot();

    let mut body = standing();
    let mut p = Vec3::new(0.0, 4.0, 0.0);
    let mut v = Vec3::ZERO;

    for _ in 0..120 {
        step(&geometry, &mut body, &mut p, &mut v);
    }
    assert!(body.on_floor, "the capsule never landed");
    // Feet exactly on the floor: centre at radius + half-segment = 1.0 above it.
    assert!(
        (p.y - 1.0).abs() < 1e-3,
        "resting centre {} is not 1.0 above the floor",
        p.y
    );

    // And it stays put, to the bit, once settled.
    let settled = p;
    for _ in 0..120 {
        step(&geometry, &mut body, &mut p, &mut v);
    }
    assert_eq!(p, settled, "a settled capsule drifted");
    assert!(body.on_floor);
}

#[test]
fn sphere_mode_is_a_degenerate_capsule() {
    let mut level = Level::new();
    with_floor(&mut level);
    let geometry = level.snapshot();

    // PORT_SPEC's roll sphere: r = 0.35, so the same radius as the capsule with
    // a zero-length segment. It must rest exactly one radius up.
    let mut body = standing().with_shape(CharacterShape::Sphere { radius: 0.35 });
    let mut p = Vec3::new(0.0, 3.0, 0.0);
    let mut v = Vec3::ZERO;
    for _ in 0..120 {
        step(&geometry, &mut body, &mut p, &mut v);
    }
    assert!(body.on_floor);
    assert!(
        (p.y - 0.35).abs() < 1e-3,
        "a rolling sphere rests at {}, not at its radius",
        p.y
    );

    // A capsule of height 2r is the same shape, and must produce the same rest.
    let mut capsule = standing().with_shape(CharacterShape::Capsule {
        radius: 0.35,
        height: 0.7,
    });
    let mut q = Vec3::new(0.0, 3.0, 0.0);
    let mut qv = Vec3::ZERO;
    for _ in 0..120 {
        step(&geometry, &mut capsule, &mut q, &mut qv);
    }
    assert_eq!(p, q, "sphere and zero-segment capsule disagree");
}

#[test]
fn the_aabb_fast_path_agrees_with_an_identity_obb() {
    // The closed-form axis-aligned contact and the ternary search over a rotated
    // box are two different routines; a box with no rotation is where they have
    // to meet.
    let mut aabb_level = Level::new();
    aabb_level.aabb(Vec3::new(0.0, -0.5, 0.0), Vec3::new(4.0, 0.5, 4.0));
    let aabb_geometry = aabb_level.snapshot();

    let mut obb_level = Level::new();
    obb_level.obb(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(4.0, 0.5, 4.0),
        Quat::IDENTITY,
    );
    let obb_geometry = obb_level.snapshot();

    for start in [
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(1.7, 3.5, -2.3),
        Vec3::new(3.9, 1.4, 3.9),
    ] {
        let (mut a_body, mut b_body) = (standing(), standing());
        let (mut a_p, mut b_p) = (start, start);
        let (mut a_v, mut b_v) = (Vec3::new(1.0, 0.0, 0.5), Vec3::new(1.0, 0.0, 0.5));
        for _ in 0..60 {
            step(&aabb_geometry, &mut a_body, &mut a_p, &mut a_v);
            step(&obb_geometry, &mut b_body, &mut b_p, &mut b_v);
        }
        assert!(
            a_p.abs_diff_eq(b_p, 1e-4),
            "AABB {a_p} and identity OBB {b_p} disagree from {start}"
        );
        assert_eq!(a_body.on_floor, b_body.on_floor);
    }
}

// ---------------------------------------------------------------------------
// PORT_SPEC's pitched ramps
// ---------------------------------------------------------------------------

/// Stand a capsule on a ramp of `pitch` degrees and report what it settled as.
///
/// The press is one tick of gravity every tick rather than an accumulating fall:
/// the solver has no friction (that is the caller's, PORT_SPEC puts it in the
/// state machine), so an accumulating slide would walk the body off the end of
/// the slab before the classification could be read. A constant press is what a
/// standing player is.
fn land_on_ramp(pitch: f32, max_floor_degrees: f32) -> (CharacterBody, MoveResult) {
    let mut level = Level::new();
    // A wide, thick slab so the capsule lands on the *face*, never near an edge.
    ramp(
        &mut level,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(6.0, 1.0, 6.0),
        pitch,
    );
    let geometry = level.snapshot();

    let mut body = standing().with_max_floor_degrees(max_floor_degrees);
    let face = geometry
        .raycast(Vec3::new(0.0, 8.0, 0.0), Vec3::NEG_Y, 16.0, ALL_LAYERS)
        .expect("the ramp is under the probe");
    let mut p = face.point + Vec3::Y * 1.05;
    let mut last = move_and_slide(&geometry, &mut body, p, Vec3::ZERO, DT);
    for _ in 0..30 {
        last = move_and_slide(&geometry, &mut body, p, Vec3::new(0.0, -2.0, 0.0), DT);
        p = last.position;
    }
    (body, last)
}

#[test]
fn pitched_ramps_report_their_authored_angle_as_the_floor_angle() {
    // The PoC level's exact pitches. The contact normal comes out of the box's
    // own frame, so the angle it reports is the angle the ramp was authored at —
    // no tessellation, no averaging.
    for pitch in [15.0f32, 16.7, 30.0, 40.0] {
        let (body, result) = land_on_ramp(pitch, 45.0);
        assert!(
            body.on_floor,
            "a {pitch}° ramp is not floor at max_floor_angle 45°"
        );
        assert_eq!(result.contacts.len(), 1, "{pitch}°: expected one contact");
        assert_eq!(result.contacts[0].kind, ContactKind::Floor);
        assert!(
            (degrees(result.floor_angle) - pitch).abs() < 0.05,
            "a {pitch}° ramp reported a floor angle of {}",
            degrees(result.floor_angle)
        );
        assert!(!result.on_wall, "{pitch}°: reported a wall");
        assert!(!result.on_ceiling, "{pitch}°: reported a ceiling");
    }
}

#[test]
fn a_forty_degree_ramp_is_a_wall_when_the_max_floor_angle_is_thirty_five() {
    // PORT_SPEC's SteepSlope and RampCliff deliberately straddle the 35°
    // roll-force threshold. At a 35° max the same geometry must reclassify.
    let (body, result) = land_on_ramp(40.0, 35.0);
    assert!(!body.on_floor, "a 40° ramp counted as floor at a 35° max");
    assert!(result.on_wall, "a 40° ramp is not a wall at a 35° max");
    assert!(!result.on_ceiling);
    assert_eq!(result.contacts[0].kind, ContactKind::Wall);

    // And the 30° ramp on the other side of the threshold is still floor.
    let (body, _) = land_on_ramp(30.0, 35.0);
    assert!(body.on_floor, "a 30° ramp is not floor at a 35° max");
}

#[test]
fn every_contact_is_floor_at_a_hundred_and_eighty_degrees() {
    // PORT_SPEC's rolling body: floor_max_angle 180°, which in Godot means
    // "nothing is ever a wall". A vertical face has to come back as floor.
    let mut level = Level::new();
    with_floor(&mut level);
    // A vertical slab the body runs into.
    level.aabb(Vec3::new(2.0, 2.0, 0.0), Vec3::new(0.25, 6.0, 3.0));
    let geometry = level.snapshot();

    let mut body = standing()
        .with_shape(CharacterShape::Sphere { radius: 0.35 })
        .with_max_floor_degrees(180.0);
    let mut p = Vec3::new(0.0, 0.35, 0.0);
    let mut v = Vec3::new(8.0, 0.0, 0.0);
    let mut last = move_and_slide(&geometry, &mut body, p, v, DT);
    for _ in 0..60 {
        last = step(&geometry, &mut body, &mut p, &mut v);
    }

    assert!(
        last.contacts.iter().all(|c| c.kind == ContactKind::Floor),
        "a 180° max floor angle still produced {:?}",
        last.contacts.iter().map(|c| c.kind).collect::<Vec<_>>()
    );
    assert!(!last.on_wall, "180°: nothing may be a wall");
    assert!(!last.on_ceiling, "180°: nothing may be a ceiling");
    assert!(
        last.contacts.iter().any(|c| c.normal.x < -0.9),
        "the body never reached the vertical slab"
    );
}

#[test]
fn a_ceiling_is_a_ceiling_only_within_the_max_floor_angle_of_down() {
    let mut level = Level::new();
    with_floor(&mut level);
    // A lid 2.4 m up: a standing capsule (2 m tall, centre 1 m up) jumping into
    // it hits its underside.
    level.aabb(Vec3::new(0.0, 3.4, 0.0), Vec3::new(4.0, 1.0, 4.0));
    let geometry = level.snapshot();

    let mut body = standing();
    let mut p = standing_on(0.0, 0.0, 0.0);
    let mut v = Vec3::new(0.0, 10.0, 0.0);
    body.on_floor = true;
    let mut saw_ceiling = false;
    for _ in 0..60 {
        let r = step(&geometry, &mut body, &mut p, &mut v);
        if r.on_ceiling {
            saw_ceiling = true;
            assert!(r.ceiling_normal.y < -0.99, "{}", r.ceiling_normal);
            assert!(!r.on_floor, "the lid must not be floor");
        }
    }
    assert!(saw_ceiling, "the capsule never hit the lid");
}

// ---------------------------------------------------------------------------
// Stop on slope
// ---------------------------------------------------------------------------

/// A single wide slab pitched by `degrees` — one face of PORT_SPEC's slope
/// battery, big enough that nothing here ever reaches an edge.
fn slope_level(pitch: f32) -> CollisionWorld {
    let mut level = Level::new();
    ramp(
        &mut level,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(8.0, 1.0, 8.0),
        pitch,
    );
    level.snapshot()
}

/// One tick of the port's Idle state, reduced to the two lines that produce the
/// bug: gravity applied *every* tick, grounded or not (`idle_state.gd` does not
/// ask), and a deceleration that takes the horizontal velocity to exactly zero.
/// What reaches the solver is therefore gravity and nothing else — which is the
/// whole of Godot's `floor_stop_on_slope` condition.
fn idle_step(
    geometry: &CollisionWorld,
    body: &mut CharacterBody,
    position: &mut Vec3,
    velocity: &mut Vec3,
) -> MoveResult {
    velocity.y -= GRAVITY * DT;
    let horizontal = Vec3::new(velocity.x, 0.0, velocity.z);
    let brake = DECELERATION * DT;
    let braked = if horizontal.length() <= brake {
        Vec3::ZERO
    } else {
        horizontal - horizontal.normalize() * brake
    };
    velocity.x = braked.x;
    velocity.z = braked.z;
    let result = move_and_slide(geometry, body, *position, *velocity, DT);
    *position = result.position;
    *velocity = result.velocity;
    result
}

/// The same tick with the stick held: a constant `wish` on the horizontal axes,
/// which is what a state machine's acceleration converges to.
fn walk_step(
    geometry: &CollisionWorld,
    body: &mut CharacterBody,
    position: &mut Vec3,
    velocity: &mut Vec3,
    wish: Vec3,
) -> MoveResult {
    velocity.y -= GRAVITY * DT;
    velocity.x = wish.x;
    velocity.z = wish.z;
    let result = move_and_slide(geometry, body, *position, *velocity, DT);
    *position = result.position;
    *velocity = result.velocity;
    result
}

/// A body standing on `pitch`, already settled.
fn standing_on_slope(pitch: f32) -> (CollisionWorld, CharacterBody, Vec3, Vec3) {
    let geometry = slope_level(pitch);
    let mut body = standing().with_snap_length(0.5);
    body.on_floor = true;
    let mut p = on_ramp(&geometry, 0.0);
    let mut v = Vec3::ZERO;
    for _ in 0..10 {
        idle_step(&geometry, &mut body, &mut p, &mut v);
    }
    (geometry, body, p, v)
}

#[test]
fn a_body_under_gravity_alone_does_not_slide_down_a_walkable_slope() {
    // The bug: `move_and_slide` projects velocity onto the slope plane, so the
    // tick's gravity comes back as a downhill tangential velocity that the
    // caller's friction is left fighting. Before this flag existed the 15° face
    // walked the body 0.73 m downhill in five seconds and the 30° one 1.59 m,
    // with the stick untouched. Godot cancels the motion outright, and so does
    // this — the body does not move by one bit.
    for pitch in [15.0f32, 30.0] {
        let (geometry, mut body, mut p, mut v) = standing_on_slope(pitch);
        let settled = p;
        assert!(body.on_floor, "{pitch}°: the body is not standing on it");
        assert_eq!(v, Vec3::ZERO, "{pitch}°: a standing body kept a velocity");

        for tick in 0..300 {
            let result = idle_step(&geometry, &mut body, &mut p, &mut v);
            assert_eq!(p, settled, "{pitch}°: crept on tick {tick}");
            assert_eq!(v, Vec3::ZERO, "{pitch}°: gained a velocity on tick {tick}");
            assert!(result.on_floor, "{pitch}°: left the floor on tick {tick}");
            assert!(
                result.stopped_on_slope,
                "{pitch}°: tick {tick} did not stop"
            );
            assert!(
                (degrees(result.floor_angle) - pitch).abs() < 0.05,
                "{pitch}°: floor angle became {}",
                degrees(result.floor_angle)
            );
        }
    }
}

#[test]
fn the_same_slope_is_walked_normally_when_there_is_input() {
    // The stop is not a freeze. A body with somewhere to go fails the direction
    // test — its velocity is nowhere near straight down — and moves exactly as
    // it did before the flag existed.
    for pitch in [15.0f32, 30.0] {
        let (geometry, mut body, mut p, mut v) = standing_on_slope(pitch);
        let start = p;

        // Straight down the fall line: the ramp descends towards +Z.
        let wish = Vec3::new(0.0, 0.0, 4.0);
        for tick in 0..60 {
            let result = walk_step(&geometry, &mut body, &mut p, &mut v, wish);
            assert!(result.on_floor, "{pitch}°: airborne on tick {tick}");
            assert!(
                !result.stopped_on_slope,
                "{pitch}°: the stop fired on tick {tick} with the stick held"
            );
        }
        // A second at 4 m/s covers 4 m of ground, plus whatever the descent adds
        // back — the body follows the surface rather than the horizontal.
        let travelled = p.z - start.z;
        assert!(
            (3.5..5.0).contains(&travelled),
            "{pitch}°: covered {travelled} m of +Z in a second at 4 m/s"
        );
        // And it is still on the face, not floating over it or ploughing in: the
        // drop is the ground it covered times the slope.
        let expected_drop = travelled * pitch.to_radians().tan();
        assert!(
            (start.y - p.y - expected_drop).abs() < 0.1,
            "{pitch}°: descended {} m over {travelled} m, expected {expected_drop}",
            start.y - p.y
        );

        // Across the fall line moves too, and the stop still keeps out of it.
        let sideways = Vec3::new(4.0, 0.0, 0.0);
        let before = p;
        for _ in 0..60 {
            walk_step(&geometry, &mut body, &mut p, &mut v, sideways);
        }
        assert!(
            (p.x - before.x - 4.0).abs() < 0.1,
            "{pitch}°: covered {} m of +X in a second at 4 m/s",
            p.x - before.x
        );
    }
}

#[test]
fn a_face_too_steep_to_stand_on_still_slides() {
    // The stop is gated on the contact being *floor*. PORT_SPEC's 40° SteepSlope
    // read against a 35° max is a wall, walls never stop the body, and a body on
    // one keeps sliding — which is what feeds the port's Roll.
    let geometry = slope_level(40.0);
    let mut body = standing()
        .with_max_floor_degrees(35.0)
        .with_snap_length(0.5);
    body.on_floor = true;
    let mut p = on_ramp(&geometry, -2.0);
    let mut v = Vec3::ZERO;
    let start = p;

    for tick in 0..300 {
        let result = idle_step(&geometry, &mut body, &mut p, &mut v);
        assert!(
            !result.stopped_on_slope,
            "tick {tick}: a wall stopped the body"
        );
        assert!(
            !result.on_floor,
            "tick {tick}: a 40° face read as floor at 35°"
        );
    }
    assert!(
        p.z - start.z > 0.25,
        "the body only slid {} m down a face it cannot stand on",
        p.z - start.z
    );
}

#[test]
fn floor_stop_on_slope_false_is_the_behaviour_the_flag_replaced() {
    // Godot's own escape hatch, and the regression guard: turning it off gives
    // back the projection, downhill velocity and all.
    let geometry = slope_level(15.0);
    let mut body = standing()
        .with_snap_length(0.5)
        .with_floor_stop_on_slope(false);
    body.on_floor = true;
    let mut p = on_ramp(&geometry, 0.0);
    let mut v = Vec3::ZERO;
    let start = p;

    for _ in 0..300 {
        let result = idle_step(&geometry, &mut body, &mut p, &mut v);
        assert!(!result.stopped_on_slope, "the flag is off");
        assert!(result.on_floor);
    }
    assert!(
        p.z - start.z > 0.5,
        "with the flag off the body should still creep; it moved {} m",
        p.z - start.z
    );
    assert!(
        v.z > 0.05,
        "with the flag off the projection should still hand back a downhill \
         velocity; it was {}",
        v.z
    );
}

#[test]
fn walking_down_a_slope_and_letting_go_comes_to_rest_without_creeping() {
    // The case the port actually plays: run down the 15° face, release the
    // stick, and stop. The stop has to engage the moment the decel has taken the
    // horizontal velocity away — not one tick later and not never.
    let geometry = slope_level(15.0);
    let mut body = standing().with_snap_length(0.5);
    body.on_floor = true;
    let mut p = on_ramp(&geometry, -4.0);
    let mut v = Vec3::ZERO;
    let start = p;

    for _ in 0..60 {
        walk_step(
            &geometry,
            &mut body,
            &mut p,
            &mut v,
            Vec3::new(0.0, 0.0, 6.0),
        );
    }
    assert!(p.z - start.z > 5.0, "the run never happened");

    // Let go. Within a handful of ticks the decel has won and the body is stopped.
    let mut stop_tick = None;
    for tick in 0..60 {
        let result = idle_step(&geometry, &mut body, &mut p, &mut v);
        if result.stopped_on_slope && stop_tick.is_none() {
            stop_tick = Some(tick);
        }
    }
    let stop_tick = stop_tick.expect("the body never came to rest on the slope");
    assert!(
        stop_tick < 30,
        "it took {stop_tick} ticks to stop; the decel needs 15"
    );

    // And it stays stopped, to the bit, for five seconds.
    let rest = p;
    for tick in 0..300 {
        idle_step(&geometry, &mut body, &mut p, &mut v);
        assert_eq!(p, rest, "crept on tick {tick} after coming to rest");
    }
    assert!(body.on_floor);
}

#[test]
fn coming_to_rest_on_a_slope_is_ragged_host_independent() {
    // DESIGN §4, applied to the new branch: the stop is a function of the tick's
    // velocity and the snapshot, never of how the host chopped wall time up.
    let geometry = slope_level(30.0);
    let script = |chunks: &[usize]| -> Vec<(Vec3, Vec3, bool)> {
        let mut body = standing().with_snap_length(0.5);
        body.on_floor = true;
        // A short drop onto the face, a run down it, then a release: an airborne
        // landing, a walk and a stop in one trace.
        let mut p = on_ramp(&geometry, -4.0) + Vec3::Y * 0.5;
        let mut v = Vec3::ZERO;
        let mut trace = Vec::new();
        let mut tick = 0u32;
        for chunk in chunks {
            for _ in 0..*chunk {
                if (30..120).contains(&tick) {
                    walk_step(
                        &geometry,
                        &mut body,
                        &mut p,
                        &mut v,
                        Vec3::new(0.0, 0.0, 5.0),
                    );
                } else {
                    idle_step(&geometry, &mut body, &mut p, &mut v);
                }
                trace.push((p, v, body.on_floor));
                tick += 1;
            }
        }
        trace
    };

    let uniform = script(&[240]);
    let ragged = script(&[7, 1, 13, 2, 40, 1, 1, 60, 5, 110]);
    assert_eq!(uniform.len(), 240);
    assert_eq!(ragged, uniform, "a ragged host produced a different run");
    assert!(
        uniform.last().unwrap().1 == Vec3::ZERO,
        "the trace never reached the resting case it is meant to cover"
    );
}

// ---------------------------------------------------------------------------
// Floor snap
// ---------------------------------------------------------------------------

/// PORT_SPEC's `Ramp`: 16.7°, the shallowest thing you can run down fast.
fn descending_ramp() -> CollisionWorld {
    let mut level = Level::new();
    // Pitched about X so the surface descends towards +Z; long enough that 8 m/s
    // for a second stays on it.
    ramp(
        &mut level,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(4.0, 1.0, 12.0),
        16.7,
    );
    level.snapshot()
}

/// Put the capsule on the ramp's surface directly above `z`.
fn on_ramp(geometry: &CollisionWorld, z: f32) -> Vec3 {
    let hit = geometry
        .raycast(Vec3::new(0.0, 10.0, z), Vec3::NEG_Y, 20.0, ALL_LAYERS)
        .expect("the ramp is under the probe");
    Vec3::new(0.0, hit.point.y + 1.0, z)
}

#[test]
fn walking_down_a_ramp_stays_grounded_every_tick_with_snap() {
    let geometry = descending_ramp();
    let mut body = standing().with_snap_length(0.5);
    let mut p = on_ramp(&geometry, -6.0);
    let start_y = p.y;
    let mut v = Vec3::new(0.0, 0.0, 8.0); // PORT_SPEC's max_speed, downhill.
    body.on_floor = true;

    for tick in 0..60 {
        let result = step(&geometry, &mut body, &mut p, &mut v);
        assert!(
            result.on_floor,
            "tick {tick}: went airborne walking down a 16.7° ramp with snap 0.5"
        );
        assert!(
            (degrees(result.floor_angle) - 16.7).abs() < 0.1,
            "tick {tick}: floor angle {}",
            degrees(result.floor_angle)
        );
        // Snap moves the body, never its horizontal speed.
        assert!(
            (v.z - 8.0).abs() < 1e-3,
            "tick {tick}: downhill speed became {}",
            v.z
        );
    }
    // 60 ticks at 8 m/s is 8 m along +Z, and the ramp drops tan(16.7°) per metre.
    assert!((p.z - 2.0).abs() < 0.05, "ended at z = {}", p.z);
    let drop = start_y - p.y;
    let expected = 8.0 * 16.7f32.to_radians().tan();
    assert!(
        (drop - expected).abs() < 0.05,
        "descended {drop} m over 8 m of ramp, expected {expected}"
    );
}

#[test]
fn walking_down_the_same_ramp_goes_airborne_with_no_snap() {
    let geometry = descending_ramp();
    let mut body = standing().with_snap_length(0.0);
    let mut p = on_ramp(&geometry, -6.0);
    let mut v = Vec3::new(0.0, 0.0, 8.0);
    body.on_floor = true;

    let mut airborne_ticks = 0;
    for _ in 0..30 {
        let result = step(&geometry, &mut body, &mut p, &mut v);
        if !result.on_floor {
            airborne_ticks += 1;
        }
    }
    assert!(
        airborne_ticks > 0,
        "snap 0.0 kept the body glued to the ramp anyway"
    );
}

#[test]
fn snap_refuses_to_engage_for_a_body_that_was_already_airborne() {
    // Godot's rule, and the reason a jump is not cancelled by the ground it just
    // left: snap is a *continuation* of being grounded, not a way to acquire it.
    let geometry = descending_ramp();
    let mut body = standing().with_snap_length(0.5);
    let mut p = on_ramp(&geometry, 0.0) + Vec3::Y * 0.3;
    let mut v = Vec3::ZERO;
    body.on_floor = false;

    let result = move_and_slide(&geometry, &mut body, p, v, DT);
    assert!(!result.snapped, "snapped a body that was not on the floor");
    assert!(!result.on_floor);
    p = result.position;
    v = result.velocity;
    let _ = (p, v);
}

#[test]
fn snap_never_lifts_the_body() {
    // A body walking *into* rising ground must be pushed up by the ordinary
    // contact solve, not by the snap probe — a snap that lifted would be a step
    // feature, and this module does not have one.
    let geometry = descending_ramp();
    let mut body = standing().with_snap_length(0.5);
    let mut p = on_ramp(&geometry, 6.0);
    let mut v = Vec3::new(0.0, 0.0, -8.0); // uphill
    body.on_floor = true;

    for _ in 0..40 {
        let before = p.y;
        let result = step(&geometry, &mut body, &mut p, &mut v);
        if result.snapped {
            assert!(
                p.y <= before + collide::CONTACT_MARGIN,
                "a snap lifted the body from {before} to {}",
                p.y
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Sliding
// ---------------------------------------------------------------------------

/// A wall in the ZY plane at x = `x`, 0.5 m thick (PORT_SPEC's shaft walls).
fn wall(level: &mut Level, x: f32) -> Entity {
    level.aabb(Vec3::new(x, 6.0, 0.0), Vec3::new(0.25, 6.0, 3.0))
}

#[test]
fn a_forty_five_degree_approach_keeps_its_tangential_speed() {
    let mut level = Level::new();
    with_floor(&mut level);
    wall(&mut level, 2.0);
    let geometry = level.snapshot();

    let mut body = standing();
    body.on_floor = true;
    let mut p = standing_on(0.0, 0.0, 0.0);
    // 45° into the wall: 8 m/s at (√½, 0, √½) — 5.657 across, 5.657 along.
    let speed = 8.0f32;
    let mut v = Vec3::new(speed * 0.5f32.sqrt(), 0.0, speed * 0.5f32.sqrt());
    let tangential = v.z;

    let mut hit = false;
    for _ in 0..60 {
        let result = step(&geometry, &mut body, &mut p, &mut v);
        if result.on_wall {
            hit = true;
            assert!(
                (v.z - tangential).abs() < 1e-3,
                "the tangential component became {} (was {tangential})",
                v.z
            );
            assert!(
                v.x.abs() < 1e-3,
                "the into-wall component survived at {}",
                v.x
            );
            assert!(result.wall_normal.x < -0.99, "{}", result.wall_normal);
        }
    }
    assert!(hit, "never reached the wall");
    assert!(p.x < 2.0, "tunnelled through to x = {}", p.x);
}

#[test]
fn a_head_on_wall_stops_the_body() {
    let mut level = Level::new();
    with_floor(&mut level);
    wall(&mut level, 2.0);
    let geometry = level.snapshot();

    let mut body = standing();
    body.on_floor = true;
    let mut p = standing_on(0.0, 0.0, 0.0);
    let mut v = Vec3::new(8.0, 0.0, 0.0);

    for _ in 0..60 {
        step(&geometry, &mut body, &mut p, &mut v);
    }
    assert!(v.x.abs() < 1e-3, "head-on impact left {} m/s", v.x);
    assert!(v.z.abs() < 1e-3, "a head-on impact invented sideways motion");
    // Resting against the wall's near face: 1.75 − radius.
    assert!(
        (p.x - (1.75 - 0.35)).abs() < 1e-2,
        "stopped at x = {}, not against the face",
        p.x
    );
}

#[test]
fn a_corner_between_two_walls_neither_jitters_nor_tunnels() {
    let mut level = Level::new();
    with_floor(&mut level);
    wall(&mut level, 2.0); // faces −X
    let corner = level.aabb(Vec3::new(0.0, 6.0, 2.0), Vec3::new(3.0, 6.0, 0.25));
    let _ = corner;
    let geometry = level.snapshot();

    let mut body = standing();
    body.on_floor = true;
    let mut p = standing_on(0.0, 0.0, 0.0);
    let mut v = Vec3::new(8.0, 0.0, 8.0); // straight into the inside corner

    let mut positions = Vec::new();
    for _ in 0..100 {
        step(&geometry, &mut body, &mut p, &mut v);
        positions.push(p);
    }

    // Wedged, not oscillating: the last twenty ticks must be one point.
    let settled = positions[80];
    for (i, q) in positions.iter().enumerate().skip(80) {
        assert!(
            q.abs_diff_eq(settled, 1e-5),
            "tick {i}: the corner jitters — {q} vs {settled}"
        );
    }
    assert!(p.x < 1.75 && p.z < 1.75, "tunnelled the corner to {p}");
    assert!(body.on_floor, "lost the floor while wedged in a corner");
    assert!(
        v.length() < 1e-3,
        "an inside corner left {} m/s of velocity",
        v.length()
    );
}

#[test]
fn a_sphere_collider_slides_like_a_box_one() {
    let mut level = Level::new();
    with_floor(&mut level);
    level.sphere(Vec3::new(2.0, 1.0, 0.0), 1.0);
    let geometry = level.snapshot();

    let mut body = standing();
    body.on_floor = true;
    let mut p = standing_on(0.0, 0.0, 0.3);
    let mut v = Vec3::new(6.0, 0.0, 0.0);

    let mut touched = false;
    for _ in 0..60 {
        let r = step(&geometry, &mut body, &mut p, &mut v);
        touched |= r.contacts.iter().any(|c| c.normal.x.abs() > 0.1);
    }
    assert!(touched, "never met the sphere");
    // Deflected around it rather than stopping dead or passing through.
    assert!(p.z > 0.3, "a round obstacle did not deflect the body: {p}");
}

// ---------------------------------------------------------------------------
// The phase wall: a yawed OBB and a layer mask
// ---------------------------------------------------------------------------

/// PORT_SPEC's PhaseWall: 0.5 × 4 × 8 at (4.56, 2, 14.48), yaw −98°, tagged on
/// the phaseable layer (bit 1 here — layer 0 is the untagged world).
const PHASEABLE: u32 = 1;

fn phase_level() -> (Level, CollisionWorld, Entity) {
    let mut level = Level::new();
    with_floor(&mut level);
    let wall = level.obb(
        Vec3::new(4.56, 2.0, 14.48),
        Vec3::new(0.25, 2.0, 4.0),
        Quat::from_rotation_y((-98.0f32).to_radians()),
    );
    level.layers(wall, CollisionLayers::layer(PHASEABLE));
    let geometry = level.snapshot();
    (level, geometry, wall)
}

/// The wall's outward face normal on the +X-ish side: local +X rotated by −98°.
fn phase_wall_normal() -> Vec3 {
    Quat::from_rotation_y((-98.0f32).to_radians()) * Vec3::X
}

#[test]
fn the_yawed_phase_wall_blocks_from_both_sides() {
    let (_level, geometry, wall) = phase_level();
    let face = phase_wall_normal();
    let centre = Vec3::new(4.56, 0.0, 14.48);

    for side in [1.0f32, -1.0] {
        let approach = face * side;
        let mut body = standing();
        body.on_floor = true;
        let mut p = centre + approach * 3.0 + Vec3::Y;
        let mut v = -approach * 8.0;

        let mut hit_it = false;
        for _ in 0..60 {
            let r = step(&geometry, &mut body, &mut p, &mut v);
            if let Some(c) = r.contacts.iter().find(|c| c.entity == wall) {
                hit_it = true;
                // The face normal must point back the way we came, exactly.
                assert!(
                    c.normal.dot(approach) > 0.999,
                    "side {side}: contact normal {} is not the wall face {}",
                    c.normal,
                    approach
                );
            }
        }
        assert!(hit_it, "side {side}: walked straight past the phase wall");
        // Stopped outside the slab: 0.25 half-thickness + 0.35 radius.
        let along = (p - centre).dot(approach);
        assert!(
            along > 0.25 + 0.35 - 1e-2,
            "side {side}: ended {along} along the face normal — inside the wall"
        );
    }
}

#[test]
fn overlap_capsule_reports_the_phase_wall() {
    let (_level, geometry, wall) = phase_level();
    let inside = Vec3::new(4.56, 1.0, 14.48);

    let hits = geometry.overlap_capsule(inside, 0.35, 2.0, ALL_LAYERS);
    assert!(
        hits.iter().any(|h| h.entity == wall),
        "overlap_capsule missed a wall the capsule is standing inside"
    );
    let hit = hits.iter().find(|h| h.entity == wall).unwrap();
    assert!(hit.depth > 0.0);
    assert!(
        !hit.trigger,
        "a solid wall must not be reported as a trigger"
    );

    // Masking the phaseable layer out is what the phase system does, and the
    // guard must then see nothing.
    let solid_only = ALL_LAYERS & !(1 << PHASEABLE);
    assert!(
        geometry
            .overlap_capsule(inside, 0.35, 2.0, solid_only)
            .iter()
            .all(|h| h.entity != wall),
        "a masked-out wall still answered an overlap query"
    );

    // …and the untagged floor still does, which is the other half of the guard.
    assert!(
        !geometry
            .overlap_capsule(Vec3::new(0.0, 0.5, 0.0), 0.35, 2.0, solid_only)
            .is_empty(),
        "masking the phaseable layer also hid the ordinary world"
    );
}

#[test]
fn masking_the_phaseable_layer_out_passes_through_the_wall_cleanly() {
    let (_level, geometry, wall) = phase_level();
    let face = phase_wall_normal();
    let centre = Vec3::new(4.56, 0.0, 14.48);

    let mut body = standing().with_layers(
        CollisionLayers::DEFAULT.with_mask(ALL_LAYERS & !(1 << PHASEABLE)),
    );
    body.on_floor = true;
    let mut p = centre + face * 3.0 + Vec3::Y;
    let mut v = -face * 8.0;

    for _ in 0..60 {
        let r = step(&geometry, &mut body, &mut p, &mut v);
        assert!(
            r.contacts.iter().all(|c| c.entity != wall),
            "a masked-out wall still produced a contact"
        );
        assert!(r.on_floor, "phasing through a wall also lost the floor");
    }
    let along = (p - centre).dot(face);
    assert!(
        along < -1.0,
        "the body did not pass through: {along} along the face normal"
    );
    assert!(
        (v + face * 8.0).length() < 1e-3,
        "passing through cost speed: {v}"
    );
}

// ---------------------------------------------------------------------------
// Layers
// ---------------------------------------------------------------------------

#[test]
fn a_default_layers_component_collides_exactly_as_no_component_does() {
    // The additive property: adding `CollisionLayers::DEFAULT` to every collider
    // in a scene must not move a single bit.
    let run = |tagged: bool| {
        let mut level = Level::new();
        let floor = with_floor(&mut level);
        let w = wall(&mut level, 2.0);
        if tagged {
            level.layers(floor, CollisionLayers::DEFAULT);
            level.layers(w, CollisionLayers::DEFAULT);
        }
        let geometry = level.snapshot();
        let mut body = standing().with_snap_length(0.5);
        let mut p = Vec3::new(-2.0, 3.0, 0.0);
        let mut v = Vec3::new(9.0, 0.0, 1.0);
        let mut trace = Vec::new();
        for _ in 0..120 {
            step(&geometry, &mut body, &mut p, &mut v);
            trace.push((p, v));
        }
        trace
    };
    assert_eq!(run(false), run(true));
}

#[test]
fn a_mask_written_between_ticks_takes_effect_on_the_next_tick_and_not_mid_tick() {
    let (_level, geometry, wall) = phase_level();
    let face = phase_wall_normal();
    let centre = Vec3::new(4.56, 0.0, 14.48);
    let solid_only = ALL_LAYERS & !(1 << PHASEABLE);

    let mut body = standing();
    body.on_floor = true;
    let mut p = centre + face * 3.0 + Vec3::Y;
    let mut v = -face * 8.0;

    // Run until the wall is actually in the way.
    let mut blocked_tick = None;
    for tick in 0..60 {
        let r = step(&geometry, &mut body, &mut p, &mut v);
        if r.contacts.iter().any(|c| c.entity == wall) {
            blocked_tick = Some(tick);
            break;
        }
    }
    let blocked_tick = blocked_tick.expect("never reached the wall");
    let blocked_at = p;

    // Phase in *between* ticks. The tick that already ran cannot change.
    body.layers.set_mask_layer(PHASEABLE, false);
    assert_eq!(body.layers.mask, solid_only);
    assert_eq!(p, blocked_at, "mutating a mask moved the body");

    // The very next tick sees the new mask.
    v = -face * 8.0;
    let after = move_and_slide(&geometry, &mut body, p, v, DT);
    assert!(
        after.contacts.iter().all(|c| c.entity != wall),
        "tick {} still collided with the phased-out wall",
        blocked_tick + 1
    );
    assert!(
        (after.position - p - v * DT).length() < 1e-5,
        "the phased body did not move freely"
    );

    // And phasing back out restores it.
    body.layers.set_mask_layer(PHASEABLE, true);
    assert_eq!(body.layers.mask, ALL_LAYERS);
    let back = move_and_slide(&geometry, &mut body, blocked_at, -face * 8.0, DT);
    assert!(
        back.contacts.iter().any(|c| c.entity == wall),
        "restoring the mask did not restore the collision"
    );
}

#[test]
fn a_query_reports_triggers_and_flags_them() {
    let mut level = Level::new();
    let solid = with_floor(&mut level);
    let pickup = level.sphere(Vec3::new(0.0, 1.0, 0.0), 0.5);
    level.trigger(pickup);
    let geometry = level.snapshot();

    // 0.1 m below its resting height, so it genuinely overlaps the floor: an
    // overlap query means *overlap*, not the solver's touching tolerance.
    let hits = geometry.overlap_capsule(Vec3::new(0.0, 0.9, 0.0), 0.35, 2.0, ALL_LAYERS);
    assert_eq!(hits.len(), 2, "{hits:?}");
    assert!(hits.iter().any(|h| h.entity == pickup && h.trigger));
    assert!(hits.iter().any(|h| h.entity == solid && !h.trigger));

    // …but the solver walks straight through it.
    let mut body = standing();
    body.on_floor = true;
    let result = move_and_slide(
        &geometry,
        &mut body,
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::ZERO,
        DT,
    );
    assert!(
        result.contacts.iter().all(|c| c.entity != pickup),
        "a trigger pushed the body out"
    );
}

// ---------------------------------------------------------------------------
// Raycasts
// ---------------------------------------------------------------------------

#[test]
fn a_ray_hits_a_ramp_face_with_the_ramp_normal() {
    for pitch in [15.0f32, 30.0, 40.0] {
        let mut level = Level::new();
        ramp(
            &mut level,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 1.0, 4.0),
            pitch,
        );
        let geometry = level.snapshot();

        let hit = geometry
            .raycast(Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y, 10.0, ALL_LAYERS)
            .unwrap_or_else(|| panic!("{pitch}°: the downward ray missed the ramp"));
        let angle = degrees(hit.normal.dot(Vec3::Y).clamp(-1.0, 1.0).acos());
        assert!(
            (angle - pitch).abs() < 1e-3,
            "{pitch}°: the ray normal is {angle}° off vertical"
        );
        // Straight down through the ramp's centre: the top face is at y = 1/cos θ
        // above the centre only along the tilt axis; at z = 0 it is exactly the
        // half-thickness divided by cos θ.
        let expected = 1.0 / pitch.to_radians().cos();
        assert!(
            (hit.point.y - expected).abs() < 1e-3,
            "{pitch}°: hit at y = {}, expected {expected}",
            hit.point.y
        );
        assert!((hit.dist - (5.0 - expected)).abs() < 1e-3);
    }
}

#[test]
fn a_ray_respects_the_layer_mask() {
    let (_level, geometry, wall) = phase_level();
    let face = phase_wall_normal();
    let origin = Vec3::new(4.56, 2.0, 14.48) + face * 4.0;

    let hit = geometry
        .raycast(origin, -face, 8.0, ALL_LAYERS)
        .expect("the ray must find the phase wall");
    assert_eq!(hit.entity, wall);
    assert!(hit.normal.dot(face) > 0.999, "{}", hit.normal);
    assert!((hit.dist - (4.0 - 0.25)).abs() < 1e-3, "dist {}", hit.dist);

    let solid_only = ALL_LAYERS & !(1 << PHASEABLE);
    assert!(
        geometry.raycast(origin, -face, 8.0, solid_only).is_none(),
        "a masked-out wall still stopped a ray"
    );
}

#[test]
fn the_ledge_vault_dual_ray_reads_the_way_port_spec_describes() {
    // PORT_SPEC's LedgeGrab: a head ray at y = 1.5 that MISSES and a chest ray at
    // y = 0.9 that HITS, both 0.6 long along −wall_normal.
    let mut level = Level::new();
    with_floor(&mut level);
    // A 1.2 m-tall ledge whose face is at x = 1.0.
    level.aabb(Vec3::new(2.0, 0.6, 0.0), Vec3::new(1.0, 0.6, 4.0));
    let geometry = level.snapshot();

    let foot = Vec3::new(0.5, 0.0, 0.0);
    let into = Vec3::X;
    let head = geometry.raycast(foot + Vec3::Y * 1.5, into, 0.6, ALL_LAYERS);
    let chest = geometry.raycast(foot + Vec3::Y * 0.9, into, 0.6, ALL_LAYERS);
    assert!(head.is_none(), "the head ray must clear the ledge");
    let chest = chest.expect("the chest ray must find the ledge face");
    // The normal points back along the approach: a ray going +X enters the −X
    // face, whose outward normal is −X.
    assert!(chest.normal.x < -0.999, "{}", chest.normal);
    assert!((chest.dist - 0.5).abs() < 1e-3, "dist {}", chest.dist);
}

#[test]
fn a_ray_that_starts_inside_a_box_reports_the_origin() {
    let mut level = Level::new();
    let block = level.aabb(Vec3::ZERO, Vec3::splat(1.0));
    let geometry = level.snapshot();
    let hit = geometry
        .raycast(Vec3::ZERO, Vec3::X, 5.0, ALL_LAYERS)
        .expect("a ray starting inside must not silently miss");
    assert_eq!(hit.entity, block);
    assert_eq!(hit.dist, 0.0);
}

// ---------------------------------------------------------------------------
// The analytic height field
// ---------------------------------------------------------------------------

/// A `Sim` whose scene is one terrain patch, at the given quality tier.
fn terrain_sim(quality: f32) -> Sim {
    const RON: &str = r#"(
        generators: [
            (name: "ground", spec: Terrain((
                seed: 11, size: (80.0, 80.0), amplitude: 3.0, octaves: 4,
                frequency: 0.05, lacunarity: 2.0, gain: 0.5, base_segments: 48,
            ))),
        ],
        entities: [ (name: Some("ground"), generator: "ground") ],
        camera: (eye: (0.0, 6.0, 10.0), target: (0.0, 0.0, 0.0)),
    )"#;
    Sim::from_config(SimConfig::default().with_scene(RON).with_quality(quality))
}

#[test]
fn a_terrain_raycast_is_independent_of_tessellation() {
    // DESIGN §9's headline claim, applied to the new query. The mesh at
    // Quality(0.3) and Quality(1.0) has different triangles; the *field* it is a
    // view of is the same, and the ray samples the field.
    let mut coarse = terrain_sim(0.3);
    let mut fine = terrain_sim(1.0);

    let a = CollisionWorld::from_world(coarse.world_mut());
    let b = CollisionWorld::from_world(fine.world_mut());
    assert_eq!(a.terrain().len(), 1);
    assert_eq!(b.terrain().len(), 1);

    let mut hits = 0;
    for i in -6..=6 {
        for j in -6..=6 {
            let origin = Vec3::new(i as f32 * 2.5, 30.0, j as f32 * 2.5);
            let ha = a.raycast(origin, Vec3::NEG_Y, 60.0, ALL_LAYERS);
            let hb = b.raycast(origin, Vec3::NEG_Y, 60.0, ALL_LAYERS);
            assert_eq!(
                ha.is_some(),
                hb.is_some(),
                "the two tiers disagree about hitting from {origin}"
            );
            if let (Some(ha), Some(hb)) = (ha, hb) {
                hits += 1;
                assert_eq!(ha.point, hb.point, "from {origin}");
                assert_eq!(ha.normal, hb.normal, "from {origin}");
                assert_eq!(ha.dist, hb.dist, "from {origin}");
                // And the hit really is on the analytic surface.
                let surface = a.terrain()[0]
                    .surface
                    .height_world(a.terrain()[0].origin, ha.point.x, ha.point.z);
                assert!(
                    (ha.point.y - surface).abs() < 1e-3,
                    "hit {} is {} off the field",
                    ha.point,
                    ha.point.y - surface
                );
            }
        }
    }
    assert_eq!(hits, 169, "every probe should have found terrain");
}

#[test]
fn a_capsule_rests_on_the_analytic_field_at_both_quality_tiers() {
    let mut coarse = terrain_sim(0.3);
    let mut fine = terrain_sim(1.0);
    let a = CollisionWorld::from_world(coarse.world_mut());
    let b = CollisionWorld::from_world(fine.world_mut());

    let drop = |geometry: &CollisionWorld| {
        let mut body = standing().with_snap_length(0.5);
        let mut p = Vec3::new(3.0, 20.0, -4.0);
        let mut v = Vec3::ZERO;
        for _ in 0..240 {
            step(geometry, &mut body, &mut p, &mut v);
        }
        (body.on_floor, p, v)
    };
    let (grounded_a, pa, _) = drop(&a);
    let (grounded_b, pb, _) = drop(&b);
    assert!(grounded_a && grounded_b, "the capsule never landed");
    assert_eq!(pa, pb, "the two tiers put the capsule in different places");

    // Resting on the tangent plane: the lower cap's centre — half the segment
    // below the body's centre, i.e. `height/2 - radius` = 0.65 — sits one radius
    // from the surface *along the normal*, not one radius above it in y.
    let entry = &a.terrain()[0];
    let feet = pa - Vec3::Y * 0.65;
    let (h, grad) = entry.surface.sample_world(entry.origin, feet.x, feet.z);
    let normal = runt_core::mesh::terrain::normal_from_gradient(grad);
    let along = (feet - Vec3::new(feet.x, h, feet.z)).dot(normal);
    assert!(
        (along - 0.35).abs() < 2e-3,
        "the capsule rests {along} from the field, not one radius"
    );
}

// ---------------------------------------------------------------------------
// Sub-stepping
// ---------------------------------------------------------------------------

#[test]
fn a_forty_metre_per_second_body_does_not_tunnel_a_half_metre_wall() {
    // Above PORT_SPEC's max_fall_speed (20) and above the ground-pound slam (30):
    // defensive, and the number the sub-step margin was sized against.
    let mut level = Level::new();
    with_floor(&mut level);
    wall(&mut level, 2.0);
    let geometry = level.snapshot();

    let mut body = standing();
    body.on_floor = true;
    let mut p = standing_on(0.0, 0.0, 0.0);
    let mut v = Vec3::new(40.0, 0.0, 0.0);

    let first = move_and_slide(&geometry, &mut body, p, v, DT);
    assert_eq!(
        first.sub_steps, 2,
        "40 m/s × 1/60 = 0.667 m needs two 0.35 m sub-steps"
    );

    for _ in 0..30 {
        step(&geometry, &mut body, &mut p, &mut v);
    }
    assert!(
        p.x < 1.75,
        "a 40 m/s body tunnelled the 0.5 m wall to x = {}",
        p.x
    );
    assert!(v.x.abs() < 1e-3, "it did not stop: {} m/s", v.x);
}

#[test]
fn sub_step_counts_come_from_the_entry_velocity_alone() {
    let mut level = Level::new();
    with_floor(&mut level);
    let geometry = level.snapshot();
    let mut body = standing(); // radius 0.35 → 0.35 m cap.
    let p = Vec3::new(0.0, 20.0, 0.0);

    // |v|·dt vs the 0.35 m cap.
    for (speed, expected) in [(8.0, 1u32), (20.0, 1), (21.1, 2), (30.0, 2), (80.0, 4)] {
        let r = move_and_slide(&geometry, &mut body, p, Vec3::new(speed, 0.0, 0.0), DT);
        assert_eq!(
            r.sub_steps, expected,
            "{speed} m/s (= {} m/tick) took {} sub-steps",
            speed * DT,
            r.sub_steps
        );
    }
    // …and the cap on sub-steps holds.
    let r = move_and_slide(&geometry, &mut body, p, Vec3::new(10_000.0, 0.0, 0.0), DT);
    assert_eq!(r.sub_steps, collide::MAX_SUBSTEPS);
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

/// A scripted run that exercises slides, a snap and a corner: down a 16.7° ramp,
/// across a floor, into an inside corner.
fn scripted_run(geometry: &CollisionWorld, ticks: u32) -> Vec<(Vec3, Vec3, bool)> {
    let mut body = standing().with_snap_length(0.5);
    body.on_floor = true;
    let mut p = Vec3::new(0.0, 3.0, -8.0);
    let mut v = Vec3::new(0.0, 0.0, 8.0);
    let mut trace = Vec::with_capacity(ticks as usize);
    for tick in 0..ticks {
        // A turn into the corner half way through, so the run is not one straight
        // line: this is what puts two walls in contact at once.
        if tick == 120 {
            v.x = 8.0;
        }
        step(geometry, &mut body, &mut p, &mut v);
        trace.push((p, v, body.on_floor));
    }
    trace
}

fn scripted_level() -> CollisionWorld {
    let mut level = Level::new();
    with_floor(&mut level);
    ramp(
        &mut level,
        Vec3::new(0.0, 0.0, -6.0),
        Vec3::new(4.0, 1.0, 6.0),
        16.7,
    );
    level.aabb(Vec3::new(6.0, 6.0, 0.0), Vec3::new(0.25, 6.0, 8.0));
    level.aabb(Vec3::new(0.0, 6.0, 6.0), Vec3::new(8.0, 6.0, 0.25));
    level.snapshot()
}

#[test]
fn the_solver_is_a_pure_function_of_the_snapshot() {
    let geometry = scripted_level();
    let a = scripted_run(&geometry, 240);
    let b = scripted_run(&geometry, 240);
    assert_eq!(a, b, "two identical runs diverged");
    // And a freshly gathered snapshot of the same world is the same value.
    assert_eq!(scripted_run(&scripted_level(), 240), a);
}

#[test]
fn a_scripted_capsule_run_is_identical_under_ragged_and_uniform_hosts() {
    // DESIGN §4: the sim is a function of the input trace and the tick, never of
    // how the host chopped wall time up. `move_and_slide` only ever sees the
    // fixed tick, so a host that delivers 240 ticks in one go and one that
    // delivers them in ragged bursts must agree to the bit.
    let geometry = scripted_level();

    let uniform = scripted_run(&geometry, 240);

    // The ragged host, replayed through the same scripted driver in chunks.
    let mut body = standing().with_snap_length(0.5);
    body.on_floor = true;
    let mut p = Vec3::new(0.0, 3.0, -8.0);
    let mut v = Vec3::new(0.0, 0.0, 8.0);
    let mut ragged = Vec::with_capacity(240);
    let mut tick = 0u32;
    for chunk in [7usize, 1, 13, 2, 40, 1, 1, 60, 5, 110] {
        for _ in 0..chunk {
            if tick == 120 {
                v.x = 8.0;
            }
            step(&geometry, &mut body, &mut p, &mut v);
            ragged.push((p, v, body.on_floor));
            tick += 1;
        }
    }
    assert_eq!(ragged.len(), 240);
    assert_eq!(ragged, uniform, "a ragged host produced a different run");
}

/// The same scripted run, but driven from inside a real `FixedSim` schedule —
/// which is how the port will call it — so that the accumulator, the input
/// buffer and the rest of the tick are in the loop too.
mod in_schedule {
    use super::*;
    use runt_core::collide::{ColliderQuery, TerrainQuery};

    #[derive(Component)]
    struct Player;

    #[derive(Resource, Default, Clone, PartialEq, Debug)]
    struct Trace(Vec<(Vec3, Vec3, bool)>);

    // `Without<Player>` on the read-only queries is what lets bevy prove they
    // cannot alias the `&mut Transform` below — the player is never its own
    // world geometry.
    fn drive_player(
        colliders: Query<ColliderQuery, Without<Player>>,
        terrain: Query<TerrainQuery, Without<Player>>,
        mut player: Query<(&mut CharacterBody, &mut Transform, &mut Velocity), With<Player>>,
        mut trace: ResMut<Trace>,
    ) {
        let geometry = CollisionWorld::gather(&colliders, &terrain);
        for (mut body, mut transform, mut velocity) in &mut player {
            if !body.on_floor {
                velocity.0.y -= GRAVITY * DT;
            }
            let result = move_and_slide(
                &geometry,
                &mut body,
                transform.translation,
                velocity.0,
                DT,
            );
            transform.translation = result.position;
            velocity.0 = result.velocity;
            trace.0.push((result.position, result.velocity, result.on_floor));
        }
    }

    fn sim_with_player() -> Sim {
        let mut sim = Sim::from_config(SimConfig::default().without_scene());
        sim.world_mut().init_resource::<Trace>();
        sim.world_mut().spawn((
            Transform::from_translation(Vec3::new(0.0, -0.5, 0.0)),
            AabbCollider {
                half_extents: Vec3::new(25.0, 0.5, 25.0),
            },
        ));
        sim.world_mut().spawn((
            Transform::from_translation(Vec3::new(4.0, 6.0, 0.0)),
            ObbCollider {
                half_extents: Vec3::new(0.25, 6.0, 8.0),
                rotation: Quat::from_rotation_y((-98.0f32).to_radians()),
            },
        ));
        sim.world_mut().spawn((
            Player,
            standing().with_snap_length(0.5),
            Transform::from_translation(Vec3::new(-4.0, 1.0, 0.0)),
            Velocity(Vec3::new(8.0, 0.0, 1.5)),
        ));
        sim.fixed_sim_mut().add_systems(drive_player);
        sim
    }

    fn trace_of(sim: &Sim) -> Trace {
        sim.world().resource::<Trace>().clone()
    }

    #[test]
    fn a_capsule_driven_from_fixed_sim_is_ragged_host_independent() {
        // Uniform host: one update per tick, 3 s of it.
        let mut uniform = sim_with_player();
        for i in 0..=180u64 {
            uniform.update(i as f64 / 60.0);
        }

        // Ragged host: the same 3 s, delivered in lumps of 5–200 ms. None
        // exceeds `MAX_ACCUMULATED`, so no tick is *dropped* — a dropped tick is
        // a different simulation, not a nondeterministic one.
        let mut ragged = sim_with_player();
        let steps = [0.005f64, 0.030, 0.007, 0.012];
        let mut t = 0.0f64;
        ragged.update(0.0);
        let mut i = 0usize;
        while t < 3.0 {
            t = (t + steps[i % steps.len()]).min(3.0);
            i += 1;
            ragged.update(t);
        }

        assert_eq!(uniform.tick_count(), ragged.tick_count());
        assert_eq!(
            trace_of(&uniform),
            trace_of(&ragged),
            "the same run under two host cadences diverged"
        );
        assert!(
            uniform.tick_count() >= 180,
            "the run was too short to prove anything"
        );
    }

    #[test]
    fn the_solver_never_touches_an_entity_it_was_not_given() {
        // The obb wall is a scene entity with no CharacterBody; nothing may move
        // it, and the ball systems in the schedule must stay no-ops.
        let mut sim = sim_with_player();
        let before: Vec<Vec3> = {
            let mut q = sim.world_mut().query::<(&Transform, &ObbCollider)>();
            q.iter(sim.world()).map(|(t, _)| t.translation).collect()
        };
        for i in 0..=120u64 {
            sim.update(i as f64 / 60.0);
        }
        let after: Vec<Vec3> = {
            let mut q = sim.world_mut().query::<(&Transform, &ObbCollider)>();
            q.iter(sim.world()).map(|(t, _)| t.translation).collect()
        };
        assert_eq!(before, after, "the solver moved a static collider");
        assert!(!before.is_empty());
    }
}

// ---------------------------------------------------------------------------
// Scene authoring
// ---------------------------------------------------------------------------

#[test]
fn a_scene_file_can_author_an_obb_and_its_layers() {
    // PORT_SPEC's PhaseWall, spelled the way a `.ron` level will spell it:
    // half-extents plus Euler degrees, exactly like a transform.
    const RON: &str = r#"(
        generators: [ (name: "block", spec: Cube(size: 1.0)) ],
        entities: [
            (name: Some("phase_wall"), generator: "block",
             transform: (translation: (4.56, 2.0, 14.48), rotation: Euler((0.0, -98.0, 0.0))),
             obb_collider: Some((half_extents: (0.25, 2.0, 4.0),
                                 rotation: Euler((0.0, -98.0, 0.0)))),
             collision_layers: Some((memberships: 2, mask: 65535))),
            (name: Some("plain"), generator: "block",
             aabb_collider: Some((1.0, 1.0, 1.0))),
        ],
        camera: (eye: (0.0, 6.0, 10.0), target: (0.0, 0.0, 0.0)),
    )"#;

    let mut sim = Sim::from_config(SimConfig::default().with_scene(RON));
    let wall = sim.scene_entity("phase_wall").expect("named in the scene");
    let plain = sim.scene_entity("plain").expect("named in the scene");

    let collider = *sim
        .world()
        .get::<ObbCollider>(wall)
        .expect("obb_collider must spawn an ObbCollider");
    assert_eq!(collider.half_extents, Vec3::new(0.25, 2.0, 4.0));
    assert!(collider
        .rotation
        .abs_diff_eq(Quat::from_rotation_y((-98.0f32).to_radians()), 1e-6));

    let layers = *sim
        .world()
        .get::<CollisionLayers>(wall)
        .expect("collision_layers must spawn a CollisionLayers");
    assert_eq!(layers, CollisionLayers::layer(1));

    // An entity that authored no layers gets none, and is therefore the default.
    assert!(sim.world().get::<CollisionLayers>(plain).is_none());

    let geometry = CollisionWorld::from_world(sim.world_mut());
    assert_eq!(geometry.colliders().len(), 2);
    assert!(geometry
        .colliders()
        .iter()
        .any(|c| c.entity == plain && c.memberships == CollisionLayers::DEFAULT.memberships));

    // And the round trip preserves what was authored.
    let saved = runt_core::scene::save_scene(sim.world()).expect("save");
    let reparsed = runt_core::scene::parse_scene(&saved).expect("reparse");
    assert_eq!(reparsed, runt_core::scene::parse_scene(RON).unwrap());
}

#[test]
fn a_scene_without_collision_v2_fields_is_unchanged() {
    // The additive property at the file level: every scene that shipped before
    // this module must parse, spawn and save exactly as it did.
    let desc = runt_core::scene::demo_scene();
    assert!(desc.entities.iter().all(|e| e.obb_collider.is_none()));
    assert!(desc.entities.iter().all(|e| e.collision_layers.is_none()));

    let mut sim = Sim::new();
    sim.update(0.0);
    let geometry = CollisionWorld::from_world(sim.world_mut());
    assert!(
        geometry.colliders().is_empty(),
        "the demo scene grew a collider"
    );
    assert_eq!(geometry.terrain().len(), 1, "the demo has one terrain patch");
}
