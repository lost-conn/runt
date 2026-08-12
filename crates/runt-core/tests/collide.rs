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

use std::sync::Arc;

use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};

use runt_core::collide::{
    self, move_and_slide, CharacterBody, CharacterShape, ColliderEntry, ColliderShape,
    CollisionLayers, CollisionWorld, ContactKind, MoveResult, ObbCollider, Trimesh, ALL_LAYERS,
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

    /// An entity carrying nothing — the id a hand-pushed [`ColliderEntry`]
    /// needs. A trimesh has no component yet (DESIGN §9a: scene authoring is a
    /// later step), so `push_collider` is how one enters a snapshot.
    fn bare(&mut self) -> Entity {
        self.world.spawn_empty().id()
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

// ---------------------------------------------------------------------------
// Static trimeshes (DESIGN §9a's 2026-08-04 trimesh amendment)
// ---------------------------------------------------------------------------
//
// The claim these make together is narrow and load-bearing: a triangulated
// surface behaves like the analytic one it approximates. So almost every test
// below is a *comparison* — the same run against a box and against the two
// triangles of that box's top face, against an OBB ramp and against a
// triangulated one — rather than a number someone chose. What is pinned rather
// than compared is the one thing there is nothing to compare against: the
// 240-tick fingerprint over a bumpy soup.

/// Push a trimesh into a snapshot at `center`.
fn push_trimesh(
    geometry: &mut CollisionWorld,
    entity: Entity,
    center: Vec3,
    mesh: &Arc<Trimesh>,
    memberships: u16,
) {
    geometry.push_collider(ColliderEntry {
        entity,
        center,
        shape: ColliderShape::Trimesh(mesh.clone()),
        memberships,
        trigger: false,
    });
}

/// A `cells × cells` grid over `[-half, half]²`, height from `height(i, j)`.
///
/// Two triangles per cell, wound so both face normals point up. This is the
/// shape a baked surface actually has — many small coplanar-ish triangles with
/// shared internal edges — which is the case the contact normal rules exist for.
fn grid_soup(
    half: f32,
    cells: usize,
    height: impl Fn(usize, usize) -> f32,
) -> (Vec<Vec3>, Vec<u32>) {
    let n = cells + 1;
    let step = 2.0 * half / cells as f32;
    let mut verts = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            verts.push(Vec3::new(
                -half + i as f32 * step,
                height(i, j),
                -half + j as f32 * step,
            ));
        }
    }
    let mut indices = Vec::with_capacity(cells * cells * 6);
    for j in 0..cells {
        for i in 0..cells {
            let a = (j * n + i) as u32;
            let b = a + 1;
            let d = ((j + 1) * n + i) as u32;
            let c = d + 1;
            indices.extend([a, d, c, a, c, b]);
        }
    }
    (verts, indices)
}

/// The **top face** of a box as a two-triangle soup: the same surface the OBB
/// path presents, with nothing else attached, so a comparison isolates the
/// contact routine rather than the shape.
fn box_top_face(half: Vec3, rotation: Quat) -> (Vec<Vec3>, Vec<u32>) {
    let corners = [
        Vec3::new(-half.x, half.y, -half.z),
        Vec3::new(half.x, half.y, -half.z),
        Vec3::new(half.x, half.y, half.z),
        Vec3::new(-half.x, half.y, half.z),
    ];
    (
        corners.iter().map(|c| rotation * *c).collect(),
        vec![0, 2, 1, 0, 3, 2],
    )
}

/// The same geometry as a **per-face** soup: every triangle carrying its own
/// three vertices, which is what a CSG bake or an `.obj` export hands you and
/// what `build` has to weld back together.
fn unwelded(verts: &[Vec3], indices: &[u32]) -> (Vec<Vec3>, Vec<u32>) {
    let out: Vec<Vec3> = indices.iter().map(|i| verts[*i as usize]).collect();
    let idx: Vec<u32> = (0..out.len() as u32).collect();
    (out, idx)
}

/// A deterministic bump field — a fixed function of the grid index, not noise,
/// so the fingerprint below is reproducible from the source alone.
fn bumpy(i: usize, j: usize) -> f32 {
    0.12 * (i as f32 * 0.7).sin() * (j as f32 * 0.9).cos()
}

// -- build -------------------------------------------------------------------

#[test]
fn building_a_trimesh_welds_drops_degenerates_and_is_reproducible() {
    let (verts, indices) = grid_soup(4.0, 6, bumpy);
    let (mut soup, mut soup_indices) = unwelded(&verts, &indices);
    assert_eq!(soup.len(), indices.len(), "the per-face soup is unwelded");

    // Three degenerates a real bake produces: a repeated corner, a collinear
    // sliver, and a zero-area triangle from a vertex the weld will merge.
    let base = soup.len() as u32;
    soup.extend([
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(-2.0, 0.5, 1.0),
        Vec3::new(-1.0, 0.5, 1.0),
        Vec3::new(0.0, 0.5, 1.0),
        Vec3::new(3.0, 1.0, 3.0),
        // Closer to the first than the weld grid, so it becomes the same vertex.
        Vec3::new(3.0 + 1e-6, 1.0, 3.0),
        Vec3::new(3.0, 1.0, 3.5),
    ]);
    soup_indices.extend(base..base + 9);

    let mesh = Trimesh::build_from_soup(&soup, &soup_indices);
    assert_eq!(
        mesh.triangle_count(),
        indices.len() / 3,
        "the three degenerate triangles survived the build"
    );
    assert_eq!(
        mesh.vertex_count(),
        verts.len() + 7,
        "welding did not collapse the per-face soup back onto the grid \
         (the seven extra are the degenerate triangles' own distinct corners — \
         nine positions, two of which weld together and none of which land on \
         the grid)"
    );
    assert!(
        mesh.face_normals().iter().all(|n| n.is_finite() && (n.length() - 1.0).abs() < 1e-5),
        "a face normal is not a unit vector"
    );
    assert!(
        mesh.face_normals().iter().all(|n| n.y > 0.0),
        "the grid's triangles are wound the wrong way"
    );

    // The whole point of the weld: an unwelded soup and the indexed mesh it came
    // from are the same collider. Not the same *index* arrays — first-occurrence
    // order differs, and the degenerates left a few vertices behind — but the
    // same triangles, in the same order, with the same tree over them.
    let indexed = Trimesh::build_from_soup(&verts, &indices);
    let positions = |m: &Trimesh| -> Vec<[Vec3; 3]> {
        m.tris()
            .iter()
            .map(|t| {
                [
                    m.verts()[t[0] as usize],
                    m.verts()[t[1] as usize],
                    m.verts()[t[2] as usize],
                ]
            })
            .collect()
    };
    assert_eq!(positions(&indexed), positions(&mesh));
    assert_eq!(indexed.face_normals(), mesh.face_normals());
    assert_eq!(indexed.nodes(), mesh.nodes());
    assert_eq!(indexed.edge_flags(), mesh.edge_flags());
}

#[test]
fn a_shared_edge_is_a_seam_unless_the_surface_turns_a_corner_at_it() {
    // What `edge_flags` decides, and the whole reason a tessellated floor can be
    // walked on: an edge two triangles share is a *seam* when the neighbour
    // continues this triangle's surface, and a *ridge* only when the surface
    // genuinely folds away behind it.

    // Flat grid: every internal edge is a seam, and only the sheet's own open
    // boundary is exposed. 4 × 4 cells is 16 boundary edges.
    let (verts, indices) = grid_soup(4.0, 4, |_, _| 0.0);
    let flat = Trimesh::build_from_soup(&verts, &indices);
    let exposed: u32 = flat.edge_flags().iter().map(|f| f.count_ones()).sum();
    assert_eq!(
        exposed, 16,
        "a flat grid grew a ridge somewhere in its interior"
    );

    // Two quads meeting along `y`: a tent (they fold *up* to the shared edge) has
    // a real ridge there, a gutter (they fold *down* to it) has a seam. Same
    // triangles, same winding, one sign apart — so the flag is reading the
    // geometry and not the topology.
    let fold = |ridge_y: f32, outer_y: f32| {
        let soup = vec![
            Vec3::new(-1.0, outer_y, -1.0),
            Vec3::new(1.0, outer_y, -1.0),
            Vec3::new(-1.0, ridge_y, 0.0),
            Vec3::new(1.0, ridge_y, 0.0),
            Vec3::new(-1.0, outer_y, 1.0),
            Vec3::new(1.0, outer_y, 1.0),
        ];
        // Both faces wound so their normals point upwards. Triangle 0 is
        // `(v0, v2, v3)`, whose edge 1 is the shared one.
        let idx = vec![0, 2, 3, 0, 3, 1, 2, 4, 5, 2, 5, 3];
        Trimesh::build_from_soup(&soup, &idx)
    };

    // Triangle 0 is `(v0, v2, v3)`: edge 0 is `v0→v2`, the sheet's open left
    // side; edge 1 is `v2→v3`, the fold; edge 2 is `v3→v0`, the quad's own
    // diagonal, which its coplanar other half shares.
    let tent = fold(1.0, 0.0);
    assert_eq!(tent.triangle_count(), 4);
    assert!(tent.face_normals().iter().all(|n| n.y > 0.0));
    assert_eq!(
        tent.edge_flags()[0],
        0b011,
        "the open side and the convex ridge are exposed; the coplanar diagonal \
         is a seam"
    );

    let gutter = fold(0.0, 1.0);
    assert!(gutter.face_normals().iter().all(|n| n.y > 0.0));
    assert_eq!(
        gutter.edge_flags()[0],
        0b001,
        "a concave fold is a seam: nothing can touch it from the front"
    );
}

#[test]
fn building_the_same_soup_twice_produces_identical_arrays() {
    // DESIGN §9a: the BVH is built once at load and *deterministically*. Median
    // index split plus a `(axis value, original index)` sort key means there is
    // exactly one tree for a given soup — no tie a hash seed or a sort's
    // internal order could break.
    let (verts, indices) = grid_soup(6.0, 9, bumpy);
    let (soup, soup_indices) = unwelded(&verts, &indices);

    let a = Trimesh::build_from_soup(&soup, &soup_indices);
    let b = Trimesh::build_from_soup(&soup, &soup_indices);

    assert_eq!(a.verts(), b.verts(), "welded vertices differ");
    assert_eq!(a.tris(), b.tris(), "triangle order differs");
    assert_eq!(a.face_normals(), b.face_normals(), "face normals differ");
    assert_eq!(a.nodes(), b.nodes(), "BVH nodes differ");
    assert_eq!(a, b);

    // …and the tree really is one: the leaves partition the triangle array,
    // every leaf is within the fixed size, and a node's box contains its
    // children's. "The lowest triangle index" is only a rule if the leaves are
    // contiguous ranges of the array the tie-break names.
    let mut covered = vec![0u32; a.triangle_count()];
    for node in a.nodes() {
        match (node.leaf_range(), node.children()) {
            (Some((first, count)), None) => {
                assert!(count as usize <= 8, "a leaf holds {count} triangles");
                for t in first..first + count {
                    covered[t as usize] += 1;
                }
            }
            (None, Some((left, right))) => {
                assert!(node.split_axis().is_some_and(|axis| axis < 3));
                let (lo, hi) = node.bounds();
                for child in [left, right] {
                    let (clo, chi) = a.nodes()[child as usize].bounds();
                    assert!(
                        clo.cmpge(lo).all() && chi.cmple(hi).all(),
                        "a child's box escapes its parent's"
                    );
                }
            }
            _ => unreachable!("a node is a leaf or an inner node"),
        }
    }
    assert!(
        covered.iter().all(|c| *c == 1),
        "the leaves do not partition the triangles exactly once"
    );
}

// -- resting on a triangulated surface ---------------------------------------

/// Drop a standing capsule from `start` and run it for `ticks`.
fn drop_onto(geometry: &CollisionWorld, start: Vec3, ticks: u32) -> (CharacterBody, Vec3) {
    let mut body = standing();
    let mut p = start;
    let mut v = Vec3::ZERO;
    for _ in 0..ticks {
        step(geometry, &mut body, &mut p, &mut v);
    }
    (body, p)
}

#[test]
fn a_capsule_settles_on_a_triangulated_plane_where_it_settles_on_a_box_floor() {
    // The headline equivalence: two triangles and a 50 m slab are the same
    // ground. Anything else — a floor half a contact margin low, a body resting
    // on the *edge* direction rather than the face — shows up here as a
    // millimetre.
    let mut boxed = Level::new();
    with_floor(&mut boxed);
    let box_geometry = boxed.snapshot();

    let mut level = Level::new();
    let ground = level.bare();
    let (verts, indices) = grid_soup(25.0, 1, |_, _| 0.0);
    assert_eq!(indices.len(), 6, "a one-cell grid is two triangles");
    let mesh = Trimesh::build_from_soup(&verts, &indices);
    let mut geometry = level.snapshot();
    push_trimesh(&mut geometry, ground, Vec3::ZERO, &mesh, 1);

    let start = Vec3::new(0.7, 4.0, -1.3);
    let (box_body, box_p) = drop_onto(&box_geometry, start, 120);
    let (tri_body, tri_p) = drop_onto(&geometry, start, 120);

    assert!(box_body.on_floor && tri_body.on_floor, "one never landed");
    assert!(
        (tri_p.y - 1.0).abs() < 1e-4,
        "the capsule rests at {} on triangles, not 1.0 above them",
        tri_p.y
    );
    assert!(
        (tri_p.y - box_p.y).abs() < 1e-4,
        "triangles rest the body at {}, the box at {}",
        tri_p.y,
        box_p.y
    );
    assert!(
        (tri_p.x - start.x).abs() < 1e-4 && (tri_p.z - start.z).abs() < 1e-4,
        "the drop walked sideways to {tri_p}"
    );

    // And it stays put to the bit, which is what `floor_stop_on_slope` promises
    // and what a contact normal wobbling between face and edge would break.
    let mut body = tri_body;
    let mut p = tri_p;
    let mut v = Vec3::ZERO;
    for tick in 0..240 {
        step(&geometry, &mut body, &mut p, &mut v);
        assert_eq!(p, tri_p, "the settled capsule drifted on tick {tick}");
    }
}

#[test]
fn a_triangulated_thirty_degree_ramp_is_walked_like_the_obb_one() {
    // PORT_SPEC's 30° ramp, twice: once as the solid box the OBB path solves in
    // the box's own frame, once as the two triangles of that box's top face. The
    // surfaces are the same plane, so 120 ticks of walking up it have to agree.
    let half = Vec3::new(6.0, 1.0, 6.0);
    let rotation = Quat::from_rotation_x(30.0f32.to_radians());

    let mut boxed = Level::new();
    boxed.obb(Vec3::ZERO, half, rotation);
    let box_geometry = boxed.snapshot();

    let mut level = Level::new();
    let face = level.bare();
    let (verts, indices) = box_top_face(half, rotation);
    let mesh = Trimesh::build_from_soup(&verts, &indices);
    assert_eq!(mesh.triangle_count(), 2);
    let mut tri_geometry = level.snapshot();
    push_trimesh(&mut tri_geometry, face, Vec3::ZERO, &mesh, 1);

    // Seeded from the box, so the two runs start at the same bits.
    let hit = box_geometry
        .raycast(Vec3::new(0.0, 8.0, 2.5), Vec3::NEG_Y, 16.0, ALL_LAYERS)
        .expect("the ramp is under the probe");
    let start = hit.point + Vec3::Y;

    // The trimesh raycast has to find the same surface, to the millimetre.
    let tri_hit = tri_geometry
        .raycast(Vec3::new(0.0, 8.0, 2.5), Vec3::NEG_Y, 16.0, ALL_LAYERS)
        .expect("the triangulated ramp is under the probe");
    assert_eq!(tri_hit.entity, face);
    assert!(
        (tri_hit.dist - hit.dist).abs() < 1e-4,
        "the two ramps are {} apart under the probe",
        (tri_hit.dist - hit.dist).abs()
    );
    assert!(tri_hit.normal.abs_diff_eq(hit.normal, 1e-5));

    let run = |geometry: &CollisionWorld| {
        let mut body = standing().with_snap_length(0.5);
        body.on_floor = true;
        let mut p = start;
        let mut v = Vec3::ZERO;
        let mut trace = Vec::with_capacity(120);
        for _ in 0..120 {
            // Uphill: the ramp descends towards +Z.
            let r = walk_step(geometry, &mut body, &mut p, &mut v, Vec3::new(0.0, 0.0, -3.0));
            trace.push((p, r.on_floor, r.floor_angle));
        }
        trace
    };

    let box_trace = run(&box_geometry);
    let tri_trace = run(&tri_geometry);
    for (tick, (b, t)) in box_trace.iter().zip(&tri_trace).enumerate() {
        assert_eq!(b.1, t.1, "tick {tick}: grounded disagrees");
        assert!(
            b.0.abs_diff_eq(t.0, 1e-3),
            "tick {tick}: the box put the body at {} and the triangles at {}",
            b.0,
            t.0
        );
        assert!(
            (degrees(t.2) - 30.0).abs() < 0.05,
            "tick {tick}: the triangulated ramp reported {}°",
            degrees(t.2)
        );
    }
    // The run has to have gone somewhere, or it proved nothing.
    let climbed = tri_trace.last().unwrap().0;
    assert!(
        start.z - climbed.z > 5.0,
        "the body only climbed {} m of ramp",
        start.z - climbed.z
    );
    assert!(climbed.y - start.y > 2.5, "it climbed without rising");
}

#[test]
fn crossing_an_internal_edge_keeps_a_stable_floor_normal() {
    // The reason `trimesh_contact` snaps to the face normal on an interior
    // closest point. A tessellated floor is nothing but shared edges; a body
    // walking over one must not read the direction to that edge as a surface,
    // which is a normal tilting away from vertical — and a normal past
    // `max_floor_angle` is a *wall*, so the failure is a phantom wall blip and a
    // hitch in a run, not a cosmetic one.
    let mut level = Level::new();
    let ground = level.bare();
    let (verts, indices) = grid_soup(6.0, 8, |_, _| 0.0);
    let mesh = Trimesh::build_from_soup(&verts, &indices);
    assert_eq!(mesh.triangle_count(), 128, "8×8 cells, two triangles each");
    let mut geometry = level.snapshot();
    push_trimesh(&mut geometry, ground, Vec3::ZERO, &mesh, 1);

    let mut body = standing().with_snap_length(0.5);
    body.on_floor = true;
    // Slightly off the vertex rows in z, so the crossings are the cells' shared
    // diagonals and boundaries rather than a lattice of exact corners.
    let mut p = Vec3::new(-3.0, 1.0, 0.05);
    let mut v = Vec3::ZERO;

    for tick in 0..120 {
        let r = walk_step(&geometry, &mut body, &mut p, &mut v, Vec3::new(3.0, 0.0, 0.0));
        assert!(r.on_floor, "tick {tick}: went airborne on flat ground");
        assert!(
            !r.on_wall,
            "tick {tick}: an internal edge read as a wall ({})",
            r.wall_normal
        );
        assert!(
            !r.on_ceiling,
            "tick {tick}: an internal edge read as a ceiling"
        );
        assert!(
            r.floor_normal.abs_diff_eq(Vec3::Y, 1e-5),
            "tick {tick}: the floor normal became {} at x = {}",
            r.floor_normal,
            p.x
        );
        assert!(
            (p.y - 1.0).abs() < 1e-4,
            "tick {tick}: the body's height became {} crossing an edge",
            p.y
        );
        assert!(
            r.contacts.iter().all(|c| c.kind == ContactKind::Floor),
            "tick {tick}: a flat floor produced {:?}",
            r.contacts.iter().map(|c| c.kind).collect::<Vec<_>>()
        );
    }
    // It really did cross the edges rather than stalling on one.
    assert!(p.x > 2.5, "the walk stalled at x = {}", p.x);
    assert!(
        (v.x - 3.0).abs() < 1e-3,
        "an internal edge ate {} m/s of forward speed",
        3.0 - v.x
    );
}

// -- queries -----------------------------------------------------------------

#[test]
fn a_ray_hits_the_triangle_it_should_with_that_triangle_s_normal() {
    let mut level = Level::new();
    let flat = level.bare();
    let tilted = level.bare();

    let (fv, fi) = grid_soup(4.0, 4, |_, _| 0.0);
    let flat_mesh = Trimesh::build_from_soup(&fv, &fi);
    let rotation = Quat::from_rotation_x(30.0f32.to_radians());
    let (tv, ti) = box_top_face(Vec3::new(3.0, 0.0, 3.0), rotation);
    let tilted_mesh = Trimesh::build_from_soup(&tv, &ti);

    let mut geometry = level.snapshot();
    // The flat sheet sits 2 m up; the tilted one is 20 m away in +X.
    push_trimesh(&mut geometry, flat, Vec3::new(0.0, 2.0, 0.0), &flat_mesh, 1);
    push_trimesh(
        &mut geometry,
        tilted,
        Vec3::new(20.0, 0.0, 0.0),
        &tilted_mesh,
        1 << PHASEABLE,
    );

    // Straight down onto the flat sheet: t is the drop, the normal is up, and
    // the point is on the plane.
    let hit = geometry
        .raycast(Vec3::new(0.7, 6.0, -1.3), Vec3::NEG_Y, 10.0, ALL_LAYERS)
        .expect("the sheet is under the ray");
    assert_eq!(hit.entity, flat);
    assert!((hit.dist - 4.0).abs() < 1e-5, "dist {}", hit.dist);
    assert!(hit.normal.abs_diff_eq(Vec3::Y, 1e-6), "{}", hit.normal);
    assert!((hit.point.y - 2.0).abs() < 1e-5);
    assert!(!hit.trigger);

    // A ray that misses the sheet's extent finds nothing, even though the BVH
    // root's box is right there.
    assert!(
        geometry
            .raycast(Vec3::new(9.0, 6.0, 0.0), Vec3::NEG_Y, 10.0, ALL_LAYERS)
            .is_none(),
        "a ray outside the sheet still hit it"
    );

    // The tilted sheet reports the plane's own normal, and the ray reaches it
    // through the mask.
    let tilt = geometry
        .raycast(Vec3::new(20.0, 6.0, 0.0), Vec3::NEG_Y, 10.0, ALL_LAYERS)
        .expect("the tilted sheet is under the ray");
    assert_eq!(tilt.entity, tilted);
    let expected = rotation * Vec3::Y;
    assert!(tilt.normal.abs_diff_eq(expected, 1e-5), "{}", tilt.normal);
    assert!((tilt.dist - 6.0).abs() < 1e-4, "dist {}", tilt.dist);

    // Mask filtering is the same one-way rule everything else obeys.
    let solid_only = ALL_LAYERS & !(1 << PHASEABLE);
    assert!(
        geometry
            .raycast(Vec3::new(20.0, 6.0, 0.0), Vec3::NEG_Y, 10.0, solid_only)
            .is_none(),
        "a masked-out trimesh still stopped a ray"
    );
    // …and the untagged sheet still answers, so the mask hid one thing only.
    assert!(geometry
        .raycast(Vec3::new(0.0, 6.0, 0.0), Vec3::NEG_Y, 10.0, solid_only)
        .is_some());

    // A ray from *below* hits too, with the normal turned to face it: the box
    // and sphere paths both answer a ray that starts inside, and a soup's
    // winding is the drawing's business, not the collider's.
    let under = geometry
        .raycast(Vec3::new(0.0, 0.0, 0.0), Vec3::Y, 10.0, ALL_LAYERS)
        .expect("a ray from below must find the sheet");
    assert_eq!(under.entity, flat);
    assert!(under.normal.abs_diff_eq(Vec3::NEG_Y, 1e-6), "{}", under.normal);
}

#[test]
fn the_bvh_raycast_answers_what_a_brute_force_scan_answers() {
    // The traversal prunes with the running best and visits the near child
    // first. Both are optimisations; neither may change an answer. A hundred and
    // sixty-nine rays over a bumpy soup, against the same soup scanned linearly.
    let (verts, indices) = grid_soup(8.0, 16, bumpy);
    let mesh = Trimesh::build_from_soup(&verts, &indices);

    let mut level = Level::new();
    let ground = level.bare();
    let mut geometry = level.snapshot();
    push_trimesh(&mut geometry, ground, Vec3::ZERO, &mesh, 1);

    // Möller–Trumbore, restated here so the comparison is against an
    // independent scan rather than against the routine under test.
    let brute = |origin: Vec3, dir: Vec3, max_dist: f32| -> Option<(f32, u32)> {
        let mut best: Option<(f32, u32)> = None;
        for (index, tri) in mesh.tris().iter().enumerate() {
            let a = mesh.verts()[tri[0] as usize];
            let b = mesh.verts()[tri[1] as usize];
            let c = mesh.verts()[tri[2] as usize];
            let (e1, e2) = (b - a, c - a);
            let pv = dir.cross(e2);
            let det = e1.dot(pv);
            if det.abs() < 1e-12 {
                continue;
            }
            let inv = 1.0 / det;
            let tv = origin - a;
            let u = tv.dot(pv) * inv;
            if !(0.0..=1.0).contains(&u) {
                continue;
            }
            let qv = tv.cross(e1);
            let v = dir.dot(qv) * inv;
            if v < 0.0 || u + v > 1.0 {
                continue;
            }
            let t = e2.dot(qv) * inv;
            if t < 0.0 || t > max_dist {
                continue;
            }
            let index = index as u32;
            if best.is_none_or(|(bt, bi)| t < bt || (t == bt && index < bi)) {
                best = Some((t, index));
            }
        }
        best
    };

    let mut hits = 0;
    for i in -6..=6 {
        for j in -6..=6 {
            let origin = Vec3::new(i as f32 * 1.1, 5.0, j as f32 * 1.1);
            // Not straight down: a slanted ray crosses several nodes, which is
            // what exercises the ordering.
            let dir = Vec3::new(0.25, -1.0, -0.15).normalize();
            let expected = brute(origin, dir, 20.0);
            let got = geometry.raycast(origin, dir, 20.0, ALL_LAYERS);
            assert_eq!(
                expected.is_some(),
                got.is_some(),
                "from {origin}: brute force and the BVH disagree about hitting"
            );
            if let (Some((t, index)), Some(hit)) = (expected, got) {
                hits += 1;
                // Exact, except for the one case where "exact" is not a
                // well-posed question: a ray crossing the shared edge of two
                // triangles hits both at the same `t` up to an ulp, either
                // answer is the surface, and which one the BVH sees first
                // depends on the pruning bound. A whole ulp is four orders of
                // magnitude inside the tolerance that would hide a real miss.
                assert!(
                    (hit.dist - t).abs() < 1e-5,
                    "from {origin}: the BVH says {} and a linear scan {t}",
                    hit.dist
                );
                if hit.dist == t {
                    let n = mesh.face_normals()[index as usize];
                    let n = if dir.dot(n) > 0.0 { -n } else { n };
                    assert_eq!(hit.normal, n, "from {origin}");
                }
            }
        }
    }
    assert!(hits > 150, "only {hits} of 169 rays found the soup");
}

#[test]
fn overlap_queries_report_a_trimesh_like_they_report_a_box() {
    let mut level = Level::new();
    let ground = level.bare();
    let (verts, indices) = grid_soup(6.0, 4, |_, _| 0.0);
    let mesh = Trimesh::build_from_soup(&verts, &indices);
    let mut geometry = level.snapshot();
    push_trimesh(&mut geometry, ground, Vec3::ZERO, &mesh, 1 << PHASEABLE);

    // A capsule 0.1 m into the surface genuinely overlaps it.
    let hits = geometry.overlap_capsule(Vec3::new(0.4, 0.9, -0.7), 0.35, 2.0, ALL_LAYERS);
    assert_eq!(hits.len(), 1, "one collider, one hit: {hits:?}");
    assert_eq!(hits[0].entity, ground);
    assert!(
        (hits[0].depth - 0.1).abs() < 1e-4,
        "depth {} for a 0.1 m overlap",
        hits[0].depth
    );
    assert!(hits[0].normal.abs_diff_eq(Vec3::Y, 1e-5));
    assert!(!hits[0].trigger);

    // Clear of it: overlap means overlap, not the solver's touching band.
    assert!(geometry
        .overlap_capsule(Vec3::new(0.4, 1.001, -0.7), 0.35, 2.0, ALL_LAYERS)
        .is_empty());
    // Off the edge of the sheet entirely.
    assert!(geometry
        .overlap_sphere(Vec3::new(20.0, 0.0, 0.0), 0.35, ALL_LAYERS)
        .is_empty());
    // A sphere sitting in the surface.
    let sphere_hits = geometry.overlap_sphere(Vec3::new(1.0, 0.2, 1.0), 0.5, ALL_LAYERS);
    assert_eq!(sphere_hits.len(), 1);
    assert!((sphere_hits[0].depth - 0.3).abs() < 1e-4);

    // And the same one-way mask rule.
    assert!(geometry
        .overlap_capsule(
            Vec3::new(0.4, 0.9, -0.7),
            0.35,
            2.0,
            ALL_LAYERS & !(1 << PHASEABLE)
        )
        .is_empty());

    // `overlap_body` says the same thing about the shape a solver would use.
    let mut body = standing();
    body.layers = CollisionLayers::DEFAULT;
    assert_eq!(
        geometry
            .overlap_body(&body, Vec3::new(0.4, 0.9, -0.7))
            .len(),
        1
    );
}

// -- determinism -------------------------------------------------------------

#[test]
fn a_run_over_a_trimesh_is_a_pure_function_of_the_snapshot() {
    let (verts, indices) = grid_soup(8.0, 16, bumpy);
    let build = || {
        let mut level = Level::new();
        let ground = level.bare();
        let mesh = Trimesh::build_from_soup(&verts, &indices);
        let mut geometry = level.snapshot();
        push_trimesh(&mut geometry, ground, Vec3::ZERO, &mesh, 1);
        geometry
    };
    let run = |geometry: &CollisionWorld| {
        let mut body = standing().with_snap_length(0.5);
        let mut p = Vec3::new(-6.0, 2.0, -1.5);
        let mut v = Vec3::new(3.0, 0.0, 0.5);
        let mut trace = Vec::with_capacity(240);
        for _ in 0..240 {
            step(geometry, &mut body, &mut p, &mut v);
            trace.push((p, v, body.on_floor));
        }
        trace
    };

    let a = run(&build());
    assert_eq!(run(&build()), a, "two identical runs over a soup diverged");
    // A freshly built mesh is the same collider, not merely an equal one.
    let one = build();
    assert_eq!(run(&one), a);
    assert_eq!(run(&one), a);
}

/// FNV-1a over the capsule's position, velocity and grounded flag, every tick,
/// for 240 ticks of a run across the bumpy soup — the same hash shape
/// `physics.rs` pins the demo scene with. Returns the trace's end state too, so
/// the test can say what the hash is a hash *of*.
fn bumpy_trimesh_fingerprint() -> (u64, Vec3, bool, u32) {
    let (verts, indices) = grid_soup(8.0, 16, bumpy);
    let mesh = Trimesh::build_from_soup(&verts, &indices);
    let mut level = Level::new();
    let ground = level.bare();
    let mut geometry = level.snapshot();
    push_trimesh(&mut geometry, ground, Vec3::ZERO, &mesh, 1);

    let mut body = standing().with_snap_length(0.5);
    let mut p = Vec3::new(-6.0, 2.0, -1.5);
    let mut v = Vec3::new(3.0, 0.0, 0.5);

    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut grounded_ticks = 0u32;
    for _ in 0..240 {
        step(&geometry, &mut body, &mut p, &mut v);
        grounded_ticks += body.on_floor as u32;
        for word in [
            p.x.to_bits(),
            p.y.to_bits(),
            p.z.to_bits(),
            v.x.to_bits(),
            v.y.to_bits(),
            v.z.to_bits(),
            body.on_floor as u32,
        ] {
            h ^= word as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    (h, p, body.on_floor, grounded_ticks)
}

#[test]
fn the_bumpy_trimesh_run_ticks_to_its_pinned_fingerprint() {
    // The one number here that is pinned rather than compared. Everything the
    // trimesh path does is in it: the weld, the BVH's triangle order, the leaf
    // scan order, the seam/ridge flags, the face-normal snap, the
    // deepest-contact tie-break, the snap probe. If any of them changes, this
    // moves — which is the point. A change that is *meant* to move it re-pins it
    // with a note saying why.
    let (hash, end, on_floor, grounded_ticks) = bumpy_trimesh_fingerprint();

    // …but a hash of a run that fell off the world would be just as stable, so
    // say what the run was: a dozen metres of +X across a 16 m sheet, deflected
    // by the bumps (nothing drives the horizontal velocity — the only thing that
    // changes it is the surface), and still on that surface at the end.
    assert!(on_floor, "the run ended airborne");
    assert!(
        end.x > 5.0 && end.x < 7.5,
        "the run ended at x = {}, not a dozen metres of +X from −6",
        end.x
    );
    assert!(
        end.z.abs() < 2.0,
        "the bumps deflected the run {} m off its lane — it is not crossing \
         the sheet any more",
        end.z
    );
    assert!(
        end.y > 0.85 && end.y < 1.15,
        "the capsule is at {} — off the ±0.12 m surface it was rolling over",
        end.y
    );
    assert!(
        grounded_ticks > 200,
        "only {grounded_ticks} of 240 ticks were grounded; the run is a fall, \
         not a roll across bumps"
    );

    // Re-pinned 2026-08-06 with the ridge cone (`ridge_cone_normal`) and the
    // cross-collider seam rule. The bumpy soup is nothing but convex ridges, and
    // on every one of them the capsule used to be able to pick up a normal
    // tilted past the neighbouring face — the same defect that made a cliff lip
    // snag and a bridge joint hop. The run's *shape* is unchanged: the five
    // assertions above were written against the old number and all still hold,
    // which is what says the geometry of the run did not move. What moved are
    // the normals on the ticks where the body crossed a crest, and the
    // widened gather band (`WITNESS_MARGIN`) that lets the seam rule see the
    // surface next door.
    assert_eq!(
        hash, 0x1ce6_195a_6e72_40fe,
        "the 240-tick run across the bumpy soup changed"
    );
}

#[test]
fn a_trimesh_snapshot_is_a_value_that_shares_its_geometry() {
    // An `Arc` in a collider entry is what makes a level's baked soup one
    // allocation no matter how many entries reference it — and cloning a
    // snapshot has to stay a refcount bump rather than a copy of the mesh.
    let (verts, indices) = grid_soup(4.0, 4, bumpy);
    let mesh = Trimesh::build_from_soup(&verts, &indices);
    let mut level = Level::new();
    let a = level.bare();
    let b = level.bare();
    let mut geometry = level.snapshot();
    push_trimesh(&mut geometry, a, Vec3::ZERO, &mesh, 1);
    push_trimesh(&mut geometry, b, Vec3::new(0.0, -4.0, 0.0), &mesh, 1);

    assert_eq!(Arc::strong_count(&mesh), 3, "the soup was copied, not shared");
    let cloned = geometry.clone();
    assert_eq!(Arc::strong_count(&mesh), 5);
    assert_eq!(cloned, geometry);

    // Entity order, as everywhere else: the scan is a `Vec` walk in it.
    assert!(geometry.colliders()[0].entity < geometry.colliders()[1].entity);
    assert_eq!(geometry.colliders().len(), 2);

    // Two sheets 4 m apart: a ray finds the near one.
    let hit = geometry
        .raycast(Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y, 20.0, ALL_LAYERS)
        .expect("both sheets are under the ray");
    assert_eq!(hit.entity, a);
}

// ---------------------------------------------------------------------------
// Joints between colliders, and the lip of a cliff
// ---------------------------------------------------------------------------
//
// Two feel bugs from playtesting the port, and the geometry they came from.
//
// 1. A phase bridge butted against a mountain ledge always bumped. The bridge is
//    a box collider, the ledge is a baked soup, and `Trimesh`'s welder — which
//    already solves this *inside* one soup — is built per collider and cannot
//    see across the joint. The far collider's edge-region normal tilts backwards
//    over the surface the body is on, and the tilt goes almost entirely into
//    *vertical* velocity: a hop over flat ground.
// 2. Walking or rolling off a steep cliff always snagged. The lip is a genuine
//    convex ridge, so the soup's own flags call it exposed and the contact takes
//    the direction to it — but a body standing a centimetre *back* from the lip
//    is over the ledge top, not against the ridge, and the direction to the
//    ridge is then tilted past the ground it is walking on. Against a
//    near-vertical face it crosses behind the face's plane entirely and the old
//    code read that as penetration: a 0.36 m ejection at the lip.
//
// Both are measured the same way — sweep the sub-tick phase at which the body
// arrives at the feature, because a body that steps exactly onto a joint never
// samples the band the defect lives in.

/// PORT_SPEC's walking body at the port's own numbers: `moves.rs`
/// `CAPSULE_RADIUS` 0.35 / `CAPSULE_HEIGHT` 1.4, `FLOOR_ANGLE_STANDING_DEG` 45,
/// `SNAP_GROUNDED` 0.5.
fn walker() -> CharacterBody {
    CharacterBody::default()
        .with_shape(CharacterShape::Capsule {
            radius: 0.35,
            height: 1.4,
        })
        .with_max_floor_degrees(45.0)
        .with_snap_length(0.5)
}

/// The same body rolling: `ROLL_RADIUS` 0.35 as a sphere,
/// `FLOOR_ANGLE_ROLLING_DEG` 180 — every contact is floor — and the same snap.
fn roller() -> CharacterBody {
    CharacterBody::default()
        .with_shape(CharacterShape::Sphere { radius: 0.35 })
        .with_max_floor_degrees(180.0)
        .with_snap_length(0.5)
}

/// Half the body's height: where its centre sits when its feet rest on `y = 0`.
fn resting_offset(body: &CharacterBody) -> f32 {
    match body.shape {
        CharacterShape::Capsule { height, .. } => height * 0.5,
        CharacterShape::Sphere { radius } => radius,
    }
}

/// A flat rectangle over `x ∈ [x0, x1]`, `z ∈ [-30, 30]`, at height `y`, cut
/// into `cells` columns of two triangles — the shape a baked surface has.
fn strip(x0: f32, x1: f32, y: f32, cells: usize) -> (Vec<Vec3>, Vec<u32>) {
    let n = cells + 1;
    let mut verts = Vec::with_capacity(2 * n);
    for j in 0..2 {
        let z = -30.0 + 60.0 * j as f32;
        for i in 0..n {
            verts.push(Vec3::new(x0 + (x1 - x0) * i as f32 / cells as f32, y, z));
        }
    }
    let mut indices = Vec::with_capacity(cells * 6);
    for i in 0..cells {
        let (a, b) = (i as u32, i as u32 + 1);
        let d = a + n as u32;
        let c = d + 1;
        indices.extend([a, d, c, a, c, b]);
    }
    (verts, indices)
}

/// A **solid** block over `x ∈ [x0, x1]`, `z ∈ [-30, 30]`, its top at `top` and
/// two metres deep — a closed soup, with the top cut into `cells` columns so it
/// carries the internal edges a bake leaves.
///
/// Solid rather than a bare sheet because the difference is not cosmetic: a
/// snap probe drops half a metre, and under a sheet it meets the *boundary* of
/// the surface it was standing on, which reads as a wall. A baked root is a
/// volume, and its lip is the ridge between a top face and a side face.
fn slab(x0: f32, x1: f32, top: f32, cells: usize) -> (Vec<Vec3>, Vec<u32>) {
    let (bot, z0, z1) = (top - 2.0, -30.0f32, 30.0f32);
    let mut verts: Vec<Vec3> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut quad = |p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3| {
        let base = verts.len() as u32;
        verts.extend([p0, p1, p2, p3]);
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    };
    for i in 0..cells {
        let xa = x0 + (x1 - x0) * i as f32 / cells as f32;
        let xb = x0 + (x1 - x0) * (i + 1) as f32 / cells as f32;
        quad(
            Vec3::new(xa, top, z1),
            Vec3::new(xb, top, z1),
            Vec3::new(xb, top, z0),
            Vec3::new(xa, top, z0),
        );
    }
    quad(
        Vec3::new(x0, bot, z0),
        Vec3::new(x1, bot, z0),
        Vec3::new(x1, bot, z1),
        Vec3::new(x0, bot, z1),
    );
    quad(
        Vec3::new(x1, bot, z1),
        Vec3::new(x1, bot, z0),
        Vec3::new(x1, top, z0),
        Vec3::new(x1, top, z1),
    );
    quad(
        Vec3::new(x0, bot, z0),
        Vec3::new(x0, bot, z1),
        Vec3::new(x0, top, z1),
        Vec3::new(x0, top, z0),
    );
    quad(
        Vec3::new(x0, bot, z1),
        Vec3::new(x1, bot, z1),
        Vec3::new(x1, top, z1),
        Vec3::new(x0, top, z1),
    );
    quad(
        Vec3::new(x1, bot, z0),
        Vec3::new(x0, bot, z0),
        Vec3::new(x0, top, z0),
        Vec3::new(x1, top, z0),
    );
    (verts, indices)
}

/// What one run over a feature did to the body, in the terms the bug reports
/// were written in.
///
/// The three velocity/teleport figures are gathered only while the body is
/// within [`DEPARTURE`] of the height it started at — the window in which it is
/// crossing the feature. Past that it is falling, and a body that *lands* on the
/// face below a cliff is meant to lose speed and be pushed out; that is the
/// landing, not the lip.
#[derive(Debug, Default, Clone, Copy)]
struct Run {
    /// Worst drop below the driven horizontal speed, m/s.
    speed_loss: f32,
    /// Worst upward velocity the solver handed back, m/s. The bump.
    kick: f32,
    /// Worst rise above the resting height, m.
    rise: f32,
    /// Ticks on which `on_floor` went false.
    airborne: u32,
    /// Largest single-tick horizontal step, m — a teleport shows up here.
    jump: f32,
    end: Vec3,
}

impl Run {
    fn worst(self, other: Run) -> Run {
        Run {
            speed_loss: self.speed_loss.max(other.speed_loss),
            kick: self.kick.max(other.kick),
            rise: self.rise.max(other.rise),
            airborne: self.airborne.max(other.airborne),
            jump: self.jump.max(other.jump),
            end: other.end,
        }
    }
}

/// Drive `body` along `+dir` at `speed`, re-asserting the drive every grounded
/// tick — the port's own loop, with gravity while airborne and the state
/// machine's input replaced by a constant.
fn drive(
    geometry: &CollisionWorld,
    body: &mut CharacterBody,
    start: Vec3,
    dir: Vec3,
    speed: f32,
    ticks: u32,
) -> Run {
    let rest = start.y;
    let mut p = start;
    let mut v = dir * speed;
    let mut run = Run::default();
    for _ in 0..ticks {
        if body.on_floor {
            v = dir * speed;
        } else {
            v.y -= GRAVITY * DT;
        }
        let before = p;
        let result = move_and_slide(geometry, body, p, v, DT);
        p = result.position;
        v = result.velocity;
        run.rise = run.rise.max(p.y - rest);
        run.airborne += u32::from(!result.on_floor);
        if p.y >= rest - DEPARTURE {
            run.speed_loss = run.speed_loss.max(speed - v.dot(dir));
            run.kick = run.kick.max(v.y);
            run.jump = run.jump.max((p - before).dot(dir).abs());
        }
    }
    run.end = p;
    run
}

/// The same drive, repeated at 32 sub-tick arrival phases, worst case kept.
///
/// A body walking at 4 m/s covers 6.7 cm a tick and the defect's band is 2.6 cm
/// wide, so a single run has about a one-in-three chance of stepping straight
/// over it — which is exactly why the joint "sometimes" bumped.
fn sweep_phases(
    geometry: &CollisionWorld,
    body: &CharacterBody,
    start: Vec3,
    dir: Vec3,
    speed: f32,
    distance: f32,
) -> Run {
    let ticks = (distance / (speed * DT)).ceil() as u32;
    let mut worst = Run::default();
    for phase in 0..32u32 {
        let mut b = *body;
        b.on_floor = true;
        let offset = dir * (speed * DT * phase as f32 / 32.0);
        worst = worst.worst(drive(geometry, &mut b, start - offset, dir, speed, ticks));
    }
    worst
}

/// How far a body may drop below the surface it left and still be *crossing*
/// the feature rather than falling away from it — see [`Run`]. A third of the
/// capsule's radius: far enough that the whole lip is inside the window, near
/// enough that nothing the body lands on later is.
const DEPARTURE: f32 = 0.12;

/// How much of a bump the tests will tolerate. A tenth of a millimetre per
/// second is float noise on a 12 m/s run; the defect measured 0.30 m/s of
/// upward kick at walking speed and 0.85 m/s rolling.
const NO_BUMP: f32 = 1.0e-4;

/// The two joints the port actually has, plus the two the level author's hand
/// produces: surfaces that overlap in plan, and surfaces a few millimetres out
/// of true. Every one of them is one flat surface as far as the player is
/// concerned.
fn joint(kind: &str) -> CollisionWorld {
    let mut level = Level::new();
    let mut geometry;
    match kind {
        // A baked ledge to x = 0, a box bridge from x = 0 — the phase bridge.
        // `lift` is the bridge out of true vertically, `gap` horizontally.
        "box|soup" | "box|soup 8mm high" | "box|soup 8mm low" | "box|soup 8mm gap"
        | "box|soup step 0.3" => {
            let (lift, gap) = match kind {
                "box|soup 8mm high" => (0.008, 0.0),
                "box|soup 8mm low" => (-0.008, 0.0),
                "box|soup 8mm gap" => (0.0, 0.008),
                "box|soup step 0.3" => (0.3, 0.0),
                _ => (0.0, 0.0),
            };
            let ledge = level.bare();
            level.aabb(
                Vec3::new(6.0 + gap, -2.0 + lift, 0.0),
                Vec3::new(6.0, 2.0, 30.0),
            );
            geometry = level.snapshot();
            let (verts, indices) = slab(-12.0, 0.0, 0.0, 12);
            let mesh = Trimesh::build_from_soup(&verts, &indices);
            push_trimesh(&mut geometry, ledge, Vec3::ZERO, &mesh, 1);
        }
        // Two boxes, which is what two placed props are.
        "box|box" => {
            level.aabb(Vec3::new(-6.0, -2.0, 0.0), Vec3::new(6.0, 2.0, 30.0));
            level.aabb(Vec3::new(6.0, -2.0, 0.0), Vec3::new(6.0, 2.0, 30.0));
            geometry = level.snapshot();
        }
        // Two baked roots, which is what two CSG roots are.
        "soup|soup" => {
            let (a, b) = (level.bare(), level.bare());
            geometry = level.snapshot();
            let (va, ia) = slab(-12.0, 0.0, 0.0, 12);
            let (vb, ib) = slab(0.0, 12.0, 0.0, 12);
            push_trimesh(&mut geometry, a, Vec3::ZERO, &Trimesh::build_from_soup(&va, &ia), 1);
            push_trimesh(&mut geometry, b, Vec3::ZERO, &Trimesh::build_from_soup(&vb, &ib), 1);
        }
        // The bridge pushed 0.4 m into the cliff, so the ledge's lip edge is
        // buried inside the box and the box's edge is under the ledge.
        "box|soup overlapped" => {
            let ledge = level.bare();
            level.aabb(Vec3::new(5.6, -2.0, 0.0), Vec3::new(6.0, 2.0, 30.0));
            geometry = level.snapshot();
            let (verts, indices) = slab(-12.0, 0.4, 0.0, 12);
            let mesh = Trimesh::build_from_soup(&verts, &indices);
            push_trimesh(&mut geometry, ledge, Vec3::ZERO, &mesh, 1);
        }
        other => panic!("no such joint: {other}"),
    }
    geometry
}

#[test]
fn crossing_a_joint_between_two_colliders_costs_no_speed_and_no_hop() {
    // `out_of_true` is the vertical mismatch the joint carries, which the body
    // is allowed to rise by. `float` is how many ticks it may spend off the
    // floor: zero everywhere the two surfaces are level, and one for a joint
    // that steps *down* by less than the tolerance — where the snap probe, a
    // static drop rather than Godot's swept cast, can find itself a few
    // centimetres inside the lower box and leave by its side face rather than
    // its top. That is the probe's own approximation (`snap_to_floor`), it
    // predates this rule, and it costs one tick of `on_floor` on an eight
    // millimetre drop.
    for (kind, out_of_true, float) in [
        ("box|soup", 0.0f32, 0u32),
        ("box|box", 0.0, 0),
        ("soup|soup", 0.0, 0),
        ("box|soup overlapped", 0.0, 0),
        ("box|soup 8mm gap", 0.0, 0),
        ("box|soup 8mm high", 0.008, 1),
        ("box|soup 8mm low", 0.008, 1),
    ] {
        let geometry = joint(kind);
        for (name, body, speed) in [
            ("walk", walker(), 4.0f32),
            ("run", walker(), 8.0),
            ("roll", roller(), 12.0),
        ] {
            for dir in [Vec3::X, Vec3::NEG_X] {
                let start = Vec3::new(-3.0 * dir.x, resting_offset(&body), 0.0);
                let run = sweep_phases(&geometry, &body, start, dir, speed, 6.0);
                assert!(
                    run.speed_loss < NO_BUMP,
                    "{kind} {name} {dir}: the joint took {} m/s of horizontal speed",
                    run.speed_loss
                );
                assert!(
                    run.kick < NO_BUMP,
                    "{kind} {name} {dir}: the joint kicked the body up at {} m/s",
                    run.kick
                );
                assert!(
                    run.rise < out_of_true + NO_BUMP,
                    "{kind} {name} {dir}: the joint lifted the body {} m",
                    run.rise
                );
                assert!(
                    run.airborne <= float,
                    "{kind} {name} {dir}: on_floor flickered off {} times \
                     crossing the joint, more than the {float} allowed",
                    run.airborne
                );
                assert!(
                    (run.end.x * dir.x) > 2.5,
                    "{kind} {name} {dir}: the run ended at x = {}, still short \
                     of the far side",
                    run.end.x
                );
            }
        }
    }
}

#[test]
fn a_wall_butted_onto_a_floor_still_stops_the_body() {
    // The false positive the seam rule has to refuse. The wall's bottom edge
    // lies exactly in the floor's plane and touches it, which is rule 2 of
    // `suppress_cross_collider_seams` satisfied — and rule 3, orientation,
    // is the only thing between "one smooth surface" and walking through a wall.
    for wall_is_soup in [false, true] {
        let mut level = Level::new();
        level.aabb(Vec3::new(0.0, -2.0, 0.0), Vec3::new(12.0, 2.0, 6.0));
        let soup_wall = level.bare();
        if !wall_is_soup {
            level.aabb(Vec3::new(5.0, 1.5, 0.0), Vec3::new(1.0, 1.5, 6.0));
        }
        let mut geometry = level.snapshot();
        if wall_is_soup {
            // A vertical face at x = 4, rising off the floor it is butted onto.
            let verts = vec![
                Vec3::new(4.0, 0.0, -6.0),
                Vec3::new(4.0, 0.0, 6.0),
                Vec3::new(4.0, 3.0, 6.0),
                Vec3::new(4.0, 3.0, -6.0),
            ];
            // Wound so the face looks back down `-x`, at the body walking into
            // it: a soup is single-sided and its winding is the level's, not
            // the test's choice.
            let indices = vec![0, 1, 2, 0, 2, 3];
            let mesh = Trimesh::build_from_soup(&verts, &indices);
            push_trimesh(&mut geometry, soup_wall, Vec3::ZERO, &mesh, 1);
        }

        let mut body = walker();
        body.on_floor = true;
        let feet = Vec3::new(0.0, resting_offset(&body), 0.0);
        let run = drive(&geometry, &mut body, feet, Vec3::X, 8.0, 120);
        assert!(
            run.end.x < 4.0,
            "the body reached x = {} — it went through the wall",
            run.end.x
        );
        assert!(
            run.speed_loss > 7.9,
            "the wall only took {} m/s: it did not stop the body",
            run.speed_loss
        );
        assert!(
            body.on_floor,
            "the body left the floor while being stopped by a wall"
        );
    }
}

#[test]
fn a_three_hundred_millimetre_rise_is_still_a_step_and_not_a_seam() {
    // The other false positive: a joint the level author *meant*. 0.3 m is
    // thirty times `SEAM_PLANE_TOLERANCE`, so the two surfaces never read as
    // one — the capsule's round bottom rides up it, as it does over any step
    // inside its radius, and the ride is what the assertions look for.
    let geometry = joint("box|soup step 0.3");
    let body = walker();
    let start = Vec3::new(-3.0, resting_offset(&body), 0.0);
    let run = sweep_phases(&geometry, &body, start, Vec3::X, 4.0, 6.0);

    // 0.3 m against a 0.35 m radius puts the contact 5.7° above horizontal:
    // a wall, and the capsule stops at it. What matters is that it is *some*
    // feature — a seam would have let the body sail across at constant height,
    // which is what the 8 mm joint above is allowed to do and this is not.
    assert!(
        run.speed_loss > 3.9,
        "the step only took {} m/s: it was flattened into a seam",
        run.speed_loss
    );
    assert!(
        run.end.x < 0.0,
        "the body reached x = {} — it walked over a 0.3 m rise",
        run.end.x
    );
    assert!(
        run.rise < 0.05,
        "the body climbed {} m of a 0.3 m step it cannot climb",
        run.rise
    );
}

/// A ledge top at `y = 0` out to `x = 0`, and a face falling away from that lip
/// at `steep` degrees from horizontal — one soup, as a baked mountain is.
fn cliff(steep: f32) -> CollisionWorld {
    let (mut verts, mut indices) = strip(-12.0, 0.0, 0.0, 12);
    let run = 12.0 / steep.to_radians().tan();
    let base = verts.len() as u32;
    for j in 0..2 {
        let z = -30.0 + 60.0 * j as f32;
        verts.push(Vec3::new(0.0, 0.0, z));
        verts.push(Vec3::new(run, -12.0, z));
    }
    indices.extend([base, base + 2, base + 3, base, base + 3, base + 1]);

    let mut level = Level::new();
    let mountain = level.bare();
    let mut geometry = level.snapshot();
    let mesh = Trimesh::build_from_soup(&verts, &indices);
    push_trimesh(&mut geometry, mountain, Vec3::ZERO, &mesh, 1);
    geometry
}

#[test]
fn walking_or_rolling_off_a_steep_cliff_detaches_cleanly() {
    for steep in [55.0f32, 70.0, 80.0, 89.0] {
        let geometry = cliff(steep);
        for (name, body, speed) in [
            ("walk", walker(), 4.0f32),
            ("run", walker(), 8.0),
            ("roll", roller(), 12.0),
        ] {
            let start = Vec3::new(-3.0, resting_offset(&body), 0.0);
            let run = sweep_phases(&geometry, &body, start, Vec3::X, speed, 3.0 + speed * 0.5);
            // No tick of lost horizontal speed: the lip must not brake.
            assert!(
                run.speed_loss < NO_BUMP,
                "{steep}° {name}: the lip took {} m/s of horizontal speed \
                 before the fall",
                run.speed_loss
            );
            // No upward kick, and no snap-back or ejection: one tick's travel is
            // `speed · dt`, and nothing may move the body further than that.
            assert!(
                run.kick < NO_BUMP,
                "{steep}° {name}: the lip kicked the body up at {} m/s",
                run.kick
            );
            assert!(
                run.rise < NO_BUMP,
                "{steep}° {name}: the lip lifted the body {} m",
                run.rise
            );
            assert!(
                run.jump < speed * DT + 1e-4,
                "{steep}° {name}: a tick moved the body {} m, more than the \
                 {} m it was travelling — the lip ejected it",
                run.jump,
                speed * DT
            );
            assert!(
                run.end.y < -1.0,
                "{steep}° {name}: the body ended at y = {} — it never fell",
                run.end.y
            );
        }
    }
}

#[test]
fn walking_along_the_top_edge_of_a_cliff_neither_falls_nor_snags() {
    // The other half of the lip: a body running *parallel* to it, one
    // centimetre inside, is on the ledge and must stay there at full speed.
    for steep in [55.0f32, 80.0, 89.0] {
        let geometry = cliff(steep);
        for (name, body, speed) in [("run", walker(), 8.0f32), ("roll", roller(), 12.0)] {
            for lane in [-0.01f32, -0.05, -0.2] {
                let mut b = body;
                b.on_floor = true;
                let start = Vec3::new(lane, resting_offset(&body), -5.0);
                let run = drive(&geometry, &mut b, start, Vec3::Z, speed, 60);
                assert!(
                    run.speed_loss < NO_BUMP && run.kick < NO_BUMP && run.rise < NO_BUMP,
                    "{steep}° {name} lane {lane}: running along the lip cost \
                     {} m/s, {} m/s of lift, {} m of rise",
                    run.speed_loss,
                    run.kick,
                    run.rise
                );
                assert_eq!(
                    run.airborne, 0,
                    "{steep}° {name} lane {lane}: the body left the ledge \
                     running along it"
                );
                assert!(
                    (run.end.x - lane).abs() < 1e-4,
                    "{steep}° {name} lane {lane}: the lip pushed the body to \
                     x = {}",
                    run.end.x
                );
            }
        }
    }
}

#[test]
fn floor_snap_still_holds_a_body_to_every_slope_it_may_stand_on() {
    // The fix must not have bought the cliff by disarming the snap. A
    // triangulated ramp at each angle inside `max_floor_angle`, walked
    // downhill: grounded every tick, which is the whole point of `snap_length`.
    for degrees in [5.0f32, 15.0, 25.0, 35.0, 44.0] {
        let pitch = Quat::from_rotation_x(degrees.to_radians());
        let (verts, indices) = grid_soup(12.0, 12, |_, _| 0.0);
        let pitched: Vec<Vec3> = verts.iter().map(|v| pitch * *v).collect();
        let mut level = Level::new();
        let ramp = level.bare();
        let mut geometry = level.snapshot();
        let mesh = Trimesh::build_from_soup(&pitched, &indices);
        push_trimesh(&mut geometry, ramp, Vec3::ZERO, &mesh, 1);

        // Positive pitch drops the +Z edge, so +Z is downhill.
        let mut body = walker();
        body.on_floor = true;
        let lift = Vec3::Y * resting_offset(&body);
        let start = pitch * Vec3::new(0.0, 0.0, -6.0) + lift;
        let run = drive(&geometry, &mut body, start, Vec3::Z, 6.0, 60);
        assert_eq!(
            run.airborne, 0,
            "{degrees}°: the body launched off the ramp on {} of 60 ticks \
             despite the snap",
            run.airborne
        );
        assert!(
            run.end.z - start.z > 5.5,
            "{degrees}°: the run only covered {} m of the 6 m it was driven",
            run.end.z - start.z
        );
    }
}

// ---------------------------------------------------------------------------
// Point containment
// ---------------------------------------------------------------------------
//
// `CollisionWorld::contains_point` is the query the overlaps cannot express.
// An overlap is a *surface* test — which feature of the collider is the shape
// penetrating — and a [`Trimesh`]'s features are its triangles, so a shape
// swallowed whole by a closed soup overlaps nothing at all and reads exactly
// like open air. The convex shapes have no such hole, which is why the first
// caller to notice was one that had only ever been pointed at an OBB.
//
// So the load-bearing test below is `a_point_deep_inside_a_trimesh_box_is_inside_it`:
// it asserts *both* halves, the empty overlap and the answered containment, on
// the same geometry.

/// A **closed** box as a triangle soup — twelve triangles, wound outward, which
/// is what a CSG bake emits for a solid and what an `overlap_*` call can only
/// see the skin of.
///
/// The winding is exact rather than incidental: the +X face's two triangles
/// share the diagonal `(+,−,−) → (+,+,+)`, whose midpoint is the centre of the
/// face — so the +X ray from the box's own centre lands exactly on a shared
/// edge, and the same is true on the other two axes. That is the parity trap
/// [`runt_core::collide::CONTAINMENT_MERGE`] exists for, and it is the *default*
/// configuration for axis-aligned geometry rather than an unlucky one.
fn box_soup(half: Vec3) -> (Vec<Vec3>, Vec<u32>) {
    let v = |x: f32, y: f32, z: f32| Vec3::new(x * half.x, y * half.y, z * half.z);
    let verts = vec![
        v(-1.0, -1.0, -1.0), // 0
        v(1.0, -1.0, -1.0),  // 1
        v(1.0, -1.0, 1.0),   // 2
        v(-1.0, -1.0, 1.0),  // 3
        v(-1.0, 1.0, -1.0),  // 4
        v(1.0, 1.0, -1.0),   // 5
        v(1.0, 1.0, 1.0),    // 6
        v(-1.0, 1.0, 1.0),   // 7
    ];
    #[rustfmt::skip]
    let indices = vec![
        0, 1, 2,  0, 2, 3, // −Y
        4, 6, 5,  4, 7, 6, // +Y
        0, 4, 5,  0, 5, 1, // −Z
        3, 2, 6,  3, 6, 7, // +Z
        0, 3, 7,  0, 7, 4, // −X
        1, 5, 6,  1, 6, 2, // +X
    ];
    (verts, indices)
}

/// An **L-shaped** prism: the 6-gon `(0,0) (4,0) (4,1.5) (1.5,1.5) (1.5,4)
/// (0,4)` in XY, extruded to `z = ±2`. One closed surface, no coincident faces,
/// and one reentrant corner — so the square `x > 1.5, y > 1.5` is inside the
/// shape's bounding box and outside the shape.
///
/// The caps are a fan from the first vertex, which for this polygon is a valid
/// triangulation and adds no vertex the side quads do not also have: a cap
/// triangulated with extra points would leave a T-junction, and a crack in a
/// closed surface is exactly what a parity count cannot survive.
fn l_prism() -> (Vec<Vec3>, Vec<u32>) {
    const POLY: [[f32; 2]; 6] = [
        [0.0, 0.0],
        [4.0, 0.0],
        [4.0, 1.5],
        [1.5, 1.5],
        [1.5, 4.0],
        [0.0, 4.0],
    ];
    const DEPTH: f32 = 2.0;
    let n = POLY.len() as u32;
    let mut verts = Vec::with_capacity(POLY.len() * 2);
    for p in POLY {
        verts.push(Vec3::new(p[0], p[1], DEPTH));
    }
    for p in POLY {
        verts.push(Vec3::new(p[0], p[1], -DEPTH));
    }

    let mut indices = Vec::new();
    for i in 1..n - 1 {
        // Front cap, +Z; back cap, −Z, the same fan reversed.
        indices.extend([0, i, i + 1]);
        indices.extend([n, n + i + 1, n + i]);
    }
    for i in 0..n {
        let j = (i + 1) % n;
        // Outward, for a polygon wound counter-clockwise seen from +Z.
        indices.extend([i, i + n, j + n]);
        indices.extend([i, j + n, j]);
    }
    (verts, indices)
}

#[test]
fn a_point_inside_a_convex_collider_is_the_collider_it_is_inside() {
    let mut level = Level::new();
    let sphere = level.sphere(Vec3::new(0.0, 2.0, 0.0), 1.5);
    let geometry = level.snapshot();

    assert_eq!(
        geometry.contains_point(Vec3::new(0.0, 2.0, 0.0), ALL_LAYERS),
        Some(sphere),
        "the centre of a sphere is inside it"
    );
    // Inclusive on the shell, and outside a millimetre past it.
    assert_eq!(
        geometry.contains_point(Vec3::new(0.0, 3.5, 0.0), ALL_LAYERS),
        Some(sphere)
    );
    assert_eq!(
        geometry.contains_point(Vec3::new(0.0, 3.501, 0.0), ALL_LAYERS),
        None
    );

    let mut level = Level::new();
    let floor = with_floor(&mut level);
    let geometry = level.snapshot();
    assert_eq!(
        geometry.contains_point(Vec3::new(4.0, -0.5, -7.0), ALL_LAYERS),
        Some(floor),
        "a point buried in the floor slab is inside it"
    );
    assert_eq!(
        geometry.contains_point(Vec3::new(4.0, 0.001, -7.0), ALL_LAYERS),
        None,
        "…and a millimetre above its top face is not"
    );
    assert_eq!(
        geometry.contains_point(Vec3::new(26.0, -0.5, 0.0), ALL_LAYERS),
        None,
        "…nor is a point beyond its edge at the same depth"
    );
}

#[test]
fn a_rotated_box_contains_the_points_its_own_frame_says_it_does() {
    // PORT_SPEC's PhaseWall: 0.5 m thick, yawed −98°. A slab that thin is the
    // shape a world-axis test would get wrong — 0.2 m along the wall's own
    // normal is inside it, and 0.2 m along world X is 8° off that and outside.
    let (_, geometry, wall) = phase_level();
    let centre = Vec3::new(4.5556774, 2.0, 14.4808);
    let normal = phase_wall_normal();

    assert_eq!(geometry.contains_point(centre, ALL_LAYERS), Some(wall));
    assert_eq!(
        geometry.contains_point(centre + normal * 0.2, ALL_LAYERS),
        Some(wall),
        "0.2 m along the wall's own normal is inside a 0.25 m half-thickness"
    );
    assert_eq!(
        geometry.contains_point(centre + normal * 0.3, ALL_LAYERS),
        None,
        "…and 0.3 m along it is out the other side"
    );
    // Along the wall's length instead: 3.9 m of the 4 m half-length, still in.
    let along = Quat::from_rotation_y(-98f32.to_radians()) * Vec3::Z;
    assert_eq!(
        geometry.contains_point(centre + along * 3.9, ALL_LAYERS),
        Some(wall)
    );
    assert_eq!(
        geometry.contains_point(centre + along * 4.1, ALL_LAYERS),
        None
    );
}

#[test]
fn a_point_deep_inside_a_trimesh_box_is_inside_it() {
    // The bug this query exists for, stated as the two halves of one geometry:
    // an 8 m cube as a closed soup, and a standing capsule at the middle of it.
    let mut level = Level::new();
    let block = level.bare();
    let mut geometry = level.snapshot();
    let (verts, indices) = box_soup(Vec3::splat(4.0));
    let mesh = Trimesh::build_from_soup(&verts, &indices);
    push_trimesh(&mut geometry, block, Vec3::ZERO, &mesh, 1);

    // Half of it: the surface query sees nothing at all. The capsule's segment
    // reaches 1.0 m from its centre and the nearest triangle is 4 m away, so
    // there is no contact to report — which is indistinguishable, to every
    // caller of `overlap_body`, from standing in an empty field.
    let body = standing();
    assert_eq!(
        geometry.overlap_body(&body, Vec3::ZERO),
        Vec::new(),
        "a capsule swallowed by a soup overlaps none of its triangles"
    );
    // …and at the top face, where it straddles the surface, the same query
    // answers immediately. The guard built on it therefore works within about a
    // capsule radius of a surface and nowhere else.
    let straddling = geometry.overlap_body(&body, Vec3::new(0.0, 4.0, 0.0));
    assert_eq!(straddling.len(), 1, "at the face: {straddling:?}");
    assert_eq!(straddling[0].entity, block);
    // A segment running *through* a triangle is zero distance from it, so the
    // depth saturates at the capsule's radius however far past the face the
    // body has gone. That is the shape of the hole in one number: "how deep am
    // I in it" stops meaning anything the moment the surface is behind you.
    assert!((straddling[0].depth - 0.35).abs() < 1e-3, "{straddling:?}");

    // The other half: containment answers everywhere inside the body.
    assert_eq!(
        geometry.contains_point(Vec3::ZERO, ALL_LAYERS),
        Some(block),
        "the centre of a closed soup is inside it"
    );
    for point in [
        Vec3::new(3.9, 3.9, 3.9),
        Vec3::new(-3.9, 0.0, 2.0),
        Vec3::new(0.0, -3.9, 0.0),
        Vec3::new(1.75, -2.5, -3.25),
    ] {
        assert_eq!(
            geometry.contains_point(point, ALL_LAYERS),
            Some(block),
            "{point} is inside the box"
        );
    }
    for point in [
        // Just outside a face, on the axis a ray leaves along.
        Vec3::new(0.0, 4.001, 0.0),
        Vec3::new(4.001, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -4.001),
        // Well outside, on the far side of a face — the "through the wall"
        // case, where a lone ray back through the box crosses twice.
        Vec3::new(0.0, 12.0, 0.0),
        Vec3::new(-9.0, 0.0, 0.0),
        // Outside a corner, where all three rays miss.
        Vec3::new(6.0, 6.0, 6.0),
    ] {
        assert_eq!(
            geometry.contains_point(point, ALL_LAYERS),
            None,
            "{point} is outside the box"
        );
    }
}

#[test]
fn the_notch_of_a_concave_soup_is_outside_it() {
    // The same claim the game's recharge polygon test makes, one dimension up:
    // a concave body's bounding box is not the body. The L's reentrant square
    // is `x > 1.5, y > 1.5`, and a parity count is the reason it can be told
    // apart from the arm beside it.
    let mut level = Level::new();
    let block = level.bare();
    let mut geometry = level.snapshot();
    let (verts, indices) = l_prism();
    let mesh = Trimesh::build_from_soup(&verts, &indices);
    push_trimesh(&mut geometry, block, Vec3::ZERO, &mesh, 1);

    for point in [
        Vec3::new(0.75, 0.75, 0.0),  // the corner both arms share
        Vec3::new(3.0, 0.75, 1.9),   // the horizontal arm, near the +Z cap
        Vec3::new(0.75, 3.0, -1.9),  // the vertical arm, near the −Z cap
    ] {
        assert_eq!(
            geometry.contains_point(point, ALL_LAYERS),
            Some(block),
            "{point} is in the solid"
        );
    }
    for point in [
        Vec3::new(3.0, 3.0, 0.0),   // the notch, dead centre
        Vec3::new(1.6, 1.6, 0.0),   // the notch, a hair off the reentrant corner
        Vec3::new(3.9, 3.9, 1.9),   // the notch's far corner, inside the bounds
        Vec3::new(2.0, 2.0, 2.1),   // past the +Z cap, above the arm
    ] {
        assert_eq!(
            geometry.contains_point(point, ALL_LAYERS),
            None,
            "{point} is in the notch, not the solid"
        );
    }
}

#[test]
fn an_open_surface_contains_nothing_even_from_underneath() {
    // A `grid_soup` is a *sheet*: it has no inside, and the honest answer under
    // it is "outside" rather than "inside the half-space below". One axis says
    // otherwise — the +Y ray crosses the sheet exactly once — and the other two
    // outvote it, which is what the three-axis rule buys beyond the shared-edge
    // case it was written for.
    let mut level = Level::new();
    let sheet = level.bare();
    let mut geometry = level.snapshot();
    let (verts, indices) = grid_soup(6.0, 8, bumpy);
    let mesh = Trimesh::build_from_soup(&verts, &indices);
    push_trimesh(&mut geometry, sheet, Vec3::ZERO, &mesh, 1);

    for y in [-4.0f32, -0.5, 0.5, 4.0] {
        for (x, z) in [(0.0f32, 0.0f32), (2.5, -1.5), (-3.0, 3.0)] {
            assert_eq!(
                geometry.contains_point(Vec3::new(x, y, z), ALL_LAYERS),
                None,
                "an open sheet contains nothing, and ({x}, {y}, {z}) is nothing"
            );
        }
    }
}

#[test]
fn a_capsule_resting_on_a_trimesh_floor_has_its_whole_segment_outside_it() {
    // The property the phase guard's tolerance depends on, restated for the new
    // query: the medial segment `CharacterBody::segment` hands out is a full
    // radius inside the capsule, so a body standing *on* a surface never has a
    // segment point inside it — no matter how deep into the contact margin the
    // solver has let it settle.
    let mut level = Level::new();
    let ground = level.bare();
    let mut geometry = level.snapshot();
    let (verts, indices) = box_soup(Vec3::new(12.0, 2.0, 12.0));
    let mesh = Trimesh::build_from_soup(&verts, &indices);
    // Top face at y = 0.
    push_trimesh(&mut geometry, ground, Vec3::new(0.0, -2.0, 0.0), &mesh, 1);

    let (body, position) = drop_onto(&geometry, Vec3::new(1.0, 6.0, -2.0), 180);
    assert!(body.on_floor, "the capsule never landed");
    let (lo, hi) = body.segment(position);
    for point in [lo, (lo + hi) * 0.5, hi] {
        assert_eq!(
            geometry.contains_point(point, ALL_LAYERS),
            None,
            "{point} of a resting capsule reads as inside the floor"
        );
    }
    // And the feet themselves — the sole, a radius below `lo` — are the point
    // that *would* be ambiguous: it sits on the face, and the query is
    // inclusive there. Which is the whole reason the caller samples the
    // segment.
    let sole = position - Vec3::Y * 1.0;
    assert!(sole.y.abs() < 2e-3, "the capsule settled at {}", sole.y);
}

#[test]
fn the_mask_hides_a_collider_from_the_containment_query() {
    let mut level = Level::new();
    let block = level.bare();
    let mut geometry = level.snapshot();
    let (verts, indices) = box_soup(Vec3::splat(4.0));
    let mesh = Trimesh::build_from_soup(&verts, &indices);
    // Layer 1 alone, as `phase_level` tags its wall.
    push_trimesh(&mut geometry, block, Vec3::ZERO, &mesh, 1 << 1);

    assert_eq!(geometry.contains_point(Vec3::ZERO, ALL_LAYERS), Some(block));
    assert_eq!(geometry.contains_point(Vec3::ZERO, 1 << 1), Some(block));
    assert_eq!(
        geometry.contains_point(Vec3::ZERO, ALL_LAYERS & !(1 << 1)),
        None,
        "a mask that cannot see the collider cannot be inside it"
    );
    assert_eq!(geometry.contains_point(Vec3::ZERO, 0), None);
}

#[test]
fn overlapping_colliders_answer_with_the_lowest_entity_every_time() {
    // DESIGN §3's tie-break, which for a query that answers with one entity out
    // of several candidates is the whole of its determinism. "Lowest" is
    // `Entity`'s own `Ord` — the key `CollisionWorld` sorts its snapshot by —
    // and not spawn order, which this bevy's inverted entity index makes the
    // reverse of it.
    let point = Vec3::new(0.5, 0.0, 0.0);
    let mut level = Level::new();
    let big = level.aabb(Vec3::ZERO, Vec3::splat(2.0));
    let offset = level.aabb(Vec3::new(1.0, 0.0, 0.0), Vec3::splat(2.0));
    let ball = level.sphere(point, 2.0);
    let geometry = level.snapshot();
    let lowest = [big, offset, ball].into_iter().min().expect("three");
    for _ in 0..8 {
        assert_eq!(
            geometry.contains_point(point, ALL_LAYERS),
            Some(lowest),
            "three colliders contain the point; the lowest entity wins, and \
             wins again on every repeat of the same query"
        );
    }

    // Take that winner out of the mask and the next one up answers.
    let mut level = Level::new();
    let a = level.aabb(Vec3::ZERO, Vec3::splat(2.0));
    let b = level.aabb(Vec3::new(1.0, 0.0, 0.0), Vec3::splat(2.0));
    let (low, high) = if a < b { (a, b) } else { (b, a) };
    level.layers(low, CollisionLayers::layer(1));
    let geometry = level.snapshot();
    assert_eq!(geometry.contains_point(point, ALL_LAYERS), Some(low));
    assert_eq!(
        geometry.contains_point(point, ALL_LAYERS & !(1 << 1)),
        Some(high),
        "masking the lower entity out hands the answer to the higher one"
    );
}

// ---------------------------------------------------------------------------
// contains_point_rejecting — the classification hole in the tie rule
// ---------------------------------------------------------------------------
//
// `overlapping_colliders_answer_with_the_lowest_entity_every_time` above
// proves the tie rule is right for "which one collider is this point in."
// `contains_point_rejecting`'s whole reason to exist is that the rule is the
// wrong one for "is this point in anything I have to treat as solid" the
// moment a collider can answer `mask` for one reason and still be something
// the caller wants to ignore — pulling the answering bit out of `mask` cannot
// help, because the collider is not on `mask` *because of* that bit, it is on
// `mask` for an unrelated one it also carries. `reject` is checked against the
// collider's own memberships instead, so it removes exactly the candidates it
// means to and nothing hides behind an accident of `Entity` order.

#[test]
fn a_rejected_lower_entity_no_longer_hides_a_real_one_behind_it() {
    // Two colliders, both containing `point`, both answering the same query
    // bit — `PHASE_BIT` — the way `rocks::ROCK_MEMBERSHIPS` answers
    // `PHASED_MASK` in `3dimenshift-runt` whether or not the rock is also
    // tagged holdable. `holdable` additionally carries `REJECT_BIT`, standing
    // in for `layers::HOLDABLE`; `solid` carries only `PHASE_BIT`, standing in
    // for a genuinely solid phaseable body.
    const PHASE_BIT: u16 = 1 << 1;
    const REJECT_BIT: u16 = 1 << 5;
    let point = Vec3::ZERO;
    let mut level = Level::new();
    let a = level.aabb(point, Vec3::splat(4.0));
    let b = level.aabb(point, Vec3::splat(4.0));
    // This bevy's entity index does not preserve spawn order (see
    // `overlapping_colliders_answer_with_the_lowest_entity_every_time`), so
    // the regression needs the *actual* lower `Entity` picked out after the
    // fact rather than assumed from spawn order — it is `holdable` that has
    // to be the one the old tie rule would have committed to and stopped on.
    let (holdable, solid) = if a < b { (a, b) } else { (b, a) };
    level.layers(
        holdable,
        CollisionLayers::DEFAULT.with_memberships(PHASE_BIT | REJECT_BIT),
    );
    level.layers(solid, CollisionLayers::DEFAULT.with_memberships(PHASE_BIT));
    let geometry = level.snapshot();

    // Baseline: plain `contains_point` — `reject = 0` — answers the lower
    // entity, same as the tie-rule test above. This is the behaviour the bug
    // relied on.
    assert_eq!(geometry.contains_point(point, PHASE_BIT), Some(holdable));

    // The naive fix does not work: `holdable` is on `mask` through `PHASE_BIT`,
    // not through `REJECT_BIT`, so subtracting `REJECT_BIT` from the query
    // mask changes nothing about which colliders `mask_accepts` sees.
    assert_eq!(
        geometry.contains_point(point, PHASE_BIT & !REJECT_BIT),
        Some(holdable),
        "masking the reject bit out of the query mask cannot exclude a \
         collider that answers the query on a different bit entirely"
    );

    // The regression this whole thing is about: with the old one-`Entity`
    // behaviour (no `reject` parameter at all) this could not even be asked;
    // `contains_point_rejecting` asks it by checking `reject` against each
    // collider's own memberships, inside the scan, before either the surface
    // test or the tie rule ever runs — so a rejected collider is skipped
    // outright and the real one behind it wins. This is also
    // "rejection wins when a collider matches both masks": `holdable` matches
    // `mask` (`PHASE_BIT`) and `reject` (`REJECT_BIT`) at once, and losing is
    // what that match costs it.
    assert_eq!(
        geometry.contains_point_rejecting(point, PHASE_BIT, REJECT_BIT),
        Some(solid),
        "a rejected lower-entity collider must not hide a real one behind it"
    );
}

#[test]
fn contains_point_rejecting_at_reject_zero_is_contains_point() {
    // `reject = 0` matches nothing (`mask_accepts` is `query & memberships`,
    // and `0 & anything == 0`), so every collider that would answer
    // `contains_point` answers `contains_point_rejecting` too, entity for
    // entity — not merely "usually agrees," the same tie-broken answer on
    // every query the existing suite already runs through `contains_point`.
    let point = Vec3::new(0.5, 0.0, 0.0);
    let mut level = Level::new();
    level.aabb(Vec3::ZERO, Vec3::splat(2.0));
    level.aabb(Vec3::new(1.0, 0.0, 0.0), Vec3::splat(2.0));
    level.sphere(point, 2.0);
    let geometry = level.snapshot();
    assert_eq!(
        geometry.contains_point_rejecting(point, ALL_LAYERS, 0),
        geometry.contains_point(point, ALL_LAYERS),
    );

    // And the empty case: nothing here at all, still `None` either way.
    let empty = CollisionWorld::default();
    assert_eq!(empty.contains_point_rejecting(point, ALL_LAYERS, 0), None);
    assert_eq!(empty.contains_point(point, ALL_LAYERS), None);
}

#[test]
fn a_point_inside_only_a_rejected_collider_is_none() {
    let point = Vec3::ZERO;
    let mut level = Level::new();
    let block = level.aabb(point, Vec3::splat(4.0));
    level.layers(block, CollisionLayers::layer(1));
    let geometry = level.snapshot();
    assert_eq!(
        geometry.contains_point_rejecting(point, ALL_LAYERS, 1 << 1),
        None,
        "the only collider containing the point is exactly the one rejected"
    );
    // The mask-only exclusion agrees here, because this collider's one
    // membership bit really is the bit being excluded — the case that made
    // the naive fix look like it worked before `a_rejected_lower_entity_…`
    // showed where it stops.
    assert_eq!(geometry.contains_point(point, ALL_LAYERS & !(1 << 1)), None);
}

#[test]
fn rejection_reaches_the_terrain_field_the_same_way_it_reaches_a_collider() {
    // The two halves of `contains_point_rejecting`'s scan — the collider loop
    // and the terrain-field loop — have to agree on `reject`, or a body could
    // be un-holdable on the ground and still holdable off it (or the other
    // way round) purely because of which loop happened to notice it first.
    let mut sim = terrain_sim(1.0);
    let patch = CollisionWorld::from_world(sim.world_mut()).terrain()[0];
    let (x, z) = (0.0, 0.0);
    let h = patch.surface.height_world(patch.origin, x, z);
    let under = Vec3::new(x, h - 1.0, z);

    const REJECT_BIT: u16 = 1 << 5;

    // A block at the same point, disjoint `Entity` order from the terrain
    // patch either way — `reject` has to win regardless of which one the tie
    // rule would otherwise have preferred.
    let block = sim
        .world_mut()
        .spawn((
            Transform::from_translation(under),
            AabbCollider {
                half_extents: Vec3::splat(4.0),
            },
        ))
        .id();

    // Reject the terrain: only the block is left, whichever `Entity` order
    // they were in.
    sim.world_mut()
        .entity_mut(patch.entity)
        .insert(CollisionLayers::DEFAULT.with_memberships(1 | REJECT_BIT));
    let geometry = CollisionWorld::from_world(sim.world_mut());
    assert_eq!(
        geometry.contains_point_rejecting(under, ALL_LAYERS, REJECT_BIT),
        Some(block),
        "a rejected terrain patch must not out-rank a real collider"
    );

    // Reject the block instead: the terrain is what is left standing.
    sim.world_mut()
        .entity_mut(patch.entity)
        .insert(CollisionLayers::DEFAULT);
    sim.world_mut()
        .entity_mut(block)
        .insert(CollisionLayers::DEFAULT.with_memberships(1 | REJECT_BIT));
    let geometry = CollisionWorld::from_world(sim.world_mut());
    assert_eq!(
        geometry.contains_point_rejecting(under, ALL_LAYERS, REJECT_BIT),
        Some(patch.entity),
        "a rejected collider must not out-rank the terrain field behind it"
    );

    // And rejecting both leaves nothing.
    sim.world_mut()
        .entity_mut(patch.entity)
        .insert(CollisionLayers::DEFAULT.with_memberships(1 | REJECT_BIT));
    let geometry = CollisionWorld::from_world(sim.world_mut());
    assert_eq!(
        geometry.contains_point_rejecting(under, ALL_LAYERS, REJECT_BIT),
        None
    );
}

#[test]
fn a_point_below_the_analytic_field_is_inside_the_terrain() {
    // The height field has no volume to test against, so containment is the
    // question the field itself answers: below the surface at that XZ, and
    // inside the patch's footprint. Sampled, never tessellated — so the two
    // quality tiers agree, exactly as the raycast does.
    let mut coarse = terrain_sim(0.3);
    let mut fine = terrain_sim(1.0);
    let a = CollisionWorld::from_world(coarse.world_mut());
    let b = CollisionWorld::from_world(fine.world_mut());
    let patch = a.terrain()[0];

    let mut inside = 0;
    for i in -5..=5 {
        for j in -5..=5 {
            let (x, z) = (i as f32 * 3.0, j as f32 * 3.0);
            let h = patch.surface.height_world(patch.origin, x, z);
            let under = Vec3::new(x, h - 1.0, z);
            let over = Vec3::new(x, h + 1.0, z);
            assert_eq!(a.contains_point(under, ALL_LAYERS), Some(patch.entity));
            assert_eq!(a.contains_point(over, ALL_LAYERS), None);
            assert_eq!(
                a.contains_point(under, ALL_LAYERS),
                b.contains_point(under, ALL_LAYERS),
                "the tiers disagree under ({x}, {z})"
            );
            inside += 1;
        }
    }
    assert_eq!(inside, 121);

    // Outside the 80 × 80 patch, at any depth.
    assert_eq!(
        a.contains_point(Vec3::new(60.0, -50.0, 0.0), ALL_LAYERS),
        None
    );
    // …and the mask hides it like any other collider.
    assert_eq!(
        a.contains_point(Vec3::new(0.0, -20.0, 0.0), ALL_LAYERS & !1),
        None
    );
}
