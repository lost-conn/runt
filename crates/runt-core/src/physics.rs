//! Hand-rolled kinematic physics (DESIGN §9).
//!
//! > *no physics crate. Collision and motion are ordinary `FixedSim` systems in
//! > `runt-core`.* — DESIGN §9
//!
//! Three pieces, chained in that order inside the tick:
//!
//! ```text
//! integrate_balls   input + gravity → semi-implicit Euler → terrain contact
//! resolve_overlaps  sphere↔sphere / sphere↔AABB → OverlapEvent + push-out
//! roll_spin         cosmetic rotation derived from velocity
//! ```
//!
//! ## Terrain is analytic, not mesh
//!
//! [`integrate_balls`] samples [`TerrainSurface::sample_world`] — one evaluation
//! for height *and* gradient — and never looks at a triangle. That is the whole
//! §9 property: the mesh is a *view* of `h(x, z)`, so `Quality(0.3)` and
//! `Quality(1.0)` produce different geometry and **bit-identical** trajectories.
//! `tests/physics.rs` asserts exactly that.
//!
//! ## Rate independence
//!
//! Every decay in here is written as `exp(-rate · dt)`, never as a per-tick
//! constant. `exp(-r·dt)` composed `n` times is `exp(-r·n·dt)`, so a 30 Hz sim
//! and a 60 Hz sim reach the same speed at the same *wall time* rather than after
//! the same number of ticks. `rolling_friction` and `air_damping` are therefore
//! **rates in 1/s**, not fractions — the same convention
//! [`FollowCamera::approach`](crate::camera::FollowCamera::approach) already uses.
//! (`powf(k, dt·60)` is the same function with a worse-behaved base; `exp` says
//! what it means and does not hide a magic 60 in the units.)
//!
//! ## Two thresholds, and why they exist
//!
//! Pure exponential decay never actually reaches zero, and a contact solve that
//! reflects every approach velocity buzzes forever against gravity. So:
//!
//! - [`BOUNCE_SPEED`] — normal impact speed below this is *absorbed* (the ball
//!   comes to rest on the surface) instead of reflected. It must exceed one
//!   tick's worth of gravity, `gravity · dt`, or a resting ball would bounce on
//!   its own accumulated fall: 20·(1/60) = 0.33 and 20·(1/30) = 0.67 are both
//!   comfortably under the 1.0 default.
//! - [`REST_SPEED`] — tangential speed below this snaps to exactly zero on
//!   contact. This is what makes a settled ball *stay* settled to the bit, and it
//!   doubles as emergent static friction: a slope whose one-tick tangential
//!   impulse `gravity·dt·|∇h|/(1+|∇h|²)` falls below it will not start the ball
//!   moving. At the defaults that is a slope under ~1.7° at 60 Hz (~0.9° at
//!   30 Hz) — small enough to read as "flat ground holds a marble", and the one
//!   place where tick rate is visible in the *feel* rather than the trajectory.
//!
//! ## What §9 says we are not doing
//!
//! No dynamic-dynamic response, no stacking, no joints, no mesh colliders. A
//! solid overlap moves the ball and *only* the ball; the other body never learns
//! it was hit. Overlap events are still emitted for solids (the game will want
//! them for impact sounds) — [`Trigger`] only suppresses the push-out.

use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};

use crate::camera::Camera;
use crate::ecs::{FixedTick, TerrainSurface, Transform};
use crate::input::{Input, Key};

// ---------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------

/// Default ball radius, in world units.
pub const BALL_RADIUS: f32 = 0.5;
/// Downward acceleration, m/s². Well above 9.81: marble games read as sluggish
/// at real gravity because the ball is hand-sized and the camera is not.
pub const BALL_GRAVITY: f32 = 20.0;
/// Tangential decay **rate** while touching ground, 1/s. ~1.2 lets a fast roll
/// coast for several seconds without feeling frictionless.
pub const BALL_ROLLING_FRICTION: f32 = 1.2;
/// Whole-velocity decay **rate**, 1/s, applied airborne and grounded alike.
/// Small on purpose: it is a terminal-velocity guard, not a brake.
pub const BALL_AIR_DAMPING: f32 = 0.1;
/// Fraction of normal speed returned by a bounce.
pub const BALL_RESTITUTION: f32 = 0.35;
/// Speed clamp, m/s. Caps how far one tick can move the ball, which is what
/// keeps the discrete contact test from being tunnelled through.
pub const BALL_MAX_SPEED: f32 = 25.0;
/// Default input acceleration, m/s². Comfortably above the tangential gravity of
/// a moderate slope, so the ball can climb.
pub const BALL_ACCEL: f32 = 22.0;

/// Normal impact speed below which a contact is absorbed rather than reflected.
/// See the module docs — this must stay above `gravity · dt`.
pub const BOUNCE_SPEED: f32 = 1.0;

/// Tangential speed below which a grounded ball snaps to exactly at rest.
pub const REST_SPEED: f32 = 0.01;

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/// A ball: the §9 point integrator's parameters.
///
/// The radius is the *physics* radius. Nothing here reads the entity's mesh or
/// its `Transform.scale` — a ball is a point plus a radius, and the sphere you
/// see is a prop drawn at the same place.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Ball {
    pub radius: f32,
    /// Downward acceleration magnitude, m/s² (applied as −Y).
    pub gravity: f32,
    /// Tangential decay rate while grounded, 1/s. See the module docs.
    pub rolling_friction: f32,
    /// Whole-velocity decay rate, 1/s, applied every tick.
    pub air_damping: f32,
    /// Fraction of normal speed a bounce returns, `[0, 1]`.
    pub restitution: f32,
    /// Speed clamp, m/s.
    pub max_speed: f32,
}

impl Default for Ball {
    fn default() -> Ball {
        Ball {
            radius: BALL_RADIUS,
            gravity: BALL_GRAVITY,
            rolling_friction: BALL_ROLLING_FRICTION,
            air_damping: BALL_AIR_DAMPING,
            restitution: BALL_RESTITUTION,
            max_speed: BALL_MAX_SPEED,
        }
    }
}

impl Ball {
    pub fn with_radius(radius: f32) -> Ball {
        Ball {
            radius,
            ..Ball::default()
        }
    }
}

/// World-space linear velocity, m/s. Sim state; the renderer never reads it.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct Velocity(pub Vec3);

/// Contact state left behind by [`integrate_balls`], for anything that wants to
/// know whether the ball is touching ground (the roll spin does; step 6's jump
/// and impact sounds will).
///
/// **Output only.** No system in this module reads it back into the integrator,
/// so it cannot become a hidden second copy of the simulation state.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Grounded {
    pub grounded: bool,
    /// Surface normal at the last contact; `+Y` while airborne.
    pub normal: Vec3,
}

impl Default for Grounded {
    fn default() -> Grounded {
        Grounded {
            grounded: false,
            normal: Vec3::Y,
        }
    }
}

/// Drives a ball from the [`Input`] resource.
///
/// **Exactly one player ball is expected in v1.** Nothing enforces it: if a
/// scene marks several, they all receive the same input and move in lockstep,
/// which is a legible outcome rather than a silent one. Whichever is intended,
/// the input is meaningless without a camera — see [`integrate_balls`].
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct BallController {
    /// Horizontal acceleration at full stick, m/s².
    pub accel: f32,
}

impl Default for BallController {
    fn default() -> BallController {
        BallController { accel: BALL_ACCEL }
    }
}

/// Cosmetic rolling rotation (DESIGN §9: "visual spin is derived from velocity …
/// never simulated state").
///
/// [`roll_spin`] writes `Transform.rotation` and nothing else, and no physics
/// system reads `Transform.rotation` at all. Removing this marker changes what
/// the ball looks like and not one bit of where it goes; `tests/physics.rs`
/// asserts that against a full trajectory stream.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct RollSpin;

/// Sphere overlap shape, centered on the entity's `Transform.translation`.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct SphereCollider {
    pub radius: f32,
}

/// Axis-aligned box overlap shape, centered on the entity's
/// `Transform.translation`, in **world** axes.
///
/// v1 contract, same as [`TerrainSurface`]'s: collider entities are
/// translation-only. A rotated AABB is not an AABB, and quietly treating one as
/// though it were is how "the pickup has an invisible corner" bugs start — so
/// [`resolve_overlaps`] debug-asserts it instead.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct AabbCollider {
    pub half_extents: Vec3,
}

/// Marks a collider as a trigger: overlapping it still emits an
/// [`OverlapEvent`], but the ball passes straight through.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct Trigger;

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

/// One ball-vs-collider overlap, for the tick it happened on.
///
/// Emitted for **solid and trigger alike** — the game wants a bounce sound as
/// much as it wants a pickup — with `trigger` saying which. For a solid, the
/// event describes the overlap the resolver then removed; `depth` is therefore
/// the push-out distance that was applied, not a residual.
///
/// ## Lifetime (verified against bevy_ecs 0.19)
///
/// `Messages<M>` is a double buffer. [`update_overlap_messages`] calls
/// `Messages::update` — swap the buffers, clear the one that becomes the writer
/// — as the **first** system of every `FixedSim`. Consequences, in the order a
/// reader cares about them:
///
/// - A system chained **after** [`resolve_overlaps`] sees this tick's events on
///   this tick.
/// - A system chained **before** it sees them on the *next* tick (its cursor is
///   still parked behind them when the tick ends).
/// - An event survives exactly two swaps, i.e. it is readable throughout the
///   tick it was written on and throughout the following tick, then dropped.
/// - `MessageReader` carries a `Local` cursor per system, so no reader can miss
///   an event or see one twice regardless of where it sits in the chain.
///
/// Nothing in the engine reads these; they exist for game code. Outside the
/// schedule, [`Sim::overlaps`](crate::Sim::overlaps) hands back the current
/// tick's batch without disturbing anyone's cursor.
#[derive(Message, Clone, Copy, Debug, PartialEq)]
pub struct OverlapEvent {
    /// The moving entity.
    pub ball: Entity,
    /// The collider it overlapped.
    pub other: Entity,
    /// Unit contact normal, pointing **from** `other` **towards** `ball` — the
    /// direction a solid push-out moves the ball in.
    pub normal: Vec3,
    /// Penetration depth along `normal`, always `> 0`.
    pub depth: f32,
    /// `true` if `other` carries [`Trigger`], i.e. nothing was pushed out.
    pub trigger: bool,
}

/// Make a world able to carry [`OverlapEvent`]s.
///
/// [`Sim`](crate::Sim) does this at construction; a test that builds a bare
/// `World` must call it before running [`resolve_overlaps`], because
/// `MessageWriter` is `ResMut<Messages<_>>` underneath and a missing resource is
/// a panic, not a no-op.
pub fn register_messages(world: &mut World) {
    world.init_resource::<Messages<OverlapEvent>>();
}

/// `FixedSim` (head): advance the [`OverlapEvent`] double buffer by one tick.
///
/// Unconditional, rather than bevy's `message_update_system`, whose "has it
/// changed since the last run" bookkeeping would make an event's lifetime depend
/// on the traffic pattern. One swap per tick, always, is the version you can
/// state in a doc comment (see [`OverlapEvent`]) and test.
pub fn update_overlap_messages(mut messages: ResMut<Messages<OverlapEvent>>) {
    messages.update();
}

// ---------------------------------------------------------------------------
// Integrator
// ---------------------------------------------------------------------------

/// `FixedSim`: one semi-implicit Euler step per ball, then the terrain contact
/// solve (DESIGN §9).
///
/// ```text
/// a  = input·accel + (0, −gravity, 0)
/// v += a·dt                    ← semi-implicit: velocity first…
/// v *= exp(−air_damping·dt)
/// p += v·dt                    ← …then position, with the *new* velocity
/// contact solve
/// clamp |v| ≤ max_speed
/// ```
///
/// **Input is camera-relative and therefore meaningless without a camera.**
/// W/Up pushes along the camera's forward axis projected onto the XZ plane;
/// A/D and Left/Right push along its right axis. With no camera entity in the
/// world the input term is simply zero and the ball still falls, rolls and
/// bounces — a headless physics test needs no camera, a playable scene does.
///
/// The camera pose read here is the one the *previous* tick left behind
/// (`follow_camera` runs later in the chain), which is what makes the control
/// basis independent of how far the camera has swung this tick.
///
/// Slope response is not a special case: gravity is applied in world space and
/// the contact solve keeps whatever part of it lies along the surface, so a ball
/// accelerates down `−∇h` for free.
#[allow(clippy::type_complexity)] // Bevy system params read worse behind aliases.
pub fn integrate_balls(
    tick: Res<FixedTick>,
    input: Res<Input>,
    // `Without<Ball>` on both read-only queries is what lets bevy prove they
    // cannot alias the `&mut Transform` below — a terrain or a camera is never
    // also a ball.
    cameras: Query<(Entity, &Transform), (With<Camera>, Without<Ball>)>,
    terrains: Query<(Entity, &TerrainSurface, &Transform), Without<Ball>>,
    mut balls: Query<(
        Entity,
        &Ball,
        Option<&BallController>,
        &mut Transform,
        &mut Velocity,
        Option<&mut Grounded>,
    )>,
) {
    let dt = tick.dt_secs;
    if !dt.is_finite() || dt <= 0.0 {
        return;
    }

    // The lowest `Entity` wins if a scene somehow has several cameras — the same
    // arbitrary but *stable* rule `Sim::camera_entity` applies.
    let mut camera: Option<(Entity, Transform)> = None;
    for (entity, transform) in cameras.iter() {
        if camera.is_none_or(|(best, _)| entity < best) {
            camera = Some((entity, *transform));
        }
    }
    let basis = camera.and_then(|(_, pose)| camera_basis(pose));
    let drive = input_direction(&input);

    // Sorted once per tick, not per ball: DESIGN §3 says sort by `Entity` where
    // ordering matters, and with several overlapping terrains it does.
    let mut fields: Vec<(Entity, TerrainSurface, Vec3)> = terrains
        .iter()
        .map(|(entity, surface, transform)| (entity, *surface, transform.translation))
        .collect();
    fields.sort_unstable_by_key(|(entity, _, _)| *entity);

    let mut ids: Vec<Entity> = balls.iter().map(|(entity, ..)| entity).collect();
    ids.sort_unstable();

    for id in ids {
        let Ok((_, ball, controller, mut transform, mut velocity, mut grounded)) =
            balls.get_mut(id)
        else {
            continue;
        };

        // -- acceleration ---------------------------------------------------
        let mut accel = Vec3::new(0.0, -ball.gravity, 0.0);
        if let (Some(controller), Some((forward, right))) = (controller, basis) {
            accel += (forward * drive.y + right * drive.x) * controller.accel;
        }

        // -- semi-implicit Euler --------------------------------------------
        let mut v = velocity.0 + accel * dt;
        v *= decay(ball.air_damping, dt);
        let mut p = transform.translation + v * dt;

        // -- terrain contact -------------------------------------------------
        let contact = ground_under(&fields, p.x, p.z);
        let mut on_ground = false;
        let mut contact_normal = Vec3::Y;

        if let Some((height, grad)) = contact {
            if p.y - ball.radius <= height {
                let normal = runt_mesh::terrain::normal_from_gradient(grad);
                p.y = height + ball.radius;

                let into = v.dot(normal);
                let mut v_normal = Vec3::ZERO;
                if into < 0.0 {
                    // Reflect a real impact; absorb a settle. Without the
                    // threshold, gravity's per-tick contribution alone would
                    // keep a resting ball twitching forever.
                    if -into > BOUNCE_SPEED {
                        v_normal = normal * (-into * ball.restitution);
                    }
                } else {
                    // Already separating (a push-out from last tick, say): leave
                    // it alone rather than inventing an impulse.
                    v_normal = normal * into;
                }

                let mut v_tangent = v - normal * into;
                v_tangent *= decay(ball.rolling_friction, dt);
                if v_tangent.length_squared() < REST_SPEED * REST_SPEED {
                    v_tangent = Vec3::ZERO;
                }

                v = v_tangent + v_normal;
                on_ground = true;
                contact_normal = normal;
            }
        }
        // No terrain under it: free fall. A kill plane is game logic (step 6),
        // not physics.

        // -- clamp ------------------------------------------------------------
        let speed_sq = v.length_squared();
        if ball.max_speed > 0.0 && speed_sq > ball.max_speed * ball.max_speed {
            v *= ball.max_speed / speed_sq.sqrt();
        }

        transform.translation = p;
        velocity.0 = v;
        if let Some(grounded) = grounded.as_mut() {
            **grounded = Grounded {
                grounded: on_ground,
                normal: contact_normal,
            };
        }
    }
}

/// `exp(−rate · dt)`: the tick-rate-independent survival factor for a decay of
/// `rate` per second. `n` steps of `dt` compose to `exp(−rate·n·dt)` exactly, so
/// wall time is what governs, not tick count.
#[inline]
pub fn decay(rate: f32, dt: f32) -> f32 {
    if rate <= 0.0 {
        return 1.0;
    }
    (-rate * dt).exp()
}

/// The surface under `(x, z)`: `(height, gradient)`, or `None` outside every
/// terrain patch.
///
/// With overlapping patches the **highest** surface wins — a ball rests on the
/// top one — and an exact tie goes to the lowest `Entity`, which is why `fields`
/// arrives sorted.
fn ground_under(fields: &[(Entity, TerrainSurface, Vec3)], x: f32, z: f32) -> Option<(f32, glam::Vec2)> {
    let mut best: Option<(f32, glam::Vec2)> = None;
    for (_, surface, origin) in fields {
        if !surface.contains_world(*origin, x, z) {
            continue;
        }
        let sample = surface.sample_world(*origin, x, z);
        if best.is_none_or(|(h, _)| sample.0 > h) {
            best = Some(sample);
        }
    }
    best
}

/// The XZ control basis `(forward, right)` for a camera pose, or `None` if the
/// pose leaves nothing usable on the horizontal plane.
pub fn camera_basis(pose: Transform) -> Option<(Vec3, Vec3)> {
    // Local −Z is the camera's view direction (Transform::looking_at builds it).
    let forward = flatten(pose.rotation * Vec3::NEG_Z)
        // Looking straight down or straight up leaves nothing on the XZ plane;
        // the camera's own up axis is then the meaningful horizontal direction.
        .or_else(|| flatten(pose.rotation * Vec3::Y))?;
    Some((forward, Vec3::new(-forward.z, 0.0, forward.x)))
}

/// Project onto the XZ plane and normalize, or `None` if nothing survives.
#[inline]
fn flatten(v: Vec3) -> Option<Vec3> {
    Vec3::new(v.x, 0.0, v.z).try_normalize()
}

/// WASD/arrows as `(right, forward)` in `[-1, 1]²`, normalized so a diagonal is
/// not faster than a straight line.
fn input_direction(input: &Input) -> glam::Vec2 {
    let axis = |neg: [Key; 2], pos: [Key; 2]| {
        let held = |ks: [Key; 2]| ks.into_iter().any(|k| input.held(k));
        held(pos) as i32 as f32 - held(neg) as i32 as f32
    };
    let raw = glam::Vec2::new(
        axis([Key::A, Key::Left], [Key::D, Key::Right]),
        axis([Key::S, Key::Down], [Key::W, Key::Up]),
    );
    raw.try_normalize().unwrap_or(glam::Vec2::ZERO)
}

// ---------------------------------------------------------------------------
// Discrete overlaps
// ---------------------------------------------------------------------------

/// A collider entity, snapshotted so the resolver can hold the ball's transform
/// mutably while it reads everyone else's.
struct Obstacle {
    entity: Entity,
    center: Vec3,
    shape: Shape,
    trigger: bool,
}

enum Shape {
    Sphere(f32),
    Aabb(Vec3),
}

/// `FixedSim`: sphere-vs-sphere and sphere-vs-AABB overlaps for every ball
/// (DESIGN §9's "discrete shapes").
///
/// Kinematic, one-way: a solid overlap pushes the **ball** out along the contact
/// normal and kills the velocity component heading into the surface. The other
/// body is never touched and receives no impulse — §9 rules dynamic-dynamic
/// response out, and a resolver that quietly did half of it would be worse than
/// one that does none.
///
/// Both loops run in `Entity` order (DESIGN §3): archetype iteration order is
/// stable but not *specified*, and the resolution of a ball wedged between two
/// colliders depends on which it is pushed out of first.
#[allow(clippy::type_complexity)] // Bevy system params read worse behind aliases.
pub fn resolve_overlaps(
    mut writer: MessageWriter<OverlapEvent>,
    colliders: Query<
        (
            Entity,
            Option<&SphereCollider>,
            Option<&AabbCollider>,
            &Transform,
            Has<Trigger>,
        ),
        Without<Ball>,
    >,
    mut balls: Query<(Entity, &SphereCollider, &mut Transform, &mut Velocity), With<Ball>>,
) {
    let mut obstacles: Vec<Obstacle> = colliders
        .iter()
        .filter_map(|(entity, sphere, aabb, transform, trigger)| {
            let shape = match (sphere, aabb) {
                // A collider entity carrying both shapes is an authoring
                // mistake; the sphere wins and the assert says so in dev builds.
                (Some(s), _) => {
                    debug_assert!(
                        aabb.is_none(),
                        "entity {entity} has both a SphereCollider and an AabbCollider"
                    );
                    Shape::Sphere(s.radius)
                }
                (None, Some(a)) => {
                    debug_assert!(
                        transform.rotation == Quat::IDENTITY && transform.scale == Vec3::ONE,
                        "AABB collider entities must be translation-only; \
                         a rotated box is not an axis-aligned box"
                    );
                    Shape::Aabb(a.half_extents)
                }
                (None, None) => return None,
            };
            Some(Obstacle {
                entity,
                center: transform.translation,
                shape,
                trigger,
            })
        })
        .collect();
    obstacles.sort_unstable_by_key(|o| o.entity);
    if obstacles.is_empty() {
        return;
    }

    let mut ids: Vec<Entity> = balls.iter().map(|(entity, ..)| entity).collect();
    ids.sort_unstable();

    for id in ids {
        let Ok((_, collider, mut transform, mut velocity)) = balls.get_mut(id) else {
            continue;
        };
        for obstacle in &obstacles {
            let Some((normal, depth)) = overlap(
                transform.translation,
                collider.radius,
                obstacle.center,
                &obstacle.shape,
            ) else {
                continue;
            };

            writer.write(OverlapEvent {
                ball: id,
                other: obstacle.entity,
                normal,
                depth,
                trigger: obstacle.trigger,
            });

            if obstacle.trigger {
                continue; // Overlap noted, nothing resolved: the ball flies on.
            }
            transform.translation += normal * depth;
            let into = velocity.0.dot(normal);
            if into < 0.0 {
                // Only the component heading into the face. What is left is
                // tangential, so the ball slides along the surface instead of
                // stopping dead against it.
                velocity.0 -= normal * into;
            }
        }
    }
}

/// Contact between a sphere and one shape: `(normal from shape towards sphere,
/// depth)`, or `None` when they do not overlap.
fn overlap(center: Vec3, radius: f32, other: Vec3, shape: &Shape) -> Option<(Vec3, f32)> {
    match *shape {
        Shape::Sphere(other_radius) => {
            let reach = radius + other_radius;
            let delta = center - other;
            let dist_sq = delta.length_squared();
            if dist_sq >= reach * reach {
                return None;
            }
            let dist = dist_sq.sqrt();
            // Exactly concentric: no direction is more right than another, so
            // pick the one that will not make a ball vanish into the floor.
            let normal = delta.try_normalize().unwrap_or(Vec3::Y);
            Some((normal, reach - dist))
        }
        Shape::Aabb(half_extents) => {
            let half = half_extents.abs();
            let local = center - other;
            let closest = local.clamp(-half, half);
            let delta = local - closest;
            let dist_sq = delta.length_squared();
            if dist_sq > 0.0 {
                if dist_sq >= radius * radius {
                    return None;
                }
                let normal = delta.try_normalize().unwrap_or(Vec3::Y);
                return Some((normal, radius - dist_sq.sqrt()));
            }
            // Center is inside the box: leave by the nearest face, which is the
            // axis of least penetration.
            let gap = half - local.abs();
            let axis = if gap.x <= gap.y && gap.x <= gap.z {
                0
            } else if gap.y <= gap.z {
                1
            } else {
                2
            };
            let sign = if local[axis] < 0.0 { -1.0 } else { 1.0 };
            let mut normal = Vec3::ZERO;
            normal[axis] = sign;
            Some((normal, gap[axis] + radius))
        }
    }
}

// ---------------------------------------------------------------------------
// Cosmetic spin
// ---------------------------------------------------------------------------

/// `FixedSim` (after every solve): roll the ball's *appearance* to match where
/// it is going (DESIGN §9: "visual spin is derived from velocity … cosmetic,
/// never simulated state").
///
/// Axis and angle both come out of one cross product: `n × v` points along the
/// roll axis and its length is the tangential speed, because `|n| = 1`. The step
/// is `|n × v| · dt / radius` — arc length over radius, i.e. exactly the rotation
/// a ball rolling without slipping would make. (§9's shorthand is `|v|·dt/r`;
/// they agree whenever `v` is tangential, which is when a ball is rolling, and
/// this form additionally declines to spin a ball that is only falling.)
///
/// The one rule: **nothing may read `Transform.rotation` back**. The integrator
/// and the overlap resolver touch `translation` and `Velocity` only, so this
/// system could be deleted and the trajectory would not shift by a bit.
pub fn roll_spin(
    tick: Res<FixedTick>,
    mut balls: Query<(&Ball, &Velocity, Option<&Grounded>, &mut Transform), With<RollSpin>>,
) {
    let dt = tick.dt_secs;
    for (ball, velocity, grounded, mut transform) in &mut balls {
        if ball.radius <= 0.0 {
            continue;
        }
        let normal = grounded.map_or(Vec3::Y, |g| g.normal);
        let roll = normal.cross(velocity.0);
        let Some(axis) = roll.try_normalize() else {
            continue;
        };
        let angle = roll.length() * dt / ball.radius;
        transform.rotation = (Quat::from_axis_angle(axis, angle) * transform.rotation).normalize();
    }
}
