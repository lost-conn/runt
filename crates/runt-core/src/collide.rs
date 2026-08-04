//! Collision v2 — capsule character solver, rotated boxes, layers, queries
//! (DESIGN §9, "Collision v2" amendment, 2026-08-04).
//!
//! [`physics`](crate::physics) is the v1 story: a point integrator plus
//! sphere/AABB overlap push-out, tuned for a rolling ball. It is untouched by
//! this module — no system here runs in `FixedSim`, nothing here reads or writes
//! a [`Ball`](crate::physics::Ball), and `assets/demo.ron` ticks to the same
//! fingerprint it always did.
//!
//! This module is the *other* shape of kinematic motion: a Godot-style
//! `move_and_slide` for a vertical capsule (or sphere), which is what a
//! platformer's state machine needs and what a ball integrator cannot express.
//! It is a **library**, not a schedule: game code calls [`move_and_slide`] from
//! its own `FixedSim` system, at whatever point in its state machine makes
//! sense, and owns the position/velocity it passes in.
//!
//! ```text
//! CollisionWorld::from_world / ::gather   snapshot every collider, Entity-sorted
//! move_and_slide(&world, &mut body, p, v, dt) -> MoveResult
//! world.overlap_sphere / overlap_capsule  → Vec<OverlapHit>
//! world.raycast                           → Option<RayHit>
//! ```
//!
//! ## Shapes
//!
//! Everything the solver collides *with* is a convex primitive placed at an
//! entity's `Transform.translation`:
//!
//! | component | shape |
//! |---|---|
//! | [`SphereCollider`] | sphere |
//! | [`AabbCollider`] | box, world axes — the zero-rotation fast path |
//! | [`ObbCollider`] | box, arbitrary [`Quat`] |
//! | [`TerrainSurface`] | the analytic height field, sampled — never its mesh |
//!
//! …plus one shape that is not convex and not a component: [`Trimesh`], a
//! **static** triangle soup with a BVH over it, pushed into a snapshot by hand
//! ([`CollisionWorld::push_collider`]). It is the answer to CSG-baked geometry,
//! which has no analytic form — DESIGN §9a's 2026-08-04 trimesh amendment. It is
//! immutable after [`Trimesh::build`] and it never moves; everything downstream
//! of contact generation (classification, push-out, snap, stop-on-slope) is the
//! code the convex shapes already went through.
//!
//! Everything that *moves* is a swept sphere along a **vertical** segment: a
//! capsule of `{radius, height}`, or the degenerate zero-length case, a sphere.
//! One shape family means one contact routine, which is what keeps the sphere
//! and capsule modes of the port's roll swap consistent with each other.
//!
//! ### Why `ObbCollider` carries its own rotation
//!
//! The earlier plan was a yaw-only rotated box. That restriction bought nothing:
//! the contact math transforms the moving segment into the box's local frame and
//! is *rotation-agnostic* there — a pitched ramp costs exactly what a yawed wall
//! costs. So [`ObbCollider`] takes a full [`Quat`], and the PoC level's five
//! pitched ramps and one −98°-yaw phase wall are the same code path.
//!
//! The rotation lives on the **collider**, not on the entity's `Transform`, for
//! the same reason [`Ball::radius`](crate::physics::Ball::radius) is not the
//! mesh's radius: the collider is not the drawing. A scene usually gives both
//! the same angle, and `obb_collider` in a `.ron` file authors it in degrees
//! exactly like a transform does — but nothing forces them to agree, and the
//! solver never reads `Transform.rotation`.
//!
//! ## Layer semantics — one-way, query-side
//!
//! [`CollisionLayers`] is `{ memberships, mask }`, both `u16`. Godot checks
//! `A.collision_mask & B.collision_layer` — *one-way, per body*, evaluated from
//! the perspective of whoever is doing the moving. A symmetric
//! "both directions must agree" rule is a different (and stricter) system, and
//! it breaks the phase mechanic: the phase system mutates only the **player's**
//! mask and expects tagged world geometry, whose memberships never change, to
//! become passable.
//!
//! So runt takes the one-way rule and states it as a property of the *query*:
//!
//! > A collider is visible to a query iff `query_mask & collider.memberships != 0`.
//!
//! Every entry point here takes a `mask: u16`; [`move_and_slide`] uses
//! `body.layers.mask`. A collider with no [`CollisionLayers`] component is
//! `CollisionLayers::DEFAULT` — member of layer 0, mask all — so a scene written
//! before layers existed behaves identically. `memberships` on a *moving* body
//! is carried for symmetry and for the day something queries against the player;
//! nothing in this module reads it.
//!
//! Layers are plain component data, mutable from any system. A mask written this
//! tick is read by the next [`move_and_slide`] call that snapshots the world —
//! [`CollisionWorld`] is a value, taken once, so a mutation can never take effect
//! *part way through* a solve.
//!
//! ## Determinism (DESIGN §3, §4)
//!
//! - [`CollisionWorld`] sorts its colliders and terrain patches by `Entity` at
//!   construction; every scan is a `Vec` walk in that order. No hash container
//!   is iterated anywhere in this file. ([`Trimesh::build`] *looks up* in a
//!   `HashMap` while welding, and never iterates it — the output order is the
//!   input's.)
//! - Contact selection picks the greatest `depth`; an exact tie goes to the
//!   lowest `Entity`, and within one [`Trimesh`] to the lowest triangle index.
//!   Velocity is projected against contacts in `Entity` order.
//! - Every iteration count is a compile-time constant
//!   ([`SLIDE_ITERATIONS`], [`MAX_SUBSTEPS`], [`SEGMENT_SEARCH_ITERATIONS`],
//!   [`RAY_BISECTIONS`], [`RAY_MAX_STEPS`]). Nothing loops until an error
//!   threshold that a different machine might reach on a different step.
//! - Sub-stepping is derived from the *entry* velocity and `dt` only, so the
//!   number of sub-steps a tick takes cannot depend on what it collides with.
//! - Everything is a pure function of its arguments: same snapshot + same
//!   position/velocity ⇒ same `MoveResult`, bit for bit.
//!
//! ## Where the two halves of §9 do *not* meet
//!
//! [`resolve_overlaps`](crate::physics::resolve_overlaps) — the `Ball` path's
//! overlap pass — knows nothing about [`ObbCollider`] or [`CollisionLayers`]. A
//! ball rolls straight through a rotated box, and a masked-out collider is still
//! solid to it. That is a deliberate boundary, not an oversight: the v1 pass is
//! pinned by a fingerprint test and a set of trajectory tests, and the game that
//! wanted collision v2 drives a [`CharacterBody`] (sphere mode is what its
//! rolling state uses), never a `Ball`. Extending the ball pass is a few lines
//! whenever something actually needs it.
//!
//! ## What this is not
//!
//! No swept CCD (see [`MAX_SUBSTEPS`] for what is done instead), no *dynamic*
//! trimesh, no convex decomposition, no mesh-vs-mesh, no dynamic-dynamic
//! response, no joints. See DESIGN §9 and §9a.

use std::collections::HashMap;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};

use crate::ecs::{TerrainSurface, Transform};
use crate::physics::{AabbCollider, SphereCollider, Trigger};

// ---------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------

/// How many push-out/slide passes one sub-step may take. Godot's
/// `max_slides` default is 6; 5 is enough for a floor plus two walls with a
/// pass to spare, and the loop exits early the moment nothing is penetrating.
pub const SLIDE_ITERATIONS: u32 = 5;

/// Separation below which a surface still counts as *touching*, in metres.
///
/// Without it a body that comes to rest exactly on the floor reports
/// `on_floor == false` on the very next tick — it is no longer penetrating, and
/// "not penetrating" is not the same as "not standing on". 1 mm is far below
/// anything a player can see and far above `f32` noise at world scale.
pub const CONTACT_MARGIN: f32 = 1e-3;

/// Penetration below which the push-out step is skipped (the contact is still
/// reported and still projects velocity).
const PUSH_EPSILON: f32 = 1e-6;

/// How far a velocity may tilt away from straight *down* and still count as
/// "gravity only" for [`CharacterBody::floor_stop_on_slope`].
///
/// Godot's literal, from `character_body_3d.cpp`:
/// `(velocity.normalized() + up_direction).length() < 0.01`. The unit velocity
/// has to land within `0.01` of `-up`, which is 0.57° of tilt — a body with any
/// real horizontal intent fails it, and a body being pushed only by gravity
/// passes it exactly.
pub const STOP_ON_SLOPE_TILT: f32 = 0.01;

/// Largest translation one sub-step may attempt, as a fraction of the moving
/// shape's radius.
///
/// ## The margin, with the port's actual numbers
///
/// The solve is discrete: it tests the *end* position of a step, never the swept
/// volume. Consecutive test positions `d` apart each cover an interval of
/// `2·radius` along the motion axis, so they overlap — and therefore cover the
/// path with no gap for an obstacle to hide in — exactly when `d < 2·radius`.
/// Half of that bound is the fraction below, i.e. consecutive positions always
/// overlap by at least half a radius.
///
/// PORT_SPEC's body is a capsule of `radius ≈ 0.35`, so the per-sub-step cap is
/// `0.35 m`:
///
/// | motion | `|v|·dt` at 60 Hz | sub-steps |
/// |---|---|---|
/// | `max_speed` 8 | 0.133 m | 1 |
/// | `max_fall_speed` 20 | 0.333 m | 1 |
/// | ground-pound slam 30 | 0.500 m | 2 |
/// | defensive 40 | 0.667 m | 2 |
///
/// Normal play therefore never sub-steps, and the thinnest geometry in the level
/// (the 0.5 m phase wall) needs `d ≥ 0.5 + 2·0.35 = 1.2 m` to be tunnelled —
/// three and a half times the cap.
pub const SUBSTEP_RADIUS_FRACTION: f32 = 1.0;

/// Ceiling on sub-steps per call. Eight of them at the cap above is `2.8 m` of
/// motion in one tick — 168 m/s. Past that the body *can* tunnel, and it does so
/// loudly (a teleporting player) rather than by silently costing a hundred
/// solves a tick.
pub const MAX_SUBSTEPS: u32 = 8;

/// Ternary-search steps used to find the contact point on a segment against a
/// rotated box. The bracket shrinks by `(2/3)^n`, so 30 steps put the parameter
/// within `5·10⁻⁶` of the true minimum; the signed distance function is
/// 1-Lipschitz, so the depth error is that times the segment length.
pub const SEGMENT_SEARCH_ITERATIONS: u32 = 30;

/// March step for the analytic-height-field raycast, in metres. This is the
/// smallest terrain feature a ray is guaranteed to notice; the field is analytic,
/// so it has nothing to do with the mesh's tessellation and the same ray hits the
/// same point at every [`Quality`](runt_mesh::Quality) tier.
pub const RAY_MARCH_STEP: f32 = 0.125;

/// Cap on march steps: `512 · 0.125 = 64 m` of terrain ray.
pub const RAY_MAX_STEPS: u32 = 512;

/// Bisection steps that refine a bracketed terrain crossing. 24 of them resolve
/// [`RAY_MARCH_STEP`] to `7·10⁻⁹ m`.
pub const RAY_BISECTIONS: u32 = 24;

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/// An **oriented** box collider, centered on the entity's
/// `Transform.translation`.
///
/// The rotation is world-space and belongs to the collider, not the entity — see
/// the module docs. Full [`Quat`], not yaw: the closest-point solve happens in
/// the box's own frame, where orientation has already been divided out, so a
/// pitched ramp is not more expensive than a yawed wall. This supersedes the
/// earlier yaw-only plan.
///
/// [`AabbCollider`] remains the zero-rotation form and stays cheaper: a vertical
/// segment against an axis-aligned box has a closed-form contact point, while an
/// OBB needs the search.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct ObbCollider {
    pub half_extents: Vec3,
    pub rotation: Quat,
}

impl Default for ObbCollider {
    fn default() -> ObbCollider {
        ObbCollider {
            half_extents: Vec3::splat(0.5),
            rotation: Quat::IDENTITY,
        }
    }
}

impl ObbCollider {
    pub fn new(half_extents: Vec3, rotation: Quat) -> ObbCollider {
        ObbCollider {
            half_extents,
            rotation,
        }
    }

    /// A box pitched about world X, the form the PoC level's five ramps take.
    pub fn pitched(half_extents: Vec3, degrees: f32) -> ObbCollider {
        ObbCollider::new(half_extents, Quat::from_rotation_x(degrees.to_radians()))
    }

    /// A box turned about world Y, the form the PoC level's phase wall takes.
    pub fn yawed(half_extents: Vec3, degrees: f32) -> ObbCollider {
        ObbCollider::new(half_extents, Quat::from_rotation_y(degrees.to_radians()))
    }
}

/// Which layers an entity belongs to, and which layers it looks for.
///
/// One-way, query-side semantics — see the module docs. `memberships` says what
/// this entity *is*; `mask` says what it *collides with* when it is the one
/// moving or querying.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct CollisionLayers {
    pub memberships: u16,
    pub mask: u16,
}

/// Every layer bit set — the default mask.
pub const ALL_LAYERS: u16 = u16::MAX;

impl Default for CollisionLayers {
    fn default() -> CollisionLayers {
        CollisionLayers::DEFAULT
    }
}

impl CollisionLayers {
    /// Member of layer 0, collides with everything. What an entity with no
    /// [`CollisionLayers`] component is treated as, which is what makes adding
    /// layers a non-event for scenes written before them.
    pub const DEFAULT: CollisionLayers = CollisionLayers {
        memberships: 1,
        mask: ALL_LAYERS,
    };

    /// Member of layer `index` only, colliding with everything.
    pub fn layer(index: u32) -> CollisionLayers {
        CollisionLayers {
            memberships: 1 << index,
            mask: ALL_LAYERS,
        }
    }

    pub fn with_memberships(mut self, memberships: u16) -> CollisionLayers {
        self.memberships = memberships;
        self
    }

    pub fn with_mask(mut self, mask: u16) -> CollisionLayers {
        self.mask = mask;
        self
    }

    /// Turn one mask bit on or off. The phase system's whole job.
    pub fn set_mask_layer(&mut self, index: u32, enabled: bool) {
        let bit = 1u16 << index;
        if enabled {
            self.mask |= bit;
        } else {
            self.mask &= !bit;
        }
    }

    /// Turn one membership bit on or off.
    pub fn set_membership_layer(&mut self, index: u32, enabled: bool) {
        let bit = 1u16 << index;
        if enabled {
            self.memberships |= bit;
        } else {
            self.memberships &= !bit;
        }
    }
}

/// The one-way visibility rule, spelled out once so nothing can restate it
/// differently: a collider is visible to a query iff the query's mask shares a
/// bit with the collider's memberships.
#[inline]
pub fn mask_accepts(query_mask: u16, memberships: u16) -> bool {
    query_mask & memberships != 0
}

/// The moving shape: a vertical capsule, or a sphere.
///
/// PORT_SPEC swaps between the two at runtime (standing ↔ rolling), so they are
/// one enum on one component rather than two components a state machine has to
/// insert and remove.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CharacterShape {
    /// `height` is the **total** height including both caps, matching Godot's
    /// `CapsuleShape3D`. The swept segment is therefore `height - 2·radius`
    /// long; a capsule with `height <= 2·radius` degenerates to a sphere rather
    /// than becoming invalid.
    Capsule { radius: f32, height: f32 },
    Sphere { radius: f32 },
}

impl Default for CharacterShape {
    fn default() -> CharacterShape {
        CharacterShape::Capsule {
            radius: 0.35,
            height: 2.0,
        }
    }
}

impl CharacterShape {
    #[inline]
    pub fn radius(self) -> f32 {
        match self {
            CharacterShape::Capsule { radius, .. } | CharacterShape::Sphere { radius } => radius,
        }
    }

    /// Half the length of the swept segment: `0` for a sphere.
    #[inline]
    pub fn half_segment(self) -> f32 {
        match self {
            CharacterShape::Capsule { radius, height } => (height * 0.5 - radius).max(0.0),
            CharacterShape::Sphere { .. } => 0.0,
        }
    }

    /// The swept segment `(lower, upper)` for a body centered at `position`.
    /// Both endpoints coincide for a sphere.
    #[inline]
    pub fn segment(self, position: Vec3, up: Vec3) -> (Vec3, Vec3) {
        let h = self.half_segment();
        (position - up * h, position + up * h)
    }
}

/// A kinematic character: the shape and the tunables [`move_and_slide`] reads.
///
/// Every field is meant to be written *between* solves — PORT_SPEC needs
/// `max_floor_angle` at 45° standing, 89° during a ground-pound slam and 180°
/// rolling, `snap_length` toggled 0.5/0.0 grounded/airborne, the shape swapped
/// capsule↔sphere, and `layers.mask` rewritten every phase frame. None of them
/// is read anywhere except inside a call, so a mid-tick write is impossible by
/// construction.
///
/// Position and velocity are **not** here: they are the caller's, passed in and
/// handed back by [`MoveResult`]. A component that mirrored `Transform` would be
/// a second copy of the simulation state, which DESIGN §9 already refuses for
/// [`Grounded`](crate::physics::Grounded).
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct CharacterBody {
    pub shape: CharacterShape,
    /// Radians. A contact whose normal is within this angle of `up` is floor.
    /// `PI` makes every contact floor (the rolling case).
    pub max_floor_angle: f32,
    /// Godot's `floor_snap_length`, metres. `0` disables snapping.
    pub snap_length: f32,
    /// Godot's `floor_stop_on_slope`, **default `true`** as it is there.
    ///
    /// A body standing on a slope it is allowed to stand on must not slide down
    /// it. Without this, projecting the tick's gravity onto the slope plane
    /// hands back a downhill tangential velocity every tick, and the caller's
    /// friction is left fighting a force the solver invented: the body creeps.
    /// With it, a *gravity-only* motion (see [`STOP_ON_SLOPE_TILT`]) that finds
    /// floor has its velocity zeroed, and — when the floor absorbed the whole of
    /// that motion, which is what standing still *is* — its position taken back
    /// to where the tick started.
    ///
    /// Set it to `false` for a body that is *meant* to slide (Godot's own
    /// escape hatch); the solver then behaves exactly as it did before the flag
    /// existed. It says nothing about steep faces: a contact past
    /// `max_floor_angle` is a wall, walls never stop the body, and a body on one
    /// still slides.
    pub floor_stop_on_slope: bool,
    /// The layers this body collides with (`mask`) and belongs to
    /// (`memberships`, unread here — see the module docs).
    pub layers: CollisionLayers,
    /// Which way is up. Constant `+Y` in practice; a field so the classifier
    /// never hardcodes it.
    pub up: Vec3,
    /// Was the body grounded at the end of the previous solve? **Written by**
    /// [`move_and_slide`]; floor snap refuses to engage without it, exactly as
    /// Godot refuses to snap a body that was already airborne.
    pub on_floor: bool,
}

impl Default for CharacterBody {
    fn default() -> CharacterBody {
        CharacterBody {
            shape: CharacterShape::default(),
            max_floor_angle: std::f32::consts::FRAC_PI_4,
            snap_length: 0.5,
            floor_stop_on_slope: true,
            layers: CollisionLayers::DEFAULT,
            up: Vec3::Y,
            on_floor: false,
        }
    }
}

impl CharacterBody {
    pub fn with_shape(mut self, shape: CharacterShape) -> CharacterBody {
        self.shape = shape;
        self
    }

    pub fn with_max_floor_degrees(mut self, degrees: f32) -> CharacterBody {
        self.max_floor_angle = degrees.to_radians();
        self
    }

    pub fn with_snap_length(mut self, snap_length: f32) -> CharacterBody {
        self.snap_length = snap_length;
        self
    }

    pub fn with_floor_stop_on_slope(mut self, stop: bool) -> CharacterBody {
        self.floor_stop_on_slope = stop;
        self
    }

    pub fn with_layers(mut self, layers: CollisionLayers) -> CharacterBody {
        self.layers = layers;
        self
    }

    /// The swept segment for a body centered at `position`.
    #[inline]
    pub fn segment(&self, position: Vec3) -> (Vec3, Vec3) {
        self.shape.segment(position, self.up)
    }
}

// ---------------------------------------------------------------------------
// Contacts
// ---------------------------------------------------------------------------

/// What a contact normal means, relative to a body's `max_floor_angle`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContactKind {
    Floor,
    Wall,
    Ceiling,
}

/// One resolved contact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contact {
    /// The collider (or terrain patch) that was touched.
    pub entity: Entity,
    /// Unit normal pointing **out of** the surface, towards the body — the
    /// direction push-out moves in. Same convention as
    /// [`OverlapEvent::normal`](crate::physics::OverlapEvent::normal).
    pub normal: Vec3,
    /// A point on the surface, at the deepest part of the overlap.
    pub point: Vec3,
    /// Penetration along `normal`. May be slightly negative (down to
    /// `-CONTACT_MARGIN`) for a body that is touching but not overlapping.
    pub depth: f32,
    pub kind: ContactKind,
}

/// What one [`move_and_slide`] produced.
///
/// `contacts` is deduplicated by entity (deepest wins) and sorted by `Entity`.
/// It is a plain `Vec`, pre-sized to the number of colliders the body could
/// possibly touch; the inline-storage optimisation is a swap of this one type
/// when a profile says it matters.
#[derive(Clone, Debug, PartialEq)]
pub struct MoveResult {
    pub position: Vec3,
    /// Velocity after slide projection. The caller keeps its own *pre-slide*
    /// copy if it needs one (PORT_SPEC's roll wall-bump does).
    pub velocity: Vec3,
    pub on_floor: bool,
    /// `+Y` when not on floor.
    pub floor_normal: Vec3,
    /// Radians between `floor_normal` and `up`; `0` when not on floor.
    pub floor_angle: f32,
    pub on_wall: bool,
    /// `Vec3::ZERO` when not on a wall.
    pub wall_normal: Vec3,
    pub on_ceiling: bool,
    /// `Vec3::ZERO` when not on a ceiling.
    pub ceiling_normal: Vec3,
    pub contacts: Vec<Contact>,
    /// How many sub-steps the translation was split into (≥ 1).
    pub sub_steps: u32,
    /// Whether floor snap is what produced `on_floor`.
    pub snapped: bool,
    /// Whether [`CharacterBody::floor_stop_on_slope`] cancelled this tick's
    /// gravity. When it is set, `velocity` is `Vec3::ZERO` by construction.
    pub stopped_on_slope: bool,
}

impl MoveResult {
    fn empty(position: Vec3, velocity: Vec3) -> MoveResult {
        MoveResult {
            position,
            velocity,
            on_floor: false,
            floor_normal: Vec3::Y,
            floor_angle: 0.0,
            on_wall: false,
            wall_normal: Vec3::ZERO,
            on_ceiling: false,
            ceiling_normal: Vec3::ZERO,
            contacts: Vec::new(),
            sub_steps: 1,
            snapped: false,
            stopped_on_slope: false,
        }
    }

    /// The floor contacts, in `Entity` order.
    pub fn floors(&self) -> impl Iterator<Item = &Contact> {
        self.contacts.iter().filter(|c| c.kind == ContactKind::Floor)
    }

    /// The wall contacts, in `Entity` order.
    pub fn walls(&self) -> impl Iterator<Item = &Contact> {
        self.contacts.iter().filter(|c| c.kind == ContactKind::Wall)
    }
}

/// A contact before classification.
#[derive(Clone, Copy, Debug)]
struct RawContact {
    entity: Entity,
    normal: Vec3,
    point: Vec3,
    depth: f32,
}

// ---------------------------------------------------------------------------
// Static trimeshes (DESIGN §9a, 2026-08-04)
// ---------------------------------------------------------------------------

/// Grid the welder snaps positions onto, in metres.
///
/// 0.1 mm: two vertices a CSG bake emitted separately for the same corner land
/// on the same key, and two vertices a level author *meant* to keep apart never
/// do. The quantised key is what the lookup is on; the vertex kept is the
/// **first occurrence's exact position**, so welding never moves geometry by up
/// to half a cell — it only ever discards duplicates.
pub const WELD_GRID: f32 = 1.0e-4;

/// Squared cross-product length below which a triangle is dropped as
/// degenerate. Same threshold and the same reasoning as `runt_mesh`'s
/// `DEGENERATE_AREA_SQ` (which is crate-private there): a real triangle's raw
/// cross is ~1e-4 squared, a float sliver's is ~1e-17.
pub const DEGENERATE_AREA_SQ: f32 = 1.0e-12;

/// Triangles per BVH leaf. Eight is the point where the leaf's linear scan is
/// cheaper than another level of node tests, and it caps the number of contacts
/// one leaf can contribute.
pub const BVH_LEAF_TRIS: usize = 8;

/// Depth of the fixed traversal stack. The build splits at the **median index**,
/// so the tree is balanced by construction and depth is `ceil(log2(n/8)) + 1` —
/// 64 covers 2^66 triangles, i.e. it cannot be reached. It is a fixed-size array
/// rather than a `Vec` so traversal allocates nothing and its bound is a
/// compile-time constant (DESIGN §9a).
pub const BVH_STACK_DEPTH: usize = 64;

/// One node of a [`Trimesh`]'s BVH.
///
/// Flat `Vec`, children as indices, no `Box`: the whole tree is one allocation
/// and one memcpy, and a node index is a `u32` a fixed-size stack can hold.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BvhNode {
    min: Vec3,
    max: Vec3,
    /// Leaf: the first triangle. Inner: the left child node.
    first: u32,
    /// Leaf: how many triangles (always `> 0`). Inner: `0`.
    count: u32,
    /// Inner: the right child node. Leaf: unused.
    right: u32,
    /// Inner: the axis the split was made on, `0..3`. Leaf: `3`.
    axis: u32,
}

impl BvhNode {
    pub fn bounds(&self) -> (Vec3, Vec3) {
        (self.min, self.max)
    }

    pub fn is_leaf(&self) -> bool {
        self.count > 0
    }

    /// `(first triangle, count)` for a leaf, `None` for an inner node.
    pub fn leaf_range(&self) -> Option<(u32, u32)> {
        self.is_leaf().then_some((self.first, self.count))
    }

    /// `(left, right)` node indices for an inner node, `None` for a leaf.
    pub fn children(&self) -> Option<(u32, u32)> {
        (!self.is_leaf()).then_some((self.first, self.right))
    }

    /// The axis an inner node was split on, `None` for a leaf.
    pub fn split_axis(&self) -> Option<usize> {
        (!self.is_leaf()).then_some(self.axis as usize)
    }

    /// Does this node's box overlap `[lo, hi]`?
    #[inline]
    fn overlaps(&self, lo: Vec3, hi: Vec3) -> bool {
        self.min.cmple(hi).all() && self.max.cmpge(lo).all()
    }
}

/// A **static** triangle soup with a BVH over it.
///
/// Immutable after [`build`](Trimesh::build), which is what lets the whole thing
/// live behind an [`Arc`] and be shared by every entry that references it: a
/// level's baked geometry is built once at load and then only read.
///
/// ## What `build` does, and why each step is deterministic
///
/// 1. **Weld** positions onto a [`WELD_GRID`] cell. The lookup is a `HashMap`,
///    but the *output* order is the input's — vertices are appended on first
///    occurrence, so nothing here iterates a hash container and two runs over
///    the same soup produce the same `verts` array.
/// 2. **Drop degenerates** — a triangle with a repeated welded index, or a raw
///    cross below [`DEGENERATE_AREA_SQ`]. A zero-area triangle has no normal,
///    and a contact routine that divides by its length is a NaN generator.
/// 3. **Face normals**, precomputed once. They are the contact normal the solver
///    reports on the face interior, so the surface a body stands on is the
///    surface the level author authored — not an average of whatever the
///    tessellator did nearby.
/// 4. **BVH**, top-down: split at the *median index* of the triangles sorted by
///    centroid along the longest axis of the node's centroid bounds, keyed
///    `(axis value, original triangle index)` so equal centroids still have one
///    order. Median-index (rather than median-value) split means the halves are
///    always the same size, which is what bounds the depth — a soup whose
///    centroids all coincide splits evenly instead of recursing forever.
///
/// The triangles are physically reordered into BVH order, so a leaf is a
/// contiguous range and "ascending triangle index" — the tie-break rule for both
/// contacts and raycasts — is the order the arrays are already in.
#[derive(Clone, Debug, PartialEq)]
pub struct Trimesh {
    verts: Vec<Vec3>,
    tris: Vec<[u32; 3]>,
    face_normals: Vec<Vec3>,
    /// Three bits per triangle: bit `k` is set when edge `k` (`0` = `a→b`, `1` =
    /// `b→c`, `2` = `c→a`) is **exposed** — an open boundary of the soup, or a
    /// ridge the surface genuinely turns a corner at. A clear bit is a *seam*: a
    /// shared edge where the neighbour continues this triangle's surface flat or
    /// concave, so touching it is touching the face. See
    /// [`build_from_soup`](Trimesh::build_from_soup).
    edge_flags: Vec<u8>,
    nodes: Vec<BvhNode>,
}

/// One triangle while the tree is being built. Dropped once the arrays are
/// written out in BVH order.
struct BuildTri {
    centroid: Vec3,
    min: Vec3,
    max: Vec3,
    tri: [u32; 3],
    normal: Vec3,
    edge_flags: u8,
    /// Index in the *welded* triangle list, before any reordering: the
    /// tie-break key, and the one thing about a triangle that a sort cannot
    /// change.
    original: u32,
}

impl Trimesh {
    /// Build from a [`MeshData`](runt_mesh::MeshData) — the mesh a generator or
    /// a CSG bake produced. Only `positions` and `indices` are read; normals,
    /// UVs and colours are the drawing's business, not the collider's.
    pub fn build(mesh: &runt_mesh::MeshData) -> Arc<Trimesh> {
        Trimesh::build_from_soup(&mesh.positions, &mesh.indices)
    }

    /// Build from bare slices, for geometry that never became a `MeshData`.
    pub fn build_from_soup(positions: &[Vec3], indices: &[u32]) -> Arc<Trimesh> {
        // -- weld ----------------------------------------------------------
        let mut verts: Vec<Vec3> = Vec::new();
        let mut welded: Vec<u32> = Vec::with_capacity(positions.len());
        let mut grid: HashMap<[i64; 3], u32> = HashMap::with_capacity(positions.len());
        for &p in positions {
            let key = weld_key(p);
            let index = *grid.entry(key).or_insert_with(|| {
                verts.push(p);
                (verts.len() - 1) as u32
            });
            welded.push(index);
        }

        // -- triangles, degenerates dropped, normals precomputed ------------
        let mut items: Vec<BuildTri> = Vec::with_capacity(indices.len() / 3);
        for face in indices.chunks_exact(3) {
            debug_assert!(
                face.iter().all(|i| (*i as usize) < welded.len()),
                "trimesh index out of range of the position array"
            );
            if face.iter().any(|i| (*i as usize) >= welded.len()) {
                continue;
            }
            let tri = [
                welded[face[0] as usize],
                welded[face[1] as usize],
                welded[face[2] as usize],
            ];
            if tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2] {
                continue;
            }
            let (a, b, c) = (
                verts[tri[0] as usize],
                verts[tri[1] as usize],
                verts[tri[2] as usize],
            );
            let cross = (b - a).cross(c - a);
            if cross.length_squared() < DEGENERATE_AREA_SQ {
                continue;
            }
            let original = items.len() as u32;
            items.push(BuildTri {
                centroid: (a + b + c) / 3.0,
                min: a.min(b).min(c),
                max: a.max(b).max(c),
                tri,
                normal: cross.normalize(),
                edge_flags: 0b111,
                original,
            });
        }

        mark_exposed_edges(&verts, &mut items);

        // -- BVH -----------------------------------------------------------
        let mut nodes: Vec<BvhNode> = Vec::new();
        if !items.is_empty() {
            nodes.reserve(items.len() / BVH_LEAF_TRIS * 2 + 2);
            build_bvh(&mut items, 0, &mut nodes);
        }

        let tris = items.iter().map(|i| i.tri).collect();
        let face_normals = items.iter().map(|i| i.normal).collect();
        let edge_flags = items.iter().map(|i| i.edge_flags).collect();
        Arc::new(Trimesh {
            verts,
            tris,
            face_normals,
            edge_flags,
            nodes,
        })
    }

    pub fn triangle_count(&self) -> usize {
        self.tris.len()
    }

    pub fn vertex_count(&self) -> usize {
        self.verts.len()
    }

    /// The welded vertices, in first-occurrence order.
    pub fn verts(&self) -> &[Vec3] {
        &self.verts
    }

    /// The triangles, in BVH order — the order every tie-break refers to.
    pub fn tris(&self) -> &[[u32; 3]] {
        &self.tris
    }

    /// One unit normal per triangle, in the same order as [`tris`](Trimesh::tris).
    pub fn face_normals(&self) -> &[Vec3] {
        &self.face_normals
    }

    /// Three bits per triangle saying which of its edges are exposed ridges
    /// rather than seams with a neighbour — see the field's own docs and
    /// [`trimesh_contact`].
    pub fn edge_flags(&self) -> &[u8] {
        &self.edge_flags
    }

    /// The flat node array. `nodes[0]` is the root; empty for an empty mesh.
    pub fn nodes(&self) -> &[BvhNode] {
        &self.nodes
    }

    /// Local-space bounds, or `None` for an empty mesh.
    pub fn bounds(&self) -> Option<(Vec3, Vec3)> {
        self.nodes.first().map(|n| n.bounds())
    }

    #[inline]
    fn triangle(&self, index: u32) -> [Vec3; 3] {
        let t = self.tris[index as usize];
        [
            self.verts[t[0] as usize],
            self.verts[t[1] as usize],
            self.verts[t[2] as usize],
        ]
    }

    /// Every triangle whose box overlaps `[lo, hi]`, in **ascending triangle
    /// index**.
    ///
    /// The stack is a fixed `[u32; BVH_STACK_DEPTH]` and children are pushed in
    /// a fixed order (right then left, so the left subtree pops first), which
    /// makes the leaves come out in index order already; the sort is there so
    /// the guarantee is the routine's rather than the traversal's.
    fn query_aabb(&self, lo: Vec3, hi: Vec3, out: &mut Vec<u32>) {
        out.clear();
        if self.nodes.is_empty() {
            return;
        }
        let mut stack = [0u32; BVH_STACK_DEPTH];
        let mut depth = 0usize;
        stack[depth] = 0;
        depth += 1;
        while depth > 0 {
            depth -= 1;
            let node = &self.nodes[stack[depth] as usize];
            if !node.overlaps(lo, hi) {
                continue;
            }
            if node.is_leaf() {
                out.extend(node.first..node.first + node.count);
                continue;
            }
            assert!(
                depth + 2 <= BVH_STACK_DEPTH,
                "BVH traversal exceeded its fixed stack — the tree is not balanced"
            );
            stack[depth] = node.right;
            stack[depth + 1] = node.first;
            depth += 2;
        }
        out.sort_unstable();
    }
}

/// The quantised weld key. `round` rather than `floor` so a position sitting on
/// a cell boundary is not split by an ulp of noise, and `i64` so a 32-bit world
/// coordinate cannot overflow it.
#[inline]
fn weld_key(p: Vec3) -> [i64; 3] {
    let q = p / WELD_GRID;
    [q.x.round() as i64, q.y.round() as i64, q.z.round() as i64]
}

/// How far off a triangle's plane a neighbour's far vertex has to lie, relative
/// to its distance, before the shared edge counts as a ridge rather than a seam.
///
/// `1e-4` of the distance is a dihedral of about 0.006° — far below anything a
/// level author authored on purpose, and far above the float noise in a surface
/// that is meant to be flat. Being *wrong* in the tolerant direction is the safe
/// side: a seam mistaken for a ridge is the internal-edge bug back again, while
/// a ridge of six thousandths of a degree is a flat surface.
const EDGE_RIDGE_TOLERANCE: f32 = 1.0e-4;

/// Work out which of each triangle's edges are **exposed** — the rule
/// [`trimesh_contact`] leans on.
///
/// An edge is a *seam* when some triangle sharing it continues this one's
/// surface: the neighbour's opposite vertex lies on or in front of this
/// triangle's plane, so a body on the front side can never legitimately touch
/// that edge from outside — whatever it touches there, it touches the face.
/// An edge is *exposed* when no triangle shares it (the open boundary of a
/// sheet) or every neighbour folds away behind the plane (a convex ridge, where
/// the direction to the edge really is the surface direction).
///
/// The lookup is a `HashMap` keyed on the welded vertex pair, filled and read in
/// triangle order. Nothing iterates it, so the flags are a pure function of the
/// soup (DESIGN §9a).
fn mark_exposed_edges(verts: &[Vec3], items: &mut [BuildTri]) {
    // Edge (low, high) → the triangles on it, as (triangle, opposite vertex).
    let mut shared: HashMap<(u32, u32), Vec<(u32, u32)>> = HashMap::with_capacity(items.len() * 2);
    for (index, item) in items.iter().enumerate() {
        for k in 0..3usize {
            let (v0, v1) = (item.tri[k], item.tri[(k + 1) % 3]);
            let key = (v0.min(v1), v0.max(v1));
            shared
                .entry(key)
                .or_default()
                .push((index as u32, item.tri[(k + 2) % 3]));
        }
    }

    for (index, item) in items.iter_mut().enumerate() {
        let (tri, normal) = (item.tri, item.normal);
        let mut flags = 0u8;
        for k in 0..3usize {
            let (v0, v1) = (tri[k], tri[(k + 1) % 3]);
            let key = (v0.min(v1), v0.max(v1));
            let on_edge = &shared[&key];
            let mut exposed = true;
            for &(other, opposite) in on_edge {
                if other == index as u32 {
                    continue;
                }
                let away = verts[opposite as usize] - verts[v0 as usize];
                let reach = away.length().max(1e-6);
                // In front of, or level with, this triangle's plane: the surface
                // does not turn a corner here.
                if away.dot(normal) > -EDGE_RIDGE_TOLERANCE * reach {
                    exposed = false;
                    break;
                }
            }
            if exposed {
                flags |= 1 << k;
            }
        }
        item.edge_flags = flags;
    }
}

/// Build the subtree covering `items`, whose first triangle will end up at
/// `offset`. Returns the index of the node it wrote.
fn build_bvh(items: &mut [BuildTri], offset: usize, nodes: &mut Vec<BvhNode>) -> u32 {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for it in items.iter() {
        min = min.min(it.min);
        max = max.max(it.max);
    }

    let me = nodes.len() as u32;
    nodes.push(BvhNode {
        min,
        max,
        first: offset as u32,
        count: items.len() as u32,
        right: 0,
        axis: 3,
    });
    if items.len() <= BVH_LEAF_TRIS {
        return me;
    }

    // The longest axis of the *centroid* bounds, not of the geometry bounds: a
    // node of long thin triangles all lying in a row is split along the row.
    let mut cmin = Vec3::splat(f32::INFINITY);
    let mut cmax = Vec3::splat(f32::NEG_INFINITY);
    for it in items.iter() {
        cmin = cmin.min(it.centroid);
        cmax = cmax.max(it.centroid);
    }
    let extent = cmax - cmin;
    let axis = if extent.x >= extent.y && extent.x >= extent.z {
        0usize
    } else if extent.y >= extent.z {
        1
    } else {
        2
    };

    // Stable in the sense that matters: a *total* order. `partial_cmp` cannot
    // fail on a coordinate that reached here (degenerates are gone), and the
    // `then` on the original index means two triangles with the same centroid
    // still have exactly one relative order.
    items.sort_by(|a, b| {
        a.centroid[axis]
            .partial_cmp(&b.centroid[axis])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.original.cmp(&b.original))
    });

    let mid = items.len() / 2;
    let (left_items, right_items) = items.split_at_mut(mid);
    let left = build_bvh(left_items, offset, nodes);
    let right = build_bvh(right_items, offset + mid, nodes);
    nodes[me as usize] = BvhNode {
        min,
        max,
        first: left,
        count: 0,
        right,
        axis: axis as u32,
    };
    me
}

// -- closest-point math ------------------------------------------------------

/// Which part of a triangle a closest point landed on.
///
/// Edges are numbered by their first vertex: `0` is `a→b`, `1` is `b→c`, `2` is
/// `c→a`. A vertex sits on two of them — vertex `i` on edges `i` and `(i+2)%3`.
///
/// The distinction is not cosmetic. An **interior** closest point means the
/// contact is against the face, and the face's own normal is the honest answer.
/// A boundary point is on an edge that some *other* triangle probably shares,
/// and whether the direction to it is a real surface direction or an artefact of
/// where the tessellator happened to cut is exactly what
/// [`Trimesh::edge_flags`](Trimesh) records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TriFeature {
    Interior,
    Edge(u8),
    Vertex(u8),
}

impl TriFeature {
    /// Is this feature on an edge the mesh marked **exposed** — a real ridge or
    /// an open boundary, rather than a seam between two triangles that continue
    /// each other?
    ///
    /// A vertex is exposed if either of its edges is: it takes only one genuine
    /// ridge through a point for the direction to that point to be real.
    #[inline]
    fn is_exposed(self, edge_flags: u8) -> bool {
        match self {
            TriFeature::Interior => false,
            TriFeature::Edge(k) => edge_flags & (1 << k) != 0,
            TriFeature::Vertex(i) => edge_flags & ((1 << i) | (1 << ((i + 2) % 3))) != 0,
        }
    }
}

/// Closest point on triangle `abc` to `p` — Ericson, *Real-Time Collision
/// Detection* 5.1.5, with the Voronoi region it landed in reported.
fn closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (Vec3, TriFeature) {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, TriFeature::Vertex(0));
    }

    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, TriFeature::Vertex(1));
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (a + ab * v, TriFeature::Edge(0));
    }

    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, TriFeature::Vertex(2));
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (a + ac * w, TriFeature::Edge(2));
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + (c - b) * w, TriFeature::Edge(1));
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (a + ab * v + ac * w, TriFeature::Interior)
}

/// Closest pair between two segments — RTCD 5.1.9. Returns `(on p1q1, on p2q2)`.
fn closest_segment_segment(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> (Vec3, Vec3) {
    const EPS: f32 = 1e-12;
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);

    if a <= EPS && e <= EPS {
        return (p1, p2);
    }
    let (s, t);
    if a <= EPS {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(r);
        if e <= EPS {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = d1.dot(d2);
            let denom = a * e - b * b;
            let s0 = if denom > EPS {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                // Parallel: any `s` is as good, and zero is the one both a
                // forward and a reversed pair of segments agree on.
                0.0
            };
            let t0 = (b * s0 + f) / e;
            if t0 < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t0 > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            } else {
                t = t0;
                s = s0;
            }
        }
    }
    (p1 + d1 * s, p2 + d2 * t)
}

/// Closest pair between a segment and a triangle — RTCD 5.1.10's decomposition:
/// the minimum over each segment **endpoint** against the triangle and the
/// segment against each of the three triangle **edges**.
///
/// Those candidates cover every configuration in which the two are disjoint,
/// including the awkward one (segment parallel to the plane, both endpoints
/// projecting outside the triangle, the middle projecting inside) — there the
/// segment-vs-edge candidate reports exactly the perpendicular distance.
///
/// The *piercing* case is deliberately not special-cased: a segment through the
/// face has an endpoint behind the plane, that endpoint's closest triangle point
/// is interior, and [`trimesh_contact`] reads a negative signed distance there
/// and pushes out along the face normal — which is the answer a special case
/// would have had to produce anyway.
///
/// Ties keep the earlier candidate (endpoints before edges, `a→b` before `b→c`
/// before `c→a`), which is a fixed order and therefore a deterministic one.
fn closest_segment_triangle(
    s0: Vec3,
    s1: Vec3,
    a: Vec3,
    b: Vec3,
    c: Vec3,
) -> (Vec3, Vec3, TriFeature) {
    let mut best_d2 = f32::INFINITY;
    let mut best = (s0, a, TriFeature::Vertex(0));
    for s in [s0, s1] {
        let (on_tri, feature) = closest_point_on_triangle(s, a, b, c);
        let d2 = (s - on_tri).length_squared();
        if d2 < best_d2 {
            best_d2 = d2;
            best = (s, on_tri, feature);
        }
    }
    if s0 != s1 {
        for (k, (e0, e1)) in [(a, b), (b, c), (c, a)].into_iter().enumerate() {
            let (on_seg, on_edge) = closest_segment_segment(s0, s1, e0, e1);
            let d2 = (on_seg - on_edge).length_squared();
            if d2 < best_d2 {
                best_d2 = d2;
                best = (on_seg, on_edge, TriFeature::Edge(k as u8));
            }
        }
    }
    best
}

/// Contact between a swept sphere (`seg_lo`..`seg_hi`, `radius`) and one
/// triangle, in the triangle's own space.
///
/// Returns `(normal, depth, point)` in the same convention every other contact
/// routine here uses: the normal points **out of** the surface towards the body,
/// the depth is penetration along it, the point is on the surface.
///
/// ## The normal, and why it is not always the direction to the closest point
///
/// Taking `(closest_on_segment − closest_on_triangle) / dist` is right when the
/// body is off the side of a face and wrong when it is over one: a body standing
/// on a tessellated floor is only ever a hair from the shared edge of two
/// coplanar triangles, and the direction to that *edge* tilts away from vertical
/// the moment the body drifts past it. That is the **internal edge** problem,
/// and it is not cosmetic: a tilted normal is a tilted floor, a floor tilted
/// past `max_floor_angle` is a *wall*, and a wall in the middle of flat ground
/// eats the speed of everything that runs over it.
///
/// Four rules, all of them geometry rather than tuning:
///
/// - **Interior closest point ⇒ the face normal.** The body is over the face;
///   the face's own normal is the surface it is on, exactly as the OBB path
///   reports the box face it touched rather than an averaged direction.
/// - **A closest point on a seam ⇒ the face normal.** `edge_flags` says which of
///   the triangle's three edges are genuine ridges (see
///   [`Trimesh::build_from_soup`]); a seam between two triangles that continue
///   each other is not a feature of the *surface*, only of the tessellation, and
///   a contact must not be able to tell they were ever separate triangles.
/// - **`dist` below 1e-6 ⇒ the face normal.** There is no direction to
///   normalise, and dividing by it would be a NaN.
/// - **A direction pointing behind the face ⇒ the face normal.** The body got
///   through and wants pushing back to the front, not further in.
///
/// What is left keeps the true direction: a body against a real exposed edge —
/// the lip of a triangulated ledge, the corner of a baked box, the boundary of
/// an open sheet — where the rounded normal is the right answer and snapping to
/// the face would shove the body along a surface it is beside rather than on.
///
/// ## Depth
///
/// Against the face the depth is measured from the **plane** (a signed
/// distance), not from the closest point: a body that has sunk behind the
/// triangle then reads a depth of `radius + |signed|` and comes back out,
/// instead of reading a small unsigned distance and staying stuck. The candidate
/// is rejected on unsigned distance first, so the depth this can produce is
/// bounded by `2·radius + CONTACT_MARGIN` — a body far behind a face is out of
/// range of it, not catapulted by it.
fn trimesh_contact(
    seg_lo: Vec3,
    seg_hi: Vec3,
    radius: f32,
    tri: [Vec3; 3],
    face_normal: Vec3,
    edge_flags: u8,
) -> Option<(Vec3, f32, Vec3)> {
    let [a, b, c] = tri;
    let (on_seg, on_tri, feature) = closest_segment_triangle(seg_lo, seg_hi, a, b, c);
    let delta = on_seg - on_tri;
    let dist = delta.length();
    if dist > radius + CONTACT_MARGIN {
        return None;
    }
    let front = delta.dot(face_normal);
    let (normal, separation) = if dist < 1e-6 || front < 0.0 || !feature.is_exposed(edge_flags) {
        (face_normal, front)
    } else {
        (delta / dist, dist)
    };
    let depth = radius - separation;
    if depth <= -CONTACT_MARGIN {
        return None;
    }
    Some((normal, depth, on_tri))
}

/// Every contact a swept sphere has with a trimesh placed at `center`, emitted
/// in **ascending triangle index**.
///
/// The broadphase is the segment's AABB grown by `radius + CONTACT_MARGIN` — the
/// exact reach of a contact, so the box neither misses one nor collects
/// triangles that cannot produce one. There is no travel term in it because
/// there is nothing swept to cover: [`move_and_slide`] tests the *end* position
/// of each sub-step and re-collects after every push-out, and the sub-step cap
/// ([`SUBSTEP_RADIUS_FRACTION`]) is what covers the path between them.
fn segment_trimesh_contacts(
    mesh: &Trimesh,
    center: Vec3,
    seg_lo: Vec3,
    seg_hi: Vec3,
    radius: f32,
    scratch: &mut Vec<u32>,
    mut emit: impl FnMut(Vec3, f32, Vec3),
) {
    if mesh.nodes.is_empty() {
        return;
    }
    let lo = seg_lo - center;
    let hi = seg_hi - center;
    let reach = Vec3::splat(radius + CONTACT_MARGIN);
    mesh.query_aabb(lo.min(hi) - reach, lo.max(hi) + reach, scratch);
    for &index in scratch.iter() {
        if let Some((normal, depth, point)) = trimesh_contact(
            lo,
            hi,
            radius,
            mesh.triangle(index),
            mesh.face_normals[index as usize],
            mesh.edge_flags[index as usize],
        ) {
            emit(normal, depth, point + center);
        }
    }
}

// -- rays --------------------------------------------------------------------

/// Reciprocal that never becomes an infinity, so the slab test cannot produce
/// `0 · inf = NaN` on a ray that runs exactly along a box face.
#[inline]
fn safe_recip(x: f32) -> f32 {
    const TINY: f32 = 1e-20;
    if x.abs() < TINY {
        if x < 0.0 {
            -1.0 / TINY
        } else {
            1.0 / TINY
        }
    } else {
        1.0 / x
    }
}

/// Slab test against an axis-aligned box, returning the entry parameter.
#[inline]
fn ray_aabb_enter(origin: Vec3, inv_dir: Vec3, max_dist: f32, min: Vec3, max: Vec3) -> Option<f32> {
    let t0 = (min - origin) * inv_dir;
    let t1 = (max - origin) * inv_dir;
    let near = t0.min(t1);
    let far = t0.max(t1);
    let t_enter = near.max_element().max(0.0);
    let t_exit = far.min_element().min(max_dist);
    if t_enter <= t_exit {
        Some(t_enter)
    } else {
        None
    }
}

/// Möller–Trumbore, double-sided.
///
/// Double-sided because the box and sphere paths are: a ray that starts inside
/// geometry reports a hit rather than silently missing, and a soup's winding is
/// the *drawing's* business — a collider that answered "no hit" for a ray coming
/// from the wrong side would make a floor probe depend on which way the level's
/// exporter wound it.
fn ray_triangle(origin: Vec3, dir: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<f32> {
    let e1 = b - a;
    let e2 = c - a;
    let pvec = dir.cross(e2);
    let det = e1.dot(pvec);
    if det.abs() < 1e-12 {
        return None; // Parallel to the plane.
    }
    let inv_det = 1.0 / det;
    let tvec = origin - a;
    let u = tvec.dot(pvec) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let qvec = tvec.cross(e1);
    let v = dir.dot(qvec) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(qvec) * inv_det;
    if t < 0.0 {
        return None;
    }
    Some(t)
}

/// Nearest triangle a ray hits, as `(dist, normal)`.
///
/// Traversal descends the **near child first** — which child that is comes from
/// the sign of `dir` along the node's stored split axis, a pure function of the
/// ray — so the running best prunes the far subtree as often as possible. The
/// answer does not depend on that ordering: the winner is chosen on
/// `(t, triangle index)` lexicographically, so an exact tie always goes to the
/// lowest triangle index no matter which order the two were visited in.
fn ray_trimesh(
    mesh: &Trimesh,
    center: Vec3,
    origin: Vec3,
    dir: Vec3,
    max_dist: f32,
) -> Option<(f32, Vec3)> {
    if mesh.nodes.is_empty() {
        return None;
    }
    let origin = origin - center;
    let inv_dir = Vec3::new(safe_recip(dir.x), safe_recip(dir.y), safe_recip(dir.z));

    let mut best: Option<(f32, u32)> = None;
    let mut stack = [0u32; BVH_STACK_DEPTH];
    let mut depth = 0usize;
    stack[depth] = 0;
    depth += 1;

    while depth > 0 {
        depth -= 1;
        let node = &mesh.nodes[stack[depth] as usize];
        // Bounded by the running best, so a subtree that can only produce a
        // *later* hit is skipped — but `<=` inside the slab test keeps a subtree
        // that can produce an exactly-tying one, which is where the lowest-index
        // rule below has to be able to see it.
        let limit = best.map_or(max_dist, |(t, _)| t);
        if ray_aabb_enter(origin, inv_dir, limit, node.min, node.max).is_none() {
            continue;
        }

        if node.is_leaf() {
            for index in node.first..node.first + node.count {
                let [a, b, c] = mesh.triangle(index);
                if let Some(t) = ray_triangle(origin, dir, a, b, c) {
                    if t <= max_dist
                        && best.is_none_or(|(bt, bi)| t < bt || (t == bt && index < bi))
                    {
                        best = Some((t, index));
                    }
                }
            }
            continue;
        }

        assert!(
            depth + 2 <= BVH_STACK_DEPTH,
            "BVH traversal exceeded its fixed stack — the tree is not balanced"
        );
        // The left subtree holds the smaller centroids along `axis`, so a ray
        // travelling in `+axis` meets it first. Push the far child, then the
        // near one, so the near one pops first.
        let (near, far) = if dir[node.axis as usize] >= 0.0 {
            (node.first, node.right)
        } else {
            (node.right, node.first)
        };
        stack[depth] = far;
        stack[depth + 1] = near;
        depth += 2;
    }

    best.map(|(t, index)| {
        let n = mesh.face_normals[index as usize];
        // The reported normal faces back along the ray, as the box path's does.
        (t, if dir.dot(n) > 0.0 { -n } else { n })
    })
}

// ---------------------------------------------------------------------------
// The world snapshot
// ---------------------------------------------------------------------------

/// A collider's shape, resolved from whichever component the entity carried —
/// or, for [`Trimesh`], handed to [`CollisionWorld::push_collider`] directly.
///
/// Not `Copy`: a trimesh is shared by [`Arc`], never copied. Cloning an entry is
/// a refcount bump.
#[derive(Clone, Debug, PartialEq)]
pub enum ColliderShape {
    Sphere { radius: f32 },
    Aabb { half_extents: Vec3 },
    Obb { half_extents: Vec3, rotation: Quat },
    /// A static triangle soup, its vertices in the collider's own space and
    /// offset by [`ColliderEntry::center`]. There is no rotation: a trimesh is
    /// baked geometry that already has the orientation the level author gave
    /// it, and rotating one at runtime would be the dynamic case §9a refuses.
    Trimesh(Arc<Trimesh>),
}

/// One collider, snapshotted.
#[derive(Clone, Debug, PartialEq)]
pub struct ColliderEntry {
    pub entity: Entity,
    pub center: Vec3,
    pub shape: ColliderShape,
    pub memberships: u16,
    /// Carries [`Trigger`]: reported by the queries, never solid to the solver.
    pub trigger: bool,
}

/// One analytic terrain patch, snapshotted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainEntry {
    pub entity: Entity,
    pub surface: TerrainSurface,
    pub origin: Vec3,
    pub memberships: u16,
}

/// Every collider in the world, in `Entity` order.
///
/// A **value**, taken once and then read: the solver cannot observe a component
/// being mutated part way through a solve, and a game system is free to write
/// layers or transforms while holding one.
///
/// The scan is linear. At the scale runt is built for (a level is tens of
/// boxes) that is the right answer, and it is the seam a broadphase would slot
/// into: everything here funnels through [`CollisionWorld::for_each`], so a
/// spatial index becomes a change to one method rather than to five call sites.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CollisionWorld {
    colliders: Vec<ColliderEntry>,
    terrain: Vec<TerrainEntry>,
}

/// The read-only collider components [`CollisionWorld::gather`] wants.
pub type ColliderQuery = (
    Entity,
    Option<&'static SphereCollider>,
    Option<&'static AabbCollider>,
    Option<&'static ObbCollider>,
    &'static Transform,
    Option<&'static CollisionLayers>,
    Has<Trigger>,
);

/// The read-only terrain components [`CollisionWorld::gather`] wants.
pub type TerrainQuery = (
    Entity,
    &'static TerrainSurface,
    &'static Transform,
    Option<&'static CollisionLayers>,
);

impl CollisionWorld {
    pub fn new() -> CollisionWorld {
        CollisionWorld::default()
    }

    /// Snapshot from inside a `FixedSim` system.
    ///
    /// Both queries are generic in their **filter** so the caller can exclude
    /// the thing that is moving — a `&Transform` read here and a `&mut
    /// Transform` write on the player are the same component, and bevy refuses
    /// the pair without a `Without<…>` proving them disjoint. That is the same
    /// `Without<Ball>` shape `resolve_overlaps` already uses.
    ///
    /// ```ignore
    /// fn player_move(
    ///     colliders: Query<ColliderQuery, Without<Player>>,
    ///     terrain: Query<TerrainQuery, Without<Player>>,
    ///     mut player: Query<(&mut CharacterBody, &mut Transform, &mut Velocity), With<Player>>,
    /// ) {
    ///     let geometry = CollisionWorld::gather(&colliders, &terrain);
    ///     …
    /// }
    /// ```
    pub fn gather<Fc: bevy_ecs::query::QueryFilter, Ft: bevy_ecs::query::QueryFilter>(
        colliders: &Query<'_, '_, ColliderQuery, Fc>,
        terrain: &Query<'_, '_, TerrainQuery, Ft>,
    ) -> CollisionWorld {
        let mut world = CollisionWorld::new();
        for (entity, sphere, aabb, obb, transform, layers, trigger) in colliders.iter() {
            let memberships = layers.copied().unwrap_or_default().memberships;
            if let Some(shape) = resolve_shape(entity, sphere, aabb, obb, transform) {
                world.colliders.push(ColliderEntry {
                    entity,
                    center: transform.translation,
                    shape,
                    memberships,
                    trigger,
                });
            }
        }
        for (entity, surface, transform, layers) in terrain.iter() {
            world.terrain.push(TerrainEntry {
                entity,
                surface: *surface,
                origin: transform.translation,
                memberships: layers.copied().unwrap_or_default().memberships,
            });
        }
        world.finish();
        world
    }

    /// Snapshot from a bare [`World`] — what a test or a non-system caller
    /// wants. Identical output to [`gather`](CollisionWorld::gather).
    pub fn from_world(world: &mut World) -> CollisionWorld {
        let mut out = CollisionWorld::new();

        let mut colliders = world.query::<(
            Entity,
            Option<&SphereCollider>,
            Option<&AabbCollider>,
            Option<&ObbCollider>,
            &Transform,
            Option<&CollisionLayers>,
            Has<Trigger>,
        )>();
        for (entity, sphere, aabb, obb, transform, layers, trigger) in colliders.iter(world) {
            let memberships = layers.copied().unwrap_or_default().memberships;
            if let Some(shape) = resolve_shape(entity, sphere, aabb, obb, transform) {
                out.colliders.push(ColliderEntry {
                    entity,
                    center: transform.translation,
                    shape,
                    memberships,
                    trigger,
                });
            }
        }

        let mut terrain =
            world.query::<(Entity, &TerrainSurface, &Transform, Option<&CollisionLayers>)>();
        for (entity, surface, transform, layers) in terrain.iter(world) {
            out.terrain.push(TerrainEntry {
                entity,
                surface: *surface,
                origin: transform.translation,
                memberships: layers.copied().unwrap_or_default().memberships,
            });
        }

        out.finish();
        out
    }

    /// Add a collider by hand. For tests and for gameplay-owned geometry that
    /// never became an entity — and the **entry point for a [`Trimesh`]**, which
    /// has no component of its own yet (scene authoring is a later step; a
    /// baked soup is loaded by game code, built once, and pushed here).
    pub fn push_collider(&mut self, entry: ColliderEntry) -> &mut CollisionWorld {
        debug_assert!(
            !matches!(&entry.shape, ColliderShape::Trimesh(mesh) if mesh.triangle_count() == 0),
            "entity {} was given a trimesh collider with no triangles — \
             the soup welded away to nothing, which is an authoring bug rather \
             than an entity that happens to be intangible",
            entry.entity
        );
        self.colliders.push(entry);
        self.finish();
        self
    }

    pub fn push_terrain(&mut self, entry: TerrainEntry) -> &mut CollisionWorld {
        self.terrain.push(entry);
        self.finish();
        self
    }

    /// DESIGN §3: sort by `Entity` where ordering matters — and here it does,
    /// because a body wedged between two colliders resolves differently
    /// depending on which one it leaves first.
    fn finish(&mut self) {
        self.colliders.sort_unstable_by_key(|c| c.entity);
        self.terrain.sort_unstable_by_key(|t| t.entity);
    }

    pub fn colliders(&self) -> &[ColliderEntry] {
        &self.colliders
    }

    pub fn terrain(&self) -> &[TerrainEntry] {
        &self.terrain
    }

    pub fn is_empty(&self) -> bool {
        self.colliders.is_empty() && self.terrain.is_empty()
    }

    /// Visit every collider a `mask` can see, in `Entity` order. The one place
    /// the linear scan lives — a broadphase replaces this method and nothing
    /// else.
    fn for_each(&self, mask: u16, mut f: impl FnMut(&ColliderEntry)) {
        for entry in &self.colliders {
            if mask_accepts(mask, entry.memberships) {
                f(entry);
            }
        }
    }

    fn for_each_terrain(&self, mask: u16, mut f: impl FnMut(&TerrainEntry)) {
        for entry in &self.terrain {
            if mask_accepts(mask, entry.memberships) {
                f(entry);
            }
        }
    }
}

fn resolve_shape(
    entity: Entity,
    sphere: Option<&SphereCollider>,
    aabb: Option<&AabbCollider>,
    obb: Option<&ObbCollider>,
    transform: &Transform,
) -> Option<ColliderShape> {
    // An entity carrying two shapes is an authoring mistake. Same rule
    // `resolve_overlaps` applies, extended by one arm: sphere, then OBB (the
    // more specific box), then AABB.
    debug_assert!(
        [sphere.is_some(), aabb.is_some(), obb.is_some()]
            .iter()
            .filter(|present| **present)
            .count()
            <= 1,
        "entity {entity} carries more than one collider shape"
    );
    if let Some(s) = sphere {
        return Some(ColliderShape::Sphere { radius: s.radius });
    }
    if let Some(o) = obb {
        return Some(ColliderShape::Obb {
            half_extents: o.half_extents.abs(),
            rotation: o.rotation,
        });
    }
    if let Some(a) = aabb {
        debug_assert!(
            transform.rotation == Quat::IDENTITY,
            "AABB collider entities must be translation-only; \
             a rotated box wants an ObbCollider"
        );
        return Some(ColliderShape::Aabb {
            half_extents: a.half_extents.abs(),
        });
    }
    None
}

// ---------------------------------------------------------------------------
// Contact math
// ---------------------------------------------------------------------------

/// Exact signed distance from `p` to an origin-centered box of `half` extents.
/// Negative inside. Convex — which is what makes the ternary search below valid.
#[inline]
fn box_sdf(p: Vec3, half: Vec3) -> f32 {
    let q = p.abs() - half;
    q.max(Vec3::ZERO).length() + q.max_element().min(0.0)
}

/// [`box_sdf`] plus the surface data a contact needs: the outward unit normal at
/// `p` and the nearest point on the box.
fn box_surface(p: Vec3, half: Vec3) -> (f32, Vec3, Vec3) {
    let q = p.abs() - half;
    if q.max_element() > 0.0 {
        // Outside: the nearest point is the clamp, and the normal points at `p`.
        let closest = p.clamp(-half, half);
        let delta = p - closest;
        let dist = delta.length();
        let normal = delta.try_normalize().unwrap_or(Vec3::Y);
        (dist, normal, closest)
    } else {
        // Inside (or exactly on the surface): leave by the nearest face, i.e.
        // the axis of least penetration. Ties take the lowest axis index, which
        // is arbitrary but fixed — determinism, not physics.
        let gap = -q;
        let axis = if gap.x <= gap.y && gap.x <= gap.z {
            0
        } else if gap.y <= gap.z {
            1
        } else {
            2
        };
        let sign = if p[axis] < 0.0 { -1.0 } else { 1.0 };
        let mut normal = Vec3::ZERO;
        normal[axis] = sign;
        let mut closest = p;
        closest[axis] = sign * half[axis];
        (-gap[axis], normal, closest)
    }
}

/// The parameter `t ∈ [0,1]` along `a→b` that **minimises** the box's signed
/// distance.
///
/// Outside the box that is the closest point; inside it is the *deepest* point,
/// which is the one push-out has to clear. One search covers both because the
/// signed distance to a convex set is a convex function, so its restriction to a
/// segment is convex and ternary search is exact in the limit.
///
/// Fixed [`SEGMENT_SEARCH_ITERATIONS`] with a bracket-width early exit — both
/// are pure functions of the inputs, so two machines that agree on `f32`
/// arithmetic agree on the answer.
fn deepest_on_segment(a: Vec3, b: Vec3, half: Vec3) -> f32 {
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    for _ in 0..SEGMENT_SEARCH_ITERATIONS {
        let third = (hi - lo) / 3.0;
        if third <= 1e-7 {
            break;
        }
        let m1 = lo + third;
        let m2 = hi - third;
        if box_sdf(a.lerp(b, m1), half) <= box_sdf(a.lerp(b, m2), half) {
            hi = m2;
        } else {
            lo = m1;
        }
    }
    (lo + hi) * 0.5
}

/// Contact between a vertical swept sphere (`seg_lo`..`seg_hi`, `radius`) and a
/// box.
///
/// `rotation` is the box's orientation; [`Quat::IDENTITY`] takes the closed-form
/// path. Returns `None` when the separation exceeds [`CONTACT_MARGIN`].
fn segment_box_contact(
    seg_lo: Vec3,
    seg_hi: Vec3,
    radius: f32,
    center: Vec3,
    rotation: Quat,
    half: Vec3,
) -> Option<(Vec3, f32, Vec3)> {
    let axis_aligned = rotation == Quat::IDENTITY;
    let (a, b) = if axis_aligned {
        (seg_lo - center, seg_hi - center)
    } else {
        let inv = rotation.inverse();
        (inv * (seg_lo - center), inv * (seg_hi - center))
    };

    let local = if axis_aligned {
        // The segment is vertical and the box is axis-aligned, so the signed
        // distance depends on `y` only through `|y| - half.y` — and it is
        // non-decreasing in that. Minimising it is therefore just clamping the
        // box's own centre height into the segment. Exact, no search.
        let y = 0.0f32.clamp(a.y.min(b.y), a.y.max(b.y));
        Vec3::new(a.x, y, a.z)
    } else {
        a.lerp(b, deepest_on_segment(a, b, half))
    };

    let (sdf, normal_local, closest_local) = box_surface(local, half);
    let depth = radius - sdf;
    if depth <= -CONTACT_MARGIN {
        return None;
    }
    let (normal, point) = if axis_aligned {
        (normal_local, closest_local + center)
    } else {
        (rotation * normal_local, rotation * closest_local + center)
    };
    Some((normal, depth, point))
}

/// Contact between a vertical swept sphere and a sphere.
fn segment_sphere_contact(
    seg_lo: Vec3,
    seg_hi: Vec3,
    radius: f32,
    center: Vec3,
    other_radius: f32,
) -> Option<(Vec3, f32, Vec3)> {
    // Closest point on a segment to a point: the usual projection, and the
    // segment is vertical so the clamp is on `y` alone.
    let closest = Vec3::new(
        seg_lo.x,
        center.y.clamp(seg_lo.y.min(seg_hi.y), seg_lo.y.max(seg_hi.y)),
        seg_lo.z,
    );
    let reach = radius + other_radius;
    let delta = closest - center;
    let dist = delta.length();
    let depth = reach - dist;
    if depth <= -CONTACT_MARGIN {
        return None;
    }
    // Concentric: no direction is more right than another, so pick the one that
    // will not push a body into the floor. Same rule `physics::overlap` uses.
    let normal = delta.try_normalize().unwrap_or(Vec3::Y);
    Some((normal, depth, center + normal * other_radius))
}

/// Contact between a vertical swept sphere and an analytic height field.
///
/// One field evaluation, at the segment's `(x, z)` — a vertical capsule has only
/// one — giving height *and* gradient, exactly as
/// [`integrate_balls`](crate::physics::integrate_balls) does. The contact is
/// against the **tangent plane** there rather than the vertical column, so a
/// slope holds the body at the right distance instead of over-lifting it by
/// `1/cos θ`; the same plane is what turns a steep field region into a wall,
/// since its normal is then nearly horizontal and classification follows.
///
/// A plane's signed distance is linear along a segment, so the deepest end is
/// simply whichever endpoint has the smaller dot product — no search.
///
/// Limitation, stated rather than hidden: the sample is taken under the body's
/// axis, so on a strongly curved field the contact point is approximate (the
/// solve re-samples each iteration, which converges). A height field has no
/// overhangs, so the head cap can never be the only thing touching.
fn segment_field_contact(
    entry: &TerrainEntry,
    seg_lo: Vec3,
    seg_hi: Vec3,
    radius: f32,
) -> Option<(Vec3, f32, Vec3)> {
    let (x, z) = (seg_lo.x, seg_lo.z);
    if !entry.surface.contains_world(entry.origin, x, z) {
        return None;
    }
    let (height, grad) = entry.surface.sample_world(entry.origin, x, z);
    let normal = runt_mesh::terrain::normal_from_gradient(grad);
    let on_surface = Vec3::new(x, height, z);

    let d_lo = (seg_lo - on_surface).dot(normal);
    let d_hi = (seg_hi - on_surface).dot(normal);
    let (dist, deepest) = if d_lo <= d_hi {
        (d_lo, seg_lo)
    } else {
        (d_hi, seg_hi)
    };
    let depth = radius - dist;
    if depth <= -CONTACT_MARGIN {
        return None;
    }
    Some((normal, depth, deepest - normal * radius))
}

/// Every contact a body at `position` has, in `Entity` order.
///
/// `solid_only` drops [`Trigger`] colliders — what the solver wants; the overlap
/// queries keep them and flag them instead.
///
/// A convex collider contributes at most one contact. A [`Trimesh`] contributes
/// **one per penetrating triangle**, in ascending triangle index, and that is
/// deliberate: a body wedged into the inside corner of two triangles must have
/// its velocity projected against both planes, exactly as a body between two
/// boxes does. The set is folded down to one contact per entity later
/// ([`merge_contacts`]), so what a caller reads back is unchanged.
fn collect_contacts(
    world: &CollisionWorld,
    shape: CharacterShape,
    up: Vec3,
    position: Vec3,
    mask: u16,
    solid_only: bool,
    out: &mut Vec<RawContact>,
) {
    out.clear();
    let (lo, hi) = shape.segment(position, up);
    let radius = shape.radius();
    let mut candidates: Vec<u32> = Vec::new();

    world.for_each(mask, |entry| {
        if solid_only && entry.trigger {
            return;
        }
        let hit = match &entry.shape {
            ColliderShape::Sphere { radius: r } => {
                segment_sphere_contact(lo, hi, radius, entry.center, *r)
            }
            ColliderShape::Aabb { half_extents } => segment_box_contact(
                lo,
                hi,
                radius,
                entry.center,
                Quat::IDENTITY,
                *half_extents,
            ),
            ColliderShape::Obb {
                half_extents,
                rotation,
            } => segment_box_contact(lo, hi, radius, entry.center, *rotation, *half_extents),
            ColliderShape::Trimesh(mesh) => {
                segment_trimesh_contacts(
                    mesh,
                    entry.center,
                    lo,
                    hi,
                    radius,
                    &mut candidates,
                    |normal, depth, point| {
                        out.push(RawContact {
                            entity: entry.entity,
                            normal,
                            point,
                            depth,
                        });
                    },
                );
                None
            }
        };
        if let Some((normal, depth, point)) = hit {
            out.push(RawContact {
                entity: entry.entity,
                normal,
                point,
                depth,
            });
        }
    });

    world.for_each_terrain(mask, |entry| {
        if let Some((normal, depth, point)) = segment_field_contact(entry, lo, hi, radius) {
            out.push(RawContact {
                entity: entry.entity,
                normal,
                point,
                depth,
            });
        }
    });
}

// ---------------------------------------------------------------------------
// The solver
// ---------------------------------------------------------------------------

/// Move a kinematic body by `velocity · dt`, sliding along whatever it hits.
///
/// The Godot `move_and_slide` analog, and the one entry point a state machine
/// needs per tick.
///
/// ## Order of operations
///
/// ```text
/// sub-steps  n = clamp(ceil(|v·dt| / (radius · SUBSTEP_RADIUS_FRACTION)), 1, MAX_SUBSTEPS)
/// per sub-step:
///   translate      p += v · (dt/n)
///   for ≤ SLIDE_ITERATIONS:
///     collect      every contact at p, layer-filtered, Entity-sorted
///     record       merge into the result set (deepest per entity)
///     stop         floor contact + gravity-only v → v = 0, and undo the drop
///                  when the floor absorbed the whole of it
///     project      for each contact in Entity order: v -= n·min(v·n, 0)
///     push out     p += n_deepest · depth_deepest      (deepest wins, ties → lowest Entity)
///     stop         when nothing is penetrating by more than PUSH_EPSILON
/// floor snap  (once, after the sub-steps)
/// classify    floor / wall / ceiling against max_floor_angle
/// ```
///
/// Push-out resolves **one** contact per iteration and then re-collects, rather
/// than resolving all of them at once: leaving a wall changes what the floor
/// contact is, and a simultaneous solve would double-count the corner. Velocity
/// is projected against *every* contact in the iteration, though — a body in a
/// corner cannot move into either wall, and projecting only the deepest would
/// let it creep into the other.
///
/// ## Stop on slope
///
/// Projecting velocity onto the floor plane is right for a body that is *going*
/// somewhere and wrong for one that is not: the tick's gravity comes back as a
/// downhill tangential velocity, and a body standing still on a slope it is
/// allowed to stand on creeps down it. Godot's `floor_stop_on_slope` (default
/// true, and default true here) is the answer, and the condition is narrow —
/// on floor, and the velocity is *gravity only*, straight down within
/// [`STOP_ON_SLOPE_TILT`]. The velocity then goes, and the motion with it when
/// the floor absorbed the whole sub-step — see [`slope_stop`] for when it does
/// not, and why that case has to be left alone.
///
/// The narrowness is what makes it safe. A body with any horizontal intent —
/// walking, running, sliding after a landing — fails the direction test and
/// moves exactly as it did before. A contact past `max_floor_angle` is a wall,
/// never floor, so a body on a face too steep to stand on still slides down it.
///
/// ## Floor snap
///
/// When `snap_length > 0`, the body **was** on the floor at the end of the
/// previous call, it is not on the floor now, and it is not moving up: probe
/// down. Godot's rule, and it is what keeps a runner glued to a descending ramp
/// instead of launching off every crest.
///
/// The probe is analytic rather than a search. Drop the body by `snap_length`,
/// take the deepest floor-classified contact there, and undo exactly the part of
/// the drop that penetration accounts for: moving back up by `δ` along `up`
/// reduces penetration by `δ · (n·up)`, so the body lands at
/// `snap_length - depth/(n·up)`. Exact for a planar surface, and the ordinary
/// push-out loop runs afterwards to settle anything curved.
///
/// The snap moves the body **straight down along `up`** — never along the
/// contact normal, which on a 17° ramp would slide it 0.13 m sideways for a
/// 0.04 m drop. Horizontal velocity is untouched (that is the whole point);
/// the component heading *into* the floor is projected out, exactly as a real
/// contact would have done, so a grounded body does not accumulate fall speed.
///
/// ## Writes to `body`
///
/// Only `body.on_floor`, and only at the very end — it is the memory the *next*
/// call's snap needs. Everything else the solver reads it never writes.
pub fn move_and_slide(
    world: &CollisionWorld,
    body: &mut CharacterBody,
    position: Vec3,
    velocity: Vec3,
    dt: f32,
) -> MoveResult {
    // A trimesh is static world geometry and never the thing being moved: the
    // moving shape is a `CharacterShape`, which has no trimesh form, so the body
    // cannot *be* one by construction. What is still reachable is a snapshot
    // holding an empty one — a soup that welded away to nothing — which would
    // read as a solid collider that is silently intangible.
    debug_assert!(
        !world
            .colliders
            .iter()
            .any(|c| matches!(&c.shape, ColliderShape::Trimesh(m) if m.triangle_count() == 0)),
        "a trimesh collider in this snapshot has no triangles"
    );

    let up = body.up.try_normalize().unwrap_or(Vec3::Y);
    let radius = body.shape.radius();
    if !dt.is_finite() || dt <= 0.0 || radius <= 0.0 {
        let mut result = MoveResult::empty(position, velocity);
        result.on_floor = body.on_floor;
        return result;
    }

    let was_on_floor = body.on_floor;
    let mask = body.layers.mask;
    let cos_max = body.max_floor_angle.cos();

    // -- sub-stepping ------------------------------------------------------
    //
    // Derived from the *entry* velocity alone: what the body then collides with
    // must not be able to change how many steps the tick took.
    let step_cap = radius * SUBSTEP_RADIUS_FRACTION;
    let distance = velocity.length() * dt;
    let sub_steps = if step_cap > 0.0 && distance > step_cap {
        ((distance / step_cap).ceil() as u32).clamp(1, MAX_SUBSTEPS)
    } else {
        1
    };
    let sub_dt = dt / sub_steps as f32;

    let mut p = position;
    let mut v = velocity;
    let mut found: Vec<RawContact> = Vec::new();
    let mut merged: Vec<RawContact> = Vec::new();
    let mut stopped_on_slope = false;

    'sub_steps: for _ in 0..sub_steps {
        let step_start = p;
        p += v * sub_dt;
        for _ in 0..SLIDE_ITERATIONS {
            collect_contacts(world, body.shape, up, p, mask, true, &mut found);
            if found.is_empty() {
                break;
            }
            merge_contacts(&mut merged, &found);

            // Stop on slope, before the projection that would create the slide.
            if body.floor_stop_on_slope && gravity_only(v, up) {
                match slope_stop(&found, up, cos_max, step_start, p) {
                    SlopeStop::Sliding => {}
                    SlopeStop::Cancelled(rest) => {
                        // Godot's `break` out of the whole slide loop, and for
                        // the same reason: there is nothing left to resolve once
                        // the motion that caused the contact has been taken back.
                        p = rest;
                        v = Vec3::ZERO;
                        stopped_on_slope = true;
                        break 'sub_steps;
                    }
                    SlopeStop::Settling => {
                        // The velocity goes, the position is left to the ordinary
                        // push-out below. `v` is zero from here, so the projection
                        // is a no-op and the sub-step resolves exactly as it did
                        // before this flag existed.
                        v = Vec3::ZERO;
                        stopped_on_slope = true;
                    }
                }
            }

            // Velocity first, against every contact, in Entity order.
            for contact in &found {
                let into = v.dot(contact.normal);
                if into < 0.0 {
                    v -= contact.normal * into;
                }
            }

            let Some(deepest) = deepest_contact(&found) else {
                break;
            };
            if deepest.depth <= PUSH_EPSILON {
                break;
            }
            p += deepest.normal * deepest.depth;
        }
    }

    let mut snapped = false;
    if body.snap_length > 0.0
        && was_on_floor
        && v.dot(up) <= 0.0
        && !merged
            .iter()
            .any(|c| classify(c.normal, up, cos_max) == ContactKind::Floor)
    {
        // A body that walked off a crest and is now falling straight down is the
        // same gravity-only case the sub-step loop stops: the floor the probe
        // finds is a floor it was standing on, so the snap must put it back on
        // that floor at rest rather than hand back the tangential remainder.
        let stopping = body.floor_stop_on_slope && gravity_only(v, up);
        if let Some((snap_p, snap_v)) = snap_to_floor(world, body, up, cos_max, p, v, &mut found) {
            p = snap_p;
            v = if stopping { Vec3::ZERO } else { snap_v };
            stopped_on_slope |= stopping;
            snapped = true;
            collect_contacts(world, body.shape, up, p, mask, true, &mut found);
            merge_contacts(&mut merged, &found);
        }
    }

    // -- classify ----------------------------------------------------------
    merged.sort_unstable_by_key(|c| c.entity);
    let mut contacts: Vec<Contact> = Vec::with_capacity(merged.len());
    let mut result = MoveResult::empty(p, v);
    result.sub_steps = sub_steps;
    result.snapped = snapped;
    result.stopped_on_slope = stopped_on_slope;

    let mut best_floor_dot = f32::NEG_INFINITY;
    let mut deepest_wall = f32::NEG_INFINITY;
    let mut deepest_ceiling = f32::NEG_INFINITY;
    for raw in &merged {
        let kind = classify(raw.normal, up, cos_max);
        match kind {
            ContactKind::Floor => {
                // The most upright floor contact wins — a ledge's rounded corner
                // must not out-vote the ground the body is standing on. Ties go
                // to the lowest Entity because `merged` is sorted and the
                // comparison is strict.
                let dot = raw.normal.dot(up);
                if dot > best_floor_dot {
                    best_floor_dot = dot;
                    result.on_floor = true;
                    result.floor_normal = raw.normal;
                    result.floor_angle = dot.clamp(-1.0, 1.0).acos();
                }
            }
            ContactKind::Wall => {
                if raw.depth > deepest_wall {
                    deepest_wall = raw.depth;
                    result.on_wall = true;
                    result.wall_normal = raw.normal;
                }
            }
            ContactKind::Ceiling => {
                if raw.depth > deepest_ceiling {
                    deepest_ceiling = raw.depth;
                    result.on_ceiling = true;
                    result.ceiling_normal = raw.normal;
                }
            }
        }
        contacts.push(Contact {
            entity: raw.entity,
            normal: raw.normal,
            point: raw.point,
            depth: raw.depth,
            kind,
        });
    }
    result.contacts = contacts;

    body.on_floor = result.on_floor;
    result
}

/// Godot's classification, restated: within `max_floor_angle` of up is floor,
/// within `max_floor_angle` of down is ceiling, everything between is wall.
///
/// At `max_floor_angle == PI` the first test is `dot >= -1`, which every unit
/// normal satisfies — so *everything* is floor, which is exactly what a rolling
/// body wants and what PORT_SPEC asks for.
#[inline]
fn classify(normal: Vec3, up: Vec3, cos_max: f32) -> ContactKind {
    let dot = normal.dot(up).clamp(-1.0, 1.0);
    if dot >= cos_max {
        ContactKind::Floor
    } else if dot <= -cos_max {
        ContactKind::Ceiling
    } else {
        ContactKind::Wall
    }
}

/// Is this velocity nothing but the tick's gravity — pointing straight *down*,
/// within [`STOP_ON_SLOPE_TILT`]?
///
/// Godot spells it `(velocity.normalized() + up_direction).length() < 0.01`, and
/// the degenerate case matters: a zero velocity normalises to zero, so the sum
/// is `up`, whose length is 1, and a body that is already at rest never takes
/// the stop path. Nothing to cancel, so nothing is cancelled.
#[inline]
fn gravity_only(velocity: Vec3, up: Vec3) -> bool {
    (velocity.normalize_or_zero() + up).length_squared() < STOP_ON_SLOPE_TILT * STOP_ON_SLOPE_TILT
}

/// What [`CharacterBody::floor_stop_on_slope`] does with a sub-step that ran the
/// body into something under gravity alone.
enum SlopeStop {
    /// Nothing floor-classified was hit. Steep faces are walls, walls do not
    /// stop a body, and this one keeps sliding.
    Sliding,
    /// The floor absorbed the whole sub-step: the body had nowhere to go. Both
    /// the motion and the velocity are taken back, and the position carried is
    /// the one the sub-step started from — *exactly*, which is what makes a
    /// standing body bit-stable tick after tick.
    Cancelled(Vec3),
    /// The body did go somewhere — it fell onto the floor, or it began the
    /// sub-step penetrating and is still coming out. Only the velocity is
    /// cancelled; where the body ends up is left to the ordinary push-out.
    Settling,
}

/// Read a sub-step against Godot's stop-on-slope rule.
///
/// Godot's own line is
/// `if (result.travel.length() <= margin) gt.origin -= result.travel;` — undo
/// the move when the swept solve found the body had nowhere to go, and otherwise
/// keep the travel, which for a straight-down sweep means resting on the surface
/// just reached. A teleport-then-push-out solver has no sweep, so the same
/// reading is reconstructed from the contact:
///
/// - `lift` is how far back along `up` the deepest floor contact has to be
///   undone to stop penetrating — the analytic un-drop [`snap_to_floor`] uses,
///   exact for a plane. Along `up` rather than along the contact normal is the
///   whole point: a normal-directed push-out walks a body sideways up the slope,
///   one drop's worth of `sin θ · cos θ` every tick.
/// - `descent − lift` is therefore the travel that actually happened, the same
///   quantity Godot compares against its margin.
///
/// The lower bound on that comparison is what keeps this honest. A *negative*
/// gap means the body was already penetrating before the sub-step, and cancelling
/// back to there would freeze the overlap in place for good — a body resting on
/// the floor would read as inside it, and every query built on that (the port's
/// phase guard, for one) would be wrong forever. Those sub-steps settle through
/// the ordinary push-out instead, which converges in a tick or two, and the
/// cancel then holds a position the solver had already produced on its own.
fn slope_stop(
    found: &[RawContact],
    up: Vec3,
    cos_max: f32,
    step_start: Vec3,
    p: Vec3,
) -> SlopeStop {
    let descent = (step_start - p).dot(up);
    if descent <= 0.0 {
        return SlopeStop::Sliding;
    }

    let mut lift = f32::NEG_INFINITY;
    for c in found {
        if classify(c.normal, up, cos_max) != ContactKind::Floor {
            continue;
        }
        // A floor whose normal is perpendicular to `up` (only reachable at
        // max_floor_angle = 180°) gives no purchase to lift back along `up`.
        let n_up = c.normal.dot(up);
        if n_up <= 1e-3 {
            continue;
        }
        lift = lift.max(c.depth / n_up);
    }
    if lift == f32::NEG_INFINITY {
        return SlopeStop::Sliding;
    }

    let gap = descent - lift;
    if (0.0..CONTACT_MARGIN).contains(&gap) {
        SlopeStop::Cancelled(step_start)
    } else {
        SlopeStop::Settling
    }
}

/// Deepest contact; an exact tie goes to the lowest `Entity`, which `found` is
/// already ordered by.
fn deepest_contact(found: &[RawContact]) -> Option<RawContact> {
    let mut best: Option<RawContact> = None;
    for c in found {
        if best.is_none_or(|b| c.depth > b.depth) {
            best = Some(*c);
        }
    }
    best
}

/// Fold this iteration's contacts into the accumulated set, keeping the deepest
/// per entity. Linear search: the set is a handful of entries and a map would be
/// a hash container in the middle of a deterministic solve.
fn merge_contacts(merged: &mut Vec<RawContact>, found: &[RawContact]) {
    for c in found {
        match merged.iter_mut().find(|m| m.entity == c.entity) {
            Some(existing) if c.depth > existing.depth => *existing = *c,
            Some(_) => {}
            None => merged.push(*c),
        }
    }
}

/// The floor-snap probe. See [`move_and_slide`] for the argument.
fn snap_to_floor(
    world: &CollisionWorld,
    body: &CharacterBody,
    up: Vec3,
    cos_max: f32,
    position: Vec3,
    velocity: Vec3,
    scratch: &mut Vec<RawContact>,
) -> Option<(Vec3, Vec3)> {
    let probe = position - up * body.snap_length;
    collect_contacts(
        world,
        body.shape,
        up,
        probe,
        body.layers.mask,
        true,
        scratch,
    );

    let mut best: Option<RawContact> = None;
    for c in scratch.iter() {
        if classify(c.normal, up, cos_max) != ContactKind::Floor {
            continue;
        }
        // A floor whose normal is perpendicular to `up` (only reachable at
        // max_floor_angle = 180°) gives no purchase to lift back along `up`.
        if c.normal.dot(up) <= 1e-3 {
            continue;
        }
        if best.is_none_or(|b| c.depth > b.depth) {
            best = Some(*c);
        }
    }
    let best = best?;

    let lift = (best.depth / best.normal.dot(up)).clamp(0.0, body.snap_length);
    let mut p = probe + up * lift;
    let mut v = velocity;

    // Settle: a planar surface is already exact, a curved field is not.
    for _ in 0..SLIDE_ITERATIONS {
        collect_contacts(
            world,
            body.shape,
            up,
            p,
            body.layers.mask,
            true,
            scratch,
        );
        let Some(deepest) = deepest_contact(scratch) else {
            break;
        };
        if deepest.depth <= PUSH_EPSILON {
            break;
        }
        p += deepest.normal * deepest.depth;
    }

    // The snap must never lift the body above where the ordinary solve left it —
    // that would be a step *up*, which is a different feature.
    if (p - position).dot(up) > CONTACT_MARGIN {
        return None;
    }
    // Horizontal velocity survives untouched; only the part driving into the
    // floor is removed, as the contact would have done had the body reached it.
    let into = v.dot(best.normal);
    if into < 0.0 {
        v -= best.normal * into;
    }
    Some((p, v))
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

/// One overlapping collider.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OverlapHit {
    pub entity: Entity,
    /// Penetration depth along `normal`, `> 0`.
    pub depth: f32,
    /// Unit normal pointing out of the collider, towards the query shape.
    pub normal: Vec3,
    /// The point on the collider's surface.
    pub point: Vec3,
    /// The collider carries [`Trigger`]. Queries report triggers rather than
    /// hiding them — the phase entry guard wants solids only and says so; a
    /// pickup sweep wants the opposite.
    pub trigger: bool,
}

/// One ray hit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    pub entity: Entity,
    pub point: Vec3,
    /// Unit surface normal at `point`, pointing back along the ray's approach.
    pub normal: Vec3,
    /// Distance from the ray origin, in world units.
    pub dist: f32,
    pub trigger: bool,
}

impl CollisionWorld {
    /// Every collider a sphere overlaps, in `Entity` order.
    ///
    /// PORT_SPEC's phase entry guard and unphase snap-back are both this call
    /// with a different mask.
    pub fn overlap_sphere(&self, center: Vec3, radius: f32, mask: u16) -> Vec<OverlapHit> {
        self.overlap_segment(center, center, radius, mask)
    }

    /// Every collider a vertical capsule overlaps, in `Entity` order. `height`
    /// is total height including caps, as in [`CharacterShape::Capsule`].
    pub fn overlap_capsule(
        &self,
        center: Vec3,
        radius: f32,
        height: f32,
        mask: u16,
    ) -> Vec<OverlapHit> {
        let half = (height * 0.5 - radius).max(0.0);
        self.overlap_segment(
            center - Vec3::Y * half,
            center + Vec3::Y * half,
            radius,
            mask,
        )
    }

    /// Every collider the shape a [`CharacterBody`] would occupy at `position`
    /// overlaps — the capsule↔sphere swap without restating the shape.
    pub fn overlap_body(&self, body: &CharacterBody, position: Vec3) -> Vec<OverlapHit> {
        let (lo, hi) = body.segment(position);
        self.overlap_segment(lo, hi, body.shape.radius(), body.layers.mask)
    }

    fn overlap_segment(&self, lo: Vec3, hi: Vec3, radius: f32, mask: u16) -> Vec<OverlapHit> {
        let mut hits = Vec::new();
        let push = |entity: Entity,
                        trigger: bool,
                        hit: Option<(Vec3, f32, Vec3)>,
                        hits: &mut Vec<OverlapHit>| {
            // Overlap means *overlap*: unlike the solver, a query does not want
            // the touching-tolerance band reported as a hit.
            if let Some((normal, depth, point)) = hit {
                if depth > 0.0 {
                    hits.push(OverlapHit {
                        entity,
                        depth,
                        normal,
                        point,
                        trigger,
                    });
                }
            }
        };

        let mut candidates: Vec<u32> = Vec::new();
        self.for_each(mask, |entry| {
            let hit = match &entry.shape {
                ColliderShape::Sphere { radius: r } => {
                    segment_sphere_contact(lo, hi, radius, entry.center, *r)
                }
                ColliderShape::Aabb { half_extents } => segment_box_contact(
                    lo,
                    hi,
                    radius,
                    entry.center,
                    Quat::IDENTITY,
                    *half_extents,
                ),
                ColliderShape::Obb {
                    half_extents,
                    rotation,
                } => segment_box_contact(lo, hi, radius, entry.center, *rotation, *half_extents),
                // One hit per *collider*, as every other shape reports: the
                // deepest triangle is what the overlap is. Ties go to the lowest
                // triangle index, which is the order they arrive in.
                ColliderShape::Trimesh(mesh) => {
                    let mut best: Option<(Vec3, f32, Vec3)> = None;
                    segment_trimesh_contacts(
                        mesh,
                        entry.center,
                        lo,
                        hi,
                        radius,
                        &mut candidates,
                        |normal, depth, point| {
                            if best.is_none_or(|(_, d, _)| depth > d) {
                                best = Some((normal, depth, point));
                            }
                        },
                    );
                    best
                }
            };
            push(entry.entity, entry.trigger, hit, &mut hits);
        });
        self.for_each_terrain(mask, |entry| {
            let hit = segment_field_contact(entry, lo, hi, radius);
            push(entry.entity, false, hit, &mut hits);
        });
        hits
    }

    /// The nearest thing a ray hits, or `None`.
    ///
    /// `dir` need not be normalized; `dist` is in world units either way. Boxes
    /// use an exact slab test in the box's own frame, spheres the usual
    /// quadratic, trimeshes a near-child-first BVH descent with Möller–Trumbore
    /// at the leaves, and the analytic height field a fixed-step march refined
    /// by bisection — see [`RAY_MARCH_STEP`]. The field is sampled, never its
    /// mesh, so the same ray returns the same hit at every quality tier.
    ///
    /// PORT_SPEC's ledge vault (a head ray that must miss and a chest ray that
    /// must hit) and the air pulse's wall find are both this call.
    pub fn raycast(&self, origin: Vec3, dir: Vec3, max_dist: f32, mask: u16) -> Option<RayHit> {
        let dir = dir.try_normalize()?;
        // `is_finite` as well as positive: a NaN length is a caller bug, and a
        // ray of NaN length must miss rather than march.
        if !max_dist.is_finite() || max_dist <= 0.0 {
            return None;
        }
        let mut best: Option<RayHit> = None;
        let consider = |hit: RayHit, best: &mut Option<RayHit>| {
            // Strict `<`: an exact tie keeps the earlier entity, and the scan is
            // in Entity order.
            if best.is_none_or(|b| hit.dist < b.dist) {
                *best = Some(hit);
            }
        };

        self.for_each(mask, |entry| {
            let found = match &entry.shape {
                ColliderShape::Sphere { radius } => {
                    ray_sphere(origin, dir, max_dist, entry.center, *radius)
                }
                ColliderShape::Aabb { half_extents } => {
                    ray_box(origin, dir, max_dist, entry.center, Quat::IDENTITY, *half_extents)
                }
                ColliderShape::Obb {
                    half_extents,
                    rotation,
                } => ray_box(origin, dir, max_dist, entry.center, *rotation, *half_extents),
                ColliderShape::Trimesh(mesh) => {
                    ray_trimesh(mesh, entry.center, origin, dir, max_dist)
                }
            };
            if let Some((dist, normal)) = found {
                consider(
                    RayHit {
                        entity: entry.entity,
                        point: origin + dir * dist,
                        normal,
                        dist,
                        trigger: entry.trigger,
                    },
                    &mut best,
                );
            }
        });
        self.for_each_terrain(mask, |entry| {
            if let Some((dist, normal)) = ray_field(entry, origin, dir, max_dist) {
                consider(
                    RayHit {
                        entity: entry.entity,
                        point: origin + dir * dist,
                        normal,
                        dist,
                        trigger: false,
                    },
                    &mut best,
                );
            }
        });
        best
    }
}

/// Slab test in the box's own frame. Exact.
fn ray_box(
    origin: Vec3,
    dir: Vec3,
    max_dist: f32,
    center: Vec3,
    rotation: Quat,
    half: Vec3,
) -> Option<(f32, Vec3)> {
    let identity = rotation == Quat::IDENTITY;
    let inv = rotation.inverse();
    let o = if identity {
        origin - center
    } else {
        inv * (origin - center)
    };
    let d = if identity { dir } else { inv * dir };

    let mut t_enter = 0.0f32;
    let mut t_exit = max_dist;
    let mut axis = usize::MAX;
    let mut sign = 1.0f32;

    for i in 0..3 {
        if d[i].abs() < 1e-8 {
            // Parallel to this slab: either always inside it or never.
            if o[i] < -half[i] || o[i] > half[i] {
                return None;
            }
            continue;
        }
        let inv_d = 1.0 / d[i];
        let t_lo = (-half[i] - o[i]) * inv_d;
        let t_hi = (half[i] - o[i]) * inv_d;
        let (near, far, near_sign) = if t_lo <= t_hi {
            (t_lo, t_hi, -1.0)
        } else {
            (t_hi, t_lo, 1.0)
        };
        if near > t_enter {
            t_enter = near;
            axis = i;
            sign = near_sign;
        }
        if far < t_exit {
            t_exit = far;
        }
        if t_enter > t_exit {
            return None;
        }
    }

    // Origin inside the box: report the start of the ray. There is no entry
    // face to take a normal from, so hand back the one that faces the ray.
    if axis == usize::MAX {
        return Some((0.0, -dir));
    }
    let mut normal_local = Vec3::ZERO;
    normal_local[axis] = sign;
    let normal = if identity {
        normal_local
    } else {
        rotation * normal_local
    };
    Some((t_enter, normal))
}

fn ray_sphere(
    origin: Vec3,
    dir: Vec3,
    max_dist: f32,
    center: Vec3,
    radius: f32,
) -> Option<(f32, Vec3)> {
    let m = origin - center;
    let b = m.dot(dir);
    let c = m.length_squared() - radius * radius;
    if c <= 0.0 {
        // Started inside.
        return Some((0.0, -dir));
    }
    if b > 0.0 {
        return None; // Pointing away.
    }
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let t = -b - disc.sqrt();
    if t < 0.0 || t > max_dist {
        return None;
    }
    let point = origin + dir * t;
    Some((t, (point - center).try_normalize().unwrap_or(-dir)))
}

/// Fixed-step march over the analytic field, then bisection on the bracketed
/// crossing.
///
/// `f(t) = ray(t).y − h(ray(t).xz)` is positive above the surface and negative
/// below it. The march walks `t` in [`RAY_MARCH_STEP`] increments looking for
/// the first sign change with both ends inside the patch, then
/// [`RAY_BISECTIONS`] halvings pin it down. Both counts are constants, so the
/// answer is a pure function of the ray and the field — and the field is what
/// the mesh is generated *from*, so tessellation cannot move the hit.
///
/// A ray that starts below the surface reports a hit at `t = 0` rather than
/// silently missing.
fn ray_field(entry: &TerrainEntry, origin: Vec3, dir: Vec3, max_dist: f32) -> Option<(f32, Vec3)> {
    let above = |t: f32| -> Option<f32> {
        let p = origin + dir * t;
        if !entry.surface.contains_world(entry.origin, p.x, p.z) {
            return None;
        }
        Some(p.y - entry.surface.height_world(entry.origin, p.x, p.z))
    };
    let normal_at = |t: f32| -> Vec3 {
        let p = origin + dir * t;
        entry.surface.normal_world(entry.origin, p.x, p.z)
    };

    let mut prev_t = 0.0f32;
    let mut prev = above(0.0);
    if prev.is_some_and(|f| f <= 0.0) {
        return Some((0.0, normal_at(0.0)));
    }

    let steps = ((max_dist / RAY_MARCH_STEP).ceil() as u32).clamp(1, RAY_MAX_STEPS);
    for i in 1..=steps {
        let t = (i as f32 * RAY_MARCH_STEP).min(max_dist);
        let cur = above(t);
        if let (Some(pf), Some(cf)) = (prev, cur) {
            if pf > 0.0 && cf <= 0.0 {
                let (mut lo, mut hi) = (prev_t, t);
                for _ in 0..RAY_BISECTIONS {
                    let mid = (lo + hi) * 0.5;
                    match above(mid) {
                        Some(f) if f > 0.0 => lo = mid,
                        Some(_) => hi = mid,
                        // Left the patch mid-bracket: keep the bracket we have.
                        None => break,
                    }
                }
                return Some((hi, normal_at(hi)));
            }
        }
        prev_t = t;
        prev = cur;
        if t >= max_dist {
            break;
        }
    }
    None
}
