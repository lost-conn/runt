//! The manipulator handles — `editor` feature only (DESIGN §10a).
//!
//! Three arrows, three rings, three boxes, and the arithmetic that turns a
//! pointer into a number. Everything here is a **pure function of a ray and a
//! pose**: there is no state, no hysteresis and no frame history, which is what
//! makes drag maths — traditionally the part of an editor nobody can test —
//! ordinary unit tests.
//!
//! ```text
//! Gizmo          where the handles are and how big they look
//! pick_axis      which handle a ray is over            (hit test)
//! grab_param     the scalar a handle reads off a ray   (1 number per frame)
//! dragged        that scalar applied to one transform
//! delta_matrix   …or to a whole fold of brushes
//! parts          the draws, as (mesh, model, colour)
//! ```
//!
//! # Constant screen size
//!
//! A gizmo that shrinks with distance is a gizmo you cannot grab. [`gizmo_scale`]
//! solves for the world size that subtends a fixed number of **pixels** at the
//! selection's distance, so the handles are the same size on a pebble at arm's
//! length and a mountain across the valley. It is the one place the tool needs
//! the camera's lens rather than just its pose.
//!
//! # Local space, always
//!
//! The handles wear the selection's own rotation. That is not a preference, it
//! is what keeps a scale sound: scaling a rotated box along a *world* axis
//! produces a shear, and a `Trs` — which is what every scene format the engine
//! is likely to meet stores — has nowhere to put one. Rotating the handles
//! instead makes the illegal edit unreachable rather than merely discouraged.
//!
//! # Always on top, without a new pipeline
//!
//! A handle buried in the terrain is a handle you cannot use, and runt has no
//! depth-off variant. It does have the *idiom*, written down at
//! [`DEPTH_GREATER`](crate::MaterialVariant::DEPTH_GREATER): "the same geometry
//! with `ADDITIVE | DEPTH_GREATER` paints the occluded part". So [`parts`]
//! emits each handle twice — once opaque and depth-tested, once additive and
//! depth-*inverted* — and the occluded half of an arrow shows through as a
//! ghost. Two draws per handle against one new pipeline state, one new shader
//! variant and one more bit in a key space that is deliberately small: the
//! cheaper trade, and it uses only bits the engine already ships.

use glam::{Mat4, Quat, Vec3, Vec4};
use runt_mesh::{primitives, MeshData};

use crate::camera::Camera;
use crate::ecs::{Transform, Viewport};
use crate::editor::{Axis, Drag, DragKind, Ray, Snap};
use crate::material::MaterialVariant;

// ---------------------------------------------------------------------------
// Proportions
// ---------------------------------------------------------------------------

/// Arrow shaft length, in gizmo units (1 unit = [`gizmo_scale`]'s answer).
pub const SHAFT_LEN: f32 = 1.0;
/// Arrow shaft radius.
pub const SHAFT_RADIUS: f32 = 0.018;
/// Arrowhead length, beyond the shaft.
pub const HEAD_LEN: f32 = 0.22;
/// Arrowhead radius.
pub const HEAD_RADIUS: f32 = 0.07;
/// Half-extent of a scale tool's cube, sitting at the end of the shaft.
pub const CUBE_HALF: f32 = 0.06;
/// Rotation ring radius.
pub const RING_RADIUS: f32 = 0.9;
/// Rotation ring tube radius.
pub const RING_TUBE: f32 = 0.022;

/// How close a ray must pass to an arrow's axis to count as over it.
///
/// Comfortably wider than the shaft: a hit test that matched the drawn radius
/// would need pixel-perfect aim at a shape two pixels across.
pub const PICK_TOLERANCE: f32 = 0.1;
/// How close to [`RING_RADIUS`] a ray's plane crossing must land.
pub const RING_TOLERANCE: f32 = 0.12;

/// How tall one gizmo unit is on screen, in logical pixels.
pub const GIZMO_PIXELS: f32 = 110.0;

/// The ghost pass's alpha — how strongly an occluded handle shows through.
pub const GHOST_ALPHA: f32 = 0.28;
/// The colour a hovered or held handle takes.
pub const HIGHLIGHT: Vec4 = Vec4::new(1.0, 0.85, 0.3, 1.0);

/// A drag can never scale a node through zero.
pub const MIN_SCALE_FACTOR: f32 = 0.02;

// ---------------------------------------------------------------------------
// The gizmo
// ---------------------------------------------------------------------------

/// Where the handles are, how they are oriented, and how big.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gizmo {
    pub origin: Vec3,
    /// The selection's rotation — see the module docs on local space.
    pub basis: Quat,
    /// World size of one gizmo unit.
    pub scale: f32,
    /// Which set of handles is showing.
    pub kind: DragKind,
}

impl Gizmo {
    /// The gizmo for a selection whose pose is `anchor`, seen by `camera` at
    /// `eye`.
    pub fn for_anchor(
        anchor: &Transform,
        kind: DragKind,
        camera: &Camera,
        eye: Vec3,
        viewport: Viewport,
    ) -> Gizmo {
        Gizmo {
            origin: anchor.translation,
            basis: anchor.rotation,
            scale: gizmo_scale(anchor.translation, eye, camera.fov_y_rad, viewport),
            kind,
        }
    }

    /// `axis` as a world direction.
    pub fn world_axis(&self, axis: Axis) -> Vec3 {
        (self.basis * axis.unit()).normalize_or_zero()
    }

    /// The far end of `axis`'s shaft, in world space.
    pub fn tip(&self, axis: Axis) -> Vec3 {
        self.origin + self.world_axis(axis) * SHAFT_LEN * self.scale
    }
}

/// The world size of one gizmo unit at `origin`, so it subtends
/// [`GIZMO_PIXELS`] logical pixels.
///
/// The vertical field of view spans `2·tan(fov/2)·d` world units at distance
/// `d`, over `viewport.height` pixels — so one pixel is that over the height,
/// and the answer is `GIZMO_PIXELS` of them. Falls back to `1.0` for a viewport
/// nothing has drawn into yet, which is what a headless test has.
pub fn gizmo_scale(origin: Vec3, eye: Vec3, fov_y_rad: f32, viewport: Viewport) -> f32 {
    if !viewport.is_known() {
        return 1.0;
    }
    let dist = (origin - eye).length().max(1e-3);
    let world_per_pixel = 2.0 * (fov_y_rad * 0.5).tan() * dist / viewport.height as f32;
    (world_per_pixel * GIZMO_PIXELS).max(1e-4)
}

// ---------------------------------------------------------------------------
// Hit test
// ---------------------------------------------------------------------------

/// Which handle `ray` is over, nearest first, or `None`.
pub fn pick_axis(gizmo: &Gizmo, ray: Ray) -> Option<Axis> {
    let mut best: Option<(f32, Axis)> = None;
    for axis in Axis::ALL {
        let hit = match gizmo.kind {
            DragKind::Rotate => ring_hit(gizmo, axis, ray),
            DragKind::Translate | DragKind::Scale => shaft_hit(gizmo, axis, ray),
        };
        let Some(t) = hit else { continue };
        // Strict `<`: an exact tie keeps X over Y over Z, which is the order
        // `Axis::ALL` is in, so the answer never depends on float noise.
        if best.is_none_or(|(best_t, _)| t < best_t) {
            best = Some((t, axis));
        }
    }
    best.map(|(_, axis)| axis)
}

/// Distance along `ray` at which it passes within tolerance of `axis`'s shaft,
/// or `None`.
fn shaft_hit(gizmo: &Gizmo, axis: Axis, ray: Ray) -> Option<f32> {
    let dir = gizmo.world_axis(axis);
    // The whole handle, arrowhead or cube included: the tip is the part a hand
    // actually aims at.
    let length = (SHAFT_LEN + HEAD_LEN) * gizmo.scale;
    let (t, dist) = ray_segment_distance(ray, gizmo.origin, gizmo.origin + dir * length)?;
    (dist <= PICK_TOLERANCE * gizmo.scale).then_some(t)
}

/// Distance along `ray` at which it crosses `axis`'s ring, or `None`.
fn ring_hit(gizmo: &Gizmo, axis: Axis, ray: Ray) -> Option<f32> {
    let normal = gizmo.world_axis(axis);
    let denom = ray.dir.dot(normal);
    // Edge-on: the ring is a line and there is nothing to aim at. Refusing is
    // better than the near-infinite `t` the divide would produce.
    if denom.abs() < 1e-4 {
        return None;
    }
    let t = (gizmo.origin - ray.origin).dot(normal) / denom;
    if t <= 0.0 {
        return None;
    }
    let radial = (ray.at(t) - gizmo.origin).length();
    ((radial - RING_RADIUS * gizmo.scale).abs() <= RING_TOLERANCE * gizmo.scale).then_some(t)
}

/// Closest approach between a ray and a segment: `(t along the ray, distance)`.
///
/// `None` only for a degenerate segment. The ray is clamped at its origin (a
/// handle behind the camera is not hovered) and the segment at both ends.
pub fn ray_segment_distance(ray: Ray, a: Vec3, b: Vec3) -> Option<(f32, f32)> {
    let u = b - a;
    let len_sq = u.length_squared();
    if len_sq <= f32::EPSILON {
        return None;
    }
    let w0 = a - ray.origin;
    let b_dot = u.dot(ray.dir);
    let d = u.dot(w0);
    let e = ray.dir.dot(w0);
    let denom = len_sq - b_dot * b_dot;
    // Parallel: any point does, so take the segment's start.
    let mut s = if denom.abs() < 1e-8 {
        0.0
    } else {
        (b_dot * e - d * 1.0) / denom
    };
    s = s.clamp(0.0, 1.0);
    let p = a + u * s;
    let t = (p - ray.origin).dot(ray.dir).max(0.0);
    Some((t, (p - ray.at(t)).length()))
}

// ---------------------------------------------------------------------------
// Drag maths
// ---------------------------------------------------------------------------

/// The scalar `axis`'s handle reads off `ray`, in the units its tool wants.
///
/// - **Translate / Scale** — signed distance from the gizmo origin along the
///   axis, to the point on that axis line closest to the ray. This is the
///   textbook "drag along a line in 3D": the pointer picks a point in space and
///   the handle takes its shadow on the axis, which is why a sideways mouse
///   movement moves an X arrow and a vertical one does not.
/// - **Rotate** — the angle, in radians, of the ray's crossing of the handle's
///   plane, measured from the axis's own reference direction.
///
/// `None` when the geometry degenerates: looking straight down a translate axis
/// (its shadow is the whole line) or straight along a rotate plane.
pub fn grab_param(gizmo: &Gizmo, axis: Axis, ray: Ray) -> Option<f32> {
    match gizmo.kind {
        DragKind::Translate | DragKind::Scale => {
            axis_param(ray, gizmo.origin, gizmo.world_axis(axis))
        }
        DragKind::Rotate => ring_angle(gizmo, axis, ray),
    }
}

/// Where along the infinite line `pivot + axis·s` the ray comes closest.
pub fn axis_param(ray: Ray, pivot: Vec3, axis: Vec3) -> Option<f32> {
    let b = axis.dot(ray.dir);
    let denom = 1.0 - b * b;
    // The ray is within ~2.5° of the axis: the closest point races off to
    // infinity and a drag would teleport.
    if denom.abs() < 2e-3 {
        return None;
    }
    let w0 = pivot - ray.origin;
    let d = axis.dot(w0);
    let e = ray.dir.dot(w0);
    Some((b * e - d) / denom)
}

/// The two in-plane reference directions for a rotation about `axis`.
///
/// Chosen so that a **positive** rotation about the axis carries `u` towards
/// `v` — i.e. `(u, v, axis)` is right-handed. Cycled off the basis rather than
/// derived per call, so a ring's zero angle never jumps when the selection turns.
pub fn ring_basis(basis: Quat, axis: Axis) -> (Vec3, Vec3) {
    let (u, v) = match axis {
        Axis::X => (Vec3::Y, Vec3::Z),
        Axis::Y => (Vec3::Z, Vec3::X),
        Axis::Z => (Vec3::X, Vec3::Y),
    };
    (basis * u, basis * v)
}

fn ring_angle(gizmo: &Gizmo, axis: Axis, ray: Ray) -> Option<f32> {
    let normal = gizmo.world_axis(axis);
    let denom = ray.dir.dot(normal);
    if denom.abs() < 1e-4 {
        return None;
    }
    let t = (gizmo.origin - ray.origin).dot(normal) / denom;
    if t <= 0.0 {
        return None;
    }
    let radial = ray.at(t) - gizmo.origin;
    let (u, v) = ring_basis(gizmo.basis, axis);
    Some(radial.dot(v).atan2(radial.dot(u)))
}

/// The signed shortest way round from `from` to `to`, in radians.
///
/// A rotate drag that crosses the ring's seam must read as a small step and not
/// as a 350° lurch, so the difference is wrapped into `(-π, π]` before it is
/// snapped.
pub fn wrap_angle(delta: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    let mut d = delta % TAU;
    if d > PI {
        d -= TAU;
    } else if d <= -PI {
        d += TAU;
    }
    d
}

/// Turn a grab's two parameters into the snapped amount its tool means.
///
/// For a scale the raw difference is in world units, and dividing by the
/// gizmo's own size is what makes it a *multiplier*: dragging the handle out by
/// one gizmo length is `+1.0`, i.e. double. That is the one unit choice in the
/// module and it is the reason a scale drag feels the same at every distance —
/// the gizmo is a constant number of pixels, so a constant number of pixels of
/// drag is a constant amount of scale.
pub fn drag_amount(gizmo: &Gizmo, from: f32, to: f32, snap: &Snap) -> f32 {
    let raw = match gizmo.kind {
        DragKind::Translate => to - from,
        DragKind::Rotate => wrap_angle(to - from),
        DragKind::Scale => (to - from) / gizmo.scale.max(1e-4),
    };
    snap.amount(gizmo.kind, raw)
}

/// The drag a held handle describes, ready for
/// [`EditableScene::drag_op`](crate::editor::EditableScene::drag_op).
pub fn drag_of(gizmo: &Gizmo, axis: Axis, amount: f32) -> Drag {
    Drag {
        kind: gizmo.kind,
        axis,
        amount,
        pivot: gizmo.origin,
        basis: gizmo.basis,
    }
}

/// `drag` applied to one node's transform.
///
/// The single-node half of the pair; [`delta_matrix`] is the other. Scale lands
/// on the node's **own** `scale` component along the handle's axis, which is
/// exactly why the handles are local (module docs): there is no world-space
/// non-uniform scale to represent, so there is none to lose.
pub fn dragged(start: &Transform, drag: &Drag) -> Transform {
    let world_axis = drag.world_axis();
    match drag.kind {
        DragKind::Translate => Transform {
            translation: start.translation + world_axis * drag.amount,
            ..*start
        },
        DragKind::Rotate => {
            let q = Quat::from_axis_angle(world_axis, drag.amount);
            Transform {
                translation: drag.pivot + q * (start.translation - drag.pivot),
                rotation: (q * start.rotation).normalize(),
                scale: start.scale,
            }
        }
        DragKind::Scale => {
            let factor = (1.0 + drag.amount).max(MIN_SCALE_FACTOR);
            let mut scale = start.scale;
            scale[drag.axis.index()] *= factor;
            // The offset from the pivot scales with everything else, so a node
            // dragged by a gizmo that is not its own origin travels the way its
            // geometry does.
            let offset = start.translation - drag.pivot;
            let along = offset.dot(world_axis);
            Transform {
                translation: drag.pivot + offset + world_axis * (along * (factor - 1.0)),
                rotation: start.rotation,
                scale,
            }
        }
    }
}

/// `drag` as a world-space matrix, for a node made of many transforms.
///
/// The group half of the pair. A fold of a hundred brushes has no single `Trs`
/// to write into, so the drag becomes a matrix each brush's own transform is
/// composed with — `delta · brush` — and the caller decomposes back.
///
/// **A group scale is uniform**, and that is a real restriction rather than an
/// oversight: `delta · brush` for a per-axis world scale is not a similarity
/// transform, so a brush rotated off-axis comes back sheared and
/// `to_scale_rotation_translation` has to invent an answer. Per-axis scale
/// reaches a node that *is* one transform, through [`dragged`]; a fold takes a
/// uniform one. An adapter that wants to say no to even that should return
/// `None` from `drag_op`.
pub fn delta_matrix(drag: &Drag) -> Mat4 {
    let world_axis = drag.world_axis();
    match drag.kind {
        DragKind::Translate => Mat4::from_translation(world_axis * drag.amount),
        DragKind::Rotate => {
            Mat4::from_translation(drag.pivot)
                * Mat4::from_quat(Quat::from_axis_angle(world_axis, drag.amount))
                * Mat4::from_translation(-drag.pivot)
        }
        DragKind::Scale => {
            let factor = (1.0 + drag.amount).max(MIN_SCALE_FACTOR);
            Mat4::from_translation(drag.pivot)
                * Mat4::from_scale(Vec3::splat(factor))
                * Mat4::from_translation(-drag.pivot)
        }
    }
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

/// The four unit meshes a gizmo is built from.
///
/// Unit shapes plus a model matrix, rather than one mesh per handle per size:
/// the gizmo resizes every frame (it is a constant number of pixels), and a
/// mesh library keyed on content would otherwise gain an entry per distance the
/// camera has ever been at.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GizmoMesh {
    /// Unit cylinder along +Y: radius 1, height 1, centred.
    Shaft,
    /// Unit cone along +Y, apex at `+0.5`.
    Head,
    /// Unit cube, centred.
    Cube,
    /// Unit ring in XZ (axis +Y), tube [`RING_TUBE`]`/`[`RING_RADIUS`] thick.
    Ring,
}

impl GizmoMesh {
    pub const ALL: [GizmoMesh; 4] = [
        GizmoMesh::Shaft,
        GizmoMesh::Head,
        GizmoMesh::Cube,
        GizmoMesh::Ring,
    ];

    /// The geometry. Deliberately coarse — a handle is 110 pixels tall and
    /// nobody has ever looked at a gizmo's silhouette.
    pub fn mesh(self) -> MeshData {
        match self {
            GizmoMesh::Shaft => primitives::cylinder(1.0, 1.0, 8),
            GizmoMesh::Head => primitives::cone(1.0, 1.0, 10),
            GizmoMesh::Cube => primitives::box3(Vec3::ONE),
            GizmoMesh::Ring => primitives::torus(1.0, RING_TUBE / RING_RADIUS, 40, 6),
        }
    }
}

/// One draw of one handle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GizmoPart {
    pub mesh: GizmoMesh,
    pub model: Mat4,
    pub color: Vec4,
    pub axis: Axis,
    /// The occluded-half draw — see the module docs. A game gives this one
    /// [`ghost_variant`] and the other [`solid_variant`].
    pub ghost: bool,
}

/// The material bits a handle's solid draw wants: unlit, so a gizmo does not
/// take the scene's lighting and read as part of the level.
pub fn solid_variant() -> MaterialVariant {
    MaterialVariant::UNLIT
}

/// …and its ghost draw's: additive with the depth test inverted, which paints
/// exactly the part the solid draw lost to the depth buffer.
pub fn ghost_variant() -> MaterialVariant {
    MaterialVariant::UNLIT | MaterialVariant::ADDITIVE | MaterialVariant::DEPTH_GREATER
}

/// Every draw this gizmo needs, solid pass then ghost pass.
///
/// Order matters and is the order it is emitted in: the ghosts are
/// [`ADDITIVE`](MaterialVariant::ADDITIVE), so they sort into the blended tail
/// anyway, but emitting them second keeps the list readable and keeps a caller
/// that ignores the flag from painting glows under solids.
pub fn parts(gizmo: &Gizmo, hover: Option<Axis>) -> Vec<GizmoPart> {
    let mut solid = Vec::with_capacity(6);
    for axis in Axis::ALL {
        let color = if hover == Some(axis) {
            HIGHLIGHT
        } else {
            axis.color()
        };
        // Y is the mesh's own axis, so the rotation is exact rather than a
        // `from_rotation_arc` that has to invent an answer for the -Y case.
        let to_axis = gizmo.basis * y_to(axis);
        let g = gizmo.scale;
        match gizmo.kind {
            DragKind::Rotate => solid.push(GizmoPart {
                mesh: GizmoMesh::Ring,
                model: Mat4::from_scale_rotation_translation(
                    Vec3::splat(RING_RADIUS * g),
                    to_axis,
                    gizmo.origin,
                ),
                color,
                axis,
                ghost: false,
            }),
            DragKind::Translate | DragKind::Scale => {
                let dir = gizmo.world_axis(axis);
                solid.push(GizmoPart {
                    mesh: GizmoMesh::Shaft,
                    model: Mat4::from_scale_rotation_translation(
                        Vec3::new(SHAFT_RADIUS * g, SHAFT_LEN * g, SHAFT_RADIUS * g),
                        to_axis,
                        gizmo.origin + dir * (SHAFT_LEN * 0.5 * g),
                    ),
                    color,
                    axis,
                    ghost: false,
                });
                let (mesh, size, offset) = if gizmo.kind == DragKind::Scale {
                    (
                        GizmoMesh::Cube,
                        Vec3::splat(CUBE_HALF * 2.0 * g),
                        SHAFT_LEN * g,
                    )
                } else {
                    (
                        GizmoMesh::Head,
                        Vec3::new(HEAD_RADIUS * g, HEAD_LEN * g, HEAD_RADIUS * g),
                        (SHAFT_LEN + HEAD_LEN * 0.5) * g,
                    )
                };
                solid.push(GizmoPart {
                    mesh,
                    model: Mat4::from_scale_rotation_translation(
                        size,
                        to_axis,
                        gizmo.origin + dir * offset,
                    ),
                    color,
                    axis,
                    ghost: false,
                });
            }
        }
    }
    let mut out = solid.clone();
    out.extend(solid.into_iter().map(|part| GizmoPart {
        color: part.color.truncate().extend(GHOST_ALPHA),
        ghost: true,
        ..part
    }));
    out
}

/// The rotation carrying +Y (every gizmo mesh's own axis) onto `axis`.
fn y_to(axis: Axis) -> Quat {
    use std::f32::consts::FRAC_PI_2;
    match axis {
        Axis::X => Quat::from_rotation_z(-FRAC_PI_2),
        Axis::Y => Quat::IDENTITY,
        Axis::Z => Quat::from_rotation_x(FRAC_PI_2),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gizmo(kind: DragKind) -> Gizmo {
        Gizmo {
            origin: Vec3::ZERO,
            basis: Quat::IDENTITY,
            scale: 1.0,
            kind,
        }
    }

    #[test]
    fn y_maps_onto_each_axis_exactly() {
        for axis in Axis::ALL {
            let mapped = y_to(axis) * Vec3::Y;
            assert!(
                (mapped - axis.unit()).length() < 1e-6,
                "{axis:?} -> {mapped:?}"
            );
        }
    }

    #[test]
    fn an_angle_wraps_the_short_way_round() {
        use std::f32::consts::PI;
        assert!((wrap_angle(0.1) - 0.1).abs() < 1e-6);
        assert!((wrap_angle(2.0 * PI - 0.1) + 0.1).abs() < 1e-5);
        assert!((wrap_angle(-2.0 * PI + 0.1) - 0.1).abs() < 1e-5);
    }

    #[test]
    fn a_scale_drag_cannot_take_a_node_through_zero() {
        let drag = Drag {
            kind: DragKind::Scale,
            axis: Axis::X,
            amount: -50.0,
            pivot: Vec3::ZERO,
            basis: Quat::IDENTITY,
        };
        let out = dragged(&Transform::IDENTITY, &drag);
        assert_eq!(out.scale.x, MIN_SCALE_FACTOR);
        assert!(out.scale.x > 0.0);
    }

    #[test]
    fn the_gizmo_grows_with_distance_so_it_does_not_shrink_on_screen() {
        let viewport = Viewport::new(800, 600);
        let near = gizmo_scale(Vec3::ZERO, Vec3::Z * 10.0, 1.0, viewport);
        let far = gizmo_scale(Vec3::ZERO, Vec3::Z * 40.0, 1.0, viewport);
        assert!((far / near - 4.0).abs() < 1e-4, "{near} {far}");
        // Nothing drawn yet: a headless caller gets one world unit, not a NaN.
        assert_eq!(gizmo_scale(Vec3::ZERO, Vec3::Z, 1.0, Viewport::ZERO), 1.0);
    }

    #[test]
    fn parts_are_emitted_solid_then_ghost() {
        let drawn = parts(&gizmo(DragKind::Translate), Some(Axis::X));
        assert_eq!(drawn.len(), 12, "shaft + head per axis, twice");
        assert!(drawn[..6].iter().all(|p| !p.ghost));
        assert!(drawn[6..].iter().all(|p| p.ghost));
        assert_eq!(drawn[0].color, HIGHLIGHT, "the hovered axis is highlighted");
        assert_eq!(drawn[6].color.w, GHOST_ALPHA);
        // The rotate tool is one ring per axis and no arrowheads.
        assert_eq!(parts(&gizmo(DragKind::Rotate), None).len(), 6);
    }
}
