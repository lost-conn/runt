//! The in-game scene editor's toolkit (DESIGN §10a).
//!
//! Headless and GPU-free, every one of them, which is the point of the way the
//! toolkit is written: picking is a ray against the collision snapshot, a drag
//! is a scalar off that ray, and undo is a `Vec` and a cursor. Nothing here
//! needs a `Renderer`, so nothing here is a screenshot test with an opinion.
//!
//! The claims, in the order the module builds them:
//!
//! - [`a_projected_point_unprojects_onto_the_ray_that_found_it`] — the screen ↔
//!   world round trip, which every pick and every drag stands on.
//! - [`the_crosshair_picks_the_collider_it_is_over`] — against known colliders,
//!   including the one behind it.
//! - [`an_arrow_is_grabbed_by_a_ray_that_grazes_it_and_missed_by_one_that_does_not`]
//!   and its siblings — the gizmo hit test.
//! - [`a_translate_drag_moves_by_the_distance_the_pointer_travelled`] — the drag
//!   maths, in world units, from screen positions.
//! - [`an_op_and_its_inverse_leave_the_scene_byte_identical`] — the op-log
//!   algebra, which is what "the op log IS the undo stack" has to mean.

#![cfg(feature = "editor")]

use bevy_ecs::prelude::*;
use glam::{Mat4, Quat, Vec2, Vec3};

use runt_core::collide::{CollisionWorld, ALL_LAYERS};
use runt_core::editor::{
    self, Axis, Drag, DragKind, EditError, EditableScene, EditorState, OpLog, PaletteEntry, Ray,
    Snap, Tool,
};
use runt_core::editor_gizmo::{self as gizmo, Gizmo};
use runt_core::physics::AabbCollider;
use runt_core::{Camera, Transform, Viewport};

const VIEWPORT: Viewport = Viewport {
    width: 800,
    height: 600,
};

/// A camera at `(0, 2, 10)` looking at the origin — the pose every ray test
/// below is taken from.
fn camera_pose() -> (Camera, Mat4) {
    let eye = Vec3::new(0.0, 2.0, 10.0);
    let pose = Transform {
        translation: eye,
        rotation: runt_core::camera::look_rotation(eye, Vec3::ZERO, Vec3::Y),
        scale: Vec3::ONE,
    };
    (Camera::default(), pose.matrix())
}

fn view_proj() -> Mat4 {
    let (camera, pose) = camera_pose();
    camera.view_proj(pose, VIEWPORT.aspect())
}

// ---------------------------------------------------------------------------
// Rays
// ---------------------------------------------------------------------------

#[test]
fn a_projected_point_unprojects_onto_the_ray_that_found_it() {
    let (camera, pose) = camera_pose();
    let vp = view_proj();

    // Points spread across the frustum, including off-centre in both axes —
    // a round trip that only tested the centre would pass with the Y flip
    // missing, which is the mistake this is here to catch.
    for world in [
        Vec3::ZERO,
        Vec3::new(3.0, 1.0, -2.0),
        Vec3::new(-4.0, -1.5, 0.5),
        Vec3::new(0.5, 4.0, -6.0),
    ] {
        let screen = editor::project(vp, VIEWPORT, world).expect("in front of the camera");
        assert!(
            screen.x >= 0.0 && screen.x <= 800.0 && screen.y >= 0.0 && screen.y <= 600.0,
            "{world:?} projected off screen to {screen:?}"
        );
        let ray = editor::screen_ray(&camera, pose, VIEWPORT, screen).expect("a ray");

        // The point is *on* the ray: the perpendicular component of the offset
        // is zero. Asserting that rather than "the ray hits it at distance d"
        // keeps the test about the round trip and not about the near plane.
        let offset = world - ray.origin;
        let perpendicular = offset - ray.dir * offset.dot(ray.dir);
        assert!(
            perpendicular.length() < 1e-3,
            "{world:?} is {} off its own ray",
            perpendicular.length()
        );
        assert!(offset.dot(ray.dir) > 0.0, "{world:?} came back behind");
    }
}

#[test]
fn a_point_behind_the_camera_does_not_project() {
    // `w <= 0`: the naive divide would put it somewhere plausible in the wrong
    // half of the screen, which is a selection bug that looks like a physics one.
    assert_eq!(
        editor::project(view_proj(), VIEWPORT, Vec3::new(0.0, 2.0, 40.0)),
        None
    );
    // …and a viewport nothing has drawn into yet has no screen to project onto.
    assert_eq!(
        editor::project(view_proj(), Viewport::ZERO, Vec3::ZERO),
        None
    );
    let (camera, pose) = camera_pose();
    assert_eq!(
        editor::screen_ray(&camera, pose, Viewport::ZERO, Vec2::ZERO),
        None
    );
}

#[test]
fn the_ray_through_the_centre_of_the_screen_is_the_camera_forward() {
    let (camera, pose) = camera_pose();
    let ray = editor::screen_ray(&camera, pose, VIEWPORT, VIEWPORT.size() * 0.5).expect("a ray");
    let forward = (Vec3::ZERO - Vec3::new(0.0, 2.0, 10.0)).normalize();
    assert!(
        (ray.dir - forward).length() < 1e-4,
        "{:?} is not the look direction",
        ray.dir
    );
}

// ---------------------------------------------------------------------------
// Picking
// ---------------------------------------------------------------------------

#[test]
fn the_crosshair_picks_the_collider_it_is_over() {
    let mut world = World::new();
    // Two boxes on the camera's axis, one behind the other, plus one off to the
    // side. The near one must win and the far one must be reachable by aiming
    // past the near one's edge.
    let near = world
        .spawn((
            Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
            AabbCollider {
                half_extents: Vec3::splat(1.0),
            },
        ))
        .id();
    world.spawn((
        Transform::from_translation(Vec3::new(0.0, 0.0, -8.0)),
        AabbCollider {
            half_extents: Vec3::splat(1.0),
        },
    ));
    let side = world
        .spawn((
            Transform::from_translation(Vec3::new(4.0, 0.0, 0.0)),
            AabbCollider {
                half_extents: Vec3::splat(1.0),
            },
        ))
        .id();
    let snapshot = CollisionWorld::from_world(&mut world);
    let (camera, pose) = camera_pose();

    let pick = |target: Vec3| -> Option<Entity> {
        let screen = editor::project(view_proj(), VIEWPORT, target)?;
        let ray = editor::screen_ray(&camera, pose, VIEWPORT, screen)?;
        editor::pick_any(&snapshot, ray, editor::PICK_DIST).map(|hit| hit.entity)
    };

    assert_eq!(pick(Vec3::ZERO), Some(near), "the nearest box wins");
    assert_eq!(
        pick(Vec3::new(4.0, 0.0, 0.0)),
        Some(side),
        "aiming at the side box picks it"
    );
    // Aim well above everything: nothing at all.
    let miss = editor::screen_ray(&camera, pose, VIEWPORT, Vec2::new(400.0, 5.0)).unwrap();
    assert!(editor::pick_any(&snapshot, miss, editor::PICK_DIST).is_none());

    // The far box is only reachable once the near one is out of the way, which
    // is the strongest statement available that "nearest" is being computed and
    // not "first in Entity order".
    let mut without_near = World::new();
    without_near.spawn((
        Transform::from_translation(Vec3::new(0.0, 0.0, -8.0)),
        AabbCollider {
            half_extents: Vec3::splat(1.0),
        },
    ));
    let snapshot2 = CollisionWorld::from_world(&mut without_near);
    let screen = editor::project(view_proj(), VIEWPORT, Vec3::new(0.0, 0.0, -8.0)).unwrap();
    let ray = editor::screen_ray(&camera, pose, VIEWPORT, screen).unwrap();
    assert!(editor::pick_any(&snapshot2, ray, editor::PICK_DIST).is_some());

    // …and the masked form is the same query with a say in the layers, which is
    // what a game that wants to select only its own scenery reaches for.
    let screen = editor::project(view_proj(), VIEWPORT, Vec3::ZERO).unwrap();
    let ray = editor::screen_ray(&camera, pose, VIEWPORT, screen).unwrap();
    assert_eq!(
        editor::pick_world(&snapshot, ray, editor::PICK_DIST, ALL_LAYERS).map(|h| h.entity),
        Some(near)
    );
    assert_eq!(
        editor::pick_world(&snapshot, ray, editor::PICK_DIST, 0),
        None
    );
}

// ---------------------------------------------------------------------------
// The gizmo's hit test
// ---------------------------------------------------------------------------

fn unit_gizmo(kind: DragKind) -> Gizmo {
    Gizmo {
        origin: Vec3::ZERO,
        basis: Quat::IDENTITY,
        scale: 1.0,
        kind,
    }
}

#[test]
fn an_arrow_is_grabbed_by_a_ray_that_grazes_it_and_missed_by_one_that_does_not() {
    let g = unit_gizmo(DragKind::Translate);
    // Straight down at the middle of the X shaft: a hit.
    let over = Ray {
        origin: Vec3::new(0.6, 5.0, 0.0),
        dir: -Vec3::Y,
    };
    assert_eq!(gizmo::pick_axis(&g, over), Some(Axis::X));

    // The same ray a quarter of a unit to the side of the shaft: past
    // `PICK_TOLERANCE`, so nothing.
    let beside = Ray {
        origin: Vec3::new(0.6, 5.0, 0.25),
        dir: -Vec3::Y,
    };
    assert_eq!(gizmo::pick_axis(&g, beside), None);

    // Past the arrowhead: also nothing. The handle is a segment, not a line.
    let beyond = Ray {
        origin: Vec3::new(2.0, 5.0, 0.0),
        dir: -Vec3::Y,
    };
    assert_eq!(gizmo::pick_axis(&g, beyond), None);

    // Each axis is reachable from a direction that is not its own.
    let on_z = Ray {
        origin: Vec3::new(0.0, 5.0, 0.6),
        dir: -Vec3::Y,
    };
    assert_eq!(gizmo::pick_axis(&g, on_z), Some(Axis::Z));
    let on_y = Ray {
        origin: Vec3::new(0.0, 0.6, 5.0),
        dir: -Vec3::Z,
    };
    assert_eq!(gizmo::pick_axis(&g, on_y), Some(Axis::Y));
}

#[test]
fn a_rotation_ring_is_grabbed_at_its_radius_and_not_inside_it() {
    let g = unit_gizmo(DragKind::Rotate);
    // Down the Y axis onto the Y ring, which lies in XZ at radius RING_RADIUS.
    let on_ring = Ray {
        origin: Vec3::new(gizmo::RING_RADIUS, 5.0, 0.0),
        dir: -Vec3::Y,
    };
    assert_eq!(gizmo::pick_axis(&g, on_ring), Some(Axis::Y));

    // Through the middle of the same ring: inside it, so no handle.
    let through_middle = Ray {
        origin: Vec3::new(0.0, 5.0, 0.0),
        dir: -Vec3::Y,
    };
    assert_eq!(gizmo::pick_axis(&g, through_middle), None);
}

#[test]
fn a_handle_behind_the_camera_is_not_hovered() {
    let g = unit_gizmo(DragKind::Translate);
    // Pointing away from the gizmo entirely.
    let away = Ray {
        origin: Vec3::new(0.6, 5.0, 0.0),
        dir: Vec3::Y,
    };
    assert_eq!(gizmo::pick_axis(&g, away), None);
}

#[test]
fn the_handles_wear_the_selections_rotation() {
    // A node yawed 90° about Y: its local X points down world −Z, so the ray
    // that used to find X on the world X axis finds nothing, and one down the
    // −Z axis finds it.
    let g = Gizmo {
        origin: Vec3::ZERO,
        basis: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
        scale: 1.0,
        kind: DragKind::Translate,
    };
    assert!((g.world_axis(Axis::X) - -Vec3::Z).length() < 1e-5);
    let on_local_x = Ray {
        origin: Vec3::new(0.0, 5.0, -0.6),
        dir: -Vec3::Y,
    };
    assert_eq!(gizmo::pick_axis(&g, on_local_x), Some(Axis::X));
}

// ---------------------------------------------------------------------------
// Drag maths
// ---------------------------------------------------------------------------

#[test]
fn a_translate_drag_moves_by_the_distance_the_pointer_travelled() {
    // The whole loop, from two screen positions to a moved transform, with a
    // real camera in the middle. The node sits at the origin; the pointer
    // grabs its X arrow and drags to where world (3, 0, 0) projects.
    let (camera, pose) = camera_pose();
    let vp = view_proj();
    let g = Gizmo::for_anchor(
        &Transform::IDENTITY,
        DragKind::Translate,
        &camera,
        Vec3::new(0.0, 2.0, 10.0),
        VIEWPORT,
    );

    let grab_screen = editor::project(vp, VIEWPORT, Vec3::new(0.5, 0.0, 0.0)).unwrap();
    let drop_screen = editor::project(vp, VIEWPORT, Vec3::new(3.5, 0.0, 0.0)).unwrap();
    let grab_ray = editor::screen_ray(&camera, pose, VIEWPORT, grab_screen).unwrap();
    let drop_ray = editor::screen_ray(&camera, pose, VIEWPORT, drop_screen).unwrap();

    let from = gizmo::grab_param(&g, Axis::X, grab_ray).expect("a grab");
    let to = gizmo::grab_param(&g, Axis::X, drop_ray).expect("a drop");
    // Both parameters are distances along the axis from the gizmo's origin, so
    // they read straight off the world points the pointer was aimed at.
    assert!((from - 0.5).abs() < 1e-2, "{from}");
    assert!((to - 3.5).abs() < 1e-2, "{to}");

    let off = Snap {
        on: false,
        ..Snap::default()
    };
    let amount = gizmo::drag_amount(&g, from, to, &off);
    assert!((amount - 3.0).abs() < 1e-2, "{amount}");

    let moved = gizmo::dragged(&Transform::IDENTITY, &gizmo::drag_of(&g, Axis::X, amount));
    assert!((moved.translation - Vec3::new(3.0, 0.0, 0.0)).length() < 1e-2);
    assert_eq!(moved.rotation, Quat::IDENTITY);
    assert_eq!(moved.scale, Vec3::ONE);

    // …and with the grid on, the same drag lands on a whole number of cells.
    let snapped = gizmo::drag_amount(&g, from, to, &Snap::default());
    assert_eq!(snapped, 3.0);
}

#[test]
fn a_drag_down_the_axis_being_dragged_refuses_rather_than_teleporting() {
    // Looking straight down X: every point on the X line projects to the same
    // pixel, so the closest-point solve is degenerate. Returning `None` is what
    // keeps a drag from jumping to infinity when the camera swings round.
    let g = unit_gizmo(DragKind::Translate);
    let down_the_axis = Ray {
        origin: Vec3::new(-10.0, 0.0, 0.0),
        dir: Vec3::X,
    };
    assert_eq!(gizmo::grab_param(&g, Axis::X, down_the_axis), None);
    // The other two are fine from the same ray.
    assert!(gizmo::grab_param(&g, Axis::Y, down_the_axis).is_some());
}

#[test]
fn a_rotate_drag_turns_about_the_pivot_and_snaps_to_whole_notches() {
    let g = unit_gizmo(DragKind::Rotate);
    let snap = Snap::default(); // 15°
    let from = 0.0;
    let to = 0.9_f32; // 51.6°, nearest notch 45°
    let amount = gizmo::drag_amount(&g, from, to, &snap);
    assert!(
        (amount - 45.0_f32.to_radians()).abs() < 1e-5,
        "{}",
        amount.to_degrees()
    );

    // Applied to a node one unit out along X, about the origin: a 90° turn
    // about Y carries it onto −Z.
    let drag = Drag {
        kind: DragKind::Rotate,
        axis: Axis::Y,
        amount: std::f32::consts::FRAC_PI_2,
        pivot: Vec3::ZERO,
        basis: Quat::IDENTITY,
    };
    let start = Transform::from_translation(Vec3::X);
    let turned = gizmo::dragged(&start, &drag);
    assert!(
        (turned.translation - -Vec3::Z).length() < 1e-5,
        "{:?}",
        turned.translation
    );

    // The group form agrees with the single-node one, which is the property
    // that lets an adapter pick either without the two disagreeing.
    let m = gizmo::delta_matrix(&drag);
    assert!((m.transform_point3(Vec3::X) - -Vec3::Z).length() < 1e-5);
}

#[test]
fn a_rotate_drag_across_the_seam_reads_as_a_small_step() {
    let g = unit_gizmo(DragKind::Rotate);
    let off = Snap {
        on: false,
        ..Snap::default()
    };
    // 175° to −175° is ten degrees the short way, not 350 the long way.
    let amount = gizmo::drag_amount(&g, 175.0_f32.to_radians(), -175.0_f32.to_radians(), &off);
    assert!(
        (amount - 10.0_f32.to_radians()).abs() < 1e-5,
        "{}",
        amount.to_degrees()
    );
}

#[test]
fn a_scale_drag_is_a_multiplier_in_gizmo_lengths() {
    // The gizmo is two world units long here, so dragging out by two units is
    // one gizmo length, which is +1.0 — double.
    let g = Gizmo {
        origin: Vec3::ZERO,
        basis: Quat::IDENTITY,
        scale: 2.0,
        kind: DragKind::Scale,
    };
    let off = Snap {
        on: false,
        ..Snap::default()
    };
    let amount = gizmo::drag_amount(&g, 2.0, 4.0, &off);
    assert!((amount - 1.0).abs() < 1e-6, "{amount}");

    let scaled = gizmo::dragged(&Transform::IDENTITY, &gizmo::drag_of(&g, Axis::Y, amount));
    assert_eq!(scaled.scale, Vec3::new(1.0, 2.0, 1.0));
    assert_eq!(scaled.translation, Vec3::ZERO, "a scale does not translate");

    // The group form is uniform by construction — see `delta_matrix`'s docs.
    let m = gizmo::delta_matrix(&gizmo::drag_of(&g, Axis::Y, amount));
    let (s, _, _) = m.to_scale_rotation_translation();
    assert!((s - Vec3::splat(2.0)).length() < 1e-5, "{s:?}");
}

#[test]
fn a_placement_snaps_absolutely_where_a_drag_snaps_relatively() {
    let state: EditorState<u32> = EditorState::new();
    let ray = Ray {
        origin: Vec3::new(0.0, 10.0, 0.0),
        dir: -Vec3::Y,
    };
    // No hit: the fallback distance ahead of the ray, on the grid.
    let placed = state.place_at(ray, None);
    assert_eq!(placed, Vec3::new(0.0, 2.0, 0.0));
    assert_eq!(placed, state.snap.position(placed), "already on the grid");
}

// ---------------------------------------------------------------------------
// The op log
// ---------------------------------------------------------------------------

/// A scene of numbered nodes, which is the least a scene can be and still
/// exercise the algebra: transforms that change, nodes that appear and nodes
/// that go away.
#[derive(Clone, Debug, Default, PartialEq)]
struct ToyScene {
    nodes: Vec<Transform>,
    palette: Vec<PaletteEntry>,
    /// The adapter is allowed to refuse; this is how the test asks it to.
    refuse: bool,
}

#[derive(Clone, Debug, PartialEq)]
enum ToyOp {
    Set {
        index: usize,
        from: Transform,
        to: Transform,
    },
    Insert {
        index: usize,
        node: Transform,
    },
    Remove {
        index: usize,
        node: Transform,
    },
}

impl ToyScene {
    fn new(count: usize) -> ToyScene {
        ToyScene {
            nodes: (0..count)
                .map(|i| Transform::from_translation(Vec3::splat(i as f32)))
                .collect(),
            palette: vec![PaletteEntry::new("toy", "node")],
            refuse: false,
        }
    }
}

impl EditableScene for ToyScene {
    type Id = usize;
    type Op = ToyOp;

    fn pick_id(&self, entity: Entity) -> Option<usize> {
        let index = entity.index_u32() as usize;
        (index < self.nodes.len()).then_some(index)
    }

    fn anchor(&self, id: usize) -> Option<Transform> {
        self.nodes.get(id).copied()
    }

    fn drag_op(&self, id: usize, drag: &Drag) -> Option<ToyOp> {
        let from = *self.nodes.get(id)?;
        Some(ToyOp::Set {
            index: id,
            from,
            to: gizmo::dragged(&from, drag),
        })
    }

    fn palette(&self) -> &[PaletteEntry] {
        &self.palette
    }

    fn place_op(&self, _index: usize, at: &Transform) -> Option<ToyOp> {
        Some(ToyOp::Insert {
            index: self.nodes.len(),
            node: *at,
        })
    }

    fn delete_op(&self, id: usize) -> Option<ToyOp> {
        Some(ToyOp::Remove {
            index: id,
            node: *self.nodes.get(id)?,
        })
    }

    fn apply(&mut self, op: &ToyOp) -> Result<(), EditError> {
        if self.refuse {
            return Err(EditError::Io("refused".into()));
        }
        match op {
            ToyOp::Set { index, to, .. } => {
                *self
                    .nodes
                    .get_mut(*index)
                    .ok_or_else(|| EditError::NoSuchId(index.to_string()))? = *to;
            }
            ToyOp::Insert { index, node } => self.nodes.insert(*index, *node),
            ToyOp::Remove { index, .. } => {
                if *index >= self.nodes.len() {
                    return Err(EditError::NoSuchId(index.to_string()));
                }
                self.nodes.remove(*index);
            }
        }
        Ok(())
    }

    fn invert(&self, op: &ToyOp) -> Option<ToyOp> {
        Some(match op {
            ToyOp::Set { index, from, to } => ToyOp::Set {
                index: *index,
                from: *to,
                to: *from,
            },
            ToyOp::Insert { index, node } => ToyOp::Remove {
                index: *index,
                node: *node,
            },
            ToyOp::Remove { index, node } => ToyOp::Insert {
                index: *index,
                node: *node,
            },
        })
    }

    fn op_id(&self, op: &ToyOp) -> Option<usize> {
        Some(match op {
            ToyOp::Set { index, .. }
            | ToyOp::Insert { index, .. }
            | ToyOp::Remove { index, .. } => *index,
        })
    }

    fn describe(&self, op: &ToyOp) -> String {
        format!("{op:?}")
    }

    fn serialize(&self) -> Result<String, EditError> {
        Ok(format!("{:?}", self.nodes))
    }
}

fn x_drag(amount: f32) -> Drag {
    Drag {
        kind: DragKind::Translate,
        axis: Axis::X,
        amount,
        pivot: Vec3::ZERO,
        basis: Quat::IDENTITY,
    }
}

#[test]
fn an_op_and_its_inverse_leave_the_scene_byte_identical() {
    let mut scene = ToyScene::new(3);
    let before = scene.serialize().unwrap();
    let mut log: OpLog<ToyOp> = OpLog::new();

    let op = scene.drag_op(1, &x_drag(2.5)).unwrap();
    editor::apply(&mut scene, &mut log, op).unwrap();
    assert_ne!(scene.serialize().unwrap(), before);
    assert_eq!(log.cursor(), 1);

    assert!(editor::undo(&mut scene, &mut log).unwrap());
    assert_eq!(
        scene.serialize().unwrap(),
        before,
        "the inverse did not restore the scene exactly"
    );
    assert_eq!(log.cursor(), 0);
    assert_eq!(log.len(), 1, "an undone op is still redoable");

    assert!(editor::redo(&mut scene, &mut log).unwrap());
    assert_ne!(scene.serialize().unwrap(), before);
    assert_eq!(log.cursor(), 1);
}

#[test]
fn a_delete_and_its_undo_restore_the_node_where_it_was() {
    let mut scene = ToyScene::new(4);
    let before = scene.serialize().unwrap();
    let mut log: OpLog<ToyOp> = OpLog::new();

    // The *middle* node, so a restore that appended rather than inserted would
    // be caught by the byte comparison.
    let op = scene.delete_op(1).unwrap();
    editor::apply(&mut scene, &mut log, op).unwrap();
    assert_eq!(scene.nodes.len(), 3);

    assert!(editor::undo(&mut scene, &mut log).unwrap());
    assert_eq!(scene.serialize().unwrap(), before);
}

#[test]
fn a_new_edit_after_an_undo_drops_the_redo_tail() {
    let mut scene = ToyScene::new(2);
    let mut log: OpLog<ToyOp> = OpLog::new();
    for amount in [1.0, 2.0, 3.0] {
        let op = scene.drag_op(0, &x_drag(amount)).unwrap();
        editor::apply(&mut scene, &mut log, op).unwrap();
    }
    assert_eq!(log.len(), 3);

    editor::undo(&mut scene, &mut log).unwrap();
    editor::undo(&mut scene, &mut log).unwrap();
    assert_eq!(log.cursor(), 1);
    assert!(log.can_redo());

    let op = scene.drag_op(0, &x_drag(9.0)).unwrap();
    editor::apply(&mut scene, &mut log, op).unwrap();
    assert_eq!(log.len(), 2, "the two undone ops are unreachable now");
    assert!(!log.can_redo());
}

#[test]
fn undo_at_the_bottom_and_redo_at_the_top_are_nothing_rather_than_errors() {
    let mut scene = ToyScene::new(1);
    let mut log: OpLog<ToyOp> = OpLog::new();
    assert!(!editor::undo(&mut scene, &mut log).unwrap());
    assert!(!editor::redo(&mut scene, &mut log).unwrap());

    let op = scene.drag_op(0, &x_drag(1.0)).unwrap();
    editor::apply(&mut scene, &mut log, op).unwrap();
    assert!(!editor::redo(&mut scene, &mut log).unwrap());
}

#[test]
fn a_refused_op_leaves_the_undo_stack_exactly_as_it_was() {
    let mut scene = ToyScene::new(2);
    let mut log: OpLog<ToyOp> = OpLog::new();
    let good = scene.drag_op(0, &x_drag(1.0)).unwrap();
    editor::apply(&mut scene, &mut log, good).unwrap();

    scene.refuse = true;
    let bad = scene.drag_op(1, &x_drag(1.0)).unwrap();
    assert!(editor::apply(&mut scene, &mut log, bad).is_err());
    assert_eq!(log.len(), 1, "a refused edit was logged");
    assert_eq!(log.cursor(), 1);

    // …and a refused *undo* does not consume the cursor either, or the next
    // undo would silently skip an op.
    assert!(editor::undo(&mut scene, &mut log).is_err());
    assert_eq!(log.cursor(), 1);
    scene.refuse = false;
    assert!(editor::undo(&mut scene, &mut log).unwrap());
    assert_eq!(log.cursor(), 0);
}

#[test]
fn the_log_is_bounded_and_drops_the_oldest_first() {
    let mut scene = ToyScene::new(1);
    let mut log: OpLog<ToyOp> = OpLog::with_limit(3);
    for amount in [1.0, 2.0, 3.0, 4.0, 5.0] {
        let op = scene.drag_op(0, &x_drag(amount)).unwrap();
        editor::apply(&mut scene, &mut log, op).unwrap();
    }
    assert_eq!(log.len(), 3);
    assert_eq!(log.cursor(), 3);
    // Three undos, and then the bottom — the two oldest are gone, not merely
    // unreachable.
    assert!(editor::undo(&mut scene, &mut log).unwrap());
    assert!(editor::undo(&mut scene, &mut log).unwrap());
    assert!(editor::undo(&mut scene, &mut log).unwrap());
    assert!(!editor::undo(&mut scene, &mut log).unwrap());
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[test]
fn leaving_edit_mode_drops_a_drag_and_keeps_the_selection() {
    let mut state: EditorState<usize> = EditorState::new();
    state.set_on(true);
    state.selection = Some(7);
    state.tool = Tool::Move;
    state.grab = Some(editor::Grab {
        kind: DragKind::Translate,
        axis: Axis::X,
        pivot: Vec3::ZERO,
        basis: Quat::IDENTITY,
        gizmo_scale: 1.0,
        from: 0.0,
        applied: 0.0,
    });
    state.set_on(false);
    assert!(state.grab.is_none(), "a drag survived the exit");
    assert_eq!(state.selection, Some(7), "the selection did not");
    assert_eq!(state.tool, Tool::Move);
}

#[test]
fn an_applied_op_marks_the_world_for_rebuilding() {
    let mut state: EditorState<usize> = EditorState::new();
    assert!(!state.dirty);
    state.touched("moved");
    assert!(state.dirty, "the hot-reload signal did not fire");
    assert_eq!(state.message, "moved");
}
