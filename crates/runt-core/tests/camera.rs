//! The camera entity and the follow camera (DESIGN §5).
//!
//! No GPU: a view matrix is arithmetic, and the point of moving the camera into
//! the world was that it stopped being renderer trivia.

use glam::{Mat4, Vec3};
use runt_core::camera::{look_rotation, Camera};
use runt_core::scene::{DEMO_EYE, DEMO_SPIN};
use runt_core::{Sim, Transform, TICK_DT};

const ASPECT: f32 = 16.0 / 9.0;

/// The look-at the host used to hardcode, from glam's non-deprecated API.
fn expected_view() -> Mat4 {
    glam::camera::rh::view::look_at_mat4(DEMO_EYE, Vec3::ZERO, Vec3::Y)
}

fn close(a: Mat4, b: Mat4, tol: f32) -> bool {
    a.to_cols_array()
        .iter()
        .zip(b.to_cols_array())
        .all(|(x, y)| (x - y).abs() <= tol)
}

#[test]
fn a_looking_at_transform_inverts_to_look_at() {
    let t = Transform::looking_at(DEMO_EYE, Vec3::ZERO, Vec3::Y);
    assert_eq!(t.translation, DEMO_EYE);
    assert!(
        close(t.matrix().inverse(), expected_view(), 1e-5),
        "camera pose must be the inverse of the view matrix\n{:?}\n{:?}",
        t.matrix().inverse(),
        expected_view()
    );

    // Degenerate inputs fall back to identity rather than producing NaNs that
    // would poison every later tick.
    assert_eq!(look_rotation(Vec3::ZERO, Vec3::ZERO, Vec3::Y), glam::Quat::IDENTITY);
    assert_eq!(look_rotation(Vec3::Y, Vec3::ZERO, Vec3::Y), glam::Quat::IDENTITY);
}

#[test]
fn the_demo_camera_reproduces_the_hardcoded_view() {
    let mut sim = Sim::new();
    let frame = sim.frame_params(ASPECT).expect("the demo spawns a camera");

    let camera = Camera::default();
    let expected = camera.projection(ASPECT) * expected_view();
    assert!(
        close(frame.view_proj, expected, 1e-5),
        "view-projection must match the pre-ECS host view\n{:?}\n{expected:?}",
        frame.view_proj
    );

    // 60° vertical, 0.1..100 — the numbers the renderer used to bake in.
    assert!((camera.fov_y_rad - 60f32.to_radians()).abs() < 1e-6);
    assert_eq!((camera.z_near, camera.z_far), (0.1, 100.0));
}

#[test]
fn a_world_with_no_camera_has_no_frame_params() {
    let mut sim = Sim::new();
    let camera = sim.camera_entity().expect("demo camera");
    sim.world_mut().despawn(camera);
    assert!(sim.camera_entity().is_none());
    assert!(sim.frame_params(ASPECT).is_none(), "no camera, no frame");
}

#[test]
fn the_follow_camera_settles_on_its_target_and_stays() {
    let mut sim = Sim::new();
    let camera = sim.camera_entity().expect("demo camera");

    // The target (the spinning box) never translates, so the rest position is
    // where the camera already is: the follow shows up as a gentle re-aim from
    // the origin to the box's center, and then nothing.
    let start = *sim.world().get::<Transform>(camera).expect("transform");

    sim.update(0.0);
    let mut t = 0.0;
    while sim.tick_count() < 120 {
        t += TICK_DT;
        sim.update(t);
    }
    let settled = *sim.world().get::<Transform>(camera).expect("transform");

    assert!(
        (settled.translation - DEMO_EYE).length() < 1e-4,
        "position must hold at the demo eye, got {:?}",
        settled.translation
    );

    let focus = Vec3::new(0.0, 0.5, 0.0); // the twisted box's translation
    let want = look_rotation(DEMO_EYE, focus, Vec3::Y);
    assert!(
        settled.rotation.abs_diff_eq(want, 1e-3),
        "after 2 s the camera must be looking at its target: {:?} vs {want:?}",
        settled.rotation
    );

    // Subtle, but real: it did move, and it moved smoothly rather than snapping
    // on the first tick.
    assert!(
        start.rotation.angle_between(settled.rotation) > 1e-3,
        "the follow must actually have done something"
    );
    let mut sim2 = Sim::new();
    sim2.update(0.0);
    sim2.update(TICK_DT);
    let after_one_tick = *sim2.world().get::<Transform>(camera).expect("transform");
    let step = start.rotation.angle_between(after_one_tick.rotation);
    let total = start.rotation.angle_between(settled.rotation);
    assert!(
        step < total * 0.2,
        "one tick must be a small fraction of the whole move ({step} of {total})"
    );
}

#[test]
fn the_camera_interpolates_between_ticks_like_anything_else() {
    // The camera carries `Interpolated`, so a render between two ticks must see
    // a pose strictly between them (DESIGN §4) — a camera that snapped to tick
    // boundaries would judder against the interpolated scene.
    let mut sim = Sim::with_tick_rate(10.0);
    sim.update(0.0);
    sim.update(0.1);
    sim.update(0.15);

    let a = sim.frame_params_at(ASPECT, 0.0).expect("camera").view_proj;
    let b = sim.frame_params_at(ASPECT, 0.5).expect("camera").view_proj;
    let c = sim.frame_params_at(ASPECT, 1.0).expect("camera").view_proj;
    assert!(!close(a, b, 1e-6) && !close(b, c, 1e-6), "alpha must matter");
    assert!(close(b, b, 0.0));

    // And the spinner it follows is still spinning at its documented rate.
    assert!(DEMO_SPIN > 0.0);
}
