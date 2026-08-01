//! The editor's orbit camera (DESIGN §10).
//!
//! The scene has a camera of its own, and in the ball demo it is a `FollowCamera`
//! welded to the player. Fighting that from the editor — nudging its transform
//! and watching the follow system drag it back next tick — would be a losing
//! game, so the editor does not fight it: on load it **strips the follow rig**
//! and drives the camera entity's pose directly (see
//! [`crate::engine_thread`]).
//!
//! The maths lives here, on the UI side, and the engine only ever receives a
//! finished `(eye, target)` pair as [`Command::SetCameraPose`]. The engine
//! therefore contains nothing that knows an editor exists, which is the property
//! §10 is actually asking for.
//!
//! Convention: `yaw` is measured about +Y from +Z, `pitch` is elevation above
//! the XZ plane, and the camera sits at `target + distance · dir(yaw, pitch)`
//! looking back at `target`.
//!
//! [`Command::SetCameraPose`]: crate::protocol::Command::SetCameraPose

use glam::Vec3;

/// Pitch is clamped just short of straight up/down. At exactly ±90° the
/// look-at basis is degenerate (the view direction is parallel to up) and the
/// camera's roll becomes undefined — it visibly snaps. Stopping a degree short
/// costs nothing and removes the whole class of problem.
pub const PITCH_LIMIT: f32 = std::f32::consts::FRAC_PI_2 - 0.017_453_3;

/// A distance of zero would put the eye on the target and produce a NaN view
/// matrix, so zoom has a floor as well as a ceiling.
pub const MIN_DISTANCE: f32 = 0.05;
pub const MAX_DISTANCE: f32 = 5_000.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Orbit {
    pub target: Vec3,
    /// Radians about +Y.
    pub yaw: f32,
    /// Radians above the horizon, clamped to [`PITCH_LIMIT`].
    pub pitch: f32,
    pub distance: f32,
    /// Radians of rotation per pixel of drag.
    pub orbit_speed: f32,
    /// Fraction of `distance` panned per pixel of drag — panning in world units
    /// per pixel would crawl when zoomed out and bolt when zoomed in.
    pub pan_speed: f32,
    /// Multiplicative zoom per wheel notch.
    pub zoom_speed: f32,
}

impl Default for Orbit {
    fn default() -> Orbit {
        Orbit {
            target: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.35,
            distance: 12.0,
            orbit_speed: 0.008,
            pan_speed: 0.0015,
            zoom_speed: 0.0015,
        }
    }
}

impl Orbit {
    /// An orbit framing a sphere of `radius` about `center`, at a distance where
    /// a 60° vertical field of view just contains it with a little air.
    pub fn framing(center: Vec3, radius: f32) -> Orbit {
        let radius = radius.max(0.1);
        // r / sin(fov/2), padded ~15 %.
        let distance = (radius / (std::f32::consts::FRAC_PI_6).sin()) * 1.15;
        Orbit {
            target: center,
            distance: distance.clamp(MIN_DISTANCE, MAX_DISTANCE),
            ..Orbit::default()
        }
    }

    /// Where the camera sits.
    pub fn eye(&self) -> Vec3 {
        self.target + self.direction() * self.distance
    }

    /// The unit vector from the target towards the eye.
    pub fn direction(&self) -> Vec3 {
        let (sp, cp) = self.pitch.sin_cos();
        let (sy, cy) = self.yaw.sin_cos();
        Vec3::new(cp * sy, sp, cp * cy)
    }

    /// Drag with the left button: swing around the target.
    ///
    /// Dragging right turns the camera right, which means the *scene* appears to
    /// turn left — the convention every DCC tool uses, hence the sign.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * self.orbit_speed;
        self.pitch = (self.pitch + dy * self.orbit_speed).clamp(-PITCH_LIMIT, PITCH_LIMIT);
        // Keep yaw in (-π, π] so it cannot drift into a range where f32 spacing
        // makes the drag feel coarse after a long session.
        self.yaw = wrap_pi(self.yaw);
    }

    /// Drag with the middle button (or shift + left): slide the target across
    /// the view plane, taking the eye with it.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let forward = -self.direction();
        let right = forward.cross(Vec3::Y).normalize_or(Vec3::X);
        let up = right.cross(forward).normalize_or(Vec3::Y);
        let scale = self.distance * self.pan_speed;
        self.target += (-right * dx + up * dy) * scale;
    }

    /// Wheel: move in or out.
    ///
    /// Multiplicative rather than additive, so one notch covers the same
    /// *proportion* of the distance whether you are 2 m or 200 m out — additive
    /// zoom is unusable at both ends of that range.
    pub fn zoom(&mut self, delta: f32) {
        let factor = (-delta * self.zoom_speed).exp();
        self.distance = (self.distance * factor).clamp(MIN_DISTANCE, MAX_DISTANCE);
    }

    /// Recenter on a point without changing the viewing angle — "frame the
    /// selection".
    pub fn look_at(&mut self, target: Vec3) {
        self.target = target;
    }
}

fn wrap_pi(angle: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    let mut a = (angle + PI) % TAU;
    if a < 0.0 {
        a += TAU;
    }
    a - PI
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Vec3, b: Vec3) -> bool {
        a.abs_diff_eq(b, 1e-4)
    }

    #[test]
    fn the_default_camera_looks_down_positive_z_at_the_origin() {
        let orbit = Orbit {
            pitch: 0.0,
            yaw: 0.0,
            distance: 5.0,
            ..Orbit::default()
        };
        assert!(close(orbit.eye(), Vec3::new(0.0, 0.0, 5.0)));
    }

    #[test]
    fn orbiting_keeps_the_target_and_the_distance() {
        let mut orbit = Orbit::default();
        let (target, distance) = (orbit.target, orbit.distance);
        orbit.orbit(120.0, -40.0);
        assert_eq!(orbit.target, target);
        assert!((orbit.eye().distance(orbit.target) - distance).abs() < 1e-3);
    }

    #[test]
    fn pitch_never_reaches_the_degenerate_pole() {
        let mut orbit = Orbit::default();
        for _ in 0..1000 {
            orbit.orbit(0.0, 1000.0);
        }
        assert!(orbit.pitch <= PITCH_LIMIT);
        assert!(orbit.pitch < std::f32::consts::FRAC_PI_2);
        for _ in 0..2000 {
            orbit.orbit(0.0, -1000.0);
        }
        assert!(orbit.pitch >= -PITCH_LIMIT);
    }

    #[test]
    fn yaw_stays_bounded_over_a_long_session() {
        let mut orbit = Orbit::default();
        for _ in 0..10_000 {
            orbit.orbit(50.0, 0.0);
        }
        assert!(orbit.yaw.abs() <= std::f32::consts::PI + 1e-5, "{}", orbit.yaw);
    }

    #[test]
    fn a_full_turn_returns_to_the_same_place() {
        let mut orbit = Orbit::default();
        let before = orbit.eye();
        let steps = (std::f32::consts::TAU / orbit.orbit_speed) as i32;
        for _ in 0..steps {
            orbit.orbit(-1.0, 0.0);
        }
        assert!(
            orbit.eye().distance(before) < 0.05,
            "after a full turn the eye was {:?}, expected {before:?}",
            orbit.eye()
        );
    }

    #[test]
    fn panning_moves_the_target_across_the_view_not_along_it() {
        let mut orbit = Orbit {
            pitch: 0.0,
            yaw: 0.0,
            distance: 10.0,
            ..Orbit::default()
        };
        let forward = -orbit.direction();
        orbit.pan(100.0, 0.0);
        // The target moved, and not along the view axis.
        assert!(orbit.target.length() > 0.0);
        assert!(
            orbit.target.dot(forward).abs() < 1e-4,
            "pan drifted along the view direction: {:?}",
            orbit.target
        );
    }

    #[test]
    fn pan_scales_with_distance() {
        let near = {
            let mut o = Orbit { distance: 1.0, ..Orbit::default() };
            o.pan(100.0, 0.0);
            o.target.length()
        };
        let far = {
            let mut o = Orbit { distance: 100.0, ..Orbit::default() };
            o.pan(100.0, 0.0);
            o.target.length()
        };
        assert!(far > near * 50.0, "near {near}, far {far}");
    }

    #[test]
    fn zoom_is_multiplicative_and_bounded() {
        let mut orbit = Orbit::default();
        let start = orbit.distance;
        orbit.zoom(100.0);
        assert!(orbit.distance < start, "positive wheel moves in");
        orbit.zoom(-100.0);
        assert!((orbit.distance - start).abs() < 1e-3, "and back out again");

        for _ in 0..10_000 {
            orbit.zoom(1000.0);
        }
        assert!(orbit.distance >= MIN_DISTANCE, "zoom must not reach the target");
        for _ in 0..20_000 {
            orbit.zoom(-1000.0);
        }
        assert!(orbit.distance <= MAX_DISTANCE);
    }

    #[test]
    fn framing_puts_the_whole_sphere_in_front_of_the_camera() {
        let orbit = Orbit::framing(Vec3::new(1.0, 2.0, 3.0), 20.0);
        assert_eq!(orbit.target, Vec3::new(1.0, 2.0, 3.0));
        assert!(
            orbit.distance > 20.0,
            "the eye must be outside the sphere, got {}",
            orbit.distance
        );
    }
}
