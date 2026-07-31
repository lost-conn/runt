//! Camera components and the follow-camera system (DESIGN §5).
//!
//! The engine renders exactly one camera per `render()` call. Its pose is an
//! ordinary [`Transform`] on an ordinary entity, so it interpolates, follows and
//! gets sim-driven like anything else — the host stopped baking view matrices
//! the moment this landed.

use bevy_ecs::prelude::*;
use glam::{Mat3, Mat4, Quat, Vec3};

use crate::ecs::{FixedTick, Transform};

/// Projection parameters. The pose lives in the entity's [`Transform`]; this is
/// only the lens.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub fov_y_rad: f32,
    pub z_near: f32,
    pub z_far: f32,
}

impl Default for Camera {
    fn default() -> Camera {
        Camera {
            fov_y_rad: std::f32::consts::FRAC_PI_3, // 60°, what the host used to hardcode
            z_near: 0.1,
            z_far: 100.0,
        }
    }
}

impl Camera {
    /// Right-handed view space, `[0,1]` clip depth, Y-up — the wgpu/WebGPU NDC
    /// convention, matching the `Less` depth compare and the `Depth32Float`
    /// attachment the renderer sets up.
    pub fn projection(&self, aspect: f32) -> Mat4 {
        glam::camera::rh::proj::directx::perspective(
            self.fov_y_rad,
            aspect.max(f32::MIN_POSITIVE),
            self.z_near,
            self.z_far,
        )
    }

    /// Projection × the view matrix implied by `pose`.
    pub fn view_proj(&self, pose: Mat4, aspect: f32) -> Mat4 {
        self.projection(aspect) * view_from_pose(pose)
    }
}

/// The view matrix for a camera whose world pose is `pose` — i.e. its inverse.
///
/// A camera "at" a transform looks down its local −Z with +Y up, the same
/// convention [`Transform::looking_at`] builds and `Mat4::look_at_rh` inverts.
pub fn view_from_pose(pose: Mat4) -> Mat4 {
    pose.inverse()
}

/// Exponential approach toward a moving target (DESIGN §5's follow camera).
///
/// `stiffness` is in reciprocal seconds: the remaining distance decays by
/// `exp(-stiffness · dt)` per tick, so the response is frame-rate independent
/// *and* tick-rate independent — the same follow reads identically at 10 Hz and
/// 60 Hz, which a naive `lerp(a, b, k)` would not.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct FollowCamera {
    /// Entity to follow. Its [`Transform`] translation is both the anchor for
    /// `offset` and the look-at point.
    pub target: Entity,
    /// World-space offset from the target to the camera's rest position.
    pub offset: Vec3,
    /// Approach rate, 1/seconds. ~2 is a lazy drift, ~12 is nearly rigid.
    pub stiffness: f32,
}

impl FollowCamera {
    /// The blend factor for one step of `dt` seconds.
    pub fn approach(&self, dt: f32) -> f32 {
        if self.stiffness <= 0.0 {
            return 0.0;
        }
        1.0 - (-self.stiffness * dt).exp()
    }
}

/// `FixedSim`: ease every follow camera toward its target's rest pose.
///
/// Position and orientation use the *same* decay, so the camera never arrives
/// facing the wrong way; both are sim state (not render-side smoothing), which
/// keeps a replay's camera identical to the original run.
pub fn follow_camera(
    tick: Res<FixedTick>,
    targets: Query<&Transform, Without<FollowCamera>>,
    mut cameras: Query<(&FollowCamera, &mut Transform)>,
) {
    for (follow, mut transform) in &mut cameras {
        let Ok(target) = targets.get(follow.target) else {
            continue; // Target despawned or has no transform: hold still.
        };
        let focus = target.translation;
        let rest = focus + follow.offset;
        let k = follow.approach(tick.dt_secs);

        transform.translation = transform.translation.lerp(rest, k);
        let desired = look_rotation(transform.translation, focus, Vec3::Y);
        transform.rotation = transform.rotation.slerp(desired, k).normalize();
    }
}

/// The rotation that points a camera at `focus` from `eye`.
///
/// Degenerate cases (eye on the focus point, up parallel to the view) fall back
/// to identity rather than producing NaNs that would poison the sim forever.
pub fn look_rotation(eye: Vec3, focus: Vec3, up: Vec3) -> Quat {
    let back = eye - focus;
    let Some(back) = back.try_normalize() else {
        return Quat::IDENTITY;
    };
    let Some(right) = up.cross(back).try_normalize() else {
        return Quat::IDENTITY;
    };
    Quat::from_mat3(&Mat3::from_cols(right, back.cross(right), back))
}
