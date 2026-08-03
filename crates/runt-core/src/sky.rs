//! The background gradient (DESIGN §5's pass list, "clear → opaque forward").
//!
//! The sky is one fullscreen triangle drawn at the head of the opaque pass with
//! the depth test off. It has no geometry, no texture and no uniform of its own:
//! the fragment shader reconstructs a world-space view ray from the frame's
//! inverse view-projection and reads a three-stop vertical gradient off the
//! *same* sky/ground ambient colors the hemisphere term already uses
//! (§5: "directional key light + hemisphere ambient (sky/ground colors)"). So a
//! scene cannot end up with a sky that disagrees with its own ambient.
//!
//! ## Why this file exists as Rust as well as WGSL
//!
//! [`gradient`] and [`view_ray`] are a line-for-line twin of `sky.wgsl`. They
//! are not used by the renderer — the GPU runs the WGSL — they are the model a
//! test can hold the rendered frame against. `tests/headless_screenshot.rs`
//! samples real pixels and demands they match this function, which is both a
//! check that the sky pass ran at all and the thing that keeps the two copies
//! from drifting: change one without the other and the screenshot test fails.
//!
//! Anything that wants to know "what color is the background in that direction"
//! outside a shader (a fog term, an editor's viewport clear) should call
//! [`gradient`] rather than grow a third copy.

use glam::{Mat4, Vec2, Vec3};

use crate::ecs::Lighting;

/// Shaping exponent for the upper half of the gradient.
///
/// Below 1, so the zenith color is reached quickly and the horizon band stays
/// tight — a linear ramp reads as a washed-out smear over the top half of the
/// frame. The lower half uses [`NADIR_EXPONENT`], slightly slacker, because the
/// ground half is mostly hidden behind terrain anyway and a hard band there
/// draws the eye to the seam.
pub const ZENITH_EXPONENT: f32 = 0.55;

/// Shaping exponent for the lower half of the gradient. See [`ZENITH_EXPONENT`].
pub const NADIR_EXPONENT: f32 = 0.75;

/// The world-space direction of the view ray through a normalized-device point.
///
/// `ndc` is `(x, y)` in `[-1, 1]`, +Y up (the WGSL clip-space convention, not
/// the pixel one). `inv_view_proj` is the inverse of the matrix the frame was
/// drawn with; unprojecting the near and far points of the same NDC column and
/// differencing them gives the ray without needing the camera pose separately.
pub fn view_ray(inv_view_proj: Mat4, ndc: Vec2) -> Vec3 {
    let near = inv_view_proj * glam::Vec4::new(ndc.x, ndc.y, 0.0, 1.0);
    let far = inv_view_proj * glam::Vec4::new(ndc.x, ndc.y, 1.0, 1.0);
    let a = near.truncate() / near.w;
    let b = far.truncate() / far.w;
    (b - a).normalize_or(Vec3::Z)
}

/// The background color along `dir`.
///
/// Three stops on the ray's vertical component: `ground_color` straight down,
/// [`Lighting::horizon`] at the horizon, `sky_color` straight up.
pub fn gradient(lighting: &Lighting, dir: Vec3) -> Vec3 {
    let horizon = lighting.horizon();
    let t = dir.y.clamp(-1.0, 1.0);
    if t >= 0.0 {
        horizon.lerp(lighting.sky_color, t.powf(ZENITH_EXPONENT))
    } else {
        horizon.lerp(lighting.ground_color, (-t).powf(NADIR_EXPONENT))
    }
}

/// The background color at a normalized-device point — [`view_ray`] then
/// [`gradient`], which is exactly what the fragment shader does per pixel.
pub fn color_at(lighting: &Lighting, inv_view_proj: Mat4, ndc: Vec2) -> Vec3 {
    gradient(lighting, view_ray(inv_view_proj, ndc))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gradient_hits_its_three_stops() {
        let light = Lighting::default();
        assert!(gradient(&light, Vec3::Y).abs_diff_eq(light.sky_color, 1e-6));
        assert!(gradient(&light, Vec3::NEG_Y).abs_diff_eq(light.ground_color, 1e-6));
        assert!(gradient(&light, Vec3::Z).abs_diff_eq(light.horizon(), 1e-6));
    }

    #[test]
    fn an_unset_horizon_is_the_midpoint_and_a_set_one_is_obeyed() {
        let mut light = Lighting::default();
        assert!(light
            .horizon()
            .abs_diff_eq((light.sky_color + light.ground_color) * 0.5, 1e-6));

        light.horizon = Some(Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(light.horizon(), Vec3::new(1.0, 0.0, 0.0));
        assert!(gradient(&light, Vec3::X).abs_diff_eq(Vec3::new(1.0, 0.0, 0.0), 1e-6));
    }

    #[test]
    fn the_view_ray_comes_back_out_of_the_matrix_it_went_in_with() {
        // Built by the engine's own camera rather than by a hand-rolled
        // projection, so this cannot silently pass against a different clip
        // convention from the one the renderer actually draws with.
        let eye = Vec3::new(0.0, 2.0, 6.0);
        let camera = crate::camera::Camera::default();
        let pose = crate::ecs::Transform::looking_at(eye, Vec3::ZERO, Vec3::Y);
        let view_proj = camera.view_proj(pose.matrix(), 16.0 / 9.0);
        let inv = view_proj.inverse();

        // The centre of the frame looks where the camera looks.
        let centre = view_ray(inv, Vec2::ZERO);
        let want = (Vec3::ZERO - eye).normalize();
        assert!(centre.abs_diff_eq(want, 1e-4), "{centre:?} vs {want:?}");

        // And the top of the frame looks higher than the bottom of it.
        assert!(view_ray(inv, Vec2::new(0.0, 1.0)).y > view_ray(inv, Vec2::new(0.0, -1.0)).y);
    }

    #[test]
    fn a_degenerate_matrix_produces_a_color_rather_than_a_nan() {
        // The no-camera render path hands the renderer an identity view-proj;
        // it must still paint something finite.
        let light = Lighting::default();
        let c = color_at(&light, Mat4::IDENTITY, Vec2::new(0.5, -0.25));
        assert!(c.is_finite(), "{c:?}");
    }
}
