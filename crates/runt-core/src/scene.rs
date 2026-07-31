//! The demo scene. Placeholder content until the ECS world lands (DESIGN §3).

use glam::{Quat, Vec2, Vec3};
use runt_mesh::{cone, cube, cylinder, plane, torus, uv_sphere, MeshData};

/// A small demo scene exercising the mesh layer: primitives, per-shape color,
/// transforms, deformers, both normal modes, and additive merge — all baked
/// into one buffer.
pub fn demo_scene() -> MeshData {
    let ground = plane(Vec2::splat(6.0), 1)
        .with_color(Vec3::new(0.18, 0.20, 0.24))
        .translate(Vec3::new(0.0, -1.2, 0.0));

    let ball = uv_sphere(0.9, 24, 32)
        .smooth_normals(180.0)
        .with_color(Vec3::new(0.90, 0.35, 0.35))
        .translate(Vec3::new(-1.9, -0.3, 0.0));

    let post = cylinder(0.35, 1.8, 24)
        .with_color(Vec3::new(0.35, 0.55, 0.95))
        .translate(Vec3::new(0.0, -0.3, -1.6));

    let spike = cone(0.6, 1.4, 20)
        .with_color(Vec3::new(0.95, 0.80, 0.30))
        .translate(Vec3::new(1.9, -0.5, 0.0));

    let ring = torus(0.7, 0.22, 32, 16)
        .with_color(Vec3::new(0.55, 0.85, 0.55))
        .rotate(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2))
        .translate(Vec3::new(0.0, 0.4, 1.7));

    // A twisted, tapered box — flat-shaded to show the faceting.
    let twisted = cube(1.0)
        .scale(Vec3::new(1.0, 1.6, 1.0))
        .twist(0.9, Vec3::Y)
        .taper(0.4, Vec3::Y)
        .flat_normals()
        .with_color(Vec3::new(0.80, 0.45, 0.90))
        .translate(Vec3::new(0.0, 0.5, 0.0));

    ground
        .merge(ball)
        .merge(post)
        .merge(spike)
        .merge(ring)
        .merge(twisted)
}
