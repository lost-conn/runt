//! The demo scene, as entities (DESIGN §3, §5).
//!
//! Each shape is its own entity with its own `MeshRef` + `Material` +
//! `Transform`; the renderer draws them one by one from the sorted draw list.
//! *Placement* (translate/rotate/scale) lives in the `Transform`, not in the
//! generator — two entities differing only in placement must produce the same
//! `MeshData`, or content-addressed dedup would never fire.
//!
//! Generation that *is* shape (the twisted box's stretch, twist and taper) stays
//! in the generator: it changes the geometry, so it legitimately changes the
//! content hash.
//!
//! Placeholder content until the generator registry lands (§6, step 4).

use bevy_ecs::prelude::*;
use glam::{Quat, Vec2, Vec3, Vec4};
use runt_mesh::{cone, cube, cylinder, plane, torus, uv_sphere, MeshData};

use crate::camera::{Camera, FollowCamera};
use crate::ecs::{
    DemoEntity, DemoScene, GlobalTransform, Interpolated, MeshRef, Spin, Transform,
};
use crate::material::Material;
use crate::registry::MeshLibrary;

/// The demo's spin rate: 0.4 rad/s about +Y, the rate the pre-ECS renderer
/// hardcoded, kept so screenshots compare like with like.
pub const DEMO_SPIN: f32 = 0.4;

/// Where the demo camera sits once it has settled on its target.
pub const DEMO_EYE: Vec3 = Vec3::new(0.0, 2.4, 6.5);

// ---------------------------------------------------------------------------
// Per-shape generators — pure, GPU-free, placement-free
// ---------------------------------------------------------------------------

pub fn ground_mesh() -> MeshData {
    plane(Vec2::splat(6.0), 1).with_color(Vec3::new(0.18, 0.20, 0.24))
}

pub fn ball_mesh() -> MeshData {
    uv_sphere(0.9, 24, 32)
        .smooth_normals(180.0)
        .with_color(Vec3::new(0.90, 0.35, 0.35))
}

pub fn post_mesh() -> MeshData {
    cylinder(0.35, 1.8, 24).with_color(Vec3::new(0.35, 0.55, 0.95))
}

pub fn spike_mesh() -> MeshData {
    cone(0.6, 1.4, 20).with_color(Vec3::new(0.95, 0.80, 0.30))
}

pub fn ring_mesh() -> MeshData {
    torus(0.7, 0.22, 32, 16).with_color(Vec3::new(0.55, 0.85, 0.55))
}

/// A twisted, tapered box — flat-shaded to show the faceting.
pub fn twisted_box_mesh() -> MeshData {
    cube(1.0)
        .scale(Vec3::new(1.0, 1.6, 1.0))
        .twist(0.9, Vec3::Y)
        .taper(0.4, Vec3::Y)
        .flat_normals()
        .with_color(Vec3::new(0.80, 0.45, 0.90))
}

// ---------------------------------------------------------------------------
// Spawning
// ---------------------------------------------------------------------------

/// `Startup`: build the demo's meshes, then spawn one entity per shape.
///
/// Seven entities, six meshes: the second sphere is generated from *identical*
/// params, so [`MeshLibrary::insert`] hands back the same handle and the
/// renderer uploads one set of buffers for both (DESIGN §5). It differs only in
/// `Transform` and `Material` — which is exactly the property that makes
/// content addressing worth having.
pub fn spawn_demo_scene(mut commands: Commands, mut library: ResMut<MeshLibrary>) {
    let ground = library.insert(ground_mesh());
    let ball = library.insert(ball_mesh());
    let post = library.insert(post_mesh());
    let spike = library.insert(spike_mesh());
    let ring = library.insert(ring_mesh());
    let twisted = library.insert(twisted_box_mesh());
    // Same generator, same params → same handle. Deliberately not deduped by
    // hand: the registry is what proves it.
    let ball_again = library.insert(ball_mesh());
    debug_assert_eq!(ball, ball_again, "identical mesh params must dedup");

    let statics = [
        (
            // ground
            ground,
            Transform::from_translation(Vec3::new(0.0, -1.2, 0.0)),
            Material::vertex_colored(),
        ),
        (
            // ball
            ball,
            Transform::from_translation(Vec3::new(-1.9, -0.3, 0.0)),
            Material::vertex_colored(),
        ),
        (
            // post
            post,
            Transform::from_translation(Vec3::new(0.0, -0.3, -1.6)),
            Material::vertex_colored(),
        ),
        (
            // spike
            spike,
            Transform::from_translation(Vec3::new(1.9, -0.5, 0.0)),
            Material::vertex_colored(),
        ),
        (
            // ring
            ring,
            Transform {
                translation: Vec3::new(0.0, 0.4, 1.7),
                rotation: Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
                scale: Vec3::ONE,
            },
            Material::vertex_colored(),
        ),
        (
            // The dedup twin: same geometry as `ball`, half the size (from the
            // *transform*, so the mesh is untouched) and a flat material that
            // ignores the mesh's vertex colors — which also means a second
            // shader variant, and a second pipeline in the cache.
            ball_again,
            Transform {
                translation: Vec3::new(-2.0, 0.5, 1.9),
                rotation: Quat::IDENTITY,
                scale: Vec3::splat(0.5),
            },
            Material::flat(Vec4::new(0.35, 0.75, 0.95, 1.0)),
        ),
    ];

    for (mesh, transform, material) in statics {
        commands.spawn((
            DemoScene,
            MeshRef(mesh),
            material,
            transform,
            GlobalTransform(transform.matrix()),
        ));
    }

    // The one moving piece. `Interpolated` only goes on entities that actually
    // move — the static six render straight from their `Transform`.
    let box_transform = Transform::from_translation(Vec3::new(0.0, 0.5, 0.0));
    let spinner = commands
        .spawn((
            DemoScene,
            MeshRef(twisted),
            Material::vertex_colored(),
            box_transform,
            GlobalTransform(box_transform.matrix()),
            Interpolated::from(&box_transform),
            Spin {
                axis: Vec3::Y,
                rad_per_sec: DEMO_SPIN,
            },
        ))
        .id();
    commands.insert_resource(DemoEntity(spinner));

    // One camera, at the pose the host used to hardcode, gently following the
    // spinner. The box does not translate, so the follow shows up as a soft
    // settle over the first second and a rock-steady frame after that.
    let eye = Transform::looking_at(DEMO_EYE, Vec3::ZERO, Vec3::Y);
    commands.spawn((
        DemoScene,
        Camera::default(),
        eye,
        GlobalTransform(eye.matrix()),
        Interpolated::from(&eye),
        FollowCamera {
            target: spinner,
            offset: DEMO_EYE - box_transform.translation,
            stiffness: 2.0,
        },
    ));
}
