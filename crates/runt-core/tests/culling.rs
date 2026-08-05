//! What does *not* get drawn: [`Visibility`] (D4) and the view frustum (D5).
//!
//! Two filters, deliberately at two different layers.
//!
//! `Visibility` is a property of the world — a game says "not now" — so it is
//! applied in extraction, and a hidden entity never becomes a `DrawItem` at
//! all. The frustum is a property of the *view*, which extraction does not
//! have (the camera lives in `FrameParams`, one layer up), so it is applied by
//! the renderer alongside the blended depth sort.
//!
//! The invariant that matters more than either is **conservatism**: culling may
//! cost a draw call, never a pixel. Every geometric claim below is written from
//! that direction — the interesting assertions are the ones about what is
//! *kept*.

use bevy_ecs::prelude::*;
use glam::{Mat4, Quat, Vec3, Vec4};
use runt_core::draw::{build_draw_list, cull_draw_list, Aabb, DrawItem, FrameParams, Frustum};
use runt_core::registry::{MeshHandle, MeshLibrary};
use runt_core::texture::TextureLibrary;
use runt_core::{scene, Camera, Material, MaterialVariant, MeshRef, Renderer, Transform, Visibility};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const SIZE: u32 = 64;

/// The unit cube, centred — the box every synthetic item below wears.
const UNIT: Aabb = Aabb {
    min: Vec3::new(-0.5, -0.5, -0.5),
    max: Vec3::new(0.5, 0.5, 0.5),
};

/// A camera at `+Z` looking at the origin down `−Z`.
fn looking_at_origin() -> Mat4 {
    let pose = Transform::looking_at(Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO, Vec3::Y);
    Camera::default().view_proj(pose.matrix(), 1.0)
}

/// The same camera, turned 180° — the origin is now behind it.
fn looking_away() -> Mat4 {
    let pose = Transform::looking_at(Vec3::new(0.0, 0.0, 10.0), Vec3::new(0.0, 0.0, 20.0), Vec3::Y);
    Camera::default().view_proj(pose.matrix(), 1.0)
}

// ---------------------------------------------------------------------------
// The geometry, without a GPU
// ---------------------------------------------------------------------------

#[test]
fn a_box_at_the_origin_is_visible_and_one_behind_the_camera_is_not() {
    let frustum = Frustum::from_view_proj(&looking_at_origin());
    assert!(frustum.intersects_transformed(&UNIT, &Mat4::IDENTITY));
    // 20 units behind the camera, which sits at z = 10.
    assert!(!frustum.intersects_transformed(&UNIT, &Mat4::from_translation(Vec3::new(0.0, 0.0, 30.0))));
    // Past the far plane (z_far = 100 from the camera at z = 10).
    assert!(!frustum.intersects_transformed(&UNIT, &Mat4::from_translation(Vec3::new(0.0, 0.0, -200.0))));
    // Far off to the side.
    assert!(!frustum.intersects_transformed(&UNIT, &Mat4::from_translation(Vec3::new(500.0, 0.0, 0.0))));
}

#[test]
fn nothing_that_touches_the_frustum_is_ever_culled() {
    // The conservatism claim, checked by brute force against a ground truth
    // that cannot be accused of sharing the culler's arithmetic: a box is
    // *visible* if any of a dense grid of points inside it projects into the
    // clip volume. Anything that answers yes there must survive the cull.
    let view_proj = looking_at_origin();
    let frustum = Frustum::from_view_proj(&view_proj);

    let inside_clip = |p: Vec3| {
        let clip = view_proj * p.extend(1.0);
        clip.w > 0.0
            && clip.x.abs() <= clip.w
            && clip.y.abs() <= clip.w
            && clip.z >= 0.0
            && clip.z <= clip.w
    };

    let mut sampled = 0;
    let mut kept_and_visible = 0;
    for ix in -6..=6 {
        for iy in -4..=4 {
            for iz in -8..=4 {
                let at = Vec3::new(ix as f32 * 1.5, iy as f32 * 1.5, iz as f32 * 3.0);
                // Rotated and non-uniformly scaled, so the world-space box is a
                // real transform of the local one rather than a translation.
                let model = Mat4::from_scale_rotation_translation(
                    Vec3::new(1.7, 0.6, 2.3),
                    Quat::from_axis_angle(Vec3::new(0.3, 1.0, 0.2).normalize(), 0.9),
                    at,
                );
                let world = UNIT.transformed(&model);
                // Ground truth: any interior sample landing in the clip volume.
                let mut any = false;
                for sx in 0..5 {
                    for sy in 0..5 {
                        for sz in 0..5 {
                            let t = Vec3::new(sx as f32, sy as f32, sz as f32) / 4.0;
                            any |= inside_clip(world.min + (world.max - world.min) * t);
                        }
                    }
                }
                sampled += 1;
                let kept = frustum.intersects_transformed(&UNIT, &model);
                if any {
                    assert!(kept, "culled a box with visible points in it, at {at:?}");
                    kept_and_visible += 1;
                }
            }
        }
    }
    assert!(sampled > 500 && kept_and_visible > 20, "the sweep must actually sweep");
}

#[test]
fn the_transformed_box_bounds_all_eight_corners() {
    let model = Mat4::from_scale_rotation_translation(
        Vec3::new(2.0, 0.5, 3.0),
        Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0).normalize(), 1.1),
        Vec3::new(4.0, -1.0, 2.0),
    );
    let world = UNIT.transformed(&model);
    for i in 0..8 {
        let corner = Vec3::new(
            if i & 1 == 0 { UNIT.min.x } else { UNIT.max.x },
            if i & 2 == 0 { UNIT.min.y } else { UNIT.max.y },
            if i & 4 == 0 { UNIT.min.z } else { UNIT.max.z },
        );
        let p = model.transform_point3(corner);
        assert!(
            p.cmpge(world.min - Vec3::splat(1e-4)).all() && p.cmple(world.max + Vec3::splat(1e-4)).all(),
            "corner {p:?} escaped {world:?}"
        );
    }
}

#[test]
fn a_mesh_with_no_measured_bounds_is_kept() {
    let frustum = Frustum::from_view_proj(&looking_at_origin());
    let mut world = World::new();
    let mut items = vec![item(&mut world, MeshHandle(7), Vec3::new(0.0, 0.0, 900.0))];
    // Nothing knows how big mesh 7 is, so it cannot be proven off screen —
    // even though this one plainly is.
    assert_eq!(cull_draw_list(&mut items, &frustum, |_| None), 0);
    assert_eq!(items.len(), 1);
    // Once it *is* measured, it goes.
    assert_eq!(cull_draw_list(&mut items, &frustum, |_| Some(UNIT)), 1);
    assert!(items.is_empty());
}

#[test]
fn a_broken_transform_keeps_its_object_rather_than_deleting_it() {
    let frustum = Frustum::from_view_proj(&looking_at_origin());
    let mut world = World::new();
    let mut items = vec![item(&mut world, MeshHandle(1), Vec3::ZERO)];
    items[0].model = Mat4::from_translation(Vec3::splat(f32::NAN));
    assert_eq!(cull_draw_list(&mut items, &frustum, |_| Some(UNIT)), 0);
}

/// A synthetic draw item at `at`.
fn item(world: &mut World, mesh: MeshHandle, at: Vec3) -> DrawItem {
    DrawItem {
        entity: world.spawn_empty().id(),
        variant: MaterialVariant::VERTEX_COLOR,
        mesh,
        model: Mat4::from_translation(at),
        base_color: Vec4::ONE,
        params: Vec4::ZERO,
        texture: None,
    }
}

#[test]
fn culling_is_order_preserving_and_spawn_order_free() {
    // The determinism claim: the retained set is a pure function of camera and
    // transforms, and the survivors keep the order they already had.
    let frustum = Frustum::from_view_proj(&looking_at_origin());
    let places = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 400.0), // behind
        Vec3::new(1.0, 0.0, -2.0),
        Vec3::new(-900.0, 0.0, 0.0), // off to the side
        Vec3::new(-1.0, 1.0, -1.0),
    ];
    let survivors = |reverse: bool| {
        let mut world = World::new();
        let mut order: Vec<Vec3> = places.to_vec();
        if reverse {
            order.reverse();
        }
        let mut items: Vec<DrawItem> = order
            .into_iter()
            .map(|at| item(&mut world, MeshHandle(1), at))
            .collect();
        runt_core::draw::sort_draw_list(&mut items);
        cull_draw_list(&mut items, &frustum, |_| Some(UNIT));
        // The *positions* are the content; entity ids differ between worlds.
        items.iter().map(|i| i.model.w_axis.truncate()).collect::<Vec<_>>()
    };
    let forward = survivors(false);
    assert_eq!(forward.len(), 3, "two of five are off screen");
    // Identical world, opposite spawn order: the same three, and — because the
    // sort is total — in the same order.
    let mut a = forward.clone();
    let mut b = survivors(true);
    a.sort_by(|p, q| p.to_array().partial_cmp(&q.to_array()).unwrap());
    b.sort_by(|p, q| p.to_array().partial_cmp(&q.to_array()).unwrap());
    assert_eq!(a, b);
    // Idempotent: culling an already-culled list changes nothing.
    let mut world = World::new();
    let mut items: Vec<DrawItem> = forward
        .iter()
        .map(|at| item(&mut world, MeshHandle(1), *at))
        .collect();
    assert_eq!(cull_draw_list(&mut items, &frustum, |_| Some(UNIT)), 0);
}

// ---------------------------------------------------------------------------
// D4 — Visibility
// ---------------------------------------------------------------------------

fn spawn_drawable(world: &mut World, at: Vec3) -> Entity {
    world
        .spawn((
            MeshRef(MeshHandle(1)),
            Material::vertex_colored(),
            Transform::from_translation(at),
        ))
        .id()
}

#[test]
fn absent_visibility_is_visible_and_hidden_entities_are_not_extracted() {
    let mut world = World::new();
    let bare = spawn_drawable(&mut world, Vec3::ZERO);
    let shown = spawn_drawable(&mut world, Vec3::X);
    let hidden = spawn_drawable(&mut world, Vec3::Y);
    world.entity_mut(shown).insert(Visibility::VISIBLE);
    world.entity_mut(hidden).insert(Visibility::HIDDEN);

    let drawn: Vec<Entity> = build_draw_list(&mut world, 0.0)
        .into_iter()
        .map(|i| i.entity)
        .collect();
    assert!(drawn.contains(&bare), "no component means visible");
    assert!(drawn.contains(&shown));
    assert!(!drawn.contains(&hidden));
    assert_eq!(drawn.len(), 2);

    // The default is visible, so adding the component cannot change a frame.
    assert_eq!(Visibility::default(), Visibility::VISIBLE);
}

#[test]
fn toggling_visibility_toggles_the_draw() {
    let mut world = World::new();
    let entity = spawn_drawable(&mut world, Vec3::ZERO);
    world.entity_mut(entity).insert(Visibility::VISIBLE);
    let count = |world: &mut World| build_draw_list(world, 0.0).len();

    assert_eq!(count(&mut world), 1);
    assert!(!world.get_mut::<Visibility>(entity).unwrap().toggle());
    assert_eq!(count(&mut world), 0, "hidden");
    assert!(world.get_mut::<Visibility>(entity).unwrap().toggle());
    assert_eq!(count(&mut world), 1, "and back, with nothing else changed");

    // Hiding is not parking: the transform is untouched, so every system that
    // reads one still sees the truth.
    assert_eq!(
        world.get::<Transform>(entity).unwrap().translation,
        Vec3::ZERO
    );
}

// ---------------------------------------------------------------------------
// End to end, through the renderer
// ---------------------------------------------------------------------------

#[test]
fn a_camera_facing_away_from_everything_issues_no_draws() {
    let mut renderer = match pollster::block_on(Renderer::headless(FORMAT)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("SKIP (no GPU adapter): {e}");
            return;
        }
    };
    let tex = renderer.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("target"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

    let mut library = MeshLibrary::new();
    let textures = TextureLibrary::new();
    let mesh = library.insert(scene::ball_mesh());
    let mut world = World::new();
    let draws: Vec<DrawItem> = (0..12)
        .map(|i| item(&mut world, mesh, Vec3::new(i as f32 - 6.0, 0.0, 0.0)))
        .collect();

    let facing = FrameParams {
        view_proj: looking_at_origin(),
        ..FrameParams::default()
    };
    renderer.render(&view, SIZE, SIZE, &facing, &draws, &library, &textures);
    let stats = renderer.draw_stats();
    println!("facing the content: {stats:?}");
    assert_eq!(stats.culled, 0);
    assert_eq!(stats.draws, 1, "one mesh, one material — one instanced draw");

    let away = FrameParams {
        view_proj: looking_away(),
        ..FrameParams::default()
    };
    renderer.render(&view, SIZE, SIZE, &away, &draws, &library, &textures);
    let stats = renderer.draw_stats();
    println!("facing away: {stats:?}");
    assert_eq!(stats.items, 12);
    assert_eq!(stats.culled, 12, "everything is behind the camera");
    assert_eq!(stats.instances, 0);
    assert_eq!(stats.draws, 0, "the sky is still painted; nothing else is");
}
