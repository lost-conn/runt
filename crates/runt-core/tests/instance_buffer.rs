//! The per-instance uniform buffer: alignment and growth (DESIGN §5, §11).
//!
//! v1 draws every entity from one uniform buffer with a dynamic offset per
//! draw — no storage buffers, no per-instance vertex stream, so the path is
//! valid under `downlevel_webgl2_defaults`. Two things have to hold: offsets
//! respect the device's `min_uniform_buffer_offset_alignment` (256 there), and
//! a scene with more entities than the buffer holds grows it instead of
//! dropping draws.

use bevy_ecs::prelude::World;
use glam::{Mat4, Vec3, Vec4};
use runt_core::draw::{DrawItem, FrameParams};
use runt_core::registry::{MeshHandle, MeshLibrary};
use runt_core::texture::TextureLibrary;
use runt_core::{scene, Camera, MaterialVariant, Renderer, Transform};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const SIZE: u32 = 64;

fn target(renderer: &Renderer) -> wgpu::TextureView {
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
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

/// `count` draws of one shared mesh, spread along X so none is degenerate.
/// Entities come from a real world so the values are ones bevy_ecs would hand
/// out, not hand-rolled bit patterns.
fn crowd(world: &mut World, mesh: MeshHandle, count: usize) -> Vec<DrawItem> {
    (0..count)
        .map(|i| DrawItem {
            entity: world.spawn_empty().id(),
            variant: MaterialVariant::VERTEX_COLOR,
            mesh,
            model: Mat4::from_translation(Vec3::new(i as f32 * 0.1 - 2.0, 0.0, 0.0)),
            base_color: Vec4::ONE,
            params: Vec4::ZERO,
            texture: None,
        })
        .collect()
}

#[test]
fn instance_slots_are_aligned_and_the_buffer_grows() {
    let mut renderer = match pollster::block_on(Renderer::headless(FORMAT)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("SKIP (no GPU adapter): {e}");
            return;
        }
    };
    let view = target(&renderer);

    let align = renderer
        .device()
        .limits()
        .min_uniform_buffer_offset_alignment;
    let stride = renderer.instance_stride();
    assert_eq!(stride % align, 0, "every dynamic offset must be aligned");
    assert!(
        stride >= std::mem::size_of::<runt_core::InstanceUniform>() as u32,
        "a slot must hold the whole struct"
    );
    // WebGL2 limits are what the engine requests everywhere (DESIGN §11).
    assert_eq!(align, 256, "runt requests downlevel WebGL2 limits");

    let mut world = World::new();
    let mut library = MeshLibrary::new();
    // No textures in this scene: the default 1x1 bind group is what every draw
    // gets, which is exactly the pre-texture path (DESIGN §7).
    let textures = TextureLibrary::new();
    let mesh = library.insert(scene::ball_mesh());

    let camera = Camera::default();
    let pose = Transform::looking_at(Vec3::new(0.0, 2.0, 8.0), Vec3::ZERO, Vec3::Y);
    let frame = FrameParams {
        view_proj: camera.view_proj(pose.matrix(), 1.0),
        ..FrameParams::default()
    };

    let start = renderer.instance_capacity();
    renderer.render(&view, SIZE, SIZE, &frame, &crowd(&mut world, mesh, 4), &library, &textures);
    assert_eq!(
        renderer.instance_capacity(),
        start,
        "a small scene must not reallocate"
    );

    // Past capacity: the buffer grows (geometrically) and the frame still draws
    // — a validation error here would surface as a panic from wgpu.
    let big = (start as usize) * 3 + 1;
    renderer.render(&view, SIZE, SIZE, &frame, &crowd(&mut world, mesh, big), &library, &textures);
    assert!(
        renderer.instance_capacity() >= big as u32,
        "capacity {} must cover {big} draws",
        renderer.instance_capacity()
    );

    // And one mesh, shared by every one of those draws.
    assert_eq!(renderer.meshes().len(), 1);

    // Growth is sticky: shrinking back does not thrash the allocation.
    let grown = renderer.instance_capacity();
    renderer.render(&view, SIZE, SIZE, &frame, &crowd(&mut world, mesh, 2), &library, &textures);
    assert_eq!(renderer.instance_capacity(), grown);
}
