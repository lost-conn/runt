//! Content-addressed mesh dedup (DESIGN §5).
//!
//! The claim under test: two entities whose generators produced identical
//! geometry share one set of GPU buffers, automatically, because the key *is*
//! the content. The CPU half needs no adapter; the GPU half skips when there is
//! none, exactly like the screenshot test.

use runt_core::registry::{MeshHandle, MeshLibrary, MeshRegistry};
use runt_core::scene;
use runt_core::Sim;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// A device or `None` when the box has no usable adapter.
fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    match pollster::block_on(runt_core::headless_device()) {
        Ok(pair) => Some(pair),
        Err(e) => {
            eprintln!("SKIP (no GPU adapter): {e}");
            None
        }
    }
}

// ---------------------------------------------------------------------------
// CPU side
// ---------------------------------------------------------------------------

#[test]
fn library_collapses_identical_meshes_onto_one_handle() {
    let mut library = MeshLibrary::new();

    let a = library.insert(scene::ball_mesh());
    let b = library.insert(scene::ball_mesh());
    assert_eq!(a, b, "same generator params must produce the same handle");
    assert_eq!(library.len(), 1, "and must not store the mesh twice");

    // Different params → different content → different handle.
    let c = library.insert(scene::post_mesh());
    assert_ne!(a, c);
    assert_eq!(library.len(), 2);

    assert_eq!(a, MeshHandle::of(&scene::ball_mesh()), "handle is pure");
}

#[test]
fn demo_scene_is_seven_entities_over_six_meshes() {
    let mut sim = Sim::new();
    assert_eq!(sim.draw_list().len(), 7, "one entity per shape, plus the twin");
    assert_eq!(
        sim.mesh_library().len(),
        6,
        "the twin sphere shares the ball's geometry"
    );

    // Two draws must be pointing at the same handle — that is the dedup, seen
    // from the render path rather than from the library.
    let mut handles: Vec<u64> = sim.draw_list().iter().map(|d| d.mesh.0).collect();
    handles.sort_unstable();
    handles.dedup();
    assert_eq!(handles.len(), 6);
}

// ---------------------------------------------------------------------------
// GPU side
// ---------------------------------------------------------------------------

#[test]
fn registering_the_same_mesh_twice_uploads_one_buffer_set() {
    let Some((device, _queue)) = device() else {
        return;
    };
    let mut registry = MeshRegistry::new();

    let a = registry.register(&device, &scene::ball_mesh());
    let b = registry.register(&device, &scene::ball_mesh());
    assert_eq!(a, b);
    assert_eq!(registry.len(), 1, "the second register must be a no-op");

    let gpu = registry.get(a).expect("resident");
    let mesh = scene::ball_mesh();
    assert_eq!(gpu.index_count as usize, mesh.indices.len());
    assert_eq!(gpu.vertex_count as usize, mesh.positions.len());

    let c = registry.register(&device, &scene::spike_mesh());
    assert_ne!(a, c);
    assert_eq!(registry.len(), 2);
    assert!(registry.contains(c));
}

#[test]
fn rendering_the_demo_uploads_exactly_six_meshes() {
    let Some(mut engine) = pollster::block_on(runt_core::Engine::headless(FORMAT))
        .map_err(|e| eprintln!("SKIP (no GPU adapter): {e}"))
        .ok()
    else {
        return;
    };

    let target = engine.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("target"),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    engine.update(0.0);
    engine.render(&view, 64, 64);

    assert_eq!(
        engine.renderer().meshes().len(),
        6,
        "seven entities, six uploads"
    );
    assert_eq!(
        engine.renderer().pipeline_count(),
        2,
        "vertex-colored and flat materials are two variants"
    );

    // A second frame must not upload anything again.
    engine.render(&view, 64, 64);
    assert_eq!(engine.renderer().meshes().len(), 6);
}
