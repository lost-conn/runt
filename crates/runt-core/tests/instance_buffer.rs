//! The per-instance vertex buffer: layout, growth, and coalescing (DESIGN §5,
//! §11 — D3).
//!
//! > *True GPU instancing (per-instance vertex buffer) is the first
//! > optimization once the same (mesh, material) repeats a lot — the sort order
//! > already groups for it.* — DESIGN §5
//!
//! That is now the *only* path. The dynamic-uniform-offset draw is gone: every
//! entity is an [`InstanceRaw`] in one tightly-packed vertex buffer, and a run
//! of adjacent items agreeing on (variant, mesh, texture) is one
//! `draw_indexed`. A lone entity is a run of length one, which is what keeps it
//! one code path rather than a fast case and a slow case.
//!
//! Four claims here:
//!
//! 1. The layout is what WebGL2 can take — 96-byte stride, ten attributes
//!    across two buffers, well inside `downlevel_webgl2_defaults`.
//! 2. 500 entities sharing one mesh and one material collapse to **one** draw
//!    call. This is the whole milestone, measured through `draw_stats`.
//! 3. Distinct state still separates: a second material is a second draw.
//! 4. The buffer grows geometrically and stays grown.
//!
//! Plus an ignored 5 000-instance microbench (`--ignored`) that reports CPU
//! submit time.

use bevy_ecs::prelude::World;
use glam::{Mat4, Vec3, Vec4};
use runt_core::draw::{coalesce_draws, DrawItem, FrameParams};
use runt_core::registry::{MeshHandle, MeshLibrary};
use runt_core::texture::TextureLibrary;
use runt_core::{scene, Camera, InstanceRaw, MaterialVariant, Renderer, Transform};

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

/// A camera far enough back that the whole crowd is inside the frustum — the
/// culler (D5) is a separate test's business, and it must not quietly answer
/// this one's question.
fn frame() -> FrameParams {
    let pose = Transform::looking_at(Vec3::new(0.0, 2.0, 40.0), Vec3::ZERO, Vec3::Y);
    FrameParams {
        view_proj: Camera::default().view_proj(pose.matrix(), 1.0),
        ..FrameParams::default()
    }
}

/// `count` draws of one shared mesh and one shared material, in a tidy cube
/// around the origin. Entities come from a real world so the values are ones
/// bevy_ecs would hand out, not hand-rolled bit patterns.
fn crowd(world: &mut World, mesh: MeshHandle, count: usize) -> Vec<DrawItem> {
    (0..count)
        .map(|i| {
            // 10 × 10 columns marching *away* from the camera, so however long
            // the crowd gets it stays in front of it and the frustum test (D5)
            // never quietly answers this file's questions for it.
            let (x, y, z) = ((i % 10) as f32, ((i / 10) % 10) as f32, (i / 100) as f32);
            DrawItem {
                entity: world.spawn_empty().id(),
                variant: MaterialVariant::VERTEX_COLOR,
                mesh,
                model: Mat4::from_translation(Vec3::new(x - 5.0, y - 5.0, -z)),
                base_color: Vec4::ONE,
                params: Vec4::ZERO,
                texture: None,
            }
        })
        .collect()
}

#[test]
fn the_instance_layout_fits_webgl2() {
    // 96 bytes, tightly packed. The old uniform path strided these by the
    // device's `min_uniform_buffer_offset_alignment` — 256 under WebGL2 limits,
    // so 62% of that buffer was padding.
    assert_eq!(std::mem::size_of::<InstanceRaw>(), 96);
    assert_eq!(InstanceRaw::LAYOUT.array_stride, 96);
    assert_eq!(InstanceRaw::LAYOUT.step_mode, wgpu::VertexStepMode::Instance);

    // Ten attributes across the two buffers: 0–3 the mesh, 4–9 the instance.
    // `downlevel_webgl2_defaults` grants 16 attributes and 8 vertex buffers.
    let mesh_locations: Vec<u32> = runt_core::Vertex::LAYOUT
        .attributes
        .iter()
        .map(|a| a.shader_location)
        .collect();
    let instance_locations: Vec<u32> = InstanceRaw::LAYOUT
        .attributes
        .iter()
        .map(|a| a.shader_location)
        .collect();
    assert_eq!(mesh_locations, vec![0, 1, 2, 3]);
    assert_eq!(instance_locations, vec![4, 5, 6, 7, 8, 9]);

    let limits = wgpu::Limits::downlevel_webgl2_defaults();
    assert!(mesh_locations.len() + instance_locations.len() <= limits.max_vertex_attributes as usize);
    assert!(limits.max_vertex_buffers >= 2);

    // The four matrix columns are contiguous and in column-major order, which
    // is what lets the vertex shader rebuild `glam`'s own `Mat4` bit for bit.
    let offsets: Vec<u64> = InstanceRaw::LAYOUT
        .attributes
        .iter()
        .map(|a| a.offset)
        .collect();
    assert_eq!(offsets, vec![0, 16, 32, 48, 64, 80]);
}

#[test]
fn identical_state_coalesces_and_distinct_state_does_not() {
    // No GPU needed: coalescing is a pure function of the sorted list.
    let mut world = World::new();
    let mut items = crowd(&mut world, MeshHandle(1), 4);
    assert_eq!(coalesce_draws(&items).len(), 1, "four identical → one draw");

    // A different variant breaks the run — and after a sort it would break it
    // into two contiguous halves, not four singletons.
    items[2].variant = MaterialVariant::NONE;
    runt_core::draw::sort_draw_list(&mut items);
    let runs = coalesce_draws(&items);
    assert_eq!(runs.len(), 2);
    assert_eq!(runs[0].count, 1, "the odd one out is alone");
    assert_eq!(runs[1].count, 3);
    assert_eq!(runs[1].first, 1, "ranges are contiguous and cover the list");

    // Every instance is accounted for exactly once, in order.
    let covered: u32 = runs.iter().map(|r| r.count).sum();
    assert_eq!(covered as usize, items.len());
    for pair in runs.windows(2) {
        assert_eq!(pair[0].first + pair[0].count, pair[1].first);
    }

    // An empty frame issues nothing at all.
    assert!(coalesce_draws(&[]).is_empty());
}

#[test]
fn five_hundred_shared_entities_are_one_draw_call() {
    let mut renderer = match pollster::block_on(Renderer::headless(FORMAT)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("SKIP (no GPU adapter): {e}");
            return;
        }
    };
    let view = target(&renderer);

    let mut world = World::new();
    let mut library = MeshLibrary::new();
    // No textures in this scene: the default 1×1 bind group is what every draw
    // gets, which is exactly the pre-texture path (DESIGN §7).
    let textures = TextureLibrary::new();
    let mesh = library.insert(scene::ball_mesh());
    let frame = frame();

    let draws = crowd(&mut world, mesh, 500);
    renderer.render(&view, SIZE, SIZE, &frame, &draws, &library, &textures);

    let stats = renderer.draw_stats();
    println!("500 shared entities: {stats:?}");
    assert_eq!(stats.items, 500);
    assert_eq!(stats.culled, 0, "the whole crowd is in front of the camera");
    assert_eq!(stats.instances, 500);
    assert_eq!(stats.draws, 1, "one pipeline, one mesh, one texture → one draw");

    // And one mesh, shared by every one of those instances.
    assert_eq!(renderer.meshes().len(), 1);

    // Half of them in a second material: two pipelines, two draws, still 500
    // instances. The sort groups them, so it is two runs and not five hundred.
    let mut mixed = draws.clone();
    for item in mixed.iter_mut().step_by(2) {
        item.variant = MaterialVariant::NONE;
    }
    runt_core::draw::sort_draw_list(&mut mixed);
    renderer.render(&view, SIZE, SIZE, &frame, &mixed, &library, &textures);
    let stats = renderer.draw_stats();
    println!("500 entities in two materials: {stats:?}");
    assert_eq!(stats.instances, 500);
    assert_eq!(stats.draws, 2);
}

#[test]
fn the_instance_buffer_grows_and_stays_grown() {
    let mut renderer = match pollster::block_on(Renderer::headless(FORMAT)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("SKIP (no GPU adapter): {e}");
            return;
        }
    };
    let view = target(&renderer);

    let mut world = World::new();
    let mut library = MeshLibrary::new();
    let textures = TextureLibrary::new();
    let mesh = library.insert(scene::ball_mesh());
    let frame = frame();

    let start = renderer.instance_capacity();
    let small = crowd(&mut world, mesh, 4);
    renderer.render(&view, SIZE, SIZE, &frame, &small, &library, &textures);
    assert_eq!(
        renderer.instance_capacity(),
        start,
        "a small scene must not reallocate"
    );

    // Past capacity: the buffer grows (geometrically) and the frame still draws
    // — a validation error here would surface as a panic from wgpu.
    let big = (start as usize) * 3 + 1;
    let many = crowd(&mut world, mesh, big);
    renderer.render(&view, SIZE, SIZE, &frame, &many, &library, &textures);
    assert!(
        renderer.instance_capacity() >= big as u32,
        "capacity {} must cover {big} instances",
        renderer.instance_capacity()
    );

    // Growth is sticky: shrinking back does not thrash the allocation.
    let grown = renderer.instance_capacity();
    renderer.render(&view, SIZE, SIZE, &frame, &small, &library, &textures);
    assert_eq!(renderer.instance_capacity(), grown);
    assert_eq!(renderer.draw_stats().draws, 1);
}

/// The cheapest possible geometry: one triangle.
///
/// The microbench below measures the **CPU submit path** — cull, sort,
/// coalesce, pack, `write_buffer`, encode — and that path does not care how
/// many triangles a mesh has. A real mesh does: 5 000 instances of the demo
/// sphere is millions of triangles a frame, the queue backs up, and `submit`
/// starts blocking on the GPU. The number that comes out of *that* is a
/// measurement of this machine's vertex throughput wearing a CPU bench's
/// clothes, which is exactly the trap this mesh exists to avoid.
fn one_triangle() -> runt_core::Mesh {
    runt_core::Mesh {
        positions: vec![
            Vec3::new(-0.5, -0.5, 0.0),
            Vec3::new(0.5, -0.5, 0.0),
            Vec3::new(0.0, 0.5, 0.0),
        ],
        normals: vec![Vec3::Z; 3],
        uvs: vec![glam::Vec2::ZERO; 3],
        colors: vec![Vec3::ONE; 3],
        indices: vec![0, 1, 2],
    }
}

/// 5 000 instances of one mesh: how long the CPU spends turning a draw list
/// into a submitted frame.
///
/// Ignored by default because it is a measurement, not an assertion — a shared
/// CI box would fail a threshold for reasons that are not regressions. Run it
/// with `cargo test -p runt-core --release -- --ignored --nocapture`. The
/// budget it exists to watch is **~1 ms** of submit time, which is what the
/// port's spike fields (4–6k instances) need to fit inside a 16.7 ms frame.
#[test]
#[ignore = "microbench: run with --ignored --nocapture"]
fn five_thousand_instance_submit_microbench() {
    let mut renderer = match pollster::block_on(Renderer::headless(FORMAT)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("SKIP (no GPU adapter): {e}");
            return;
        }
    };
    let view = target(&renderer);

    let mut world = World::new();
    let mut library = MeshLibrary::new();
    let textures = TextureLibrary::new();
    let mesh = library.insert(one_triangle());
    let frame = frame();
    const COUNT: usize = 5_000;
    let draws = crowd(&mut world, mesh, COUNT);

    // Warm: first frame uploads the mesh, compiles the pipeline and grows the
    // buffer, none of which is per-frame work.
    for _ in 0..5 {
        renderer.render(&view, SIZE, SIZE, &frame, &draws, &library, &textures);
    }
    renderer
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");

    const ITERS: u32 = 50;
    let start = std::time::Instant::now();
    for _ in 0..ITERS {
        renderer.render(&view, SIZE, SIZE, &frame, &draws, &library, &textures);
    }
    let cpu = start.elapsed() / ITERS;
    renderer
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");

    let stats = renderer.draw_stats();
    println!(
        "{COUNT} instances → {} draw call(s); CPU submit {:.3} ms/frame ({:.1} ns/instance)",
        stats.draws,
        cpu.as_secs_f64() * 1e3,
        cpu.as_nanos() as f64 / COUNT as f64,
    );
    assert_eq!(stats.draws, 1);
    assert_eq!(stats.instances, COUNT as u32);
}
