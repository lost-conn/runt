//! Offscreen scene targets (`Renderer::render_to_texture`) and the multi-texture
//! UI batch they exist to be sampled by — the engine half of a live 3-D demo
//! viewport inside a UI card.
//!
//! Six claims, in ascending order of how much of the engine they touch:
//!
//! 1. **An offscreen scene is the scene.** The same draw list, the same camera
//!    and the same size rendered into a target and into a host view come out
//!    byte for byte identical. One `encode_scene`, two destinations.
//! 2. **A viewport beside a frame does not touch it.** The host frame is the
//!    same bytes whether an offscreen target was rendered before it, after it,
//!    or not at all — which is the "no RTT in use is byte-identical" gate
//!    stated in the form that a *used* target also has to pass.
//!    (`tests/ui.rs` holds the golden demo frame itself against E2's pin; that
//!    hash is the other half of this claim and it is not restated here.)
//! 3. **Two worlds, one upload per content.** A second `Sim` whose scene
//!    generated the same geometry adds nothing to the mesh registry: handles
//!    are content hashes, so sharing is what happens when nobody arranges
//!    anything.
//! 4. **Each camera sees its own frame block.** Two cameras in one frame, in
//!    either order, produce exactly the two pictures they produce alone. The
//!    guarantee has two halves: each target owns a uniform buffer, and each
//!    render is its own submission (a `queue.write_buffer` lands on the queue
//!    timeline in call order, so a submission reads what was written before
//!    it).
//! 5. **Two textures in one UI frame.** A viewport quad and glyph-atlas quads
//!    in the same batch, sampling different textures, with painter's order
//!    preserved *across* the texture changes.
//! 6. **The handle is reserved, sticky and droppable.** A target's handle can
//!    never be a content hash, its allocation survives a same-size frame, and
//!    dropping it takes the registry entry with it.
//!
//! Every GPU test skips (loudly) with no adapter, like the rest of the suite.

use glam::{Mat4, Vec3, Vec4};
use runt_core::draw::{DrawItem, FrameParams};
use runt_core::registry::MeshLibrary;
use runt_core::texture::{TextureHandle, TextureLibrary};
use runt_core::{
    scene, Camera, Engine, MaterialVariant, RenderTarget, Renderer, Sim, Transform, UiBatch,
};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const SIZE: u32 = 96;

fn renderer() -> Option<Renderer> {
    match pollster::block_on(Renderer::headless(FORMAT)) {
        Ok(r) => Some(r),
        Err(e) => {
            eprintln!("SKIP (no GPU adapter): {e}");
            None
        }
    }
}

/// A plain colour target to stand in for the host's view.
fn host_target(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("host view"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Tightly-packed RGBA8 out of any `COPY_SRC` texture.
fn read(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Pixels {
    let (width, height) = (texture.width(), texture.height());
    let unpadded_row = width * 4;
    let padded_row = unpadded_row.div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (padded_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let (tx, rx) = std::sync::mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll device");
    rx.recv().expect("map callback").expect("buffer mapped");

    let padded = readback.get_mapped_range(..).expect("mapped range");
    let mut data = Vec::with_capacity((unpadded_row * height) as usize);
    for row in 0..height as usize {
        let start = row * padded_row as usize;
        data.extend_from_slice(&padded[start..start + unpadded_row as usize]);
    }
    drop(padded);
    readback.unmap();
    Pixels { data, width }
}

/// A frame's pixels, with the width needed to index them.
#[derive(PartialEq)]
struct Pixels {
    data: Vec<u8>,
    width: u32,
}

impl Pixels {
    fn at(&self, x: u32, y: u32) -> [f32; 4] {
        let i = (y as usize * self.width as usize + x as usize) * 4;
        [
            self.data[i] as f32 / 255.0,
            self.data[i + 1] as f32 / 255.0,
            self.data[i + 2] as f32 / 255.0,
            self.data[i + 3] as f32 / 255.0,
        ]
    }
}

/// The pixels a scene target currently holds — through the registry, which is
/// where its colour texture lives.
fn read_target(renderer: &Renderer, target: RenderTarget) -> Pixels {
    let gpu = renderer
        .textures()
        .get(target.handle())
        .expect("the target is registered under its handle");
    read(renderer.device(), renderer.queue(), &gpu.albedo)
}

#[track_caller]
fn assert_rgba(got: [f32; 4], want: [f32; 4], what: &str) {
    let tolerance = 2.0 / 255.0;
    assert!(
        (0..4).all(|c| (got[c] - want[c]).abs() <= tolerance),
        "{what}: got {got:?}, want {want:?}"
    );
}

/// A camera pose, as the view-projection a frame carries.
fn camera_at(eye: Vec3, at: Vec3, aspect: f32) -> Mat4 {
    Camera::default().view_proj(Transform::looking_at(eye, at, Vec3::Y).matrix(), aspect)
}

/// One ball at the origin: enough geometry that a camera move is visible and
/// enough that "the scene drew" is not a claim about the sky alone.
fn ball_scene() -> (MeshLibrary, TextureLibrary, Vec<DrawItem>) {
    let mut library = MeshLibrary::new();
    let mesh = library.insert(scene::ball_mesh());
    let mut world = bevy_ecs::world::World::new();
    let draws = vec![DrawItem {
        entity: world.spawn_empty().id(),
        variant: MaterialVariant::VERTEX_COLOR,
        mesh,
        model: Mat4::from_scale(Vec3::splat(2.0)),
        base_color: Vec4::ONE,
        params: Vec4::ZERO,
        texture: None,
    }];
    (library, TextureLibrary::new(), draws)
}

// ---------------------------------------------------------------------------
// 1. An offscreen scene is the scene
// ---------------------------------------------------------------------------

#[test]
fn a_scene_drawn_into_a_target_is_the_scene_drawn_into_a_view() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let device = renderer.device().clone();
    let queue = renderer.queue().clone();
    let (host, view) = host_target(&device, SIZE, SIZE);
    let (library, textures, draws) = ball_scene();
    let frame = FrameParams {
        view_proj: camera_at(Vec3::new(0.0, 0.0, 6.0), Vec3::ZERO, 1.0),
        ..FrameParams::default()
    };

    renderer.render(&view, SIZE, SIZE, &frame, &draws, &library, &textures);
    let in_view = read(&device, &queue, &host);

    let target = RenderTarget(0);
    let stats = renderer.render_to_texture(target, SIZE, SIZE, &frame, &draws, &library, &textures);
    let offscreen = read_target(&renderer, target);

    println!("offscreen stats: {stats:?}");
    assert_eq!(stats.items, 1);
    assert_eq!(
        stats.draws, 1,
        "one mesh, one material — one instanced draw"
    );
    assert!(
        in_view.data == offscreen.data,
        "the offscreen scene is not the frame the same camera draws into a view"
    );

    // …and it is not vacuous: the ball is actually in both of them, so this is
    // a claim about geometry and not only about a shared sky.
    let sky_only = {
        renderer.render(&view, SIZE, SIZE, &frame, &[], &library, &textures);
        read(&device, &queue, &host)
    };
    let middle = SIZE / 2;
    assert_ne!(
        offscreen.at(middle, middle),
        sky_only.at(middle, middle),
        "nothing was drawn into the target but the sky"
    );

    // The handle the caller was promised before any of this happened.
    assert_eq!(target.handle(), TextureHandle::render_target(0));
    assert!(renderer.textures().contains(target.handle()));
}

// ---------------------------------------------------------------------------
// 2. A viewport beside a frame does not touch it
// ---------------------------------------------------------------------------

#[test]
fn a_host_frame_is_the_same_bytes_with_a_viewport_beside_it() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let device = renderer.device().clone();
    let queue = renderer.queue().clone();
    let (host, view) = host_target(&device, SIZE, SIZE);
    let (library, textures, draws) = ball_scene();

    // Deliberately different cameras and different sizes: the target is a
    // 48×32 card and the host frame is a square, so anything the two passes
    // shared — the depth attachment, the instance buffer, the frame block —
    // would show up as a difference here.
    let host_frame = FrameParams {
        view_proj: camera_at(Vec3::new(0.0, 0.0, 6.0), Vec3::ZERO, 1.0),
        ..FrameParams::default()
    };
    let card_frame = FrameParams {
        view_proj: camera_at(Vec3::new(3.0, 2.0, 3.0), Vec3::ZERO, 48.0 / 32.0),
        ..FrameParams::default()
    };
    let target = RenderTarget(7);

    renderer.render(&view, SIZE, SIZE, &host_frame, &draws, &library, &textures);
    let alone = read(&device, &queue, &host);

    // Viewport first, then the host frame.
    renderer.render_to_texture(target, 48, 32, &card_frame, &draws, &library, &textures);
    renderer.render(&view, SIZE, SIZE, &host_frame, &draws, &library, &textures);
    let after_viewport = read(&device, &queue, &host);
    assert!(
        alone.data == after_viewport.data,
        "rendering a viewport before the host frame moved the host frame"
    );

    // Host frame first, then the viewport — the order a game that renders its
    // cards at the end of the frame would produce.
    renderer.render(&view, SIZE, SIZE, &host_frame, &draws, &library, &textures);
    renderer.render_to_texture(target, 48, 32, &card_frame, &draws, &library, &textures);
    let before_viewport = read(&device, &queue, &host);
    assert!(
        alone.data == before_viewport.data,
        "rendering a viewport after the host frame reached back into it"
    );

    // The frame's own introspection still describes the *host* frame: the
    // offscreen pass returns its stats rather than storing them.
    assert_eq!(renderer.draw_stats().items, draws.len() as u32);
    assert_eq!(
        renderer.scaled_target_size(),
        None,
        "an offscreen target is not the render-scale target and must not allocate one"
    );
}

// ---------------------------------------------------------------------------
// 3. Two worlds, one upload per content
// ---------------------------------------------------------------------------

#[test]
fn two_worlds_with_the_same_content_share_one_upload() {
    let mut engine = match pollster::block_on(Engine::headless(FORMAT)) {
        Ok(e) => e,
        Err(e) => return eprintln!("SKIP two-world dedup: {e}"),
    };
    let device = engine.device().clone();
    let (_host, view) = host_target(&device, SIZE, SIZE);

    // The game's world, drawn the way a host draws it.
    engine.update(0.0);
    engine.render(&view, SIZE, SIZE);
    let after_host = engine.renderer().meshes().len();
    assert!(after_host > 0, "the demo scene drew nothing");

    // Two more, fully separate `Sim`s — each its own `World`, its own
    // libraries, its own clock — whose scenes generated byte-identical
    // geometry. Handles are content hashes, so the renderer has nothing left to
    // upload for either of them.
    let mut first = Sim::new();
    let mut second = Sim::new();
    first.update(0.0);
    second.update(0.0);

    engine.render_to_texture(RenderTarget(1), &mut first, 64, 64);
    let meshes = engine.renderer().meshes().len();
    let textures = engine.renderer().textures().len();
    println!("two worlds: {meshes} mesh(es), {textures} texture(s) resident");
    assert_eq!(
        meshes, after_host,
        "a second world re-uploaded geometry the game's world had already uploaded"
    );

    // The third world at the *same* size and tick as the second, so the two
    // draw lists are identical down to which items the frustum kept — which is
    // what makes "no new bakes" a claim about content addressing rather than
    // about two cameras happening to see the same things.
    engine.render_to_texture(RenderTarget(2), &mut second, 64, 64);
    assert_eq!(
        engine.renderer().meshes().len(),
        meshes,
        "a third world uploaded geometry that was already resident"
    );
    assert_eq!(
        engine.renderer().textures().len(),
        textures + 1,
        "the only new texture must be the second scene target itself"
    );
    assert!(engine
        .renderer()
        .textures()
        .contains(RenderTarget(1).handle()));

    // And the worlds really did draw — the dedup claim is about work *skipped*,
    // not about work never asked for. Two identical worlds produce identical
    // viewports; an empty one does not.
    let card = read_target(engine.renderer(), RenderTarget(1));
    let twin = read_target(engine.renderer(), RenderTarget(2));
    assert!(
        card.data == twin.data,
        "two worlds of the same content drew two different pictures"
    );
    let empty = {
        let mut nothing = Sim::without_scene();
        engine.render_to_texture(RenderTarget(3), &mut nothing, 64, 64);
        read_target(engine.renderer(), RenderTarget(3))
    };
    assert!(
        card.data != empty.data,
        "the demo world's viewport is the empty world's viewport"
    );
}

// ---------------------------------------------------------------------------
// 4. Each camera sees its own frame block
// ---------------------------------------------------------------------------

#[test]
fn two_cameras_in_one_frame_each_see_their_own() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let device = renderer.device().clone();
    let queue = renderer.queue().clone();
    let (host, view) = host_target(&device, SIZE, SIZE);
    let (library, textures, draws) = ball_scene();

    // Two very different views of the same ball: close on, and far off to one
    // side. A pass that read the other's view-projection would paint the other
    // picture, which is exactly what the comparisons below can see.
    let near = FrameParams {
        view_proj: camera_at(Vec3::new(0.0, 0.0, 4.0), Vec3::ZERO, 1.0),
        ..FrameParams::default()
    };
    let far = FrameParams {
        view_proj: camera_at(Vec3::new(12.0, 6.0, 12.0), Vec3::ZERO, 1.0),
        ..FrameParams::default()
    };
    let target = RenderTarget(3);

    // Each alone, as the reference.
    renderer.render(&view, SIZE, SIZE, &near, &draws, &library, &textures);
    let solo_host = read(&device, &queue, &host);
    renderer.render_to_texture(target, SIZE, SIZE, &far, &draws, &library, &textures);
    let solo_card = read_target(&renderer, target);
    assert!(
        solo_host.data != solo_card.data,
        "the two cameras draw the same picture; the test cannot see a mix-up"
    );

    // Both in one frame, with nothing waited on in between — the case where a
    // shared frame uniform would hand the second camera's matrices to the first
    // pass to execute.
    for (label, card_first) in [("card first", true), ("host first", false)] {
        if card_first {
            renderer.render_to_texture(target, SIZE, SIZE, &far, &draws, &library, &textures);
            renderer.render(&view, SIZE, SIZE, &near, &draws, &library, &textures);
        } else {
            renderer.render(&view, SIZE, SIZE, &near, &draws, &library, &textures);
            renderer.render_to_texture(target, SIZE, SIZE, &far, &draws, &library, &textures);
        }
        let got_host = read(&device, &queue, &host);
        let got_card = read_target(&renderer, target);
        assert!(
            got_host.data == solo_host.data,
            "{label}: the host frame is not the near camera's"
        );
        assert!(
            got_card.data == solo_card.data,
            "{label}: the viewport is not the far camera's"
        );
    }
}

// ---------------------------------------------------------------------------
// 5. Two textures in one UI frame
// ---------------------------------------------------------------------------

#[test]
fn a_viewport_quad_and_atlas_quads_share_one_batch() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let device = renderer.device().clone();
    let queue = renderer.queue().clone();
    let (host, view) = host_target(&device, 128, 128);
    let (library, textures, draws) = ball_scene();

    // The viewport: a 32×32 picture of the ball, sampled 1:1 by a 32-pixel quad
    // so a probe at (x, y) has an exact expected texel at (x − 16, y − 16).
    let target = RenderTarget(4);
    let card = FrameParams {
        view_proj: camera_at(Vec3::new(2.0, 1.5, 4.0), Vec3::ZERO, 1.0),
        ..FrameParams::default()
    };
    renderer.render_to_texture(target, 32, 32, &card, &draws, &library, &textures);
    let card_pixels = read_target(&renderer, target);

    // The glyph atlas: 2×2 opaque texels, the `UiAtlasImage` door.
    let atlas = TextureHandle(0xa71a_5000);
    renderer.upload_ui_atlas(
        atlas,
        2,
        2,
        &[
            255, 0, 0, 255, // (0,0) red
            0, 255, 0, 255, // (1,0) green
            0, 0, 255, 255, // (0,1) blue
            255, 255, 0, 255, // (1,1) yellow
        ],
    );
    let red = [1.0, 0.0, 0.0, 1.0];
    let green = [0.0, 1.0, 0.0, 1.0];
    let texel = |x: f32, y: f32| [x * 0.5, y * 0.5, x * 0.5 + 0.5, y * 0.5 + 0.5];

    // Panel (atlas red) → viewport → caption (atlas green): three runs, and the
    // third is a *return* to the batch's own atlas.
    let mut batch = UiBatch::new();
    batch.atlas = Some(atlas);
    batch.textured([0.0, 0.0, 32.0, 32.0], texel(0.0, 0.0), [1.0; 4]);
    batch.set_texture(Some(target.handle()));
    batch.textured([16.0, 16.0, 32.0, 32.0], [0.0, 0.0, 1.0, 1.0], [1.0; 4]);
    batch.set_texture(None);
    batch.textured([24.0, 24.0, 8.0, 8.0], texel(1.0, 0.0), [1.0; 4]);
    assert_eq!(batch.runs().count(), 3, "three runs, three draws");

    renderer.set_ui_batch(&batch);
    renderer.render(&view, 128, 128, &card, &[], &library, &textures);
    let painted = read(&device, &queue, &host);

    // The panel where only the panel is.
    assert_rgba(painted.at(4, 4), red, "the atlas quad under everything");
    // The viewport where only the viewport is — texel for texel.
    assert_rgba(
        painted.at(40, 40),
        card_pixels.at(24, 24),
        "the viewport quad samples the scene target",
    );
    assert_rgba(
        painted.at(46, 20),
        card_pixels.at(30, 4),
        "…and it is the whole picture, not one colour",
    );
    // Painter's order across a texture change: the viewport covers the panel.
    assert_rgba(
        painted.at(20, 20),
        card_pixels.at(4, 4),
        "the later viewport quad must win the overlap",
    );
    // …and the caption, on the batch's atlas again, covers the viewport.
    assert_rgba(painted.at(26, 26), green, "the caption over the viewport");

    // The claim is not "textures win over atlases": reverse the two and the
    // overlap reverses with them. Same textures, same rects, opposite order.
    let mut reversed = UiBatch::new();
    reversed.atlas = Some(atlas);
    reversed.set_texture(Some(target.handle()));
    reversed.textured([16.0, 16.0, 32.0, 32.0], [0.0, 0.0, 1.0, 1.0], [1.0; 4]);
    reversed.set_texture(None);
    reversed.textured([0.0, 0.0, 32.0, 32.0], texel(0.0, 0.0), [1.0; 4]);

    renderer.set_ui_batch(&reversed);
    renderer.render(&view, 128, 128, &card, &[], &library, &textures);
    let flipped = read(&device, &queue, &host);
    assert_rgba(
        flipped.at(20, 20),
        red,
        "reversing the list reverses the stack",
    );
    assert_rgba(
        flipped.at(40, 40),
        card_pixels.at(24, 24),
        "and leaves the rest of the viewport alone",
    );

    // A batch that never switches is still one draw and still works — the old
    // path, unchanged, through the new code.
    let mut plain = UiBatch::new();
    plain.atlas = Some(atlas);
    plain.textured([0.0, 0.0, 32.0, 32.0], texel(0.0, 0.0), [1.0; 4]);
    assert_eq!(plain.runs().count(), 1);
    renderer.set_ui_batch(&plain);
    renderer.render(&view, 128, 128, &card, &[], &library, &textures);
    let single = read(&device, &queue, &host);
    assert_rgba(single.at(4, 4), red, "one run, one atlas, one draw");
}

// ---------------------------------------------------------------------------
// 6. The handle: reserved, sticky, droppable
// ---------------------------------------------------------------------------

#[test]
fn a_targets_handle_is_reserved_and_its_allocation_is_sticky() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let target = RenderTarget(11);
    assert!(
        target.handle().is_reserved(),
        "a target's handle must be outside the content-addressed half"
    );
    assert_eq!(renderer.render_target_size(target), None);

    let handle = renderer.ensure_render_target(target, 64, 48);
    assert_eq!(handle, target.handle(), "the handle is knowable up front");
    assert_eq!(renderer.render_target_size(target), Some((64, 48)));

    // Same size: nothing is reallocated, and the registry entry is the one that
    // was already there (a fresh texture would be a fresh, blank viewport every
    // frame).
    let before = renderer.textures().len();
    renderer.ensure_render_target(target, 64, 48);
    assert_eq!(renderer.textures().len(), before);

    // A different size recreates it; the handle does not move, because the
    // handle is a name.
    renderer.ensure_render_target(target, 32, 32);
    assert_eq!(renderer.render_target_size(target), Some((32, 32)));
    assert_eq!(
        renderer.textures().len(),
        before,
        "the old entry was replaced"
    );
    assert_eq!(
        renderer
            .textures()
            .get(handle)
            .expect("resident")
            .albedo
            .width(),
        32
    );

    // Degenerate sizes are clamped rather than handed to wgpu.
    renderer.ensure_render_target(target, 0, 0);
    assert_eq!(renderer.render_target_size(target), Some((1, 1)));

    // Dropping takes the registry entry with it — and a quad still naming the
    // handle degrades to the white texel, which `ui.rs` already pins.
    renderer.drop_render_target(target);
    assert_eq!(renderer.render_target_size(target), None);
    assert!(!renderer.textures().contains(handle));
    renderer.drop_render_target(target); // idempotent

    // Two targets are two textures; the ids are the whole identity.
    renderer.ensure_render_target(RenderTarget(0), 16, 16);
    renderer.ensure_render_target(RenderTarget(1), 16, 16);
    assert_ne!(RenderTarget(0).handle(), RenderTarget(1).handle());
    assert_eq!(renderer.textures().len(), before + 1);
}
