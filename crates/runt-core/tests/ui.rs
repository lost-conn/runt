//! The screen-space UI pass (plan D11, `runt_core::ui`) on a real GPU.
//!
//! Five claims, in ascending order of how much they would cost to get wrong:
//!
//! 1. **An empty batch is free.** The demo frame with no HUD in it hashes to
//!    the pin E2 left behind (`0x12ff80684b74a167`) — byte for byte the frame
//!    an engine without this module draws. No pass, no pipeline, no allocation.
//!    This is the gate: the UI pipeline is allowed to exist only on the terms
//!    that nothing which does not use it can tell.
//! 2. **Premultiplied compositing is what it says.** Quads are held against
//!    `src + dst·(1−srcα)` over the *actual* background, opaque and translucent,
//!    including a translucent quad over an opaque one — which is the case that
//!    separates premultiplied from straight alpha.
//! 3. **Painter's order is `Vec` order.** The later quad wins the overlap.
//! 4. **An atlas quad samples the atlas.** A 4×4 texture baked through
//!    `TextureRegistry`, one texel selected by uv, read back and compared to the
//!    pixels that landed on screen.
//! 5. **Render scale does not touch it.** At 0.5 the world is 2×2 chonk and a
//!    UI quad's edges still land on exact *surface* pixel columns — the property
//!    that only holds because the pass runs after the blit.
//!
//! Plus the WebGL2 translation check `render_scale.rs` runs over the blit: a
//! shader that compiles on Vulkan and fails to translate is a shader that only
//! breaks in a browser.
//!
//! Every GPU test skips (loudly) with no adapter, like the rest of the suite.

use runt_core::texture::TextureSpec;
use runt_core::{
    Engine, FrameParams, MeshLibrary, NoopCache, RenderScale, Renderer, TextureHandle,
    TextureLibrary, UiQuad,
};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

// The demo frame `headless_screenshot.rs` captures, restated so the empty-batch
// gate compares against the same pixels E2 pinned.
const GOLDEN_SIZE: u32 = 512;
const GOLDEN_TICKS: u64 = 42;
/// The pre-E4 frame's FNV-1a: pinned by E1 (blended pass) and E2 (instancing),
/// both of which had to leave it standing. E4 has to leave it standing too.
const GOLDEN_FNV1A: u64 = 0x12ff_8068_4b74_a167;

/// `copy_texture_to_buffer` requires `bytes_per_row` to be a multiple of 256.
fn align_256(n: u32) -> u32 {
    n.div_ceil(256) * 256
}

/// An offscreen RGBA8 target plus the readback that empties it.
struct Target {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

impl Target {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Target {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ui test target"),
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
        Target {
            texture,
            view,
            width,
            height,
        }
    }

    /// Tightly-packed RGBA8, rows unpadded.
    fn read(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<u8> {
        let unpadded_row = self.width * 4;
        let padded_row = align_256(unpadded_row);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ui readback"),
            size: (padded_row * self.height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ui readback"),
        });
        encoder.copy_texture_to_buffer(
            self.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
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
        let mut pixels = Vec::with_capacity((unpadded_row * self.height) as usize);
        for row in 0..self.height as usize {
            let start = row * padded_row as usize;
            pixels.extend_from_slice(&padded[start..start + unpadded_row as usize]);
        }
        drop(padded);
        readback.unmap();
        pixels
    }
}

/// A frame's pixels, with the width needed to index them.
struct Pixels {
    data: Vec<u8>,
    width: u32,
}

impl Pixels {
    /// RGBA at a pixel, `0..=1`.
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

fn renderer() -> Option<Renderer> {
    match pollster::block_on(Renderer::headless(FORMAT)) {
        Ok(r) => Some(r),
        Err(e) => {
            eprintln!("SKIP (no GPU adapter): {e}");
            None
        }
    }
}

/// Draw an empty 3-D scene (the sky, which covers every pixel) plus `quads`,
/// and read the result back. `atlas` is the batch's texture, if any.
fn frame(
    renderer: &mut Renderer,
    target: &Target,
    scale: f32,
    quads: &[UiQuad],
    atlas: Option<TextureHandle>,
) -> Pixels {
    let device = renderer.device().clone();
    let queue = renderer.queue().clone();
    renderer.set_ui_quads(quads, atlas);
    renderer.render_scaled(
        &target.view,
        target.width,
        target.height,
        RenderScale::new(scale),
        &FrameParams::default(),
        &[],
        &MeshLibrary::new(),
        &TextureLibrary::new(),
    );
    Pixels {
        data: target.read(&device, &queue),
        width: target.width,
    }
}

/// `src + dst·(1 − srcα)` — the composite the pass claims, in the test's own
/// arithmetic rather than the shader's.
fn over(src: [f32; 4], dst: [f32; 4]) -> [f32; 4] {
    let inv = 1.0 - src[3];
    [
        src[0] + dst[0] * inv,
        src[1] + dst[1] * inv,
        src[2] + dst[2] * inv,
        src[3] + dst[3] * inv,
    ]
}

#[track_caller]
fn assert_rgba(got: [f32; 4], want: [f32; 4], what: &str) {
    // 2 LSBs of RGBA8: the blend runs in float and rounds once on the way out,
    // so an exact match is not something to demand of every backend.
    let tolerance = 2.0 / 255.0;
    assert!(
        (0..4).all(|c| (got[c] - want[c]).abs() <= tolerance),
        "{what}: got {got:?}, want {want:?}"
    );
}

// ---------------------------------------------------------------------------
// 1. The gate — an empty batch changes nothing
// ---------------------------------------------------------------------------

#[test]
fn an_empty_batch_leaves_the_frame_byte_identical() {
    let mut engine = match pollster::block_on(Engine::headless(FORMAT)) {
        Ok(e) => e,
        Err(e) => return eprintln!("SKIP ui empty-batch gate: {e}"),
    };
    let device = engine.device().clone();
    let queue = engine.queue().clone();
    let target = Target::new(&device, GOLDEN_SIZE, GOLDEN_SIZE);

    // The same drive `headless_screenshot.rs` uses: quarter-tick steps to 42
    // ticks, so the pose is the pinned one rather than a nearby one.
    engine.update(0.0);
    let mut t = 0.0;
    while engine.tick_count() < GOLDEN_TICKS {
        t += runt_core::TICK_DT * 0.25;
        engine.update(t);
    }
    assert_eq!(engine.tick_count(), GOLDEN_TICKS);

    // The world has a `UiBatch` resource from tick zero (a game's HUD system
    // takes `ResMut<UiBatch>` without inserting it) and nothing has filled it.
    assert!(
        engine
            .sim()
            .world()
            .get_resource::<runt_core::UiBatch>()
            .expect("UiBatch is a standard resource")
            .is_empty(),
        "nothing in the demo draws a HUD"
    );

    engine.render(&target.view, GOLDEN_SIZE, GOLDEN_SIZE);

    let pixels = target.read(&device, &queue);
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in &pixels {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    println!("empty-batch frame fnv1a=0x{hash:016x}");
    assert_eq!(
        hash, GOLDEN_FNV1A,
        "an empty UI batch moved the golden frame; the UI pass is not free"
    );

    assert_eq!(engine.renderer().ui_quad_count(), 0);
    assert!(
        !engine.renderer().ui_ready(),
        "an empty batch must not even compile the UI pipeline"
    );
}

#[test]
fn a_batch_that_empties_again_stops_costing_anything() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let device = renderer.device().clone();
    let target = Target::new(&device, 64, 64);

    let bare = frame(&mut renderer, &target, 1.0, &[], None);
    let painted = frame(
        &mut renderer,
        &target,
        1.0,
        &[UiQuad::solid([8.0, 8.0, 16.0, 16.0], [1.0, 0.0, 0.0, 1.0])],
        None,
    );
    assert!(renderer.ui_ready(), "a non-empty batch compiles the pipeline");
    assert_ne!(bare.at(16, 16), painted.at(16, 16), "the quad painted");

    // …and the frame after it goes back to exactly the bare frame. The pipeline
    // is still resident (it is sticky, like the blit's), but nothing is
    // encoded, so the pixels are the ones from before the HUD ever existed.
    let cleared = frame(&mut renderer, &target, 1.0, &[], None);
    assert_eq!(
        cleared.data, bare.data,
        "an emptied batch must leave no trace in the frame"
    );
}

// ---------------------------------------------------------------------------
// 2. Premultiplied compositing
// ---------------------------------------------------------------------------

#[test]
fn quads_composite_premultiplied_over_the_frame() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let device = renderer.device().clone();
    let target = Target::new(&device, 64, 64);

    // The background this pass will be measured against — the sky, whatever it
    // happens to be at each pixel. Holding the composite against the *actual*
    // destination is stronger than picking a background we control: it is the
    // blend equation under test, not the clear colour.
    let bare = frame(&mut renderer, &target, 1.0, &[], None);

    // An opaque red panel and a half-transparent white sheet, overlapping in
    // the middle. Integer rects, so a quad covers pixels [x, x+w) exactly.
    let red = [1.0, 0.0, 0.0, 1.0];
    let veil = [1.0, 1.0, 1.0, 0.5];
    let quads = [
        UiQuad::solid([8.0, 8.0, 24.0, 24.0], red),
        UiQuad::solid([24.0, 24.0, 24.0, 24.0], veil),
    ];
    let painted = frame(&mut renderer, &target, 1.0, &quads, None);

    // Opaque over anything is itself.
    assert_rgba(painted.at(12, 12), red, "the opaque panel");
    assert_rgba(painted.at(31, 12), red, "the panel's last covered column");

    // Translucent over the sky: premultiplied source, `1 − α` of the frame.
    let veil_src = UiQuad::rgba(veil);
    assert_rgba(
        painted.at(40, 40),
        over(veil_src, bare.at(40, 40)),
        "the veil over the sky",
    );

    // Translucent over the opaque panel — the case that tells premultiplied
    // apart from straight alpha. Straight-alpha blending would multiply the
    // already-multiplied source by its alpha again and come out at 0.25 white
    // rather than 0.5.
    assert_rgba(
        painted.at(28, 28),
        over(veil_src, red),
        "the veil over the panel",
    );

    // Outside every rect the frame is untouched, to the byte.
    assert_eq!(painted.at(4, 4), bare.at(4, 4), "outside the quads");
    assert_eq!(painted.at(60, 60), bare.at(60, 60), "outside the quads");

    // A fully transparent quad is a no-op, whatever colour it claims: `rgba`
    // sends its premultiplied source to zero and the destination factor to one.
    let ghost = frame(
        &mut renderer,
        &target,
        1.0,
        &[UiQuad::solid([0.0, 0.0, 64.0, 64.0], [1.0, 1.0, 1.0, 0.0])],
        None,
    );
    assert_eq!(
        ghost.data, bare.data,
        "a zero-alpha quad must composite to nothing"
    );
}

// ---------------------------------------------------------------------------
// 3. Painter's order
// ---------------------------------------------------------------------------

#[test]
fn painters_order_is_vec_order() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let device = renderer.device().clone();
    let target = Target::new(&device, 64, 64);

    let first = [1.0, 0.0, 0.0, 1.0];
    let second = [0.0, 0.0, 1.0, 1.0];
    let rect_a = [8.0, 8.0, 24.0, 24.0];
    let rect_b = [20.0, 20.0, 24.0, 24.0];

    let ab = frame(
        &mut renderer,
        &target,
        1.0,
        &[UiQuad::solid(rect_a, first), UiQuad::solid(rect_b, second)],
        None,
    );
    assert_rgba(ab.at(24, 24), second, "the later quad owns the overlap");
    assert_rgba(ab.at(10, 10), first, "and the earlier one keeps the rest");

    // Reversing the list reverses the overlap and nothing else. No depth, no
    // sort, no tiebreak: the only thing that decided this was the index.
    let ba = frame(
        &mut renderer,
        &target,
        1.0,
        &[UiQuad::solid(rect_b, second), UiQuad::solid(rect_a, first)],
        None,
    );
    assert_rgba(ba.at(24, 24), first, "reversing the list reverses the stack");
    assert_rgba(ba.at(40, 40), second, "and leaves the rest alone");
}

// ---------------------------------------------------------------------------
// 4. The atlas
// ---------------------------------------------------------------------------

#[test]
fn an_atlas_quad_samples_the_texel_its_uv_names() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let device = renderer.device().clone();
    let queue = renderer.queue().clone();
    let target = Target::new(&device, 64, 64);

    // A 4×4 bake through the ordinary texture path — the same road P7's glyph
    // atlas takes, at the smallest size that still has distinguishable texels.
    let spec = TextureSpec {
        frequency: 4.0,
        octaves: 2,
        world_scale: 1.0,
        anti_tiling: false,
        ..TextureSpec::default()
    };
    let atlas = renderer.bake_texture(&spec, 4, &NoopCache);
    let baked = runt_core::bake::read_target(
        &device,
        &queue,
        &renderer.textures().get(atlas).expect("just baked").albedo,
        4,
    )
    .expect("read the atlas back");

    let texel = |x: usize, y: usize| {
        let i = (y * 4 + x) * 4;
        [
            baked[i] as f32 / 255.0,
            baked[i + 1] as f32 / 255.0,
            baked[i + 2] as f32 / 255.0,
            baked[i + 3] as f32 / 255.0,
        ]
    };

    let bare = frame(&mut renderer, &target, 1.0, &[], None);
    // uv covering exactly texel (1,1) — a nearest sampler magnifying it 8× per
    // axis paints one flat colour over the whole 32×32 quad.
    let quad = UiQuad::textured(
        [16.0, 16.0, 32.0, 32.0],
        [0.25, 0.25, 0.5, 0.5],
        [1.0, 1.0, 1.0, 1.0],
    );
    let painted = frame(&mut renderer, &target, 1.0, &[quad], Some(atlas));

    let want = over(texel(1, 1), bare.at(24, 24));
    assert_rgba(painted.at(24, 24), want, "the atlas texel");
    assert_rgba(painted.at(40, 40), over(texel(1, 1), bare.at(40, 40)), "flat across the quad");

    // A different texel is a different colour, or the uv is being ignored.
    let other = UiQuad::textured(
        [16.0, 16.0, 32.0, 32.0],
        [0.75, 0.75, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    );
    let elsewhere = frame(&mut renderer, &target, 1.0, &[other], Some(atlas));
    assert_rgba(
        elsewhere.at(24, 24),
        over(texel(3, 3), bare.at(24, 24)),
        "uv (0.75..1) is texel (3,3)",
    );

    // A solid quad in the *same* batch ignores the bound atlas entirely — the
    // sentinel is what lets a panel and its glyphs share one draw.
    let mixed = frame(
        &mut renderer,
        &target,
        1.0,
        &[
            UiQuad::solid([0.0, 0.0, 8.0, 8.0], [0.0, 1.0, 0.0, 1.0]),
            quad,
        ],
        Some(atlas),
    );
    assert_rgba(mixed.at(4, 4), [0.0, 1.0, 0.0, 1.0], "a solid quad beside an atlas one");
    assert_rgba(mixed.at(24, 24), want, "and the atlas quad is unaffected");

    // An atlas handle nothing has baked degrades to the white texel rather than
    // taking the frame down: the batch is data a game rebuilt this frame.
    let missing = frame(
        &mut renderer,
        &target,
        1.0,
        &[UiQuad::textured(
            [16.0, 16.0, 32.0, 32.0],
            [0.25, 0.25, 0.5, 0.5],
            [1.0, 0.5, 0.25, 1.0],
        )],
        Some(TextureHandle(0xdead_beef)),
    );
    assert_rgba(
        missing.at(24, 24),
        [1.0, 0.5, 0.25, 1.0],
        "an unbaked atlas falls back to white, so the tint survives",
    );
}

// ---------------------------------------------------------------------------
// 5. Render scale
// ---------------------------------------------------------------------------

#[test]
fn a_raw_image_atlas_lands_texel_for_texel() {
    // `UiAtlasImage`'s path (P7's glyph bake): pixels the *game* generated,
    // uploaded under a handle the game chose, sampled by uv exactly as a baked
    // `TextureSpec` would be. The one door into the texture registry that does
    // not go through a procedural spec, and the only one a bitmap font can use.
    let Some(mut renderer) = renderer() else {
        return;
    };
    let device = renderer.device().clone();
    let target = Target::new(&device, 64, 64);

    // 2×2, one distinct premultiplied texel per quadrant — and deliberately
    // **not** square-with-mips: the raw path is 1 mip and any size, which is
    // what a 128×56 glyph grid needs.
    let handle = TextureHandle(0xf0_0d);
    let pixels: Vec<u8> = vec![
        255, 0, 0, 255, // (0,0) red
        0, 255, 0, 255, // (1,0) green
        0, 0, 255, 255, // (0,1) blue
        128, 128, 128, 128, // (1,1) half-alpha grey, premultiplied
    ];
    renderer.upload_ui_atlas(handle, 2, 2, &pixels);
    assert!(renderer.textures().contains(handle), "not resident");
    // Idempotent: a host may pump the resource every frame.
    renderer.upload_ui_atlas(handle, 2, 2, &pixels);
    assert_eq!(renderer.textures().len(), 1);

    let bare = frame(&mut renderer, &target, 1.0, &[], None);
    let quad = |uv: [f32; 4]| UiQuad::textured([16.0, 16.0, 32.0, 32.0], uv, [1.0; 4]);

    let red = frame(&mut renderer, &target, 1.0, &[quad([0.0, 0.0, 0.5, 0.5])], Some(handle));
    assert_rgba(red.at(24, 24), over([1.0, 0.0, 0.0, 1.0], bare.at(24, 24)), "texel (0,0)");
    let green = frame(&mut renderer, &target, 1.0, &[quad([0.5, 0.0, 1.0, 0.5])], Some(handle));
    assert_rgba(green.at(24, 24), over([0.0, 1.0, 0.0, 1.0], bare.at(24, 24)), "texel (1,0)");
    let blue = frame(&mut renderer, &target, 1.0, &[quad([0.0, 0.5, 0.5, 1.0])], Some(handle));
    assert_rgba(blue.at(24, 24), over([0.0, 0.0, 1.0, 1.0], bare.at(24, 24)), "texel (0,1)");

    // The half-alpha texel composites as premultiplied rather than being
    // multiplied by its own alpha a second time.
    let grey = frame(&mut renderer, &target, 1.0, &[quad([0.5, 0.5, 1.0, 1.0])], Some(handle));
    let want = over(
        [128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0],
        bare.at(24, 24),
    );
    assert_rgba(grey.at(24, 24), want, "the premultiplied texel");
}

#[test]
fn a_malformed_raw_image_is_refused_rather_than_handed_to_wgpu() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    // Too few bytes for the size it claims: a warning and nothing resident,
    // rather than a validation error inside `write_texture`.
    renderer.upload_ui_atlas(TextureHandle(1), 4, 4, &[0u8; 8]);
    assert!(!renderer.textures().contains(TextureHandle(1)));
    renderer.upload_ui_atlas(TextureHandle(2), 0, 4, &[]);
    assert!(!renderer.textures().contains(TextureHandle(2)));
    // …and a batch naming a handle that never arrived draws untextured rather
    // than panicking, which the pass already promised.
    let device = renderer.device().clone();
    let target = Target::new(&device, 32, 32);
    let quad = UiQuad::textured([0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 1.0, 1.0], [1.0; 4]);
    frame(&mut renderer, &target, 1.0, &[quad], Some(TextureHandle(1)));
}

#[test]
fn ui_stays_surface_crisp_at_half_render_scale() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let device = renderer.device().clone();
    // 64×48 at 0.5 is a 32×24 internal target: every world pixel is a 2×2 block
    // on the surface, so anything drawn *before* the blit can only have edges on
    // even columns.
    let target = Target::new(&device, 64, 48);

    // Deliberately odd bounds: x 17..31, y 9..39. An edge here is unreachable
    // for a quad that went through the 2× upscale.
    let rect = [17.0, 9.0, 14.0, 30.0];
    let color = [1.0, 0.0, 1.0, 1.0];

    let bare = frame(&mut renderer, &target, 0.5, &[], None);
    let painted = frame(&mut renderer, &target, 0.5, &[UiQuad::solid(rect, color)], None);
    assert_eq!(
        renderer.scaled_target_size(),
        Some((32, 24)),
        "the scene really was drawn at half resolution"
    );

    // The world behind it is chonky — 2×2 blocks — which is what makes the
    // next assertion mean something.
    let mut blocky = 0;
    for y in (0..48).step_by(2) {
        for x in (0..64).step_by(2) {
            if bare.at(x, y) == bare.at(x + 1, y) && bare.at(x, y) == bare.at(x, y + 1) {
                blocky += 1;
            }
        }
    }
    assert_eq!(blocky, 32 * 24, "every 2×2 block of the scaled frame is flat");

    // The quad's edges, column by column, at *surface* resolution.
    let row = 20;
    assert_eq!(painted.at(16, row), bare.at(16, row), "column 16 is untouched");
    assert_rgba(painted.at(17, row), color, "column 17 is the quad's first");
    assert_rgba(painted.at(30, row), color, "column 30 is the quad's last");
    assert_eq!(painted.at(31, row), bare.at(31, row), "column 31 is untouched");

    // …and its rows.
    let col = 20;
    assert_eq!(painted.at(col, 8), bare.at(col, 8), "row 8 is untouched");
    assert_rgba(painted.at(col, 9), color, "row 9 is the quad's first");
    assert_rgba(painted.at(col, 38), color, "row 38 is the quad's last");
    assert_eq!(painted.at(col, 39), bare.at(col, 39), "row 39 is untouched");

    // The odd edges are the claim: a UI drawn into the half-scale target and
    // blitted up could not put one at column 17 or row 9 at all.
    assert_ne!(
        painted.at(17, row),
        painted.at(16, row),
        "the left edge is a surface-pixel boundary, not a 2×2 block boundary"
    );
}

// ---------------------------------------------------------------------------
// Batch size — a HUD is one draw whatever is in it
// ---------------------------------------------------------------------------

#[test]
fn two_hundred_quads_are_one_batch() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let device = renderer.device().clone();
    let target = Target::new(&device, 256, 256);

    // A meter ring's worth of slices plus a HUD around it — the shape P7 will
    // actually build, since there is no arc mode and an arc is a fan of quads.
    let mut quads = Vec::new();
    for i in 0..200u32 {
        let a = i as f32 * std::f32::consts::TAU / 200.0;
        let (x, y) = (128.0 + 100.0 * a.cos(), 128.0 + 100.0 * a.sin());
        quads.push(UiQuad::solid(
            [x - 2.0, y - 2.0, 4.0, 4.0],
            [1.0, 1.0, 1.0, 1.0],
        ));
    }

    let painted = frame(&mut renderer, &target, 1.0, &quads, None);

    // What the batch actually costs, warm: the same frame with and without it,
    // submitted and waited on. One `write_buffer` and one `draw` either way —
    // there is no per-quad state to set, so the difference is fill rate and a
    // 9.6 kB upload, not draw calls.
    let device = renderer.device().clone();
    let mut bench = |quads: &[UiQuad]| {
        renderer.set_ui_quads(quads, None);
        let start = std::time::Instant::now();
        for _ in 0..50 {
            renderer.render_scaled(
                &target.view,
                target.width,
                target.height,
                RenderScale::default(),
                &FrameParams::default(),
                &[],
                &MeshLibrary::new(),
                &TextureLibrary::new(),
            );
            device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        }
        start.elapsed() / 50
    };
    let with_hud = bench(&quads);
    let with_one = bench(&quads[..1]);
    let without = bench(&[]);
    println!(
        "256×256 frame: {without:?} bare, {with_one:?} with 1 quad, {with_hud:?} \
         with 200 (pass overhead {:?}, the other 199 quads {:?})",
        with_one.saturating_sub(without),
        with_hud.saturating_sub(with_one),
    );

    assert_eq!(renderer.ui_quad_count(), 0, "the bench left the batch empty");
    // The buffer grew once, by doubling, and is now sticky.
    assert!(
        renderer.ui_ready(),
        "the pass exists after a batch that used it"
    );
    // Spot-check that the ring actually landed: the first slice's centre.
    assert_rgba(painted.at(228, 128), [1.0, 1.0, 1.0, 1.0], "slice 0");
}

// ---------------------------------------------------------------------------
// WebGL2: the UI shader has to translate, not just compile
// ---------------------------------------------------------------------------

#[test]
fn the_ui_shader_translates_to_glsl_es_for_webgl2() {
    let module = naga::front::wgsl::parse_str(runt_core::UI_SHADER).expect("ui WGSL parses");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        // Nothing beyond the baseline: a HUD must not be the thing that needs a
        // capability WebGL2 lacks (DESIGN §11).
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .expect("ui WGSL validates");

    let options = naga::back::glsl::Options {
        version: naga::back::glsl::Version::Embedded {
            version: 300,
            is_webgl: true,
        },
        ..Default::default()
    };
    for (stage, entry) in [
        (naga::ShaderStage::Vertex, "vs_ui"),
        (naga::ShaderStage::Fragment, "fs_ui"),
    ] {
        let pipeline_options = naga::back::glsl::PipelineOptions {
            shader_stage: stage,
            entry_point: entry.to_string(),
            multiview: None,
        };
        let mut out = String::new();
        let mut writer = naga::back::glsl::Writer::new(
            &mut out,
            &module,
            &info,
            &options,
            &pipeline_options,
            naga::proc::BoundsCheckPolicies::default(),
        )
        .unwrap_or_else(|e| panic!("{entry}: GLSL-ES writer: {e}"));
        writer
            .write()
            .unwrap_or_else(|e| panic!("{entry}: GLSL-ES emit: {e}"));
        assert!(
            out.contains("#version 300 es"),
            "{entry} did not come out as ES 3.00:\n{out}"
        );
    }
}
