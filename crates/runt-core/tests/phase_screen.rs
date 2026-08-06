//! The phase circle's screen effect (DESIGN §5, `blit.wgsl`): inside the
//! circle the frame's luminance is inverted and pulled 40% towards grey, and
//! outside it nothing happens at all.
//!
//! This is a port of `3dimenshift/shaders/phase_screen_effect.gdshader`, so the
//! claims are about *that* formula rather than about a plausible one. Five of
//! them, in ascending order of how much of the engine they touch:
//!
//! 1. **The resting state is the old renderer.** At radius zero a native frame
//!    is byte-for-byte the frame that was drawn before this feature existed,
//!    and no internal target is allocated — including after a circle-on frame
//!    has allocated one, which is the case a sticky allocation could quietly
//!    break.
//! 2. **The arithmetic.** Every pixel deep inside the circle is the Godot
//!    formula applied to the pixel that was there, and every pixel outside it
//!    is the identical byte. Held against the *plain* frame rather than against
//!    a hash, so it says something on a machine whose rasterizer differs.
//! 3. **It is round on screen.** Measured on a deliberately non-square target:
//!    a missing aspect correction is an ellipse, and on a square one an ellipse
//!    and a circle are the same picture.
//! 4. **Render scale is orthogonal.** One fullscreen pass does the copy and the
//!    effect, so a half-resolution frame gets the same treatment on the same
//!    boundary.
//! 5. **The HUD is out of it.** The UI pass is encoded after the effect and
//!    straight onto the host's view, so a screen-space HUD sitting inside the
//!    circle is untouched.
//!
//! The circle's *placement* is checked here against the same arithmetic
//! `shader.wgsl` uses (aspect-corrected NDC, radius in NDC-Y units), which is
//! what makes the effect's boundary and the material shaders' discard boundary
//! the same boundary. `tests/transparency.rs` sees both in one frame.

use glam::{Vec2, Vec3};
use runt_core::draw::FrameParams;
use runt_core::ecs::{phase_screen_color, PHASE_EDGE, PHASE_MIN_RADIUS};
use runt_core::registry::MeshLibrary;
use runt_core::texture::TextureLibrary;
use runt_core::ui::UiQuad;
use runt_core::{RenderScale, Renderer};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// Deliberately not square, and 320×192 so both axes divide by 2 exactly — the
/// aspect claim needs the first and the render-scale claim needs the second.
const WIDTH: u32 = 320;
const HEIGHT: u32 = 192;
const ASPECT: f32 = WIDTH as f32 / HEIGHT as f32;

/// The frame these tests are drawn on is the sky and nothing else.
///
/// Not an oversight: the effect reads the finished framebuffer and has no idea
/// what wrote it, so geometry would add nothing but a second thing to keep
/// still. What it *does* need is a picture that varies — a flat colour would
/// let a wrongly placed circle pass — and a three-stop gradient across a
/// perspective view varies on both axes.
fn frame_params() -> FrameParams {
    let view_proj = runt_core::Camera::default().projection(ASPECT)
        * glam::camera::rh::view::look_at_mat4(
            Vec3::new(0.0, 1.0, 5.0),
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::Y,
        );
    FrameParams {
        view_proj,
        ..Default::default()
    }
}

struct Rig {
    renderer: Renderer,
    target: wgpu::Texture,
    view: wgpu::TextureView,
}

impl Rig {
    fn new() -> Option<Rig> {
        let renderer = match pollster::block_on(Renderer::headless(FORMAT)) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP phase_screen (no GPU adapter): {e}");
                return None;
            }
        };
        let target = renderer.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("phase screen target"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        Some(Rig {
            renderer,
            target,
            view,
        })
    }

    /// One frame with the circle at `center`/`radius`, at `scale`, with `hud`
    /// on top, read back as tightly packed RGBA8.
    fn frame(&mut self, center: Vec2, radius: f32, scale: f32, hud: &[UiQuad]) -> Vec<u8> {
        // Strength is deliberately hot: it drives the material shaders' edge
        // fringe and *nothing here*, and a frame that changed with it would
        // mean the screen effect had picked up `phase.w` by accident.
        self.renderer.set_phase_fx(center, radius, 1.0);
        self.renderer.set_ui_quads(hud, None);
        self.renderer.render_scaled(
            &self.view,
            WIDTH,
            HEIGHT,
            RenderScale::new(scale),
            &frame_params(),
            &[],
            &MeshLibrary::new(),
            &TextureLibrary::new(),
        );
        self.read_back()
    }

    fn read_back(&self) -> Vec<u8> {
        let device = self.renderer.device();
        let unpadded_row = WIDTH * 4;
        let padded_row = unpadded_row.div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded_row * HEIGHT) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            self.target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(HEIGHT),
                },
            },
            wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
        );
        self.renderer.queue().submit(Some(encoder.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll device");
        rx.recv().expect("map callback").expect("buffer mapped");

        let padded = readback.get_mapped_range(..).expect("mapped range");
        let mut pixels = Vec::with_capacity((unpadded_row * HEIGHT) as usize);
        for row in 0..HEIGHT as usize {
            let start = row * padded_row as usize;
            pixels.extend_from_slice(&padded[start..start + unpadded_row as usize]);
        }
        drop(padded);
        readback.unmap();
        pixels
    }
}

/// The pixel *centre* in NDC, +Y up — the value the fullscreen triangle's
/// interpolated uv lands on, and the same place `shader.wgsl` computes from
/// `@builtin(position)`.
fn pixel_ndc(x: u32, y: u32) -> Vec2 {
    Vec2::new(
        (x as f32 + 0.5) / WIDTH as f32 * 2.0 - 1.0,
        1.0 - (y as f32 + 0.5) / HEIGHT as f32 * 2.0,
    )
}

/// Aspect-corrected distance from the circle's centre, in NDC-Y units.
fn phase_distance(x: u32, y: u32, center: Vec2) -> f32 {
    let mut d = pixel_ndc(x, y) - center;
    d.x *= ASPECT;
    d.length()
}

fn rgb(pixels: &[u8], x: u32, y: u32) -> Vec3 {
    let i = ((y * WIDTH + x) * 4) as usize;
    Vec3::new(pixels[i] as f32, pixels[i + 1] as f32, pixels[i + 2] as f32) / 255.0
}

fn texel(pixels: &[u8], x: u32, y: u32) -> &[u8] {
    let i = ((y * WIDTH + x) * 4) as usize;
    &pixels[i..i + 4]
}

/// The whole per-pixel claim, swept over the frame: outside is the identical
/// byte, inside is the formula, and the band between them is left alone
/// because a smoothstep's exact shape is the shader's business.
///
/// `slack` widens that band by a pixel either way, so a half-texel disagreement
/// about where a pixel *is* cannot fail a claim about what it *holds*.
fn assert_the_effect_matches_the_original(
    plain: &[u8],
    effected: &[u8],
    center: Vec2,
    radius: f32,
) {
    let slack = 2.0 / HEIGHT as f32;
    let (mut inside, mut outside) = (0u32, 0u32);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let dist = phase_distance(x, y, center);
            if dist > radius + PHASE_EDGE + slack {
                assert_eq!(
                    texel(plain, x, y),
                    texel(effected, x, y),
                    "({x},{y}) is {dist:.4} from the centre, outside a circle of \
                     {radius}, and the effect reached it anyway"
                );
                outside += 1;
            } else if dist < radius - PHASE_EDGE - slack {
                let want = phase_screen_color(rgb(plain, x, y), 1.0).clamp(Vec3::ZERO, Vec3::ONE);
                let got = rgb(effected, x, y);
                let error = (got - want).abs().max_element();
                assert!(
                    error <= 3.0 / 255.0,
                    "({x},{y}) is {dist:.4} inside a circle of {radius}: \
                     expected {want:?}, got {got:?} (off by {error})"
                );
                inside += 1;
            }
        }
    }
    // The sweep has to have had something to say on both sides of the edge; a
    // circle badly enough placed to have emptied one of them would otherwise
    // pass vacuously.
    assert!(inside > 1000, "only {inside} pixels were inside the circle");
    assert!(outside > 1000, "only {outside} pixels were outside it");
}

// ---------------------------------------------------------------------------
// 1. The resting state
// ---------------------------------------------------------------------------

#[test]
fn a_circle_at_rest_draws_the_frame_that_was_always_drawn() {
    let Some(mut rig) = Rig::new() else {
        return;
    };

    let plain = rig.frame(Vec2::ZERO, 0.0, 1.0, &[]);
    assert_eq!(
        rig.renderer.scaled_target_size(),
        None,
        "a native frame with no circle must not allocate an internal target"
    );

    // Exactly at the threshold is still off — the shaders test `> 0.001` and so
    // does the renderer, and a boundary that disagreed with itself would draw
    // the effect into a frame the material shaders consider circle-free.
    let at_threshold = rig.frame(Vec2::ZERO, PHASE_MIN_RADIUS, 1.0, &[]);
    assert_eq!(
        plain, at_threshold,
        "radius {PHASE_MIN_RADIUS} is still off"
    );
    assert_eq!(rig.renderer.scaled_target_size(), None);

    // On: the detour is taken even though the scale is native, because there is
    // no other way to read the finished frame.
    let on = rig.frame(Vec2::ZERO, 0.5, 1.0, &[]);
    assert_ne!(plain, on, "a circle of radius 0.5 changed nothing");
    assert_eq!(
        rig.renderer.scaled_target_size(),
        Some((WIDTH, HEIGHT)),
        "the circle's frame draws offscreen at the host's own size"
    );

    // And back off again. The internal target is sticky, so this is the case
    // where a stale `Some(..)` could silently keep routing native frames
    // through a pass they no longer need.
    let back = rig.frame(Vec2::ZERO, 0.0, 1.0, &[]);
    assert_eq!(
        plain, back,
        "turning the circle off did not restore the exact original frame"
    );
}

// ---------------------------------------------------------------------------
// 2. The arithmetic
// ---------------------------------------------------------------------------

#[test]
fn inside_the_circle_is_the_originals_inversion() {
    let Some(mut rig) = Rig::new() else {
        return;
    };
    let center = Vec2::new(0.1, -0.2);
    let radius = 0.45;

    let plain = rig.frame(Vec2::ZERO, 0.0, 1.0, &[]);
    let effected = rig.frame(center, radius, 1.0, &[]);
    assert_the_effect_matches_the_original(&plain, &effected, center, radius);

    // Spelled out once at a single pixel too, so a failure of the sweep above
    // has a value to read rather than a coordinate.
    let (x, y) = (
        ((center.x * 0.5 + 0.5) * WIDTH as f32) as u32,
        ((0.5 - center.y * 0.5) * HEIGHT as f32) as u32,
    );
    let before = rgb(&plain, x, y);
    let after = rgb(&effected, x, y);
    println!("centre pixel ({x},{y}): {before:?} → {after:?}");
    // The look, not just the formula: the middle of the circle is on the far
    // side of mid-grey from where it started.
    let luma = |c: Vec3| c.dot(Vec3::new(0.2126, 0.7152, 0.0722));
    assert!(
        (luma(before) - 0.5).signum() != (luma(after) - 0.5).signum(),
        "the inversion did not cross mid-grey: {before:?} → {after:?}"
    );
}

// ---------------------------------------------------------------------------
// 3. Round on screen, not round in NDC
// ---------------------------------------------------------------------------

#[test]
fn the_circle_is_round_on_screen() {
    let Some(mut rig) = Rig::new() else {
        return;
    };
    let radius = 0.4;
    let plain = rig.frame(Vec2::ZERO, 0.0, 1.0, &[]);
    let effected = rig.frame(Vec2::ZERO, radius, 1.0, &[]);

    // How far the effect reaches along the centre row and the centre column, in
    // NDC units of each axis. The mask fades to nothing over `PHASE_EDGE`, so
    // the last *visibly* changed pixel sits a hair inside the analytic edge —
    // which is fine, because both measurements are made the same way and the
    // claim is about their ratio.
    let reach = |along_x: bool| {
        let mut worst = 0.0f32;
        for i in 0..if along_x { WIDTH } else { HEIGHT } {
            let (x, y) = if along_x {
                (i, HEIGHT / 2)
            } else {
                (WIDTH / 2, i)
            };
            if texel(&plain, x, y) != texel(&effected, x, y) {
                let ndc = pixel_ndc(x, y);
                worst = worst.max(if along_x { ndc.x.abs() } else { ndc.y.abs() });
            }
        }
        worst
    };
    let (across, down) = (reach(true), reach(false));
    println!("effect reach: {across:.4} in NDC-X, {down:.4} in NDC-Y (aspect {ASPECT})");

    // Vertically the radius is in its own units, so the reach is the radius
    // plus (almost all of) the edge band.
    assert!(
        (down - (radius + PHASE_EDGE)).abs() < 3.0 / HEIGHT as f32,
        "the circle is {down:.4} tall, expected about {:.4}",
        radius + PHASE_EDGE
    );
    // Horizontally it is that same distance divided by the aspect ratio, which
    // is the entire aspect correction. Without it this ratio would be 1.
    assert!(
        (across * ASPECT / down - 1.0).abs() < 0.05,
        "the circle is an ellipse: {across:.4} across, {down:.4} down, \
         which is {:.3} of a circle",
        across * ASPECT / down
    );
}

// ---------------------------------------------------------------------------
// 4. One pass does the copy and the effect
// ---------------------------------------------------------------------------

#[test]
fn the_effect_survives_render_scale() {
    let Some(mut rig) = Rig::new() else {
        return;
    };
    let center = Vec2::new(-0.15, 0.1);
    let radius = 0.45;

    // Both frames at 0.5, so the picture underneath is the same chonky upscale
    // and the only difference is the effect. The mask is still evaluated per
    // *host* pixel — the pass runs at the view's resolution whatever the
    // internal target's size is — which is why the same sweep applies.
    let half = Some((WIDTH / 2, HEIGHT / 2));
    let plain = rig.frame(Vec2::ZERO, 0.0, 0.5, &[]);
    assert_eq!(rig.renderer.scaled_target_size(), half);
    let effected = rig.frame(center, radius, 0.5, &[]);
    assert_eq!(rig.renderer.scaled_target_size(), half);

    assert_the_effect_matches_the_original(&plain, &effected, center, radius);
}

// ---------------------------------------------------------------------------
// 5. The HUD is above it
// ---------------------------------------------------------------------------

#[test]
fn the_hud_is_not_inverted() {
    let Some(mut rig) = Rig::new() else {
        return;
    };
    // A panel across the middle of the screen, well inside a circle that covers
    // everything.
    let rect = [100.0, 60.0, 80.0, 40.0];
    let hud = [UiQuad::solid(rect, [0.9, 0.3, 0.1, 1.0])];

    let plain = rig.frame(Vec2::ZERO, 0.0, 1.0, &hud);
    let effected = rig.frame(Vec2::ZERO, 2.0, 1.0, &hud);

    // Inset by a pixel: the quad's own edges are the UI pass's business and a
    // half-covered edge pixel is not what this is asking about.
    let (x0, y0) = (rect[0] as u32 + 1, rect[1] as u32 + 1);
    let (x1, y1) = (
        (rect[0] + rect[2]) as u32 - 1,
        (rect[1] + rect[3]) as u32 - 1,
    );
    for y in y0..y1 {
        for x in x0..x1 {
            assert_eq!(
                texel(&plain, x, y),
                texel(&effected, x, y),
                "the HUD pixel ({x},{y}) was inverted; the UI pass must be \
                 encoded after the effect, not into it"
            );
        }
    }

    // …and the claim is not vacuous: the world behind the panel *was* inverted.
    assert_ne!(
        texel(&plain, 10, 10),
        texel(&effected, 10, 10),
        "the circle covered the whole screen and changed nothing"
    );
}
