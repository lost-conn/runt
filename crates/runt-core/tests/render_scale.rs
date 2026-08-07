//! Render scale — "pixel chonkiness" (DESIGN §11's resolution lever).
//!
//! The engine draws the scene into an internal target of `scale × view size`
//! and blits it up with a nearest filter. Four claims are worth pinning, and
//! they are in ascending order of how expensive they are to check:
//!
//! 1. **The arithmetic.** Clamping, round-half-up, a floor of one pixel per
//!    axis, and the step ladder a host's `[` / `]` keys walk. No GPU.
//! 2. **Native is untouched.** At 1.0 there is no internal target, no blit and
//!    no pixel difference — including *after* a scaled frame has allocated one,
//!    which is the case a sticky allocation could quietly break. The screenshot
//!    suite pins the same property from the other side (its hashes must not
//!    move); this pins it directly, frame against frame.
//! 3. **The target follows the inputs.** Scale changes and viewport changes are
//!    the same event to the allocator, and both must be picked up.
//! 4. **It is actually blocky.** A nearest upscale of a half-resolution frame
//!    puts identical pixels in 2×2 blocks. That is the whole visible feature,
//!    and it is measurable: at 0.5 *every* even/odd neighbour pair matches, on
//!    a view chosen for having gradients everywhere.
//!
//! Plus the WebGL2 check that costs nothing to run and would otherwise only
//! fail in a browser: the blit shader has to survive naga's WGSL → GLSL-ES 3.00
//! translation, not merely compile on the machine running the test.

use runt_core::{Engine, RenderScale};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// Wide enough for the sky gradient and the terrain to fill it, and divisible
/// by 4 so 0.25 lands exactly.
const WIDTH: u32 = 256;
const HEIGHT: u32 = 192;
/// Ticks before capture, so the demo scene has moved into frame.
const TICKS: u64 = 42;

// ---------------------------------------------------------------------------
// 1. The arithmetic — no GPU
// ---------------------------------------------------------------------------

#[test]
fn a_scale_is_clamped_into_its_range() {
    assert_eq!(RenderScale::default().get(), 1.0);
    assert_eq!(RenderScale::new(0.5).get(), 0.5);

    // Above the ceiling and below the floor both saturate: supersampling is a
    // different feature, and a zero-sized attachment is not a feature at all.
    assert_eq!(RenderScale::new(4.0).get(), RenderScale::MAX);
    assert_eq!(RenderScale::new(0.0).get(), RenderScale::MIN);
    assert_eq!(RenderScale::new(-1.0).get(), RenderScale::MIN);
    assert_eq!(RenderScale::new(f32::INFINITY).get(), RenderScale::MAX);
    assert_eq!(RenderScale::new(f32::NEG_INFINITY).get(), RenderScale::MIN);
    // A NaN must resolve to the safe end rather than propagate into a texture
    // size, where it would become 0 and take the device down with it.
    assert_eq!(RenderScale::new(f32::NAN).get(), RenderScale::MAX);

    assert!(RenderScale::new(1.0).is_native());
    assert!(!RenderScale::new(0.99).is_native());
}

#[test]
fn the_scaled_size_rounds_half_up_and_never_reaches_zero() {
    let at = |scale: f32, w: u32, h: u32| RenderScale::new(scale).size(w, h);

    // Native is the identity — the property the whole "1.0 is untouched" claim
    // rests on, since it is what makes the renderer take the old path.
    assert_eq!(at(1.0, 1920, 1080), (1920, 1080));
    assert_eq!(at(1.0, 1, 1), (1, 1));

    assert_eq!(at(0.5, 1920, 1080), (960, 540));
    assert_eq!(at(0.25, 1920, 1080), (480, 270));
    assert_eq!(at(0.75, 1920, 1080), (1440, 810));
    // 1/3 of 1920 is 640 exactly; of 1080 it is 360 exactly.
    assert_eq!(at(1.0 / 3.0, 1920, 1080), (640, 360));

    // Half up, not half even and not truncated: 641 × 0.5 = 320.5 → 321.
    assert_eq!(at(0.5, 641, 361), (321, 181));
    assert_eq!(at(0.5, 3, 3), (2, 2));
    // …and never below one pixel, however small the view or the scale.
    assert_eq!(at(0.1, 4, 4), (1, 1));
    assert_eq!(at(0.25, 1, 1), (1, 1));
    assert_eq!(at(0.5, 0, 0), (1, 1));
}

#[test]
fn stepping_walks_the_preset_ladder_and_saturates_at_both_ends() {
    let steps = RenderScale::STEPS;
    assert_eq!(steps.len(), 5);
    assert_eq!(steps[0], 0.25);
    assert_eq!(*steps.last().unwrap(), 1.0);
    assert!(
        steps.windows(2).all(|w| w[0] < w[1]),
        "the ladder is ascending, so -1 always means chunkier"
    );

    let native = RenderScale::default();
    assert_eq!(native.stepped(-1).get(), 0.75);
    assert_eq!(native.stepped(-1).stepped(-1).get(), 0.5);
    // Bottoming out rather than wrapping: mashing `[` on a phone must not jump
    // back to native resolution.
    let bottom = (0..10).fold(native, |s, _| s.stepped(-1));
    assert_eq!(bottom.get(), 0.25);
    let top = (0..10).fold(bottom, |s, _| s.stepped(1));
    assert_eq!(top.get(), 1.0);

    // An off-ladder value (a URL query, a config file) steps to the next rung
    // in the direction of travel, never past it.
    assert_eq!(RenderScale::new(0.42).stepped(1).get(), 0.5);
    assert_eq!(RenderScale::new(0.42).stepped(-1).get(), 1.0 / 3.0);
    assert_eq!(RenderScale::new(0.0).stepped(-1).get(), 0.25);
    assert_eq!(RenderScale::new(0.99).stepped(1).get(), 1.0);
    // 1/3 is not exactly representable; stepping off it must not stick.
    assert_eq!(RenderScale::new(1.0 / 3.0).stepped(1).get(), 0.5);
    assert_eq!(RenderScale::new(1.0 / 3.0).stepped(-1).get(), 0.25);
    assert_eq!(native.stepped(0).get(), 1.0);
    assert_eq!(native.stepped(-2).get(), 0.5, "a two-place step is two rungs");
}

#[test]
fn the_scale_is_not_simulation_state() {
    // The sim must not be able to see it. Two runs of the same ticks, one of
    // them changing the scale every tick, have to agree on every transform —
    // otherwise a render knob has become a gameplay input and every recorded
    // trace is suspect. (The port pins the same property through its key
    // binding; this pins the resource itself.)
    let fingerprint = |vary: bool| {
        let mut sim = runt_core::Sim::new();
        for tick in 0..90u32 {
            if vary {
                sim.set_render_scale(RenderScale::STEPS[tick as usize % 5]);
            }
            sim.tick();
        }
        let mut rows: Vec<String> = sim
            .world_mut()
            .query::<(bevy_ecs::entity::Entity, &runt_core::Transform)>()
            .iter(sim.world())
            .map(|(e, t)| format!("{e:?} {:?} {:?} {:?}", t.translation, t.rotation, t.scale))
            .collect();
        rows.sort();
        rows
    };
    assert_eq!(fingerprint(false), fingerprint(true));
}

#[test]
fn the_engine_reports_the_pixels_it_will_draw() {
    let mut sim = runt_core::Sim::without_scene();
    assert_eq!(sim.render_scale().get(), 1.0, "native by default");
    sim.set_render_scale(0.5);
    assert_eq!(sim.render_scale().get(), 0.5);
    sim.set_render_scale(9.0);
    assert_eq!(sim.render_scale().get(), 1.0, "clamped on the way in");
}

// ---------------------------------------------------------------------------
// GPU harness
// ---------------------------------------------------------------------------

/// `copy_texture_to_buffer` wants a 256-byte row stride.
fn align_256(n: u32) -> u32 {
    n.div_ceil(256) * 256
}

/// An engine on the demo scene, ticked to a fixed pose, plus a target it can
/// draw into and read back from. `None` when the machine has no adapter.
struct Harness {
    engine: Engine,
    target: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

impl Harness {
    fn new(width: u32, height: u32) -> Option<Harness> {
        let mut engine = match pollster::block_on(Engine::headless(FORMAT)) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("SKIP render_scale (no GPU adapter): {e}");
                return None;
            }
        };
        // Stepped like a host would rather than jumped, so the spiral-of-death
        // clamp never eats the interval — the same drive `headless_screenshot`
        // uses, and the same fixed pose.
        engine.update(0.0);
        let mut t = 0.0;
        while engine.tick_count() < TICKS {
            t += runt_core::TICK_DT * 0.25;
            engine.update(t);
        }

        let (target, view) = Harness::make_target(engine.device(), width, height);
        Some(Harness {
            engine,
            target,
            view,
            width,
            height,
        })
    }

    fn make_target(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("render scale test target"),
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
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        (target, view)
    }

    /// Point the harness at a differently sized target — a host window resize.
    fn resize(&mut self, width: u32, height: u32) {
        let (target, view) = Harness::make_target(self.engine.device(), width, height);
        self.target = target;
        self.view = view;
        self.width = width;
        self.height = height;
    }

    /// One frame at `scale`, read back as tightly packed RGBA8.
    fn frame(&mut self, scale: f32) -> Vec<u8> {
        self.engine.set_render_scale(scale);
        self.engine.render(&self.view, self.width, self.height);
        self.read_back()
    }

    fn read_back(&self) -> Vec<u8> {
        let device = self.engine.device().clone();
        let queue = self.engine.queue().clone();
        let unpadded_row = self.width * 4;
        let padded_row = align_256(unpadded_row);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded_row * self.height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback"),
        });
        encoder.copy_texture_to_buffer(
            self.target.as_image_copy(),
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

// ---------------------------------------------------------------------------
// 2. Native resolution is bit-for-bit what it was
// ---------------------------------------------------------------------------

#[test]
fn native_scale_draws_the_same_pixels_and_allocates_nothing() {
    let Some(mut h) = Harness::new(WIDTH, HEIGHT) else {
        return;
    };

    // The path every existing test and every existing screenshot hash takes:
    // `render` with nobody having mentioned a scale.
    h.engine.render(&h.view, h.width, h.height);
    let baseline = h.read_back();
    assert_eq!(
        h.engine.renderer().scaled_target_size(),
        None,
        "a native frame must not allocate an internal target"
    );

    let explicit = h.frame(1.0);
    assert_eq!(baseline, explicit, "setting the scale to 1.0 changed pixels");
    assert_eq!(
        h.engine.renderer().scaled_target_size(),
        None,
        "an explicit 1.0 must not allocate either"
    );

    // And after a scaled frame has allocated one: the internal target is sticky
    // by design, so this is the case where a stale `Some(..)` could silently
    // redirect a native frame through the blit.
    let scaled = h.frame(0.5);
    assert_ne!(baseline, scaled, "0.5 rendered the identical frame");
    assert_eq!(h.engine.renderer().scaled_target_size(), Some((128, 96)));

    let back_to_native = h.frame(1.0);
    assert_eq!(
        baseline, back_to_native,
        "going back to 1.0 did not restore the exact original frame"
    );
}

// ---------------------------------------------------------------------------
// 3. The internal target follows the scale and the viewport
// ---------------------------------------------------------------------------

#[test]
fn the_internal_target_tracks_the_scale_and_a_resize() {
    let Some(mut h) = Harness::new(WIDTH, HEIGHT) else {
        return;
    };

    h.frame(0.5);
    assert_eq!(h.engine.renderer().scaled_target_size(), Some((128, 96)));

    // A scale change reallocates…
    h.frame(0.25);
    assert_eq!(h.engine.renderer().scaled_target_size(), Some((64, 48)));

    // …and so does a resize at an unchanged scale, which is the same event as
    // far as the allocator is concerned.
    h.resize(640, 360);
    h.frame(0.25);
    assert_eq!(h.engine.renderer().scaled_target_size(), Some((160, 90)));

    // Half-up rounding survives the round trip to a real allocation.
    h.resize(641, 361);
    h.frame(0.5);
    assert_eq!(h.engine.renderer().scaled_target_size(), Some((321, 181)));

    // A clamp is applied on the way in, so an absurd request is a legal frame.
    h.frame(0.0);
    assert_eq!(
        h.engine.renderer().scaled_target_size(),
        Some((64, 36)),
        "0.0 clamps to the 0.1 floor"
    );
    assert_eq!(h.engine.render_scale(), RenderScale::MIN);
}

#[test]
fn a_viewport_too_small_to_scale_stays_native() {
    // 1×1 at any scale rounds back to 1×1, and blitting a texture onto itself
    // at 1:1 is waste with a chance of being wrong. Nothing should be allocated
    // at all.
    let Some(mut h) = Harness::new(1, 1) else {
        return;
    };
    h.frame(0.25);
    assert_eq!(h.engine.renderer().scaled_target_size(), None);
    assert_eq!(h.engine.render_size(1, 1), (1, 1));
}

// ---------------------------------------------------------------------------
// 4. The blockiness probe — the visible half of the feature
// ---------------------------------------------------------------------------

/// Fraction of even/odd neighbour *pairs* that are byte-identical, horizontally
/// and vertically. A nearest upscale of a half-resolution frame maps source
/// texel `i` onto destination pixels `2i` and `2i+1` on both axes, so at 0.5
/// this is 1.0 by construction — and at 1.0 it is whatever the picture happens
/// to be, which on a frame full of gradients is well below it.
fn paired_fraction(pixels: &[u8], width: u32, height: u32) -> f64 {
    let px = |x: u32, y: u32| {
        let i = ((y * width + x) * 4) as usize;
        &pixels[i..i + 4]
    };
    let mut matched = 0u64;
    let mut total = 0u64;
    for y in (0..height - 1).step_by(2) {
        for x in (0..width - 1).step_by(2) {
            // The three other members of the 2×2 block against its top-left.
            for (dx, dy) in [(1, 0), (0, 1), (1, 1)] {
                total += 1;
                if px(x, y) == px(x + dx, y + dy) {
                    matched += 1;
                }
            }
        }
    }
    matched as f64 / total as f64
}

#[test]
fn half_scale_comes_out_in_two_by_two_blocks() {
    let Some(mut h) = Harness::new(WIDTH, HEIGHT) else {
        return;
    };

    let native = paired_fraction(&h.frame(1.0), WIDTH, HEIGHT);
    let half = paired_fraction(&h.frame(0.5), WIDTH, HEIGHT);
    let quarter = paired_fraction(&h.frame(0.25), WIDTH, HEIGHT);
    println!(
        "2×2 block agreement: native {native:.3}, half {half:.3}, quarter {quarter:.3}"
    );

    // Exact, not approximate: nearest sampling at a 2:1 ratio with an aligned
    // grid produces identical bytes, so anything below 1.0 means the sampler is
    // filtering, the uv mapping is off by half a texel, or the internal target
    // is not the size it claims.
    assert_eq!(half, 1.0, "0.5 must be exact 2×2 blocks");
    assert_eq!(quarter, 1.0, "0.25 blocks contain 2×2 blocks too");

    // And the frame really is gradient-heavy, so the comparison means
    // something: at native resolution most neighbours differ.
    assert!(
        native < 0.8,
        "the probe view is too flat to say anything: {native:.3}"
    );
}

#[test]
fn the_blit_puts_the_picture_back_the_right_way_up() {
    // Blockiness alone cannot catch this: a frame flipped in v (texture space
    // runs top-down, NDC runs bottom-up, and the fullscreen triangle has to
    // undo exactly one of those) is still made of perfect 2×2 blocks. So this
    // asks the only question that distinguishes them — is a scaled pixel more
    // like the native pixel *there*, or the one at the mirror image?
    let Some(mut h) = Harness::new(WIDTH, HEIGHT) else {
        return;
    };
    let native = h.frame(1.0);
    let scaled = h.frame(0.5);

    let rgb = |buf: &[u8], x: u32, y: u32| {
        let i = ((y * WIDTH + x) * 4) as usize;
        [buf[i] as i32, buf[i + 1] as i32, buf[i + 2] as i32]
    };
    let distance = |a: [i32; 3], b: [i32; 3]| {
        (a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs()
    };

    let (mut upright, mut flipped_v, mut flipped_h, mut total) = (0, 0, 0, 0);
    for y in (4..HEIGHT - 4).step_by(7) {
        for x in (4..WIDTH - 4).step_by(7) {
            let here = rgb(&scaled, x, y);
            let d_same = distance(here, rgb(&native, x, y));
            let d_v = distance(here, rgb(&native, x, HEIGHT - 1 - y));
            let d_h = distance(here, rgb(&native, WIDTH - 1 - x, y));
            total += 1;
            if d_same <= d_v && d_same <= d_h {
                upright += 1;
            }
            if d_v < d_same {
                flipped_v += 1;
            }
            if d_h < d_same {
                flipped_h += 1;
            }
        }
    }
    let fraction = upright as f64 / total as f64;
    println!(
        "orientation: {upright}/{total} upright ({fraction:.3}), \
         {flipped_v} closer to a v-flip, {flipped_h} to an h-flip"
    );
    // Not 1.0: a half-resolution sample of a fine-detailed frame genuinely
    // differs from the full-resolution pixel under it, and a sky gradient is
    // near-symmetric about the vertical axis, so a minority of points can
    // legitimately prefer a mirror. A *flipped* frame would land near zero.
    assert!(
        fraction > 0.8,
        "the blit is not upright: only {fraction:.3} of samples match in place"
    );
}

// ---------------------------------------------------------------------------
// WebGL2: the blit has to survive translation, not just compilation
// ---------------------------------------------------------------------------

#[test]
fn the_blit_shader_translates_to_glsl_es_for_webgl2() {
    let module = naga::front::wgsl::parse_str(runt_core::BLIT_SHADER).expect("blit WGSL parses");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        // Nothing beyond the baseline: the point is that this shader needs no
        // capability WebGL2 lacks (DESIGN §11).
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .expect("blit WGSL validates");

    let options = naga::back::glsl::Options {
        version: naga::back::glsl::Version::Embedded {
            version: 300,
            is_webgl: true,
        },
        ..Default::default()
    };
    for (stage, entry) in [
        (naga::ShaderStage::Vertex, "vs_blit"),
        (naga::ShaderStage::Fragment, "fs_blit"),
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

// ---------------------------------------------------------------------------
// 6. Display density is the *other* scale, and the world is told the logical one
// ---------------------------------------------------------------------------

/// `Engine::render` reports the surface it was handed as the **logical** screen
/// the world lays its HUD out on — surface ÷ scale factor.
///
/// The two scales in this file are independent and it is worth being blunt
/// about which is which: render scale shrinks *the picture the scene is drawn
/// into* and never moves the HUD (that is the test above), while the scale
/// factor is how dense the host's pixels are and moves nothing except what a
/// pixel **means**. A frame can be at 0.5 render scale on a 2× display, and the
/// viewport the world sees answers only to the second.
#[test]
fn the_world_is_told_the_screen_in_logical_pixels() {
    use runt_core::ecs::Viewport;

    let Some(mut h) = Harness::new(WIDTH, HEIGHT) else {
        return;
    };

    let seen = |h: &Harness| {
        h.engine
            .sim()
            .world()
            .get_resource::<Viewport>()
            .copied()
            .expect("render writes the viewport")
    };

    // No host has said anything about density, so the two spaces coincide.
    h.engine.render(&h.view, h.width, h.height);
    assert_eq!(seen(&h), Viewport::new(WIDTH, HEIGHT));

    // A 2× panel showing the same window: half the logical screen, and the
    // regression this exists to catch — reported physical, a HUD anchored to
    // the right edge was laid out at x=256 on a screen 128 logical pixels wide,
    // which on a touch build is a button no finger can reach.
    h.engine.set_scale_factor(2.0);
    h.engine.render(&h.view, h.width, h.height);
    assert_eq!(seen(&h), Viewport::new(WIDTH / 2, HEIGHT / 2));

    // Render scale is orthogonal: the scene is drawn into a quarter of the
    // pixels and the screen the HUD is measured against does not move.
    let scaled = h.frame(0.5);
    assert_eq!(seen(&h), Viewport::new(WIDTH / 2, HEIGHT / 2));
    assert_eq!(
        h.engine.renderer().scaled_target_size(),
        Some((WIDTH / 2, HEIGHT / 2)),
        "the internal target follows the *surface*, not the logical screen",
    );
    assert_eq!(scaled.len(), (WIDTH * HEIGHT * 4) as usize);

    // A window dragged onto a 1× monitor mid-run reports the full screen again.
    h.engine.set_render_scale(1.0);
    h.engine.set_scale_factor(1.0);
    h.engine.render(&h.view, h.width, h.height);
    assert_eq!(seen(&h), Viewport::new(WIDTH, HEIGHT));

    // A fractional factor — a 125%-scaled desktop, the common Linux case.
    h.engine.set_scale_factor(1.25);
    h.engine.render(&h.view, h.width, h.height);
    assert_eq!(seen(&h), Viewport::new(205, 154));
    assert_eq!(h.engine.scale_factor(), 1.25);
}
