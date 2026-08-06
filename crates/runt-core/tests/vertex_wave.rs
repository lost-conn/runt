//! `MaterialVariant::VERTEX_WAVE` — the only vertex-stage look in the key
//! space (DESIGN §5), and a port of `3dimenshift/shaders/fx/water.gdshader`'s
//! `vertex()`.
//!
//! Four claims, in ascending order of how much of the engine they touch:
//!
//! 1. **The key space knows about it.** A permanent bit, a `const` the
//!    preprocessor emits, and a member of `IMPLEMENTED` — checked without a
//!    device, because the bit *number* is a compatibility promise
//!    (`tests/material_variants.rs` owns the rest of that argument).
//! 2. **Zero amplitude is the un-waved frame, byte for byte.** The same scene
//!    with the bit set and `params.x == 0` produces the identical bytes to the
//!    same scene with the bit cleared. That is the strong form of "the bit
//!    changes nothing else": not the normal, not the colour, not the fragment
//!    path, not the pipeline state.
//! 3. **Without the bit, the render clock is invisible.** Two frames at wall
//!    times 0 s and 3.7 s are byte-identical. So anything that *does* move in
//!    claim 4 moved because of the bit and for no other reason.
//! 4. **A displaced vertex is displaced by Godot's own arithmetic.** The top
//!    edge of a quad is measured in pixels at both of its ends, at two clocks,
//!    against the closed form
//!    `Δy = 0.5·A·(sin(x·f + t·s) + sin(z·f·1.3 + t·s·0.85))`. Both ends and
//!    both clocks, because a single probe cannot tell a *wave* from a rigid
//!    lift, and one clock cannot tell a wave from a static bend.
//!
//! # Why the projection is orthographic
//!
//! Every other GPU test here aims a perspective camera, because every other one
//! is about a *frame*. This one is about a number, and an ortho matrix makes
//! the pixel row of a world Y an exact linear function with no `w` in it —
//! so an expectation is arithmetic rather than a re-derivation of the
//! projection. `render` takes whatever `view_proj` it is handed.

use glam::{Mat4, Vec3, Vec4};

use bevy_ecs::prelude::World;
use runt_core::draw::build_draw_list;
use runt_core::ecs::Lighting;
use runt_core::registry::{MeshHandle, MeshLibrary};
use runt_core::texture::TextureLibrary;
use runt_core::{Material, MaterialVariant, Mesh, MeshRef, Renderer, Transform};

const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Half-extent of the orthographic box, in world units. With a 256 px square
/// target that is exactly 32 px per metre, and the two conversions below are
/// the whole of the projection.
const ORTHO_HALF: f32 = 4.0;
const PX_PER_UNIT: f32 = WIDTH as f32 * 0.5 / ORTHO_HALF;

/// The quad's resting half-height and half-width, in world units.
const QUAD_HALF_H: f32 = 1.0;
const QUAD_HALF_W: f32 = 3.0;

/// `water.gdshader`'s authored trio, scaled up: the amplitude is 0.13 m in the
/// original and 2 m here, because this test measures the displacement in whole
/// pixels and 0.13 m is four of them. The *shape* is the original's — the
/// crossed-sine ratios are shader constants, not params, so there is nothing to
/// scale about them.
const AMPLITUDE: f32 = 2.0;
const FREQUENCY: f32 = 0.5;
const SPEED: f32 = 1.0;

/// The second sine's ratios, restated from `shader.wgsl` so that a change to
/// either one fails here rather than being silently absorbed.
const CROSS_FREQ: f32 = 1.3;
const CROSS_SPEED: f32 = 0.85;

/// `fx/water.gdshader:28-31`, on the CPU. `z` is the world Z of the vertex,
/// which is zero for every vertex in this scene — spelled out anyway, because a
/// formula with a term missing is not the formula.
fn displacement(x: f32, z: f32, t: f32) -> f32 {
    let w = (x * FREQUENCY + t * SPEED).sin()
        + (z * FREQUENCY * CROSS_FREQ + t * SPEED * CROSS_SPEED).sin();
    w * AMPLITUDE * 0.5
}

// ---------------------------------------------------------------------------
// 1. The key space — no GPU
// ---------------------------------------------------------------------------

#[test]
fn the_wave_bit_is_declared_implemented_and_permanent() {
    assert_eq!(MaterialVariant::VERTEX_WAVE.bits(), 1 << 12);
    assert_eq!(
        MaterialVariant::VERTEX_WAVE.unimplemented(),
        MaterialVariant::NONE,
        "the bit does something, so it must not report as reserved"
    );
    // It is a *vertex* look, so it is not one of the bits that replace the
    // lighting term — a waving surface is still lit (or still unlit) exactly as
    // its other bits say.
    assert!(!MaterialVariant::UNLIT.contains(MaterialVariant::VERTEX_WAVE));
    // …and it is not a blend bit either: it does not move a draw out of the
    // opaque state-sort.
    assert!(!MaterialVariant::VERTEX_WAVE.intersects(MaterialVariant::BLENDED));

    let on = runt_core::material::variant_source(
        runt_core::material::BASE_SHADER,
        MaterialVariant::VERTEX_WAVE,
    );
    assert!(on.contains("const F_VERTEX_WAVE: bool = true;"));
    let off = runt_core::material::variant_source(
        runt_core::material::BASE_SHADER,
        MaterialVariant::NONE,
    );
    assert!(off.contains("const F_VERTEX_WAVE: bool = false;"));
}

#[test]
fn the_crossed_sine_is_the_originals() {
    // `water.gdshader` samples the pair at the *world* position and offsets the
    // second sine in both space and time; a pair with one ratio dropped would
    // still wobble, and would wobble as one diagonal wave. At a point where the
    // two disagree, the ratios are observable.
    // A clock where the two sines are far apart — near `t·s ≈ π/2` they agree to
    // three decimals whatever the ratio, and an assertion there would pass on a
    // shader that had dropped it.
    let t = 3.0;
    let at_origin = ((t * SPEED).sin() + (t * SPEED * CROSS_SPEED).sin()) * AMPLITUDE * 0.5;
    assert!(
        (displacement(0.0, 0.0, t) - at_origin).abs() < 1.0e-6,
        "at the origin the pair is the two clocks alone"
    );
    assert!(
        (displacement(0.0, 2.0, t) - at_origin).abs() > 0.1,
        "…and away from z = 0 it is not, or the cross frequency is dead"
    );
    assert!(
        (displacement(2.0, 0.0, t) - at_origin).abs() > 0.1,
        "…nor away from x = 0, or the first sine is dead"
    );
    // The two clocks are different clocks: were `CROSS_SPEED` 1.0, the pair
    // would be one sine of twice the amplitude and the crossing would be a lie.
    assert!(
        ((t * SPEED).sin() - (t * SPEED * CROSS_SPEED).sin()).abs() > 0.01,
        "the second sine runs on the same clock as the first"
    );
    // t = 0 is the flat surface: both sines are sin(0).
    assert_eq!(displacement(0.0, 0.0, 0.0), 0.0);
}

// ---------------------------------------------------------------------------
// 2–4. The pixels
// ---------------------------------------------------------------------------

/// A quad in the XY plane at `z = 0`, wound counter-clockwise so the backface
/// culling every variant carries keeps it. Two triangles sharing the diagonal
/// `0–2`, so the **top edge is the single segment 3→2** and the silhouette
/// there is the straight line between two displaced vertices — which is what
/// makes a per-vertex displacement measurable at all.
fn quad() -> Mesh {
    Mesh {
        positions: vec![
            Vec3::new(-QUAD_HALF_W, -QUAD_HALF_H, 0.0),
            Vec3::new(QUAD_HALF_W, -QUAD_HALF_H, 0.0),
            Vec3::new(QUAD_HALF_W, QUAD_HALF_H, 0.0),
            Vec3::new(-QUAD_HALF_W, QUAD_HALF_H, 0.0),
        ],
        normals: vec![Vec3::Z; 4],
        uvs: vec![glam::Vec2::ZERO; 4],
        colors: vec![Vec3::ONE; 4],
        indices: vec![0, 1, 2, 0, 2, 3],
    }
}

/// White on black: the lighting rig is flattened to nothing so the sky pass
/// paints the background a single colour, and the quad is `BILLBOARD_UNLIT` so
/// it is exactly its base colour. "Is this pixel the quad?" is then a threshold
/// rather than a judgement.
fn dark_sky() -> Lighting {
    Lighting {
        key_color: Vec3::ZERO,
        sky_color: Vec3::ZERO,
        ground_color: Vec3::ZERO,
        horizon: Some(Vec3::ZERO),
        clouds: 0.0,
        sun: 0.0,
        ..Lighting::default()
    }
}

fn view_proj() -> Mat4 {
    glam::camera::rh::proj::directx::orthographic(
        -ORTHO_HALF,
        ORTHO_HALF,
        -ORTHO_HALF,
        ORTHO_HALF,
        -10.0,
        10.0,
    )
}

/// World X at the centre of pixel column `col`.
fn column_x(col: u32) -> f32 {
    (col as f32 + 0.5 - WIDTH as f32 * 0.5) / PX_PER_UNIT
}

/// The row a top edge at world `y` first covers: the topmost pixel whose centre
/// is at or below it.
fn expected_row(y: f32) -> f32 {
    HEIGHT as f32 * 0.5 - y * PX_PER_UNIT - 0.5
}

struct Frame {
    pixels: Vec<u8>,
}

impl Frame {
    /// The topmost lit row in a column, or `None` if the column is all
    /// background.
    fn top_edge(&self, col: u32) -> Option<u32> {
        (0..HEIGHT).find(|row| {
            let i = ((row * WIDTH + col) * 4) as usize;
            self.pixels[i] > 128
        })
    }
}

/// One device, one mesh, one target — and as many frames as a test wants off
/// them.
///
/// **One `Renderer` per test, held for the test.** That is how every GPU test
/// file here is arranged (`phase_screen.rs`'s `Rig`, `texture_bake.rs`'s
/// `renderer()`), and this file learned the hard way that it is load-bearing
/// rather than tidy: an earlier draft built a whole `Renderer` — and therefore a
/// whole `wgpu::Instance` — *per rendered frame*, eight of them across four
/// tests that all start within microseconds of each other. `Instance::new`
/// drives the platform loader's lazy, process-global initialization, and two
/// threads inside that at once read a function-pointer table the other has not
/// finished filling in. The result was a SIGSEGV at address zero in
/// `libvulkan.so`, about one run in three, in whichever test happened to lose.
///
/// Sharing one `Instance` across the binary is **not** the fix and was tried:
/// the adapter here is the GL backend, one instance is one EGL context, and
/// `eglMakeCurrent` from two threads returns `BadAccess`. The fix is simply to
/// stop asking for devices in a loop.
///
/// It also makes the tests say more than they did. Comparing two frames for
/// byte equality across two separately created devices was quietly asserting
/// that two drivers agree; comparing them on **one** device asserts what the
/// claims are actually about.
struct Rig {
    renderer: Renderer,
    handle: MeshHandle,
    target: wgpu::Texture,
    view: wgpu::TextureView,
}

impl Rig {
    fn new() -> Option<Rig> {
        let mut renderer = match pollster::block_on(Renderer::headless(FORMAT)) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP vertex_wave (no GPU adapter): {e}");
                return None;
            }
        };

        let mesh = quad();
        let handle = renderer.register_mesh(&mesh);
        assert_eq!(handle, MeshHandle::of(&mesh));

        let target = renderer.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("vertex_wave target"),
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
            handle,
            target,
            view,
        })
    }

    /// Draw the quad once. `variant` is ORed onto `BILLBOARD_UNLIT`, `amplitude`
    /// goes into `params.x`, and `seconds` is the render clock.
    ///
    /// The target is reused between frames — the pass clears it — so a second
    /// frame costs a draw and a readback rather than a device.
    fn frame(&mut self, variant: MaterialVariant, amplitude: f32, seconds: f32) -> Frame {
        let mut world = World::new();
        world.spawn((
            MeshRef(self.handle),
            Material {
                base_color: Vec4::ONE,
                params: Vec4::new(amplitude, FREQUENCY, SPEED, 0.0),
                texture: None,
                variant: variant | MaterialVariant::BILLBOARD_UNLIT,
            },
            Transform::IDENTITY,
        ));
        let draws = build_draw_list(&mut world, 0.0);

        let frame = runt_core::FrameParams {
            view_proj: view_proj(),
            lighting: dark_sky(),
        };
        // The alpha is pinned at zero: this feature reads `time.x` and a test
        // that let `time.y` drift would be testing two things.
        self.renderer.set_render_clock(seconds, 0.0);
        self.renderer.render(
            &self.view,
            WIDTH,
            HEIGHT,
            &frame,
            &draws,
            &MeshLibrary::new(),
            &TextureLibrary::new(),
        );

        Frame {
            pixels: read_back(&self.renderer, &self.target),
        }
    }
}

fn read_back(renderer: &Renderer, target: &wgpu::Texture) -> Vec<u8> {
    let device = renderer.device();
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
        target.as_image_copy(),
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
    renderer.queue().submit(Some(encoder.finish()));

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

#[test]
fn a_zero_amplitude_wave_is_the_unwaved_frame_byte_for_byte() {
    let Some(mut rig) = Rig::new() else { return };
    let waved = rig.frame(MaterialVariant::VERTEX_WAVE, 0.0, 2.5);
    let plain = rig.frame(MaterialVariant::NONE, 0.0, 2.5);
    assert_eq!(
        waved.pixels, plain.pixels,
        "the bit at zero amplitude changed something other than the vertex"
    );
}

#[test]
fn without_the_bit_the_render_clock_is_invisible() {
    let Some(mut rig) = Rig::new() else { return };
    let at_zero = rig.frame(MaterialVariant::NONE, AMPLITUDE, 0.0);
    let later = rig.frame(MaterialVariant::NONE, AMPLITUDE, 3.7);
    assert_eq!(
        at_zero.pixels, later.pixels,
        "an un-waved draw moved with the clock — something else reads time.x"
    );
}

#[test]
fn a_waved_vertex_moves_by_the_originals_arithmetic() {
    // Both ends of the quad, one pixel in from each so the probe is inside the
    // triangle rather than on its corner.
    let left = ((WIDTH as f32 * 0.5) - QUAD_HALF_W * PX_PER_UNIT) as u32 + 1;
    let right = ((WIDTH as f32 * 0.5) + QUAD_HALF_W * PX_PER_UNIT) as u32 - 1;

    // Two clocks: `0.0` is the flat surface (both sines are `sin(0)`) and
    // `1.9` is one where the two ends disagree by more than a metre.
    let Some(mut rig) = Rig::new() else { return };
    for seconds in [0.0f32, 1.9] {
        let frame = rig.frame(MaterialVariant::VERTEX_WAVE, AMPLITUDE, seconds);
        for col in [left, right] {
            // The top edge is the segment between the two displaced top
            // vertices, so the expectation at an interior column is the *linear*
            // interpolation of their displacements — which is what the
            // rasterizer interpolates, and is not the sine sampled there.
            let x = column_x(col);
            let a = displacement(-QUAD_HALF_W, 0.0, seconds);
            let b = displacement(QUAD_HALF_W, 0.0, seconds);
            let s = (x + QUAD_HALF_W) / (2.0 * QUAD_HALF_W);
            let want = expected_row(QUAD_HALF_H + a + (b - a) * s);

            let got = frame.top_edge(col).unwrap_or_else(|| {
                panic!("column {col} at t = {seconds} is empty — the quad left the target")
            }) as f32;
            assert!(
                (got - want).abs() <= 1.0,
                "t = {seconds}, column {col}: edge at row {got}, expected {want}"
            );
        }
    }
}

#[test]
fn the_wave_is_a_wave_and_not_a_lift() {
    // A rigid lift would move both ends of the quad by the same amount, and a
    // static bend would move neither with the clock. Measured as a *pair of
    // differences*, so neither can pass for the other.
    let left = ((WIDTH as f32 * 0.5) - QUAD_HALF_W * PX_PER_UNIT) as u32 + 1;
    let right = ((WIDTH as f32 * 0.5) + QUAD_HALF_W * PX_PER_UNIT) as u32 - 1;

    let Some(mut rig) = Rig::new() else { return };
    let early = rig.frame(MaterialVariant::VERTEX_WAVE, AMPLITUDE, 0.0);
    let late = rig.frame(MaterialVariant::VERTEX_WAVE, AMPLITUDE, 1.9);

    let tilt = |f: &Frame| f.top_edge(left).unwrap() as i64 - f.top_edge(right).unwrap() as i64;
    assert!(
        tilt(&early).abs() > 2,
        "at t = 0 the two ends are sin(∓1.5) apart; the surface is not tilted"
    );
    assert_ne!(
        tilt(&early),
        tilt(&late),
        "the tilt did not change with the clock — the wave is frozen"
    );
    assert_ne!(
        early.top_edge(left),
        late.top_edge(left),
        "the left end did not move with the clock"
    );
}
