//! `MaterialVariant::TEXTURE_SCROLL` — the clock on the texture tap (DESIGN §7,
//! §5), and a port of `3dimenshift/shaders/fx/water.gdshader:35`'s
//!
//! ```text
//! uv = world_pos.xz · noise_scale + vec2(TIME·scroll_speed, TIME·scroll_speed·0.7)
//! ```
//!
//! # The claim, and why a variant-key test would not be one
//!
//! The bit exists so a *standing* surface can have a *travelling* pattern: a
//! pond whose mottling drifts is water, and one whose mottling is nailed down is
//! a sheet of painted glass. Asserting that bit 18 reaches the pipeline cache
//! would prove none of that, so the load-bearing tests here measure the drift
//! directly, once per texture path and in the strongest form each path admits:
//!
//!   * **Baked, as an exact integer-pixel registration.** The surface faces `+Z`,
//!     so `triplanar_blend` gives the XY plane a weight of exactly one and the
//!     other two exactly zero — the tap is a plain 2D lookup at `(x, y)·scale`,
//!     and a world-X offset is therefore a pure horizontal translation of the
//!     image. Choose the offset to be a whole number of pixels' worth of metres
//!     and the scrolled frame must be the still frame *shifted*, with no
//!     resampling anywhere to excuse with a tolerance. (The cross-axis half of
//!     the offset lands in `p.z`, which on this surface is weighted at zero —
//!     which is exactly why the live test below is the one that measures it.)
//!   * **Live, against the CPU twin, absolutely.** One frame, two hypotheses: is
//!     this pixel `live_albedo_at(world + offset)` or `live_albedo_at(world)`?
//!     The first fits to about one LSB and the second does not fit at all, which
//!     pins the *semantics* — including the `0.7` cross ratio, which is a term of
//!     the offset the twin predicts and is not observable on the baked framing.
//!
//! Both are bracketed by the two claims that say the bit changes nothing else:
//! a zero speed is the still frame byte for byte, and without the bit the render
//! clock is invisible.
//!
//! # Why the offset is in metres and not in tile units
//!
//! The two branches scale `p_source` by `world_scale` and by
//! `live_cells_per_metre` respectively, so one authored number applied after
//! either scaling would mean two different speeds either side of §7's live gate.
//! Applied before, as `shader.wgsl` does, it means metres per second on both —
//! which is why the same `SCROLL_SPEED` below drives the baked registration and
//! the live twin and why the two are the same claim about one number.

use glam::{Mat4, Vec2, Vec3, Vec4};

use runt_core::draw::{DrawItem, FrameParams};
use runt_core::mesh::MeshData;
use runt_core::registry::{MeshHandle, MeshLibrary};
use runt_core::texture::{self, TextureLibrary, TextureSpec, LIVE_LOD_CELL_PIXELS};
use runt_core::{Lighting, MaterialVariant, NoopCache, Renderer};

mod common;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Square, and `tests/local_space.rs`'s size for its reason: every framing here
/// is fully covered by the surface, so no frame contains a silhouette and no
/// rasterizer fill rule has to be argued about.
const SIZE: u32 = 192;

/// Half-extent of the orthographic box for the **CPU-twin** framing, in metres.
/// Small enough that the live octave window has faded nothing, which
/// [`the_twin_framing_leaves_every_octave_on`] asserts rather than assumes.
const TWIN_HALF: f32 = 1.2;

/// Half-extent for the **registration** framing. Forty metres across shows well
/// over one 27.8 m tile of the fine fixture, so "the two frames differ" is a
/// claim about a picture rather than about bilinear noise.
const WIDE_HALF: f32 = 20.0;

/// The surface's half-extent — larger than `WIDE_HALF · √2`, so every framing is
/// covered edge to edge.
const QUAD_HALF: f32 = 48.0;

/// The baked tile's resolution. The live tests never read a tile and bake
/// `MIN_RESOLUTION`; this one does read it, and 512 texels across a 27.8 m tile
/// is 18 per metre against `WIDE_HALF`'s 0.21 m per pixel.
const BAKE_RESOLUTION: u32 = 512;

/// How far the pattern travels in the registration experiment, **in pixels**.
///
/// A whole number of pixels is the whole trick: under the orthographic camera the
/// pixel grid and the world are related by an exact scale, so a drift of
/// `DRIFT_PX` pixels' worth of metres makes the expected image an exact integer
/// shift of the still one.
const DRIFT_PX: u32 = 37;

/// The clock the drift is measured at. Not 1.0, so that a shader which had
/// dropped the multiply and scrolled by `params.w` alone would fail rather than
/// pass by coincidence.
const SECONDS: f32 = 2.5;

/// `shader.wgsl`'s cross ratio, restated so that a change to it fails here rather
/// than being silently absorbed. `fx/water.gdshader:35`'s `0.7`.
const CROSS_SPEED: f32 = 0.7;

/// Metres per second, chosen so that `SPEED · SECONDS` is exactly `DRIFT_PX`
/// pixels of the wide framing.
fn scroll_speed() -> f32 {
    DRIFT_PX as f32 * metres_per_pixel(WIDE_HALF) / SECONDS
}

/// Where the sampling point has walked to by `seconds` — the offset
/// `shader.wgsl` adds to `p_source`, on the CPU.
fn drift(seconds: f32) -> Vec3 {
    let d = scroll_speed() * seconds;
    Vec3::new(d, 0.0, d * CROSS_SPEED)
}

// ---------------------------------------------------------------------------
// 1. The key space — no GPU
// ---------------------------------------------------------------------------

#[test]
fn the_scroll_bit_is_declared_implemented_and_permanent() {
    assert_eq!(MaterialVariant::TEXTURE_SCROLL.bits(), 1 << 18);
    assert_eq!(
        MaterialVariant::TEXTURE_SCROLL.unimplemented(),
        MaterialVariant::NONE,
        "the bit does something, so it must not report as reserved"
    );
    // It slides a sampling point and nothing else: it replaces no lighting term
    // and it moves no draw out of the opaque state-sort.
    assert!(!MaterialVariant::UNLIT.contains(MaterialVariant::TEXTURE_SCROLL));
    assert!(!MaterialVariant::TEXTURE_SCROLL.intersects(MaterialVariant::BLENDED));

    // …and it touches no fixed-function state either, so a scrolled draw is the
    // still one's pipeline with one `const` flipped.
    let plain = runt_core::render_state(MaterialVariant::TEXTURE);
    let scrolled =
        runt_core::render_state(MaterialVariant::TEXTURE | MaterialVariant::TEXTURE_SCROLL);
    assert_eq!(scrolled, plain, "a sampling offset is not a pipeline state");
}

#[test]
fn the_base_shader_branches_on_the_scroll_const_and_holds_the_ratio_itself() {
    let on = runt_core::material::variant_source(
        runt_core::material::BASE_SHADER,
        MaterialVariant::TEXTURE_SCROLL,
    );
    assert!(on.contains("const F_TEXTURE_SCROLL: bool = true;"));
    let off = runt_core::material::variant_source(
        runt_core::material::BASE_SHADER,
        MaterialVariant::NONE,
    );
    assert!(off.contains("const F_TEXTURE_SCROLL: bool = false;"));

    // A *shader* bit, so the base source has to read the const — the distinction
    // `tests/local_space.rs` draws against `TWO_SIDED`, which is declared and
    // read by nothing.
    assert!(
        runt_core::material::BASE_SHADER.contains("F_TEXTURE_SCROLL"),
        "the bit is a shader branch; the base source has to read the const"
    );
    // The cross ratio is the shader's own constant and not a param. If it ever
    // became authorable, this file's `CROSS_SPEED` — and the twin hypothesis
    // built on it — would be measuring a default rather than the design.
    assert!(
        runt_core::material::BASE_SHADER.contains("const SCROLL_CROSS_SPEED: f32 = 0.7;"),
        "the 0.7 cross ratio is `fx/water.gdshader:35`'s literal and must stay a constant"
    );
}

// ---------------------------------------------------------------------------
// The rig
// ---------------------------------------------------------------------------

/// A square in the XY plane at `z = 0`, facing `+Z` and centred on the origin.
///
/// Facing `+Z` exactly is load-bearing for the baked half: `triplanar_blend` of
/// `(0, 0, 1)` is `(0, 0, 1)`, so the XY plane carries the whole tap and the
/// world-X drift is an exact horizontal translation of the image. Counter-
/// clockwise seen from `+Z`, which is where the camera is.
fn quad() -> MeshData {
    let h = QUAD_HALF;
    MeshData {
        positions: vec![
            Vec3::new(-h, -h, 0.0),
            Vec3::new(h, -h, 0.0),
            Vec3::new(h, h, 0.0),
            Vec3::new(-h, h, 0.0),
        ],
        normals: vec![Vec3::Z; 4],
        uvs: vec![Vec2::ZERO; 4],
        colors: vec![Vec3::ONE; 4],
        indices: vec![0, 1, 2, 0, 2, 3],
    }
}

/// Orthographic, axis-aligned on the origin, `w = 1` everywhere.
fn view_proj(half: f32) -> Mat4 {
    glam::camera::rh::proj::directx::orthographic(-half, half, -half, half, -1.0, 1.0)
}

/// The world point at the centre of pixel `(x, y)`.
fn world_at(half: f32, x: u32, y: u32) -> Vec3 {
    let span = half * 2.0;
    Vec3::new(
        -half + (x as f32 + 0.5) / SIZE as f32 * span,
        // Row 0 is the top of the target and `+Y` is up, so v runs backwards.
        -half + (1.0 - (y as f32 + 0.5) / SIZE as f32) * span,
        0.0,
    )
}

fn metres_per_pixel(half: f32) -> f32 {
    half * 2.0 / SIZE as f32
}

/// A light rig that multiplies albedo by exactly one. Every material below is
/// `BILLBOARD_UNLIT`, so a frame is a direct readout of albedo and the only thing
/// being measured is where the texture was sampled.
fn flat_lighting() -> Lighting {
    Lighting {
        key_dir: Vec3::Y,
        key_color: Vec3::ZERO,
        sky_color: Vec3::ONE,
        ground_color: Vec3::ONE,
        horizon: None,
        ..Lighting::default()
    }
}

/// One device, one mesh, one target, one bake — held for the whole test, for the
/// reason `tests/vertex_wave.rs`'s `Rig` gives at length: a `Renderer` per frame
/// races the platform loader's process-global init and SIGSEGVs inside
/// `libvulkan.so` about one run in three.
struct Rig {
    renderer: Renderer,
    meshes: MeshLibrary,
    mesh: MeshHandle,
    textures: TextureLibrary,
    handle: runt_core::TextureHandle,
    spec: TextureSpec,
    target: wgpu::Texture,
    view: wgpu::TextureView,
}

impl Rig {
    fn new(label: &str, resolution: u32) -> Option<Rig> {
        let mut renderer = match pollster::block_on(Renderer::headless(FORMAT)) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP texture_scroll::{label} (no GPU adapter): {e}");
                return None;
            }
        };

        let mut meshes = MeshLibrary::new();
        let mesh = meshes.insert(quad());

        let spec = common::fine();
        let mut textures = TextureLibrary::new();
        let handle = textures.insert(spec.clone(), resolution);
        renderer.bake_texture(&spec, resolution, &NoopCache);

        let target = renderer.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("texture_scroll target"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
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
            meshes,
            mesh,
            textures,
            handle,
            spec,
            target,
            view,
        })
    }

    /// Draw the surface once, wearing `variant | BILLBOARD_UNLIT`, with `speed`
    /// in `params.w` and the render clock at `seconds`. `textured` decides
    /// whether the baked pair is bound at all, so "inert without a texture" can
    /// be claimed through the same path as everything else.
    fn frame(
        &mut self,
        variant: MaterialVariant,
        speed: f32,
        seconds: f32,
        half: f32,
        textured: bool,
    ) -> Vec<u8> {
        let draws = [DrawItem {
            entity: bevy_ecs::entity::Entity::from_raw_u32(0).expect("entity 0"),
            variant: variant | MaterialVariant::BILLBOARD_UNLIT,
            mesh: self.mesh,
            model: Mat4::IDENTITY,
            base_color: Vec4::ONE,
            params: Vec4::new(0.0, 0.0, 0.0, speed),
            texture: if textured { Some(self.handle) } else { None },
        }];
        // The interpolation alpha is pinned at zero: this feature reads `time.x`
        // and a test that let `time.y` drift would be testing two things.
        self.renderer.set_render_clock(seconds, 0.0);
        self.renderer.render(
            &self.view,
            SIZE,
            SIZE,
            &FrameParams {
                view_proj: view_proj(half),
                lighting: flat_lighting(),
            },
            &draws,
            &self.meshes,
            &self.textures,
        );
        read_back(&self.renderer, &self.target)
    }
}

fn read_back(renderer: &Renderer, target: &wgpu::Texture) -> Vec<u8> {
    let (device, queue) = (renderer.device(), renderer.queue());
    let padded = (SIZE * 4).div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("texture_scroll readback"),
        size: (padded * SIZE) as u64,
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
                bytes_per_row: Some(padded),
                rows_per_image: Some(SIZE),
            },
        },
        wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
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
        .expect("poll");
    rx.recv().expect("map callback").expect("mapped");

    let mapped = readback.get_mapped_range(..).expect("mapped range");
    let mut pixels = Vec::with_capacity((SIZE * SIZE * 4) as usize);
    for row in 0..SIZE as usize {
        let start = row * padded as usize;
        pixels.extend_from_slice(&mapped[start..start + (SIZE * 4) as usize]);
    }
    drop(mapped);
    readback.unmap();
    pixels
}

fn pixel(pixels: &[u8], x: u32, y: u32) -> [f32; 3] {
    let i = (y as usize * SIZE as usize + x as usize) * 4;
    [
        pixels[i] as f32 / 255.0,
        pixels[i + 1] as f32 / 255.0,
        pixels[i + 2] as f32 / 255.0,
    ]
}

/// Sorted per-pixel max-channel errors against a colour the caller predicts, over
/// a coprime scatter of the whole target — `tests/local_space.rs`'s shape, and
/// `tests/live_texture.rs`'s before it.
fn divergence(pixels: &[u8], samples: u32, mut want: impl FnMut(u32, u32) -> Vec3) -> Vec<f32> {
    let mut errors = Vec::with_capacity(samples as usize);
    for i in 0..samples {
        let x = i.wrapping_mul(97) % SIZE;
        let y = i.wrapping_mul(59).wrapping_add(i / SIZE * 31) % SIZE;
        let expected = want(x, y);
        let got = pixel(pixels, x, y);
        errors.push(
            (0..3)
                .map(|c| (got[c] - expected[c]).abs())
                .fold(0.0f32, f32::max),
        );
    }
    errors.sort_by(f32::total_cmp);
    errors
}

/// How well a hypothesis fitted, as `(fraction within one LSB, p99, share past
/// 4 LSB)` — printed, because the numbers are the evidence.
/// `tests/local_space.rs::report` argues at length why the fit *fraction* rather
/// than the median is the load-bearing statistic on a `CellValue` field.
fn report(label: &str, errors: &[f32]) -> (f32, f32, f32) {
    const LSB: f32 = 1.0 / 255.0;
    let share = |t: f32| errors.iter().filter(|e| **e > t).count() as f32 / errors.len() as f32;
    let fit = 1.0 - share(LSB);
    let p99 = errors[errors.len() * 99 / 100];
    let visible = share(4.0 * LSB);
    println!(
        "{label}: {} probes — {:.2}% within 1 LSB, p99 {:.5} ({:.2} LSB), {:.2}% past 4 LSB",
        errors.len(),
        fit * 100.0,
        p99,
        p99 * 255.0,
        visible * 100.0
    );
    (fit, p99, visible)
}

/// How much two frames disagree over a region, as `(mean, fraction over 4 LSB)`.
fn disagreement(
    a: &[u8],
    b: &[u8],
    mut map: impl FnMut(u32, u32) -> Option<(u32, u32)>,
) -> (f32, f32) {
    let mut sum = 0.0f32;
    let mut over = 0usize;
    let mut n = 0usize;
    for y in 0..SIZE {
        for x in 0..SIZE {
            let Some((bx, by)) = map(x, y) else { continue };
            let p = pixel(a, x, y);
            let q = pixel(b, bx, by);
            let delta = (0..3).map(|c| (p[c] - q[c]).abs()).fold(0.0f32, f32::max);
            sum += delta;
            if delta > 4.0 / 255.0 {
                over += 1;
            }
            n += 1;
        }
    }
    assert!(n > (SIZE * SIZE / 2) as usize, "the comparison region collapsed");
    (sum / n as f32, over as f32 / n as f32)
}

// ---------------------------------------------------------------------------
// 2. The precondition the twin comparison rests on — no GPU
// ---------------------------------------------------------------------------

#[test]
fn the_twin_framing_leaves_every_octave_on() {
    // The CPU twin has no camera and weights every octave at 1, so a comparison
    // against a live fragment is only meaningful where the octave window has
    // faded nothing. `tests/local_space.rs` makes the same check for the same
    // framing; restated here because this file would otherwise fail for a reason
    // that has nothing to do with the scroll.
    let spec = common::fine();
    let footprint = metres_per_pixel(TWIN_HALF) * spec.live_cells_per_metre();
    let (min, _) = spec.live_octave_window(footprint, LIVE_LOD_CELL_PIXELS);
    let last = spec.octave_plan().len() as f32 - 1.0;
    println!(
        "twin framing: {:.5} m/px → {footprint:.5} cells/px → window floor {min:.2}",
        metres_per_pixel(TWIN_HALF)
    );
    assert!(
        min >= last,
        "octave {last} is already fading at the twin framing ({min:.2} < {last})"
    );
}

#[test]
fn the_drift_is_a_whole_number_of_pixels_of_the_wide_framing() {
    // The registration test's own assumption, as an assertion: if `DRIFT_PX`,
    // `SIZE` or `WIDE_HALF` ever moved so that the drift stopped landing on the
    // pixel grid, the exact comparison would start failing for a reason that has
    // nothing to do with the shader.
    let want = DRIFT_PX as f32 * metres_per_pixel(WIDE_HALF);
    let got = drift(SECONDS).x;
    assert!(
        (got - want).abs() < 1.0e-4,
        "the drift at t = {SECONDS} is {got} m, not {want} m ({DRIFT_PX} px)"
    );
    // …and the cross axis is the ratio and not a copy, or the twin test below
    // would be asserting the same term twice.
    assert!((drift(SECONDS).z - want * CROSS_SPEED).abs() < 1.0e-4);
    assert_ne!(drift(SECONDS).z, drift(SECONDS).x);
}

// ---------------------------------------------------------------------------
// 3. The two claims that say the bit changes nothing else
// ---------------------------------------------------------------------------

#[test]
fn a_zero_speed_scroll_is_the_still_frame_byte_for_byte() {
    // The strong form of "the bit changes nothing but the sampling point": not
    // the mip level, not the triplanar blend, not the pipeline state. A surface
    // authored with the bit and no speed has to be the surface without it.
    let Some(mut rig) = Rig::new(
        "a_zero_speed_scroll_is_the_still_frame_byte_for_byte",
        BAKE_RESOLUTION,
    ) else {
        return;
    };
    let scrolled = rig.frame(
        MaterialVariant::TEXTURE | MaterialVariant::TEXTURE_SCROLL,
        0.0,
        SECONDS,
        WIDE_HALF,
        true,
    );
    let still = rig.frame(MaterialVariant::TEXTURE, 0.0, SECONDS, WIDE_HALF, true);
    assert_eq!(
        scrolled, still,
        "the bit at zero speed changed something other than the sampling point"
    );

    // …and it is inert with nothing to sample, however fast it is told to go —
    // the same shape of claim `local_space.rs` makes about a textureless draw.
    let untextured = rig.frame(MaterialVariant::NONE, scroll_speed(), SECONDS, WIDE_HALF, false);
    let untextured_scrolled = rig.frame(
        MaterialVariant::TEXTURE_SCROLL,
        scroll_speed(),
        SECONDS,
        WIDE_HALF,
        false,
    );
    assert_eq!(
        untextured, untextured_scrolled,
        "the bit slides where a texture is sampled; with none bound it must be inert"
    );
}

#[test]
fn without_the_bit_the_render_clock_is_invisible() {
    // So anything that moves in the two tests below moved because of the bit and
    // for no other reason. `tests/vertex_wave.rs` makes the same claim for the
    // vertex stage; this is the fragment half of it, on a textured draw.
    let Some(mut rig) = Rig::new("without_the_bit_the_render_clock_is_invisible", BAKE_RESOLUTION)
    else {
        return;
    };
    let at_zero = rig.frame(MaterialVariant::TEXTURE, scroll_speed(), 0.0, WIDE_HALF, true);
    let later = rig.frame(MaterialVariant::TEXTURE, scroll_speed(), 3.7, WIDE_HALF, true);
    assert_eq!(
        at_zero, later,
        "an unscrolled textured draw moved with the clock — something else reads time.x"
    );
}

// ---------------------------------------------------------------------------
// 4. The baked path, as an exact registration
// ---------------------------------------------------------------------------

/// **The pattern travels, and it travels by the clock times the speed.**
///
/// The still frame and the scrolled one are compared under an integer column
/// shift, which on this framing is exactly what a world-X offset of `DRIFT_PX`
/// pixels' worth of metres does to a `+Z`-facing triplanar tap. Both halves are
/// load-bearing: "the shifted frames match" alone would pass on a build where the
/// texture had become a flat wash, so the frame's own content is asserted first
/// against an unrelated registration of itself.
#[test]
fn a_scrolled_baked_tap_is_the_still_frame_translated_by_the_clock() {
    let Some(mut rig) = Rig::new(
        "a_scrolled_baked_tap_is_the_still_frame_translated_by_the_clock",
        BAKE_RESOLUTION,
    ) else {
        return;
    };

    let still = rig.frame(MaterialVariant::TEXTURE, 0.0, 0.0, WIDE_HALF, true);
    let scrolled = rig.frame(
        MaterialVariant::TEXTURE | MaterialVariant::TEXTURE_SCROLL,
        scroll_speed(),
        SECONDS,
        WIDE_HALF,
        true,
    );

    // Content first: the still frame against a third-of-a-frame shift of itself,
    // which is the scale on which "these are two different images" measures here.
    // A third rather than a half because half the columns would leave exactly
    // half the target to compare over, and `disagreement` refuses that.
    let unrelated = |x: u32, y: u32| (x + SIZE / 3 < SIZE).then(|| (x + SIZE / 3, y));
    let (self_mean, self_over) = disagreement(&still, &still, unrelated);
    println!("still frame vs an unrelated shift of itself — mean {self_mean:.5}, over {self_over:.3}");
    assert!(
        self_mean > 0.02 && self_over > 0.5,
        "the still frame is too uniform for any registration test to mean anything \
         (mean {self_mean:.5}, {:.1}% over 4 LSB)",
        self_over * 100.0
    );

    // Column `x` of the scrolled frame samples the field at `world_at(x) + drift`,
    // which is the world point column `x + DRIFT_PX` of the still frame showed.
    // The `DRIFT_PX` columns that walked in from off-frame are skipped.
    let shifted = |x: u32, y: u32| (x + DRIFT_PX < SIZE).then(|| (x + DRIFT_PX, y));
    let (mean, over) = disagreement(&scrolled, &still, shifted);
    println!(
        "scrolled vs still, registered by {DRIFT_PX} px: mean {mean:.5} ({:.2} LSB), \
         {:.3}% over 4 LSB",
        mean * 255.0,
        over * 100.0
    );
    // Not zero, and the residue is named: `CellValue` is a step function of which
    // lattice cell won, so a fragment within a float of a boundary can land either
    // side once its coordinate has been through a multiply-add. Single-pixel
    // events on a boundary, which the over-4-LSB fraction bounds — a wrong offset
    // moves the whole image, not a scatter of pixels.
    assert!(
        mean < 1.0 / 255.0 && over < 0.01,
        "the scrolled frame is not the still one shifted by {DRIFT_PX} px \
         (mean {:.2} LSB, {:.1}% over 4 LSB) — the drift is not `params.w · time.x`",
        mean * 255.0,
        over * 100.0
    );

    // …and the un-registered comparison is the artifact itself: without the shift
    // the two frames are two different images, so the registration above is a
    // result and not a tautology about a picture that never moved.
    let (still_mean, still_over) = disagreement(&scrolled, &still, |x, y| Some((x, y)));
    println!(
        "scrolled vs still, unregistered: mean {still_mean:.5}, {:.3}% over 4 LSB",
        still_over * 100.0
    );
    assert!(
        still_over > 0.4 && still_mean > 0.3 * self_mean,
        "the scrolled frame is the still frame in place — the bit did nothing \
         (mean {still_mean:.5} against an unrelated-registration scale of {self_mean:.5})"
    );
}

// ---------------------------------------------------------------------------
// 5. The live path, against the CPU twin
// ---------------------------------------------------------------------------

/// The strongest form of the claim, and the only one that can see the cross axis:
/// one rendered frame, two hypotheses, and the CPU says which one the GPU drew.
///
/// The live path has no triplanar projection — it evaluates the 3D field at the
/// shading point — so the `z` term of the offset is not weighted away here as it
/// is on a `+Z`-facing baked tap. `live_albedo_at(world + drift)` therefore
/// carries the whole of `vec3(d, 0, d·0.7)`, and a shader that had dropped the
/// cross ratio would fit the un-offset hypothesis no better and this one much
/// worse.
#[test]
fn a_scrolled_live_fragment_samples_the_field_at_the_drifted_point() {
    let Some(mut rig) = Rig::new(
        "a_scrolled_live_fragment_samples_the_field_at_the_drifted_point",
        texture::MIN_RESOLUTION,
    ) else {
        return;
    };
    let spec = rig.spec.clone();

    const PROBES: u32 = 4096;
    let offset = drift(SECONDS);
    println!("drift at t = {SECONDS}: {offset:?} m");

    let pixels = rig.frame(
        MaterialVariant::LIVE_TEX | MaterialVariant::TEXTURE_SCROLL,
        scroll_speed(),
        SECONDS,
        TWIN_HALF,
        true,
    );

    let drifted = divergence(&pixels, PROBES, |x, y| {
        spec.live_albedo_at(world_at(TWIN_HALF, x, y) + offset)
    });
    let (fit, p99, visible) = report("drifted hypothesis", &drifted);
    // One LSB of an 8-bit target is 0.0039, and the fragment and the twin are the
    // same arithmetic on the same field — the only things between them are the
    // write to the attachment and the interpolator's last bit on `world_pos`.
    assert!(
        p99 < 2.5 / 255.0 && fit > 0.95,
        "the drifted hypothesis explains only {:.1}% of probes to within 1 LSB \
         (p99 {p99:.5}) — the fragment is not sampling at `p + params.w · time.x`",
        fit * 100.0
    );
    assert!(
        visible < 0.02,
        "{:.2}% of probes disagree visibly with the drifted hypothesis, which is a wrong \
         sampling point rather than a boundary tail",
        visible * 100.0
    );

    // The discriminator. The un-drifted hypothesis has to explain a *minority* of
    // probes — the pixels it gets right are the ones where two arbitrary points
    // of the field happen to agree, which is a property of the noise and not
    // evidence for the hypothesis.
    let still = divergence(&pixels, PROBES, |x, y| {
        spec.live_albedo_at(world_at(TWIN_HALF, x, y))
    });
    let (still_fit, still_p99, _) = report("un-drifted hypothesis", &still);
    assert!(
        still_fit < 0.5 && still_p99 > 10.0 * p99.max(1.0 / 255.0),
        "the un-drifted hypothesis fits about as well as the drifted one \
         ({:.1}% vs {:.1}% within 1 LSB) — this framing is not discriminating",
        still_fit * 100.0,
        fit * 100.0
    );

    // …and the cross axis specifically. A shader that scrolled X alone would pass
    // everything above if the field happened not to vary much in Z, so the
    // hypothesis with the `0.7` term deleted is scored too and has to fit worse.
    let flat = divergence(&pixels, PROBES, |x, y| {
        spec.live_albedo_at(world_at(TWIN_HALF, x, y) + Vec3::new(offset.x, 0.0, 0.0))
    });
    let (flat_fit, _, _) = report("X-only hypothesis (no cross ratio)", &flat);
    assert!(
        flat_fit < 0.5,
        "an X-only drift explains {:.1}% of probes — `SCROLL_CROSS_SPEED` is not \
         reaching the sample point",
        flat_fit * 100.0
    );

    // …and with the bit off, the same draw is the un-drifted field. Without this
    // the file could pass on a build that had made the offset unconditional.
    let unscrolled = rig.frame(
        MaterialVariant::LIVE_TEX,
        scroll_speed(),
        SECONDS,
        TWIN_HALF,
        true,
    );
    let world = divergence(&unscrolled, PROBES, |x, y| {
        spec.live_albedo_at(world_at(TWIN_HALF, x, y))
    });
    let (off_fit, off_p99, _) = report("bit off vs un-drifted hypothesis", &world);
    assert!(
        off_fit > 0.95 && off_p99 < 2.5 / 255.0,
        "without the bit the fragment must sample where it always did ({:.1}% within \
         1 LSB, p99 {off_p99:.5})",
        off_fit * 100.0
    );
}
