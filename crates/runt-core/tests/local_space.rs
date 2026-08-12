//! `MaterialVariant::LOCAL_SPACE` — sampling a procedural texture in the
//! entity's own space instead of the world's (DESIGN §7, §5).
//!
//! # The claim, and why a variant-key test would not be one
//!
//! The bit exists to fix one artifact: a textured object that moves drags its
//! surface through a pattern nailed to the world, and the pattern slides across
//! it. `3dimenshift-runt/shift/src/model.rs`'s `spawn` doc comment is the
//! written record of that being a blocker — the port's player is a rolling ball,
//! so it was left untextured and `mottled()` (a hashed patch brightness baked
//! into vertex colours) stands in for the blotches it should have had.
//!
//! Asserting that bit 15 reaches the pipeline cache would prove none of that. So
//! the load-bearing tests here measure the artifact directly, and they do it two
//! independent ways:
//!
//!   * **Against the CPU twin, absolutely.** One frame, two hypotheses: is this
//!     pixel `TextureSpec::live_albedo_at(local)` or `live_albedo_at(world)`?
//!     Under an arbitrary in-plane rigid transform those are two completely
//!     different colours, and the answer comes out at ~1 LSB for one of them and
//!     two orders of magnitude worse for the other. That pins the *semantics*:
//!     not "the frame changed", but "the field is evaluated at the object's own
//!     coordinate, and the CPU can say what that colour is".
//!   * **Against another frame, relatively, with no CPU model in the loop.**
//!     Draw the object, move it, draw it again — and check that the second frame
//!     is the *rigid re-registration* of the first. That is what "does not slide"
//!     means as a sentence about pixels, and it needs no evaluator to be true.
//!     Two motions make it exact rather than approximate:
//!
//!       - a **translation of a whole number of pixels**, which under an
//!         orthographic camera is an exact shift of the image; and
//!       - a **quarter turn about Z**, which on a square target is an exact
//!         index permutation (`rot90`) and needs no resampling either.
//!
//!     Both are run on both texture paths, which is how "both fragment branches
//!     honour the bit" gets pixels rather than a code reading.
//!
//! # Why the quad is bigger than the viewport
//!
//! Every framing here is fully covered by the surface, so there is no silhouette
//! anywhere in the frame. That removes the one thing that would make an exact
//! image comparison a judgement call: a rasterizer's fill rule at a moved or
//! rotated edge decides differently about edge pixels, and a handful of
//! disagreeing pixels along a boundary would have to be excused by a tolerance
//! wide enough to hide a real failure.
//!
//! # What is deliberately *not* claimed
//!
//! `shader.wgsl`'s triplanar comment records that the baked path's plane
//! *weights* stay world-space under the bit. The two motions tested here are
//! exactly the ones that leave `abs(n)` alone — a translation, and a Z-spin of a
//! +Z-facing sheet — so the baked results below are exact and are not evidence
//! about a tumbling baked surface. `F_LIVE_TEX` has no triplanar projection at
//! all and is exact under any rigid transform, which is why it is the path a
//! rotating object should ask for.

use glam::{Mat4, Vec2, Vec3, Vec4};

use runt_core::draw::{DrawItem, FrameParams};
use runt_core::mesh::MeshData;
use runt_core::registry::{MeshHandle, MeshLibrary};
use runt_core::texture::{self, TextureLibrary, TextureSpec, LIVE_LOD_CELL_PIXELS};
use runt_core::{Lighting, MaterialVariant, NoopCache, Renderer};

mod common;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Square, because the quarter-turn comparison is an index permutation of a
/// square grid and nothing else.
const SIZE: u32 = 192;

/// Half-extent of the orthographic box for the **CPU-twin** framing, in metres.
///
/// Small: the live path has no mip chain and fades octaves out by pixel
/// footprint (`texture::LIVE_LOD_CELL_PIXELS`), while the CPU twin has no camera
/// and weights every octave at 1. A comparison between them is only meaningful
/// where nothing has faded, which
/// [`the_probe_framings_leave_every_octave_on`] asserts rather than assumes.
const TWIN_HALF: f32 = 1.2;

/// Half-extent of the orthographic box for the **frame-to-frame** framings.
///
/// Twenty metres rather than one, and that is about the *baked* path: the fine
/// fixture tiles every 27.8 m, so a metre-wide view of it is a few texels of an
/// almost flat image, and "the two frames differ" would be a claim about
/// bilinear noise. Forty metres across shows well over a tile. The live path's
/// octave window has faded octaves at this footprint and it does not matter
/// here — a rigid transform preserves the footprint exactly, so whatever has
/// faded has faded identically in both frames being compared.
const WIDE_HALF: f32 = 20.0;

/// The surface's half-extent. Larger than `WIDE_HALF · √2` plus the largest
/// translation below, so every framing is covered edge to edge and no frame
/// contains a silhouette (see the module docs).
const QUAD_HALF: f32 = 48.0;

/// The baked tile's resolution for the `F_TEXTURE` framings.
///
/// `texture::MIN_RESOLUTION` (64) is what the live tests bake, because the live
/// path never reads the tile. This one does read it: 512 texels across a 27.8 m
/// tile is 18 per metre against `WIDE_HALF`'s 0.21 m per pixel, so a probe is
/// comparing filtered texels rather than one mip level of mush.
const BAKE_RESOLUTION: u32 = 512;

/// How far the object slides in the translation experiment, **in pixels**.
///
/// A whole number of pixels is the whole trick: at this framing the pixel grid
/// and the world are related by an exact scale, so moving the object by
/// `SLIDE_PX` pixels' worth of metres makes the expected image an exact integer
/// shift of the original with no resampling anywhere.
const SLIDE_PX: u32 = 37;

// ---------------------------------------------------------------------------
// 1. The key space — no GPU
// ---------------------------------------------------------------------------

#[test]
fn the_local_space_bit_is_declared_implemented_and_permanent() {
    assert_eq!(MaterialVariant::LOCAL_SPACE.bits(), 1 << 15);
    assert_eq!(
        MaterialVariant::LOCAL_SPACE.unimplemented(),
        MaterialVariant::NONE,
        "the bit does something, so it must not report as reserved"
    );
    // It chooses a sampling basis and nothing else: it replaces no lighting term
    // and it moves no draw out of the opaque state-sort.
    assert!(!MaterialVariant::UNLIT.contains(MaterialVariant::LOCAL_SPACE));
    assert!(!MaterialVariant::LOCAL_SPACE.intersects(MaterialVariant::BLENDED));

    // …and it touches no fixed-function state either, which is the half
    // `two_sided.rs` had to add a field for.
    let plain = runt_core::render_state(MaterialVariant::TEXTURE);
    let local = runt_core::render_state(MaterialVariant::TEXTURE | MaterialVariant::LOCAL_SPACE);
    assert_eq!(local, plain, "a sampling basis is not a pipeline state");
}

#[test]
fn the_base_shader_really_branches_on_the_local_space_const() {
    let on = runt_core::material::variant_source(
        runt_core::material::BASE_SHADER,
        MaterialVariant::LOCAL_SPACE,
    );
    assert!(on.contains("const F_LOCAL_SPACE: bool = true;"));
    let off = runt_core::material::variant_source(
        runt_core::material::BASE_SHADER,
        MaterialVariant::NONE,
    );
    assert!(off.contains("const F_LOCAL_SPACE: bool = false;"));

    // The distinction from `TWO_SIDED`, spelled out. That bit is declared in
    // `FLAGS` and read by nothing in the WGSL — `two_sided.rs` asserts the
    // generated sources differ in one line and no more. This one is a *shader*
    // bit, so the base source must actually mention it, and the varying it
    // selects must actually be declared.
    assert!(
        runt_core::material::BASE_SHADER.contains("F_LOCAL_SPACE"),
        "the bit is a shader branch; the base source has to read the const"
    );
    assert!(
        !runt_core::material::BASE_SHADER.contains("F_TWO_SIDED"),
        "TWO_SIDED is the pipeline-only bit this is being contrasted with; if it \
         has grown a branch, the contrast in this test is stale"
    );
    assert!(
        runt_core::material::BASE_SHADER.contains("local_pos: vec3<f32>"),
        "the local basis needs a varying to arrive on"
    );
}

// ---------------------------------------------------------------------------
// The rig
// ---------------------------------------------------------------------------

/// A square in the XY plane at `z = 0`, centred on the **origin**, facing `+Z`.
///
/// Centred rather than cornered so that a rotation about Z maps the surface onto
/// itself: an off-centre quad spun a quarter turn would swing part of itself out
/// of frame and the comparison would be about coverage instead of about colour.
/// Counter-clockwise seen from `+Z`, which is where the camera is — the opaque
/// pipeline culls back faces.
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

/// Orthographic, axis-aligned on the origin, `w = 1` everywhere: pixel centre →
/// world point is an exact affine map in both directions, which is what lets
/// this file make per-pixel claims and integer-pixel translations at all.
fn view_proj(half: f32) -> Mat4 {
    glam::camera::rh::proj::directx::orthographic(-half, half, -half, half, -1.0, 1.0)
}

/// The world point at the centre of pixel `(x, y)` — the inverse of the map
/// above, spelled out rather than inverted numerically.
fn world_at(half: f32, x: u32, y: u32) -> Vec3 {
    let span = half * 2.0;
    Vec3::new(
        -half + (x as f32 + 0.5) / SIZE as f32 * span,
        // Row 0 is the top of the target and `+Y` is up, so v runs backwards.
        -half + (1.0 - (y as f32 + 0.5) / SIZE as f32) * span,
        0.0,
    )
}

/// One metre per pixel's worth of world, at a given framing.
fn metres_per_pixel(half: f32) -> f32 {
    half * 2.0 / SIZE as f32
}

/// A light rig that multiplies albedo by exactly one, for the sky behind a
/// surface that in fact covers everything. Every material below is
/// `BILLBOARD_UNLIT`, so the frame is a direct readout of albedo and no normal,
/// key light or hemisphere term can contribute to a difference between two
/// frames — the *only* thing being measured is where the texture was sampled.
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

/// One device, one mesh, one target, one bake — and as many frames as a test
/// wants off them.
///
/// Held for the whole test rather than rebuilt per frame for the reason
/// `tests/two_sided.rs`'s `Rig` gives: `Instance::new` races the platform
/// loader's process-global init, and a `Renderer` per frame SIGSEGVs inside
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
                eprintln!("SKIP local_space::{label} (no GPU adapter): {e}");
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
            label: Some("local_space target"),
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

    /// Draw the surface once, wearing `variant | BILLBOARD_UNLIT`, placed by
    /// `model` and framed by `half`. `textured` is what decides whether the
    /// baked pair is bound at all, so the "inert without a texture" claim can be
    /// made through the same path as everything else.
    fn frame(&mut self, variant: MaterialVariant, model: Mat4, half: f32, textured: bool) -> Vec<u8> {
        let draws = [DrawItem {
            entity: bevy_ecs::entity::Entity::from_raw_u32(0).expect("entity 0"),
            variant: variant | MaterialVariant::BILLBOARD_UNLIT,
            mesh: self.mesh,
            model,
            base_color: Vec4::ONE,
            params: Vec4::ZERO,
            texture: if textured { Some(self.handle) } else { None },
        }];
        self.renderer.set_render_clock(0.0, 0.0);
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
        label: Some("local_space readback"),
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

/// Sorted per-pixel max-channel errors against a colour the *caller* predicts,
/// over the whole frame on a coprime stride — the reporting shape
/// `tests/live_texture.rs` uses, generalized over the hypothesis being tested.
fn divergence(pixels: &[u8], samples: u32, mut want: impl FnMut(u32, u32) -> Vec3) -> Vec<f32> {
    let mut errors = Vec::with_capacity(samples as usize);
    for i in 0..samples {
        // Coprime strides, so the scatter walks the whole target.
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

/// How well a hypothesis fitted, as `(fraction within one LSB, p99, max)` of a
/// sorted error list — printed, because the numbers are the evidence.
///
/// **The fit fraction is the load-bearing statistic and the median is not.** The
/// field is `CellValue` noise through a ramp, so a wrong sampling point is wrong
/// by however much two arbitrary points of that field differ — which is often
/// very little and occasionally a fifth of the ramp. A median therefore
/// under-reports a hypothesis that is *completely* wrong. "What share of probes
/// the hypothesis explains to the last bit the 8-bit target can hold" does not:
/// the right basis explains essentially all of them and a wrong one explains
/// only the pixels where the field happens to agree with itself.
/// The third number is the share of probes past **4 LSB** — a visible
/// disagreement — and it is deliberately reported instead of the maximum. A max
/// is a one-sample statistic on a step function: `CellValue` is decided by which
/// lattice cell won, so a single probe within a float of a boundary can land
/// either side and move by a whole octave's share of the ramp. Whether that
/// happens once in four thousand probes or four hundred times is the difference
/// between a rounding tail and a wrong basis, and only the *count* can tell them
/// apart.
fn report(label: &str, errors: &[f32]) -> (f32, f32, f32) {
    const LSB: f32 = 1.0 / 255.0;
    let share = |t: f32| errors.iter().filter(|e| **e > t).count() as f32 / errors.len() as f32;
    let fit = 1.0 - share(LSB);
    let p99 = errors[errors.len() * 99 / 100];
    let visible = share(4.0 * LSB);
    println!(
        "{label}: {} probes — {:.2}% within 1 LSB, median {:.2} LSB, p99 {:.5} ({:.2} LSB), \
         {:.2}% past 4 LSB, max {:.5}",
        errors.len(),
        fit * 100.0,
        errors[errors.len() / 2] * 255.0,
        p99,
        p99 * 255.0,
        visible * 100.0,
        errors.last().expect("sampled something")
    );
    (fit, p99, visible)
}

// ---------------------------------------------------------------------------
// 2. The precondition the CPU-twin comparison rests on — no GPU
// ---------------------------------------------------------------------------

#[test]
fn the_probe_framings_leave_every_octave_on() {
    // `TWIN_HALF`'s own assumption, as an assertion rather than a comment: at
    // that framing the live octave window has faded nothing, so the CPU twin —
    // which has no camera and weights every octave at 1 — is the right thing to
    // compare a pixel against. If `TWIN_HALF` or `SIZE` ever moves far enough to
    // break this, the divergence test would start failing for a reason that has
    // nothing to do with the sampling basis.
    let spec = common::fine();
    let footprint = metres_per_pixel(TWIN_HALF) * spec.live_cells_per_metre();
    let (min, _) = spec.live_octave_window(footprint, LIVE_LOD_CELL_PIXELS);
    let last = spec.octave_plan().len() as f32 - 1.0;
    println!(
        "twin framing: {:.5} m/px → {footprint:.5} cells/px → window floor {min:.2}, \
         {} octaves",
        metres_per_pixel(TWIN_HALF),
        spec.octave_plan().len()
    );
    assert!(
        min >= last,
        "octave {last} is already fading at the twin framing ({min:.2} < {last})"
    );

    // And `WIDE_HALF`'s: it is *expected* to have faded something, which is why
    // no test compares it to the twin. Stated so that the two framings cannot be
    // mixed up later by someone reusing the wrong constant.
    let wide = metres_per_pixel(WIDE_HALF) * spec.live_cells_per_metre();
    let (wide_min, _) = spec.live_octave_window(wide, LIVE_LOD_CELL_PIXELS);
    println!("wide framing: {wide:.5} cells/px → window floor {wide_min:.2}");
    assert!(
        wide_min < last,
        "the wide framing now resolves every octave too; the frame-to-frame tests \
         still hold, but the comment explaining why they do not need the twin is stale"
    );
}

// ---------------------------------------------------------------------------
// 3. The semantics, against the CPU twin
// ---------------------------------------------------------------------------

/// The strongest form of the claim: one rendered frame, two hypotheses, and the
/// CPU says which one the GPU drew.
///
/// For each placement the same frame is scored twice — against
/// `live_albedo_at(model⁻¹ · world)` and against `live_albedo_at(world)`. Under
/// an in-plane rigid transform those predict two unrelated colours per pixel, so
/// this is not a tolerance question: one of them lands inside the LSB noise the
/// 8-bit target imposes and the other is off by a fraction of the ramp's whole
/// range.
///
/// The scaled placement is here because the scale decision is a documented one
/// (`shader.wgsl`'s `p_source`): feature size follows the object's own units, so
/// a 0.5× entity samples the field at twice the coordinate and gets half-size
/// features. `model⁻¹` carries that automatically, which means this test would
/// fail if the shader ever started normalizing the scale out — the intended
/// behaviour is pinned rather than merely commented.
///
/// Rotations are about **Z** only, so the surface stays in the `z = 0` plane and
/// `world_at` remains the exact inverse of the projection. A tumbling quad would
/// need a ray/plane intersection to say where a pixel is, and that arithmetic
/// would then be part of what is under test.
#[test]
fn a_local_space_fragment_samples_the_field_at_the_objects_own_coordinate() {
    let Some(mut rig) = Rig::new(
        "a_local_space_fragment_samples_the_field_at_the_objects_own_coordinate",
        texture::MIN_RESOLUTION,
    ) else {
        return;
    };
    let spec = rig.spec.clone();

    const PROBES: u32 = 4096;
    // The displacements are metres, and they are several of them for a reason:
    // the fine fixture puts six lattice cells across a 27.8 m tile, so its base
    // octave — which carries most of the fBm — has a 4.6 m feature. A placement
    // that moved the surface by a fraction of that would leave the world
    // hypothesis *nearly* right, and the two hypotheses would then be separated
    // by a couple of LSB rather than by a fifth of the ramp. Each translation
    // below is of the order of one base cell, which is what makes "which of these
    // two colours is it" a question with an unambiguous answer.
    let placements: [(&str, Mat4); 4] = [
        ("identity", Mat4::IDENTITY),
        (
            "translated",
            Mat4::from_translation(Vec3::new(7.3, -4.1, 0.0)),
        ),
        (
            "spun",
            Mat4::from_translation(Vec3::new(5.1, 3.7, 0.0)) * Mat4::from_rotation_z(0.7),
        ),
        (
            "spun and halved",
            Mat4::from_translation(Vec3::new(-6.2, 4.9, 0.0))
                * Mat4::from_rotation_z(-1.1)
                * Mat4::from_scale(Vec3::splat(0.5)),
        ),
    ];

    for (name, model) in placements {
        let inverse = model.inverse();
        let pixels = rig.frame(
            MaterialVariant::LIVE_TEX | MaterialVariant::LOCAL_SPACE,
            model,
            TWIN_HALF,
            true,
        );

        let local = divergence(&pixels, PROBES, |x, y| {
            spec.live_albedo_at(inverse.transform_point3(world_at(TWIN_HALF, x, y)))
        });
        let (local_fit, local_p99, local_visible) =
            report(&format!("{name} vs local hypothesis"), &local);

        // One LSB of an 8-bit target is 0.0039, and the fragment and the twin are
        // the same arithmetic on the same field: the only things between them are
        // the write to the attachment and the interpolator's last bit on
        // `local_pos`. So essentially every probe has to land inside a couple of
        // LSB, and the handful that do not have to be *few* — a flip on a lattice
        // boundary, which the scaled placement sees more of because its local
        // coordinate is twice as large and its `model.inverse()` has a scale's
        // worth of extra rounding in it.
        assert!(
            local_p99 < 2.5 / 255.0 && local_fit > 0.95,
            "{name}: the local hypothesis explains only {:.1}% of probes to within 1 LSB \
             (p99 {local_p99:.5}) — the fragment is not sampling at the object's own \
             coordinate",
            local_fit * 100.0
        );
        assert!(
            local_visible < 0.02,
            "{name}: {:.2}% of probes disagree visibly with the local hypothesis, which is \
             a wrong sampling point somewhere rather than a boundary tail",
            local_visible * 100.0
        );

        if name == "identity" {
            // The one placement where the two hypotheses are the same claim, so
            // there is nothing to discriminate and asserting a difference would
            // be asserting a coincidence. It is in the list because it is the
            // control: it says `local_pos` arrives intact and the local path
            // agrees with the world path where they must.
            continue;
        }

        let world =
            divergence(&pixels, PROBES, |x, y| spec.live_albedo_at(world_at(TWIN_HALF, x, y)));
        let (world_fit, world_p99, _) = report(&format!("{name} vs world hypothesis"), &world);
        // The discriminator, on both statistics. The world hypothesis must
        // explain a *minority* of probes — the pixels it gets right are the ones
        // where two arbitrary points of the field happen to agree, and that is a
        // property of the noise rather than evidence for the hypothesis. And its
        // tail must be an order of magnitude past the local fit's, which is what
        // stops "both hypotheses are bad" from passing.
        assert!(
            world_fit < 0.5 && world_p99 > 10.0 * local_p99.max(1.0 / 255.0),
            "{name}: the world hypothesis fits about as well as the local one \
             ({:.1}% vs {:.1}% within 1 LSB, p99 {world_p99:.5} vs {local_p99:.5}) — this \
             placement is not actually discriminating, so the test proves nothing",
            world_fit * 100.0,
            local_fit * 100.0
        );
    }

    // …and with the bit *off*, the same non-identity placement is the world
    // hypothesis instead. Without this the whole file could pass on a build that
    // had made local sampling unconditional, which would silently re-key every
    // textured surface in the port.
    let model = Mat4::from_translation(Vec3::new(7.3, -4.1, 0.0)) * Mat4::from_rotation_z(0.7);
    let pixels = rig.frame(MaterialVariant::LIVE_TEX, model, TWIN_HALF, true);
    let world = divergence(&pixels, PROBES, |x, y| {
        spec.live_albedo_at(world_at(TWIN_HALF, x, y))
    });
    let (fit, p99, max) = report("bit off vs world hypothesis", &world);
    assert!(
        fit > 0.95 && p99 < 2.5 / 255.0 && max < 0.06,
        "without the bit the fragment must still sample in world space ({:.1}% within \
         1 LSB, p99 {p99:.5}, max {max:.5})",
        fit * 100.0
    );
}

// ---------------------------------------------------------------------------
// 4. The artifact, measured between two frames
// ---------------------------------------------------------------------------

/// Where pixel `(x, y)` of the resting frame ends up when the object is turned a
/// quarter turn about `+Z`.
///
/// Derived rather than guessed: the camera puts world `(X, Y)` at column
/// `(X/2h + 0.5)·S` and row `(0.5 − Y/2h)·S`, and a `+90°` turn about Z sends a
/// surface point from `(X, Y)` to `(−Y, X)`. Substituting one into the other
/// gives `(col, row) → (row, S − 1 − col)`, which is an exact permutation of the
/// grid — no resampling, no interpolation, nothing to excuse with a tolerance.
fn rot90(x: u32, y: u32) -> (u32, u32) {
    (y, SIZE - 1 - x)
}

/// How much two frames disagree over a region, as `(mean, fraction over 4 LSB)`.
///
/// Two numbers because either alone can be fooled. A mean can be dragged down by
/// a large area that happens to be flat; a count of differing pixels says nothing
/// about how far. Together they separate "the same image" from "a different one"
/// without a golden anything.
fn disagreement(a: &[u8], b: &[u8], mut map: impl FnMut(u32, u32) -> Option<(u32, u32)>) -> (f32, f32) {
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

/// The headline claim, on both texture paths: **a moving object carries its
/// pattern with it under the bit, and slides through a fixed one without it.**
///
/// No CPU model of the field appears anywhere in this test. It draws the surface
/// three times — at rest, translated by a whole number of pixels, and turned a
/// quarter turn — and asks whether the moved frames are the rigid
/// re-registration of the resting one. That is the artifact stated as a fact
/// about pixels, which is the form it was reported in.
///
/// Both halves are load-bearing. "The registered frames match with the bit" alone
/// would pass on a build where the texture had quietly become a flat colour;
/// "they do not match without it" alone would pass on a build where the bit did
/// something arbitrary.
#[test]
fn a_moving_object_carries_its_pattern_and_a_world_space_one_slides_through_it() {
    let Some(mut rig) = Rig::new(
        "a_moving_object_carries_its_pattern_and_a_world_space_one_slides_through_it",
        BAKE_RESOLUTION,
    ) else {
        return;
    };

    let slide = SLIDE_PX as f32 * metres_per_pixel(WIDE_HALF);
    let translated = Mat4::from_translation(Vec3::new(slide, 0.0, 0.0));
    // Exactly a quarter turn, so the permutation above is exact. `Mat4::from_
    // rotation_z(FRAC_PI_2)` leaves a `cos` of −4.4e−8 in the basis rather than a
    // clean zero, which moves a surface point by under a micrometre over
    // `QUAD_HALF` — four orders of magnitude below one pixel of `WIDE_HALF`.
    let spun = Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2);

    // Column `x` of the translated frame shows what column `x − SLIDE_PX` of the
    // resting frame showed, so the comparison runs over the overlap and the
    // `SLIDE_PX` columns that slid in from off-frame are skipped.
    let shifted = |x: u32, y: u32| (x + SLIDE_PX < SIZE).then(|| (x + SLIDE_PX, y));

    for (path, bit) in [
        ("live", MaterialVariant::LIVE_TEX),
        ("baked", MaterialVariant::TEXTURE),
    ] {
        let rest_local = rig.frame(bit | MaterialVariant::LOCAL_SPACE, Mat4::IDENTITY, WIDE_HALF, true);
        let rest_world = rig.frame(bit, Mat4::IDENTITY, WIDE_HALF, true);

        // A frame with nothing in it would make every registration below pass
        // trivially, so the *content* is asserted first: the resting frame has to
        // be a texture rather than a flat wash. Measured as the frame against
        // itself under the quarter-turn permutation, which is exactly the
        // "different image" scale the assertions below are calibrated in.
        let (self_mean, self_over) = disagreement(&rest_local, &rest_local, |x, y| Some(rot90(x, y)));
        println!("{path}: resting frame vs its own quarter turn — mean {self_mean:.5}, over {self_over:.3}");
        assert!(
            self_mean > 0.02 && self_over > 0.5,
            "{path}: the resting frame is too uniform for any registration test to \
             mean anything (mean {self_mean:.5}, {:.1}% over 4 LSB)",
            self_over * 100.0
        );

        for (motion, model, map) in [
            (
                "translated",
                translated,
                &shifted as &dyn Fn(u32, u32) -> Option<(u32, u32)>,
            ),
            ("spun", spun, &|x, y| Some(rot90(x, y))),
        ] {
            let moved_local = rig.frame(bit | MaterialVariant::LOCAL_SPACE, model, WIDE_HALF, true);
            let (local_mean, local_over) = disagreement(&rest_local, &moved_local, map);
            println!(
                "{path} {motion} LOCAL_SPACE: mean {local_mean:.5} ({:.2} LSB), \
                 {:.3}% of probes over 4 LSB",
                local_mean * 255.0,
                local_over * 100.0
            );
            // Two LSB of mean, and under a percent of pixels past four. Not zero,
            // and the residue is named: `CellValue` is a step function of which
            // lattice cell won, so a fragment within a float of a boundary can
            // land either side once its coordinate has been through a rotation
            // matrix, and the baked path additionally re-selects a mip level from
            // gradients that have been rotated. Both are single-pixel events on a
            // boundary, which is what the over-4-LSB *fraction* is there to bound
            // — a real slide moves the whole image, not a scatter of pixels.
            assert!(
                local_mean < 1.0 / 255.0,
                "{path} {motion}: the registered frames differ by {:.2} LSB on average — \
                 the pattern did not travel with the object, which is the whole claim",
                local_mean * 255.0
            );
            assert!(
                local_over < 0.01,
                "{path} {motion}: {:.1}% of pixels disagree by more than 4 LSB, which is \
                 an image that changed rather than a boundary tail",
                local_over * 100.0
            );

            let moved_world = rig.frame(bit, model, WIDE_HALF, true);
            let (world_mean, world_over) = disagreement(&rest_world, &moved_world, map);
            println!(
                "{path} {motion} world space: mean {world_mean:.5} ({:.2} LSB), \
                 {:.3}% of probes over 4 LSB",
                world_mean * 255.0,
                world_over * 100.0
            );
            // The artifact itself. Without the bit the pattern stayed in the
            // world and the surface travelled through it, so the two registered
            // frames are two different images.
            //
            // "Different" is calibrated against the frame's *own* content rather
            // than against a hardcoded number, which is what makes the bound
            // meaningful on a blurrier path: `self_mean` above is this same
            // statistic between the resting frame and an unrelated registration of
            // itself, i.e. the scale on which "these are two different images"
            // measures. World-space sampling has to reach a third of that. The
            // fraction past 4 LSB is the discriminator that needs no calibration
            // at all — under a percent with the bit, more than half of the frame
            // without it.
            assert!(
                world_over > 0.4 && world_mean > 0.3 * self_mean,
                "{path} {motion}: world-space sampling produced the *same* registered \
                 image (mean {world_mean:.5} against an unrelated-registration scale of \
                 {self_mean:.5}, {:.1}% over 4 LSB) — either the bit is being ignored or \
                 this motion cannot show the artifact",
                world_over * 100.0
            );
        }
    }
}

/// Two compatibility claims that have to hold or the bit is not free to add to a
/// key space every scene file and pipeline cache already depends on.
///
/// 1. **With an identity model matrix the two bases are one basis**, so the bit
///    cannot change a pixel. That is also the narrowest possible check that
///    `local_pos` is wired up at all: a varying left at zero, or fed the wrong
///    attribute, would make this frame a flat colour.
/// 2. **Without a texture the bit is inert.** It selects where a texture is
///    sampled, and a draw that samples none has nothing for it to select — the
///    same shape of claim `textures_scene.rs` makes about `NORMAL_MAP`.
#[test]
fn the_bit_changes_no_pixel_where_the_two_bases_agree() {
    let Some(mut rig) = Rig::new(
        "the_bit_changes_no_pixel_where_the_two_bases_agree",
        texture::MIN_RESOLUTION,
    ) else {
        return;
    };

    let world = rig.frame(MaterialVariant::LIVE_TEX, Mat4::IDENTITY, TWIN_HALF, true);
    let local = rig.frame(
        MaterialVariant::LIVE_TEX | MaterialVariant::LOCAL_SPACE,
        Mat4::IDENTITY,
        TWIN_HALF,
        true,
    );
    assert_eq!(
        world, local,
        "at the identity the local basis *is* the world basis; a differing byte \
         means the varying carries something other than the pre-transform position"
    );
    // …and it is a texture, not a wash, so the equality above is not the equality
    // of two blank frames.
    assert!(
        local.chunks_exact(4).map(|p| p[1]).max() > local.chunks_exact(4).map(|p| p[1]).min(),
        "the probe frame has no texture in it"
    );

    let untextured = rig.frame(MaterialVariant::NONE, Mat4::from_rotation_z(0.7), TWIN_HALF, false);
    let untextured_local = rig.frame(
        MaterialVariant::LOCAL_SPACE,
        Mat4::from_rotation_z(0.7),
        TWIN_HALF,
        false,
    );
    assert_eq!(
        untextured, untextured_local,
        "the bit chooses where a texture is sampled; with no texture bound it must \
         be inert, however the object is placed"
    );
}
