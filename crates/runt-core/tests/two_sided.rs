//! `MaterialVariant::TWO_SIDED` — the bit that turns backface culling off for
//! one variant's pipeline (DESIGN §5), and a port of
//! `3dimenshift/shaders/fx/water.gdshader:2`'s `cull_disabled`.
//!
//! It is the first bit in the key space that changes **only** the pipeline
//! descriptor: `TRANSPARENT`, `ADDITIVE` and `DEPTH_GREATER` are fixed-function
//! too, but they move blend and depth state, and `render_state`'s table already
//! carried a field for each. This one adds `cull`, and the reason it needs
//! pixels rather than a table assertion is that a cull mode is invisible to
//! every layer above the rasterizer — nothing on the CPU, nothing in the WGSL,
//! and nothing in the cache can tell you whether a triangle facing away
//! actually got drawn.
//!
//! Three claims:
//!
//! 1. **The key space knows about it**, with the permanent bit number and the
//!    `const` the preprocessor emits — no device needed. It is *not* a shader
//!    branch, and the source it generates proves that by being byte-identical
//!    to the source without it.
//! 2. **A surface seen from behind is gone without the bit and there with it.**
//!    The same mesh, the same material, the same camera — one bit apart.
//! 3. **From the front the bit changes nothing, byte for byte.** Culling only
//!    ever removes triangles the camera is behind, so a front-facing draw must
//!    be pixel-identical with the bit set. That is the in-repo form of "without
//!    the bit, frames are what they were": the no-GPU half — that every key
//!    lacking the bit still asks for `Face::Back` — lives in
//!    `tests/material_variants.rs`.

use glam::{Mat4, Vec3, Vec4};

use bevy_ecs::prelude::World;
use runt_core::draw::build_draw_list;
use runt_core::ecs::Lighting;
use runt_core::registry::{MeshHandle, MeshLibrary};
use runt_core::texture::TextureLibrary;
use runt_core::{Material, MaterialVariant, Mesh, MeshRef, Renderer, Transform};

const WIDTH: u32 = 128;
const HEIGHT: u32 = 128;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Half-extent of the orthographic box. The quad is 2 × 2, so it covers the
/// middle quarter of the target and leaves a background margin on every side —
/// which is what makes "empty" and "covered" a count rather than a judgement.
const ORTHO_HALF: f32 = 2.0;

// ---------------------------------------------------------------------------
// 1. The key space — no GPU
// ---------------------------------------------------------------------------

#[test]
fn the_two_sided_bit_is_declared_implemented_and_permanent() {
    assert_eq!(MaterialVariant::TWO_SIDED.bits(), 1 << 13);
    assert_eq!(
        MaterialVariant::TWO_SIDED.unimplemented(),
        MaterialVariant::NONE,
        "the bit does something, so it must not report as reserved"
    );
    // It replaces no lighting term and moves no draw out of the opaque
    // state-sort: a two-sided surface is scheduled and shaded exactly as its
    // other bits say.
    assert!(!MaterialVariant::UNLIT.contains(MaterialVariant::TWO_SIDED));
    assert!(!MaterialVariant::TWO_SIDED.intersects(MaterialVariant::BLENDED));

    // The preprocessor declares it like every other flag…
    let on = runt_core::material::variant_source(
        runt_core::material::BASE_SHADER,
        MaterialVariant::TWO_SIDED,
    );
    assert!(on.contains("const F_TWO_SIDED: bool = true;"));
    let off =
        runt_core::material::variant_source(runt_core::material::BASE_SHADER, MaterialVariant::NONE);
    assert!(off.contains("const F_TWO_SIDED: bool = false;"));

    // …and that const is the *only* difference, because no branch reads it.
    // A shader-only assumption anywhere would show up here as two sources that
    // differ in more than one line.
    let differing = on
        .lines()
        .zip(off.lines())
        .filter(|(a, b)| a != b)
        .collect::<Vec<_>>();
    assert_eq!(
        differing,
        vec![(
            "const F_TWO_SIDED: bool = true;",
            "const F_TWO_SIDED: bool = false;"
        )],
        "TWO_SIDED is a pipeline bit; it must not change the generated WGSL"
    );
}

// ---------------------------------------------------------------------------
// 2–3. The pixels
// ---------------------------------------------------------------------------

/// A unit quad in the XY plane, wound counter-clockwise as seen from **+Z**.
/// Its front face is therefore towards the default camera and away from the
/// rotated one, and that is the whole of the experiment.
fn quad() -> Mesh {
    Mesh {
        positions: vec![
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(-1.0, 1.0, 0.0),
        ],
        normals: vec![Vec3::Z; 4],
        uvs: vec![glam::Vec2::ZERO; 4],
        colors: vec![Vec3::ONE; 4],
        indices: vec![0, 1, 2, 0, 2, 3],
    }
}

/// White on black: the lighting rig is flattened so the sky pass paints one
/// colour, and the quad is `BILLBOARD_UNLIT` so it is exactly `base_color`.
/// Unlit also means the answer cannot come from a normal that happens to face
/// the wrong way — the *only* thing being measured is whether the triangle was
/// rasterized.
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

fn ortho() -> Mat4 {
    glam::camera::rh::proj::directx::orthographic(
        -ORTHO_HALF,
        ORTHO_HALF,
        -ORTHO_HALF,
        ORTHO_HALF,
        -10.0,
        10.0,
    )
}

/// Looking at the quad's front.
fn from_front() -> Mat4 {
    ortho()
}

/// The same camera walked round to the other side: half a turn about Y, which
/// reverses the quad's winding in NDC and is exactly what a back face is.
fn from_behind() -> Mat4 {
    ortho() * Mat4::from_rotation_y(std::f32::consts::PI)
}

/// One device, one mesh, one target — and as many frames as a test wants off
/// them. See `tests/vertex_wave.rs`'s `Rig` for why this is a struct held for
/// the whole test rather than a `Renderer` per frame: `Instance::new` races the
/// platform loader's process-global init, and building one per frame SIGSEGVs in
/// `libvulkan.so` about one run in three.
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
                eprintln!("SKIP two_sided (no GPU adapter): {e}");
                return None;
            }
        };

        let mesh = quad();
        let handle = renderer.register_mesh(&mesh);
        assert_eq!(handle, MeshHandle::of(&mesh));

        let target = renderer.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("two_sided target"),
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

    /// Draw the quad once. `variant` is ORed onto `BILLBOARD_UNLIT`; the target
    /// is reused between frames because the pass clears it.
    fn frame(&mut self, variant: MaterialVariant, view_proj: Mat4) -> Frame {
        let mut world = World::new();
        world.spawn((
            MeshRef(self.handle),
            Material {
                base_color: Vec4::ONE,
                params: Vec4::ZERO,
                texture: None,
                variant: variant | MaterialVariant::BILLBOARD_UNLIT,
            },
            Transform::IDENTITY,
        ));
        let draws = build_draw_list(&mut world, 0.0);

        let frame = runt_core::FrameParams {
            view_proj,
            lighting: dark_sky(),
        };
        self.renderer.set_render_clock(0.0, 0.0);
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

struct Frame {
    pixels: Vec<u8>,
}

impl Frame {
    /// Pixels brighter than the black background.
    fn lit(&self) -> usize {
        self.pixels.chunks_exact(4).filter(|px| px[0] > 128).count()
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

/// The claim, in one test and on one device: seen from behind, the plain
/// variant is nothing and the two-sided one is the whole quad.
///
/// Both halves are needed. "Two-sided is visible" alone would pass on a build
/// where culling was off for everything, and "plain is empty" alone would pass
/// on one where the quad had simply missed the target.
#[test]
fn a_back_face_is_gone_without_the_bit_and_there_with_it() {
    let Some(mut rig) = Rig::new() else { return };

    // A quarter of a 128² target, less the half-open edge: the exact number is
    // the rasterizer's business, so the assertions bracket it generously.
    let front = rig.frame(MaterialVariant::NONE, from_front()).lit();
    assert!(
        front > WIDTH as usize * HEIGHT as usize / 8,
        "the quad is not on screen from the front at all ({front} lit pixels)"
    );

    let culled = rig.frame(MaterialVariant::NONE, from_behind()).lit();
    assert_eq!(
        culled, 0,
        "a back face survived without TWO_SIDED — culling is off for every key"
    );

    let two_sided = rig
        .frame(MaterialVariant::TWO_SIDED, from_behind())
        .lit();
    assert_eq!(
        two_sided, front,
        "TWO_SIDED from behind must cover exactly what the front view covers — \
         the quad is symmetric about Y, so half a turn maps it onto itself"
    );
}

/// From the front, the bit is inert: culling only ever drops triangles the
/// camera is behind, and there are none.
#[test]
fn from_the_front_the_bit_changes_no_pixel() {
    let Some(mut rig) = Rig::new() else { return };
    let plain = rig.frame(MaterialVariant::NONE, from_front());
    let two_sided = rig.frame(MaterialVariant::TWO_SIDED, from_front());
    assert_eq!(
        plain.pixels, two_sided.pixels,
        "the bit changed something other than the cull mode"
    );
}
