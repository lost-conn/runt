//! The key light's shadow map (DESIGN §5, §11; `src/shadow.rs`).
//!
//! Four claims, in ascending order of cost — the same ladder
//! `tests/render_scale.rs` climbs, because the feature makes the same promise:
//! a capability gate whose *off* is not "cheap" but "nothing".
//!
//! 1. **The arithmetic.** The tier table, and the texel snap: the light box is
//!    a pure function of (camera, light, extent, resolution), so "a sub-texel
//!    camera pan does not move the sampling grid" is a matrix equality rather
//!    than a screenshot. No GPU.
//! 2. **WebGL2 survives it.** The depth pass and the lit material variant —
//!    which now carries the comparison sample — have to make the WGSL →
//!    GLSL-ES 3.00 crossing, not merely compile here. `sampler2DShadow` with
//!    hardware PCF is core ES 3.00; this is what says the emitted code stays
//!    inside it.
//! 3. **Off is byte-identical and allocates nothing** — including *after* the
//!    gate has been open, which is the case a sticky allocation could quietly
//!    break. Every pinned frame hash in the suite rests on this.
//! 4. **On actually shadows.** An occluder darkens its receiver; the darkness
//!    is the key light's absence and not the ambient's (a shadowed floor keeps
//!    the hemisphere's sky colour — the two-colour ambient is the look, and a
//!    shadow that killed it would read as a hole); a receiver in the open is
//!    untouched to the byte; and a bare plane under the light shows **no
//!    acne** at either tier, which is the biases' whole job
//!    ([`ShadowSettings`]' defaults are tuned against exactly this).

use bevy_ecs::prelude::World;
use glam::{Mat4, Vec2, Vec3, Vec4};
use runt_core::draw::build_draw_list;
use runt_core::ecs::Lighting;
use runt_core::registry::{MeshHandle, MeshLibrary};
use runt_core::texture::TextureLibrary;
use runt_core::{
    shadow, Camera, Engine, Material, MaterialVariant, Mesh, MeshRef, Renderer, ShadowQuality,
    ShadowSettings, Transform,
};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

// ---------------------------------------------------------------------------
// 1. The arithmetic — no GPU
// ---------------------------------------------------------------------------

#[test]
fn the_gate_is_off_by_default_and_maps_tiers_to_resolutions() {
    // §11's row, verbatim: off / 512² single cascade / 2048².
    assert_eq!(ShadowQuality::default(), ShadowQuality::Off);
    assert_eq!(ShadowQuality::Off.resolution(), None);
    assert_eq!(ShadowQuality::Low.resolution(), Some(512));
    assert_eq!(ShadowQuality::High.resolution(), Some(2048));

    // The settings' defaults are the tuned ones the GPU half of this file
    // validates; a drive-by "improvement" here should have to re-earn them.
    let s = ShadowSettings::default();
    assert_eq!(
        (s.extent, s.bias, s.slope_bias, s.fade),
        (20.0, 0.0015, 0.004, 0.1)
    );
}

#[test]
fn the_gate_promotes_lit_variants_and_only_lit_variants() {
    use runt_core::resolve_shadow_variant;

    // The gate open: a lit key gains SHADOW, whatever else it carries…
    let lit = MaterialVariant::VERTEX_COLOR | MaterialVariant::TEXTURE;
    assert_eq!(
        resolve_shadow_variant(lit, true),
        lit | MaterialVariant::SHADOW
    );
    assert_eq!(
        resolve_shadow_variant(MaterialVariant::NONE, true),
        MaterialVariant::SHADOW
    );

    // …and an unlit key does not: its fragment path replaces the lighting
    // term the shadow would scale, so the bit would only compile a twin
    // pipeline that draws identical pixels.
    for unlit in [
        MaterialVariant::BILLBOARD_UNLIT,
        MaterialVariant::FRESNEL | MaterialVariant::ADDITIVE,
        MaterialVariant::EMISSIVE_SWEEP | MaterialVariant::VERTEX_COLOR,
    ] {
        assert_eq!(resolve_shadow_variant(unlit, true), unlit);
    }

    // The gate closed: every key passes through *unchanged* — the same key is
    // the same cached pipeline, which is the byte-identity guarantee's spine.
    for variant in [lit, MaterialVariant::NONE, MaterialVariant::BILLBOARD_UNLIT] {
        assert_eq!(resolve_shadow_variant(variant, false), variant);
    }
}

/// A camera the demo scene could plausibly hold: perspective, above and behind
/// the focus, looking down at it.
fn camera_view_proj(eye: Vec3, focus: Vec3) -> Mat4 {
    Camera::default().view_proj(Transform::looking_at(eye, focus, Vec3::Y).matrix(), 16.0 / 9.0)
}

#[test]
fn the_light_box_is_snapped_to_whole_texels() {
    // A vertical key light, so world XZ is the map's plane and the arithmetic
    // below is legible: texel = 2·extent ÷ resolution.
    let key = Vec3::Y;
    let (extent, resolution) = (20.0, 512);
    let texel = 2.0 * extent / resolution as f32;

    let at = |eye: Vec3, focus: Vec3| {
        shadow::light_view_proj(&camera_view_proj(eye, focus), key, extent, resolution)
    };

    // A sub-texel camera pan changes the light matrix not at all: the box's
    // light-space translation rounded to the same texel, so the same world
    // renders into the same grid — which is the entire anti-shimmer claim.
    let base = at(Vec3::new(0.0, 10.0, 8.0), Vec3::ZERO);
    let nudge = Vec3::new(0.3 * texel, 0.0, 0.0);
    let panned = at(Vec3::new(0.0, 10.0, 8.0) + nudge, nudge);
    assert_eq!(base, panned, "a sub-texel pan moved the shadow grid");

    // A whole-texel pan shifts the grid by exactly one texel — 2/resolution in
    // NDC — not by "about one": rounding must not accumulate drift.
    let step = Vec3::new(texel, 0.0, 0.0);
    let stepped = at(Vec3::new(0.0, 10.0, 8.0) + step, step);
    let probe = Vec4::new(3.7, 0.0, -2.2, 1.0);
    let a = (base * probe).truncate();
    let b = (stepped * probe).truncate();
    let shift = (Vec2::new(a.x, a.y) - Vec2::new(b.x, b.y)).length();
    assert!(
        (shift - 2.0 / resolution as f32).abs() < 1.0e-5,
        "a one-texel pan shifted the grid by {shift}, not one texel"
    );
    assert!((a.z - b.z).abs() < 1.0e-6, "a horizontal pan moved light depth");

    // The grid is pinned to the *world*: any fixed world point lands a whole
    // number of texels from the map's corner, whatever the camera does.
    let t = (base * Vec4::new(0.0, 0.0, 0.0, 1.0)).truncate();
    for axis in [t.x, t.y] {
        let texels = (axis * 0.5 + 0.5) * resolution as f32;
        assert!(
            (texels - texels.round()).abs() < 1.0e-2,
            "world origin sits {texels} texels in — not on the grid"
        );
    }
}

#[test]
fn a_degenerate_camera_still_yields_a_finite_matrix() {
    // The no-camera path hands the renderer an identity view-projection, and a
    // broken transform can hand it NaNs; the light matrix must degrade to
    // *somewhere finite* rather than poison the frame block.
    let identity = shadow::light_view_proj(&Mat4::IDENTITY, Vec3::Y, 20.0, 512);
    assert!(identity.is_finite());

    let nan = shadow::light_view_proj(&(Mat4::IDENTITY * f32::NAN), Vec3::Y, 20.0, 512);
    assert!(nan.is_finite(), "a NaN camera leaked into the light matrix");

    // A zero light direction (an author dragging a slider through the origin)
    // falls back rather than normalizing to NaN, and a nonsense extent
    // (reflected write gone wrong) is floored.
    assert!(shadow::light_view_proj(&Mat4::IDENTITY, Vec3::ZERO, 20.0, 512).is_finite());
    assert!(shadow::light_view_proj(&Mat4::IDENTITY, Vec3::Y, f32::NAN, 512).is_finite());
    assert!(shadow::light_view_proj(&Mat4::IDENTITY, Vec3::Y, -3.0, 0).is_finite());
}

// ---------------------------------------------------------------------------
// 2. WebGL2: the new shader code has to survive translation
// ---------------------------------------------------------------------------

#[test]
fn the_shadow_pass_and_the_lit_variant_translate_to_glsl_es_for_webgl2() {
    // The depth-only pass: trivial WGSL, but it is the one pipeline in the
    // engine with an empty fragment stage, which is its own translation shape.
    glsl_es(
        runt_core::SHADOW_SHADER,
        &[
            ("vs_shadow", naga::ShaderStage::Vertex),
            ("fs_shadow", naga::ShaderStage::Fragment),
        ],
    );

    // The SHADOW variants carry `textureSampleCompareLevel` on a depth
    // texture — GLSL's `sampler2DShadow`, which for a plain 2D map needs no
    // extension in ES 3.00 (the `GL_EXT_texture_shadow_lod` question only
    // arises for cube/array shadows, which the engine does not use). The
    // plainest shadowed key and the busiest one both make the crossing — the
    // busy one matters because the lookup runs after the phase circle's
    // discard, which is the control-flow shape a GLSL backend could refuse.
    for variant in [
        MaterialVariant::SHADOW,
        MaterialVariant::SHADOW
            | MaterialVariant::PHASE_CIRCLE
            | MaterialVariant::VERTEX_COLOR
            | MaterialVariant::TEXTURE
            | MaterialVariant::NORMAL_MAP,
    ] {
        let source = runt_core::material::variant_source(runt_core::material::BASE_SHADER, variant);
        glsl_es(
            &source,
            &[
                ("vs_main", naga::ShaderStage::Vertex),
                ("fs_main", naga::ShaderStage::Fragment),
            ],
        );
    }
}

/// `tests/frame_uniform.rs`'s translation harness, restated: parse, validate
/// with **no** extra capabilities, emit GLSL ES 3.00 for WebGL2.
fn glsl_es(source: &str, entries: &[(&str, naga::ShaderStage)]) {
    let module = naga::front::wgsl::parse_str(source).expect("WGSL parses");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .expect("WGSL validates with no extra capabilities");

    let options = naga::back::glsl::Options {
        version: naga::back::glsl::Version::Embedded {
            version: 300,
            is_webgl: true,
        },
        ..Default::default()
    };
    for &(entry, stage) in entries {
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
        .unwrap_or_else(|e| panic!("{entry} has no GLSL ES 3.00 form: {e}"));
        writer
            .write()
            .unwrap_or_else(|e| panic!("{entry} failed to translate: {e}"));
        assert!(
            out.contains("#version 300 es"),
            "{entry} did not come out as ES 3.00"
        );
    }
}

// ---------------------------------------------------------------------------
// GPU harness — readback
// ---------------------------------------------------------------------------

fn read_back(renderer: &Renderer, target: &wgpu::Texture, width: u32, height: u32) -> Vec<u8> {
    let device = renderer.device();
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
        target.as_image_copy(),
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
    let mut pixels = Vec::with_capacity((unpadded_row * height) as usize);
    for row in 0..height as usize {
        let start = row * padded_row as usize;
        pixels.extend_from_slice(&padded[start..start + unpadded_row as usize]);
    }
    drop(padded);
    readback.unmap();
    pixels
}

// ---------------------------------------------------------------------------
// 3. Off is byte-identical and allocates nothing — the whole engine
// ---------------------------------------------------------------------------

const ENGINE_WIDTH: u32 = 256;
const ENGINE_HEIGHT: u32 = 192;

/// `tests/render_scale.rs`'s harness, trimmed: the demo scene ticked to a
/// fixed pose, drawn through the full [`Engine`] path — resource mirroring
/// included, which is the door the port actually uses.
struct EngineRig {
    engine: Engine,
    target: wgpu::Texture,
    view: wgpu::TextureView,
}

impl EngineRig {
    fn new() -> Option<EngineRig> {
        let mut engine = match pollster::block_on(Engine::headless(FORMAT)) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("SKIP shadows (no GPU adapter): {e}");
                return None;
            }
        };
        engine.update(0.0);
        let mut t = 0.0;
        while engine.tick_count() < 42 {
            t += runt_core::TICK_DT * 0.25;
            engine.update(t);
        }
        let target = engine.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow test target"),
            size: wgpu::Extent3d {
                width: ENGINE_WIDTH,
                height: ENGINE_HEIGHT,
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
        Some(EngineRig { engine, target, view })
    }

    fn frame(&mut self) -> Vec<u8> {
        self.engine.render(&self.view, ENGINE_WIDTH, ENGINE_HEIGHT);
        read_back(
            self.engine.renderer(),
            &self.target,
            ENGINE_WIDTH,
            ENGINE_HEIGHT,
        )
    }
}

#[test]
fn shadows_off_draws_the_same_pixels_and_allocates_nothing() {
    let Some(mut rig) = EngineRig::new() else {
        return;
    };

    // The path every pinned frame hash takes: nobody has mentioned shadows.
    let baseline = rig.frame();
    assert_eq!(
        rig.engine.renderer().shadow_map_resolution(),
        None,
        "an untouched engine must not allocate a shadow map"
    );

    // Saying `Off` out loud must be the same nothing.
    rig.engine.set_shadow_quality(ShadowQuality::Off);
    assert_eq!(baseline, rig.frame(), "an explicit Off changed pixels");
    assert_eq!(rig.engine.renderer().shadow_map_resolution(), None);

    // Open the gate: the map exists at the tier's size. (What the shadow looks
    // like is the renderer rig's business below; the demo scene's is not
    // pinned here, only that the plumbing allocated what it said.)
    rig.engine.set_shadow_quality(ShadowQuality::Low);
    let low = rig.frame();
    assert_eq!(rig.engine.renderer().shadow_map_resolution(), Some(512));
    rig.engine.set_shadow_quality(ShadowQuality::High);
    rig.frame();
    assert_eq!(rig.engine.renderer().shadow_map_resolution(), Some(2048));

    // …and closing it again frees the map and restores the exact original
    // frame — off costs nothing in both directions, like `RenderScale` at 1.0.
    rig.engine.set_shadow_quality(ShadowQuality::Off);
    assert_eq!(
        baseline,
        rig.frame(),
        "going back to Off did not restore the exact original frame"
    );
    assert_eq!(
        rig.engine.renderer().shadow_map_resolution(),
        None,
        "Off must free the map, not park it"
    );

    // The demo scene has geometry over geometry, so the gate being open is
    // visible at all — otherwise every assertion above would also pass on a
    // shadow pass that draws nothing.
    assert_ne!(baseline, low, "Low rendered the identical frame");
}

// ---------------------------------------------------------------------------
// 4. An occluder darkens its receiver — a bare renderer, a built scene
// ---------------------------------------------------------------------------

const WIDTH: u32 = 128;
const HEIGHT: u32 = 128;

/// A `2·hx` × `2·hz` horizontal rectangle at y = 0, normal +Y, wound
/// counter-clockwise seen from above.
fn rect(hx: f32, hz: f32) -> Mesh {
    Mesh {
        positions: vec![
            Vec3::new(-hx, 0.0, -hz),
            Vec3::new(-hx, 0.0, hz),
            Vec3::new(hx, 0.0, hz),
            Vec3::new(hx, 0.0, -hz),
        ],
        normals: vec![Vec3::Y; 4],
        uvs: vec![glam::Vec2::ZERO; 4],
        colors: vec![Vec3::ONE; 4],
        indices: vec![0, 1, 2, 0, 2, 3],
    }
}

/// A `2·half` × `2·half` horizontal quad, normal +Y, wound counter-clockwise
/// seen from above.
fn slab(half: f32) -> Mesh {
    rect(half, half)
}

/// A caster whose only light-facing geometry floats high over its base: an
/// up-facing cap at `y = height` above a down-facing plate at `y = 0`. Under a
/// vertical key light the cap is the entire silhouette (the plate back-faces
/// the light and writes no depth); the plate's whole job is to stretch the
/// mesh's measured box downward, so the caster *straddles* the light's near
/// plane instead of sitting wholly above it — which keeps even the pre-fix
/// caster cull from being the thing under test.
fn capped_column(half: f32, height: f32) -> Mesh {
    Mesh {
        positions: vec![
            Vec3::new(-half, height, -half),
            Vec3::new(-half, height, half),
            Vec3::new(half, height, half),
            Vec3::new(half, height, -half),
            Vec3::new(-half, 0.0, -half),
            Vec3::new(-half, 0.0, half),
            Vec3::new(half, 0.0, half),
            Vec3::new(half, 0.0, -half),
        ],
        normals: [[Vec3::Y; 4], [Vec3::NEG_Y; 4]].concat(),
        uvs: vec![glam::Vec2::ZERO; 8],
        colors: vec![Vec3::ONE; 8],
        indices: vec![0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6],
    }
}

/// The rig's light: a vertical key so the shadow lands exactly under the
/// occluder, a bright key against a legible two-colour ambient — the shadowed
/// floor should read as `sky_color`, nothing else.
fn rig_lighting() -> Lighting {
    Lighting {
        key_dir: Vec3::Y,
        key_color: Vec3::new(0.7, 0.7, 0.7),
        sky_color: Vec3::new(0.30, 0.33, 0.40),
        ground_color: Vec3::new(0.10, 0.10, 0.10),
        horizon: None,
        clouds: 0.0,
        sun: 0.0,
    }
}

struct Rig {
    renderer: Renderer,
    ground: MeshHandle,
    occluder: MeshHandle,
    target: wgpu::Texture,
    view: wgpu::TextureView,
    view_proj: Mat4,
}

impl Rig {
    fn new() -> Option<Rig> {
        let mut renderer = match pollster::block_on(Renderer::headless(FORMAT)) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP shadows (no GPU adapter): {e}");
                return None;
            }
        };
        let ground = renderer.register_mesh(&slab(15.0));
        let occluder = renderer.register_mesh(&slab(2.0));
        let target = renderer.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow rig target"),
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
        let view_proj = Camera::default()
            .view_proj(Transform::looking_at(Vec3::new(0.0, 10.0, 7.0), Vec3::ZERO, Vec3::Y).matrix(), 1.0);
        Some(Rig {
            renderer,
            ground,
            occluder,
            target,
            view,
            view_proj,
        })
    }

    /// One frame: the ground, plus the occluder floating at y = 3 when asked.
    fn frame(&mut self, quality: ShadowQuality, with_occluder: bool) -> Vec<u8> {
        self.renderer.set_shadow_quality(quality);
        let mut world = World::new();
        let white = Material {
            base_color: Vec4::ONE,
            params: Vec4::ZERO,
            texture: None,
            variant: MaterialVariant::NONE,
        };
        world.spawn((MeshRef(self.ground), white, Transform::IDENTITY));
        if with_occluder {
            world.spawn((
                MeshRef(self.occluder),
                white,
                Transform::from_translation(Vec3::new(0.0, 3.0, 0.0)),
            ));
        }
        let draws = build_draw_list(&mut world, 0.0);
        let frame = runt_core::FrameParams {
            view_proj: self.view_proj,
            lighting: rig_lighting(),
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
        read_back(&self.renderer, &self.target, WIDTH, HEIGHT)
    }

    /// One frame of an arbitrary spawn list under its own camera — the rig's
    /// lighting and flat white material throughout, like [`Rig::frame`].
    fn frame_scene(
        &mut self,
        quality: ShadowQuality,
        spawns: &[(MeshHandle, Transform)],
        view_proj: Mat4,
    ) -> Vec<u8> {
        self.renderer.set_shadow_quality(quality);
        let mut world = World::new();
        let white = Material {
            base_color: Vec4::ONE,
            params: Vec4::ZERO,
            texture: None,
            variant: MaterialVariant::NONE,
        };
        for &(mesh, transform) in spawns {
            world.spawn((MeshRef(mesh), white, transform));
        }
        let draws = build_draw_list(&mut world, 0.0);
        let frame = runt_core::FrameParams {
            view_proj,
            lighting: rig_lighting(),
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
        read_back(&self.renderer, &self.target, WIDTH, HEIGHT)
    }

    /// The frame pixel a world point lands on.
    fn pixel_of(&self, p: Vec3) -> (u32, u32) {
        Rig::pixel_at(self.view_proj, p)
    }

    /// [`Rig::pixel_of`] under an arbitrary camera. Asserts the point is
    /// actually on frame — a float→u32 cast saturates, so an off-screen point
    /// would otherwise sample the frame's edge and lie.
    fn pixel_at(view_proj: Mat4, p: Vec3) -> (u32, u32) {
        let clip = view_proj * p.extend(1.0);
        let ndc = clip.truncate() / clip.w;
        assert!(
            clip.w > 0.0 && ndc.x.abs() < 1.0 && ndc.y.abs() < 1.0,
            "sample point {p} projects off-frame (ndc {ndc})"
        );
        (
            ((ndc.x * 0.5 + 0.5) * WIDTH as f32) as u32,
            ((0.5 - ndc.y * 0.5) * HEIGHT as f32) as u32,
        )
    }
}

fn rgb(pixels: &[u8], (x, y): (u32, u32)) -> [i32; 3] {
    let i = ((y * WIDTH + x) * 4) as usize;
    [pixels[i] as i32, pixels[i + 1] as i32, pixels[i + 2] as i32]
}

#[test]
fn an_occluder_darkens_its_receiver_and_the_ambient_survives() {
    let Some(mut rig) = Rig::new() else { return };

    // The slab floats at y=3 under a vertical light: the ground directly below
    // it is in shadow, and the ground out at x=4 is in the open.
    let shaded_px = rig.pixel_of(Vec3::ZERO);
    let open_px = rig.pixel_of(Vec3::new(4.0, 0.0, 0.0));

    let off = rig.frame(ShadowQuality::Off, true);
    for quality in [ShadowQuality::Low, ShadowQuality::High] {
        let on = rig.frame(quality, true);

        let shaded = rgb(&on, shaded_px);
        let lit = rgb(&on, open_px);
        let was = rgb(&off, shaded_px);

        // Darker under the slab than before, and than the open floor beside
        // it: `key + hemi` fell to `hemi`.
        assert!(
            shaded[0] < was[0] - 60 && shaded[0] < lit[0] - 60,
            "{quality:?}: no shadow under the occluder (shaded {shaded:?}, was {was:?}, lit {lit:?})"
        );

        // …but only to `hemi`: the shadowed floor keeps the hemisphere's sky
        // colour (≈ 0.30 · 255 in red), because the shadow term scales the key
        // light alone. A black floor here means the ambient was shadowed too.
        assert!(
            (40..=130).contains(&shaded[0]) && shaded[2] > shaded[0],
            "{quality:?}: shadow crushed the ambient — {shaded:?} should be the sky-blue hemi"
        );

        // A receiver in the open is untouched to the byte: an unoccluded PCF
        // tap is exactly 1.0 and the arithmetic collapses to the off path.
        assert_eq!(
            lit,
            rgb(&off, open_px),
            "{quality:?}: shadows moved a pixel that nothing occludes"
        );
    }
}

#[test]
fn a_bare_floor_shows_no_acne_at_either_tier() {
    let Some(mut rig) = Rig::new() else { return };

    // No occluder at all: with the biases right, every tap on the floor passes
    // its own depth and the frame is byte-identical to shadows-off. Acne — a
    // floor self-shadowing in moiré stripes — would light this up instantly,
    // which is exactly what `ShadowSettings`' defaults are on the hook for.
    let off = rig.frame(ShadowQuality::Off, false);
    for quality in [ShadowQuality::Low, ShadowQuality::High] {
        let on = rig.frame(quality, false);
        let differing = off
            .iter()
            .zip(&on)
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            differing, 0,
            "{quality:?}: {differing} bytes of acne on an unoccluded floor"
        );
    }
}

#[test]
fn a_second_camera_viewport_skips_the_shadow_pass() {
    let Some(mut rig) = Rig::new() else { return };
    rig.renderer.set_shadow_quality(ShadowQuality::Low);

    // `render_to_texture` deliberately renders shadowless (its frame block
    // says so and its bind group holds the dummy — see the method's docs).
    // The claim worth a test is narrower: with the gate open and the map
    // resident, an offscreen scene still renders validly and its stats add up.
    let mut world = World::new();
    world.spawn((
        MeshRef(rig.ground),
        Material {
            base_color: Vec4::ONE,
            params: Vec4::ZERO,
            texture: None,
            variant: MaterialVariant::NONE,
        },
        Transform::IDENTITY,
    ));
    let draws = build_draw_list(&mut world, 0.0);
    let frame = runt_core::FrameParams {
        view_proj: rig.view_proj,
        lighting: rig_lighting(),
    };
    let stats = rig.renderer.render_to_texture(
        runt_core::RenderTarget(7),
        64,
        64,
        &frame,
        &draws,
        &MeshLibrary::new(),
        &TextureLibrary::new(),
    );
    assert_eq!(stats.items, 1);
    assert_eq!(stats.draws, 1);
}

// ---------------------------------------------------------------------------
// 5. Tall casters and the box rim — the failure modes the port actually hit
// ---------------------------------------------------------------------------
//
// Geometry note, shared by the next two tests: the rig's camera sits at
// (0, 10, 7) looking at the origin, so the light box (extent 20) centres near
// (0, −6.5, −4.5) and the vertical key light's eye — the map's near plane —
// sits at y ≈ 33.5. "Tall" below means *taller than that*.

#[test]
fn a_caster_reaching_above_the_lights_near_plane_still_casts() {
    let Some(mut rig) = Rig::new() else { return };

    // From y = 2 to y = 50 the column straddles the near plane: its measured
    // box dips under the plane, so the caster cull keeps it either way, and
    // what is under test is the *clip* — without pancaking, the cap at y = 50
    // (the entire light-facing silhouette) is above the plane and rasterizes
    // nothing. Both faces back-face the rig's camera, so the frame shows the
    // ground and only the ground.
    let column = rig.renderer.register_mesh(&capped_column(2.0, 48.0));
    let spawns = [
        (rig.ground, Transform::IDENTITY),
        (column, Transform::from_translation(Vec3::new(0.0, 2.0, 0.0))),
    ];
    let vp = rig.view_proj;
    let shaded_px = rig.pixel_of(Vec3::ZERO);

    let off = rig.frame_scene(ShadowQuality::Off, &spawns, vp);
    for quality in [ShadowQuality::Low, ShadowQuality::High] {
        let on = rig.frame_scene(quality, &spawns, vp);
        let shaded = rgb(&on, shaded_px);
        let was = rgb(&off, shaded_px);
        assert!(
            shaded[0] < was[0] - 60,
            "{quality:?}: a caster reaching above the light lost its shadow \
             (shaded {shaded:?}, was {was:?})"
        );
    }
}

#[test]
fn a_caster_wholly_between_the_light_and_the_box_still_casts() {
    let Some(mut rig) = Rig::new() else { return };

    // The occluder slab parked at y = 50: every point of it is above the
    // light's near plane (≈ y = 33.5), which is exactly the caster the old
    // light-frustum cull rejected — and, once kept, exactly the geometry
    // pancaking exists to flatten onto the plane. Both fixes in one caster.
    let spawns = [
        (rig.ground, Transform::IDENTITY),
        (
            rig.occluder,
            Transform::from_translation(Vec3::new(0.0, 50.0, 0.0)),
        ),
    ];
    let vp = rig.view_proj;
    let shaded_px = rig.pixel_of(Vec3::ZERO);

    let off = rig.frame_scene(ShadowQuality::Off, &spawns, vp);
    for quality in [ShadowQuality::Low, ShadowQuality::High] {
        let on = rig.frame_scene(quality, &spawns, vp);
        let shaded = rgb(&on, shaded_px);
        let was = rgb(&off, shaded_px);
        assert!(
            shaded[0] < was[0] - 60,
            "{quality:?}: a caster between the light and the box was culled away \
             (shaded {shaded:?}, was {was:?})"
        );
    }
}

#[test]
fn the_shadow_fades_over_the_rim_band_instead_of_stepping_off() {
    let Some(mut rig) = Rig::new() else { return };

    // A wide world: a big floor, and a long thin strip floating at y = 8 whose
    // shadow runs — under the vertical key light — all the way across and past
    // the light box's rim. The camera sits low and off to one side, so the
    // ground around the rim is on screen and the strip hides none of it.
    let floor = rig.renderer.register_mesh(&rect(45.0, 45.0));
    let strip = rig.renderer.register_mesh(&rect(45.0, 2.0));
    let vp = Camera::default().view_proj(
        Transform::looking_at(Vec3::new(0.0, 12.0, 22.0), Vec3::new(16.0, 0.0, 0.0), Vec3::Y)
            .matrix(),
        1.0,
    );
    let spawns = [
        (floor, Transform::IDENTITY),
        (strip, Transform::from_translation(Vec3::new(0.0, 8.0, 0.0))),
    ];

    let extent = ShadowSettings::default().extent;
    let band = 0.1; // `ShadowSettings::default().fade`, in map-uv units

    let off = rig.frame_scene(ShadowQuality::Off, &spawns, vp);
    for (quality, resolution) in [(ShadowQuality::Low, 512), (ShadowQuality::High, 2048)] {
        // Where the rim is, *exactly*: the same pure matrix the renderer
        // samples through, under which uv.x along the ground line z = 0 is
        // affine in world x — two probes recover the map, and the map places
        // three receivers by their fade coordinate.
        let lvp = shadow::light_view_proj(&vp, Vec3::Y, extent, resolution);
        let uv_x = |x: f32| (lvp * Vec4::new(x, 0.0, 0.0, 1.0)).x * 0.5 + 0.5;
        let (u0, du) = (uv_x(0.0), uv_x(1.0) - uv_x(0.0));
        // The rim ahead of the camera (positive world x — the light's uv axis
        // may point either way along it), measured inward in band widths.
        let rim_uv = |bands: f32| {
            if du > 0.0 { 1.0 - bands * band } else { bands * band }
        };
        let x_at = |uv: f32| (uv - u0) / du;

        // Inside the band's inner edge (full shadow), halfway through the
        // fade, and past the rim entirely (open daylight).
        let p_in = Vec3::new(x_at(rim_uv(1.5)), 0.0, 0.0);
        let p_mid = Vec3::new(x_at(rim_uv(0.5)), 0.0, 0.0);
        let p_out = Vec3::new(x_at(rim_uv(-0.5)), 0.0, 0.0);

        // The uv.x rim must be the *only* fade coordinate in play: the other
        // three (uv.y both ways, far depth) stay interior at all three points.
        for p in [p_in, p_mid, p_out] {
            let ndc = lvp * p.extend(1.0);
            let uv_y = 0.5 - ndc.y * 0.5;
            assert!(
                (0.2..0.8).contains(&uv_y) && (0.2..0.8).contains(&ndc.z),
                "test geometry drifted: {p} has uv.y {uv_y}, depth {}",
                ndc.z
            );
        }

        let on = rig.frame_scene(quality, &spawns, vp);
        let dark = rgb(&on, Rig::pixel_at(vp, p_in));
        let mid = rgb(&on, Rig::pixel_at(vp, p_mid));
        let lit = rgb(&on, Rig::pixel_at(vp, p_out));

        // Past the rim is not merely "bright": it is byte-identical to
        // shadows-off, because the fade scales the shadow term toward lit and
        // an out-of-box receiver never samples at all.
        assert_eq!(
            lit,
            rgb(&off, Rig::pixel_at(vp, p_out)),
            "{quality:?}: the world past the box rim is not exact daylight"
        );
        // The strip does shadow the inner point…
        assert!(
            dark[0] < lit[0] - 40,
            "{quality:?}: no shadow inside the rim band (dark {dark:?}, lit {lit:?})"
        );
        // …and halfway through the band the shadow is *half gone* — strictly
        // between the two, which a hard step at the rim can never produce.
        assert!(
            mid[0] > dark[0] + 12 && mid[0] < lit[0] - 12,
            "{quality:?}: the rim is a step, not a fade (dark {dark:?}, mid {mid:?}, lit {lit:?})"
        );
    }
}
