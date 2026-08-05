//! The frame uniform's layout, and the three places that restate it (D1).
//!
//! `FrameUniform` is declared once in Rust and twice in WGSL — `shader.wgsl`
//! for every material variant, `sky.wgsl` for the background pass — because
//! there is one buffer, one bind group and one layout behind all of them. A
//! field added to one and forgotten in another is not a compile error anywhere:
//! it is a silent misalignment that shows up as a rotated matrix or a light
//! coming from the wrong side, and only in whichever pass was left behind.
//!
//! So the sync is a test. The Rust half is checked by size and offset, and the
//! two WGSL halves by reading their `struct Frame` blocks and holding the field
//! *names, in order* against the Rust struct's.

use runt_core::{FrameUniform, MaterialVariant, Renderer};

/// The fields, in order, as `FrameUniform` declares them.
const FIELDS: [&str; 10] = [
    "view_proj",
    "inv_view_proj",
    "light_dir",
    "light_color",
    "sky_color",
    "ground_color",
    "horizon_color",
    "phase",
    "time",
    "viewport",
];

#[test]
fn the_block_is_std140_shaped() {
    // Two mat4x4 (64 B each) and eight vec4 (16 B each). Nothing needs padding
    // because nothing in it is smaller than a vec4 — which is the whole reason
    // the light direction is a `vec4` with a wasted `w` rather than a `vec3`.
    assert_eq!(std::mem::size_of::<FrameUniform>(), 2 * 64 + 8 * 16);
    assert_eq!(std::mem::size_of::<FrameUniform>(), 256);
    assert_eq!(
        std::mem::size_of::<FrameUniform>() % 16,
        0,
        "a uniform block's size must be a multiple of 16 under std140"
    );

    // Every member starts on a 16-byte boundary, which is what makes the WGSL
    // side's implicit `@align(16)` on vec4/mat4 agree with `repr(C)`.
    let offsets = [
        std::mem::offset_of!(FrameUniform, view_proj),
        std::mem::offset_of!(FrameUniform, inv_view_proj),
        std::mem::offset_of!(FrameUniform, light_dir),
        std::mem::offset_of!(FrameUniform, light_color),
        std::mem::offset_of!(FrameUniform, sky_color),
        std::mem::offset_of!(FrameUniform, ground_color),
        std::mem::offset_of!(FrameUniform, horizon_color),
        std::mem::offset_of!(FrameUniform, phase),
        std::mem::offset_of!(FrameUniform, time),
        std::mem::offset_of!(FrameUniform, viewport),
    ];
    assert_eq!(offsets, [0, 64, 128, 144, 160, 176, 192, 208, 224, 240]);
    for (field, offset) in FIELDS.iter().zip(offsets) {
        assert_eq!(offset % 16, 0, "{field} is not 16-byte aligned");
    }

    // D1's three new vec4s are appended, never inserted: a field that moved
    // would re-key nothing (there is no cache on this block) but would put the
    // sky pass and the material pass on different pages of the same buffer for
    // exactly as long as it took someone to notice.
    assert_eq!(std::mem::offset_of!(FrameUniform, horizon_color), 192);
}

/// The field list of a WGSL `struct Frame { … };` block, comments stripped.
fn wgsl_frame_fields(source: &str) -> Vec<String> {
    let start = source
        .find("struct Frame {")
        .expect("the source declares a Frame block");
    let body = &source[start..];
    let end = body.find("};").expect("the Frame block is terminated");
    body[..end]
        .lines()
        .skip(1)
        .filter_map(|line| {
            let line = line.trim();
            if line.starts_with("//") {
                return None;
            }
            let (name, _) = line.split_once(':')?;
            Some(name.trim().to_string())
        })
        .collect()
}

#[test]
fn all_three_declarations_agree() {
    let material = wgsl_frame_fields(runt_core::material::BASE_SHADER);
    let sky = wgsl_frame_fields(runt_core::SKY_SHADER);

    assert_eq!(material, FIELDS, "shader.wgsl drifted from FrameUniform");
    assert_eq!(sky, FIELDS, "sky.wgsl drifted from FrameUniform");
    assert_eq!(material, sky, "the two shaders drifted from each other");
}

#[test]
fn the_phase_circle_is_a_render_value_with_a_resting_state() {
    let mut renderer = match pollster::block_on(Renderer::headless(
        wgpu::TextureFormat::Rgba8Unorm,
    )) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("SKIP (no GPU adapter): {e}");
            return;
        }
    };

    // Off by default: a radius of zero is a circle nothing is inside, so world
    // geometry is solid and phase-only geometry is gone — the same resting
    // state the original's `phase_radius = 0` means.
    assert_eq!(renderer.phase_fx(), (glam::Vec2::ZERO, 0.0, 0.0));

    renderer.set_phase_fx(glam::Vec2::new(0.25, -0.5), 0.3, 0.7);
    let (center, radius, strength) = renderer.phase_fx();
    assert_eq!(center, glam::Vec2::new(0.25, -0.5));
    assert_eq!(radius, 0.3);
    assert_eq!(strength, 0.7);

    // Nonsense in, resting state out: a negative radius is no circle and a
    // strength outside 0..1 is clamped, so a game doing its own easing cannot
    // hand the shader a value the fringe blend has no meaning for.
    renderer.set_phase_fx(glam::Vec2::ZERO, -1.0, 4.0);
    assert_eq!(renderer.phase_fx(), (glam::Vec2::ZERO, 0.0, 1.0));
    renderer.set_phase_fx(glam::Vec2::ZERO, 1.0, -1.0);
    assert_eq!(renderer.phase_fx(), (glam::Vec2::ZERO, 1.0, 0.0));
}

/// The one check that costs nothing to run here and would otherwise only fail
/// in a browser: the phase variant has to survive naga's WGSL → GLSL-ES 3.00
/// translation, not merely compile on the machine running the test.
///
/// It earns its place because of what the variant does — `discard` before the
/// texture taps, in control flow the compiler cannot prove uniform. That is
/// exactly the shape a GLSL backend can refuse over derivative uniformity, and
/// WebGL2 is a first-class target (DESIGN §11).
#[test]
fn the_phase_variant_translates_to_glsl_es_for_webgl2() {
    let variant = MaterialVariant::PHASE_CIRCLE
        | MaterialVariant::VERTEX_COLOR
        | MaterialVariant::TEXTURE
        | MaterialVariant::NORMAL_MAP;
    let source = runt_core::material::variant_source(runt_core::material::BASE_SHADER, variant);
    let module = naga::front::wgsl::parse_str(&source).expect("variant WGSL parses");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .expect("variant WGSL validates with no extra capabilities");

    let options = naga::back::glsl::Options {
        version: naga::back::glsl::Version::Embedded {
            version: 300,
            is_webgl: true,
        },
        ..Default::default()
    };
    for (stage, entry) in [
        (naga::ShaderStage::Vertex, "vs_main"),
        (naga::ShaderStage::Fragment, "fs_main"),
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
        .unwrap_or_else(|e| panic!("{entry} has no GLSL ES 3.00 form: {e}"));
        writer
            .write()
            .unwrap_or_else(|e| panic!("{entry} failed to translate: {e}"));
    }
}
