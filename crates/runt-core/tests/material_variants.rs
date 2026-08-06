//! Shader variants: the preprocessor's output, and that every variant it can
//! produce actually compiles (DESIGN §5).
//!
//! The string half is a plain unit test; the compile half needs a device and
//! skips without one. Both matter — a variant that generates fine but fails to
//! compile is a crash the first time a material uses it, which is exactly the
//! failure mode a variant system invites.

use runt_core::material::{self, MaterialVariant};
use runt_core::Renderer;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// The bits the *shader source* branches on: everything except the three that
/// only select fixed-function state (`TRANSPARENT`, `ADDITIVE`,
/// `DEPTH_GREATER`), which cannot change whether a module compiles.
const SHADER_BITS: [MaterialVariant; 8] = [
    MaterialVariant::VERTEX_COLOR,
    MaterialVariant::TEXTURE,
    MaterialVariant::RAMP,
    MaterialVariant::LIVE_TEX,
    MaterialVariant::NORMAL_MAP,
    MaterialVariant::PHASE_CIRCLE,
    MaterialVariant::BILLBOARD_UNLIT,
    // The one bit the *vertex* stage branches on. In here rather than only in
    // `tests/vertex_wave.rs` because a vertex branch can fail to compile
    // against any fragment branch, and 256 modules is still under a second.
    MaterialVariant::VERTEX_WAVE,
];

/// Every combination of the flags the WGSL can branch on — 256 of them.
///
/// The sweep follows [`SHADER_BITS`] rather than `FLAGS` so that adding a
/// *look* extends the coverage automatically. It stops short of all 8192 keys
/// on purpose: the three render-state bits produce byte-identical WGSL (nothing
/// reads `F_TRANSPARENT`), so crossing them in would octuple a GPU test's
/// runtime to compile the same 256 modules eight times each. What they select
/// instead of a branch is covered without a device by
/// `the_render_state_comes_from_the_key`, and their effect on the *cache* by
/// `every_state_combination_is_its_own_pipeline`.
///
/// `FRESNEL` and `EMISSIVE_SWEEP` are the two branching bits *not* in here, and
/// that is a gap rather than a decision: they were appended after this list was
/// written and crossing them in would quadruple it again. Both are exercised
/// end to end by `tests/transparency.rs` and by the port's own frames.
///
/// The sweep includes combinations the draw list will never emit
/// (`TEXTURE | LIVE_TEX`, with and without the rest). They stay in on purpose:
/// "unreachable" is a property of `draw::resolve_variant` today and of nothing
/// in the shader, and a key that *cannot* be built is a better guarantee than a
/// key nobody currently builds.
fn all_variants() -> Vec<MaterialVariant> {
    (0..(1u32 << SHADER_BITS.len()))
        .map(|mask| {
            let mut variant = MaterialVariant::NONE;
            for (i, flag) in SHADER_BITS.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    variant |= *flag;
                }
            }
            variant
        })
        .collect()
}

#[test]
fn variant_source_declares_every_flag_with_the_right_value() {
    let src = material::variant_source(material::BASE_SHADER, MaterialVariant::VERTEX_COLOR);
    assert!(src.contains("const F_VERTEX_COLOR: bool = true;"), "{src:.400}");
    assert!(src.contains("const F_TEXTURE: bool = false;"));
    assert!(src.contains("const F_RAMP: bool = false;"));
    assert!(src.contains("const F_LIVE_TEX: bool = false;"));
    assert!(src.contains("const F_NORMAL_MAP: bool = false;"));
    assert!(src.contains("const F_TRANSPARENT: bool = false;"));
    assert!(src.contains("const F_ADDITIVE: bool = false;"));
    assert!(src.contains("const F_DEPTH_GREATER: bool = false;"));
    assert!(src.contains("const F_PHASE_CIRCLE: bool = false;"));
    assert!(src.contains("const F_BILLBOARD_UNLIT: bool = false;"));
    assert!(src.contains("const F_VERTEX_WAVE: bool = false;"));
    assert!(src.ends_with(material::BASE_SHADER), "base source is appended verbatim");

    let none = material::variant_source(material::BASE_SHADER, MaterialVariant::NONE);
    assert!(none.contains("const F_VERTEX_COLOR: bool = false;"));

    let two = material::variant_source(
        material::BASE_SHADER,
        MaterialVariant::VERTEX_COLOR | MaterialVariant::RAMP,
    );
    assert!(two.contains("const F_VERTEX_COLOR: bool = true;"));
    assert!(two.contains("const F_RAMP: bool = true;"));

    // Every declared flag gets a const, whether or not v1 implements it —
    // otherwise the base source could not reference one before it works.
    for (name, _) in MaterialVariant::FLAGS {
        assert!(none.contains(&format!("const {name}: bool = ")), "missing {name}");
    }

    // Generation is a pure function of (base, key): same key, same bytes.
    assert_eq!(
        material::variant_source(material::BASE_SHADER, MaterialVariant::VERTEX_COLOR),
        src
    );
}

#[test]
fn variant_keys_behave_as_bitflags() {
    let v = MaterialVariant::VERTEX_COLOR | MaterialVariant::TEXTURE;
    assert!(v.contains(MaterialVariant::VERTEX_COLOR));
    assert!(v.contains(MaterialVariant::TEXTURE));
    assert!(!v.contains(MaterialVariant::RAMP));
    assert!(MaterialVariant::NONE.is_empty());

    // Vertex colour, both texture paths (§7) and the normal map are
    // implemented; only the ramp bit is declared and inert.
    assert_eq!(v.unimplemented(), MaterialVariant::NONE);
    assert_eq!(
        MaterialVariant::NORMAL_MAP.unimplemented(),
        MaterialVariant::NONE
    );
    assert_eq!(
        MaterialVariant::LIVE_TEX.unimplemented(),
        MaterialVariant::NONE,
        "live eval landed; the bit is no longer reserved"
    );
    assert_eq!(
        (v | MaterialVariant::RAMP).unimplemented(),
        MaterialVariant::RAMP,
        "reserved bits must report as unimplemented, not silently pass"
    );

    // Bit positions are permanent — a cache key that meant one thing must never
    // come to mean another. `NORMAL_MAP` is appended at bit 4 for that reason,
    // and the render-state set at 5..9 for the same one.
    assert_eq!(MaterialVariant::VERTEX_COLOR.bits(), 1 << 0);
    assert_eq!(MaterialVariant::TEXTURE.bits(), 1 << 1);
    assert_eq!(MaterialVariant::RAMP.bits(), 1 << 2);
    assert_eq!(MaterialVariant::LIVE_TEX.bits(), 1 << 3);
    assert_eq!(MaterialVariant::NORMAL_MAP.bits(), 1 << 4);
    assert_eq!(MaterialVariant::TRANSPARENT.bits(), 1 << 5);
    assert_eq!(MaterialVariant::ADDITIVE.bits(), 1 << 6);
    assert_eq!(MaterialVariant::DEPTH_GREATER.bits(), 1 << 7);
    assert_eq!(MaterialVariant::PHASE_CIRCLE.bits(), 1 << 8);
    assert_eq!(MaterialVariant::BILLBOARD_UNLIT.bits(), 1 << 9);
    assert_eq!(MaterialVariant::FRESNEL.bits(), 1 << 10);
    assert_eq!(MaterialVariant::EMISSIVE_SWEEP.bits(), 1 << 11);
    assert_eq!(MaterialVariant::VERTEX_WAVE.bits(), 1 << 12);

    // The flag list and the bits agree, so no key can be generated that the
    // preprocessor would not emit a const for.
    let mut union = MaterialVariant::NONE;
    for (_, flag) in MaterialVariant::FLAGS {
        assert!(!union.contains(flag), "duplicate flag bit {:#06b}", flag.bits());
        union |= flag;
    }
    assert_eq!(union.bits(), 0b1_1111_1111_1111);

    // The four looks that replace the lighting term rather than feeding it.
    // Exactly one wins per fragment; the mask is what says which four they are.
    assert_eq!(
        MaterialVariant::UNLIT.bits(),
        MaterialVariant::FRESNEL.bits()
            | MaterialVariant::EMISSIVE_SWEEP.bits()
            | MaterialVariant::BILLBOARD_UNLIT.bits(),
    );

    // `contains` is all-of, `intersects` is any-of. `BLENDED` is a mask where
    // either bit alone is the whole answer, so the two are not interchangeable.
    let both = MaterialVariant::BLENDED;
    assert!(MaterialVariant::TRANSPARENT.intersects(both));
    assert!(MaterialVariant::ADDITIVE.intersects(both));
    assert!(!MaterialVariant::TRANSPARENT.contains(both));
    assert!(both.contains(both));
    assert!(!MaterialVariant::DEPTH_GREATER.intersects(both));
}

/// The keystone claim of D2: fixed-function state is a pure function of the
/// key, so two looks share a pipeline only when they really are the same
/// pipeline. No GPU — this is a table, and a table is worth testing as one.
#[test]
fn the_render_state_comes_from_the_key() {
    use runt_core::render_state;
    use wgpu::CompareFunction::{Greater, Less};

    let opaque = render_state(MaterialVariant::VERTEX_COLOR);
    assert_eq!(opaque.blend, wgpu::BlendState::REPLACE);
    assert!(opaque.depth_write);
    assert_eq!(opaque.depth_compare, Less);
    // Every pre-D2 key must still describe exactly the pipeline that was
    // hardcoded before D2 existed — that is what "opaque frames are unchanged"
    // means at this layer.
    for bits in 0..(1u32 << 5) {
        assert_eq!(
            render_state(MaterialVariant::from_bits(bits)),
            opaque,
            "key {bits:#07b} moved off the opaque state"
        );
    }
    // …and so must the two looks that are opaque despite being above the blend
    // bits numerically.
    assert_eq!(render_state(MaterialVariant::PHASE_CIRCLE), opaque);
    assert_eq!(render_state(MaterialVariant::BILLBOARD_UNLIT), opaque);

    let transparent = render_state(MaterialVariant::TRANSPARENT);
    assert_eq!(transparent.blend, wgpu::BlendState::ALPHA_BLENDING);
    assert!(!transparent.depth_write, "a blended draw is not an occluder");
    assert_eq!(transparent.depth_compare, Less);

    let additive = render_state(MaterialVariant::ADDITIVE);
    assert_eq!(additive.blend.color.dst_factor, wgpu::BlendFactor::One);
    assert_eq!(additive.blend.color.src_factor, wgpu::BlendFactor::SrcAlpha);
    assert!(!additive.depth_write);

    // Both bits: additive wins, the way LIVE_TEX wins over TEXTURE.
    assert_eq!(
        render_state(MaterialVariant::TRANSPARENT | MaterialVariant::ADDITIVE).blend,
        additive.blend
    );

    // DEPTH_GREATER composes with anything and implies nothing.
    let silhouette = render_state(MaterialVariant::ADDITIVE | MaterialVariant::DEPTH_GREATER);
    assert_eq!(silhouette.depth_compare, Greater);
    assert_eq!(silhouette.blend, additive.blend);
    assert!(!silhouette.depth_write);
    let opaque_greater = render_state(MaterialVariant::DEPTH_GREATER);
    assert_eq!(opaque_greater.depth_compare, Greater);
    assert_eq!(opaque_greater.blend, wgpu::BlendState::REPLACE);
    assert!(
        opaque_greater.depth_write,
        "an inverted depth test is not a blend"
    );
}

#[test]
fn every_variant_compiles_into_a_pipeline() {
    let mut renderer = match pollster::block_on(Renderer::headless(FORMAT)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("SKIP (no GPU adapter): {e}");
            return;
        }
    };

    for variant in all_variants() {
        let source = material::variant_source(material::BASE_SHADER, variant);

        // Validation errors are reported out of band; scope them so a failure
        // names the variant instead of aborting somewhere in the driver.
        let scope = renderer
            .device()
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let _module = renderer
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("variant under test"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let err = pollster::block_on(scope.pop());
        assert!(
            err.is_none(),
            "variant {:#06b} failed to compile: {err:?}",
            variant.bits()
        );

        // And the whole pipeline, which also checks the vertex layout and the
        // two bind-group layouts against the generated module.
        renderer.ensure_pipeline(variant);
    }

    assert_eq!(
        renderer.pipeline_count(),
        all_variants().len(),
        "one cached pipeline per key"
    );

    // The cache is a cache: asking again compiles nothing new.
    renderer.ensure_pipeline(MaterialVariant::VERTEX_COLOR);
    assert_eq!(renderer.pipeline_count(), all_variants().len());
}

/// The render-state bits key the cache like any other bit: every combination is
/// its own pipeline, and each is compiled exactly once.
///
/// The failure this exists to catch is the quiet one — a key that shares a
/// cached pipeline with a differently-blended twin would draw with whichever
/// state happened to be compiled first, which is invisible until it is a
/// shipped frame with the wrong transparency in it.
#[test]
fn every_state_combination_is_its_own_pipeline() {
    let mut renderer = match pollster::block_on(Renderer::headless(FORMAT)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("SKIP (no GPU adapter): {e}");
            return;
        }
    };

    // Every combination of the three state bits, over two different looks, so a
    // cache collision *between* the two halves of the key would show up too.
    let mut keys = Vec::new();
    for look in [MaterialVariant::NONE, MaterialVariant::VERTEX_COLOR] {
        for bits in 0..8u32 {
            let mut variant = look;
            if bits & 1 != 0 {
                variant |= MaterialVariant::TRANSPARENT;
            }
            if bits & 2 != 0 {
                variant |= MaterialVariant::ADDITIVE;
            }
            if bits & 4 != 0 {
                variant |= MaterialVariant::DEPTH_GREATER;
            }
            keys.push(variant);
        }
    }
    assert_eq!(keys.len(), 16);

    for variant in &keys {
        renderer.ensure_pipeline(*variant);
    }
    assert_eq!(
        renderer.pipeline_count(),
        keys.len(),
        "one pipeline per key, and no two keys sharing one"
    );
    // Exactly once: a second sweep compiles nothing.
    for variant in &keys {
        renderer.ensure_pipeline(*variant);
    }
    assert_eq!(renderer.pipeline_count(), keys.len());
}
