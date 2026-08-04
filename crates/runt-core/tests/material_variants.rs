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

/// Every combination of the declared flags.
///
/// Five flags since §7's texture bits landed, so 32 pipelines rather than 16 —
/// the count follows `FLAGS` on purpose, so adding a look extends the coverage
/// automatically instead of leaving a new combination untested.
///
/// §7's live path did **not** add a flag — `LIVE_TEX` was reserved at bit 3
/// from the start, which is exactly what reserving it was for — so the count
/// stays 32. It does add four combinations the draw list will never emit
/// (`TEXTURE | LIVE_TEX`, with and without the other two bits); they stay in
/// this sweep on purpose. "Unreachable" is a property of `draw::resolve_variant`
/// today and of nothing in the shader, and a key that cannot be built is a
/// better guarantee than a key nobody currently builds.
fn all_variants() -> Vec<MaterialVariant> {
    let bits = MaterialVariant::FLAGS.len();
    (0..(1u32 << bits)).map(MaterialVariant::from_bits).collect()
}

#[test]
fn variant_source_declares_every_flag_with_the_right_value() {
    let src = material::variant_source(material::BASE_SHADER, MaterialVariant::VERTEX_COLOR);
    assert!(src.contains("const F_VERTEX_COLOR: bool = true;"), "{src:.400}");
    assert!(src.contains("const F_TEXTURE: bool = false;"));
    assert!(src.contains("const F_RAMP: bool = false;"));
    assert!(src.contains("const F_LIVE_TEX: bool = false;"));
    assert!(src.contains("const F_NORMAL_MAP: bool = false;"));
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
    // come to mean another. `NORMAL_MAP` is appended at bit 4 for that reason.
    assert_eq!(MaterialVariant::VERTEX_COLOR.bits(), 0b00001);
    assert_eq!(MaterialVariant::TEXTURE.bits(), 0b00010);
    assert_eq!(MaterialVariant::RAMP.bits(), 0b00100);
    assert_eq!(MaterialVariant::LIVE_TEX.bits(), 0b01000);
    assert_eq!(MaterialVariant::NORMAL_MAP.bits(), 0b10000);

    // The flag list and the bits agree, so no key can be generated that the
    // preprocessor would not emit a const for.
    let mut union = MaterialVariant::NONE;
    for (_, flag) in MaterialVariant::FLAGS {
        assert!(!union.contains(flag), "duplicate flag bit {:#06b}", flag.bits());
        union |= flag;
    }
    assert_eq!(union.bits(), 0b11111);
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
