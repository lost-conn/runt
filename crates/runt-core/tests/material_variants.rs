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

/// The bits the *shader source* branches on: everything except the four that
/// only select fixed-function state (`TRANSPARENT`, `ADDITIVE`,
/// `DEPTH_GREATER`, `TWO_SIDED`), which cannot change whether a module
/// compiles.
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
    // against any fragment branch, and 256 modules is a few seconds.
    MaterialVariant::VERTEX_WAVE,
];

/// `LOCAL_SPACE`'s whole reach, as a product of its own — the reason it is
/// **not** in [`SHADER_BITS`].
///
/// It belongs there by the letter of that list's rule: it does branch, so it does
/// change a module's bytes, which is the test `TWO_SIDED` fails. What it fails
/// instead is the cost/coverage trade — and the trade was measured rather than
/// guessed. `SHADER_BITS` with the bit appended compiles 512 modules in **21.3 s**
/// where 256 take **9.3 s**; the 32-key product below costs **1.3 s**.
///
/// The twelve seconds buy nothing. The bit's entire reach is `shader.wgsl`'s
/// `p_source` and the two derivative pairs, so the only flags it can interact with
/// are the two texture paths, `NORMAL_MAP` (which reads a pair) and `VERTEX_WAVE`
/// (which writes the varying it selects) — four flags, all four crossed in full
/// below. The other 240 modules re-prove that a `const` the fragment already
/// folded is still folded.
///
/// This is therefore a *narrower* claim than `SHADER_BITS`' and it is written
/// down as one: if a future bit ever reads `local_pos` from outside the texture
/// branches, this list is what has to grow, or the bit goes in `SHADER_BITS`
/// and the sweep pays double.
const LOCAL_SPACE_REACH: [MaterialVariant; 4] = [
    MaterialVariant::TEXTURE,
    MaterialVariant::LIVE_TEX,
    MaterialVariant::NORMAL_MAP,
    MaterialVariant::VERTEX_WAVE,
];

/// Every combination of the flags the WGSL can branch on — 256 of them.
///
/// The sweep follows [`SHADER_BITS`] rather than `FLAGS` so that adding a
/// *look* extends the coverage automatically. It stops short of all 16384 keys
/// on purpose: the four render-state bits produce byte-identical WGSL (nothing
/// reads `F_TRANSPARENT`), so crossing them in would multiply a GPU test's
/// runtime sixteenfold to compile the same 256 modules over and over. What they
/// select instead of a branch is covered without a device by
/// `the_render_state_comes_from_the_key`, and their effect on the *cache* by
/// `every_state_combination_is_its_own_pipeline`. `TWO_SIDED`'s effect on the
/// *pixels* — the half no cache test can see — is `tests/two_sided.rs`.
///
/// `FRESNEL` and `EMISSIVE_SWEEP` are the two branching bits *not* in here, and
/// that is a gap rather than a decision: they were appended after this list was
/// written and crossing them in would quadruple it again. Both are exercised
/// end to end by `tests/transparency.rs` and by the port's own frames.
/// `LOCAL_SPACE` is also absent, and that one *is* a decision — see
/// [`LOCAL_SPACE_REACH`].
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
    assert!(src.contains("const F_TWO_SIDED: bool = false;"));
    assert!(src.contains("const F_LOCAL_SPACE: bool = false;"));
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
    assert_eq!(MaterialVariant::TWO_SIDED.bits(), 1 << 13);
    assert_eq!(MaterialVariant::SHADOW.bits(), 1 << 14);
    assert_eq!(MaterialVariant::LOCAL_SPACE.bits(), 1 << 15);

    // The flag list and the bits agree, so no key can be generated that the
    // preprocessor would not emit a const for.
    let mut union = MaterialVariant::NONE;
    for (_, flag) in MaterialVariant::FLAGS {
        assert!(!union.contains(flag), "duplicate flag bit {:#06b}", flag.bits());
        union |= flag;
    }
    assert_eq!(union.bits(), 0b1111_1111_1111_1111);

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
    assert_eq!(opaque.cull, Some(wgpu::Face::Back));
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

    // TWO_SIDED touches the cull field and nothing else — it composes with every
    // other state bit the way DEPTH_GREATER does.
    let two_sided = render_state(MaterialVariant::TWO_SIDED);
    assert_eq!(two_sided.cull, None);
    assert_eq!(two_sided.blend, opaque.blend);
    assert_eq!(two_sided.depth_compare, opaque.depth_compare);
    assert_eq!(two_sided.depth_write, opaque.depth_write);
    let two_sided_additive = render_state(MaterialVariant::TWO_SIDED | MaterialVariant::ADDITIVE);
    assert_eq!(two_sided_additive.cull, None);
    assert_eq!(two_sided_additive.blend, additive.blend);
    assert!(!two_sided_additive.depth_write);
    // …and nothing else touches it: every key without the bit still culls back
    // faces, which is what "opaque frames are unchanged" means one field over.
    for bits in 0..(1u32 << 13) {
        let variant = MaterialVariant::from_bits(bits);
        assert_eq!(
            render_state(variant).cull,
            Some(wgpu::Face::Back),
            "key {bits:#015b} lost its back-face cull without asking"
        );
    }
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

/// The sweep [`LOCAL_SPACE_REACH`] buys instead of doubling the one above: the
/// local-space basis compiled against every flag that can reach it, with the bit
/// on and off, and one cached pipeline per key.
///
/// The failures it is aimed at are specific rather than generic. Under
/// `LOCAL_SPACE | NORMAL_MAP | LIVE_TEX` the fragment takes **two** derivative
/// pairs where every other key takes one, and both `dpdx` calls sit inside a
/// `const bool` branch — WGSL forbids a derivative in non-uniform control flow,
/// so if naga ever stopped treating a module-level `const` condition as uniform,
/// this is the key that would stop compiling. Under `LOCAL_SPACE | VERTEX_WAVE`
/// the varying the fragment selects is written from a value the vertex stage
/// mutated, which is the one place the two stages' halves of the bit meet.
#[test]
fn every_local_space_texture_combination_compiles() {
    let mut renderer = match pollster::block_on(Renderer::headless(FORMAT)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("SKIP (no GPU adapter): {e}");
            return;
        }
    };

    let mut keys = Vec::new();
    for local in [MaterialVariant::NONE, MaterialVariant::LOCAL_SPACE] {
        for bits in 0..(1u32 << LOCAL_SPACE_REACH.len()) {
            let mut variant = local;
            for (i, flag) in LOCAL_SPACE_REACH.iter().enumerate() {
                if bits & (1 << i) != 0 {
                    variant |= *flag;
                }
            }
            keys.push(variant);
        }
    }
    assert_eq!(keys.len(), 32, "two halves of a four-bit product");

    for variant in &keys {
        let scope = renderer
            .device()
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let _module = renderer
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("local-space variant under test"),
                source: wgpu::ShaderSource::Wgsl(
                    material::variant_source(material::BASE_SHADER, *variant).into(),
                ),
            });
        let err = pollster::block_on(scope.pop());
        assert!(
            err.is_none(),
            "variant {:#018b} failed to compile: {err:?}",
            variant.bits()
        );
        renderer.ensure_pipeline(*variant);
    }

    // Half of these keys differ from the other half in bit 15 alone, so a key
    // space that had quietly masked `LOCAL_SPACE` off — the way
    // `resolve_variant` masks the two texture bits — would show up here as
    // sixteen pipelines instead of thirty-two.
    assert_eq!(
        renderer.pipeline_count(),
        keys.len(),
        "one pipeline per key, and the bit is part of the key"
    );
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

    // Every combination of the four state bits, over two different looks, so a
    // cache collision *between* the two halves of the key would show up too.
    // `TWO_SIDED` is in here rather than in the compile sweep for exactly this
    // reason: it generates the same WGSL as its twin without the bit, so the
    // cache is the only place a collision could hide.
    const STATE_BITS: [MaterialVariant; 4] = [
        MaterialVariant::TRANSPARENT,
        MaterialVariant::ADDITIVE,
        MaterialVariant::DEPTH_GREATER,
        MaterialVariant::TWO_SIDED,
    ];
    let mut keys = Vec::new();
    for look in [MaterialVariant::NONE, MaterialVariant::VERTEX_COLOR] {
        for bits in 0..(1u32 << STATE_BITS.len()) {
            let mut variant = look;
            for (i, flag) in STATE_BITS.iter().enumerate() {
                if bits & (1 << i) != 0 {
                    variant |= *flag;
                }
            }
            keys.push(variant);
        }
    }
    assert_eq!(keys.len(), 32);

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
