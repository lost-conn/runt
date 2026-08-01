//! The generator registry (DESIGN §6).
//!
//! Two things are on trial here: that `generate` is pure, and that `param_key`
//! is a *stable* function of the params rather than of this process's memory
//! layout. The second is what an on-disk cache and a cross-machine content hash
//! both rest on.

use glam::{Vec2, Vec3};
use runt_core::gen::{GeneratorSpec, Shading};
use runt_core::{MeshHandle, Quality, TerrainParams};

/// One of every variant, so a new generator cannot be added without deciding
/// what it does here.
fn every_variant() -> Vec<GeneratorSpec> {
    vec![
        GeneratorSpec::Plane {
            size: Vec2::new(4.0, 6.0),
            subdivisions: 3,
            shading: Shading::Generated,
            color: None,
        },
        GeneratorSpec::Cube {
            size: 1.5,
            shading: Shading::Generated,
            color: Some(Vec3::new(0.2, 0.4, 0.6)),
        },
        GeneratorSpec::UvSphere {
            radius: 0.9,
            rings: 12,
            sectors: 16,
            shading: Shading::Smooth(180.0),
            color: None,
        },
        GeneratorSpec::Cylinder {
            radius: 0.35,
            height: 1.8,
            segments: 16,
            shading: Shading::Generated,
            color: None,
        },
        GeneratorSpec::Cone {
            radius: 0.6,
            height: 1.4,
            segments: 12,
            shading: Shading::Flat,
            color: None,
        },
        GeneratorSpec::Torus {
            major_radius: 0.7,
            minor_radius: 0.22,
            major_segments: 16,
            minor_segments: 8,
            shading: Shading::Generated,
            color: None,
        },
        GeneratorSpec::TwistedBox {
            dims: Vec3::new(1.0, 1.6, 1.0),
            twist: 0.9,
            taper: 0.4,
            shading: Shading::Flat,
            color: Some(Vec3::new(0.8, 0.45, 0.9)),
        },
        GeneratorSpec::Terrain(TerrainParams {
            seed: 7,
            size: Vec2::splat(20.0),
            amplitude: 1.0,
            octaves: 3,
            frequency: 0.1,
            lacunarity: 2.0,
            gain: 0.5,
            base_segments: 12,
            color: None,
        }),
    ]
}

#[test]
fn every_variant_generates_valid_geometry() {
    for spec in every_variant() {
        let mesh = spec.generate(Quality::FULL);
        mesh.validate();
        assert!(!mesh.is_empty(), "{} generated nothing", spec.kind());
        let n = mesh.vertex_count();
        assert_eq!(mesh.normals.len(), n, "{}: normals sized", spec.kind());
        assert_eq!(mesh.uvs.len(), n, "{}: uvs sized", spec.kind());
        assert_eq!(mesh.colors.len(), n, "{}: colors sized", spec.kind());
        for normal in &mesh.normals {
            assert!(
                (normal.length() - 1.0).abs() < 1e-3,
                "{}: normal {normal:?} is not unit",
                spec.kind()
            );
        }
    }
}

#[test]
fn generation_is_pure() {
    for spec in every_variant() {
        let a = spec.generate(Quality::FULL);
        let b = spec.generate(Quality::FULL);
        assert_eq!(a, b, "{}: two runs must be identical", spec.kind());
        assert_eq!(a.content_hash(), b.content_hash());
    }
}

#[test]
fn param_key_is_stable_across_runs_and_orderings() {
    // The whole reason `param_key` goes through postcard rather than
    // `derive(Hash)`: no float bit patterns, no struct padding, no field-order
    // surprise. A freshly built value must key the same as one built earlier.
    for spec in every_variant() {
        let key = spec.param_key(Quality::FULL);
        assert_eq!(key, spec.clone().param_key(Quality::FULL));
        assert_eq!(key, spec.param_key(Quality(1.0)));
        assert_ne!(key, 0, "{}: a degenerate key would collide everything", spec.kind());
    }

    // Distinct variants and distinct params are distinct keys.
    let mut keys: Vec<u64> = every_variant()
        .iter()
        .map(|s| s.param_key(Quality::FULL))
        .collect();
    let before = keys.len();
    keys.sort_unstable();
    keys.dedup();
    assert_eq!(keys.len(), before, "variants must not share a param key");
}

#[test]
fn param_key_notices_a_single_changed_number() {
    let base = GeneratorSpec::UvSphere {
        radius: 0.9,
        rings: 12,
        sectors: 16,
        shading: Shading::Smooth(180.0),
        color: None,
    };
    let variations = [
        GeneratorSpec::UvSphere {
            radius: 0.9000001,
            rings: 12,
            sectors: 16,
            shading: Shading::Smooth(180.0),
            color: None,
        },
        GeneratorSpec::UvSphere {
            radius: 0.9,
            rings: 13,
            sectors: 16,
            shading: Shading::Smooth(180.0),
            color: None,
        },
        GeneratorSpec::UvSphere {
            radius: 0.9,
            rings: 12,
            sectors: 16,
            shading: Shading::Flat,
            color: None,
        },
        GeneratorSpec::UvSphere {
            radius: 0.9,
            rings: 12,
            sectors: 16,
            shading: Shading::Smooth(180.0),
            color: Some(Vec3::ONE),
        },
    ];
    for v in variations {
        assert_ne!(
            base.param_key(Quality::FULL),
            v.param_key(Quality::FULL),
            "changed params must change the key: {v:?}"
        );
    }
}

#[test]
fn quality_is_part_of_the_key_and_of_the_output() {
    // DESIGN §6: "Different quality → different content hash → coexisting LODs
    // for free." The key has to separate them or one LOD would evict the other.
    let spec = GeneratorSpec::UvSphere {
        radius: 1.0,
        rings: 24,
        sectors: 32,
        shading: Shading::Generated,
        color: None,
    };
    let full = spec.param_key(Quality::FULL);
    let half = spec.param_key(Quality(0.5));
    assert_ne!(full, half);

    let a = spec.generate(Quality::FULL);
    let b = spec.generate(Quality(0.5));
    assert!(a.vertex_count() > b.vertex_count(), "lower quality is coarser");
    assert_ne!(MeshHandle::of(&a), MeshHandle::of(&b), "distinct LODs, distinct hashes");
}

#[test]
fn quality_floors_keep_a_terrible_tier_renderable() {
    // §11 again: gates scale content down, they never fail it.
    for spec in every_variant() {
        let mesh = spec.generate(Quality(0.001));
        mesh.validate();
        assert!(!mesh.is_empty(), "{} vanished at the bottom tier", spec.kind());
    }
}

#[test]
fn specs_round_trip_through_ron() {
    // Scene files are hand-edited, so the serde form is the interface: a spec
    // written out has to read back as the same spec, and therefore the same key.
    for spec in every_variant() {
        let text = ron::to_string(&spec).expect("serialize");
        let back: GeneratorSpec = ron::from_str(&text).expect("deserialize");
        assert_eq!(spec, back, "round trip: {text}");
        assert_eq!(spec.param_key(Quality::FULL), back.param_key(Quality::FULL));
    }
}

#[test]
fn omitted_optional_fields_take_their_defaults() {
    // The hand-editing promise: shading and color are omittable everywhere.
    let terse: GeneratorSpec =
        ron::from_str("Cylinder(radius: 0.35, height: 1.8, segments: 24)").expect("parse");
    assert_eq!(
        terse,
        GeneratorSpec::Cylinder {
            radius: 0.35,
            height: 1.8,
            segments: 24,
            shading: Shading::Generated,
            color: None,
        }
    );
}

#[test]
fn shape_and_placement_stay_separate() {
    // A scene's `scale` is placement and must not reach the generator; the
    // twisted box's `dims` is shape and must. If the two ever merged,
    // content-addressed dedup would stop firing for every scaled instance.
    let a = GeneratorSpec::Cube {
        size: 1.0,
        shading: Shading::Generated,
        color: None,
    };
    let scaled_by_shape = GeneratorSpec::TwistedBox {
        dims: Vec3::new(1.0, 1.6, 1.0),
        twist: 0.0,
        taper: 1.0,
        shading: Shading::Generated,
        color: None,
    };
    assert_ne!(
        MeshHandle::of(&a.generate(Quality::FULL)),
        MeshHandle::of(&scaled_by_shape.generate(Quality::FULL)),
        "a shape change must change the content hash"
    );
}
