//! The `reflect` feature's contract (DESIGN §3, §10).
//!
//! The editor builds its param panels by walking a [`GeneratorSpec`] with
//! `bevy_reflect` and never by knowing what a `Torus` is. That only works if
//! four things hold, and this file holds them:
//!
//! 1. every generator variant reflects, with its fields *visible* — including
//!    the glam ones, which need the remote definitions in `runt_core::reflect`;
//! 2. edits made through reflection land on the real value;
//! 3. the slider bounds are readable from static type info, with no value in
//!    hand (a panel is laid out before a value is chosen);
//! 4. `KINDS` and `default_of_kind` agree with the reflected variant list, so
//!    the editor's variant dropdown cannot drift from the enum.
//!
//! The whole file is compiled out without the feature — which is itself part of
//! the contract, since the wasm player must not link `bevy_reflect`.

#![cfg(feature = "reflect")]

use bevy_reflect::enums::{Enum, VariantInfo};
use bevy_reflect::{PartialReflect, TypeInfo, Typed};
use glam::{Vec2, Vec3};
use runt_core::gen::{GeneratorSpec, Shading};
use runt_core::reflect::{FieldRange, TerrainParamsDef};
use runt_mesh::TerrainParams;

// ---------------------------------------------------------------------------
// 1. the derives exist and expose the fields
// ---------------------------------------------------------------------------

#[test]
fn generator_spec_reflects_as_an_enum() {
    let TypeInfo::Enum(info) = GeneratorSpec::type_info() else {
        panic!("GeneratorSpec must reflect as an enum");
    };
    assert_eq!(
        info.variant_len(),
        GeneratorSpec::KINDS.len(),
        "every generator variant must appear in KINDS"
    );
    for kind in GeneratorSpec::KINDS {
        assert!(
            info.variant(kind).is_some(),
            "KINDS names {kind:?}, which the enum does not define"
        );
    }
}

#[test]
fn every_kind_has_a_default_and_round_trips_its_name() {
    for kind in GeneratorSpec::KINDS {
        let spec = GeneratorSpec::default_of_kind(kind)
            .unwrap_or_else(|| panic!("no default for {kind:?}"));
        assert_eq!(&spec.kind(), kind);
        // A default must be a *usable* generator, not just a well-typed one.
        let mesh = spec.generate(runt_mesh::Quality::FULL);
        assert!(
            !mesh.indices.is_empty(),
            "{kind:?}'s default generates an empty mesh"
        );
    }
    assert!(GeneratorSpec::default_of_kind("NotAGenerator").is_none());
}

/// The UvSphere panel: five widgets, in declaration order, with the leaf kinds
/// the mapper switches on.
#[test]
fn walking_uv_sphere_yields_the_expected_widget_tree() {
    let spec = GeneratorSpec::UvSphere {
        radius: 0.9,
        rings: 24,
        sectors: 32,
        shading: Shading::Smooth(180.0),
        color: Some(Vec3::new(0.9, 0.35, 0.35)),
    };

    let fields = field_names(&spec);
    assert_eq!(fields, ["radius", "rings", "sectors", "shading", "color"]);

    let e: &dyn Enum = &spec;
    assert_eq!(e.variant_name(), "UvSphere");

    // f32 → slider, u32 → stepper: both are downcastable leaves.
    assert_eq!(e.field("radius").unwrap().try_downcast_ref::<f32>(), Some(&0.9));
    assert_eq!(e.field("rings").unwrap().try_downcast_ref::<u32>(), Some(&24));
    assert_eq!(e.field("sectors").unwrap().try_downcast_ref::<u32>(), Some(&32));

    // `shading` is an enum → variant dropdown with one f32 payload.
    let shading = e.field("shading").unwrap().reflect_ref().as_enum().unwrap();
    assert_eq!(shading.variant_name(), "Smooth");
    assert_eq!(shading.field_len(), 1);
    assert_eq!(
        shading.field_at(0).unwrap().try_downcast_ref::<f32>(),
        Some(&180.0)
    );

    // `color` is Option<Vec3> → an optional group of three numeric fields. This
    // is the case that only works because of the nested remote definitions.
    let color = e.field("color").unwrap().reflect_ref().as_enum().unwrap();
    assert_eq!(color.variant_name(), "Some");
    let rgb = color.field_at(0).unwrap().reflect_ref().as_struct().unwrap();
    assert_eq!(
        (0..rgb.field_len())
            .map(|i| rgb.name_at(i).unwrap())
            .collect::<Vec<_>>(),
        ["x", "y", "z"]
    );
    assert_eq!(rgb.field("y").unwrap().try_downcast_ref::<f32>(), Some(&0.35));
}

/// The Terrain panel. Its params live in `runt-mesh`, which knows nothing about
/// reflection — everything here comes through `TerrainParamsDef`.
#[test]
fn walking_terrain_yields_the_expected_widget_tree() {
    let spec = GeneratorSpec::Terrain(TerrainParams {
        seed: 20260731,
        size: Vec2::splat(40.0),
        amplitude: 1.2,
        octaves: 4,
        frequency: 0.055,
        lacunarity: 2.0,
        gain: 0.5,
        base_segments: 64,
        color: Some(Vec3::new(0.17, 0.21, 0.18)),
    });

    let e: &dyn Enum = &spec;
    assert_eq!(e.variant_name(), "Terrain");
    // A tuple variant: one payload, which is itself the struct of params.
    assert_eq!(e.field_len(), 1);

    let params = e.field_at(0).unwrap().reflect_ref().as_struct().unwrap();
    let names: Vec<_> = (0..params.field_len())
        .map(|i| params.name_at(i).unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "seed",
            "size",
            "amplitude",
            "octaves",
            "frequency",
            "lacunarity",
            "gain",
            "base_segments",
            "color",
        ]
    );

    // `seed: u64` is the reroll widget's trigger — name *and* type.
    assert_eq!(
        params.field("seed").unwrap().try_downcast_ref::<u64>(),
        Some(&20260731)
    );
    // `size: Vec2` is a two-field group, reached through Vec2Def.
    let size = params.field("size").unwrap().reflect_ref().as_struct().unwrap();
    assert_eq!(
        (0..size.field_len())
            .map(|i| size.name_at(i).unwrap())
            .collect::<Vec<_>>(),
        ["x", "y"]
    );
    assert_eq!(size.field("x").unwrap().try_downcast_ref::<f32>(), Some(&40.0));
}

// ---------------------------------------------------------------------------
// 2. edits land
// ---------------------------------------------------------------------------

#[test]
fn reflected_edits_reach_the_real_value() {
    let mut spec = GeneratorSpec::UvSphere {
        radius: 1.0,
        rings: 16,
        sectors: 24,
        shading: Shading::Generated,
        color: Some(Vec3::ONE),
    };

    // scalar leaf
    (&mut spec as &mut dyn Enum)
        .field_mut("radius")
        .unwrap()
        .try_apply(&2.5f32)
        .unwrap();
    // integer leaf
    (&mut spec as &mut dyn Enum)
        .field_mut("rings")
        .unwrap()
        .try_apply(&48u32)
        .unwrap();
    // a component of a remote-reflected vector, two levels down
    {
        let color = (&mut spec as &mut dyn Enum).field_mut("color").unwrap();
        let some = color.reflect_mut().as_enum().unwrap();
        let rgb = some.field_at_mut(0).unwrap().reflect_mut().as_struct().unwrap();
        rgb.field_mut("z").unwrap().try_apply(&0.25f32).unwrap();
    }

    let GeneratorSpec::UvSphere {
        radius,
        rings,
        color,
        ..
    } = spec
    else {
        panic!("the variant must not change under a field edit")
    };
    assert_eq!(radius, 2.5);
    assert_eq!(rings, 48);
    assert_eq!(color, Some(Vec3::new(1.0, 1.0, 0.25)));
}

/// Terrain's params round-trip through the remote wrapper unchanged — the
/// property that makes an editor edit and a hand-written RON edit equivalent.
#[test]
fn terrain_params_edit_through_the_remote_wrapper() {
    let mut spec = GeneratorSpec::Terrain(TerrainParams::default());
    {
        let e: &mut dyn Enum = &mut spec;
        let params = e.field_at_mut(0).unwrap().reflect_mut().as_struct().unwrap();
        params.field_mut("seed").unwrap().try_apply(&99u64).unwrap();
        params
            .field_mut("amplitude")
            .unwrap()
            .try_apply(&3.5f32)
            .unwrap();
    }
    let GeneratorSpec::Terrain(params) = spec else {
        panic!()
    };
    assert_eq!(params.seed, 99);
    assert_eq!(params.amplitude, 3.5);
    // Everything untouched keeps its value: an edit is a field write, not a
    // rebuild of the struct.
    assert_eq!(params.octaves, TerrainParams::default().octaves);
}

// ---------------------------------------------------------------------------
// 3. ranges are readable without a value
// ---------------------------------------------------------------------------

#[test]
fn slider_bounds_come_from_static_type_info() {
    let TypeInfo::Enum(info) = GeneratorSpec::type_info() else {
        panic!()
    };
    let VariantInfo::Struct(sphere) = info.variant("UvSphere").unwrap() else {
        panic!("UvSphere is a struct variant")
    };

    let radius = sphere
        .field("radius")
        .unwrap()
        .get_attribute::<FieldRange>()
        .copied()
        .expect("radius declares a range");
    assert!(radius.min > 0.0, "a zero-radius sphere is not a sphere");
    assert!(radius.max > radius.min);

    let rings = sphere
        .field("rings")
        .unwrap()
        .get_attribute::<FieldRange>()
        .copied()
        .expect("rings declares a range");
    assert!(rings.min >= 2.0, "a sphere needs at least two rings");

    // …and on a remote-defined struct's fields too.
    let TypeInfo::Struct(terrain) = TerrainParamsDef::type_info() else {
        panic!()
    };
    let gain = FieldRange::lookup(TerrainParamsDef::type_info(), "gain")
        .expect("gain declares a range");
    assert_eq!((gain.min, gain.max), (0.0, 1.0));
    assert!(terrain.field("octaves").unwrap().get_attribute::<FieldRange>().is_some());
    // A field with no attribute reports none, so the mapper can fall back.
    assert!(FieldRange::lookup(TerrainParamsDef::type_info(), "seed").is_none());
}

#[test]
fn field_range_maps_to_and_from_a_slider_track() {
    let r = FieldRange::new(2.0, 10.0);
    assert_eq!(r.normalize(2.0), 0.0);
    assert_eq!(r.normalize(10.0), 1.0);
    assert_eq!(r.denormalize(0.5), 6.0);
    // Out-of-range values are pinned to the ends rather than escaping the track.
    assert_eq!(r.normalize(-100.0), 0.0);
    assert_eq!(r.clamp(1e9), 10.0);
    // A degenerate range must not divide by zero.
    assert_eq!(FieldRange::new(1.0, 1.0).normalize(1.0), 0.0);
}

// ---------------------------------------------------------------------------
// 4. the registry is complete enough to rebuild a value
// ---------------------------------------------------------------------------

#[test]
fn the_type_registry_knows_every_editable_type() {
    let registry = runt_core::reflect::type_registry();
    for path in [
        "runt_core::gen::GeneratorSpec",
        "runt_core::gen::Shading",
        "runt_core::scene::SceneDesc",
        "runt_core::scene::EntityDesc",
        "runt_core::scene::TransformDesc",
        "runt_core::reflect::Vec3Def",
        "runt_core::reflect::OptVec3Def",
        "runt_core::reflect::TerrainParamsDef",
    ] {
        assert!(
            registry.get_with_type_path(path).is_some(),
            "{path} is not registered"
        );
    }
}

/// Every variant can be cloned through reflection and come back identical —
/// what an editor does when it snapshots a spec for undo or for a debounce.
#[test]
fn every_variant_survives_a_reflect_round_trip() {
    for kind in GeneratorSpec::KINDS {
        let spec = GeneratorSpec::default_of_kind(kind).unwrap();
        let dynamic = spec.to_dynamic();
        let back = <GeneratorSpec as bevy_reflect::FromReflect>::from_reflect(dynamic.as_ref())
            .unwrap_or_else(|| panic!("{kind:?} did not survive FromReflect"));
        assert_eq!(back, spec, "{kind:?} changed across a reflect round trip");
        // And the param key — the cache's identity — is unchanged, which is the
        // property that makes an editor edit indistinguishable from a file edit.
        assert_eq!(
            back.param_key(runt_mesh::Quality::FULL),
            spec.param_key(runt_mesh::Quality::FULL)
        );
    }
}

// ---------------------------------------------------------------------------

/// The named fields of an enum value's current variant, in declaration order.
fn field_names(value: &dyn Enum) -> Vec<&str> {
    (0..value.field_len())
        .map(|i| value.name_at(i).unwrap())
        .collect()
}
