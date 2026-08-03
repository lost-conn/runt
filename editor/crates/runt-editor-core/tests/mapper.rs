//! The reflection → widget mapping (DESIGN §10).
//!
//! These assert *structure*, never pixels: what controls a generator produces,
//! in what order, with what bounds, and that driving those controls changes the
//! generator. That is the whole contract between `runt-core`'s `Reflect` derives
//! and the rinch layer — if it holds, the panel is correct whatever it looks
//! like.

use glam::{Vec2, Vec3};
use runt_core::gen::{GeneratorSpec, Shading};
use runt_editor_core::mapper::{self, Edit, Widget};
use runt_editor_core::path::{FieldPath, Step};
use runt_mesh::TerrainParams;

/// A one-line summary of each control in a panel: `kind label path`.
/// Comparing these catches an accidental reordering or a widget silently
/// changing kind, which a field-by-field assertion would not.
fn outline(root: &Widget) -> Vec<String> {
    root.flatten()
        .into_iter()
        .map(|w| {
            let kind = match w {
                Widget::Float { .. } => "float",
                Widget::Int { .. } => "int",
                Widget::Seed { .. } => "seed",
                Widget::Bool { .. } => "bool",
                Widget::Text { .. } => "text",
                Widget::Vector { .. } => "vector",
                Widget::Variant { .. } => "variant",
                Widget::Group { .. } => "group",
                Widget::Unsupported { .. } => "UNSUPPORTED",
            };
            format!("{kind} {}{}", w.label(), w.path().display())
        })
        .collect()
}

fn sphere() -> GeneratorSpec {
    GeneratorSpec::UvSphere {
        radius: 0.9,
        rings: 24,
        sectors: 32,
        shading: Shading::Smooth(180.0),
        color: Some(Vec3::new(0.9, 0.35, 0.35)),
    }
}

fn terrain() -> GeneratorSpec {
    GeneratorSpec::Terrain(TerrainParams {
        seed: 20260731,
        size: Vec2::splat(40.0),
        amplitude: 1.2,
        octaves: 4,
        frequency: 0.055,
        lacunarity: 2.0,
        gain: 0.5,
        base_segments: 64,
        color: Some(Vec3::new(0.17, 0.21, 0.18)),
        // Carried, not edited: `TerrainParamsDef` marks `tint` `#[reflect(ignore)]`,
        // so it does not appear in the widget tree asserted below.
        tint: None,
    })
}

// ---------------------------------------------------------------------------
// UvSphere
// ---------------------------------------------------------------------------

#[test]
fn uv_sphere_produces_the_expected_widget_tree() {
    let panel = mapper::build(&sphere(), "ball");
    assert_eq!(
        outline(&panel),
        [
            // The generator itself is the variant selector.
            "variant ball",
            "float radius.radius",
            "int rings.rings",
            "int sectors.sectors",
            // `shading` is a nested enum: its own selector plus its payload.
            "variant shading.shading",
            "float .shading#0",
            // `color` is Option<Vec3>: a None/Some selector over a vector row.
            "variant color.color",
            "vector .color#0",
        ]
    );
}

#[test]
fn the_root_selector_offers_every_generator() {
    let panel = mapper::build(&sphere(), "ball");
    let Widget::Variant {
        selected, options, ..
    } = &panel
    else {
        panic!("the root of a generator panel is a variant selector")
    };
    assert_eq!(selected, "UvSphere");
    assert_eq!(options, GeneratorSpec::KINDS);
}

#[test]
fn numeric_widgets_carry_the_declared_bounds() {
    let panel = mapper::build(&sphere(), "ball");
    let widgets = panel.flatten();

    let radius = widgets
        .iter()
        .find(|w| matches!(w, Widget::Float { label, .. } if label == "radius"))
        .expect("a radius slider");
    let Widget::Float { value, range, .. } = radius else {
        unreachable!()
    };
    // f32 → f64: the panel widens, so compare with the tolerance that widening
    // costs rather than pretending 0.9f32 is 0.9f64.
    assert!((*value - 0.9).abs() < 1e-6, "radius read back as {value}");
    assert!(
        range.min > 0.0 && range.max >= 20.0,
        "radius bounds came from the attribute, got {range:?}"
    );

    let rings = widgets
        .iter()
        .find(|w| matches!(w, Widget::Int { label, .. } if label == "rings"))
        .expect("a rings stepper");
    let Widget::Int { value, range, .. } = rings else {
        unreachable!()
    };
    assert_eq!(*value, 24);
    assert!(range.min >= 2.0, "a sphere needs rings, got {range:?}");
}

#[test]
fn the_option_colour_is_a_selector_over_a_vector_row() {
    let panel = mapper::build(&sphere(), "ball");
    let color = panel
        .flatten()
        .into_iter()
        .find(|w| w.path() == &FieldPath::root().field("color"))
        .expect("a colour widget");

    let Widget::Variant {
        selected,
        options,
        fields,
        ..
    } = color
    else {
        panic!("Option<Vec3> maps to a variant selector, got {color:?}")
    };
    assert_eq!(selected, "Some");
    assert_eq!(options, &["None".to_string(), "Some".to_string()]);

    let [Widget::Vector { components, .. }] = &fields[..] else {
        panic!("the Some payload is a vector row, got {fields:?}")
    };
    assert_eq!(
        components.iter().map(|c| c.label.as_str()).collect::<Vec<_>>(),
        ["x", "y", "z"]
    );
    assert!((components[0].value - 0.9).abs() < 1e-6);
}

/// A `None` colour has no payload widgets at all — the panel shrinks rather than
/// showing three greyed-out numbers for a value that does not exist.
#[test]
fn a_none_colour_has_no_components() {
    let spec = GeneratorSpec::Cube {
        size: 1.0,
        shading: Shading::Flat,
        color: None,
    };
    let panel = mapper::build(&spec, "cube");
    let color = panel
        .flatten()
        .into_iter()
        .find(|w| w.path() == &FieldPath::root().field("color"))
        .unwrap();
    let Widget::Variant {
        selected, fields, ..
    } = color
    else {
        panic!()
    };
    assert_eq!(selected, "None");
    assert!(fields.is_empty());
}

// ---------------------------------------------------------------------------
// Terrain
// ---------------------------------------------------------------------------

#[test]
fn terrain_produces_the_expected_widget_tree() {
    let panel = mapper::build(&terrain(), "ground");
    assert_eq!(
        outline(&panel),
        [
            "variant ground",
            // A tuple variant with one payload: the params struct.
            "group #0",
            "seed seed#0.seed",
            "vector size#0.size",
            "float amplitude#0.amplitude",
            "int octaves#0.octaves",
            "float frequency#0.frequency",
            "float lacunarity#0.lacunarity",
            "float gain#0.gain",
            "int base segments#0.base_segments",
            "variant color#0.color",
            "vector #0.color#0",
        ]
    );
}

/// The one name-based rule in the mapper, and the reason it exists: a slider
/// over a `u64` seed would be useless.
#[test]
fn the_terrain_seed_is_a_seed_widget_not_an_integer_slider() {
    let panel = mapper::build(&terrain(), "ground");
    let seed = panel
        .flatten()
        .into_iter()
        .find(|w| w.label() == "seed")
        .expect("a seed widget");
    let Widget::Seed { value, path, .. } = seed else {
        panic!("expected a Seed widget, got {seed:?}")
    };
    assert_eq!(*value, 20260731);
    assert_eq!(path.0, vec![Step::Index(0), Step::Field("seed".into())]);
}

#[test]
fn terrain_bounds_come_from_the_remote_definition() {
    let panel = mapper::build(&terrain(), "ground");
    let gain = panel
        .flatten()
        .into_iter()
        .find(|w| w.label() == "gain")
        .unwrap();
    let Widget::Float { range, .. } = gain else {
        panic!()
    };
    assert_eq!((range.min, range.max), (0.0, 1.0));
}

/// Underscored field names read as words in the UI, but the *path* keeps the
/// real identifier — a label change must never break an edit.
#[test]
fn labels_are_prettified_but_paths_are_not() {
    let panel = mapper::build(&terrain(), "ground");
    let widget = panel
        .flatten()
        .into_iter()
        .find(|w| w.label() == "base segments")
        .expect("a base_segments widget");
    assert_eq!(
        widget.path().0,
        vec![Step::Index(0), Step::Field("base_segments".into())]
    );
}

// ---------------------------------------------------------------------------
// every generator maps to something
// ---------------------------------------------------------------------------

#[test]
fn no_generator_produces_an_unsupported_widget() {
    for kind in GeneratorSpec::KINDS {
        let spec = GeneratorSpec::default_of_kind(kind).unwrap();
        let panel = mapper::build(&spec, *kind);
        let unsupported: Vec<_> = panel
            .flatten()
            .into_iter()
            .filter(|w| matches!(w, Widget::Unsupported { .. }))
            .collect();
        assert!(
            unsupported.is_empty(),
            "{kind} has controls the mapper cannot render: {unsupported:?}"
        );
        assert!(
            panel.flatten().len() > 1,
            "{kind} produced an empty panel"
        );
    }
}

// ---------------------------------------------------------------------------
// applying edits
// ---------------------------------------------------------------------------

#[test]
fn a_float_edit_reaches_the_generator() {
    let mut spec = sphere();
    mapper::apply(
        &mut spec,
        &FieldPath::root().field("radius"),
        &Edit::Float(2.25),
    )
    .unwrap();
    let GeneratorSpec::UvSphere { radius, .. } = spec else {
        panic!()
    };
    assert_eq!(radius, 2.25);
}

/// A slider is `f64`; the field may be a `u32`. The narrowing rounds rather than
/// truncating, so 23.999 is 24 segments and not 23.
#[test]
fn an_integer_edit_rounds_rather_than_truncating() {
    let mut spec = sphere();
    mapper::apply(
        &mut spec,
        &FieldPath::root().field("rings"),
        &Edit::Float(23.999_999),
    )
    .unwrap();
    let GeneratorSpec::UvSphere { rings, .. } = spec else {
        panic!()
    };
    assert_eq!(rings, 24);
}

#[test]
fn an_integer_edit_cannot_wrap_negative() {
    let mut spec = sphere();
    mapper::apply(
        &mut spec,
        &FieldPath::root().field("rings"),
        &Edit::Float(-50.0),
    )
    .unwrap();
    let GeneratorSpec::UvSphere { rings, .. } = spec else {
        panic!()
    };
    assert_eq!(rings, 0, "clamped to the type's floor, not wrapped to u32::MAX");
}

#[test]
fn a_vector_component_edit_reaches_one_axis_only() {
    let mut spec = terrain();
    mapper::apply(
        &mut spec,
        &FieldPath::root().index(0).field("size").field("y"),
        &Edit::Float(64.0),
    )
    .unwrap();
    let GeneratorSpec::Terrain(params) = spec else {
        panic!()
    };
    assert_eq!(params.size, Vec2::new(40.0, 64.0));
}

#[test]
fn a_seed_edit_writes_the_full_width() {
    let mut spec = terrain();
    let huge = u64::MAX - 3;
    mapper::apply(
        &mut spec,
        &FieldPath::root().index(0).field("seed"),
        &Edit::Seed(huge),
    )
    .unwrap();
    let GeneratorSpec::Terrain(params) = spec else {
        panic!()
    };
    assert_eq!(params.seed, huge, "a seed must survive as a u64, not via f64");
}

#[test]
fn switching_a_nested_variant_works() {
    let mut spec = sphere();
    mapper::apply(
        &mut spec,
        &FieldPath::root().field("shading"),
        &Edit::Variant("Flat".into()),
    )
    .unwrap();
    let GeneratorSpec::UvSphere { shading, .. } = spec else {
        panic!()
    };
    assert_eq!(shading, Shading::Flat);
}

#[test]
fn switching_a_colour_on_and_off_works() {
    let mut spec = sphere();
    let path = FieldPath::root().field("color");

    mapper::apply(&mut spec, &path, &Edit::Variant("None".into())).unwrap();
    assert!(matches!(spec, GeneratorSpec::UvSphere { color: None, .. }));

    mapper::apply(&mut spec, &path, &Edit::Variant("Some".into())).unwrap();
    let GeneratorSpec::UvSphere { color, .. } = spec else {
        panic!()
    };
    assert_eq!(
        color,
        Some(Vec3::ZERO),
        "a re-enabled colour starts at the type's zero; the user drags from there"
    );
}

/// The root variant switch is the one place a zero-filled default would be
/// actively wrong (`Torus(0,0,0,0)` is not a torus), so it goes through
/// `GeneratorSpec::default_of_kind` instead.
#[test]
fn switching_the_generator_itself_uses_the_hand_written_defaults() {
    for kind in GeneratorSpec::KINDS {
        let spec = GeneratorSpec::default_of_kind(kind).unwrap();
        assert!(
            !spec.generate(runt_mesh::Quality::FULL).indices.is_empty(),
            "{kind}'s default must be a renderable mesh"
        );
    }
}

#[test]
fn an_edit_to_a_path_that_no_longer_exists_is_an_error_not_a_panic() {
    // `color` is None, so `.color#0` addresses nothing.
    let mut spec = GeneratorSpec::Cube {
        size: 1.0,
        shading: Shading::Flat,
        color: None,
    };
    let stale = FieldPath::root().field("color").index(0).field("x");
    let err = mapper::apply(&mut spec, &stale, &Edit::Float(1.0)).unwrap_err();
    assert!(err.contains("color"), "{err}");

    let err = mapper::apply(
        &mut spec,
        &FieldPath::root().field("no_such_field"),
        &Edit::Float(1.0),
    )
    .unwrap_err();
    assert!(err.contains("no_such_field"), "{err}");
}

/// Editing must not change what the value *is* — the param key is the cache's
/// identity, and an edit that produced a different key for the same numbers
/// would make every slider position a cache miss.
#[test]
fn an_edit_and_a_hand_written_value_are_indistinguishable() {
    let mut edited = sphere();
    mapper::apply(
        &mut edited,
        &FieldPath::root().field("radius"),
        &Edit::Float(1.5),
    )
    .unwrap();

    let authored = GeneratorSpec::UvSphere {
        radius: 1.5,
        rings: 24,
        sectors: 32,
        shading: Shading::Smooth(180.0),
        color: Some(Vec3::new(0.9, 0.35, 0.35)),
    };
    assert_eq!(edited, authored);
    assert_eq!(
        edited.param_key(runt_mesh::Quality::FULL),
        authored.param_key(runt_mesh::Quality::FULL)
    );
    assert_eq!(
        edited.generate(runt_mesh::Quality::FULL).content_hash(),
        authored.generate(runt_mesh::Quality::FULL).content_hash()
    );
}

/// A round trip through the mapper with no edits must leave the panel able to
/// reproduce every value it was built from.
#[test]
fn every_leaf_value_survives_being_read_and_written_back() {
    for kind in GeneratorSpec::KINDS {
        let original = GeneratorSpec::default_of_kind(kind).unwrap();
        let mut copy = original.clone();
        let panel = mapper::build(&original, *kind);

        for widget in panel.flatten() {
            let (path, edit) = match widget {
                Widget::Float { path, value, .. } => (path, Edit::Float(*value)),
                Widget::Int { path, value, .. } => (path, Edit::Int(*value)),
                Widget::Seed { path, value, .. } => (path, Edit::Seed(*value)),
                Widget::Bool { path, value, .. } => (path, Edit::Bool(*value)),
                Widget::Text { path, value, .. } => (path, Edit::Text(value.clone())),
                Widget::Vector { components, .. } => {
                    for c in components {
                        mapper::apply(&mut copy, &c.path, &Edit::Float(c.value)).unwrap();
                    }
                    continue;
                }
                _ => continue,
            };
            mapper::apply(&mut copy, path, &edit)
                .unwrap_or_else(|e| panic!("{kind}: writing back {path:?} failed: {e}"));
        }
        assert_eq!(copy, original, "{kind} changed under an identity round trip");
    }
}
