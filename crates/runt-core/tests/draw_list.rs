//! Draw-list extraction and ordering (DESIGN §5).
//!
//! No GPU: the sort is the contract, and the contract is that the same world
//! state always produces the same command stream. Pipeline swaps are the
//! expensive change, so variant sorts first; mesh binds next; entity last as a
//! tie-break, because "whatever order the query happened to yield" is not an
//! ordering anyone can reason about.

use bevy_ecs::prelude::*;
use glam::{Quat, Vec3, Vec4};
use runt_core::draw::{build_draw_list, cull_draw_list, resolve_variant, Aabb, DrawItem, Frustum};
use runt_core::registry::MeshHandle;
use runt_core::texture::{TextureHandle, TextureLibrary};
use runt_core::{Interpolated, Material, MaterialVariant, MeshRef, Transform, Visibility};

/// Spawn a drawable with an explicit mesh handle and variant.
fn spawn(world: &mut World, mesh: u64, variant: MaterialVariant, x: f32) -> Entity {
    world
        .spawn((
            MeshRef(MeshHandle(mesh)),
            Material {
                base_color: Vec4::ONE,
                params: Vec4::ZERO,
                texture: None,
                variant,
            },
            Transform::from_translation(Vec3::new(x, 0.0, 0.0)),
        ))
        .id()
}

fn keys(items: &[DrawItem]) -> Vec<(u32, u64)> {
    items.iter().map(|i| (i.variant.bits(), i.mesh.0)).collect()
}

#[test]
fn draws_sort_by_variant_then_mesh_then_entity() {
    let mut world = World::new();

    // Deliberately interleaved on spawn: every adjacent pair disagrees on at
    // least one sort field, so a stable-but-unsorted result cannot pass.
    let vc = MaterialVariant::VERTEX_COLOR;
    let none = MaterialVariant::NONE;
    let e_vc_9 = spawn(&mut world, 9, vc, 0.0);
    let e_none_5 = spawn(&mut world, 5, none, 1.0);
    let e_vc_2 = spawn(&mut world, 2, vc, 2.0);
    let e_none_9 = spawn(&mut world, 9, none, 3.0);
    let e_vc_2_again = spawn(&mut world, 2, vc, 4.0);
    let e_none_5_again = spawn(&mut world, 5, none, 5.0);

    let items = build_draw_list(&mut world, 0.0);
    assert_eq!(items.len(), 6);

    assert_eq!(
        keys(&items),
        vec![
            (0, 5),
            (0, 5),
            (0, 9),
            (1, 2),
            (1, 2),
            (1, 9),
        ],
        "variant first, then mesh"
    );

    // Ties broken by entity, ascending — the deterministic part.
    let order: Vec<Entity> = items.iter().map(|i| i.entity).collect();
    assert_eq!(
        order,
        vec![e_none_5, e_none_5_again, e_none_9, e_vc_2, e_vc_2_again, e_vc_9]
    );
    for pair in items.windows(2) {
        assert!(
            pair[0].sort_key() <= pair[1].sort_key(),
            "sort keys must be non-decreasing"
        );
    }
}

#[test]
fn spawn_order_does_not_change_the_draw_order() {
    // Two worlds, same content, opposite spawn order. The *keys* must match;
    // only the entity tie-break may differ, and it must be internally sorted.
    let build = |reverse: bool| {
        let mut world = World::new();
        let mut specs = vec![
            (7u64, MaterialVariant::VERTEX_COLOR),
            (3, MaterialVariant::NONE),
            (7, MaterialVariant::NONE),
            (1, MaterialVariant::VERTEX_COLOR),
        ];
        if reverse {
            specs.reverse();
        }
        for (i, (mesh, variant)) in specs.into_iter().enumerate() {
            spawn(&mut world, mesh, variant, i as f32);
        }
        keys(&build_draw_list(&mut world, 0.0))
    };
    assert_eq!(build(false), build(true));
    assert_eq!(build(false), vec![(0, 3), (0, 7), (1, 1), (1, 7)]);
}

#[test]
fn spawn_order_does_not_change_the_draw_order_with_culling_on() {
    // The same claim as above, now that a camera gets to remove things (D5) and
    // a component gets to hide them (D4). Both are order-preserving filters
    // over the sorted list, so the retained set must still be a pure function
    // of the world — not of the order it was built in.
    let pose = Transform::looking_at(Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO, Vec3::Y);
    let view_proj = runt_core::Camera::default().view_proj(pose.matrix(), 1.0);
    let frustum = Frustum::from_view_proj(&view_proj);
    let unit = Aabb {
        min: Vec3::splat(-0.5),
        max: Vec3::splat(0.5),
    };

    // (mesh, variant, x, visible). Two are parked far off to the side and one
    // is hidden outright, so all three filters have something to do.
    let mut specs = vec![
        (7u64, MaterialVariant::VERTEX_COLOR, 0.0f32, true),
        (3, MaterialVariant::NONE, 900.0, true),
        (7, MaterialVariant::NONE, -1.0, false),
        (1, MaterialVariant::VERTEX_COLOR, 1.5, true),
        (3, MaterialVariant::VERTEX_COLOR, -900.0, true),
        (1, MaterialVariant::NONE, -2.0, true),
    ];
    let build = |specs: &[(u64, MaterialVariant, f32, bool)]| {
        let mut world = World::new();
        for (mesh, variant, x, visible) in specs {
            let entity = spawn(&mut world, *mesh, *variant, *x);
            if !visible {
                world.entity_mut(entity).insert(Visibility::HIDDEN);
            }
        }
        let mut items = build_draw_list(&mut world, 0.0);
        cull_draw_list(&mut items, &frustum, |_| Some(unit));
        // Keys plus position: the entity ids differ between two worlds, the
        // content does not.
        items
            .iter()
            .map(|i| (i.variant.bits(), i.mesh.0, i.model.w_axis.x))
            .collect::<Vec<_>>()
    };

    let forward = build(&specs);
    assert_eq!(forward.len(), 3, "one hidden, two off screen, three drawn");
    specs.reverse();
    assert_eq!(build(&specs), forward);
    // Rotated rather than reversed — a third order, same answer.
    specs.rotate_left(2);
    assert_eq!(build(&specs), forward);

    // And the answer itself, so a change of *policy* has to be deliberate.
    assert_eq!(
        forward,
        vec![(0, 1, -2.0), (1, 1, 1.5), (1, 7, 0.0)],
        "variant, then mesh, then entity — with the culled and hidden gone"
    );
}

#[test]
fn interpolated_entities_blend_and_the_rest_do_not() {
    let mut world = World::new();

    let still = spawn(&mut world, 1, MaterialVariant::NONE, 0.0);

    let moving = world
        .spawn((
            MeshRef(MeshHandle(1)),
            Material::vertex_colored(),
            Transform::from_translation(Vec3::new(10.0, 0.0, 0.0)),
            Interpolated {
                prev_translation: Vec3::ZERO,
                prev_rotation: Quat::IDENTITY,
                prev_scale: Vec3::ONE,
            },
        ))
        .id();

    let at = |world: &mut World, alpha: f32, entity: Entity| {
        build_draw_list(world, alpha)
            .into_iter()
            .find(|i| i.entity == entity)
            .expect("entity in draw list")
            .model
            .w_axis
            .x
    };

    // Halfway between the previous tick's 0 and this tick's 10.
    assert!((at(&mut world, 0.5, moving) - 5.0).abs() < 1e-5);
    assert!((at(&mut world, 0.0, moving) - 0.0).abs() < 1e-5);
    assert!((at(&mut world, 1.0, moving) - 10.0).abs() < 1e-5);

    // An entity with no `Interpolated` ignores alpha entirely.
    for alpha in [0.0, 0.5, 1.0] {
        assert_eq!(at(&mut world, alpha, still), 0.0);
    }
}

// ---------------------------------------------------------------------------
// §7's live/baked gate
// ---------------------------------------------------------------------------

const TEX: MaterialVariant = MaterialVariant::TEXTURE;
const LIVE: MaterialVariant = MaterialVariant::LIVE_TEX;
const VC: MaterialVariant = MaterialVariant::VERTEX_COLOR;
const NM: MaterialVariant = MaterialVariant::NORMAL_MAP;

#[test]
fn the_two_texture_paths_are_mutually_exclusive_on_a_draw() {
    // Whatever goes in, exactly one of the two bits comes out — never both,
    // never neither. Live evaluates the spec and never reads the bake, so a
    // draw carrying both would be paying for a pipeline half of which is dead.
    for authored in [
        MaterialVariant::NONE,
        TEX,
        LIVE,
        TEX | LIVE,
        TEX | NM,
        LIVE | NM,
        TEX | LIVE | VC | NM,
    ] {
        for gate in [false, true] {
            let got = resolve_variant(authored, true, gate);
            assert!(
                got.contains(TEX) != got.contains(LIVE),
                "authored {:#07b} + gate {gate} gave {:#07b}",
                authored.bits(),
                got.bits()
            );
            // Everything else survives untouched.
            let others = MaterialVariant::from_bits(
                got.bits() & !(TEX.bits() | LIVE.bits()),
            );
            let authored_others = MaterialVariant::from_bits(
                authored.bits() & !(TEX.bits() | LIVE.bits()),
            );
            assert_eq!(others, authored_others, "the gate touched an unrelated bit");
        }
    }
}

#[test]
fn the_gate_promotes_but_a_material_that_asked_for_live_keeps_it() {
    assert_eq!(resolve_variant(TEX | NM, true, false), TEX | NM);
    assert_eq!(resolve_variant(TEX | NM, true, true), LIVE | NM);
    // A scene file that asked for live is not demoted by the global default —
    // v1 has no perf tier to demote *against* (DESIGN §11's probe is future
    // work), so the gate can only ever say yes to more.
    assert_eq!(resolve_variant(LIVE | NM, true, false), LIVE | NM);
    assert_eq!(resolve_variant(LIVE | NM, true, true), LIVE | NM);
}

#[test]
fn an_untextured_material_is_returned_untouched() {
    // No handle means no bake to sample and no spec to evaluate; both bits are
    // already inert, and rewriting them would be this function having an
    // opinion about a draw it has no business touching.
    for gate in [false, true] {
        assert_eq!(resolve_variant(VC, false, gate), VC);
        assert_eq!(resolve_variant(MaterialVariant::NONE, false, gate), MaterialVariant::NONE);
        assert_eq!(resolve_variant(TEX | NM, false, gate), TEX | NM);
    }
}

/// A textured drawable, plus a library the gate can be flipped on.
fn textured_world(authored: MaterialVariant) -> World {
    let mut world = World::new();
    world.insert_resource(TextureLibrary::new());
    world.spawn((
        MeshRef(MeshHandle(1)),
        Material {
            base_color: Vec4::ONE,
            params: Vec4::ZERO,
            texture: Some(TextureHandle(0xABCD)),
            variant: authored,
        },
        Transform::IDENTITY,
    ));
    world
}

#[test]
fn the_toggle_switches_the_variant_bit_in_the_draw_list() {
    let mut world = textured_world(TEX | NM);

    let baked = build_draw_list(&mut world, 0.0);
    assert_eq!(baked[0].variant, TEX | NM, "off is baked, and off is default");

    world
        .resource_mut::<TextureLibrary>()
        .set_live_textures(true);
    let live = build_draw_list(&mut world, 0.0);
    assert_eq!(live[0].variant, LIVE | NM);

    // …and back, byte for byte. The whole value of the toggle is that flipping
    // it twice is a no-op, so an A/B comparison is comparing two renders of one
    // scene rather than two scenes.
    world
        .resource_mut::<TextureLibrary>()
        .set_live_textures(false);
    assert_eq!(build_draw_list(&mut world, 0.0), baked);
}

#[test]
fn a_world_with_no_texture_library_draws_baked() {
    // A bare `World` (every unit test in this file) has no library and no
    // textures, so the flag it would have carried cannot matter — but the
    // extraction must not panic reaching for a resource that is not there.
    let mut world = World::new();
    world.spawn((
        MeshRef(MeshHandle(1)),
        Material {
            base_color: Vec4::ONE,
            params: Vec4::ZERO,
            texture: Some(TextureHandle(7)),
            variant: TEX,
        },
        Transform::IDENTITY,
    ));
    assert_eq!(build_draw_list(&mut world, 0.0)[0].variant, TEX);
}

#[test]
fn entities_without_a_mesh_or_material_are_not_drawn() {
    let mut world = World::new();
    spawn(&mut world, 1, MaterialVariant::NONE, 0.0);
    // A camera-like entity: transform but no geometry.
    world.spawn(Transform::IDENTITY);
    // Geometry with no material is not a draw either — the variant is what
    // picks the pipeline, so there is nothing to draw it with.
    world.spawn((MeshRef(MeshHandle(2)), Transform::IDENTITY));

    assert_eq!(build_draw_list(&mut world, 0.0).len(), 1);
}
