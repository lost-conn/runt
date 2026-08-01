//! Scene RON: load, save, round-trip (DESIGN §6).
//!
//! The scene file is the "save-as-params" pillar made concrete, so these tests
//! are as much about the *file* as about the loader: that `assets/demo.ron` is
//! the demo (not a stale copy of it), that a load→save→load cycle is a fixed
//! point, and that the sharing the file advertises actually happens.

use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};
use runt_core::ecs::{GeneratorRef, Spin, TerrainSurface};
use runt_core::gen::GeneratorSpec;
use runt_core::material::Material;
use runt_core::scene::{
    self, save_scene, LoadedScene, QualityPolicy, SceneDesc, SceneError, DEMO_EYE, DEMO_SCENE_RON,
    DEMO_SPIN,
};
use runt_core::{MeshRef, Sim, SimConfig, Transform};

/// Every drawable entity's (mesh handle, material, transform), in a stable
/// order — the fingerprint a round trip has to preserve.
fn world_fingerprint(world: &mut World) -> Vec<(u64, Material, Transform)> {
    let mut rows: Vec<(u64, Material, Transform)> = world
        .query::<(&MeshRef, &Material, &Transform)>()
        .iter(world)
        .map(|(mesh, material, transform)| (mesh.0 .0, *material, *transform))
        .collect();
    rows.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.2.translation.to_array().partial_cmp(&b.2.translation.to_array()).unwrap())
    });
    rows
}

fn sim_with(scene: &str) -> Sim {
    Sim::from_config(SimConfig::default().with_scene(scene))
}

// ---------------------------------------------------------------------------
// The demo scene
// ---------------------------------------------------------------------------

#[test]
fn the_demo_scene_is_the_demo() {
    let mut sim = Sim::new();
    let desc = scene::demo_scene();
    assert_eq!(desc.generators.len(), 6, "six generators");
    assert_eq!(desc.entities.len(), 7, "seven placements");

    // Seven entities plus the camera.
    let drawables = sim.world_mut().query::<&MeshRef>().iter(sim.world()).count();
    assert_eq!(drawables, 7);
    assert_eq!(sim.draw_list().len(), 7);

    // Constants the camera tests pin down have to be what the file says.
    assert_eq!(desc.camera.eye, DEMO_EYE);
    let spinner = desc
        .entities
        .iter()
        .find(|e| e.name.as_deref() == Some("spinner"))
        .expect("the demo has a spinner");
    assert_eq!(spinner.spin.expect("it spins").rad_per_sec, DEMO_SPIN);
    assert_eq!(sim.demo_entity(), sim.scene_entity("spinner").expect("named"));
}

#[test]
fn one_generator_entry_means_one_mesh_for_both_spheres() {
    // The file references "ball" twice on purpose. Content addressing has to
    // collapse them: seven entities, six meshes, one pair of GPU buffers for the
    // pair. This is DESIGN §5's "determinism paying rent", checked at the level
    // a scene author can actually see it.
    let sim = Sim::new();
    assert_eq!(sim.mesh_library().len(), 6, "seven entities, six meshes");

    let ball = sim.scene_entity("ball").expect("ball");
    let twin = sim.scene_entity("ball_twin").expect("twin");
    let mesh_of = |sim: &Sim, e| sim.world().get::<MeshRef>(e).expect("MeshRef").0;
    assert_eq!(mesh_of(&sim, ball), mesh_of(&sim, twin), "shared geometry");

    // ...and they still look different, because placement and material are not
    // shape.
    let t_of = |sim: &Sim, e| *sim.world().get::<Transform>(e).expect("Transform");
    assert_ne!(t_of(&sim, ball).scale, t_of(&sim, twin).scale);
    assert_ne!(
        sim.world().get::<Material>(ball).expect("mat").variant,
        sim.world().get::<Material>(twin).expect("mat").variant
    );
}

#[test]
fn entities_remember_which_generator_built_them() {
    // `MeshRef` is a content hash and stays one; `GeneratorRef` is the
    // provenance a quality change or an editor tweak needs in order to
    // regenerate an entity without reloading the scene.
    let mut sim = Sim::new();
    let ball = sim.scene_entity("ball").expect("ball");
    let twin = sim.scene_entity("ball_twin").expect("twin");
    let gen_of = |sim: &Sim, e| sim.world().get::<GeneratorRef>(e).expect("GeneratorRef").clone();

    assert_eq!(gen_of(&sim, ball).name, "ball");
    assert_eq!(gen_of(&sim, ball), gen_of(&sim, twin), "same generator, same key");

    // The param key really is the cache key that produced the mesh.
    let spec = scene::demo_generator("ball");
    assert_eq!(
        gen_of(&sim, ball).param_key,
        spec.param_key(sim.quality_tier().quality())
    );

    // Every scene entity has one.
    let missing = sim
        .world_mut()
        .query_filtered::<Entity, (With<MeshRef>, Without<GeneratorRef>)>()
        .iter(sim.world())
        .count();
    assert_eq!(missing, 0);
}

#[test]
fn the_demos_props_rest_on_the_terrain_rather_than_in_it() {
    // The demo's ground is a height field now, so "the floor" is seed-dependent.
    // If a future seed or amplitude change buries the spike, this fails here
    // rather than in a screenshot nobody looks at.
    let sim = Sim::new();
    let surface = *sim
        .world()
        .get::<TerrainSurface>(sim.scene_entity("ground").expect("ground"))
        .expect("terrain carries its field");
    let origin = sim
        .world()
        .get::<Transform>(sim.scene_entity("ground").expect("ground"))
        .expect("transform")
        .translation;

    // (entity, footprint radius, lowest point of its geometry in world Y)
    for (name, radius, underside) in [
        ("ball", 0.9f32, -1.2f32),
        ("post", 0.35, -1.2),
        ("spike", 0.6, -1.2),
        ("ring", 0.92, -0.52),
        ("ball_twin", 0.45, 0.05),
        ("spinner", 0.8, -0.3),
    ] {
        let center = sim
            .world()
            .get::<Transform>(sim.scene_entity(name).expect(name))
            .expect("transform")
            .translation;
        let mut peak = f32::MIN;
        for i in 0..=16 {
            for j in 0..=16 {
                let x = center.x - radius + i as f32 * radius / 8.0;
                let z = center.z - radius + j as f32 * radius / 8.0;
                peak = peak.max(surface.height_world(origin, x, z));
            }
        }
        assert!(
            peak < underside,
            "{name} is buried: terrain peaks at {peak:.3}, its underside is {underside:.3}"
        );
    }
}

#[test]
fn the_terrain_field_is_reachable_the_way_physics_will_reach_it() {
    // The step-5 seam, exercised as a physics system would: query
    // (&TerrainSurface, &Transform), sample the field in world coordinates,
    // never touch the mesh.
    let mut sim = Sim::new();
    let mut query = sim.world_mut().query::<(&TerrainSurface, &Transform)>();
    let found: Vec<(TerrainSurface, Transform)> = query
        .iter(sim.world())
        .map(|(surface, transform)| (*surface, *transform))
        .collect();
    assert_eq!(found.len(), 1, "exactly one terrain in the demo");
    let (surface, transform) = found[0];

    let (h, grad) = surface.sample_world(transform.translation, 3.0, -4.0);
    assert_eq!(h, surface.height_world(transform.translation, 3.0, -4.0));
    assert_eq!(grad, surface.gradient_world(transform.translation, 3.0, -4.0));
    assert!(surface.normal_world(transform.translation, 3.0, -4.0).y > 0.0);
    assert!(surface.contains_world(transform.translation, 19.0, -19.0));
    assert!(!surface.contains_world(transform.translation, 21.0, 0.0));

    // And the mesh really is a view of that field: every ground vertex, put back
    // into world space, sits exactly on it.
    let ground = sim.scene_entity("ground").expect("ground");
    let handle = sim.world().get::<MeshRef>(ground).expect("MeshRef").0;
    let mesh = sim.mesh_library().get(handle).expect("in the library");
    for p in mesh.positions.iter().step_by(37) {
        let world = transform.translation + *p;
        assert!(
            (world.y - surface.height_world(transform.translation, world.x, world.z)).abs() < 1e-5,
            "vertex {p:?} is off the field"
        );
    }
}

// ---------------------------------------------------------------------------
// Round trip
// ---------------------------------------------------------------------------

#[test]
fn load_save_load_is_a_fixed_point() {
    let mut a = Sim::new();
    let saved = save_scene(a.world()).expect("save");

    let mut b = sim_with(&saved);
    let saved_again = save_scene(b.world()).expect("save again");

    // Same text, therefore same semantic content: the second save has nothing
    // left to normalize.
    assert_eq!(saved, saved_again, "saving is idempotent");

    // Same world, too.
    assert_eq!(world_fingerprint(a.world_mut()), world_fingerprint(b.world_mut()));
    assert_eq!(a.mesh_library().len(), b.mesh_library().len());
    assert_eq!(a.draw_list().len(), b.draw_list().len());

    // And the descriptions agree, field for field.
    let da = scene::scene_desc(a.world()).expect("desc");
    let db = scene::scene_desc(b.world()).expect("desc");
    assert_eq!(da, db);
}

#[test]
fn a_saved_scene_re_parses_into_the_same_description() {
    let sim = Sim::new();
    let saved = save_scene(sim.world()).expect("save");
    let reparsed = scene::parse_scene(&saved).expect("re-parse");
    assert_eq!(reparsed, scene::scene_desc(sim.world()).expect("desc"));

    // The authored file and the saved file describe the same scene even though
    // one has comments and the other does not.
    let authored = scene::parse_scene(DEMO_SCENE_RON).expect("authored parses");
    assert_eq!(authored.generators, reparsed.generators);
    assert_eq!(authored.entities, reparsed.entities);
    assert_eq!(authored.camera, reparsed.camera);
}

#[test]
fn saving_preserves_hand_authored_euler_angles_but_records_real_movement() {
    // A save must not rewrite `Euler((90, 0, 0))` into a quaternion just because
    // it round-tripped — that would churn a hand-edited file on every save. It
    // must record an actual change, though.
    let mut sim = Sim::new();
    let ring = scene::parse_scene(&save_scene(sim.world()).expect("save"))
        .expect("parse")
        .entities
        .into_iter()
        .find(|e| e.name.as_deref() == Some("ring"))
        .expect("ring");
    assert_eq!(
        ring.transform.rotation,
        runt_core::scene::RotationDesc::Euler(Vec3::new(90.0, 0.0, 0.0)),
        "an untouched rotation keeps its authored form"
    );

    // Now move something and save again.
    let spike = sim.scene_entity("spike").expect("spike");
    sim.world_mut().get_mut::<Transform>(spike).expect("transform").translation =
        Vec3::new(4.0, 1.0, -2.0);
    let saved = scene::parse_scene(&save_scene(sim.world()).expect("save")).expect("parse");
    let moved = saved
        .entities
        .iter()
        .find(|e| e.name.as_deref() == Some("spike"))
        .expect("spike");
    assert_eq!(moved.transform.translation, Vec3::new(4.0, 1.0, -2.0));
}

#[test]
fn ticking_the_spinner_shows_up_in_a_save() {
    let mut sim = Sim::new();
    sim.update(0.0);
    for i in 1..=30 {
        sim.update(i as f64 * runt_core::TICK_DT);
    }
    let saved = scene::parse_scene(&save_scene(sim.world()).expect("save")).expect("parse");
    let spinner = saved
        .entities
        .iter()
        .find(|e| e.name.as_deref() == Some("spinner"))
        .expect("spinner");
    let live = sim
        .world()
        .get::<Transform>(sim.demo_entity())
        .expect("transform")
        .rotation;
    assert!(live.angle_between(Quat::IDENTITY) > 0.1, "it did spin");
    assert!(
        spinner.transform.rotation.quat().abs_diff_eq(live, 1e-6),
        "a save captures where things actually are"
    );
}

#[test]
fn a_world_with_no_scene_cannot_be_saved() {
    let sim = Sim::without_scene();
    assert!(sim.world().get_resource::<LoadedScene>().is_none());
    assert!(matches!(save_scene(sim.world()), Err(SceneError::NoSceneLoaded)));
    assert!(sim.try_demo_entity().is_none());
    assert!(sim.mesh_library().is_empty(), "nothing generated");
}

// ---------------------------------------------------------------------------
// Quality
// ---------------------------------------------------------------------------

#[test]
fn quality_changes_the_geometry_and_nothing_else() {
    // DESIGN §6: quality selects *data*, never a different scene. The structure
    // — how many entities, which generator each uses, where they sit — has to be
    // identical, while the terrain (and every other tessellated mesh) is not.
    let full = Sim::from_config(SimConfig::default().with_quality(1.0));
    let low = Sim::from_config(SimConfig::default().with_quality(0.25));

    let terrain_handle = |sim: &Sim| {
        sim.world()
            .get::<MeshRef>(sim.scene_entity("ground").expect("ground"))
            .expect("MeshRef")
            .0
    };
    assert_ne!(
        terrain_handle(&full),
        terrain_handle(&low),
        "a different tier is a different terrain mesh"
    );
    let verts = |sim: &Sim| {
        sim.mesh_library()
            .get(terrain_handle(sim))
            .expect("in library")
            .vertex_count()
    };
    assert!(verts(&full) > verts(&low), "and a coarser one at the low tier");

    let structure = |sim: &Sim| {
        let loaded = sim.world().get_resource::<LoadedScene>().expect("loaded");
        loaded
            .desc
            .entities
            .iter()
            .zip(&loaded.spawned)
            .map(|(desc, &e)| {
                (
                    desc.generator.clone(),
                    desc.name.clone(),
                    *sim.world().get::<Transform>(e).expect("transform"),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(structure(&full), structure(&low), "structure is tier-invariant");

    // The §9 clincher: the *surface* is the same at both tiers, even though the
    // meshes are not. Physics cannot tell which device it is running on.
    let field = |sim: &Sim| {
        sim.world()
            .get::<TerrainSurface>(sim.scene_entity("ground").expect("ground"))
            .expect("terrain")
            .field
    };
    assert_eq!(field(&full), field(&low));
    for (x, z) in [(0.0, 0.0), (7.5, -3.25), (-11.0, 14.0)] {
        assert_eq!(
            field(&full).height(x, z).to_bits(),
            field(&low).height(x, z).to_bits()
        );
    }
}

#[test]
fn a_per_generator_policy_overrides_the_tier() {
    let tier = runt_core::QualityTier(0.5);
    assert_eq!(QualityPolicy::Inherit.resolve(tier).0, 0.5);
    assert_eq!(QualityPolicy::Scaled(0.5).resolve(tier).0, 0.25);
    assert_eq!(QualityPolicy::Fixed(1.0).resolve(tier).0, 1.0);

    // Fixed geometry really does ignore the session tier.
    let src = r#"(
        generators: [
            (name: "pinned", spec: UvSphere(radius: 1.0, rings: 24, sectors: 32), quality: Fixed(1.0)),
            (name: "loose",  spec: UvSphere(radius: 2.0, rings: 24, sectors: 32)),
        ],
        entities: [
            (generator: "pinned"),
            (generator: "loose", transform: (translation: (5.0, 0.0, 0.0))),
        ],
    )"#;
    let a = Sim::from_config(SimConfig::default().with_scene(src).with_quality(1.0));
    let b = Sim::from_config(SimConfig::default().with_scene(src).with_quality(0.25));
    let handles = |sim: &Sim| {
        let loaded = sim.world().get_resource::<LoadedScene>().expect("loaded");
        loaded
            .spawned
            .iter()
            .map(|&e| sim.world().get::<MeshRef>(e).expect("MeshRef").0)
            .collect::<Vec<_>>()
    };
    assert_eq!(handles(&a)[0], handles(&b)[0], "Fixed ignores the tier");
    assert_ne!(handles(&a)[1], handles(&b)[1], "Inherit follows it");
}

// ---------------------------------------------------------------------------
// Format ergonomics and failure modes
// ---------------------------------------------------------------------------

#[test]
fn a_minimal_scene_is_three_lines() {
    // The hand-editability claim, tested: everything except the generator's own
    // shape params has a default.
    let mut sim = sim_with(
        r#"(
            generators: [(name: "b", spec: Cube(size: 1.0))],
            entities: [(generator: "b")],
        )"#,
    );
    assert_eq!(sim.draw_list().len(), 1);
    assert_eq!(sim.mesh_library().len(), 1);
    let e = sim
        .world()
        .get_resource::<LoadedScene>()
        .expect("loaded")
        .spawned[0];
    assert_eq!(*sim.world().get::<Transform>(e).expect("t"), Transform::IDENTITY);
    assert_eq!(*sim.world().get::<Material>(e).expect("m"), Material::vertex_colored());
    assert!(sim.world().get::<Spin>(e).is_none());
    assert!(sim.world().get::<runt_core::Interpolated>(e).is_none());
    // The default camera still gives the world something to render from.
    assert!(sim.frame_params(1.0).is_some());
}

#[test]
fn a_bad_generator_reference_fails_without_spawning_anything() {
    let mut world = World::new();
    let err = scene::load_scene(
        &mut world,
        r#"(
            generators: [(name: "a", spec: Cube(size: 1.0))],
            entities: [(generator: "a"), (generator: "typo")],
        )"#,
    )
    .expect_err("unknown generator must fail");
    assert!(matches!(err, SceneError::UnknownGenerator { entity: 1, .. }), "{err}");
    assert_eq!(
        world.query::<&MeshRef>().iter(&world).count(),
        0,
        "validation runs before anything is spawned"
    );
}

#[test]
fn a_bad_follow_target_fails_the_same_way() {
    let mut world = World::new();
    let err = scene::load_scene(
        &mut world,
        r#"(
            generators: [(name: "a", spec: Cube(size: 1.0))],
            entities: [(name: Some("real"), generator: "a")],
            camera: (eye: (0.0, 2.0, 5.0), follow: Some((entity: "ghost", offset: (0.0, 2.0, 5.0), stiffness: 2.0))),
        )"#,
    )
    .expect_err("unknown follow target must fail");
    assert!(matches!(err, SceneError::UnknownFollowTarget(_)), "{err}");
}

#[test]
fn broken_ron_is_an_error_not_a_panic() {
    assert!(matches!(scene::parse_scene("(generators: ["), Err(SceneError::Parse(_))));
    // And a sim configured with it still comes up — logged, empty, running.
    let mut sim = sim_with("nonsense");
    assert_eq!(sim.draw_list().len(), 0);
    assert_eq!(sim.tick_count(), 0);
    sim.update(0.0);
    sim.update(1.0 / 60.0);
    assert_eq!(sim.tick_count(), 1, "a broken scene must not stop the sim");
}

#[test]
fn loading_twice_replaces_rather_than_accumulates() {
    let mut sim = Sim::new();
    let before = sim.draw_list().len();
    let desc: SceneDesc = scene::demo_scene();
    scene::spawn_scene(sim.world_mut(), desc).expect("reload");
    assert_eq!(sim.draw_list().len(), before, "no duplicated entities");
    assert_eq!(sim.mesh_library().len(), 6, "and no duplicated meshes");
    // The reload was served entirely from layer A.
    assert_eq!(sim.cache_stats().generated, 6);
    assert_eq!(sim.cache_stats().memo_hits, 6);
    assert!(sim.frame_params(1.0).is_some(), "and there is still one camera");
}

#[test]
fn a_scene_ron_is_readable_when_written_back() {
    // Not a formatting shrine — just a guard that `save_scene` keeps producing
    // something a person would be willing to edit.
    let saved = save_scene(Sim::new().world()).expect("save");
    assert!(saved.contains("generators: ["), "named fields survive:\n{saved}");
    assert!(saved.contains("Terrain(("), "the terrain spec is spelled out");
    assert!(!saved.contains("Vec3("), "vectors stay as bare tuples");
    assert!(saved.lines().count() > 40, "pretty-printed, not one line");
}

#[test]
fn the_scene_files_generators_are_what_the_demo_helpers_return() {
    // `scene::ball_mesh()` and friends read the RON rather than restating it, so
    // this mostly guards the names — but a renamed generator entry silently
    // breaking every test helper is exactly the kind of thing to catch here.
    for name in ["ground", "ball", "post", "spike", "ring", "twisted_box"] {
        let spec = scene::demo_generator(name);
        assert!(!spec.generate(runt_core::Quality::FULL).is_empty(), "{name}");
    }
    assert!(matches!(
        scene::demo_generator("ground"),
        GeneratorSpec::Terrain(_)
    ));
    assert_eq!(scene::demo_terrain_params().seed, 20260731);
}
