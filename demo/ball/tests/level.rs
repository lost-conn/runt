//! `assets/level1.ron` is content, and content can be wrong (DESIGN §12 step 6).
//!
//! Everything a hand-placed coordinate could get wrong is re-derived here from
//! the *same analytic field* the game rolls on: a pickup floating out of reach
//! or buried in a hill, a spawn point on a slope the ball slides off before the
//! player touches a key, a post standing in mid-air, an obstacle sitting inside
//! a collectible. None of that is visible in a diff of the RON, and all of it is
//! a one-line failure here.
//!
//! The field is not restated — it is read out of the scene file's own generator
//! entry, so a seed or amplitude change is picked up automatically and these
//! numbers cannot drift away from what ships.

use glam::{Vec2, Vec3};
use runt_core::gen::GeneratorSpec;
use runt_core::scene::{parse_scene, EntityDesc, SceneDesc};
use runt_core::{HeightField, TerrainParams};

use runt_ball::game::{Pickup, GameState, PICKUP_GENERATOR, PLAYER_ENTITY};

/// Slopes a jumpless ball must be able to sit on and climb off. `tan(30°)`.
const MAX_PICKUP_SLOPE: f32 = 0.577_350_3;
/// The spawn wants to be genuinely level, not merely climbable. `tan(5°)`.
const MAX_SPAWN_SLOPE: f32 = 0.087_488_66;

/// Clearance a pickup's centre must have over the ground under it.
const MIN_CLEARANCE: f32 = 0.3;
const MAX_CLEARANCE: f32 = 1.5;

const BALL_RADIUS: f32 = 0.5;
const POST_HALF_HEIGHT: f32 = 1.0;
/// Pickup trigger radius, from the RON.
const PICKUP_TRIGGER: f32 = 0.7;

fn scene() -> SceneDesc {
    parse_scene(runt_ball::LEVEL1_RON).expect("level1.ron must parse")
}

fn terrain(scene: &SceneDesc) -> TerrainParams {
    let entry = scene
        .generators
        .iter()
        .find(|g| g.name == "ground")
        .expect("level1.ron defines a `ground` generator");
    match &entry.spec {
        GeneratorSpec::Terrain(params) => *params,
        other => panic!("`ground` is {}, not Terrain", other.kind()),
    }
}

fn field(scene: &SceneDesc) -> HeightField {
    terrain(scene).field()
}

fn entities<'a>(scene: &'a SceneDesc, generator: &str) -> Vec<&'a EntityDesc> {
    scene
        .entities
        .iter()
        .filter(|e| e.generator == generator)
        .collect()
}

fn named<'a>(scene: &'a SceneDesc, name: &str) -> &'a EntityDesc {
    scene
        .entities
        .iter()
        .find(|e| e.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("level1.ron names no entity {name:?}"))
}

fn horizontal(a: Vec3, b: Vec3) -> f32 {
    Vec2::new(a.x - b.x, a.z - b.z).length()
}

// ---------------------------------------------------------------------------
// The file itself
// ---------------------------------------------------------------------------

#[test]
fn the_level_parses_and_holds_what_the_game_expects() {
    let scene = scene();
    assert_eq!(
        entities(&scene, PICKUP_GENERATOR).len(),
        12,
        "twelve collectibles"
    );
    assert_eq!(entities(&scene, "post").len(), 7, "seven obstacle posts");
    assert_eq!(entities(&scene, "ground").len(), 1);
    assert_eq!(entities(&scene, "player").len(), 1);

    // The camera follows the player, and the player is the ball the game drives.
    let follow = scene
        .camera
        .follow
        .as_ref()
        .expect("the level uses a follow camera");
    assert_eq!(follow.entity, PLAYER_ENTITY);
    let player = named(&scene, PLAYER_ENTITY);
    assert!(player.ball.is_some() && player.ball_controller.is_some());

    // Every pickup is a trigger, or it would knock the ball around instead of
    // being collected; every post is solid, or it would be a ghost.
    for pickup in entities(&scene, PICKUP_GENERATOR) {
        assert!(pickup.trigger, "{:?} must be a trigger", pickup.name);
        assert_eq!(pickup.sphere_collider, Some(PICKUP_TRIGGER));
        assert!(pickup.spin.is_some(), "collectibles spin so they read as such");
    }
    for post in entities(&scene, "post") {
        assert!(!post.trigger, "{:?} must be solid", post.name);
        assert!(post.aabb_collider.is_some());
    }
}

#[test]
fn every_pickup_floats_just_above_its_own_patch_of_ground() {
    let scene = scene();
    let field = field(&scene);
    for pickup in entities(&scene, PICKUP_GENERATOR) {
        let p = pickup.transform.translation;
        let clearance = p.y - field.height(p.x, p.z);
        assert!(
            (MIN_CLEARANCE..=MAX_CLEARANCE).contains(&clearance),
            "{:?} at ({}, {}) sits {clearance} above the ground, outside \
             [{MIN_CLEARANCE}, {MAX_CLEARANCE}]",
            pickup.name,
            p.x,
            p.z
        );
        // The bob moves it while the game runs, so the *extremes* have to stay
        // sane too — a pickup that dips into the hill at the bottom of its
        // float is unreachable for a third of every cycle.
        let low = clearance - runt_ball::game::BOB_AMPLITUDE;
        let high = clearance + runt_ball::game::BOB_AMPLITUDE;
        assert!(
            low > 0.15 && high < 1.6,
            "{:?} bobs through [{low}, {high}]",
            pickup.name
        );
    }
}

#[test]
fn no_pickup_sits_on_a_slope_a_jumpless_ball_could_not_hold() {
    // There is no jump in v0, so a collectible on a steep face is a collectible
    // you can only get by luck. `tan(30°)` is the line.
    let scene = scene();
    let field = field(&scene);
    for pickup in entities(&scene, PICKUP_GENERATOR) {
        let p = pickup.transform.translation;
        let slope = field.gradient(p.x, p.z).length();
        assert!(
            slope <= MAX_PICKUP_SLOPE,
            "{:?} is on a slope of {slope} (> tan 30° = {MAX_PICKUP_SLOPE})",
            pickup.name
        );
    }
}

#[test]
fn the_spawn_point_is_level_and_resting_on_the_ground() {
    let scene = scene();
    let field = field(&scene);
    let p = named(&scene, PLAYER_ENTITY).transform.translation;

    let slope = field.gradient(p.x, p.z).length();
    assert!(
        slope < MAX_SPAWN_SLOPE,
        "the ball spawns on a slope of {slope}; it would roll away before the \
         player touched a key"
    );
    // Not the lattice origin, whose gradient is identically zero for *every*
    // seed — that would make the assertion above vacuous.
    assert!(
        p.x != 0.0 || p.z != 0.0,
        "spawning at (0, 0) makes the flatness check meaningless"
    );

    let drop = p.y - field.height(p.x, p.z) - BALL_RADIUS;
    assert!(
        (0.0..=0.05).contains(&drop),
        "the ball starts {drop} above its resting height; over ~0.05 it opens \
         the game with a visible bounce"
    );
}

#[test]
fn posts_stand_on_the_ground_and_clear_of_everything_else() {
    let scene = scene();
    let field = field(&scene);
    let posts = entities(&scene, "post");
    let pickups = entities(&scene, PICKUP_GENERATOR);
    let spawn = named(&scene, PLAYER_ENTITY).transform.translation;

    for (i, post) in posts.iter().enumerate() {
        let p = post.transform.translation;
        let half = post.aabb_collider.expect("posts are AABBs");

        // Base on the ground: a floating post is a visible bug, a buried one is
        // an invisible wall.
        let base = p.y - POST_HALF_HEIGHT - field.height(p.x, p.z);
        assert!(
            base.abs() < 0.15,
            "{:?}'s base is {base} off the ground",
            post.name
        );

        // Nothing may stand inside a collectible: the ball has to be able to
        // reach a pickup's centre, and a solid box in the way makes that
        // impossible without ever looking wrong in the file.
        for pickup in &pickups {
            let q = pickup.transform.translation;
            let closest = (q - p).clamp(-half, half) + p;
            let gap = (q - closest).length();
            assert!(
                gap > PICKUP_TRIGGER + BALL_RADIUS,
                "{:?} is {gap} from {:?}; the ball ({BALL_RADIUS}) could not \
                 reach inside the trigger ({PICKUP_TRIGGER})",
                post.name,
                pickup.name
            );
        }

        // Nor on top of the player.
        assert!(
            horizontal(p, spawn) > half.x + BALL_RADIUS + 1.0,
            "{:?} is on the spawn point",
            post.name
        );

        // Nor inside each other: two overlapping AABBs make a push-out fight
        // whose winner depends on entity order.
        for other in posts.iter().skip(i + 1) {
            let q = other.transform.translation;
            let other_half = other.aabb_collider.expect("posts are AABBs");
            let overlap = (p - q).abs().cmplt(half + other_half).all();
            assert!(
                !overlap,
                "{:?} and {:?} overlap",
                post.name,
                other.name
            );
        }
    }
}

#[test]
fn everything_is_inside_the_terrain_patch() {
    // Off the patch there is no ground: `contains_world` finds nothing, the
    // integrator free-falls, and the kill plane eats whatever was there.
    let scene = scene();
    let half = terrain(&scene).size * 0.5;
    let margin = 2.0; // Room to roll around a pickup rather than off the edge.
    for entity in &scene.entities {
        if entity.generator == "ground" {
            continue;
        }
        let p = entity.transform.translation;
        assert!(
            p.x.abs() < half.x - margin && p.z.abs() < half.y - margin,
            "{:?} at ({}, {}) is within {margin} of the edge of the {}×{} patch",
            entity.name,
            p.x,
            p.z,
            half.x * 2.0,
            half.y * 2.0
        );
    }
}

#[test]
fn pickups_are_spread_out_rather_than_clustered() {
    // Twelve collectibles in one corner is a five-second game. The minimum
    // separation is also what keeps one overlap event from ever covering two.
    let scene = scene();
    let pickups = entities(&scene, PICKUP_GENERATOR);
    for (i, a) in pickups.iter().enumerate() {
        for b in pickups.iter().skip(i + 1) {
            let gap = horizontal(a.transform.translation, b.transform.translation);
            assert!(
                gap > 4.0,
                "{:?} and {:?} are only {gap} apart",
                a.name,
                b.name
            );
        }
    }
}

// ---------------------------------------------------------------------------
// …and what `setup` makes of it
// ---------------------------------------------------------------------------

#[test]
fn setup_turns_the_scene_into_a_game() {
    let mut sim = runt_ball::headless_sim();
    let state = sim.world().resource::<GameState>().clone();

    assert_eq!(state.total, 12);
    assert_eq!(state.score, 0);
    assert_eq!(state.elapsed_ticks, 0);
    assert_eq!(state.resets, 0);
    assert!(!state.won());
    assert_eq!(state.player, sim.scene_entity(PLAYER_ENTITY).expect("player"));

    // The kill plane is under the whole field, and not absurdly so.
    let scene = scene();
    let field = field(&scene);
    let mut floor = f32::MAX;
    let half = terrain(&scene).size * 0.5;
    for j in 0..=128 {
        for i in 0..=128 {
            let x = -half.x + half.x * 2.0 * i as f32 / 128.0;
            let z = -half.y + half.y * 2.0 * j as f32 / 128.0;
            floor = floor.min(field.height(x, z));
        }
    }
    assert!(
        state.kill_y < floor && state.kill_y > floor - 12.0,
        "kill plane {} against a field floor of {floor}",
        state.kill_y
    );

    // Every pickup entity carries the marker, with the base height the file
    // placed it at — the bob oscillates about the authored value, not about
    // wherever it happened to be on tick one.
    let mut pickups = sim.world_mut().query::<&Pickup>();
    let mut found: Vec<f32> = pickups.iter(sim.world()).map(|p| p.base_y).collect();
    assert_eq!(found.len(), 12);
    found.sort_by(f32::total_cmp);

    let mut authored: Vec<f32> = entities(&scene, PICKUP_GENERATOR)
        .iter()
        .map(|e| e.transform.translation.y)
        .collect();
    authored.sort_by(f32::total_cmp);
    assert_eq!(found, authored);
}
