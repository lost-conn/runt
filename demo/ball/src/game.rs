//! The game (DESIGN §12 step 6).
//!
//! Every line of gameplay in runt's v0 demo lives in this file, and nothing in
//! `runt-core` or `runt-app` knows any of it exists. That is DESIGN §2's rule
//! ("hosts contain no engine logic") pushed one layer further: the *engine*
//! contains no game logic either. What the engine supplies is a scene loader,
//! a ball integrator, overlap messages, a follow camera, and two seams —
//! [`Sim::fixed_sim_mut`] for systems and [`StatusLine`] for text. What this
//! file supplies is the rules.
//!
//! ```text
//! FixedSim (engine)                    …then (game, chained)
//! update_overlap_messages              collect_pickups   trigger overlap → score
//! spin                                 kill_plane        fell off → back to spawn
//! integrate_balls                      game_clock        tick the timer
//! resolve_overlaps  ──────────────┐    win_check         all collected → Won
//! roll_spin                       └──▶ pickup_bob        cosmetic float
//! follow_camera  ◀────────────────┐    update_status     the "HUD"
//! propagate_transforms            └── (game runs in here)
//! advance_tick_count
//! ```
//!
//! The chain position is load-bearing twice over:
//!
//! - **After `resolve_overlaps`**, so a `MessageReader<OverlapEvent>` sees this
//!   tick's overlaps *on this tick* rather than the next one (the lifetime rules
//!   are spelled out on [`OverlapEvent`]). A pickup collected on tick *N* scores
//!   on tick *N*.
//! - **Before `follow_camera`**, so a kill-plane reset teleports the ball before
//!   the camera decides where to look. Chained the other way, every fall would
//!   cost one frame of the camera lunging at the old position.
//!
//! ## Determinism
//!
//! Nothing here reads a clock, a `HashMap`, or a random number. The one
//! transcendental — the pickup bob — is a function of [`TickCount`], not of
//! elapsed wall time, so it lands on the same float every run at every frame
//! rate. `tests/replay.rs` runs the whole game twice under different host
//! cadences and compares the transform stream bit for bit.
//!
//! ## What the scene file says and what this file says
//!
//! `assets/level1.ron` places geometry and colliders; it has no idea what a
//! "pickup" is. The join is the generator name: [`setup`] marks every entity
//! built from the `pickup` generator with the [`Pickup`] component. So a
//! thirteenth pickup is one more entry in the RON and no code change, and the
//! engine's scene format never grows a game-specific field.

use bevy_ecs::prelude::*;
use glam::Vec3;

use runt_core::ecs::{FixedTick, TerrainSurface, TickCount};
use runt_core::physics::{self, Ball, OverlapEvent, Velocity};
use runt_core::scene::LoadedScene;
use runt_core::{camera, Sim, StatusLine, Transform};

/// The scene generator whose entities are collectibles. The one piece of shared
/// vocabulary between `level1.ron` and this file.
pub const PICKUP_GENERATOR: &str = "pickup";

/// The scene entity name of the player's ball.
pub const PLAYER_ENTITY: &str = "player";

/// How far below the terrain's lowest point counts as "fell off the world".
///
/// Generous: the field's minimum is a *sample* minimum, and a ball can legally
/// dip below the surface it is resting on by a hair mid-solve. Nothing except a
/// genuine fall off the edge of the patch gets anywhere near this.
pub const KILL_MARGIN: f32 = 6.0;

/// Pickup float: metres of travel either side of the resting height.
///
/// Small on purpose. The bob is written into `Transform.translation.y` every
/// tick, and the trigger sphere is centred on that transform, so the bob is
/// physically real — 0.12 m keeps the collectible's reach honest while still
/// reading as "hovering" rather than "sitting in the dirt".
pub const BOB_AMPLITUDE: f32 = 0.12;

/// Bob angular rate, rad/s.
pub const BOB_RATE: f32 = 2.2;

// ---------------------------------------------------------------------------
// Components and resources
// ---------------------------------------------------------------------------

/// A collectible. Carries the two numbers the bob needs so the system is a pure
/// function of `(tick, component)` and never has to remember anything.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Pickup {
    /// The `y` the scene file placed it at — the centre the bob oscillates
    /// about, captured once so repeated bobs cannot drift it.
    pub base_y: f32,
    /// Phase offset, radians. Derived from spawn index so twelve pickups do not
    /// pulse in unison.
    pub phase: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Phase {
    #[default]
    Playing,
    Won,
}

/// The whole game state. One resource, no hidden state anywhere else — which is
/// what lets a test assert a run reproduced exactly by comparing this struct.
#[derive(Resource, Clone, Debug, PartialEq)]
pub struct GameState {
    pub score: u32,
    pub total: u32,
    /// Ticks spent in [`Phase::Playing`]. Frozen at the win.
    pub elapsed_ticks: u64,
    pub phase: Phase,
    /// Times the kill plane put the ball back. No penalty in v0 — falling off
    /// already costs the walk back, and a marble game that punishes twice is a
    /// marble game nobody finishes.
    pub resets: u32,
    /// Where a reset puts the ball.
    pub spawn_point: Vec3,
    /// The player's ball entity.
    pub player: Entity,
    /// Below this `y`, the ball is gone.
    pub kill_y: f32,
    /// Tick length, cached so status formatting needs no system param.
    pub tick_dt: f32,
}

impl GameState {
    /// Seconds of play, from the tick count. Not a wall clock — this is the
    /// number a replay reproduces.
    pub fn elapsed_secs(&self) -> f32 {
        self.elapsed_ticks as f32 * self.tick_dt
    }

    pub fn won(&self) -> bool {
        self.phase == Phase::Won
    }

    /// The line the host paints (see [`StatusLine`]).
    pub fn status(&self) -> String {
        let falls = match self.resets {
            0 => String::new(),
            1 => " · 1 fall".to_string(),
            n => format!(" · {n} falls"),
        };
        match self.phase {
            Phase::Playing => format!(
                "runt ball — {}/{} · {:.1}s{falls}",
                self.score,
                self.total,
                self.elapsed_secs()
            ),
            Phase::Won => format!(
                "runt ball — WON {}/{} in {:.1}s{falls}",
                self.score,
                self.total,
                self.elapsed_secs()
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

/// Turn a freshly loaded `level1.ron` into a game.
///
/// Runs once, after `Startup` and before the first tick — the only moment when
/// the scene's entities exist and nothing has simulated yet. Three jobs: mark
/// the pickups, measure the kill plane, install the systems.
///
/// Panics if the scene has no player: a level that cannot be played is a build
/// error, not a runtime condition to degrade around, and `tests/level.rs` runs
/// this on the shipped file.
pub fn setup(sim: &mut Sim) {
    let player = sim
        .scene_entity(PLAYER_ENTITY)
        .unwrap_or_else(|| panic!("level1.ron must name an entity {PLAYER_ENTITY:?}"));
    let spawn_point = sim
        .world()
        .get::<Transform>(player)
        .expect("the player has a Transform")
        .translation;
    let tick_dt = sim.tick_dt() as f32;

    // --- mark the pickups --------------------------------------------------
    //
    // Spawn order, straight off the scene description: index-aligned with
    // `LoadedScene::spawned`, so the phase offsets are a deterministic function
    // of the file's ordering rather than of query iteration.
    let loaded = sim
        .world()
        .get_resource::<LoadedScene>()
        .expect("a scene was loaded");
    let pickups: Vec<(Entity, f32)> = loaded
        .desc
        .entities
        .iter()
        .zip(&loaded.spawned)
        .filter(|(desc, _)| desc.generator == PICKUP_GENERATOR)
        .map(|(desc, &entity)| (entity, desc.transform.translation.y))
        .collect();
    assert!(
        !pickups.is_empty(),
        "level1.ron must place at least one {PICKUP_GENERATOR:?} entity"
    );
    let total = pickups.len() as u32;

    // Golden-angle phases: any count of pickups ends up spread around the
    // circle instead of falling into rows that pulse together.
    for (index, (entity, base_y)) in pickups.into_iter().enumerate() {
        sim.world_mut().entity_mut(entity).insert(Pickup {
            base_y,
            phase: index as f32 * 2.399_963_2,
        });
    }

    // --- measure the kill plane -------------------------------------------
    let kill_y = terrain_floor(sim) - KILL_MARGIN;

    sim.world_mut().insert_resource(GameState {
        score: 0,
        total,
        elapsed_ticks: 0,
        phase: Phase::Playing,
        resets: 0,
        spawn_point,
        player,
        kill_y,
        tick_dt,
    });

    // --- install the rules -------------------------------------------------
    //
    // See the module docs for why the chain sits exactly here.
    sim.fixed_sim_mut().add_systems(
        (
            collect_pickups,
            kill_plane,
            game_clock,
            win_check,
            pickup_bob,
            update_status,
        )
            .chain()
            .after(physics::resolve_overlaps)
            .after(physics::roll_spin)
            .before(camera::follow_camera),
    );
}

/// The lowest point of every terrain patch in the world, sampled on a grid.
///
/// Sampled rather than derived from `amplitude`: the field normalises by its
/// octave weights, so the true minimum is well inside `±amplitude` and a
/// margin computed from the parameter would sit needlessly deep. A 129² grid
/// over a 48 m patch samples every 0.37 m, far finer than the field's smallest
/// feature at these frequencies.
fn terrain_floor(sim: &mut Sim) -> f32 {
    const N: i32 = 128;
    let mut query = sim.world_mut().query::<(&TerrainSurface, &Transform)>();
    let patches: Vec<(TerrainSurface, Vec3)> = query
        .iter(sim.world())
        .map(|(surface, transform)| (*surface, transform.translation))
        .collect();

    let mut floor = f32::MAX;
    for (surface, origin) in patches {
        let half = surface.size * 0.5;
        for j in 0..=N {
            for i in 0..=N {
                // `size` is (X extent, Z extent) — the patch is a plan view.
                let x = origin.x - half.x + surface.size.x * i as f32 / N as f32;
                let z = origin.z - half.y + surface.size.y * j as f32 / N as f32;
                floor = floor.min(surface.height_world(origin, x, z));
            }
        }
    }
    if floor == f32::MAX {
        // No terrain at all: a scene like that has no floor to fall through,
        // so put the plane somewhere harmless rather than at +∞.
        return -1000.0;
    }
    floor
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// `FixedSim`: a trigger overlap with a [`Pickup`] scores it and removes it.
///
/// Reads `MessageReader<OverlapEvent>` rather than
/// [`Sim::overlaps`](runt_core::Sim::overlaps) because this is *in* the tick:
/// the reader's cursor guarantees each overlap is seen exactly once, so a
/// pickup cannot be scored twice even if the despawn had not already made that
/// impossible.
pub fn collect_pickups(
    mut reader: MessageReader<OverlapEvent>,
    pickups: Query<(), With<Pickup>>,
    mut state: ResMut<GameState>,
    mut commands: Commands,
) {
    for event in reader.read() {
        if !event.trigger || pickups.get(event.other).is_err() {
            continue; // A post, a wall, a bounce — not a collectible.
        }
        state.score += 1;
        commands.entity(event.other).despawn();
    }
}

/// `FixedSim`: a ball below the kill plane goes back to the spawn point.
///
/// Position *and* velocity, because a reset that kept the fall's velocity would
/// fire the ball straight back off the map. No score penalty (see
/// [`GameState::resets`]).
pub fn kill_plane(
    mut state: ResMut<GameState>,
    mut balls: Query<(&mut Transform, &mut Velocity), With<Ball>>,
) {
    let Ok((mut transform, mut velocity)) = balls.get_mut(state.player) else {
        return;
    };
    if transform.translation.y >= state.kill_y {
        return;
    }
    transform.translation = state.spawn_point;
    velocity.0 = Vec3::ZERO;
    state.resets += 1;
}

/// `FixedSim`: one tick of the clock, while there is still a game on.
pub fn game_clock(mut state: ResMut<GameState>) {
    if state.phase == Phase::Playing {
        state.elapsed_ticks += 1;
    }
}

/// `FixedSim`: all of them collected ends the run.
///
/// Chained after [`game_clock`], so the tick the last pickup was taken on is
/// counted — the final time is the time it took to take it.
pub fn win_check(mut state: ResMut<GameState>) {
    if state.phase == Phase::Playing && state.score >= state.total {
        state.phase = Phase::Won;
    }
}

/// `FixedSim`: float the pickups.
///
/// A function of [`TickCount`] and the component, with no accumulator: replaying
/// from tick 4000 gives the same height as reaching tick 4000, and the bob
/// cannot drift. Writing `translation.y` outright is also exactly what
/// [`Interpolated`](runt_core::Interpolated) expects — `PostSim` snapshots the
/// previous value at the head of each tick, so the renderer blends between two
/// real tick poses and the float reads smooth at any frame rate.
pub fn pickup_bob(
    tick: Res<TickCount>,
    fixed: Res<FixedTick>,
    mut pickups: Query<(&Pickup, &mut Transform)>,
) {
    let t = tick.0 as f32 * fixed.dt_secs;
    for (pickup, mut transform) in &mut pickups {
        transform.translation.y = pickup.base_y + BOB_AMPLITUDE * (t * BOB_RATE + pickup.phase).sin();
    }
}

/// `FixedSim` (tail): publish the status line for the host to paint.
///
/// [`StatusLine::set`] is a no-op when the text is unchanged, and the host skips
/// the platform call on an unchanged string too, so the common tick costs one
/// `String` compare.
pub fn update_status(state: Res<GameState>, mut status: ResMut<StatusLine>) {
    status.set(state.status());
}
