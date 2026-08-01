//! The engine's own thread (DESIGN §10).
//!
//! rinch owns the main thread; the engine gets one of its own, with its own
//! wgpu 30 device, and the two talk over [`Command`]/[`Event`] channels plus a
//! [`FrameSink`]. The loop is the ordinary host loop from DESIGN §4 with a
//! command drain bolted on the front:
//!
//! ```text
//! loop {
//!     drain commands  →  apply
//!     update(now)     →  (skipped while paused)
//!     render(view)
//!     read back       →  sink.submit
//! }
//! ```
//!
//! Everything the editor does to the world happens in the "apply" step, between
//! ticks — never during one. That is what keeps the sim's determinism claim
//! intact: an edit is indistinguishable from having loaded a different scene
//! file, because it *is* a different scene file (§6, save-as-params).
//!
//! ## What the editor does to a loaded scene
//!
//! Two things, both on load, both to the camera:
//!
//! - **`FollowCamera` is removed.** The ball demo's camera is welded to the
//!   player; an editor that set its transform each frame would spend every tick
//!   being dragged back. Removing the rig is honest and reversible — the scene
//!   file still says `follow:`, and `save_scene` writes it back untouched,
//!   because saving reads the *description*, not the live components.
//! - **`Interpolated` is removed.** Camera edits arrive between ticks, and an
//!   `Interpolated` camera would blend from its previous-tick pose towards the
//!   new one, smearing every drag by a frame. Nothing else about the camera
//!   needs interpolation once it stops being sim-driven.
//!
//! Nothing else in the world is touched. In particular the sim keeps ticking
//! (spinners spin, balls roll) unless [`Command::SetPaused`] says otherwise.

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::time::{Duration, Instant};

use glam::{Vec3, Vec4};
use runt_core::camera::FollowCamera;
use runt_core::ecs::{Interpolated, MeshRef, Transform};
use runt_core::gen::GeneratorSpec;
use runt_core::material::Material;
use runt_core::scene::{LoadedScene, TransformDesc};
use runt_core::{Engine, Sim};

use crate::bridge::FrameBridge;
use crate::protocol::{
    Command, EntitySnapshot, Event, FrameSink, GeneratorSnapshot, SceneSnapshot, Stats,
};

/// How often the loop reports [`Stats`]. Once per frame would be one UI
/// re-render per frame, which is a lot of work to display a number that changes
/// in the third decimal place.
const STATS_INTERVAL: Duration = Duration::from_millis(400);

/// How bright a selected entity's material gets (§10 explicitly does *not* want
/// an outline pass; this is the cheap thing that works).
///
/// The shader computes `albedo = base_color.rgb × vertex_color` and multiplies
/// that by the lighting, so the highlight deliberately pushes `base_color`
/// **above 1.0**. Clamping it to 1 would make the whole feature a no-op on the
/// commonest case in runt's scenes: a material left at its default white, doing
/// nothing but passing the mesh's vertex colours through. Values over 1 are
/// perfectly legal in the uniform and simply saturate at the framebuffer, which
/// is exactly the "blown out" look a selection wants.
const SELECTION_GAIN: f32 = 2.2;
const SELECTION_LIFT: f32 = 0.35;

/// The UI's end of the engine thread.
pub struct EngineHandle {
    commands: Sender<Command>,
    events: Receiver<Event>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl EngineHandle {
    /// Queue a command. Fails silently once the thread is gone, which is what a
    /// UI wants during shutdown.
    pub fn send(&self, command: Command) {
        if let Err(e) = self.commands.send(command) {
            log::debug!("runt-editor: engine thread is gone, dropping {:?}", e.0);
        }
    }

    /// Every event waiting, without blocking. Called once per UI frame.
    pub fn drain(&self) -> Vec<Event> {
        self.events.try_iter().collect()
    }

    /// Ask the thread to stop and wait for it. Called on window close; also runs
    /// from `Drop` so a panicking UI cannot leave a GPU device alive.
    pub fn shutdown(&mut self) {
        let _ = self.commands.send(Command::Shutdown);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

impl Drop for EngineHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// What the engine thread starts with.
pub struct EngineConfig {
    pub width: u32,
    pub height: u32,
    /// Scene to load before the first frame. `None` starts empty, which is what
    /// the editor does — the UI picks a scene.
    pub scene: Option<PathBuf>,
    /// Quality multiplier for every generator (DESIGN §6).
    pub quality: f32,
    /// Persist generated meshes to disk. On in the editor: revisiting a slider
    /// value is then free, which is the whole reason §6 built the cache.
    pub persistent_cache: bool,
    /// Frames per second to aim for.
    ///
    /// The loop would otherwise run flat out — a headless render of the demo
    /// scene at 320x180 clears ten thousand frames a second — and burn a core to
    /// produce frames a 60 Hz display throws away. Sleeping the remainder of the
    /// budget is the whole throttle; if a frame overruns, nothing is skipped and
    /// the editor simply runs slower.
    pub target_fps: f32,
}

impl Default for EngineConfig {
    fn default() -> EngineConfig {
        EngineConfig {
            width: 1280,
            height: 720,
            scene: None,
            quality: 1.0,
            persistent_cache: true,
            target_fps: 60.0,
        }
    }
}

/// Start the engine thread.
///
/// Returns as soon as the thread is spawned; the device is created on that
/// thread, so a machine with no usable adapter reports through
/// [`Event::Error`] rather than by failing to start the editor.
pub fn spawn(config: EngineConfig, mut sink: Box<dyn FrameSink>) -> EngineHandle {
    let (command_tx, command_rx) = std::sync::mpsc::channel();
    let (event_tx, event_rx) = std::sync::mpsc::channel();

    let join = std::thread::Builder::new()
        .name("runt-engine".into())
        .spawn(move || {
            match EngineLoop::new(config, command_rx, event_tx.clone()) {
                Ok(mut loop_) => loop_.run(sink.as_mut()),
                Err(e) => {
                    let _ = event_tx.send(Event::Error(format!("engine startup failed: {e}")));
                }
            }
        })
        .expect("spawning the engine thread");

    EngineHandle {
        commands: command_tx,
        events: event_rx,
        join: Some(join),
    }
}

struct EngineLoop {
    engine: Engine,
    bridge: FrameBridge,
    commands: Receiver<Command>,
    events: Sender<Event>,

    scene_path: Option<PathBuf>,
    selected: Option<usize>,
    /// The selected entity's untouched material color, so deselecting restores
    /// exactly what the scene said rather than an approximation of it.
    selection_restore: Option<(usize, Vec4)>,

    paused: bool,
    /// Sim time, which advances only while unpaused. The engine takes wall time
    /// from its host and computes ticks from it (DESIGN §4), so pausing is a
    /// matter of not advancing the number we hand it — no engine support needed.
    sim_clock: f64,
    last_frame: Instant,

    /// An explicit camera pose from the editor, applied before every render.
    camera_pose: Option<(Vec3, Vec3)>,

    frame_budget: Duration,
    frames: u32,
    window_started: Instant,
    last_stats: Instant,
    last_render_ms: f32,
}

impl EngineLoop {
    fn new(
        config: EngineConfig,
        commands: Receiver<Command>,
        events: Sender<Event>,
    ) -> Result<EngineLoop, String> {
        let cache: Box<dyn runt_core::cache::CacheStore> = if config.persistent_cache {
            runt_core::cache::platform_default()
        } else {
            Box::new(runt_core::cache::NoopCache)
        };
        let sim_config = runt_core::sim::SimConfig::default()
            .with_quality(config.quality)
            .with_cache(cache)
            .without_scene();

        let engine = pollster::block_on(Engine::headless_with_config(
            FrameBridge::FORMAT,
            sim_config,
        ))?;
        let bridge = FrameBridge::new(engine.device(), config.width, config.height);

        let now = Instant::now();
        let mut loop_ = EngineLoop {
            engine,
            bridge,
            commands,
            events,
            scene_path: None,
            selected: None,
            selection_restore: None,
            paused: false,
            sim_clock: 0.0,
            last_frame: now,
            camera_pose: None,
            frame_budget: if config.target_fps > 0.0 {
                Duration::from_secs_f32(1.0 / config.target_fps)
            } else {
                Duration::ZERO
            },
            frames: 0,
            window_started: now,
            last_stats: now,
            last_render_ms: 0.0,
        };

        if let Some(path) = config.scene {
            loop_.load_scene(&path);
        }
        Ok(loop_)
    }

    fn run(&mut self, sink: &mut dyn FrameSink) {
        loop {
            let frame_started = Instant::now();
            match self.pump_commands() {
                Flow::Continue => {}
                Flow::Stop => return,
            }

            let now = Instant::now();
            let delta = now.duration_since(self.last_frame).as_secs_f64();
            self.last_frame = now;
            if !self.paused {
                // Clamp the delta the same way a host would after a stall: the
                // sim has its own spiral guard, but feeding it a two-second jump
                // because the window was dragged is pointless work either way.
                self.sim_clock += delta.min(0.25);
            }
            self.engine.update(self.sim_clock);

            self.apply_camera_pose();

            let render_started = Instant::now();
            let (width, height) = self.bridge.size();
            // `engine` and `bridge` are separate fields, so the immutable borrow
            // of one and the mutable borrow of the other coexist happily.
            self.engine.render(self.bridge.view(), width, height);
            self.last_render_ms = render_started.elapsed().as_secs_f32() * 1000.0;

            let device = self.engine.device().clone();
            let queue = self.engine.queue().clone();
            let pixels = self.bridge.read(&device, &queue);
            sink.submit(pixels, width, height);

            self.frames += 1;
            self.report_stats();

            // Give the rest of the budget back to the machine. `checked_sub`
            // rather than a saturating subtract so an overrunning frame simply
            // does not sleep, instead of sleeping for a wrapped eternity.
            if let Some(rest) = self.frame_budget.checked_sub(frame_started.elapsed()) {
                std::thread::sleep(rest);
            }
        }
    }

    // -- commands -----------------------------------------------------------

    fn pump_commands(&mut self) -> Flow {
        loop {
            match self.commands.try_recv() {
                Ok(Command::Shutdown) => return Flow::Stop,
                Ok(command) => self.apply(command),
                Err(TryRecvError::Empty) => return Flow::Continue,
                // The UI dropped its end: the window is gone.
                Err(TryRecvError::Disconnected) => return Flow::Stop,
            }
        }
    }

    fn apply(&mut self, command: Command) {
        match command {
            Command::Shutdown => unreachable!("handled by pump_commands"),

            Command::Resize { width, height } => {
                let device = self.engine.device().clone();
                self.bridge.resize(&device, width, height);
            }

            Command::LoadScene(path) => self.load_scene(&path),
            Command::SaveScene(path) => self.save_scene(&path),

            Command::SetPaused(paused) => self.paused = paused,

            Command::SetCameraPose { eye, target } => {
                self.camera_pose = Some((eye, target));
            }

            Command::Select(index) => self.select(index),

            Command::ParamEdit { generator, spec } => self.param_edit(generator, spec),

            Command::TransformEdit { entity, transform } => {
                self.transform_edit(entity, transform)
            }
        }
    }

    // -- scene --------------------------------------------------------------

    fn load_scene(&mut self, path: &std::path::Path) {
        let source = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                self.error(format!("cannot read {}: {e}", path.display()));
                return;
            }
        };

        self.selected = None;
        self.selection_restore = None;

        let world = self.engine.sim_mut().world_mut();
        if let Err(e) = runt_core::scene::load_scene(world, &source) {
            self.error(format!("{}: {e}", path.display()));
            return;
        }

        self.make_camera_inert();
        self.scene_path = Some(path.to_path_buf());
        // A freshly loaded scene has no editor camera pose yet; the UI frames it
        // from the snapshot and sends one.
        self.camera_pose = None;
        self.send_snapshot();
    }

    /// Take the scene camera out of the sim's hands. See the module docs.
    fn make_camera_inert(&mut self) {
        let world = self.engine.sim_mut().world_mut();
        let Some(camera) = world.get_resource::<LoadedScene>().map(|s| s.camera) else {
            return;
        };
        if let Ok(mut entity) = world.get_entity_mut(camera) {
            entity.remove::<FollowCamera>();
            entity.remove::<Interpolated>();
        }
    }

    fn save_scene(&mut self, path: &std::path::Path) {
        let world = self.engine.sim().world();
        match runt_core::scene::save_scene(world) {
            Ok(ron) => match std::fs::write(path, ron.as_bytes()) {
                Ok(()) => {
                    self.scene_path = Some(path.to_path_buf());
                    let _ = self.events.send(Event::SceneSaved {
                        path: path.to_path_buf(),
                        bytes: ron.len(),
                    });
                }
                Err(e) => self.error(format!("cannot write {}: {e}", path.display())),
            },
            Err(e) => self.error(format!("save failed: {e}")),
        }
    }

    // -- edits --------------------------------------------------------------

    /// Swap a generator's spec and rebuild everything that depends on it.
    ///
    /// The path is the same one `load_scene` takes — spec → `GenCache` →
    /// `MeshLibrary` → `MeshRef` — so a live edit and a file edit cannot produce
    /// different geometry. Revisiting a value a slider has already passed
    /// through is a layer-A cache hit and costs nothing (§6), which is what
    /// makes scrubbing usable.
    fn param_edit(&mut self, index: usize, spec: GeneratorSpec) {
        let world = self.engine.sim_mut().world_mut();

        let Some(loaded) = world.get_resource::<LoadedScene>() else {
            self.error("no scene loaded".into());
            return;
        };
        let Some(entry) = loaded.desc.generators.get(index) else {
            self.error(format!("no generator #{index}"));
            return;
        };
        if entry.spec == spec {
            return; // A debounce flush of an unchanged value.
        }
        let quality_policy = entry.quality;
        let generator_name = entry.name.clone();
        let tier = *world.resource::<runt_core::ecs::QualityTier>();
        let quality = quality_policy.resolve(tier);

        // Regenerate, borrowing the cache and library out of the world together
        // — the same dance `spawn_scene` does, for the same borrow reason.
        let handle = world.resource_scope(
            |world, mut cache: bevy_ecs::prelude::Mut<runt_core::cache::GenCache>| {
                world.resource_scope(
                    |_world, mut library: bevy_ecs::prelude::Mut<runt_core::registry::MeshLibrary>| {
                        cache.resolve(&spec, quality, &mut library)
                    },
                )
            },
        );

        // Record the edit in the description, so a save writes it out.
        let mut affected = Vec::new();
        {
            let mut loaded = world.resource_mut::<LoadedScene>();
            loaded.desc.generators[index].spec = spec.clone();
            for (i, placement) in loaded.desc.entities.iter().enumerate() {
                if placement.generator == generator_name {
                    affected.push((i, loaded.spawned[i]));
                }
            }
        }

        let param_key = spec.param_key(quality);
        let terrain = match &spec {
            GeneratorSpec::Terrain(params) => Some(runt_core::ecs::TerrainSurface::new(params)),
            _ => None,
        };

        for (_, entity) in &affected {
            let Ok(mut entity) = world.get_entity_mut(*entity) else {
                continue;
            };
            entity.insert(MeshRef(handle));
            entity.insert(runt_core::ecs::GeneratorRef {
                name: generator_name.clone(),
                param_key,
            });
            // Terrain is not just a mesh: it is the analytic surface physics
            // samples (DESIGN §9). Editing the field has to move the collision
            // with it, or the ball rolls on the old hills.
            if let Some(surface) = terrain {
                entity.insert(surface);
            }
        }

        self.send_snapshot();
    }

    fn transform_edit(&mut self, index: usize, desc: TransformDesc) {
        let world = self.engine.sim_mut().world_mut();
        let Some(loaded) = world.get_resource::<LoadedScene>() else {
            self.error("no scene loaded".into());
            return;
        };
        let Some(&entity) = loaded.spawned.get(index) else {
            self.error(format!("no entity #{index}"));
            return;
        };

        let transform = desc.to_transform();
        {
            let mut loaded = world.resource_mut::<LoadedScene>();
            loaded.desc.entities[index].transform = desc;
        }
        if let Ok(mut entity) = world.get_entity_mut(entity) {
            entity.insert(transform);
            entity.insert(runt_core::ecs::GlobalTransform(transform.matrix()));
            // A moving entity's interpolation state must not lag a whole tick
            // behind an edit, or the object visibly slides to its new home.
            if let Some(mut interp) = entity.get_mut::<Interpolated>() {
                *interp = Interpolated::from(&transform);
            }
        }
        self.send_snapshot();
    }

    /// Brighten the selected entity's material, restoring the previous one.
    fn select(&mut self, index: Option<usize>) {
        if self.selected == index {
            return;
        }
        let world = self.engine.sim_mut().world_mut();

        // Put the old selection back exactly as the scene had it.
        if let Some((old, color)) = self.selection_restore.take() {
            if let Some(entity) = world.get_resource::<LoadedScene>().and_then(|s| s.spawned.get(old).copied()) {
                if let Some(mut material) = world.get_mut::<Material>(entity) {
                    material.base_color = color;
                }
            }
        }

        self.selected = index;
        let Some(index) = index else {
            return;
        };
        let Some(entity) = world
            .get_resource::<LoadedScene>()
            .and_then(|s| s.spawned.get(index).copied())
        else {
            return;
        };
        if let Some(mut material) = world.get_mut::<Material>(entity) {
            let original = material.base_color;
            self.selection_restore = Some((index, original));
            let lifted = original.truncate() * SELECTION_GAIN + Vec3::splat(SELECTION_LIFT);
            material.base_color = lifted.extend(original.w);
        }
    }

    // -- per-frame ----------------------------------------------------------

    fn apply_camera_pose(&mut self) {
        let Some((eye, target)) = self.camera_pose else {
            return;
        };
        let world = self.engine.sim_mut().world_mut();
        let Some(camera) = world.get_resource::<LoadedScene>().map(|s| s.camera) else {
            return;
        };
        let pose = Transform::looking_at(eye, target, Vec3::Y);
        if let Ok(mut entity) = world.get_entity_mut(camera) {
            entity.insert(pose);
            entity.insert(runt_core::ecs::GlobalTransform(pose.matrix()));
            // Belt and braces: `make_camera_inert` strips this on load, but a
            // scene reloaded under a live pose would put it back.
            if let Some(mut interp) = entity.get_mut::<Interpolated>() {
                *interp = Interpolated::from(&pose);
            }
        }
    }

    fn report_stats(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.last_stats) < STATS_INTERVAL {
            return;
        }
        let elapsed = now.duration_since(self.window_started).as_secs_f32();
        let fps = if elapsed > 0.0 {
            self.frames as f32 / elapsed
        } else {
            0.0
        };

        let (triangles, draws) = self.frame_geometry();
        let (width, height) = self.bridge.size();
        let stats = Stats {
            fps,
            render_ms: self.last_render_ms,
            readback_ms: self.bridge.last_readback_ms(),
            triangles,
            draws,
            width,
            height,
            cache: self.engine.sim().cache_stats(),
            tick: self.engine.tick_count(),
            paused: self.paused,
        };
        let _ = self.events.send(Event::Stats(stats));

        self.frames = 0;
        self.window_started = now;
        self.last_stats = now;
    }

    /// Triangles and draws in the current world. Counted from the mesh library
    /// rather than from the renderer, which does not keep the number around.
    fn frame_geometry(&mut self) -> (u32, u32) {
        let draws = self.engine.sim_mut().draw_list();
        let library = self.engine.sim().mesh_library();
        let triangles: u32 = draws
            .iter()
            .filter_map(|item| library.get(item.mesh))
            .map(|mesh| (mesh.indices.len() / 3) as u32)
            .sum();
        (triangles, draws.len() as u32)
    }

    // -- reporting ----------------------------------------------------------

    fn send_snapshot(&mut self) {
        let snapshot = snapshot(self.engine.sim(), self.scene_path.clone());
        let _ = self.events.send(Event::SceneLoaded(Box::new(snapshot)));
    }

    fn error(&self, message: String) {
        log::warn!("runt-editor: {message}");
        let _ = self.events.send(Event::Error(message));
    }
}

enum Flow {
    Continue,
    Stop,
}

/// Flatten the loaded scene into the form the panels read.
///
/// A free function so tests can build a `Sim`, load a scene into it and check
/// the snapshot without a GPU anywhere in sight.
pub fn snapshot(sim: &Sim, path: Option<PathBuf>) -> SceneSnapshot {
    let world = sim.world();
    let Some(loaded) = world.get_resource::<LoadedScene>() else {
        return SceneSnapshot {
            path,
            ..SceneSnapshot::default()
        };
    };
    let library = sim.mesh_library();

    let generators = loaded
        .desc
        .generators
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            // Find any entity using this generator and read the triangle count
            // off its resolved mesh — cheaper and more truthful than
            // regenerating, since it reports what is actually on screen.
            let triangles = loaded
                .desc
                .entities
                .iter()
                .position(|e| e.generator == entry.name)
                .and_then(|index| loaded.spawned.get(index))
                .and_then(|&entity| world.get::<MeshRef>(entity))
                .and_then(|mesh| library.get(mesh.0))
                .map(|mesh| (mesh.indices.len() / 3) as u32)
                .unwrap_or(0);
            let _ = i;
            GeneratorSnapshot {
                name: entry.name.clone(),
                spec: entry.spec.clone(),
                triangles,
            }
        })
        .collect();

    let entities = loaded
        .desc
        .entities
        .iter()
        .enumerate()
        .map(|(index, placement)| {
            let generator_index = loaded
                .desc
                .generators
                .iter()
                .position(|g| g.name == placement.generator)
                .unwrap_or(0);
            // Prefer the live transform: an entity the sim has moved should list
            // where it *is*, not where the file dropped it.
            let transform = loaded
                .spawned
                .get(index)
                .and_then(|&e| world.get::<Transform>(e))
                .map(TransformDesc::from_transform)
                .unwrap_or(placement.transform);
            EntitySnapshot {
                index,
                label: placement
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("{} #{index}", placement.generator)),
                generator: placement.generator.clone(),
                generator_index,
                transform,
            }
        })
        .collect();

    SceneSnapshot {
        path,
        generators,
        entities,
    }
}
