//! The editor↔engine wire (DESIGN §10).
//!
//! The UI runs on rinch's main thread; the engine runs on a thread of its own
//! with its own wgpu device. Nothing is shared between them but two channels and
//! a pixel buffer — no `Arc<Mutex<World>>`, no engine types leaking into
//! widgets. That is what keeps the engine unable to tell an editor from a game
//! (§10: the same input events, the same `render`).
//!
//! ```text
//!   UI thread ──[Command]──▶ engine thread ──[Event]──▶ UI thread
//!                                   │
//!                                   └──[RGBA8 pixels]──▶ SurfaceWriter
//! ```
//!
//! Two rules keep this small:
//!
//! - **Entities are named by index, not by `Entity`.** A command says
//!   "entity 3", meaning `SceneDesc::entities[3]`, because that is the identity
//!   the *file* has and the only one the UI can meaningfully hold across a
//!   reload. `LoadedScene::spawned` is index-aligned with it, so the engine's
//!   translation is a subscript.
//! - **Commands are absolute, never incremental.** [`Command::ParamEdit`]
//!   carries a whole [`GeneratorSpec`], not a field path and a delta. A dropped
//!   or reordered command can therefore only ever cost a frame of staleness, and
//!   the debounce in front of it is free to coalesce as aggressively as it likes.

use std::path::PathBuf;

use runt_core::cache::CacheStats;
use runt_core::gen::GeneratorSpec;
use runt_core::scene::TransformDesc;

/// UI → engine.
#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    /// Read a scene RON off disk and rebuild the world from it. Answered with
    /// [`Event::SceneLoaded`] or [`Event::Error`].
    LoadScene(PathBuf),
    /// Serialize the live scene back to `path`. Answered with
    /// [`Event::SceneSaved`] or [`Event::Error`].
    SaveScene(PathBuf),

    /// Replace generator `index`'s spec, regenerate it through the cache, and
    /// repoint every entity that uses it.
    ParamEdit {
        /// Index into `SceneDesc::generators`.
        generator: usize,
        spec: GeneratorSpec,
    },
    /// Replace entity `index`'s placement.
    TransformEdit {
        /// Index into `SceneDesc::entities`.
        entity: usize,
        transform: TransformDesc,
    },

    /// Highlight one entity in the viewport, or clear the highlight. Purely
    /// cosmetic — the engine brightens the entity's material and restores it on
    /// deselect (§10 stops well short of an outline pass).
    Select(Option<usize>),

    /// Drive the editor camera. The orbit maths lives in the UI (see
    /// [`crate::orbit`]); the engine only receives a pose, so it never has to
    /// know an editor exists.
    SetCameraPose {
        eye: glam::Vec3,
        target: glam::Vec3,
    },

    /// Stop ticking the sim. Rendering continues, so the scene stays live to
    /// look at and to edit while frozen.
    SetPaused(bool),

    /// The viewport's size changed. Sent by the UI whenever rinch reports a new
    /// layout size for the surface; the engine resizes its offscreen target and
    /// the readback buffer to match.
    Resize { width: u32, height: u32 },

    /// Leave the loop and drop the device.
    Shutdown,
}

/// Engine → UI.
#[derive(Clone, Debug)]
pub enum Event {
    /// A scene is in the world. Carries everything the panels need, so the UI
    /// never reaches back to ask.
    SceneLoaded(Box<SceneSnapshot>),
    SceneSaved { path: PathBuf, bytes: usize },
    /// Per-frame counters for the status bar. Emitted at most a few times a
    /// second — a stat per frame would be a re-render per frame.
    Stats(Stats),
    /// Something the user asked for did not work. Never fatal: the engine thread
    /// reports and carries on.
    Error(String),
}

/// The whole editable state of the loaded scene, flattened for the UI.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SceneSnapshot {
    pub path: Option<PathBuf>,
    pub generators: Vec<GeneratorSnapshot>,
    pub entities: Vec<EntitySnapshot>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GeneratorSnapshot {
    /// Its name in the scene file — what entities refer to.
    pub name: String,
    pub spec: GeneratorSpec,
    /// Triangles the generator actually produced at the session's quality.
    pub triangles: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EntitySnapshot {
    /// Index into [`SceneSnapshot::entities`] — the identity every command uses.
    pub index: usize,
    /// The `name` in the scene file, or a generated stand-in for the list.
    pub label: String,
    /// Which generator supplies its geometry, by name and by index.
    pub generator: String,
    pub generator_index: usize,
    pub transform: TransformDesc,
}

/// Status-bar counters. Everything here is diagnostics; nothing in the engine
/// may branch on it (DESIGN §6 says the same of `CacheStats`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stats {
    /// Frames per second measured over the last reporting window.
    pub fps: f32,
    /// Wall time inside `Engine::render` plus the submit, in milliseconds.
    pub render_ms: f32,
    /// Wall time in the texture→CPU readback, in milliseconds. The number that
    /// decides whether the CPU bridge is viable (§10).
    pub readback_ms: f32,
    /// Triangles in the last draw list.
    pub triangles: u32,
    /// Entities drawn last frame.
    pub draws: u32,
    pub width: u32,
    pub height: u32,
    pub cache: CacheStats,
    pub tick: u64,
    pub paused: bool,
}

/// Where a frame's pixels go.
///
/// The engine thread does not depend on rinch: it hands tightly-packed RGBA8 to
/// whatever this is, and the editor binary implements it over
/// `rinch::SurfaceWriter`. Tests implement it over a counter, which is how the
/// loop can be exercised at all without a window.
pub trait FrameSink: Send {
    /// `pixels` is exactly `width * height * 4` bytes, row-major, no padding —
    /// which is also `SurfaceWriter::submit_frame`'s contract.
    fn submit(&mut self, pixels: &[u8], width: u32, height: u32);
}

impl<F: FnMut(&[u8], u32, u32) + Send> FrameSink for F {
    fn submit(&mut self, pixels: &[u8], width: u32, height: u32) {
        self(pixels, width, height)
    }
}
