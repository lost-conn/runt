//! The UI's own state, and the rules for keeping it in step with the engine.
//!
//! rinch signals must be touched from the main thread only, so everything here
//! is main-thread state. The engine thread never writes it; it sends
//! [`Event`]s, and [`EditorState::absorb`] folds them in. That one-way rule is
//! what keeps "what the panel shows" and "what the world contains" from drifting
//! in ways nobody can reproduce.
//!
//! ## Why the context is leaked
//!
//! [`Ctx`] is a `&'static Editor`, obtained by leaking one `Box` at startup.
//! That looks like a cheat and is a deliberate fit to rinch's grain: `rsx!`
//! closures capture by **move**, and a panel hands the same handle to a dozen
//! nested closures inside two nested `for` loops. With an `Rc` that is a dozen
//! hand-placed `.clone()`s in exactly the right scopes, and one mistake is a
//! borrow-checker error thirty lines inside a macro expansion. rinch's own
//! examples dodge this by making their shared state `Copy` (a struct of
//! `Signal`s); `&'static` is the same trick for state that cannot be a signal —
//! a `RefCell`, a channel, a GPU thread handle.
//!
//! The cost is that the editor's one context lives until the process exits,
//! which is exactly as long as it was going to live anyway. The visible
//! consequence is that [`EngineHandle`]'s `Drop` never runs, so the engine
//! thread is reaped by process exit rather than joined; it holds no resource
//! that outlives the process.

use std::cell::{Cell, RefCell};
use std::path::PathBuf;

use runt_core::gen::GeneratorSpec;
use runt_core::scene::TransformDesc;
use runt_editor_core::debounce::Debouncer;
use runt_editor_core::mapper::{self, Widget};
use runt_editor_core::protocol::{Event, SceneSnapshot, Stats};
use runt_editor_core::{EngineHandle, FieldPath, Orbit};

/// Which mouse gesture the viewport is currently in.
#[derive(Clone, Copy, PartialEq)]
pub enum Drag {
    None,
    Orbit,
    Pan,
}

/// Everything the editor owns, in one place, for the life of the process.
pub struct Editor {
    pub state: RefCell<EditorState>,
    pub engine: EngineHandle,
    /// Param edits waiting out their quiet period, keyed by generator index.
    pub pending: RefCell<Debouncer<usize, GeneratorSpec>>,

    // -- viewport gesture state --------------------------------------------
    pub orbit: Cell<Orbit>,
    pub drag: Cell<Drag>,
    pub last_mouse: Cell<(f32, f32)>,
    /// The last size rinch reported for the surface, so a resize is sent once
    /// rather than every frame.
    pub surface_size: Cell<(u32, u32)>,
    /// Whether the camera has been framed on the current scene.
    pub framed: Cell<bool>,
    /// Where built-in scene paths are resolved from.
    pub root: PathBuf,
}

/// A `Copy` handle to the editor. See the module docs for why it is `'static`.
pub type Ctx = &'static Editor;

impl Editor {
    pub fn leak(engine: EngineHandle, root: PathBuf) -> Ctx {
        Box::leak(Box::new(Editor {
            state: RefCell::new(EditorState::default()),
            engine,
            pending: RefCell::new(Debouncer::default()),
            orbit: Cell::new(Orbit::default()),
            drag: Cell::new(Drag::None),
            last_mouse: Cell::new((0.0, 0.0)),
            surface_size: Cell::new((0, 0)),
            framed: Cell::new(false),
            root,
        }))
    }
}

/// One row of the param panel, as the list renders it.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    /// Stable across re-renders *unless* the panel genuinely changed shape —
    /// see [`EditorState::panel_key`].
    pub key: String,
    pub depth: usize,
    pub widget: Widget,
}

/// Everything the UI draws from.
#[derive(Default)]
pub struct EditorState {
    pub scene: SceneSnapshot,
    pub stats: Stats,
    pub status: String,
    pub error: Option<String>,
    /// Entity index, not `Entity` — see the protocol docs.
    pub selected: Option<usize>,
    /// The generator being edited and the working copy of its params. Edits land
    /// here first and are debounced on their way to the engine, so the panel
    /// stays responsive while the mesh catches up.
    pub draft: Option<(usize, GeneratorSpec)>,
    /// The transform being edited, with the rotation as Euler degrees — a
    /// quaternion is not something anyone types.
    pub draft_transform: Option<(usize, TransformDesc, glam::Vec3)>,
    /// Bumped whenever the panel's *shape* changes (a different generator, a
    /// switched variant, a rerolled seed) as opposed to a value moving under a
    /// slider the user is still holding.
    pub panel_revision: u32,
    /// Set by any edit, cleared by a save.
    pub dirty: bool,
    pub scene_path: Option<PathBuf>,
}

impl EditorState {
    /// Fold one engine event in. Returns whether anything the UI shows changed.
    pub fn absorb(&mut self, event: Event) -> bool {
        match event {
            Event::SceneLoaded(snapshot) => {
                let reloaded = snapshot.path != self.scene.path;
                self.scene = *snapshot;
                self.scene_path = self.scene.path.clone();
                if reloaded {
                    self.selected = None;
                    self.draft = None;
                    self.draft_transform = None;
                    self.dirty = false;
                    self.error = None;
                    self.status = match &self.scene.path {
                        Some(p) => format!(
                            "loaded {} — {} generators, {} entities",
                            p.display(),
                            self.scene.generators.len(),
                            self.scene.entities.len()
                        ),
                        None => "empty scene".into(),
                    };
                    self.panel_revision = self.panel_revision.wrapping_add(1);
                }
                true
            }
            Event::SceneSaved { path, bytes } => {
                self.dirty = false;
                self.scene_path = Some(path.clone());
                self.status = format!("saved {} ({bytes} bytes)", path.display());
                self.error = None;
                true
            }
            Event::Stats(stats) => {
                let changed = self.stats != stats;
                self.stats = stats;
                changed
            }
            Event::Error(message) => {
                self.error = Some(message);
                true
            }
        }
    }

    /// Select an entity, and load its generator and transform into the drafts.
    pub fn select(&mut self, index: Option<usize>) {
        self.selected = index;
        self.draft = None;
        self.draft_transform = None;
        if let Some(entity) = index.and_then(|i| self.scene.entities.get(i)) {
            if let Some(generator) = self.scene.generators.get(entity.generator_index) {
                self.draft = Some((entity.generator_index, generator.spec.clone()));
            }
            self.draft_transform =
                Some((entity.index, entity.transform, euler_degrees(&entity.transform)));
        }
        self.panel_revision = self.panel_revision.wrapping_add(1);
    }

    /// The param panel for the current draft.
    pub fn rows(&self) -> Vec<Row> {
        let Some((generator, spec)) = &self.draft else {
            return Vec::new();
        };
        let name = self
            .scene
            .generators
            .get(*generator)
            .map(|g| g.name.clone())
            .unwrap_or_else(|| format!("generator #{generator}"));

        mapper::build(spec, name)
            .rows()
            .into_iter()
            .map(|(depth, widget)| Row {
                key: self.panel_key(*generator, &widget),
                depth,
                widget,
            })
            .collect()
    }

    /// A row's identity for rinch's list reconciliation.
    ///
    /// Deliberately **not** value-dependent. rinch skips re-rendering a row whose
    /// key is unchanged, and that is exactly what a slider mid-drag needs: the
    /// element must survive while its value moves, or the drag is cancelled on
    /// the first pixel. The revision counter is the escape hatch for the cases
    /// where the value really must be pushed back into the control — a variant
    /// switch, a reroll, a new selection — and it is bumped by hand at each of
    /// those, so it cannot fire on an ordinary drag.
    fn panel_key(&self, generator: usize, widget: &Widget) -> String {
        format!(
            "{}:{generator}:{}",
            self.panel_revision,
            widget.path().display()
        )
    }

    /// Apply an edit to the draft. Returns the spec to send, if it changed.
    pub fn edit(
        &mut self,
        path: &FieldPath,
        edit: &mapper::Edit,
    ) -> Option<(usize, GeneratorSpec)> {
        let (generator, spec) = self.draft.as_mut()?;
        let generator = *generator;
        let before = spec.clone();
        if let Err(e) = mapper::apply(spec, path, edit) {
            // A stale path is normal — the user switched a variant while a
            // control was still holding an old address. Say so quietly.
            log::debug!(
                "runt-editor: dropped edit {edit:?} at {}: {e}",
                path.display()
            );
            return None;
        }
        if *spec == before {
            return None;
        }
        // A variant switch changes which controls exist, so the panel has to be
        // rebuilt; a value change must not rebuild it.
        if matches!(edit, mapper::Edit::Variant(_) | mapper::Edit::Seed(_)) {
            self.panel_revision = self.panel_revision.wrapping_add(1);
        }
        self.dirty = true;
        Some((generator, spec.clone()))
    }

    /// The status-bar line.
    pub fn status_line(&self) -> String {
        if let Some(error) = &self.error {
            return format!("! {error}");
        }
        let s = &self.stats;
        if s.width == 0 {
            return self.status.clone();
        }
        format!(
            "{:.0} fps · render {:.1} ms · readback {:.1} ms · {}×{} · {} tris / {} draws · \
             cache {} gen {} hit · tick {}{}",
            s.fps,
            s.render_ms,
            s.readback_ms,
            s.width,
            s.height,
            s.triangles,
            s.draws,
            s.cache.generated,
            s.cache.hits(),
            s.tick,
            if s.paused { " · PAUSED" } else { "" },
        )
    }

    pub fn title(&self) -> String {
        let name = self
            .scene_path
            .as_ref()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "no scene".into());
        format!("{}{}", name, if self.dirty { " *" } else { "" })
    }
}

/// A transform's rotation as XYZ Euler degrees — the form the fields edit.
///
/// `RotationDesc::Euler` is already degrees and is returned as-is; anything else
/// is decomposed, which is lossy in the usual gimbal sense but is what a numeric
/// rotation field has to do. The draft keeps the degrees alongside the
/// `TransformDesc` so repeated edits accumulate on the *typed* values rather
/// than round-tripping through a quaternion each time.
pub fn euler_degrees(transform: &TransformDesc) -> glam::Vec3 {
    use runt_core::scene::RotationDesc;
    match transform.rotation {
        RotationDesc::Identity => glam::Vec3::ZERO,
        RotationDesc::Euler(degrees) => degrees,
        RotationDesc::Quat(_) => {
            let (x, y, z) = transform.rotation.quat().to_euler(glam::EulerRot::XYZ);
            glam::Vec3::new(x.to_degrees(), y.to_degrees(), z.to_degrees())
        }
    }
}
