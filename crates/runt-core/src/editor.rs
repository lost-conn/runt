//! The in-game scene editor's toolkit — `editor` feature only (DESIGN §10a).
//!
//! §10a settled where tools live: **inside the running game**, drawn with the
//! engine's own [`UiBatch`](crate::ui::UiBatch) and driven by the engine's own
//! [`Input`](crate::Input). [`tweak`](crate::tweak) was the first half of that —
//! the *property* surface, a slider you drag while standing under the sky it
//! changes. This is the second half, and it is the one §10a listed as still
//! unbuilt: *"scene arrangement — place/transform/pick, save scene RON — is
//! still wanted and is still unbuilt; it lands in-game too when it lands."*
//!
//! ```text
//! editor          the toolkit: rays, tools, snapping, the op log, the seam
//! editor_gizmo    the handles: their geometry, their hit test, their drag math
//! EditableScene   the seam a game implements over *its* scene format
//! ```
//!
//! # Every edit is a data operation, never a poke at the world
//!
//! The one architectural decision everything else falls out of. A move is not
//! "write a `Transform`"; it is an **op against the scene description**, applied
//! by the game's adapter, after which the game re-realizes the affected part of
//! the world. Three things come free:
//!
//! - **Undo/redo is the op log.** There is no second data structure and no
//!   snapshot-the-world: [`OpLog`] is a `Vec` and a cursor, and undo is
//!   `apply(invert(op))`. What makes that sound is a contract on the *adapter* —
//!   see [`EditableScene::invert`] — that an op carries enough to invert
//!   exactly, byte for byte.
//! - **Saving is free.** The thing being edited is the thing that gets written
//!   out; there is no "export" step that could disagree with what is on screen.
//! - **Shipping it to a player later is a policy change, not a rewrite.** An op
//!   log is already a serializable diff against an authored level.
//!
//! # What is engine and what is game
//!
//! The toolkit is **game-agnostic and mostly pure functions**. It knows about
//! rays, screen pixels, axes, handles, snapping and a stack of opaque ops. It
//! does not know what a level *is*, what may be placed in one, what an id means
//! or how bytes reach a disk. All of that is behind [`EditableScene`], which the
//! game implements over its own format.
//!
//! This module deliberately installs **no systems and no resources**. Unlike
//! [`tweak`](crate::tweak), whose registry is genuinely engine-side state, every
//! piece of editor state is parameterized by the game's id type — so the
//! resource is the *game's* (`#[derive(Resource)] struct Editor { … }`) and the
//! engine hands it [`EditorState`], [`OpLog`] and a pile of functions to drive
//! them with. A generic `Resource` here would buy one `insert_resource` call and
//! cost every caller a turbofish.
//!
//! # The pointer
//!
//! [`EditorState::cursor`] is the editor's **own** cursor, in logical pixels,
//! moved by [`Input::mouse_delta`](crate::Input::mouse_delta) and clamped to the
//! viewport. It is not the OS cursor, because runt has no cursor seam at all —
//! `runt-app` reads `CursorMoved` into a delta and never reports a position (the
//! port's `ui/mouse.rs` writes the same finding down from the other side). A
//! delta-driven cursor is the shape that works anyway: it is what a game with a
//! captured mouse has, it is identical on native and web, and — because
//! `mouse_delta` is already what a [trace](crate::trace) records — a run edited
//! with the editor open replays into the same edits, which an absolute position
//! read from the host would not.
//!
//! The cost is honest and small: the drawn crosshair and the desktop arrow can
//! drift apart, so a game draws the crosshair (it has a font and an atlas; the
//! engine has neither) and the player looks at that one.
//!
//! # Feature gate
//!
//! `editor` implies `reflect`, because the tweak panel *is* the property editor
//! for a selection and there is no version of this tool that is worth having
//! without it. The player build carries neither — see the port's
//! `shift/Cargo.toml`, which switches the feature on by **target** rather than
//! by a flag somebody has to remember.

use bevy_ecs::prelude::Entity;
use glam::{Mat4, Quat, Vec2, Vec3, Vec4Swizzles};

use crate::camera::Camera;
use crate::collide::{CollisionWorld, RayHit, ALL_LAYERS};
use crate::ecs::{Transform, Viewport};

// ---------------------------------------------------------------------------
// Rays: screen -> world, and back
// ---------------------------------------------------------------------------

/// A world-space ray. `dir` is unit length.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3,
}

impl Ray {
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.dir * t
    }
}

/// The ray under `cursor`, from a camera's lens and world pose.
///
/// `cursor` is in **logical pixels, top-left origin, +Y down** — the same space
/// [`UiQuad`](crate::ui::UiQuad) and [`Touch`](crate::Touch) are in, so a
/// finger and a mouse arrive in the same coordinates.
///
/// This takes the *camera*, not a [`FrameParams`](crate::FrameParams), and that
/// is deliberate: `FrameParams` is built by the render side, one layer up and
/// one interpolation-alpha away, whereas a pick is a simulation question asked
/// at a tick boundary. Taking the camera means the answer is a pure function of
/// world state, which is what makes it replayable and what makes the test
/// below a unit test. [`ray_from_view_proj`] is here for a caller that already
/// holds the matrix.
pub fn screen_ray(camera: &Camera, pose: Mat4, viewport: Viewport, cursor: Vec2) -> Option<Ray> {
    ray_from_view_proj(camera.view_proj(pose, viewport.aspect()), viewport, cursor)
}

/// [`screen_ray`] from a view-projection matrix.
///
/// `None` for a viewport with no area (nothing has been drawn yet) or a matrix
/// that does not invert.
pub fn ray_from_view_proj(view_proj: Mat4, viewport: Viewport, cursor: Vec2) -> Option<Ray> {
    if !viewport.is_known() {
        return None;
    }
    let inv = view_proj.inverse();
    if !inv.is_finite() {
        return None;
    }
    let size = viewport.size();
    // Pixels -> NDC. X maps straight through; Y flips, because the pointer
    // space is +Y down and clip space is +Y up.
    let ndc = Vec2::new(
        (cursor.x / size.x) * 2.0 - 1.0,
        1.0 - (cursor.y / size.y) * 2.0,
    );
    // `0.0` is the near plane and `1.0` the far one: runt's projection is the
    // wgpu/WebGPU `[0,1]` clip-depth convention (see `Camera::projection`), not
    // OpenGL's `[-1,1]`.
    let near = inv * glam::Vec4::new(ndc.x, ndc.y, 0.0, 1.0);
    let far = inv * glam::Vec4::new(ndc.x, ndc.y, 1.0, 1.0);
    if near.w.abs() < f32::EPSILON || far.w.abs() < f32::EPSILON {
        return None;
    }
    let origin = near.xyz() / near.w;
    let dir = (far.xyz() / far.w - origin).try_normalize()?;
    Some(Ray { origin, dir })
}

/// Where `world` lands on screen, in the same pixels [`screen_ray`] takes.
///
/// `None` for a point behind the camera (`w <= 0`), which is the case a naive
/// divide turns into a plausible-looking point in the wrong half of the screen.
pub fn project(view_proj: Mat4, viewport: Viewport, world: Vec3) -> Option<Vec2> {
    if !viewport.is_known() {
        return None;
    }
    let clip = view_proj * world.extend(1.0);
    if clip.w <= f32::EPSILON {
        return None;
    }
    let ndc = clip.xyz() / clip.w;
    let size = viewport.size();
    Some(Vec2::new(
        (ndc.x * 0.5 + 0.5) * size.x,
        (0.5 - ndc.y * 0.5) * size.y,
    ))
}

/// The nearest collider a screen ray hits.
///
/// A thin wrapper over [`CollisionWorld::raycast`] — the engine already had the
/// query (its docs name a ledge vault and a wall find as the callers), so
/// picking needed no new collision code, only the ray. `mask` is
/// [`ALL_LAYERS`] by default because an editor wants to select the thing you
/// are looking at, not the thing your character could stand on.
pub fn pick_world(world: &CollisionWorld, ray: Ray, max_dist: f32, mask: u16) -> Option<RayHit> {
    world.raycast(ray.origin, ray.dir, max_dist, mask)
}

/// [`pick_world`] against every layer — the call an editor actually makes.
pub fn pick_any(world: &CollisionWorld, ray: Ray, max_dist: f32) -> Option<RayHit> {
    pick_world(world, ray, max_dist, ALL_LAYERS)
}

// ---------------------------------------------------------------------------
// Tools, axes, snapping
// ---------------------------------------------------------------------------

/// Which of the five things the pointer is currently for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Tool {
    /// Click picks; nothing drags. The tool you can never lose anything in, and
    /// therefore the default.
    #[default]
    Select,
    Move,
    Rotate,
    Scale,
    /// Click places the current palette entry where the ray lands.
    Place,
}

impl Tool {
    /// The tools, in the order the keys 1..5 select them.
    pub const ALL: [Tool; 5] = [
        Tool::Select,
        Tool::Move,
        Tool::Rotate,
        Tool::Scale,
        Tool::Place,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Tool::Select => "select",
            Tool::Move => "move",
            Tool::Rotate => "rotate",
            Tool::Scale => "scale",
            Tool::Place => "place",
        }
    }

    /// The drag this tool performs, or `None` for the two that do not drag.
    pub fn drag_kind(self) -> Option<DragKind> {
        match self {
            Tool::Move => Some(DragKind::Translate),
            Tool::Rotate => Some(DragKind::Rotate),
            Tool::Scale => Some(DragKind::Scale),
            Tool::Select | Tool::Place => None,
        }
    }
}

/// What a drag does to a transform. The tool's verb, without the tool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DragKind {
    Translate,
    Rotate,
    Scale,
}

/// One of the three handle directions.
///
/// Always **local** to the selection: the gizmo's basis is the selected node's
/// own rotation, which is what makes an axis scale shear-free by construction —
/// scaling along a world axis a rotated brush is not aligned with has nowhere to
/// put the shear it produces, and a `Trs` has no field for one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    pub const ALL: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

    pub fn unit(self) -> Vec3 {
        match self {
            Axis::X => Vec3::X,
            Axis::Y => Vec3::Y,
            Axis::Z => Vec3::Z,
        }
    }

    pub fn index(self) -> usize {
        match self {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Axis::X => "x",
            Axis::Y => "y",
            Axis::Z => "z",
        }
    }

    /// The handle's colour, in the convention every 3D tool since the nineties
    /// has used: X red, Y green, Z blue.
    pub fn color(self) -> glam::Vec4 {
        match self {
            Axis::X => glam::Vec4::new(0.92, 0.25, 0.28, 1.0),
            Axis::Y => glam::Vec4::new(0.36, 0.85, 0.32, 1.0),
            Axis::Z => glam::Vec4::new(0.28, 0.5, 0.95, 1.0),
        }
    }
}

/// The quantization a drag lands on.
///
/// **The step is applied to the drag's amount, not to the resulting position.**
/// That is the rule that keeps a group edit honest: an absolute snap would drag
/// a hundred brushes onto one grid point, where a snapped *delta* moves them all
/// by the same whole number of cells and preserves the shape they were authored
/// in. A placement is the exception and snaps absolutely — see
/// [`snap_to`] and [`EditorState::place_at`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Snap {
    /// Translation step, world units.
    pub grid: f32,
    /// Rotation step, degrees.
    pub angle_deg: f32,
    /// Scale step, as a multiplier increment (`0.1` = 10% notches).
    pub scale: f32,
    /// Off means continuous. Toggled with one key; the steps stay as they were,
    /// so turning it back on resumes the same grid.
    pub on: bool,
}

impl Default for Snap {
    fn default() -> Snap {
        Snap {
            // Half a metre: the playground's shelves and steps were authored on
            // roughly this, and it is coarse enough that a dragged box lands
            // somewhere a second box can meet.
            grid: 0.5,
            angle_deg: 15.0,
            scale: 0.1,
            on: true,
        }
    }
}

impl Snap {
    /// Quantize a drag amount for `kind`. A no-op while [`Snap::on`] is false.
    pub fn amount(&self, kind: DragKind, raw: f32) -> f32 {
        if !self.on {
            return raw;
        }
        match kind {
            DragKind::Translate => snap_to(raw, self.grid),
            DragKind::Rotate => snap_to(raw, self.angle_deg.to_radians()),
            DragKind::Scale => snap_to(raw, self.scale),
        }
    }

    /// Quantize a world position — what the place tool puts a new node on.
    pub fn position(&self, p: Vec3) -> Vec3 {
        if !self.on {
            return p;
        }
        Vec3::new(
            snap_to(p.x, self.grid),
            snap_to(p.y, self.grid),
            snap_to(p.z, self.grid),
        )
    }
}

/// Round `value` to the nearest multiple of `step`. A non-positive or
/// non-finite step is "no snapping", never a division by zero.
pub fn snap_to(value: f32, step: f32) -> f32 {
    if !step.is_finite() || step <= 0.0 {
        return value;
    }
    (value / step).round() * step
}

// ---------------------------------------------------------------------------
// The seam
// ---------------------------------------------------------------------------

/// One placeable thing, as the palette shows it.
///
/// The engine deliberately knows nothing but the label: what a "box" or a
/// "moving platform" *is* belongs entirely to the adapter, which is handed back
/// the index it published.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PaletteEntry {
    pub label: String,
    /// A one-word group for the status line — `"csg"`, `"object"`. Purely
    /// cosmetic; the toolkit never branches on it.
    pub group: String,
}

impl PaletteEntry {
    pub fn new(group: impl Into<String>, label: impl Into<String>) -> PaletteEntry {
        PaletteEntry {
            group: group.into(),
            label: label.into(),
        }
    }
}

/// A completed drag, in the terms an adapter needs to turn it into an op.
///
/// The toolkit resolves the pointer arithmetic (which axis, how far along it,
/// snapped) and hands over the *result*, because how a delta reaches the scene
/// depends on what the id addresses: a single node takes
/// [`dragged`](crate::editor_gizmo::dragged), a fold of a hundred brushes takes
/// [`delta_matrix`](crate::editor_gizmo::delta_matrix) applied to each. Both are
/// engine functions; only the choice between them is the game's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Drag {
    pub kind: DragKind,
    pub axis: Axis,
    /// Snapped. World units, radians, or a multiplier increment — per `kind`.
    pub amount: f32,
    /// The point the drag pivots about: the gizmo's origin.
    pub pivot: Vec3,
    /// The gizmo's basis, so `axis` is a world direction the adapter can use
    /// without re-deriving it.
    pub basis: Quat,
}

impl Drag {
    /// The axis as a world-space unit vector.
    pub fn world_axis(&self) -> Vec3 {
        self.basis * self.axis.unit()
    }

    /// Is this drag a no-op? A snapped drag spends most of its life here, and an
    /// adapter that minted an op per frame anyway would fill the undo stack with
    /// nothing.
    pub fn is_null(&self) -> bool {
        self.amount == 0.0
    }
}

/// What went wrong applying an op. Deliberately small: an editor that cannot
/// apply an edit says so on the status line and carries on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EditError {
    /// The op names something the scene no longer has.
    NoSuchId(String),
    /// The op is not one this scene can invert or apply — see
    /// [`EditableScene::invert`].
    NotInvertible,
    /// Serialization, or a host-side write.
    Io(String),
}

impl std::fmt::Display for EditError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EditError::NoSuchId(id) => write!(f, "no such node {id}"),
            EditError::NotInvertible => write!(f, "the op cannot be inverted"),
            EditError::Io(e) => write!(f, "{e}"),
        }
    }
}

/// The seam between the toolkit and a game's own scene format.
///
/// # The contract, in three sentences
///
/// 1. An **id is stable across a reload**. The toolkit holds one as a selection
///    across a hot-reload that despawns and respawns the whole world, so an id
///    that were an `Entity` would select a different node every edit.
/// 2. Every **op carries enough to invert exactly**. Not "approximately": the
///    op log is the undo stack, so `apply(invert(op))` after `apply(op)` must
///    restore the scene byte for byte. In practice that means an op which
///    changes a value carries the old one, and an op which removes a node
///    carries the node.
/// 3. `apply` changes the **description**, never the world. Re-realizing the
///    world is the game's job and happens after — see the hot-reload note on
///    [`EditorState::dirty`].
///
/// # Why the ops are the game's type
///
/// The sketch this landed from had `enum EditOp` in the engine. It is an
/// associated type instead, and the reason is the same one that keeps
/// `TextureSpec` out of `runt-app`: the engine cannot name a "CSG root with a
/// phase-layer mask", so an engine-side op enum would either be a `Box<dyn Any>`
/// with extra steps or a vocabulary the engine has no business having. What the
/// toolkit actually needs from an op is that it can be stored, applied and
/// inverted — three lines of trait — and the adapter supplies the meaning.
pub trait EditableScene {
    /// Stable across reloads. In practice an index into the scene file, or a
    /// name.
    type Id: Copy + Eq + std::fmt::Debug;

    /// One edit. `Clone` because the log stores it and undo applies its inverse.
    type Op: Clone;

    /// Which scene node a world entity belongs to, if any.
    ///
    /// The world hit is whatever collider the ray found — a merged terrain soup,
    /// a trigger volume, one visual part of a five-part object — and the adapter
    /// walks that back to the node the scene file names.
    fn pick_id(&self, entity: Entity) -> Option<Self::Id>;

    /// Where the gizmo goes and how it is oriented: the node's own pose.
    ///
    /// `scale` is read by the scale tool and ignored by the other two, so an
    /// adapter whose node has no meaningful scale may return
    /// [`Vec3::ONE`](glam::Vec3::ONE).
    fn anchor(&self, id: Self::Id) -> Option<Transform>;

    /// Turn a finished drag into an op, or `None` if this node does not take it
    /// (a fold of many brushes refusing a per-axis scale, say).
    fn drag_op(&self, id: Self::Id, drag: &Drag) -> Option<Self::Op>;

    /// What the place tool can place.
    fn palette(&self) -> &[PaletteEntry];

    /// Place palette entry `index` at `at`.
    fn place_op(&self, index: usize, at: &Transform) -> Option<Self::Op>;

    /// Remove `id`. The op must carry the node it removed (contract 2).
    fn delete_op(&self, id: Self::Id) -> Option<Self::Op>;

    /// Apply an op to the description. Never touches the world.
    fn apply(&mut self, op: &Self::Op) -> Result<(), EditError>;

    /// The op that exactly undoes `op` **against the scene as it is now**.
    ///
    /// Called immediately before the inverse is applied, so an adapter may read
    /// current state — but it must not need to: an op that carried the old value
    /// inverts without looking, which is the shape that survives being stored
    /// in a log and replayed later.
    fn invert(&self, op: &Self::Op) -> Option<Self::Op>;

    /// Which node an op is about, for the status line and for re-selecting after
    /// an undo. `None` for ops with no single subject.
    fn op_id(&self, op: &Self::Op) -> Option<Self::Id>;

    /// A one-line description, for the status line.
    fn describe(&self, op: &Self::Op) -> String;

    /// The scene, as the text its own loader reads back.
    fn serialize(&self) -> Result<String, EditError>;
}

// ---------------------------------------------------------------------------
// The op log
// ---------------------------------------------------------------------------

/// How many ops a log keeps before the oldest falls off the bottom.
///
/// A guard rather than a budget, [`tweak::MAX_FIELDS_PER_ROOT`]'s reason: an
/// unbounded log in a tool you leave open all afternoon is a slow leak of whole
/// scene nodes, and nobody has ever wanted the two-hundredth undo.
///
/// [`tweak::MAX_FIELDS_PER_ROOT`]: crate::tweak::MAX_FIELDS_PER_ROOT
pub const MAX_LOG: usize = 256;

/// Applied ops, and where in them we are. **The undo stack, and the only one.**
///
/// ```text
/// entries  [ a  b  c  d  e ]
/// cursor            ^        three applied, two undone
/// ```
///
/// `cursor` is the number of ops currently *in effect*. `undo` steps it back and
/// hands out the op that must be inverted; `redo` steps it forward and hands out
/// the op to re-apply; [`push`](OpLog::push) **truncates the tail**, because a
/// new edit after an undo makes the redone future unreachable — the classic
/// linear-history rule, and the one users have muscle memory for.
#[derive(Clone, Debug)]
pub struct OpLog<Op> {
    entries: Vec<Op>,
    cursor: usize,
    limit: usize,
}

impl<Op> Default for OpLog<Op> {
    fn default() -> OpLog<Op> {
        OpLog::new()
    }
}

impl<Op> OpLog<Op> {
    pub fn new() -> OpLog<Op> {
        OpLog {
            entries: Vec::new(),
            cursor: 0,
            limit: MAX_LOG,
        }
    }

    pub fn with_limit(limit: usize) -> OpLog<Op> {
        OpLog {
            entries: Vec::new(),
            cursor: 0,
            limit: limit.max(1),
        }
    }

    /// Ops in effect. Also the undo depth.
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Ops held, applied and undone together.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn can_undo(&self) -> bool {
        self.cursor > 0
    }

    pub fn can_redo(&self) -> bool {
        self.cursor < self.entries.len()
    }

    /// Record an op that has just been applied. Drops the redo tail, and the
    /// oldest entry once the log is [`MAX_LOG`] deep.
    pub fn push(&mut self, op: Op) {
        self.entries.truncate(self.cursor);
        self.entries.push(op);
        if self.entries.len() > self.limit {
            self.entries.remove(0);
        }
        self.cursor = self.entries.len();
    }

    /// The op to invert, and the step back. `None` at the bottom.
    pub fn step_back(&mut self) -> Option<&Op> {
        if self.cursor == 0 {
            return None;
        }
        self.cursor -= 1;
        self.entries.get(self.cursor)
    }

    /// The op to re-apply, and the step forward. `None` at the top.
    pub fn step_forward(&mut self) -> Option<&Op> {
        let op = self.entries.get(self.cursor)?;
        self.cursor += 1;
        Some(op)
    }

    /// The most recently applied op, if any.
    pub fn last(&self) -> Option<&Op> {
        self.cursor.checked_sub(1).and_then(|i| self.entries.get(i))
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.cursor = 0;
    }
}

/// Apply an op and record it. The one way an edit reaches a scene.
///
/// On failure nothing is logged, so a refused edit leaves the undo stack exactly
/// as it was — an undo that "undid" an op which never landed is the worst bug an
/// editor can have.
pub fn apply<S: EditableScene>(
    scene: &mut S,
    log: &mut OpLog<S::Op>,
    op: S::Op,
) -> Result<(), EditError> {
    scene.apply(&op)?;
    log.push(op);
    Ok(())
}

/// Undo one op. `Ok(false)` at the bottom of the stack — not an error, just
/// nothing left.
pub fn undo<S: EditableScene>(scene: &mut S, log: &mut OpLog<S::Op>) -> Result<bool, EditError> {
    let Some(op) = log.step_back().cloned() else {
        return Ok(false);
    };
    let Some(inverse) = scene.invert(&op) else {
        // Put the cursor back: refusing an undo must not silently consume it.
        log.step_forward();
        return Err(EditError::NotInvertible);
    };
    match scene.apply(&inverse) {
        Ok(()) => Ok(true),
        Err(e) => {
            log.step_forward();
            Err(e)
        }
    }
}

/// Redo one op. `Ok(false)` at the top.
pub fn redo<S: EditableScene>(scene: &mut S, log: &mut OpLog<S::Op>) -> Result<bool, EditError> {
    let Some(op) = log.step_forward().cloned() else {
        return Ok(false);
    };
    match scene.apply(&op) {
        Ok(()) => Ok(true),
        Err(e) => {
            log.step_back();
            Err(e)
        }
    }
}

// ---------------------------------------------------------------------------
// Editor state
// ---------------------------------------------------------------------------

/// How far in front of the camera a placement lands when the ray hits nothing.
pub const PLACE_FALLBACK_DIST: f32 = 8.0;

/// How far a pick ray reaches. Past the playground's diameter, and finite so a
/// terrain field march terminates.
pub const PICK_DIST: f32 = 500.0;

/// A live drag: which handle is held, and where it was grabbed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grab {
    pub kind: DragKind,
    pub axis: Axis,
    /// The gizmo pose the drag started against. Frozen for the duration, so a
    /// node moving under its own gizmo cannot make the drag chase itself.
    pub pivot: Vec3,
    pub basis: Quat,
    pub gizmo_scale: f32,
    /// The handle parameter at the moment of the grab.
    pub from: f32,
    /// The amount already committed by ops this drag has emitted.
    pub applied: f32,
}

/// Everything the editor is currently doing, minus the scene.
///
/// Parameterized by the adapter's id type. Not a `Resource` — see the module
/// docs; the game wraps this and its [`OpLog`] in one of its own.
#[derive(Clone, Debug, PartialEq)]
pub struct EditorState<Id> {
    /// Is edit mode on at all? Everything below is inert while this is false.
    pub on: bool,
    pub tool: Tool,
    pub selection: Option<Id>,
    /// Which handle the pointer is over, recomputed every tick.
    pub hover: Option<Axis>,
    pub grab: Option<Grab>,
    pub snap: Snap,
    /// The editor's own pointer, logical pixels, top-left origin. See the module
    /// docs on why it is not the host's.
    pub cursor: Vec2,
    /// Pointer speed, pixels per pixel of mouse delta.
    pub sensitivity: f32,
    /// Which palette entry the place tool will place.
    pub palette_index: usize,
    /// **The hot-reload signal.** Set by every applied op; the game clears it
    /// once it has re-realized the world. The engine says *that* the description
    /// changed and never how much of the world to rebuild — a full respawn and a
    /// surgical one are both correct answers and only the game knows which it
    /// can afford.
    pub dirty: bool,
    /// The last thing that happened, for the status line.
    pub message: String,
}

impl<Id> Default for EditorState<Id> {
    fn default() -> EditorState<Id> {
        EditorState::new()
    }
}

impl<Id> EditorState<Id> {
    pub fn new() -> EditorState<Id> {
        EditorState {
            on: false,
            tool: Tool::Select,
            selection: None,
            hover: None,
            grab: None,
            snap: Snap::default(),
            cursor: Vec2::ZERO,
            sensitivity: 1.0,
            palette_index: 0,
            dirty: false,
            message: String::new(),
        }
    }

    /// Turn edit mode on or off. Leaving drops any drag in progress but keeps
    /// the selection, so toggling out to play and back in resumes where you were.
    pub fn set_on(&mut self, on: bool) {
        if self.on == on {
            return;
        }
        self.on = on;
        self.grab = None;
        self.hover = None;
        self.message = if on { "edit".into() } else { "play".into() };
    }

    pub fn toggle(&mut self) {
        self.set_on(!self.on);
    }

    /// Move the pointer by a mouse delta and clamp it inside the viewport.
    ///
    /// Clamped rather than wrapped: a crosshair that leaves the screen is a
    /// crosshair you have to hunt for, and there is no OS cursor to fall back on.
    pub fn move_cursor(&mut self, delta: Vec2, viewport: Viewport) {
        if !viewport.is_known() {
            return;
        }
        let size = viewport.size();
        self.cursor = (self.cursor + delta * self.sensitivity).clamp(Vec2::ZERO, size);
    }

    /// Centre the pointer — what entering edit mode does, since the cursor has
    /// nowhere else it could have been.
    pub fn centre_cursor(&mut self, viewport: Viewport) {
        if viewport.is_known() {
            self.cursor = viewport.size() * 0.5;
        }
    }

    /// Where a placement lands for a ray that hit `hit` (or missed).
    ///
    /// Snapped **absolutely**, unlike a drag: a new node has no authored
    /// position to preserve, so putting it on the grid is strictly better than
    /// putting it wherever the pixel was. See [`Snap`].
    pub fn place_at(&self, ray: Ray, hit: Option<&RayHit>) -> Vec3 {
        let raw = match hit {
            Some(hit) => hit.point,
            None => ray.at(PLACE_FALLBACK_DIST),
        };
        self.snap.position(raw)
    }

    /// Record an applied op's effect on the state: the world needs rebuilding,
    /// and the status line has news.
    pub fn touched(&mut self, message: impl Into<String>) {
        self.dirty = true;
        self.message = message.into();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapping_rounds_to_the_nearest_multiple_and_survives_a_zero_step() {
        assert_eq!(snap_to(2.3, 0.5), 2.5);
        assert_eq!(snap_to(-2.3, 0.5), -2.5);
        assert_eq!(snap_to(0.24, 0.5), 0.0);
        // A step of zero is "no snapping", never a NaN.
        assert_eq!(snap_to(2.3, 0.0), 2.3);
        assert_eq!(snap_to(2.3, -1.0), 2.3);
        assert_eq!(snap_to(2.3, f32::NAN), 2.3);
    }

    #[test]
    fn a_snap_off_is_continuous() {
        let mut snap = Snap::default();
        assert_eq!(snap.amount(DragKind::Translate, 2.3), 2.5);
        snap.on = false;
        assert_eq!(snap.amount(DragKind::Translate, 2.3), 2.3);
        assert_eq!(snap.position(Vec3::splat(2.3)), Vec3::splat(2.3));
    }

    #[test]
    fn the_cursor_stays_inside_the_viewport() {
        let mut state: EditorState<u32> = EditorState::new();
        let viewport = Viewport::new(800, 600);
        state.centre_cursor(viewport);
        assert_eq!(state.cursor, Vec2::new(400.0, 300.0));
        state.move_cursor(Vec2::new(10_000.0, -10_000.0), viewport);
        assert_eq!(state.cursor, Vec2::new(800.0, 0.0));
        // …and an unknown viewport moves nothing rather than clamping to zero.
        state.move_cursor(Vec2::new(5.0, 5.0), Viewport::ZERO);
        assert_eq!(state.cursor, Vec2::new(800.0, 0.0));
    }
}
