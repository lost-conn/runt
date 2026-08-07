//! Live-tunable parameters — runt's `@export` (DESIGN §3, §10) — `reflect`
//! feature only.
//!
//! Godot's `@export var follow_speed: float = 10.0` does three things at once:
//! it declares a field, it puts a slider in the inspector, and it remembers what
//! the slider was left at. This module is the first two of those for runt, and
//! [`TweakOverrides`] is the third — split apart, because the engine has no
//! inspector and no files, and each of those is somebody else's job.
//!
//! ```text
//! Tweakables      which resources/components are tunable, and how to reach them
//! TweakField      one leaf: a path, a kind, a range, and its value right now
//! TweakOverrides  path → value, serializable; the host decides where bytes live
//! tweak_panel     the debug overlay that drives all of the above
//! ```
//!
//! # Opt-in, by name, at runtime
//!
//! Nothing is tunable until a game says so:
//!
//! ```ignore
//! runt_core::tweak::install(sim.world_mut());
//! runt_core::tweak::register_resource::<Lighting>(sim.world_mut(), "sky");
//! runt_core::tweak::register_component::<CameraRig>(sim.world_mut(), "camera", rig);
//! ```
//!
//! Registration is explicit rather than inventory- or registry-driven for the
//! reason [`crate::reflect::type_registry`] is: the set is greppable, it cannot
//! vary with link order, and a game decides what it is willing to have a stranger
//! drag a slider on. There is deliberately no `TypeRegistry` in this module at
//! all — every piece of metadata a panel needs (field names, nesting,
//! [`FieldRange`]) comes off `T`'s static [`TypeInfo`], which
//! `#[derive(Reflect)]` already generates, and the *access* comes off two
//! monomorphized function pointers per root. A registry would be a second place
//! for a type to be missing from.
//!
//! # Paths
//!
//! A root's fields are flattened depth-first into dotted paths —
//! `sky.sky_color.x`, `camera.zoom_target`, `render_scale.0`. The walk descends
//! through structs and tuple structs, so a `Vec3` field carrying
//! `#[reflect(remote = Vec3Def)]` arrives as three `f32` leaves and needs no
//! special case anywhere. A path is the **stable identity** of a tunable: it is
//! what an override file keys on, so renaming a field orphans its override
//! (which is then reported and ignored, never guessed at).
//!
//! # What is a leaf, and what is out
//!
//! | reflected as | treated as |
//! |---|---|
//! | `f32` | [`TweakValue::Float`] |
//! | `bool` | [`TweakValue::Bool`] |
//! | `u8`/`u32`/`u64`/`usize`/`i32`/`i64` | [`TweakValue::Int`] |
//! | struct / tuple struct | descended into |
//! | enum, all variants unit | [`TweakValue::Choice`] — cycled by name |
//! | **enum with data** | **skipped** |
//! | `f64`, `String`, anything else | **skipped** |
//!
//! The data-carrying enum is the deliberate cut. Descending into one means the
//! set of paths changes when the *value* changes — `terrain.color.Some.x` exists
//! only while the option is `Some` — and a path that comes and goes is not a
//! stable key for an override file. `Option<Vec3>`
//! ([`OptVec3Def`](crate::reflect::OptVec3Def)) and
//! [`GeneratorSpec`](crate::gen::GeneratorSpec) are therefore both invisible
//! here; editing a generator is the scene editor's job, which has a variant
//! dropdown and a rebuild, and is a different tool from a slider that nudges a
//! live number. Skipping is silent by design: a root is registered for the two
//! fields somebody wants to tune, not audited for the six they do not.
//!
//! # Bounds
//!
//! Straight off [`FieldRange`], which lives at the param
//! (`crate::reflect`'s doctrine: bounds are declared next to the value, never in
//! a side table the editor owns). A range declared on a *composite* field is
//! **inherited by its leaves** — `#[reflect(remote = Vec3Def, @FieldRange::new(0.0, 1.0))]`
//! on a colour bounds all three channels — which is exactly how
//! [`TerrainParamsDef`](crate::reflect::TerrainParamsDef) already annotates its
//! own. Unannotated leaves fall back to [`DEFAULT_RANGE`] /
//! [`DEFAULT_INT_RANGE`] / [`DEFAULT_SIGNED_RANGE`].
//!
//! Unlike a `FieldRange` in the generator panel — where the doc string calls it
//! "advisory, not a constraint" — [`set`](Tweakables::set) **clamps**. The
//! difference is that this writes into a *live* world: a generator param out of
//! range makes an ugly mesh, and a `follow_speed` of −4000 makes a camera that
//! never comes back. A tunable a panel cannot get out of is worth more than the
//! last decimal of reach, and the authored value is always one keypress away.
//!
//! # No files
//!
//! [`TweakOverrides`] serializes to and from RON strings and that is the whole
//! of its I/O. Where the bytes go — `localStorage`, a config file, a `.ron`
//! beside the level — is the host's decision, the same seam
//! [`runt_app::storage`](https://docs.rs) gives the bindings table. The engine
//! has no `std::fs` in its contract and does not grow one for a debug feature.

use std::collections::BTreeMap;

use bevy_ecs::component::Mutable;
use bevy_ecs::prelude::*;
use bevy_reflect::{PartialReflect, Reflect, ReflectMut, ReflectRef, TypeInfo, Typed};
use serde::{Deserialize, Serialize};

use crate::reflect::{FieldRange, DEFAULT_INT_RANGE, DEFAULT_RANGE};

/// The bounds an unannotated **signed** integer param gets.
///
/// [`DEFAULT_INT_RANGE`] starts at zero, which is right for a segment count and
/// wrong for anything that can go the other way; a slider that cannot reach −1
/// is worse than a wide one.
pub const DEFAULT_SIGNED_RANGE: FieldRange = FieldRange {
    min: -256.0,
    max: 256.0,
    step: 1.0,
};

/// How many leaves one root may contribute before the walk gives up.
///
/// A guard, not a budget: reflection can walk into a type whose nesting the
/// author did not picture, and a debug panel that allocates a hundred thousand
/// rows because somebody registered the wrong resource is a hang rather than a
/// mistake you can see. Every real root is a dozen.
pub const MAX_FIELDS_PER_ROOT: usize = 512;

/// How deep the walk will descend before it stops.
///
/// The other half of the same guard, and the half that protects the *stack*
/// rather than the heap. A reflected `Vec3` inside a struct inside a resource is
/// three; anything past this is a type nobody meant to register.
pub const MAX_DEPTH: usize = 8;

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

/// One tunable leaf's value, in the four shapes a panel knows how to draw.
///
/// `Serialize`/`Deserialize` because this *is* the override file's payload —
/// the variant tag is what makes an override that no longer matches its field's
/// type detectable rather than silently coerced.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TweakValue {
    Float(f32),
    Bool(bool),
    /// Every integer width, widened. A panel has one integer widget and an
    /// override file should not care that a field went from `u8` to `u32`.
    Int(i64),
    /// A unit enum variant, **by name**. Not by index: appending a variant must
    /// not silently repoint every stored override one variant along.
    Choice(String),
}

impl TweakValue {
    /// The value as a panel prints it: floats trimmed, bools as words.
    pub fn display(&self) -> String {
        match self {
            TweakValue::Float(v) => {
                let text = format!("{v:.3}");
                let trimmed = text.trim_end_matches('0').trim_end_matches('.');
                if trimmed.is_empty() || trimmed == "-" {
                    "0".to_string()
                } else {
                    trimmed.to_string()
                }
            }
            TweakValue::Bool(v) => if *v { "on" } else { "off" }.to_string(),
            TweakValue::Int(v) => v.to_string(),
            TweakValue::Choice(v) => v.clone(),
        }
    }
}

/// One tunable leaf, as a panel sees it.
///
/// Produced fresh by [`fields_of`] rather than retained and diffed — the same
/// call [`UiBatch`](crate::ui::UiBatch) makes, and for the same reason: a list
/// rebuilt from the world cannot go stale, and a debug overlay allocating a few
/// dozen `String`s on the ticks it is open is not a cost anybody can measure.
#[derive(Clone, Debug, PartialEq)]
pub struct TweakField {
    /// The stable dotted path — `sky.clouds`. What an override keys on.
    pub path: String,
    /// The root this belongs to, by index into [`Tweakables::roots`].
    pub root: usize,
    /// The leaf's own tail of the path — `clouds`, or `sky_color.x` for a leaf
    /// one level down. What a panel puts in the left column.
    pub label: String,
    /// The value in the world, right now.
    pub value: TweakValue,
    /// Where a slider's ends sit, and what one arrow-key press is worth.
    pub range: FieldRange,
    /// For [`TweakValue::Choice`], every variant name in declaration order.
    /// Empty otherwise.
    pub choices: Vec<&'static str>,
    /// Whether [`TweakOverrides`] currently carries an entry for this path.
    pub overridden: bool,
}

impl TweakField {
    /// One arrow-key press, in the value's own units.
    ///
    /// The declared [`FieldRange::step`] when there is one; otherwise a
    /// hundredth of the range for a float (so a full sweep is a hundred presses,
    /// which is about as long as anybody will hold an arrow key) and 1 for an
    /// integer.
    pub fn step(&self) -> f32 {
        if self.range.step > 0.0 {
            return self.range.step;
        }
        match self.value {
            TweakValue::Float(_) => ((self.range.max - self.range.min) / 100.0).abs(),
            _ => 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// The registry
// ---------------------------------------------------------------------------

/// How one root is reached in a world.
///
/// Two monomorphized function pointers per root rather than a `Box<dyn Fn>`:
/// registration is a `Vec` push with no allocation, the pointers are the
/// *only* thing that has to know `T`, and nothing here needs a `TypeRegistry`
/// to get from a name to a value.
#[derive(Clone, Copy)]
enum Reach {
    Resource {
        read: fn(&World) -> Option<&dyn PartialReflect>,
        write: fn(&mut World) -> Option<&mut dyn PartialReflect>,
    },
    Component {
        entity: Entity,
        read: fn(&World, Entity) -> Option<&dyn PartialReflect>,
        write: fn(&mut World, Entity) -> Option<&mut dyn PartialReflect>,
    },
}

/// One registered tunable root.
#[derive(Clone, Copy)]
pub struct TweakRoot {
    /// The first path segment — `sky` in `sky.clouds`. Chosen by the game.
    pub name: &'static str,
    reach: Reach,
}

impl TweakRoot {
    fn read<'w>(&self, world: &'w World) -> Option<&'w dyn PartialReflect> {
        match self.reach {
            Reach::Resource { read, .. } => read(world),
            Reach::Component { entity, read, .. } => read(world, entity),
        }
    }

    fn write<'w>(&self, world: &'w mut World) -> Option<&'w mut dyn PartialReflect> {
        match self.reach {
            Reach::Resource { write, .. } => write(world),
            Reach::Component { entity, write, .. } => write(world, entity),
        }
    }
}

impl std::fmt::Debug for TweakRoot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self.reach {
            Reach::Resource { .. } => "resource",
            Reach::Component { entity, .. } => {
                return write!(f, "{}: component {entity}", self.name)
            }
        };
        write!(f, "{}: {kind}", self.name)
    }
}

/// Every tunable root a game has opted in, and what each one was worth when it
/// was registered.
///
/// The world's own inventory of "things a panel may edit". Empty by default, so
/// a game that never calls [`register_resource`] carries one empty `Vec` and
/// nothing else.
#[derive(Resource, Default)]
pub struct Tweakables {
    roots: Vec<TweakRoot>,
    /// path → the value the world was **built with**, snapshotted at
    /// registration.
    ///
    /// Registration-time rather than first-edit, and that is the whole meaning
    /// of [`clear`]: "put it back the way the game shipped it", not "put it back
    /// the way I found it this session". The two differ exactly when an override
    /// file was loaded, which is the case the distinction exists for — a
    /// developer who has been dragging sliders for an hour wants the ground
    /// truth back, not their own last guess.
    ///
    /// The consequence is an ordering rule, stated here because it is the one
    /// way to get this wrong: **register before applying overrides.** A root
    /// registered after [`TweakOverrides::apply`] has snapshotted the override
    /// as the authored value, and `clear` on it is a no-op.
    authored: BTreeMap<String, TweakValue>,
}

impl Tweakables {
    pub fn new() -> Tweakables {
        Tweakables::default()
    }

    /// The registered roots, in registration order — which is also the order
    /// [`fields_of`] walks and the order the panel lists.
    pub fn roots(&self) -> &[TweakRoot] {
        &self.roots
    }

    /// The value `path` had when its root was registered, if it had one.
    pub fn authored(&self, path: &str) -> Option<&TweakValue> {
        self.authored.get(path)
    }

    /// Every leaf under every root, depth-first in registration order.
    ///
    /// A root whose resource or entity is missing contributes nothing and is not
    /// an error: a component root outlives the entity it named, and a panel that
    /// panicked when a level unloaded would be a panel nobody leaves open.
    pub fn fields(&self, world: &World) -> Vec<TweakField> {
        let mut out = Vec::new();
        for (index, root) in self.roots.iter().enumerate() {
            let Some(value) = root.read(world) else {
                continue;
            };
            let start = out.len();
            let mut path = String::from(root.name);
            walk(value, &mut path, index, 0, None, &mut out);
            if out.len() - start > MAX_FIELDS_PER_ROOT {
                log::warn!(
                    "tweak: root {:?} walked to {} leaves — truncated at {MAX_FIELDS_PER_ROOT}",
                    root.name,
                    out.len() - start
                );
                out.truncate(start + MAX_FIELDS_PER_ROOT);
            }
        }
        out
    }

    /// Read one leaf.
    pub fn get(&self, world: &World, path: &str) -> Option<TweakValue> {
        let (root, rest) = split_root(path)?;
        let index = self.roots.iter().position(|r| r.name == root)?;
        let value = self.roots[index].read(world)?;
        let (leaf, _) = resolve(value, rest)?;
        leaf_value(leaf)
    }

    /// Write one leaf, clamped to its [`FieldRange`], returning the value that
    /// actually landed.
    ///
    /// The return is the *clamped* value rather than the requested one so a
    /// caller recording an override records what the world holds — otherwise a
    /// file could carry a number the world has never had.
    pub fn set(
        &self,
        world: &mut World,
        path: &str,
        value: TweakValue,
    ) -> Result<TweakValue, TweakError> {
        let (root, rest) = split_root(path).ok_or(TweakError::NoSuchPath)?;
        let index = self
            .roots
            .iter()
            .position(|r| r.name == root)
            .ok_or(TweakError::NoSuchPath)?;
        let target = self.roots[index]
            .write(world)
            .ok_or(TweakError::RootAbsent)?;
        let (leaf, range) = resolve_mut(target, rest).ok_or(TweakError::NoSuchPath)?;
        write_leaf(leaf, value, range)
    }
}

/// Why an edit did not land.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TweakError {
    /// No root of that name, or no such field under it.
    NoSuchPath,
    /// The root is registered but its resource — or the entity holding its
    /// component — is not in the world right now.
    RootAbsent,
    /// The path names a real field of a different shape: a `Bool` for an `f32`,
    /// a variant name the enum does not have.
    WrongKind,
}

impl std::fmt::Display for TweakError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            TweakError::NoSuchPath => "no such tweakable",
            TweakError::RootAbsent => "the root is not in the world",
            TweakError::WrongKind => "the value is the wrong kind for that field",
        };
        f.write_str(text)
    }
}

impl std::error::Error for TweakError {}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Put the three tweakable resources in `world`. Idempotent.
///
/// Separate from registration so a game can install the system once and register
/// roots from wherever each one is built.
pub fn install(world: &mut World) {
    world.init_resource::<Tweakables>();
    world.init_resource::<TweakOverrides>();
}

/// Make resource `T` tunable under `name`, and snapshot what it holds now.
///
/// Call **after** the resource is inserted and **before**
/// [`TweakOverrides::apply`] — see [`Tweakables::authored`] for why the order is
/// load-bearing. A resource that is not there yet is registered anyway (so a
/// later insert becomes visible) with a log line and no authored snapshot, which
/// makes `clear` on it a no-op rather than a lie.
pub fn register_resource<T: Resource<Mutability = Mutable> + Reflect + Typed>(
    world: &mut World,
    name: &'static str,
) {
    let root = TweakRoot {
        name,
        reach: Reach::Resource {
            read: |world| world.get_resource::<T>().map(|r| r as &dyn PartialReflect),
            write: |world| {
                world
                    .get_resource_mut::<T>()
                    .map(|r| r.into_inner() as &mut dyn PartialReflect)
            },
        },
    };
    push_root(world, root, std::any::type_name::<T>());
}

/// Make component `T` on `entity` tunable under `name`.
///
/// The entity is captured at registration: a root names *this* camera rig, not
/// "whatever has a `CameraRig`". A panel over "every entity with component X" is
/// a different tool (it needs a picker, and paths that survive a respawn), and
/// the one-named-entity case is what a game actually tunes.
pub fn register_component<T: Component<Mutability = Mutable> + Reflect + Typed>(
    world: &mut World,
    name: &'static str,
    entity: Entity,
) {
    let root = TweakRoot {
        name,
        reach: Reach::Component {
            entity,
            read: |world, entity| world.get::<T>(entity).map(|c| c as &dyn PartialReflect),
            write: |world, entity| {
                world
                    .get_mut::<T>(entity)
                    .map(|c| c.into_inner() as &mut dyn PartialReflect)
            },
        },
    };
    push_root(world, root, std::any::type_name::<T>());
}

/// The half of registration that is not generic: refuse a duplicate name, push,
/// snapshot.
fn push_root(world: &mut World, root: TweakRoot, type_name: &str) {
    let duplicate = world
        .get_resource::<Tweakables>()
        .is_some_and(|t| t.roots.iter().any(|r| r.name == root.name));
    if duplicate {
        // Two roots of one name would make every path under the second one
        // unreachable — `split_root` finds the first — which is a silently
        // dead panel section rather than a visible error. Refuse it.
        log::warn!(
            "tweak: root {:?} is already registered — ignoring {type_name}",
            root.name
        );
        return;
    }

    world.init_resource::<Tweakables>();
    world.resource_scope(|world, mut tweaks: Mut<Tweakables>| {
        tweaks.roots.push(root);
        let index = tweaks.roots.len() - 1;
        match root.read(world) {
            Some(value) => {
                let mut fields = Vec::new();
                let mut path = String::from(root.name);
                walk(value, &mut path, index, 0, None, &mut fields);
                log::info!(
                    "tweak: root {:?} = {type_name} ({} fields)",
                    root.name,
                    fields.len()
                );
                for field in fields {
                    tweaks.authored.insert(field.path, field.value);
                }
            }
            None => log::warn!(
                "tweak: root {:?} = {type_name} is not in the world yet — \
                 registered without an authored snapshot",
                root.name
            ),
        }
    });
}

// ---------------------------------------------------------------------------
// Overrides
// ---------------------------------------------------------------------------

/// Every tunable a run has moved off its authored value, keyed by path.
///
/// A `BTreeMap` rather than a `HashMap` so the RON comes out in one order on
/// every machine — a settings file that reshuffles itself on every save is a
/// file nobody can diff, and DESIGN §3's determinism rule reads on hash
/// iteration everywhere else in the engine for the same reason.
#[derive(Resource, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TweakOverrides {
    pub values: BTreeMap<String, TweakValue>,
}

impl TweakOverrides {
    pub fn new() -> TweakOverrides {
        TweakOverrides::default()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Pretty RON — the format the bindings table and the settings blob already
    /// use, and for their reason: the whole point of persisting a tweak is that
    /// somebody can open the file, read `"sky.clouds": Float(0.4)`, and paste it
    /// into the level.
    pub fn to_ron(&self) -> Result<String, ron::Error> {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::new())
    }

    pub fn from_ron(text: &str) -> Result<TweakOverrides, ron::error::SpannedError> {
        ron::from_str(text)
    }

    /// Write every override into the world, returning how many landed.
    ///
    /// Every failure is a dropped entry and a log line, never a stop: an
    /// override file outlives the field it names — a renamed param, a root a
    /// build no longer registers, a `bool` that became an enum — and a game that
    /// refuses to start because of a stale debug file is a worse outcome than a
    /// slider that is back where it started.
    ///
    /// The dead entries are **kept**, not pruned. A developer who switches
    /// branches should not lose an hour of tuning because the other branch had
    /// not landed the field yet.
    pub fn apply(&self, world: &mut World) -> usize {
        let Some(tweaks) = world.remove_resource::<Tweakables>() else {
            return 0;
        };
        let mut applied = 0usize;
        for (path, value) in &self.values {
            match tweaks.set(world, path, value.clone()) {
                Ok(_) => applied += 1,
                Err(e) => log::warn!("tweak: override {path:?} dropped — {e}"),
            }
        }
        world.insert_resource(tweaks);
        log::info!("tweak: applied {applied}/{} overrides", self.values.len());
        applied
    }
}

// ---------------------------------------------------------------------------
// The world-side edit door
// ---------------------------------------------------------------------------

/// Every leaf in the world, with [`TweakField::overridden`] filled in.
///
/// The call a panel makes. Returns empty rather than panicking when the
/// resources are absent, so a world without [`install`] simply has no tunables.
pub fn fields_of(world: &World) -> Vec<TweakField> {
    let Some(tweaks) = world.get_resource::<Tweakables>() else {
        return Vec::new();
    };
    let mut fields = tweaks.fields(world);
    if let Some(overrides) = world.get_resource::<TweakOverrides>() {
        for field in &mut fields {
            field.overridden = overrides.values.contains_key(&field.path);
        }
    }
    fields
}

/// Set one leaf **and record it** as an override.
///
/// The pair is one function because they must not come apart: an edit the world
/// took and the file did not is a tweak that evaporates at the next launch, and
/// an entry in the file the world refused is a lie about the run. What is
/// recorded is the *clamped* value [`Tweakables::set`] returned.
pub fn set_and_record(
    world: &mut World,
    path: &str,
    value: TweakValue,
) -> Result<TweakValue, TweakError> {
    let Some(tweaks) = world.remove_resource::<Tweakables>() else {
        return Err(TweakError::RootAbsent);
    };
    let result = tweaks.set(world, path, value);
    world.insert_resource(tweaks);
    if let Ok(landed) = &result {
        world
            .get_resource_or_insert_with(TweakOverrides::default)
            .values
            .insert(path.to_string(), landed.clone());
    }
    result
}

/// Put one leaf back to what the world was built with, and forget the override.
///
/// `Ok(None)` means there was nothing authored to go back to — a root registered
/// before its resource existed. The override is dropped either way, because an
/// entry nobody can restore from is exactly the stale entry
/// [`TweakOverrides::apply`] would warn about forever.
pub fn clear(world: &mut World, path: &str) -> Result<Option<TweakValue>, TweakError> {
    let Some(tweaks) = world.remove_resource::<Tweakables>() else {
        return Err(TweakError::RootAbsent);
    };
    let authored = tweaks.authored.get(path).cloned();
    let result = match &authored {
        Some(value) => tweaks.set(world, path, value.clone()).map(Some),
        None => Ok(None),
    };
    world.insert_resource(tweaks);
    if result.is_ok() {
        if let Some(mut overrides) = world.get_resource_mut::<TweakOverrides>() {
            overrides.values.remove(path);
        }
    }
    result
}

/// [`clear`], for every override there is. Returns how many were restored.
pub fn clear_all(world: &mut World) -> usize {
    let paths: Vec<String> = world
        .get_resource::<TweakOverrides>()
        .map(|o| o.values.keys().cloned().collect())
        .unwrap_or_default();
    let mut restored = 0usize;
    for path in paths {
        if matches!(clear(world, &path), Ok(Some(_))) {
            restored += 1;
        }
    }
    restored
}

// ---------------------------------------------------------------------------
// The walk
// ---------------------------------------------------------------------------

/// `"sky.clouds"` → `("sky", "clouds")`. A bare root name yields an empty tail,
/// which [`resolve`] reads as "the root itself".
fn split_root(path: &str) -> Option<(&str, &str)> {
    match path.split_once('.') {
        Some((root, rest)) => Some((root, rest)),
        None if path.is_empty() => None,
        None => Some((path, "")),
    }
}

/// The range declared on `field` of `info`, whether `info` names its fields or
/// numbers them.
///
/// The tuple-struct half is what makes a newtype like
/// [`RenderScale`](crate::RenderScale) annotatable: its one field is `.0`, and
/// without this it would fall back to [`DEFAULT_RANGE`] and give a panel a
/// −100…100 slider on a value the renderer clamps to 0.1…1.
fn declared_range(info: &TypeInfo, field: &str) -> Option<FieldRange> {
    match info {
        TypeInfo::Struct(_) => FieldRange::lookup(info, field),
        TypeInfo::TupleStruct(s) => {
            let index: usize = field.parse().ok()?;
            s.field_at(index)?.get_attribute::<FieldRange>().copied()
        }
        _ => None,
    }
}

/// Depth-first over `value`, appending one [`TweakField`] per supported leaf.
///
/// `path` is used as a scratch buffer — pushed and truncated rather than
/// reallocated per node — because this runs once per open panel frame over every
/// registered root.
fn walk(
    value: &dyn PartialReflect,
    path: &mut String,
    root: usize,
    depth: usize,
    declared: Option<FieldRange>,
    out: &mut Vec<TweakField>,
) {
    if depth > MAX_DEPTH {
        log::warn!("tweak: {path} is nested deeper than {MAX_DEPTH} — not descending");
        return;
    }
    let info = value.get_represented_type_info();
    match value.reflect_ref() {
        ReflectRef::Struct(s) => {
            for i in 0..s.field_len() {
                let (Some(name), Some(child)) = (s.name_at(i), s.field_at(i)) else {
                    continue;
                };
                let child_declared = info
                    .and_then(|info| declared_range(info, name))
                    .or(declared);
                let mark = path.len();
                path.push('.');
                path.push_str(name);
                walk(child, path, root, depth + 1, child_declared, out);
                path.truncate(mark);
            }
        }
        ReflectRef::TupleStruct(s) => {
            for i in 0..s.field_len() {
                let Some(child) = s.field(i) else { continue };
                let name = i.to_string();
                let child_declared = info
                    .and_then(|info| declared_range(info, &name))
                    .or(declared);
                let mark = path.len();
                path.push('.');
                path.push_str(&name);
                walk(child, path, root, depth + 1, child_declared, out);
                path.truncate(mark);
            }
        }
        // A unit-only enum is a choice; anything else is out (module docs).
        ReflectRef::Enum(e) => {
            let Some(TypeInfo::Enum(enum_info)) = info else {
                return;
            };
            let choices: Vec<&'static str> = enum_info.iter().map(|v| v.name()).collect();
            let all_unit = enum_info
                .iter()
                .all(|v| matches!(v, bevy_reflect::enums::VariantInfo::Unit(_)));
            if !all_unit {
                return;
            }
            push_leaf(
                out,
                path,
                root,
                TweakValue::Choice(e.variant_name().to_string()),
                declared.unwrap_or(FieldRange {
                    min: 0.0,
                    max: choices.len().saturating_sub(1) as f32,
                    step: 1.0,
                }),
                choices,
            );
        }
        _ => {
            let Some((value, fallback)) = leaf_of(value) else {
                return;
            };
            push_leaf(
                out,
                path,
                root,
                value,
                declared.unwrap_or(fallback),
                Vec::new(),
            );
        }
    }
}

fn push_leaf(
    out: &mut Vec<TweakField>,
    path: &str,
    root: usize,
    value: TweakValue,
    range: FieldRange,
    choices: Vec<&'static str>,
) {
    // The label is everything after the root name — `sky_color.x`, not
    // `sky.sky_color.x` — because the panel prints the root as a group header
    // and repeating it on every row is half the width of a phone screen.
    let label = path
        .split_once('.')
        .map(|(_, rest)| rest.to_string())
        .unwrap_or_else(|| path.to_string());
    out.push(TweakField {
        path: path.to_string(),
        root,
        label,
        value,
        range,
        choices,
        overridden: false,
    });
}

/// A leaf's value and the range it gets when the param declares none.
fn leaf_of(value: &dyn PartialReflect) -> Option<(TweakValue, FieldRange)> {
    if let Some(v) = value.try_downcast_ref::<f32>() {
        return Some((TweakValue::Float(*v), DEFAULT_RANGE));
    }
    if let Some(v) = value.try_downcast_ref::<bool>() {
        return Some((
            TweakValue::Bool(*v),
            FieldRange::new(0.0, 1.0).with_step(1.0),
        ));
    }
    if let Some(v) = value.try_downcast_ref::<u8>() {
        return Some((TweakValue::Int(*v as i64), DEFAULT_INT_RANGE));
    }
    if let Some(v) = value.try_downcast_ref::<u32>() {
        return Some((TweakValue::Int(*v as i64), DEFAULT_INT_RANGE));
    }
    if let Some(v) = value.try_downcast_ref::<u64>() {
        return Some((TweakValue::Int(*v as i64), DEFAULT_INT_RANGE));
    }
    if let Some(v) = value.try_downcast_ref::<usize>() {
        return Some((TweakValue::Int(*v as i64), DEFAULT_INT_RANGE));
    }
    if let Some(v) = value.try_downcast_ref::<i32>() {
        return Some((TweakValue::Int(*v as i64), DEFAULT_SIGNED_RANGE));
    }
    if let Some(v) = value.try_downcast_ref::<i64>() {
        return Some((TweakValue::Int(*v), DEFAULT_SIGNED_RANGE));
    }
    None
}

/// The value at a leaf that has already been resolved.
fn leaf_value(value: &dyn PartialReflect) -> Option<TweakValue> {
    if let ReflectRef::Enum(e) = value.reflect_ref() {
        return Some(TweakValue::Choice(e.variant_name().to_string()));
    }
    leaf_of(value).map(|(value, _)| value)
}

/// Walk `rest` — a dotted tail, possibly empty — down from `value`.
fn resolve<'a>(
    value: &'a dyn PartialReflect,
    rest: &str,
) -> Option<(&'a dyn PartialReflect, FieldRange)> {
    let mut current = value;
    let mut declared = None;
    for segment in rest.split('.').filter(|s| !s.is_empty()) {
        let info = current.get_represented_type_info();
        declared = info
            .and_then(|info| declared_range(info, segment))
            .or(declared);
        current = match current.reflect_ref() {
            ReflectRef::Struct(s) => s.field(segment)?,
            ReflectRef::TupleStruct(s) => s.field(segment.parse().ok()?)?,
            _ => return None,
        };
    }
    let fallback = leaf_of(current)
        .map(|(_, range)| range)
        .unwrap_or(DEFAULT_RANGE);
    Some((current, declared.unwrap_or(fallback)))
}

/// [`resolve`], mutably. The same loop; `get_represented_type_info` returns a
/// `&'static TypeInfo` and so borrows nothing, which is what lets the range be
/// read off the parent on the way down.
fn resolve_mut<'a>(
    value: &'a mut dyn PartialReflect,
    rest: &str,
) -> Option<(&'a mut dyn PartialReflect, FieldRange)> {
    let mut current = value;
    let mut declared = None;
    for segment in rest.split('.').filter(|s| !s.is_empty()) {
        let info = current.get_represented_type_info();
        declared = info
            .and_then(|info| declared_range(info, segment))
            .or(declared);
        current = match current.reflect_mut() {
            ReflectMut::Struct(s) => s.field_mut(segment)?,
            ReflectMut::TupleStruct(s) => s.field_mut(segment.parse().ok()?)?,
            _ => return None,
        };
    }
    let fallback = leaf_of(current)
        .map(|(_, range)| range)
        .unwrap_or(DEFAULT_RANGE);
    Some((current, declared.unwrap_or(fallback)))
}

/// Write `value` into an already-resolved leaf, clamped.
fn write_leaf(
    leaf: &mut dyn PartialReflect,
    value: TweakValue,
    range: FieldRange,
) -> Result<TweakValue, TweakError> {
    match value {
        TweakValue::Float(v) => {
            let v = if v.is_finite() {
                range.clamp(v)
            } else {
                range.min
            };
            let slot = leaf
                .try_downcast_mut::<f32>()
                .ok_or(TweakError::WrongKind)?;
            *slot = v;
            Ok(TweakValue::Float(v))
        }
        TweakValue::Bool(v) => {
            let slot = leaf
                .try_downcast_mut::<bool>()
                .ok_or(TweakError::WrongKind)?;
            *slot = v;
            Ok(TweakValue::Bool(v))
        }
        TweakValue::Int(v) => {
            // Clamp in f64 against the declared range, then again into the
            // field's own width — a `u8` asked for 300 must land at 255, not
            // wrap to 44.
            let clamped = (v as f64).clamp(range.min as f64, range.max as f64) as i64;
            write_int(leaf, clamped).ok_or(TweakError::WrongKind)
        }
        TweakValue::Choice(name) => {
            let ReflectMut::Enum(_) = leaf.reflect_mut() else {
                return Err(TweakError::WrongKind);
            };
            let Some(TypeInfo::Enum(info)) = leaf.get_represented_type_info() else {
                return Err(TweakError::WrongKind);
            };
            let variant = info
                .iter()
                .find(|v| v.name() == name)
                .ok_or(TweakError::WrongKind)?;
            if !matches!(variant, bevy_reflect::enums::VariantInfo::Unit(_)) {
                return Err(TweakError::WrongKind);
            }
            let dynamic = bevy_reflect::enums::DynamicEnum::new(
                variant.name(),
                bevy_reflect::enums::DynamicVariant::Unit,
            );
            leaf.apply(&dynamic);
            Ok(TweakValue::Choice(name))
        }
    }
}

/// Store `v` in whichever integer width the leaf actually is, saturating.
fn write_int(leaf: &mut dyn PartialReflect, v: i64) -> Option<TweakValue> {
    macro_rules! try_width {
        ($($ty:ty),*) => {$(
            if let Some(slot) = leaf.try_downcast_mut::<$ty>() {
                let stored = v.clamp(<$ty>::MIN as i64, <$ty>::MAX as i64) as $ty;
                *slot = stored;
                return Some(TweakValue::Int(stored as i64));
            }
        )*};
    }
    try_width!(u8, u32, i32);
    if let Some(slot) = leaf.try_downcast_mut::<u64>() {
        let stored = v.max(0) as u64;
        *slot = stored;
        return Some(TweakValue::Int(stored as i64));
    }
    if let Some(slot) = leaf.try_downcast_mut::<usize>() {
        let stored = v.max(0) as usize;
        *slot = stored;
        return Some(TweakValue::Int(stored as i64));
    }
    if let Some(slot) = leaf.try_downcast_mut::<i64>() {
        *slot = v;
        return Some(TweakValue::Int(v));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Reflect, Clone, Copy, Debug, PartialEq, Default)]
    enum Mode {
        #[default]
        Off,
        Slow,
        Fast,
    }

    #[derive(Reflect, Clone, Copy, Debug, PartialEq)]
    struct Nested {
        #[reflect(@FieldRange::new(0.0, 1.0))]
        gain: f32,
        count: u32,
    }

    #[derive(Resource, Reflect, Clone, Copy, Debug, PartialEq)]
    struct Rig {
        #[reflect(@FieldRange::new(0.0, 20.0).with_step(0.5))]
        follow_stiffness: f32,
        enabled: bool,
        mode: Mode,
        // The composite whose range its leaves inherit.
        #[reflect(@FieldRange::new(-2.0, 2.0))]
        nested: Nested,
        #[reflect(remote = crate::reflect::Vec3Def, @FieldRange::new(0.0, 1.0))]
        color: glam::Vec3,
    }

    impl Default for Rig {
        fn default() -> Rig {
            Rig {
                follow_stiffness: 10.0,
                enabled: true,
                mode: Mode::Slow,
                nested: Nested {
                    gain: 0.5,
                    count: 4,
                },
                color: glam::Vec3::new(0.2, 0.4, 0.6),
            }
        }
    }

    #[derive(Component, Reflect, Clone, Copy, Debug, PartialEq, Default)]
    struct Knob {
        amount: f32,
    }

    fn rig_world() -> World {
        let mut world = World::new();
        world.insert_resource(Rig::default());
        install(&mut world);
        register_resource::<Rig>(&mut world, "rig");
        world
    }

    fn paths(world: &World) -> Vec<String> {
        fields_of(world).into_iter().map(|f| f.path).collect()
    }

    #[test]
    fn registering_a_root_enumerates_its_leaves_as_dotted_paths() {
        let world = rig_world();
        assert_eq!(
            paths(&world),
            vec![
                "rig.follow_stiffness",
                "rig.enabled",
                "rig.mode",
                "rig.nested.gain",
                "rig.nested.count",
                // A remote-defined `Vec3` needs no special case: it is a struct
                // of three `f32`s and the walk flattens it like any other.
                "rig.color.x",
                "rig.color.y",
                "rig.color.z",
            ]
        );
    }

    #[test]
    fn a_range_is_read_off_the_param_and_inherited_by_nested_leaves() {
        let world = rig_world();
        let fields = fields_of(&world);
        let by_path = |path: &str| {
            fields
                .iter()
                .find(|f| f.path == path)
                .unwrap_or_else(|| panic!("no {path}"))
                .clone()
        };

        let stiffness = by_path("rig.follow_stiffness");
        assert_eq!(stiffness.range, FieldRange::new(0.0, 20.0).with_step(0.5));
        assert_eq!(
            stiffness.step(),
            0.5,
            "a declared step is the arrow's value"
        );

        // Declared on the leaf, so the composite's -2..2 does not reach it.
        assert_eq!(by_path("rig.nested.gain").range, FieldRange::new(0.0, 1.0));
        // Not declared on the leaf, so the composite's does.
        assert_eq!(
            by_path("rig.nested.count").range,
            FieldRange::new(-2.0, 2.0)
        );
        // …and the colour's three channels all inherit the one attribute.
        for axis in ["x", "y", "z"] {
            assert_eq!(
                by_path(&format!("rig.color.{axis}")).range,
                FieldRange::new(0.0, 1.0),
                "{axis} did not inherit the colour's range"
            );
        }
    }

    #[test]
    fn an_edit_lands_in_the_live_world_and_is_clamped_at_the_range() {
        let mut world = rig_world();
        let landed = set_and_record(&mut world, "rig.follow_stiffness", TweakValue::Float(4.5))
            .expect("set");
        assert_eq!(landed, TweakValue::Float(4.5));
        assert_eq!(world.resource::<Rig>().follow_stiffness, 4.5);
        // …and reading one leaf back by path agrees with the world.
        let tweaks = world.resource::<Tweakables>();
        assert_eq!(
            tweaks.get(&world, "rig.follow_stiffness"),
            Some(TweakValue::Float(4.5))
        );
        assert_eq!(tweaks.get(&world, "rig.nope"), None);

        // Past the declared end, on both sides.
        let high = set_and_record(&mut world, "rig.follow_stiffness", TweakValue::Float(1e9));
        assert_eq!(high, Ok(TweakValue::Float(20.0)));
        assert_eq!(world.resource::<Rig>().follow_stiffness, 20.0);
        let low = set_and_record(&mut world, "rig.follow_stiffness", TweakValue::Float(-50.0));
        assert_eq!(low, Ok(TweakValue::Float(0.0)));

        // …and what was recorded is what landed, not what was asked for.
        let overrides = world.resource::<TweakOverrides>();
        assert_eq!(
            overrides.values.get("rig.follow_stiffness"),
            Some(&TweakValue::Float(0.0))
        );
    }

    #[test]
    fn bools_ints_and_unit_enums_all_edit() {
        let mut world = rig_world();
        set_and_record(&mut world, "rig.enabled", TweakValue::Bool(false)).expect("bool");
        assert!(!world.resource::<Rig>().enabled);

        set_and_record(&mut world, "rig.nested.count", TweakValue::Int(2)).expect("int");
        assert_eq!(world.resource::<Rig>().nested.count, 2);
        // The composite's range clamps the integer too — and a negative into a
        // `u32` saturates at the width rather than wrapping.
        set_and_record(&mut world, "rig.nested.count", TweakValue::Int(-9)).expect("int");
        assert_eq!(world.resource::<Rig>().nested.count, 0);

        set_and_record(&mut world, "rig.mode", TweakValue::Choice("Fast".into())).expect("enum");
        assert_eq!(world.resource::<Rig>().mode, Mode::Fast);
        assert_eq!(
            set_and_record(
                &mut world,
                "rig.mode",
                TweakValue::Choice("Sideways".into())
            ),
            Err(TweakError::WrongKind),
            "a variant the enum does not have is refused, not guessed"
        );
    }

    #[test]
    fn the_wrong_kind_and_the_wrong_path_are_errors_rather_than_writes() {
        let mut world = rig_world();
        assert_eq!(
            set_and_record(&mut world, "rig.enabled", TweakValue::Float(1.0)),
            Err(TweakError::WrongKind)
        );
        assert_eq!(
            set_and_record(&mut world, "rig.nope", TweakValue::Float(1.0)),
            Err(TweakError::NoSuchPath)
        );
        assert_eq!(
            set_and_record(&mut world, "nope.field", TweakValue::Float(1.0)),
            Err(TweakError::NoSuchPath)
        );
        assert_eq!(*world.resource::<Rig>(), Rig::default(), "nothing moved");
    }

    #[test]
    fn overrides_round_trip_through_ron_and_apply_to_a_fresh_world() {
        let mut world = rig_world();
        set_and_record(&mut world, "rig.follow_stiffness", TweakValue::Float(3.25)).expect("set");
        set_and_record(&mut world, "rig.enabled", TweakValue::Bool(false)).expect("set");
        set_and_record(&mut world, "rig.mode", TweakValue::Choice("Fast".into())).expect("set");

        let text = world.resource::<TweakOverrides>().to_ron().expect("ron");
        let parsed = TweakOverrides::from_ron(&text).expect("parse");
        assert_eq!(&parsed, world.resource::<TweakOverrides>());

        // A fresh world, built from the authored defaults, then loaded.
        let mut fresh = rig_world();
        assert_eq!(fresh.resource::<Rig>().follow_stiffness, 10.0);
        assert_eq!(parsed.apply(&mut fresh), 3);
        let rig = *fresh.resource::<Rig>();
        assert_eq!(rig.follow_stiffness, 3.25);
        assert!(!rig.enabled);
        assert_eq!(rig.mode, Mode::Fast);
    }

    #[test]
    fn an_override_for_a_field_that_no_longer_exists_is_dropped_and_kept() {
        let mut world = rig_world();
        let mut overrides = TweakOverrides::new();
        overrides
            .values
            .insert("rig.follow_stiffness".into(), TweakValue::Float(7.0));
        overrides
            .values
            .insert("rig.renamed_away".into(), TweakValue::Float(7.0));
        overrides
            .values
            .insert("gone.entirely".into(), TweakValue::Bool(true));

        assert_eq!(overrides.apply(&mut world), 1, "only the live one lands");
        assert_eq!(world.resource::<Rig>().follow_stiffness, 7.0);
        // The map itself is untouched: switching branches must not delete an
        // hour of tuning for a field the other branch has not landed yet.
        assert_eq!(overrides.len(), 3);
    }

    #[test]
    fn clearing_restores_the_value_the_world_was_registered_with() {
        let mut world = rig_world();
        // …even after an override file has moved it, which is the case the
        // registration-time snapshot exists for: "authored" means what the game
        // ships, not what this session started at.
        let mut overrides = TweakOverrides::new();
        overrides
            .values
            .insert("rig.follow_stiffness".into(), TweakValue::Float(1.0));
        overrides.apply(&mut world);
        world.insert_resource(overrides);
        assert_eq!(world.resource::<Rig>().follow_stiffness, 1.0);

        set_and_record(&mut world, "rig.follow_stiffness", TweakValue::Float(9.0)).expect("set");
        assert_eq!(world.resource::<Rig>().follow_stiffness, 9.0);

        let restored = clear(&mut world, "rig.follow_stiffness").expect("clear");
        assert_eq!(restored, Some(TweakValue::Float(10.0)));
        assert_eq!(world.resource::<Rig>().follow_stiffness, 10.0);
        assert!(
            world.resource::<TweakOverrides>().is_empty(),
            "the override outlived the value it described"
        );
    }

    #[test]
    fn clear_all_puts_every_root_back() {
        let mut world = rig_world();
        set_and_record(&mut world, "rig.follow_stiffness", TweakValue::Float(1.0)).expect("set");
        set_and_record(&mut world, "rig.color.y", TweakValue::Float(1.0)).expect("set");
        assert_eq!(clear_all(&mut world), 2);
        assert_eq!(*world.resource::<Rig>(), Rig::default());
        assert!(world.resource::<TweakOverrides>().is_empty());
    }

    #[test]
    fn a_component_root_reaches_one_named_entity() {
        let mut world = World::new();
        install(&mut world);
        let a = world.spawn(Knob { amount: 1.0 }).id();
        let b = world.spawn(Knob { amount: 2.0 }).id();
        register_component::<Knob>(&mut world, "knob", a);

        assert_eq!(paths(&world), vec!["knob.amount"]);
        set_and_record(&mut world, "knob.amount", TweakValue::Float(5.0)).expect("set");
        assert_eq!(world.get::<Knob>(a).expect("a").amount, 5.0);
        assert_eq!(
            world.get::<Knob>(b).expect("b").amount,
            2.0,
            "a root names one entity, not every entity with the component"
        );

        // The entity going away is a panel with fewer rows, not a panic.
        world.despawn(a);
        assert!(paths(&world).is_empty());
        assert_eq!(
            set_and_record(&mut world, "knob.amount", TweakValue::Float(1.0)),
            Err(TweakError::RootAbsent)
        );
    }

    #[test]
    fn a_duplicate_root_name_is_refused() {
        let mut world = rig_world();
        register_resource::<Rig>(&mut world, "rig");
        assert_eq!(world.resource::<Tweakables>().roots().len(), 1);
    }

    #[test]
    fn a_world_without_install_simply_has_no_tunables() {
        let world = World::new();
        assert!(fields_of(&world).is_empty());
    }
}
