//! The selection inspector's widget model — `reflect` feature only (DESIGN
//! §10a).
//!
//! > *Panels are generated from `Reflect` param structs: a `Reflect`-walking
//! > widget mapper (f32 → slider with range attributes, enum → select, Vec3 →
//! > triple, seed → reroll button). Hand-written panels only where reflection
//! > genuinely can't express the interaction.* — DESIGN §10
//!
//! This module is that mapper, engine-side. It is deliberately **not** a UI: it
//! turns a `&dyn PartialReflect` into a [`Widget`] tree of pure data, and turns
//! an [`Edit`] back into a mutation of the original value.
//! [`inspect_panel`](crate::inspect_panel) draws the tree and produces the
//! edits; a different front end could do both differently and this module would
//! not know.
//!
//! # Why this exists next to `tweak`, and why they must stay two things
//!
//! [`crate::tweak`] refuses to descend into a data-carrying enum, and its module
//! docs say why: its dotted paths are the **keys of an override file**, and a
//! path that exists only while `color` is `Some` is not a stable key. That
//! invariant is right for what tweak is — a registry whose edits outlive the
//! session — and it makes tweak permanently blind to the single most edited
//! value in the engine, [`GeneratorSpec`](crate::gen::GeneratorSpec), which *is*
//! an enum with data.
//!
//! The inspector has no file. A [`FieldPath`] here identifies a widget inside
//! **this frame's tree of the selected value**, is rebuilt every frame the way
//! [`UiBatch`](crate::ui::UiBatch) is, and is never written anywhere. A path
//! that stops resolving because the user switched a variant is an edit dropped
//! on the floor, not an orphaned line in a settings file — so the variant cut
//! does not apply here, and switching a variant is not merely allowed, it is
//! the point of the tool.
//!
//! # The mapping
//!
//! | reflected shape | widget |
//! |---|---|
//! | `f32`, `f64` | [`Widget::Float`] |
//! | `u64` named `seed` | [`Widget::Seed`] — reroll, not a slider |
//! | other integers | [`Widget::Int`] |
//! | `bool` | [`Widget::Bool`] |
//! | enum, all variants unit | [`Widget::Choice`] — cycled by name, as tweak does |
//! | enum with data | [`Widget::Variant`] — a select, plus the active variant's fields |
//! | struct of 2–4 `f32` named `x`,`y`,`z`,`w` | [`Widget::Vector`] |
//! | `[f32; 2..=4]` | [`Widget::Vector`] — the fixed array a schema without glam writes |
//! | any other struct / tuple struct | [`Widget::Group`] |
//! | anything else | [`Widget::Unsupported`] — visible, not editable |
//!
//! `Vec2`/`Vec3` land in the vector row because of the remote definitions in
//! [`crate::reflect`]; the mapper never names a glam type. `String` lands in
//! `Unsupported` on purpose: the panel deliberately has no text entry (see
//! [`crate::tweak_panel`]'s doctrine on carets), and a control that cannot be
//! driven is better shown read-only than invented badly.
//!
//! [`Widget::Color`] is deliberately **not** in this table: the walk never
//! produces one. The one reflected field this codebase has that is a colour —
//! `color: Option<Vec3>` on [`GeneratorSpec`](crate::gen::GeneratorSpec)'s
//! variants — is optional, so it is a [`Widget::Variant`] of `None`/`Some`
//! before it is ever a vector at all; teaching the walk to reach past that
//! wrapper and *also* recognise the field by name would be a second
//! name-based rule stacked on the one `seed` already is, for a shape that
//! only ever shows up wrapped in `Option` and never bare. The two fields a
//! game actually wants a swatch for — a gradient's `Vec<(f32, Vec3)>` ramp and
//! a material's `Vec4` tint — are both `#[reflect(ignore)]` and unreachable
//! from any walk regardless (see [`Widget::Color`]'s own doc comment). So
//! `Color` stays what the module doc above already says every hand-authored
//! game panel needs sometimes: a widget a game builds directly, not one the
//! reflected walk infers.
//!
//! # Ranges are advisory here — the opposite of `tweak`
//!
//! Bounds come off `#[reflect(@FieldRange…)]` exactly as tweak reads them, with
//! the same inheritance: a range declared on a *composite* field bounds every
//! leaf under it that declares none. But where [`Tweakables::set`] **clamps** —
//! it writes into a live world, and a camera stiffness of −4000 is a camera
//! that never comes back — [`apply`] does **not**. A `FieldRange` on a
//! generator param is "where the interesting part of the space is"
//! ([`crate::gen`]'s doctrine: advisory, never validated), a scene file can say
//! anything, and an inspector that silently rounds a typed-in 300 down to 256
//! would make the panel disagree with the file it is editing. The slider ends
//! sit at the range; the value goes where it is sent.
//!
//! [`Tweakables::set`]: crate::tweak::Tweakables::set

use bevy_reflect::enums::{DynamicEnum, DynamicVariant, VariantInfo};
use bevy_reflect::{PartialReflect, ReflectMut, ReflectRef, TypeInfo};

use crate::reflect::{FieldRange, DEFAULT_INT_RANGE, DEFAULT_RANGE};
use crate::tweak::{declared_range, DEFAULT_SIGNED_RANGE, MAX_DEPTH};

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

/// One step from a reflected value to one of its children.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Step {
    /// A named field of a struct or of an enum's *active* struct variant.
    Field(String),
    /// A positional field of a tuple struct or an enum's active tuple variant.
    Index(usize),
}

/// A path from the root of the inspected value to one field, root-first.
///
/// Not a dotted string, unlike tweak's paths, and the difference is the whole
/// point of the previous section: a tweak path is a *stable identity* that
/// files key on, while this is a *frame-local address* that gets re-walked
/// against whatever the value is now. An explicit step list makes a mismatch —
/// the variant changed under a mid-drag slider — an ordinary `Err` the panel
/// drops, rather than a string parse anybody has to think about.
///
/// An empty path addresses the root itself.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FieldPath(pub Vec<Step>);

impl FieldPath {
    pub fn root() -> FieldPath {
        FieldPath(Vec::new())
    }

    pub fn is_root(&self) -> bool {
        self.0.is_empty()
    }

    /// This path with one more named step on the end. The walk is depth-first,
    /// so paths grow by copy rather than mutation.
    pub fn field(&self, name: impl Into<String>) -> FieldPath {
        let mut next = self.0.clone();
        next.push(Step::Field(name.into()));
        FieldPath(next)
    }

    pub fn index(&self, i: usize) -> FieldPath {
        let mut next = self.0.clone();
        next.push(Step::Index(i));
        FieldPath(next)
    }

    /// A display form: `.size.x`, `#0.amplitude`. Used in error messages,
    /// never parsed back.
    pub fn display(&self) -> String {
        let mut out = String::new();
        for step in &self.0 {
            match step {
                Step::Field(name) => {
                    out.push('.');
                    out.push_str(name);
                }
                Step::Index(i) => {
                    out.push('#');
                    out.push_str(&i.to_string());
                }
            }
        }
        out
    }
}

/// Follow `path` from `root`, mutably. The enum arms resolve only the *active*
/// variant's fields, which is exactly the drop-the-edit behaviour a panel
/// wants when a variant changes mid-drag.
fn resolve_mut<'a>(
    root: &'a mut dyn PartialReflect,
    path: &FieldPath,
) -> Result<&'a mut dyn PartialReflect, String> {
    let mut cursor = root;
    for (depth, step) in path.0.iter().enumerate() {
        // The message wants `step` after the borrow of `cursor` has moved on.
        let message = format!("{}: nothing at depth {depth}", path.display());
        cursor = match (cursor.reflect_mut(), step) {
            (ReflectMut::Struct(s), Step::Field(name)) => s.field_mut(name),
            (ReflectMut::TupleStruct(t), Step::Index(i)) => t.field_mut(*i),
            (ReflectMut::Enum(e), Step::Field(name)) => e.field_mut(name),
            (ReflectMut::Enum(e), Step::Index(i)) => e.field_at_mut(*i),
            // A fixed array's element — the other half of the vector row an
            // `[f32; 3]` becomes in the walk below.
            (ReflectMut::Array(a), Step::Index(i)) => a.get_mut(*i),
            _ => None,
        }
        .ok_or(message)?;
    }
    Ok(cursor)
}

// ---------------------------------------------------------------------------
// Widgets
// ---------------------------------------------------------------------------

/// One control in a generated panel. Pure data: a label, a path, a value, and
/// whatever the control's kind needs to draw itself.
#[derive(Clone, Debug, PartialEq)]
pub enum Widget {
    Float {
        label: String,
        path: FieldPath,
        value: f64,
        range: FieldRange,
    },
    Int {
        label: String,
        path: FieldPath,
        value: i64,
        range: FieldRange,
    },
    Bool {
        label: String,
        path: FieldPath,
        value: bool,
    },
    /// A `u64` field named `seed`: too wide for a slider and meaningless as a
    /// magnitude, so its one verb is [reroll]. The one name-based rule in the
    /// mapper; `seed` is the name every runt generator uses.
    Seed {
        label: String,
        path: FieldPath,
        value: u64,
    },
    /// A unit-only enum — the same shape tweak calls a choice, cycled by name.
    Choice {
        label: String,
        path: FieldPath,
        selected: String,
        options: Vec<String>,
    },
    /// An enum with data: a variant select plus whatever the active variant
    /// contains. The widget tweak cannot have, and the reason this module
    /// exists.
    Variant {
        label: String,
        path: FieldPath,
        selected: String,
        options: Vec<String>,
        fields: Vec<Widget>,
    },
    /// A small struct of `f32`s named `x`/`y`/`z`/`w` — one labelled group of
    /// numeric components rather than an anonymous nested struct.
    Vector {
        label: String,
        path: FieldPath,
        components: Vec<VectorComponent>,
        range: FieldRange,
    },
    /// A colour: like [`Vector`](Widget::Vector), a header row that flattens
    /// into one editable [`Float`](Widget::Float) per channel via
    /// [`push_rows`](Widget::push_rows) — but the header also draws a filled
    /// swatch ([`inspect_panel`](crate::inspect_panel)'s `draw`), because a
    /// colour is the one small float aggregate a person actually wants to
    /// *see* rather than read three numbers for.
    ///
    /// # rgb, not rgba — and always hand-built
    ///
    /// Two real values could have driven this: the procedural texture's
    /// gradient stops (`TextureSpec::ramp: Vec<(f32, Vec3)>`, rgb) and a
    /// material's tint (`MaterialDesc::base_color: Vec4`, rgba). Neither is
    /// reachable from [`build_at`]'s reflected walk, and for two different
    /// reasons that both predate this widget: the ramp's `Vec3` is buried
    /// inside a `Vec` of tuples, which `#[reflect(remote = …)]` cannot reach
    /// through (`ramp`'s own doc comment, `crate::texture::TextureSpec`);
    /// `Vec4` has no remote definition to reach through at all, because on
    /// this glam it is one opaque SIMD register with no `x`/`y`/`z`/`w`
    /// fields to delegate to (`crate::reflect`'s "glam remote definitions"
    /// section explains why `Vec4` alone has none). Both fields carry
    /// `#[reflect(ignore)]` for exactly that reason. So every `Color` in this
    /// engine is, and will stay, something a game builds by hand from a value
    /// it already has — the way the ramp's rows are hand-built today
    /// (`ramp_rows` in the game's `materials.rs`), now with a swatch because
    /// there is finally a widget for one.
    ///
    /// Because it is always hand-built there is no single call site that
    /// must commit to "three channels" or "four": `components` holds however
    /// many the caller has, in `r, g, b[, a]` order, exactly as `Vector`'s
    /// `components` holds however many axes it has. The swatch itself is
    /// always rgb — alpha has no pixel to blend against in a debug-overlay
    /// quad, and drawing it as translucency would read as "this swatch is
    /// hard to see," which is the one thing a colour preview must never be.
    /// An `a` channel, when present, is still an ordinary [`Float`](Widget::Float)
    /// row underneath, edited the same way `r`, `g` and `b` are.
    Color {
        label: String,
        path: FieldPath,
        /// The swatch colour, rgb. `f32`, unlike every other widget value's
        /// `f64`: its only reader is [`inspect_panel`](crate::inspect_panel)'s
        /// draw, which writes it straight into a
        /// [`UiQuad`](crate::ui::UiQuad)'s `f32` color, and carrying `f64`
        /// this far would be a cast nobody needs. **Not clamped** — see
        /// `range`, below; the draw clamps for display, once, at paint time.
        value: [f32; 3],
        /// `r`, `g`, `b`, and `a` if this colour has one — the channels
        /// [`push_rows`](Widget::push_rows) turns into `Float` rows, the way
        /// `Vector`'s `components` do.
        components: Vec<VectorComponent>,
        /// **Advisory**, the same as every other range in this module — see
        /// the module docs' "Ranges are advisory" section. Channel values are
        /// not clamped to it: a channel past 1.0 is how this renderer, which
        /// has no emission channel, spells "blown out."
        range: FieldRange,
    },
    /// A struct that is not a vector: a heading and its fields.
    Group {
        label: String,
        path: FieldPath,
        fields: Vec<Widget>,
    },
    /// Something the mapper has no control for. Kept in the tree so a new param
    /// type shows up as "not editable yet" rather than vanishing.
    Unsupported {
        label: String,
        path: FieldPath,
        type_name: String,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct VectorComponent {
    pub label: String,
    pub path: FieldPath,
    pub value: f64,
}

impl Widget {
    pub fn label(&self) -> &str {
        match self {
            Widget::Float { label, .. }
            | Widget::Int { label, .. }
            | Widget::Bool { label, .. }
            | Widget::Seed { label, .. }
            | Widget::Choice { label, .. }
            | Widget::Variant { label, .. }
            | Widget::Vector { label, .. }
            | Widget::Color { label, .. }
            | Widget::Group { label, .. }
            | Widget::Unsupported { label, .. } => label,
        }
    }

    pub fn path(&self) -> &FieldPath {
        match self {
            Widget::Float { path, .. }
            | Widget::Int { path, .. }
            | Widget::Bool { path, .. }
            | Widget::Seed { path, .. }
            | Widget::Choice { path, .. }
            | Widget::Variant { path, .. }
            | Widget::Vector { path, .. }
            | Widget::Color { path, .. }
            | Widget::Group { path, .. }
            | Widget::Unsupported { path, .. } => path,
        }
    }

    /// The tree flattened into panel rows, depth-first, each with its nesting
    /// depth — what [`inspect_panel`](crate::inspect_panel) draws and walks a
    /// cursor over.
    ///
    /// A [`Vector`](Widget::Vector) contributes its own header row and then one
    /// [`Float`](Widget::Float) row per component, carrying the vector's range —
    /// so the panel has exactly one idea of "edit a number" and a colour's
    /// channels are edited the way tweak already edits them, not through a
    /// second, vector-shaped code path. [`Color`](Widget::Color) shares this
    /// arm outright rather than duplicating it: a colour *is* a vector that
    /// also draws a swatch, and giving it its own flattening would be the
    /// second code path the sentence above already argues against.
    pub fn rows(&self) -> Vec<(usize, Widget)> {
        let mut out = Vec::new();
        self.push_rows(0, &mut out);
        out
    }

    fn push_rows(&self, depth: usize, out: &mut Vec<(usize, Widget)>) {
        out.push((depth, self.clone()));
        match self {
            Widget::Variant { fields, .. } | Widget::Group { fields, .. } => {
                for f in fields {
                    f.push_rows(depth + 1, out);
                }
            }
            Widget::Vector { components, range, .. } | Widget::Color { components, range, .. } => {
                for c in components {
                    out.push((
                        depth + 1,
                        Widget::Float {
                            label: c.label.clone(),
                            path: c.path.clone(),
                            value: c.value,
                            range: *range,
                        },
                    ));
                }
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// The walk
// ---------------------------------------------------------------------------

/// Build the widget tree for a reflected value.
///
/// `label` names the root — the generator's kind, usually. Rebuild it every
/// frame the panel is open, the way [`tweak::fields_of`](crate::tweak::fields_of)
/// is called: a tree derived from the value cannot go stale, and a selection
/// panel allocating a few dozen `String`s while it is up is not a cost anybody
/// can measure.
pub fn build(value: &dyn PartialReflect, label: impl Into<String>) -> Widget {
    build_at(value, label.into(), FieldPath::root(), None, 0)
}

fn build_at(
    value: &dyn PartialReflect,
    label: String,
    path: FieldPath,
    inherited: Option<FieldRange>,
    depth: usize,
) -> Widget {
    if depth > MAX_DEPTH {
        // The same guard tweak's walk carries, for the same reason: a type
        // nobody meant to inspect must be a visible stub, not a stack overflow.
        return Widget::Unsupported {
            label,
            path,
            type_name: format!("nested deeper than {MAX_DEPTH}"),
        };
    }

    // Leaves first: a scalar is never also a struct, and deciding here keeps
    // the opaque arm below from re-testing everything.
    if let Some(widget) = leaf(value, &label, &path, inherited) {
        return widget;
    }

    let info = value.get_represented_type_info();
    match value.reflect_ref() {
        ReflectRef::Struct(s) => {
            if let Some(components) = vector_components(value, &path) {
                return Widget::Vector {
                    label,
                    path,
                    components,
                    range: inherited.unwrap_or(DEFAULT_RANGE),
                };
            }
            let fields = (0..s.field_len())
                .filter_map(|i| {
                    let name = s.name_at(i)?;
                    let child = s.field_at(i)?;
                    let range = info.and_then(|info| declared_range(info, name)).or(inherited);
                    Some(build_at(child, pretty(name), path.field(name), range, depth + 1))
                })
                .collect();
            Widget::Group { label, path, fields }
        }

        ReflectRef::TupleStruct(t) => {
            let fields = (0..t.field_len())
                .filter_map(|i| {
                    let child = t.field(i)?;
                    let range = info
                        .and_then(|info| declared_range(info, &i.to_string()))
                        .or(inherited);
                    Some(build_at(child, format!("#{i}"), path.index(i), range, depth + 1))
                })
                .collect();
            Widget::Group { label, path, fields }
        }

        ReflectRef::Enum(e) => {
            let Some(TypeInfo::Enum(enum_info)) = info else {
                return Widget::Unsupported {
                    label,
                    path,
                    type_name: value.reflect_type_path().to_string(),
                };
            };
            let options: Vec<String> = enum_info.iter().map(|v| v.name().to_string()).collect();
            let selected = e.variant_name().to_string();

            // The unit-only enum is tweak's choice, and it stays one here so
            // the two panels agree about what `Mode` is.
            if enum_info.iter().all(|v| matches!(v, VariantInfo::Unit(_))) {
                return Widget::Choice { label, path, selected, options };
            }

            let variant = enum_info.variant(e.variant_name());
            let fields = (0..e.field_len())
                .filter_map(|i| {
                    let child = e.field_at(i)?;
                    match e.name_at(i) {
                        // A struct variant: named fields, each with its own range.
                        Some(name) => Some(build_at(
                            child,
                            pretty(name),
                            path.field(name),
                            variant
                                .and_then(|v| variant_field_range(v, Some(name), i))
                                .or(inherited),
                            depth + 1,
                        )),
                        // A tuple variant: positional. A single-field tuple
                        // variant (`Smooth(f32)`, `Terrain(params)`) drops the
                        // `#0` — "Smooth → #0" reads worse than "Smooth" with a
                        // value next to it.
                        None => Some(build_at(
                            child,
                            if e.field_len() == 1 {
                                String::new()
                            } else {
                                format!("#{i}")
                            },
                            path.index(i),
                            variant
                                .and_then(|v| variant_field_range(v, None, i))
                                .or(inherited),
                            depth + 1,
                        )),
                    }
                })
                .collect();
            Widget::Variant { label, path, selected, options, fields }
        }

        // A fixed float array of vector arity is a vector row: `[f32; 3]` is
        // what a schema that keeps serde and drops glam writes for a size or a
        // colour, and it deserves the same one labelled group of components a
        // `Vec3` gets. Any other array — too long for axes, or not floats —
        // stays a visible stub.
        ReflectRef::Array(_) => match array_components(value, &path) {
            Some(components) => Widget::Vector {
                label,
                path,
                components,
                range: inherited.unwrap_or(DEFAULT_RANGE),
            },
            None => Widget::Unsupported {
                label,
                path,
                type_name: value.reflect_type_path().to_string(),
            },
        },

        _ => Widget::Unsupported {
            label,
            path,
            type_name: value.reflect_type_path().to_string(),
        },
    }
}

/// A scalar leaf, if this value is one.
///
/// Integer defaults follow tweak, not the frozen editor: an unannotated
/// *signed* integer gets [`DEFAULT_SIGNED_RANGE`], because a slider that cannot
/// reach −1 is worse than a wide one.
fn leaf(
    value: &dyn PartialReflect,
    label: &str,
    path: &FieldPath,
    range: Option<FieldRange>,
) -> Option<Widget> {
    if let Some(v) = value.try_downcast_ref::<f32>() {
        return Some(Widget::Float {
            label: label.to_string(),
            path: path.clone(),
            value: *v as f64,
            range: range.unwrap_or(DEFAULT_RANGE),
        });
    }
    if let Some(v) = value.try_downcast_ref::<f64>() {
        return Some(Widget::Float {
            label: label.to_string(),
            path: path.clone(),
            value: *v,
            range: range.unwrap_or(DEFAULT_RANGE),
        });
    }
    if let Some(v) = value.try_downcast_ref::<bool>() {
        return Some(Widget::Bool {
            label: label.to_string(),
            path: path.clone(),
            value: *v,
        });
    }
    if let Some(v) = value.try_downcast_ref::<u64>() {
        let is_seed = matches!(path.0.last(), Some(Step::Field(name)) if name == "seed");
        return Some(if is_seed {
            Widget::Seed {
                label: label.to_string(),
                path: path.clone(),
                value: *v,
            }
        } else {
            Widget::Int {
                label: label.to_string(),
                path: path.clone(),
                value: *v as i64,
                range: range.unwrap_or(DEFAULT_INT_RANGE),
            }
        });
    }

    macro_rules! int_leaf {
        ($default:expr; $($ty:ty),*) => {$(
            if let Some(v) = value.try_downcast_ref::<$ty>() {
                return Some(Widget::Int {
                    label: label.to_string(),
                    path: path.clone(),
                    value: *v as i64,
                    range: range.unwrap_or($default),
                });
            }
        )*};
    }
    int_leaf!(DEFAULT_INT_RANGE; u8, u16, u32, usize);
    int_leaf!(DEFAULT_SIGNED_RANGE; i8, i16, i32, i64, isize);

    None
}

/// The `x`/`y`/`z`/`w` components of a small float struct, if that is what this
/// is. Anything else — a different arity, a non-`f32` field, a different name —
/// falls through to the ordinary group path.
fn vector_components(value: &dyn PartialReflect, path: &FieldPath) -> Option<Vec<VectorComponent>> {
    const AXES: [&str; 4] = ["x", "y", "z", "w"];
    let ReflectRef::Struct(s) = value.reflect_ref() else {
        return None;
    };
    if !(2..=4).contains(&s.field_len()) {
        return None;
    }
    let mut components = Vec::with_capacity(s.field_len());
    for (i, axis) in AXES.iter().take(s.field_len()).enumerate() {
        let name = s.name_at(i)?;
        if name != *axis {
            return None;
        }
        let v = *s.field_at(i)?.try_downcast_ref::<f32>()?;
        components.push(VectorComponent {
            label: name.to_string(),
            path: path.field(name),
            value: v as f64,
        });
    }
    Some(components)
}

/// The components of a small `f32` array, if that is what this is — the
/// [`vector_components`] rule for the struct-free spelling. The axis names are
/// borrowed rather than `#0`..`#3`, because an `[f32; 3]` in a param struct
/// means a point or an extent, and `x`/`y`/`z` is what a human calls those.
fn array_components(value: &dyn PartialReflect, path: &FieldPath) -> Option<Vec<VectorComponent>> {
    const AXES: [&str; 4] = ["x", "y", "z", "w"];
    let ReflectRef::Array(a) = value.reflect_ref() else {
        return None;
    };
    if !(2..=4).contains(&a.len()) {
        return None;
    }
    let mut components = Vec::with_capacity(a.len());
    for (i, axis) in AXES.iter().take(a.len()).enumerate() {
        let v = *a.get(i)?.try_downcast_ref::<f32>()?;
        components.push(VectorComponent {
            label: (*axis).to_string(),
            path: path.index(i),
            value: v as f64,
        });
    }
    Some(components)
}

/// The range declared on one field of an enum *variant* — the half of the
/// lookup [`declared_range`] cannot do, because `bevy_reflect` models a
/// variant's fields with different types than a struct's.
fn variant_field_range(
    variant: &VariantInfo,
    name: Option<&str>,
    index: usize,
) -> Option<FieldRange> {
    match (variant, name) {
        (VariantInfo::Struct(s), Some(name)) => {
            s.field(name)?.get_attribute::<FieldRange>().copied()
        }
        (VariantInfo::Tuple(t), _) => t.field_at(index)?.get_attribute::<FieldRange>().copied(),
        _ => None,
    }
}

/// `major_radius` → `major radius`. Cosmetic; the path keeps the real name.
fn pretty(name: &str) -> String {
    name.replace('_', " ")
}

// ---------------------------------------------------------------------------
// Applying edits
// ---------------------------------------------------------------------------

/// One change a widget wants to make.
#[derive(Clone, Debug, PartialEq)]
pub enum Edit {
    Float(f64),
    Int(i64),
    Bool(bool),
    Seed(u64),
    /// Switch an enum — a [`Choice`](Widget::Choice) or a
    /// [`Variant`](Widget::Variant) — to the named variant. Only the *name*
    /// travels; what the new variant's fields contain is [`apply`]'s problem.
    Variant(String),
}

/// The next value a reroll lands on: one step of splitmix64.
///
/// A pure function of the current seed rather than a random draw, because the
/// panel runs inside a fixed tick and a reroll that consulted a clock or an OS
/// RNG would be the one edit a [trace](crate::trace) could not replay. The
/// sequence walks the whole `u64` space and never repeats within it, which is
/// everything "give me a different mesh" needs.
pub fn reroll(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Apply `edit` at `path` inside `root`.
///
/// Writes through to the leaf: a `Float` edit becomes a real `f32` assignment
/// on the real struct field, which is why the caller can take the edited value
/// afterwards and hash it like any other. Numbers are **not clamped** to their
/// [`FieldRange`] — the module docs carry the argument; the short form is that
/// the range is advisory ([`crate::gen`]) and an inspector must not disagree
/// with the file it is editing.
///
/// Errors are strings and are meant to be *ignorable*: a path that no longer
/// resolves (the user switched a variant while a slider was mid-drag) is a
/// dropped edit, not a broken panel.
pub fn apply(root: &mut dyn PartialReflect, path: &FieldPath, edit: &Edit) -> Result<(), String> {
    if let Edit::Variant(name) = edit {
        return apply_variant(root, path, name);
    }

    let target = resolve_mut(root, path)?;
    let type_path = target.reflect_type_path().to_string();

    let ok = match edit {
        Edit::Float(v) => set_number(target, *v),
        Edit::Int(v) => set_number(target, *v as f64),
        Edit::Bool(v) => try_set(target, *v),
        Edit::Seed(v) => try_set(target, *v),
        Edit::Variant(_) => unreachable!("handled above"),
    };

    if ok {
        Ok(())
    } else {
        Err(format!(
            "{}: cannot apply {edit:?} to a {type_path}",
            path.display()
        ))
    }
}

/// Write a numeric edit into whatever width the field actually is.
///
/// Widgets are all `f64` — a slider has no idea whether it is driving a `u32`
/// segment count or an `f32` radius — so the narrowing happens here, once,
/// where the concrete type is known. The only clamp is the *type's own width*:
/// a 300 into a `u8` saturates at 255 rather than wrapping, which is a fact
/// about `u8`, not an opinion about the param.
fn set_number(target: &mut dyn PartialReflect, value: f64) -> bool {
    macro_rules! narrow {
        ($($ty:ty),*) => {$(
            if target.try_downcast_ref::<$ty>().is_some() {
                // `round` before the cast: a slider that reports 23.999999 for
                // a segment count must not silently mean 23.
                let clamped = value.round().clamp(<$ty>::MIN as f64, <$ty>::MAX as f64);
                return try_set(target, clamped as $ty);
            }
        )*};
    }

    if target.try_downcast_ref::<f32>().is_some() {
        return try_set(target, value as f32);
    }
    if target.try_downcast_ref::<f64>().is_some() {
        return try_set(target, value);
    }
    narrow!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);
    false
}

fn try_set<T: PartialReflect>(target: &mut dyn PartialReflect, value: T) -> bool {
    target.try_apply(&value).is_ok()
}

/// Switch the enum at `path` to variant `name`.
///
/// A concrete Rust enum cannot have its variant changed through `Reflect`
/// alone — `try_apply` on a differently-shaped value is an error by design,
/// because the new variant's fields have to come from *somewhere*. Two sources
/// know:
///
/// - [`GeneratorSpec::default_of_kind`], for a generator root — hand-written
///   precisely because a `Torus(0, 0, 0, 0)` is not a torus. Recognised by
///   downcast so a spec nested anywhere in a bigger value gets the same
///   treatment.
/// - [`default_variant`], for the small enums inside one (`Shading`, an
///   optional colour): every field at its type's zero, and the user drags from
///   there.
///
/// [`GeneratorSpec::default_of_kind`]: crate::gen::GeneratorSpec::default_of_kind
fn apply_variant(
    root: &mut dyn PartialReflect,
    path: &FieldPath,
    name: &str,
) -> Result<(), String> {
    let target = resolve_mut(root, path)?;
    let ReflectRef::Enum(current) = target.reflect_ref() else {
        return Err(format!("{}: not an enum", path.display()));
    };
    if current.variant_name() == name {
        // Same variant, keep the values: "switch a Torus to a Torus" must not
        // reset a half-tuned mesh.
        return Ok(());
    }

    if let Some(spec) = target.try_downcast_mut::<crate::gen::GeneratorSpec>() {
        *spec = crate::gen::GeneratorSpec::default_of_kind(name)
            .ok_or_else(|| format!("{}: no generator kind {name:?}", path.display()))?;
        return Ok(());
    }

    let info = target
        .get_represented_type_info()
        .ok_or_else(|| format!("{}: no type info", path.display()))?;
    let TypeInfo::Enum(enum_info) = info else {
        return Err(format!("{}: not an enum type", path.display()));
    };
    let variant = enum_info
        .variant(name)
        .ok_or_else(|| format!("{}: no variant {name:?}", path.display()))?;

    let replacement = default_variant(variant)?;
    target
        .try_apply(&replacement)
        .map_err(|e| format!("{}: cannot become {name:?}: {e:?}", path.display()))
}

/// A `DynamicEnum` for `variant`, every field filled with its type's zero.
///
/// "Zero" rather than "the value a human would want": reflection knows the
/// shape, not the intent. Switching `Shading` to `Smooth` therefore lands on
/// `Smooth(0.0)` and the user drags from there. The one case where zeros would
/// be actively unhelpful — the whole generator — is caught above by the
/// [`default_of_kind`](crate::gen::GeneratorSpec::default_of_kind) downcast.
fn default_variant(variant: &VariantInfo) -> Result<DynamicEnum, String> {
    use bevy_reflect::structs::DynamicStruct;
    use bevy_reflect::tuple::DynamicTuple;

    let value = match variant {
        VariantInfo::Unit(_) => DynamicVariant::Unit,
        VariantInfo::Tuple(t) => {
            let mut tuple = DynamicTuple::default();
            for i in 0..t.field_len() {
                let field = t.field_at(i).expect("index from field_len");
                tuple.insert_boxed(zero_of(field.type_info(), field.type_path())?);
            }
            DynamicVariant::Tuple(tuple)
        }
        VariantInfo::Struct(s) => {
            let mut fields = DynamicStruct::default();
            for i in 0..s.field_len() {
                let field = s.field_at(i).expect("index from field_len");
                fields.insert_boxed(field.name(), zero_of(field.type_info(), field.type_path())?);
            }
            DynamicVariant::Struct(fields)
        }
    };
    Ok(DynamicEnum::new(variant.name(), value))
}

/// The zero value for a type, built from its static info.
///
/// Scalars have a table; composites are built recursively from their fields, so
/// turning a colour back on produces a real `Some(Vec3(0,0,0))` rather than
/// failing at the first non-primitive. Enums default to their **first declared
/// variant**, which is arbitrary but stable and is why `None` comes before
/// `Some` in [`OptVec3Def`](crate::reflect::OptVec3Def).
///
/// A type that is neither — a `Vec<T>`, a map — reports rather than guessing.
/// Nothing in a generator's params is one today, and inventing a plausible
/// default for a collection is exactly the kind of guess a tool should not
/// make.
fn zero_of(
    info: Option<&'static TypeInfo>,
    type_path: &str,
) -> Result<Box<dyn PartialReflect>, String> {
    use bevy_reflect::structs::DynamicStruct;
    use bevy_reflect::tuple_struct::DynamicTupleStruct;

    if let Some(scalar) = zero_scalar(type_path) {
        return Ok(scalar);
    }

    match info {
        Some(TypeInfo::Struct(s)) => {
            let mut out = DynamicStruct::default();
            out.set_represented_type(info);
            for i in 0..s.field_len() {
                let field = s.field_at(i).expect("index from field_len");
                out.insert_boxed(field.name(), zero_of(field.type_info(), field.type_path())?);
            }
            Ok(Box::new(out))
        }
        Some(TypeInfo::TupleStruct(t)) => {
            let mut out = DynamicTupleStruct::default();
            out.set_represented_type(info);
            for i in 0..t.field_len() {
                let field = t.field_at(i).expect("index from field_len");
                out.insert_boxed(zero_of(field.type_info(), field.type_path())?);
            }
            Ok(Box::new(out))
        }
        Some(TypeInfo::Enum(e)) => {
            let first = e
                .iter()
                .next()
                .ok_or_else(|| format!("{type_path} has no variants"))?;
            let mut out = default_variant(first)?;
            out.set_represented_type(info);
            Ok(Box::new(out))
        }
        _ => Err(format!(
            "cannot build a default {type_path}; switch this variant from code instead"
        )),
    }
}

fn zero_scalar(type_path: &str) -> Option<Box<dyn PartialReflect>> {
    Some(match type_path {
        "f32" => Box::new(0.0f32),
        "f64" => Box::new(0.0f64),
        "bool" => Box::new(false),
        "u8" => Box::new(0u8),
        "u16" => Box::new(0u16),
        "u32" => Box::new(0u32),
        "u64" => Box::new(0u64),
        "usize" => Box::new(0usize),
        "i8" => Box::new(0i8),
        "i16" => Box::new(0i16),
        "i32" => Box::new(0i32),
        "i64" => Box::new(0i64),
        "isize" => Box::new(0isize),
        "alloc::string::String" => Box::new(String::new()),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen::{GeneratorSpec, Shading};
    use bevy_reflect::Reflect;
    use glam::Vec3;

    fn torus() -> GeneratorSpec {
        GeneratorSpec::default_of_kind("Torus").expect("Torus is a kind")
    }

    #[test]
    fn a_generator_walks_to_a_variant_root_with_the_kinds_as_options() {
        let spec = torus();
        let Widget::Variant { selected, options, fields, .. } = build(&spec, "generator") else {
            panic!("a data enum is a Variant");
        };
        assert_eq!(selected, "Torus");
        assert_eq!(options, GeneratorSpec::KINDS, "the dropdown is the kind list");
        assert_eq!(
            fields.iter().map(|f| f.label()).collect::<Vec<_>>(),
            vec![
                "major radius",
                "minor radius",
                "major segments",
                "minor segments",
                "shading",
                "color"
            ],
        );
    }

    #[test]
    fn ranges_come_off_the_variant_field_annotations() {
        let Widget::Variant { fields, .. } = build(&torus(), "generator") else {
            panic!("not a variant");
        };
        let Widget::Float { value, range, .. } = &fields[0] else {
            panic!("major_radius is a float");
        };
        assert_eq!(*value, 1.0);
        assert_eq!(*range, FieldRange::new(0.01, 20.0));
        let Widget::Int { range, .. } = &fields[2] else {
            panic!("major_segments is an int");
        };
        assert_eq!(*range, FieldRange::new(3.0, 256.0));
    }

    #[test]
    fn shading_and_color_are_nested_variants_with_their_own_options() {
        let Widget::Variant { fields, .. } = build(&torus(), "generator") else {
            panic!("not a variant");
        };
        // `Shading` carries data (`Smooth(f32)`), so it is a Variant even while
        // its *active* variant is unit — the whole enum decides, not the value.
        let Widget::Variant { selected, options, fields: inner, .. } = &fields[4] else {
            panic!("shading is a variant");
        };
        assert_eq!(selected, "Generated");
        assert_eq!(options, &["Generated", "Flat", "Smooth"]);
        assert!(inner.is_empty(), "a unit variant has no field rows");

        let Widget::Variant { selected, options, .. } = &fields[5] else {
            panic!("an optional colour is a variant");
        };
        assert_eq!(selected, "None");
        assert_eq!(options, &["None", "Some"]);
    }

    #[test]
    fn a_terrain_walks_to_a_seed_and_a_vector() {
        let spec = GeneratorSpec::default_of_kind("Terrain").expect("kind");
        let Widget::Variant { fields, .. } = build(&spec, "generator") else {
            panic!("not a variant");
        };
        // A single-field tuple variant drops the `#0` label.
        let Widget::Group { label, fields: params, .. } = &fields[0] else {
            panic!("terrain params are a group");
        };
        assert_eq!(label, "");

        let Widget::Seed { path, .. } = &params[0] else {
            panic!("a u64 named seed is a Seed, not an Int");
        };
        assert_eq!(path.0, vec![Step::Index(0), Step::Field("seed".into())]);

        // A remote-defined `Vec2` needs no special case: it is a struct of two
        // `f32`s named x/y, and the composite's range rides along.
        let Widget::Vector { components, range, .. } = &params[1] else {
            panic!("size is a vector");
        };
        assert_eq!(
            components.iter().map(|c| c.label.as_str()).collect::<Vec<_>>(),
            vec!["x", "y"]
        );
        assert_eq!(*range, FieldRange::new(1.0, 256.0));
    }

    #[test]
    fn a_vector_flattens_into_float_rows_that_carry_its_range() {
        let spec = GeneratorSpec::default_of_kind("Terrain").expect("kind");
        let rows = build(&spec, "generator").rows();
        let (depth, x_row) = rows
            .iter()
            .find(|(_, w)| matches!(w, Widget::Float { path, .. } if path.0.last() == Some(&Step::Field("x".into()))))
            .expect("the x component became a row");
        let Widget::Float { range, .. } = x_row else { unreachable!() };
        assert_eq!(*range, FieldRange::new(1.0, 256.0));
        assert_eq!(*depth, 3, "root → params → size → x");
    }

    /// `Color` is never produced by the walk (see the module docs), so this
    /// pins the one thing it shares with `Vector`: [`Widget::push_rows`]
    /// flattens a hand-built one exactly the same way, channel order and all.
    #[test]
    fn a_color_flattens_into_its_channel_rows_in_order() {
        let path = FieldPath::root().field("tint");
        let component = |name: &str, value: f64| VectorComponent {
            label: name.to_string(),
            path: path.field(name),
            value,
        };
        let widget = Widget::Color {
            label: "tint".into(),
            path: path.clone(),
            value: [0.2, 0.4, 0.8],
            components: vec![
                component("r", 0.2),
                component("g", 0.4),
                component("b", 0.8),
                component("a", 1.0),
            ],
            range: FieldRange::new(0.0, 1.0),
        };

        let rows = widget.rows();
        assert!(matches!(&rows[0], (0, Widget::Color { .. })), "the header comes first, at its own depth");
        let channels: Vec<(&str, f64, usize)> = rows[1..]
            .iter()
            .map(|(depth, w)| {
                let Widget::Float { label, value, range, .. } = w else {
                    panic!("a channel row is a Float: {w:?}");
                };
                assert_eq!(*range, FieldRange::new(0.0, 1.0), "the channel carries the colour's range");
                (label.as_str(), *value, *depth)
            })
            .collect();
        assert_eq!(
            channels,
            vec![("r", 0.2, 1), ("g", 0.4, 1), ("b", 0.8, 1), ("a", 1.0, 1)],
            "r, g, b, a — in that order, one below the header",
        );
    }

    #[test]
    fn composite_ranges_are_inherited_by_undeclared_leaves() {
        // The same shape tweak's inheritance test pins, walked by this mapper.
        #[derive(Reflect)]
        struct Nested {
            #[reflect(@FieldRange::new(0.0, 1.0))]
            gain: f32,
            count: u32,
            offset: i32,
        }
        #[derive(Reflect)]
        struct Params {
            #[reflect(@FieldRange::new(-2.0, 2.0))]
            nested: Nested,
        }
        let value = Params {
            nested: Nested { gain: 0.5, count: 4, offset: -1 },
        };
        let Widget::Group { fields, .. } = build(&value, "params") else {
            panic!("not a group");
        };
        let Widget::Group { fields: leaves, .. } = &fields[0] else {
            panic!("nested is a group");
        };
        // Declared on the leaf, so the composite's −2…2 does not reach it.
        assert!(matches!(&leaves[0], Widget::Float { range, .. } if *range == FieldRange::new(0.0, 1.0)));
        // Not declared on the leaf, so the composite's does.
        assert!(matches!(&leaves[1], Widget::Int { range, .. } if *range == FieldRange::new(-2.0, 2.0)));
        assert!(matches!(&leaves[2], Widget::Int { range, .. } if *range == FieldRange::new(-2.0, 2.0)));
    }

    #[test]
    fn a_small_float_array_is_a_vector_whose_components_take_edits() {
        #[derive(Reflect)]
        struct Params {
            #[reflect(@FieldRange::new(0.1, 20.0))]
            size: [f32; 3],
            weights: [f32; 6],
        }
        let mut value = Params {
            size: [2.0, 2.0, 2.0],
            weights: [0.0; 6],
        };
        let Widget::Group { fields, .. } = build(&value, "params") else {
            panic!("not a group");
        };
        // The array walks to the same row a `Vec3` does, axes and all, with
        // the field's own range riding along.
        let Widget::Vector { components, range, .. } = &fields[0] else {
            panic!("a small float array is a vector: {:?}", fields[0]);
        };
        assert_eq!(
            components.iter().map(|c| c.label.as_str()).collect::<Vec<_>>(),
            vec!["x", "y", "z"]
        );
        assert_eq!(*range, FieldRange::new(0.1, 20.0));
        // Too long for axes: visible, not editable — the walk's usual refusal.
        assert!(matches!(&fields[1], Widget::Unsupported { .. }));

        // An edit lands on the element through an Index step…
        let path = FieldPath::root().field("size").index(1);
        apply(&mut value, &path, &Edit::Float(5.0)).expect("set");
        assert_eq!(value.size, [2.0, 5.0, 2.0]);
        // …and an index off the end is a dropped edit, not a panic.
        let stale = FieldPath::root().field("size").index(9);
        assert!(apply(&mut value, &stale, &Edit::Float(1.0)).is_err());
        assert_eq!(value.size, [2.0, 5.0, 2.0], "nothing moved");
    }

    #[test]
    fn a_unit_enum_is_a_choice_and_a_signed_int_gets_the_signed_default() {
        #[derive(Reflect, Default)]
        enum Mode {
            #[default]
            Off,
            Slow,
            Fast,
        }
        #[derive(Reflect, Default)]
        struct Rig {
            mode: Mode,
            bias: i32,
        }
        let Widget::Group { fields, .. } = build(&Rig::default(), "rig") else {
            panic!("not a group");
        };
        let Widget::Choice { selected, options, .. } = &fields[0] else {
            panic!("a unit enum is a choice, exactly as tweak has it");
        };
        assert_eq!(selected, "Off");
        assert_eq!(options, &["Off", "Slow", "Fast"]);
        assert!(matches!(&fields[1], Widget::Int { range, .. } if *range == DEFAULT_SIGNED_RANGE));
    }

    #[test]
    fn a_float_edit_lands_and_is_not_clamped_at_the_advisory_range() {
        let mut spec = torus();
        let path = FieldPath::root().field("major_radius");
        apply(&mut spec, &path, &Edit::Float(5.0)).expect("set");
        assert!(matches!(spec, GeneratorSpec::Torus { major_radius, .. } if major_radius == 5.0));

        // Past the declared 20.0 end: the range is advisory (gen.rs doctrine),
        // so the value goes where it was sent — the opposite of tweak's `set`.
        apply(&mut spec, &path, &Edit::Float(500.0)).expect("set");
        assert!(matches!(spec, GeneratorSpec::Torus { major_radius, .. } if major_radius == 500.0));
    }

    #[test]
    fn an_int_edit_rounds_and_saturates_at_the_type_not_the_range() {
        let mut spec = torus();
        let path = FieldPath::root().field("major_segments");
        apply(&mut spec, &path, &Edit::Int(48)).expect("set");
        assert!(matches!(spec, GeneratorSpec::Torus { major_segments, .. } if major_segments == 48));
        // A negative into a `u32` saturates at the width rather than wrapping.
        apply(&mut spec, &path, &Edit::Int(-5)).expect("set");
        assert!(matches!(spec, GeneratorSpec::Torus { major_segments, .. } if major_segments == 0));
    }

    #[test]
    fn switching_the_generator_lands_on_its_hand_written_default() {
        let mut spec = torus();
        apply(&mut spec, &FieldPath::root(), &Edit::Variant("Cube".into())).expect("switch");
        assert_eq!(spec, GeneratorSpec::default_of_kind("Cube").expect("kind"));
        // …and a kind the registry does not have is refused, not guessed.
        assert!(apply(&mut spec, &FieldPath::root(), &Edit::Variant("Blob".into())).is_err());
    }

    #[test]
    fn switching_a_nested_variant_builds_its_fields_from_zeros() {
        let mut spec = torus();
        let shading = FieldPath::root().field("shading");
        apply(&mut spec, &shading, &Edit::Variant("Smooth".into())).expect("switch");
        assert!(matches!(spec, GeneratorSpec::Torus { shading: Shading::Smooth(v), .. } if v == 0.0));

        // Turning a colour on produces a real `Some(Vec3::ZERO)` through the
        // remote definitions — the recursive half of `zero_of`.
        let color = FieldPath::root().field("color");
        apply(&mut spec, &color, &Edit::Variant("Some".into())).expect("switch");
        assert!(matches!(spec, GeneratorSpec::Torus { color: Some(c), .. } if c == Vec3::ZERO));
    }

    #[test]
    fn switching_to_the_current_variant_keeps_the_values() {
        let mut spec = torus();
        apply(
            &mut spec,
            &FieldPath::root().field("major_radius"),
            &Edit::Float(7.5),
        )
        .expect("set");
        apply(&mut spec, &FieldPath::root(), &Edit::Variant("Torus".into())).expect("no-op");
        assert!(
            matches!(spec, GeneratorSpec::Torus { major_radius, .. } if major_radius == 7.5),
            "a half-tuned mesh survived selecting its own kind"
        );
    }

    #[test]
    fn a_reroll_changes_the_seed_and_is_deterministic() {
        let mut spec = GeneratorSpec::default_of_kind("Terrain").expect("kind");
        let path = FieldPath::root().index(0).field("seed");
        let GeneratorSpec::Terrain(before) = spec else { unreachable!() };

        let next = reroll(before.seed);
        assert_ne!(next, before.seed);
        assert_eq!(next, reroll(before.seed), "a reroll must replay");

        apply(&mut spec, &path, &Edit::Seed(next)).expect("seed");
        let GeneratorSpec::Terrain(after) = spec else { unreachable!() };
        assert_eq!(after.seed, next);
    }

    #[test]
    fn a_stale_path_is_a_dropped_edit_not_a_panic() {
        let mut spec = torus();
        // The path a Torus slider held, after the user switched to a Cube.
        apply(&mut spec, &FieldPath::root(), &Edit::Variant("Cube".into())).expect("switch");
        let stale = FieldPath::root().field("major_radius");
        assert!(apply(&mut spec, &stale, &Edit::Float(2.0)).is_err());
        assert_eq!(
            spec,
            GeneratorSpec::default_of_kind("Cube").expect("kind"),
            "nothing moved"
        );
    }

    #[test]
    fn the_wrong_kind_of_edit_is_an_error_rather_than_a_write() {
        let mut spec = torus();
        let path = FieldPath::root().field("shading");
        assert!(apply(&mut spec, &path, &Edit::Float(1.0)).is_err());
        assert_eq!(spec, torus(), "nothing moved");
    }
}
