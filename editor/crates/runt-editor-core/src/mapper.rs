//! The `Reflect`-walking widget mapper (DESIGN §10).
//!
//! > *Panels are generated from `Reflect` param structs: a `Reflect`-walking
//! > widget mapper (f32 → slider with range attributes, enum → select, Vec3 →
//! > triple, seed → reroll button). Hand-written panels only where reflection
//! > genuinely can't express the interaction.* — DESIGN §10
//!
//! This module is that mapper, and it is deliberately **not** a UI: it turns a
//! `&dyn PartialReflect` into a [`Widget`] tree of pure data, and turns an
//! [`Edit`] back into a mutation of the original value. The rinch layer renders
//! the tree and emits `Edit`s. Two consequences worth stating:
//!
//! - The mapping is testable without a window, and it is (see the tests below).
//! - There is exactly one place that decides "an `f32` is a slider" — here — so
//!   the answer cannot vary by generator.
//!
//! ## The mapping
//!
//! | Reflected shape | Widget |
//! |---|---|
//! | `f32`, `f64` | [`Widget::Float`] — slider bounded by `FieldRange`, plus text entry |
//! | `u32`, `u64`, `usize`, `i32`… | [`Widget::Int`] — stepper |
//! | `u64` named `seed` | [`Widget::Seed`] — text field + reroll |
//! | `bool` | [`Widget::Bool`] — checkbox |
//! | `String` | [`Widget::Text`] |
//! | struct of 2–4 `f32` named `x`,`y`,`z`,`w` | [`Widget::Vector`] — grouped numerics |
//! | any other struct / tuple struct | [`Widget::Group`] |
//! | enum | [`Widget::Variant`] — a select over the variant names, plus the active variant's fields |
//!
//! `Vec2`/`Vec3` land in the vector row because of the remote definitions in
//! `runt_core::reflect`; the mapper never names a glam type.
//!
//! ## Ranges
//!
//! Bounds come from `#[reflect(@FieldRange…)]` on the param itself, read out of
//! static `TypeInfo` as the walk descends. Unannotated numbers fall back to
//! `DEFAULT_RANGE` / `DEFAULT_INT_RANGE`, so a new param is usable before anyone
//! has thought about its bounds — it just is not *well* bounded. There is no
//! table in this crate to keep in sync; see `runt_core::reflect` for why.

use bevy_reflect::enums::VariantInfo;
use bevy_reflect::{PartialReflect, ReflectRef, TypeInfo};
use runt_core::reflect::{FieldRange, DEFAULT_INT_RANGE, DEFAULT_RANGE};

use crate::path::{resolve_mut, FieldPath, Step};

/// One control in a generated panel.
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
    /// A `u64` field named `seed`: too wide for a slider and meaningless as a
    /// magnitude, so it gets a text field and a reroll button instead.
    Seed {
        label: String,
        path: FieldPath,
        value: u64,
    },
    Bool {
        label: String,
        path: FieldPath,
        value: bool,
    },
    Text {
        label: String,
        path: FieldPath,
        value: String,
    },
    /// A small struct of `f32`s named `x`/`y`/`z`/`w` — one labelled row of
    /// numeric fields rather than a nested group.
    Vector {
        label: String,
        path: FieldPath,
        components: Vec<VectorComponent>,
        range: FieldRange,
    },
    /// An enum: a variant selector plus whatever the active variant contains.
    Variant {
        label: String,
        path: FieldPath,
        selected: String,
        options: Vec<String>,
        fields: Vec<Widget>,
    },
    /// A struct that is not a vector: a heading and its fields.
    Group {
        label: String,
        path: FieldPath,
        fields: Vec<Widget>,
    },
    /// Something the mapper has no control for. Rendered as read-only text so a
    /// new param type shows up as "not editable yet" rather than vanishing.
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
            | Widget::Seed { label, .. }
            | Widget::Bool { label, .. }
            | Widget::Text { label, .. }
            | Widget::Vector { label, .. }
            | Widget::Variant { label, .. }
            | Widget::Group { label, .. }
            | Widget::Unsupported { label, .. } => label,
        }
    }

    pub fn path(&self) -> &FieldPath {
        match self {
            Widget::Float { path, .. }
            | Widget::Int { path, .. }
            | Widget::Seed { path, .. }
            | Widget::Bool { path, .. }
            | Widget::Text { path, .. }
            | Widget::Vector { path, .. }
            | Widget::Variant { path, .. }
            | Widget::Group { path, .. }
            | Widget::Unsupported { path, .. } => path,
        }
    }

    /// This widget and every widget under it, depth first. What the UI iterates
    /// when it flattens a panel into rows.
    pub fn flatten(&self) -> Vec<&Widget> {
        let mut out = vec![self];
        match self {
            Widget::Variant { fields, .. } | Widget::Group { fields, .. } => {
                for f in fields {
                    out.extend(f.flatten());
                }
            }
            _ => {}
        }
        out
    }

    /// [`flatten`](Widget::flatten), owned, with each row's nesting depth.
    ///
    /// The UI needs owned rows (its list is rebuilt from a signal, not borrowed
    /// from one) and the depth to indent by, and getting both from one walk is
    /// cheaper than cloning the tree and walking it twice.
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
            _ => {}
        }
    }
}

/// Build the panel for a reflected value.
///
/// `label` names the root — the generator's name, usually.
pub fn build(value: &dyn PartialReflect, label: impl Into<String>) -> Widget {
    build_at(value, label.into(), FieldPath::root(), None)
}

fn build_at(
    value: &dyn PartialReflect,
    label: String,
    path: FieldPath,
    range: Option<FieldRange>,
) -> Widget {
    // Leaves first: a scalar is never also a struct, and checking here keeps the
    // opaque/value arm below from having to re-test everything.
    if let Some(widget) = leaf(value, &label, &path, range) {
        return widget;
    }

    match value.reflect_ref() {
        ReflectRef::Struct(s) => {
            let info = s.get_represented_type_info();
            if let Some(components) = vector_components(value, &path) {
                return Widget::Vector {
                    label,
                    path,
                    components,
                    range: range.unwrap_or(DEFAULT_RANGE),
                };
            }
            let fields = (0..s.field_len())
                .filter_map(|i| {
                    let name = s.name_at(i)?;
                    let child = s.field_at(i)?;
                    Some(build_at(
                        child,
                        pretty(name),
                        path.field(name),
                        struct_field_range(info, name),
                    ))
                })
                .collect();
            Widget::Group { label, path, fields }
        }

        ReflectRef::TupleStruct(t) => {
            let fields = (0..t.field_len())
                .filter_map(|i| {
                    let child = t.field(i)?;
                    Some(build_at(child, format!("#{i}"), path.index(i), range))
                })
                .collect();
            Widget::Group { label, path, fields }
        }

        ReflectRef::Enum(e) => {
            let info = e.get_represented_type_info();
            let options = variant_names(info);
            let variant = active_variant(info, e.variant_name());
            let fields = (0..e.field_len())
                .filter_map(|i| {
                    let child = e.field_at(i)?;
                    match e.name_at(i) {
                        // A struct variant: named fields, each with its own range.
                        Some(name) => Some(build_at(
                            child,
                            pretty(name),
                            path.field(name),
                            variant.and_then(|v| variant_field_range(v, Some(name), i)),
                        )),
                        // A tuple variant: positional. A single-field tuple
                        // variant (`Smooth(f32)`, `Some(Vec3)`) inherits the
                        // enum's own label, because "Smooth → #0" reads worse
                        // than "Smooth" with a number next to it.
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
                                .or(range),
                        )),
                    }
                })
                .collect();
            Widget::Variant {
                label,
                path,
                selected: e.variant_name().to_string(),
                options,
                fields,
            }
        }

        _ => Widget::Unsupported {
            label,
            path,
            type_name: value.reflect_type_path().to_string(),
        },
    }
}

/// A scalar leaf, if this value is one.
fn leaf(
    value: &dyn PartialReflect,
    label: &str,
    path: &FieldPath,
    range: Option<FieldRange>,
) -> Option<Widget> {
    let name = path.0.last().and_then(|s| match s {
        Step::Field(n) => Some(n.as_str()),
        Step::Index(_) => None,
    });

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
    if let Some(v) = value.try_downcast_ref::<String>() {
        return Some(Widget::Text {
            label: label.to_string(),
            path: path.clone(),
            value: v.clone(),
        });
    }
    if let Some(v) = value.try_downcast_ref::<u64>() {
        // The one name-based rule in the mapper. A seed is a `u64` whose
        // *magnitude* means nothing, so a slider over it would be a cruel joke;
        // it wants a text field and a reroll. `seed` is the name every runt
        // generator uses, and the reflect tests pin that.
        return Some(if name == Some("seed") {
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
        ($($ty:ty),*) => {$(
            if let Some(v) = value.try_downcast_ref::<$ty>() {
                return Some(Widget::Int {
                    label: label.to_string(),
                    path: path.clone(),
                    value: *v as i64,
                    range: range.unwrap_or(DEFAULT_INT_RANGE),
                });
            }
        )*};
    }
    int_leaf!(u8, u16, u32, usize, i8, i16, i32, i64, isize);

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
    for i in 0..s.field_len() {
        let name = s.name_at(i)?;
        if name != AXES[i] {
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

fn struct_field_range(info: Option<&TypeInfo>, field: &str) -> Option<FieldRange> {
    match info? {
        TypeInfo::Struct(s) => s.field(field)?.get_attribute::<FieldRange>().copied(),
        _ => None,
    }
}

fn active_variant<'a>(info: Option<&'a TypeInfo>, name: &str) -> Option<&'a VariantInfo> {
    match info? {
        TypeInfo::Enum(e) => e.variant(name),
        _ => None,
    }
}

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

fn variant_names(info: Option<&TypeInfo>) -> Vec<String> {
    match info {
        Some(TypeInfo::Enum(e)) => e.iter().map(|v| v.name().to_string()).collect(),
        _ => Vec::new(),
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
    Seed(u64),
    Bool(bool),
    Text(String),
    /// Switch an enum to a different variant. Only the *name* travels; what the
    /// new variant's fields should contain is [`apply`]'s problem.
    Variant(String),
}

/// Apply `edit` at `path` inside `root`.
///
/// Writes through to the leaf: a `Float` edit becomes a real `f32` assignment on
/// the real struct field, which is why the engine can take the whole edited
/// value afterwards and hash it like any other.
///
/// Errors are strings and are meant to be *ignorable*: a path that no longer
/// resolves (the user switched a variant while a slider was mid-drag) is a
/// dropped edit, not a broken editor.
pub fn apply(root: &mut dyn PartialReflect, path: &FieldPath, edit: &Edit) -> Result<(), String> {
    if let Edit::Variant(name) = edit {
        return apply_variant(root, path, name);
    }

    let target = resolve_mut(root, path)?;
    let type_path = target.reflect_type_path().to_string();

    let ok = match edit {
        Edit::Float(v) => set_number(target, *v),
        Edit::Int(v) => set_number(target, *v as f64),
        Edit::Seed(v) => try_set(target, *v),
        Edit::Bool(v) => try_set(target, *v),
        Edit::Text(v) => try_set(target, v.clone()),
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
/// segment count or an `f32` radius — so the narrowing happens here, once, where
/// the concrete type is known.
fn set_number(target: &mut dyn PartialReflect, value: f64) -> bool {
    macro_rules! narrow {
        ($($ty:ty),*) => {$(
            if target.try_downcast_ref::<$ty>().is_some() {
                // `round` before the cast: a slider that reports 23.999999 for a
                // segment count must not silently mean 23.
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
/// A concrete Rust enum cannot have its variant changed through `Reflect` alone
/// — `try_apply` on a differently-shaped value is an error by design, because
/// the new variant's fields have to come from *somewhere*. So this reaches for
/// the one thing that knows: [`GeneratorSpec::default_of_kind`], for the root,
/// and [`default_variant`] for the small enums nested inside it.
///
/// [`GeneratorSpec::default_of_kind`]: runt_core::gen::GeneratorSpec::default_of_kind
pub fn apply_variant(
    root: &mut dyn PartialReflect,
    path: &FieldPath,
    name: &str,
) -> Result<(), String> {
    let target = resolve_mut(root, path)?;
    let ReflectRef::Enum(current) = target.reflect_ref() else {
        return Err(format!("{}: not an enum", path.display()));
    };
    if current.variant_name() == name {
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
/// `Smooth(0.0)` and the user drags from there. The one case where that would be
/// actively unhelpful — swapping the whole *generator* — is handled a level up,
/// by `GeneratorSpec::default_of_kind`, which is hand-written precisely because
/// a `Torus(0, 0, 0, 0)` is not a torus.
fn default_variant(variant: &VariantInfo) -> Result<bevy_reflect::enums::DynamicEnum, String> {
    use bevy_reflect::enums::{DynamicEnum, DynamicVariant};
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
                fields.insert_boxed(
                    field.name(),
                    zero_of(field.type_info(), field.type_path())?,
                );
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
/// `Some` in `OptVec3Def`.
///
/// A type that is neither — a `Vec<T>`, a map — reports rather than guessing.
/// Nothing in a generator's params is one today, and inventing a plausible
/// default for a collection is exactly the kind of guess a tool should not make.
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
