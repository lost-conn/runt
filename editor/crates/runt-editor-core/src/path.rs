//! Addressing a field inside a reflected value.
//!
//! A widget in the param panel is not a `&mut f32` — the value it edits lives
//! behind a channel, on another thread, and may have been replaced by the time
//! the edit arrives. So a widget carries a **path** instead: a list of steps
//! from the root of the edited value down to the leaf, which the engine thread
//! re-walks against whatever the current value is.
//!
//! `bevy_reflect` ships its own string path syntax (`GetPath`). This is a
//! deliberate reimplementation, for two reasons:
//!
//! - **Enums.** A path into an enum has to survive the variant changing under
//!   it. Re-walking an explicit step list lets a mismatch be an ordinary
//!   `Err("no such field")` that the UI can drop on the floor, rather than a
//!   parse-time surprise.
//! - **It is the widget's identity.** The UI keys widgets by path to decide what
//!   is the *same* control across a rebuild, so it wants a comparable, hashable
//!   value, not a string that has to be re-parsed on every keystroke.

use bevy_reflect::PartialReflect;

/// One step from a reflected value to one of its children.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Step {
    /// A named field of a struct or of an enum's *active* struct variant.
    Field(String),
    /// A positional field of a tuple struct or an enum's active tuple variant.
    Index(usize),
}

/// A path from the root of an edited value to one field, root-first.
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

    /// This path with one more step on the end. Building the widget tree is a
    /// depth-first walk, so paths are grown by copy rather than mutated.
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

    /// The path to this path's parent, and the final step — what an edit needs
    /// when it has to replace a whole child rather than write through to a leaf.
    pub fn split_last(&self) -> Option<(FieldPath, &Step)> {
        let (last, rest) = self.0.split_last()?;
        Some((FieldPath(rest.to_vec()), last))
    }

    /// A display form: `.size.x`, `#0.amplitude`. Used in labels and in error
    /// messages, never parsed back.
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

/// Follow `path` from `root`, immutably.
pub fn resolve<'a>(
    root: &'a dyn PartialReflect,
    path: &FieldPath,
) -> Result<&'a dyn PartialReflect, String> {
    let mut cursor = root;
    for (depth, step) in path.0.iter().enumerate() {
        cursor = step_into(cursor, step)
            .ok_or_else(|| missing(path, depth, step))?;
    }
    Ok(cursor)
}

/// Follow `path` from `root`, mutably.
pub fn resolve_mut<'a>(
    root: &'a mut dyn PartialReflect,
    path: &FieldPath,
) -> Result<&'a mut dyn PartialReflect, String> {
    let mut cursor = root;
    for (depth, step) in path.0.iter().enumerate() {
        // The error message needs `step` after the borrow of `cursor` is gone,
        // so build it before moving into the child.
        let message = missing(path, depth, step);
        cursor = step_into_mut(cursor, step).ok_or(message)?;
    }
    Ok(cursor)
}

fn missing(path: &FieldPath, depth: usize, step: &Step) -> String {
    format!(
        "{}: no {} at depth {depth}",
        path.display(),
        match step {
            Step::Field(name) => format!("field {name:?}"),
            Step::Index(i) => format!("element #{i}"),
        }
    )
}

fn step_into<'a>(value: &'a dyn PartialReflect, step: &Step) -> Option<&'a dyn PartialReflect> {
    use bevy_reflect::ReflectRef;
    match (value.reflect_ref(), step) {
        (ReflectRef::Struct(s), Step::Field(name)) => s.field(name),
        (ReflectRef::Struct(s), Step::Index(i)) => s.field_at(*i),
        (ReflectRef::TupleStruct(t), Step::Index(i)) => t.field(*i),
        (ReflectRef::Tuple(t), Step::Index(i)) => t.field(*i),
        // An enum exposes only its *active* variant's fields, which is exactly
        // the behaviour a panel wants: a path into `Some(..)` stops resolving
        // the moment the value becomes `None`.
        (ReflectRef::Enum(e), Step::Field(name)) => e.field(name),
        (ReflectRef::Enum(e), Step::Index(i)) => e.field_at(*i),
        (ReflectRef::List(l), Step::Index(i)) => l.get(*i),
        _ => None,
    }
}

fn step_into_mut<'a>(
    value: &'a mut dyn PartialReflect,
    step: &Step,
) -> Option<&'a mut dyn PartialReflect> {
    use bevy_reflect::ReflectMut;
    match (value.reflect_mut(), step) {
        (ReflectMut::Struct(s), Step::Field(name)) => s.field_mut(name),
        (ReflectMut::Struct(s), Step::Index(i)) => s.field_at_mut(*i),
        (ReflectMut::TupleStruct(t), Step::Index(i)) => t.field_mut(*i),
        (ReflectMut::Tuple(t), Step::Index(i)) => t.field_mut(*i),
        (ReflectMut::Enum(e), Step::Field(name)) => e.field_mut(name),
        (ReflectMut::Enum(e), Step::Index(i)) => e.field_at_mut(*i),
        (ReflectMut::List(l), Step::Index(i)) => l.get_mut(*i),
        _ => None,
    }
}
