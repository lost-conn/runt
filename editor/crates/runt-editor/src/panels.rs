//! The panels: entity list on the left, generated inspector on the right.
//!
//! The inspector is the point of the whole exercise. It contains **no knowledge
//! of any generator**: it is a `for` loop over rows that
//! `runt_editor_core::mapper` produced by walking a `Reflect` value, and a
//! `match` from widget kind to rinch component. Adding a generator to
//! `runt-core` adds a panel here for free; adding a *param type* is the only
//! thing that needs a new arm.
//!
//! The transform block below it is the one hand-written panel, and DESIGN §10
//! sanctions exactly that ("hand-written panels only where reflection genuinely
//! can't express the interaction"): a transform's rotation is stored as a
//! quaternion and edited as Euler degrees, which is a *conversion*, not a
//! rendering of a field.
//!
//! Every component here takes [`Ctx`] — a `Copy` `&'static` handle — precisely
//! so the closures below can be written without a single `.clone()`. See
//! `state.rs` for why.

use std::time::Instant;

use rinch::prelude::*;
use runt_core::gen::GeneratorSpec;
use runt_editor_core::mapper::{Edit, Widget};
use runt_editor_core::protocol::{Command, EntitySnapshot};
use runt_editor_core::FieldPath;

use crate::state::{Ctx, Row};

/// Indentation for a nested row, in pixels.
fn indent(depth: usize) -> u32 {
    (depth.saturating_sub(1) * 10) as u32
}

/// Bump the UI's redraw token.
fn touch(revision: Signal<u32>) {
    revision.update(|r| *r = r.wrapping_add(1));
}

// ---------------------------------------------------------------------------
// reactive reads
//
// `rsx!`'s `for` and `if` take *expressions*, so each of these is a free
// function rather than an inline block. Reading `revision` inside is what
// subscribes the surrounding node to the engine's snapshots — the state itself
// lives outside the reactive system, which is what lets it hold a `RefCell` and
// a GPU thread handle.
// ---------------------------------------------------------------------------

fn entity_rows(ctx: Ctx, revision: Signal<u32>) -> Vec<EntitySnapshot> {
    let _ = revision.get();
    ctx.state.borrow().scene.entities.clone()
}

fn scene_is_empty(ctx: Ctx, revision: Signal<u32>) -> bool {
    let _ = revision.get();
    ctx.state.borrow().scene.entities.is_empty()
}

fn nothing_selected(ctx: Ctx, revision: Signal<u32>) -> bool {
    let _ = revision.get();
    ctx.state.borrow().selected.is_none()
}

fn param_rows(ctx: Ctx, revision: Signal<u32>) -> Vec<Row> {
    let _ = revision.get();
    ctx.state.borrow().rows()
}

fn has_transform(ctx: Ctx, revision: Signal<u32>) -> bool {
    let _ = revision.get();
    ctx.state.borrow().draft_transform.is_some()
}

// ---------------------------------------------------------------------------
// left panel
// ---------------------------------------------------------------------------

#[component]
pub fn entity_list(ctx: Ctx, revision: Signal<u32>) -> NodeHandle {
    rsx! {
        Stack {
            gap: "2",
            Text { size: "xs", color: "dimmed", weight: "bold", "ENTITIES" }

            for entity in entity_rows(ctx, revision) {
                div {
                    key: entity.index,
                    style: {
                        let index = entity.index;
                        move || {
                            let _ = revision.get();
                            let selected = ctx.state.borrow().selected == Some(index);
                            format!(
                                "padding: 4px 6px; border-radius: 3px; cursor: pointer; \
                                 font-size: 12px; background: {}; color: {};",
                                if selected { "var(--rinch-primary-color)" } else { "transparent" },
                                if selected { "#fff" } else { "inherit" },
                            )
                        }
                    },
                    onclick: {
                        let index = entity.index;
                        move || {
                            ctx.state.borrow_mut().select(Some(index));
                            ctx.engine.send(Command::Select(Some(index)));
                            touch(revision);
                        }
                    },
                    {entity.label.clone()}
                    span {
                        style: "opacity: 0.55; font-size: 11px; margin-left: 6px;",
                        {format!("({})", entity.generator)}
                    }
                }
            }

            if scene_is_empty(ctx, revision) {
                Text {
                    size: "xs",
                    color: "dimmed",
                    style: "padding: 12px 4px;",
                    "No scene loaded. Pick one from the toolbar."
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// right panel
// ---------------------------------------------------------------------------

#[component]
pub fn inspector(ctx: Ctx, revision: Signal<u32>) -> NodeHandle {
    rsx! {
        Stack {
            gap: "4",

            if nothing_selected(ctx, revision) {
                Text {
                    size: "xs",
                    color: "dimmed",
                    style: "padding: 12px 4px;",
                    "Select an entity to edit its generator."
                }
            }

            for row in param_rows(ctx, revision) {
                div {
                    key: row.key.clone(),
                    style: {format!(
                        "padding-left: {}px; width: 100%; box-sizing: border-box;",
                        indent(row.depth)
                    )},
                    {param_row(__scope, ctx, revision, row.widget.clone())}
                }
            }

            {transform_panel(__scope, ctx, revision)}
        }
    }
}

/// Route a change through the draft, the debouncer and (eventually) the engine.
///
/// Every control funnels through here, which is why there is exactly one place
/// that decides how an edit is coalesced. A **variant** change is sent
/// immediately rather than debounced: it is a discrete click, not a drag, and
/// waiting 120 ms to see a Torus become a Cone reads as a bug.
fn commit(ctx: Ctx, revision: Signal<u32>, path: &FieldPath, edit: Edit) {
    let immediate = matches!(edit, Edit::Variant(_));
    let Some((generator, spec)) = ctx.state.borrow_mut().edit(path, &edit) else {
        return;
    };
    if immediate {
        ctx.engine.send(Command::ParamEdit { generator, spec });
    } else {
        ctx.pending
            .borrow_mut()
            .push(generator, spec, Instant::now());
    }
    touch(revision);
}

/// Switch the whole generator to another kind.
///
/// The one place that does *not* go through the reflection mapper: a zero-filled
/// `Torus` is not a torus, so this reaches for `GeneratorSpec::default_of_kind`,
/// which is hand-written and exhaustively matched in `runt-core`.
fn switch_generator(ctx: Ctx, revision: Signal<u32>, kind: &str) {
    let mut state = ctx.state.borrow_mut();
    let Some((generator, spec)) = state.draft.as_mut() else {
        return;
    };
    if spec.kind() == kind {
        return;
    }
    let Some(replacement) = GeneratorSpec::default_of_kind(kind) else {
        return;
    };
    *spec = replacement.clone();
    let generator = *generator;
    state.dirty = true;
    state.panel_revision = state.panel_revision.wrapping_add(1);
    drop(state);

    ctx.engine.send(Command::ParamEdit {
        generator,
        spec: replacement,
    });
    touch(revision);
}

#[component]
fn param_row(ctx: Ctx, revision: Signal<u32>, widget: Widget) -> NodeHandle {
    match widget.clone() {
        // -- slider + exact entry -------------------------------------------
        Widget::Float {
            label,
            path,
            value,
            range,
        } => {
            let track = Signal::new(value);
            let slider_path = path.clone();
            rsx! {
                div {
                    style: "display: flex; align-items: center; gap: 6px; padding: 2px 0; width: 100%; box-sizing: border-box; min-width: 0;",
                    Text { size: "xs", style: "width: 68px; flex-shrink: 0;", {label.clone()} }
                    div {
                        style: "flex: 1; min-width: 0;",
                        Slider {
                            min: range.min as f64,
                            max: range.max as f64,
                            step: slider_step(range.min, range.max),
                            value_signal: Some(track),
                            size: "xs",
                            onchange: move |v: f64| {
                                track.set(v);
                                commit(ctx, revision, &slider_path, Edit::Float(v));
                            },
                        }
                    }
                    div {
                        style: "width: 58px; flex-shrink: 0;",
                        TextInput {
                            size: "xs",
                            value_fn: move || format!("{:.4}", track.get()),
                            oninput: move |text: String| {
                                // Text entry is deliberately not bounded by the
                                // slider: a `FieldRange` says where the track
                                // ends, not what the param may be.
                                if let Ok(v) = text.trim().parse::<f64>() {
                                    track.set(v);
                                    commit(ctx, revision, &path, Edit::Float(v));
                                }
                            },
                        }
                    }
                }
            }
        }

        // -- integer stepper -------------------------------------------------
        Widget::Int {
            label,
            path,
            value,
            range,
        } => rsx! {
            div {
                style: "display: flex; align-items: center; gap: 6px; padding: 2px 0; width: 100%; box-sizing: border-box; min-width: 0;",
                Text { size: "xs", style: "width: 68px; flex-shrink: 0;", {label.clone()} }
                div {
                    style: "flex: 1; min-width: 0;",
                    NumberInput {
                        size: "xs",
                        value: value as f64,
                        min: range.min as f64,
                        max: range.max as f64,
                        step: 1.0,
                        decimal_scale: 0,
                        oninput: move |text: String| {
                            if let Ok(v) = text.trim().parse::<f64>() {
                                commit(ctx, revision, &path, Edit::Int(v as i64));
                            }
                        },
                    }
                }
            }
        },

        // -- seed + reroll ---------------------------------------------------
        Widget::Seed { label, path, value } => {
            let shown = Signal::new(value.to_string());
            let reroll_path = path.clone();
            rsx! {
                div {
                    style: "display: flex; align-items: center; gap: 6px; padding: 2px 0; width: 100%; box-sizing: border-box; min-width: 0;",
                    Text { size: "xs", style: "width: 68px; flex-shrink: 0;", {label.clone()} }
                    div {
                        style: "flex: 1; min-width: 0;",
                        TextInput {
                            size: "xs",
                            value_fn: move || shown.get(),
                            oninput: move |text: String| {
                                shown.set(text.clone());
                                if let Ok(v) = text.trim().parse::<u64>() {
                                    commit(ctx, revision, &path, Edit::Seed(v));
                                }
                            },
                        }
                    }
                    Button {
                        size: "xs",
                        variant: "default",
                        onclick: move || {
                            // splitmix64 of the current seed — deterministic,
                            // and no `rand` dependency (see `reroll_seed`).
                            let current = shown.get().trim().parse::<u64>().unwrap_or(0);
                            let next = runt_editor_core::reroll_seed(current);
                            shown.set(next.to_string());
                            commit(ctx, revision, &reroll_path, Edit::Seed(next));
                        },
                        "reroll"
                    }
                }
            }
        }

        // -- checkbox ---------------------------------------------------------
        Widget::Bool { label, path, value } => {
            let on = Signal::new(value);
            rsx! {
                div {
                    style: "padding: 2px 0;",
                    Checkbox {
                        label: label.clone(),
                        size: "xs",
                        checked_fn: move || on.get(),
                        onchange: move || {
                            let next = !on.get();
                            on.set(next);
                            commit(ctx, revision, &path, Edit::Bool(next));
                        },
                    }
                }
            }
        }

        // -- free text --------------------------------------------------------
        Widget::Text { label, path, value } => {
            let text = Signal::new(value);
            rsx! {
                div {
                    style: "display: flex; align-items: center; gap: 6px; padding: 2px 0; width: 100%; box-sizing: border-box; min-width: 0;",
                    Text { size: "xs", style: "width: 68px; flex-shrink: 0;", {label.clone()} }
                    div {
                        style: "flex: 1; min-width: 0;",
                        TextInput {
                            size: "xs",
                            value_fn: move || text.get(),
                            oninput: move |v: String| {
                                text.set(v.clone());
                                commit(ctx, revision, &path, Edit::Text(v));
                            },
                        }
                    }
                }
            }
        }

        // -- Vec2 / Vec3 as one row -------------------------------------------
        Widget::Vector {
            label, components, ..
        } => rsx! {
            div {
                style: "display: flex; align-items: center; gap: 4px; padding: 2px 0; width: 100%; box-sizing: border-box; min-width: 0;",
                Text { size: "xs", style: "width: 68px; flex-shrink: 0;", {label.clone()} }
                for component in components.clone() {
                    div {
                        key: component.path.display(),
                        style: "flex: 1; min-width: 0;",
                        {numeric_field(__scope, ctx, revision, component.path.clone(), component.value)}
                    }
                }
            }
        },

        // -- variant selector --------------------------------------------------
        Widget::Variant {
            label,
            path,
            selected,
            options,
            ..
        } => {
            let is_root = path.is_root();
            rsx! {
                div {
                    style: "display: flex; align-items: center; gap: 6px; padding: 4px 0;",
                    Text {
                        size: "xs",
                        weight: {if is_root { "bold" } else { "normal" }},
                        style: "width: 68px; flex-shrink: 0;",
                        {if label.is_empty() { "value".to_string() } else { label.clone() }}
                    }
                    div {
                        style: "flex: 1; min-width: 0;",
                        Select {
                            size: "xs",
                            value: selected.clone(),
                            data: options
                                .iter()
                                .map(|o| SelectOption::new(o.clone(), o.clone()))
                                .collect::<Vec<_>>(),
                            onchange: move |choice: String| {
                                if is_root {
                                    switch_generator(ctx, revision, &choice);
                                } else {
                                    commit(ctx, revision, &path, Edit::Variant(choice));
                                }
                            },
                        }
                    }
                }
            }
        }

        // -- headings ----------------------------------------------------------
        Widget::Group { label, .. } => rsx! {
            Text {
                size: "xs",
                color: "dimmed",
                weight: "bold",
                style: "padding-top: 6px;",
                {if label.is_empty() { "params".to_string() } else { label.clone() }}
            }
        },

        Widget::Unsupported {
            label, type_name, ..
        } => rsx! {
            Text {
                size: "xs",
                color: "dimmed",
                {format!("{label}: {type_name} (not editable yet)")}
            }
        },
    }
}

/// One bare numeric box, used for vector components.
#[component]
fn numeric_field(ctx: Ctx, revision: Signal<u32>, path: FieldPath, value: f64) -> NodeHandle {
    let shown = Signal::new(format!("{value:.4}"));
    rsx! {
        TextInput {
            size: "xs",
            value_fn: move || shown.get(),
            oninput: move |text: String| {
                shown.set(text.clone());
                if let Ok(v) = text.trim().parse::<f64>() {
                    commit(ctx, revision, &path, Edit::Float(v));
                }
            },
        }
    }
}

/// A slider step fine enough to feel continuous over the declared range without
/// producing values with fifteen significant figures.
fn slider_step(min: f32, max: f32) -> f64 {
    let span = (max - min).abs() as f64;
    if span <= 0.0 {
        return 0.01;
    }
    // ~500 stops across the track, rounded down to a power of ten.
    let raw = span / 500.0;
    10f64.powf(raw.log10().floor()).max(1e-6)
}

// ---------------------------------------------------------------------------
// transform
// ---------------------------------------------------------------------------

/// Which transform sub-vector a field edits.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Part {
    Translation,
    Rotation,
    Scale,
}

/// One row of the transform block.
#[derive(Clone, Debug, PartialEq)]
pub struct TransformRow {
    pub key: String,
    pub label: String,
    pub part: Part,
    pub values: [f64; 3],
}

fn transform_rows(ctx: Ctx, revision: Signal<u32>) -> Vec<TransformRow> {
    let _ = revision.get();
    let state = ctx.state.borrow();
    let Some((index, transform, euler)) = state.draft_transform else {
        return Vec::new();
    };
    let v = |v: glam::Vec3| [v.x as f64, v.y as f64, v.z as f64];
    vec![
        TransformRow {
            key: format!("{index}:t"),
            label: "translation".into(),
            part: Part::Translation,
            values: v(transform.translation),
        },
        TransformRow {
            key: format!("{index}:r"),
            label: "rotation deg".into(),
            part: Part::Rotation,
            values: v(euler),
        },
        TransformRow {
            key: format!("{index}:s"),
            label: "scale".into(),
            part: Part::Scale,
            values: v(transform.scale),
        },
    ]
}

/// The one hand-written panel (DESIGN §10 sanctions this case explicitly).
///
/// Placement is not a generator param — it lives on the *entity*, not the spec —
/// and its rotation is stored as a quaternion but edited as Euler degrees. That
/// conversion is a genuine interaction the reflection mapper cannot express, so
/// this block is written out.
///
/// Transform edits are sent immediately rather than debounced: nothing is
/// regenerated, so there is nothing to coalesce.
#[component]
fn transform_panel(ctx: Ctx, revision: Signal<u32>) -> NodeHandle {
    rsx! {
        div {
            if has_transform(ctx, revision) {
                div {
                    style: "padding-top: 10px; margin-top: 8px; \
                            border-top: 1px solid var(--rinch-color-default-border);",
                    Text { size: "xs", color: "dimmed", weight: "bold", "TRANSFORM" }

                    for row in transform_rows(ctx, revision) {
                        div {
                            key: row.key.clone(),
                            style: "display: flex; align-items: center; gap: 4px; padding: 2px 0; width: 100%; box-sizing: border-box; min-width: 0;",
                            Text {
                                size: "xs",
                                style: "width: 68px; flex-shrink: 0;",
                                {row.label.clone()}
                            }
                            {axis_fields(__scope, ctx, revision, row.part, row.values)}
                        }
                    }
                }
            }
        }
    }
}

/// The three numeric boxes of one transform row.
#[component]
fn axis_fields(ctx: Ctx, revision: Signal<u32>, part: Part, values: [f64; 3]) -> NodeHandle {
    rsx! {
        div {
            style: "display: flex; gap: 4px; flex: 1; min-width: 0; overflow: hidden;",
            for axis in 0usize..3 {
                div {
                    key: axis,
                    style: "flex: 1; min-width: 0;",
                    {transform_field(__scope, ctx, revision, part, axis, values[axis])}
                }
            }
        }
    }
}

#[component]
fn transform_field(
    ctx: Ctx,
    revision: Signal<u32>,
    part: Part,
    axis: usize,
    value: f64,
) -> NodeHandle {
    let shown = Signal::new(format!("{value:.4}"));
    rsx! {
        TextInput {
            size: "xs",
            value_fn: move || shown.get(),
            oninput: move |text: String| {
                shown.set(text.clone());
                let Ok(v) = text.trim().parse::<f32>() else {
                    return;
                };
                let mut state = ctx.state.borrow_mut();
                let Some((index, transform, euler)) = state.draft_transform.as_mut() else {
                    return;
                };
                match part {
                    Part::Translation => transform.translation[axis] = v,
                    Part::Scale => transform.scale[axis] = v,
                    Part::Rotation => {
                        euler[axis] = v;
                        // Write the *typed degrees* back, not a quaternion:
                        // `RotationDesc::Euler` is the authoring form, and
                        // round-tripping through a quaternion on every keystroke
                        // would make a 90 deg field drift as you edit its
                        // neighbour.
                        transform.rotation = runt_core::scene::RotationDesc::Euler(*euler);
                    }
                }
                let (index, transform) = (*index, *transform);
                state.dirty = true;
                drop(state);

                ctx.engine.send(Command::TransformEdit {
                    entity: index,
                    transform,
                });
                touch(revision);
            },
        }
    }
}
