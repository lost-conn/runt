//! The overlay that drives [`crate::inspect`] — `reflect` feature only.
//!
//! [`crate::tweak_panel`]'s sibling, and deliberately its twin: the same row
//! pitch, the same palette, the same four keys, the same tap-to-select /
//! drag-to-sweep fingers, the same [`PanelFont`] seam. A developer who has
//! learned one panel has learned the other, and the shared internals
//! ([`row_pitch`], [`row_at`](tweak_panel), the value bar) are shared precisely
//! so the two cannot drift apart by a pixel or a habit.
//!
//! What is different is what is *behind* the rows. The tweak panel edits a
//! registry of world paths; this one edits **the selected value** — a
//! [`Widget`] tree the game rebuilds each frame from whatever its selection is
//! (a generator spec, a material, a texture spec), hands to [`decide`], and
//! applies the resulting edit with [`inspect::apply`]. The panel owns no value
//! and knows no world: it is a pure function of (state, rows, input), which is
//! what makes it a table of unit tests below and a replay-safe overlay above.
//!
//! ```text
//! game        selection → inspect::build(&spec, name) → tree.rows()
//! Set::Input  decide(&mut panel, &rows, &input, viewport) → Some((path, edit))
//! game        inspect::apply(&mut spec, &path, &edit) → regenerate
//! Set::Ui     draw(…)
//! ```
//!
//! # Two widgets the tweak panel has no verb for
//!
//! - A [`Variant`](Widget::Variant) row cycles its enum's variants with
//!   Left/Right/Enter, exactly the way a choice row cycles — switching a
//!   variant *is* choosing, and the fields appearing under it are just rows.
//! - A [`Seed`](Widget::Seed) row rerolls on Enter. The next seed is
//!   [`inspect::reroll`] of the current one — a pure function, so a recorded
//!   run rerolls its way to the same meshes.
//!
//! # Anchored right, not left
//!
//! The tweak panel sits at the top-left; this one hugs the top-**right**
//! ([`panel_left`]), because a selection inspector and a tweak list are open
//! *together* in an editing session and one screen edge cannot hold both. That
//! is also why [`decide`] takes the [`Viewport`] the tweak panel's does not
//! need: a right edge is a place only the screen's width knows.
//!
//! Everything the tweak panel's docs say about determinism, key claiming and
//! the font seam holds here unchanged and is not restated.

use bevy_ecs::prelude::*;

use crate::ecs::Viewport;
use crate::input::{Input, Key};
use crate::inspect::{self, Edit, FieldPath, Widget};
use crate::tweak::TweakValue;
use crate::tweak_panel::{
    self, PanelFont, BACKDROP, BAR_WIDTH, COARSE, DIM, DRAG_SPAN, FINE, KEY_ACTIVATE, KEY_DEC,
    KEY_DOWN, KEY_INC, KEY_UP, MARGIN, PAD, RIM, SCALE, SELECTED, TEXT, WIDTH,
};
use crate::ui::UiBatch;

/// One flattened line of the tree: `(nesting depth, the widget)` — exactly what
/// [`Widget::rows`] returns. Rebuilt each frame like the tree it comes from.
pub type Row = (usize, Widget);

/// The panel's left edge: hugging the top-right corner, mirrored from the
/// tweak panel's top-left. Never past [`MARGIN`], so a viewport narrower than
/// the panel degrades to overlap rather than to a panel off the glass.
pub fn panel_left(viewport: Viewport) -> f32 {
    (viewport.size().x - MARGIN - WIDTH).max(MARGIN)
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// What the panel is doing between ticks.
///
/// Ordinary sim state, exactly as [`TweakPanel`](crate::tweak_panel::TweakPanel)
/// is: a resource so a replay carries it, `Default` so a game that never opens
/// it pays a handful of words.
#[derive(Resource, Clone, Debug, Default)]
pub struct InspectPanel {
    /// Whether the panel is up. The game owns the key that flips this — and
    /// owns clearing it when the selection goes away.
    pub open: bool,
    /// Where the cursor is, as a row index. Clamped into range every frame, so
    /// a variant switch shortening the tree moves it rather than breaking it.
    cursor: usize,
    /// The finger currently dragging a value: `(id, start x, the value the
    /// drag began from)` — absolute against its own origin, not accumulated,
    /// for the tweak panel's reason.
    drag: Option<(u64, f32, f64)>,
    /// The row a finger went down on.
    touch_row: Option<usize>,
    /// The first row the last drawn frame put on screen; see
    /// [`TweakPanel::scroll`](crate::tweak_panel::TweakPanel::scroll).
    scroll: usize,
}

impl InspectPanel {
    pub fn new() -> InspectPanel {
        InspectPanel::default()
    }

    /// Open or close. Closing forgets the drag, so a finger that was mid-sweep
    /// does not resume against a stale origin.
    pub fn set_open(&mut self, open: bool) {
        if self.open == open {
            return;
        }
        self.open = open;
        self.drag = None;
        self.touch_row = None;
    }

    pub fn toggle(&mut self) {
        self.set_open(!self.open);
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// The first row on screen as of the last [`draw`].
    pub fn scroll(&self) -> usize {
        self.scroll
    }
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

/// The whole state machine: move the cursor, or produce one edit.
///
/// The keys are the tweak panel's: Up/Down move, Left/Right step or cycle
/// (Shift coarse, Ctrl fine), Enter is the row's own verb — toggle a bool,
/// cycle a choice or a variant, **reroll a seed**. Group and vector header
/// rows are selectable and inert, like a folded thought.
///
/// The caller applies the returned edit with [`inspect::apply`] and rebuilds
/// the tree; this function never sees the value.
pub fn decide(
    panel: &mut InspectPanel,
    rows: &[Row],
    input: &Input,
    viewport: Viewport,
) -> Option<(FieldPath, Edit)> {
    if rows.is_empty() {
        panel.cursor = 0;
        return None;
    }
    // Clamp first: a variant switch since last tick may have shortened the
    // tree under the cursor.
    panel.cursor = panel.cursor.min(rows.len() - 1);

    if input.just_pressed(KEY_UP) {
        panel.cursor = (panel.cursor + rows.len() - 1) % rows.len();
    }
    if input.just_pressed(KEY_DOWN) {
        panel.cursor = (panel.cursor + 1) % rows.len();
    }

    // Touch before the keys' edits, so a finger and a keyboard cannot both
    // move the same value on one tick.
    if let Some(action) = touch(panel, rows, input, viewport) {
        return Some(action);
    }

    let (_, widget) = &rows[panel.cursor];
    if input.just_pressed(KEY_ACTIVATE) {
        return activate(widget);
    }

    let delta = i32::from(input.just_pressed(KEY_INC)) - i32::from(input.just_pressed(KEY_DEC));
    if delta == 0 {
        return None;
    }
    let scale = if input.held(Key::Shift) {
        COARSE
    } else if input.held(Key::Ctrl) {
        FINE
    } else {
        1.0
    };
    step(widget, delta, scale)
}

/// Enter on a row: bools flip, enums advance, seeds reroll, numbers do nothing
/// (the tweak panel's argument against a text entry holds here verbatim).
fn activate(widget: &Widget) -> Option<(FieldPath, Edit)> {
    match widget {
        Widget::Bool { path, value, .. } => Some((path.clone(), Edit::Bool(!value))),
        Widget::Seed { path, value, .. } => {
            Some((path.clone(), Edit::Seed(inspect::reroll(*value))))
        }
        Widget::Choice {
            path, selected, options, ..
        }
        | Widget::Variant {
            path, selected, options, ..
        } => cycle(path, selected, options, 1),
        _ => None,
    }
}

/// Nudge a row by `delta` steps.
fn step(widget: &Widget, delta: i32, scale: f32) -> Option<(FieldPath, Edit)> {
    match widget {
        Widget::Float {
            path, value, range, ..
        } => {
            let step = if range.step > 0.0 {
                range.step
            } else {
                // A full sweep in a hundred presses — the tweak panel's default.
                ((range.max - range.min) / 100.0).abs()
            };
            Some((
                path.clone(),
                Edit::Float(value + (delta as f32 * step * scale) as f64),
            ))
        }
        // Rounded away from zero, so a fine step on an integer still moves it
        // by one rather than by nothing — the tweak panel's rule, kept.
        Widget::Int { path, value, .. } => Some((
            path.clone(),
            Edit::Int(value + if delta >= 0 { 1 } else { -1 }),
        )),
        Widget::Choice {
            path, selected, options, ..
        }
        | Widget::Variant {
            path, selected, options, ..
        } => cycle(path, selected, options, delta),
        _ => None,
    }
}

/// The next variant over, by name, wrapping.
fn cycle(path: &FieldPath, selected: &str, options: &[String], delta: i32) -> Option<(FieldPath, Edit)> {
    let len = options.len();
    if len == 0 {
        return None;
    }
    let at = options.iter().position(|o| o == selected).unwrap_or(0);
    let next = if delta >= 0 {
        (at + 1) % len
    } else {
        (at + len - 1) % len
    };
    Some((path.clone(), Edit::Variant(options[next].clone())))
}

/// Tap to select, drag to sweep — the tweak panel's finger, on the other edge.
fn touch(
    panel: &mut InspectPanel,
    rows: &[Row],
    input: &Input,
    viewport: Viewport,
) -> Option<(FieldPath, Edit)> {
    if !viewport.is_known() {
        return None;
    }
    let left = panel_left(viewport);
    for touch in input.touches_started() {
        let Some(row) = tweak_panel::row_at(panel.scroll, rows.len(), touch.pos.y) else {
            continue;
        };
        if touch.pos.x < left || touch.pos.x > left + WIDTH {
            continue;
        }
        panel.cursor = row;
        panel.touch_row = Some(row);
        let start = match &rows[row].1 {
            Widget::Float { value, .. } => *value,
            Widget::Int { value, .. } => *value as f64,
            _ => 0.0,
        };
        panel.drag = Some((touch.id, touch.pos.x, start));
    }

    // A lift ends the drag; a tap that never moved has already selected.
    if let Some((id, _, _)) = panel.drag {
        if input.touch_ended(id).is_some() {
            panel.drag = None;
            panel.touch_row = None;
            return None;
        }
    }

    let (id, origin_x, origin_value) = panel.drag?;
    let touch = input.touch(id)?;
    let (_, widget) = rows.get(panel.touch_row?)?;
    let moved = touch.pos.x - origin_x;
    if moved.abs() < 1.0 {
        return None;
    }
    match widget {
        Widget::Float { path, range, .. } => {
            let span = (range.max - range.min) as f64;
            let want = origin_value + (moved / DRAG_SPAN) as f64 * span;
            Some((path.clone(), Edit::Float(want)))
        }
        Widget::Int { path, range, .. } => {
            let span = (range.max - range.min) as f64;
            let want = origin_value + (moved / DRAG_SPAN) as f64 * span;
            Some((path.clone(), Edit::Int(want.round() as i64)))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Draw
// ---------------------------------------------------------------------------

/// `Set::Ui`: paint the panel over the frame, after the game's own HUD.
///
/// `title` names what is selected — the generator's kind, an entity's name —
/// because a panel that says only INSPECT is a panel over the wrong thing
/// waiting to happen. Draws nothing when closed, when the viewport is unknown,
/// or when there are no rows (no selection is an absent panel, not an empty
/// box).
pub fn draw(
    panel: &mut InspectPanel,
    rows: &[Row],
    batch: &mut UiBatch,
    font: &dyn PanelFont,
    viewport: Viewport,
    title: &str,
) {
    if !panel.open || !viewport.is_known() || rows.is_empty() {
        return;
    }

    let pitch = tweak_panel::row_pitch();
    let left = panel_left(viewport);
    // Never taller than the screen — the tweak panel's window, verbatim.
    let max_rows =
        (((viewport.size().y - MARGIN * 2.0 - PAD * 2.0) / pitch).floor() - 1.0).max(1.0);
    let shown = rows.len().min(max_rows as usize);
    let first = panel
        .cursor
        .saturating_sub(shown.saturating_sub(1))
        .min(rows.len() - shown);
    // Remembered for the touch hit test, which runs a tick later against the
    // picture this frame is about to draw.
    panel.scroll = first;
    let height = PAD * 2.0 + pitch * (shown as f32 + 1.0);

    batch.solid([left, MARGIN, WIDTH, height], RIM);
    batch.solid(
        [left + 1.0, MARGIN + 1.0, WIDTH - 2.0, height - 2.0],
        BACKDROP,
    );

    let x = left + PAD;
    let mut y = MARGIN + PAD;
    font.text(batch, x, y, title, SCALE, DIM);
    y += pitch;

    for (offset, (depth, widget)) in rows[first..first + shown].iter().enumerate() {
        let index = first + offset;
        if index == panel.cursor {
            batch.solid([left + 1.0, y - 1.0, WIDTH - 2.0, pitch], SELECTED);
        }
        let indent = x + 8.0 * *depth as f32;
        let color = match widget {
            Widget::Group { .. } | Widget::Vector { .. } | Widget::Unsupported { .. } => DIM,
            _ => TEXT,
        };
        font.text(batch, indent, y, widget.label(), SCALE, color);

        if let Some(text) = value_text(widget) {
            let right = left + WIDTH - PAD;
            font.text(batch, right - font.width(&text, SCALE), y, &text, SCALE, color);
            match widget {
                Widget::Float { value, range, .. } => tweak_panel::value_bar(
                    batch,
                    range,
                    *value as f32,
                    right - BAR_WIDTH - 76.0,
                    y,
                    pitch,
                ),
                Widget::Int { value, range, .. } => tweak_panel::value_bar(
                    batch,
                    range,
                    *value as f32,
                    right - BAR_WIDTH - 76.0,
                    y,
                    pitch,
                ),
                _ => {}
            }
        }
        y += pitch;
    }
}

/// The value column's text, or `None` for a row that is only a heading.
///
/// Numbers borrow [`TweakValue::display`]'s trimming so the two panels print
/// `0.35`, not one `0.35` and one `0.350`. A seed shows its low 32 bits in hex
/// — the magnitude means nothing, the tail is what a human compares, and the
/// full width would eat the row.
fn value_text(widget: &Widget) -> Option<String> {
    match widget {
        Widget::Float { value, .. } => Some(TweakValue::Float(*value as f32).display()),
        Widget::Int { value, .. } => Some(TweakValue::Int(*value).display()),
        Widget::Bool { value, .. } => Some(TweakValue::Bool(*value).display()),
        Widget::Seed { value, .. } => Some(format!("{:08x}", *value as u32)),
        Widget::Choice { selected, .. } | Widget::Variant { selected, .. } => {
            Some(format!("< {selected} >"))
        }
        Widget::Unsupported { .. } => Some("-".to_string()),
        Widget::Vector { .. } | Widget::Group { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{InputEvent, TouchPhase};
    use crate::reflect::FieldRange;

    fn float(label: &str, value: f64) -> Widget {
        Widget::Float {
            label: label.to_string(),
            path: FieldPath::root().field(label),
            value,
            range: FieldRange::new(0.0, 10.0),
        }
    }

    /// A hand-built tree the way the walk would flatten one: a variant root,
    /// then its rows.
    fn rows() -> Vec<Row> {
        vec![
            (
                0,
                Widget::Variant {
                    label: "generator".into(),
                    path: FieldPath::root(),
                    selected: "Torus".into(),
                    options: vec!["Torus".into(), "Cube".into(), "Plane".into()],
                    fields: Vec::new(),
                },
            ),
            (1, float("radius", 2.0)),
            (
                1,
                Widget::Bool {
                    label: "lit".into(),
                    path: FieldPath::root().field("lit"),
                    value: false,
                },
            ),
            (
                1,
                Widget::Seed {
                    label: "seed".into(),
                    path: FieldPath::root().field("seed"),
                    value: 42,
                },
            ),
            (
                1,
                Widget::Group {
                    label: "nested".into(),
                    path: FieldPath::root().field("nested"),
                    fields: Vec::new(),
                },
            ),
        ]
    }

    fn view() -> Viewport {
        Viewport::new(1280, 720)
    }

    /// One tick of input built from events, the way `Sim::tick` builds it.
    fn tick(events: impl IntoIterator<Item = InputEvent>) -> Input {
        let mut input = Input::new();
        input.begin_tick(events);
        input
    }

    #[test]
    fn the_cursor_wraps_and_survives_the_tree_shrinking_under_it() {
        let mut panel = InspectPanel::new();
        panel.open = true;
        let rows = rows();
        decide(&mut panel, &rows, &tick([InputEvent::KeyDown(Key::Up)]), view());
        assert_eq!(panel.cursor(), 4, "up from the top wraps to the bottom");

        // A variant switch shortened the tree; the cursor lands on the end.
        let shorter = &rows[..2];
        decide(&mut panel, shorter, &tick([]), view());
        assert_eq!(panel.cursor(), 1);

        decide(&mut panel, &[], &tick([]), view());
        assert_eq!(panel.cursor(), 0, "no rows is not a panic");
    }

    #[test]
    fn arrows_step_a_float_and_the_modifiers_scale_the_step() {
        let mut panel = InspectPanel::new();
        panel.open = true;
        let rows = rows();
        panel.cursor = 1; // radius

        let plus = decide(&mut panel, &rows, &tick([InputEvent::KeyDown(Key::Right)]), view());
        let Some((path, Edit::Float(v))) = plus else {
            panic!("right produced no edit");
        };
        assert_eq!(path, FieldPath::root().field("radius"));
        // Approximate: the step is computed in `f32` (the range's width) and
        // widened, so exact `f64` equality would test the rounding, not the step.
        assert!(
            (v - 2.1).abs() < 1e-6,
            "{v} is not a hundredth of the range up"
        );

        let mut held = Input::new();
        held.begin_tick([InputEvent::KeyDown(Key::Shift)]);
        held.begin_tick([InputEvent::KeyDown(Key::Left)]);
        let Some((_, Edit::Float(v))) = decide(&mut panel, &rows, &held, view()) else {
            panic!("shift+left produced no edit");
        };
        assert!((v - 1.0).abs() < 1e-5, "{v} is not ten steps down");
    }

    #[test]
    fn enter_toggles_a_bool_and_leaves_a_number_alone() {
        let mut panel = InspectPanel::new();
        panel.open = true;
        let rows = rows();

        panel.cursor = 2; // lit
        assert_eq!(
            decide(&mut panel, &rows, &tick([InputEvent::KeyDown(Key::Enter)]), view()),
            Some((FieldPath::root().field("lit"), Edit::Bool(true)))
        );

        panel.cursor = 1; // radius — a float has no "activate"
        assert_eq!(
            decide(&mut panel, &rows, &tick([InputEvent::KeyDown(Key::Enter)]), view()),
            None
        );
    }

    #[test]
    fn a_variant_row_cycles_its_options_in_both_directions() {
        let mut panel = InspectPanel::new();
        panel.open = true;
        let rows = rows();
        panel.cursor = 0; // the generator select

        assert_eq!(
            decide(&mut panel, &rows, &tick([InputEvent::KeyDown(Key::Right)]), view()),
            Some((FieldPath::root(), Edit::Variant("Cube".into())))
        );
        assert_eq!(
            decide(&mut panel, &rows, &tick([InputEvent::KeyDown(Key::Left)]), view()),
            Some((FieldPath::root(), Edit::Variant("Plane".into()))),
            "left from the first option wraps to the last"
        );
        assert_eq!(
            decide(&mut panel, &rows, &tick([InputEvent::KeyDown(Key::Enter)]), view()),
            Some((FieldPath::root(), Edit::Variant("Cube".into()))),
            "enter is the same verb as right"
        );
    }

    #[test]
    fn enter_on_a_seed_rerolls_it_deterministically() {
        let mut panel = InspectPanel::new();
        panel.open = true;
        let rows = rows();
        panel.cursor = 3; // seed

        let expected = inspect::reroll(42);
        assert_ne!(expected, 42);
        assert_eq!(
            decide(&mut panel, &rows, &tick([InputEvent::KeyDown(Key::Enter)]), view()),
            Some((FieldPath::root().field("seed"), Edit::Seed(expected)))
        );
    }

    #[test]
    fn a_group_row_is_selectable_and_inert() {
        let mut panel = InspectPanel::new();
        panel.open = true;
        let rows = rows();
        panel.cursor = 4; // nested
        for key in [Key::Enter, Key::Right, Key::Left] {
            assert_eq!(
                decide(&mut panel, &rows, &tick([InputEvent::KeyDown(key)]), view()),
                None
            );
        }
    }

    #[test]
    fn a_tap_selects_the_row_and_a_drag_sweeps_its_value() {
        let mut panel = InspectPanel::new();
        panel.open = true;
        let rows = rows();
        let viewport = view();

        // Row 1 (`radius`) — the second line, one pitch below the title — at
        // the panel's right-anchored x.
        let pitch = tweak_panel::row_pitch();
        let y = MARGIN + PAD + pitch * 2.0 + 1.0;
        let x = panel_left(viewport) + 20.0;

        let mut live = Input::new();
        live.begin_tick([InputEvent::Touch {
            id: 7,
            phase: TouchPhase::Started,
            x,
            y,
        }]);
        assert_eq!(
            decide(&mut panel, &rows, &live.clone(), viewport),
            None,
            "a tap only selects"
        );
        assert_eq!(panel.cursor(), 1);

        // Dragging right by a quarter of the sweep span moves the value a
        // quarter of its range, measured from where the finger landed.
        live.begin_tick([InputEvent::Touch {
            id: 7,
            phase: TouchPhase::Moved,
            x: x + DRAG_SPAN * 0.25,
            y,
        }]);
        let Some((path, Edit::Float(v))) = decide(&mut panel, &rows, &live.clone(), viewport)
        else {
            panic!("the drag produced no edit");
        };
        assert_eq!(path, FieldPath::root().field("radius"));
        assert!((v - 4.5).abs() < 1e-4, "{v}");

        // A lift ends it.
        live.begin_tick([InputEvent::Touch {
            id: 7,
            phase: TouchPhase::Ended,
            x: 400.0,
            y,
        }]);
        assert_eq!(decide(&mut panel, &rows, &live.clone(), viewport), None);
    }

    #[test]
    fn a_touch_at_the_tweak_panels_edge_is_not_this_panels() {
        let mut panel = InspectPanel::new();
        panel.open = true;
        let rows = rows();
        let y = MARGIN + PAD + tweak_panel::row_pitch() * 2.0 + 1.0;
        // The top-left corner is the tweak panel's turf.
        decide(
            &mut panel,
            &rows,
            &tick([InputEvent::Touch {
                id: 1,
                phase: TouchPhase::Started,
                x: MARGIN + 20.0,
                y,
            }]),
            view(),
        );
        assert_eq!(panel.cursor(), 0, "the cursor moved for somebody else's tap");
    }

    /// The stub the drawing test borrows from the tweak panel's suite: one quad
    /// per character.
    struct BlockFont;

    impl PanelFont for BlockFont {
        fn width(&self, text: &str, scale: f32) -> f32 {
            text.chars().count() as f32 * 8.0 * scale
        }
        fn text(
            &self,
            batch: &mut UiBatch,
            x: f32,
            y: f32,
            text: &str,
            scale: f32,
            color: [f32; 4],
        ) {
            for (i, _) in text.chars().enumerate() {
                batch.solid(
                    [x + i as f32 * 8.0 * scale, y, 7.0 * scale, 8.0 * scale],
                    color,
                );
            }
        }
    }

    #[test]
    fn a_closed_panel_and_an_empty_selection_both_draw_nothing() {
        let mut panel = InspectPanel::new();
        let mut batch = UiBatch::new();
        draw(&mut panel, &rows(), &mut batch, &BlockFont, view(), "TORUS");
        assert!(batch.is_empty(), "a closed panel is not a pass");

        panel.open = true;
        draw(&mut panel, &[], &mut batch, &BlockFont, view(), "TORUS");
        assert!(batch.is_empty(), "no selection is an absent panel");

        draw(&mut panel, &rows(), &mut batch, &BlockFont, view(), "TORUS");
        assert!(!batch.is_empty(), "an open panel with rows draws");
    }
}
