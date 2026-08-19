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
//! Set::Input  decide(&mut panel, &rows, nav, &input, viewport) → Some((path, edit))
//! game        inspect::apply(&mut spec, &path, &edit) → regenerate
//! Set::Ui     draw(…)
//! ```
//!
//! `nav` is a [`PanelNav`] — what the device said, with the device forgotten.
//! [`PanelNav::from_keys`] is the keyboard this panel was written against and
//! is what a caller with nothing else to say passes.
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
use crate::input::Input;
use crate::inspect::{self, Edit, FieldPath, Widget};
use crate::tweak::TweakValue;
use crate::tweak_panel::{
    self, PanelFont, PanelNav, BACKDROP, BAR_WIDTH, DIM, DRAG_SPAN, MARGIN, PAD, RIM, SCALE,
    SELECTED, SWEEP_SECONDS, TEXT, WIDTH,
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
    /// Fractional units an analog sweep has accumulated on an **integer** row
    /// but not yet spent.
    ///
    /// A float row needs none of this — it takes whatever fraction the tick
    /// produced. An integer one cannot: a slow sweep is a few thousandths of a
    /// unit per tick, and rounding each tick's contribution on its own would
    /// round every one of them to nothing, so the row would simply refuse to
    /// move below some stick deflection. Carried here and spent a whole unit at
    /// a time instead. Cleared whenever the sweep stops or the cursor moves, so
    /// it can never leak into a row it was not earned on.
    sweep_carry: f64,
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
/// The vocabulary is the tweak panel's: up/down move, dec/inc step or cycle
/// (scaled coarse or fine), activate is the row's own verb — toggle a bool,
/// cycle a choice or a variant, **reroll a seed**. Group, vector and colour
/// header rows are selectable and inert, like a folded thought: a swatch is a
/// preview, not a picker, and inventing "Enter opens a colour dialog" here
/// would be exactly the kind of control this panel has never had one of.
///
/// `nav` is what the device said and `input` is the finger — see [`PanelNav`]
/// on why those are two arguments. [`PanelNav::fold`] and
/// [`PanelNav::reset`] have no meaning here and are ignored: this panel has no
/// groups to fold and no authored value to go back to.
///
/// The caller applies the returned edit with [`inspect::apply`] and rebuilds
/// the tree; this function never sees the value.
pub fn decide(
    panel: &mut InspectPanel,
    rows: &[Row],
    nav: PanelNav,
    input: &Input,
    dt: f32,
    viewport: Viewport,
) -> Option<(FieldPath, Edit)> {
    if rows.is_empty() {
        panel.cursor = 0;
        return None;
    }
    // Clamp first: a variant switch since last tick may have shortened the
    // tree under the cursor.
    panel.cursor = panel.cursor.min(rows.len() - 1);

    let was = panel.cursor;
    if nav.up {
        panel.cursor = (panel.cursor + rows.len() - 1) % rows.len();
    }
    if nav.down {
        panel.cursor = (panel.cursor + 1) % rows.len();
    }
    // A sweep is about the row it was aimed at; moving off one abandons
    // whatever fraction it had earned there.
    if panel.cursor != was || nav.sweep == 0.0 {
        panel.sweep_carry = 0.0;
    }

    // Touch before the keys' edits, so a finger and a keyboard cannot both
    // move the same value on one tick.
    if let Some(action) = touch(panel, rows, input, viewport) {
        return Some(action);
    }

    let (_, widget) = &rows[panel.cursor];
    if nav.activate {
        return activate(widget);
    }

    let delta = i32::from(nav.inc) - i32::from(nav.dec);
    if delta != 0 {
        return step(widget, delta, nav.scale);
    }
    if nav.sweep != 0.0 {
        return sweep(&mut panel.sweep_carry, widget, nav.sweep, dt);
    }
    None
}

/// One tick of an analog sweep: a rate, integrated.
///
/// The rate is a fraction of the row's own declared range per
/// [`SWEEP_SECONDS`], so a colour channel and a distance in metres both feel
/// the same under the same stick deflection without either being tuned by hand.
///
/// Only numbers sweep. A bool, a choice, a variant and a seed are all *steps* —
/// there is no half of the way from one enum variant to the next — and a stick
/// pushed at one of them does nothing rather than cycling continuously, which
/// would make holding a direction spin through the options unusably.
fn sweep(
    carry: &mut f64,
    widget: &Widget,
    amount: f32,
    dt: f32,
) -> Option<(FieldPath, Edit)> {
    match widget {
        Widget::Float {
            path, value, range, ..
        } => {
            let span = (range.max - range.min) as f64;
            let delta = amount as f64 * span / SWEEP_SECONDS as f64 * dt as f64;
            (delta != 0.0).then(|| (path.clone(), Edit::Float(value + delta)))
        }
        Widget::Int {
            path, value, range, ..
        } => {
            let span = (range.max - range.min) as f64;
            *carry += amount as f64 * span / SWEEP_SECONDS as f64 * dt as f64;
            let whole = carry.trunc();
            if whole == 0.0 {
                return None;
            }
            *carry -= whole;
            Some((path.clone(), Edit::Int(value + whole as i64)))
        }
        _ => None,
    }
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
///
/// # `focused`, and why the cursor can be invisible
///
/// Open and *being navigated* are two different states, and a host with more
/// than one panel up has to be able to say so: [`decide`] already answers a
/// [`PanelNav::NONE`] with nothing, so the panel that is not holding the verbs
/// is inert — and a highlighted row on an inert panel is a claim on ↑↓ that
/// the next press will not honour. Pass `false` and the row bar is simply not
/// drawn; the list stays legible, and the cursor comes back where it was the
/// moment the verbs do.
///
/// A host that only ever shows one panel passes `true` and never thinks about
/// it. Which panel has them is the host's question — the arbitration is a
/// property of a screen, not of a widget — so this takes the answer rather
/// than keeping a bit it could not maintain.
pub fn draw(
    panel: &mut InspectPanel,
    rows: &[Row],
    batch: &mut UiBatch,
    font: &dyn PanelFont,
    viewport: Viewport,
    title: &str,
    focused: bool,
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
        // The cursor bar is drawn only while this panel is the one being
        // navigated — see the `focused` argument.
        if focused && index == panel.cursor {
            batch.solid([left + 1.0, y - 1.0, WIDTH - 2.0, pitch], SELECTED);
        }
        let indent = x + 8.0 * *depth as f32;
        // A colour header is a heading like `Group`/`Vector`: the swatch is
        // the row's content, the label is only naming it, so it dims the same
        // way "nested" or "size" do.
        let color = match widget {
            Widget::Group { .. } | Widget::Vector { .. } | Widget::Color { .. } | Widget::Unsupported { .. } => DIM,
            _ => TEXT,
        };
        font.text(batch, indent, y, widget.label(), SCALE, color);

        let right = left + WIDTH - PAD;
        if let Some(text) = value_text(widget) {
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
        } else if let Widget::Color { value, .. } = widget {
            // No text to leave room for, unlike a numeric row's bar — the
            // swatch fills the whole value column, flush with the right edge
            // every other row's text is flush with.
            color_swatch(batch, *value, right - BAR_WIDTH, y, pitch);
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
        // `Color` draws a swatch instead of a value column, the way `Vector`
        // and `Group` draw nothing there: it is a heading with its content
        // below it, not a leaf with a reading next to it.
        Widget::Vector { .. } | Widget::Group { .. } | Widget::Color { .. } => None,
    }
}

/// Clamp one channel for **display only**.
///
/// Channel values are never clamped in the data — see
/// [`inspect::Widget::Color`]'s doc comment for why an out-of-range channel
/// is a real, meaningful number in this renderer (no emission channel, so a
/// colour past white is how "blown out" is spelled) — but a quad's fill
/// colour has to be *some* finite `[0, 1]` number or it paints noise instead
/// of a swatch. NaN clamps to `0.0` rather than propagating: `f32::clamp`
/// itself returns NaN unchanged (it is unordered against both bounds), and a
/// black square is a far better failure than a quad whose colour silently
/// isn't one.
fn display_channel(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// The preview beside a [`Color`](Widget::Color) row's label.
///
/// `BAR_WIDTH` wide, the same box a numeric row's
/// [`tweak_panel::value_bar`] fills, so a colour and a number agree about
/// where the value column starts — the same reuse-not-invent rule the rest
/// of this module's spacing follows.
///
/// Drawn with a one-pixel [`RIM`] border rather than flush against
/// [`BACKDROP`]: the backdrop is already very dark
/// (`[0.05, 0.05, 0.07, 0.88]`), so a near-black *authored* colour with no
/// border would draw as a gap in the panel — exactly the "reads as a hole"
/// failure a swatch must not have. A chequer behind it would say the same
/// thing with a second texture and a shader branch a debug overlay has no
/// other reason to own; one extra quad in a colour the panel's own outline
/// already uses is simpler and gives the same read, "this is a filled thing."
fn color_swatch(batch: &mut UiBatch, rgb: [f32; 3], x: f32, y: f32, pitch: f32) {
    let h = (pitch - 6.0).max(2.0);
    let top = y + (pitch - h) * 0.5 - 1.0;
    batch.solid([x - 1.0, top - 1.0, BAR_WIDTH + 2.0, h + 2.0], RIM);
    let fill = [
        display_channel(rgb[0]),
        display_channel(rgb[1]),
        display_channel(rgb[2]),
        1.0,
    ];
    batch.solid([x, top, BAR_WIDTH, h], fill);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{InputEvent, Key, TouchPhase};
    use crate::inspect::VectorComponent;
    use crate::reflect::FieldRange;

    fn float(label: &str, value: f64) -> Widget {
        Widget::Float {
            label: label.to_string(),
            path: FieldPath::root().field(label),
            value,
            range: FieldRange::new(0.0, 10.0),
        }
    }

    /// A hand-built `Color`, the way a game builds one — see
    /// [`inspect::Widget::Color`]'s doc comment for why the walk never does.
    fn color(label: &str, rgb: [f32; 3]) -> Widget {
        let path = FieldPath::root().field(label);
        let component = |name: &str, v: f32| VectorComponent {
            label: name.to_string(),
            path: path.field(name),
            value: v as f64,
        };
        Widget::Color {
            label: label.to_string(),
            path: path.clone(),
            value: rgb,
            components: vec![component("r", rgb[0]), component("g", rgb[1]), component("b", rgb[2])],
            range: FieldRange::new(0.0, 1.0),
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

    /// One tick at the engine's fixed rate — what `drive` reads off `FixedTick`
    /// in a real world. Only the sweep looks at it; every edge test is
    /// unaffected by its value.
    const TEST_DT: f32 = 1.0 / 60.0;

    /// One tick of input built from events, the way `Sim::tick` builds it.
    fn tick(events: impl IntoIterator<Item = InputEvent>) -> Input {
        let mut input = Input::new();
        input.begin_tick(events);
        input
    }

    /// [`decide`] as every table below drives it — the keyboard, through
    /// [`PanelNav::from_keys`]. The tweak panel's own helper carries the
    /// argument for why this indirection is worth having.
    fn decide_keys(
        panel: &mut InspectPanel,
        rows: &[Row],
        input: &Input,
        viewport: Viewport,
    ) -> Option<(FieldPath, Edit)> {
        decide(panel, rows, PanelNav::from_keys(input), input, TEST_DT, viewport)
    }

    #[test]
    fn the_cursor_wraps_and_survives_the_tree_shrinking_under_it() {
        let mut panel = InspectPanel::new();
        panel.open = true;
        let rows = rows();
        decide_keys(&mut panel, &rows, &tick([InputEvent::KeyDown(Key::Up)]), view());
        assert_eq!(panel.cursor(), 4, "up from the top wraps to the bottom");

        // A variant switch shortened the tree; the cursor lands on the end.
        let shorter = &rows[..2];
        decide_keys(&mut panel, shorter, &tick([]), view());
        assert_eq!(panel.cursor(), 1);

        decide_keys(&mut panel, &[], &tick([]), view());
        assert_eq!(panel.cursor(), 0, "no rows is not a panic");
    }

    #[test]
    fn arrows_step_a_float_and_the_modifiers_scale_the_step() {
        let mut panel = InspectPanel::new();
        panel.open = true;
        let rows = rows();
        panel.cursor = 1; // radius

        let plus = decide_keys(&mut panel, &rows, &tick([InputEvent::KeyDown(Key::Right)]), view());
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
        let Some((_, Edit::Float(v))) = decide_keys(&mut panel, &rows, &held, view()) else {
            panic!("shift+left produced no edit");
        };
        assert!((v - 1.0).abs() < 1e-5, "{v} is not ten steps down");
    }

    /// The seam's whole point: a device that is not a keyboard.
    ///
    /// Nothing here presses a key — the [`Input`] handed over is a silent tick,
    /// present only because the finger's half of `decide` still wants one.
    #[test]
    fn a_nav_built_by_hand_drives_the_panel_with_no_keyboard_at_all() {
        let mut panel = InspectPanel::new();
        panel.open = true;
        let rows = rows();
        let quiet = tick([]);

        decide(
            &mut panel,
            &rows,
            PanelNav {
                down: true,
                ..PanelNav::NONE
            },
            &quiet,
            TEST_DT,
            view(),
        );
        assert_eq!(panel.cursor(), 1, "radius, one row past the variant");

        let edit = decide(
            &mut panel,
            &rows,
            PanelNav {
                inc: true,
                ..PanelNav::NONE
            },
            &quiet,
            TEST_DT,
            view(),
        );
        let Some((path, Edit::Float(v))) = edit else {
            panic!("a hand-built inc produced no edit");
        };
        assert_eq!(path, FieldPath::root().field("radius"));
        assert!((v - 2.1).abs() < 1e-6, "{v} is not one step up");

        // `fold` and `reset` are the tweak panel's; this one has no verb for
        // either and must not invent one.
        assert_eq!(
            decide(
                &mut panel,
                &rows,
                PanelNav {
                    fold: true,
                    reset: true,
                    ..PanelNav::NONE
                },
                &quiet,
                TEST_DT,
                view(),
            ),
            None
        );
    }

    #[test]
    fn enter_toggles_a_bool_and_leaves_a_number_alone() {
        let mut panel = InspectPanel::new();
        panel.open = true;
        let rows = rows();

        panel.cursor = 2; // lit
        assert_eq!(
            decide_keys(&mut panel, &rows, &tick([InputEvent::KeyDown(Key::Enter)]), view()),
            Some((FieldPath::root().field("lit"), Edit::Bool(true)))
        );

        panel.cursor = 1; // radius — a float has no "activate"
        assert_eq!(
            decide_keys(&mut panel, &rows, &tick([InputEvent::KeyDown(Key::Enter)]), view()),
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
            decide_keys(&mut panel, &rows, &tick([InputEvent::KeyDown(Key::Right)]), view()),
            Some((FieldPath::root(), Edit::Variant("Cube".into())))
        );
        assert_eq!(
            decide_keys(&mut panel, &rows, &tick([InputEvent::KeyDown(Key::Left)]), view()),
            Some((FieldPath::root(), Edit::Variant("Plane".into()))),
            "left from the first option wraps to the last"
        );
        assert_eq!(
            decide_keys(&mut panel, &rows, &tick([InputEvent::KeyDown(Key::Enter)]), view()),
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
            decide_keys(&mut panel, &rows, &tick([InputEvent::KeyDown(Key::Enter)]), view()),
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
                decide_keys(&mut panel, &rows, &tick([InputEvent::KeyDown(key)]), view()),
                None
            );
        }
    }

    /// Mirrors [`a_group_row_is_selectable_and_inert`]: a swatch is a preview,
    /// not a control, so its header has no verb either.
    #[test]
    fn a_color_row_is_selectable_and_inert() {
        let mut panel = InspectPanel::new();
        panel.open = true;
        let rows = vec![(0, color("tint", [0.2, 0.4, 0.8]))];
        panel.cursor = 0;
        for key in [Key::Enter, Key::Right, Key::Left] {
            assert_eq!(
                decide_keys(&mut panel, &rows, &tick([InputEvent::KeyDown(key)]), view()),
                None
            );
        }
    }

    /// `Widget::rows` flattens a `Color` exactly the way it flattens a
    /// `Vector` (they share the match arm — `inspect.rs`'s `push_rows`); this
    /// pins the panel's own view of that: the header at its own depth, then
    /// one channel row per component, in order, one level deeper.
    #[test]
    fn a_color_flattens_into_its_channel_rows_in_order() {
        let rows = color("tint", [0.2, 0.4, 0.8]).rows();
        assert!(matches!(&rows[0], (0, Widget::Color { .. })), "the header comes first");
        let channels: Vec<(usize, &str)> = rows[1..].iter().map(|(depth, w)| (*depth, w.label())).collect();
        assert_eq!(
            channels,
            vec![(1, "r"), (1, "g"), (1, "b")],
            "r, g, b — in order, one below the header",
        );
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
            decide_keys(&mut panel, &rows, &live.clone(), viewport),
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
        let Some((path, Edit::Float(v))) = decide_keys(&mut panel, &rows, &live.clone(), viewport)
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
        assert_eq!(decide_keys(&mut panel, &rows, &live.clone(), viewport), None);
    }

    #[test]
    fn a_touch_at_the_tweak_panels_edge_is_not_this_panels() {
        let mut panel = InspectPanel::new();
        panel.open = true;
        let rows = rows();
        let y = MARGIN + PAD + tweak_panel::row_pitch() * 2.0 + 1.0;
        // The top-left corner is the tweak panel's turf.
        decide_keys(
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
        draw(&mut panel, &rows(), &mut batch, &BlockFont, view(), "TORUS", true);
        assert!(batch.is_empty(), "a closed panel is not a pass");

        panel.open = true;
        draw(&mut panel, &[], &mut batch, &BlockFont, view(), "TORUS", true);
        assert!(batch.is_empty(), "no selection is an absent panel");

        draw(&mut panel, &rows(), &mut batch, &BlockFont, view(), "TORUS", true);
        assert!(!batch.is_empty(), "an open panel with rows draws");
    }

    /// `inspect::apply` never clamps a channel (the module docs' advisory-range
    /// argument), so a `Color` the panel is asked to draw can carry a channel
    /// past `[0, 1]` or, if a drag or a scene file ever produced one, NaN. The
    /// draw must turn that into a swatch, not a panic and not a quad the GPU
    /// would choke on.
    #[test]
    fn an_out_of_range_or_nan_channel_draws_a_finite_swatch_not_a_panic() {
        let mut panel = InspectPanel::new();
        panel.open = true;
        let rows = vec![(0, color("blown", [5.0, -2.0, f32::NAN]))];
        let mut batch = UiBatch::new();
        draw(&mut panel, &rows, &mut batch, &BlockFont, view(), "MATERIAL", true);
        assert!(!batch.is_empty(), "the rim and the swatch still draw");
        for quad in &batch.quads {
            assert!(
                quad.rect.iter().all(|v| v.is_finite()),
                "geometry must stay finite even off-range: {:?}",
                quad.rect
            );
            assert!(
                quad.color.iter().all(|v| v.is_finite()),
                "a NaN channel must not reach the GPU colour: {:?}",
                quad.color
            );
        }
    }
}
