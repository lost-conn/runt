//! The debug overlay that drives [`crate::tweak`] — `reflect` feature only.
//!
//! A list of rows on the left of the screen, a cursor, and four keys. That is
//! deliberately the whole of it. This is a **debug overlay, not a product**:
//! there is no tree widget, no layout engine, no scrollbar, no text entry and no
//! theme. What it has to be is legible at a glance and impossible to get lost
//! in, because the moment it is more interesting than the thing it is tuning it
//! has failed.
//!
//! ```text
//! ┌──────────────────────────────┐
//! │ TWEAKS           (3 changed) │
//! │ v sky                        │
//! │   clouds        ▓▓▓░░░  0.35 │
//! │   sun           ▓░░░░░  0.02 │
//! │ > camera                     │
//! │ v render_scale               │
//! │   0             ▓▓▓▓░░  0.75 │
//! └──────────────────────────────┘
//! ```
//!
//! # It is engine-side, and it draws with the game's font
//!
//! The panel logic lives here so every runt game gets it for the price of one
//! system, but the engine has **no typeface**: [`UiBatch`] is quads and a glyph
//! atlas is game-authored content ([`UiAtlasImage`](crate::ui::UiAtlasImage)'s
//! whole reason to exist). So the glyphs come in through [`PanelFont`], a
//! two-method trait.
//!
//! Since [`crate::font`] landed there is nothing left for a game to implement:
//! every [`BitmapFont`](crate::font::BitmapFont) *is* a `PanelFont`, so a game
//! hands the panel the same font it draws its HUD with and the trait is only
//! still a trait so that a game with some other idea of text can still be one.
//!
//! [`row_pitch`] is where that lands geometrically: `SCALE` is in
//! [`font::UNIT`](crate::font::UNIT) cells, so a font baked *for* that scale
//! draws texel-for-pixel and the pitch is the same number it has always been.
//!
//! # The seam
//!
//! Two calls, at the two ends of a tick, because the panel does two unrelated
//! things:
//!
//! ```text
//! Set::Input   panel_input(world)   consume this tick's keys → edit the world
//! Set::Ui      draw(…)              …and paint what it now says
//! ```
//!
//! `panel_input` is an exclusive system (`&mut World`) because an edit writes
//! into an arbitrary resource by path — there is no query that expresses that.
//! It costs a `Vec` of rows on the ticks the panel is **open** and one boolean
//! read on the ticks it is not.
//!
//! # Input, and what it does not touch
//!
//! The panel reads [`Input`] and never writes it. That is the determinism seam,
//! and it is worth stating in the negative: nothing here can change what a
//! [trace](crate::trace) records, because a trace is re-derived from `Input` and
//! `Input` is untouched. A run recorded with the panel open replays with the
//! panel open, opening on the same tick, moving the same cursor and landing the
//! same edits at the same ticks — the panel's whole state is a pure function of
//! (recorded input, its own previous state), which is the same thing every other
//! system in a fixed tick is.
//!
//! What the panel does *not* do is stop the game reading the same keys. Arrows
//! are a camera in most games. Claiming them is the **host's** job and it is one
//! line — resolve the game's actions against an empty input while
//! [`TweakPanel::open`] — for the same reason the engine does not decide what
//! Escape means. [`TweakPanel::open`] is public so a game can ask.

use bevy_ecs::prelude::*;

use crate::ecs::Viewport;
use crate::input::{Input, Key};
use crate::tweak::{self, TweakField, TweakValue};
use crate::ui::UiBatch;

// ---------------------------------------------------------------------------
// The font seam
// ---------------------------------------------------------------------------

/// How the panel draws text: the game's own atlas, behind two methods.
///
/// Both are in the same coordinate space [`UiQuad`](crate::ui::UiQuad) is —
/// logical pixels, top-left origin, +Y down — and both take a `scale` in
/// [`font::UNIT`](crate::font::UNIT) cells, so the panel matches whatever the
/// rest of the game's HUD looks like.
///
/// [`crate::font::BitmapFont`] implements this, which is normally all a game
/// needs to know: `draw(…, &font, …)`.
///
/// Two methods, not three: the panel deliberately does **not** ask the font for
/// a line height. The row pitch is [`row_pitch`], a constant, because the touch
/// hit test has no font in hand, and a hit test that disagrees with the layout
/// by a pixel is a worse bug than a row of the wrong height.
pub trait PanelFont {
    /// How wide `text` will be at `scale`, in logical pixels. Used for
    /// right-aligning the value column, so an answer that disagrees with
    /// [`text`](PanelFont::text) only makes the panel ragged.
    fn width(&self, text: &str, scale: f32) -> f32;

    /// Push `text`'s quads into `batch` with its top-left at `(x, y)`.
    fn text(&self, batch: &mut UiBatch, x: f32, y: f32, text: &str, scale: f32, color: [f32; 4]);
}

// ---------------------------------------------------------------------------
// Look
// ---------------------------------------------------------------------------

/// The panel's palette. Deliberately not configurable: a debug overlay that can
/// be themed is a debug overlay somebody has spent an afternoon on.
pub const BACKDROP: [f32; 4] = [0.05, 0.05, 0.07, 0.88];
pub const RIM: [f32; 4] = [0.45, 0.42, 0.55, 0.9];
pub const TEXT: [f32; 4] = [0.86, 0.86, 0.9, 1.0];
pub const DIM: [f32; 4] = [0.86, 0.86, 0.9, 0.45];
pub const SELECTED: [f32; 4] = [0.24, 0.22, 0.36, 1.0];
/// A value that is off its authored setting. The one colour that carries
/// information rather than structure.
pub const OVERRIDDEN: [f32; 4] = [1.0, 0.78, 0.32, 1.0];
/// The filled part of a row's bar.
pub const BAR: [f32; 4] = [0.46, 0.4, 0.85, 0.9];
pub const BAR_TRACK: [f32; 4] = [1.0, 1.0, 1.0, 0.12];

/// Text scale, in [`font::UNIT`](crate::font::UNIT) cells. 2 × an 8-px cell is
/// 16 logical pixels, which is the smallest thing still readable on a Deck at
/// arm's length — and the size a game is therefore expected to have baked.
pub const SCALE: f32 = 2.0;
/// Panel width, in logical pixels — wide enough for `nested.something: 0.001`
/// and narrow enough to leave the game visible, which is the entire point of
/// tuning against a live frame.
pub const WIDTH: f32 = 340.0;
/// Distance from the top-left corner of the screen.
pub const MARGIN: f32 = 12.0;
/// Padding inside the rim.
pub const PAD: f32 = 6.0;
/// How wide the value bar is.
pub const BAR_WIDTH: f32 = 84.0;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// One line of the panel: a group header, or one of its fields.
///
/// Rows are derived from the field list every frame rather than stored, so a
/// root whose entity went away simply stops having rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Row {
    /// A root's header — the thing Enter collapses.
    Group(usize),
    /// An index into the field list.
    Field(usize),
}

/// What the panel is doing between ticks.
///
/// Everything here is ordinary sim state: a `usize`, a `bool` and a small
/// `Vec<usize>`. It is a resource so a replay carries it, and it is `Default` so
/// a game that never opens the panel pays three words.
#[derive(Resource, Clone, Debug, Default)]
pub struct TweakPanel {
    /// Whether the panel is up. The game owns the key that flips this; see the
    /// module docs on why the engine does not.
    pub open: bool,
    /// Where the cursor is, as a row index. Clamped into range every frame, so
    /// a root disappearing under it moves it rather than breaking it.
    cursor: usize,
    /// Root indices whose fields are folded away.
    collapsed: Vec<usize>,
    /// The finger currently dragging a value, and where it started.
    ///
    /// `(id, start x, the value the drag began from)` — the value is captured so
    /// a drag is absolute against its own origin rather than an accumulation of
    /// per-tick deltas, which would drift with the tick rate.
    drag: Option<(u64, f32, f32)>,
    /// The row a finger went down on, so a tap that does not move is a select
    /// and a tap that does is a drag.
    touch_row: Option<usize>,
    /// Fractional units an analog sweep has earned on an **integer** field and
    /// not yet spent — see
    /// [`InspectPanel`](crate::inspect_panel::InspectPanel)'s own, which
    /// carries the argument.
    sweep_carry: f64,
    /// The first row the **last drawn frame** put on screen.
    ///
    /// Written by [`draw`] and read by the touch hit test, which is the only
    /// order that can be right: a finger lands on the picture the player is
    /// looking at, which is last frame's. It is zero until something has been
    /// drawn, and zero is also the answer whenever the list fits.
    scroll: usize,
}

impl TweakPanel {
    pub fn new() -> TweakPanel {
        TweakPanel::default()
    }

    /// Open or close. Closing forgets the drag, so a finger that was mid-adjust
    /// when the panel closed does not resume against a stale origin.
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

    /// The first row on screen as of the last [`draw`]. See the field.
    pub fn scroll(&self) -> usize {
        self.scroll
    }

    pub fn is_collapsed(&self, root: usize) -> bool {
        self.collapsed.contains(&root)
    }

    fn set_collapsed(&mut self, root: usize, collapsed: bool) {
        match (collapsed, self.collapsed.iter().position(|r| *r == root)) {
            (true, None) => self.collapsed.push(root),
            (false, Some(at)) => {
                self.collapsed.remove(at);
            }
            _ => {}
        }
    }

    /// The visible lines, in list order: each root's header followed by its
    /// fields, unless it is folded.
    ///
    /// Derived rather than cached because the field list is derived; a cache
    /// would be one more thing that can disagree with the world.
    pub fn rows(&self, fields: &[TweakField]) -> Vec<Row> {
        let mut rows = Vec::new();
        let mut root = None;
        for (index, field) in fields.iter().enumerate() {
            if root != Some(field.root) {
                root = Some(field.root);
                rows.push(Row::Group(field.root));
            }
            if !self.is_collapsed(field.root) {
                rows.push(Row::Field(index));
            }
        }
        rows
    }
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

/// The keys the panel claims while it is open.
///
/// | key | what it does |
/// |---|---|
/// | Up / Down | move the cursor |
/// | Left / Right | step the value down / up |
/// | Shift + Left/Right | ten steps — coarse |
/// | Ctrl + Left/Right | a tenth of a step — fine |
/// | Enter | toggle a bool, cycle an enum, fold a group |
/// | Tab | fold / unfold the group the cursor is in |
/// | R | put this field back to its authored value |
///
/// Repeat is per press, not per frame held: every one of these reads
/// [`Input::just_pressed`]. A slider you can hold to sweep would want a
/// key-repeat clock, which is a second timebase in something that has to replay.
/// Ten-at-a-time with Shift is the same reach with none of that.
pub const KEY_UP: Key = Key::Up;
pub const KEY_DOWN: Key = Key::Down;
pub const KEY_DEC: Key = Key::Left;
pub const KEY_INC: Key = Key::Right;
pub const KEY_ACTIVATE: Key = Key::Enter;
pub const KEY_FOLD: Key = Key::Tab;
/// **R** for reset, because runt's [`Key`] has no Backspace — the vocabulary
/// is deliberately "what a ball game needs" and punctuation never made it in
/// ([`Key::Other`] is where the rest of a keyboard goes). A letter is a fine
/// binding here anyway: the panel claims every key it uses while it is open.
pub const KEY_RESET: Key = Key::R;

/// How long a full [`PanelNav::sweep`] takes to cross a value's whole range.
///
/// Two seconds is a deliberate compromise: fast enough that reaching the far
/// end of a range is not a chore, slow enough that the last third of the stick
/// is still aimable. The fine end is bought by the caller's curve rather than
/// by this number — see [`PanelNav::sweep`].
pub const SWEEP_SECONDS: f32 = 2.0;

/// Shift's multiplier on one step.
pub const COARSE: f32 = 10.0;
/// Ctrl's.
pub const FINE: f32 = 0.1;

/// Logical pixels of horizontal drag that sweep a value across its whole range.
///
/// Wider than a phone is, on purpose: a finger crossing the screen should be a
/// deliberate full sweep, and everything shorter should be a nudge.
pub const DRAG_SPAN: f32 = 480.0;

/// What one tick of panel navigation *means*, with the device it came from
/// already forgotten.
///
/// Both [`decide`]s used to read [`Input`] directly, which quietly made "a
/// panel" and "a keyboard" the same thing: the table at [`KEY_UP`] is a list of
/// `Key`s, and a game driving these panels from a gamepad, a remote or a
/// scripted test had nowhere to say so. This is the seam. The keys keep their
/// constants and keep their meaning — [`from_keys`](PanelNav::from_keys) *is*
/// the old reads, moved intact — and a caller with another device builds one of
/// these by hand instead.
///
/// Every *boolean* field is an **edge**, not a level, for the reason
/// [`KEY_UP`]'s table gives: repeat is per press, and a held key that swept a
/// value would want a key-repeat clock, which is a second timebase in something
/// that has to replay.
///
/// [`sweep`](PanelNav::sweep) is the one continuous channel, and it does not
/// break that rule — it answers it. A key-repeat clock is a second timebase
/// because it is invented inside the panel; an analog stick is a *value in the
/// input*, integrated over the simulation's own fixed tick, so a recorded trace
/// reproduces the sweep exactly the way it reproduces a press. There is no
/// second clock, which was the whole objection.
///
/// [`fold`](PanelNav::fold) and [`reset`](PanelNav::reset) are this panel's
/// alone — [`crate::inspect_panel`] has no verb for either and ignores them.
/// One struct over both panels rather than two that differ by two fields: they
/// are the same vocabulary, and a panel that happens not to have a verb is not
/// a different vocabulary.
///
/// The pointer is deliberately **not** here. A touch carries a position, an id
/// and a lifetime, and flattening that into booleans would lose the drag — so
/// both `decide`s take a `PanelNav` *and* an [`Input`], and the second one is
/// the finger's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PanelNav {
    /// Move the cursor up a row.
    pub up: bool,
    /// …and down.
    pub down: bool,
    /// Step the value down; cycle an enum backwards.
    pub dec: bool,
    /// …and up, and forwards.
    pub inc: bool,
    /// The row's own verb: flip a bool, cycle a choice, reroll a seed, fold a
    /// group header.
    pub activate: bool,
    /// Fold or unfold the group the cursor is in. This panel only.
    pub fold: bool,
    /// Put this field back to its authored value. This panel only.
    pub reset: bool,
    /// A continuous nudge, `-1..=1`, already curved by whoever built the value.
    ///
    /// Where [`dec`](PanelNav::dec) / [`inc`](PanelNav::inc) are one step per
    /// press, this is a **rate**: [`decide`] integrates it over `dt`, so a stick
    /// held a third of the way moves a value slowly and a full push crosses its
    /// range in [`SWEEP_SECONDS`]. `0.0` — [`NONE`](PanelNav::NONE)'s value, and
    /// [`from_keys`](PanelNav::from_keys)' — means a keyboard, which sweeps
    /// nothing and steps as it always did.
    ///
    /// Curved by the caller rather than here because the shape of the curve is a
    /// feel decision about a particular device, and this struct's whole job is
    /// not to know what device it is.
    pub sweep: f32,
    /// The multiplier on one step of [`dec`](PanelNav::dec) /
    /// [`inc`](PanelNav::inc), already resolved to [`COARSE`], [`FINE`] or
    /// `1.0`.
    ///
    /// Resolved by whoever built the value rather than carried as two modifier
    /// flags, so that a device with no modifiers to spare passes `1.0` and
    /// never has to learn that `COARSE` exists.
    pub scale: f32,
}

impl PanelNav {
    /// Nothing pressed, and one step is one step.
    ///
    /// The base for a hand-built value: `PanelNav { up: true, ..PanelNav::NONE }`.
    /// There is deliberately no `Default` — a derived one would zero
    /// [`scale`](PanelNav::scale), and a nav whose every step multiplies to
    /// nothing is a panel that silently refuses to edit.
    pub const NONE: PanelNav = PanelNav {
        up: false,
        down: false,
        dec: false,
        inc: false,
        activate: false,
        fold: false,
        reset: false,
        sweep: 0.0,
        scale: 1.0,
    };

    /// Both devices at once: pressed here or pressed there.
    ///
    /// A game that drives a panel from its own action table still wants the
    /// keyboard underneath — and wants the two to *add*, so that a verb only
    /// one of them can express (this panel's fold and reset have no obvious
    /// gamepad button) survives being combined rather than being dropped by
    /// whichever side spoke last.
    ///
    /// `scale` is the exception and takes `self`'s unless `self` left it at
    /// `1.0`: a step is one multiplier, not two multiplied together, and a
    /// device with no modifiers must not scale down a device that has them.
    pub fn or(self, other: PanelNav) -> PanelNav {
        PanelNav {
            up: self.up || other.up,
            down: self.down || other.down,
            dec: self.dec || other.dec,
            inc: self.inc || other.inc,
            activate: self.activate || other.activate,
            fold: self.fold || other.fold,
            reset: self.reset || other.reset,
            // The larger deflection wins rather than the sum: two devices
            // pushing the same way are one hand asking once, and adding them
            // would sweep at twice the rate either asked for.
            sweep: if other.sweep.abs() > self.sweep.abs() {
                other.sweep
            } else {
                self.sweep
            },
            scale: if self.scale == 1.0 {
                other.scale
            } else {
                self.scale
            },
        }
    }

    /// The keyboard, exactly as both panels have always read it.
    ///
    /// Kept as one function rather than inlined at the call sites precisely so
    /// that "the keyboard still means what it meant" is one place to look and
    /// one place to be wrong.
    pub fn from_keys(input: &Input) -> PanelNav {
        PanelNav {
            up: input.just_pressed(KEY_UP),
            down: input.just_pressed(KEY_DOWN),
            dec: input.just_pressed(KEY_DEC),
            inc: input.just_pressed(KEY_INC),
            activate: input.just_pressed(KEY_ACTIVATE),
            fold: input.just_pressed(KEY_FOLD),
            reset: input.just_pressed(KEY_RESET),
            // A keyboard has no analog anything: it steps.
            sweep: 0.0,
            scale: if input.held(Key::Shift) {
                COARSE
            } else if input.held(Key::Ctrl) {
                FINE
            } else {
                1.0
            },
        }
    }
}

/// `Set::Input` (exclusive): read this tick's keys and touches, and write what
/// they say into the world.
///
/// Does nothing at all — one resource read and a branch — when the panel is
/// closed. Opening it is the game's business (see the module docs); this only
/// runs the panel that is already up.
///
/// The keyboard's version of [`drive`]. A game whose panel answers to something
/// else — a pad, a remote, a test — schedules its own system around `drive`
/// instead of adding this one, which is why the nav is the only thing that
/// differs between them.
pub fn panel_input(world: &mut World) {
    let Some(nav) = world.get_resource::<Input>().map(PanelNav::from_keys) else {
        return;
    };
    drive(world, nav);
}

/// The simulation's tick length, for [`PanelNav::sweep`]'s integration.
///
/// Read from the world rather than passed, because [`drive`] is already the
/// world-shaped half and a caller assembling a nav has no business also
/// knowing the tick rate. The fallback is the engine's own default, which is
/// what a world without a [`FixedTick`] would have been ticking at anyway.
fn tick_dt(world: &World) -> f32 {
    world
        .get_resource::<crate::ecs::FixedTick>()
        .map(|tick| tick.dt_secs)
        .unwrap_or(1.0 / 60.0)
}

/// One tick of the panel, driven by a nav the caller decided.
///
/// [`panel_input`] is this with [`PanelNav::from_keys`] in front of it. The
/// split exists because the world plumbing here — enumerate the fields, run
/// [`decide`], apply through [`tweak::set_and_record`] — is the same work
/// whatever pressed the button, and a game that reads its own input table
/// should not have to copy it to say so.
///
/// The [`Input`] is still read from the world for the finger's half of
/// `decide`; only the nav comes from the caller.
pub fn drive(world: &mut World, nav: PanelNav) {
    let open = world
        .get_resource::<TweakPanel>()
        .is_some_and(|panel| panel.open);
    if !open {
        return;
    }
    let Some(input) = world.get_resource::<Input>().cloned() else {
        return;
    };
    let fields = tweak::fields_of(world);

    let dt = tick_dt(world);
    let action = world.resource_scope(|_world, mut panel: Mut<TweakPanel>| {
        decide(&mut panel, &fields, nav, &input, dt)
    });

    match action {
        Some(Action::Set(path, value)) => {
            if let Err(e) = tweak::set_and_record(world, &path, value) {
                log::warn!("tweak panel: {path} — {e}");
            }
        }
        Some(Action::Reset(path)) => {
            if let Err(e) = tweak::clear(world, &path) {
                log::warn!("tweak panel: {path} — {e}");
            }
        }
        None => {}
    }
}

/// What one tick of panel input decided to do to the world.
///
/// Separated from [`panel_input`] so the state machine is a pure function of
/// (state, fields, input) and can be tested with a hand-built [`Input`] and no
/// world at all — which is what `tests` does.
#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    Set(String, TweakValue),
    Reset(String),
}

/// The whole state machine: move the cursor, fold a group, or produce an edit.
///
/// `nav` is what the device said; `input` is the finger — see [`PanelNav`] on
/// why those are two arguments and not one.
pub fn decide(
    panel: &mut TweakPanel,
    fields: &[TweakField],
    nav: PanelNav,
    input: &Input,
    dt: f32,
) -> Option<Action> {
    let rows = panel.rows(fields);
    if rows.is_empty() {
        panel.cursor = 0;
        return None;
    }
    // Clamp first: a root that lost its entity since last tick shortens the
    // list under the cursor, and a cursor past the end must land on the last
    // row rather than select nothing.
    panel.cursor = panel.cursor.min(rows.len() - 1);

    let was = panel.cursor;
    if nav.up {
        panel.cursor = (panel.cursor + rows.len() - 1) % rows.len();
    }
    if nav.down {
        panel.cursor = (panel.cursor + 1) % rows.len();
    }
    // See `InspectPanel::sweep_carry`: a sweep belongs to the row it was aimed
    // at, and moving off one abandons the fraction it had earned.
    if panel.cursor != was || nav.sweep == 0.0 {
        panel.sweep_carry = 0.0;
    }

    // Touch, before the keys' edits so a finger and a keyboard cannot both move
    // the same value on one tick. Selection happens here; the drag below turns
    // into the same `Set` a key would.
    if let Some(action) = touch(panel, fields, &rows, input) {
        return Some(action);
    }

    let row = rows[panel.cursor];
    let fold = nav.fold;
    match row {
        Row::Group(root) => {
            // Enter and Tab do the same thing on a header. Two keys for one
            // verb, because Enter is what a player reaches for and Tab is what
            // the same fold means on a field row.
            if fold || nav.activate {
                let collapsed = panel.is_collapsed(root);
                panel.set_collapsed(root, !collapsed);
            }
            None
        }
        Row::Field(index) => {
            let field = &fields[index];
            if fold {
                panel.set_collapsed(field.root, true);
                return None;
            }
            if nav.reset {
                return Some(Action::Reset(field.path.clone()));
            }
            if nav.activate {
                return activate(field);
            }
            let delta = i32::from(nav.inc) - i32::from(nav.dec);
            if delta != 0 {
                return step(field, delta as f32 * field.step() * nav.scale);
            }
            if nav.sweep != 0.0 {
                return sweep(&mut panel.sweep_carry, field, nav.sweep, dt);
            }
            None
        }
    }
}

/// One tick of an analog sweep: a rate, integrated.
///
/// [`crate::inspect_panel::decide`]'s twin, over this panel's value type — and
/// its docs carry the argument for why only numbers sweep and why an integer
/// needs a carry.
fn sweep(carry: &mut f64, field: &TweakField, amount: f32, dt: f32) -> Option<Action> {
    let span = field.range.max - field.range.min;
    let delta = amount * span / SWEEP_SECONDS * dt;
    match &field.value {
        TweakValue::Float(v) => {
            (delta != 0.0).then(|| Action::Set(field.path.clone(), TweakValue::Float(v + delta)))
        }
        TweakValue::Int(v) => {
            *carry += delta as f64;
            let whole = carry.trunc();
            if whole == 0.0 {
                return None;
            }
            *carry -= whole;
            Some(Action::Set(
                field.path.clone(),
                TweakValue::Int(v + whole as i64),
            ))
        }
        _ => None,
    }
}

/// Enter on a field: bools flip, enums advance, numbers do nothing.
///
/// A number does nothing on purpose. The obvious alternative is "open a text
/// entry", and a text entry needs a caret, a character buffer, an accept and a
/// cancel — four states in a module whose entire claim is that it has none.
fn activate(field: &TweakField) -> Option<Action> {
    match &field.value {
        TweakValue::Bool(v) => Some(Action::Set(field.path.clone(), TweakValue::Bool(!v))),
        TweakValue::Choice(name) => {
            let at = field.choices.iter().position(|c| c == name).unwrap_or(0);
            let next = field.choices.get((at + 1) % field.choices.len().max(1))?;
            Some(Action::Set(
                field.path.clone(),
                TweakValue::Choice((*next).to_string()),
            ))
        }
        _ => None,
    }
}

/// Nudge a numeric field by `delta` in its own units.
///
/// The clamp is [`crate::tweak::Tweakables::set`]'s job, not this one's — one
/// clamp, in the place that owns the range, rather than two that could
/// disagree.
fn step(field: &TweakField, delta: f32) -> Option<Action> {
    let value = match field.value {
        TweakValue::Float(v) => TweakValue::Float(v + delta),
        // Rounded away from zero, so a step smaller than 1 on an integer still
        // moves it by one rather than by nothing.
        TweakValue::Int(v) => TweakValue::Int(v + if delta >= 0.0 { 1 } else { -1 }),
        // A bool has no "next"; Enter is its verb.
        TweakValue::Bool(_) => return None,
        TweakValue::Choice(ref name) => {
            let at = field.choices.iter().position(|c| c == name).unwrap_or(0);
            let len = field.choices.len();
            if len == 0 {
                return None;
            }
            let next = if delta >= 0.0 {
                (at + 1) % len
            } else {
                (at + len - 1) % len
            };
            TweakValue::Choice(field.choices[next].to_string())
        }
    };
    Some(Action::Set(field.path.clone(), value))
}

/// Tap to select, drag to adjust.
///
/// A finger that lands inside the panel claims the row under it. If it then
/// moves horizontally it sweeps that row's value across its range — absolute
/// against where the drag started, not accumulated, so the same finger travel
/// is the same change however many ticks it took.
fn touch(
    panel: &mut TweakPanel,
    fields: &[TweakField],
    rows: &[Row],
    input: &Input,
) -> Option<Action> {
    for touch in input.touches_started() {
        let Some(row) = row_at(panel.scroll, rows.len(), touch.pos.y) else {
            continue;
        };
        if touch.pos.x < MARGIN || touch.pos.x > MARGIN + WIDTH {
            continue;
        }
        panel.cursor = row;
        panel.touch_row = Some(row);
        let start = match rows[row] {
            Row::Field(index) => match fields[index].value {
                TweakValue::Float(v) => v,
                TweakValue::Int(v) => v as f32,
                _ => 0.0,
            },
            Row::Group(_) => 0.0,
        };
        panel.drag = Some((touch.id, touch.pos.x, start));
    }

    // A lift ends the drag. A tap that never moved has already done its job
    // (it moved the cursor), so there is nothing else to do here.
    if let Some((id, _, _)) = panel.drag {
        if input.touch_ended(id).is_some() {
            panel.drag = None;
            panel.touch_row = None;
            return None;
        }
    }

    let (id, origin_x, origin_value) = panel.drag?;
    let touch = input.touch(id)?;
    let row = *rows.get(panel.touch_row?)?;
    let Row::Field(index) = row else {
        return None;
    };
    let field = &fields[index];
    let span = field.range.max - field.range.min;
    let moved = touch.pos.x - origin_x;
    if moved.abs() < 1.0 {
        return None;
    }
    let want = origin_value + (moved / DRAG_SPAN) * span;
    match field.value {
        TweakValue::Float(_) => Some(Action::Set(field.path.clone(), TweakValue::Float(want))),
        TweakValue::Int(_) => Some(Action::Set(
            field.path.clone(),
            TweakValue::Int(want.round() as i64),
        )),
        _ => None,
    }
}

/// Which row a screen `y` is over, or `None` for outside the list.
///
/// `scroll` is the index the top visible line stands for — the panel scrolls
/// when the list is longer than the screen, and a hit test that ignored that
/// would select the wrong field on every panel tall enough to matter.
///
/// `pub(crate)` because [`crate::inspect_panel`] lays its rows out on the same
/// pitch below the same title line; two hit tests would be two chances to be a
/// pixel apart.
pub(crate) fn row_at(scroll: usize, count: usize, y: f32) -> Option<usize> {
    let pitch = row_pitch();
    let top = MARGIN + PAD + pitch; // the title occupies the first line
    let row = ((y - top) / pitch).floor();
    if row < 0.0 {
        return None;
    }
    let row = scroll + row as usize;
    (row < count).then_some(row)
}

/// The row pitch the layout and the hit test both use.
///
/// A constant rather than the font's `line_height`, because the hit test in
/// [`touch`] has no font in hand and a hit test that disagrees with the layout
/// by a pixel is worse than one that owns the number. The draw uses this too, so
/// there is exactly one.
///
/// It is [`font::UNIT`](crate::font::UNIT) rather than a bare `8.0` so the tie
/// to the font's nominal cell is written down: a `PanelFont` drawn at [`SCALE`]
/// occupies `UNIT · SCALE` pixels of line box by definition, and the extra two
/// are the air between rows. The number is unchanged — 18 — which is the point:
/// switching a game from a hand-authored bitmap to a real typeface must not move
/// anybody's finger.
pub fn row_pitch() -> f32 {
    crate::font::UNIT * SCALE + 2.0
}

// ---------------------------------------------------------------------------
// Draw
// ---------------------------------------------------------------------------

/// `Set::Ui` (after the game's own HUD): paint the panel over the frame.
///
/// Appends to `batch` rather than clearing it, because a HUD system has already
/// built the screen this tick and the panel goes *on top* — painter's order is
/// `Vec` order and this is the last thing pushed.
///
/// Draws nothing when the panel is closed, when the viewport is unknown (before
/// the first frame), or when nothing is registered — a game with the system
/// installed and no roots gets an empty batch, not an empty box.
///
/// `focused` is *this panel is the one being navigated*, and `false` withholds
/// the cursor bar: a highlighted row on a panel whose next ↑ belongs to some
/// other panel is a lie about where the hands are. A host with one panel passes
/// `true` and forgets about it; [`inspect_panel::draw`](crate::inspect_panel::draw)
/// takes the same argument and states the case at length.
pub fn draw(
    panel: &mut TweakPanel,
    fields: &[TweakField],
    batch: &mut UiBatch,
    font: &dyn PanelFont,
    viewport: Viewport,
    override_count: usize,
    focused: bool,
) {
    if !panel.open || !viewport.is_known() {
        return;
    }
    let rows = panel.rows(fields);
    if rows.is_empty() {
        return;
    }

    let pitch = row_pitch();
    // The title line plus every row, and never taller than the screen: at some
    // number of tunables the list runs off the bottom, and a debug overlay that
    // covers the game it is tuning is the one failure mode worth spending three
    // lines on.
    let max_rows =
        (((viewport.size().y - MARGIN * 2.0 - PAD * 2.0) / pitch).floor() - 1.0).max(1.0);
    let shown = rows.len().min(max_rows as usize);
    // Keep the cursor on screen by scrolling the window, not by moving it.
    let first = panel
        .cursor
        .saturating_sub(shown.saturating_sub(1))
        .min(rows.len() - shown);
    // Remembered for the touch hit test, which runs a tick later against the
    // picture this frame is about to draw. See [`TweakPanel::scroll`].
    panel.scroll = first;
    let height = PAD * 2.0 + pitch * (shown as f32 + 1.0);

    batch.solid([MARGIN, MARGIN, WIDTH, height], RIM);
    batch.solid(
        [MARGIN + 1.0, MARGIN + 1.0, WIDTH - 2.0, height - 2.0],
        BACKDROP,
    );

    let x = MARGIN + PAD;
    let mut y = MARGIN + PAD;
    let title = if override_count == 0 {
        "TWEAKS".to_string()
    } else {
        format!("TWEAKS  ({override_count} changed)")
    };
    font.text(batch, x, y, &title, SCALE, DIM);
    y += pitch;

    for (offset, row) in rows[first..first + shown].iter().enumerate() {
        let index = first + offset;
        // Only while this panel is the one being navigated —
        // [`inspect_panel::draw`](crate::inspect_panel::draw)'s `focused`
        // carries the argument in full, and the two panels answer it alike.
        if focused && index == panel.cursor {
            batch.solid([MARGIN + 1.0, y - 1.0, WIDTH - 2.0, pitch], SELECTED);
        }
        match row {
            Row::Group(root) => {
                let arrow = if panel.is_collapsed(*root) { ">" } else { "v" };
                // The root's name is the head of any of its fields' paths.
                let name = fields
                    .iter()
                    .find(|f| f.root == *root)
                    .and_then(|f| f.path.split('.').next())
                    .unwrap_or("?");
                font.text(batch, x, y, &format!("{arrow} {name}"), SCALE, TEXT);
            }
            Row::Field(field_index) => {
                let field = &fields[*field_index];
                let color = if field.overridden { OVERRIDDEN } else { TEXT };
                font.text(batch, x + 8.0, y, &field.label, SCALE, color);

                let text = field.value.display();
                let right = MARGIN + WIDTH - PAD;
                font.text(
                    batch,
                    right - font.width(&text, SCALE),
                    y,
                    &text,
                    SCALE,
                    color,
                );
                bar(batch, field, right - BAR_WIDTH - 76.0, y, pitch);
            }
        }
        y += pitch;
    }
}

/// The little fill that says where a value sits in its range.
///
/// Skipped for bools and enums: `on`/`off` is already the whole state, and a
/// half-full bar next to it would be a second, worse spelling of it.
fn bar(batch: &mut UiBatch, field: &TweakField, x: f32, y: f32, pitch: f32) {
    let value = match field.value {
        TweakValue::Float(v) => v,
        TweakValue::Int(v) => v as f32,
        _ => return,
    };
    value_bar(batch, &field.range, value, x, y, pitch);
}

/// The bar itself, from a bare range and value.
///
/// Split out of [`bar`] because [`crate::inspect_panel`] draws the same fill
/// for its numeric rows and has widgets rather than [`TweakField`]s in hand —
/// one geometry, so the two panels cannot drift a pixel apart.
pub(crate) fn value_bar(
    batch: &mut UiBatch,
    range: &crate::reflect::FieldRange,
    value: f32,
    x: f32,
    y: f32,
    pitch: f32,
) {
    let h = (pitch - 6.0).max(2.0);
    let top = y + (pitch - h) * 0.5 - 1.0;
    batch.solid([x, top, BAR_WIDTH, h], BAR_TRACK);
    let t = range.normalize(value);
    if t > 0.0 {
        batch.solid([x, top, BAR_WIDTH * t, h], BAR);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{InputEvent, TouchPhase};
    use crate::reflect::FieldRange;

    fn field(path: &str, root: usize, value: TweakValue) -> TweakField {
        TweakField {
            path: path.to_string(),
            root,
            label: path
                .split_once('.')
                .map(|(_, r)| r.to_string())
                .unwrap_or_default(),
            value,
            range: FieldRange::new(0.0, 10.0),
            choices: Vec::new(),
            overridden: false,
        }
    }

    fn fields() -> Vec<TweakField> {
        vec![
            field("sky.clouds", 0, TweakValue::Float(2.0)),
            field("sky.lit", 0, TweakValue::Bool(false)),
            field("cam.steps", 1, TweakValue::Int(3)),
        ]
    }

    /// One tick of input built from events, the way `Sim::tick` builds it.
    fn tick(events: impl IntoIterator<Item = InputEvent>) -> Input {
        let mut input = Input::new();
        input.begin_tick(events);
        input
    }

    /// One tick at the engine's fixed rate — what `drive` reads off `FixedTick`
    /// in a real world. Only the sweep looks at it.
    const TEST_DT: f32 = 1.0 / 60.0;

    /// …and a second tick against the same `Input`, so a held modifier stays
    /// held while an edge fires.
    fn again(input: &mut Input, events: impl IntoIterator<Item = InputEvent>) -> Input {
        input.begin_tick(events);
        input.clone()
    }

    /// [`decide`] as every table below drives it: the keyboard, through the
    /// very conversion [`panel_input`] uses.
    ///
    /// These tests were written against a `decide` that read [`Input`] itself.
    /// That they are unchanged in what they press and what they assert is the
    /// proof that [`PanelNav`] *moved* the reads rather than changing them —
    /// which is the whole claim of the seam, and is worth one indirection to
    /// keep checkable.
    fn decide_keys(panel: &mut TweakPanel, fields: &[TweakField], input: &Input) -> Option<Action> {
        decide(panel, fields, PanelNav::from_keys(input), input, TEST_DT)
    }

    #[test]
    fn a_closed_panel_has_no_rows_and_draws_nothing() {
        let mut panel = TweakPanel::new();
        let mut batch = UiBatch::new();
        draw(
            &mut panel,
            &fields(),
            &mut batch,
            &BlockFont,
            Viewport::new(1280, 720),
            0,
            true,
        );
        assert!(batch.is_empty(), "a closed panel is not a pass");
    }

    #[test]
    fn the_rows_are_a_header_per_root_and_folding_removes_its_fields() {
        let mut panel = TweakPanel::new();
        let fields = fields();
        assert_eq!(
            panel.rows(&fields),
            vec![
                Row::Group(0),
                Row::Field(0),
                Row::Field(1),
                Row::Group(1),
                Row::Field(2),
            ]
        );
        panel.set_collapsed(0, true);
        assert_eq!(
            panel.rows(&fields),
            vec![Row::Group(0), Row::Group(1), Row::Field(2)],
            "a folded group keeps its header and loses its fields"
        );
    }

    #[test]
    fn the_cursor_wraps_and_survives_the_list_shrinking_under_it() {
        let mut panel = TweakPanel::new();
        panel.open = true;
        let fields = fields();

        decide_keys(&mut panel, &fields, &tick([InputEvent::KeyDown(Key::Down)]));
        assert_eq!(panel.cursor(), 1);
        // Up from the top wraps to the bottom, which is what makes a five-row
        // list navigable with two keys.
        let mut panel_top = TweakPanel::new();
        panel_top.open = true;
        decide_keys(
            &mut panel_top,
            &fields,
            &tick([InputEvent::KeyDown(Key::Up)]),
        );
        assert_eq!(panel_top.cursor(), 4);

        // A root going away shortens the list; the cursor lands on the end
        // rather than off it.
        panel.cursor = 4;
        let shorter = vec![fields[0].clone()];
        decide_keys(&mut panel, &shorter, &tick([]));
        assert_eq!(panel.cursor(), 1);

        // …and no rows at all is not a panic.
        decide_keys(&mut panel, &[], &tick([]));
        assert_eq!(panel.cursor(), 0);
    }

    #[test]
    fn arrows_step_a_number_and_the_modifiers_scale_the_step() {
        let mut panel = TweakPanel::new();
        panel.open = true;
        let fields = fields();
        panel.cursor = 1; // sky.clouds

        // Default step for an undeclared range is a hundredth of it: 0.1 here.
        let plus = decide_keys(
            &mut panel,
            &fields,
            &tick([InputEvent::KeyDown(Key::Right)]),
        );
        assert_eq!(
            plus,
            Some(Action::Set("sky.clouds".into(), TweakValue::Float(2.1)))
        );

        let mut held = Input::new();
        held.begin_tick([InputEvent::KeyDown(Key::Shift)]);
        let coarse = again(&mut held, [InputEvent::KeyDown(Key::Right)]);
        assert_eq!(
            decide_keys(&mut panel, &fields, &coarse),
            Some(Action::Set("sky.clouds".into(), TweakValue::Float(3.0))),
            "shift is ten steps"
        );

        let mut held = Input::new();
        held.begin_tick([InputEvent::KeyDown(Key::Ctrl)]);
        let fine = again(&mut held, [InputEvent::KeyDown(Key::Left)]);
        let Some(Action::Set(_, TweakValue::Float(v))) = decide_keys(&mut panel, &fields, &fine) else {
            panic!("ctrl+left produced no edit");
        };
        assert!((v - 1.99).abs() < 1e-5, "{v} is not a tenth of a step down");
    }

    /// The seam's whole point: a device that is not a keyboard.
    ///
    /// Nothing here presses a key — the [`Input`] handed over is a silent tick,
    /// present only because the finger's half of `decide` still wants one — and
    /// the panel moves and edits regardless, by the same arithmetic the arrows
    /// take above.
    #[test]
    fn a_nav_built_by_hand_drives_the_panel_with_no_keyboard_at_all() {
        let mut panel = TweakPanel::new();
        panel.open = true;
        let fields = fields();
        let quiet = tick([]);

        decide(
            &mut panel,
            &fields,
            PanelNav {
                down: true,
                ..PanelNav::NONE
            },
            &quiet,
            TEST_DT,
        );
        assert_eq!(panel.cursor, 1, "sky.clouds, one row past its header");

        assert_eq!(
            decide(
                &mut panel,
                &fields,
                PanelNav {
                    inc: true,
                    ..PanelNav::NONE
                },
                &quiet,
                TEST_DT,
            ),
            Some(Action::Set("sky.clouds".into(), TweakValue::Float(2.1))),
            "one step, the very step the Right arrow takes"
        );

        assert_eq!(
            decide(
                &mut panel,
                &fields,
                PanelNav {
                    inc: true,
                    scale: COARSE,
                    ..PanelNav::NONE
                },
                &quiet,
                TEST_DT,
            ),
            Some(Action::Set("sky.clouds".into(), TweakValue::Float(3.0))),
            "a device with a coarse modifier of its own says so in `scale`"
        );

        // And a nav with nothing pressed edits nothing — `NONE`'s scale is
        // 1.0, so a caller that spreads it has a live step waiting rather than
        // one multiplied to zero.
        assert_eq!(
            decide(&mut panel, &fields, PanelNav::NONE, &quiet, TEST_DT),
            None
        );
    }

    /// A stick sweeps a value as a **rate**, and the keyboard still steps.
    ///
    /// The rate is a fraction of the field's own range per `SWEEP_SECONDS`, so
    /// a full push crosses `sky.clouds`' 0..10 in two seconds — five a second,
    /// or 1/12 of a unit per tick at 60 Hz.
    #[test]
    fn a_full_sweep_crosses_the_range_in_sweep_seconds() {
        let mut panel = TweakPanel::new();
        panel.open = true;
        let fields = fields();
        panel.cursor = 1; // sky.clouds, a Float over 0..10 starting at 2.0
        let quiet = tick([]);

        let nav = PanelNav {
            sweep: 1.0,
            ..PanelNav::NONE
        };
        let Some(Action::Set(_, TweakValue::Float(v))) =
            decide(&mut panel, &fields, nav, &quiet, TEST_DT)
        else {
            panic!("a full sweep produced no edit");
        };
        let per_tick = 10.0 / SWEEP_SECONDS * TEST_DT;
        assert!(
            (v - (2.0 + per_tick)).abs() < 1e-5,
            "{v} is not one tick of a full-range sweep"
        );

        // Half the stick is half the rate — it is a rate, not a step.
        let half = PanelNav {
            sweep: 0.5,
            ..PanelNav::NONE
        };
        let Some(Action::Set(_, TweakValue::Float(v))) =
            decide(&mut panel, &fields, half, &quiet, TEST_DT)
        else {
            panic!("half a sweep produced no edit");
        };
        assert!((v - (2.0 + per_tick * 0.5)).abs() < 1e-5, "{v}");

        // …and a nav with no sweep in it edits nothing, so the keyboard path
        // is untouched by any of this.
        assert_eq!(
            decide(&mut panel, &fields, PanelNav::NONE, &quiet, TEST_DT),
            None
        );
    }

    /// An integer field moves in whole units, and a sweep too slow to fill one
    /// in a tick still gets there — the carry is what stops small deflections
    /// rounding to nothing forever.
    #[test]
    fn a_slow_sweep_still_eventually_moves_an_integer() {
        let mut panel = TweakPanel::new();
        panel.open = true;
        let fields = fields();
        panel.cursor = 4; // cam.steps, an Int
        let quiet = tick([]);
        let crawl = PanelNav {
            sweep: 0.05,
            ..PanelNav::NONE
        };

        // 0.05 × 10 / 2s × 1/60 ≈ 0.0042 a tick: nothing lands for a while.
        assert_eq!(decide(&mut panel, &fields, crawl, &quiet, TEST_DT), None);
        let mut moved = None;
        for _ in 0..600 {
            if let Some(action) = decide(&mut panel, &fields, crawl, &quiet, TEST_DT) {
                moved = Some(action);
                break;
            }
        }
        assert_eq!(
            moved,
            Some(Action::Set("cam.steps".into(), TweakValue::Int(4))),
            "the carry never spent a whole unit"
        );
    }

    #[test]
    fn two_devices_or_together_without_either_losing_a_verb() {
        let keys = PanelNav {
            fold: true,
            reset: true,
            scale: COARSE,
            ..PanelNav::NONE
        };
        let pad = PanelNav {
            up: true,
            activate: true,
            ..PanelNav::NONE
        };

        let both = keys.or(pad);
        assert!(both.up && both.activate, "the pad's verbs survive");
        assert!(
            both.fold && both.reset,
            "…and so do the two the pad cannot express"
        );
        assert_eq!(
            both.scale, COARSE,
            "a device with no modifiers must not scale down one that has them"
        );
        assert_eq!(
            pad.or(keys).scale,
            COARSE,
            "…from either side, since a step is one multiplier and not two"
        );
    }

    #[test]
    fn enter_toggles_a_bool_and_does_nothing_to_a_number() {
        let mut panel = TweakPanel::new();
        panel.open = true;
        let fields = fields();

        panel.cursor = 2; // sky.lit
        assert_eq!(
            decide_keys(
                &mut panel,
                &fields,
                &tick([InputEvent::KeyDown(Key::Enter)])
            ),
            Some(Action::Set("sky.lit".into(), TweakValue::Bool(true)))
        );

        panel.cursor = 1; // sky.clouds — a float has no "activate"
        assert_eq!(
            decide_keys(
                &mut panel,
                &fields,
                &tick([InputEvent::KeyDown(Key::Enter)])
            ),
            None
        );
    }

    #[test]
    fn enter_on_a_header_folds_it_and_backspace_on_a_field_resets_it() {
        let mut panel = TweakPanel::new();
        panel.open = true;
        let fields = fields();

        panel.cursor = 0; // the `sky` header
        decide_keys(
            &mut panel,
            &fields,
            &tick([InputEvent::KeyDown(Key::Enter)]),
        );
        assert!(panel.is_collapsed(0));
        decide_keys(
            &mut panel,
            &fields,
            &tick([InputEvent::KeyDown(Key::Enter)]),
        );
        assert!(!panel.is_collapsed(0));

        panel.cursor = 1;
        assert_eq!(
            decide_keys(&mut panel, &fields, &tick([InputEvent::KeyDown(Key::R)])),
            Some(Action::Reset("sky.clouds".into()))
        );
        // Tab on a field folds the group it belongs to, which is the way back
        // out of a long list without walking the cursor up it.
        panel.cursor = 1;
        decide_keys(&mut panel, &fields, &tick([InputEvent::KeyDown(Key::Tab)]));
        assert!(panel.is_collapsed(0));
    }

    #[test]
    fn an_integer_steps_by_one_however_small_the_step_is() {
        let mut panel = TweakPanel::new();
        panel.open = true;
        let fields = fields();
        panel.cursor = 4; // cam.steps
        let mut held = Input::new();
        held.begin_tick([InputEvent::KeyDown(Key::Ctrl)]);
        let fine = again(&mut held, [InputEvent::KeyDown(Key::Right)]);
        assert_eq!(
            decide_keys(&mut panel, &fields, &fine),
            Some(Action::Set("cam.steps".into(), TweakValue::Int(4))),
            "a tenth of a step on an integer is still one"
        );
    }

    #[test]
    fn a_tap_selects_the_row_and_a_drag_sweeps_its_value() {
        let mut panel = TweakPanel::new();
        panel.open = true;
        let fields = fields();

        // Row 1 (`sky.clouds`) — the second line of the list, one pitch below
        // the title.
        let pitch = row_pitch();
        let y = MARGIN + PAD + pitch * 2.0 + 1.0;
        let down = tick([InputEvent::Touch {
            id: 7,
            phase: TouchPhase::Started,
            x: 40.0,
            y,
        }]);
        assert_eq!(
            decide_keys(&mut panel, &fields, &down),
            None,
            "a tap only selects"
        );
        assert_eq!(panel.cursor(), 1);

        // …and dragging right by a quarter of the sweep span moves the value a
        // quarter of its range, measured from where the finger landed.
        let mut live = Input::new();
        live.begin_tick([InputEvent::Touch {
            id: 7,
            phase: TouchPhase::Started,
            x: 40.0,
            y,
        }]);
        decide_keys(&mut panel, &fields, &live.clone());
        let moved = again(
            &mut live,
            [InputEvent::Touch {
                id: 7,
                phase: TouchPhase::Moved,
                x: 40.0 + DRAG_SPAN * 0.25,
                y,
            }],
        );
        let Some(Action::Set(path, TweakValue::Float(v))) = decide_keys(&mut panel, &fields, &moved)
        else {
            panic!("the drag produced no edit");
        };
        assert_eq!(path, "sky.clouds");
        assert!((v - (2.0 + 2.5)).abs() < 1e-4, "{v}");

        // A lift ends it; further motion from a new finger does not resume.
        let up = again(
            &mut live,
            [InputEvent::Touch {
                id: 7,
                phase: TouchPhase::Ended,
                x: 400.0,
                y,
            }],
        );
        assert_eq!(decide_keys(&mut panel, &fields, &up), None);
    }

    /// The stub the two drawing tests share: one quad per character, so a
    /// batch's length is a proxy for "text was asked for".
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
    fn a_list_taller_than_the_screen_scrolls_and_the_hit_test_scrolls_with_it() {
        // The bug this exists for: `row_at` counts rows from the top of the
        // panel, and the top of the panel is not row 0 once the list is longer
        // than the screen. A hit test that forgot that would select the wrong
        // field on every panel tall enough to matter.
        let mut panel = TweakPanel::new();
        panel.open = true;
        let fields: Vec<TweakField> = (0..40)
            .map(|i| field(&format!("sky.f{i}"), 0, TweakValue::Float(0.0)))
            .collect();

        // A viewport with room for a handful of rows, and the cursor at the end
        // of the list so the window has to have moved.
        let short = Viewport::new(320, (MARGIN * 2.0 + PAD * 2.0 + row_pitch() * 7.0) as u32);
        panel.cursor = fields.len();
        let mut batch = UiBatch::new();
        draw(&mut panel, &fields, &mut batch, &BlockFont, short, 0, true);
        assert!(
            !batch.is_empty(),
            "a panel taller than the screen drew nothing"
        );
        assert!(panel.scroll() > 0, "the window never scrolled");

        // A tap on the first visible line selects the row that line *is*.
        let y = MARGIN + PAD + row_pitch() + 1.0;
        let first_visible = panel.scroll();
        decide_keys(
            &mut panel,
            &fields,
            &tick([InputEvent::Touch {
                id: 3,
                phase: TouchPhase::Started,
                x: MARGIN + 20.0,
                y,
            }]),
        );
        assert_eq!(panel.cursor(), first_visible);
    }

    #[test]
    fn a_touch_outside_the_panel_is_not_the_panels() {
        let mut panel = TweakPanel::new();
        panel.open = true;
        let fields = fields();
        let y = MARGIN + PAD + row_pitch() * 2.0 + 1.0;
        // Right of the panel's own width: the game's, not the panel's.
        decide_keys(
            &mut panel,
            &fields,
            &tick([InputEvent::Touch {
                id: 1,
                phase: TouchPhase::Started,
                x: MARGIN + WIDTH + 40.0,
                y,
            }]),
        );
        assert_eq!(
            panel.cursor(),
            0,
            "the cursor moved for somebody else's tap"
        );
    }
}
