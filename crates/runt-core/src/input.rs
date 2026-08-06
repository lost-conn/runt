//! Engine-side input (DESIGN §4).
//!
//! Hosts translate their native events (winit, rinch `SurfaceEvent`, …) into
//! [`InputEvent`]s and push them at them engine. The engine never sees a host
//! type, and — crucially for determinism — buffered events are **only** applied
//! at tick boundaries. A replay is therefore just the recorded event trace plus
//! the seeds: nothing about *when* the host happened to poll can leak into sim
//! state.

use bevy_ecs::resource::Resource;
use glam::Vec2;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Keys
// ---------------------------------------------------------------------------

/// The engine's own key identity — a physical-layout key code, not a character.
///
/// Deliberately small: what a ball game needs. Anything else the host cannot map
/// arrives as [`Key::Other`], which is a single bucket by design (we never want
/// unmapped keys silently acquiring meaning).
///
/// `Serialize`/`Deserialize` exist so an [`InputTrace`](crate::trace::InputTrace)
/// can be written to a file. Postcard encodes a unit variant as its **index**, so
/// append new keys at the end (before [`Key::Other`] is fine to break; a stored
/// trace is a debugging artifact, not a save format).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Key {
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    Q,
    R,
    S,
    T,
    U,
    V,
    W,
    X,
    Y,
    Z,
    Digit0,
    Digit1,
    Digit2,
    Digit3,
    Digit4,
    Digit5,
    Digit6,
    Digit7,
    Digit8,
    Digit9,
    Up,
    Down,
    Left,
    Right,
    Space,
    Enter,
    Tab,
    Escape,
    Shift,
    Ctrl,
    Alt,
    /// `[` — the physical key left of `]`, whatever it prints on this layout.
    /// Conventionally a "step something down" binding (render scale, in the
    /// hosts that have one).
    BracketLeft,
    /// `]`. See [`Key::BracketLeft`].
    BracketRight,
    /// Every key the host could not map. Never given gameplay meaning.
    Other,
}

impl Key {
    /// Total number of distinct [`Key`] values — the width of the key sets.
    pub const COUNT: usize = Key::Other as usize + 1;

    /// Dense index, stable for the life of the build (used by [`KeySet`]).
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// All keys, in declaration order. Iteration order is fixed, so anything
    /// built by walking this is deterministic.
    pub const ALL: [Key; Key::COUNT] = {
        use Key::*;
        [
            A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, Digit0,
            Digit1, Digit2, Digit3, Digit4, Digit5, Digit6, Digit7, Digit8, Digit9, Up, Down, Left,
            Right, Space, Enter, Tab, Escape, Shift, Ctrl, Alt, BracketLeft, BracketRight, Other,
        ]
    };
}

/// A fixed-width bitset over [`Key`]. Iteration order is the declaration order
/// of `Key`, so it is deterministic — DESIGN §3 forbids hash iteration feeding
/// sim state, and this sidesteps the question entirely.
mod keyset {
    use super::Key;

    /// 64 bits is comfortably wider than [`Key::COUNT`]; assert it stays so.
    const _: () = assert!(Key::COUNT <= 64);

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct KeySet(u64);

    impl KeySet {
        #[inline]
        pub const fn new() -> KeySet {
            KeySet(0)
        }

        #[inline]
        pub fn insert(&mut self, key: Key) -> bool {
            let bit = 1u64 << key.index();
            let had = self.0 & bit != 0;
            self.0 |= bit;
            !had
        }

        #[inline]
        pub fn remove(&mut self, key: Key) -> bool {
            let bit = 1u64 << key.index();
            let had = self.0 & bit != 0;
            self.0 &= !bit;
            had
        }

        #[inline]
        pub fn contains(&self, key: Key) -> bool {
            self.0 & (1u64 << key.index()) != 0
        }

        #[inline]
        pub fn clear(&mut self) {
            self.0 = 0;
        }

        #[inline]
        pub fn is_empty(&self) -> bool {
            self.0 == 0
        }

        #[inline]
        pub fn len(&self) -> u32 {
            self.0.count_ones()
        }

        /// Keys in `Key::ALL` order.
        pub fn iter(&self) -> impl Iterator<Item = Key> + '_ {
            Key::ALL.into_iter().filter(|k| self.contains(*k))
        }
    }
}

pub use keyset::KeySet;

// ---------------------------------------------------------------------------
// Gamepad
// ---------------------------------------------------------------------------

/// A gamepad face/shoulder/stick/d-pad button, in SDL "standard mapping" order.
///
/// Named by **position, not by legend**: `South`/`East`/`West`/`North` rather
/// than A/B/X/Y, because the letters move between vendors (and the *meaning* of
/// A/B swaps between Nintendo and everyone else) while the position under the
/// thumb does not. A binding written against `South` is the same physical button
/// on every pad; a binding written against "A" is a bug on half of them.
///
/// The order is SDL's / the Web Gamepad API's standard mapping so a host can
/// translate by index rather than by table, which is one fewer place to get the
/// mapping wrong.
///
/// [`PadButton::Other`] is a single bucket for everything a host could not map
/// (guide/home, paddles, a sixth shoulder), on the same doctrine as
/// [`Key::Other`]: unmapped controls must never silently acquire gameplay
/// meaning.
///
/// Postcard encodes a unit variant as its **index** — append new buttons at the
/// end, as for [`Key`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum PadButton {
    /// Bottom face button (A on Xbox, cross on PlayStation, B on Nintendo).
    South,
    /// Right face button (B / circle / A).
    East,
    /// Left face button (X / square / Y).
    West,
    /// Top face button (Y / triangle / X).
    North,
    /// Left shoulder (LB / L1).
    L1,
    /// Right shoulder (RB / R1).
    R1,
    /// Left stick pressed in (LS / L3).
    L3,
    /// Right stick pressed in (RS / R3).
    R3,
    Start,
    Select,
    DpadUp,
    DpadDown,
    DpadLeft,
    DpadRight,
    /// Every pad button the host could not map. Never given gameplay meaning.
    Other,
}

impl PadButton {
    /// Total number of distinct [`PadButton`] values — the width of
    /// [`PadButtonSet`].
    pub const COUNT: usize = PadButton::Other as usize + 1;

    /// Dense index, stable for the life of the build (used by
    /// [`PadButtonSet`]).
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// All pad buttons, in declaration order. Iteration order is fixed, so
    /// anything built by walking this is deterministic (DESIGN §3).
    pub const ALL: [PadButton; PadButton::COUNT] = {
        use PadButton::*;
        [
            South, East, West, North, L1, R1, L3, R3, Start, Select, DpadUp, DpadDown, DpadLeft,
            DpadRight, Other,
        ]
    };
}

/// Which analog stick a [`InputEvent::PadStick`] is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum PadStick {
    Left,
    Right,
}

impl PadStick {
    pub const COUNT: usize = PadStick::Right as usize + 1;

    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    pub const ALL: [PadStick; PadStick::COUNT] = [PadStick::Left, PadStick::Right];
}

/// Which analog trigger a [`InputEvent::PadTrigger`] is about.
///
/// The shoulder *buttons* are [`PadButton::L1`]/[`PadButton::R1`]; these are the
/// pressure-sensitive ones underneath them, which are levels rather than edges
/// and so cannot live in the button set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum PadTrigger {
    L2,
    R2,
}

impl PadTrigger {
    pub const COUNT: usize = PadTrigger::R2 as usize + 1;

    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    pub const ALL: [PadTrigger; PadTrigger::COUNT] = [PadTrigger::L2, PadTrigger::R2];
}

/// A fixed-width bitset over [`PadButton`], the pad's answer to [`KeySet`].
/// Iteration order is `PadButton::ALL` order, hence deterministic (DESIGN §3).
mod padbuttonset {
    use super::PadButton;

    /// 16 bits covers [`PadButton::COUNT`] with room to append; assert it stays
    /// so (widen to `u32` here and in the shifts if a pad ever needs it).
    const _: () = assert!(PadButton::COUNT <= 16);

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct PadButtonSet(u16);

    impl PadButtonSet {
        #[inline]
        pub const fn new() -> PadButtonSet {
            PadButtonSet(0)
        }

        #[inline]
        pub fn insert(&mut self, button: PadButton) -> bool {
            let bit = 1u16 << button.index();
            let had = self.0 & bit != 0;
            self.0 |= bit;
            !had
        }

        #[inline]
        pub fn remove(&mut self, button: PadButton) -> bool {
            let bit = 1u16 << button.index();
            let had = self.0 & bit != 0;
            self.0 &= !bit;
            had
        }

        #[inline]
        pub fn contains(&self, button: PadButton) -> bool {
            self.0 & (1u16 << button.index()) != 0
        }

        #[inline]
        pub fn clear(&mut self) {
            self.0 = 0;
        }

        #[inline]
        pub fn is_empty(&self) -> bool {
            self.0 == 0
        }

        #[inline]
        pub fn len(&self) -> u32 {
            self.0.count_ones()
        }

        /// Buttons in `PadButton::ALL` order.
        pub fn iter(&self) -> impl Iterator<Item = PadButton> + '_ {
            PadButton::ALL.into_iter().filter(|b| self.contains(*b))
        }
    }
}

pub use padbuttonset::PadButtonSet;

// ---------------------------------------------------------------------------
// Touch
// ---------------------------------------------------------------------------

/// What just happened to one finger.
///
/// A mirror of winit's `TouchPhase` (and of the browser's
/// `touchstart`/`touchmove`/`touchend`/`touchcancel`) rather than a re-export of
/// either: DESIGN §2 says the engine never sees a host type, and a winit
/// dependency here would drag a windowing library into every replay, every
/// headless test and the editor.
///
/// Postcard encodes a unit variant as its **index**, so append rather than
/// insert (see [`Key`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum TouchPhase {
    Started,
    Moved,
    Ended,
    /// The system took the finger away — a gesture recogniser claimed it, the
    /// screen locked, the browser decided the page was scrolling. [`Input`]
    /// treats it exactly as [`TouchPhase::Ended`]; see
    /// [`Input::touches_ended`].
    Cancelled,
}

/// One finger, as a `FixedSim` system sees it.
///
/// `pos` is in **logical pixels, top-left origin, +Y down** — the coordinate
/// system every windowing system reports touches in, and the one
/// [`UiBatch`](crate::ui::UiBatch) already lays HUD rectangles out in. Nothing
/// flips it on the way through, because the question a touch game asks is "is
/// this finger inside the button I drew?", and the button is in screen space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Touch {
    /// The host's identifier for this contact.
    ///
    /// Opaque, and unique only among *live* touches: a platform is free to hand
    /// the same number to the next finger once this one lifts. A game that
    /// remembers an id across ticks must drop it when the touch ends (that is
    /// what [`Input::touches_ended`] is for) rather than assume ids never
    /// repeat.
    pub id: u64,
    /// Where the finger is now, or — for a touch in
    /// [`Input::touches_started`] / [`Input::touches_ended`] — where it landed
    /// or was last seen.
    pub pos: Vec2,
}

/// How many simultaneous fingers the engine will track.
///
/// Ten is every hand a player has and more contacts than most panels report.
/// The cap exists so [`Input`] stays a fixed-size, allocation-free value —
/// [`crate::trace::apply`] clones it every replayed tick — and so a host with a
/// runaway digitiser cannot grow sim state without bound. Touches past the cap
/// are dropped on arrival, which a trace then reproduces exactly, because the
/// trace is recorded from `Input` and `Input` never saw them.
pub const MAX_TOUCHES: usize = 10;

/// A fixed-capacity list of [`Touch`]es in arrival order.
///
/// Arrival order rather than id order, and an array rather than a `Vec`, for the
/// two reasons that run through this file: iteration has to be deterministic
/// (DESIGN §3 — and id order would reshuffle a gesture the moment a platform
/// recycled a number), and [`Input`] has to stay allocation-free because replay
/// clones it per tick.
mod touchlist {
    use super::{Touch, MAX_TOUCHES};
    use glam::Vec2;

    /// A live touch plus the one bit the recorder needs: whether this tick
    /// moved it. Kept beside the position rather than in a parallel array so a
    /// removal cannot desynchronise the two.
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Slot {
        touch: Touch,
        moved: bool,
    }

    const EMPTY: Slot = Slot {
        touch: Touch {
            id: 0,
            pos: Vec2::ZERO,
        },
        moved: false,
    };

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct TouchList {
        slots: [Slot; MAX_TOUCHES],
        len: u8,
    }

    impl Default for TouchList {
        fn default() -> TouchList {
            TouchList::new()
        }
    }

    impl TouchList {
        #[inline]
        pub const fn new() -> TouchList {
            TouchList {
                slots: [EMPTY; MAX_TOUCHES],
                len: 0,
            }
        }

        /// Append `touch`. `false` when the list is full — see [`MAX_TOUCHES`].
        pub fn push(&mut self, touch: Touch) -> bool {
            if self.len as usize >= MAX_TOUCHES {
                return false;
            }
            self.slots[self.len as usize] = Slot {
                touch,
                moved: false,
            };
            self.len += 1;
            true
        }

        pub fn get(&self, id: u64) -> Option<Touch> {
            self.slice()
                .iter()
                .find(|s| s.touch.id == id)
                .map(|s| s.touch)
        }

        /// Move a live touch. `false` when no touch has that id — a host
        /// reporting motion for a finger that already lifted is ignored rather
        /// than resurrecting it.
        ///
        /// A move to where the touch already was is not a move: the `moved` flag
        /// is what keeps a finger resting still from writing an event into the
        /// trace every tick, exactly as the drive stick's change flag does.
        pub fn set_pos(&mut self, id: u64, pos: Vec2) -> bool {
            let Some(slot) = self.slice_mut().iter_mut().find(|s| s.touch.id == id) else {
                return false;
            };
            if slot.touch.pos != pos {
                slot.touch.pos = pos;
                slot.moved = true;
            }
            true
        }

        /// Remove the touch with `id`, returning it with its last position.
        /// Order among the survivors is preserved (`remove`, not `swap_remove`)
        /// because arrival order is the thing this list promises.
        pub fn remove(&mut self, id: u64) -> Option<Touch> {
            let at = self.slice().iter().position(|s| s.touch.id == id)?;
            let touch = self.slots[at].touch;
            for i in at..self.len as usize - 1 {
                self.slots[i] = self.slots[i + 1];
            }
            self.len -= 1;
            Some(touch)
        }

        /// Whether this tick moved the touch with `id`.
        pub fn moved(&self, id: u64) -> bool {
            self.slice()
                .iter()
                .find(|s| s.touch.id == id)
                .is_some_and(|s| s.moved)
        }

        pub fn clear_moved(&mut self) {
            for slot in self.slice_mut() {
                slot.moved = false;
            }
        }

        pub fn clear(&mut self) {
            self.len = 0;
        }

        #[inline]
        pub fn len(&self) -> usize {
            self.len as usize
        }

        #[inline]
        pub fn is_empty(&self) -> bool {
            self.len == 0
        }

        /// Touches in arrival order.
        pub fn iter(&self) -> impl Iterator<Item = Touch> + '_ {
            self.slice().iter().map(|s| s.touch)
        }

        #[inline]
        fn slice(&self) -> &[Slot] {
            &self.slots[..self.len as usize]
        }

        #[inline]
        fn slice_mut(&mut self) -> &mut [Slot] {
            &mut self.slots[..self.len as usize]
        }
    }
}

pub use touchlist::TouchList;

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// One host input event. This is the *entire* engine input vocabulary; a replay
/// trace is a `Vec<(tick, InputEvent)>` — see [`crate::trace`].
///
/// Postcard encodes a variant as its **index**, so new events are appended here
/// rather than inserted: a trace file recorded before `TouchDrive`, `FocusLost`
/// or the pad events existed still reads back correctly. (The reverse does not
/// hold, and a trace remains a debugging artifact rather than a save format.)
/// `tests/input_trace_compat.rs` pins every index so a reorder fails the build
/// rather than silently rewriting old traces.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum InputEvent {
    KeyDown(Key),
    KeyUp(Key),
    /// Pointer motion in **pixels**, already a delta (hosts that only get
    /// absolute positions difference them before pushing).
    MouseMove { dx: f32, dy: f32 },
    /// `button`: 0 = left, 1 = right, 2 = middle, 3.. host-defined.
    MouseButton { button: u8, pressed: bool },
    /// Wheel notches (positive = scroll up / zoom in).
    Wheel { dy: f32 },
    /// Analog drive stick: `x` = right, `y` = forward, magnitude `0..=1`. What a
    /// touch host's virtual stick produces (and what a gamepad would, later).
    ///
    /// **A level, not an edge.** Unlike every other event here it does not
    /// describe a change; it sets a value that persists across ticks until
    /// something sets it again, exactly as a physical stick's position does.
    /// A host sends one when the value moves and a zero when the finger lifts.
    /// See [`Input::drive`].
    TouchDrive { dir: Vec2 },
    /// The window stopped receiving input (alt-tab, a phone call, a browser tab
    /// going to the background).
    ///
    /// It is an *event* rather than a method on [`Input`] so that everything a
    /// run depends on arrives through one channel: a host that calls a method
    /// out of band would produce runs a trace cannot describe. The engine's
    /// response is [`Input::release_all`] — every held key and button is
    /// released and the drive stick is zeroed, because the release the host
    /// would otherwise deliver is never coming.
    FocusLost,
    /// A gamepad button changed state. An edge, like [`InputEvent::KeyDown`] —
    /// and like a key, a host that repeats a "still down" report does not
    /// produce a second press (see [`Input::pad_just_pressed`]).
    ///
    /// The variant and [`PadButton`] the type share a name; they live in
    /// different namespaces, and any other pairing reads worse at both call
    /// sites.
    PadButton { button: PadButton, pressed: bool },
    /// An analog stick's position: `x` = right, `y` = up/forward, magnitude
    /// `0..=1`.
    ///
    /// **A level, not an edge** — exactly like [`InputEvent::TouchDrive`], and
    /// for the same reason: a physical stick has a position at every instant,
    /// not a stream of changes. It persists across ticks until another
    /// `PadStick` for the same stick replaces it, so a host sends one when the
    /// value moves and a zero when the stick is let go. See [`Input::stick`].
    ///
    /// Deliberately *not* folded into `TouchDrive`: a pad has two of them, and a
    /// game may want the touch stick and the pad's left stick bound to different
    /// things (or summed) rather than fighting over one slot.
    PadStick { stick: PadStick, dir: Vec2 },
    /// An analog trigger's pull, `0..=1`.
    ///
    /// **A level, not an edge**, on the same terms as [`InputEvent::PadStick`]
    /// and [`InputEvent::TouchDrive`]. See [`Input::trigger`].
    PadTrigger { trigger: PadTrigger, value: f32 },
    /// One finger, raw: the host's contact id, what just happened to it, and
    /// where it is in **logical pixels, top-left origin, +Y down** (see
    /// [`Touch`]).
    ///
    /// Deliberately *not* a replacement for [`InputEvent::TouchDrive`], and
    /// deliberately not exclusive with it. `TouchDrive` is one host-synthesised
    /// stick, which is all a one-thumb game ever wanted; this is every contact,
    /// which is what a game that draws its own dpad, button grid and camera drag
    /// needs — and it needs them *simultaneously*, which no single collapsed
    /// value can express. A host may emit both (the default) because a game that
    /// reads only `drive()` is unaffected by touches it never looks at; a game
    /// that builds gestures out of these turns the synthesis off so the two
    /// paths cannot double-drive the same finger.
    ///
    /// **Edges, not a level** — unlike the sticks. The engine keeps the live set
    /// (see [`Input::touches`]); each event is one change to it.
    Touch {
        id: u64,
        phase: TouchPhase,
        x: f32,
        y: f32,
    },
}

// ---------------------------------------------------------------------------
// Input resource
// ---------------------------------------------------------------------------

/// The number of distinct mouse buttons tracked.
pub const MOUSE_BUTTONS: usize = 8;

/// Per-tick input state, rebuilt at every tick boundary from the buffered
/// events. Systems in `FixedSim` read this; nothing else may.
#[derive(Resource, Clone, Debug, Default)]
pub struct Input {
    held: KeySet,
    just_pressed: KeySet,
    just_released: KeySet,
    buttons_held: u8,
    buttons_pressed: u8,
    buttons_released: u8,
    mouse_delta: Vec2,
    wheel: f32,
    drive: Vec2,
    drive_changed: bool,
    pad_held: PadButtonSet,
    pad_pressed: PadButtonSet,
    pad_released: PadButtonSet,
    /// Indexed by [`PadStick::index`]. An array rather than two named fields so
    /// the accessors and the trace recorder can walk `PadStick::ALL` instead of
    /// duplicating themselves per stick.
    sticks: [Vec2; PadStick::COUNT],
    sticks_changed: [bool; PadStick::COUNT],
    /// Indexed by [`PadTrigger::index`]; see `sticks`.
    triggers: [f32; PadTrigger::COUNT],
    triggers_changed: [bool; PadTrigger::COUNT],
    /// Fingers currently on the glass, in arrival order. Level state, like
    /// `held`: a finger that does not move stays exactly where it was.
    touches: TouchList,
    /// The two per-tick edge lists, cleared in [`Input::begin_tick`] like the
    /// key edge sets. A tap that lands and lifts inside one tick appears in
    /// both and in `touches` in neither, which is the whole reason they are
    /// separate lists rather than a phase tag on a live touch.
    touches_started: TouchList,
    touches_ended: TouchList,
}

impl Input {
    pub fn new() -> Input {
        Input {
            held: KeySet::new(),
            just_pressed: KeySet::new(),
            just_released: KeySet::new(),
            buttons_held: 0,
            buttons_pressed: 0,
            buttons_released: 0,
            mouse_delta: Vec2::ZERO,
            wheel: 0.0,
            drive: Vec2::ZERO,
            drive_changed: false,
            pad_held: PadButtonSet::new(),
            pad_pressed: PadButtonSet::new(),
            pad_released: PadButtonSet::new(),
            sticks: [Vec2::ZERO; PadStick::COUNT],
            sticks_changed: [false; PadStick::COUNT],
            triggers: [0.0; PadTrigger::COUNT],
            triggers_changed: [false; PadTrigger::COUNT],
            touches: TouchList::new(),
            touches_started: TouchList::new(),
            touches_ended: TouchList::new(),
        }
    }

    /// Apply one tick's worth of buffered events.
    ///
    /// `held`, `drive`, the pad's held buttons, sticks and triggers persist
    /// across ticks; the edge sets, the analog accumulators and every change
    /// flag are per-tick and reset here. Called once per tick, by the tick loop
    /// only.
    pub fn begin_tick(&mut self, events: impl IntoIterator<Item = InputEvent>) {
        self.just_pressed.clear();
        self.just_released.clear();
        self.buttons_pressed = 0;
        self.buttons_released = 0;
        self.mouse_delta = Vec2::ZERO;
        self.wheel = 0.0;
        self.drive_changed = false;
        self.pad_pressed.clear();
        self.pad_released.clear();
        self.sticks_changed = [false; PadStick::COUNT];
        self.triggers_changed = [false; PadTrigger::COUNT];
        self.touches_started.clear();
        self.touches_ended.clear();
        self.touches.clear_moved();

        for ev in events {
            match ev {
                InputEvent::KeyDown(k) => {
                    // Auto-repeat must not re-fire `just_pressed`.
                    if self.held.insert(k) {
                        self.just_pressed.insert(k);
                    }
                }
                InputEvent::KeyUp(k) => {
                    if self.held.remove(k) {
                        self.just_released.insert(k);
                    }
                }
                InputEvent::MouseMove { dx, dy } => {
                    self.mouse_delta += Vec2::new(dx, dy);
                }
                InputEvent::MouseButton { button, pressed } => {
                    if (button as usize) < MOUSE_BUTTONS {
                        let bit = 1u8 << button;
                        if pressed {
                            if self.buttons_held & bit == 0 {
                                self.buttons_pressed |= bit;
                            }
                            self.buttons_held |= bit;
                        } else {
                            if self.buttons_held & bit != 0 {
                                self.buttons_released |= bit;
                            }
                            self.buttons_held &= !bit;
                        }
                    }
                }
                InputEvent::Wheel { dy } => self.wheel += dy,
                // Level, not edge: the last value in the tick is the value, and
                // it stands until something replaces it.
                InputEvent::TouchDrive { dir } => self.set_drive(dir),
                InputEvent::FocusLost => self.release_all(),
                InputEvent::PadButton { button, pressed } => {
                    // Same edge logic as a key: a host that re-reports a button
                    // still being down must not re-fire `pad_just_pressed`.
                    if pressed {
                        if self.pad_held.insert(button) {
                            self.pad_pressed.insert(button);
                        }
                    } else if self.pad_held.remove(button) {
                        self.pad_released.insert(button);
                    }
                }
                // Levels, not edges: the last value in the tick is the value,
                // and it stands until something replaces it.
                InputEvent::PadStick { stick, dir } => self.set_stick(stick, dir),
                InputEvent::PadTrigger { trigger, value } => self.set_trigger(trigger, value),
                InputEvent::Touch { id, phase, x, y } => {
                    self.apply_touch(id, phase, Vec2::new(x, y))
                }
            }
        }
    }

    /// Drop every held key/mouse/pad button, centre the drive stick and both pad
    /// sticks, and let both triggers go — for focus loss, where the host will
    /// never deliver the matching release.
    ///
    /// The pad is included even though it is not "focused" in the window sense:
    /// a backgrounded tab stops being polled, so the release is exactly as
    /// undeliverable as a key's, and a trigger left at 1.0 would keep driving
    /// the sim forever.
    pub fn release_all(&mut self) {
        for k in self.held.iter().collect::<Vec<_>>() {
            self.just_released.insert(k);
        }
        self.held.clear();
        self.buttons_released |= self.buttons_held;
        self.buttons_held = 0;
        self.set_drive(Vec2::ZERO);
        for b in self.pad_held.iter().collect::<Vec<_>>() {
            self.pad_released.insert(b);
        }
        self.pad_held.clear();
        for stick in PadStick::ALL {
            self.set_stick(stick, Vec2::ZERO);
        }
        for trigger in PadTrigger::ALL {
            self.set_trigger(trigger, 0.0);
        }
        // Every finger ends too, for the same reason a key is released: the
        // `Ended` the host would have delivered is never coming, and a dpad
        // holding a direction because the tab went to the background is the
        // touch version of the ball rolling into the sunset. It is a cancel
        // rather than a lift, and `Input` does not distinguish the two (see
        // [`Input::touches_ended`]), so the touch simply leaves.
        for touch in self.touches.iter() {
            self.touches_ended.push(touch);
        }
        self.touches.clear();
    }

    /// Apply one raw touch event to the live set.
    ///
    /// Non-finite coordinates are *ignored* rather than zeroed, unlike the
    /// sticks: zero is a real position (the top-left corner, where a game may
    /// well have put a button), so a host reporting NaN must not be able to
    /// place a finger there. A `Started` at NaN never becomes a touch at all; a
    /// `Moved` to NaN leaves the finger where it was; an `Ended` at NaN still
    /// ends the finger, at its last known position.
    fn apply_touch(&mut self, id: u64, phase: TouchPhase, pos: Vec2) {
        match phase {
            TouchPhase::Started => {
                if !pos.is_finite() {
                    return;
                }
                // A host re-announcing a live id is a move, not a second
                // finger — the same doctrine that keeps key auto-repeat from
                // re-firing `just_pressed`.
                if self.touches.set_pos(id, pos) {
                    return;
                }
                let touch = Touch { id, pos };
                // Past [`MAX_TOUCHES`] the finger does not exist as far as the
                // sim is concerned, so it must not get a start edge either.
                if self.touches.push(touch) {
                    self.touches_started.push(touch);
                }
            }
            TouchPhase::Moved => {
                if pos.is_finite() {
                    self.touches.set_pos(id, pos);
                }
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                if pos.is_finite() {
                    self.touches.set_pos(id, pos);
                }
                if let Some(touch) = self.touches.remove(id) {
                    self.touches_ended.push(touch);
                }
            }
        }
    }

    /// Set the analog drive, normalising what a host sent.
    ///
    /// Anything non-finite becomes zero and anything past the unit circle is
    /// pulled back onto it, so a sloppy host cannot inject a stick that pushes
    /// harder than the keyboard — and the value a trace records is the value the
    /// sim saw, already sanitised.
    fn set_drive(&mut self, dir: Vec2) {
        let dir = if !dir.is_finite() {
            Vec2::ZERO
        } else if dir.length_squared() > 1.0 {
            dir.normalize()
        } else {
            dir
        };
        if dir != self.drive {
            self.drive = dir;
            self.drive_changed = true;
        }
    }

    /// Set one pad stick, sanitised exactly as [`Input::set_drive`] does — a pad
    /// driver reporting garbage (an uncalibrated axis, a NaN out of a HID
    /// report) must not be able to hand the sim a stick that outruns the
    /// keyboard, and the trace must record the value the sim saw.
    fn set_stick(&mut self, stick: PadStick, dir: Vec2) {
        let dir = if !dir.is_finite() {
            Vec2::ZERO
        } else if dir.length_squared() > 1.0 {
            dir.normalize()
        } else {
            dir
        };
        let slot = stick.index();
        if dir != self.sticks[slot] {
            self.sticks[slot] = dir;
            self.sticks_changed[slot] = true;
        }
    }

    /// Set one pad trigger, clamped to `0..=1` (non-finite becomes zero). Same
    /// doctrine as [`Input::set_stick`].
    fn set_trigger(&mut self, trigger: PadTrigger, value: f32) {
        let value = if value.is_finite() {
            value.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let slot = trigger.index();
        if value != self.triggers[slot] {
            self.triggers[slot] = value;
            self.triggers_changed[slot] = true;
        }
    }

    #[inline]
    pub fn held(&self, key: Key) -> bool {
        self.held.contains(key)
    }
    #[inline]
    pub fn just_pressed(&self, key: Key) -> bool {
        self.just_pressed.contains(key)
    }
    #[inline]
    pub fn just_released(&self, key: Key) -> bool {
        self.just_released.contains(key)
    }
    #[inline]
    pub fn any_held(&self) -> bool {
        !self.held.is_empty()
    }
    #[inline]
    pub fn held_count(&self) -> u32 {
        self.held.len()
    }
    /// Held keys in `Key::ALL` order.
    pub fn held_keys(&self) -> impl Iterator<Item = Key> + '_ {
        self.held.iter()
    }
    /// Keys pressed *this tick*, in `Key::ALL` order. What a trace recorder
    /// turns back into [`InputEvent::KeyDown`]s (see [`crate::trace`]).
    pub fn just_pressed_keys(&self) -> impl Iterator<Item = Key> + '_ {
        self.just_pressed.iter()
    }
    /// Keys released *this tick*, in `Key::ALL` order.
    pub fn just_released_keys(&self) -> impl Iterator<Item = Key> + '_ {
        self.just_released.iter()
    }

    #[inline]
    pub fn button_held(&self, button: u8) -> bool {
        (button as usize) < MOUSE_BUTTONS && self.buttons_held & (1 << button) != 0
    }
    #[inline]
    pub fn button_just_pressed(&self, button: u8) -> bool {
        (button as usize) < MOUSE_BUTTONS && self.buttons_pressed & (1 << button) != 0
    }
    #[inline]
    pub fn button_just_released(&self, button: u8) -> bool {
        (button as usize) < MOUSE_BUTTONS && self.buttons_released & (1 << button) != 0
    }

    /// Pointer motion accumulated during this tick, in pixels.
    #[inline]
    pub fn mouse_delta(&self) -> Vec2 {
        self.mouse_delta
    }
    /// Wheel motion accumulated during this tick.
    #[inline]
    pub fn wheel(&self) -> f32 {
        self.wheel
    }

    /// The analog drive stick: `x` right, `y` forward, magnitude `0..=1`.
    ///
    /// Level state, like [`held`](Input::held) and unlike
    /// [`mouse_delta`](Input::mouse_delta): it survives a tick with no events and
    /// only changes when a [`InputEvent::TouchDrive`] says so.
    #[inline]
    pub fn drive(&self) -> Vec2 {
        self.drive
    }

    /// Whether [`drive`](Input::drive) took a new value *this tick* — what the
    /// trace recorder keys on, so a stick held still costs nothing per tick.
    #[inline]
    pub fn drive_changed(&self) -> bool {
        self.drive_changed
    }

    #[inline]
    pub fn pad_held(&self, button: PadButton) -> bool {
        self.pad_held.contains(button)
    }
    #[inline]
    pub fn pad_just_pressed(&self, button: PadButton) -> bool {
        self.pad_pressed.contains(button)
    }
    #[inline]
    pub fn pad_just_released(&self, button: PadButton) -> bool {
        self.pad_released.contains(button)
    }
    #[inline]
    pub fn any_pad_held(&self) -> bool {
        !self.pad_held.is_empty()
    }
    #[inline]
    pub fn pad_held_count(&self) -> u32 {
        self.pad_held.len()
    }
    /// Held pad buttons in `PadButton::ALL` order.
    pub fn pad_held_buttons(&self) -> impl Iterator<Item = PadButton> + '_ {
        self.pad_held.iter()
    }
    /// Pad buttons pressed *this tick*, in `PadButton::ALL` order.
    pub fn pad_just_pressed_buttons(&self) -> impl Iterator<Item = PadButton> + '_ {
        self.pad_pressed.iter()
    }
    /// Pad buttons released *this tick*, in `PadButton::ALL` order.
    pub fn pad_just_released_buttons(&self) -> impl Iterator<Item = PadButton> + '_ {
        self.pad_released.iter()
    }

    /// One analog stick: `x` right, `y` up/forward, magnitude `0..=1`.
    ///
    /// Level state, like [`drive`](Input::drive): it survives a tick with no
    /// events and only changes when an [`InputEvent::PadStick`] says so.
    #[inline]
    pub fn stick(&self, stick: PadStick) -> Vec2 {
        self.sticks[stick.index()]
    }

    /// Whether [`stick`](Input::stick) took a new value *this tick* — what the
    /// trace recorder keys on, so a stick held still costs nothing per tick.
    #[inline]
    pub fn stick_changed(&self, stick: PadStick) -> bool {
        self.sticks_changed[stick.index()]
    }

    /// One analog trigger's pull, `0..=1`. Level state; see
    /// [`stick`](Input::stick).
    #[inline]
    pub fn trigger(&self, trigger: PadTrigger) -> f32 {
        self.triggers[trigger.index()]
    }

    /// Whether [`trigger`](Input::trigger) took a new value *this tick*.
    #[inline]
    pub fn trigger_changed(&self, trigger: PadTrigger) -> bool {
        self.triggers_changed[trigger.index()]
    }

    /// Every finger on the glass, in arrival order (oldest first).
    ///
    /// Level state, like [`held`](Input::held): a finger that does not move
    /// stays exactly where it was, tick after tick, until it lifts. Arrival
    /// order is deterministic and is what a trace reproduces — a game may rely
    /// on "the first one down is first" and get the same answer on replay.
    ///
    /// This is raw contact data on purpose. Everything a touch *game* wants —
    /// a floating dpad, a chorded button grid, a camera drag, all three at once
    /// under different fingers — is a policy over this set, and policy belongs
    /// to the game (the one built-in exception is the host's optional virtual
    /// stick, which arrives separately as [`drive`](Input::drive)).
    pub fn touches(&self) -> impl Iterator<Item = Touch> + '_ {
        self.touches.iter()
    }

    /// The live touch with this id, if it is still down. What a game that
    /// latched onto a finger last tick calls to find out where it went.
    #[inline]
    pub fn touch(&self, id: u64) -> Option<Touch> {
        self.touches.get(id)
    }

    /// How many fingers are down, `0..=`[`MAX_TOUCHES`].
    #[inline]
    pub fn touch_count(&self) -> usize {
        self.touches.len()
    }

    #[inline]
    pub fn any_touch(&self) -> bool {
        !self.touches.is_empty()
    }

    /// Fingers that landed *this tick*, at the position they landed on — a
    /// per-tick edge list, like [`just_pressed_keys`](Input::just_pressed_keys).
    ///
    /// A tap that lands and lifts inside one tick appears here **and** in
    /// [`touches_ended`](Input::touches_ended) while never appearing in
    /// [`touches`](Input::touches), so a button grid polled once per tick cannot
    /// miss a fast stab.
    pub fn touches_started(&self) -> impl Iterator<Item = Touch> + '_ {
        self.touches_started.iter()
    }

    /// Fingers that left *this tick*, at the last position they were seen.
    ///
    /// A lift, a system cancel and a focus loss all arrive here identically.
    /// They are not distinguished because nothing a game does with a released
    /// finger differs between them — every one of them means "that gesture is
    /// over" — and collapsing them keeps `Input` (and therefore the trace) a
    /// record of what the tick *saw* rather than of what the window manager did.
    pub fn touches_ended(&self) -> impl Iterator<Item = Touch> + '_ {
        self.touches_ended.iter()
    }

    /// The touch with this id if it ended this tick, at its last position.
    ///
    /// The counterpart to [`touch`](Input::touch) for a game holding an id: one
    /// call answers "is my finger gone, and where did it let go?".
    #[inline]
    pub fn touch_ended(&self, id: u64) -> Option<Touch> {
        self.touches_ended.get(id)
    }

    /// Whether this tick moved the live touch with `id`.
    ///
    /// What the trace recorder keys on, so a finger resting still costs nothing
    /// per tick — the same trick as [`drive_changed`](Input::drive_changed).
    #[inline]
    pub fn touch_moved(&self, id: u64) -> bool {
        self.touches.moved(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edges_are_per_tick_and_held_persists() {
        let mut input = Input::new();

        input.begin_tick([InputEvent::KeyDown(Key::W)]);
        assert!(input.held(Key::W));
        assert!(input.just_pressed(Key::W));
        assert!(!input.just_released(Key::W));

        // Next tick with no events: still held, no longer a fresh press.
        input.begin_tick([]);
        assert!(input.held(Key::W));
        assert!(!input.just_pressed(Key::W));

        input.begin_tick([InputEvent::KeyUp(Key::W)]);
        assert!(!input.held(Key::W));
        assert!(input.just_released(Key::W));
    }

    #[test]
    fn key_repeat_does_not_refire_just_pressed() {
        let mut input = Input::new();
        input.begin_tick([InputEvent::KeyDown(Key::Space)]);
        input.begin_tick([InputEvent::KeyDown(Key::Space)]);
        assert!(input.held(Key::Space));
        assert!(!input.just_pressed(Key::Space), "auto-repeat is not a press");
    }

    #[test]
    fn press_and_release_within_one_tick_shows_both_edges() {
        let mut input = Input::new();
        input.begin_tick([InputEvent::KeyDown(Key::Escape), InputEvent::KeyUp(Key::Escape)]);
        assert!(input.just_pressed(Key::Escape));
        assert!(input.just_released(Key::Escape));
        assert!(!input.held(Key::Escape));
    }

    #[test]
    fn analog_accumulates_within_a_tick_and_resets_between() {
        let mut input = Input::new();
        input.begin_tick([
            InputEvent::MouseMove { dx: 1.0, dy: 2.0 },
            InputEvent::MouseMove { dx: 0.5, dy: -1.0 },
            InputEvent::Wheel { dy: 1.0 },
            InputEvent::Wheel { dy: 2.0 },
        ]);
        assert_eq!(input.mouse_delta(), Vec2::new(1.5, 1.0));
        assert_eq!(input.wheel(), 3.0);

        input.begin_tick([]);
        assert_eq!(input.mouse_delta(), Vec2::ZERO);
        assert_eq!(input.wheel(), 0.0);
    }

    #[test]
    fn mouse_buttons_track_edges() {
        let mut input = Input::new();
        input.begin_tick([InputEvent::MouseButton { button: 0, pressed: true }]);
        assert!(input.button_held(0) && input.button_just_pressed(0));
        input.begin_tick([]);
        assert!(input.button_held(0) && !input.button_just_pressed(0));
        input.begin_tick([InputEvent::MouseButton { button: 0, pressed: false }]);
        assert!(!input.button_held(0) && input.button_just_released(0));
    }

    #[test]
    fn held_key_iteration_is_declaration_order() {
        let mut input = Input::new();
        input.begin_tick([
            InputEvent::KeyDown(Key::Space),
            InputEvent::KeyDown(Key::A),
            InputEvent::KeyDown(Key::W),
        ]);
        let order: Vec<Key> = input.held_keys().collect();
        assert_eq!(order, vec![Key::A, Key::W, Key::Space]);
    }

    #[test]
    fn release_all_clears_held() {
        let mut input = Input::new();
        input.begin_tick([InputEvent::KeyDown(Key::D), InputEvent::MouseButton { button: 1, pressed: true }]);
        input.release_all();
        assert!(!input.held(Key::D));
        assert!(input.just_released(Key::D));
        assert!(!input.button_held(1));
        assert!(input.button_just_released(1));
    }

    #[test]
    fn the_drive_stick_is_a_level_that_persists_across_ticks() {
        let mut input = Input::new();
        assert_eq!(input.drive(), Vec2::ZERO);

        input.begin_tick([InputEvent::TouchDrive { dir: Vec2::new(0.5, -0.25) }]);
        assert_eq!(input.drive(), Vec2::new(0.5, -0.25));
        assert!(input.drive_changed());

        // No events at all: unlike the mouse delta, the stick stays where it was.
        input.begin_tick([]);
        assert_eq!(input.drive(), Vec2::new(0.5, -0.25));
        assert!(!input.drive_changed(), "nothing moved it, so nothing changed");

        // Several in one tick: the last one wins.
        input.begin_tick([
            InputEvent::TouchDrive { dir: Vec2::new(0.1, 0.1) },
            InputEvent::TouchDrive { dir: Vec2::new(0.0, 1.0) },
        ]);
        assert_eq!(input.drive(), Vec2::new(0.0, 1.0));

        // Setting it to what it already was is not a change.
        input.begin_tick([InputEvent::TouchDrive { dir: Vec2::new(0.0, 1.0) }]);
        assert!(!input.drive_changed());

        input.begin_tick([InputEvent::TouchDrive { dir: Vec2::ZERO }]);
        assert_eq!(input.drive(), Vec2::ZERO);
        assert!(input.drive_changed());
    }

    #[test]
    fn a_sloppy_stick_is_clamped_rather_than_trusted() {
        let mut input = Input::new();
        input.begin_tick([InputEvent::TouchDrive { dir: Vec2::new(3.0, 4.0) }]);
        assert!((input.drive().length() - 1.0).abs() < 1e-6, "{:?}", input.drive());

        input.begin_tick([InputEvent::TouchDrive { dir: Vec2::new(f32::NAN, 0.0) }]);
        assert_eq!(input.drive(), Vec2::ZERO);
    }

    #[test]
    fn focus_loss_releases_everything_including_the_stick() {
        let mut input = Input::new();
        input.begin_tick([
            InputEvent::KeyDown(Key::W),
            InputEvent::MouseButton { button: 0, pressed: true },
            InputEvent::TouchDrive { dir: Vec2::new(0.0, 1.0) },
        ]);
        assert!(input.held(Key::W));

        input.begin_tick([InputEvent::FocusLost]);
        assert!(!input.held(Key::W), "a key held across a focus loss would stick");
        assert!(input.just_released(Key::W));
        assert!(!input.button_held(0));
        assert_eq!(input.drive(), Vec2::ZERO);
        assert!(input.drive_changed());

        // And the release is not re-announced on the tick after.
        input.begin_tick([]);
        assert!(!input.just_released(Key::W));
        assert!(!input.drive_changed());
    }

    #[test]
    fn pad_button_edges_are_per_tick_and_held_persists() {
        let mut input = Input::new();

        input.begin_tick([InputEvent::PadButton {
            button: PadButton::South,
            pressed: true,
        }]);
        assert!(input.pad_held(PadButton::South));
        assert!(input.pad_just_pressed(PadButton::South));
        assert!(!input.pad_just_released(PadButton::South));

        input.begin_tick([]);
        assert!(input.pad_held(PadButton::South));
        assert!(!input.pad_just_pressed(PadButton::South));

        input.begin_tick([InputEvent::PadButton {
            button: PadButton::South,
            pressed: false,
        }]);
        assert!(!input.pad_held(PadButton::South));
        assert!(input.pad_just_released(PadButton::South));
    }

    #[test]
    fn a_repeated_pad_button_report_does_not_refire_just_pressed() {
        let mut input = Input::new();
        let down = InputEvent::PadButton {
            button: PadButton::R1,
            pressed: true,
        };
        input.begin_tick([down]);
        input.begin_tick([down]);
        assert!(input.pad_held(PadButton::R1));
        assert!(
            !input.pad_just_pressed(PadButton::R1),
            "a host re-reporting a held button is not a new press"
        );
    }

    #[test]
    fn held_pad_button_iteration_is_declaration_order() {
        let mut input = Input::new();
        input.begin_tick([
            InputEvent::PadButton {
                button: PadButton::DpadLeft,
                pressed: true,
            },
            InputEvent::PadButton {
                button: PadButton::South,
                pressed: true,
            },
            InputEvent::PadButton {
                button: PadButton::L1,
                pressed: true,
            },
        ]);
        let order: Vec<PadButton> = input.pad_held_buttons().collect();
        assert_eq!(
            order,
            vec![PadButton::South, PadButton::L1, PadButton::DpadLeft]
        );
    }

    #[test]
    fn pad_sticks_and_triggers_are_levels_that_persist_across_ticks() {
        let mut input = Input::new();
        assert_eq!(input.stick(PadStick::Left), Vec2::ZERO);
        assert_eq!(input.trigger(PadTrigger::R2), 0.0);

        input.begin_tick([
            InputEvent::PadStick {
                stick: PadStick::Left,
                dir: Vec2::new(0.5, -0.25),
            },
            InputEvent::PadTrigger {
                trigger: PadTrigger::R2,
                value: 0.75,
            },
        ]);
        assert_eq!(input.stick(PadStick::Left), Vec2::new(0.5, -0.25));
        assert!(input.stick_changed(PadStick::Left));
        assert_eq!(input.trigger(PadTrigger::R2), 0.75);
        assert!(input.trigger_changed(PadTrigger::R2));
        // The other stick and trigger are untouched — they are separate levels.
        assert_eq!(input.stick(PadStick::Right), Vec2::ZERO);
        assert!(!input.stick_changed(PadStick::Right));
        assert_eq!(input.trigger(PadTrigger::L2), 0.0);

        // No events at all: like the drive stick, they stay where they were.
        input.begin_tick([]);
        assert_eq!(input.stick(PadStick::Left), Vec2::new(0.5, -0.25));
        assert_eq!(input.trigger(PadTrigger::R2), 0.75);
        assert!(!input.stick_changed(PadStick::Left));
        assert!(!input.trigger_changed(PadTrigger::R2));

        // Several in one tick: the last one wins.
        input.begin_tick([
            InputEvent::PadStick {
                stick: PadStick::Left,
                dir: Vec2::new(0.1, 0.1),
            },
            InputEvent::PadStick {
                stick: PadStick::Left,
                dir: Vec2::new(0.0, 1.0),
            },
            InputEvent::PadTrigger {
                trigger: PadTrigger::R2,
                value: 0.1,
            },
            InputEvent::PadTrigger {
                trigger: PadTrigger::R2,
                value: 1.0,
            },
        ]);
        assert_eq!(input.stick(PadStick::Left), Vec2::new(0.0, 1.0));
        assert_eq!(input.trigger(PadTrigger::R2), 1.0);

        // Setting them to what they already were is not a change.
        input.begin_tick([
            InputEvent::PadStick {
                stick: PadStick::Left,
                dir: Vec2::new(0.0, 1.0),
            },
            InputEvent::PadTrigger {
                trigger: PadTrigger::R2,
                value: 1.0,
            },
        ]);
        assert!(!input.stick_changed(PadStick::Left));
        assert!(!input.trigger_changed(PadTrigger::R2));
    }

    #[test]
    fn a_sloppy_pad_axis_is_clamped_rather_than_trusted() {
        let mut input = Input::new();
        input.begin_tick([
            InputEvent::PadStick {
                stick: PadStick::Right,
                dir: Vec2::new(3.0, 4.0),
            },
            InputEvent::PadTrigger {
                trigger: PadTrigger::L2,
                value: 4.0,
            },
        ]);
        assert!(
            (input.stick(PadStick::Right).length() - 1.0).abs() < 1e-6,
            "{:?}",
            input.stick(PadStick::Right)
        );
        assert_eq!(input.trigger(PadTrigger::L2), 1.0);

        input.begin_tick([
            InputEvent::PadStick {
                stick: PadStick::Right,
                dir: Vec2::new(f32::NAN, 0.0),
            },
            InputEvent::PadTrigger {
                trigger: PadTrigger::L2,
                value: f32::NAN,
            },
        ]);
        assert_eq!(input.stick(PadStick::Right), Vec2::ZERO);
        assert_eq!(input.trigger(PadTrigger::L2), 0.0);

        // A driver reporting a negative trigger pull is a zero, not a push.
        input.begin_tick([InputEvent::PadTrigger {
            trigger: PadTrigger::L2,
            value: -0.5,
        }]);
        assert_eq!(input.trigger(PadTrigger::L2), 0.0);
    }

    #[test]
    fn focus_loss_neutralises_the_pad() {
        let mut input = Input::new();
        input.begin_tick([
            InputEvent::PadButton {
                button: PadButton::North,
                pressed: true,
            },
            InputEvent::PadStick {
                stick: PadStick::Left,
                dir: Vec2::new(0.0, 1.0),
            },
            InputEvent::PadStick {
                stick: PadStick::Right,
                dir: Vec2::new(1.0, 0.0),
            },
            InputEvent::PadTrigger {
                trigger: PadTrigger::R2,
                value: 1.0,
            },
        ]);
        assert!(input.pad_held(PadButton::North));

        input.begin_tick([InputEvent::FocusLost]);
        assert!(
            !input.pad_held(PadButton::North),
            "a pad button held across a focus loss would stick"
        );
        assert!(input.pad_just_released(PadButton::North));
        for stick in PadStick::ALL {
            assert_eq!(input.stick(stick), Vec2::ZERO);
            assert!(input.stick_changed(stick));
        }
        assert_eq!(input.trigger(PadTrigger::R2), 0.0);
        assert!(input.trigger_changed(PadTrigger::R2));
        // The untouched trigger was already zero, so nothing changed about it.
        assert!(!input.trigger_changed(PadTrigger::L2));

        // And none of it is re-announced on the tick after.
        input.begin_tick([]);
        assert!(!input.pad_just_released(PadButton::North));
        assert!(!input.stick_changed(PadStick::Left));
        assert!(!input.trigger_changed(PadTrigger::R2));
    }

    #[test]
    fn a_pad_button_pressed_and_released_within_one_tick_shows_both_edges() {
        let mut input = Input::new();
        input.begin_tick([
            InputEvent::PadButton {
                button: PadButton::Start,
                pressed: true,
            },
            InputEvent::PadButton {
                button: PadButton::Start,
                pressed: false,
            },
        ]);
        assert!(input.pad_just_pressed(PadButton::Start));
        assert!(input.pad_just_released(PadButton::Start));
        assert!(!input.pad_held(PadButton::Start));
    }

    // -- touch ---------------------------------------------------------------

    /// Shorthand for the raw event, which is otherwise four fields of noise.
    fn touch(id: u64, phase: TouchPhase, x: f32, y: f32) -> InputEvent {
        InputEvent::Touch { id, phase, x, y }
    }

    #[test]
    fn a_finger_is_a_level_with_edges_at_both_ends() {
        let mut input = Input::new();
        assert_eq!(input.touch_count(), 0);
        assert!(!input.any_touch());

        input.begin_tick([touch(4, TouchPhase::Started, 10.0, 20.0)]);
        assert_eq!(
            input.touches().collect::<Vec<_>>(),
            vec![Touch {
                id: 4,
                pos: Vec2::new(10.0, 20.0)
            }]
        );
        assert_eq!(input.touches_started().count(), 1);
        assert_eq!(input.touches_ended().count(), 0);
        assert!(
            !input.touch_moved(4),
            "arriving is not moving — the start edge already carries where it landed"
        );

        // A tick with no events at all: the finger is still down, still there,
        // and no longer a fresh arrival.
        input.begin_tick([]);
        assert_eq!(input.touch(4).map(|t| t.pos), Some(Vec2::new(10.0, 20.0)));
        assert_eq!(input.touches_started().count(), 0);
        assert!(!input.touch_moved(4), "a resting finger has not moved");

        input.begin_tick([touch(4, TouchPhase::Moved, 12.0, 20.0)]);
        assert_eq!(input.touch(4).map(|t| t.pos), Some(Vec2::new(12.0, 20.0)));
        assert!(input.touch_moved(4));

        // A move to where it already is is not a move — the same doctrine that
        // keeps a still stick out of the trace.
        input.begin_tick([touch(4, TouchPhase::Moved, 12.0, 20.0)]);
        assert!(!input.touch_moved(4));

        input.begin_tick([touch(4, TouchPhase::Ended, 12.0, 30.0)]);
        assert_eq!(input.touch(4), None, "a lifted finger is not live");
        assert_eq!(
            input.touch_ended(4).map(|t| t.pos),
            Some(Vec2::new(12.0, 30.0)),
            "the end carries where it let go"
        );
        assert!(!input.any_touch());

        // And the release is not re-announced on the tick after.
        input.begin_tick([]);
        assert_eq!(input.touch_ended(4), None);
    }

    #[test]
    fn a_tap_inside_one_tick_shows_both_edges_and_no_live_finger() {
        // A button grid polled once per tick must not miss a fast stab.
        let mut input = Input::new();
        input.begin_tick([
            touch(1, TouchPhase::Started, 5.0, 5.0),
            touch(1, TouchPhase::Ended, 5.0, 6.0),
        ]);
        assert_eq!(
            input.touches_started().map(|t| t.pos).collect::<Vec<_>>(),
            vec![Vec2::new(5.0, 5.0)],
            "the start edge survives even though the finger is gone"
        );
        assert_eq!(
            input.touches_ended().map(|t| t.pos).collect::<Vec<_>>(),
            vec![Vec2::new(5.0, 6.0)]
        );
        assert_eq!(input.touch_count(), 0);
    }

    #[test]
    fn fingers_are_independent_and_iterate_in_arrival_order() {
        // The whole point of raw touch: a dpad thumb and a camera thumb at once,
        // neither able to disturb the other.
        let mut input = Input::new();
        input.begin_tick([
            touch(7, TouchPhase::Started, 100.0, 100.0),
            touch(3, TouchPhase::Started, 300.0, 100.0),
        ]);
        assert_eq!(
            input.touches().map(|t| t.id).collect::<Vec<_>>(),
            vec![7, 3],
            "arrival order, not id order — a gesture must not reshuffle"
        );

        input.begin_tick([touch(3, TouchPhase::Moved, 320.0, 140.0)]);
        assert_eq!(input.touch(7).map(|t| t.pos), Some(Vec2::new(100.0, 100.0)));
        assert_eq!(input.touch(3).map(|t| t.pos), Some(Vec2::new(320.0, 140.0)));
        assert!(input.touch_moved(3) && !input.touch_moved(7));

        // One lifting leaves the other exactly where it was, still first.
        input.begin_tick([touch(3, TouchPhase::Ended, 320.0, 140.0)]);
        assert_eq!(input.touches().map(|t| t.id).collect::<Vec<_>>(), vec![7]);
        assert_eq!(input.touch(7).map(|t| t.pos), Some(Vec2::new(100.0, 100.0)));

        // A third finger appends behind the survivor.
        input.begin_tick([touch(9, TouchPhase::Started, 0.0, 0.0)]);
        assert_eq!(
            input.touches().map(|t| t.id).collect::<Vec<_>>(),
            vec![7, 9]
        );
    }

    #[test]
    fn a_cancel_ends_a_finger_exactly_as_a_lift_does() {
        let mut input = Input::new();
        input.begin_tick([touch(2, TouchPhase::Started, 1.0, 1.0)]);
        input.begin_tick([touch(2, TouchPhase::Cancelled, 1.0, 1.0)]);
        assert_eq!(input.touch(2), None);
        assert_eq!(input.touch_ended(2).map(|t| t.id), Some(2));
    }

    #[test]
    fn focus_loss_ends_every_finger() {
        let mut input = Input::new();
        input.begin_tick([
            touch(1, TouchPhase::Started, 10.0, 10.0),
            touch(2, TouchPhase::Started, 20.0, 20.0),
        ]);
        assert_eq!(input.touch_count(), 2);

        input.begin_tick([InputEvent::FocusLost]);
        assert_eq!(
            input.touch_count(),
            0,
            "a finger held across a focus loss would drive a dpad forever"
        );
        assert_eq!(
            input.touches_ended().map(|t| t.id).collect::<Vec<_>>(),
            vec![1, 2],
            "and it has to arrive as an end, or nothing is ever told to let go"
        );

        // Not re-announced afterwards, and a `Moved` for a forgotten finger
        // cannot resurrect it.
        input.begin_tick([touch(1, TouchPhase::Moved, 11.0, 11.0)]);
        assert_eq!(input.touches_ended().count(), 0);
        assert_eq!(input.touch_count(), 0);
    }

    #[test]
    fn a_re_announced_id_is_a_move_not_a_second_finger() {
        let mut input = Input::new();
        input.begin_tick([
            touch(5, TouchPhase::Started, 0.0, 0.0),
            touch(5, TouchPhase::Started, 4.0, 0.0),
        ]);
        assert_eq!(input.touch_count(), 1);
        assert_eq!(input.touches_started().count(), 1);
        assert_eq!(input.touch(5).map(|t| t.pos), Some(Vec2::new(4.0, 0.0)));
    }

    #[test]
    fn a_non_finite_position_never_places_a_finger() {
        // Zero would be a *real* position — the top-left corner, where a game
        // may well have put a button — so garbage is dropped, not zeroed.
        let mut input = Input::new();
        input.begin_tick([touch(1, TouchPhase::Started, f32::NAN, 0.0)]);
        assert_eq!(input.touch_count(), 0);

        input.begin_tick([
            touch(2, TouchPhase::Started, 50.0, 50.0),
            touch(2, TouchPhase::Moved, f32::INFINITY, 50.0),
        ]);
        assert_eq!(
            input.touch(2).map(|t| t.pos),
            Some(Vec2::new(50.0, 50.0)),
            "a garbage move leaves the finger where it was"
        );

        // A garbage end still ends it, at the last position anyone believed.
        input.begin_tick([touch(2, TouchPhase::Ended, f32::NAN, f32::NAN)]);
        assert_eq!(input.touch_count(), 0);
        assert_eq!(
            input.touch_ended(2).map(|t| t.pos),
            Some(Vec2::new(50.0, 50.0))
        );
    }

    #[test]
    fn the_live_set_is_capped_and_the_overflow_never_exists() {
        let mut input = Input::new();
        let events: Vec<InputEvent> = (0..MAX_TOUCHES as u64 + 3)
            .map(|id| touch(id, TouchPhase::Started, id as f32, 0.0))
            .collect();
        input.begin_tick(events);
        assert_eq!(input.touch_count(), MAX_TOUCHES);
        assert_eq!(
            input.touches_started().count(),
            MAX_TOUCHES,
            "a dropped finger must not get a start edge either"
        );
        // …and the ones past the cap are not there to move or to end.
        input.begin_tick([touch(MAX_TOUCHES as u64 + 1, TouchPhase::Ended, 0.0, 0.0)]);
        assert_eq!(input.touches_ended().count(), 0);
        assert_eq!(input.touch_count(), MAX_TOUCHES);
    }

    #[test]
    fn touch_and_the_drive_stick_are_separate_channels() {
        // The host may synthesise a stick *and* forward the fingers; nothing
        // one does may leak into the other, or a game reading both would be
        // driven twice by one thumb.
        let mut input = Input::new();
        input.begin_tick([
            touch(1, TouchPhase::Started, 10.0, 10.0),
            InputEvent::TouchDrive {
                dir: Vec2::new(0.0, 1.0),
            },
        ]);
        assert_eq!(input.drive(), Vec2::new(0.0, 1.0));
        assert_eq!(input.touch_count(), 1);

        input.begin_tick([touch(1, TouchPhase::Moved, 10.0, 60.0)]);
        assert_eq!(
            input.drive(),
            Vec2::new(0.0, 1.0),
            "a raw finger does not move the synthesised stick"
        );
        assert!(!input.drive_changed());
    }

    #[test]
    fn pad_and_keyboard_state_are_independent() {
        // The pad is additional vocabulary, not a re-skin of the keyboard: a
        // focus loss aside, nothing one does may show up in the other.
        let mut input = Input::new();
        input.begin_tick([
            InputEvent::KeyDown(Key::W),
            InputEvent::PadButton {
                button: PadButton::South,
                pressed: true,
            },
            InputEvent::PadStick {
                stick: PadStick::Left,
                dir: Vec2::new(1.0, 0.0),
            },
        ]);
        assert!(input.held(Key::W) && input.pad_held(PadButton::South));
        assert_eq!(input.held_count(), 1);
        assert_eq!(input.pad_held_count(), 1);
        assert_eq!(
            input.drive(),
            Vec2::ZERO,
            "the pad's left stick is not the touch drive stick"
        );
        assert!(!input.drive_changed());
    }
}
