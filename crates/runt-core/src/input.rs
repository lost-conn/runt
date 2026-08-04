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
