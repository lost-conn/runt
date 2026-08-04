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
// Events
// ---------------------------------------------------------------------------

/// One host input event. This is the *entire* engine input vocabulary; a replay
/// trace is a `Vec<(tick, InputEvent)>` — see [`crate::trace`].
///
/// Postcard encodes a variant as its **index**, so new events are appended here
/// rather than inserted: a trace file recorded before `TouchDrive` and
/// `FocusLost` existed still reads back correctly. (The reverse does not hold,
/// and a trace remains a debugging artifact rather than a save format.)
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
        }
    }

    /// Apply one tick's worth of buffered events.
    ///
    /// `held` and `drive` persist across ticks; the edge sets, the analog
    /// accumulators and the drive's change flag are per-tick and reset here.
    /// Called once per tick, by the tick loop only.
    pub fn begin_tick(&mut self, events: impl IntoIterator<Item = InputEvent>) {
        self.just_pressed.clear();
        self.just_released.clear();
        self.buttons_pressed = 0;
        self.buttons_released = 0;
        self.mouse_delta = Vec2::ZERO;
        self.wheel = 0.0;
        self.drive_changed = false;

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
            }
        }
    }

    /// Drop every held key/button and centre the drive stick — for focus loss,
    /// where the host will never deliver the matching release.
    pub fn release_all(&mut self) {
        for k in self.held.iter().collect::<Vec<_>>() {
            self.just_released.insert(k);
        }
        self.held.clear();
        self.buttons_released |= self.buttons_held;
        self.buttons_held = 0;
        self.set_drive(Vec2::ZERO);
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
}
