//! Gamepad polling: a snapshot the host takes each frame, diffed into engine
//! events.
//!
//! A pad is the one input device nobody delivers as events. gilrs has an event
//! queue but it is a queue over *its own* polling; the Web Gamepad API has no
//! events at all — `navigator.getGamepads()` hands you an array of current
//! state and it is your problem to notice what moved. So the host polls both,
//! flattens the result into a [`PadSnapshot`], and [`PadDiffer`] turns two
//! consecutive snapshots into the edges and levels the engine's vocabulary
//! wants (DESIGN §2: a host translates events and does nothing else).
//!
//! The diffing half is platform-free and unit-tested; the two pollers are the
//! only cfg'd code, and each does nothing but fill in a [`PadSnapshot`].
//!
//! ## One virtual pad
//!
//! Every connected pad is merged into one snapshot before diffing (see
//! [`merge`]). runt hosts single-player games, and "player two's stick also
//! drives" is a better failure than "the game ignores the pad you picked up".
//! Splitting them is a change to this function and nothing else.

use glam::Vec2;

use runt_core::input::{InputEvent, PadButton, PadStick, PadTrigger};

/// Smallest level change worth an event, shared with the touch stick.
///
/// The same 1/64 for the same reason: it is finer than anything a thumb can
/// aim, and a pad held *almost* still would otherwise write an event into the
/// trace every single tick. Imported rather than redefined so the two input
/// paths can never drift apart.
pub use crate::input::STICK_EPSILON;

/// Deflection below which a stick or trigger reads as exactly zero.
///
/// A pad at rest does not report zero. Worn potentiometers sit at 0.01–0.03 and
/// wander, and a trigger's resting value is whatever its spring settled on
/// today. Without this gate a controller sitting on the desk writes a
/// `PadStick` into the trace on most frames — the recording bloats, and every
/// replay of it is at the mercy of a float that was noise to begin with
/// (DESIGN §4: what reaches the sim must be what the player did).
///
/// Applied *before* the epsilon comparison, so a pad drifting inside the gate
/// is not merely quiet — it is identically zero, frame after frame.
pub const PAD_NOISE_GATE: f32 = 0.05;

// ---------------------------------------------------------------------------
// Snapshot
// ---------------------------------------------------------------------------

/// One pad's complete state at one instant, in the engine's vocabulary.
///
/// Everything a poller produces and everything [`PadDiffer`] consumes. Plain
/// data with no history, so a platform poller cannot get edge detection subtly
/// wrong in its own private way.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PadSnapshot {
    /// Bit `i` is [`PadButton::ALL`]`[i]` — i.e. `PadButton::index()` — held.
    pub buttons: u16,
    /// Left stick, `x` right and `y` **forward**.
    pub left: Vec2,
    /// Right stick, same convention.
    pub right: Vec2,
    /// Indexed by [`PadTrigger::index`]: `[L2, R2]`, each `0..=1`.
    pub triggers: [f32; PadTrigger::COUNT],
}

impl PadSnapshot {
    pub fn new() -> PadSnapshot {
        PadSnapshot::default()
    }

    #[inline]
    pub fn is_pressed(&self, button: PadButton) -> bool {
        self.buttons & (1 << button.index()) != 0
    }

    /// Press or release one button. Pollers build a snapshot with this rather
    /// than shifting bits at the call site.
    #[inline]
    pub fn set(&mut self, button: PadButton, pressed: bool) {
        let bit = 1u16 << button.index();
        if pressed {
            self.buttons |= bit;
        } else {
            self.buttons &= !bit;
        }
    }

    /// Builder form of [`set`](PadSnapshot::set), for tests and tables.
    #[inline]
    pub fn with(mut self, button: PadButton, pressed: bool) -> PadSnapshot {
        self.set(button, pressed);
        self
    }

    #[inline]
    pub fn stick(&self, stick: PadStick) -> Vec2 {
        match stick {
            PadStick::Left => self.left,
            PadStick::Right => self.right,
        }
    }

    #[inline]
    pub fn set_stick(&mut self, stick: PadStick, dir: Vec2) {
        match stick {
            PadStick::Left => self.left = dir,
            PadStick::Right => self.right = dir,
        }
    }

    #[inline]
    pub fn trigger(&self, trigger: PadTrigger) -> f32 {
        self.triggers[trigger.index()]
    }

    #[inline]
    pub fn set_trigger(&mut self, trigger: PadTrigger, value: f32) {
        self.triggers[trigger.index()] = value;
    }

    /// Snap the analog axes to zero inside [`PAD_NOISE_GATE`] and drop
    /// non-finite garbage a driver may have handed us.
    ///
    /// The stick is gated on *magnitude*, not per axis, so a diagonal rest
    /// position dies as one thing rather than surviving on whichever component
    /// happened to be larger.
    fn gated(mut self) -> PadSnapshot {
        for stick in PadStick::ALL {
            let dir = self.stick(stick);
            let quiet = !dir.is_finite() || dir.length() <= PAD_NOISE_GATE;
            self.set_stick(stick, if quiet { Vec2::ZERO } else { dir });
        }
        for trigger in PadTrigger::ALL {
            let value = self.trigger(trigger);
            let quiet = !value.is_finite() || value <= PAD_NOISE_GATE;
            self.set_trigger(trigger, if quiet { 0.0 } else { value });
        }
        self
    }
}

/// Fold every connected pad into the one virtual pad the engine sees.
///
/// - **Buttons**: OR. Any pad's press is the press.
/// - **Sticks**: the largest deflection wins, per stick. Not a sum — two players
///   pushing opposite ways would cancel to a dead stick, which reads as a broken
///   controller rather than as a shared one. Ties keep the earlier pad, so the
///   result does not depend on iteration timing.
/// - **Triggers**: max, for the same reason.
pub fn merge(snaps: impl Iterator<Item = PadSnapshot>) -> PadSnapshot {
    let mut out = PadSnapshot::new();
    for snap in snaps {
        out.buttons |= snap.buttons;
        for stick in PadStick::ALL {
            // `>` (not `>=`) so the first pad to reach a magnitude keeps it, and
            // NaN — which fails every comparison — never displaces a real value.
            if snap.stick(stick).length_squared() > out.stick(stick).length_squared() {
                out.set_stick(stick, snap.stick(stick));
            }
        }
        for trigger in PadTrigger::ALL {
            if snap.trigger(trigger) > out.trigger(trigger) {
                out.set_trigger(trigger, snap.trigger(trigger));
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Differ
// ---------------------------------------------------------------------------

/// Turns consecutive [`PadSnapshot`]s into [`InputEvent`]s.
///
/// Holds the last state it *emitted*, not the last state it saw — which is what
/// makes the epsilon a floor on event rate rather than a floor on precision. A
/// stick creeping by 1/1000 per frame is silent for sixty frames and then
/// reports the whole 1/16 it actually travelled, instead of creeping forever
/// unheard because each individual frame was too small to mention.
#[derive(Clone, Copy, Debug, Default)]
pub struct PadDiffer {
    /// The state the engine believes, i.e. the last thing pushed at it.
    sent: PadSnapshot,
}

impl PadDiffer {
    pub fn new() -> PadDiffer {
        PadDiffer::default()
    }

    /// The state the engine was last told about. Mostly for tests.
    pub fn sent(&self) -> PadSnapshot {
        self.sent
    }

    /// Push whatever changed between the last snapshot and `now`.
    ///
    /// Buttons come out in [`PadButton::ALL`] order, then sticks in
    /// [`PadStick::ALL`] order, then triggers — a fixed order, because two
    /// events in one frame land in the same tick's buffer and the sim must see
    /// them the same way on every machine and every replay (DESIGN §3).
    pub fn diff(&mut self, now: PadSnapshot, mut push: impl FnMut(InputEvent)) {
        let now = now.gated();

        let changed = now.buttons ^ self.sent.buttons;
        if changed != 0 {
            for button in PadButton::ALL {
                let bit = 1u16 << button.index();
                if changed & bit != 0 {
                    push(InputEvent::PadButton {
                        button,
                        pressed: now.buttons & bit != 0,
                    });
                }
            }
            self.sent.buttons = now.buttons;
        }

        for stick in PadStick::ALL {
            let next = now.stick(stick);
            let sent = self.sent.stick(stick);
            // A return to *exactly* centre always goes out: it is the event that
            // stops the ball, and letting the epsilon eat it because the last
            // twitch was small is how a pad drives the sim after it was let go.
            if next == sent || ((next - sent).length() < STICK_EPSILON && next != Vec2::ZERO) {
                continue;
            }
            self.sent.set_stick(stick, next);
            push(InputEvent::PadStick { stick, dir: next });
        }

        for trigger in PadTrigger::ALL {
            let next = now.trigger(trigger);
            let sent = self.sent.trigger(trigger);
            // Same zero-crossing rule as the sticks, for the same reason.
            if next == sent || ((next - sent).abs() < STICK_EPSILON && next != 0.0) {
                continue;
            }
            self.sent.set_trigger(trigger, next);
            push(InputEvent::PadTrigger { trigger, value: next });
        }
    }

    /// Let go of everything the engine currently believes is held, and forget it.
    ///
    /// For a pad that vanished mid-press — unplugged, batteries flat, a phone
    /// backgrounding the tab — where the release will never be polled. Emits
    /// *only* what is actually held, so calling it on a resting pad writes
    /// nothing into the trace.
    pub fn reset(&mut self, mut push: impl FnMut(InputEvent)) {
        for button in PadButton::ALL {
            if self.sent.is_pressed(button) {
                push(InputEvent::PadButton {
                    button,
                    pressed: false,
                });
            }
        }
        for stick in PadStick::ALL {
            if self.sent.stick(stick) != Vec2::ZERO {
                push(InputEvent::PadStick {
                    stick,
                    dir: Vec2::ZERO,
                });
            }
        }
        for trigger in PadTrigger::ALL {
            if self.sent.trigger(trigger) != 0.0 {
                push(InputEvent::PadTrigger {
                    trigger,
                    value: 0.0,
                });
            }
        }
        self.sent = PadSnapshot::new();
    }

    /// Forget the held state without telling the engine — for focus loss, where
    /// [`InputEvent::FocusLost`] already makes the engine drop every pad button,
    /// centre both sticks and release both triggers (`Input::release_all`).
    ///
    /// Exactly [`input::VirtualStick::reset`](crate::input::VirtualStick::reset)'s
    /// stance: the neutralisation is not in doubt, so sending it twice would put
    /// a dozen redundant events in every trace for no gain.
    pub fn forget(&mut self) {
        self.sent = PadSnapshot::new();
    }
}

// ---------------------------------------------------------------------------
// Native poller: gilrs
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
pub use native::NativePads;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use glam::Vec2;
    use gilrs::{Axis, Button, Gamepad, Gilrs};

    use runt_core::input::PadButton;

    use super::{merge, PadSnapshot};

    /// gilrs button → [`PadButton`].
    ///
    /// The naming trap: gilrs `LeftTrigger` is the **bumper** (L1) and
    /// `LeftTrigger2` is the analog trigger underneath it (L2) — which is not in
    /// this table at all, because L2/R2 are levels and live in
    /// [`PadTrigger`](runt_core::input::PadTrigger). `LeftThumb`/`RightThumb`
    /// are the stick clicks, L3/R3.
    ///
    /// `C`, `Z` and `Mode` (guide/home) collapse to
    /// [`PadButton::Other`]: gilrs reports them, our vocabulary deliberately
    /// refuses to give them gameplay meaning. `Unknown` is absent because
    /// `Gamepad::is_pressed` panics on it.
    const BUTTONS: [(Button, PadButton); 17] = [
        (Button::South, PadButton::South),
        (Button::East, PadButton::East),
        (Button::West, PadButton::West),
        (Button::North, PadButton::North),
        (Button::LeftTrigger, PadButton::L1),
        (Button::RightTrigger, PadButton::R1),
        (Button::LeftThumb, PadButton::L3),
        (Button::RightThumb, PadButton::R3),
        (Button::Start, PadButton::Start),
        (Button::Select, PadButton::Select),
        (Button::DPadUp, PadButton::DpadUp),
        (Button::DPadDown, PadButton::DpadDown),
        (Button::DPadLeft, PadButton::DpadLeft),
        (Button::DPadRight, PadButton::DpadRight),
        (Button::Mode, PadButton::Other),
        (Button::C, PadButton::Other),
        (Button::Z, PadButton::Other),
        // `LeftTrigger2`/`RightTrigger2` are absent on purpose: L2/R2 are levels
        // and go out as `PadTrigger`, not as bits.
    ];

    /// Every connected pad, polled once per frame.
    ///
    /// Owns the gilrs context because gilrs's state is only as fresh as the last
    /// time its event queue was drained — see [`poll`](NativePads::poll).
    pub struct NativePads {
        gilrs: Gilrs,
    }

    impl NativePads {
        /// `None` when the platform has no gamepad support at all (no udev, a
        /// container with no `/dev/input`, a build gilrs does not implement).
        /// The host runs perfectly well without a pad, so this is a warning and
        /// not an error.
        pub fn new() -> Option<NativePads> {
            match Gilrs::new() {
                Ok(gilrs) => Some(NativePads { gilrs }),
                Err(e) => {
                    log::warn!("gamepad support unavailable: {e}");
                    None
                }
            }
        }

        /// Drain gilrs's queue, then read the state it left behind.
        ///
        /// The drain is not optional and the events are not interesting: gilrs
        /// updates its cached state *while* dequeuing, so a `next_event` loop is
        /// how the numbers below become this frame's numbers rather than the
        /// numbers from whenever the queue was last touched. Edge detection is
        /// [`PadDiffer`](super::PadDiffer)'s job on both platforms, so consuming
        /// the events here costs nothing.
        pub fn poll(&mut self) -> PadSnapshot {
            while self.gilrs.next_event().is_some() {}
            merge(
                self.gilrs
                    .gamepads()
                    .map(|(_id, pad)| snapshot(&pad))
                    .collect::<Vec<_>>()
                    .into_iter(),
            )
        }
    }

    /// One gilrs gamepad's current state.
    ///
    /// **Axis convention:** gilrs reports sticks with `y` positive *up*, which is
    /// already the engine's convention — `y` forward, as
    /// [`VirtualStick`](crate::input::VirtualStick) produces from a touch drag.
    /// No flip here; the web poller has to flip because the browser does not
    /// agree.
    fn snapshot(pad: &Gamepad<'_>) -> PadSnapshot {
        let mut snap = PadSnapshot::new();
        if !pad.is_connected() {
            return snap;
        }
        for (from, to) in BUTTONS {
            if pad.is_pressed(from) {
                snap.set(to, true);
            }
        }
        snap.left = Vec2::new(pad.value(Axis::LeftStickX), pad.value(Axis::LeftStickY));
        snap.right = Vec2::new(pad.value(Axis::RightStickX), pad.value(Axis::RightStickY));
        snap.triggers = [
            trigger(pad, Button::LeftTrigger2, Axis::LeftZ),
            trigger(pad, Button::RightTrigger2, Axis::RightZ),
        ];
        snap
    }

    /// An analog trigger's pull, `0..=1`.
    ///
    /// Preferring the button form is what makes a digital trigger work: a pad
    /// whose L2 is a switch has no `LeftZ` axis at all, and reports 0/1 through
    /// `button_data`. The axis is the fallback for the pads that only expose it
    /// there, clamped because a raw `LeftZ` may be a full -1..=1 axis.
    fn trigger(pad: &Gamepad<'_>, button: Button, axis: Axis) -> f32 {
        let raw = match pad.button_data(button) {
            Some(data) => data.value(),
            None => pad.value(axis),
        };
        if raw.is_finite() {
            raw.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Web poller: navigator.getGamepads()
// ---------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
pub use web::WebPads;

#[cfg(target_arch = "wasm32")]
mod web {
    use glam::Vec2;
    use wasm_bindgen::JsCast;
    use web_sys::{Gamepad, GamepadButton, GamepadMappingType};

    use runt_core::input::PadButton;

    use super::{merge, PadSnapshot};

    /// W3C standard-mapping button index → [`PadButton`].
    ///
    /// **Not** the identity, despite both orders claiming to be "standard": ours
    /// puts the stick clicks where the browser puts the analog triggers, and
    /// swaps Start with Select.
    ///
    /// ```text
    /// w3c  control          PadButton      our index
    ///  0   bottom face      South           0
    ///  1   right face       East            1
    ///  2   left face        West            2
    ///  3   top face         North           3
    ///  4   left bumper      L1              4
    ///  5   right bumper     R1              5
    ///  6   left trigger     -- (PadTrigger::L2, a level)
    ///  7   right trigger    -- (PadTrigger::R2, a level)
    ///  8   select/back      Select          9   <- swapped
    ///  9   start            Start           8   <- swapped
    /// 10   left stick in    L3              6   <- moved
    /// 11   right stick in   R3              7   <- moved
    /// 12   dpad up          DpadUp         10
    /// 13   dpad down        DpadDown       11
    /// 14   dpad left        DpadLeft       12
    /// 15   dpad right       DpadRight      13
    /// 16   guide/home       Other          14
    /// ```
    ///
    /// `None` is an index that is not a button in our vocabulary. Anything past
    /// 16 (a pad reporting extras through the standard mapping) is `Other`.
    const STANDARD: [Option<PadButton>; 17] = [
        Some(PadButton::South),
        Some(PadButton::East),
        Some(PadButton::West),
        Some(PadButton::North),
        Some(PadButton::L1),
        Some(PadButton::R1),
        None, // L2 — read as a level below.
        None, // R2 — ditto.
        Some(PadButton::Select),
        Some(PadButton::Start),
        Some(PadButton::L3),
        Some(PadButton::R3),
        Some(PadButton::DpadUp),
        Some(PadButton::DpadDown),
        Some(PadButton::DpadLeft),
        Some(PadButton::DpadRight),
        Some(PadButton::Other), // guide/home
    ];

    /// W3C standard-mapping index of the left analog trigger; the right is +1.
    const TRIGGER_BASE: u32 = 6;

    pub struct WebPads {
        /// A non-standard pad is reported once and then ignored, because this
        /// runs every frame and a console line per frame is a denial of service
        /// against the developer.
        warned: bool,
    }

    impl WebPads {
        /// Always `Some`: the Gamepad API is either there or `getGamepads()`
        /// returns an empty list, and neither is a failure the host can act on.
        /// `Option` only so the two platforms present the same constructor.
        pub fn new() -> Option<WebPads> {
            Some(WebPads { warned: false })
        }

        /// Read `navigator.getGamepads()` and fold it into one snapshot.
        ///
        /// The array is sparse — disconnected slots come back `null` — and
        /// browsers hand out a *fresh* set of objects per call, which is why
        /// there is nothing to cache between frames.
        pub fn poll(&mut self) -> PadSnapshot {
            let Some(win) = web_sys::window() else {
                return PadSnapshot::new();
            };
            let Ok(list) = win.navigator().get_gamepads() else {
                return PadSnapshot::new();
            };

            let mut snaps = Vec::new();
            for entry in list.iter() {
                let Ok(pad) = entry.dyn_into::<Gamepad>() else {
                    continue; // A null slot.
                };
                if !pad.connected() {
                    continue;
                }
                // Standard mapping only. Without it the indices below mean
                // whatever the driver felt like, and guessing is how a pad ends
                // up firing the weapon when you press start.
                if pad.mapping() != GamepadMappingType::Standard {
                    if !self.warned {
                        self.warned = true;
                        log::warn!(
                            "gamepad {:?} has no standard mapping; ignoring it",
                            pad.id()
                        );
                    }
                    continue;
                }
                snaps.push(snapshot(&pad));
            }
            merge(snaps.into_iter())
        }
    }

    fn snapshot(pad: &Gamepad) -> PadSnapshot {
        let mut snap = PadSnapshot::new();

        let buttons = pad.buttons();
        for index in 0..buttons.length() {
            if !pressed(&buttons, index) {
                continue;
            }
            let mapped = match STANDARD.get(index as usize) {
                Some(Some(button)) => *button,
                Some(None) => continue,   // An analog trigger; a level, not a bit.
                None => PadButton::Other, // Past the standard mapping's 17.
            };
            snap.set(mapped, true);
        }

        snap.triggers = [
            value(&buttons, TRIGGER_BASE),
            value(&buttons, TRIGGER_BASE + 1),
        ];

        // Browsers report `y` positive *downwards*; gilrs and the engine want
        // `y` forward. Flip here so the two hosts hand the sim the same numbers
        // for the same physical push.
        let axes = pad.axes();
        snap.left = Vec2::new(axis(&axes, 0), -axis(&axes, 1));
        snap.right = Vec2::new(axis(&axes, 2), -axis(&axes, 3));
        snap
    }

    fn button_at(buttons: &js_sys::Array, index: u32) -> Option<GamepadButton> {
        buttons.get(index).dyn_into::<GamepadButton>().ok()
    }

    fn pressed(buttons: &js_sys::Array, index: u32) -> bool {
        button_at(buttons, index).is_some_and(|b| b.pressed())
    }

    fn value(buttons: &js_sys::Array, index: u32) -> f32 {
        button_at(buttons, index).map_or(0.0, |b| b.value() as f32)
    }

    fn axis(axes: &js_sys::Array, index: u32) -> f32 {
        axes.get(index).as_f64().unwrap_or(0.0) as f32
    }
}

/// The pad poller for whatever this build is targeting.
///
/// The host names this and never the platform type, which is what keeps the
/// wiring in `lib.rs` free of `cfg`.
#[cfg(not(target_arch = "wasm32"))]
pub type Pads = NativePads;

/// See the native [`Pads`](type@Pads).
#[cfg(target_arch = "wasm32")]
pub type Pads = WebPads;

#[cfg(test)]
mod tests {
    use super::*;

    /// Collect what a closure-taking method pushed.
    fn collect(f: impl FnOnce(&mut dyn FnMut(InputEvent))) -> Vec<InputEvent> {
        let mut out = Vec::new();
        {
            let mut push = |e: InputEvent| out.push(e);
            f(&mut push);
        }
        out
    }

    fn stick_event(events: &[InputEvent], want: PadStick) -> Option<Vec2> {
        events.iter().find_map(|e| match e {
            InputEvent::PadStick { stick, dir } if *stick == want => Some(*dir),
            _ => None,
        })
    }

    fn trigger_event(events: &[InputEvent], want: PadTrigger) -> Option<f32> {
        events.iter().find_map(|e| match e {
            InputEvent::PadTrigger { trigger, value } if *trigger == want => Some(*value),
            _ => None,
        })
    }

    #[test]
    fn a_resting_pad_writes_nothing() {
        let mut differ = PadDiffer::new();
        let events = collect(|push| differ.diff(PadSnapshot::new(), push));
        assert!(events.is_empty(), "{events:?}");
        // …and neither does one whose sticks are merely *near* zero.
        let noisy = PadSnapshot {
            left: Vec2::new(0.02, -0.03),
            right: Vec2::new(0.0, 0.04),
            triggers: [0.03, 0.01],
            ..PadSnapshot::new()
        };
        let events = collect(|push| differ.diff(noisy, push));
        assert!(events.is_empty(), "the noise gate leaked: {events:?}");
    }

    #[test]
    fn button_edges_come_out_in_padbutton_order() {
        let mut differ = PadDiffer::new();
        // Pressed in the *opposite* order to `PadButton::ALL`, to prove the
        // emission order is the enum's and not the caller's.
        let snap = PadSnapshot::new()
            .with(PadButton::DpadLeft, true)
            .with(PadButton::Start, true)
            .with(PadButton::South, true);
        let events = collect(|push| differ.diff(snap, push));
        assert_eq!(
            events,
            vec![
                InputEvent::PadButton {
                    button: PadButton::South,
                    pressed: true
                },
                InputEvent::PadButton {
                    button: PadButton::Start,
                    pressed: true
                },
                InputEvent::PadButton {
                    button: PadButton::DpadLeft,
                    pressed: true
                },
            ]
        );

        // An unchanged snapshot is silent; a partial release reports only what
        // moved.
        assert!(collect(|push| differ.diff(snap, push)).is_empty());
        let events = collect(|push| differ.diff(snap.with(PadButton::Start, false), push));
        assert_eq!(
            events,
            vec![InputEvent::PadButton {
                button: PadButton::Start,
                pressed: false
            }]
        );
    }

    #[test]
    fn a_sub_epsilon_wiggle_is_silent_but_accumulates() {
        let mut differ = PadDiffer::new();
        let mut snap = PadSnapshot::new();
        snap.left = Vec2::new(0.5, 0.0);
        assert_eq!(
            stick_event(&collect(|push| differ.diff(snap, push)), PadStick::Left),
            Some(Vec2::new(0.5, 0.0))
        );

        // Half an epsilon further: below the floor, nothing sent.
        snap.left.x += STICK_EPSILON * 0.5;
        assert!(collect(|push| differ.diff(snap, push)).is_empty());

        // …but the differ compares against what it *sent*, so a second nudge of
        // the same size crosses the floor and reports the whole travel.
        snap.left.x += STICK_EPSILON * 0.6;
        let dir = stick_event(&collect(|push| differ.diff(snap, push)), PadStick::Left)
            .expect("the accumulated drift must eventually be reported");
        assert!((dir.x - snap.left.x).abs() < 1e-6, "{dir:?}");
    }

    #[test]
    fn a_release_to_zero_is_never_eaten_by_the_epsilon() {
        let mut differ = PadDiffer::new();
        // Deflect by *less* than one epsilon past the gate, so the return trip
        // is a sub-epsilon change — the exact case that would strand the stick.
        let mut snap = PadSnapshot::new();
        snap.left = Vec2::new(PAD_NOISE_GATE + STICK_EPSILON * 0.25, 0.0);
        snap.triggers[PadTrigger::L2.index()] = PAD_NOISE_GATE + STICK_EPSILON * 0.25;
        let events = collect(|push| differ.diff(snap, push));
        assert!(stick_event(&events, PadStick::Left).is_some());
        assert!(trigger_event(&events, PadTrigger::L2).is_some());

        let events = collect(|push| differ.diff(PadSnapshot::new(), push));
        assert_eq!(
            stick_event(&events, PadStick::Left),
            Some(Vec2::ZERO),
            "a centred stick must always reach the engine"
        );
        assert_eq!(trigger_event(&events, PadTrigger::L2), Some(0.0));
        assert_eq!(differ.sent(), PadSnapshot::new());
    }

    #[test]
    fn the_noise_gate_snaps_to_exactly_zero_before_comparing() {
        let mut differ = PadDiffer::new();
        let mut snap = PadSnapshot::new();
        snap.right = Vec2::new(0.6, 0.0);
        collect(|push| differ.diff(snap, push));

        // Letting go leaves the pad resting inside the gate, not at 0.0. The
        // engine must still be told zero — the *gated* zero — and the differ
        // must then be quiet however the rest wanders inside the gate.
        snap.right = Vec2::new(0.031, -0.02);
        assert_eq!(
            stick_event(&collect(|push| differ.diff(snap, push)), PadStick::Right),
            Some(Vec2::ZERO)
        );
        snap.right = Vec2::new(-0.04, 0.01);
        assert!(collect(|push| differ.diff(snap, push)).is_empty());
    }

    #[test]
    fn levels_ride_out_on_the_same_frame_as_edges_in_a_fixed_order() {
        let mut differ = PadDiffer::new();
        let snap = PadSnapshot {
            buttons: 0,
            left: Vec2::new(0.0, 1.0),
            right: Vec2::new(1.0, 0.0),
            triggers: [1.0, 0.5],
        }
        .with(PadButton::South, true);
        let events = collect(|push| differ.diff(snap, push));
        assert_eq!(events.len(), 5);
        assert!(matches!(events[0], InputEvent::PadButton { .. }));
        assert!(matches!(
            events[1],
            InputEvent::PadStick {
                stick: PadStick::Left,
                ..
            }
        ));
        assert!(matches!(
            events[2],
            InputEvent::PadStick {
                stick: PadStick::Right,
                ..
            }
        ));
        assert!(matches!(
            events[3],
            InputEvent::PadTrigger {
                trigger: PadTrigger::L2,
                ..
            }
        ));
        assert!(matches!(
            events[4],
            InputEvent::PadTrigger {
                trigger: PadTrigger::R2,
                ..
            }
        ));
    }

    #[test]
    fn reset_neutralises_exactly_what_is_held() {
        let mut differ = PadDiffer::new();
        let snap = PadSnapshot {
            buttons: 0,
            left: Vec2::new(0.0, 1.0),
            right: Vec2::ZERO,
            triggers: [0.0, 0.75],
        }
        .with(PadButton::East, true)
        .with(PadButton::R1, true);
        collect(|push| differ.diff(snap, push));

        let events = collect(|push| differ.reset(push));
        assert_eq!(
            events,
            vec![
                InputEvent::PadButton {
                    button: PadButton::East,
                    pressed: false
                },
                InputEvent::PadButton {
                    button: PadButton::R1,
                    pressed: false
                },
                InputEvent::PadStick {
                    stick: PadStick::Left,
                    dir: Vec2::ZERO
                },
                InputEvent::PadTrigger {
                    trigger: PadTrigger::R2,
                    value: 0.0
                },
            ],
            "reset must send the neutral half of the held state and nothing else"
        );

        // Nothing is held now, so a second reset is silent…
        assert!(collect(|push| differ.reset(push)).is_empty());
        // …and the engine is believed neutral, so a still-held pad reports afresh.
        let events = collect(|push| differ.diff(snap, push));
        assert_eq!(events.len(), 4);
    }

    #[test]
    fn forget_drops_the_state_without_saying_anything() {
        let mut differ = PadDiffer::new();
        let snap = PadSnapshot::new().with(PadButton::North, true);
        collect(|push| differ.diff(snap, push));
        differ.forget();
        assert_eq!(differ.sent(), PadSnapshot::new());
        // The engine zeroed itself on FocusLost, so the pad must re-announce.
        assert_eq!(
            collect(|push| differ.diff(snap, push)),
            vec![InputEvent::PadButton {
                button: PadButton::North,
                pressed: true
            }]
        );
    }

    #[test]
    fn merge_ors_buttons_and_takes_the_strongest_axis() {
        let one = PadSnapshot {
            buttons: 0,
            left: Vec2::new(0.3, 0.0),
            right: Vec2::new(0.9, 0.0),
            triggers: [1.0, 0.0],
        }
        .with(PadButton::South, true);
        let two = PadSnapshot {
            buttons: 0,
            left: Vec2::new(0.0, -0.8),
            right: Vec2::new(0.1, 0.0),
            triggers: [0.2, 0.6],
        }
        .with(PadButton::DpadUp, true);

        let merged = merge([one, two].into_iter());
        assert!(merged.is_pressed(PadButton::South) && merged.is_pressed(PadButton::DpadUp));
        assert_eq!(merged.left, two.left, "the larger deflection wins");
        assert_eq!(merged.right, one.right);
        assert_eq!(merged.triggers, [1.0, 0.6], "per-trigger max");

        // Opposed sticks do not cancel — one of them still drives.
        let left = PadSnapshot {
            left: Vec2::new(-1.0, 0.0),
            ..PadSnapshot::new()
        };
        let right = PadSnapshot {
            left: Vec2::new(1.0, 0.0),
            ..PadSnapshot::new()
        };
        assert_eq!(merge([left, right].into_iter()).left, left.left);

        // A tie keeps the first, so the merge does not depend on poll order
        // jitter, and no pads at all is a neutral pad.
        assert_eq!(merge(std::iter::empty()), PadSnapshot::new());
    }
}
