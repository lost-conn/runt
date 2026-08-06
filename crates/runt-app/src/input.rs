//! winit → engine input translation.
//!
//! This table is the *only* place a winit type meets a runt type. Keeping it
//! host-side is what lets the editor (rinch `SurfaceEvent`s) and the player
//! (winit) feed an engine that cannot tell them apart (DESIGN §2, §10).

use glam::Vec2;
use runt_core::{InputEvent, Key};
use winit::event::{MouseButton, TouchPhase};
use winit::keyboard::KeyCode;

/// Map a winit physical key code onto the engine's key vocabulary.
///
/// Physical (not logical) codes: WASD must stay WASD on an AZERTY keyboard's
/// layout position, and a replay must not depend on the layout in effect.
/// Anything outside the vocabulary collapses to [`Key::Other`] by design.
pub fn translate_key(code: KeyCode) -> Key {
    match code {
        KeyCode::KeyA => Key::A,
        KeyCode::KeyB => Key::B,
        KeyCode::KeyC => Key::C,
        KeyCode::KeyD => Key::D,
        KeyCode::KeyE => Key::E,
        KeyCode::KeyF => Key::F,
        KeyCode::KeyG => Key::G,
        KeyCode::KeyH => Key::H,
        KeyCode::KeyI => Key::I,
        KeyCode::KeyJ => Key::J,
        KeyCode::KeyK => Key::K,
        KeyCode::KeyL => Key::L,
        KeyCode::KeyM => Key::M,
        KeyCode::KeyN => Key::N,
        KeyCode::KeyO => Key::O,
        KeyCode::KeyP => Key::P,
        KeyCode::KeyQ => Key::Q,
        KeyCode::KeyR => Key::R,
        KeyCode::KeyS => Key::S,
        KeyCode::KeyT => Key::T,
        KeyCode::KeyU => Key::U,
        KeyCode::KeyV => Key::V,
        KeyCode::KeyW => Key::W,
        KeyCode::KeyX => Key::X,
        KeyCode::KeyY => Key::Y,
        KeyCode::KeyZ => Key::Z,

        KeyCode::Digit0 | KeyCode::Numpad0 => Key::Digit0,
        KeyCode::Digit1 | KeyCode::Numpad1 => Key::Digit1,
        KeyCode::Digit2 | KeyCode::Numpad2 => Key::Digit2,
        KeyCode::Digit3 | KeyCode::Numpad3 => Key::Digit3,
        KeyCode::Digit4 | KeyCode::Numpad4 => Key::Digit4,
        KeyCode::Digit5 | KeyCode::Numpad5 => Key::Digit5,
        KeyCode::Digit6 | KeyCode::Numpad6 => Key::Digit6,
        KeyCode::Digit7 | KeyCode::Numpad7 => Key::Digit7,
        KeyCode::Digit8 | KeyCode::Numpad8 => Key::Digit8,
        KeyCode::Digit9 | KeyCode::Numpad9 => Key::Digit9,

        KeyCode::ArrowUp => Key::Up,
        KeyCode::ArrowDown => Key::Down,
        KeyCode::ArrowLeft => Key::Left,
        KeyCode::ArrowRight => Key::Right,

        KeyCode::Space => Key::Space,
        KeyCode::Enter | KeyCode::NumpadEnter => Key::Enter,
        KeyCode::Tab => Key::Tab,
        KeyCode::Escape => Key::Escape,

        // Physical positions, like every other entry: on an AZERTY board these
        // two are where `^` and `$` print, and a replay recorded on one layout
        // has to step the render scale the same way on another.
        KeyCode::BracketLeft => Key::BracketLeft,
        KeyCode::BracketRight => Key::BracketRight,

        // Left/right modifiers collapse: a ball game wants "is shift down".
        KeyCode::ShiftLeft | KeyCode::ShiftRight => Key::Shift,
        KeyCode::ControlLeft | KeyCode::ControlRight => Key::Ctrl,
        KeyCode::AltLeft | KeyCode::AltRight => Key::Alt,

        _ => Key::Other,
    }
}

/// Map a winit mouse button onto the engine's button index.
///
/// 0/1/2 = left/right/middle, matching [`runt_core::InputEvent::MouseButton`].
pub fn translate_button(button: MouseButton) -> u8 {
    match button {
        MouseButton::Left => 0,
        MouseButton::Right => 1,
        MouseButton::Middle => 2,
        MouseButton::Back => 3,
        MouseButton::Forward => 4,
        MouseButton::Other(n) => 5u8.saturating_add((n % 3) as u8),
    }
}

/// Map a winit touch phase onto the engine's.
///
/// A one-to-one table rather than a re-export, because `runt-core` deliberately
/// has no winit dependency (DESIGN §2): the engine's
/// [`TouchPhase`](runt_core::TouchPhase) is its own type, and this is the seam.
pub fn translate_touch_phase(phase: TouchPhase) -> runt_core::TouchPhase {
    match phase {
        TouchPhase::Started => runt_core::TouchPhase::Started,
        TouchPhase::Moved => runt_core::TouchPhase::Moved,
        TouchPhase::Ended => runt_core::TouchPhase::Ended,
        TouchPhase::Cancelled => runt_core::TouchPhase::Cancelled,
    }
}

/// Everything one host touch becomes, engine-side.
///
/// The first return is the raw [`InputEvent::Touch`] and it is **always**
/// produced: every contact reaches the sim, so a game can build a floating dpad,
/// a chorded button grid and a camera drag out of them at the same time. The
/// second is the [`InputEvent::TouchDrive`] the virtual stick synthesised, and
/// it exists only when the host was given a `stick` (see
/// [`RunConfig::without_virtual_stick`](crate::RunConfig::without_virtual_stick))
/// *and* the deflection moved enough to be worth an event.
///
/// Both at once is the safe default, not a compromise: a game that reads only
/// [`Input::drive`](runt_core::Input::drive) is unaffected by touch events it
/// never looks at, so the existing one-thumb path keeps working untouched. A
/// game that reads the raw touches turns the synthesis off, and then the same
/// finger cannot drive the ball twice down two paths.
///
/// A free function taking the stick by `Option<&mut _>` rather than a method on
/// the host: the host owns a GPU device and cannot be built in a unit test, and
/// this is the whole decision worth testing.
pub fn touch_events(
    stick: Option<&mut VirtualStick>,
    id: u64,
    phase: TouchPhase,
    x: f32,
    y: f32,
) -> (InputEvent, Option<InputEvent>) {
    let raw = InputEvent::Touch {
        id,
        phase: translate_touch_phase(phase),
        x,
        y,
    };
    let drive = stick
        .and_then(|stick| stick.touch(id, phase, x, y))
        .map(|dir| InputEvent::TouchDrive { dir });
    (raw, drive)
}

// ---------------------------------------------------------------------------
// Touch → virtual stick
// ---------------------------------------------------------------------------

/// Logical pixels of drag that read as full deflection.
///
/// 60 is about a thumb's comfortable travel without lifting: far enough that a
/// deliberate half-push is reachable, near enough that a full push does not need
/// the wrist. It is *logical*, not physical, so a 3× phone screen and a laptop
/// touchpad feel the same.
pub const STICK_RADIUS: f32 = 60.0;

/// Logical pixels of slop before the stick reads at all. A finger resting on
/// glass drifts a pixel or two; without this the ball creeps.
pub const STICK_DEADZONE: f32 = 6.0;

/// Smallest change worth sending to the engine.
///
/// A drag produces a touch event per frame, and most of them move the stick by
/// less than a pixel. 1/64 of full deflection is finer than anything a player can
/// aim and keeps a slow drag from writing an event into the trace every tick.
pub const STICK_EPSILON: f32 = 1.0 / 64.0;

/// A touch screen as an analog stick: the first finger down anchors the centre,
/// and dragging away from that anchor deflects it.
///
/// It lives here — on the host side, next to the winit key table — because
/// DESIGN §2 says a host translates events and does nothing else. The engine
/// receives [`InputEvent::TouchDrive`](runt_core::InputEvent::TouchDrive) and
/// cannot tell a finger from a gamepad.
///
/// **One finger, on purpose:** the first one down is the stick and every other
/// one is ignored until it lifts. That is not a limitation to lift here — a
/// second thumb for a camera or a chord of buttons is *not* expressible as a
/// single deflection, so it goes to the game raw, as
/// [`InputEvent::Touch`](runt_core::InputEvent::Touch), and the game decides
/// what a finger means. A program that does that turns this off entirely
/// ([`RunConfig::without_virtual_stick`](crate::RunConfig::without_virtual_stick))
/// so one thumb cannot drive both paths at once.
#[derive(Clone, Copy, Debug, Default)]
pub struct VirtualStick {
    /// `(finger id, anchor x, anchor y)` in logical pixels.
    anchor: Option<(u64, f32, f32)>,
    value: Vec2,
}

impl VirtualStick {
    pub fn new() -> VirtualStick {
        VirtualStick::default()
    }

    /// The deflection last sent, `x` right and `y` forward.
    pub fn value(&self) -> Vec2 {
        self.value
    }

    /// Whether a finger currently owns the stick.
    pub fn is_active(&self) -> bool {
        self.anchor.is_some()
    }

    /// Feed one winit touch. Returns the deflection to push at the engine, or
    /// `None` when nothing moved enough to be worth an event.
    ///
    /// `x`/`y` are **logical** pixels with `y` growing downwards, as every
    /// windowing system reports them; the returned `y` grows *forwards*, which
    /// is why the sign flips below.
    pub fn touch(&mut self, id: u64, phase: TouchPhase, x: f32, y: f32) -> Option<Vec2> {
        match phase {
            TouchPhase::Started => {
                if self.anchor.is_none() {
                    self.anchor = Some((id, x, y));
                    // A fresh anchor is a centred stick. Report it only if the
                    // engine is not already holding zero.
                    return self.emit(Vec2::ZERO);
                }
                None // Someone else's finger; v1 ignores it.
            }
            TouchPhase::Moved => {
                let (anchor, ax, ay) = self.anchor?;
                if anchor != id {
                    return None;
                }
                self.emit(deflection(x - ax, ay - y))
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                let (anchor, _, _) = self.anchor?;
                if anchor != id {
                    return None;
                }
                self.anchor = None;
                self.emit(Vec2::ZERO)
            }
        }
    }

    /// Forget the finger and centre the stick — for focus loss, where the
    /// `Ended` that would otherwise arrive never does.
    ///
    /// Returns nothing to push: the engine zeroes its own drive when it is told
    /// the window lost focus (see
    /// [`InputEvent::FocusLost`](runt_core::InputEvent::FocusLost)), and sending
    /// a redundant `TouchDrive` alongside it would put a second event in every
    /// trace for no gain.
    pub fn reset(&mut self) {
        self.anchor = None;
        self.value = Vec2::ZERO;
    }

    fn emit(&mut self, next: Vec2) -> Option<Vec2> {
        if (next - self.value).length() < STICK_EPSILON && next != Vec2::ZERO {
            return None;
        }
        if next == self.value {
            return None;
        }
        self.value = next;
        Some(next)
    }
}

/// Drag offset (right, forward) in logical pixels → stick deflection.
///
/// The dead zone is *subtracted* rather than clipped, so the stick starts from
/// zero the moment it engages instead of jumping to `deadzone/radius`.
fn deflection(right: f32, forward: f32) -> Vec2 {
    let raw = Vec2::new(right, forward);
    let length = raw.length();
    if !length.is_finite() || length <= STICK_DEADZONE {
        return Vec2::ZERO;
    }
    let travel = (length - STICK_DEADZONE) / (STICK_RADIUS - STICK_DEADZONE);
    raw / length * travel.min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letters_and_arrows_map_through() {
        assert_eq!(translate_key(KeyCode::KeyW), Key::W);
        assert_eq!(translate_key(KeyCode::ArrowLeft), Key::Left);
        assert_eq!(translate_key(KeyCode::Space), Key::Space);
        assert_eq!(translate_key(KeyCode::Escape), Key::Escape);
    }

    #[test]
    fn modifier_sides_collapse() {
        assert_eq!(translate_key(KeyCode::ShiftLeft), Key::Shift);
        assert_eq!(translate_key(KeyCode::ShiftRight), Key::Shift);
    }

    #[test]
    fn unmapped_keys_are_other() {
        assert_eq!(translate_key(KeyCode::F13), Key::Other);
        assert_eq!(translate_key(KeyCode::ScrollLock), Key::Other);
    }

    #[test]
    fn buttons_use_the_documented_indices() {
        assert_eq!(translate_button(MouseButton::Left), 0);
        assert_eq!(translate_button(MouseButton::Right), 1);
        assert_eq!(translate_button(MouseButton::Middle), 2);
    }

    #[test]
    fn touch_phases_map_one_for_one() {
        assert_eq!(
            translate_touch_phase(TouchPhase::Started),
            runt_core::TouchPhase::Started
        );
        assert_eq!(
            translate_touch_phase(TouchPhase::Moved),
            runt_core::TouchPhase::Moved
        );
        assert_eq!(
            translate_touch_phase(TouchPhase::Ended),
            runt_core::TouchPhase::Ended
        );
        assert_eq!(
            translate_touch_phase(TouchPhase::Cancelled),
            runt_core::TouchPhase::Cancelled
        );
    }

    // -- raw touches beside the stick ---------------------------------------

    #[test]
    fn every_finger_reaches_the_engine_raw_even_the_ones_the_stick_ignores() {
        let mut stick = VirtualStick::new();

        let (raw, drive) = touch_events(Some(&mut stick), 1, TouchPhase::Started, 100.0, 100.0);
        assert_eq!(
            raw,
            InputEvent::Touch {
                id: 1,
                phase: runt_core::TouchPhase::Started,
                x: 100.0,
                y: 100.0
            }
        );
        assert_eq!(drive, None, "a fresh anchor is a centred stick");

        // The second finger is invisible to the stick — and must not be
        // invisible to the game, which is the entire point of the raw stream.
        let (raw, drive) = touch_events(Some(&mut stick), 2, TouchPhase::Started, 500.0, 500.0);
        assert_eq!(
            raw,
            InputEvent::Touch {
                id: 2,
                phase: runt_core::TouchPhase::Started,
                x: 500.0,
                y: 500.0
            }
        );
        assert_eq!(drive, None);
        let (raw, drive) = touch_events(Some(&mut stick), 2, TouchPhase::Moved, 500.0, 100.0);
        assert!(matches!(raw, InputEvent::Touch { id: 2, .. }));
        assert_eq!(drive, None, "the second finger does not drive the stick");

        // The first one still does, and both events go out together.
        let (raw, drive) = touch_events(
            Some(&mut stick),
            1,
            TouchPhase::Moved,
            100.0,
            100.0 - STICK_RADIUS,
        );
        assert!(matches!(raw, InputEvent::Touch { id: 1, .. }));
        let Some(InputEvent::TouchDrive { dir }) = drive else {
            panic!("a full-radius drag drives the stick: {drive:?}");
        };
        assert!((dir.y - 1.0).abs() < 1e-5, "{dir:?}");
    }

    #[test]
    fn without_the_virtual_stick_only_the_raw_touch_goes_out() {
        // The switch's whole job: a game that builds its own controls must not
        // also be driving `Input::drive` with the same thumb.
        let script = [
            (1u64, TouchPhase::Started, 100.0, 100.0),
            (1, TouchPhase::Moved, 100.0, 100.0 - STICK_RADIUS),
            (1, TouchPhase::Ended, 100.0, 40.0),
            (2, TouchPhase::Started, 0.0, 0.0),
            (2, TouchPhase::Cancelled, 0.0, 0.0),
        ];
        for (id, phase, x, y) in script {
            let (raw, drive) = touch_events(None, id, phase, x, y);
            assert_eq!(
                raw,
                InputEvent::Touch {
                    id,
                    phase: translate_touch_phase(phase),
                    x,
                    y
                }
            );
            assert_eq!(drive, None, "synthesis is off: {phase:?} still drove");
        }
    }

    // -- the virtual stick --------------------------------------------------

    #[test]
    fn a_drag_deflects_the_stick_from_where_it_started() {
        let mut stick = VirtualStick::new();
        // Anchoring does not move anything, and the engine is already at zero.
        assert_eq!(stick.touch(1, TouchPhase::Started, 200.0, 400.0), None);
        assert!(stick.is_active());

        // Dragging up is *forward*, whatever the screen's y axis thinks.
        let up = stick
            .touch(1, TouchPhase::Moved, 200.0, 400.0 - STICK_RADIUS)
            .expect("a full-radius drag is a change");
        assert!((up.y - 1.0).abs() < 1e-5 && up.x.abs() < 1e-5, "{up:?}");

        // Right is +x, and past the radius it saturates instead of overshooting.
        let right = stick
            .touch(1, TouchPhase::Moved, 200.0 + STICK_RADIUS * 4.0, 400.0)
            .expect("change");
        assert!((right.x - 1.0).abs() < 1e-5 && right.y.abs() < 1e-5, "{right:?}");
        assert!(right.length() <= 1.0 + 1e-6);

        // Lifting centres it.
        let released = stick
            .touch(1, TouchPhase::Ended, 200.0 + STICK_RADIUS * 4.0, 400.0)
            .expect("a release must reach the engine");
        assert_eq!(released, Vec2::ZERO);
        assert!(!stick.is_active());
    }

    #[test]
    fn the_dead_zone_holds_the_stick_still_and_then_starts_from_zero() {
        let mut stick = VirtualStick::new();
        stick.touch(7, TouchPhase::Started, 0.0, 0.0);
        assert_eq!(
            stick.touch(7, TouchPhase::Moved, STICK_DEADZONE * 0.5, 0.0),
            None,
            "a resting thumb must not drive the ball"
        );
        // Just past the dead zone the deflection is near zero, not a jump to
        // deadzone/radius.
        let nudge = stick
            .touch(7, TouchPhase::Moved, STICK_DEADZONE + 1.0, 0.0)
            .expect("change");
        assert!(nudge.x > 0.0 && nudge.x < 0.05, "{nudge:?}");
    }

    #[test]
    fn only_the_first_finger_drives() {
        let mut stick = VirtualStick::new();
        stick.touch(1, TouchPhase::Started, 100.0, 100.0);
        let driven = stick.touch(1, TouchPhase::Moved, 100.0, 40.0).expect("change");

        // A second finger anywhere, doing anything, changes nothing.
        assert_eq!(stick.touch(2, TouchPhase::Started, 500.0, 500.0), None);
        assert_eq!(stick.touch(2, TouchPhase::Moved, 500.0, 100.0), None);
        assert_eq!(stick.touch(2, TouchPhase::Ended, 500.0, 100.0), None);
        assert_eq!(stick.value(), driven, "the second finger moved the stick");

        // …and the first one still owns it.
        assert!(stick.touch(1, TouchPhase::Moved, 160.0, 100.0).is_some());
    }

    #[test]
    fn a_cancelled_touch_centres_the_stick() {
        let mut stick = VirtualStick::new();
        stick.touch(3, TouchPhase::Started, 0.0, 0.0);
        stick.touch(3, TouchPhase::Moved, 0.0, -STICK_RADIUS);
        assert_eq!(
            stick.touch(3, TouchPhase::Cancelled, 0.0, -STICK_RADIUS),
            Some(Vec2::ZERO),
            "a cancelled finger must not leave the ball driving forever"
        );
        assert!(!stick.is_active());
    }

    #[test]
    fn a_sub_pixel_drag_does_not_write_an_event_per_frame() {
        let mut stick = VirtualStick::new();
        stick.touch(1, TouchPhase::Started, 0.0, 0.0);
        stick.touch(1, TouchPhase::Moved, 30.0, 0.0).expect("change");
        // A tenth of a pixel further: below `STICK_EPSILON` of deflection.
        assert_eq!(stick.touch(1, TouchPhase::Moved, 30.1, 0.0), None);
        // A whole radius further is not.
        assert!(stick.touch(1, TouchPhase::Moved, 60.0, 0.0).is_some());
    }

    #[test]
    fn reset_centres_without_emitting() {
        // The engine zeroes its own drive on `FocusLost`, so the host has
        // nothing to send — but it must not keep the stale anchor either.
        let mut stick = VirtualStick::new();
        stick.touch(1, TouchPhase::Started, 0.0, 0.0);
        stick.touch(1, TouchPhase::Moved, 0.0, -STICK_RADIUS);
        stick.reset();
        assert!(!stick.is_active());
        assert_eq!(stick.value(), Vec2::ZERO);
        // A new finger starts clean rather than resuming the old deflection.
        assert_eq!(stick.touch(2, TouchPhase::Started, 9.0, 9.0), None);
    }
}
