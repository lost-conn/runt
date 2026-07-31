//! winit → engine input translation.
//!
//! This table is the *only* place a winit type meets a runt type. Keeping it
//! host-side is what lets the editor (rinch `SurfaceEvent`s) and the player
//! (winit) feed an engine that cannot tell them apart (DESIGN §2, §10).

use runt_core::Key;
use winit::event::MouseButton;
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
}
