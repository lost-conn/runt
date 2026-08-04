//! Trace-format tripwire (DESIGN §4).
//!
//! Postcard encodes an enum variant as its **index**, so the declaration order
//! of [`InputEvent`] and [`Key`] *is* the wire format of an
//! [`InputTrace`](runt_core::trace::InputTrace). Reordering or inserting a
//! variant silently reinterprets every trace ever recorded: a stored
//! `TouchDrive` becomes a `FocusLost`, a replay diverges, and nothing anywhere
//! says why.
//!
//! These tests pin the indices so that mistake is a failing build instead. They
//! are not testing postcard — they are testing that nobody edits the middle of
//! those enums. Appending is always fine, and is what the doc comments ask for;
//! if a new variant needs to go in the middle, the honest move is to bump a
//! trace format version, not to renumber in place.

use glam::Vec2;
use runt_core::input::{Input, InputEvent, Key, PadButton, PadStick, PadTrigger};

/// Postcard writes the variant index as a leading varint; every index here is
/// < 128, so it is exactly the first byte.
fn variant_index(event: &InputEvent) -> u8 {
    let bytes = postcard::to_stdvec(event).expect("serialize");
    assert!(!bytes.is_empty(), "an event always encodes to something");
    assert!(bytes[0] < 0x80, "index outgrew a one-byte varint: {bytes:?}");
    bytes[0]
}

#[test]
fn input_event_variant_indices_are_pinned() {
    // The pre-existing seven. Changing any of these breaks stored traces.
    assert_eq!(variant_index(&InputEvent::KeyDown(Key::A)), 0);
    assert_eq!(variant_index(&InputEvent::KeyUp(Key::A)), 1);
    assert_eq!(variant_index(&InputEvent::MouseMove { dx: 0.0, dy: 0.0 }), 2);
    assert_eq!(
        variant_index(&InputEvent::MouseButton {
            button: 0,
            pressed: true
        }),
        3
    );
    assert_eq!(variant_index(&InputEvent::Wheel { dy: 0.0 }), 4);
    assert_eq!(variant_index(&InputEvent::TouchDrive { dir: Vec2::ZERO }), 5);
    assert_eq!(variant_index(&InputEvent::FocusLost), 6);

    // The gamepad vocabulary, appended after `FocusLost`.
    assert_eq!(
        variant_index(&InputEvent::PadButton {
            button: PadButton::South,
            pressed: true
        }),
        7
    );
    assert_eq!(
        variant_index(&InputEvent::PadStick {
            stick: PadStick::Left,
            dir: Vec2::ZERO
        }),
        8
    );
    assert_eq!(
        variant_index(&InputEvent::PadTrigger {
            trigger: PadTrigger::L2,
            value: 0.0
        }),
        9
    );
}

#[test]
fn key_indices_are_pinned_at_both_ends() {
    // A key is encoded as its index too, inside the event's payload.
    assert_eq!(postcard::to_stdvec(&Key::A).expect("serialize"), vec![0]);
    assert_eq!(
        postcard::to_stdvec(&Key::Other).expect("serialize"),
        vec![(Key::COUNT - 1) as u8],
        "Other is the last key; new keys go before it, never after"
    );
    assert_eq!(Key::index(Key::A), 0);
    assert_eq!(Key::index(Key::Other), Key::COUNT - 1);
}

#[test]
fn pad_button_indices_follow_the_standard_mapping() {
    // Standard-mapping order, so a host can translate by index. Pinned because
    // a reorder here is a silently rebound controller as well as a broken
    // trace.
    for (expected, button) in PadButton::ALL.iter().enumerate() {
        assert_eq!(button.index(), expected);
        assert_eq!(
            postcard::to_stdvec(button).expect("serialize"),
            vec![expected as u8]
        );
    }
    assert_eq!(PadButton::South.index(), 0);
    assert_eq!(PadButton::East.index(), 1);
    assert_eq!(PadButton::West.index(), 2);
    assert_eq!(PadButton::North.index(), 3);
    assert_eq!(PadButton::DpadUp.index(), 10);
    assert_eq!(
        PadButton::Other.index(),
        PadButton::COUNT - 1,
        "Other is the last pad button; new ones go before it"
    );
    assert_eq!(PadStick::Left.index(), 0);
    assert_eq!(PadStick::Right.index(), 1);
    assert_eq!(PadTrigger::L2.index(), 0);
    assert_eq!(PadTrigger::R2.index(), 1);
}

/// A trace of every variant survives a round trip — the tripwire above pins the
/// numbering, this pins that the numbering is actually usable end to end.
#[test]
fn a_trace_of_every_variant_round_trips() {
    use runt_core::trace::InputTrace;

    let trace = InputTrace::from_pairs([
        (0, InputEvent::KeyDown(Key::Space)),
        (0, InputEvent::KeyUp(Key::Space)),
        (1, InputEvent::MouseMove { dx: -1.5, dy: 2.0 }),
        (
            1,
            InputEvent::MouseButton {
                button: 2,
                pressed: true,
            },
        ),
        (2, InputEvent::Wheel { dy: 0.25 }),
        (
            2,
            InputEvent::TouchDrive {
                dir: Vec2::new(0.0, 1.0),
            },
        ),
        (3, InputEvent::FocusLost),
        (
            4,
            InputEvent::PadButton {
                button: PadButton::DpadDown,
                pressed: true,
            },
        ),
        (
            4,
            InputEvent::PadStick {
                stick: PadStick::Right,
                dir: Vec2::new(0.5, -0.5),
            },
        ),
        (
            5,
            InputEvent::PadTrigger {
                trigger: PadTrigger::R2,
                value: 0.75,
            },
        ),
    ]);
    let bytes = trace.to_bytes().expect("serialize");
    assert_eq!(InputTrace::from_bytes(&bytes).expect("deserialize"), trace);
}

/// The recorder's half of the pad story: buttons are edges and are written on
/// the ticks they moved, sticks and triggers are *levels* and are written only
/// when they change — a pad held still must cost nothing per tick, exactly as
/// `tests/trace.rs` asserts for the touch stick.
#[test]
fn the_recorder_writes_pad_levels_only_where_they_moved() {
    use runt_core::trace::{InputTrace, TickEvent};
    use runt_core::{Sim, SimConfig};

    let mut sim = Sim::from_config(SimConfig::default().without_scene());
    sim.record_input_trace();

    let pushes: &[(u64, InputEvent)] = &[
        (
            1,
            InputEvent::PadButton {
                button: PadButton::South,
                pressed: true,
            },
        ),
        (
            1,
            InputEvent::PadStick {
                stick: PadStick::Left,
                dir: Vec2::new(0.0, 1.0),
            },
        ),
        // Redundant: the level did not move, so nothing is written.
        (
            2,
            InputEvent::PadStick {
                stick: PadStick::Left,
                dir: Vec2::new(0.0, 1.0),
            },
        ),
        // Auto-repeat on a button: not a new press, nothing is written.
        (
            2,
            InputEvent::PadButton {
                button: PadButton::South,
                pressed: true,
            },
        ),
        (
            3,
            InputEvent::PadTrigger {
                trigger: PadTrigger::R2,
                value: 0.5,
            },
        ),
        (
            4,
            InputEvent::PadButton {
                button: PadButton::South,
                pressed: false,
            },
        ),
        // A focus loss is never written as itself: it has to come back out as
        // the release and the zeroed levels it left behind.
        (5, InputEvent::FocusLost),
    ];
    for tick in 0..8u64 {
        for (t, ev) in pushes {
            if *t == tick {
                sim.push_input(*ev);
            }
        }
        sim.tick();
    }

    let trace = sim.input_trace().expect("recording").clone();
    assert_eq!(
        trace.events,
        vec![
            TickEvent {
                tick: 1,
                event: InputEvent::PadButton {
                    button: PadButton::South,
                    pressed: true
                }
            },
            TickEvent {
                tick: 1,
                event: InputEvent::PadStick {
                    stick: PadStick::Left,
                    dir: Vec2::new(0.0, 1.0)
                }
            },
            TickEvent {
                tick: 3,
                event: InputEvent::PadTrigger {
                    trigger: PadTrigger::R2,
                    value: 0.5
                }
            },
            TickEvent {
                tick: 4,
                event: InputEvent::PadButton {
                    button: PadButton::South,
                    pressed: false
                }
            },
            TickEvent {
                tick: 5,
                event: InputEvent::PadStick {
                    stick: PadStick::Left,
                    dir: Vec2::ZERO
                }
            },
            TickEvent {
                tick: 5,
                event: InputEvent::PadTrigger {
                    trigger: PadTrigger::R2,
                    value: 0.0
                }
            },
        ],
        "a pad held still must not write an event per tick"
    );

    // Through postcard and back, the trace replays into the same held state —
    // including the ticks in between, where the level came from nowhere but the
    // event that set it.
    let bytes = trace.to_bytes().expect("serialize");
    let replayed = InputTrace::from_bytes(&bytes).expect("deserialize");
    let mut state = Input::new();
    let mut stick_at = Vec::new();
    for tick in 0..8u64 {
        state.begin_tick(replayed.events_at(tick).collect::<Vec<_>>());
        stick_at.push((
            state.pad_held(PadButton::South),
            state.stick(PadStick::Left).y,
            state.trigger(PadTrigger::R2),
        ));
    }
    assert_eq!(
        stick_at,
        vec![
            (false, 0.0, 0.0),
            (true, 1.0, 0.0),
            (true, 1.0, 0.0), // Held across a tick the trace says nothing about.
            (true, 1.0, 0.5),
            (false, 1.0, 0.5),
            (false, 0.0, 0.0), // The focus loss, reconstructed from its effects.
            (false, 0.0, 0.0),
            (false, 0.0, 0.0),
        ]
    );
}
