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
use runt_core::input::{Input, InputEvent, Key, PadButton, PadStick, PadTrigger, TouchPhase};

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

    // Raw multi-touch, appended after the pad.
    assert_eq!(
        variant_index(&InputEvent::Touch {
            id: 0,
            phase: TouchPhase::Started,
            x: 0.0,
            y: 0.0
        }),
        10
    );
}

#[test]
fn touch_phase_indices_are_pinned() {
    // A phase is encoded as its index inside the event's payload, so a reorder
    // would turn every stored `Ended` into a `Moved` — a replay in which no
    // finger ever lets go.
    for (expected, phase) in [
        TouchPhase::Started,
        TouchPhase::Moved,
        TouchPhase::Ended,
        TouchPhase::Cancelled,
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(
            postcard::to_stdvec(&phase).expect("serialize"),
            vec![expected as u8]
        );
    }
}

/// The tolerance the [`InputEvent`] doc promises, tested against bytes rather
/// than against a rebuilt value: this is a trace recorded **before**
/// `InputEvent::Touch` existed, pasted in as the literal it was, and it has to
/// keep meaning what it meant. Appending a variant cannot disturb it; inserting
/// one in the middle would, and this is what would say so.
#[test]
fn a_trace_recorded_before_touch_existed_still_reads_back() {
    use runt_core::trace::InputTrace;

    const OLD_TRACE: &[u8] = &[
        0x06, 0x00, 0x00, 0x16, 0x01, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f, 0x02,
        0x03, 0x00, 0x01, 0x03, 0x06, 0x04, 0x07, 0x00, 0x01, 0x07, 0x01, 0x16,
    ];

    let trace = InputTrace::from_bytes(OLD_TRACE).expect("an old trace still parses");
    assert_eq!(
        trace,
        InputTrace::from_pairs([
            (0, InputEvent::KeyDown(Key::W)),
            (
                1,
                InputEvent::TouchDrive {
                    dir: Vec2::new(0.0, 1.0)
                }
            ),
            (
                2,
                InputEvent::MouseButton {
                    button: 0,
                    pressed: true
                }
            ),
            (3, InputEvent::FocusLost),
            (
                4,
                InputEvent::PadButton {
                    button: PadButton::South,
                    pressed: true
                }
            ),
            (7, InputEvent::KeyUp(Key::W)),
        ]),
        "an appended variant renumbered the old ones"
    );

    // And it still *replays* — the state it rebuilds is the state it always
    // rebuilt, with the new touch machinery sitting inert beside it.
    let mut state = Input::new();
    let mut seen = Vec::new();
    for tick in 0..8u64 {
        state.begin_tick(trace.events_at(tick).collect::<Vec<_>>());
        seen.push((state.held(Key::W), state.drive().y, state.touch_count()));
    }
    assert_eq!(
        seen,
        vec![
            (true, 0.0, 0),
            (true, 1.0, 0),
            (true, 1.0, 0),
            (false, 0.0, 0), // The focus loss, exactly as it always was.
            (false, 0.0, 0),
            (false, 0.0, 0),
            (false, 0.0, 0),
            (false, 0.0, 0),
        ]
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
        (
            6,
            InputEvent::Touch {
                id: u64::MAX,
                phase: TouchPhase::Started,
                x: 12.5,
                y: -3.25,
            },
        ),
        (
            6,
            InputEvent::Touch {
                id: 2,
                phase: TouchPhase::Moved,
                x: 0.0,
                y: 1080.0,
            },
        ),
        (
            7,
            InputEvent::Touch {
                id: 2,
                phase: TouchPhase::Ended,
                x: 0.0,
                y: 1080.0,
            },
        ),
        (
            7,
            InputEvent::Touch {
                id: 3,
                phase: TouchPhase::Cancelled,
                x: 1.0,
                y: 2.0,
            },
        ),
    ]);
    let bytes = trace.to_bytes().expect("serialize");
    assert_eq!(InputTrace::from_bytes(&bytes).expect("deserialize"), trace);
}

/// The recorder's touch half. Fingers are neither edges nor levels — they are a
/// membership with a position — so the property to pin is that a tick's live set
/// is *exactly* reconstructible from what was written, while a hand resting on
/// the glass writes nothing.
#[test]
fn the_recorder_writes_touches_only_where_they_changed() {
    use runt_core::trace::{InputTrace, TickEvent};
    use runt_core::{Sim, SimConfig};

    let touch = |id, phase, x, y| InputEvent::Touch { id, phase, x, y };

    let mut sim = Sim::from_config(SimConfig::default().without_scene());
    sim.record_input_trace();

    let pushes: &[(u64, InputEvent)] = &[
        (1, touch(1, TouchPhase::Started, 10.0, 10.0)),
        (1, touch(2, TouchPhase::Started, 200.0, 10.0)),
        // Redundant: neither finger moved, so tick 2 costs nothing.
        (2, touch(1, TouchPhase::Moved, 10.0, 10.0)),
        (3, touch(1, TouchPhase::Moved, 40.0, 10.0)),
        // A tap that lands and lifts inside one tick still has both edges.
        (4, touch(9, TouchPhase::Started, 5.0, 5.0)),
        (4, touch(9, TouchPhase::Ended, 5.0, 5.0)),
        (5, touch(2, TouchPhase::Cancelled, 200.0, 10.0)),
        // A focus loss is never written as itself — finger 1 has to come back
        // out as the end it caused.
        (6, InputEvent::FocusLost),
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
                event: touch(1, TouchPhase::Started, 10.0, 10.0)
            },
            TickEvent {
                tick: 1,
                event: touch(2, TouchPhase::Started, 200.0, 10.0)
            },
            TickEvent {
                tick: 3,
                event: touch(1, TouchPhase::Moved, 40.0, 10.0)
            },
            TickEvent {
                tick: 4,
                event: touch(9, TouchPhase::Started, 5.0, 5.0)
            },
            TickEvent {
                tick: 4,
                event: touch(9, TouchPhase::Ended, 5.0, 5.0)
            },
            // A cancel is recorded as an end: `Input` does not distinguish
            // them, so a trace that claimed to would be claiming to know
            // something the tick never saw.
            TickEvent {
                tick: 5,
                event: touch(2, TouchPhase::Ended, 200.0, 10.0)
            },
            TickEvent {
                tick: 6,
                event: touch(1, TouchPhase::Ended, 40.0, 10.0)
            },
        ],
        "a hand resting on the glass must not write an event per tick"
    );

    // Through postcard and back, the trace replays into the same live set on
    // every tick — including the ones it says nothing about.
    let bytes = trace.to_bytes().expect("serialize");
    let replayed = InputTrace::from_bytes(&bytes).expect("deserialize");
    let mut state = Input::new();
    let mut seen = Vec::new();
    for tick in 0..8u64 {
        state.begin_tick(replayed.events_at(tick).collect::<Vec<_>>());
        seen.push((
            state.touches().map(|t| (t.id, t.pos.x)).collect::<Vec<_>>(),
            state.touches_started().count(),
            state.touches_ended().count(),
        ));
    }
    assert_eq!(
        seen,
        vec![
            (vec![], 0, 0),
            (vec![(1, 10.0), (2, 200.0)], 2, 0),
            (vec![(1, 10.0), (2, 200.0)], 0, 0), // Held across a silent tick.
            (vec![(1, 40.0), (2, 200.0)], 0, 0),
            (vec![(1, 40.0), (2, 200.0)], 1, 1), // The tap, in and out.
            (vec![(1, 40.0)], 0, 1),
            (vec![], 0, 1), // The focus loss, reconstructed from its effects.
            (vec![], 0, 0),
        ]
    );
}

/// A platform may hand the same contact number to a new finger the moment the
/// old one lifts — inside a single tick, if the tick was long enough. The
/// recorder has to write the departure *before* the arrival in that case, for
/// the same reason the key table exists: written the other way round, the replay
/// would end the finger that just arrived and lose it for the rest of the run.
#[test]
fn a_recycled_contact_id_replays_as_the_new_finger() {
    use runt_core::trace::InputTrace;

    let touch = |id, phase, x, y| InputEvent::Touch { id, phase, x, y };
    let mut live = Input::new();
    live.begin_tick([touch(1, TouchPhase::Started, 10.0, 10.0)]);
    live.begin_tick([
        touch(1, TouchPhase::Ended, 12.0, 10.0),
        touch(1, TouchPhase::Started, 900.0, 500.0),
    ]);
    assert_eq!(live.touch(1).map(|t| t.pos.x), Some(900.0));
    assert_eq!(live.touch_ended(1).map(|t| t.pos.x), Some(12.0));

    // Record that tick the way `trace::record` would, and replay it.
    let mut sim = runt_core::Sim::from_config(runt_core::SimConfig::default().without_scene());
    sim.record_input_trace();
    sim.tick();
    sim.push_input(touch(1, TouchPhase::Started, 10.0, 10.0));
    sim.tick();
    sim.push_input(touch(1, TouchPhase::Ended, 12.0, 10.0));
    sim.push_input(touch(1, TouchPhase::Started, 900.0, 500.0));
    sim.tick();

    let trace = sim.input_trace().expect("recording").clone();
    let bytes = trace.to_bytes().expect("serialize");
    let replayed = InputTrace::from_bytes(&bytes).expect("deserialize");
    let mut state = Input::new();
    for tick in 0..3u64 {
        state.begin_tick(replayed.events_at(tick).collect::<Vec<_>>());
    }
    assert_eq!(
        state.touch(1).map(|t| t.pos.x),
        Some(900.0),
        "the replay ended the finger that had just arrived"
    );
    assert_eq!(state.touch_ended(1).map(|t| t.pos.x), Some(12.0));
    assert_eq!(state.touches_started().count(), 1);
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
