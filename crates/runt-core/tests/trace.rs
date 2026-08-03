//! Input traces (DESIGN §4).
//!
//! The engine-level half of the replay story: that [`record`](runt_core::trace::record)
//! captures exactly what a tick saw, that [`apply`](runt_core::trace::apply)
//! hands it back, and that the pair is a fixed point — record a run, replay it,
//! record *that*, and you must get the same trace. `demo/ball/tests/replay.rs`
//! is the same claim through a whole game.

use bevy_ecs::prelude::*;

use runt_core::ecs::advance_tick_count;
use runt_core::input::{Input, InputEvent, Key};
use runt_core::trace::{self, InputTrace, Playback, TickEvent};
use runt_core::{Sim, SimConfig, TickCount};

/// One tick of `Input`, in full: held state *and* both edge sets, for keys and
/// buttons alike. Everything a `FixedSim` system could possibly read.
#[derive(Clone, Debug, Default, PartialEq)]
struct Snapshot {
    tick: u64,
    held: Vec<Key>,
    pressed: Vec<Key>,
    released: Vec<Key>,
    buttons: [bool; 3],
    button_edges: [(bool, bool); 3],
    /// Scaled to integers so the comparison is exact without arguing about
    /// float equality on a value that came through an accumulator.
    mouse: [i32; 2],
    wheel: i32,
    /// The analog drive stick, scaled like the mouse. A *level*, so unlike the
    /// two above it is expected to be identical on ticks with no events at all
    /// — which is exactly the part a trace could get wrong.
    drive: [i32; 2],
}

/// What `Input` looked like on each tick — the thing a trace has to reproduce.
#[derive(Resource, Default, Clone, Debug, PartialEq)]
struct Seen(Vec<Snapshot>);

/// `FixedSim` (tail): snapshot the tick's input state.
fn watch(mut seen: ResMut<Seen>, tick: Res<TickCount>, input: Res<Input>) {
    let delta = input.mouse_delta();
    seen.0.push(Snapshot {
        tick: tick.0,
        held: input.held_keys().collect(),
        pressed: input.just_pressed_keys().collect(),
        released: input.just_released_keys().collect(),
        buttons: std::array::from_fn(|i| input.button_held(i as u8)),
        button_edges: std::array::from_fn(|i| {
            (
                input.button_just_pressed(i as u8),
                input.button_just_released(i as u8),
            )
        }),
        mouse: [(delta.x * 1000.0) as i32, (delta.y * 1000.0) as i32],
        wheel: (input.wheel() * 1000.0) as i32,
        drive: [
            (input.drive().x * 1000.0) as i32,
            (input.drive().y * 1000.0) as i32,
        ],
    });
}

fn sim() -> Sim {
    // No scene: this is about input plumbing, and an empty world makes any
    // failure unambiguous.
    let mut sim = Sim::from_config(SimConfig::default().without_scene());
    sim.world_mut().init_resource::<Seen>();
    sim.fixed_sim_mut()
        .add_systems(watch.after(advance_tick_count));
    sim
}

fn seen(sim: &Sim) -> Seen {
    sim.world().resource::<Seen>().clone()
}

/// A run with a bit of everything, and deliberately including both of the awkward
/// same-tick cases [`trace::record`] has a table for: a key **tapped** inside one
/// tick (down then up, ending unheld) and a key **re-pressed** inside one tick
/// (up then down, ending held). Those two look identical in `Input`'s edge sets
/// and differ only in `held`.
///
/// It also drives a touch stick across the whole thing — including a redundant
/// `TouchDrive` the recorder must *not* write out, and a focus loss in the
/// middle of everything, which the recorder never writes out at all and has to
/// reproduce out of the `KeyUp`s and the zeroed stick it left behind.
fn live_run(sim: &mut Sim, ticks: u64) {
    let events: &[(u64, InputEvent)] = &[
        (1, InputEvent::KeyDown(Key::W)),
        (2, InputEvent::KeyDown(Key::E)),
        (3, InputEvent::KeyDown(Key::A)),
        (3, InputEvent::MouseMove { dx: 4.0, dy: -2.5 }),
        (4, InputEvent::TouchDrive { dir: glam::Vec2::new(0.5, 0.25) }),
        // Tapped: both edges, ends unheld.
        (5, InputEvent::KeyDown(Key::Space)),
        (5, InputEvent::KeyUp(Key::Space)),
        (6, InputEvent::Wheel { dy: 1.5 }),
        // The value it already has: a level that did not move is not an event.
        (6, InputEvent::TouchDrive { dir: glam::Vec2::new(0.5, 0.25) }),
        (7, InputEvent::KeyUp(Key::A)),
        (8, InputEvent::MouseButton { button: 0, pressed: true }),
        (9, InputEvent::KeyDown(Key::W)), // Auto-repeat: not a new press.
        (9, InputEvent::TouchDrive { dir: glam::Vec2::new(0.0, 1.0) }),
        // Re-pressed: both edges, ends *held*.
        (10, InputEvent::KeyUp(Key::E)),
        (10, InputEvent::KeyDown(Key::E)),
        (11, InputEvent::MouseButton { button: 0, pressed: false }),
        (12, InputEvent::KeyUp(Key::W)),
        // The same trick on a mouse button.
        (14, InputEvent::MouseButton { button: 1, pressed: true }),
        // Alt-tab with a key held, a button down and the stick pushed forward.
        (15, InputEvent::FocusLost),
        (16, InputEvent::MouseButton { button: 1, pressed: false }),
        (16, InputEvent::MouseButton { button: 1, pressed: true }),
        (17, InputEvent::TouchDrive { dir: glam::Vec2::new(-0.25, -0.75) }),
        (18, InputEvent::KeyUp(Key::E)),
        (18, InputEvent::MouseButton { button: 1, pressed: false }),
    ];
    for tick in 0..ticks {
        for (t, event) in events {
            if *t == tick {
                sim.push_input(*event);
            }
        }
        sim.tick();
    }
}

// ---------------------------------------------------------------------------

#[test]
fn a_recorded_trace_replays_to_the_same_input_stream() {
    let mut live = sim();
    live.record_input_trace();
    live_run(&mut live, 20);
    let trace = live.input_trace().expect("recording").clone();
    assert!(!trace.is_empty());

    let mut replay = sim();
    replay.play_input_trace(trace.clone());
    for _ in 0..20 {
        replay.tick();
    }

    assert_eq!(
        seen(&replay),
        seen(&live),
        "the replay did not reproduce the input the run actually saw"
    );
}

#[test]
fn re_recording_a_replay_reproduces_the_trace() {
    // The fixed-point property. If `record` and `apply` disagreed about the
    // encoding of a tick — an edge dropped, an ordering swapped, a delta
    // rounded — this is where it would show, because the second trace is the
    // first one round-tripped through both halves.
    let mut live = sim();
    live.record_input_trace();
    live_run(&mut live, 20);
    let first = live.input_trace().expect("recording").clone();

    let mut replay = sim();
    replay.play_input_trace(first.clone());
    replay.record_input_trace();
    for _ in 0..20 {
        replay.tick();
    }
    let second = replay.input_trace().expect("recording").clone();

    assert_eq!(second, first);
}

#[test]
fn a_replay_overrides_whatever_the_host_pushes() {
    // `apply` replaces the tick's `Input` from playback state it owns, so a host
    // event cannot leak a held key into a replay. Without that, a replay would
    // be at the mercy of whoever was leaning on the keyboard.
    let trace = InputTrace::from_pairs([
        (2, InputEvent::KeyDown(Key::W)),
        (6, InputEvent::KeyUp(Key::W)),
    ]);

    let quiet = {
        let mut sim = sim();
        sim.play_input_trace(trace.clone());
        for _ in 0..12 {
            sim.tick();
        }
        seen(&sim)
    };

    let noisy = {
        let mut sim = sim();
        sim.play_input_trace(trace);
        for tick in 0..12 {
            sim.push_input(InputEvent::KeyDown(Key::D));
            if tick % 3 == 0 {
                sim.push_input(InputEvent::KeyUp(Key::D));
                sim.push_input(InputEvent::MouseMove { dx: 9.0, dy: 9.0 });
            }
            sim.tick();
        }
        seen(&sim)
    };

    assert_eq!(quiet, noisy);
    assert!(
        quiet.0.iter().any(|s| s.held == [Key::W]),
        "the trace itself did nothing, so this proves nothing"
    );
    assert!(
        !quiet.0.iter().any(|s| s.held.contains(&Key::D)),
        "a host key survived into the replay"
    );
    assert!(
        quiet.0.iter().all(|s| s.mouse == [0, 0]),
        "host mouse motion survived into the replay"
    );
}

#[test]
fn a_trace_survives_postcard() {
    let trace = InputTrace::from_pairs([
        (0, InputEvent::KeyDown(Key::Escape)),
        (17, InputEvent::MouseMove { dx: -0.5, dy: 12.25 }),
        (17, InputEvent::Wheel { dy: -3.0 }),
        (99, InputEvent::MouseButton { button: 2, pressed: true }),
    ]);
    let bytes = trace.to_bytes().expect("serialize");
    assert_eq!(InputTrace::from_bytes(&bytes).expect("deserialize"), trace);

    // Compact enough to be worth using: four events well under a hundred bytes.
    assert!(bytes.len() < 100, "{} bytes for 4 events", bytes.len());
    assert!(InputTrace::from_bytes(b"not a trace at all").is_err());
}

#[test]
fn events_are_keyed_on_ticks_and_found_in_order() {
    let trace = InputTrace::from_pairs([
        (5, InputEvent::KeyDown(Key::A)),
        (0, InputEvent::KeyDown(Key::W)),
        (5, InputEvent::KeyUp(Key::W)),
    ]);
    assert_eq!(
        trace.events,
        vec![
            TickEvent { tick: 0, event: InputEvent::KeyDown(Key::W) },
            TickEvent { tick: 5, event: InputEvent::KeyDown(Key::A) },
            TickEvent { tick: 5, event: InputEvent::KeyUp(Key::W) },
        ],
        "sorted by tick, stable within one"
    );
    assert_eq!(trace.events_at(0).count(), 1);
    assert_eq!(trace.events_at(5).count(), 2);
    assert_eq!(trace.events_at(4).count(), 0);
    assert_eq!(trace.last_tick(), Some(5));
    assert_eq!(Playback::new(trace).remaining(2), 3);
}

#[test]
fn a_stick_is_recorded_where_it_moved_and_nowhere_else() {
    // The stick is a level, not an edge (see `InputEvent::TouchDrive`), so what
    // a trace has to carry is the ticks it *changed* on — and the replay has to
    // hold the value on every tick in between, which is the part a recorder that
    // treated it like the mouse delta would get wrong.
    let mut live = sim();
    live.record_input_trace();
    let pushes: &[(u64, glam::Vec2)] = &[
        (1, glam::Vec2::new(0.0, 1.0)),
        (2, glam::Vec2::new(0.0, 1.0)), // Unchanged.
        (3, glam::Vec2::new(0.0, 1.0)), // Still unchanged.
        (7, glam::Vec2::new(-0.6, 0.2)),
        (12, glam::Vec2::ZERO), // Finger lifted.
    ];
    for tick in 0..20u64 {
        for (t, dir) in pushes {
            if *t == tick {
                live.push_input(InputEvent::TouchDrive { dir: *dir });
            }
        }
        live.tick();
    }

    let trace = live.input_trace().expect("recording").clone();
    assert_eq!(
        trace.events,
        vec![
            TickEvent { tick: 1, event: InputEvent::TouchDrive { dir: glam::Vec2::new(0.0, 1.0) } },
            TickEvent { tick: 7, event: InputEvent::TouchDrive { dir: glam::Vec2::new(-0.6, 0.2) } },
            TickEvent { tick: 12, event: InputEvent::TouchDrive { dir: glam::Vec2::ZERO } },
        ],
        "a stick held still must not write an event per tick"
    );

    // Replaying those three events reproduces all twenty ticks bit for bit,
    // through postcard on the way — the file a `--record` run would leave.
    let bytes = trace.to_bytes().expect("serialize");
    let mut replay = sim();
    replay.play_input_trace(InputTrace::from_bytes(&bytes).expect("deserialize"));
    for _ in 0..20 {
        replay.tick();
    }
    assert_eq!(seen(&replay), seen(&live));

    // And the stick really was held across the gaps, or the above would be a
    // comparison of two runs that both did nothing.
    let held = seen(&live)
        .0
        .iter()
        .filter(|s| s.drive == [0, 1000])
        .count();
    assert_eq!(held, 6, "ticks 1..=6 should all read full forward");
}

#[test]
fn a_focus_loss_reaches_a_replay_through_its_effects() {
    // `FocusLost` is never written into a trace: by the time the recorder runs
    // it has already become ordinary state, and replaying the `KeyUp`s and the
    // zeroed stick it left behind reproduces the tick exactly. That is the
    // property — not that the event survives, but that the run does.
    let mut live = sim();
    live.record_input_trace();
    for tick in 0..10u64 {
        match tick {
            0 => {
                live.push_input(InputEvent::KeyDown(Key::W));
                live.push_input(InputEvent::TouchDrive { dir: glam::Vec2::new(1.0, 0.0) });
            }
            4 => live.push_input(InputEvent::FocusLost),
            _ => {}
        }
        live.tick();
    }

    let trace = live.input_trace().expect("recording").clone();
    assert!(
        !trace
            .events
            .iter()
            .any(|e| e.event == InputEvent::FocusLost),
        "the recorder should not be writing host state changes into a trace"
    );
    assert!(
        trace.events.contains(&TickEvent { tick: 4, event: InputEvent::KeyUp(Key::W) })
            && trace.events.contains(&TickEvent {
                tick: 4,
                event: InputEvent::TouchDrive { dir: glam::Vec2::ZERO }
            }),
        "…but its effects have to be in there: {:?}",
        trace.events
    );

    let mut replay = sim();
    replay.play_input_trace(trace);
    for _ in 0..10 {
        replay.tick();
    }
    assert_eq!(seen(&replay), seen(&live));

    // The point of the whole exercise: nothing is still held afterwards.
    let last = seen(&live).0.last().expect("ten ticks").clone();
    assert!(last.held.is_empty(), "a key survived the focus loss: {last:?}");
    assert_eq!(last.drive, [0, 0]);
}

#[test]
fn recording_costs_nothing_on_a_tick_with_no_input() {
    let mut sim = sim();
    sim.record_input_trace();
    for _ in 0..300 {
        sim.tick();
    }
    assert!(
        sim.input_trace().expect("recording").is_empty(),
        "an idle run must not fill a trace with silence"
    );
}

#[test]
fn the_systems_are_usable_without_the_sim_helpers() {
    // `record` and `apply` are ordinary systems, so a caller with its own
    // schedule layout can place them itself. This is the shape the helpers on
    // `Sim` build, spelled out.
    let mut sim = Sim::from_config(SimConfig::default().without_scene());
    sim.world_mut()
        .insert_resource(Playback::new(InputTrace::from_pairs([(
            1,
            InputEvent::KeyDown(Key::Q),
        )])));
    sim.world_mut().init_resource::<InputTrace>();
    sim.world_mut().init_resource::<Seen>();
    sim.fixed_sim_mut().add_systems((
        trace::apply.before(trace::record),
        trace::record.before(advance_tick_count),
        watch.after(advance_tick_count),
    ));

    for _ in 0..4 {
        sim.tick();
    }
    assert!(seen(&sim).0.iter().any(|s| s.held == [Key::Q]));
    assert_eq!(
        sim.world().resource::<InputTrace>().events,
        vec![TickEvent { tick: 1, event: InputEvent::KeyDown(Key::Q) }]
    );
}
