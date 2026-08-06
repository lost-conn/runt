//! Input traces — record a run, replay it exactly (DESIGN §4).
//!
//! > *Input is captured by the host, translated to engine input events, and
//! > consumed at tick boundaries — replays are just recorded input traces +
//! > seeds.* — DESIGN §4
//!
//! This module is that sentence, made into two systems and a `Vec`.
//!
//! ## Keyed on ticks, not on wall time
//!
//! An [`InputTrace`] is `(tick index, event)` pairs. The *arrival* time of a key
//! press is a property of the host's polling — two machines will genuinely hand
//! the same press to different ticks, and will genuinely diverge, correctly so,
//! because they were given different input. What has to be reproducible is the
//! **tick sequence**, and that is exactly what a trace pins down. A 60 fps host
//! and a stuttering one replaying the same trace produce bit-identical runs;
//! `tests/trace.rs` and `demo/ball/tests/replay.rs` assert it.
//!
//! ## Record
//!
//! [`record`] reconstructs the tick's events *from the [`Input`] resource*, not
//! from the host's buffer. That is the stronger place to tap: `Input` is what
//! `FixedSim` systems actually see, so a trace can never disagree with the run
//! it came from — anything the host pushed that `Input` collapsed away (auto
//! repeat, an unmappable key, two mouse moves in one tick) is collapsed in the
//! trace the same way.
//!
//! ## Apply
//!
//! [`apply`] *replaces* the tick's `Input`, so whatever the host is doing while
//! a replay runs is irrelevant — you can wiggle the mouse over a replaying
//! window without changing the outcome. It keeps its own [`Input`] in
//! [`Playback`] and feeds it the trace's slice of the tick, so held state comes
//! from the trace and only from the trace.
//!
//! ## Where they go in the tick
//!
//! Both belong at the **head** of `FixedSim`, before anything reads input:
//! [`Sim::record_input_trace`](crate::Sim::record_input_trace) and
//! [`Sim::play_input_trace`](crate::Sim::play_input_trace) install them there.
//! Recording a replay (both at once) works and is a useful self-check: the
//! recorder is downstream of the player, so it re-derives the trace it was
//! handed.

use bevy_ecs::prelude::*;
use serde::{Deserialize, Serialize};

use crate::ecs::TickCount;
use crate::input::{Input, InputEvent, TouchPhase, MOUSE_BUTTONS};

/// One event, and the tick it belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TickEvent {
    pub tick: u64,
    pub event: InputEvent,
}

/// A recorded run's input, in tick order.
///
/// Also a `Resource`: [`record`] appends to it in place.
#[derive(Resource, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct InputTrace {
    /// Sorted by `tick`, and stable within a tick (the order the events were
    /// applied in). [`push`](InputTrace::push) preserves both.
    pub events: Vec<TickEvent>,
}

impl InputTrace {
    pub fn new() -> InputTrace {
        InputTrace::default()
    }

    /// Build from `(tick, event)` pairs — how a test writes a script.
    ///
    /// The pairs are sorted by tick with a **stable** sort, so events given for
    /// the same tick keep the order they were written in.
    pub fn from_pairs(pairs: impl IntoIterator<Item = (u64, InputEvent)>) -> InputTrace {
        let mut events: Vec<TickEvent> = pairs
            .into_iter()
            .map(|(tick, event)| TickEvent { tick, event })
            .collect();
        events.sort_by_key(|e| e.tick);
        InputTrace { events }
    }

    /// Append an event for `tick`. Ticks must not go backwards — the recorder
    /// walks time forwards by construction, and a sorted trace is what
    /// [`events_at`](InputTrace::events_at) binary-searches.
    pub fn push(&mut self, tick: u64, event: InputEvent) {
        debug_assert!(
            self.events.last().is_none_or(|last| last.tick <= tick),
            "an InputTrace is recorded in tick order"
        );
        self.events.push(TickEvent { tick, event });
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// The last tick with an event on it, or `None` for an empty trace. What a
    /// headless replay uses to decide how long to run.
    pub fn last_tick(&self) -> Option<u64> {
        self.events.last().map(|e| e.tick)
    }

    /// This tick's events, in order. `O(log n)` to find the run, then a walk —
    /// a linear scan per tick would make replay quadratic in trace length.
    pub fn events_at(&self, tick: u64) -> impl Iterator<Item = InputEvent> + '_ {
        let start = self.events.partition_point(|e| e.tick < tick);
        self.events[start..]
            .iter()
            .take_while(move |e| e.tick == tick)
            .map(|e| e.event)
    }

    /// Postcard bytes — the same compact encoding the mesh cache uses (§6).
    pub fn to_bytes(&self) -> Result<Vec<u8>, TraceError> {
        postcard::to_stdvec(self).map_err(|e| TraceError(e.to_string()))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<InputTrace, TraceError> {
        postcard::from_bytes(bytes).map_err(|e| TraceError(e.to_string()))
    }
}

#[derive(Debug)]
pub struct TraceError(pub String);

impl std::fmt::Display for TraceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "input trace: {}", self.0)
    }
}

impl std::error::Error for TraceError {}

// ---------------------------------------------------------------------------
// Record
// ---------------------------------------------------------------------------

/// `FixedSim` (head): append this tick's input to the [`InputTrace`] resource.
///
/// The events are re-derived from [`Input`]'s edge sets and analog accumulators
/// in a fixed order — keys in [`Key::ALL`] order, then mouse buttons, then
/// motion, then the wheel, then the drive stick, then the pad (buttons in
/// [`PadButton::ALL`](crate::input::PadButton::ALL) order, sticks, triggers),
/// then the touches (see [`record_touches`]) — so the encoding of a tick is a
/// pure function of that tick's `Input` and does not depend on how the host
/// happened to interleave its pushes. Within one key or button, order is *not*
/// free:
///
/// | edges seen | held after | emitted |
/// |---|---|---|
/// | pressed | yes | `KeyDown` |
/// | released | no | `KeyUp` |
/// | both | **no** | `KeyDown`, `KeyUp` — tapped inside the tick |
/// | both | **yes** | `KeyUp`, `KeyDown` — re-pressed inside the tick |
///
/// The last two are why this is a table and not two loops. Both edges with the
/// key *not* held is a tap; both edges with it *still* held is a release and a
/// fresh press. Emitting them in one fixed order would replay one of the two
/// cases into the wrong held state and quietly desynchronise everything after
/// it — `tests/trace.rs` covers both.
///
/// ## What a focus loss looks like in a trace
///
/// [`InputEvent::FocusLost`] is never written out, because by the time this runs
/// it has already become ordinary state: the keys it dropped appear as `KeyUp`s,
/// the pad buttons as released `PadButton`s, the fingers it dropped as ended
/// `Touch`es, and the sticks and triggers it centred as zeroed
/// `TouchDrive`/`PadStick`/`PadTrigger`s. Replaying those reproduces the tick
/// exactly, which is the property that matters — the trace records what the tick
/// *saw*, not what the window manager did.
pub fn record(mut trace: ResMut<InputTrace>, tick: Res<TickCount>, input: Res<Input>) {
    let now = tick.0;
    for key in crate::input::Key::ALL {
        let (pressed, released) = (input.just_pressed(key), input.just_released(key));
        if !pressed && !released {
            continue;
        }
        let (down, up) = (InputEvent::KeyDown(key), InputEvent::KeyUp(key));
        match (pressed, released, input.held(key)) {
            (true, true, true) => {
                trace.push(now, up);
                trace.push(now, down);
            }
            (true, true, false) => {
                trace.push(now, down);
                trace.push(now, up);
            }
            (true, false, _) => trace.push(now, down),
            (false, true, _) => trace.push(now, up),
            (false, false, _) => unreachable!("filtered above"),
        }
    }
    for button in 0..MOUSE_BUTTONS as u8 {
        let (pressed, released) = (
            input.button_just_pressed(button),
            input.button_just_released(button),
        );
        if !pressed && !released {
            continue;
        }
        let event = |pressed| InputEvent::MouseButton { button, pressed };
        match (pressed, released, input.button_held(button)) {
            (true, true, true) => {
                trace.push(now, event(false));
                trace.push(now, event(true));
            }
            (true, true, false) => {
                trace.push(now, event(true));
                trace.push(now, event(false));
            }
            (true, false, _) => trace.push(now, event(true)),
            (false, true, _) => trace.push(now, event(false)),
            (false, false, _) => unreachable!("filtered above"),
        }
    }
    let delta = input.mouse_delta();
    if delta != glam::Vec2::ZERO {
        trace.push(
            now,
            InputEvent::MouseMove {
                dx: delta.x,
                dy: delta.y,
            },
        );
    }
    if input.wheel() != 0.0 {
        trace.push(now, InputEvent::Wheel { dy: input.wheel() });
    }
    // The stick is a *level* (see [`InputEvent::TouchDrive`]), so what a trace
    // has to carry is the ticks it moved on — a value that has not changed is
    // already in the replayed `Input` from the tick that set it.
    if input.drive_changed() {
        trace.push(now, InputEvent::TouchDrive { dir: input.drive() });
    }
    // The pad, on exactly the terms above: buttons are edges and need the same
    // four-case table as keys (a tap and a re-press inside one tick differ only
    // in the order the two events go out), sticks and triggers are levels and
    // are written only on the ticks they moved.
    for button in crate::input::PadButton::ALL {
        let (pressed, released) = (
            input.pad_just_pressed(button),
            input.pad_just_released(button),
        );
        if !pressed && !released {
            continue;
        }
        let event = |pressed| InputEvent::PadButton { button, pressed };
        match (pressed, released, input.pad_held(button)) {
            (true, true, true) => {
                trace.push(now, event(false));
                trace.push(now, event(true));
            }
            (true, true, false) => {
                trace.push(now, event(true));
                trace.push(now, event(false));
            }
            (true, false, _) => trace.push(now, event(true)),
            (false, true, _) => trace.push(now, event(false)),
            (false, false, _) => unreachable!("filtered above"),
        }
    }
    for stick in crate::input::PadStick::ALL {
        if input.stick_changed(stick) {
            trace.push(
                now,
                InputEvent::PadStick {
                    stick,
                    dir: input.stick(stick),
                },
            );
        }
    }
    for trigger in crate::input::PadTrigger::ALL {
        if input.trigger_changed(trigger) {
            trace.push(
                now,
                InputEvent::PadTrigger {
                    trigger,
                    value: input.trigger(trigger),
                },
            );
        }
    }
    record_touches(&mut trace, now, &input);
}

/// The touch half of [`record`], split out because it is four passes and a
/// reason.
///
/// A finger is neither an edge like a key nor a level like a stick: it is a
/// *membership* with a position. So the tick is written as the three things that
/// can have happened to the live set — arrivals, motions, departures — in four
/// passes, whose order is what replays into the state the tick actually had:
///
/// 1. departures of ids that are **still live**,
/// 2. arrivals,
/// 3. motions of live touches that moved,
/// 4. departures of ids that are **not** live.
///
/// Splitting the departures is the same problem the key table solves. An id in
/// both the started and the ended list is either a tap inside one tick (not live
/// afterwards → `Started` then `Ended`) or a platform recycling a contact number
/// onto a new finger inside one tick (live afterwards → the old one's `Ended`
/// must go out *first*, or the replay would end the finger that just arrived and
/// quietly lose it for the rest of the run).
///
/// Only touches that moved are written, for the reason the sticks are written
/// only when they change: ten fingers resting on the glass must not cost ten
/// events a tick. A focus loss, as everywhere else here, is never written as
/// itself — it comes back as the departures it caused.
fn record_touches(trace: &mut InputTrace, now: u64, input: &Input) {
    let event = |touch: crate::input::Touch, phase| InputEvent::Touch {
        id: touch.id,
        phase,
        x: touch.pos.x,
        y: touch.pos.y,
    };
    for touch in input.touches_ended() {
        if input.touch(touch.id).is_some() {
            trace.push(now, event(touch, TouchPhase::Ended));
        }
    }
    for touch in input.touches_started() {
        trace.push(now, event(touch, TouchPhase::Started));
    }
    for touch in input.touches() {
        if input.touch_moved(touch.id) {
            trace.push(now, event(touch, TouchPhase::Moved));
        }
    }
    for touch in input.touches_ended() {
        if input.touch(touch.id).is_none() {
            trace.push(now, event(touch, TouchPhase::Ended));
        }
    }
}

// ---------------------------------------------------------------------------
// Apply
// ---------------------------------------------------------------------------

/// A trace being replayed, plus the [`Input`] it has rebuilt so far.
///
/// The private `state` is why replay is airtight: held keys accumulate *here*,
/// from the trace, and are copied over the world's `Input` wholesale. A host
/// that keeps pushing real events during a replay cannot leak a single held key
/// into the run.
#[derive(Resource, Clone, Debug, Default)]
pub struct Playback {
    pub trace: InputTrace,
    state: Input,
}

impl Playback {
    pub fn new(trace: InputTrace) -> Playback {
        Playback {
            trace,
            state: Input::new(),
        }
    }

    /// Ticks the trace still has events for, counted from `tick`.
    pub fn remaining(&self, tick: u64) -> u64 {
        self.trace.last_tick().unwrap_or(0).saturating_sub(tick)
    }
}

/// `FixedSim` (head): overwrite this tick's [`Input`] with the trace's.
pub fn apply(mut playback: ResMut<Playback>, tick: Res<TickCount>, mut input: ResMut<Input>) {
    let now = tick.0;
    let events: Vec<InputEvent> = playback.trace.events_at(now).collect();
    playback.state.begin_tick(events);
    *input = playback.state.clone();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::Key;

    #[test]
    fn events_at_finds_the_ticks_run() {
        let trace = InputTrace::from_pairs([
            (0, InputEvent::KeyDown(Key::W)),
            (4, InputEvent::KeyDown(Key::A)),
            (4, InputEvent::KeyUp(Key::W)),
            (9, InputEvent::KeyUp(Key::A)),
        ]);
        assert_eq!(trace.events_at(4).count(), 2);
        assert_eq!(
            trace.events_at(4).next(),
            Some(InputEvent::KeyDown(Key::A)),
            "same-tick order is the order it was written in"
        );
        assert_eq!(trace.events_at(5).count(), 0);
        assert_eq!(trace.last_tick(), Some(9));
    }

    #[test]
    fn postcard_round_trips() {
        let trace = InputTrace::from_pairs([
            (1, InputEvent::KeyDown(Key::Space)),
            (2, InputEvent::MouseMove { dx: -1.5, dy: 2.0 }),
            (2, InputEvent::Wheel { dy: 0.25 }),
            (7, InputEvent::MouseButton { button: 1, pressed: true }),
        ]);
        let bytes = trace.to_bytes().expect("serialize");
        assert_eq!(InputTrace::from_bytes(&bytes).expect("deserialize"), trace);
    }
}
