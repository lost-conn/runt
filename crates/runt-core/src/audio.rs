//! The sim-side audio seam (DESIGN §8).
//!
//! > *an `AudioOut` resource queues `AudioEvent`s during `FixedSim` and flushes
//! > once per tick — never mid-tick, so replays stay deterministic. Hosts
//! > implement one `AudioBackend::submit(&[AudioEvent])` trait; web serializes
//! > to `postMessage`, native pushes to an SPSC queue read by the cpal
//! > callback.* — DESIGN §8
//!
//! **There is no synthesizer in this module and there never will be.** The core
//! knows how to *say* "make the sound called `pickup`, seeded 7, this loud, over
//! there"; what that sounds like is `runt-audio`'s business, and the core does
//! not depend on it (DESIGN §2 — and fundsp drags in a `glam` a semver major
//! behind ours, which is reason enough on its own).
//!
//! ```text
//!  FixedSim system          AudioOut          flush_audio        host
//!  ───────────────          ────────          ───────────        ────
//!  audio.play_at(…)  ──▶  queued: Vec  ──▶  outbox: (tick, ev)  ──▶ AudioBackend
//!                          (this tick)       (once per tick)         ::submit
//! ```
//!
//! ## Audio is *output*, like [`StatusLine`](crate::StatusLine)
//!
//! Nothing in the sim reads the queue back, no system branches on what was
//! played, and a host that implements [`SilentBackend`] runs a bit-identical
//! simulation to one wired to speakers. That is the property that keeps DESIGN
//! §4's replay guarantee intact once sound exists: a trace replays to the same
//! transforms *and* to the same event stream, because the events are a pure
//! function of the tick and never an input to it.
//!
//! The one piece of state that looks like an exception is
//! [`AudioOut::next_voice`] — a monotonic counter minted inside the sim so a
//! game can name a note it wants to stop later. It is deterministic (a function
//! of the event stream, which is a function of the tick), never read by
//! gameplay, and never fed by anything downstream. Nothing flows upstream from
//! the audio thread; there is no handle to wait for and no id to learn.
//!
//! ## Where the flush sits in the tick, and why
//!
//! Last but one — after `propagate_transforms`, immediately **before**
//! `advance_tick_count`:
//!
//! ```text
//! update_overlap_messages  spin  integrate_balls  resolve_overlaps  roll_spin
//!   … game systems …  follow_camera  propagate_transforms
//!   ▸ flush_audio ◂   advance_tick_count
//! ```
//!
//! Three reasons, in order of how badly it breaks if you move it:
//!
//! 1. **A tick's events must leave as one batch.** Flushing mid-tick would make
//!    the *split* depend on system ordering, so a schedule change — which is
//!    allowed to be invisible — would silently reorder audio. One flush per
//!    tick, at a fixed point, is the version you can state and test.
//! 2. **The camera has already moved.** `follow_camera` runs late in the chain,
//!    so a pan computed against [`Listener`] uses this tick's final camera pose
//!    rather than last tick's. A sound placed relative to a stale camera lags
//!    the picture by a frame at exactly the moments it is most noticeable.
//! 3. **`TickCount` has not turned over yet**, so each event is stamped with the
//!    *zero-based index of the tick that produced it* — which is the index an
//!    [`InputTrace`](crate::InputTrace) uses, so an audio log and an input trace
//!    line up without an off-by-one.
//!
//! A game system deliberately scheduled `.after(advance_tick_count)` is
//! downstream of the flush and its events go out on the following tick. That is
//! consistent rather than surprising — it is downstream of the tick counter too
//! — and no engine system is scheduled there.

use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};

use crate::ecs::{TickCount, Transform};

// ---------------------------------------------------------------------------
// Vocabulary
// ---------------------------------------------------------------------------

/// A patch preset's stable identity: FNV-1a over its name.
///
/// The identical function is `runt_audio::PatchId::new`. The two crates do not
/// depend on each other, so both sides pin the same constants in their test
/// suites (`runt-core/tests/audio.rs` and `runt-audio/tests/wire.rs`); a change
/// to either turns both red.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PatchId(pub u64);

impl PatchId {
    pub const fn new(name: &str) -> PatchId {
        let bytes = name.as_bytes();
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        let mut i = 0;
        while i < bytes.len() {
            hash ^= bytes[i] as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            i += 1;
        }
        PatchId(hash)
    }
}

/// A note the *sim* named. See [`AudioOut::play`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VoiceId(pub u32);

/// A parameter a running voice can be re-aimed at.
///
/// The small shared vocabulary every patch understands; a patch ignores an id it
/// has no meaning for. Mirrored by `runt_audio::ParamId`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParamId(pub u16);

impl ParamId {
    /// Linear amplitude multiplier.
    pub const GAIN: ParamId = ParamId(0);
    /// `-1` left … `+1` right.
    pub const PAN: ParamId = ParamId(1);
    /// Multiplier on the patch's base frequency.
    pub const PITCH: ParamId = ParamId(2);
    /// Multiplier on the patch's filter cutoff.
    pub const CUTOFF: ParamId = ParamId(3);
    /// **Not a sound parameter**: `≥ 0.5` exempts the voice from being stolen
    /// when its group fills up, anything else releases the exemption.
    ///
    /// For a *looped* sound the sim keeps re-aiming — a swim stroke that runs
    /// while the player is swimming — where losing the slot to a steal would
    /// leave the game sending `SetParam` to a voice that no longer exists. Sent
    /// immediately after the [`AudioOut::play`] that starts the loop, and
    /// cleared (or [`AudioOut::stop`]ped) when it ends. The engine does nothing
    /// with it: like every id here it is a number the synth understands, and
    /// `runt_audio`'s pool is what acts on it.
    pub const HOLD: ParamId = ParamId(4);
}

/// One instruction for whatever is making sound.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AudioEvent {
    Play {
        voice: VoiceId,
        patch: PatchId,
        /// Seeds the patch. DESIGN §8: *"a param struct plus an explicit seed"*.
        /// The engine has no opinion about what a patch does with it — the
        /// built-in pluck turns it into a scale degree.
        seed: u64,
        /// Linear amplitude on top of the patch's own level.
        gain: f32,
        /// `-1` left … `+1` right. [`Listener::spatialize`] computes it from a
        /// world position; a UI sound writes `0.0`.
        pan: f32,
    },
    SetParam {
        voice: VoiceId,
        id: ParamId,
        value: f32,
    },
    Stop {
        voice: VoiceId,
    },
}

impl AudioEvent {
    pub fn voice(&self) -> VoiceId {
        match *self {
            AudioEvent::Play { voice, .. }
            | AudioEvent::SetParam { voice, .. }
            | AudioEvent::Stop { voice } => voice,
        }
    }
}

// ---------------------------------------------------------------------------
// Positioning
// ---------------------------------------------------------------------------

/// The distance and width laws [`Listener::spatialize`] applies.
///
/// DESIGN §8's phase-3 item 4, in full: *"camera-relative pan (screen-relative
/// X, clamped), 1/d gain rolloff, **no HRTF**"*. Two numbers, no filters, no
/// head model, no front/back cue. A sound behind the camera pans to the side it
/// is on and is not otherwise marked — which is wrong in a way nobody notices
/// and cheap in a way everybody benefits from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rolloff {
    /// Within this distance a source is at full gain; beyond it, gain falls as
    /// `reference / distance`.
    pub reference: f32,
    /// How far a source has to be off-axis to reach a hard pan. `1.0` puts a
    /// source at 90° hard over; above 1 the field is wider than the screen,
    /// which reads better on headphones for a third-person camera.
    pub pan_width: f32,
}

impl Default for Rolloff {
    fn default() -> Rolloff {
        Rolloff {
            reference: 4.0,
            pan_width: 1.25,
        }
    }
}

/// Where the ears are. Built from the camera's pose — DESIGN §5 has exactly one
/// camera per render, so there is exactly one listener and it does not need to
/// be an entity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Listener {
    pub position: Vec3,
    pub rotation: Quat,
}

impl Default for Listener {
    fn default() -> Listener {
        Listener {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        }
    }
}

impl Listener {
    /// The listener implied by a camera's transform. The camera looks down local
    /// −Z with +X to the right, the convention
    /// [`Transform::looking_at`](crate::Transform::looking_at) builds.
    pub fn from_pose(pose: &Transform) -> Listener {
        Listener {
            position: pose.translation,
            rotation: pose.rotation,
        }
    }

    /// `(gain, pan)` for a source at `world`.
    ///
    /// Pan is the **sine of the horizontal angle** off the view axis — the
    /// source's camera-space X over its distance — scaled by `pan_width` and
    /// clamped. That is "screen-relative X" done in a way that does not blow up
    /// as the source approaches the camera (dividing by depth would); a source
    /// at the camera's own position lands dead centre instead of at infinity.
    ///
    /// Gain is `reference / max(distance, reference)`: flat inside the reference
    /// radius, `1/d` outside it. Never zero, so a distant sound is quiet rather
    /// than culled — culling is a game's decision, and this function is not
    /// where it belongs.
    pub fn spatialize(&self, world: Vec3, rolloff: Rolloff) -> (f32, f32) {
        let local = self.rotation.inverse() * (world - self.position);
        let distance = local.length();

        let reference = rolloff.reference.max(f32::MIN_POSITIVE);
        let gain = (reference / distance.max(reference)).clamp(0.0, 1.0);

        let pan = if distance > f32::MIN_POSITIVE {
            (local.x / distance * rolloff.pan_width).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // A NaN in either would reach an IIR filter and stay there forever.
        let gain = if gain.is_finite() { gain } else { 0.0 };
        let pan = if pan.is_finite() { pan } else { 0.0 };
        (gain, pan)
    }
}

// ---------------------------------------------------------------------------
// The queue
// ---------------------------------------------------------------------------

/// How many `(tick, event)` pairs the outbox holds before it starts dropping the
/// oldest.
///
/// A host drains this every frame, so in a running game it never exceeds one
/// tick's traffic. The cap exists for the case that has no host — a headless
/// test, a `--replay` run with no device — where an unbounded `Vec` would grow
/// for as long as the sim does. Dropping the *oldest* keeps the most recent
/// sounds, which is the right failure mode for audio.
pub const OUTBOX_CAP: usize = 4096;

/// The queue a `FixedSim` system writes sound into.
///
/// Read [`the module docs`](self) for where the flush sits and why. In short:
/// systems call [`play`](AudioOut::play) during the tick, [`flush_audio`] moves
/// the batch to the outbox once at the end of it, and the host drains the outbox
/// after [`Sim::update`](crate::Sim::update).
#[derive(Resource, Clone, Debug, Default)]
pub struct AudioOut {
    /// This tick's events, in the order they were requested.
    queued: Vec<AudioEvent>,
    /// Flushed batches waiting for a host, each stamped with the tick that
    /// produced it.
    outbox: Vec<(u64, AudioEvent)>,
    next_voice: u32,
    flushes: u64,
    dropped: u64,
}

impl AudioOut {
    pub fn new() -> AudioOut {
        AudioOut::default()
    }

    /// Queue a `Play` and return the id to address it by.
    ///
    /// The id is minted here, in the tick, precisely so that nothing has to be
    /// learned back from the audio thread — see the module docs.
    pub fn play(&mut self, patch: PatchId, seed: u64, gain: f32, pan: f32) -> VoiceId {
        let voice = VoiceId(self.next_voice);
        self.next_voice = self.next_voice.wrapping_add(1);
        self.queued.push(AudioEvent::Play {
            voice,
            patch,
            seed,
            gain: sane(gain, 0.0),
            pan: sane(pan, 0.0).clamp(-1.0, 1.0),
        });
        voice
    }

    /// [`play`](AudioOut::play) with the gain and pan of a source at `world`,
    /// heard from `listener` (DESIGN §8 phase-3 item 4).
    ///
    /// `gain` is the sound's own level; the distance law multiplies it.
    pub fn play_at(
        &mut self,
        patch: PatchId,
        seed: u64,
        gain: f32,
        world: Vec3,
        listener: &Listener,
        rolloff: Rolloff,
    ) -> VoiceId {
        let (attenuation, pan) = listener.spatialize(world, rolloff);
        self.play(patch, seed, gain * attenuation, pan)
    }

    pub fn set_param(&mut self, voice: VoiceId, id: ParamId, value: f32) {
        self.queued.push(AudioEvent::SetParam {
            voice,
            id,
            value: sane(value, 0.0),
        });
    }

    pub fn stop(&mut self, voice: VoiceId) {
        self.queued.push(AudioEvent::Stop { voice });
    }

    /// Events queued so far *this tick*, before the flush. Empty at the top of
    /// every tick.
    pub fn queued(&self) -> &[AudioEvent] {
        &self.queued
    }

    /// Flushed `(tick, event)` pairs waiting for a host.
    pub fn outbox(&self) -> &[(u64, AudioEvent)] {
        &self.outbox
    }

    /// Take everything the outbox holds.
    pub fn drain(&mut self) -> std::vec::Drain<'_, (u64, AudioEvent)> {
        self.outbox.drain(..)
    }

    /// Ticks flushed since the sim started. Equal to the tick count in any run
    /// where the schedule was not tampered with — which is what
    /// `tests/audio.rs` asserts.
    pub fn flushes(&self) -> u64 {
        self.flushes
    }

    /// Events dropped because no host drained the outbox. Should be zero in a
    /// hosted run; a non-zero value in a test means the test is not draining.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Move this tick's queue into the outbox. Called by [`flush_audio`] and by
    /// nothing else.
    fn flush(&mut self, tick: u64) {
        self.flushes += 1;
        if self.queued.is_empty() {
            return;
        }
        self.outbox
            .extend(self.queued.drain(..).map(|event| (tick, event)));
        if self.outbox.len() > OUTBOX_CAP {
            let excess = self.outbox.len() - OUTBOX_CAP;
            self.outbox.drain(..excess);
            self.dropped += excess as u64;
        }
    }
}

/// Replace a non-finite number with a fallback. Game code computing a gain from
/// a division is entitled to produce a NaN once; an IIR filter that receives one
/// is broken for the rest of the session.
#[inline]
fn sane(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

/// `FixedSim` (tail, before [`advance_tick_count`](crate::ecs::advance_tick_count)):
/// publish this tick's audio as one batch.
///
/// The whole of §8's "flushes once per tick — never mid-tick". See the module
/// docs for the three reasons it sits exactly here.
pub fn flush_audio(mut out: ResMut<AudioOut>, tick: Res<TickCount>) {
    let tick = tick.0;
    out.flush(tick);
}

// ---------------------------------------------------------------------------
// Backends
// ---------------------------------------------------------------------------

/// What a host plugs sound into.
///
/// One method, deliberately. DESIGN §8: *"the same patch code serves both hosts;
/// the host is a dumb pump"* — a backend that needed to be asked anything would
/// be a backend the engine could tell apart from the other one.
///
/// Implementors live in `runt-app`: a cpal stream natively, an `AudioWorklet`
/// port on web. Both receive the same slice.
pub trait AudioBackend {
    /// Hand over a batch. Called once per host frame with everything the ticks
    /// in that frame produced, in tick order.
    fn submit(&mut self, events: &[AudioEvent]);
}

/// The backend a program with no audio uses. Drops everything.
///
/// Its existence is the statement that audio is optional: the engine demo runs
/// on this, and it simulates identically to a run with speakers.
#[derive(Clone, Copy, Debug, Default)]
pub struct SilentBackend;

impl AudioBackend for SilentBackend {
    fn submit(&mut self, _events: &[AudioEvent]) {}
}

/// A backend that keeps everything it is given. Tests, and the phase-4 editor's
/// event inspector.
#[derive(Clone, Debug, Default)]
pub struct RecordingBackend {
    pub events: Vec<AudioEvent>,
    pub batches: usize,
}

impl AudioBackend for RecordingBackend {
    fn submit(&mut self, events: &[AudioEvent]) {
        self.batches += 1;
        self.events.extend_from_slice(events);
    }
}
