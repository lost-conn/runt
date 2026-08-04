//! The event wire format — one definition, three readers.
//!
//! A tick's worth of audio events has to reach a synth that may be in another
//! *thread* (cpal's callback) or another *agent* (an `AudioWorkletProcessor`).
//! Both paths carry bytes, so both use this format:
//!
//! ```text
//! runt-core AudioEvent ──runt-app──▶ wire::encode ──▶ bytes
//!                                                      │
//!               native: mpsc → cpal callback ──────────┤
//!               web:    postMessage(ArrayBuffer) ──────┘
//!                                                      ▼
//!                                                 wire::decode ──▶ VoicePool
//! ```
//!
//! ## Why a fixed-size record and not postcard
//!
//! The bank crosses once, at startup, and uses postcard. Events cross **every
//! tick** and are decoded on the audio render thread, where the rules are: no
//! allocation, no fallible parsing that could loop, bounded work. A fixed
//! 32-byte little-endian record gives all three — [`decode`] is a `for` loop
//! over `chunks_exact(32)` with no heap and no error path deeper than "the blob
//! had a ragged tail, ignore it".
//!
//! ## Why little-endian, explicitly
//!
//! wasm is little-endian and every target runt runs on is little-endian, so
//! `to_le_bytes` is free. Writing it out anyway means the format is *specified*
//! rather than inherited from whatever `#[repr(C)]` did today — the same reason
//! `MeshData::content_hash` does not hash raw struct memory.
//!
//! ## Layout (32 bytes, little-endian)
//!
//! ```text
//! 0   u8   kind          0 = Play, 1 = SetParam, 2 = Stop
//! 1   u8   —             reserved, written as 0
//! 2   u16  param id      SetParam only
//! 4   u32  voice id
//! 8   u64  patch id      Play only
//! 16  u64  seed          Play only
//! 24  f32  gain / value
//! 28  f32  pan
//! ```
//!
//! Unused fields are written as zero, so the encoding of an event is a pure
//! function of the event — which is what lets `tests/wire.rs` pin the bytes of a
//! known event and catch a silent format change on either side.

use crate::bank::PatchId;
use crate::params::ParamId;

/// Bytes per event record.
pub const EVENT_SIZE: usize = 32;

const KIND_PLAY: u8 = 0;
const KIND_SET_PARAM: u8 = 1;
const KIND_STOP: u8 = 2;

/// A voice the *sim* named, not one the pool assigned.
///
/// The tick mints these from a monotonic counter (see `runt_core::AudioOut`) so
/// that a later `SetParam`/`Stop` can address a note without the sim ever
/// learning anything back from the audio thread. Nothing flows upstream: DESIGN
/// §4's "per-frame code must not feed back into sim state", applied to a
/// subsystem that is not even on the same thread.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VoiceId(pub u32);

/// One instruction for the synth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Event {
    /// Start `patch` on `voice`, seeded, at `gain` and `pan`.
    Play {
        voice: VoiceId,
        patch: PatchId,
        seed: u64,
        /// Linear amplitude, applied on top of the patch's own `gain`.
        gain: f32,
        /// `-1` left … `+1` right, applied at the mix with a constant-power law.
        pan: f32,
    },
    /// Re-aim a running voice. Ignored if the voice has already finished.
    SetParam {
        voice: VoiceId,
        id: ParamId,
        value: f32,
    },
    /// Release `voice`. A patch with a release stage fades; one without ends.
    Stop { voice: VoiceId },
}

impl Event {
    /// The voice this event addresses.
    pub fn voice(&self) -> VoiceId {
        match *self {
            Event::Play { voice, .. } | Event::SetParam { voice, .. } | Event::Stop { voice } => {
                voice
            }
        }
    }

    /// Write this event's 32 bytes.
    pub fn encode(&self, out: &mut [u8; EVENT_SIZE]) {
        out.fill(0);
        match *self {
            Event::Play {
                voice,
                patch,
                seed,
                gain,
                pan,
            } => {
                out[0] = KIND_PLAY;
                out[4..8].copy_from_slice(&voice.0.to_le_bytes());
                out[8..16].copy_from_slice(&patch.0.to_le_bytes());
                out[16..24].copy_from_slice(&seed.to_le_bytes());
                out[24..28].copy_from_slice(&gain.to_le_bytes());
                out[28..32].copy_from_slice(&pan.to_le_bytes());
            }
            Event::SetParam { voice, id, value } => {
                out[0] = KIND_SET_PARAM;
                out[2..4].copy_from_slice(&id.0.to_le_bytes());
                out[4..8].copy_from_slice(&voice.0.to_le_bytes());
                out[24..28].copy_from_slice(&value.to_le_bytes());
            }
            Event::Stop { voice } => {
                out[0] = KIND_STOP;
                out[4..8].copy_from_slice(&voice.0.to_le_bytes());
            }
        }
    }

    /// Read one event back, or `None` for a kind byte this build does not know.
    ///
    /// Forward compatibility, cheaply: an older synth handed a newer event kind
    /// skips it rather than misreading it as a `Play` and making a noise nobody
    /// asked for.
    pub fn decode(bytes: &[u8; EVENT_SIZE]) -> Option<Event> {
        let voice = VoiceId(u32::from_le_bytes(bytes[4..8].try_into().ok()?));
        let a = f32::from_le_bytes(bytes[24..28].try_into().ok()?);
        match bytes[0] {
            KIND_PLAY => Some(Event::Play {
                voice,
                patch: PatchId(u64::from_le_bytes(bytes[8..16].try_into().ok()?)),
                seed: u64::from_le_bytes(bytes[16..24].try_into().ok()?),
                gain: a,
                pan: f32::from_le_bytes(bytes[28..32].try_into().ok()?),
            }),
            KIND_SET_PARAM => Some(Event::SetParam {
                voice,
                id: ParamId(u16::from_le_bytes(bytes[2..4].try_into().ok()?)),
                value: a,
            }),
            KIND_STOP => Some(Event::Stop { voice }),
            _ => None,
        }
    }
}

/// Append `events` to `out`. The one allocation is `out`'s own growth, and a
/// caller that reuses its buffer across ticks makes even that go away.
pub fn encode_into(events: &[Event], out: &mut Vec<u8>) {
    out.reserve(events.len() * EVENT_SIZE);
    let mut record = [0u8; EVENT_SIZE];
    for event in events {
        event.encode(&mut record);
        out.extend_from_slice(&record);
    }
}

/// `encode_into` into a fresh `Vec`.
pub fn encode(events: &[Event]) -> Vec<u8> {
    let mut out = Vec::with_capacity(events.len() * EVENT_SIZE);
    encode_into(events, &mut out);
    out
}

/// Call `apply` for every event in `bytes`.
///
/// **The realtime entry point**: no allocation, no iterator adapters that could
/// hide one, and a ragged tail (a truncated postMessage, a partial ring read) is
/// silently ignored rather than treated as an error the audio thread would have
/// to do something about. Returns the number of events applied.
pub fn decode(bytes: &[u8], mut apply: impl FnMut(Event)) -> usize {
    let mut applied = 0;
    for chunk in bytes.chunks_exact(EVENT_SIZE) {
        // `try_into` rather than an `expect`: `chunks_exact` makes the length
        // provable, but a proof that lives in a panic branch is still a panic
        // branch, and this code runs where a panic is a dead audio thread.
        let Ok(record) = <&[u8; EVENT_SIZE]>::try_from(chunk) else {
            continue;
        };
        if let Some(event) = Event::decode(record) {
            apply(event);
            applied += 1;
        }
    }
    applied
}

/// `decode` collected into a `Vec`. Tests and the native (non-realtime) side.
pub fn decode_all(bytes: &[u8]) -> Vec<Event> {
    let mut out = Vec::with_capacity(bytes.len() / EVENT_SIZE);
    decode(bytes, |e| out.push(e));
    out
}
