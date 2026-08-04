//! Patch parameters — **audio params are content** (DESIGN §8, §6).
//!
//! > *a patch is a `Reflect + Serialize + Hash` param struct plus an explicit
//! > seed, serialized in scene RON, edited through the same reflected panels as
//! > mesh generators, content-addressed by params hash.* — DESIGN §8
//!
//! So these structs are the same shape a generator's params are: plain fields,
//! `serde` with `#[serde(default)]` on every one (an old file gains a new knob
//! without being rewritten), and a `param_hash` taken over the postcard bytes —
//! the same route `GeneratorSpec::param_key` uses in `runt-core`, so "params
//! hash" means one thing in this codebase rather than two.
//!
//! `Reflect` is behind the `reflect` feature, mirroring `runt-core`'s: the
//! editor panels of §8's phase-3 item 5 need it and the wasm player must never
//! pay for it.
//!
//! ## No `f64`, no `Vec<f32>` in the hot path
//!
//! Everything here is read at *trigger* time — the moment a voice starts — and
//! never per sample. The per-sample state lives in [`crate::patches`], built
//! from these once.

use serde::{Deserialize, Serialize};

/// A parameter a running voice can be re-aimed at, mid-note.
///
/// The vocabulary is deliberately tiny and *shared*: a host that wants to fade a
/// drone out does not need to know which patch it is talking to. A patch is free
/// to ignore an id it has no meaning for — silently, because "the game asked a
/// pluck to change its LFO rate" is a content mistake, not a runtime fault, and
/// a panic on the audio thread is unacceptable either way.
///
/// The same four constants are restated in `runt_core::audio::ParamId`. They are
/// the wire's shared vocabulary, and `tests/wire.rs` pins the numbers on this
/// side while `runt-core`'s `tests/audio.rs` pins them on the other.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ParamId(pub u16);

impl ParamId {
    /// Linear amplitude multiplier, `0..`. Applied at the mix, so every patch
    /// understands it.
    pub const GAIN: ParamId = ParamId(0);
    /// Stereo position, `-1` hard left … `+1` hard right. Applied at the mix.
    pub const PAN: ParamId = ParamId(1);
    /// Pitch multiplier on the patch's base frequency. `1.0` is "as authored".
    pub const PITCH: ParamId = ParamId(2);
    /// Filter cutoff multiplier on the patch's authored cutoff.
    pub const CUTOFF: ParamId = ParamId(3);
}

// ---------------------------------------------------------------------------
// Pluck
// ---------------------------------------------------------------------------

/// A pitched ping: two detuned saws through a lowpass whose cutoff rides the
/// amplitude envelope. The pickup/impact material.
///
/// ## Why the cutoff tracks the envelope
///
/// A plucked string is brightest at the instant it is struck and dulls as it
/// decays. One envelope driving both amplitude *and* cutoff is the cheapest
/// convincing version of that, and it is what stops a saw-through-a-static-filter
/// from sounding like a doorbell. [`cutoff_env`](PluckParams::cutoff_env) is the
/// multiplier the cutoff reaches at the envelope's peak.
///
/// ## Why a scale, and how the seed uses it
///
/// [`AudioEvent::Play`](crate::wire::Event::Play) carries a **seed**, not a
/// pitch — the engine has no opinion about notes. So the patch turns the seed
/// into a note: `steps[hash(seed) % steps.len()]` semitones above
/// [`base_hz`](PluckParams::base_hz). Twelve pickups collected in any order
/// therefore ring a pentatonic scale instead of the same tone twelve times, and
/// a patch that wants one fixed pitch (an impact thud) writes `steps: vec![0]`.
///
/// This is the whole answer to "how does a game vary a sound without the engine
/// learning what a note is".
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect))]
#[serde(default)]
pub struct PluckParams {
    /// Root frequency, Hz. `steps` is measured from here.
    pub base_hz: f32,
    /// Semitone offsets the seed selects from. Empty behaves as `[0]`.
    pub steps: Vec<i8>,
    /// Detune of the second oscillator, as a frequency ratio. `1.0` disables it
    /// (the two saws then cancel nothing and just double the level).
    pub detune: f32,
    /// Level of the detuned second oscillator relative to the first.
    pub detune_gain: f32,
    /// Seconds to full level. Short but **not zero** — a zero-length attack is a
    /// step discontinuity, which is exactly what a click is.
    pub attack_s: f32,
    /// Seconds to −60 dB. The note is cut (and its slot freed) once the envelope
    /// passes [`SILENCE`], which is well above subnormal territory.
    pub decay_s: f32,
    /// Filter cutoff at rest, Hz.
    pub cutoff_hz: f32,
    /// Multiplier the cutoff reaches at the envelope's peak. `1.0` = static
    /// filter.
    pub cutoff_env: f32,
    /// Filter Q. Above ~2 this rings audibly; 0.7 is flat.
    pub resonance: f32,
    /// Patch level before the per-event gain.
    pub gain: f32,
    /// Fraction of a semitone the seed may detune the whole note by. Keeps a
    /// scale from sounding like a MIDI file.
    pub jitter_semitones: f32,
}

impl Default for PluckParams {
    fn default() -> PluckParams {
        PluckParams {
            base_hz: 440.0,
            // A minor pentatonic over an octave and a half: any subset of these
            // played in any order is consonant, which is the property a game
            // firing them in player-chosen order needs.
            steps: vec![0, 3, 5, 7, 10, 12, 15, 17],
            detune: 1.005,
            detune_gain: 0.5,
            attack_s: 0.004,
            decay_s: 0.35,
            cutoff_hz: 900.0,
            cutoff_env: 5.0,
            resonance: 0.9,
            gain: 0.5,
            jitter_semitones: 0.04,
        }
    }
}

// ---------------------------------------------------------------------------
// Drone
// ---------------------------------------------------------------------------

/// A sustained pad: detuned saws plus a sub-oscillator, through a lowpass swept
/// by an LFO. The ambience material.
///
/// Unlike a pluck it has no end: it holds until a
/// [`Stop`](crate::wire::Event::Stop), then releases over
/// [`release_s`](DroneParams::release_s). A pool full of drones is therefore a
/// pool that can only steal, which is why ambience is expected to be *one* voice
/// and the pool prefers to steal the quietest thing it can find.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect))]
#[serde(default)]
pub struct DroneParams {
    /// Fundamental, Hz.
    pub base_hz: f32,
    /// Second saw's frequency ratio.
    pub detune: f32,
    /// Level of the sub-oscillator an octave down.
    pub sub_gain: f32,
    /// Filter cutoff at LFO zero, Hz.
    pub cutoff_hz: f32,
    /// Filter Q.
    pub resonance: f32,
    /// Sweep rate, Hz. Fractions of a hertz are the point.
    pub lfo_hz: f32,
    /// Fraction of `cutoff_hz` the LFO sweeps, `0..1`. Clamped below 1 in the
    /// graph so the filter frequency can never reach zero.
    pub lfo_depth: f32,
    /// Seconds to full level. Long: a drone that arrives is a sound effect.
    pub attack_s: f32,
    /// Seconds from a `Stop` to silence.
    pub release_s: f32,
    /// Patch level before the per-event gain.
    pub gain: f32,
    /// Fraction of a semitone the seed may detune by, so two ambience voices
    /// with different seeds beat against each other instead of phasing.
    pub jitter_semitones: f32,
}

impl Default for DroneParams {
    fn default() -> DroneParams {
        DroneParams {
            base_hz: 55.0,
            detune: 1.006,
            sub_gain: 0.7,
            cutoff_hz: 420.0,
            resonance: 1.4,
            lfo_hz: 0.13,
            lfo_depth: 0.5,
            attack_s: 2.5,
            release_s: 1.5,
            gain: 0.25,
            jitter_semitones: 0.15,
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Envelope level below which a voice is considered finished and its slot freed.
///
/// −80 dB: inaudible under any master gain a game will use, and **six orders of
/// magnitude above `f32::MIN_POSITIVE`**. That gap is the point. A multiplicative
/// decay left to run forever walks into subnormals, where some targets fall off
/// a performance cliff and — worse for us — where flush-to-zero behaviour is not
/// identical everywhere, which would put a hole in DESIGN §8's determinism claim.
/// Cutting the tail at a round number keeps every sample this crate emits a
/// normal float. `tests/determinism.rs` scans for subnormals and finds none.
pub const SILENCE: f32 = 1.0e-4;

/// `2^(semitones/12)` — the frequency ratio of a semitone interval.
#[inline]
pub fn semitone_ratio(semitones: f32) -> f32 {
    (semitones * std::f32::consts::LN_2 / 12.0).exp()
}

/// A cheap, explicit, platform-independent bit mixer (splitmix64's finalizer).
///
/// Used to turn an event seed into note choices and detune jitter. It is written
/// out rather than reached for from a crate because DESIGN §3 wants the RNG
/// visible and seeded: this is a pure function of its input on every target, and
/// there is no hidden state anywhere.
#[inline]
pub fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// `mix64` mapped onto `[-1, 1)`. Deterministic, and identical on every target
/// because it is integer arithmetic followed by one exact division.
#[inline]
pub fn mix_bipolar(seed: u64, salt: u64) -> f32 {
    let bits = (mix64(seed ^ salt.wrapping_mul(0x517c_c1b7_2722_0a95)) >> 40) as u32; // 24 bits
    (bits as f32 / 8_388_608.0) - 1.0
}
