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
// Kick
// ---------------------------------------------------------------------------

/// A bass drum: one pitched oscillator whose frequency falls off a cliff, plus
/// an optional noise click on the transient.
///
/// ## Why the pitch envelope is the whole instrument
///
/// A kick is not a note, it is a *glide*. `addons/godot_synth`'s kick patch
/// (`resources/audio/patches/kick.tres`) is a plain sine with
/// `pitch_decay_semitones = 48` over `pitch_decay_time = 0.08` — four octaves in
/// eighty milliseconds — and everything that makes it read as a drum rather than
/// as a very low note happens inside that sweep. So the sweep is the parameter
/// with the most authority here, and it is spelled in semitones-over-seconds
/// exactly as the Godot patch spells it, so the two can be compared without
/// arithmetic.
///
/// The drop is **linear in semitones**, matching `synth_engine.gd:696-699`
/// (`semis = pitch_decay_semitones * (1 - t)`), not linear in Hz. That is
/// audibly different — a linear-in-Hz drop spends almost all its time at the
/// bottom — and it is the reason this is a parameter rather than a hardcoded
/// exponential.
///
/// ## The click is ours, not Godot's
///
/// `kick.tres` has `noise_mix = 0.0`: no click at all. It does not need one,
/// because Godot's kick plays at MIDI 14 through a full-range bus on a desktop.
/// A game that renders through a laptop speaker hears nothing below ~150 Hz, so
/// [`click_gain`](KickParams::click_gain) adds a short highpassed noise burst
/// that survives a small driver. Set it to `0.0` for the Godot-faithful sound.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect))]
#[serde(default)]
pub struct KickParams {
    /// The frequency the sweep lands on, Hz — the drum's actual pitch.
    pub base_hz: f32,
    /// How far above `base_hz` the sweep starts, in semitones.
    pub pitch_drop_semitones: f32,
    /// Seconds the sweep takes. Linear in semitones.
    pub pitch_drop_s: f32,
    /// Level of the sine partial.
    pub sine_gain: f32,
    /// Level of the triangle partial. Triangle has odd harmonics a sine has
    /// none of, which is what a "clicky" kick body is; Godot's is pure sine.
    pub triangle_gain: f32,
    /// Seconds to full level. Short but not zero — see [`PluckParams::attack_s`].
    pub attack_s: f32,
    /// Seconds to −60 dB.
    pub decay_s: f32,
    /// Level of the noise transient. `0.0` disables it and the click costs
    /// nothing but a multiply.
    pub click_gain: f32,
    /// Seconds the click takes to reach −80 dB. Milliseconds, by construction.
    pub click_decay_s: f32,
    /// Highpass on the click, Hz. Above the body, so the two do not fight.
    pub click_highpass_hz: f32,
    /// Patch level before the per-event gain.
    pub gain: f32,
}

impl Default for KickParams {
    fn default() -> KickParams {
        KickParams {
            base_hz: 55.0,
            pitch_drop_semitones: 36.0,
            pitch_drop_s: 0.07,
            sine_gain: 1.0,
            triangle_gain: 0.15,
            attack_s: 0.001,
            decay_s: 0.28,
            click_gain: 0.25,
            click_decay_s: 0.004,
            click_highpass_hz: 1800.0,
            gain: 0.9,
        }
    }
}

// ---------------------------------------------------------------------------
// Snare
// ---------------------------------------------------------------------------

/// A snare: a noise burst crossfaded against a short pitched body, both through
/// one lowpass.
///
/// The structure is `snare.tres` verbatim — `noise_mix = 0.7`, a four-partial
/// additive body, a 24-semitone pitch drop over 40 ms, and a lowpass at
/// `filter_cutoff = 0.7974`, which `synth_engine.gd:563` maps to
/// `20 * 1000^0.7974` ≈ 4.9 kHz. The crossfade is Godot's exact formula
/// (`synth_engine.gd:744`): `body * (1 - mix) + noise * mix`, *then* the filter,
/// *then* the amplitude envelope. Order matters — filtering after the mix is
/// what gives the noise and the body the same top end, which is most of why a
/// snare sounds like one object rather than two.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect))]
#[serde(default)]
pub struct SnareParams {
    /// Fundamental of the body, Hz.
    pub base_hz: f32,
    /// Amplitudes of the body's partials, starting at the fundamental. Padded
    /// with zeros and truncated to [`SNARE_PARTIALS`]; a graph cannot grow after
    /// construction, so the *count* is fixed and only the levels are content.
    pub partials: Vec<f32>,
    /// How far above `base_hz` the body starts, in semitones.
    pub pitch_drop_semitones: f32,
    /// Seconds the body's pitch drop takes.
    pub pitch_drop_s: f32,
    /// `0` = body only, `1` = noise only.
    pub noise_mix: f32,
    pub attack_s: f32,
    /// Seconds to −60 dB.
    pub decay_s: f32,
    /// Lowpass cutoff over the whole mix, Hz.
    pub cutoff_hz: f32,
    /// Filter Q.
    pub resonance: f32,
    pub gain: f32,
}

/// Partials in a [`Snare`](crate::patches::Snare)'s body. Godot's
/// `snare.tres` ships exactly four (`[1, 0.5, 0.25, 0.125]`).
pub const SNARE_PARTIALS: usize = 4;

impl Default for SnareParams {
    fn default() -> SnareParams {
        SnareParams {
            base_hz: 190.0,
            partials: vec![1.0, 0.5, 0.25, 0.125],
            pitch_drop_semitones: 12.0,
            pitch_drop_s: 0.04,
            noise_mix: 0.7,
            attack_s: 0.001,
            decay_s: 0.15,
            cutoff_hz: 4900.0,
            resonance: 0.7,
            gain: 0.8,
        }
    }
}

// ---------------------------------------------------------------------------
// Hihat
// ---------------------------------------------------------------------------

/// A hi-hat: filtered noise with a very short envelope, and nothing else.
///
/// `hihat.tres` is `noise_mix = 1.0` — the oscillator is disconnected entirely
/// (`synth_engine.gd:741-742` replaces the sample rather than mixing it) — with
/// a lowpass at `filter_cutoff = 0.7749` ≈ 4.2 kHz and `decay = 0.06`. So this
/// patch has no pitch at all and ignores [`ParamId::PITCH`]: a "hi-hat an octave
/// up" is not a thing the source project can express and inventing one here
/// would be inventing a sound.
///
/// [`highpass_hz`](HihatParams::highpass_hz) is the one addition. Godot's
/// `noise_highpass` exists and is left at `0.0`; a real closed hat is mostly
/// upper-mid, and a lowpassed-only noise burst reads as a *shaker*. Set it to
/// 20 Hz to bypass and get the source sound exactly.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect))]
#[serde(default)]
pub struct HihatParams {
    pub attack_s: f32,
    /// Seconds to −60 dB. This is the open/closed knob: 60 ms is closed,
    /// 300 ms is open, and there is no third parameter hiding behind it.
    pub decay_s: f32,
    /// Lowpass cutoff, Hz.
    pub lowpass_hz: f32,
    /// Highpass cutoff, Hz. 20 Hz is a bypass.
    pub highpass_hz: f32,
    /// Q shared by both filters.
    pub resonance: f32,
    pub gain: f32,
}

impl Default for HihatParams {
    fn default() -> HihatParams {
        HihatParams {
            attack_s: 0.001,
            decay_s: 0.06,
            lowpass_hz: 9000.0,
            highpass_hz: 4000.0,
            resonance: 0.7,
            gain: 0.35,
        }
    }
}

// ---------------------------------------------------------------------------
// Bass
// ---------------------------------------------------------------------------

/// A sustaining bass: a detuned pair of additive stacks through a lowpass, with
/// a full ADSR and a phaser.
///
/// This is the first patch in the crate with a **sustain stage**, and it is the
/// reason it exists: a [`Pluck`](crate::patches::Pluck) cannot hold a note and a
/// [`Drone`](crate::patches::Drone) takes seconds to arrive, so a bassline whose
/// notes are 0.25–2.5 beats long has nothing to play on. `bass.tres` is
/// `attack 0.005 / decay 0.2 / sustain 0.6 / release 0.1`, and the sequencer
/// that drives it emits a `Stop` at the note's notated end — which is exactly
/// what a release stage is for.
///
/// ## The phaser
///
/// Godot puts an `AudioEffectPhaser` (depth 0.5) on the bass channel's bus
/// (`scenes/audio/main_music.tscn:16-17,35`). fundsp *has* a `phaser()`, but it
/// bakes its LFO into the graph as a `Fn(f32) -> f32` at construction, so its
/// rate could not be a preset parameter. Instead
/// [`crate::patches::Bass`] cascades [`PHASER_STAGES`] sweepable allpasses whose
/// frequency is a `Shared` written from a Rust LFO — the same arrangement
/// [`Drone`](crate::patches::Drone) uses for its filter sweep — and closes the
/// feedback loop outside the graph, one sample deep.
///
/// The knobs are Godot's `AudioEffectPhaser` fields by name: `range_min_hz`,
/// `range_max_hz`, `rate_hz`, `feedback`, `depth`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect))]
#[serde(default)]
pub struct BassParams {
    /// Fundamental, Hz. A sequencer retunes per note with [`ParamId::PITCH`].
    pub base_hz: f32,
    /// Partial amplitudes from the fundamental up. Padded/truncated to
    /// [`BASS_PARTIALS`].
    pub partials: Vec<f32>,
    /// Unison spread, in cents. Two stacks, `±unison_cents / 2` apart.
    pub unison_cents: f32,
    pub cutoff_hz: f32,
    pub resonance: f32,
    pub attack_s: f32,
    /// Seconds from the peak down to `sustain`.
    pub decay_s: f32,
    /// Level held after the decay, `0..1`.
    pub sustain: f32,
    /// Seconds from a `Stop` to silence.
    pub release_s: f32,
    /// A short pitch drop on the attack — Godot's `pitch_decay_semitones = 5`
    /// over `0.034 s`, which is what gives the bass its pluck.
    pub pitch_drop_semitones: f32,
    pub pitch_drop_s: f32,
    /// Fraction of a semitone the seed may detune the note by. Godot's
    /// `pitch_randomize_cents = 4.67`.
    pub jitter_semitones: f32,
    pub gain: f32,
    /// Phaser sweep floor, Hz. `0` disables the phaser entirely.
    pub phaser_min_hz: f32,
    /// Phaser sweep ceiling, Hz.
    pub phaser_max_hz: f32,
    /// Sweep rate, Hz.
    pub phaser_rate_hz: f32,
    /// Allpass-chain feedback, `0..0.95`.
    pub phaser_feedback: f32,
    /// How much of the swept signal reaches the output, `0..1`. Godot's `depth`.
    pub phaser_depth: f32,
}

/// Partials in a [`Bass`](crate::patches::Bass)'s additive stack. Godot's
/// `bass.tres` ships six.
pub const BASS_PARTIALS: usize = 6;

/// Allpass stages in the [`Bass`](crate::patches::Bass) phaser.
///
/// Four. Godot's `AudioEffectPhaser` uses six; four notches is where the effect
/// is unmistakable and each further stage costs a full SVF per sample on a voice
/// that plays continuously. See the module docs on honest gaps.
pub const PHASER_STAGES: usize = 4;

impl Default for BassParams {
    fn default() -> BassParams {
        BassParams {
            base_hz: 73.42, // D2
            partials: vec![1.0, 0.65, 0.31, 0.14, 0.08, 0.04],
            unison_cents: 2.0,
            cutoff_hz: 3950.0,
            resonance: 0.7,
            attack_s: 0.005,
            decay_s: 0.2,
            sustain: 0.6,
            release_s: 0.1,
            pitch_drop_semitones: 5.0,
            pitch_drop_s: 0.034,
            jitter_semitones: 0.0467,
            gain: 0.5,
            phaser_min_hz: 440.0,
            phaser_max_hz: 1600.0,
            phaser_rate_hz: 0.5,
            phaser_feedback: 0.7,
            phaser_depth: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Copy `src` into a fixed-size array of partial amplitudes: extra partials are
/// dropped and missing ones are silent.
///
/// The count is a property of the *graph*, which is built once and never rebuilt
/// (see [`crate::patches`]); the levels are content. This is the function that
/// keeps those two facts from colliding when a preset carries the wrong number.
#[inline]
pub fn partials_into<const N: usize>(src: &[f32]) -> [f32; N] {
    let mut out = [0.0f32; N];
    for (slot, value) in out.iter_mut().zip(src) {
        *slot = if value.is_finite() { value.clamp(-4.0, 4.0) } else { 0.0 };
    }
    out
}

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
