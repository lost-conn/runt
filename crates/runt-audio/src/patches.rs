//! The two synthesis models (DESIGN §8's phase-3 item 1).
//!
//! Both are built from the spike's DSP (`spikes/audio/patch/src/lib.rs`) and
//! both follow the same shape, which is the shape a game synth wants:
//!
//! ```text
//! fundsp graph   the *timbre*: oscillators and a filter, built once,
//!                driven by `Shared`s, never rebuilt
//! Rust state     the *articulation*: the envelope, the note choice, the
//!                LFO phase — updated per sample, fully inspectable
//! ```
//!
//! ## Why the envelope is not a fundsp node
//!
//! It is the one piece that has to change *per trigger*, and every fundsp
//! envelope bakes its times into the graph at construction
//! (`afollow(attack, release)` is a type, not a parameter). Rebuilding a graph
//! to play a note allocates, which is forbidden on an audio thread, so the spike
//! held one gate open for a block and lived with fixed times.
//!
//! Doing it in Rust instead costs one multiply per sample and buys: per-event
//! attack/decay, sample-accurate onsets (rather than the spike's 2.7 ms block
//! granularity), a `level()` the voice pool can steal on, and a tail that is
//! *cut* at a known level rather than decaying into subnormals. See
//! [`SILENCE`](crate::params::SILENCE).
//!
//! The filter cutoff still rides that envelope — the `Shared` is written every
//! sample, which is what `var()` reads, so the sweep is smooth and no graph is
//! touched.
//!
//! ## fundsp 0.23 landmines observed here
//!
//! `prelude32`, not `hacker32` (gone). `lowpass()` takes cutoff and Q as
//! *inputs* so both are sweepable. `Shared` is f32-only and is the live-param
//! channel. `.allocate()` before realtime, then `ping(false, AttoHash)` and
//! `reset()` in that order. All from `spikes/audio/FINDINGS.md`.

use fundsp::prelude32::*;

use crate::params::{
    mix_bipolar, partials_into, semitone_ratio, BassParams, DroneParams, HihatParams, KickParams,
    ParamId, PluckParams, SnareParams, BASS_PARTIALS, PHASER_STAGES, SILENCE, SNARE_PARTIALS,
};

/// Highest fraction of the sample rate a filter cutoff may reach. Above Nyquist
/// an SVF is undefined; a little below it is merely ugly.
const MAX_CUTOFF_FRACTION: f32 = 0.45;

/// Lowest cutoff, Hz. Zero would divide by nothing useful.
const MIN_CUTOFF: f32 = 20.0;

/// dB-to-ratio constants for the two envelope shapes, precomputed as natural
/// logs: −60 dB is a decay, −80 dB (= [`SILENCE`]) is a release that must
/// actually arrive.
const LN_60DB: f32 = 6.907_755; // ln(1000)
const LN_SILENCE: f32 = 9.210_34; // ln(1 / 1e-4)

/// Clamp a cutoff into the range an SVF is defined over.
#[inline]
fn clamp_cutoff(hz: f32, sample_rate: f32) -> f32 {
    if !hz.is_finite() {
        return MIN_CUTOFF;
    }
    hz.clamp(MIN_CUTOFF, sample_rate * MAX_CUTOFF_FRACTION)
}

/// Per-sample multiplier that takes 1.0 down to `target` in `seconds`.
#[inline]
fn decay_coefficient(seconds: f32, sample_rate: f32, ln_target: f32) -> f32 {
    let samples = (seconds.max(0.001) * sample_rate).max(1.0);
    (-ln_target / samples).exp()
}

// ---------------------------------------------------------------------------
// Pluck
// ---------------------------------------------------------------------------

/// A pitched ping (see [`PluckParams`]).
///
/// One instance is built per voice slot at pool construction and re-triggered
/// forever after — including with a *different* preset, which is why every knob
/// that could vary between presets is a `Shared` rather than a `dc()` constant.
pub struct Pluck {
    net: Box<dyn AudioUnit>,
    /// Fundamental of oscillator 1, Hz.
    freq_a: Shared,
    /// Detuned oscillator 2.
    freq_b: Shared,
    /// Level of oscillator 2.
    detune_gain: Shared,
    cutoff: Shared,
    resonance: Shared,

    sample_rate: f32,
    /// Note frequency chosen at trigger time, before [`ParamId::PITCH`].
    note_hz: f32,
    detune: f32,
    pitch_mul: f32,
    cutoff_base: f32,
    cutoff_peak_mul: f32,
    cutoff_mul: f32,
    patch_gain: f32,

    /// `0..1`. Linear up over the attack, exponential down after.
    env: f32,
    attack_step: f32,
    decay_coef: f32,
    attacking: bool,
    active: bool,
}

impl Pluck {
    /// Build the graph. Allocates; call once, off the audio thread.
    pub fn new(sample_rate: f32) -> Pluck {
        let freq_a = shared(440.0);
        let freq_b = shared(442.0);
        let detune_gain = shared(0.5);
        let cutoff = shared(1000.0);
        let resonance = shared(0.9);

        // Two saws, the second detuned and attenuated. `saw()` is a bandlimited
        // wavetable oscillator (a naive saw would alias into a buzz the moment
        // the seed picked a high note).
        let tone = (var(&freq_a) >> saw()) + (var(&freq_b) >> saw()) * var(&detune_gain);
        // `lowpass()` is the 3-input sweepable SVF: (audio, cutoff, Q).
        // `dcblock()` keeps an asymmetric resonant peak from parking the mix bus
        // off zero, where the master soft clip would waste half its headroom.
        let mono = (tone | var(&cutoff) | var(&resonance)) >> lowpass() >> dcblock();

        let mut net = Box::new(mono) as Box<dyn AudioUnit>;
        net.set_sample_rate(sample_rate as f64);
        net.allocate();
        net.reset();

        Pluck {
            net,
            freq_a,
            freq_b,
            detune_gain,
            cutoff,
            resonance,
            sample_rate,
            note_hz: 440.0,
            detune: 1.005,
            pitch_mul: 1.0,
            cutoff_base: 1000.0,
            cutoff_peak_mul: 5.0,
            cutoff_mul: 1.0,
            patch_gain: 0.5,
            env: 0.0,
            attack_step: 1.0,
            decay_coef: 0.999,
            attacking: false,
            active: false,
        }
    }

    /// Aim this voice at `params`, seeded, and strike it.
    ///
    /// The seed does two things and neither of them is "call an RNG": it selects
    /// a scale degree from `params.steps` and it detunes the result by a
    /// fraction of a semitone. Both are pure functions of the seed through
    /// [`mix64`](crate::params::mix64), so the same event makes the same sound
    /// on every run — DESIGN §8's determinism, arranged so that varying a sound
    /// per pickup costs the game one integer.
    pub fn trigger(&mut self, params: &PluckParams, seed: u64) {
        let step = if params.steps.is_empty() {
            0.0
        } else {
            let index = (crate::params::mix64(seed) % params.steps.len() as u64) as usize;
            params.steps[index] as f32
        };
        let jitter = mix_bipolar(seed, 0x9e37) * params.jitter_semitones;
        self.note_hz = (params.base_hz.max(1.0) * semitone_ratio(step + jitter)).clamp(1.0, 20_000.0);
        self.detune = if params.detune.is_finite() && params.detune > 0.0 {
            params.detune
        } else {
            1.0
        };
        self.pitch_mul = 1.0;
        self.cutoff_base = params.cutoff_hz;
        self.cutoff_peak_mul = params.cutoff_env.max(1.0);
        self.cutoff_mul = 1.0;
        self.patch_gain = params.gain.max(0.0);

        self.detune_gain.set_value(params.detune_gain.clamp(0.0, 4.0));
        self.resonance.set_value(params.resonance.clamp(0.1, 8.0));
        self.write_freqs();

        let attack_samples = (params.attack_s.max(0.0005) * self.sample_rate).max(1.0);
        self.attack_step = 1.0 / attack_samples;
        self.decay_coef = decay_coefficient(params.decay_s, self.sample_rate, LN_60DB);
        self.env = 0.0;
        self.attacking = true;
        self.active = true;

        // FINDINGS landmine 4: seed, then reset, in that order. `reset()` also
        // zeroes the oscillator phases and the filter state, which is what makes
        // every strike of the same note identical rather than dependent on how
        // long the slot happened to sit idle.
        self.net.ping(false, AttoHash::new(seed));
        self.net.reset();
    }

    fn write_freqs(&mut self) {
        let f = (self.note_hz * self.pitch_mul).clamp(1.0, self.sample_rate * 0.45);
        self.freq_a.set_value(f);
        self.freq_b.set_value((f * self.detune).clamp(1.0, self.sample_rate * 0.45));
    }

    pub fn set_param(&mut self, id: ParamId, value: f32) {
        if !value.is_finite() {
            return; // A NaN from game code must not reach an IIR.
        }
        match id {
            ParamId::PITCH => {
                self.pitch_mul = value.clamp(0.01, 16.0);
                self.write_freqs();
            }
            ParamId::CUTOFF => self.cutoff_mul = value.clamp(0.01, 16.0),
            _ => {}
        }
    }

    /// A pluck has no sustain, so a `Stop` short-circuits the tail rather than
    /// cutting it: the decay is re-aimed at 25 ms, which is long enough not to
    /// click and short enough to free the slot.
    pub fn release(&mut self) {
        if self.active {
            self.attacking = false;
            self.decay_coef = decay_coefficient(0.025, self.sample_rate, LN_SILENCE);
        }
    }

    pub fn active(&self) -> bool {
        self.active
    }

    /// Current envelope level — what the pool steals on.
    pub fn level(&self) -> f32 {
        self.env * self.patch_gain
    }

    /// Render `out.len()` mono samples, overwriting.
    pub fn render_mono(&mut self, out: &mut [f32]) {
        if !self.active {
            out.fill(0.0);
            return;
        }
        let peak_span = self.cutoff_peak_mul - 1.0;
        let mut i = 0;
        while i < out.len() {
            // Envelope first: the cutoff for *this* sample rides it.
            if self.attacking {
                self.env += self.attack_step;
                if self.env >= 1.0 {
                    self.env = 1.0;
                    self.attacking = false;
                }
            } else {
                self.env *= self.decay_coef;
                if self.env < SILENCE {
                    // Cut, do not fade to nothing: see `SILENCE`. The rest of
                    // the block is filled with exact zeros below.
                    self.env = 0.0;
                    self.active = false;
                    break;
                }
            }
            let cutoff = clamp_cutoff(
                self.cutoff_base * self.cutoff_mul * (1.0 + peak_span * self.env),
                self.sample_rate,
            );
            self.cutoff.set_value(cutoff);
            out[i] = self.net.get_mono() * self.env * self.patch_gain;
            i += 1;
        }
        out[i..].fill(0.0);
    }
}

// ---------------------------------------------------------------------------
// Drone
// ---------------------------------------------------------------------------

/// Sustained ambience (see [`DroneParams`]).
///
/// The LFO is a Rust phase accumulator rather than a `sine_hz()` node, for the
/// same reason the envelope is: its rate is a per-preset parameter and its phase
/// is seeded per voice, neither of which a baked-frequency node can express
/// without a rebuild.
pub struct Drone {
    net: Box<dyn AudioUnit>,
    freq_a: Shared,
    freq_b: Shared,
    freq_sub: Shared,
    sub_gain: Shared,
    cutoff: Shared,
    resonance: Shared,

    sample_rate: f32,
    note_hz: f32,
    detune: f32,
    pitch_mul: f32,
    cutoff_base: f32,
    cutoff_mul: f32,
    lfo_depth: f32,
    lfo_phase: f32,
    lfo_step: f32,
    patch_gain: f32,

    env: f32,
    attack_step: f32,
    release_coef: f32,
    attacking: bool,
    releasing: bool,
    active: bool,
}

impl Drone {
    pub fn new(sample_rate: f32) -> Drone {
        let freq_a = shared(55.0);
        let freq_b = shared(55.3);
        let freq_sub = shared(27.5);
        let sub_gain = shared(0.7);
        let cutoff = shared(400.0);
        let resonance = shared(1.4);

        // Two detuned saws plus a sub an octave down. The 0.4 keeps three
        // stacked oscillators from arriving at the filter already clipping.
        let saws = ((var(&freq_a) >> saw())
            + (var(&freq_b) >> saw())
            + (var(&freq_sub) >> saw()) * var(&sub_gain))
            * dc(0.4);
        let mono = (saws | var(&cutoff) | var(&resonance)) >> lowpass() >> dcblock();

        let mut net = Box::new(mono) as Box<dyn AudioUnit>;
        net.set_sample_rate(sample_rate as f64);
        net.allocate();
        net.reset();

        Drone {
            net,
            freq_a,
            freq_b,
            freq_sub,
            sub_gain,
            cutoff,
            resonance,
            sample_rate,
            note_hz: 55.0,
            detune: 1.006,
            pitch_mul: 1.0,
            cutoff_base: 400.0,
            cutoff_mul: 1.0,
            lfo_depth: 0.5,
            lfo_phase: 0.0,
            lfo_step: 0.0,
            patch_gain: 0.25,
            env: 0.0,
            attack_step: 1.0,
            release_coef: 0.999,
            attacking: false,
            releasing: false,
            active: false,
        }
    }

    pub fn trigger(&mut self, params: &DroneParams, seed: u64) {
        let jitter = mix_bipolar(seed, 0x51ed) * params.jitter_semitones;
        self.note_hz = (params.base_hz.max(1.0) * semitone_ratio(jitter)).clamp(1.0, 8_000.0);
        self.detune = if params.detune.is_finite() && params.detune > 0.0 {
            params.detune
        } else {
            1.0
        };
        self.pitch_mul = 1.0;
        self.cutoff_base = params.cutoff_hz;
        self.cutoff_mul = 1.0;
        self.lfo_depth = params.lfo_depth.clamp(0.0, 0.95);
        // Seeded phase: two ambience voices started on the same tick sweep
        // against each other instead of in lockstep.
        self.lfo_phase = (mix_bipolar(seed, 0x7f4a) * 0.5 + 0.5).clamp(0.0, 1.0);
        self.lfo_step = params.lfo_hz.abs().min(50.0) / self.sample_rate;
        self.patch_gain = params.gain.max(0.0);

        self.sub_gain.set_value(params.sub_gain.clamp(0.0, 4.0));
        self.resonance.set_value(params.resonance.clamp(0.1, 8.0));
        self.write_freqs();

        self.attack_step = 1.0 / (params.attack_s.max(0.001) * self.sample_rate).max(1.0);
        self.release_coef = decay_coefficient(params.release_s, self.sample_rate, LN_SILENCE);
        self.env = 0.0;
        self.attacking = true;
        self.releasing = false;
        self.active = true;

        self.net.ping(false, AttoHash::new(seed));
        self.net.reset();
    }

    fn write_freqs(&mut self) {
        let f = (self.note_hz * self.pitch_mul).clamp(1.0, self.sample_rate * 0.45);
        self.freq_a.set_value(f);
        self.freq_b
            .set_value((f * self.detune).clamp(1.0, self.sample_rate * 0.45));
        self.freq_sub.set_value((f * 0.5).max(1.0));
    }

    pub fn set_param(&mut self, id: ParamId, value: f32) {
        if !value.is_finite() {
            return;
        }
        match id {
            ParamId::PITCH => {
                self.pitch_mul = value.clamp(0.01, 16.0);
                self.write_freqs();
            }
            ParamId::CUTOFF => self.cutoff_mul = value.clamp(0.01, 16.0),
            _ => {}
        }
    }

    /// Enter the release stage. Idempotent — a second `Stop` does not restart
    /// the fade or shorten it.
    pub fn release(&mut self) {
        if self.active {
            self.attacking = false;
            self.releasing = true;
        }
    }

    pub fn active(&self) -> bool {
        self.active
    }

    pub fn level(&self) -> f32 {
        self.env * self.patch_gain
    }

    pub fn render_mono(&mut self, out: &mut [f32]) {
        if !self.active {
            out.fill(0.0);
            return;
        }
        let mut i = 0;
        while i < out.len() {
            if self.releasing {
                self.env *= self.release_coef;
                if self.env < SILENCE {
                    self.env = 0.0;
                    self.active = false;
                    break;
                }
            } else if self.attacking {
                self.env += self.attack_step;
                if self.env >= 1.0 {
                    self.env = 1.0;
                    self.attacking = false;
                }
            }

            self.lfo_phase += self.lfo_step;
            if self.lfo_phase >= 1.0 {
                self.lfo_phase -= 1.0;
            }
            let lfo = (self.lfo_phase * std::f32::consts::TAU).sin();
            let cutoff = clamp_cutoff(
                self.cutoff_base * self.cutoff_mul * (1.0 + self.lfo_depth * lfo),
                self.sample_rate,
            );
            self.cutoff.set_value(cutoff);
            out[i] = self.net.get_mono() * self.env * self.patch_gain;
            i += 1;
        }
        out[i..].fill(0.0);
    }
}

// ---------------------------------------------------------------------------
// Percussion and bass — shared articulation
// ---------------------------------------------------------------------------

/// A downward pitch sweep, linear in **semitones**.
///
/// `addons/godot_synth`'s `pitch_decay_semitones` / `pitch_decay_time`
/// (`synth_engine.gd:696-699`), restated: the offset above the note starts at
/// `semitones` and reaches zero after `seconds`, and the interpolation happens in
/// semitone space rather than in hertz. A kick's four-octave drop sounds
/// completely different done the other way — linear-in-Hz spends 90 % of its
/// life within a tone of the destination — so the space this lerps in is a
/// decision, not an implementation detail.
///
/// The `exp` this costs runs only while the sweep is live. `t` saturates at 1.0
/// and [`finished`](PitchDrop::finished) is what the patches branch on, so an
/// 80 ms sweep costs a transcendental for 80 ms of a 250 ms note and nothing
/// afterwards.
#[derive(Clone, Copy, Debug)]
struct PitchDrop {
    semitones: f32,
    step: f32,
    t: f32,
}

impl PitchDrop {
    fn new() -> PitchDrop {
        PitchDrop {
            semitones: 0.0,
            step: 1.0,
            t: 1.0,
        }
    }

    fn trigger(&mut self, semitones: f32, seconds: f32, sample_rate: f32) {
        let semitones = if semitones.is_finite() {
            semitones.clamp(0.0, 96.0)
        } else {
            0.0
        };
        let samples = (seconds.max(0.0) * sample_rate).max(1.0);
        self.semitones = semitones;
        self.step = 1.0 / samples;
        self.t = if semitones > 0.0 && seconds > 0.0 {
            0.0
        } else {
            1.0
        };
    }

    fn finished(&self) -> bool {
        self.t >= 1.0
    }

    /// The frequency multiplier for this sample, and advance.
    fn next(&mut self) -> f32 {
        if self.t >= 1.0 {
            return 1.0;
        }
        let ratio = semitone_ratio(self.semitones * (1.0 - self.t));
        self.t += self.step;
        ratio
    }
}

/// Which stage a sustaining envelope is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Attack,
    Decay,
    Sustain,
    Release,
    Done,
}

/// A four-stage envelope with a **sustain floor** — the thing
/// [`Pluck`] and [`Drone`] between them cannot express.
///
/// Attack is linear (a step is a click), decay and release are exponential (a
/// linear fade sounds like it stops rather than like it ends). Godot's
/// `synth_engine.gd` uses linear segments throughout; the deviation is
/// deliberate and shared with the two patches that came before this one, so the
/// whole crate has one envelope shape rather than two.
///
/// The decay aims at `sustain` and the release aims at zero from *wherever the
/// envelope currently is*, which is what makes a `Stop` during the attack fade
/// out instead of jumping.
#[derive(Clone, Copy, Debug)]
struct Adsr {
    value: f32,
    sustain: f32,
    attack_step: f32,
    decay_coef: f32,
    release_coef: f32,
    stage: Stage,
}

impl Adsr {
    fn new() -> Adsr {
        Adsr {
            value: 0.0,
            sustain: 0.0,
            attack_step: 1.0,
            decay_coef: 0.999,
            release_coef: 0.999,
            stage: Stage::Done,
        }
    }

    fn trigger(&mut self, attack_s: f32, decay_s: f32, sustain: f32, release_s: f32, sr: f32) {
        self.sustain = if sustain.is_finite() {
            sustain.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.attack_step = 1.0 / (attack_s.max(0.0005) * sr).max(1.0);
        self.decay_coef = decay_coefficient(decay_s, sr, LN_60DB);
        self.release_coef = decay_coefficient(release_s, sr, LN_SILENCE);
        self.value = 0.0;
        self.stage = Stage::Attack;
    }

    fn release(&mut self) {
        if self.stage != Stage::Done {
            self.stage = Stage::Release;
        }
    }

    fn active(&self) -> bool {
        self.stage != Stage::Done
    }

    /// Advance one sample. Returns the level *before* the step, so a note's very
    /// first sample is not already 1/48 000th of the way through its attack.
    #[inline]
    fn next(&mut self) -> f32 {
        let level = self.value;
        match self.stage {
            Stage::Attack => {
                self.value += self.attack_step;
                if self.value >= 1.0 {
                    self.value = 1.0;
                    // A zero sustain means this is a one-shot: the decay runs to
                    // SILENCE and frees the slot, which is what every drum here
                    // wants and what `Stage::Sustain` would prevent.
                    self.stage = Stage::Decay;
                }
            }
            Stage::Decay => {
                // Decay towards `sustain`, not towards zero: the exponential is
                // applied to the *distance above* the floor.
                self.value = self.sustain + (self.value - self.sustain) * self.decay_coef;
                if self.value - self.sustain < SILENCE {
                    self.value = self.sustain;
                    self.stage = if self.sustain < SILENCE {
                        Stage::Done
                    } else {
                        Stage::Sustain
                    };
                }
            }
            Stage::Sustain => {}
            Stage::Release => {
                self.value *= self.release_coef;
                if self.value < SILENCE {
                    self.value = 0.0;
                    self.stage = Stage::Done;
                }
            }
            Stage::Done => return 0.0,
        }
        level
    }
}

/// One `tick` through a mono unit.
#[inline]
fn tick1(unit: &mut dyn AudioUnit, input: f32) -> f32 {
    let mut out = [0.0f32];
    unit.tick(&[input], &mut out);
    out[0]
}

// ---------------------------------------------------------------------------
// Kick
// ---------------------------------------------------------------------------

/// A bass drum (see [`KickParams`]).
///
/// Two outputs out of one graph — body and click — because they need *different
/// envelopes* and a fundsp envelope is baked at construction. Stacking them with
/// `|` and summing in Rust costs one extra float per sample and buys a click
/// that is over in four milliseconds while the body rings for a quarter second.
pub struct Kick {
    /// 0 inputs, 2 outputs: `[body, click]`.
    net: Box<dyn AudioUnit>,
    frame: [f32; 2],
    freq: Shared,
    sine_gain: Shared,
    triangle_gain: Shared,
    click_highpass: Shared,

    sample_rate: f32,
    note_hz: f32,
    pitch_mul: f32,
    drop: PitchDrop,
    env: Adsr,
    click_env: f32,
    click_coef: f32,
    click_gain: f32,
    patch_gain: f32,
    active: bool,
}

impl Kick {
    pub fn new(sample_rate: f32) -> Kick {
        let freq = shared(55.0);
        let sine_gain = shared(1.0);
        let triangle_gain = shared(0.15);
        let click_highpass = shared(1800.0);

        let body = (var(&freq) >> sine()) * var(&sine_gain)
            + (var(&freq) >> triangle()) * var(&triangle_gain);
        // `dc(0.7)` is the highpass Q: flat, because a resonant peak on a noise
        // burst is a whistle.
        let click = (noise() | var(&click_highpass) | dc(0.7)) >> highpass();

        let mut net = Box::new(body | click) as Box<dyn AudioUnit>;
        net.set_sample_rate(sample_rate as f64);
        net.allocate();
        net.reset();

        Kick {
            net,
            frame: [0.0; 2],
            freq,
            sine_gain,
            triangle_gain,
            click_highpass,
            sample_rate,
            note_hz: 55.0,
            pitch_mul: 1.0,
            drop: PitchDrop::new(),
            env: Adsr::new(),
            click_env: 0.0,
            click_coef: 0.9,
            click_gain: 0.0,
            patch_gain: 0.9,
            active: false,
        }
    }

    pub fn trigger(&mut self, params: &KickParams, seed: u64) {
        self.note_hz = params.base_hz.max(1.0).min(self.sample_rate * 0.45);
        self.pitch_mul = 1.0;
        self.sine_gain.set_value(params.sine_gain.clamp(0.0, 4.0));
        self.triangle_gain
            .set_value(params.triangle_gain.clamp(0.0, 4.0));
        self.click_highpass
            .set_value(clamp_cutoff(params.click_highpass_hz, self.sample_rate));
        self.patch_gain = params.gain.max(0.0);

        self.drop
            .trigger(params.pitch_drop_semitones, params.pitch_drop_s, self.sample_rate);
        // sustain 0: a drum has no hold, so the decay runs straight to silence
        // and the slot frees itself.
        self.env
            .trigger(params.attack_s, params.decay_s, 0.0, 0.02, self.sample_rate);
        self.click_gain = params.click_gain.max(0.0);
        self.click_env = 1.0;
        self.click_coef = decay_coefficient(params.click_decay_s, self.sample_rate, LN_SILENCE);
        self.active = true;
        self.write_freq(1.0);

        self.net.ping(false, AttoHash::new(seed));
        self.net.reset();
    }

    #[inline]
    fn write_freq(&mut self, drop_ratio: f32) {
        let f = (self.note_hz * self.pitch_mul * drop_ratio).clamp(1.0, self.sample_rate * 0.45);
        self.freq.set_value(f);
    }

    pub fn set_param(&mut self, id: ParamId, value: f32) {
        if !value.is_finite() {
            return;
        }
        if id == ParamId::PITCH {
            self.pitch_mul = value.clamp(0.01, 16.0);
            if self.drop.finished() {
                self.write_freq(1.0);
            }
        }
    }

    /// A drum's `Stop` is a fast fade, not a cut — same treatment
    /// [`Pluck::release`] gives its tail, for the same reason.
    pub fn release(&mut self) {
        self.env.release();
    }

    pub fn active(&self) -> bool {
        self.active
    }

    pub fn level(&self) -> f32 {
        self.env.value * self.patch_gain
    }

    pub fn render_mono(&mut self, out: &mut [f32]) {
        if !self.active {
            out.fill(0.0);
            return;
        }
        let mut i = 0;
        while i < out.len() {
            if !self.env.active() {
                self.active = false;
                break;
            }
            if !self.drop.finished() {
                let ratio = self.drop.next();
                self.write_freq(ratio);
            }
            let env = self.env.next();
            self.net.tick(&[], &mut self.frame);
            let click = self.frame[1] * self.click_env * self.click_gain;
            self.click_env *= self.click_coef;
            out[i] = (self.frame[0] * env + click) * self.patch_gain;
            i += 1;
        }
        out[i..].fill(0.0);
    }
}

// ---------------------------------------------------------------------------
// Snare
// ---------------------------------------------------------------------------

/// A snare drum (see [`SnareParams`]).
///
/// Four sine partials driven from one frequency `Shared` — an additive stack, so
/// the body's timbre is a preset rather than a waveform choice — crossfaded
/// against white noise and filtered as a single signal.
pub struct Snare {
    net: Box<dyn AudioUnit>,
    freq: Shared,
    partials: [Shared; SNARE_PARTIALS],
    body_gain: Shared,
    noise_gain: Shared,
    cutoff: Shared,
    resonance: Shared,

    sample_rate: f32,
    note_hz: f32,
    pitch_mul: f32,
    cutoff_base: f32,
    cutoff_mul: f32,
    drop: PitchDrop,
    env: Adsr,
    patch_gain: f32,
    active: bool,
}

impl Snare {
    pub fn new(sample_rate: f32) -> Snare {
        let freq = shared(190.0);
        let partials = [shared(1.0), shared(0.5), shared(0.25), shared(0.125)];
        let body_gain = shared(0.3);
        let noise_gain = shared(0.7);
        let cutoff = shared(4900.0);
        let resonance = shared(0.7);

        let body = (var(&freq) >> sine()) * var(&partials[0])
            + ((var(&freq) * dc(2.0)) >> sine()) * var(&partials[1])
            + ((var(&freq) * dc(3.0)) >> sine()) * var(&partials[2])
            + ((var(&freq) * dc(4.0)) >> sine()) * var(&partials[3]);
        // Godot's crossfade, verbatim (`synth_engine.gd:744`): the mix happens
        // before the filter, so both halves share one top end.
        let mixed = body * var(&body_gain) + noise() * var(&noise_gain);
        let mono = (mixed | var(&cutoff) | var(&resonance)) >> lowpass() >> dcblock();

        let mut net = Box::new(mono) as Box<dyn AudioUnit>;
        net.set_sample_rate(sample_rate as f64);
        net.allocate();
        net.reset();

        Snare {
            net,
            freq,
            partials,
            body_gain,
            noise_gain,
            cutoff,
            resonance,
            sample_rate,
            note_hz: 190.0,
            pitch_mul: 1.0,
            cutoff_base: 4900.0,
            cutoff_mul: 1.0,
            drop: PitchDrop::new(),
            env: Adsr::new(),
            patch_gain: 0.8,
            active: false,
        }
    }

    pub fn trigger(&mut self, params: &SnareParams, seed: u64) {
        self.note_hz = params.base_hz.max(1.0);
        self.pitch_mul = 1.0;
        let levels = partials_into::<SNARE_PARTIALS>(&params.partials);
        for (shared_level, value) in self.partials.iter().zip(levels) {
            shared_level.set_value(value);
        }
        let mix = if params.noise_mix.is_finite() {
            params.noise_mix.clamp(0.0, 1.0)
        } else {
            0.7
        };
        self.body_gain.set_value(1.0 - mix);
        self.noise_gain.set_value(mix);
        self.cutoff_base = params.cutoff_hz;
        self.cutoff_mul = 1.0;
        self.cutoff
            .set_value(clamp_cutoff(self.cutoff_base, self.sample_rate));
        self.resonance.set_value(params.resonance.clamp(0.1, 8.0));
        self.patch_gain = params.gain.max(0.0);

        self.drop
            .trigger(params.pitch_drop_semitones, params.pitch_drop_s, self.sample_rate);
        self.env
            .trigger(params.attack_s, params.decay_s, 0.0, 0.02, self.sample_rate);
        self.active = true;
        self.write_freq(1.0);

        self.net.ping(false, AttoHash::new(seed));
        self.net.reset();
    }

    #[inline]
    fn write_freq(&mut self, drop_ratio: f32) {
        // The fourth partial has to stay under Nyquist, not just the fundamental.
        let ceiling = self.sample_rate * 0.45 / SNARE_PARTIALS as f32;
        let f = (self.note_hz * self.pitch_mul * drop_ratio).clamp(1.0, ceiling);
        self.freq.set_value(f);
    }

    pub fn set_param(&mut self, id: ParamId, value: f32) {
        if !value.is_finite() {
            return;
        }
        match id {
            ParamId::PITCH => {
                self.pitch_mul = value.clamp(0.01, 16.0);
                if self.drop.finished() {
                    self.write_freq(1.0);
                }
            }
            ParamId::CUTOFF => {
                self.cutoff_mul = value.clamp(0.01, 16.0);
                self.cutoff.set_value(clamp_cutoff(
                    self.cutoff_base * self.cutoff_mul,
                    self.sample_rate,
                ));
            }
            _ => {}
        }
    }

    pub fn release(&mut self) {
        self.env.release();
    }

    pub fn active(&self) -> bool {
        self.active
    }

    pub fn level(&self) -> f32 {
        self.env.value * self.patch_gain
    }

    pub fn render_mono(&mut self, out: &mut [f32]) {
        if !self.active {
            out.fill(0.0);
            return;
        }
        let mut i = 0;
        while i < out.len() {
            if !self.env.active() {
                self.active = false;
                break;
            }
            if !self.drop.finished() {
                let ratio = self.drop.next();
                self.write_freq(ratio);
            }
            let env = self.env.next();
            out[i] = self.net.get_mono() * env * self.patch_gain;
            i += 1;
        }
        out[i..].fill(0.0);
    }
}

// ---------------------------------------------------------------------------
// Hihat
// ---------------------------------------------------------------------------

/// A hi-hat (see [`HihatParams`]): band-limited noise and a very short envelope.
///
/// The only patch here with no oscillator and no pitch. `set_param` accepts
/// [`ParamId::CUTOFF`] and ignores [`ParamId::PITCH`], which is not an oversight
/// — see [`HihatParams`].
pub struct Hihat {
    net: Box<dyn AudioUnit>,
    lowpass: Shared,
    highpass: Shared,
    resonance: Shared,

    sample_rate: f32,
    lowpass_base: f32,
    highpass_base: f32,
    cutoff_mul: f32,
    env: Adsr,
    patch_gain: f32,
    active: bool,
}

impl Hihat {
    pub fn new(sample_rate: f32) -> Hihat {
        let lowpass_hz = shared(9000.0);
        let highpass_hz = shared(4000.0);
        let resonance = shared(0.7);

        let mono = (noise() | var(&highpass_hz) | var(&resonance))
            >> highpass()
            >> (pass() | var(&lowpass_hz) | var(&resonance))
            >> lowpass();

        let mut net = Box::new(mono) as Box<dyn AudioUnit>;
        net.set_sample_rate(sample_rate as f64);
        net.allocate();
        net.reset();

        Hihat {
            net,
            lowpass: lowpass_hz,
            highpass: highpass_hz,
            resonance,
            sample_rate,
            lowpass_base: 9000.0,
            highpass_base: 4000.0,
            cutoff_mul: 1.0,
            env: Adsr::new(),
            patch_gain: 0.35,
            active: false,
        }
    }

    pub fn trigger(&mut self, params: &HihatParams, seed: u64) {
        self.lowpass_base = params.lowpass_hz;
        self.highpass_base = params.highpass_hz;
        self.cutoff_mul = 1.0;
        self.resonance.set_value(params.resonance.clamp(0.1, 8.0));
        self.write_cutoffs();
        self.patch_gain = params.gain.max(0.0);
        self.env
            .trigger(params.attack_s, params.decay_s, 0.0, 0.01, self.sample_rate);
        self.active = true;

        // Reseeding is what makes two hats in a row *different* noise rather than
        // the same 60 ms of it twice — and still deterministic, because the seed
        // comes from the event.
        self.net.ping(false, AttoHash::new(seed));
        self.net.reset();
    }

    fn write_cutoffs(&mut self) {
        self.lowpass.set_value(clamp_cutoff(
            self.lowpass_base * self.cutoff_mul,
            self.sample_rate,
        ));
        self.highpass.set_value(clamp_cutoff(
            self.highpass_base * self.cutoff_mul,
            self.sample_rate,
        ));
    }

    pub fn set_param(&mut self, id: ParamId, value: f32) {
        if value.is_finite() && id == ParamId::CUTOFF {
            self.cutoff_mul = value.clamp(0.01, 16.0);
            self.write_cutoffs();
        }
    }

    pub fn release(&mut self) {
        self.env.release();
    }

    pub fn active(&self) -> bool {
        self.active
    }

    pub fn level(&self) -> f32 {
        self.env.value * self.patch_gain
    }

    pub fn render_mono(&mut self, out: &mut [f32]) {
        if !self.active {
            out.fill(0.0);
            return;
        }
        let mut i = 0;
        while i < out.len() {
            if !self.env.active() {
                self.active = false;
                break;
            }
            let env = self.env.next();
            out[i] = self.net.get_mono() * env * self.patch_gain;
            i += 1;
        }
        out[i..].fill(0.0);
    }
}

// ---------------------------------------------------------------------------
// Bass
// ---------------------------------------------------------------------------

/// A sustaining bass with a phaser (see [`BassParams`]).
///
/// ## Three graphs, not one
///
/// ```text
/// tone   0-in 1-out   12 sines (2 unison x 6 partials) -> lowpass
/// phaser 1-in 1-out   PHASER_STAGES swept allpasses
/// Rust                the ADSR, the pitch drop, the LFO, and the feedback path
/// ```
///
/// The split exists because the phaser has **feedback**, and a feedback loop
/// inside a fundsp graph needs `feedback()`, which owns its own delay and cannot
/// have its coefficient swept from outside. One sample of feedback in Rust is
/// the same filter, is one multiply, and leaves the coefficient a `Shared` like
/// every other live parameter in this crate.
///
/// ## Cost, stated
///
/// Twelve sine oscillators is the most expensive voice here by a wide margin,
/// and it is what `bass.tres`'s six harmonics times two unison voices *is*. It
/// is affordable because a bassline is close to monophonic: the pool gives this
/// model three slots (see [`crate::voice`]), not sixteen.
pub struct Bass {
    tone: Box<dyn AudioUnit>,
    phaser: Box<dyn AudioUnit>,
    freq_a: Shared,
    freq_b: Shared,
    partials: [Shared; BASS_PARTIALS],
    cutoff: Shared,
    resonance: Shared,
    allpass_hz: Shared,

    sample_rate: f32,
    note_hz: f32,
    detune: f32,
    pitch_mul: f32,
    cutoff_base: f32,
    cutoff_mul: f32,
    drop: PitchDrop,
    env: Adsr,
    patch_gain: f32,

    lfo_phase: f32,
    lfo_step: f32,
    sweep_centre: f32,
    sweep_span: f32,
    feedback: f32,
    depth: f32,
    last_wet: f32,
    active: bool,
}

impl Bass {
    pub fn new(sample_rate: f32) -> Bass {
        let freq_a = shared(73.0);
        let freq_b = shared(73.1);
        let partials = [
            shared(1.0),
            shared(0.65),
            shared(0.31),
            shared(0.14),
            shared(0.08),
            shared(0.04),
        ];
        let cutoff = shared(3950.0);
        let resonance = shared(0.7);
        let allpass_hz = shared(1000.0);

        // A non-capturing closure: every call has the same argument types, so the
        // (enormous) fundsp return type is inferred once and written out nowhere.
        let partial = |f: &Shared, multiple: f32, level: &Shared| {
            ((var(f) * dc(multiple)) >> sine()) * var(level)
        };
        let stack = |f: &Shared| {
            partial(f, 1.0, &partials[0])
                + partial(f, 2.0, &partials[1])
                + partial(f, 3.0, &partials[2])
                + partial(f, 4.0, &partials[3])
                + partial(f, 5.0, &partials[4])
                + partial(f, 6.0, &partials[5])
        };
        // 0.5 rather than 1.0: two full-level stacks summed would arrive at the
        // filter already past unity, and the master soft clip is a safety net
        // rather than a gain stage.
        let tone_mix = (stack(&freq_a) + stack(&freq_b)) * dc(0.5);
        let tone = (tone_mix | var(&cutoff) | var(&resonance)) >> lowpass() >> dcblock();

        // The allpass chain. `dc(0.7)` is a flat Q: the notches come from the
        // phase rotation, not from resonance.
        let ap = || (pass() | var(&allpass_hz) | dc(0.7)) >> allpass();
        let chain = ap() >> ap() >> ap() >> ap();
        debug_assert_eq!(PHASER_STAGES, 4, "the chain above is written out by hand");

        let mut tone = Box::new(tone) as Box<dyn AudioUnit>;
        tone.set_sample_rate(sample_rate as f64);
        tone.allocate();
        tone.reset();

        let mut phaser = Box::new(chain) as Box<dyn AudioUnit>;
        phaser.set_sample_rate(sample_rate as f64);
        phaser.allocate();
        phaser.reset();

        Bass {
            tone,
            phaser,
            freq_a,
            freq_b,
            partials,
            cutoff,
            resonance,
            allpass_hz,
            sample_rate,
            note_hz: 73.42,
            detune: 1.0,
            pitch_mul: 1.0,
            cutoff_base: 3950.0,
            cutoff_mul: 1.0,
            drop: PitchDrop::new(),
            env: Adsr::new(),
            patch_gain: 0.5,
            lfo_phase: 0.0,
            lfo_step: 0.0,
            sweep_centre: 0.0,
            sweep_span: 0.0,
            feedback: 0.0,
            depth: 0.0,
            last_wet: 0.0,
            active: false,
        }
    }

    pub fn trigger(&mut self, params: &BassParams, seed: u64) {
        let jitter = mix_bipolar(seed, 0x2f19) * params.jitter_semitones;
        self.note_hz = (params.base_hz.max(1.0) * semitone_ratio(jitter)).clamp(1.0, 8_000.0);
        // Unison as a ratio: `unison_cents` is the *total* spread, so the two
        // stacks sit half of it either side of the note.
        self.detune = semitone_ratio(params.unison_cents.clamp(0.0, 100.0) / 200.0);
        self.pitch_mul = 1.0;

        let levels = partials_into::<BASS_PARTIALS>(&params.partials);
        for (shared_level, value) in self.partials.iter().zip(levels) {
            shared_level.set_value(value);
        }
        self.cutoff_base = params.cutoff_hz;
        self.cutoff_mul = 1.0;
        self.cutoff
            .set_value(clamp_cutoff(self.cutoff_base, self.sample_rate));
        self.resonance.set_value(params.resonance.clamp(0.1, 8.0));
        self.patch_gain = params.gain.max(0.0);

        self.drop
            .trigger(params.pitch_drop_semitones, params.pitch_drop_s, self.sample_rate);
        self.env.trigger(
            params.attack_s,
            params.decay_s,
            params.sustain,
            params.release_s,
            self.sample_rate,
        );

        // -- the phaser ----------------------------------------------------
        let lo = clamp_cutoff(params.phaser_min_hz, self.sample_rate);
        let hi = clamp_cutoff(params.phaser_max_hz.max(params.phaser_min_hz), self.sample_rate);
        self.sweep_centre = (lo + hi) * 0.5;
        self.sweep_span = (hi - lo) * 0.5;
        self.lfo_step = params.phaser_rate_hz.abs().min(20.0) / self.sample_rate;
        // Seeded phase, for the same reason `Drone`'s is: two bass notes started
        // together should not sweep in lockstep.
        self.lfo_phase = (mix_bipolar(seed, 0x6c31) * 0.5 + 0.5).clamp(0.0, 1.0);
        self.feedback = params.phaser_feedback.clamp(0.0, 0.95);
        self.depth = if params.phaser_min_hz > 0.0 {
            params.phaser_depth.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.last_wet = 0.0;

        self.active = true;
        self.write_freqs(1.0);

        self.tone.ping(false, AttoHash::new(seed));
        self.tone.reset();
        self.phaser.ping(false, AttoHash::new(seed));
        self.phaser.reset();
    }

    #[inline]
    fn write_freqs(&mut self, drop_ratio: f32) {
        // The sixth partial is the one that has to stay under Nyquist.
        let ceiling = self.sample_rate * 0.45 / BASS_PARTIALS as f32;
        let f = (self.note_hz * self.pitch_mul * drop_ratio).clamp(1.0, ceiling);
        self.freq_a.set_value(f / self.detune);
        self.freq_b.set_value(f * self.detune);
    }

    pub fn set_param(&mut self, id: ParamId, value: f32) {
        if !value.is_finite() {
            return;
        }
        match id {
            ParamId::PITCH => {
                self.pitch_mul = value.clamp(0.01, 16.0);
                if self.drop.finished() {
                    self.write_freqs(1.0);
                }
            }
            ParamId::CUTOFF => {
                self.cutoff_mul = value.clamp(0.01, 16.0);
                self.cutoff.set_value(clamp_cutoff(
                    self.cutoff_base * self.cutoff_mul,
                    self.sample_rate,
                ));
            }
            _ => {}
        }
    }

    pub fn release(&mut self) {
        self.env.release();
    }

    pub fn active(&self) -> bool {
        self.active
    }

    pub fn level(&self) -> f32 {
        self.env.value * self.patch_gain
    }

    pub fn render_mono(&mut self, out: &mut [f32]) {
        if !self.active {
            out.fill(0.0);
            return;
        }
        let mut i = 0;
        while i < out.len() {
            if !self.env.active() {
                self.active = false;
                break;
            }
            if !self.drop.finished() {
                let ratio = self.drop.next();
                self.write_freqs(ratio);
            }

            let dry = self.tone.get_mono();
            let sample = if self.depth > 0.0 {
                self.lfo_phase += self.lfo_step;
                if self.lfo_phase >= 1.0 {
                    self.lfo_phase -= 1.0;
                }
                let lfo = (self.lfo_phase * std::f32::consts::TAU).sin();
                self.allpass_hz.set_value(clamp_cutoff(
                    self.sweep_centre + self.sweep_span * lfo,
                    self.sample_rate,
                ));
                let wet = tick1(
                    self.phaser.as_mut(),
                    dry + self.feedback * self.last_wet,
                );
                // A feedback path is an IIR and an IIR can be pushed into a NaN
                // by a hostile parameter; clamping the state (not the output)
                // keeps that impossible without colouring the sound at sane
                // levels.
                self.last_wet = if wet.is_finite() {
                    wet.clamp(-4.0, 4.0)
                } else {
                    0.0
                };
                dry + self.depth * self.last_wet
            } else {
                dry
            };

            let env = self.env.next();
            out[i] = sample * env * self.patch_gain;
            i += 1;
        }
        out[i..].fill(0.0);
    }
}
