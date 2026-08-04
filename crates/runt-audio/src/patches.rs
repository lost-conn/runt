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

use crate::params::{mix_bipolar, semitone_ratio, DroneParams, ParamId, PluckParams, SILENCE};

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
