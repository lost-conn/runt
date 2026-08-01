//! The spike's DSP patch, shared verbatim by the native host (cpal) and the
//! wasm worklet. The whole point is that ONE piece of Rust is the synth and the
//! hosts are dumb pumps — that is the property runt's audio module needs.
//!
//! Patch: a detuned saw drone through an LFO-swept lowpass, plus a triggerable
//! pluck (gate -> asymmetric follower envelope -> pitched saw -> its own
//! envelope-tracked lowpass). Cutoff, drone pitch, pluck pitch and the trigger
//! are all live parameters.

use fundsp::prelude32::*;

/// Everything that defines the sound. `Copy` + plain fields on purpose: this is
/// the shape a runt scene file would serialize (DESIGN §8: "patches are param
/// structs, seeded, serialized in scene files like generators").
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PatchParams {
    pub drone_hz: f32,
    pub drone_detune: f32,
    pub drone_gain: f32,
    pub cutoff_hz: f32,
    pub resonance: f32,
    pub lfo_rate_hz: f32,
    /// Fraction of `cutoff_hz` the LFO sweeps, 0..1. Clamped below 1 so the
    /// filter frequency can never reach 0.
    pub lfo_depth: f32,
    pub pluck_hz: f32,
    pub pluck_decay_s: f32,
    pub pluck_gain: f32,
    /// Seeds every stochastic node in the graph. Same seed -> same samples.
    pub seed: u64,
}

impl Default for PatchParams {
    fn default() -> Self {
        Self {
            drone_hz: 55.0,
            drone_detune: 1.006,
            drone_gain: 0.22,
            cutoff_hz: 700.0,
            resonance: 1.6,
            lfo_rate_hz: 0.17,
            lfo_depth: 0.55,
            pluck_hz: 440.0,
            pluck_decay_s: 0.35,
            pluck_gain: 0.5,
            seed: 0xA17D_10DE,
        }
    }
}

/// A built, running instance of the patch. Not `Send`-friendly by design —
/// it lives on whichever thread pumps audio, and parameters cross the boundary
/// as plain numbers (postMessage on web, an atomic/queue natively).
pub struct Patch {
    net: Box<dyn AudioUnit>,
    drone_hz: Shared,
    cutoff_hz: Shared,
    pluck_hz: Shared,
    gate: Shared,
    /// Quanta remaining that the gate should stay open. The gate is a
    /// sample-rate signal, so a trigger is "hold 1.0 for one block".
    gate_quanta: u32,
    params: PatchParams,
    sample_rate: f64,
}

impl Patch {
    pub fn new(params: PatchParams, sample_rate: f64) -> Self {
        let drone_hz = shared(params.drone_hz);
        let cutoff_hz = shared(params.cutoff_hz);
        let pluck_hz = shared(params.pluck_hz);
        let gate = shared(0.0);

        let depth = params.lfo_depth.clamp(0.0, 0.95);

        // --- drone: two detuned saws + a sub, swept lowpass -----------------
        let saws = (var(&drone_hz) >> saw())
            + (var(&drone_hz) * dc(params.drone_detune) >> saw())
            + (var(&drone_hz) * dc(0.5) >> saw()) * dc(0.7);

        // cutoff * (1 + depth*sin(lfo)) -- a signal, so it is smooth and the
        // host can move `cutoff_hz` at any time without a zipper.
        let cutoff_sig =
            var(&cutoff_hz) * (dc(1.0) + sine_hz(params.lfo_rate_hz) * dc(depth));

        let drone =
            (saws * dc(params.drone_gain) | cutoff_sig | dc(params.resonance)) >> lowpass();

        // --- pluck: gate -> AD envelope, pitched saw, envelope-tracked LP ---
        // Two identical followers off the same `gate` rather than a split, so
        // the envelope can shape both amplitude and brightness.
        let env = || var(&gate) >> afollow(0.002, params.pluck_decay_s);
        let tone = var(&pluck_hz) >> saw();
        let pluck_voice = ((tone | env() * dc(5000.0) + dc(250.0) | dc(0.9)) >> lowpass())
            * env()
            * dc(params.pluck_gain);

        // --- sum, DC-block, soft clip, spread to stereo ---------------------
        let mono = (drone + pluck_voice) >> dcblock() >> shape(Tanh(0.9));
        let mut net = Box::new(mono >> pan(0.0)) as Box<dyn AudioUnit>;

        net.set_sample_rate(sample_rate);
        net.allocate();
        // A seeded reset is what makes two runs with the same params identical.
        net.ping(false, AttoHash::new(params.seed));
        net.reset();

        Self {
            net,
            drone_hz,
            cutoff_hz,
            pluck_hz,
            gate,
            gate_quanta: 0,
            params,
            sample_rate,
        }
    }

    pub fn params(&self) -> &PatchParams {
        &self.params
    }

    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    pub fn set_cutoff_hz(&mut self, hz: f32) {
        self.params.cutoff_hz = hz;
        self.cutoff_hz.set_value(hz);
    }

    pub fn set_drone_hz(&mut self, hz: f32) {
        self.params.drone_hz = hz;
        self.drone_hz.set_value(hz);
    }

    pub fn set_pluck_hz(&mut self, hz: f32) {
        self.params.pluck_hz = hz;
        self.pluck_hz.set_value(hz);
    }

    /// Fire the pluck. The gate is held open for one render block; the
    /// follower turns that into an attack.
    pub fn trigger(&mut self) {
        self.gate_quanta = 1;
    }

    /// Render one block of interleaved stereo. `out.len()` must be even.
    /// Trigger state is applied at block granularity — with 128-frame quanta
    /// that is 2.7 ms at 48 kHz, below the perceptual floor for SFX onset.
    pub fn render_stereo(&mut self, out: &mut [f32]) {
        self.gate
            .set_value(if self.gate_quanta > 0 { 1.0 } else { 0.0 });
        self.gate_quanta = self.gate_quanta.saturating_sub(1);

        for frame in out.chunks_mut(2) {
            let (l, r) = self.net.get_stereo();
            frame[0] = l;
            frame[1] = r;
        }
    }
}

/// Deterministic offline render used by the determinism check and the wasm CPU
/// benchmark. `triggers` lists block indices at which the pluck fires, so the
/// interesting (stochastic-looking) part of the patch is exercised too.
pub fn render_offline(
    params: PatchParams,
    sample_rate: f64,
    frames: usize,
    block: usize,
    triggers: &[usize],
) -> Vec<f32> {
    let mut patch = Patch::new(params, sample_rate);
    let mut out = vec![0.0f32; frames * 2];
    let mut block_index = 0usize;

    for chunk in out.chunks_mut(block * 2) {
        if triggers.contains(&block_index) {
            patch.trigger();
        }
        patch.render_stereo(chunk);
        block_index += 1;
    }
    out
}

/// FNV-1a over the raw sample bits. Bit-exact on purpose: this is a
/// determinism check, not a similarity check.
pub fn hash_samples(samples: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for s in samples {
        for b in s.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
    }
    h
}

/// The exact offline job both the native binary and the wasm build run, so the
/// two hashes are directly comparable (cross-platform determinism check).
pub fn canonical_render() -> Vec<f32> {
    render_offline(
        PatchParams::default(),
        48_000.0,
        48_000, // 1 second
        128,
        &[10, 100, 200, 300],
    )
}
