//! The voice pool — the mixer **both hosts drive** (DESIGN §8's phase-3 item 4).
//!
//! ```text
//!  wire bytes ──▶ submit_bytes ──▶ apply(Event) ──▶ slot
//!                                                     │ render_mono
//!                                                     ▼
//!                        Σ (gain, constant-power pan) ──▶ master ──▶ tanh ──▶ out
//! ```
//!
//! One [`VoicePool`] lives in the cpal callback natively and one lives in the
//! `AudioWorkletProcessor` on web. They are the *same object* fed the *same
//! bytes*, which is the §8 property that matters: "the same patch code serves
//! both hosts; the host is a dumb pump".
//!
//! ## Fixed slots, no allocation after construction
//!
//! [`MAX_VOICES`] slots are built up front, and **each slot owns one of every
//! patch model**. A `Play` re-triggers the model it needs; it never builds a
//! graph, so nothing on the audio thread allocates. The cost is memory —
//! 16 plucks and 16 drones resident whether or not they sound — and it is small,
//! because fundsp's wavetable oscillators share one global table (the per-voice
//! state is a phase and a filter).
//!
//! The alternative, boxing a fresh patch per note, is the ordinary way to do
//! this and it allocates on the audio thread. That is the thing you are not
//! allowed to do.
//!
//! ## Stealing
//!
//! With every slot busy, a new `Play` takes the **quietest** one, ties going to
//! the **oldest**. Quietest first because the discontinuity a steal introduces is
//! proportional to what was cut: taking the voice already 60 dB down is nearly
//! inaudible, while taking the loudest is a click by construction. Age only
//! breaks ties, which in practice means a rank of silent-but-not-yet-freed
//! voices is consumed oldest-first.
//!
//! ## The master bus
//!
//! `tanh` — DESIGN §8's "soft clip", and the same shape the spike used inside
//! its patch. Two properties earn it the spot: `|tanh(x)| < 1` for every finite
//! `x`, so **no sample can leave this pool outside ±1** no matter how many
//! voices pile up (`tests/pool.rs` asserts it against a deliberately overdriven
//! mix); and it is smooth everywhere, so unlike a hard clamp it adds harmonics
//! gracefully instead of buzzing at the ceiling.
//!
//! A non-finite sample is replaced with zero and counted
//! ([`PoolStats::nan_guarded`]). A NaN in a feedback filter is permanent and a
//! NaN reaching a speaker is worse than a bug report; the counter is there so a
//! test can assert the guard never had to fire rather than let it hide one.

use crate::bank::{PatchBank, PatchDef, PatchId};
use crate::params::ParamId;
use crate::patches::{Drone, Pluck};
use crate::wire::{Event, VoiceId, EVENT_SIZE};

/// Simultaneous voices. DESIGN §8: *"voice count is bounded by design taste, not
/// CPU"* — the spike measured 0.45% of a core for one patch, so sixteen is
/// roughly 7% in the worst case and the real limit is how much a listener can
/// pick apart.
pub const MAX_VOICES: usize = 16;

/// Largest block a single `render` call will accept. A worklet quantum is 128;
/// cpal asks for whatever the device wants, commonly 512 or 1024. Requests above
/// this are rendered in several passes rather than refused.
pub const MAX_BLOCK: usize = 1024;

/// Seconds over which a slot's gain and pan glide to a new target.
///
/// Only [`ParamId::GAIN`]/[`ParamId::PAN`] on a *running* voice glide; a fresh
/// `Play` jumps straight to its target, because a new note gliding in from the
/// previous occupant's stereo position is the bug this smoothing exists to
/// prevent, not a feature.
const SMOOTH_SECS: f32 = 0.005;

/// Which model a slot is currently running.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Pluck,
    Drone,
}

struct Slot {
    pluck: Pluck,
    drone: Drone,
    kind: Option<Kind>,
    voice: VoiceId,
    /// Monotonic counter at the moment this slot was last triggered. Ties in the
    /// steal ranking break towards the smaller value.
    started: u64,
    gain: f32,
    pan: f32,
    /// Target and current mix gains per channel. Panning is smoothed as a pair
    /// of gains rather than as an angle so the per-sample cost is two lerps
    /// instead of a `sin`/`cos`.
    target: (f32, f32),
    current: (f32, f32),
}

impl Slot {
    fn new(sample_rate: f32) -> Slot {
        Slot {
            pluck: Pluck::new(sample_rate),
            drone: Drone::new(sample_rate),
            kind: None,
            voice: VoiceId(u32::MAX),
            started: 0,
            gain: 1.0,
            pan: 0.0,
            target: (0.707, 0.707),
            current: (0.707, 0.707),
        }
    }

    fn active(&self) -> bool {
        match self.kind {
            Some(Kind::Pluck) => self.pluck.active(),
            Some(Kind::Drone) => self.drone.active(),
            None => false,
        }
    }

    /// Envelope level scaled by the slot's own gain — the quantity a steal
    /// compares, and therefore a measure of *how loud this voice is right now*
    /// rather than of how loud it was asked to be.
    fn level(&self) -> f32 {
        let env = match self.kind {
            Some(Kind::Pluck) => self.pluck.level(),
            Some(Kind::Drone) => self.drone.level(),
            None => 0.0,
        };
        env * self.gain
    }

    fn set_mix(&mut self, gain: f32, pan: f32, snap: bool) {
        self.gain = if gain.is_finite() { gain.max(0.0) } else { 0.0 };
        self.pan = if pan.is_finite() { pan.clamp(-1.0, 1.0) } else { 0.0 };
        // Constant power: a source swept across the field keeps its loudness
        // instead of dipping in the middle the way a linear law does.
        let angle = (self.pan + 1.0) * (std::f32::consts::FRAC_PI_4);
        self.target = (self.gain * angle.cos(), self.gain * angle.sin());
        if snap {
            self.current = self.target;
        }
    }

    fn render_mono(&mut self, out: &mut [f32]) {
        match self.kind {
            Some(Kind::Pluck) => self.pluck.render_mono(out),
            Some(Kind::Drone) => self.drone.render_mono(out),
            None => out.fill(0.0),
        }
    }

    fn set_param(&mut self, id: ParamId, value: f32) {
        match id {
            ParamId::GAIN => self.set_mix(value, self.pan, false),
            ParamId::PAN => self.set_mix(self.gain, value, false),
            _ => match self.kind {
                Some(Kind::Pluck) => self.pluck.set_param(id, value),
                Some(Kind::Drone) => self.drone.set_param(id, value),
                None => {}
            },
        }
    }

    fn release(&mut self) {
        match self.kind {
            Some(Kind::Pluck) => self.pluck.release(),
            Some(Kind::Drone) => self.drone.release(),
            None => {}
        }
    }
}

/// Counters. Diagnostics and test assertions only — nothing in the pool branches
/// on them, which is the same rule `CacheStats` follows in `runt-core`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PoolStats {
    /// `Play` events that started a voice.
    pub played: u64,
    /// `Play` events that had to take a slot from a sounding voice.
    pub stolen: u64,
    /// `Play` events naming a patch the bank does not contain. Dropped, never
    /// fatal: a missing sound must not be able to kill an audio thread.
    pub dropped_unknown: u64,
    /// `SetParam`/`Stop` for a voice that had already finished (or been stolen).
    /// Expected in normal play; a large number means a game is talking to notes
    /// it has lost track of.
    pub stale_addressed: u64,
    /// Samples the NaN guard had to replace. **Should be zero.**
    pub nan_guarded: u64,
    /// Largest absolute sample seen *before* the soft clip. Above ~3 the mix is
    /// leaning on the limiter for loudness rather than for safety.
    pub peak_pre_clip: f32,
}

/// A fixed bank of voices, a mixer and a master soft clip.
pub struct VoicePool {
    slots: Vec<Slot>,
    bank: PatchBank,
    scratch: Vec<f32>,
    left: Vec<f32>,
    right: Vec<f32>,
    sample_rate: f32,
    smooth: f32,
    master_gain: f32,
    age: u64,
    stats: PoolStats,
    last_rms: f32,
}

impl VoicePool {
    /// Build the pool. **Allocates** — every graph in every slot is constructed
    /// and `allocate()`d here so that nothing after this point has to.
    pub fn new(bank: PatchBank, sample_rate: f32) -> VoicePool {
        let sample_rate = if sample_rate.is_finite() && sample_rate > 1.0 {
            sample_rate
        } else {
            crate::REFERENCE_SAMPLE_RATE as f32
        };
        let slots = (0..MAX_VOICES).map(|_| Slot::new(sample_rate)).collect();
        VoicePool {
            slots,
            bank,
            scratch: vec![0.0; MAX_BLOCK],
            left: vec![0.0; MAX_BLOCK],
            right: vec![0.0; MAX_BLOCK],
            sample_rate,
            smooth: 1.0 - (-1.0 / (SMOOTH_SECS * sample_rate)).exp(),
            master_gain: 1.0,
            age: 0,
            stats: PoolStats::default(),
            last_rms: 0.0,
        }
    }

    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    pub fn bank(&self) -> &PatchBank {
        &self.bank
    }

    /// Swap the bank. Running voices keep the params they were triggered with —
    /// a bank is read at trigger time and never again — so this cannot glitch
    /// a note that is already sounding.
    pub fn set_bank(&mut self, bank: PatchBank) {
        self.bank = bank;
    }

    /// Master level applied before the soft clip. Above 1.0 this is a drive
    /// control, not a volume control.
    pub fn set_master_gain(&mut self, gain: f32) {
        if gain.is_finite() {
            self.master_gain = gain.clamp(0.0, 8.0);
        }
    }

    pub fn master_gain(&self) -> f32 {
        self.master_gain
    }

    pub fn stats(&self) -> PoolStats {
        self.stats
    }

    /// RMS of the left channel of the most recent render. The headless proof of
    /// life: a browser with no speaker can still report this back over the port.
    pub fn last_rms(&self) -> f32 {
        self.last_rms
    }

    pub fn active_voices(&self) -> usize {
        self.slots.iter().filter(|s| s.active()).count()
    }

    /// The [`VoiceId`] sounding in slot `index`, or `None` if it is free.
    ///
    /// Diagnostics: it is how `tests/pool.rs` observes *which* voice a steal
    /// took, which is the only externally visible consequence of the stealing
    /// policy. Nothing in the pool reads it.
    pub fn slot_voice(&self, index: usize) -> Option<VoiceId> {
        let slot = self.slots.get(index)?;
        slot.active().then_some(slot.voice)
    }

    /// Whether `voice` is still sounding.
    pub fn is_playing(&self, voice: VoiceId) -> bool {
        self.find(voice).is_some()
    }

    // -- control ------------------------------------------------------------

    /// Apply one event.
    pub fn apply(&mut self, event: Event) {
        match event {
            Event::Play {
                voice,
                patch,
                seed,
                gain,
                pan,
            } => self.play(voice, patch, seed, gain, pan),
            Event::SetParam { voice, id, value } => match self.find(voice) {
                Some(at) => self.slots[at].set_param(id, value),
                None => self.stats.stale_addressed += 1,
            },
            Event::Stop { voice } => match self.find(voice) {
                Some(at) => self.slots[at].release(),
                None => self.stats.stale_addressed += 1,
            },
        }
    }

    /// Decode and apply a wire blob (see [`crate::wire`]). Returns the number of
    /// events applied. Allocation-free.
    pub fn submit_bytes(&mut self, bytes: &[u8]) -> usize {
        let mut applied = 0;
        for chunk in bytes.chunks_exact(EVENT_SIZE) {
            let Ok(record) = <&[u8; EVENT_SIZE]>::try_from(chunk) else {
                continue;
            };
            if let Some(event) = Event::decode(record) {
                self.apply(event);
                applied += 1;
            }
        }
        applied
    }

    fn find(&self, voice: VoiceId) -> Option<usize> {
        self.slots
            .iter()
            .position(|s| s.voice == voice && s.active())
    }

    fn play(&mut self, voice: VoiceId, patch: PatchId, seed: u64, gain: f32, pan: f32) {
        let Some(def) = self.bank.get(patch).cloned() else {
            self.stats.dropped_unknown += 1;
            return;
        };
        let (at, stolen) = self.claim();
        self.age += 1;

        let slot = &mut self.slots[at];
        slot.voice = voice;
        slot.started = self.age;
        // `snap`: a new note takes its stereo position immediately. Gliding it in
        // from whatever the slot's previous occupant was doing is audible and
        // wrong.
        slot.set_mix(gain, pan, true);
        match def {
            PatchDef::Pluck(params) => {
                slot.kind = Some(Kind::Pluck);
                slot.pluck.trigger(&params, seed);
            }
            PatchDef::Drone(params) => {
                slot.kind = Some(Kind::Drone);
                slot.drone.trigger(&params, seed);
            }
        }

        self.stats.played += 1;
        if stolen {
            self.stats.stolen += 1;
        }
    }

    /// A slot to start a voice in, and whether taking it interrupted one.
    ///
    /// Free slots first (in order, so a quiet scene reuses slot 0 and the rest
    /// stay cold); otherwise the quietest, ties to the oldest.
    fn claim(&mut self) -> (usize, bool) {
        if let Some(at) = self.slots.iter().position(|s| !s.active()) {
            return (at, false);
        }
        let mut best = 0usize;
        for at in 1..self.slots.len() {
            let (a, b) = (&self.slots[at], &self.slots[best]);
            let quieter = a.level() < b.level();
            let tied_and_older = a.level() == b.level() && a.started < b.started;
            if quieter || tied_and_older {
                best = at;
            }
        }
        (best, true)
    }

    // -- rendering ----------------------------------------------------------

    /// Render `frames` frames into planar channel buffers.
    ///
    /// Planar because that is what `AudioWorkletProcessor.process()` hands out
    /// (`outputs[0][channel]`) — FINDINGS again. The native side interleaves
    /// afterwards, which is the cheaper direction to convert in.
    pub fn render_planar(&mut self, left: &mut [f32], right: &mut [f32]) {
        let frames = left.len().min(right.len());
        let mut done = 0;
        while done < frames {
            let n = (frames - done).min(MAX_BLOCK);
            self.render_block(n);
            left[done..done + n].copy_from_slice(&self.left[..n]);
            right[done..done + n].copy_from_slice(&self.right[..n]);
            done += n;
        }
    }

    /// Render into an interleaved stereo buffer. `out.len()` must be even; a
    /// trailing odd sample is left untouched.
    pub fn render_interleaved(&mut self, out: &mut [f32]) {
        let frames = out.len() / 2;
        let mut done = 0;
        while done < frames {
            let n = (frames - done).min(MAX_BLOCK);
            self.render_block(n);
            for i in 0..n {
                out[(done + i) * 2] = self.left[i];
                out[(done + i) * 2 + 1] = self.right[i];
            }
            done += n;
        }
    }

    /// Mix `n ≤ MAX_BLOCK` frames into `self.left`/`self.right`, then run the
    /// master bus over them. The pool has no absolute timeline — a block is only
    /// ever "the next `n` frames".
    fn render_block(&mut self, n: usize) {
        let n = n.min(MAX_BLOCK);
        self.left[..n].fill(0.0);
        self.right[..n].fill(0.0);

        for index in 0..self.slots.len() {
            if !self.slots[index].active() {
                continue;
            }
            // Split borrows: the scratch buffer belongs to the pool, the voice to
            // the slot, and the mix loop needs both.
            let VoicePool {
                slots,
                scratch,
                left,
                right,
                smooth,
                ..
            } = self;
            let slot = &mut slots[index];
            let buf = &mut scratch[..n];
            slot.render_mono(buf);

            let (mut gl, mut gr) = slot.current;
            let (tl, tr) = slot.target;
            for i in 0..n {
                gl += (tl - gl) * *smooth;
                gr += (tr - gr) * *smooth;
                left[i] += buf[i] * gl;
                right[i] += buf[i] * gr;
            }
            slot.current = (gl, gr);
        }

        // -- master bus ------------------------------------------------------
        let master = self.master_gain;
        let mut peak = self.stats.peak_pre_clip;
        let mut guarded = 0u64;
        let mut energy = 0.0f32;
        for i in 0..n {
            let mut l = self.left[i] * master;
            let mut r = self.right[i] * master;
            if !l.is_finite() || !r.is_finite() {
                // A single non-finite sample would otherwise be latched into
                // every downstream filter forever.
                l = 0.0;
                r = 0.0;
                guarded += 1;
            }
            peak = peak.max(l.abs()).max(r.abs());
            let l = l.tanh();
            let r = r.tanh();
            self.left[i] = l;
            self.right[i] = r;
            energy += l * l;
        }
        self.stats.peak_pre_clip = peak;
        self.stats.nan_guarded += guarded;
        self.last_rms = if n > 0 {
            (energy / n as f32).sqrt()
        } else {
            0.0
        };
    }
}

// ---------------------------------------------------------------------------
// Offline rendering — the determinism harness (DESIGN §8)
// ---------------------------------------------------------------------------

/// Render a scripted session with no device attached.
///
/// `script` is `(block index, event)` pairs, applied at the top of that block —
/// which is exactly what both hosts do with a tick's worth of events, so an
/// offline render and a live one differ only in where the samples go.
pub fn render_offline(
    bank: &PatchBank,
    script: &[(usize, Event)],
    frames: usize,
    block: usize,
    sample_rate: f32,
) -> Vec<f32> {
    let block = block.clamp(1, MAX_BLOCK);
    let mut pool = VoicePool::new(bank.clone(), sample_rate);
    let mut out = vec![0.0f32; frames * 2];
    for (index, chunk) in out.chunks_mut(block * 2).enumerate() {
        for (at, event) in script {
            if *at == index {
                pool.apply(*event);
            }
        }
        pool.render_interleaved(chunk);
    }
    out
}

/// FNV-1a over the raw sample bits.
///
/// Bit-exact on purpose: this is a determinism check, not a similarity check.
/// Taken verbatim from the spike so a number produced here is comparable with
/// one recorded in FINDINGS.
pub fn hash_samples(samples: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for s in samples {
        for b in s.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// The one-second job the determinism tests render twice.
///
/// Deliberately exercises everything that could go non-deterministic: a drone
/// whose LFO phase came from a seed, four plucks whose *notes* came from seeds,
/// a mid-flight `SetParam`, a `Stop`, and enough overlap to make the mix bus and
/// the soft clip do real work.
pub fn canonical_script() -> Vec<(usize, Event)> {
    let pluck = PatchId::new("pluck");
    let drone = PatchId::new("drone");
    vec![
        (
            0,
            Event::Play {
                voice: VoiceId(0),
                patch: drone,
                seed: 0xD0DE,
                gain: 0.5,
                pan: 0.0,
            },
        ),
        (
            8,
            Event::Play {
                voice: VoiceId(1),
                patch: pluck,
                seed: 1,
                gain: 0.8,
                pan: -0.6,
            },
        ),
        (
            60,
            Event::Play {
                voice: VoiceId(2),
                patch: pluck,
                seed: 2,
                gain: 0.8,
                pan: 0.6,
            },
        ),
        (
            120,
            Event::Play {
                voice: VoiceId(3),
                patch: pluck,
                seed: 3,
                gain: 1.0,
                pan: 0.0,
            },
        ),
        (
            140,
            Event::SetParam {
                voice: VoiceId(0),
                id: ParamId::CUTOFF,
                value: 2.0,
            },
        ),
        (
            200,
            Event::Play {
                voice: VoiceId(4),
                patch: pluck,
                seed: 4,
                gain: 0.9,
                pan: -0.25,
            },
        ),
        (300, Event::Stop { voice: VoiceId(0) }),
    ]
}

/// [`canonical_script`] rendered for one second at 48 kHz in 128-frame blocks —
/// the worklet's quantum, so the offline result is the live result.
pub fn canonical_render() -> Vec<f32> {
    render_offline(
        &PatchBank::builtin(),
        &canonical_script(),
        48_000,
        128,
        crate::REFERENCE_SAMPLE_RATE as f32,
    )
}
