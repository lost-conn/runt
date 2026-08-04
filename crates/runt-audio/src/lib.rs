//! runt audio — synthesis, worklet-resident (DESIGN §8).
//!
//! > *`fundsp` for synthesis, running inside an `AudioWorklet` on web and on the
//! > cpal callback natively. The same patch code serves both hosts; the host is
//! > a dumb pump.* — DESIGN §8
//!
//! ## Two halves, one crate
//!
//! Everything here is either **description** or **synthesis**, and the split is
//! a cargo feature:
//!
//! | | default build | `--features dsp` |
//! |---|---|---|
//! | [`PatchId`], [`PatchBank`], [`PluckParams`], [`DroneParams`] | yes | yes |
//! | [`wire`] — the byte codec both hosts speak | yes | yes |
//! | [`analyze`] — measurement, for tests with no speakers | yes | yes |
//! | [`VoicePool`], the fundsp graphs | — | yes |
//!
//! The default half has two dependencies (serde, postcard) and no float DSP in
//! it at all. That is deliberate: `demo/ball`'s **wasm module** links the
//! default half — it describes a bank and encodes events — while the synth lives
//! in a *separate* wasm module (`runt-audio-worklet`) that the browser fetches
//! only when audio starts. A game's download does not carry a synthesizer it
//! runs in another thread.
//!
//! It also keeps fundsp's `glam 0.28` out of the engine build. fundsp pulls a
//! glam a semver major behind the workspace's 0.33 (FINDINGS landmine 8); with
//! `dsp` off there is no fundsp, and with it on the only crates affected are the
//! two that actually make sound.
//!
//! ## What crosses which boundary
//!
//! ```text
//!  game (runt-core)          host (runt-app)              synth (this crate, +dsp)
//!  ────────────────          ───────────────              ────────────────────────
//!  AudioOut::play()   ──▶  AudioEvent   ──wire::encode──▶  wire::decode ──▶ VoicePool
//!  (FixedSim)              (flushed once                   native: same process
//!                           per tick)                      web:    postMessage → worklet
//! ```
//!
//! `runt-core` deliberately does **not** depend on this crate (DESIGN §2 keeps
//! the core free of sibling feature crates), so it owns its own `AudioEvent`
//! enum and this crate owns the byte format. [`wire`] is the single definition
//! of that format — `runt-core` encodes *through* it, via `runt-app`, rather
//! than restating it. See [`wire`] for the layout and the round-trip test.
//!
//! ## Determinism (DESIGN §8, §4)
//!
//! Same params + seed + build + platform → bit-identical samples. **Not**
//! cross-platform: the spike measured ~1e-10 divergence between native and wasm
//! libm through IIR state. [`render_offline`] + [`hash_samples`] are the check;
//! `tests/determinism.rs` runs it without pinning a constant that another box
//! would fail on principle rather than on merit.

#![forbid(unsafe_code)]

pub mod analyze;
pub mod bank;
pub mod params;
pub mod wire;

#[cfg(feature = "dsp")]
pub mod patches;
#[cfg(feature = "dsp")]
pub mod voice;

pub use bank::{PatchBank, PatchDef, PatchEntry, PatchId};
pub use params::{DroneParams, ParamId, PluckParams};
pub use wire::{Event, VoiceId, EVENT_SIZE};

#[cfg(feature = "dsp")]
pub use voice::{
    canonical_render, canonical_script, hash_samples, render_offline, PoolStats, VoicePool,
    MAX_VOICES,
};

/// The sample rate everything offline is measured at. Real hosts use whatever
/// the device reports; nothing in here assumes this value except the tests.
pub const REFERENCE_SAMPLE_RATE: f64 = 48_000.0;
