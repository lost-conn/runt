//! The AudioWorklet's wasm module — raw `cdylib`, **zero imports** (DESIGN §8).
//!
//! Written in the style of `spikes/audio/worklet/src/lib.rs`, which proved the
//! approach. The whole ABI is: a handle (a pointer), a handful of numeric
//! functions, and `Float32Array`/`Uint8Array` views over
//! `instance.exports.memory`. Nothing crosses as a string, nothing crosses as an
//! object, and the module imports nothing at all — which is what lets the main
//! thread `WebAssembly.compile()` it once and pass the *structured-cloneable*
//! `WebAssembly.Module` to the processor through `processorOptions`, where it is
//! instantiated **synchronously** and is live on the first `process()` call.
//!
//! That single trick is what removes `SharedArrayBuffer`, COOP/COEP and the
//! service worker from DESIGN §8 — see FINDINGS, "COOP/COEP story for GitHub
//! Pages".
//!
//! ## The one rule the JS side must obey
//!
//! **Rust's allocator can grow linear memory, and growing it detaches every
//! existing `ArrayBuffer` view.** Any export that may allocate must be followed
//! by `refreshViews()` on the JS side. In this module exactly two exports
//! allocate:
//!
//! | export | allocates | JS must refresh after |
//! |---|---|---|
//! | [`runt_audio_new`] | builds 16 voice slots | yes (and it returns the handle, so this is unavoidable anyway) |
//! | [`runt_audio_load_bank`] | postcard → `PatchBank` | **yes** |
//! | [`runt_audio_submit`] | no | no |
//! | [`runt_audio_render`] | no | no |
//!
//! `render` and `submit` are the two that run per quantum, and neither
//! allocates — which is the property that makes the views safe to hold across
//! thousands of `process()` calls.
//!
//! ## Panics
//!
//! The `wasm-worklet` profile builds with `panic = "abort"`. A panic on the
//! audio render thread is unrecoverable either way (the processor is dead and
//! the context goes silent); aborting costs no unwind tables. Every export below
//! is written to have no panic path a caller can reach — lengths are clamped,
//! decoding is total, and a bad bank is a return code rather than a fault.

use runt_audio::{PatchBank, VoicePool};

/// Largest render block. A worklet quantum is 128 today and the spec reserves
/// the right to change it, so leave headroom and clamp rather than trust the
/// host.
const MAX_FRAMES: usize = 1024;

/// Scratch for one tick's worth of events. 256 events per quantum is far beyond
/// anything a 60 Hz tick can produce (the pool only has 16 voices); a JS side
/// that somehow has more splits the blob across `process()` calls.
const EVENT_BYTES: usize = 256 * runt_audio::EVENT_SIZE;

/// Scratch for the patch bank blob, copied in once before the first render.
const BANK_BYTES: usize = 64 * 1024;

/// Everything the processor owns, in one allocation the JS side holds a pointer
/// to. Leaked on purpose — the worklet owns it for the lifetime of the page.
pub struct Synth {
    pool: VoicePool,
    left: Vec<f32>,
    right: Vec<f32>,
    events: Vec<u8>,
    bank: Vec<u8>,
    /// Quanta rendered. Lets a headless harness prove the processor is being
    /// called rather than merely constructed.
    quanta: u64,
}

/// Create the synth with an **empty** bank; call [`runt_audio_load_bank`] before
/// expecting sound.
///
/// # Safety
/// The returned pointer must be passed back unmodified to every other export.
#[no_mangle]
pub extern "C" fn runt_audio_new(sample_rate: f32) -> *mut Synth {
    let synth = Synth {
        pool: VoicePool::new(PatchBank::new(), sample_rate),
        left: vec![0.0; MAX_FRAMES],
        right: vec![0.0; MAX_FRAMES],
        events: vec![0; EVENT_BYTES],
        bank: vec![0; BANK_BYTES],
        quanta: 0,
    };
    Box::into_raw(Box::new(synth))
}

/// # Safety
/// `s` must come from [`runt_audio_new`].
unsafe fn synth<'a>(s: *mut Synth) -> &'a mut Synth {
    &mut *s
}

// ---------------------------------------------------------------------- bank

/// Where JS writes the postcard-encoded [`PatchBank`] before calling
/// [`runt_audio_load_bank`].
///
/// # Safety
/// `s` must come from [`runt_audio_new`].
#[no_mangle]
pub unsafe extern "C" fn runt_audio_bank_ptr(s: *mut Synth) -> *mut u8 {
    synth(s).bank.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn runt_audio_bank_capacity() -> u32 {
    BANK_BYTES as u32
}

/// Decode `len` bytes from the bank scratch. Returns the number of presets
/// loaded, or `0` if the blob did not decode — a return code, not a panic,
/// because a malformed bank must leave the audio thread running and silent
/// rather than take the page down.
///
/// **Allocates.** Refresh the JS views afterwards.
///
/// # Safety
/// `s` must come from [`runt_audio_new`].
#[no_mangle]
pub unsafe extern "C" fn runt_audio_load_bank(s: *mut Synth, len: u32) -> u32 {
    let s = synth(s);
    let len = (len as usize).min(s.bank.len());
    match PatchBank::from_bytes(&s.bank[..len]) {
        Ok(bank) => {
            let count = bank.len() as u32;
            s.pool.set_bank(bank);
            count
        }
        Err(_) => 0,
    }
}

// -------------------------------------------------------------------- events

/// Where JS writes wire records before calling [`runt_audio_submit`].
///
/// # Safety
/// `s` must come from [`runt_audio_new`].
#[no_mangle]
pub unsafe extern "C" fn runt_audio_events_ptr(s: *mut Synth) -> *mut u8 {
    synth(s).events.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn runt_audio_events_capacity() -> u32 {
    EVENT_BYTES as u32
}

/// Apply `len` bytes of wire records. Allocation-free; safe to call from inside
/// `process()`. Returns the number of events applied.
///
/// # Safety
/// `s` must come from [`runt_audio_new`].
#[no_mangle]
pub unsafe extern "C" fn runt_audio_submit(s: *mut Synth, len: u32) -> u32 {
    let s = synth(s);
    let len = (len as usize).min(s.events.len());
    // Split the borrow: the pool and the scratch both live in `Synth`.
    let Synth { pool, events, .. } = s;
    pool.submit_bytes(&events[..len]) as u32
}

// -------------------------------------------------------------------- render

/// # Safety
/// `s` must come from [`runt_audio_new`].
#[no_mangle]
pub unsafe extern "C" fn runt_audio_left_ptr(s: *mut Synth) -> *const f32 {
    synth(s).left.as_ptr()
}

/// # Safety
/// `s` must come from [`runt_audio_new`].
#[no_mangle]
pub unsafe extern "C" fn runt_audio_right_ptr(s: *mut Synth) -> *const f32 {
    synth(s).right.as_ptr()
}

#[no_mangle]
pub extern "C" fn runt_audio_max_frames() -> u32 {
    MAX_FRAMES as u32
}

/// Render `frames` frames into the planar buffers. Allocation-free.
///
/// Planar because `outputs[0][channel]` is planar — the JS side `set()`s each
/// channel straight out of linear memory with no interleave pass.
///
/// # Safety
/// `s` must come from [`runt_audio_new`].
#[no_mangle]
pub unsafe extern "C" fn runt_audio_render(s: *mut Synth, frames: u32) {
    let s = synth(s);
    let frames = (frames as usize).min(MAX_FRAMES);
    let Synth {
        pool, left, right, ..
    } = s;
    pool.render_planar(&mut left[..frames], &mut right[..frames]);
    s.quanta += 1;
}

// --------------------------------------------------------------- diagnostics

/// RMS of the last rendered block. The headless proof of life: a browser with a
/// null audio sink still reports a number > 0 once something is sounding.
///
/// # Safety
/// `s` must come from [`runt_audio_new`].
#[no_mangle]
pub unsafe extern "C" fn runt_audio_last_rms(s: *mut Synth) -> f32 {
    synth(s).pool.last_rms()
}

/// # Safety
/// `s` must come from [`runt_audio_new`].
#[no_mangle]
pub unsafe extern "C" fn runt_audio_active_voices(s: *mut Synth) -> u32 {
    synth(s).pool.active_voices() as u32
}

/// # Safety
/// `s` must come from [`runt_audio_new`].
#[no_mangle]
pub unsafe extern "C" fn runt_audio_quanta(s: *mut Synth) -> u32 {
    synth(s).quanta as u32
}

/// Voices started since construction.
///
/// # Safety
/// `s` must come from [`runt_audio_new`].
#[no_mangle]
pub unsafe extern "C" fn runt_audio_played(s: *mut Synth) -> u32 {
    synth(s).pool.stats().played as u32
}

/// Non-finite samples the master guard had to replace. Should stay `0`.
///
/// # Safety
/// `s` must come from [`runt_audio_new`].
#[no_mangle]
pub unsafe extern "C" fn runt_audio_nan_guarded(s: *mut Synth) -> u32 {
    synth(s).pool.stats().nan_guarded as u32
}

/// # Safety
/// `s` must come from [`runt_audio_new`].
#[no_mangle]
pub unsafe extern "C" fn runt_audio_set_master_gain(s: *mut Synth, gain: f32) {
    synth(s).pool.set_master_gain(gain);
}
