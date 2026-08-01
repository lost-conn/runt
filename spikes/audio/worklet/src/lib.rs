//! Raw wasm module for the AudioWorklet. **No wasm-bindgen.**
//!
//! Why: `AudioWorkletGlobalScope` has no `TextEncoder`/`TextDecoder`, no
//! `fetch`, and no ES-module `import` in older engines. wasm-bindgen's glue
//! needs the first two (rustwasm/wasm-bindgen#2367, open since 2020) and its
//! `--target web` loader needs the third. A plain `cdylib` with `extern "C"`
//! exports needs *none* of them: the JS side gets a `WebAssembly.Instance`, a
//! handful of numeric functions, and one `Float32Array` view over
//! `instance.exports.memory`. That is the whole ABI.
//!
//! The audio buffers are exposed as raw pointers into linear memory and are
//! **planar**, because that is what `AudioWorkletProcessor.process()` hands us
//! (`outputs[0][channel]`), not interleaved.

use spike_patch::{canonical_render, hash_samples, Patch, PatchParams};

/// Largest render block we will ever be asked for. A worklet quantum is 128
/// today; the spec reserves the right to change it, so leave headroom and
/// clamp rather than trusting the host.
const MAX_FRAMES: usize = 1024;

pub struct Synth {
    patch: Patch,
    interleaved: Vec<f32>,
    left: Vec<f32>,
    right: Vec<f32>,
    last_rms: f32,
}

/// Create a synth. Returns an opaque handle (pointer) the JS side passes back.
/// Leaked on purpose — the worklet owns it for the lifetime of the page.
///
/// # Safety
/// Caller must pass the returned pointer back unmodified.
#[no_mangle]
pub extern "C" fn synth_new(sample_rate: f32) -> *mut Synth {
    let synth = Synth {
        patch: Patch::new(PatchParams::default(), sample_rate as f64),
        interleaved: vec![0.0; MAX_FRAMES * 2],
        left: vec![0.0; MAX_FRAMES],
        right: vec![0.0; MAX_FRAMES],
        last_rms: 0.0,
    };
    Box::into_raw(Box::new(synth))
}

/// # Safety
/// `s` must come from `synth_new`.
unsafe fn synth<'a>(s: *mut Synth) -> &'a mut Synth {
    &mut *s
}

/// Pointer to the planar left-channel buffer (MAX_FRAMES f32s).
///
/// # Safety
/// `s` must come from `synth_new`.
#[no_mangle]
pub unsafe extern "C" fn synth_left_ptr(s: *mut Synth) -> *const f32 {
    synth(s).left.as_ptr()
}

/// # Safety
/// `s` must come from `synth_new`.
#[no_mangle]
pub unsafe extern "C" fn synth_right_ptr(s: *mut Synth) -> *const f32 {
    synth(s).right.as_ptr()
}

#[no_mangle]
pub extern "C" fn synth_max_frames() -> u32 {
    MAX_FRAMES as u32
}

/// Render `frames` frames into the planar buffers. Also stashes the block RMS
/// so a headless test can prove sound is being produced without a speaker.
///
/// # Safety
/// `s` must come from `synth_new`.
#[no_mangle]
pub unsafe extern "C" fn synth_render(s: *mut Synth, frames: u32) {
    let s = synth(s);
    let frames = (frames as usize).min(MAX_FRAMES);

    s.patch.render_stereo(&mut s.interleaved[..frames * 2]);

    let mut sum = 0.0f32;
    for i in 0..frames {
        let (l, r) = (s.interleaved[i * 2], s.interleaved[i * 2 + 1]);
        s.left[i] = l;
        s.right[i] = r;
        sum += l * l;
    }
    s.last_rms = (sum / frames as f32).sqrt();
}

/// # Safety
/// `s` must come from `synth_new`.
#[no_mangle]
pub unsafe extern "C" fn synth_last_rms(s: *mut Synth) -> f32 {
    synth(s).last_rms
}

/// # Safety
/// `s` must come from `synth_new`.
#[no_mangle]
pub unsafe extern "C" fn synth_set_cutoff(s: *mut Synth, hz: f32) {
    synth(s).patch.set_cutoff_hz(hz);
}

/// # Safety
/// `s` must come from `synth_new`.
#[no_mangle]
pub unsafe extern "C" fn synth_set_drone(s: *mut Synth, hz: f32) {
    synth(s).patch.set_drone_hz(hz);
}

/// # Safety
/// `s` must come from `synth_new`.
#[no_mangle]
pub unsafe extern "C" fn synth_set_pluck_hz(s: *mut Synth, hz: f32) {
    synth(s).patch.set_pluck_hz(hz);
}

/// # Safety
/// `s` must come from `synth_new`.
#[no_mangle]
pub unsafe extern "C" fn synth_trigger(s: *mut Synth) {
    synth(s).patch.trigger();
}

// ------------------------------------------------------------ determinism ---
// The canonical 1-second render, hashed. Split into two u32s so the JS side
// never has to deal with i64/BigInt in the exported signature (BigInt works in
// modern engines, but keeping the ABI to i32/f32 keeps this loadable anywhere).

#[no_mangle]
pub extern "C" fn canonical_hash_lo() -> u32 {
    hash_samples(&canonical_render()) as u32
}

#[no_mangle]
pub extern "C" fn canonical_hash_hi() -> u32 {
    (hash_samples(&canonical_render()) >> 32) as u32
}

// Expose the canonical buffer itself so the harness can diff wasm samples
// against native ones and tell "rounding drift" apart from "different sound".
static mut CANON: Option<Vec<f32>> = None;

/// # Safety
/// Single-threaded worklet scope only.
#[no_mangle]
pub unsafe extern "C" fn canonical_ptr() -> *const f32 {
    let c = &raw mut CANON;
    if (*c).is_none() {
        *c = Some(canonical_render());
    }
    (*c).as_ref().unwrap().as_ptr()
}

/// # Safety
/// Call `canonical_ptr` first.
#[no_mangle]
pub unsafe extern "C" fn canonical_len() -> u32 {
    let c = &raw const CANON;
    (*c).as_ref().map(|v| v.len() as u32).unwrap_or(0)
}

// -------------------------------------------------------------- benchmark ---

/// Render `quanta` blocks of 128 frames and return the accumulated RMS (so the
/// optimizer cannot delete the work). JS times the call.
///
/// # Safety
/// `s` must come from `synth_new`.
#[no_mangle]
pub unsafe extern "C" fn synth_bench(s: *mut Synth, quanta: u32) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..quanta {
        if i % 400 == 0 {
            synth(s).patch.trigger();
        }
        synth_render(s, 128);
        acc += synth(s).last_rms;
    }
    acc
}
