//! The host's audio pump (DESIGN §8).
//!
//! > *Hosts implement one `AudioBackend::submit(&[AudioEvent])` trait; web
//! > serializes to `postMessage`, native pushes to an SPSC queue read by the
//! > cpal callback.* — DESIGN §8
//!
//! Two backends, one seam. Neither contains a note of engine logic: they move
//! bytes to a mixer and that is all.
//!
//! ```text
//!            Sim::drain_audio(&mut backend)
//!                       │
//!        ┌──────────────┴──────────────┐
//!  native                              web
//!  wire::encode                        wire::encode
//!  mpsc::Sender ──▶ cpal callback      postMessage(ArrayBuffer, transfer)
//!                   VoicePool                    │
//!                                        window.runtAudio.submit
//!                                                │
//!                                        AudioWorkletProcessor
//!                                        runt_audio_submit → VoicePool
//! ```
//!
//! The synth is the same `runt_audio::VoicePool` in both cases; on web it is
//! compiled into a **separate wasm module** the browser fetches on first sound,
//! so the game's own module never carries fundsp. On native it is linked in
//! directly, because there is nothing to save by not doing so.
//!
//! ## Why the queue is `std::sync::mpsc` and not a crate
//!
//! DESIGN §8 says "an SPSC queue"; it does not say a dependency. `mpsc` is
//! single-producer here by construction (only the event-loop thread submits) and
//! its receiver side — the only side the audio callback touches — does not
//! allocate. Sends allocate at most a block, on the game thread, which is where
//! allocation is allowed. This is exactly what the spike measured at 1126
//! callbacks with **0 xruns**, so it is a known quantity rather than a guess.
//!
//! ## Why the browser plumbing is JavaScript
//!
//! `crates/runt-app/web/runt-audio.js` builds the `AudioContext`, compiles the
//! worklet module and installs the first-gesture handler; this file finds it on
//! `window` and calls two functions. Writing the same thing through `web-sys`
//! would be the same code with `Reflect::set` in place of object literals, and
//! it would still be JavaScript at runtime. Keeping it in a JS file also means a
//! page that does not include the script simply has no audio — the same
//! opt-in-by-DOM rule the `#runt-status` HUD already follows (see
//! [`Host::sync_status`](crate::Host::sync_status)).

use runt_core::audio::{AudioBackend, AudioEvent, SilentBackend};

/// What a game hands the host to get sound.
///
/// The bank is **bytes**, not a typed bank: `runt-app`'s wasm build must not
/// link the synthesizer, and a game that only describes presets does not need to
/// either. `runt_audio::PatchBank::to_bytes()` produces these; both backends
/// hand them straight to a `VoicePool` that does know how to read them.
pub struct AudioConfig {
    /// postcard-encoded `runt_audio::PatchBank`.
    pub bank: Vec<u8>,
    /// Master level applied before the bus soft clip.
    pub master_gain: f32,
}

impl AudioConfig {
    pub fn new(bank: Vec<u8>) -> AudioConfig {
        AudioConfig {
            bank,
            master_gain: 1.0,
        }
    }

    pub fn with_master_gain(mut self, gain: f32) -> AudioConfig {
        self.master_gain = gain;
        self
    }
}

/// Bring up the platform's backend, or fall back to silence.
///
/// **Never fails.** A missing sound card, a browser without an `AudioWorklet`, a
/// page that did not include `runt-audio.js` — none of those are reasons for a
/// game not to run, so every one of them logs once and returns
/// [`SilentBackend`]. Audio is the one subsystem whose absence a player can
/// tolerate completely.
pub fn start(config: &AudioConfig) -> Box<dyn AudioBackend> {
    match start_inner(config) {
        Ok(backend) => backend,
        Err(e) => {
            log::warn!("audio unavailable, running silent: {e}");
            Box::new(SilentBackend)
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn start_inner(config: &AudioConfig) -> Result<Box<dyn AudioBackend>, String> {
    native::CpalBackend::start(config).map(|b| Box::new(b) as Box<dyn AudioBackend>)
}

#[cfg(target_arch = "wasm32")]
fn start_inner(config: &AudioConfig) -> Result<Box<dyn AudioBackend>, String> {
    web::WorkletBackend::start(config).map(|b| Box::new(b) as Box<dyn AudioBackend>)
}

// ---------------------------------------------------------------------------
// Native — cpal
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::sync::mpsc;

    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use cpal::{ErrorKind, OutputCallbackInfo, StreamConfig};
    use runt_audio::wire::Event;
    use runt_audio::{PatchBank, VoicePool};
    use runt_core::audio::{AudioBackend, AudioEvent};

    use super::AudioConfig;

    pub struct CpalBackend {
        tx: mpsc::Sender<Event>,
        /// Dropping the stream stops the device. Held for exactly that reason.
        _stream: cpal::Stream,
    }

    impl CpalBackend {
        pub fn start(config: &AudioConfig) -> Result<CpalBackend, String> {
            let bank =
                PatchBank::from_bytes(&config.bank).map_err(|e| format!("patch bank: {e}"))?;

            let host = cpal::default_host();
            let device = host
                .default_output_device()
                .ok_or_else(|| "no default output device".to_string())?;
            let supported = device
                .default_output_config()
                .map_err(|e| format!("default_output_config: {e}"))?;
            log::info!("audio device {:?}: {supported:?}", device.id());

            // cpal 0.18 takes the config *by value* (0.15 took `&config`), and
            // `sample_rate` is a bare `u32`. FINDINGS, "cpal 0.18.1 vs
            // documented 0.15".
            let stream_config: StreamConfig = supported.into();
            let sample_rate = stream_config.sample_rate as f32;
            let channels = stream_config.channels.max(1) as usize;

            let mut pool = VoicePool::new(bank, sample_rate);
            pool.set_master_gain(config.master_gain);
            // Sized once, here, so the callback's `resize` is a no-op forever
            // after: allocation on the audio thread is the thing this whole
            // module is arranged to avoid.
            let mut scratch = vec![0.0f32; 4096 * 2];

            let (tx, rx) = mpsc::channel::<Event>();
            let stream = device
                .build_output_stream(
                    stream_config,
                    move |data: &mut [f32], _: &OutputCallbackInfo| {
                        // Drain at block granularity — exactly what the worklet
                        // does at the top of `process()`, and what the tick
                        // boundary already did upstream.
                        while let Ok(event) = rx.try_recv() {
                            pool.apply(event);
                        }
                        let frames = data.len() / channels;
                        if scratch.len() < frames * 2 {
                            scratch.resize(frames * 2, 0.0);
                        }
                        let mix = &mut scratch[..frames * 2];
                        pool.render_interleaved(mix);
                        for (i, frame) in data.chunks_mut(channels).enumerate() {
                            let (l, r) = (mix[i * 2], mix[i * 2 + 1]);
                            for (ch, sample) in frame.iter_mut().enumerate() {
                                // Mono devices get the left channel; anything
                                // wider than stereo gets the pair repeated,
                                // which is what a surround device expects from a
                                // stereo source.
                                *sample = if ch % 2 == 0 { l } else { r };
                            }
                        }
                    },
                    move |err| {
                        if matches!(err.kind(), ErrorKind::Xrun) {
                            log::warn!("audio xrun");
                        } else {
                            log::error!("audio stream error: {err}");
                        }
                    },
                    None,
                )
                .map_err(|e| format!("build_output_stream: {e}"))?;
            stream.play().map_err(|e| format!("stream play: {e}"))?;

            Ok(CpalBackend {
                tx,
                _stream: stream,
            })
        }
    }

    impl AudioBackend for CpalBackend {
        fn submit(&mut self, events: &[AudioEvent]) {
            for event in events {
                // A closed channel means the stream died; there is nothing
                // useful to do about it from the game thread and nothing worth
                // logging sixty times a second.
                let _ = self.tx.send(super::to_wire(*event));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Web — AudioWorklet
// ---------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
mod web {
    use runt_audio::wire;
    use runt_core::audio::{AudioBackend, AudioEvent};
    use wasm_bindgen::JsValue;

    use super::AudioConfig;

    pub struct WorkletBackend {
        api: JsValue,
        submit: js_sys::Function,
        /// Reused between frames so a quiet tick costs no allocation on the
        /// Rust side either.
        scratch: Vec<u8>,
        wire: Vec<runt_audio::wire::Event>,
    }

    impl WorkletBackend {
        pub fn start(config: &AudioConfig) -> Result<WorkletBackend, String> {
            let window = web_sys::window().ok_or_else(|| "no window".to_string())?;
            let api = js_sys::Reflect::get(&window, &JsValue::from_str("runtAudio"))
                .map_err(|_| "window.runtAudio lookup failed".to_string())?;
            if api.is_undefined() || api.is_null() {
                return Err("this page does not include runt-audio.js".to_string());
            }

            let init = function_on(&api, "init")?;
            let submit = function_on(&api, "submit")?;

            // The bank crosses as a plain byte array; JS never looks inside it.
            let bank = js_sys::Uint8Array::from(config.bank.as_slice());
            let options = js_sys::Object::new();
            set(&options, "masterGain", &JsValue::from_f64(config.master_gain as f64))?;
            init.call2(&api, &bank, &options)
                .map_err(|e| format!("runtAudio.init threw: {e:?}"))?;

            Ok(WorkletBackend {
                api,
                submit,
                scratch: Vec::new(),
                wire: Vec::new(),
            })
        }
    }

    impl AudioBackend for WorkletBackend {
        fn submit(&mut self, events: &[AudioEvent]) {
            self.wire.clear();
            self.wire.extend(events.iter().map(|e| super::to_wire(*e)));
            self.scratch.clear();
            wire::encode_into(&self.wire, &mut self.scratch);

            // `Uint8Array::from` copies into a fresh JS buffer, which is exactly
            // what we want: the JS side transfers that buffer to the worklet, so
            // it must not be a view onto wasm memory.
            let bytes = js_sys::Uint8Array::from(self.scratch.as_slice());
            if let Err(e) = self.submit.call1(&self.api, &bytes) {
                log::warn!("runtAudio.submit threw: {e:?}");
            }
        }
    }

    fn function_on(api: &JsValue, name: &str) -> Result<js_sys::Function, String> {
        js_sys::Reflect::get(api, &JsValue::from_str(name))
            .ok()
            .and_then(|v| v.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| format!("window.runtAudio.{name} is not a function"))
    }

    fn set(object: &js_sys::Object, key: &str, value: &JsValue) -> Result<(), String> {
        js_sys::Reflect::set(object, &JsValue::from_str(key), value)
            .map(|_| ())
            .map_err(|_| format!("could not set {key}"))
    }

    use wasm_bindgen::JsCast;
}

// ---------------------------------------------------------------------------
// Translation
// ---------------------------------------------------------------------------

/// `runt-core`'s event → `runt-audio`'s.
///
/// The two enums are deliberately separate types in crates that do not depend on
/// each other (see `runt_core::audio`), and this function is the *only* place
/// the two vocabularies meet. It is total and field-for-field, so a new variant
/// on either side fails to compile here rather than going quiet at runtime.
fn to_wire(event: AudioEvent) -> runt_audio::wire::Event {
    use runt_audio::wire::{Event, VoiceId};
    match event {
        AudioEvent::Play {
            voice,
            patch,
            seed,
            gain,
            pan,
        } => Event::Play {
            voice: VoiceId(voice.0),
            patch: runt_audio::PatchId(patch.0),
            seed,
            gain,
            pan,
        },
        AudioEvent::SetParam { voice, id, value } => Event::SetParam {
            voice: VoiceId(voice.0),
            id: runt_audio::ParamId(id.0),
            value,
        },
        AudioEvent::Stop { voice } => Event::Stop {
            voice: VoiceId(voice.0),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runt_core::audio::{ParamId, PatchId, VoiceId};

    #[test]
    fn the_two_event_vocabularies_translate_field_for_field() {
        // The only bridge between `runt_core::AudioEvent` and
        // `runt_audio::wire::Event`. If the two ever drift, this is where it
        // shows up — as a compile error if a variant appears, and as this test
        // if a field is dropped.
        assert_eq!(
            to_wire(AudioEvent::Play {
                voice: VoiceId(3),
                patch: PatchId::new("pickup"),
                seed: 9,
                gain: 0.5,
                pan: -0.25,
            }),
            runt_audio::wire::Event::Play {
                voice: runt_audio::VoiceId(3),
                patch: runt_audio::PatchId::new("pickup"),
                seed: 9,
                gain: 0.5,
                pan: -0.25,
            }
        );
        assert_eq!(
            to_wire(AudioEvent::SetParam {
                voice: VoiceId(3),
                id: ParamId::CUTOFF,
                value: 2.0,
            }),
            runt_audio::wire::Event::SetParam {
                voice: runt_audio::VoiceId(3),
                id: runt_audio::ParamId::CUTOFF,
                value: 2.0,
            }
        );
        assert_eq!(
            to_wire(AudioEvent::Stop {
                voice: VoiceId(3)
            }),
            runt_audio::wire::Event::Stop {
                voice: runt_audio::VoiceId(3)
            }
        );
    }

    #[test]
    fn a_missing_backend_is_silence_and_not_a_failure() {
        // An empty bank is a legal bank; what this asserts is that `start`
        // returns *something* usable on a box with no sound card, which is the
        // box this test runs on.
        let bank = runt_audio::PatchBank::new().to_bytes().expect("encode");
        let mut backend = start(&AudioConfig::new(bank));
        backend.submit(&[AudioEvent::Stop {
            voice: VoiceId(0),
        }]);
    }
}
