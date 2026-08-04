//! Manual listen step for the built-in patches — the one thing a headless box
//! cannot do for you.
//!
//! ```text
//! cargo run -p runt-audio --features dsp --example audition            # play
//! cargo run -p runt-audio --features dsp --example audition -- wav /tmp/x.wav
//! cargo run -p runt-audio --features dsp --example audition -- analyze
//! ```
//!
//! `play` drives a real cpal stream through the same [`VoicePool`] the game and
//! the worklet use, so what you hear is the shipped mixer and not a rig that
//! resembles it. `analyze` and `wav` need no device.
//!
//! cpal 0.18 (not the 0.15 every tutorial documents): `build_output_stream`
//! takes the config **by value**, `sample_rate` is a bare `u32`, `device.id()`
//! replaces `name()`, and `ErrorKind::Xrun` is the native glitch signal — all
//! per `spikes/audio/FINDINGS.md`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{ErrorKind, OutputCallbackInfo, StreamConfig};
use runt_audio::analyze;
use runt_audio::voice::{canonical_render, hash_samples};
use runt_audio::wire::{Event, VoiceId};
use runt_audio::{PatchBank, PatchId, VoicePool};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None | Some("play") => play(),
        Some("wav") => wav(args.get(1).map(String::as_str).unwrap_or("audition.wav")),
        Some("analyze") => report(),
        Some("bench") => bench(),
        Some(other) => {
            eprintln!("unknown command {other:?}; try play | wav <path> | analyze | bench")
        }
    }
}

/// CPU cost of a full pool, against the realtime budget.
///
/// The spike measured one patch at 10.10 µs native / 11.99 µs wasm against a
/// 2 666 µs budget; this is the same measurement for sixteen voices, which is
/// the number that decides whether `MAX_VOICES` is a taste decision or a
/// performance one.
fn bench() {
    const SR: f32 = 48_000.0;
    const QUANTUM: usize = 128;
    const QUANTA: usize = 20_000; // ~53 s of audio

    let mut pool = VoicePool::new(PatchBank::builtin(), SR);
    let pluck = PatchId::new("pluck");
    let mut buf = vec![0.0f32; QUANTUM * 2];

    // Fill every slot with a long-decaying voice and keep them alive, so this
    // measures a *saturated* pool rather than an idle one.
    let retrigger = |pool: &mut VoicePool, block: usize| {
        for i in 0..runt_audio::MAX_VOICES as u32 {
            pool.apply(Event::Play {
                voice: VoiceId(block as u32 * 32 + i),
                patch: pluck,
                seed: i as u64,
                gain: 0.4,
                pan: (i as f32 / 15.0) * 2.0 - 1.0,
            });
        }
    };

    retrigger(&mut pool, 0);
    for _ in 0..1000 {
        pool.render_interleaved(&mut buf); // warm caches
    }

    let start = std::time::Instant::now();
    for block in 0..QUANTA {
        if block % 100 == 0 {
            retrigger(&mut pool, block);
        }
        pool.render_interleaved(&mut buf);
    }
    let elapsed = start.elapsed();

    let per_quantum_us = elapsed.as_secs_f64() * 1e6 / QUANTA as f64;
    let budget_us = QUANTUM as f64 / SR as f64 * 1e6;
    println!("voices           : {}", runt_audio::MAX_VOICES);
    println!("quanta           : {QUANTA} x {QUANTUM} frames stereo");
    println!("per quantum      : {per_quantum_us:.2} us");
    println!("realtime budget  : {budget_us:.2} us");
    println!(
        "CPU load         : {:.2} % of one core",
        per_quantum_us / budget_us * 100.0
    );
    println!("stats            : {:?}", pool.stats());
}

/// The script both `play` and `wav` render: a drone under a run of plucks that
/// walk across the stereo field, then a chord that forces the mix bus to work.
fn script() -> Vec<(usize, Event)> {
    let pluck = PatchId::new("pluck");
    let drone = PatchId::new("drone");
    let mut events = vec![(
        0,
        Event::Play {
            voice: VoiceId(0),
            patch: drone,
            seed: 0xA11CE,
            gain: 0.6,
            pan: 0.0,
        },
    )];
    for i in 0..12u32 {
        events.push((
            40 + i as usize * 24,
            Event::Play {
                voice: VoiceId(i + 1),
                patch: pluck,
                seed: i as u64,
                gain: 0.8,
                pan: (i as f32 / 11.0) * 2.0 - 1.0,
            },
        ));
    }
    for i in 0..5u32 {
        events.push((
            360,
            Event::Play {
                voice: VoiceId(100 + i),
                patch: pluck,
                seed: 40 + i as u64,
                gain: 1.0,
                pan: 0.0,
            },
        ));
    }
    events.push((520, Event::Stop { voice: VoiceId(0) }));
    events
}

fn play() {
    let host = cpal::default_host();
    let Some(device) = host.default_output_device() else {
        eprintln!("no default output device (headless box?) — try `wav` or `analyze`");
        std::process::exit(2);
    };
    let config = match device.default_output_config() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("default_output_config failed: {e}");
            std::process::exit(2);
        }
    };
    println!("device: {:?}", device.id());
    println!("config: {config:?}");

    let config: StreamConfig = config.into();
    let sample_rate = config.sample_rate as f32;
    let channels = config.channels as usize;

    let mut pool = VoicePool::new(PatchBank::builtin(), sample_rate);
    let mut scratch: Vec<f32> = Vec::new();
    let (tx, rx) = mpsc::channel::<Event>();
    let xruns = Arc::new(AtomicUsize::new(0));
    let xruns_cb = xruns.clone();

    let stream = device
        .build_output_stream(
            config,
            move |data: &mut [f32], _: &OutputCallbackInfo| {
                while let Ok(event) = rx.try_recv() {
                    pool.apply(event);
                }
                let frames = data.len() / channels.max(1);
                scratch.resize(frames * 2, 0.0);
                pool.render_interleaved(&mut scratch);
                for (i, frame) in data.chunks_mut(channels.max(1)).enumerate() {
                    let (l, r) = (scratch[i * 2], scratch[i * 2 + 1]);
                    for (ch, s) in frame.iter_mut().enumerate() {
                        *s = if ch % 2 == 0 { l } else { r };
                    }
                }
            },
            move |err| {
                if matches!(err.kind(), ErrorKind::Xrun) {
                    xruns_cb.fetch_add(1, Ordering::Relaxed);
                }
                eprintln!("stream error: {err}");
            },
            None,
        )
        .expect("build_output_stream");
    stream.play().expect("play");

    // The script's block indices are 128-frame quanta; replay them in wall time
    // so the same sequence the offline render produces is what comes out.
    println!("playing ~12 s: a drone, twelve plucks across the field, a chord");
    let script = script();
    let last = script.iter().map(|(at, _)| *at).max().unwrap_or(0);
    let mut sent = 0usize;
    for block in 0..=last + 120 {
        for (at, event) in &script {
            if *at == block {
                let _ = tx.send(*event);
                sent += 1;
            }
        }
        std::thread::sleep(std::time::Duration::from_micros(
            (128.0 / sample_rate * 1e6) as u64,
        ));
    }
    println!("sent {sent} events; xruns: {}", xruns.load(Ordering::Relaxed));
}

fn wav(path: &str) {
    let sr = 48_000u32;
    let data = runt_audio::render_offline(&PatchBank::builtin(), &script(), sr as usize * 12, 128, sr as f32);
    std::fs::write(path, wav16(&data, sr, 2)).expect("write wav");
    println!(
        "wrote {path} ({} frames, hash {:#018x})",
        data.len() / 2,
        hash_samples(&data)
    );
}

fn report() {
    let buf = canonical_render();
    let mono = analyze::to_mono(&buf);
    let (subnormal, nonfinite) = analyze::anomalies(&buf);
    println!("canonical hash   : {:#018x}", hash_samples(&buf));
    println!("peak             : {:.4}", analyze::peak(&buf));
    println!("rms              : {:.4}", analyze::rms(&mono));
    println!("subnormal/nonfin : {subnormal} / {nonfinite}");
    println!(
        "centroid         : {:.1} Hz",
        analyze::spectral_centroid(&mono[8_000..24_000], 48_000.0)
    );
}

fn wav16(samples: &[f32], sample_rate: u32, channels: u16) -> Vec<u8> {
    let bits = 16u16;
    let byte_rate = sample_rate * channels as u32 * (bits / 8) as u32;
    let block_align = channels * bits / 8;
    let data_len = samples.len() as u32 * 2;
    let mut v = Vec::with_capacity(44 + data_len as usize);
    v.extend(b"RIFF");
    v.extend(&(36 + data_len).to_le_bytes());
    v.extend(b"WAVEfmt ");
    v.extend(&16u32.to_le_bytes());
    v.extend(&1u16.to_le_bytes());
    v.extend(&channels.to_le_bytes());
    v.extend(&sample_rate.to_le_bytes());
    v.extend(&byte_rate.to_le_bytes());
    v.extend(&block_align.to_le_bytes());
    v.extend(&bits.to_le_bytes());
    v.extend(b"data");
    v.extend(&data_len.to_le_bytes());
    for s in samples {
        v.extend(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    v
}
