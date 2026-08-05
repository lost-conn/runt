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
        Some("bench") => match args.get(1).map(String::as_str) {
            Some("bgm") => bench_bgm(),
            _ => bench(),
        },
        Some(other) => {
            eprintln!(
                "unknown command {other:?}; try play | wav <path> | analyze | bench [bgm]"
            )
        }
    }
}

/// CPU cost of a full pool, against the realtime budget.
///
/// The spike measured one patch at 10.10 µs native / 11.99 µs wasm against a
/// 2 666 µs budget; this is the same measurement for a **saturated SFX group**,
/// which is the number that decides whether [`PLUCK_VOICES`] is a taste decision
/// or a performance one. Deliberately pathological: it retriggers every slot
/// just under the pluck's decay time, so the group never goes quiet. `bench bgm`
/// is the realistic worst case.
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
    println!("voices           : {} plucks in a {}-slot pool", runt_audio::PLUCK_VOICES, runt_audio::MAX_VOICES);
    println!("quanta           : {QUANTA} x {QUANTUM} frames stereo");
    println!("per quantum      : {per_quantum_us:.2} us");
    println!("realtime budget  : {budget_us:.2} us");
    println!(
        "CPU load         : {:.2} % of one core",
        per_quantum_us / budget_us * 100.0
    );
    println!("stats            : {:?}", pool.stats());
}

/// CPU cost of a **full background-music load**, against the realtime budget.
///
/// This is the measurement the BGM work exists to justify. `bench` above
/// saturates the pool with sixteen copies of the cheapest model; this one runs
/// the expensive ones at the density a song actually produces:
///
/// ```text
/// 3 bass      sustained, retriggered every bar   12 sine oscillators + a
///                                                4-stage allpass chain each
/// 2 kick   \
/// 2 snare   >  every group full, on an 8th-note grid at 170.35 bpm
/// 3 hihat  /
/// 16 pluck    an SFX burst on top of all of it
/// ```
///
/// So it is the **worst case and then some**: every one of the pool's 28 slots
/// sounding at once, which a real game reaches only if a player triggers sixteen
/// overlapping SFX during a bar. The number to read is the per-quantum figure
/// against the 2 666 µs budget a 128-frame quantum has at 48 kHz.
fn bench_bgm() {
    const SR: f32 = 48_000.0;
    const QUANTUM: usize = 128;
    const QUANTA: usize = 20_000; // ~53 s of audio
    /// 170.35 bpm, in quanta. One beat is 48000 * 60 / 170.35 / 128 = 165.1
    /// quanta; the eighth-note grid the drums live on is half of that.
    const EIGHTH_QUANTA: usize = 83;

    // `PatchBank::builtin()`'s music presets, not the 3dimenshift port's. They
    // are the same models with the same partial counts and the same phaser
    // stage count, which is what the cost is made of; the port's numbers differ
    // only in frequencies and envelope times.
    let (bass, kick, snare, hihat, pluck) = (
        PatchId::new("bass"),
        PatchId::new("kick"),
        PatchId::new("snare"),
        PatchId::new("hihat"),
        PatchId::new("pluck"),
    );
    let budget_us = QUANTUM as f64 / SR as f64 * 1e6;

    // Three phases, each on a **fresh pool**. Sharing one would leak the
    // previous phase's voices into the next — and a `Bass` sustains until it is
    // stopped, so the first phase's bass would still be sounding through the
    // third and every number after the first would be the same number.
    let run = |label: &str, basses: u32, drums: u32, sfx: u32| -> f64 {
        let mut pool = VoicePool::new(PatchBank::builtin(), SR);
        let mut buf = vec![0.0f32; QUANTUM * 2];
        let mut voice = 0u32;
        let mut live: Vec<(usize, VoiceId)> = Vec::new(); // (release quantum, voice)

        let fire = |pool: &mut VoicePool,
                        voice: &mut u32,
                        patch,
                        count: u32,
                        gain: f32,
                        hold: Option<(usize, &mut Vec<(usize, VoiceId)>)>| {
            let mut held = hold;
            for i in 0..count {
                let id = VoiceId(*voice);
                pool.apply(Event::Play {
                    voice: id,
                    patch,
                    seed: (*voice as u64).wrapping_mul(0x9e37_79b9),
                    gain,
                    pan: (i as f32 / count.max(2) as f32) * 1.4 - 0.7,
                });
                if let Some((due, ref mut queue)) = held {
                    queue.push((due, id));
                }
                *voice = voice.wrapping_add(1);
            }
        };

        // Warm the caches without counting them.
        fire(&mut pool, &mut voice, pluck, 4, 0.3, None);
        for _ in 0..1_000 {
            pool.render_interleaved(&mut buf);
        }

        let start = std::time::Instant::now();
        let mut peak_active = 0usize;
        for block in 0..QUANTA {
            // The sequencer stops a bass note at its notated end; without that
            // the bass group fills up and stays full, which is a different
            // (and quietly heavier) instrument.
            live.retain(|(due, id)| {
                if *due <= block {
                    pool.apply(Event::Stop { voice: *id });
                    false
                } else {
                    true
                }
            });
            if block % EIGHTH_QUANTA == 0 {
                fire(&mut pool, &mut voice, hihat, drums, 0.4, None);
                fire(&mut pool, &mut voice, kick, drums, 0.9, None);
                fire(&mut pool, &mut voice, snare, drums, 0.7, None);
            }
            // A bass note every half-bar, held for a half-bar — the density
            // `bassline.tres` averages.
            if block % (EIGHTH_QUANTA * 4) == 0 {
                fire(
                    &mut pool,
                    &mut voice,
                    bass,
                    basses,
                    0.5,
                    Some((block + EIGHTH_QUANTA * 4, &mut live)),
                );
            }
            if sfx > 0 && block % (EIGHTH_QUANTA * 8) == 0 {
                fire(&mut pool, &mut voice, pluck, sfx, 0.3, None);
            }
            pool.render_interleaved(&mut buf);
            peak_active = peak_active.max(pool.active_voices());
        }
        let per_quantum_us = start.elapsed().as_secs_f64() * 1e6 / QUANTA as f64;
        println!(
            "{label:<28} {per_quantum_us:>7.2} us   {:>5.2} % of one core   peak {peak_active:>2} voices   peak pre-clip {:.2}",
            per_quantum_us / budget_us * 100.0,
            pool.stats().peak_pre_clip
        );
        per_quantum_us
    };

    println!("quanta           : {QUANTA} x {QUANTUM} frames stereo, {SR} Hz");
    println!("realtime budget  : {budget_us:.2} us per quantum");
    println!("slots            : {}", runt_audio::MAX_VOICES);
    println!();
    let idle = run("silence (an empty pool)", 0, 0, 0);
    // The song as written: the bassline is monophonic and the three drum
    // patterns never collide, so one voice per group is what it asks for.
    let song = run("the song as written", 1, 1, 0);
    // Every music group full at once. The song cannot produce this; a denser
    // one could.
    let music = run("every BGM group full", 3, 3, 0);
    let label = format!("...+ a {}-voice SFX burst", runt_audio::PLUCK_VOICES);
    let everything = run(&label, 3, 3, runt_audio::PLUCK_VOICES as u32);
    println!();
    println!("the song costs      {:.2} us over silence", song - idle);
    println!("the SFX burst adds  {:.2} us over the full BGM", everything - music);
    println!(
        "worst case is       {:.2} % of the {:.0} us budget",
        everything / budget_us * 100.0,
        budget_us
    );
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
