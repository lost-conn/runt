//! Native host for the audio spike.
//!
//!   cargo run -p spike-native --release -- hash      determinism check
//!   cargo run -p spike-native --release -- bench     CPU cost per quantum
//!   cargo run -p spike-native --release -- analyze   measured proof params work
//!   cargo run -p spike-native --release -- play      cpal realtime + xrun count
//!   cargo run -p spike-native --release -- wav out.wav
//!
//! `analyze` exists because the CI/agent box has no audio device. It measures
//! the rendered buffer instead of trusting ears: fundamental tracking, spectral
//! centroid vs. cutoff, and pluck onset energy.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{ErrorKind, OutputCallbackInfo, SampleFormat, StreamConfig};
use spike_patch::{canonical_render, hash_samples, render_offline, Patch, PatchParams};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("hash") => cmd_hash(),
        Some("bench") => cmd_bench(),
        Some("analyze") => cmd_analyze(),
        Some("play") => cmd_play(),
        Some("wav") => cmd_wav(args.get(1).map(String::as_str).unwrap_or("out.wav")),
        _ => eprintln!("usage: spike-native <hash|bench|analyze|play|wav [path]>"),
    }
}

// ---------------------------------------------------------------- determinism

fn cmd_hash() {
    // Two independent renders in one process. Fresh `Patch` each time, so this
    // covers construction order and any lazily-seeded internal state.
    let a = canonical_render();
    let b = canonical_render();
    let ha = hash_samples(&a);
    let hb = hash_samples(&b);

    println!("samples          : {}", a.len());
    println!("run A hash       : {ha:#018x}");
    println!("run B hash       : {hb:#018x}");
    println!("in-process match : {}", if ha == hb { "YES" } else { "NO" });

    // Different seed must give a different render, otherwise the seed is a lie
    // and "same params -> same output" is vacuous.
    let seeded = PatchParams {
        seed: 0x1234_5678,
        ..PatchParams::default()
    };
    let c = render_offline(seeded, 48_000.0, 48_000, 128, &[10, 100, 200, 300]);
    let hc = hash_samples(&c);
    println!("other-seed hash  : {hc:#018x}");
    println!("seed changes out : {}", if hc != ha { "YES" } else { "NO" });

    // Denormal probe: if a decayed tail were flushing to zero inconsistently we
    // would see it as run-to-run drift above. Report the count so FINDINGS can
    // say something concrete rather than hand-wave about FTZ.
    let subnormals = a.iter().filter(|s| s.is_subnormal()).count();
    println!("subnormal samples: {subnormals}");
    println!(
        "nan/inf samples  : {}",
        a.iter().filter(|s| !s.is_finite()).count()
    );

    // First samples verbatim, for diffing against the wasm build. Whether the
    // two platforms differ by 1 ULP or by something structural is the whole
    // question, and a hash cannot tell you.
    println!(
        "head             : {}",
        a[..16]
            .iter()
            .map(|s| format!("{s:.9}"))
            .collect::<Vec<_>>()
            .join(",")
    );

    // Printed last and alone so a second *process* invocation can be diffed.
    println!("CANONICAL {ha:#018x}");
}

// ----------------------------------------------------------------- benchmarks

fn cmd_bench() {
    const SR: f64 = 48_000.0;
    const QUANTUM: usize = 128;
    const QUANTA: usize = 20_000; // ~53 s of audio

    let mut patch = Patch::new(PatchParams::default(), SR);
    let mut buf = vec![0.0f32; QUANTUM * 2];

    // warm up (first blocks touch cold caches / lazy allocations)
    for _ in 0..1000 {
        patch.render_stereo(&mut buf);
    }

    let start = std::time::Instant::now();
    for i in 0..QUANTA {
        if i % 400 == 0 {
            patch.trigger();
        }
        patch.render_stereo(&mut buf);
    }
    let elapsed = start.elapsed();

    let per_quantum_us = elapsed.as_secs_f64() * 1e6 / QUANTA as f64;
    let audio_seconds = (QUANTA * QUANTUM) as f64 / SR;
    println!("quanta           : {QUANTA} x {QUANTUM} frames stereo");
    println!("wall time        : {:.3} s", elapsed.as_secs_f64());
    println!("per quantum      : {per_quantum_us:.2} us");
    println!(
        "realtime budget  : {:.2} us per quantum at {} Hz",
        QUANTUM as f64 / SR * 1e6,
        SR as u32
    );
    println!(
        "CPU load         : {:.3} % of one core",
        elapsed.as_secs_f64() / audio_seconds * 100.0
    );
}

// ------------------------------------------------------------------- analysis

/// Goertzel magnitude at `freq` over `x` (mono).
fn goertzel(x: &[f32], sr: f32, freq: f32) -> f32 {
    let k = 2.0 * std::f32::consts::PI * freq / sr;
    let coeff = 2.0 * k.cos();
    let (mut s1, mut s2) = (0.0f32, 0.0f32);
    for &v in x {
        let s0 = v + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    (s1 * s1 + s2 * s2 - coeff * s1 * s2).max(0.0).sqrt() / x.len() as f32
}

fn to_mono(stereo: &[f32]) -> Vec<f32> {
    stereo.chunks(2).map(|f| (f[0] + f[1]) * 0.5).collect()
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
}

/// Energy-weighted mean frequency over log-spaced probes. A blunt instrument,
/// but it moves monotonically with filter cutoff, which is all we need.
fn spectral_centroid(x: &[f32], sr: f32) -> f32 {
    let (mut num, mut den) = (0.0f32, 0.0f32);
    for i in 0..96 {
        let f = 30.0 * (8000.0f32 / 30.0).powf(i as f32 / 95.0);
        let m = goertzel(x, sr, f);
        num += f * m;
        den += m;
    }
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

/// Normalized-autocorrelation pitch detector. Returns the fundamental in Hz.
fn detect_pitch(x: &[f32], sr: f32, min_hz: f32, max_hz: f32) -> f32 {
    let min_lag = (sr / max_hz) as usize;
    let max_lag = ((sr / min_hz) as usize).min(x.len() / 2);
    let n = x.len() - max_lag;
    let energy: f32 = x[..n].iter().map(|v| v * v).sum();

    let mut best = (min_lag, f32::MIN);
    for lag in min_lag..max_lag {
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        for i in 0..n {
            num += x[i] * x[i + lag];
            den += x[i + lag] * x[i + lag];
        }
        let norm = num / (energy * den).max(1e-20).sqrt();
        if norm > best.1 {
            best = (lag, norm);
        }
    }
    sr / best.0 as f32
}

fn cmd_analyze() {
    const SR: f32 = 48_000.0;
    let quiet: &[usize] = &[];

    println!("== drone fundamental tracks `drone_hz` ==");
    for hz in [55.0f32, 82.5, 110.0] {
        let p = PatchParams {
            drone_hz: hz,
            pluck_gain: 0.0,
            lfo_depth: 0.0,
            ..PatchParams::default()
        };
        let buf = to_mono(&render_offline(p, SR as f64, 24_000, 128, quiet));
        let tail = &buf[8_000..]; // skip filter settling
        // The patch stacks a sub-oscillator at drone_hz/2 under the two saws,
        // so the waveform's true period is 2/drone_hz. Autocorrelation finds
        // that period directly; peak-picking a spectrum does not, because the
        // resonant lowpass can make a harmonic louder than the fundamental.
        let expected = hz * 0.5;
        let detected = detect_pitch(tail, SR, 15.0, 400.0);
        println!(
            "  drone_hz={hz:6.1}  expected f0={expected:6.2} Hz  detected f0={detected:6.2} Hz  err={:+.2} %",
            (detected - expected) / expected * 100.0
        );
    }

    println!("== cutoff moves spectral centroid ==");
    for cutoff in [200.0f32, 700.0, 2500.0, 6000.0] {
        let p = PatchParams {
            cutoff_hz: cutoff,
            pluck_gain: 0.0,
            lfo_depth: 0.0, // hold the sweep still so the number is clean
            ..PatchParams::default()
        };
        let buf = to_mono(&render_offline(p, SR as f64, 24_000, 128, quiet));
        let tail = &buf[8_000..];
        println!(
            "  cutoff={cutoff:7.1} Hz  centroid={:7.1} Hz  rms={:.4}",
            spectral_centroid(tail, SR),
            rms(tail)
        );
    }

    println!("== live cutoff change takes effect mid-stream ==");
    {
        let mut patch = Patch::new(
            PatchParams {
                lfo_depth: 0.0,
                pluck_gain: 0.0,
                ..PatchParams::default()
            },
            SR as f64,
        );
        let mut before = vec![0.0f32; 12_000 * 2];
        for c in before.chunks_mut(256) {
            patch.render_stereo(c);
        }
        patch.set_cutoff_hz(5000.0);
        let mut after = vec![0.0f32; 12_000 * 2];
        for c in after.chunks_mut(256) {
            patch.render_stereo(c);
        }
        let b = to_mono(&before);
        let a = to_mono(&after);
        println!(
            "  centroid before set_cutoff_hz(5000) = {:7.1} Hz",
            spectral_centroid(&b[4000..], SR)
        );
        println!(
            "  centroid after  set_cutoff_hz(5000) = {:7.1} Hz",
            spectral_centroid(&a[4000..], SR)
        );
    }

    println!("== trigger() produces a pluck transient ==");
    {
        let p = PatchParams {
            drone_gain: 0.0, // isolate the pluck
            ..PatchParams::default()
        };
        let buf = to_mono(&render_offline(p, SR as f64, 24_000, 128, &[40]));
        // block 40 * 128 = frame 5120
        let pre = rms(&buf[3_000..5_000]);
        let post = rms(&buf[5_200..7_200]);
        let decayed = rms(&buf[20_000..22_000]);
        println!("  rms before trigger = {pre:.6}");
        println!(
            "  rms after  trigger = {post:.6}  ({:.0}x)",
            post / pre.max(1e-9)
        );
        println!(
            "  rms 310 ms later   = {decayed:.6}  (decaying: {})",
            if decayed < post { "YES" } else { "NO" }
        );
    }
}

// ---------------------------------------------------------------------- audio

enum Cmd {
    Cutoff(f32),
    Trigger,
    PluckHz(f32),
}

fn cmd_play() {
    let host = cpal::default_host();
    let device = match host.default_output_device() {
        Some(d) => d,
        None => {
            eprintln!("no default output device (headless box?) — nothing to play");
            std::process::exit(2);
        }
    };
    let supported = match device.default_output_config() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("default_output_config failed: {e}");
            std::process::exit(2);
        }
    };
    println!("device: {:?}", device.id());
    println!("config: {supported:?}");
    if supported.sample_format() != SampleFormat::F32 {
        eprintln!(
            "spike only wires up the f32 path; device wants {:?}",
            supported.sample_format()
        );
    }

    let config: StreamConfig = supported.into();
    let sample_rate = config.sample_rate as f64;
    let channels = config.channels as usize;

    let (tx, rx) = mpsc::channel::<Cmd>();
    let xruns = Arc::new(AtomicUsize::new(0));
    let callbacks = Arc::new(AtomicU32::new(0));
    let xruns_cb = xruns.clone();
    let callbacks_cb = callbacks.clone();

    let mut patch = Patch::new(PatchParams::default(), sample_rate);
    let mut scratch: Vec<f32> = Vec::new();

    let stream = device
        .build_output_stream(
            // cpal 0.18 takes the config *by value* (0.15 took `&config`).
            config,
            move |data: &mut [f32], _: &OutputCallbackInfo| {
                callbacks_cb.fetch_add(1, Ordering::Relaxed);
                // Drain the control queue at block granularity — exactly what
                // the worklet does with postMessage, and what runt's sim would
                // do at a tick boundary.
                while let Ok(cmd) = rx.try_recv() {
                    match cmd {
                        Cmd::Cutoff(hz) => patch.set_cutoff_hz(hz),
                        Cmd::PluckHz(hz) => patch.set_pluck_hz(hz),
                        Cmd::Trigger => patch.trigger(),
                    }
                }
                let frames = data.len() / channels;
                scratch.resize(frames * 2, 0.0);
                patch.render_stereo(&mut scratch);
                for (i, frame) in data.chunks_mut(channels).enumerate() {
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
    println!("playing 12 s: cutoff sweep + a pluck every 750 ms");

    let notes = [440.0f32, 554.37, 659.25, 880.0];
    for step in 0..48 {
        let t = step as f32 / 48.0;
        let cutoff = 300.0 * (12.0f32).powf(t * 2.0 % 1.0);
        let _ = tx.send(Cmd::Cutoff(cutoff));
        if step % 3 == 0 {
            let _ = tx.send(Cmd::PluckHz(notes[(step / 3) % notes.len()]));
            let _ = tx.send(Cmd::Trigger);
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }

    println!("callbacks: {}", callbacks.load(Ordering::Relaxed));
    println!("xruns    : {}", xruns.load(Ordering::Relaxed));
}

// ------------------------------------------------------------------ wav dump

fn cmd_wav(path: &str) {
    let sr = 48_000u32;
    let data = render_offline(
        PatchParams::default(),
        sr as f64,
        sr as usize * 6,
        128,
        &[100, 300, 500, 700, 900, 1100],
    );
    let bytes = wav16(&data, sr, 2);
    std::fs::write(path, bytes).expect("write wav");
    println!(
        "wrote {path} ({} frames, hash {:#018x})",
        data.len() / 2,
        hash_samples(&data)
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
