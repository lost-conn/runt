//! What the six patches actually *sound* like, measured rather than heard.
//!
//! Nobody can listen on a CI box, so this is the spike's `analyze` command
//! turned into assertions: does the pluck land on the pitch the params asked
//! for, does the filter cutoff move the spectrum, does the envelope decay, does
//! the drone hold. Run with `--nocapture` to see the numbers — they are quoted
//! in the phase-3 report and are the thing to eyeball after a DSP change.
//!
//! The measurements are blunt by design (see [`runt_audio::analyze`]): what is
//! asserted is direction and rough magnitude, never a float to four places,
//! because the latter would be a determinism test wearing a disguise.

// Everything below drives the synthesizer, which only exists behind `dsp`.
// A default-feature build of this crate is the *description* half — the patch
// bank and the wire codec — and has nothing here to test.
#![cfg(feature = "dsp")]

use runt_audio::analyze;
use runt_audio::voice::render_offline;
use runt_audio::wire::{Event, VoiceId};
use runt_audio::{
    BassParams, DroneParams, HihatParams, KickParams, PatchBank, PatchDef, PatchId, PluckParams,
    SnareParams, VoicePool, REFERENCE_SAMPLE_RATE,
};

const SR: f32 = REFERENCE_SAMPLE_RATE as f32;
const ONE: PatchId = PatchId::new("one");

fn bank_of(def: PatchDef) -> PatchBank {
    PatchBank::new().with("one", def)
}

fn strike(def: PatchDef, seed: u64, frames: usize) -> Vec<f32> {
    render_offline(
        &bank_of(def),
        &[(
            0,
            Event::Play {
                voice: VoiceId(0),
                patch: ONE,
                seed,
                gain: 1.0,
                pan: 0.0,
            },
        )],
        frames,
        128,
        SR,
    )
}

// ---------------------------------------------------------------------------
// Pluck
// ---------------------------------------------------------------------------

/// A pluck pinned to one note (`steps: [0]`, no jitter) so the measurement has
/// a single right answer.
fn fixed_pluck(base_hz: f32) -> PluckParams {
    PluckParams {
        base_hz,
        steps: vec![0],
        jitter_semitones: 0.0,
        // A long decay so there is a steady-ish stretch to analyse; the default
        // 0.35 s is mostly transient.
        decay_s: 3.0,
        cutoff_env: 1.0,
        cutoff_hz: 4000.0,
        ..PluckParams::default()
    }
}

#[test]
fn the_pluck_lands_on_the_pitch_its_params_ask_for() {
    println!("== pluck fundamental tracks base_hz ==");
    for hz in [110.0f32, 220.0, 440.0, 880.0] {
        let buf = strike(PatchDef::Pluck(fixed_pluck(hz)), 0, 24_000);
        let mono = analyze::to_mono(&buf);
        // Skip the attack and the brightest part of the decay.
        let detected = analyze::detect_pitch(&mono[4_000..16_000], SR, 50.0, 2_000.0);
        let error = (detected - hz) / hz * 100.0;
        println!("  base_hz={hz:7.1}  detected={detected:7.2} Hz  err={error:+.2} %");
        assert!(
            error.abs() < 2.0,
            "expected ~{hz} Hz, autocorrelation found {detected} Hz"
        );
    }
}

#[test]
fn the_seed_walks_the_pluck_up_its_scale() {
    // The engine sends a seed, not a note (see `PluckParams`). What has to hold
    // is that the seed reaches the pitch: several seeds must produce several
    // *distinct* pitches, all of them members of the authored scale.
    let params = PluckParams {
        base_hz: 220.0,
        steps: vec![0, 3, 5, 7, 10, 12],
        jitter_semitones: 0.0,
        decay_s: 3.0,
        cutoff_env: 1.0,
        cutoff_hz: 4000.0,
        ..PluckParams::default()
    };
    let allowed: Vec<f32> = params
        .steps
        .iter()
        .map(|s| params.base_hz * runt_audio::params::semitone_ratio(*s as f32))
        .collect();

    println!("== seed selects a scale degree ==");
    let mut seen = std::collections::BTreeSet::new();
    for seed in 0..12u64 {
        let buf = strike(PatchDef::Pluck(params.clone()), seed, 24_000);
        let mono = analyze::to_mono(&buf);
        let detected = analyze::detect_pitch(&mono[4_000..16_000], SR, 100.0, 1_200.0);
        let (nearest, error) = allowed
            .iter()
            .map(|f| (*f, (detected - f).abs() / f))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .expect("the scale is not empty");
        println!("  seed={seed:2}  detected={detected:7.2} Hz  nearest={nearest:7.2} Hz  err={:+.2} %", error * 100.0);
        assert!(
            error < 0.02,
            "seed {seed} produced {detected} Hz, which is not on the scale"
        );
        seen.insert(nearest.to_bits());
    }
    assert!(
        seen.len() >= 3,
        "twelve seeds landed on only {} distinct notes; the seed is barely doing anything",
        seen.len()
    );
}

#[test]
fn the_cutoff_moves_the_spectrum_monotonically() {
    println!("== pluck cutoff opens the spectrum ==");
    let mut previous = (0.0f32, 0.0f32);
    let mut first = 0.0f32;
    for cutoff in [300.0f32, 900.0, 2500.0, 6000.0] {
        let params = PluckParams {
            cutoff_hz: cutoff,
            cutoff_env: 1.0, // hold the sweep still so the number is clean
            ..fixed_pluck(220.0)
        };
        let buf = strike(PatchDef::Pluck(params), 0, 24_000);
        let mono = analyze::to_mono(&buf);
        let window = &mono[4_000..16_000];
        let centroid = analyze::spectral_centroid(window, SR);
        let brightness = analyze::brightness(window, SR);
        println!(
            "  cutoff={cutoff:7.1} Hz  centroid={centroid:7.1} Hz  high/low={brightness:.4}  rms={:.4}",
            analyze::rms(window)
        );
        assert!(
            centroid > previous.0,
            "raising the cutoff must raise the centroid ({centroid} did not exceed {})",
            previous.0
        );
        assert!(
            brightness > previous.1,
            "and must let more high band through ({brightness} vs {})",
            previous.1
        );
        if first == 0.0 {
            first = brightness;
        }
        previous = (centroid, brightness);
    }
    // Each step gains less than the last — a 220 Hz saw simply runs out of
    // harmonics to let through — so the per-step claim is only "more", and the
    // claim with teeth is the span across the whole sweep.
    assert!(
        previous.1 > first * 10.0,
        "the sweep must move the spectrum by an order of magnitude ({first} → {})",
        previous.1
    );
}

#[test]
fn the_envelope_strikes_and_decays() {
    println!("== pluck envelope ==");
    let buf = strike(PatchDef::Pluck(PluckParams::default()), 0, 48_000);
    let mono = analyze::to_mono(&buf);
    let onset = analyze::rms(&mono[100..2_000]);
    let mid = analyze::rms(&mono[8_000..10_000]);
    let late = analyze::rms(&mono[30_000..32_000]);
    println!("  rms  0–40 ms  = {onset:.6}");
    println!("  rms  170 ms   = {mid:.6}");
    println!("  rms  630 ms   = {late:.6}");
    assert!(onset > 0.01, "a pluck must actually sound: {onset}");
    assert!(mid < onset, "and decay");
    assert!(late < mid * 0.2, "and keep decaying");
}

#[test]
fn the_cutoff_envelope_makes_the_onset_brighter_than_the_tail() {
    // The property that distinguishes a pluck from a saw through a static
    // filter, and the reason `cutoff_env` exists.
    let params = PluckParams {
        base_hz: 220.0,
        steps: vec![0],
        jitter_semitones: 0.0,
        decay_s: 1.2,
        cutoff_hz: 400.0,
        cutoff_env: 8.0,
        ..PluckParams::default()
    };
    let buf = strike(PatchDef::Pluck(params), 0, 48_000);
    let mono = analyze::to_mono(&buf);
    let onset = analyze::spectral_centroid(&mono[200..4_000], SR);
    let tail = analyze::spectral_centroid(&mono[20_000..24_000], SR);
    println!("== cutoff envelope ==\n  centroid onset={onset:.1} Hz  tail={tail:.1} Hz");
    assert!(
        onset > tail * 1.3,
        "the strike must be brighter than the tail ({onset} vs {tail})"
    );
}

#[test]
fn the_attack_starts_from_silence() {
    // A zero-length attack is a step discontinuity, which is what a click is.
    // The first few samples must be small.
    let buf = strike(PatchDef::Pluck(PluckParams::default()), 0, 4_800);
    assert!(
        buf[0].abs() < 1e-3 && buf[1].abs() < 1e-3,
        "a note must not begin at full level: {} {}",
        buf[0],
        buf[1]
    );
}

// ---------------------------------------------------------------------------
// Pluck — the noise half (E7)
// ---------------------------------------------------------------------------

/// The gate the whole feature was designed around: a preset that does not ask
/// for noise must render the *same bytes* it rendered before the noise existed.
///
/// This is not "sounds the same". `Pluck` grew a second fundsp graph, and had it
/// gone into the first one, `AudioUnit::ping` would have threaded a different
/// hash into the two saws and `WaveSynth::reset` would have started them at a
/// different phase — every pluck in the crate quietly different, for a field
/// nobody set. The check that it did not is a byte comparison against a render
/// taken with the noise fields absent, which is what the two halves below are.
#[test]
fn a_preset_without_noise_renders_exactly_what_it_rendered_before_noise_existed() {
    // The 0.0 defaults, spelled out rather than inherited, so that a future
    // change to `Default` cannot quietly turn this test into a tautology.
    let silent = PluckParams {
        noise_mix: 0.0,
        noise_decay_s: 0.0,
        noise_highpass_hz: 0.0,
        ..PluckParams::default()
    };
    assert_eq!(silent, PluckParams::default(), "the default is noiseless");

    // Every knob a preset in the port actually moves, so this covers the paths
    // (detune off, filter wide open, filter shut, long tail) and not just one.
    let shapes = [
        PluckParams::default(),
        fixed_pluck(220.0),
        PluckParams {
            base_hz: 46.25,
            steps: vec![0],
            detune: 1.012,
            detune_gain: 0.8,
            cutoff_hz: 150.0,
            cutoff_env: 2.4,
            resonance: 0.8,
            ..PluckParams::default()
        },
        PluckParams {
            base_hz: 880.0,
            detune_gain: 0.0,
            cutoff_hz: 9000.0,
            cutoff_env: 1.0,
            resonance: 6.0,
            decay_s: 0.05,
            ..PluckParams::default()
        },
    ];
    for (i, shape) in shapes.iter().enumerate() {
        for seed in [0u64, 1, 7, 0xdead_beef] {
            let a = strike(PatchDef::Pluck(shape.clone()), seed, 24_000);
            // The same params reached through the noise fields' defaults rather
            // than through `Default` — the bytes must be identical, not close.
            let b = strike(
                PatchDef::Pluck(PluckParams {
                    noise_mix: 0.0,
                    noise_decay_s: 0.0,
                    noise_highpass_hz: 0.0,
                    ..shape.clone()
                }),
                seed,
                24_000,
            );
            assert_eq!(a, b, "shape {i}, seed {seed}: the noise path is not inert");
            assert!(analyze::peak(&a) > 0.01, "shape {i} must sound at all");
        }
    }
}

#[test]
fn noise_mix_puts_noise_in_the_signal_and_takes_tone_out() {
    // Three claims at once, because they are one claim: the crossfade is a
    // crossfade. More noise → less pitch, more broadband energy, and at 1.0 the
    // oscillators are gone rather than quiet.
    let base = PluckParams {
        base_hz: 220.0,
        steps: vec![0],
        jitter_semitones: 0.0,
        decay_s: 0.6,
        cutoff_hz: 6000.0,
        cutoff_env: 1.0,
        resonance: 0.7,
        ..PluckParams::default()
    };
    println!("== noise_mix crossfade ==");
    let mut tonality = Vec::new();
    for mix in [0.0f32, 0.35, 0.7, 1.0] {
        let buf = strike(
            PatchDef::Pluck(PluckParams {
                noise_mix: mix,
                ..base.clone()
            }),
            0,
            24_000,
        );
        let mono = analyze::to_mono(&buf);
        // Goertzel at the fundamental against the total: a pure saw parks a lot
        // of magnitude in one bin, white noise parks almost none in any.
        let fundamental = analyze::goertzel(&mono[2_000..14_000], SR, 220.0);
        let total = analyze::rms(&mono[2_000..14_000]).max(1e-9);
        let ratio = fundamental / total;
        println!("  mix={mix:.2}  220 Hz/rms={ratio:.4}  peak={:.4}", analyze::peak(&buf));
        assert!(analyze::peak(&buf) > 0.01, "mix={mix} must make a sound");
        tonality.push(ratio);
    }
    for pair in tonality.windows(2) {
        assert!(
            pair[1] < pair[0],
            "more noise must mean less of the note: {pair:?}"
        );
    }
    assert!(
        tonality[3] < tonality[0] * 0.25,
        "at mix 1.0 the oscillators are disconnected, not attenuated: {tonality:?}"
    );
}

#[test]
fn the_noise_decay_is_independent_of_the_note_decay() {
    // The field's whole reason to exist: Godot's noisy patches are a *burst*
    // over a body, and the burst is shorter. A long note with a short
    // `noise_decay_s` must be noisy at the front and tonal at the back.
    let params = PluckParams {
        base_hz: 220.0,
        steps: vec![0],
        jitter_semitones: 0.0,
        attack_s: 0.002,
        decay_s: 1.5,
        cutoff_hz: 6000.0,
        cutoff_env: 1.0,
        noise_mix: 0.75,
        ..PluckParams::default()
    };
    // Three renders that differ *only* in `noise_decay_s`, so the amplitude
    // envelope — which is decaying under all of this — and the `1 - mix` share
    // of the tone cancel out of every comparison. The reference is not "no
    // noise": it is the same preset with a burst one millisecond long, i.e. the
    // same quarter-level note with the noise already over.
    let render = |noise_decay_s: f32| {
        let buf = strike(
            PatchDef::Pluck(PluckParams {
                noise_decay_s,
                ..params.clone()
            }),
            0,
            48_000,
        );
        let mono = analyze::to_mono(&buf);
        (
            analyze::rms(&mono[200..2_000]),
            analyze::rms(&mono[20_000..30_000]),
        )
    };
    let (gone_early, gone_late) = render(0.001); // the note alone, at 1 - mix
    let (burst_early, burst_late) = render(0.3); // a 300 ms burst under a 1.5 s note
    let (held_early, held_late) = render(0.0); // Godot's "follow the amp envelope"

    println!("== noise burst under a long note ==  rms at 4–40 ms / 420–630 ms");
    println!("  noise_decay_s = 0.001   {gone_early:.5}  {gone_late:.5}");
    println!("  noise_decay_s = 0.3     {burst_early:.5}  {burst_late:.5}");
    println!("  noise_decay_s = 0       {held_early:.5}  {held_late:.5}");

    assert!(
        burst_early > gone_early * 1.5,
        "the strike must carry the burst ({burst_early} vs {gone_early})"
    );
    assert!(
        (burst_late - gone_late).abs() < gone_late * 0.05,
        "…and by 420 ms the burst is over and only the note is left \
         ({burst_late} vs {gone_late})"
    );
    // …and it is `noise_decay_s` that ended it: at Godot's `0.0` the burst
    // follows the amplitude envelope instead and is still going.
    assert!(
        held_late > gone_late * 1.5,
        "noise_decay_s = 0 must hold the burst open ({held_late} vs {gone_late})"
    );
}

#[test]
fn the_noise_highpass_takes_the_bottom_out_of_the_noise_only() {
    let base = PluckParams {
        base_hz: 110.0,
        steps: vec![0],
        jitter_semitones: 0.0,
        decay_s: 0.5,
        cutoff_hz: 8000.0,
        cutoff_env: 1.0,
        noise_mix: 1.0, // noise only, so the measurement is about the noise
        ..PluckParams::default()
    };
    println!("== noise highpass ==");
    let mut previous = f32::INFINITY;
    for hp in [0.0f32, 600.0, 3000.0] {
        let buf = strike(
            PatchDef::Pluck(PluckParams {
                noise_highpass_hz: hp,
                ..base.clone()
            }),
            0,
            24_000,
        );
        let mono = analyze::to_mono(&buf);
        let low = analyze::band_energy(&mono[1_000..16_000], SR, 30.0, 400.0);
        println!("  highpass={hp:6.0} Hz  energy below 400 Hz = {low:.5}");
        assert!(low < previous, "a higher highpass must leave less bottom");
        previous = low;
    }
}

#[test]
fn a_noisy_pluck_is_deterministic_and_clean() {
    // Same seed → the same noise, twice; different seed → different noise; and
    // nothing subnormal or infinite anywhere, which is the property `SILENCE`
    // exists to protect and which a second, independently-cut envelope could
    // have broken.
    let params = PluckParams {
        noise_mix: 0.85,
        noise_decay_s: 0.14,
        noise_highpass_hz: 600.0,
        decay_s: 2.0, // far longer than the burst: the cut has room to happen
        ..PluckParams::default()
    };
    let a = strike(PatchDef::Pluck(params.clone()), 42, 48_000);
    let b = strike(PatchDef::Pluck(params.clone()), 42, 48_000);
    let c = strike(PatchDef::Pluck(params), 43, 48_000);
    assert_eq!(a, b, "same params + seed must give the same noise");
    assert_ne!(a, c, "a different seed must give different noise");
    assert!(analyze::peak(&a) > 0.05, "and it must actually sound");
    assert_eq!(
        analyze::anomalies(&a),
        (0, 0),
        "no subnormals, no non-finite samples"
    );
}

// ---------------------------------------------------------------------------
// Drone
// ---------------------------------------------------------------------------

#[test]
fn the_drone_holds_its_fundamental() {
    println!("== drone fundamental tracks base_hz ==");
    for hz in [55.0f32, 82.5, 110.0] {
        let params = DroneParams {
            base_hz: hz,
            sub_gain: 0.0, // the sub is an octave down and would halve the period
            lfo_depth: 0.0,
            jitter_semitones: 0.0,
            attack_s: 0.05,
            cutoff_hz: 3000.0,
            ..DroneParams::default()
        };
        let buf = strike(PatchDef::Drone(params), 0, 48_000);
        let mono = analyze::to_mono(&buf);
        let detected = analyze::detect_pitch(&mono[12_000..36_000], SR, 20.0, 400.0);
        let error = (detected - hz) / hz * 100.0;
        println!("  base_hz={hz:6.1}  detected={detected:6.2} Hz  err={error:+.2} %");
        assert!(error.abs() < 2.0);
    }
}

#[test]
fn the_sub_oscillator_halves_the_period() {
    // Not a bug and worth pinning: with the sub audible the waveform repeats at
    // `base_hz / 2`, exactly as the spike recorded for its own drone.
    let params = DroneParams {
        base_hz: 110.0,
        sub_gain: 1.0,
        lfo_depth: 0.0,
        jitter_semitones: 0.0,
        attack_s: 0.05,
        cutoff_hz: 3000.0,
        ..DroneParams::default()
    };
    let buf = strike(PatchDef::Drone(params), 0, 48_000);
    let mono = analyze::to_mono(&buf);
    let detected = analyze::detect_pitch(&mono[12_000..36_000], SR, 20.0, 400.0);
    println!("== sub oscillator ==\n  base_hz=110 with sub → detected {detected:.2} Hz");
    assert!((detected - 55.0).abs() / 55.0 < 0.02);
}

#[test]
fn the_drone_lfo_moves_the_filter_over_time() {
    let params = DroneParams {
        base_hz: 55.0,
        lfo_hz: 1.0, // fast enough to see inside a two-second render
        lfo_depth: 0.9,
        cutoff_hz: 800.0,
        attack_s: 0.05,
        jitter_semitones: 0.0,
        ..DroneParams::default()
    };
    let buf = strike(PatchDef::Drone(params), 0, 96_000);
    let mono = analyze::to_mono(&buf);
    // A quarter and three quarters through one LFO cycle: opposite extremes.
    let a = analyze::spectral_centroid(&mono[12_000..18_000], SR);
    let b = analyze::spectral_centroid(&mono[36_000..42_000], SR);
    println!("== drone LFO ==\n  centroid at LFO peak={a:.1} Hz  at LFO trough={b:.1} Hz");
    assert!(
        (a - b).abs() / a.max(b) > 0.15,
        "the sweep must be audible in the spectrum ({a} vs {b})"
    );
}

#[test]
fn the_drone_fades_in_rather_than_arriving() {
    let params = DroneParams {
        attack_s: 1.0,
        ..DroneParams::default()
    };
    let buf = strike(PatchDef::Drone(params), 0, 96_000);
    let mono = analyze::to_mono(&buf);
    let early = analyze::rms(&mono[0..2_000]);
    let late = analyze::rms(&mono[40_000..44_000]);
    println!("== drone attack ==\n  rms early={early:.6}  late={late:.6}");
    assert!(early < late * 0.25, "a drone that arrives is a sound effect");
}

// ---------------------------------------------------------------------------
// A running voice, re-aimed
// ---------------------------------------------------------------------------

#[test]
fn a_live_cutoff_change_takes_effect_mid_stream() {
    // The `Shared`-per-sample path, end to end: no graph is rebuilt and the
    // spectrum still moves. Mirrors the spike's "live cutoff change" check.
    let params = DroneParams {
        lfo_depth: 0.0,
        cutoff_hz: 300.0,
        attack_s: 0.05,
        jitter_semitones: 0.0,
        ..DroneParams::default()
    };
    let mut pool = VoicePool::new(bank_of(PatchDef::Drone(params)), SR);
    pool.apply(Event::Play {
        voice: VoiceId(0),
        patch: ONE,
        seed: 0,
        gain: 1.0,
        pan: 0.0,
    });

    let mut before = vec![0.0f32; 24_000 * 2];
    pool.render_interleaved(&mut before);
    pool.apply(Event::SetParam {
        voice: VoiceId(0),
        id: runt_audio::ParamId::CUTOFF,
        value: 8.0,
    });
    let mut after = vec![0.0f32; 24_000 * 2];
    pool.render_interleaved(&mut after);

    let (before, after) = (analyze::to_mono(&before), analyze::to_mono(&after));
    let b = analyze::brightness(&before[8_000..], SR);
    let a = analyze::brightness(&after[8_000..], SR);
    println!(
        "== live SetParam(CUTOFF) ==\n  centroid {:.1} → {:.1} Hz   high/low {b:.4} → {a:.4}",
        analyze::spectral_centroid(&before[8_000..], SR),
        analyze::spectral_centroid(&after[8_000..], SR),
    );
    assert!(a > b * 4.0, "the edit must be audible ({b} → {a})");
}

// ---------------------------------------------------------------------------
// Kick, Snare, Hihat, Bass — the BGM models
// ---------------------------------------------------------------------------
//
// These four came from `addons/godot_synth`'s patches (see each `*Params` doc)
// and each has one property that *is* the instrument. A kick without a pitch
// drop is a low sine; a snare without noise is a tom; a hi-hat that is not
// bright is a shaker; a bass that does not sustain is a pluck. So that is what
// each test below measures — the defining property, in the direction that makes
// it defining, never a float to four places.

/// Mono samples in `[from_s, to_s)` of a rendered stereo buffer.
fn window(buf: &[f32], from_s: f32, to_s: f32) -> Vec<f32> {
    let frame = |s: f32| ((s * SR) as usize * 2).min(buf.len());
    analyze::to_mono(&buf[frame(from_s)..frame(to_s)])
}

#[test]
fn the_kick_starts_high_and_lands_on_its_note() {
    // `pitch_drop_semitones` is the whole instrument. 36 semitones is three
    // octaves, so the first few milliseconds sit at 8x the destination and the
    // tail sits on it. The click is off: this measurement is about the body, and
    // a noise burst is exactly what a pitch tracker cannot read.
    let params = KickParams {
        base_hz: 55.0,
        pitch_drop_semitones: 36.0,
        pitch_drop_s: 0.08,
        triangle_gain: 0.0,
        click_gain: 0.0,
        decay_s: 0.6,
        ..KickParams::default()
    };
    let buf = strike(PatchDef::Kick(params), 1, 24_000);

    // 5–15 ms: the sweep has fallen a little from 440 Hz but is still up there.
    let early = analyze::detect_pitch(&window(&buf, 0.005, 0.015), SR, 40.0, 1200.0);
    // 200–400 ms: long past the 80 ms sweep, so this is the note itself.
    let late = analyze::detect_pitch(&window(&buf, 0.20, 0.40), SR, 30.0, 1200.0);

    println!("kick: early {early:.1} Hz, late {late:.1} Hz");
    assert!(
        early > late * 4.0,
        "the drop must be audible as a drop (early {early:.1}, late {late:.1})"
    );
    assert!(
        (late - 55.0).abs() < 55.0 * 0.08,
        "and it must land on base_hz, not near it (got {late:.1})"
    );
}

#[test]
fn the_kicks_click_is_the_only_thing_it_has_above_a_kilohertz() {
    // The click is this crate's addition, not Godot's (`kick.tres` is
    // `noise_mix = 0`), and it exists so a 55 Hz drum survives a laptop speaker.
    // What it must not do is change the body — so measure only the top end.
    let quiet = KickParams {
        click_gain: 0.0,
        triangle_gain: 0.0,
        ..KickParams::default()
    };
    let loud = KickParams {
        click_gain: 1.0,
        ..quiet.clone()
    };

    let without = strike(PatchDef::Kick(quiet), 3, 12_000);
    let with = strike(PatchDef::Kick(loud), 3, 12_000);

    // The measurement window is the *click's* window. `click_decay_s` is 4 ms by
    // default and the body rings for 280, so averaging over 30 ms would dilute a
    // real difference into a 1.5x one — which is what the first version of this
    // test measured before the window was tightened to match the thing measured.
    let top = |buf: &[f32]| analyze::band_energy(&window(buf, 0.0, 0.006), SR, 1500.0, 9000.0);
    let (a, b) = (top(&without), top(&with));
    println!("kick click (first 6 ms): {a:.6} -> {b:.6}");
    assert!(b > a * 3.0, "the click must add real high end ({a:.6} -> {b:.6})");

    // …and it is over almost immediately: nothing of it survives to 200 ms.
    let tail_a = analyze::band_energy(&window(&without, 0.2, 0.3), SR, 1500.0, 9000.0);
    let tail_b = analyze::band_energy(&window(&with, 0.2, 0.3), SR, 1500.0, 9000.0);
    assert!(
        tail_b < tail_a * 2.0 + 1e-6,
        "the click must not still be ringing at 200 ms ({tail_a:.8} vs {tail_b:.8})"
    );
}

#[test]
fn the_snare_is_a_tuned_body_when_the_mix_says_body_and_noise_when_it_says_noise() {
    // `noise_mix` is Godot's crossfade (`synth_engine.gd:744`) and it is the
    // parameter that decides whether this patch is a snare or a tom. Both ends
    // of it are measurable and they are measurable in *different* ways, which is
    // the point: a body has a pitch, and noise has a spectrum.
    let base = SnareParams {
        base_hz: 200.0,
        pitch_drop_semitones: 0.0, // a fixed pitch, so the tracker has one answer
        decay_s: 0.5,
        cutoff_hz: 8000.0,
        ..SnareParams::default()
    };

    let body_only = strike(
        PatchDef::Snare(SnareParams {
            noise_mix: 0.0,
            ..base.clone()
        }),
        5,
        24_000,
    );
    let pitch = analyze::detect_pitch(&window(&body_only, 0.02, 0.30), SR, 80.0, 900.0);
    println!("snare body pitch: {pitch:.1} Hz");
    assert!(
        (pitch - 200.0).abs() < 200.0 * 0.05,
        "the body is four partials on base_hz (got {pitch:.1})"
    );

    // The noise half: white noise is flat, four partials of a 200 Hz tone are
    // not, so the high/low ratio has to climb monotonically with the mix.
    let mut previous = 0.0f32;
    for mix in [0.0f32, 0.35, 0.7, 1.0] {
        let buf = strike(
            PatchDef::Snare(SnareParams {
                noise_mix: mix,
                ..base.clone()
            }),
            5,
            12_000,
        );
        let brightness = analyze::brightness(&window(&buf, 0.0, 0.15), SR);
        println!("snare noise_mix {mix:.2}: brightness {brightness:.4}");
        assert!(
            brightness > previous,
            "more noise must mean more top end ({mix}: {brightness:.4} vs {previous:.4})"
        );
        previous = brightness;
    }
}

#[test]
fn the_hihat_is_the_brightest_thing_in_the_kit_and_the_shortest() {
    // Two claims, and the ordering one is the interesting half: "bright" is only
    // meaningful next to the two drums it shares a bar with.
    let kick = strike(PatchDef::Kick(KickParams::default()), 9, 24_000);
    let snare = strike(PatchDef::Snare(SnareParams::default()), 9, 24_000);
    let hihat = strike(PatchDef::Hihat(HihatParams::default()), 9, 24_000);

    let centroid = |buf: &[f32]| analyze::spectral_centroid(&window(buf, 0.0, 0.06), SR);
    let (k, s, h) = (centroid(&kick), centroid(&snare), centroid(&hihat));
    println!("centroid: kick {k:.0} Hz, snare {s:.0} Hz, hihat {h:.0} Hz");
    assert!(k < s, "a kick is darker than a snare ({k:.0} < {s:.0})");
    assert!(s < h, "and a snare is darker than a hat ({s:.0} < {h:.0})");
    assert!(h > 3000.0, "a closed hat lives in the top octaves (got {h:.0})");

    // Short: the default `decay_s` is 60 ms, so nothing survives to 300 ms.
    let head = analyze::rms(&window(&hihat, 0.0, 0.03));
    let tail = analyze::rms(&window(&hihat, 0.3, 0.4));
    println!("hihat rms: {head:.5} -> {tail:.8}");
    assert!(head > 0.01, "a hat has to be audible at all (got {head:.5})");
    assert!(tail < head * 1e-3, "and gone by 300 ms ({tail:.8})");
}

#[test]
fn the_hihat_has_no_pitch_and_says_so() {
    // `HihatParams` documents that PITCH is ignored, because Godot's hat is
    // `noise_mix = 1.0` and has no oscillator to retune. A test rather than a
    // comment, because "silently ignored" is exactly the kind of claim that rots.
    let bank = bank_of(PatchDef::Hihat(HihatParams::default()));
    let script = |pitch: f32| {
        vec![
            (
                0,
                Event::Play {
                    voice: VoiceId(0),
                    patch: ONE,
                    seed: 4,
                    gain: 1.0,
                    pan: 0.0,
                },
            ),
            (
                0,
                Event::SetParam {
                    voice: VoiceId(0),
                    id: runt_audio::ParamId::PITCH,
                    value: pitch,
                },
            ),
        ]
    };
    let a = render_offline(&bank, &script(1.0), 6_000, 128, SR);
    let b = render_offline(&bank, &script(4.0), 6_000, 128, SR);
    assert_eq!(a, b, "a hat two octaves up must be the same hat");
}

#[test]
fn the_bass_holds_at_its_sustain_level_and_plays_the_note_it_was_given() {
    // The property no other model in this crate has. `sustain: 0.6` means the
    // envelope settles at 60 % of its peak and stays there — not "decays
    // slowly", which is what a `Pluck` with a long `decay_s` would give.
    let params = BassParams {
        base_hz: 73.42, // D2
        jitter_semitones: 0.0,
        pitch_drop_semitones: 0.0,
        sustain: 0.6,
        phaser_depth: 0.0, // the sweep is measured separately; here it is noise
        // …and so is the unison. Two stacks two cents apart beat against each
        // other at f * (2^(2/1200) - 1) ~= 0.085 Hz — a twelve-second cycle,
        // which is real, wanted, and completely swamps a "is the level flat"
        // measurement taken over three seconds. Turn it off to measure the
        // envelope; `the_bass_unison_beats` measures it on its own.
        unison_cents: 0.0,
        ..BassParams::default()
    };
    let buf = strike(PatchDef::Bass(params), 2, 48_000 * 3);

    let pitch = analyze::detect_pitch(&window(&buf, 0.5, 1.0), SR, 40.0, 500.0);
    println!("bass pitch: {pitch:.2} Hz");
    assert!(
        (pitch - 73.42).abs() < 73.42 * 0.04,
        "the additive stack's fundamental is base_hz (got {pitch:.2})"
    );

    let peak = analyze::rms(&window(&buf, 0.0, 0.05));
    let held = analyze::rms(&window(&buf, 1.0, 1.5));
    let later = analyze::rms(&window(&buf, 2.4, 2.9));
    println!("bass rms: peak {peak:.4}, 1 s {held:.4}, 2.5 s {later:.4}");
    assert!(held > peak * 0.35, "it must still be there at a second");
    assert!(
        (later / held - 1.0).abs() < 0.05,
        "and flat between one and two and a half ({held:.4} vs {later:.4})"
    );
}

#[test]
fn the_bass_phaser_sweeps_and_does_nothing_at_zero_depth() {
    // `phaser_depth` is Godot's `AudioEffectPhaser.depth`. Its audible signature
    // is a slow *amplitude* wobble as the notches move through the harmonics —
    // so measure the spread of short-window RMS across two LFO periods, which is
    // flat on a sustaining note and is not flat once the notches move.
    let dry = BassParams {
        jitter_semitones: 0.0,
        phaser_depth: 0.0,
        phaser_rate_hz: 0.5,
        // The unison beat is a second slow amplitude modulator (see
        // `the_bass_unison_beats`); with it on, "the level wobbles" would not
        // isolate the phaser.
        unison_cents: 0.0,
        ..BassParams::default()
    };
    let wet = BassParams {
        phaser_depth: 0.7,
        ..dry.clone()
    };

    let spread = |params: BassParams| {
        let buf = strike(PatchDef::Bass(params), 2, 48_000 * 4);
        // Windows over 0.5–4.0 s: past the attack, across two 2 s LFO cycles.
        let levels: Vec<f32> = (0..14)
            .map(|i| {
                let from = 0.5 + i as f32 * 0.25;
                analyze::rms(&window(&buf, from, from + 0.25))
            })
            .collect();
        let mean = levels.iter().sum::<f32>() / levels.len() as f32;
        let hi = levels.iter().cloned().fold(0.0f32, f32::max);
        let lo = levels.iter().cloned().fold(f32::MAX, f32::min);
        (hi - lo) / mean.max(1e-9)
    };

    let flat = spread(dry);
    let swept = spread(wet);
    println!("bass phaser: dry spread {flat:.4}, wet spread {swept:.4}");
    assert!(flat < 0.02, "a sustaining note with no phaser is flat ({flat:.4})");
    assert!(
        swept > flat * 5.0 && swept > 0.05,
        "the phaser must actually sweep ({flat:.4} -> {swept:.4})"
    );
}

#[test]
fn the_bass_pitch_drop_is_a_pluck_and_not_a_glissando() {
    // Godot's `bass.tres` drops 5 semitones over 34 ms — short enough to read as
    // an attack transient rather than as a slide. Both halves of that are
    // measurable: the start is sharp, and by 100 ms it is over.
    let params = BassParams {
        base_hz: 73.42,
        // One partial, no unison: an additive stack whose second partial is at
        // 0.65 is exactly the signal an autocorrelation tracker reports an
        // octave high on (see `analyze::detect_pitch`'s "octave trap"), and this
        // test is about the sweep, not about the tracker.
        partials: vec![1.0],
        unison_cents: 0.0,
        jitter_semitones: 0.0,
        pitch_drop_semitones: 5.0,
        pitch_drop_s: 0.034,
        phaser_depth: 0.0,
        ..BassParams::default()
    };
    let buf = strike(PatchDef::Bass(params), 2, 48_000);
    let early = analyze::detect_pitch(&window(&buf, 0.0, 0.02), SR, 40.0, 500.0);
    let late = analyze::detect_pitch(&window(&buf, 0.2, 0.6), SR, 40.0, 500.0);
    println!("bass drop: {early:.2} -> {late:.2} Hz");
    assert!(early > late * 1.10, "five semitones is 33 % ({early:.2} -> {late:.2})");
    assert!((late - 73.42).abs() < 73.42 * 0.04, "and it lands (got {late:.2})");
}

#[test]
fn the_bass_unison_beats() {
    // `unison_cents` is Godot's `detune_voices = 2` / `detune_cents = 2.0`, and
    // what two cents *does* is put a slow beat on a held note: two tones a
    // ratio r apart beat at `f * (r - 1)`, which at D2 and two cents is about
    // 0.085 Hz — a twelve-second cycle. Two other tests turn it off to measure
    // something else; this one is why it is there.
    let held = |cents: f32| {
        let buf = strike(
            PatchDef::Bass(BassParams {
                base_hz: 73.42,
                jitter_semitones: 0.0,
                pitch_drop_semitones: 0.0,
                phaser_depth: 0.0,
                unison_cents: cents,
                ..BassParams::default()
            }),
            2,
            48_000 * 6,
        );
        let levels: Vec<f32> = (0..10)
            .map(|i| {
                let from = 1.0 + i as f32 * 0.5;
                analyze::rms(&window(&buf, from, from + 0.5))
            })
            .collect();
        let hi = levels.iter().cloned().fold(0.0f32, f32::max);
        let lo = levels.iter().cloned().fold(f32::MAX, f32::min);
        (hi - lo) / hi.max(1e-9)
    };

    let single = held(0.0);
    let unison = held(2.0);
    println!("bass unison: 0 cents {single:.4}, 2 cents {unison:.4}");
    assert!(single < 0.02, "one stack holds a flat level ({single:.4})");
    assert!(unison > 0.05, "two cents apart, it breathes ({unison:.4})");
}
