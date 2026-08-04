//! What the two patches actually *sound* like, measured rather than heard.
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
    DroneParams, PatchBank, PatchDef, PatchId, PluckParams, VoicePool, REFERENCE_SAMPLE_RATE,
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
