//! Determinism, scoped exactly as DESIGN §8 scopes it.
//!
//! > *same params + seed + build + platform → bit-identical samples. Cross-
//! > platform bit-identity is **not** promised and does not hold (~1e-10 libm
//! > divergence through IIR state).* — DESIGN §8
//!
//! So there is **no hard-pinned hash constant here**, and that is a deliberate
//! choice rather than a gap. A pinned constant would encode this machine's libm
//! and fail on the next one for a reason that has nothing to do with the code
//! being wrong — it would be a test that lies. What *is* checkable on every
//! machine is the property itself: two independent renders of the same script
//! agree bit for bit, a different seed does not, and nothing in the buffer is
//! subnormal or non-finite.
//!
//! (The spike did record concrete hashes — `0xcc9ec2a6ec256bfd` native,
//! `0xfe98d88ffabef45d` wasm — precisely to *demonstrate* that the two differ.
//! FINDINGS is the place for a measured number; a test is not.)

// Everything below drives the synthesizer, which only exists behind `dsp`.
// A default-feature build of this crate is the *description* half — the patch
// bank and the wire codec — and has nothing here to test.
#![cfg(feature = "dsp")]

use runt_audio::analyze;
use runt_audio::voice::{canonical_render, canonical_script, hash_samples, render_offline};
use runt_audio::wire::{Event, VoiceId};
use runt_audio::{PatchBank, PatchDef, PatchId, PluckParams, REFERENCE_SAMPLE_RATE};

fn sr() -> f32 {
    REFERENCE_SAMPLE_RATE as f32
}

#[test]
fn two_fresh_renders_of_the_same_script_are_bit_identical() {
    // Fresh pools each time, so this covers construction order and any lazily
    // initialised global (fundsp's wavetables are shared and built on first use
    // — if that leaked state into the first render, this is what would catch it).
    let a = canonical_render();
    let b = canonical_render();
    assert_eq!(a.len(), 96_000, "one second of stereo at 48 kHz");
    assert_eq!(
        hash_samples(&a),
        hash_samples(&b),
        "same params + seed + build + platform must give the same samples"
    );
    assert_eq!(a, b);
}

#[test]
fn the_seed_actually_changes_the_sound() {
    // Otherwise "same seed → same output" is true and vacuous.
    let bank = PatchBank::builtin();
    let with = |seed: u64| {
        render_offline(
            &bank,
            &[(
                0,
                Event::Play {
                    voice: VoiceId(0),
                    patch: PatchId::new("pluck"),
                    seed,
                    gain: 1.0,
                    pan: 0.0,
                },
            )],
            12_000,
            128,
            sr(),
        )
    };
    let a = hash_samples(&with(1));
    let b = hash_samples(&with(2));
    assert_ne!(a, b, "a different seed must pick a different note");
    assert_eq!(a, hash_samples(&with(1)), "and the same seed must not");
}

#[test]
fn nothing_rendered_is_subnormal_or_non_finite() {
    // The claim `SILENCE` exists to support: cutting a decaying envelope at
    // −80 dB keeps every sample a normal float, so no target's flush-to-zero
    // policy can make two machines disagree about a tail nobody can hear.
    let buf = canonical_render();
    assert_eq!(analyze::anomalies(&buf), (0, 0));
}

#[test]
fn a_long_idle_tail_stays_exactly_zero() {
    // A pluck left to ring out for eight seconds. Once the envelope is cut, the
    // pool renders nothing at all for that slot — not "very small numbers".
    let bank = PatchBank::builtin();
    let buf = render_offline(
        &bank,
        &[(
            0,
            Event::Play {
                voice: VoiceId(0),
                patch: PatchId::new("pluck"),
                seed: 9,
                gain: 1.0,
                pan: 0.0,
            },
        )],
        48_000 * 8,
        128,
        sr(),
    );
    let tail = &buf[48_000 * 2 * 4..];
    assert!(
        tail.iter().all(|s| *s == 0.0),
        "a finished voice must contribute exact zeros, not a subnormal drizzle"
    );
    assert_eq!(analyze::anomalies(&buf), (0, 0));
}

#[test]
fn a_noise_preset_is_deterministic_across_two_independent_pools() {
    // A noise source is the one thing in the crate that could plausibly reach
    // for entropy, so it gets the same treatment `canonical_render` gets: two
    // fresh pools, one hash. The noise is fundsp's, seeded by the event's seed
    // through `ping` — DESIGN §3's "the RNG is visible and seeded", arriving at
    // the audio thread by the only route this crate has.
    let bank = PatchBank::new().with(
        "splash",
        PatchDef::Pluck(PluckParams {
            noise_mix: 0.85,
            noise_decay_s: 0.14,
            noise_highpass_hz: 600.0,
            decay_s: 1.0,
            ..PluckParams::default()
        }),
    );
    let render = |seed: u64| {
        render_offline(
            &bank,
            &[(
                0,
                Event::Play {
                    voice: VoiceId(0),
                    patch: PatchId::new("splash"),
                    seed,
                    gain: 1.0,
                    pan: 0.0,
                },
            )],
            48_000 * 2,
            128,
            sr(),
        )
    };
    let a = render(3);
    let b = render(3);
    assert_eq!(hash_samples(&a), hash_samples(&b));
    assert_eq!(a, b, "the same seed must produce the same noise");
    assert_ne!(
        hash_samples(&a),
        hash_samples(&render(4)),
        "and a different seed must produce different noise"
    );
    assert!(analyze::peak(&a) > 0.05, "a noise preset must not be silent");
    // The second envelope is cut at `SILENCE` for the same reason the first one
    // is; this is the check that it actually is.
    assert_eq!(analyze::anomalies(&a), (0, 0));
    let tail = &a[48_000 * 2 * 3 / 2..];
    assert!(
        tail.iter().all(|s| *s == 0.0),
        "a finished noisy voice must contribute exact zeros too"
    );
}

#[test]
fn the_bank_hash_is_a_content_address() {
    // DESIGN §6, one level down: same presets → same hash, regardless of the
    // order they were inserted in; a changed parameter → a different hash.
    let a = PatchBank::new()
        .with("one", PatchDef::Pluck(PluckParams::default()))
        .with("two", PatchDef::Pluck(PluckParams::default()));
    let b = PatchBank::new()
        .with("two", PatchDef::Pluck(PluckParams::default()))
        .with("one", PatchDef::Pluck(PluckParams::default()));
    assert_eq!(a.param_hash(), b.param_hash());

    let mut c = a.clone();
    c.insert(
        "one",
        PatchDef::Pluck(PluckParams {
            decay_s: 0.36,
            ..PluckParams::default()
        }),
    );
    assert_ne!(a.param_hash(), c.param_hash());

    // …including the noise fields, which are the newest and therefore the ones a
    // hash taken over a stale field list would silently miss.
    for noisy in [
        PluckParams {
            noise_mix: 0.5,
            ..PluckParams::default()
        },
        PluckParams {
            noise_decay_s: 0.12,
            ..PluckParams::default()
        },
        PluckParams {
            noise_highpass_hz: 600.0,
            ..PluckParams::default()
        },
    ] {
        let mut d = a.clone();
        d.insert("one", PatchDef::Pluck(noisy.clone()));
        assert_ne!(a.param_hash(), d.param_hash(), "{noisy:?} did not move the hash");
        // …and the whole preset survives the byte form the worklet loads.
        let bytes = d.to_bytes().expect("encode");
        assert_eq!(PatchBank::from_bytes(&bytes).expect("decode"), d);
    }
}

#[test]
fn the_bank_survives_the_byte_form_both_hosts_load_from() {
    let bank = PatchBank::builtin();
    let bytes = bank.to_bytes().expect("encode");
    let back = PatchBank::from_bytes(&bytes).expect("decode");
    assert_eq!(bank, back);
    assert_eq!(bank.param_hash(), back.param_hash());
    assert!(
        bytes.len() < 1024,
        "the bank crosses the worklet boundary; {} bytes is not compact",
        bytes.len()
    );
}

#[test]
fn a_bad_bank_blob_is_an_error_and_not_a_panic() {
    // The worklet turns this into a return code; nothing may take the audio
    // thread down.
    assert!(PatchBank::from_bytes(&[0xff; 16]).is_err() || PatchBank::from_bytes(&[]).is_err());
}

#[test]
fn the_canonical_script_exercises_more_than_one_voice() {
    // Guards the guard: a determinism test over a script that only ever played
    // one note would miss every mixing and stealing bug.
    let script = canonical_script();
    let plays = script
        .iter()
        .filter(|(_, e)| matches!(e, Event::Play { .. }))
        .count();
    assert!(plays >= 5, "{plays} plays is a thin canonical render");
    assert!(script
        .iter()
        .any(|(_, e)| matches!(e, Event::SetParam { .. })));
    assert!(script.iter().any(|(_, e)| matches!(e, Event::Stop { .. })));
}
