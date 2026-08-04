//! The voice pool: stealing, the master bus, and what happens when game code
//! says something silly.
//!
//! Everything here runs offline. The pool a test drives is the pool the cpal
//! callback and the worklet drive — same struct, same entry points — so a green
//! run is a statement about the shipped mixer.

// Everything below drives the synthesizer, which only exists behind `dsp`.
// A default-feature build of this crate is the *description* half — the patch
// bank and the wire codec — and has nothing here to test.
#![cfg(feature = "dsp")]

use runt_audio::analyze;
use runt_audio::params::ParamId;
use runt_audio::voice::MAX_VOICES;
use runt_audio::wire::{self, Event, VoiceId};
use runt_audio::{PatchBank, PatchId, VoicePool, REFERENCE_SAMPLE_RATE};

const SR: f32 = REFERENCE_SAMPLE_RATE as f32;
const PLUCK: PatchId = PatchId::new("pluck");
const DRONE: PatchId = PatchId::new("drone");

fn pool() -> VoicePool {
    VoicePool::new(PatchBank::builtin(), SR)
}

fn play(voice: u32, patch: PatchId, seed: u64, gain: f32, pan: f32) -> Event {
    Event::Play {
        voice: VoiceId(voice),
        patch,
        seed,
        gain,
        pan,
    }
}

/// Render `frames` frames and return them interleaved.
fn render(pool: &mut VoicePool, frames: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; frames * 2];
    for chunk in out.chunks_mut(128 * 2) {
        pool.render_interleaved(chunk);
    }
    out
}

// ---------------------------------------------------------------------------
// Voices
// ---------------------------------------------------------------------------

#[test]
fn a_play_starts_exactly_one_voice_and_it_ends_on_its_own() {
    let mut p = pool();
    assert_eq!(p.active_voices(), 0);
    p.apply(play(1, PLUCK, 0, 1.0, 0.0));
    assert_eq!(p.active_voices(), 1);
    assert!(p.is_playing(VoiceId(1)));

    // The default pluck decays over 0.35 s; two seconds is well past the cut.
    render(&mut p, 96_000);
    assert_eq!(p.active_voices(), 0, "a pluck must free its own slot");
    assert!(!p.is_playing(VoiceId(1)));
    assert_eq!(p.stats().played, 1);
    assert_eq!(p.stats().stolen, 0);
}

#[test]
fn a_drone_holds_until_it_is_stopped() {
    let mut p = pool();
    p.apply(play(1, DRONE, 0, 1.0, 0.0));
    render(&mut p, 48_000 * 6);
    assert_eq!(p.active_voices(), 1, "a drone has no end of its own");

    p.apply(Event::Stop { voice: VoiceId(1) });
    render(&mut p, 48_000 * 4); // release_s is 1.5
    assert_eq!(p.active_voices(), 0, "and releases when told to");
}

#[test]
fn the_pool_fills_to_its_cap_and_then_steals_the_quietest() {
    let mut p = pool();
    // Sixteen voices at descending gains. Voice 0 is the quietest by a factor of
    // sixteen, so once the envelopes have moved it is unambiguously the one a
    // steal should take.
    for i in 0..MAX_VOICES as u32 {
        p.apply(play(i, PLUCK, i as u64, 0.05 + i as f32 * 0.06, 0.0));
    }
    assert_eq!(p.active_voices(), MAX_VOICES);
    render(&mut p, 2_400); // 50 ms: past every attack, before any decay is done

    for i in 0..MAX_VOICES as u32 {
        assert!(p.is_playing(VoiceId(i)), "voice {i} should still be sounding");
    }

    p.apply(play(99, PLUCK, 99, 1.0, 0.0));
    assert_eq!(p.stats().stolen, 1);
    assert_eq!(p.active_voices(), MAX_VOICES, "the cap is a cap");
    assert!(p.is_playing(VoiceId(99)));
    assert!(
        !p.is_playing(VoiceId(0)),
        "the quietest voice is the one that should have been taken"
    );
    for i in 1..MAX_VOICES as u32 {
        assert!(p.is_playing(VoiceId(i)), "voice {i} was not the quietest");
    }
}

#[test]
fn a_tie_on_loudness_goes_to_the_oldest() {
    let mut p = pool();
    // Identical gains and identical seeds → identical levels at every sample, so
    // only the age tiebreak can decide. Voice 0 was triggered first.
    for i in 0..MAX_VOICES as u32 {
        p.apply(play(i, PLUCK, 7, 1.0, 0.0));
    }
    render(&mut p, 2_400);
    p.apply(play(99, PLUCK, 7, 1.0, 0.0));
    assert!(!p.is_playing(VoiceId(0)), "oldest first among equals");
    assert!(p.is_playing(VoiceId(1)));
}

#[test]
fn a_free_slot_is_always_preferred_to_a_steal() {
    let mut p = pool();
    for i in 0..MAX_VOICES as u32 {
        p.apply(play(i, PLUCK, i as u64, 1.0, 0.0));
    }
    render(&mut p, 96_000); // everything decays away
    assert_eq!(p.active_voices(), 0);
    p.apply(play(99, PLUCK, 99, 1.0, 0.0));
    assert_eq!(p.stats().stolen, 0, "nothing was sounding; nothing was stolen");
}

// ---------------------------------------------------------------------------
// The master bus
// ---------------------------------------------------------------------------

#[test]
fn no_sample_leaves_the_bus_outside_plus_or_minus_one() {
    let mut p = pool();
    p.set_master_gain(8.0); // the hard cap on master gain
    for i in 0..MAX_VOICES as u32 {
        // Sixteen voices at four times unity, deliberately absurd.
        p.apply(play(i, PLUCK, i as u64, 4.0, 0.0));
    }
    let buf = render(&mut p, 48_000);

    let peak = analyze::peak(&buf);
    assert!(peak <= 1.0, "soft clip must bound the bus; peak was {peak}");
    assert!(
        p.stats().peak_pre_clip > 1.0,
        "this test is only meaningful if the limiter had work to do (pre-clip peak {})",
        p.stats().peak_pre_clip
    );
    assert!(peak > 0.5, "and it must not have flattened the mix to nothing");
    assert_eq!(analyze::anomalies(&buf), (0, 0));
    assert_eq!(p.stats().nan_guarded, 0);
}

#[test]
fn a_quiet_mix_passes_through_essentially_untouched() {
    // tanh is ~linear near zero: a game at sane levels is not being compressed.
    let mut p = pool();
    p.apply(play(0, PLUCK, 3, 0.2, 0.0));
    let buf = render(&mut p, 24_000);
    let peak = analyze::peak(&buf);
    assert!(peak > 0.0 && peak < 0.3, "peak was {peak}");
}

#[test]
fn pan_places_energy_on_the_side_it_says() {
    let mut p = pool();
    p.apply(play(0, PLUCK, 5, 1.0, -1.0));
    let buf = render(&mut p, 24_000);
    let left = analyze::rms(&analyze::channel(&buf, 0));
    let right = analyze::rms(&analyze::channel(&buf, 1));
    assert!(left > 0.0);
    assert!(
        right < left * 1e-3,
        "hard left must be silent on the right (l={left}, r={right})"
    );

    let mut p = pool();
    p.apply(play(0, PLUCK, 5, 1.0, 1.0));
    let buf = render(&mut p, 24_000);
    let left = analyze::rms(&analyze::channel(&buf, 0));
    let right = analyze::rms(&analyze::channel(&buf, 1));
    assert!(left < right * 1e-3, "and the mirror image (l={left}, r={right})");
}

#[test]
fn centre_pan_is_constant_power_not_constant_amplitude() {
    // A source swept across the field should not dip in the middle. Compare the
    // summed power of a centred voice against a hard-panned one.
    let power = |pan: f32| {
        let mut p = pool();
        p.apply(play(0, PLUCK, 5, 0.3, pan));
        let buf = render(&mut p, 24_000);
        let l = analyze::rms(&analyze::channel(&buf, 0));
        let r = analyze::rms(&analyze::channel(&buf, 1));
        l * l + r * r
    };
    let centre = power(0.0);
    let edge = power(-1.0);
    let ratio = centre / edge;
    assert!(
        (0.9..1.1).contains(&ratio),
        "constant power means the ratio is ~1, got {ratio}"
    );
}

// ---------------------------------------------------------------------------
// Bad input
// ---------------------------------------------------------------------------

#[test]
fn a_play_for_an_unknown_patch_is_dropped_and_counted() {
    let mut p = pool();
    p.apply(play(0, PatchId::new("nope"), 0, 1.0, 0.0));
    assert_eq!(p.active_voices(), 0);
    assert_eq!(p.stats().dropped_unknown, 1);
    assert_eq!(p.stats().played, 0);
}

#[test]
fn addressing_a_finished_voice_is_counted_not_fatal() {
    let mut p = pool();
    p.apply(Event::Stop { voice: VoiceId(42) });
    p.apply(Event::SetParam {
        voice: VoiceId(42),
        id: ParamId::GAIN,
        value: 0.5,
    });
    assert_eq!(p.stats().stale_addressed, 2);
}

#[test]
fn params_changed_mid_render_produce_no_nan_and_no_subnormals() {
    // The clickless-ish claim, stated as the part that is actually assertable:
    // a live parameter edit — including a hostile one — must not put a NaN or a
    // subnormal into the stream, and must not have to be caught by the guard.
    let mut p = pool();
    p.apply(play(0, PLUCK, 1, 0.8, 0.0));
    p.apply(play(1, DRONE, 2, 0.6, 0.0));

    let mut buf = vec![0.0f32; 48_000 * 2];
    let hostile = [
        (ParamId::PITCH, 0.0f32),
        (ParamId::PITCH, f32::NAN),
        (ParamId::CUTOFF, -1.0),
        (ParamId::CUTOFF, 1.0e30),
        (ParamId::GAIN, f32::INFINITY),
        (ParamId::PAN, f32::NAN),
        (ParamId::PAN, 40.0),
        (ParamId::PITCH, 4.0),
        (ParamId::CUTOFF, 0.02),
        (ParamId::GAIN, 1.0),
    ];
    for (index, chunk) in buf.chunks_mut(128 * 2).enumerate() {
        if let Some((id, value)) = hostile.get(index % 37) {
            p.apply(Event::SetParam {
                voice: VoiceId(1),
                id: *id,
                value: *value,
            });
        }
        p.render_interleaved(chunk);
    }

    assert_eq!(analyze::anomalies(&buf), (0, 0));
    assert_eq!(
        p.stats().nan_guarded,
        0,
        "the master guard is a safety net, not a load-bearing part"
    );
    assert!(analyze::peak(&buf) <= 1.0);
}

#[test]
fn gain_and_pan_glide_on_a_running_voice_but_snap_on_a_new_one() {
    // The glide exists so a live pan sweep does not step; the snap exists so a
    // new note does not slide in from wherever the slot's last occupant was.
    let mut p = pool();
    p.apply(play(0, DRONE, 1, 1.0, -1.0));
    render(&mut p, 48_000 * 3); // past the attack

    p.apply(Event::SetParam {
        voice: VoiceId(0),
        id: ParamId::PAN,
        value: 1.0,
    });
    // One millisecond after a hard-left-to-hard-right jump: with a 5 ms glide
    // the right channel must still be well below the left.
    let buf = render(&mut p, 48);
    let l = analyze::rms(&analyze::channel(&buf, 0));
    let r = analyze::rms(&analyze::channel(&buf, 1));
    assert!(l > r, "a pan jump must glide, not teleport (l={l}, r={r})");

    // The same 100 ms later: it has arrived.
    let buf = render(&mut p, 4_800);
    let l = analyze::rms(&analyze::channel(&buf, 0));
    let r = analyze::rms(&analyze::channel(&buf, 1));
    assert!(r > l, "and it must actually arrive (l={l}, r={r})");

    // A brand new voice hard left has no trace of the drone's position.
    let mut p = pool();
    p.apply(play(7, PLUCK, 1, 1.0, 1.0));
    let buf = render(&mut p, 128);
    let l = analyze::rms(&analyze::channel(&buf, 0));
    assert!(l < 1e-4, "a new voice starts at its own pan (l={l})");
}

// ---------------------------------------------------------------------------
// The wire, end to end
// ---------------------------------------------------------------------------

#[test]
fn submitting_bytes_is_the_same_as_applying_events() {
    let events = [
        play(0, PLUCK, 11, 0.7, -0.3),
        play(1, DRONE, 12, 0.4, 0.3),
        Event::SetParam {
            voice: VoiceId(0),
            id: ParamId::CUTOFF,
            value: 1.5,
        },
    ];

    let mut direct = pool();
    for event in events {
        direct.apply(event);
    }
    let a = render(&mut direct, 24_000);

    let mut wired = pool();
    assert_eq!(wired.submit_bytes(&wire::encode(&events)), events.len());
    let b = render(&mut wired, 24_000);

    assert_eq!(a, b, "the byte path must be the event path");
}

#[test]
fn planar_and_interleaved_rendering_agree() {
    // The worklet takes the planar path and cpal the interleaved one; they must
    // not be two different mixers.
    let mut a = pool();
    a.apply(play(0, PLUCK, 21, 0.9, -0.5));
    let mut interleaved = vec![0.0f32; 128 * 2];
    a.render_interleaved(&mut interleaved);

    let mut b = pool();
    b.apply(play(0, PLUCK, 21, 0.9, -0.5));
    let (mut left, mut right) = (vec![0.0f32; 128], vec![0.0f32; 128]);
    b.render_planar(&mut left, &mut right);

    for i in 0..128 {
        assert_eq!(interleaved[i * 2], left[i]);
        assert_eq!(interleaved[i * 2 + 1], right[i]);
    }
}

#[test]
fn a_block_larger_than_the_internal_maximum_still_renders() {
    // cpal is entitled to ask for whatever the device wants.
    let mut p = pool();
    p.apply(play(0, PLUCK, 1, 1.0, 0.0));
    let mut out = vec![0.0f32; 4096 * 2];
    p.render_interleaved(&mut out);
    assert!(analyze::rms(&out) > 0.0);
    assert_eq!(analyze::anomalies(&out), (0, 0));
}
