//! The wire format is a contract between two crates that do not depend on each
//! other (`runt-core` encodes, `runt-audio` decodes), so the *bytes* are what is
//! pinned here — a round trip alone would happily agree with itself while
//! drifting away from the other side.
//!
//! `runt-core`'s `tests/audio.rs` pins the same constants from its end. If one
//! side changes the layout, one of the two suites goes red immediately.

use runt_audio::params::ParamId;
use runt_audio::wire::{self, Event, VoiceId, EVENT_SIZE};
use runt_audio::{PatchBank, PatchDef, PatchId};

#[test]
fn the_record_is_thirty_two_little_endian_bytes() {
    assert_eq!(EVENT_SIZE, 32);

    let event = Event::Play {
        voice: VoiceId(0x0102_0304),
        patch: PatchId(0x1122_3344_5566_7788),
        seed: 0xdead_beef_0000_0001,
        gain: 1.0,
        pan: -1.0,
    };
    let bytes = wire::encode(&[event]);
    assert_eq!(bytes.len(), EVENT_SIZE);

    #[rustfmt::skip]
    let expected: [u8; 32] = [
        0x00, 0x00, 0x00, 0x00,                          // kind = Play, reserved, param id
        0x04, 0x03, 0x02, 0x01,                          // voice, LE
        0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11,  // patch, LE
        0x01, 0x00, 0x00, 0x00, 0xef, 0xbe, 0xad, 0xde,  // seed, LE
        0x00, 0x00, 0x80, 0x3f,                          // gain = 1.0f32
        0x00, 0x00, 0x80, 0xbf,                          // pan = -1.0f32
    ];
    assert_eq!(bytes.as_slice(), &expected[..]);
}

#[test]
fn every_variant_round_trips() {
    let events = [
        Event::Play {
            voice: VoiceId(7),
            patch: PatchId::new("pickup"),
            seed: 42,
            gain: 0.25,
            pan: 0.5,
        },
        Event::SetParam {
            voice: VoiceId(7),
            id: ParamId::CUTOFF,
            value: 2.5,
        },
        Event::Stop { voice: VoiceId(7) },
    ];
    let bytes = wire::encode(&events);
    assert_eq!(bytes.len(), events.len() * EVENT_SIZE);
    assert_eq!(wire::decode_all(&bytes), events);
}

#[test]
fn the_unused_half_of_a_record_is_zero() {
    // Encoding has to be a pure function of the event, or the golden bytes above
    // would only hold for a freshly zeroed buffer.
    let mut record = [0xffu8; EVENT_SIZE];
    Event::Stop { voice: VoiceId(1) }.encode(&mut record);
    assert_eq!(&record[8..], &[0u8; 24][..], "Stop carries nothing but a voice");
}

#[test]
fn a_ragged_tail_is_ignored_rather_than_fatal() {
    let mut bytes = wire::encode(&[Event::Stop { voice: VoiceId(3) }]);
    bytes.extend_from_slice(&[1, 2, 3]); // a truncated postMessage
    assert_eq!(wire::decode_all(&bytes).len(), 1);
}

#[test]
fn an_unknown_kind_is_skipped_not_guessed_at() {
    let mut bytes = wire::encode(&[
        Event::Stop { voice: VoiceId(1) },
        Event::Stop { voice: VoiceId(2) },
    ]);
    bytes[0] = 200; // a kind from a future build
    let decoded = wire::decode_all(&bytes);
    assert_eq!(decoded, vec![Event::Stop { voice: VoiceId(2) }]);
}

#[test]
fn the_shared_param_vocabulary_is_pinned() {
    // Restated in `runt_core::audio::ParamId`. These four numbers are the wire.
    assert_eq!(ParamId::GAIN.0, 0);
    assert_eq!(ParamId::PAN.0, 1);
    assert_eq!(ParamId::PITCH.0, 2);
    assert_eq!(ParamId::CUTOFF.0, 3);
}

#[test]
fn patch_ids_are_the_fnv_of_the_name() {
    // Also restated in `runt_core::audio::PatchId`; both sides must agree or a
    // `Play` would name a preset the synth does not have.
    assert_eq!(PatchId::new(""), PatchId(0xcbf2_9ce4_8422_2325));
    assert_eq!(PatchId::new("pluck"), PatchId(0x980a_104d_ddba_6b6a));
    assert_eq!(PatchId::new("drone"), PatchId(0x6d09_40c9_3eca_e8d1));
    assert_ne!(PatchId::new("pluck"), PatchId::new("drone"));
}

// ---------------------------------------------------------------------------
// The bank's schema
// ---------------------------------------------------------------------------

#[test]
fn the_patch_def_discriminants_are_the_bank_format_and_do_not_move() {
    // postcard writes an enum as a leading varint discriminant, so the *order*
    // of `PatchDef`'s variants is part of the bank's byte format. New models go
    // on the end; inserting one in the middle silently reinterprets every bank
    // ever written. This is the test that turns that from a convention into a
    // rule — and it is why there is no version field (see `PatchDef`'s docs).
    use runt_audio::{
        BassParams, DroneParams, HihatParams, KickParams, PluckParams, SnareParams,
    };
    let expected = [
        (0u32, PatchDef::Pluck(PluckParams::default()), "pluck"),
        (1, PatchDef::Drone(DroneParams::default()), "drone"),
        (2, PatchDef::Kick(KickParams::default()), "kick"),
        (3, PatchDef::Snare(SnareParams::default()), "snare"),
        (4, PatchDef::Hihat(HihatParams::default()), "hihat"),
        (5, PatchDef::Bass(BassParams::default()), "bass"),
    ];
    for (index, def, model) in &expected {
        assert_eq!(def.discriminant(), *index, "{model} moved");
        assert_eq!(def.model(), *model);
        // …and the claim is about the *bytes*, so check them: a one-entry bank
        // encodes as [count][name len][name][discriminant][params…].
        let bytes = PatchBank::new()
            .with("x", def.clone())
            .to_bytes()
            .expect("encode");
        assert_eq!(bytes[0], 1, "one entry");
        assert_eq!(&bytes[1..3], b"\x01x", "the name, length-prefixed");
        assert_eq!(bytes[3] as u32, *index, "{model}'s discriminant on the wire");
    }
    assert_eq!(
        PatchBank::SCHEMA as usize,
        expected.len(),
        "SCHEMA counts the models"
    );
}

#[test]
fn a_bank_written_before_the_music_models_still_decodes() {
    // The forward-compatibility half, from the other side: appending variants
    // must not change what an *existing* blob means. Rather than paste a hex
    // dump that nobody can check, this reconstructs the pre-BGM schema as a
    // shadow enum — two variants, exactly what `PatchDef` was — serializes a
    // bank with it, and hands the bytes to the real decoder.
    use runt_audio::{DroneParams, PluckParams};
    use serde::Serialize;

    #[derive(Serialize)]
    enum OldPatchDef {
        Pluck(PluckParams),
        Drone(DroneParams),
    }
    #[derive(Serialize)]
    struct OldEntry {
        name: String,
        def: OldPatchDef,
    }
    #[derive(Serialize)]
    struct OldBank {
        entries: Vec<OldEntry>,
    }

    // Sorted by `PatchId`, which is the invariant `PatchBank::insert` maintains
    // and which `get` binary-searches on.
    let mut entries = vec![
        ("drone", OldPatchDef::Drone(DroneParams::default())),
        ("pluck", OldPatchDef::Pluck(PluckParams::default())),
    ];
    entries.sort_by_key(|(name, _)| PatchId::new(name));
    let old = OldBank {
        entries: entries
            .into_iter()
            .map(|(name, def)| OldEntry {
                name: name.to_string(),
                def,
            })
            .collect(),
    };
    let bytes = postcard::to_stdvec(&old).expect("encode with the old schema");

    let decoded = PatchBank::from_bytes(&bytes).expect("a pre-BGM bank must still decode");
    assert_eq!(decoded.len(), 2);
    assert_eq!(
        decoded.get_by_name("pluck"),
        Some(&PatchDef::Pluck(PluckParams::default())),
        "the old Pluck bytes still mean Pluck"
    );
    assert_eq!(
        decoded.get_by_name("drone"),
        Some(&PatchDef::Drone(DroneParams::default())),
    );
    // …and it round-trips back to the same bytes, so nothing shifted.
    assert_eq!(decoded.to_bytes().expect("re-encode"), bytes);
}
