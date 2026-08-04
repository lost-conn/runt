//! The patch bank: named presets, content-addressed (DESIGN §8, §6).
//!
//! A [`PatchId`] is the hash of a name, and a [`PatchBank`] maps ids to param
//! structs. That indirection is what lets `AudioEvent::Play` carry four numbers
//! and nothing else: the *sound* is content, addressed by name, and the tick
//! only ever says "make the one called `pickup`, seeded thus, this loud, over
//! there".
//!
//! ## Why a name hash and not an index
//!
//! An index is a promise about the bank's ordering that a scene file would have
//! to keep. A name hash is stable under insertion, reordering, and a game that
//! ships two banks; it survives a recorded input trace outliving the bank it was
//! recorded against, which is what DESIGN §4's replay story quietly requires. A
//! `Play` for an id the bank has never heard of is *dropped* — with a counter,
//! not a panic (see [`PoolStats`](crate::voice::PoolStats)) — because a missing
//! sound must never be able to take an audio thread down.
//!
//! ## Crossing the worklet boundary
//!
//! The bank is handed to the synth as **postcard bytes**
//! ([`to_bytes`](PatchBank::to_bytes)). The game's wasm module builds it, JS
//! copies the blob into the worklet's linear memory, the worklet decodes it once
//! before its first `process()`. Nothing about that path is web-specific: the
//! native host decodes the same bytes.

use serde::{Deserialize, Serialize};

use crate::params::{BassParams, DroneParams, HihatParams, KickParams, PluckParams, SnareParams};

/// A patch preset's stable identity: FNV-1a over its name.
///
/// `const`, so a game can write `const PICKUP: PatchId = PatchId::new("pickup")`
/// and pay nothing at runtime. The identical function lives in
/// `runt_core::audio::PatchId` — `runt-core` does not depend on this crate
/// (DESIGN §2), so the two are pinned against the same constants by a test on
/// each side rather than by a shared type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PatchId(pub u64);

impl PatchId {
    /// The id of a patch called `name`.
    pub const fn new(name: &str) -> PatchId {
        let bytes = name.as_bytes();
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        let mut i = 0;
        while i < bytes.len() {
            hash ^= bytes[i] as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            i += 1;
        }
        PatchId(hash)
    }
}

/// Which synthesis model a preset uses, plus its params.
///
/// An enum rather than a trait object because a bank is *data* — it round-trips
/// through postcard and (phase 4) through scene RON, and a serialized trait
/// object is a plugin system nobody asked for. Adding a model is one variant
/// here, one arm in [`crate::voice`], and one struct in [`crate::params`].
///
/// ## The variant order is the wire format
///
/// postcard encodes an enum as a leading varint discriminant, so **the order of
/// these variants is part of the bank's byte format** and new models go on the
/// end, never in the middle. That rule is what makes growing this enum a
/// non-breaking change: a bank written before [`Kick`](PatchDef::Kick) existed
/// still decodes byte-for-byte, because nothing it contains moved.
///
/// The reverse direction — a *newer* bank handed to an *older* synth — fails
/// cleanly rather than silently: postcard rejects the unknown discriminant,
/// `PatchBank::from_bytes` returns `Err`, and `runt_audio_load_bank` returns 0
/// instead of panicking on an audio thread. There is no version field because
/// there is nothing a version field could do that this does not already do; see
/// [`SCHEMA`](PatchBank::SCHEMA) for the number a test pins.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect))]
pub enum PatchDef {
    Pluck(PluckParams),
    Drone(DroneParams),
    // -- appended 2026-08-04 with the BGM models. Order is the wire format. --
    Kick(KickParams),
    Snare(SnareParams),
    Hihat(HihatParams),
    Bass(BassParams),
}

impl PatchDef {
    /// The postcard discriminant this variant serializes as. Pinned by
    /// `tests/wire.rs`; see the type docs for why it is load-bearing.
    pub fn discriminant(&self) -> u32 {
        match self {
            PatchDef::Pluck(_) => 0,
            PatchDef::Drone(_) => 1,
            PatchDef::Kick(_) => 2,
            PatchDef::Snare(_) => 3,
            PatchDef::Hihat(_) => 4,
            PatchDef::Bass(_) => 5,
        }
    }

    /// A short stable name for the model, for diagnostics and test messages.
    pub fn model(&self) -> &'static str {
        match self {
            PatchDef::Pluck(_) => "pluck",
            PatchDef::Drone(_) => "drone",
            PatchDef::Kick(_) => "kick",
            PatchDef::Snare(_) => "snare",
            PatchDef::Hihat(_) => "hihat",
            PatchDef::Bass(_) => "bass",
        }
    }
}

/// One named entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PatchEntry {
    pub name: String,
    pub def: PatchDef,
}

/// A set of named presets, sorted by id.
///
/// Sorted so lookup is a binary search with no hashing on the audio thread, and
/// so [`to_bytes`](PatchBank::to_bytes) is a pure function of the *contents*
/// rather than of the order they were inserted in — which is what makes
/// [`param_hash`](PatchBank::param_hash) a content address (DESIGN §6).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PatchBank {
    entries: Vec<PatchEntry>,
}

impl PatchBank {
    /// How many synthesis models [`PatchDef`] knows about.
    ///
    /// This is the closest thing to a bank *version* the format has, and it is
    /// deliberately a count rather than a number somebody has to remember to
    /// bump: adding a variant changes it, and `tests/wire.rs` pins both it and
    /// the individual discriminants. Read [`PatchDef`] for why appending is
    /// enough and a version field would not add anything.
    pub const SCHEMA: u32 = 6;

    pub fn new() -> PatchBank {
        PatchBank::default()
    }

    /// The presets this crate ships. Generic on purpose — a game names its own
    /// (`"pickup"`, `"thud"`) with its own params; these are what the
    /// `audition` example plays and what the tests measure.
    ///
    /// One per model, so `builtin()` doubles as the answer to "does every
    /// variant in [`PatchDef`] actually make a sound" — which is what
    /// `tests/patches.rs` asks it.
    pub fn builtin() -> PatchBank {
        PatchBank::new()
            .with("pluck", PatchDef::Pluck(PluckParams::default()))
            .with("drone", PatchDef::Drone(DroneParams::default()))
            .with("kick", PatchDef::Kick(KickParams::default()))
            .with("snare", PatchDef::Snare(SnareParams::default()))
            .with("hihat", PatchDef::Hihat(HihatParams::default()))
            .with("bass", PatchDef::Bass(BassParams::default()))
    }

    /// Builder form of [`insert`](PatchBank::insert).
    pub fn with(mut self, name: &str, def: PatchDef) -> PatchBank {
        self.insert(name, def);
        self
    }

    /// Add or replace the preset called `name`.
    pub fn insert(&mut self, name: &str, def: PatchDef) {
        let id = PatchId::new(name);
        match self.entries.binary_search_by_key(&id, |e| PatchId::new(&e.name)) {
            Ok(at) => self.entries[at].def = def,
            Err(at) => self.entries.insert(
                at,
                PatchEntry {
                    name: name.to_string(),
                    def,
                },
            ),
        }
    }

    /// The preset `id` names, or `None`.
    pub fn get(&self, id: PatchId) -> Option<&PatchDef> {
        self.entries
            .binary_search_by_key(&id, |e| PatchId::new(&e.name))
            .ok()
            .map(|at| &self.entries[at].def)
    }

    /// The preset called `name`.
    pub fn get_by_name(&self, name: &str) -> Option<&PatchDef> {
        self.get(PatchId::new(name))
    }

    pub fn entries(&self) -> &[PatchEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The compact byte form both hosts load from.
    pub fn to_bytes(&self) -> Result<Vec<u8>, postcard::Error> {
        postcard::to_stdvec(self)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<PatchBank, postcard::Error> {
        postcard::from_bytes(bytes)
    }

    /// FNV-1a over the postcard bytes — the bank's content address (DESIGN §6:
    /// *"params hash → content hash"*, one level down from meshes).
    ///
    /// Two banks with the same presets hash the same regardless of insertion
    /// order, because the entries are kept sorted.
    pub fn param_hash(&self) -> u64 {
        let bytes = self.to_bytes().unwrap_or_default();
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for b in bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }
}
