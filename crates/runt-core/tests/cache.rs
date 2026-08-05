//! The content cache (DESIGN §6).
//!
//! One rule dominates: **the cache is purely an optimization**. §6 puts it
//! plainly — "deleting it must never change output (determinism makes this
//! checkable in CI)". So the headline test runs the same generators against a
//! cold cache, a warm cache, and a cache deleted out from under a live session,
//! and demands byte-identical geometry from all three.
//!
//! Everything else here is the same claim from a different angle: corrupt
//! entries are refused, a store that lies is ignored, and no hit is observable
//! except through the counters.

use std::path::{Path, PathBuf};

use glam::{Vec2, Vec3};
use runt_core::cache::{CacheStore, GenCache, MemCache, NativeDiskCache, NoopCache};
use runt_core::gen::{GeneratorSpec, Shading};
use runt_core::{MeshHandle, MeshLibrary, Quality, TerrainParams};
use runt_mesh::MeshData;

/// A scratch directory unique to this test binary and test name, removed on
/// drop even when the test panics.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> TempDir {
        let dir = std::env::temp_dir().join(format!("runt-cache-test-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        TempDir(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn specs() -> Vec<GeneratorSpec> {
    vec![
        GeneratorSpec::UvSphere {
            radius: 0.9,
            rings: 24,
            sectors: 32,
            shading: Shading::Smooth(180.0),
            color: Some(Vec3::new(0.9, 0.35, 0.35)),
        },
        GeneratorSpec::TwistedBox {
            dims: Vec3::new(1.0, 1.6, 1.0),
            twist: 0.9,
            taper: 0.4,
            shading: Shading::Flat,
            color: None,
        },
        GeneratorSpec::Terrain(TerrainParams {
            seed: 20260731,
            size: Vec2::splat(40.0),
            amplitude: 1.2,
            octaves: 4,
            frequency: 0.055,
            lacunarity: 2.0,
            gain: 0.5,
            base_segments: 32,
            color: Some(Vec3::new(0.17, 0.21, 0.18)),
            tint: None,
        }),
    ]
}

/// Resolve every spec through a fresh cache over `store`, returning the handles
/// and the geometry they landed on.
fn run(store: Box<dyn CacheStore>) -> (GenCache, Vec<MeshHandle>, Vec<MeshData>) {
    let mut cache = GenCache::new(store);
    let mut library = MeshLibrary::new();
    let handles: Vec<MeshHandle> = specs()
        .iter()
        .map(|spec| cache.resolve(spec, Quality::FULL, &mut library))
        .collect();
    let meshes = handles
        .iter()
        .map(|h| library.get(*h).expect("resolved mesh is in the library").clone())
        .collect();
    (cache, handles, meshes)
}

#[test]
fn cold_warm_and_deleted_caches_all_produce_identical_geometry() {
    let dir = TempDir::new("invariant");

    // 1. COLD — nothing on disk. Everything is generated.
    let (cold_cache, cold_handles, cold_meshes) =
        run(Box::new(NativeDiskCache::new(dir.path())));
    assert_eq!(
        cold_cache.stats().generated as usize,
        specs().len(),
        "a cold cache must run every generator"
    );
    assert_eq!(cold_cache.stats().hits(), 0);

    // 2. WARM — a second process-equivalent session over the same directory.
    let (warm_cache, warm_handles, warm_meshes) =
        run(Box::new(NativeDiskCache::new(dir.path())));
    assert_eq!(
        warm_cache.stats().generated,
        0,
        "a warm cache must not run a single generator, ran {}",
        warm_cache.stats().generated
    );
    assert_eq!(warm_cache.stats().store_hits as usize, specs().len());

    // 3. DELETED MID-RUN — a warm session whose files vanish between resolves
    //    (a `cargo clean`, a cache eviction, a full disk). It has to notice and
    //    regenerate, not serve a hole.
    let mut cache = GenCache::new(Box::new(NativeDiskCache::new(dir.path())));
    let mut library = MeshLibrary::new();
    let mut deleted_handles = Vec::new();
    let mut deleted_meshes = Vec::new();
    for (i, spec) in specs().iter().enumerate() {
        if i == 1 {
            std::fs::remove_dir_all(dir.path()).expect("delete the cache mid-run");
        }
        let handle = cache.resolve(spec, Quality::FULL, &mut library);
        deleted_handles.push(handle);
        deleted_meshes.push(library.get(handle).expect("still resolvable").clone());
    }
    assert!(
        cache.stats().generated > 0,
        "the deleted half must have been regenerated"
    );

    // The verdict: three different cache histories, one set of meshes.
    assert_eq!(cold_handles, warm_handles, "content hashes must not depend on the cache");
    assert_eq!(cold_handles, deleted_handles);
    assert_eq!(cold_meshes, warm_meshes, "geometry must not depend on the cache");
    assert_eq!(cold_meshes, deleted_meshes);
}

#[test]
fn layer_a_skips_regeneration_within_a_session() {
    // The memoization layer: resolving the same spec again is a map lookup, not
    // a generator run. Terrain at 64 segments is ~8k triangles — this is the
    // layer that keeps a scene with fifty props from meshing fifty terrains.
    let mut cache = GenCache::new(Box::new(NoopCache));
    let mut library = MeshLibrary::new();
    let spec = &specs()[2];

    let first = cache.resolve(spec, Quality::FULL, &mut library);
    for _ in 0..10 {
        assert_eq!(cache.resolve(spec, Quality::FULL, &mut library), first);
    }
    assert_eq!(cache.stats().generated, 1, "one generation for eleven resolves");
    assert_eq!(cache.stats().memo_hits, 10);
    assert_eq!(library.len(), 1, "and one mesh in the library");
}

#[test]
fn two_qualities_of_one_spec_coexist() {
    // DESIGN §6: "Different quality → different content hash → coexisting LODs
    // for free." Neither may evict the other.
    let mut cache = GenCache::new(Box::new(NoopCache));
    let mut library = MeshLibrary::new();
    let spec = &specs()[0];

    let full = cache.resolve(spec, Quality::FULL, &mut library);
    let half = cache.resolve(spec, Quality(0.5), &mut library);
    assert_ne!(full, half);
    assert_eq!(library.len(), 2);
    assert!(library.contains(full) && library.contains(half));
    assert_eq!(cache.stats().generated, 2);
}

#[test]
fn a_forgotten_memo_costs_time_and_nothing_else() {
    let dir = TempDir::new("forget");
    let mut cache = GenCache::new(Box::new(NativeDiskCache::new(dir.path())));
    let mut library = MeshLibrary::new();
    let spec = &specs()[1];

    let first = cache.resolve(spec, Quality::FULL, &mut library);
    cache.clear_memo();
    assert_eq!(cache.memo_len(), 0);

    // Layer A is gone but the library still holds the geometry, so this resolves
    // through the store's key entry without deserializing anything.
    let again = cache.resolve(spec, Quality::FULL, &mut library);
    assert_eq!(first, again);
    assert_eq!(cache.stats().generated, 1, "no regeneration was needed");
    assert_eq!(library.len(), 1);
}

#[test]
fn a_corrupted_entry_is_refused_rather_than_trusted() {
    // The cache is untrusted storage. An entry that does not hash to its own
    // file name is a lie, and believing it would silently swap one object's
    // geometry for another's — the one failure mode content addressing exists
    // to make impossible.
    let dir = TempDir::new("corrupt");
    let spec = &specs()[0];

    let (_, handles, meshes) = {
        let mut cache = GenCache::new(Box::new(NativeDiskCache::new(dir.path())));
        let mut library = MeshLibrary::new();
        let h = cache.resolve(spec, Quality::FULL, &mut library);
        let m = library.get(h).expect("present").clone();
        (cache, vec![h], vec![m])
    };

    // File the *wrong* mesh under the right hash.
    let impostor = GeneratorSpec::Cube {
        size: 1.0,
        shading: Shading::Generated,
        color: None,
    }
    .generate(Quality::FULL);
    assert_ne!(MeshHandle::of(&impostor), handles[0]);
    let store = NativeDiskCache::new(dir.path());
    store.store_mesh(handles[0].0, &impostor);

    let mut cache = GenCache::new(Box::new(NativeDiskCache::new(dir.path())));
    let mut library = MeshLibrary::new();
    let handle = cache.resolve(spec, Quality::FULL, &mut library);

    assert_eq!(handle, handles[0], "the handle is still the honest one");
    assert_eq!(
        library.get(handle).expect("present"),
        &meshes[0],
        "the impostor must not have been served"
    );
    assert_eq!(cache.stats().rejected, 1);
    assert_eq!(cache.stats().generated, 1, "it fell through to regeneration");
}

#[test]
fn a_truncated_entry_is_refused_rather_than_trusted() {
    let dir = TempDir::new("truncated");
    let spec = &specs()[1];

    let expected = {
        let mut cache = GenCache::new(Box::new(NativeDiskCache::new(dir.path())));
        let mut library = MeshLibrary::new();
        cache.resolve(spec, Quality::FULL, &mut library)
    };

    // Chop every stored mesh in half.
    let mesh_dir = dir.path().join("mesh");
    for entry in std::fs::read_dir(&mesh_dir).expect("mesh dir") {
        let path = entry.expect("entry").path();
        let bytes = std::fs::read(&path).expect("read");
        std::fs::write(&path, &bytes[..bytes.len() / 2]).expect("truncate");
    }

    let mut cache = GenCache::new(Box::new(NativeDiskCache::new(dir.path())));
    let mut library = MeshLibrary::new();
    let handle = cache.resolve(spec, Quality::FULL, &mut library);
    assert_eq!(handle, expected);
    assert_eq!(cache.stats().generated, 1);
}

#[test]
fn a_stale_key_entry_costs_one_regeneration() {
    // Layer A on disk can point at a content hash layer B no longer has (an
    // eviction that only took mesh files). The resolve must fall through.
    let dir = TempDir::new("stale-key");
    let spec = &specs()[0];

    let expected = {
        let mut cache = GenCache::new(Box::new(NativeDiskCache::new(dir.path())));
        let mut library = MeshLibrary::new();
        cache.resolve(spec, Quality::FULL, &mut library)
    };
    std::fs::remove_dir_all(dir.path().join("mesh")).expect("evict layer B only");

    let mut cache = GenCache::new(Box::new(NativeDiskCache::new(dir.path())));
    let mut library = MeshLibrary::new();
    assert_eq!(cache.resolve(spec, Quality::FULL, &mut library), expected);
    assert_eq!(cache.stats().generated, 1);
    assert_eq!(cache.stats().store_hits, 0);
}

#[test]
fn the_noop_store_never_hits_and_never_lies() {
    let mut cache = GenCache::new(Box::new(NoopCache));
    let mut library = MeshLibrary::new();
    let spec = &specs()[0];
    let a = cache.resolve(spec, Quality::FULL, &mut library);

    // A second, independent session over the same (absent) storage regenerates.
    let mut cold = GenCache::new(Box::new(NoopCache));
    let mut fresh = MeshLibrary::new();
    let b = cold.resolve(spec, Quality::FULL, &mut fresh);

    assert_eq!(a, b, "no persistence, same answer — that is the invariant");
    assert_eq!(cold.stats().store_hits, 0);
    assert_eq!(cache.store_label(), "noop");
}

// ---------------------------------------------------------------------------
// Blobs: bakes the engine stores but cannot read (D13)
// ---------------------------------------------------------------------------

/// Stand-in for a game's baked level: the engine only ever sees the bytes.
fn fake_bake(tag: u8, len: usize) -> Vec<u8> {
    (0..len).map(|i| (i as u8).wrapping_mul(31).wrapping_add(tag)).collect()
}

#[test]
fn blobs_round_trip_and_survive_a_reload_byte_identical() {
    // The determinism doctrine at the blob level: what goes in comes back out,
    // and a *second session* over the same directory sees the same bytes. The
    // stronger claim — that those bytes equal what a fresh bake would produce —
    // needs the baker, so it lives port-side (shift/src/level.rs) where the
    // `Built` struct is; this is the half the engine can state on its own.
    let dir = TempDir::new("blob-roundtrip");
    let payloads = [fake_bake(7, 0), fake_bake(11, 1), fake_bake(23, 64 * 1024)];

    {
        let store = NativeDiskCache::new(dir.path());
        for (i, bytes) in payloads.iter().enumerate() {
            store.store_blob(i as u64, bytes);
            assert_eq!(store.load_blob(i as u64).as_ref(), Some(bytes), "same session");
        }
    }

    let reopened = NativeDiskCache::new(dir.path());
    for (i, bytes) in payloads.iter().enumerate() {
        assert_eq!(
            reopened.load_blob(i as u64).as_ref(),
            Some(bytes),
            "a new session must read back the same bytes"
        );
    }
    assert_eq!(reopened.load_blob(999), None, "a key nobody wrote is a miss");
}

#[test]
fn a_corrupted_blob_is_refused_rather_than_trusted() {
    // The blob half of the untrusted-storage stance. A mesh proves itself by
    // re-hashing; a blob is opaque, so the frame's checksum stands in — and it
    // has to catch all three ways an entry can be wrong.
    let dir = TempDir::new("blob-corrupt");
    let store = NativeDiskCache::new(dir.path());
    let payload = fake_bake(3, 4096);
    let path = |hash: u64| dir.path().join("blob").join(format!("{hash:016x}.blob"));

    // 1. Truncated (a torn write, a full disk).
    store.store_blob(1, &payload);
    let bytes = std::fs::read(path(1)).expect("read");
    std::fs::write(path(1), &bytes[..bytes.len() / 2]).expect("truncate");
    assert_eq!(store.load_blob(1), None, "a truncated blob must be refused");

    // 2. Flipped bit inside the payload.
    store.store_blob(2, &payload);
    let mut bytes = std::fs::read(path(2)).expect("read");
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    std::fs::write(path(2), &bytes).expect("corrupt");
    assert_eq!(store.load_blob(2), None, "a bit-rotted blob must be refused");

    // 3. Misfiled: a whole, valid entry sitting under somebody else's key. This
    //    is the one plain length/parse checks would wave through, and the one
    //    that would silently serve the wrong level.
    store.store_blob(3, &payload);
    std::fs::copy(path(3), path(4)).expect("misfile");
    assert_eq!(store.load_blob(3), Some(payload), "the honest key still reads");
    assert_eq!(store.load_blob(4), None, "the impostor key must not");
}

#[test]
fn the_user_cache_directory_is_per_app_and_refuses_a_path() {
    // Where a *shipped* program's cache goes, as opposed to `in_target`'s
    // build-directory answer. The app name becomes a directory component, so
    // anything that could climb out of it has to be refused rather than escaped.
    if let Some(store) = NativeDiskCache::in_cache_dir("runt-selftest") {
        let root = store.root();
        assert!(root.ends_with("runt-selftest/content"), "{}", root.display());
        assert!(root.is_absolute());
    } else {
        // No HOME and no XDG_CACHE_HOME: nowhere to put it, which is a legal
        // answer (the caller falls back to a store that keeps nothing).
        assert!(std::env::var_os("HOME").is_none());
    }
    for hostile in ["", "..", "../../etc", "a/b"] {
        assert!(
            NativeDiskCache::in_cache_dir(hostile).is_none(),
            "{hostile:?} must not become a directory"
        );
    }
}

#[test]
fn a_store_with_no_blob_support_is_a_miss_and_not_a_lie() {
    // Every existing store predates blobs and gets the default impls, so the
    // engine's contract has to hold for them unchanged: a write is accepted and
    // discarded, a read is a miss, and the caller re-bakes.
    let store = NoopCache;
    store.store_blob(1, &fake_bake(1, 32));
    assert_eq!(store.load_blob(1), None);
}

// ---------------------------------------------------------------------------
// MemCache: the web's synchronous face on an asynchronous database (D13)
// ---------------------------------------------------------------------------

#[test]
fn a_mem_cache_records_writes_but_not_preloads() {
    // The flush contract. Entries that came *from* storage must not be written
    // back to it — a page load would otherwise rewrite the whole database it
    // just read, every time.
    let mesh = specs()[0].generate(Quality::FULL);
    let hash = MeshHandle::of(&mesh).0;

    let cold = MemCache::new();
    assert_eq!(cold.pending_writes(), 0);
    cold.store_mesh(hash, &mesh);
    cold.store_blob(77, &fake_bake(5, 512));
    cold.store_key(9, hash);
    assert_eq!(cold.load_mesh(hash).as_ref(), Some(&mesh));
    assert_eq!(cold.load_blob(77), Some(fake_bake(5, 512)));
    assert_eq!(cold.load_key(9), Some(hash));
    assert_eq!(cold.pending_writes(), 3);

    // What the host would hand to IndexedDB.
    let written = cold.take_written();
    assert_eq!(written.len(), 3, "one entry per distinct key written");
    assert_eq!(cold.pending_writes(), 0, "draining is idempotent");
    assert!(cold.take_written().is_empty());
    assert_eq!(cold.len(), 3, "draining does not empty the cache");

    // Next page load: the same entries come back as preloads and read fine,
    // and nothing is queued to be written back.
    let warm = MemCache::preloaded(written);
    assert_eq!(warm.len(), 3);
    assert_eq!(warm.pending_writes(), 0);
    assert_eq!(warm.load_mesh(hash).as_ref(), Some(&mesh));
    assert_eq!(warm.load_blob(77), Some(fake_bake(5, 512)));
    assert_eq!(warm.load_key(9), Some(hash));
    assert!(warm.take_written().is_empty(), "a pure read flushes nothing");
}

#[test]
fn a_mem_cache_rewrite_is_queued_once() {
    let cache = MemCache::new();
    for i in 0..5 {
        cache.store_blob(1, &fake_bake(i, 16));
    }
    let written = cache.take_written();
    assert_eq!(written.len(), 1, "five writes to one key are one entry to flush");
    assert_eq!(cache.load_blob(1), Some(fake_bake(4, 16)), "the last one wins");
}

#[test]
fn clones_of_a_mem_cache_share_one_map() {
    // The host keeps a handle while the engine holds the store as a
    // `Box<dyn CacheStore>`; if those were separate maps the flush would always
    // be empty and web would never warm up.
    let host = MemCache::new();
    let engine: Box<dyn CacheStore> = Box::new(host.clone());
    engine.store_blob(1, &fake_bake(2, 8));
    assert_eq!(host.pending_writes(), 1);
    assert_eq!(host.take_written().len(), 1);
    assert_eq!(engine.load_blob(1), Some(fake_bake(2, 8)));
}

#[test]
fn a_second_web_session_generates_nothing() {
    // The whole point of D13's web half, end to end: session one runs every
    // generator and hands its writes to (a stand-in for) IndexedDB; session two
    // starts from those bytes and runs none — with identical geometry, which is
    // the invariant this file exists to defend.
    let host = MemCache::new();
    let (cold, cold_handles, cold_meshes) = run(Box::new(host.clone()));
    assert_eq!(cold.stats().generated as usize, specs().len());

    let stored = host.take_written();
    assert!(!stored.is_empty(), "a cold session must leave something behind");

    let (warm, warm_handles, warm_meshes) = run(Box::new(MemCache::preloaded(stored)));
    assert_eq!(warm.stats().generated, 0, "a warm page load must not generate");
    assert_eq!(warm.stats().store_hits as usize, specs().len());
    assert_eq!(cold_handles, warm_handles);
    assert_eq!(cold_meshes, warm_meshes);
    assert_eq!(warm.store_label(), "memory");
}

#[test]
fn a_mem_cache_refuses_a_corrupted_preload() {
    // Bytes out of a browser database are no more trustworthy than bytes off a
    // disk: same validate-on-load, same fall-through to regeneration.
    let good = MemCache::new();
    good.store_blob(42, &fake_bake(9, 256));
    let entries = good.take_written();
    assert_eq!(entries, vec![("blob/000000000000002a".to_string(), entries[0].1.clone())]);

    // Truncated in storage.
    let mut torn = entries.clone();
    torn[0].1.truncate(4); // Shorter than the frame's checksum.
    assert_eq!(MemCache::preloaded(torn).load_blob(42), None);

    // Whole, valid, and filed under a key it was not written for.
    let misfiled = vec![("blob/00000000000000ff".to_string(), entries[0].1.clone())];
    let store = MemCache::preloaded(misfiled);
    assert_eq!(store.load_blob(0xff), None, "the frame is bound to its key");
    assert_eq!(store.load_blob(42), None, "and the real key is simply absent");
}

#[test]
fn textures_never_persist_through_a_mem_cache() {
    // Not a preference: baking a texture back out of the GPU needs a blocking
    // device poll, and the platform this store exists for has none. The flag is
    // the promise, so it stays false wherever the memory store might run.
    assert!(!MemCache::new().caches_textures());
}

#[test]
fn stored_meshes_round_trip_through_postcard() {
    // Layer B's serialization is `MeshData`'s own; if it ever stopped being
    // lossless the content-hash check would turn every warm start into a cold
    // one, silently. Check it directly rather than inferring it from a miss.
    let dir = TempDir::new("roundtrip");
    let store = NativeDiskCache::new(dir.path());
    for spec in specs() {
        let mesh = spec.generate(Quality::FULL);
        let hash = MeshHandle::of(&mesh);
        store.store_mesh(hash.0, &mesh);
        let back = store.load_mesh(hash.0).expect("stored entry reads back");
        assert_eq!(back, mesh, "{}", spec.kind());
        assert_eq!(MeshHandle::of(&back), hash);
    }
}
