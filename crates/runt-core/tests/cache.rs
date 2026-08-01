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
use runt_core::cache::{CacheStore, GenCache, NativeDiskCache, NoopCache};
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
