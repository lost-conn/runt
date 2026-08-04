//! The baked-texture cache (DESIGN §7, under §6's rule).
//!
//! Same headline claim as `cache.rs`, one pipeline stage down: **the cache is
//! purely an optimization**. A cold bake, a warm cache and a cache deleted out
//! from under a live session must all produce byte-identical pixels — otherwise
//! "content-addressed" is decoration and a user's texture depends on what is on
//! their disk.
//!
//! The rest is the same claim from other angles: a corrupt entry is refused, an
//! entry filed under the right key but baked from the wrong spec is refused, an
//! entry written by an older engine is refused, and no hit is observable except
//! by counting store traffic.
//!
//! "Byte-identical pixels" means the whole **mip chain**, not level 0. A warm
//! start that rebuilt the chain itself would be a second implementation of the
//! downsample filter and a place for the two to disagree, so the store holds
//! every level and the tests compare every level.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use runt_core::bake::TextureData;
use runt_core::cache::{CacheStore, NativeDiskCache, NoopCache};
use runt_core::texture::{self, TextureSpec};
use runt_core::Renderer;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// Small enough that a handful of bakes and readbacks stay quick, large enough
/// that "the pixels are identical" is a real claim.
const RES: u32 = 128;

fn renderer() -> Option<Renderer> {
    match pollster::block_on(Renderer::headless(FORMAT)) {
        Ok(r) => Some(r),
        Err(e) => {
            eprintln!("SKIP (no GPU adapter): {e}");
            None
        }
    }
}

fn spec() -> TextureSpec {
    TextureSpec {
        base_resolution: RES,
        ..texture::grass()
    }
}

/// Bake through `store` on a fresh renderer and read both maps back — every
/// mip level of both, because every level is part of what a bake *is*.
#[allow(clippy::type_complexity)]
fn bake_through(store: &dyn CacheStore) -> Option<(Vec<Vec<u8>>, Vec<Vec<u8>>)> {
    let mut renderer = renderer()?;
    let spec = spec();
    let handle = renderer.bake_texture(&spec, RES, store);
    let gpu = renderer.textures().get(handle).expect("resident");
    let albedo =
        runt_core::bake::read_chain(renderer.device(), renderer.queue(), &gpu.albedo, RES)?;
    let normal =
        runt_core::bake::read_chain(renderer.device(), renderer.queue(), &gpu.normal, RES)?;
    Some((albedo, normal))
}

/// A mip chain of the right shape for `resolution`, every texel `fill`. The
/// shape a store entry has to have before its *contents* are even looked at.
fn filled_chain(resolution: u32, fill: u8) -> Vec<Vec<u8>> {
    (0..runt_core::bake::mip_level_count(resolution))
        .map(|level| {
            let size = runt_core::bake::mip_size(resolution, level) as usize;
            vec![fill; size * size * 4]
        })
        .collect()
}

/// A scratch directory unique to this binary and test, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> TempDir {
        let dir =
            std::env::temp_dir().join(format!("runt-texcache-test-{}-{tag}", std::process::id()));
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

/// An in-memory store that counts what it is asked for, so a hit can be
/// *observed* — which is the only way anything is allowed to observe one.
#[derive(Default)]
struct Recording {
    entries: Mutex<Vec<(u64, TextureData)>>,
    loads: AtomicUsize,
    hits: AtomicUsize,
    stores: AtomicUsize,
}

impl CacheStore for Recording {
    fn load_mesh(&self, _content_hash: u64) -> Option<runt_core::Mesh> {
        None
    }
    fn store_mesh(&self, _content_hash: u64, _mesh: &runt_core::Mesh) {}

    fn load_texture(&self, content_key: u64) -> Option<TextureData> {
        self.loads.fetch_add(1, Ordering::Relaxed);
        let found = self
            .entries
            .lock()
            .expect("lock")
            .iter()
            .find(|(k, _)| *k == content_key)
            .map(|(_, d)| d.clone());
        if found.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        }
        found
    }

    fn store_texture(&self, content_key: u64, data: &TextureData) {
        self.stores.fetch_add(1, Ordering::Relaxed);
        self.entries
            .lock()
            .expect("lock")
            .push((content_key, data.clone()));
    }

    fn caches_textures(&self) -> bool {
        true
    }

    fn label(&self) -> &'static str {
        "recording"
    }
}

#[test]
fn cold_warm_and_deleted_all_bake_the_same_pixels() {
    let Some((cold_albedo, cold_normal)) = bake_through(&NoopCache) else {
        return;
    };

    let dir = TempDir::new("invariant");
    let disk = NativeDiskCache::new(dir.path());

    // Cold through a real disk store: bakes, and writes the pixels back.
    let (a1, n1) = bake_through(&disk).expect("adapter");
    assert_eq!(a1, cold_albedo, "a store must not change what is baked");
    assert_eq!(n1, cold_normal);
    assert_eq!(
        cold_albedo.len(),
        runt_core::bake::mip_level_count(RES) as usize,
        "the invariant below only covers the levels that are read back"
    );
    assert!(
        dir.path().join("texture").exists(),
        "the disk store wrote nothing, so the warm case below proves nothing"
    );

    // Warm: served out of the store, on a brand-new renderer that has never
    // baked anything.
    let (a2, n2) = bake_through(&disk).expect("adapter");
    assert_eq!(a2, cold_albedo, "a cache hit produced different pixels");
    assert_eq!(n2, cold_normal);

    // Deleted mid-run: the store is gone, every read fails, and the output is
    // still identical — only slower.
    std::fs::remove_dir_all(dir.path().join("texture")).expect("delete the cache");
    let (a3, n3) = bake_through(&disk).expect("adapter");
    assert_eq!(a3, cold_albedo, "deleting the cache changed the output");
    assert_eq!(n3, cold_normal);
}

#[test]
fn a_hit_is_a_hit_and_it_skips_the_bake() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let store = Recording::default();
    let spec = spec();

    renderer.bake_texture(&spec, RES, &store);
    assert_eq!(store.loads.load(Ordering::Relaxed), 1);
    assert_eq!(store.hits.load(Ordering::Relaxed), 0, "nothing to hit yet");
    assert_eq!(store.stores.load(Ordering::Relaxed), 1, "the bake was filed");

    // Same renderer, same spec: layer one answers, the store is not touched.
    renderer.bake_texture(&spec, RES, &store);
    assert_eq!(store.loads.load(Ordering::Relaxed), 1, "memory should answer");

    // A fresh renderer has no memory, so this one goes to the store — and must
    // not bake, which shows up as no second store write.
    let Some(mut other) = self::renderer() else {
        return;
    };
    other.bake_texture(&spec, RES, &store);
    assert_eq!(store.loads.load(Ordering::Relaxed), 2);
    assert_eq!(store.hits.load(Ordering::Relaxed), 1);
    assert_eq!(
        store.stores.load(Ordering::Relaxed),
        1,
        "a hit re-baked and re-filed, so it was not a hit"
    );
}

#[test]
fn an_entry_that_does_not_match_its_spec_is_refused() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let store = Recording::default();
    let spec = spec();

    // File *rock* under *grass*'s key: the right size, the wrong material. A
    // store is untrusted input, and a key collision must not repaint terrain.
    let wrong = TextureSpec {
        base_resolution: RES,
        ..texture::rock()
    };
    let handle = runt_core::TextureHandle(spec.content_key(RES));
    store.store_texture(
        handle.0,
        &TextureData {
            version: runt_core::bake::TEXTURE_DATA_VERSION,
            resolution: RES,
            spec: postcard::to_stdvec(&wrong).expect("encode"),
            albedo: filled_chain(RES, 7),
            normal: filled_chain(RES, 7),
        },
    );

    let baked = renderer.bake_texture(&spec, RES, &store);
    assert_eq!(baked, handle);
    let gpu = renderer.textures().get(baked).expect("resident");
    let albedo =
        runt_core::bake::read_target(renderer.device(), renderer.queue(), &gpu.albedo, RES)
            .expect("readback");
    assert!(
        albedo.iter().any(|b| *b != 7),
        "the mismatched entry was served instead of being rebaked"
    );

    // And the real bake replaced it, so the next run is correct rather than
    // permanently poisoned.
    assert_eq!(store.stores.load(Ordering::Relaxed), 2);
}

#[test]
fn a_truncated_or_wrong_sized_entry_is_refused() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let store = Recording::default();
    let spec = spec();
    let handle = runt_core::TextureHandle(spec.content_key(RES));

    // Right spec, right key, half the pixels — what a partial write looks like.
    let mut truncated = filled_chain(RES, 3);
    truncated[0].truncate((RES * RES * 2) as usize);
    store.store_texture(
        handle.0,
        &TextureData {
            version: runt_core::bake::TEXTURE_DATA_VERSION,
            resolution: RES,
            spec: postcard::to_stdvec(&spec).expect("encode"),
            albedo: truncated,
            normal: filled_chain(RES, 3),
        },
    );

    renderer.bake_texture(&spec, RES, &store);
    assert_eq!(
        store.stores.load(Ordering::Relaxed),
        2,
        "the truncated entry was accepted"
    );
}

#[test]
fn an_entry_missing_mip_levels_is_refused() {
    // The failure mode the version bump exists for, reached the other way: a
    // level-0-only entry is exactly what the pre-mip engine wrote. Serving it
    // would bind a texture whose upper levels were never written — undefined
    // memory, sampled the moment the camera backs away from the surface.
    let Some(mut renderer) = renderer() else {
        return;
    };
    let store = Recording::default();
    let spec = spec();
    let handle = runt_core::TextureHandle(spec.content_key(RES));

    store.store_texture(
        handle.0,
        &TextureData {
            version: runt_core::bake::TEXTURE_DATA_VERSION,
            resolution: RES,
            spec: postcard::to_stdvec(&spec).expect("encode"),
            albedo: vec![vec![3u8; (RES * RES * 4) as usize]],
            normal: vec![vec![3u8; (RES * RES * 4) as usize]],
        },
    );

    renderer.bake_texture(&spec, RES, &store);
    assert_eq!(
        store.stores.load(Ordering::Relaxed),
        2,
        "a chain with one level was accepted as a full chain"
    );
}

#[test]
fn an_entry_from_an_older_format_is_refused() {
    // Belt and braces over the length checks: an entry whose *shape* is right
    // for this version but whose stamp says otherwise is still refused, so a
    // future layout change has a lever that does not depend on the sizes
    // happening to differ.
    let Some(mut renderer) = renderer() else {
        return;
    };
    let store = Recording::default();
    let spec = spec();
    let handle = runt_core::TextureHandle(spec.content_key(RES));

    let stale = TextureData {
        version: runt_core::bake::TEXTURE_DATA_VERSION - 1,
        resolution: RES,
        spec: postcard::to_stdvec(&spec).expect("encode"),
        albedo: filled_chain(RES, 9),
        normal: filled_chain(RES, 9),
    };
    assert!(
        !stale.matches(&spec, RES),
        "a stale version passed the match it exists to fail"
    );
    store.store_texture(handle.0, &stale);

    renderer.bake_texture(&spec, RES, &store);
    assert_eq!(
        store.stores.load(Ordering::Relaxed),
        2,
        "the stale-version entry was accepted"
    );
    let gpu = renderer.textures().get(handle).expect("resident");
    let albedo =
        runt_core::bake::read_target(renderer.device(), renderer.queue(), &gpu.albedo, RES)
            .expect("readback");
    assert!(albedo.iter().any(|b| *b != 9), "the stale entry was served");
}

#[test]
fn garbage_on_disk_degrades_to_a_rebake() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let dir = TempDir::new("garbage");
    let disk = NativeDiskCache::new(dir.path());
    let spec = spec();
    let handle = runt_core::TextureHandle(spec.content_key(RES));

    std::fs::create_dir_all(dir.path().join("texture")).expect("mkdir");
    std::fs::write(
        dir.path()
            .join("texture")
            .join(format!("{:016x}.postcard", handle.0)),
        b"this is not postcard",
    )
    .expect("write garbage");

    // Must not panic, must not serve garbage: an unreadable entry is exactly a
    // missing one.
    let baked = renderer.bake_texture(&spec, RES, &disk);
    assert_eq!(baked, handle);
    assert!(renderer.textures().contains(baked));
}

#[test]
fn the_default_store_persists_nothing_and_says_so() {
    // The web path (DESIGN §7): reading a bake back needs a blocking device
    // poll, which a browser does not have, so the default store must decline
    // rather than be attempted and fail.
    assert!(!NoopCache.caches_textures());
    assert!(NoopCache.load_texture(1234).is_none());
    // Storing into it is a no-op, not a panic.
    NoopCache.store_texture(
        1234,
        &TextureData {
            version: runt_core::bake::TEXTURE_DATA_VERSION,
            resolution: 1,
            spec: Vec::new(),
            albedo: vec![vec![0; 4]],
            normal: vec![vec![0; 4]],
        },
    );
}

#[test]
fn the_disk_store_round_trips_an_entry() {
    // No GPU needed: this is the serialization contract on its own.
    let dir = TempDir::new("roundtrip");
    let disk = NativeDiskCache::new(dir.path());
    assert!(disk.caches_textures());

    // 4x4 carries three levels: 4, 2, 1. A chain that round-trips is the
    // serialization contract; a chain that survives `matches` is the loader's.
    let data = TextureData {
        version: runt_core::bake::TEXTURE_DATA_VERSION,
        resolution: 4,
        spec: postcard::to_stdvec(&spec()).expect("encode"),
        albedo: vec![
            (0..64).map(|i| i as u8).collect(),
            (0..16).map(|i| i as u8).collect(),
            (0..4).map(|i| i as u8).collect(),
        ],
        normal: vec![
            (0..64).map(|i| (255 - i) as u8).collect(),
            (0..16).map(|i| (255 - i) as u8).collect(),
            (0..4).map(|i| (255 - i) as u8).collect(),
        ],
    };
    disk.store_texture(0xabc_def, &data);
    let back = disk.load_texture(0xabc_def).expect("round trip");
    assert_eq!(back, data);
    assert!(back.matches(&spec(), 4));
    assert!(!back.matches(&spec(), 8), "resolution is part of the match");
    assert_eq!(back.albedo.len(), 3, "4x4 is three levels");
    assert!(
        !back.matches(&texture::rock(), 4),
        "the spec is part of the match"
    );

    assert!(disk.load_texture(0x999).is_none(), "a miss is a miss");
}
