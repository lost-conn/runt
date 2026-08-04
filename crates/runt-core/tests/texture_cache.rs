//! The baked-texture cache (DESIGN §7, under §6's rule).
//!
//! Same headline claim as `cache.rs`, one pipeline stage down: **the cache is
//! purely an optimization**. A cold bake, a warm cache and a cache deleted out
//! from under a live session must all produce byte-identical pixels — otherwise
//! "content-addressed" is decoration and a user's texture depends on what is on
//! their disk.
//!
//! The rest is the same claim from other angles: a corrupt entry is refused, an
//! entry filed under the right key but baked from the wrong spec is refused,
//! and no hit is observable except by counting store traffic.

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

/// Bake through `store` on a fresh renderer and read both maps back.
fn bake_through(store: &dyn CacheStore) -> Option<(Vec<u8>, Vec<u8>)> {
    let mut renderer = renderer()?;
    let spec = spec();
    let handle = renderer.bake_texture(&spec, RES, store);
    let gpu = renderer.textures().get(handle).expect("resident");
    let albedo =
        runt_core::bake::read_target(renderer.device(), renderer.queue(), &gpu.albedo, RES)?;
    let normal =
        runt_core::bake::read_target(renderer.device(), renderer.queue(), &gpu.normal, RES)?;
    Some((albedo, normal))
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
            resolution: RES,
            spec: postcard::to_stdvec(&wrong).expect("encode"),
            albedo: vec![7u8; (RES * RES * 4) as usize],
            normal: vec![7u8; (RES * RES * 4) as usize],
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
    store.store_texture(
        handle.0,
        &TextureData {
            resolution: RES,
            spec: postcard::to_stdvec(&spec).expect("encode"),
            albedo: vec![3u8; (RES * RES * 2) as usize],
            normal: vec![3u8; (RES * RES * 4) as usize],
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
            resolution: 1,
            spec: Vec::new(),
            albedo: vec![0; 4],
            normal: vec![0; 4],
        },
    );
}

#[test]
fn the_disk_store_round_trips_an_entry() {
    // No GPU needed: this is the serialization contract on its own.
    let dir = TempDir::new("roundtrip");
    let disk = NativeDiskCache::new(dir.path());
    assert!(disk.caches_textures());

    let data = TextureData {
        resolution: 4,
        spec: postcard::to_stdvec(&spec()).expect("encode"),
        albedo: (0..64).map(|i| i as u8).collect(),
        normal: (0..64).map(|i| (255 - i) as u8).collect(),
    };
    disk.store_texture(0xabc_def, &data);
    let back = disk.load_texture(0xabc_def).expect("round trip");
    assert_eq!(back, data);
    assert!(back.matches(&spec(), 4));
    assert!(!back.matches(&spec(), 8), "resolution is part of the match");
    assert!(
        !back.matches(&texture::rock(), 4),
        "the spec is part of the match"
    );

    assert!(disk.load_texture(0x999).is_none(), "a miss is a miss");
}
