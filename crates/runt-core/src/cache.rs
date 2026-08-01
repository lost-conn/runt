//! The content-addressed generation cache (DESIGN §6).
//!
//! ```text
//!   GeneratorSpec + Quality
//!            │  param_key()
//!            ▼
//!   ┌─ LAYER A ──────────────┐   param_key → content_hash
//!   │  memoization           │   "this spec has already been generated"
//!   └────────────┬───────────┘
//!                ▼
//!   ┌─ LAYER B ──────────────┐   content_hash → MeshData
//!   │  MeshLibrary (memory)  │   the resource the renderer uploads from
//!   │  CacheStore (persist)  │   disk natively, nothing on web (v1)
//!   └────────────────────────┘
//! ```
//!
//! **The invariant that governs this whole module:** the cache is *purely* an
//! optimization. Deleting it, corrupting it, or never writing it must not change
//! a single vertex — only how long generation takes. Determinism is what makes
//! that checkable, and `tests/cache.rs` checks it: cold, warm, and
//! deleted-mid-run all have to produce identical content hashes and identical
//! `MeshData`.
//!
//! Everything that could make the cache authoritative is therefore refused:
//! a deserialized mesh is re-hashed and dropped if it does not match the key it
//! was filed under, every I/O error degrades to "regenerate" with a warning, and
//! no code path can observe a cache hit except through the same `MeshHandle` a
//! cold generation would have produced.
//!
//! ## Layer A persistence
//!
//! §6 describes the persistent layer as `hash → serialized MeshData`. Taken
//! literally that layer can never be *hit*, because reaching it needs a content
//! hash you can only get by generating the mesh. So [`CacheStore`] persists the
//! layer-A mapping as well ([`load_key`](CacheStore::load_key) /
//! [`store_key`](CacheStore::store_key)) — a param key pointing at a content
//! hash. It stays a strict optimization: a missing, stale or wrong key entry
//! costs one regeneration and nothing else.
//!
//! ## Known deviation from §6
//!
//! §6 wants IndexedDB on web. IndexedDB is async top to bottom, which means the
//! scene loader would have to become async before anything else could use it;
//! that lands with phase 2's worker story (§13's open question). Web therefore
//! gets [`NoopCache`] in v1 — meshes are regenerated per page load, which is
//! exactly the "purely an optimization" path the tests already cover.

use std::collections::HashMap;

use bevy_ecs::prelude::Resource;
use runt_mesh::{MeshData, Quality};

use crate::gen::GeneratorSpec;
use crate::registry::{MeshHandle, MeshLibrary};

/// Persistence for the cache's layers. Implementations are best-effort by
/// contract: every method may silently do nothing, and the engine still works.
///
/// `Send + Sync` so a `GenCache` stays a legal ECS resource (and so the
/// generation worker of §6 can share one later).
pub trait CacheStore: Send + Sync + 'static {
    /// Layer B read. Returning `None` — including on any I/O or decode error —
    /// means "regenerate", which is always correct.
    fn load_mesh(&self, content_hash: u64) -> Option<MeshData>;

    /// Layer B write. Errors are the store's problem, not the caller's.
    fn store_mesh(&self, content_hash: u64, mesh: &MeshData);

    /// Layer A read: which content hash a param key produced last time.
    fn load_key(&self, param_key: u64) -> Option<u64> {
        let _ = param_key;
        None
    }

    /// Layer A write.
    fn store_key(&self, param_key: u64, content_hash: u64) {
        let _ = (param_key, content_hash);
    }

    /// Name for logs.
    fn label(&self) -> &'static str {
        "cache"
    }
}

/// A store that stores nothing. The web backend in v1, and the default
/// everywhere so that constructing a [`Sim`](crate::Sim) never touches a disk.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopCache;

impl CacheStore for NoopCache {
    fn load_mesh(&self, _content_hash: u64) -> Option<MeshData> {
        None
    }
    fn store_mesh(&self, _content_hash: u64, _mesh: &MeshData) {}
    fn label(&self) -> &'static str {
        "noop"
    }
}

// ---------------------------------------------------------------------------
// Native disk store
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
pub use native::NativeDiskCache;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::path::{Path, PathBuf};

    use runt_mesh::MeshData;

    use super::CacheStore;

    /// Files under a directory, one per entry, named by hash in hex.
    ///
    /// Deliberately dependency-light: no `dirs`, no database, no index file to
    /// keep consistent. Content addressing does the hard part — every file's
    /// name determines its contents, so two processes writing the same entry
    /// write the same bytes and concurrent runs cannot disagree.
    ///
    /// ```text
    /// <root>/mesh/<content hash>.postcard   layer B
    /// <root>/key/<param key>.hash           layer A
    /// ```
    pub struct NativeDiskCache {
        root: PathBuf,
    }

    impl NativeDiskCache {
        /// A cache rooted at `root`, created on demand.
        pub fn new(root: impl Into<PathBuf>) -> NativeDiskCache {
            NativeDiskCache { root: root.into() }
        }

        /// The workspace's `target/runt-cache`.
        ///
        /// `CARGO_MANIFEST_DIR` is resolved at *compile* time, so a binary moved
        /// off the build machine will point at a path that does not exist —
        /// which the store handles the same way it handles every other I/O
        /// failure: warn once per operation and regenerate. Shipping builds get
        /// a real path when there is a shipping story to hang it on.
        pub fn in_target() -> NativeDiskCache {
            NativeDiskCache::new(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../target/runt-cache"
            ))
        }

        pub fn root(&self) -> &Path {
            &self.root
        }

        fn mesh_path(&self, content_hash: u64) -> PathBuf {
            self.root.join("mesh").join(format!("{content_hash:016x}.postcard"))
        }

        fn key_path(&self, param_key: u64) -> PathBuf {
            self.root.join("key").join(format!("{param_key:016x}.hash"))
        }

        /// Write via a unique temp file plus a rename, so a reader never sees a
        /// half-written entry and two processes racing on the same hash cannot
        /// interleave bytes.
        fn write_atomic(path: &Path, bytes: &[u8]) {
            let Some(dir) = path.parent() else { return };
            if let Err(e) = std::fs::create_dir_all(dir) {
                log::warn!("runt-cache: cannot create {}: {e}", dir.display());
                return;
            }
            let tmp = path.with_extension(format!("tmp{}", std::process::id()));
            if let Err(e) = std::fs::write(&tmp, bytes) {
                log::warn!("runt-cache: cannot write {}: {e}", tmp.display());
                return;
            }
            if let Err(e) = std::fs::rename(&tmp, path) {
                log::warn!("runt-cache: cannot publish {}: {e}", path.display());
                let _ = std::fs::remove_file(&tmp);
            }
        }
    }

    impl CacheStore for NativeDiskCache {
        fn load_mesh(&self, content_hash: u64) -> Option<MeshData> {
            let bytes = std::fs::read(self.mesh_path(content_hash)).ok()?;
            match postcard::from_bytes::<MeshData>(&bytes) {
                Ok(mesh) => Some(mesh),
                Err(e) => {
                    log::warn!("runt-cache: {content_hash:016x} failed to decode ({e}); regenerating");
                    None
                }
            }
        }

        fn store_mesh(&self, content_hash: u64, mesh: &MeshData) {
            match postcard::to_stdvec(mesh) {
                Ok(bytes) => NativeDiskCache::write_atomic(&self.mesh_path(content_hash), &bytes),
                Err(e) => log::warn!("runt-cache: {content_hash:016x} failed to encode ({e})"),
            }
        }

        fn load_key(&self, param_key: u64) -> Option<u64> {
            let text = std::fs::read_to_string(self.key_path(param_key)).ok()?;
            u64::from_str_radix(text.trim(), 16).ok()
        }

        fn store_key(&self, param_key: u64, content_hash: u64) {
            NativeDiskCache::write_atomic(
                &self.key_path(param_key),
                format!("{content_hash:016x}").as_bytes(),
            );
        }

        fn label(&self) -> &'static str {
            "disk"
        }
    }
}

/// The platform's default persistent store: disk natively, nothing on web.
///
/// Hosts opt in; [`Sim::new`](crate::Sim::new) does not, so a test never touches
/// the filesystem unless it asked to.
pub fn platform_default() -> Box<dyn CacheStore> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        Box::new(NativeDiskCache::in_target())
    }
    #[cfg(target_arch = "wasm32")]
    {
        Box::new(NoopCache)
    }
}

// ---------------------------------------------------------------------------
// The cache itself
// ---------------------------------------------------------------------------

/// Where a resolved mesh came from. Only observable through
/// [`GenCache::stats`] — no engine behaviour may depend on it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Generators actually run.
    pub generated: u32,
    /// Layer A hits: the spec was already resolved this session.
    pub memo_hits: u32,
    /// Layer B hits: the mesh was deserialized from the store.
    pub store_hits: u32,
    /// Store entries thrown away because they did not hash to their own key.
    pub rejected: u32,
}

impl CacheStats {
    /// Resolutions served without running a generator.
    pub fn hits(&self) -> u32 {
        self.memo_hits + self.store_hits
    }
}

/// Layer A plus the handle to layer B's persistence, as a world resource.
///
/// Layer B's *memory* half is [`MeshLibrary`] — deliberately a separate resource,
/// because the renderer reads it every frame and has no business knowing a cache
/// exists.
#[derive(Resource)]
pub struct GenCache {
    /// param_key → content hash.
    memo: HashMap<u64, MeshHandle>,
    store: Box<dyn CacheStore>,
    stats: CacheStats,
}

impl Default for GenCache {
    fn default() -> GenCache {
        GenCache::new(Box::new(NoopCache))
    }
}

impl GenCache {
    pub fn new(store: Box<dyn CacheStore>) -> GenCache {
        GenCache {
            memo: HashMap::new(),
            store,
            stats: CacheStats::default(),
        }
    }

    /// A cache backed by the platform's persistent store.
    pub fn persistent() -> GenCache {
        GenCache::new(platform_default())
    }

    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    pub fn store_label(&self) -> &'static str {
        self.store.label()
    }

    /// Distinct param keys resolved this session.
    pub fn memo_len(&self) -> usize {
        self.memo.len()
    }

    /// Drop layer A. The next resolve re-checks the store and, failing that,
    /// regenerates — which is the point: forgetting must be free of consequence.
    pub fn clear_memo(&mut self) {
        self.memo.clear();
    }

    /// Resolve a spec to geometry, filling `library` if it is not already there.
    ///
    /// The one entry point. Order of attempts:
    ///
    /// 1. **Layer A, memory.** Param key already resolved *and* the library
    ///    still holds that mesh → return the handle, generate nothing.
    /// 2. **Layer A, store.** Param key on disk → the content hash it names →
    ///    layer B read → validate → insert.
    /// 3. **Cold.** Run the generator, insert, and write both layers back.
    ///
    /// Every path returns the same handle for the same input, so a caller cannot
    /// tell which one ran except by reading [`stats`](GenCache::stats).
    pub fn resolve(
        &mut self,
        spec: &GeneratorSpec,
        quality: Quality,
        library: &mut MeshLibrary,
    ) -> MeshHandle {
        let key = spec.param_key(quality);

        // 1. Layer A, in memory.
        if let Some(&handle) = self.memo.get(&key) {
            if library.contains(handle) {
                self.stats.memo_hits += 1;
                return handle;
            }
            // Memoized, but the library no longer has the geometry (a scene
            // reload, say). The hash is still valid, so try layer B for it.
            if let Some(handle) = self.hydrate(handle, library) {
                self.stats.store_hits += 1;
                return handle;
            }
        } else if let Some(content_hash) = self.store.load_key(key) {
            // 2. Layer A, from the store.
            let handle = MeshHandle(content_hash);
            if library.contains(handle) {
                self.memo.insert(key, handle);
                self.stats.memo_hits += 1;
                return handle;
            }
            if let Some(handle) = self.hydrate(handle, library) {
                self.memo.insert(key, handle);
                self.stats.store_hits += 1;
                return handle;
            }
        }

        // 3. Cold: run the generator.
        let mesh = spec.generate(quality);
        mesh.validate();
        let handle = library.insert(mesh);
        self.stats.generated += 1;
        self.memo.insert(key, handle);
        self.store.store_key(key, handle.0);
        if let Some(mesh) = library.get(handle) {
            self.store.store_mesh(handle.0, mesh);
        }
        handle
    }

    /// Pull one layer-B entry into the library, refusing anything that does not
    /// hash to the key it was filed under.
    ///
    /// This check is what lets the cache be untrusted storage: a truncated file,
    /// a stale entry from an older `MeshData` layout, or an outright hostile one
    /// all fail here and fall through to regeneration.
    fn hydrate(&mut self, handle: MeshHandle, library: &mut MeshLibrary) -> Option<MeshHandle> {
        let mesh = self.store.load_mesh(handle.0)?;
        let actual = MeshHandle::of(&mesh);
        if actual != handle {
            log::warn!(
                "runt-cache: entry {:#018x} hashes to {:#018x}; discarding and regenerating",
                handle.0,
                actual.0
            );
            self.stats.rejected += 1;
            return None;
        }
        Some(library.insert(mesh))
    }
}
