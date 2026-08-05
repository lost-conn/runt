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
//!   │  CacheStore (persist)  │   a directory natively, IndexedDB on web
//!   └────────────────────────┘
//!                ▲
//!                └── same store, same rules: hash → opaque bytes, for bakes
//!                    the engine cannot type (a level). See "Blobs" below.
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
//! ## Blobs: cached bakes the engine cannot read
//!
//! [`load_blob`](CacheStore::load_blob) / [`store_blob`](CacheStore::store_blob)
//! are the same layer-B story for bytes the *engine* has no type for. The case
//! that motivated them is a level bake: a game turns its level file into meshes
//! plus a collision trimesh plus a spawn table, which is minutes of CSG on a
//! phone and one `postcard` buffer afterwards. That buffer is game-defined —
//! putting its shape in the engine would be the engine learning what a level is.
//!
//! So the engine holds `u64 → Vec<u8>` and knows nothing else, and the *key* is
//! what carries the correctness. A game hashes everything the bytes depend on:
//!
//! ```text
//! hash = fnv64( level RON bytes
//!             ‖ quality bits (Quality(f32).to_bits())
//!             ‖ runt_mesh::MESH_PIPELINE_VERSION   // engine geometry changed
//!             ‖ LEVEL_BAKE_VERSION )               // the game's own bake changed
//! value = postcard::to_stdvec(&Built { meshes, trimesh, spawns })
//! ```
//!
//! Miss the salt and a stale bake outlives the code that produced it; include
//! it and every mesh-pipeline bump invalidates every level for free. The store
//! still refuses to be authoritative: entries are checksummed against their own
//! key and dropped on any mismatch ([`load_blob`](CacheStore::load_blob)), so
//! the worst a corrupt cache can cost is one re-bake.
//!
//! ## Web (§6's IndexedDB, as of phase 2)
//!
//! IndexedDB is async top to bottom and this trait is sync, which is why web
//! got [`NoopCache`] in v1. The resolution is not an async trait: the *host*
//! reads the whole database into a [`MemCache`] before it builds the [`Sim`]
//! (its wasm init is already a future), hands that in as the store, and writes
//! new entries back out after a frame. `runt-app`'s `cache` module is the seam;
//! the engine still sees one synchronous [`CacheStore`] and never learns what
//! platform it is on.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

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

    /// Opaque bytes, content-addressed exactly like a mesh but with a payload
    /// the engine never inspects — a baked level, in practice (see the module
    /// docs for the key recipe a game owes this).
    ///
    /// Implementations **must** validate before returning: the bytes are stored
    /// framed with a checksum over `(hash ‖ payload)` (see
    /// [`frame_blob`]/[`unframe_blob`]), so a truncated, bit-rotted or
    /// misfiled entry reads back as `None` rather than as somebody else's level.
    /// `None` means "re-bake", which is always correct.
    fn load_blob(&self, hash: u64) -> Option<Vec<u8>> {
        let _ = hash;
        None
    }

    /// Blob write. Best-effort like every other write here.
    fn store_blob(&self, hash: u64, bytes: &[u8]) {
        let _ = (hash, bytes);
    }

    /// Layer A read: which content hash a param key produced last time.
    fn load_key(&self, param_key: u64) -> Option<u64> {
        let _ = param_key;
        None
    }

    /// Layer A write.
    fn store_key(&self, param_key: u64, content_hash: u64) {
        let _ = (param_key, content_hash);
    }

    /// Baked texture pixels (DESIGN §7), keyed by
    /// [`TextureSpec::content_key`](crate::texture::TextureSpec::content_key).
    ///
    /// Same contract as the mesh half: `None` means "rebake", which is always
    /// correct. The returned entry is *untrusted* — the caller re-checks it
    /// against the spec it actually asked for (see
    /// [`TextureData::matches`](crate::bake::TextureData::matches)).
    fn load_texture(&self, content_key: u64) -> Option<crate::bake::TextureData> {
        let _ = content_key;
        None
    }

    fn store_texture(&self, content_key: u64, data: &crate::bake::TextureData) {
        let _ = (content_key, data);
    }

    /// Whether this store can hold baked textures.
    ///
    /// Persisting a bake means reading the target back off the GPU, which needs
    /// a **blocking** device poll — something a browser does not have. So this
    /// is not just "would you like to?": a store that answers `true` is
    /// promising that a blocking readback is legal where it runs. Default
    /// `false`, which is what keeps the web path from ever attempting it.
    fn caches_textures(&self) -> bool {
        false
    }

    /// Name for logs.
    fn label(&self) -> &'static str {
        "cache"
    }
}

// ---------------------------------------------------------------------------
// Blob framing
// ---------------------------------------------------------------------------

/// FNV-1a, 64-bit. Small, dependency-free, and *not* a security primitive: it
/// exists so a store can tell a torn write from a whole one.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    fnv1a64_from(0xcbf2_9ce4_8422_2325, bytes)
}

/// [`fnv1a64`] continued from an earlier state, so a key and a megabyte of
/// payload can be hashed as one stream without concatenating them first.
pub fn fnv1a64_from(seed: u64, bytes: &[u8]) -> u64 {
    let mut h = seed;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Wrap `bytes` for storage under `hash`: an 8-byte little-endian checksum of
/// `hash` followed by the payload, then the payload.
///
/// Layer B validates a mesh by re-hashing it, which is only possible because
/// the engine knows what a `MeshData` is. It does not know what a blob is, so
/// the equivalent guarantee is bought with a checksum instead: the frame binds
/// the payload to the key it was filed under, which is exactly the property the
/// mesh re-hash provides.
pub fn frame_blob(hash: u64, bytes: &[u8]) -> Vec<u8> {
    let mut framed = Vec::with_capacity(8 + bytes.len());
    framed.extend_from_slice(&blob_checksum(hash, bytes).to_le_bytes());
    framed.extend_from_slice(bytes);
    framed
}

/// The payload inside a [`frame_blob`] frame, or `None` if it is short, torn or
/// filed under a different key.
pub fn unframe_blob(hash: u64, framed: &[u8]) -> Option<&[u8]> {
    if framed.len() < 8 {
        return None;
    }
    let (head, payload) = framed.split_at(8);
    let stored = u64::from_le_bytes(head.try_into().ok()?);
    (stored == blob_checksum(hash, payload)).then_some(payload)
}

fn blob_checksum(hash: u64, payload: &[u8]) -> u64 {
    fnv1a64_from(fnv1a64(&hash.to_le_bytes()), payload)
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
    /// <root>/texture/<content key>.postcard baked pixels (DESIGN §7)
    /// <root>/blob/<hash>.blob               game-defined bakes, checksum-framed
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

        /// `$XDG_CACHE_HOME/<app>/content`, falling back to `$HOME/.cache`.
        ///
        /// What a *shipped* program should use, where [`in_target`] is what a
        /// program built in this workspace gets: a cache belongs in the user's
        /// cache directory, not next to the build artifacts, and it is named
        /// per app so two games cannot collide on a param key. Same
        /// dependency-light XDG resolution as `runt_app::storage`, and `None`
        /// when the environment offers nowhere to write — the caller then falls
        /// back to a store that keeps nothing, which costs time and nothing
        /// else.
        pub fn in_cache_dir(app: &str) -> Option<NativeDiskCache> {
            if app.is_empty() || app.contains('/') || app.contains("..") {
                log::warn!("runt-cache: refusing unsafe app name {app:?}");
                return None;
            }
            let base = match std::env::var_os("XDG_CACHE_HOME") {
                Some(dir) if !dir.is_empty() => PathBuf::from(dir),
                _ => PathBuf::from(std::env::var_os("HOME")?).join(".cache"),
            };
            Some(NativeDiskCache::new(base.join(app).join("content")))
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

        fn blob_path(&self, hash: u64) -> PathBuf {
            self.root.join("blob").join(format!("{hash:016x}.blob"))
        }

        fn texture_path(&self, content_key: u64) -> PathBuf {
            self.root
                .join("texture")
                .join(format!("{content_key:016x}.postcard"))
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

        fn load_blob(&self, hash: u64) -> Option<Vec<u8>> {
            let framed = std::fs::read(self.blob_path(hash)).ok()?;
            match super::unframe_blob(hash, &framed) {
                Some(payload) => Some(payload.to_vec()),
                None => {
                    log::warn!("runt-cache: blob {hash:016x} failed its checksum; re-baking");
                    None
                }
            }
        }

        fn store_blob(&self, hash: u64, bytes: &[u8]) {
            NativeDiskCache::write_atomic(&self.blob_path(hash), &super::frame_blob(hash, bytes));
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

        fn load_texture(&self, content_key: u64) -> Option<crate::bake::TextureData> {
            let bytes = std::fs::read(self.texture_path(content_key)).ok()?;
            match postcard::from_bytes::<crate::bake::TextureData>(&bytes) {
                Ok(data) => Some(data),
                Err(e) => {
                    log::warn!(
                        "runt-cache: texture {content_key:016x} failed to decode ({e}); rebaking"
                    );
                    None
                }
            }
        }

        fn store_texture(&self, content_key: u64, data: &crate::bake::TextureData) {
            match postcard::to_stdvec(data) {
                Ok(bytes) => {
                    NativeDiskCache::write_atomic(&self.texture_path(content_key), &bytes)
                }
                Err(e) => log::warn!("runt-cache: texture {content_key:016x} failed to encode ({e})"),
            }
        }

        fn caches_textures(&self) -> bool {
            true
        }

        fn label(&self) -> &'static str {
            "disk"
        }
    }
}

// ---------------------------------------------------------------------------
// In-memory store (the web's half of persistence)
// ---------------------------------------------------------------------------

/// A store held entirely in memory, pre-loadable from somewhere slower and
/// asynchronous, and able to hand back everything written to it.
///
/// This is the shape that makes IndexedDB work behind a synchronous trait
/// (module docs, "Web"). The host awaits the database, fills one of these,
/// hands it to the [`Sim`](crate::Sim) as an ordinary [`CacheStore`], and later
/// drains [`take_written`](MemCache::take_written) back out to storage. Nothing
/// in the engine knows any of that happened.
///
/// Entries are `String → Vec<u8>` because that is what every browser store is
/// natively, and the keys are namespaced by hand (`mesh/…`, `key/…`, `blob/…`)
/// so one flat database holds all three layers. Clones share one map — the
/// point is for the host to keep a handle on the copy the engine is using.
///
/// [`caches_textures`](CacheStore::caches_textures) stays `false`: persisting a
/// bake needs a blocking GPU readback, and where this store runs there is no
/// such thing (see the trait method's docs). That constraint is about the
/// *platform*, not about where the bytes end up, so a memory store cannot opt
/// out of it.
#[derive(Clone, Default)]
pub struct MemCache {
    inner: Arc<Mutex<MemInner>>,
}

#[derive(Default)]
struct MemInner {
    entries: BTreeMap<String, Vec<u8>>,
    /// Keys written since the last drain. Deliberately *not* every key: an
    /// entry that was preloaded and never rewritten must not be flushed back to
    /// storage it already came from. A set, because a key rewritten five times
    /// is still one entry to flush — and a *sorted* one, because a flush that
    /// depends on hash-iteration order is a flush that cannot be diffed.
    written: BTreeSet<String>,
}

impl MemCache {
    pub fn new() -> MemCache {
        MemCache::default()
    }

    /// A store already holding `entries` — a database read, typically. These
    /// count as loaded, not written, so they will not be flushed back.
    pub fn preloaded(entries: impl IntoIterator<Item = (String, Vec<u8>)>) -> MemCache {
        let cache = MemCache::new();
        for (key, bytes) in entries {
            cache.preload(key, bytes);
        }
        cache
    }

    /// Add one already-persisted entry.
    pub fn preload(&self, key: String, bytes: Vec<u8>) {
        self.with(|inner| {
            inner.entries.insert(key, bytes);
        });
    }

    /// Take every entry written since the last call, leaving them in the cache.
    ///
    /// The host's flush: whatever comes back has to reach storage for the next
    /// page load to be warm, and dropping it on the floor is legal (it costs a
    /// regeneration and nothing else, like every other miss here).
    pub fn take_written(&self) -> Vec<(String, Vec<u8>)> {
        self.with(|inner| {
            let keys = std::mem::take(&mut inner.written);
            keys.into_iter()
                .filter_map(|k| inner.entries.get(&k).map(|v| (k.clone(), v.clone())))
                .collect()
        })
    }

    /// Entries held, loaded and written together.
    pub fn len(&self) -> usize {
        self.with(|inner| inner.entries.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Entries waiting for the next [`take_written`](MemCache::take_written).
    pub fn pending_writes(&self) -> usize {
        self.with(|inner| inner.written.len())
    }

    fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.with(|inner| inner.entries.get(key).cloned())
    }

    fn put(&self, key: String, bytes: Vec<u8>) {
        self.with(|inner| {
            inner.written.insert(key.clone());
            inner.entries.insert(key, bytes);
        });
    }

    /// A poisoned lock would mean a panic *inside* one of these tiny closures,
    /// which cannot happen — but the cache is an optimization, so even that
    /// recovers rather than propagating: the map is still readable.
    fn with<T>(&self, f: impl FnOnce(&mut MemInner) -> T) -> T {
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        f(&mut guard)
    }
}

/// `mesh/0123456789abcdef` and friends: the key one entry lives under.
pub fn mem_key(namespace: &str, hash: u64) -> String {
    format!("{namespace}/{hash:016x}")
}

impl CacheStore for MemCache {
    fn load_mesh(&self, content_hash: u64) -> Option<MeshData> {
        let bytes = self.get(&mem_key("mesh", content_hash))?;
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
            Ok(bytes) => self.put(mem_key("mesh", content_hash), bytes),
            Err(e) => log::warn!("runt-cache: {content_hash:016x} failed to encode ({e})"),
        }
    }

    fn load_blob(&self, hash: u64) -> Option<Vec<u8>> {
        let framed = self.get(&mem_key("blob", hash))?;
        match unframe_blob(hash, &framed) {
            Some(payload) => Some(payload.to_vec()),
            None => {
                log::warn!("runt-cache: blob {hash:016x} failed its checksum; re-baking");
                None
            }
        }
    }

    fn store_blob(&self, hash: u64, bytes: &[u8]) {
        self.put(mem_key("blob", hash), frame_blob(hash, bytes));
    }

    fn load_key(&self, param_key: u64) -> Option<u64> {
        let bytes = self.get(&mem_key("key", param_key))?;
        let text = std::str::from_utf8(&bytes).ok()?;
        u64::from_str_radix(text.trim(), 16).ok()
    }

    fn store_key(&self, param_key: u64, content_hash: u64) {
        self.put(
            mem_key("key", param_key),
            format!("{content_hash:016x}").into_bytes(),
        );
    }

    fn label(&self) -> &'static str {
        "memory"
    }
}

/// The platform's default persistent store: the workspace's build-directory
/// cache natively, nothing on web.
///
/// Hosts opt in; [`Sim::new`](crate::Sim::new) does not, so a test never touches
/// the filesystem unless it asked to.
///
/// This is the *in-process* answer, and it is deliberately the dumb one: web
/// gets [`NoopCache`] because reaching IndexedDB needs a future, and no
/// synchronous function can await one. A host that can await — every host, in
/// practice, since graphics init is already async — should use
/// `runt_app::cache::open` instead, which gives web a preloaded [`MemCache`]
/// and native a per-app store under the user's cache directory
/// (`NativeDiskCache::in_cache_dir`).
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

    /// The persistent store, for the texture bake (DESIGN §7).
    ///
    /// Textures are baked by the *renderer* (they need a device) while the
    /// store lives here with the rest of the content pipeline, so the bake
    /// borrows it rather than the cache growing a GPU dependency.
    pub fn store(&self) -> &dyn CacheStore {
        self.store.as_ref()
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
