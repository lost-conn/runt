//! The host's half of the content cache (DESIGN §6): where the bytes actually
//! live on each platform.
//!
//! `runt-core` defines the [`CacheStore`] trait and refuses to have an opinion
//! about storage; picking one is a host job, exactly like picking an audio
//! device. This module is that pick:
//!
//! ```text
//! native   $XDG_CACHE_HOME/<app>/content/…   NativeDiskCache, read+written in place
//! web      IndexedDB "<app>-cache"/entries   read once into a MemCache,
//!                                            flushed back out after a frame
//! ```
//!
//! ## Why the web path is shaped like this
//!
//! [`CacheStore`] is synchronous, because it is called from the middle of scene
//! loading — mesh generation, level bakes — and making it async would make the
//! scene loader async, and then the sim, and then everything. IndexedDB is
//! asynchronous and cannot be made otherwise.
//!
//! The way out is that the host's init *already* is a future ([`Host::new`] is
//! awaited by both platforms), so the asynchrony can happen strictly before the
//! [`Sim`](runt_core::Sim) exists and strictly after it has produced whatever
//! it was going to produce. Read the whole database up front, hand the engine a
//! plain in-memory store, drain the writes back afterwards. The engine never
//! learns that a database was involved and no sync/async boundary moves.
//!
//! ## Bounds, and what is out of scope
//!
//! [`MAX_PRELOAD_ENTRIES`] caps how much of the database is read into memory on
//! start. There is deliberately **no eviction** in v1: nothing here deletes an
//! entry, and a database that grows past the cap simply stops being fully read
//! (the excess is dead weight in storage, and the game re-bakes what it cannot
//! see). Eviction wants an LRU stamp and a size budget, both of which want a
//! shipping story about how big a level bake gets — when that exists this is
//! where it goes.
//!
//! ## What a game calls
//!
//! Nothing, for meshes: the store reaches generation through
//! [`Sim`](runt_core::Sim)'s `GenCache` automatically, so a game that only
//! places generators gets warm starts by existing.
//!
//! A game with a bake of its own — a level compiled to meshes plus collision
//! plus a spawn table — wraps that bake in two calls on the same store, reached
//! through `Sim::cache_store()`:
//!
//! ```text
//! let key = bake_key(LEVEL_RON, quality);   // see runt_core::cache's recipe
//! let built = match sim.cache_store().load_blob(key) {
//!     Some(bytes) => postcard::from_bytes::<Built>(&bytes).ok(),
//!     None => None,
//! };
//! let built = built.unwrap_or_else(|| {
//!     let built = build_level(LEVEL_RON, quality);          // the slow path
//!     if let Ok(bytes) = postcard::to_stdvec(&built) {
//!         sim.cache_store().store_blob(key, &bytes);
//!     }
//!     built
//! });
//! ```
//!
//! Note what the game does *not* do: it never asks which platform it is on,
//! never opens storage, and never handles a failure — a miss and a corrupt
//! entry are the same `None`, and the answer to both is the bake it would have
//! run anyway. That is the contract that lets this module swap a directory for
//! a database without the game noticing.

use runt_core::cache::CacheStore;

/// Entries read into memory on start (web). Roughly "a couple of levels and
/// their meshes"; see the module docs on why this is a cap and not a policy.
pub const MAX_PRELOAD_ENTRIES: usize = 64;

/// The host's cache: a store to hand the engine, plus whatever is needed to get
/// writes back to durable storage.
///
/// Native is one object doing both jobs (a disk store persists as it is
/// written, so [`flush`](HostCache::flush) is a no-op). Web is a memory store
/// plus a database name.
pub struct HostCache {
    store: Option<Box<dyn CacheStore>>,
    #[cfg(target_arch = "wasm32")]
    mem: Option<runt_core::cache::MemCache>,
    #[cfg(target_arch = "wasm32")]
    db_name: String,
}

/// The single object store inside the database — see `crate::idb` on why there
/// is exactly one.
#[cfg(target_arch = "wasm32")]
const IDB_STORE: &str = "entries";

impl HostCache {
    /// Open the best store this platform has for `app`, reading web storage in
    /// full before returning.
    ///
    /// Awaited from graphics init, so it lands before the
    /// [`Sim`](runt_core::Sim) is built — which is the whole point: after that
    /// moment nothing may block on a database.
    pub async fn open(app: &str) -> HostCache {
        #[cfg(not(target_arch = "wasm32"))]
        {
            HostCache {
                store: Some(native_store(app)),
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            let db_name = format!("{app}-cache");
            let entries =
                crate::idb::idb_load_all(&db_name, IDB_STORE, MAX_PRELOAD_ENTRIES).await;
            log::info!("cache: {} entries preloaded from {db_name}", entries.len());
            let mem = runt_core::cache::MemCache::preloaded(entries);
            HostCache {
                store: Some(Box::new(mem.clone())),
                mem: Some(mem),
                db_name,
            }
        }
    }

    /// A cache that keeps nothing, for a host that has opted out.
    pub fn disabled() -> HostCache {
        HostCache {
            store: Some(Box::new(runt_core::cache::NoopCache)),
            #[cfg(target_arch = "wasm32")]
            mem: None,
            #[cfg(target_arch = "wasm32")]
            db_name: String::new(),
        }
    }

    /// The store to give [`SimConfig::with_cache`](runt_core::SimConfig::with_cache).
    ///
    /// Takeable once; a second call yields a [`NoopCache`](runt_core::NoopCache),
    /// because two live stores over one database would be two answers to the
    /// same question.
    pub fn take_store(&mut self) -> Box<dyn CacheStore> {
        self.store
            .take()
            .unwrap_or_else(|| Box::new(runt_core::cache::NoopCache))
    }

    /// Push anything written since the last flush to durable storage.
    ///
    /// Native: nothing to do, the disk store wrote as it went. Web: drain the
    /// memory store and spawn the IndexedDB write, which completes whenever the
    /// browser gets to it — deliberately unawaited, since a frame must not wait
    /// on storage and a lost write only costs a re-bake.
    ///
    /// Cheap enough to call every frame: with no pending writes it is one
    /// uncontended lock and a length check.
    pub fn flush(&self) {
        #[cfg(target_arch = "wasm32")]
        {
            let Some(mem) = self.mem.as_ref() else {
                return;
            };
            if mem.pending_writes() == 0 {
                return;
            }
            let entries = mem.take_written();
            let db_name = self.db_name.clone();
            log::debug!("cache: flushing {} entries to {db_name}", entries.len());
            wasm_bindgen_futures::spawn_local(async move {
                crate::idb::idb_put_all(&db_name, IDB_STORE, entries).await;
            });
        }
    }
}

/// The disk store for `app`, under the user's cache directory — or, when the
/// environment has no such directory, the workspace one this build was made in.
#[cfg(not(target_arch = "wasm32"))]
fn native_store(app: &str) -> Box<dyn CacheStore> {
    use runt_core::cache::NativeDiskCache;
    match NativeDiskCache::in_cache_dir(app) {
        Some(store) => {
            log::debug!("cache: {}", store.root().display());
            Box::new(store)
        }
        None => Box::new(NativeDiskCache::in_target()),
    }
}

/// A filesystem- and database-safe name derived from a window title.
///
/// Hosts name programs for humans ("runt ball"), and that string becomes a
/// directory and a database name, so it gets squeezed down to lowercase
/// `[a-z0-9-]` with no leading dot and no `..`. Two titles can collide after
/// squeezing; a program that cares passes its own name instead
/// ([`RunConfig::with_cache_name`](crate::RunConfig::with_cache_name)).
pub fn slug(title: &str) -> String {
    let mut out = String::with_capacity(title.len());
    for c in title.chars() {
        match c {
            'a'..='z' | '0'..='9' => out.push(c),
            'A'..='Z' => out.push(c.to_ascii_lowercase()),
            _ if out.ends_with('-') || out.is_empty() => {}
            _ => out.push('-'),
        }
    }
    let trimmed = out.trim_end_matches('-');
    if trimmed.is_empty() {
        "runt".to_string()
    } else {
        trimmed[..trimmed.len().min(64)].to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::{slug, HostCache};

    #[test]
    fn a_disabled_cache_hands_out_a_store_that_keeps_nothing() {
        let mut cache = HostCache::disabled();
        assert_eq!(cache.take_store().label(), "noop");
        // Taken twice by mistake: still a legal store, still keeps nothing.
        assert_eq!(cache.take_store().label(), "noop");
        cache.flush(); // Must be callable every frame, with or without a store.
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn the_native_store_is_a_disk_store_under_the_app_name() {
        let store = super::native_store("runt-selftest");
        assert_eq!(store.label(), "disk");
        // Nothing was written: `native_store` only resolves a path, and the
        // disk store creates directories when it first stores something.
        assert_eq!(store.load_blob(1), None);
    }

    #[test]
    fn slugs_are_safe_names() {
        assert_eq!(slug("runt ball"), "runt-ball");
        assert_eq!(slug("3dimenshift"), "3dimenshift");
        assert_eq!(slug("  ../../etc/passwd  "), "etc-passwd");
        assert_eq!(slug("....."), "runt");
        assert_eq!(slug(""), "runt");
        assert_eq!(slug("Ünïcödé!!"), "n-c-d");
        for name in ["runt ball", "../..", "a/b/c", "..."] {
            let s = slug(name);
            assert!(!s.is_empty() && !s.contains("..") && !s.contains('/'), "{s}");
        }
    }
}
