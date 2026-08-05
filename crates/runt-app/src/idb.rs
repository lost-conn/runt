//! IndexedDB, in the smallest shape a content cache needs (web only).
//!
//! Two operations — read the whole store, write some entries — over a database
//! of `String → Uint8Array`. That is the entire API surface, because the thing
//! on the other side of it is [`runt_core::cache::MemCache`], which is also a
//! flat map of bytes: the browser's only *synchronously readable* storage is
//! memory, so the plan is to read the database once into memory and write back
//! out of it (see `crate::cache`).
//!
//! ## Why not a cursor, a schema, or an index
//!
//! `get_all` + `get_all_keys` are two requests and no state machine; a cursor
//! is a callback loop that would have to be rebuilt as a stream to be awaited.
//! There is one object store, its keys are strings and its values are bytes, so
//! there is nothing to index and nothing to migrate — a version bump would just
//! be a cache miss anyway.
//!
//! ## Failure is a miss
//!
//! Private browsing refuses to open a database, quota exhaustion refuses a
//! write, and a user may clear site data mid-session. Every one of those paths
//! returns "no entries" or does nothing, which the cache's contract already
//! covers: the game regenerates and the frame it costs is the whole penalty.

use js_sys::{Array, Uint8Array};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{IdbDatabase, IdbObjectStore, IdbRequest, IdbTransactionMode};

/// Read every entry in `store`, or an empty vec if the database cannot be read.
///
/// `limit` caps how many entries are *kept* (see `crate::cache` for why one
/// exists and why eviction does not).
pub async fn idb_load_all(db_name: &str, store: &str, limit: usize) -> Vec<(String, Vec<u8>)> {
    match load_all_inner(db_name, store, limit).await {
        Ok(entries) => entries,
        Err(e) => {
            log::info!("idb: {db_name}/{store} unreadable ({e:?}); starting cold");
            Vec::new()
        }
    }
}

/// Write one entry. Convenience over [`idb_put_all`] — a single-entry batch.
pub async fn idb_put(db_name: &str, store: &str, key: String, bytes: Vec<u8>) -> bool {
    idb_put_all(db_name, store, vec![(key, bytes)]).await
}

/// Write `entries` in one transaction. `false` if the write did not stick.
///
/// One transaction rather than one each: a flush is "everything this frame
/// baked", and a browser that closes the tab mid-flush should leave either the
/// old contents or the new ones, not half a level.
pub async fn idb_put_all(db_name: &str, store: &str, entries: Vec<(String, Vec<u8>)>) -> bool {
    if entries.is_empty() {
        return true;
    }
    let count = entries.len();
    match put_all_inner(db_name, store, entries).await {
        Ok(()) => true,
        Err(e) => {
            log::warn!("idb: {db_name}/{store} write of {count} entries failed ({e:?})");
            false
        }
    }
}

// ---------------------------------------------------------------------------
// The plumbing
// ---------------------------------------------------------------------------

async fn load_all_inner(
    db_name: &str,
    store: &str,
    limit: usize,
) -> Result<Vec<(String, Vec<u8>)>, JsValue> {
    let db = open(db_name, store).await?;
    let tx = db.transaction_with_str_and_mode(store, IdbTransactionMode::Readonly)?;
    let os: IdbObjectStore = tx.object_store(store)?;

    // Both requests are issued before either is awaited, so they share the
    // transaction's single turn rather than serializing on the microtask queue.
    let keys_req = os.get_all_keys()?;
    let values_req = os.get_all()?;
    let keys: Array = await_request(&keys_req).await?.dyn_into()?;
    let values: Array = await_request(&values_req).await?.dyn_into()?;

    let mut out = Vec::new();
    for i in 0..keys.length().min(values.length()) {
        if out.len() >= limit {
            log::info!("idb: {db_name}/{store} has more than {limit} entries; ignoring the rest");
            break;
        }
        let Some(key) = keys.get(i).as_string() else {
            continue; // Not ours: something wrote a non-string key.
        };
        let value = values.get(i);
        if let Some(bytes) = value.dyn_ref::<Uint8Array>() {
            out.push((key, bytes.to_vec()));
        }
    }
    db.close();
    Ok(out)
}

async fn put_all_inner(
    db_name: &str,
    store: &str,
    entries: Vec<(String, Vec<u8>)>,
) -> Result<(), JsValue> {
    let db = open(db_name, store).await?;
    let tx = db.transaction_with_str_and_mode(store, IdbTransactionMode::Readwrite)?;
    let os = tx.object_store(store)?;
    let mut last = None;
    for (key, bytes) in entries {
        // `Uint8Array::from` copies into the JS heap, which is what a structured
        // clone would do anyway a moment later.
        let value = Uint8Array::from(bytes.as_slice());
        last = Some(os.put_with_key(&value, &JsValue::from_str(&key))?);
    }
    // Awaiting the final request is enough: requests in a transaction complete
    // in order, so if the last one succeeded the earlier ones did too.
    if let Some(req) = last {
        await_request(&req).await?;
    }
    db.close();
    Ok(())
}

/// Open (and if necessary create) `db_name` with one object store called
/// `store`.
async fn open(db_name: &str, store: &str) -> Result<IdbDatabase, JsValue> {
    let factory = web_sys::window()
        .ok_or_else(|| JsValue::from_str("no window"))?
        .indexed_db()?
        .ok_or_else(|| JsValue::from_str("no indexedDB (private mode?)"))?;
    let req = factory.open_with_u32(db_name, 1)?;

    // `onupgradeneeded` fires before the open request succeeds, and it is the
    // only moment an object store may be created.
    let store_name = store.to_string();
    let req_for_upgrade = req.clone();
    let upgrade = Closure::once(move |_event: web_sys::Event| {
        let Ok(result) = req_for_upgrade.result() else {
            return;
        };
        let db: IdbDatabase = result.unchecked_into();
        if !db.object_store_names().contains(&store_name) {
            let _ = db.create_object_store(&store_name);
        }
    });
    req.set_onupgradeneeded(Some(upgrade.as_ref().unchecked_ref()));

    let db = await_request(req.as_ref()).await?;
    drop(upgrade); // The upgrade, if any, has already run by now.
    Ok(db.unchecked_into())
}

/// Await an `IDBRequest` as a future: resolve with its `result`, reject with
/// its `error`.
///
/// The bridge every call here needs, because IndexedDB is an event-target API
/// and Rust wants a `Future`. `Closure::once` on both handlers — a request
/// fires exactly one of them, exactly once.
async fn await_request(req: &IdbRequest) -> Result<JsValue, JsValue> {
    let promise = js_sys::Promise::new(&mut |resolve, reject| {
        let req_ok = req.clone();
        let on_success = Closure::once(move |_event: web_sys::Event| {
            let value = req_ok.result().unwrap_or(JsValue::UNDEFINED);
            let _ = resolve.call1(&JsValue::NULL, &value);
        });
        let req_err = req.clone();
        let on_error = Closure::once(move |_event: web_sys::Event| {
            let err = req_err
                .error()
                .ok()
                .flatten()
                .map(JsValue::from)
                .unwrap_or_else(|| JsValue::from_str("idb request failed"));
            let _ = reject.call1(&JsValue::NULL, &err);
        });
        req.set_onsuccess(Some(on_success.as_ref().unchecked_ref()));
        req.set_onerror(Some(on_error.as_ref().unchecked_ref()));
        // The handlers have to outlive this scope: the request calls them from
        // the event loop, long after `Promise::new` has returned. `forget` is
        // the leak that buys that, bounded by the number of requests a session
        // makes (one open plus a handful per flush).
        on_success.forget();
        on_error.forget();
    });
    JsFuture::from(promise).await
}
