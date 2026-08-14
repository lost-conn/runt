//! A string key-value store, per platform.
//!
//! The smallest thing that lets a game keep a high score, a settings blob or a
//! chosen quality tier across runs: two functions over `String`, backed by
//! `localStorage` on web and a file under the user's config directory natively.
//!
//! ## Why it is this small
//!
//! Because the alternative is a save system, and a save system is a design
//! decision the engine has not made. `String` in and `String` out means a game
//! picks its own format (RON, postcard-in-base64, a single integer) without the
//! host having an opinion, and it means the two backends can be genuinely
//! equivalent — every browser storage API is string-shaped, so anything richer
//! would be an abstraction with a lie on one side.
//!
//! ## What it is not
//!
//! - **Not the content cache.** That is [`runt_core::cache`], which is content-
//!   addressed, has a size story and holds megabytes of baked mesh. This holds
//!   bytes-to-kilobytes of *player* state and never regenerates anything.
//! - **Not sim state.** Nothing here may reach a fixed tick without going
//!   through the input path, or a replay of that trace diverges on a machine
//!   with a different save file (DESIGN §4).
//! - **Not transactional.** Both backends are best-effort; every failure is a
//!   `None`/`false` and a log line. A game that cannot start without its save is
//!   a game that will not start on a browser in private mode.

/// Read `key` for `app`, or `None` if it was never written (or cannot be read).
pub fn load(app: &str, key: &str) -> Option<String> {
    if !is_safe(app) || !is_safe(key) {
        log::warn!("storage: refusing unsafe name {app:?}/{key:?}");
        return None;
    }
    imp::load(app, key)
}

/// Write `value` at `key` for `app`. `false` if it did not stick.
pub fn save(app: &str, key: &str, value: &str) -> bool {
    if !is_safe(app) || !is_safe(key) {
        log::warn!("storage: refusing unsafe name {app:?}/{key:?}");
        return false;
    }
    imp::save(app, key, value)
}

/// Whether a name may be pasted into a path (or a `localStorage` key) as-is.
///
/// The native backend turns `app`/`key` into `<config>/<app>/<key>`, so a key of
/// `../../.ssh/authorized_keys` would be a write primitive pointed at the user's
/// home directory. Rather than escape the names — which would make the file
/// names unreadable and still need this predicate to decide what to escape —
/// the store simply refuses anything that is not an identifier-ish word. Keys
/// are written by the program, not the player, so this costs a game nothing.
///
/// Stricter than [`runt_core::dirs::is_app_name`], which the resolver applies
/// to `app` underneath this, and deliberately not merged with it: that one is
/// the floor every caller needs (a name that cannot climb out of its directory)
/// while this one is a policy about *file* names in a directory a human will
/// occasionally open. Loosening this to the floor would let a config file be
/// called `°`, and tightening the floor to this would silently narrow what a
/// cache directory may be named.
fn is_safe(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        && !name.starts_with('.')
        && !name.contains("..")
}

// ---------------------------------------------------------------------------
// Native: a file under the config directory
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    use std::path::PathBuf;

    /// `<the user's config directory>/<app>/<key>`, per
    /// [`runt_core::dirs::config_dir`]: `$XDG_CONFIG_HOME` or `$HOME/.config`
    /// on Linux, `%APPDATA%` on Windows, `~/Library/Application Support` on
    /// macOS.
    ///
    /// This used to read the two XDG variables here and say that on
    /// macOS/Windows — where the "right" directory is somewhere else entirely —
    /// `$HOME/.config` was all this needed to be "until there is a shipping
    /// story that says otherwise". There is now such a story, and it turned out
    /// worse than the comment allowed for: Windows has no `$HOME` either, so
    /// this returned `None` and every save silently did nothing.
    ///
    /// What has *not* changed is the dependency budget — [`runt_core::dirs`] is
    /// still `env::var_os` and nothing else, no `dirs`, no `directories`, no
    /// platform crate — nor the ability to point the writes somewhere else:
    /// `$XDG_CONFIG_HOME` wins wherever it is set, which is what the round-trip
    /// test below relies on and why that test keeps passing on a macOS runner.
    fn path(app: &str, key: &str) -> Option<PathBuf> {
        Some(runt_core::dirs::config_dir(app)?.join(key))
    }

    pub fn load(app: &str, key: &str) -> Option<String> {
        // A missing file is the normal case (nothing saved yet), so it is not
        // worth a log line; anything else is.
        let path = path(app, key)?;
        match std::fs::read_to_string(&path) {
            Ok(text) => Some(text),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => {
                log::warn!("storage: cannot read {}: {e}", path.display());
                None
            }
        }
    }

    pub fn save(app: &str, key: &str, value: &str) -> bool {
        let Some(path) = path(app, key) else {
            log::warn!("storage: the environment names no config directory for {app:?}");
            return false;
        };
        let Some(dir) = path.parent() else {
            return false;
        };
        if let Err(e) = std::fs::create_dir_all(dir) {
            log::warn!("storage: cannot create {}: {e}", dir.display());
            return false;
        }
        // Temp file plus rename, as the mesh cache does: a crash mid-write
        // leaves the previous save intact rather than a truncated one, and two
        // processes cannot interleave bytes.
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        if let Err(e) = std::fs::write(&tmp, value) {
            log::warn!("storage: cannot write {}: {e}", tmp.display());
            return false;
        }
        if let Err(e) = std::fs::rename(&tmp, &path) {
            log::warn!("storage: cannot publish {}: {e}", path.display());
            let _ = std::fs::remove_file(&tmp);
            return false;
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Web: localStorage
// ---------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
mod imp {
    /// `localStorage` is a flat namespace shared with whatever else the page
    /// runs, so the app name is a prefix rather than a container: `app.key`.
    fn name(app: &str, key: &str) -> String {
        format!("{app}.{key}")
    }

    /// `None` whenever storage is unavailable — which is a real, common state,
    /// not an error: private browsing and third-party-iframe contexts both make
    /// `localStorage` throw on access rather than return null.
    fn store() -> Option<web_sys::Storage> {
        web_sys::window()?.local_storage().ok().flatten()
    }

    pub fn load(app: &str, key: &str) -> Option<String> {
        store()?.get_item(&name(app, key)).ok().flatten()
    }

    pub fn save(app: &str, key: &str, value: &str) -> bool {
        let Some(store) = store() else {
            log::warn!("storage: localStorage unavailable");
            return false;
        };
        // `setItem` throws on quota exhaustion, which is the one failure a game
        // can actually do something about, so it is worth naming.
        match store.set_item(&name(app, key), value) {
            Ok(()) => true,
            Err(e) => {
                log::warn!("storage: cannot write {}: {e:?}", name(app, key));
                false
            }
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    /// One test function on purpose: it sets `XDG_CONFIG_HOME`, and the
    /// environment is per *process*, so two of these racing in the test
    /// harness's thread pool would read each other's directory.
    #[test]
    fn native_round_trip() {
        let root = std::env::temp_dir().join(format!(
            "runt-storage-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::env::set_var("XDG_CONFIG_HOME", &root);

        // Nothing saved yet.
        assert_eq!(load("runt-test", "score"), None);

        assert!(save("runt-test", "score", "1234"));
        assert_eq!(load("runt-test", "score").as_deref(), Some("1234"));
        assert!(root.join("runt-test").join("score").is_file());

        // Overwrite, and a value with newlines and unicode survives verbatim.
        let blob = "(quality: 0.5)\nnickname: \"ünicöde\"\n";
        assert!(save("runt-test", "score", blob));
        assert_eq!(load("runt-test", "score").as_deref(), Some(blob));

        // Keys are independent, and an unwritten one is still absent.
        assert!(save("runt-test", "settings", "x"));
        assert_eq!(load("runt-test", "score").as_deref(), Some(blob));
        assert_eq!(load("runt-test", "never-written"), None);

        // A traversing name is refused rather than escaping the directory.
        assert!(!save("runt-test", "../escaped", "nope"));
        assert!(!save("..", "escaped", "nope"));
        assert_eq!(load("runt-test", "../escaped"), None);
        assert!(!root.join("escaped").exists());

        std::fs::remove_dir_all(&root).ok();
    }
}
