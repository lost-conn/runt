//! Where a program may write, per platform: one resolver, two callers.
//!
//! Two things in this workspace need a directory of the user's rather than one
//! of the build tree's — [`crate::cache`]'s disk store, which wants a cache
//! directory, and `runt_app::storage`, which wants a config directory — and
//! before this module they each answered the question themselves, identically
//! and wrongly. Both read `$XDG_CACHE_HOME`/`$XDG_CONFIG_HOME` and fell back to
//! `$HOME/.cache`/`$HOME/.config`, which is correct on Linux and nowhere else.
//! Windows sets neither variable *and* has no `$HOME`, so both resolvers
//! returned `None` there: a Windows build re-baked its entire content cache on
//! every launch (the caller degrades to a store that keeps nothing) and silently
//! never saved the player's settings. macOS worked, in the sense that
//! `~/.cache` is a directory a program may write to, but it is not where the
//! platform says to write.
//!
//! `runt_app::storage` used to close that gap with a sentence — the right
//! directory is "somewhere else entirely" on macOS and Windows, and `$HOME` is
//! "all this needs to be until there is a shipping story that says otherwise".
//! There is now such a story (a game on this engine is being exported for
//! Windows and macOS), so this module is that sentence coming due.
//!
//! ## Still no `dirs` crate
//!
//! What does *not* change is the dependency budget. `dirs`/`directories` pull
//! in `windows-sys` and `core-foundation` to ask the OS its own question
//! through `SHGetKnownFolderPath` and `NSSearchPathForDirectoriesInDomains`,
//! and the engine's whole storage story is "a couple of `env::var_os` calls and
//! best-effort I/O". Every path below is an environment variable the platform
//! itself sets — `%LOCALAPPDATA%` and `%APPDATA%` come from the Windows session,
//! `$HOME` from the login — so the shell API buys correctness we already have,
//! at the price of two platform crates and a build that cross-compiles less
//! readily. If a case ever appears where the variables are wrong and the API is
//! right (a redirected known folder on a locked-down domain, say), that is the
//! moment to reconsider, and it will be one file.
//!
//! ## The rules
//!
//! ```text
//!                        cache                          config
//! (any platform) $XDG_CACHE_HOME            $XDG_CONFIG_HOME
//! Windows        %LOCALAPPDATA%             %APPDATA%
//!                %USERPROFILE%\AppData\Local  %USERPROFILE%\AppData\Roaming
//! macOS          $HOME/Library/Caches       $HOME/Library/Application Support
//! everything else $HOME/.cache              $HOME/.config
//! ```
//!
//! …and `<app>` appended to whichever won, because two games sharing a param
//! key space is a bug that looks like a corrupt level.
//!
//! **The XDG variables win when set, on every platform including Windows and
//! macOS.** Not because XDG governs those platforms, but because an explicitly
//! set variable is a person (or a test harness, or a flatpak, or a CI runner)
//! saying where the writes go, and a resolver that ignores that is a resolver
//! you cannot point at a temp directory. `runt_app::storage`'s round-trip test
//! does exactly that, and it has to keep working when it runs on the macOS
//! builder rather than the Linux one.
//!
//! **The Linux answer is byte-for-byte what it was.** `$HOME/.cache` and
//! `$HOME/.config`, not the `$XDG_*` defaults spelled out longhand, so nothing
//! already sitting on a Linux disk moves and no cache goes cold on upgrade.
//!
//! **`None` is still a legal answer.** An environment with none of these
//! variables offers nowhere to write, and both callers already handle that: the
//! cache falls back to a store that keeps nothing (slow, correct) and the KV
//! store logs and reports a failed save.

use std::ffi::OsString;
use std::path::PathBuf;

/// The user's cache directory for `app` — regenerable bytes, safe to delete.
///
/// `None` when the environment names nowhere to write, or when `app` is not
/// usable as a single path component (see [`is_app_name`]).
pub fn cache_dir(app: &str) -> Option<PathBuf> {
    resolve(Kind::Cache, Platform::HOST, app, env_var)
}

/// The user's config directory for `app` — state the player would miss.
///
/// `None` on the same terms as [`cache_dir`].
pub fn config_dir(app: &str) -> Option<PathBuf> {
    resolve(Kind::Config, Platform::HOST, app, env_var)
}

/// Whether `app` may be pasted into a path as one directory component.
///
/// The minimum both callers need and neither may skip: the name becomes a
/// directory under someone's home, so a name that can climb out of it (`..`) or
/// name a deeper path (`/`, and `\` because this now resolves Windows paths
/// too) would turn a cache root into a write primitive. Empty is refused
/// because it would silently resolve to the shared parent, where two apps would
/// then collide.
///
/// This is deliberately a *floor* rather than a policy. `runt_app::storage`
/// applies a stricter allowlist of its own on top (identifier-ish ASCII, no
/// leading dot) because it also pastes a *key* into the path and wants file
/// names a human can read in a directory listing; the cache store is happy with
/// anything that cannot escape. Tightening this to storage's rule would be a
/// silent behaviour change for a caller that never asked for one.
pub fn is_app_name(app: &str) -> bool {
    !app.is_empty() && !app.contains('/') && !app.contains('\\') && !app.contains("..")
}

// ---------------------------------------------------------------------------
// The mapping, as a value
// ---------------------------------------------------------------------------

/// Which of the two directories is being resolved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Cache,
    Config,
}

/// Whose conventions to resolve against.
///
/// A parameter rather than a `#[cfg]` on purpose. The point of this module is
/// that a Windows path is *verifiable*, and a mapping selected by `cfg` can
/// only ever be tested on the platform it is compiled for — which for the
/// Windows arm means never, since nothing in this workspace builds for Windows
/// today. As a value, every arm is exercised by `cargo test` on any host, and
/// the one thing `cfg` still decides is the single line that picks [`HOST`].
///
/// [`HOST`]: Platform::HOST
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Platform {
    Windows,
    MacOs,
    /// Linux, the BSDs, and anything else that keeps a `$HOME` with dotfiles in
    /// it. The XDG basedir spec's fallbacks, which is what these already were.
    Xdg,
}

impl Platform {
    /// The platform this build is for.
    const HOST: Platform = if cfg!(target_os = "windows") {
        Platform::Windows
    } else if cfg!(target_os = "macos") {
        Platform::MacOs
    } else {
        Platform::Xdg
    };
}

/// The real process environment: the only thing in this module that reads it,
/// and the only line [`cache_dir`] and [`config_dir`] add to [`resolve`].
///
/// Deliberately a bare `var_os` and not the place the "empty means unset" rule
/// lives — that belongs to the mapping, which is the part under test.
fn env_var(name: &str) -> Option<OsString> {
    std::env::var_os(name)
}

/// The whole mapping, with the environment as an argument.
///
/// Factored out for one reason: `std::env::set_var` mutates a process-global
/// and the test harness runs threads, so any test that sets a variable is
/// racing every other test that reads one (`runt_app::storage`'s round-trip
/// test crams itself into a single `#[test]` for exactly this reason, and says
/// so). A `Fn(&str) -> Option<OsString>` makes the environment an ordinary
/// value, so all four platforms' rules can be checked in parallel from one
/// Linux host with no global state involved.
fn resolve(
    kind: Kind,
    platform: Platform,
    app: &str,
    env: impl Fn(&str) -> Option<OsString>,
) -> Option<PathBuf> {
    if !is_app_name(app) {
        log::warn!("runt-dirs: refusing unsafe app name {app:?}");
        return None;
    }

    // A non-empty variable, or nothing — every lookup below wants that reading.
    // `XDG_CACHE_HOME=` is a broken launcher rather than an instruction to write
    // to the filesystem root, POSIX-adjacent tooling has always treated `FOO=`
    // and an absent `FOO` alike, and both of the resolvers this module replaces
    // already did the same.
    let var = |name: &str| env(name).filter(|v| !v.is_empty()).map(PathBuf::from);

    let xdg = match kind {
        Kind::Cache => "XDG_CACHE_HOME",
        Kind::Config => "XDG_CONFIG_HOME",
    };
    let base = match var(xdg) {
        Some(explicit) => explicit,
        None => match platform {
            // `%LOCALAPPDATA%` is the machine-local half of the roaming profile
            // split and `%APPDATA%` the half that follows a domain user between
            // machines — which is exactly the cache/config distinction: a baked
            // mesh must not be copied across the network at logon, a keybinding
            // blob should be. `%USERPROFILE%` is the last resort rather than a
            // first choice: both variables come from the session and are
            // effectively always present, but a service or a stripped
            // environment can lack them, and deriving the same two directories
            // from the profile root lands in the place the shell would have
            // named rather than inventing a new one.
            Platform::Windows => {
                let (var_name, under) = match kind {
                    Kind::Cache => ("LOCALAPPDATA", "AppData\\Local"),
                    Kind::Config => ("APPDATA", "AppData\\Roaming"),
                };
                var(var_name).or_else(|| Some(var("USERPROFILE")?.join(under)))?
            }
            // Apple's File System Programming Guide: `~/Library/Caches` for
            // "discardable cache files" and `~/Library/Application Support` for
            // data the app needs to keep. No bundle identifier in the path —
            // that convention belongs to a program that has one, and `<app>` is
            // the name this engine's hosts already collide-check.
            Platform::MacOs => {
                let home = var("HOME")?;
                match kind {
                    Kind::Cache => home.join("Library").join("Caches"),
                    Kind::Config => home.join("Library").join("Application Support"),
                }
            }
            // Unchanged from what both callers did before this module existed,
            // deliberately: `$HOME/.cache` is where a Linux user's runt cache
            // already is.
            Platform::Xdg => {
                let home = var("HOME")?;
                match kind {
                    Kind::Cache => home.join(".cache"),
                    Kind::Config => home.join(".config"),
                }
            }
        },
    };
    Some(base.join(app))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake environment: the variables a test cares about, and nothing else
    /// set. Returns a closure so [`resolve`] sees the same shape it sees in
    /// production.
    fn env<'a>(vars: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<OsString> + 'a {
        move |name| {
            vars.iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| OsString::from(*v))
        }
    }

    /// The resolved path with every separator written `/`.
    ///
    /// `PathBuf::join` uses the *host*'s separator, not the platform being
    /// resolved for, so a Windows path built on this Linux box comes out as
    /// `C:\Users\ada\AppData\Local/runt-test` and on a Windows box as the same
    /// thing with a backslash. Both are the same path to every Windows API
    /// (which has accepted `/` since DOS), and normalizing here is what keeps
    /// these expectations readable and identical on either host. What the
    /// assertions are actually about is *which directory*, not how it is
    /// spelled.
    fn shown(path: Option<PathBuf>) -> Option<String> {
        path.map(|p| p.display().to_string().replace('\\', "/"))
    }

    fn cache(platform: Platform, vars: &[(&str, &str)]) -> Option<String> {
        shown(resolve(Kind::Cache, platform, "runt-test", env(vars)))
    }

    fn config(platform: Platform, vars: &[(&str, &str)]) -> Option<String> {
        shown(resolve(Kind::Config, platform, "runt-test", env(vars)))
    }

    #[test]
    fn windows_splits_the_cache_and_the_config_across_the_roaming_boundary() {
        // The bug this module exists for: Windows sets neither XDG variable nor
        // `$HOME`, so the old resolvers returned `None` and a Windows build both
        // re-baked everything every launch and never saved a setting.
        let vars = [
            ("LOCALAPPDATA", "C:\\Users\\ada\\AppData\\Local"),
            ("APPDATA", "C:\\Users\\ada\\AppData\\Roaming"),
        ];
        assert_eq!(
            cache(Platform::Windows, &vars).as_deref(),
            Some("C:/Users/ada/AppData/Local/runt-test")
        );
        assert_eq!(
            config(Platform::Windows, &vars).as_deref(),
            Some("C:/Users/ada/AppData/Roaming/runt-test")
        );
    }

    #[test]
    fn windows_falls_back_to_the_user_profile() {
        // A service or a stripped environment can be missing the two AppData
        // variables; the profile root still names the same two directories.
        let vars = [("USERPROFILE", "C:\\Users\\ada")];
        assert_eq!(
            cache(Platform::Windows, &vars).as_deref(),
            Some("C:/Users/ada/AppData/Local/runt-test")
        );
        assert_eq!(
            config(Platform::Windows, &vars).as_deref(),
            Some("C:/Users/ada/AppData/Roaming/runt-test")
        );
        // `$HOME` is not a Windows concept, and a git-bash shell that sets one
        // must not drag the cache into a POSIX-shaped directory.
        assert_eq!(cache(Platform::Windows, &[("HOME", "C:\\Users\\ada")]), None);
    }

    #[test]
    fn macos_writes_where_apple_says_rather_than_into_dotfiles() {
        let vars = [("HOME", "/Users/ada")];
        assert_eq!(
            cache(Platform::MacOs, &vars).as_deref(),
            Some("/Users/ada/Library/Caches/runt-test")
        );
        assert_eq!(
            config(Platform::MacOs, &vars).as_deref(),
            Some("/Users/ada/Library/Application Support/runt-test")
        );
    }

    #[test]
    fn linux_keeps_the_directories_it_already_has() {
        // Load-bearing: a change here moves every existing cache on disk and
        // costs every Linux user a cold bake for nothing.
        let vars = [("HOME", "/home/ada")];
        assert_eq!(
            cache(Platform::Xdg, &vars).as_deref(),
            Some("/home/ada/.cache/runt-test")
        );
        assert_eq!(
            config(Platform::Xdg, &vars).as_deref(),
            Some("/home/ada/.config/runt-test")
        );
    }

    #[test]
    fn an_explicit_xdg_variable_wins_on_every_platform() {
        // What lets a test, a sandbox or a packager point the writes somewhere
        // — including on a macOS CI runner, where `runt_app::storage`'s
        // round-trip test would otherwise be resolving to `~/Library`.
        let vars = [
            ("XDG_CACHE_HOME", "/tmp/xdg-cache"),
            ("XDG_CONFIG_HOME", "/tmp/xdg-config"),
            ("HOME", "/home/ada"),
            ("LOCALAPPDATA", "C:\\Users\\ada\\AppData\\Local"),
            ("APPDATA", "C:\\Users\\ada\\AppData\\Roaming"),
            ("USERPROFILE", "C:\\Users\\ada"),
        ];
        for platform in [Platform::Windows, Platform::MacOs, Platform::Xdg] {
            assert_eq!(
                cache(platform, &vars).as_deref(),
                Some("/tmp/xdg-cache/runt-test"),
                "{platform:?}"
            );
            assert_eq!(
                config(platform, &vars).as_deref(),
                Some("/tmp/xdg-config/runt-test"),
                "{platform:?}"
            );
        }
    }

    #[test]
    fn the_two_kinds_do_not_read_each_others_variable() {
        // `XDG_CACHE_HOME` set alone must not relocate the config, or a player's
        // settings end up in a directory documented as safe to delete.
        let vars = [("XDG_CACHE_HOME", "/tmp/xdg-cache"), ("HOME", "/home/ada")];
        assert_eq!(
            cache(Platform::Xdg, &vars).as_deref(),
            Some("/tmp/xdg-cache/runt-test")
        );
        assert_eq!(
            config(Platform::Xdg, &vars).as_deref(),
            Some("/home/ada/.config/runt-test")
        );
    }

    #[test]
    fn an_empty_variable_is_an_unset_variable() {
        // `XDG_CACHE_HOME=` is a broken launcher, not a request to write to the
        // filesystem root — and `/runt-test` is a path a program would be
        // refused, or worse, allowed.
        let vars = [("XDG_CACHE_HOME", ""), ("HOME", "/home/ada")];
        assert_eq!(
            cache(Platform::Xdg, &vars).as_deref(),
            Some("/home/ada/.cache/runt-test")
        );
        // …all the way down: an empty `$HOME` behind an empty XDG is nowhere.
        assert_eq!(cache(Platform::Xdg, &[("HOME", "")]), None);
        assert_eq!(
            cache(Platform::Windows, &[("LOCALAPPDATA", ""), ("USERPROFILE", "")]),
            None
        );
    }

    #[test]
    fn an_environment_with_nowhere_to_write_resolves_to_nothing() {
        // A legal answer, not an error: the cache degrades to keeping nothing
        // and the KV store reports a failed save.
        for platform in [Platform::Windows, Platform::MacOs, Platform::Xdg] {
            assert_eq!(cache(platform, &[]), None, "{platform:?}");
            assert_eq!(config(platform, &[]), None, "{platform:?}");
        }
    }

    #[test]
    fn an_app_name_that_could_leave_its_directory_is_refused() {
        // The name is pasted into a path under someone's home directory, so
        // this is the difference between a cache root and a write primitive.
        // `\` counts now that Windows paths are being built here.
        let vars = [
            ("HOME", "/home/ada"),
            ("LOCALAPPDATA", "C:\\Users\\ada\\AppData\\Local"),
        ];
        for hostile in ["", "..", "../../etc", "a/b", "a\\b", "..\\..\\windows"] {
            for platform in [Platform::Windows, Platform::MacOs, Platform::Xdg] {
                assert_eq!(
                    resolve(Kind::Cache, platform, hostile, env(&vars)),
                    None,
                    "{hostile:?} on {platform:?}"
                );
            }
            assert!(!is_app_name(hostile), "{hostile:?}");
        }
        // A dot inside a name is not a traversal and stays legal — `runt.ball`
        // is a plausible app name and always was.
        assert!(is_app_name("runt.ball"));
        assert!(is_app_name("3dimenshift"));
    }

    #[test]
    fn the_host_resolvers_agree_with_the_mapping_for_this_platform() {
        // The one thing `cfg` still decides: that `cache_dir` is the arm above
        // for the platform this test binary was compiled for. Reads the real
        // environment, so it asserts a relationship rather than a path.
        let expected = resolve(Kind::Cache, Platform::HOST, "runt-test", env_var);
        assert_eq!(cache_dir("runt-test"), expected);
        assert_eq!(
            config_dir("runt-test"),
            resolve(Kind::Config, Platform::HOST, "runt-test", env_var)
        );
        assert_eq!(cache_dir(".."), None);
    }
}
