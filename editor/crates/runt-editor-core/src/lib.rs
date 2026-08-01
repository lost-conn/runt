//! runt editor v1, minus the GUI (DESIGN §10).
//!
//! Everything the editor does that is *not* "draw a button" lives here, and none
//! of it depends on rinch. That split is not tidiness for its own sake: rinch
//! pulls stylo, vello and a forked winit, and a test suite that has to build
//! them to check that a slider is bounded correctly is a test suite nobody runs.
//! `cargo test -p runt-editor-core` builds the engine and nothing else.
//!
//! ```text
//!  runt-editor        rinch app: panels, viewport, event handlers
//!       │             (the only crate that knows what a widget looks like)
//!       ▼
//!  runt-editor-core   ┌ protocol       the two channels between the threads
//!       │             ├ engine_thread  the loop that owns the engine
//!       │             ├ bridge         texture → RGBA8 → SurfaceWriter
//!       │             ├ mapper         Reflect → widget tree → Reflect
//!       │             ├ path           addressing a field in a reflected value
//!       │             ├ orbit          the editor camera's maths
//!       │             └ debounce       coalescing slider drags
//!       ▼
//!  runt-core          the engine, with the `reflect` feature on
//! ```

pub mod bridge;
pub mod debounce;
pub mod engine_thread;
pub mod mapper;
pub mod orbit;
pub mod path;
pub mod protocol;

pub use bridge::FrameBridge;
pub use debounce::Debouncer;
pub use engine_thread::{spawn, EngineConfig, EngineHandle};
pub use mapper::{Edit, Widget};
pub use orbit::Orbit;
pub use path::FieldPath;
pub use protocol::{Command, Event, FrameSink, SceneSnapshot, Stats};

/// The scenes the editor offers out of the box, relative to the repo root
/// (DESIGN §12 step 4 and step 6).
pub const BUILTIN_SCENES: &[(&str, &str)] = &[
    ("demo", "assets/demo.ron"),
    ("ball level 1", "demo/ball/assets/level1.ron"),
];

/// splitmix64 — the same mixer `runt_mesh::terrain::hash_u64` uses.
///
/// Rerolling a seed has to be *deterministic*, so it is a step of this rather
/// than a draw from a thread RNG: the editor gains no `rand` dependency, and a
/// given seed always rerolls to the same next seed (DESIGN §3's rule about
/// explicitly seeded PRNGs, applied to a tool).
pub fn reroll_seed(current: u64) -> u64 {
    runt_mesh::terrain::hash_u64(current.wrapping_add(0x9E37_79B9_7F4A_7C15))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rerolling_is_deterministic_and_moves() {
        assert_eq!(reroll_seed(0), reroll_seed(0));
        assert_ne!(reroll_seed(0), 0);
        assert_ne!(reroll_seed(0), reroll_seed(1));
        // A chain of rerolls must not fall into a short cycle.
        let mut seen = std::collections::HashSet::new();
        let mut seed = 20260731u64;
        for _ in 0..1000 {
            seed = reroll_seed(seed);
            assert!(seen.insert(seed), "reroll cycled after {} steps", seen.len());
        }
    }
}
