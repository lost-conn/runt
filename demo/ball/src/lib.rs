//! runt ball — the v0 demo game (DESIGN §12 step 6).
//!
//! A rolling-ball collector: twelve rings on 48 m of procedural terrain, a
//! follow camera, a kill plane, and a seeded run you can replay from an input
//! trace. Ships native (`runt-ball`) and web (trunk).
//!
//! The crate is three files and one of them is content:
//!
//! ```text
//! src/game.rs         every rule, as FixedSim systems  ← the game
//! assets/level1.ron   the level, as save-as-params     ← the content
//! src/lib.rs          ~30 lines of "run this"          ← this file
//! src/main.rs         the native binary + replay CLI
//! ```
//!
//! There is no fourth file, and in particular there is no engine fork: the host
//! is `runt-app` exactly as the engine demo uses it, reached through
//! [`runt_app::run_with`] with a [`RunConfig`] naming this crate's scene and
//! [`game::setup`].

/// The game's sound: the patch bank and the systems that play it (DESIGN §8).
pub mod audio;
pub mod game;

use runt_app::{AudioConfig, RunConfig};
use runt_core::{Sim, SimConfig};

/// The level, embedded at build time so the wasm build needs no fetch — the
/// same trick `runt-core` plays with `assets/demo.ron`.
pub const LEVEL1_RON: &str = include_str!("../assets/level1.ron");

/// Window title, and the fallback the host shows before the first status line.
pub const TITLE: &str = "runt ball";

/// Master level for the demo. Below unity because the mix is deliberately
/// sparse and the bus soft clip should be a safety net rather than a sound.
pub const MASTER_GAIN: f32 = 0.8;

/// The default run: level 1, engine-default quality, game rules installed,
/// sound on.
///
/// The audio is a *bank of bytes* (DESIGN §8: params are content) plus the
/// engine's ordinary event queue. The host decides what to do with it — a cpal
/// stream natively, an `AudioWorklet` on web — and this crate never finds out
/// which. `runt-native` and the engine demo leave `audio` unset and are silent.
pub fn config() -> RunConfig {
    let bank = audio::bank()
        .to_bytes()
        .expect("the patch bank is plain data and always encodes");
    RunConfig::new(TITLE, LEVEL1_RON)
        .with_setup(game::setup)
        .with_audio(AudioConfig::new(bank).with_master_gain(MASTER_GAIN))
}

/// A playable [`Sim`] with no window and no GPU: level loaded, rules installed.
///
/// This is what the tests drive, and it is the *same* two calls the host makes —
/// so a passing determinism test is a statement about the shipped game and not
/// about a test rig that resembles it.
pub fn headless_sim() -> Sim {
    let mut sim = Sim::from_config(SimConfig::default().with_scene(LEVEL1_RON));
    game::setup(&mut sim);
    sim
}

/// The browser entry point.
///
/// Lives here rather than in `runt-app` because a wasm module may have exactly
/// one `#[wasm_bindgen(start)]` and wasm-bindgen collects them from dependency
/// rlibs too; `runt-app`'s own is behind its default `wasm-entry` feature, which
/// this crate switches off.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn wasm_start() {
    runt_app::wasm_init();
    runt_app::run_with(config());
}
