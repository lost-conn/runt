//! runt editor v1 — a native rinch app with a `RenderSurface` viewport
//! (DESIGN §10).
//!
//! ```text
//!  ┌ toolbar ─────────────────────────────────────────────────┐
//!  │ demo | level 1 | Open… | Save | Save As… | pause         │
//!  ├──────────┬─────────────────────────────┬─────────────────┤
//!  │ entities │  RenderSurface              │ generator params│
//!  │  (click  │  (engine frames, orbit /    │ (generated from │
//!  │  to      │   pan / zoom)               │  Reflect)       │
//!  │  select) │                             │ transform       │
//!  ├──────────┴─────────────────────────────┴─────────────────┤
//!  │ fps · render ms · readback ms · tris · cache · tick       │
//!  └──────────────────────────────────────────────────────────┘
//! ```
//!
//! This crate is the only one that knows what a widget looks like. Everything
//! else — the bridge, the protocol, the reflection mapping, the orbit maths —
//! lives in `runt-editor-core` and is tested without a window.
//!
//! ## How the two threads meet
//!
//! rinch's event loop is `ControlFlow::Wait`: nothing polls, and a frame is
//! painted only when something asks for one. The engine thread's
//! `SurfaceWriter::submit_frame` *is* that ask — it calls `request_redraw`
//! itself. So the paint cycle runs at whatever rate the engine submits, and the
//! surface's **render callback** is a reliable once-per-frame hook on the main
//! thread. Four things ride on it:
//!
//! 1. the viewport's layout size, forwarded as `Command::Resize`;
//! 2. draining engine events into the UI state;
//! 3. framing the camera on a newly opened scene;
//! 4. flushing the edit debouncer.
//!
//! No timers, no polling thread, no `Signal::send` from the engine — the frame
//! itself is the clock. (`Signal::set` would panic off the main thread anyway.)

mod panels;
mod state;

use std::path::PathBuf;
use std::time::Instant;

use rinch::prelude::*;
use rinch::render_surface::{create_render_surface, SurfaceEvent, SurfaceMouseButton};
use runt_editor_core::protocol::{Command, Event, FrameSink};
use runt_editor_core::{EngineConfig, Orbit, BUILTIN_SCENES};

use crate::state::{Ctx, Drag, Editor};

/// Where the built-in scene paths are resolved from.
///
/// `CARGO_MANIFEST_DIR` is a *compile-time* path, so a binary carried off this
/// machine would look in the wrong place — which is fine for a developer tool
/// that is always run from its own checkout, and the Open… dialog covers
/// everything else. `RUNT_ROOT` overrides it for anyone who disagrees.
fn repo_root() -> PathBuf {
    if let Ok(root) = std::env::var("RUNT_ROOT") {
        return PathBuf::from(root);
    }
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("the manifest is three levels below the repo root")
        .to_path_buf()
}

/// The engine thread's end of the frame path.
///
/// `SurfaceWriter` is `Send + Sync + Clone` precisely so this can live on the
/// engine thread, and `submit_frame` wants exactly what the bridge produces:
/// `width × height × 4` bytes of tightly-packed RGBA8.
struct SurfaceSink(rinch::render_surface::SurfaceWriter);

impl FrameSink for SurfaceSink {
    fn submit(&mut self, pixels: &[u8], width: u32, height: u32) {
        self.0.submit_frame(pixels, width, height);
    }
}

/// Push the orbit's pose to the engine.
fn send_camera(ctx: Ctx) {
    let orbit = ctx.orbit.get();
    ctx.engine.send(Command::SetCameraPose {
        eye: orbit.eye(),
        target: orbit.target,
    });
}

/// A radius that contains the scene, for the initial camera framing.
///
/// Entity translations only: a bounding box over the actual geometry would need
/// the meshes on this side of the bridge, and "far enough out to see the
/// placements" is all a starting camera has to be.
fn scene_radius(scene: &runt_editor_core::SceneSnapshot) -> f32 {
    scene
        .entities
        .iter()
        .map(|e| e.transform.translation.length())
        .fold(4.0f32, f32::max)
        * 1.2
}

/// Send any debounced edits immediately — what Save does before writing, so a
/// slider released half a second ago is in the file.
fn flush_pending(ctx: Ctx) {
    for (generator, spec) in ctx.pending.borrow_mut().flush() {
        ctx.engine.send(Command::ParamEdit { generator, spec });
    }
}

#[component]
fn app() -> NodeHandle {
    let surface = create_render_surface();
    let engine = runt_editor_core::spawn(
        EngineConfig::default(),
        Box::new(SurfaceSink(surface.writer())),
    );
    let ctx: Ctx = Editor::leak(engine, repo_root());

    // Signals are the reactive surface over `ctx`. Rinch re-runs a closure when
    // a signal it read changes, and versioning the state is far cheaper than
    // teaching it to diff a `SceneSnapshot`.
    let revision = Signal::new(0u32);
    let status = Signal::new(String::from("pick a scene from the toolbar"));
    let title = Signal::new(String::from("no scene"));
    let paused = Signal::new(false);

    // ── the once-per-frame hook ────────────────────────────────────────────
    surface.set_render_callback(move |_writer, width, height| {
        // 1. keep the offscreen target matching the viewport
        if ctx.surface_size.get() != (width, height) {
            ctx.surface_size.set((width, height));
            ctx.engine.send(Command::Resize { width, height });
        }

        // 2. fold in whatever the engine has said
        let mut changed = false;
        let mut loaded = false;
        for event in ctx.engine.drain() {
            loaded |= matches!(event, Event::SceneLoaded(_));
            changed |= ctx.state.borrow_mut().absorb(event);
        }

        // 3. point the camera at a freshly opened scene, once per scene
        if loaded && !ctx.framed.get() {
            let radius = scene_radius(&ctx.state.borrow().scene);
            if !ctx.state.borrow().scene.entities.is_empty() {
                ctx.orbit.set(Orbit::framing(glam::Vec3::ZERO, radius));
                send_camera(ctx);
                ctx.framed.set(true);
            }
        }

        if changed {
            let state = ctx.state.borrow();
            status.set(state.status_line());
            title.set(state.title());
            drop(state);
            revision.update(|r| *r = r.wrapping_add(1));
        }

        // 4. release any edit whose quiet period has elapsed
        let ready = ctx.pending.borrow_mut().take_expired(Instant::now());
        for (generator, spec) in ready {
            ctx.engine.send(Command::ParamEdit { generator, spec });
        }
    });

    // ── viewport input ─────────────────────────────────────────────────────
    surface.set_event_handler(move |event| match event {
        SurfaceEvent::MouseDown { x, y, button } => {
            ctx.last_mouse.set((x, y));
            ctx.drag.set(match button {
                SurfaceMouseButton::Left => Drag::Orbit,
                // Middle *or* right: a laptop trackpad often has no middle
                // button, and right-drag-to-pan costs nothing since the viewport
                // has no context menu.
                SurfaceMouseButton::Middle | SurfaceMouseButton::Right => Drag::Pan,
            });
        }
        SurfaceEvent::MouseUp { .. } | SurfaceEvent::MouseLeave => ctx.drag.set(Drag::None),
        SurfaceEvent::MouseMove { x, y } => {
            let (px, py) = ctx.last_mouse.replace((x, y));
            let (dx, dy) = (x - px, y - py);
            let mut orbit = ctx.orbit.get();
            match ctx.drag.get() {
                Drag::None => return,
                Drag::Orbit => orbit.orbit(dx, dy),
                Drag::Pan => orbit.pan(dx, dy),
            }
            ctx.orbit.set(orbit);
            send_camera(ctx);
        }
        SurfaceEvent::MouseWheel { delta_y, .. } => {
            let mut orbit = ctx.orbit.get();
            orbit.zoom(delta_y);
            ctx.orbit.set(orbit);
            send_camera(ctx);
        }
        SurfaceEvent::KeyDown(key) => {
            // Frame the origin — the universal "I am lost" key.
            if key.key.eq_ignore_ascii_case("f") {
                let mut orbit = ctx.orbit.get();
                orbit.look_at(glam::Vec3::ZERO);
                ctx.orbit.set(orbit);
                send_camera(ctx);
            }
        }
        _ => {}
    });

    rsx! {
        div {
            style: "display: flex; flex-direction: column; width: 100vw; height: 100vh; \
                    background: var(--rinch-color-body); overflow: hidden;",

            // ── toolbar ───────────────────────────────────────────────────
            div {
                style: "display: flex; flex-direction: row; align-items: center; gap: 8px; \
                        padding: 6px 12px; min-height: 44px; flex-shrink: 0; \
                        border-bottom: 1px solid var(--rinch-color-default-border);",

                Text { size: "sm", weight: "bold", {move || title.get()} }
                div { style: "width: 12px;" }

                for scene in BUILTIN_SCENES.iter().copied() {
                    Button {
                        key: scene.0,
                        variant: "default",
                        size: "xs",
                        onclick: move || {
                            ctx.framed.set(false);
                            ctx.engine.send(Command::LoadScene(ctx.root.join(scene.1)));
                        },
                        {scene.0}
                    }
                }

                Button {
                    variant: "default",
                    size: "xs",
                    onclick: move || {
                        if let Some(path) = rinch::dialogs::open_file()
                            .set_title("Open a runt scene")
                            .set_directory(&ctx.root)
                            .add_filter("runt scene", &["ron"])
                            .pick_file()
                        {
                            ctx.framed.set(false);
                            ctx.engine.send(Command::LoadScene(path));
                        }
                    },
                    "Open…"
                }
                Button {
                    variant: "filled",
                    size: "xs",
                    onclick: move || {
                        flush_pending(ctx);
                        let path = ctx.state.borrow().scene_path.clone();
                        match path {
                            Some(path) => ctx.engine.send(Command::SaveScene(path)),
                            None => ctx.state.borrow_mut().error = Some("no scene to save".into()),
                        }
                    },
                    "Save"
                }
                Button {
                    variant: "default",
                    size: "xs",
                    onclick: move || {
                        if let Some(path) = rinch::dialogs::save_file()
                            .set_title("Save the scene as")
                            .set_directory(&ctx.root)
                            .set_file_name("scene.ron")
                            .add_filter("runt scene", &["ron"])
                            .save()
                        {
                            flush_pending(ctx);
                            ctx.engine.send(Command::SaveScene(path));
                        }
                    },
                    "Save As…"
                }

                div { style: "flex: 1;" }

                Button {
                    variant: "subtle",
                    size: "xs",
                    onclick: move || {
                        let next = !paused.get();
                        paused.set(next);
                        ctx.engine.send(Command::SetPaused(next));
                    },
                    {move || if paused.get() { "resume sim" } else { "pause sim" }}
                }
            }

            // ── body ──────────────────────────────────────────────────────
            div {
                style: "display: flex; flex-direction: row; flex: 1; overflow: hidden;",

                // left: entities
                div {
                    style: "width: 220px; flex-shrink: 0; overflow-y: auto; padding: 8px; \
                            box-sizing: border-box; \
                            border-right: 1px solid var(--rinch-color-default-border);",
                    {panels::entity_list(__scope, ctx, revision)}
                }

                // centre: the engine's frames
                div {
                    style: "flex: 1; overflow: hidden; display: flex; position: relative; \
                            background: #101214;",
                    RenderSurface { surface: Some(surface) }
                }

                // right: params
                div {
                    style: "width: 430px; flex-shrink: 0; overflow-x: hidden; overflow-y: auto; \
                            box-sizing: border-box; \
                            padding: 8px; \
                            border-left: 1px solid var(--rinch-color-default-border);",
                    {panels::inspector(__scope, ctx, revision)}
                }
            }

            // ── status bar ────────────────────────────────────────────────
            div {
                style: "height: 26px; flex-shrink: 0; display: flex; align-items: center; \
                        padding: 0 12px; font-size: 11px; font-family: monospace; \
                        color: var(--rinch-color-dimmed); \
                        border-top: 1px solid var(--rinch-color-default-border);",
                span { {move || status.get()} }
            }
        }
    }
}

fn main() {
    // `runt_core::DEFAULT_LOG_FILTER` rather than a bare "info": debug builds
    // keep wgpu's Vulkan validation layers on (they are worth having), but the
    // loader and `wgpu_core` narrate every probe and every resource at info,
    // which buries the scene-load lines and the warnings that matter. `RUST_LOG`
    // still overrides the whole thing.
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or(runt_core::DEFAULT_LOG_FILTER),
    )
    .init();

    let theme = ThemeProviderProps {
        primary_color: Some("blue".into()),
        default_radius: Some("sm".into()),
        dark_mode: true,
        ..Default::default()
    };
    run_with_theme("runt editor", 1600, 900, app, theme);
}
