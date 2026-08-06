//! runt player host — native window + web canvas.
//!
//! Everything here is windowing, surface management and presentation. The
//! engine itself is `runt-core`, which never sees a `Window` or a `Surface`:
//! this host acquires the frame's `TextureView` and hands it to
//! [`runt_core::Renderer::render`].
//!
//! ## No game logic lives here (DESIGN §2)
//!
//! > *Hosts contain no engine logic. If a feature needs host code beyond event
//! > translation and surface management, it's designed wrong.*
//!
//! So the host does not know what it is running. [`RunConfig`] is the whole
//! contract: a title, a scene, a quality tier and a `setup` hook that runs once
//! against the [`Sim`](runt_core::Sim) before its first tick. `runt-native` is
//! `run_with(RunConfig::engine_demo())`; `demo/ball` is the same call with its
//! own scene and a setup that registers its `FixedSim` systems. Neither program
//! adds a line of code to this file.
//!
//! There is one more hook, and it is a *frame* hook rather than a load one:
//! [`RunConfig::with_frame`] runs a game's closure against the whole
//! [`Engine`] between the tick and the draw. It exists because offscreen
//! viewports (a tutorial card showing a second world — see
//! [`Engine::render_to_texture`](runt_core::Engine::render_to_texture)) are
//! work that has to happen inside the frame and needs the *engine*, which no
//! system inside a `FixedSim` can borrow. The host still knows nothing about
//! what the closure does; it knows only when to call it.
//!
//! The one thing that *looks* like an exception is the status line (see
//! [`Host::sync_status`]): the engine has no text renderer, so a game writes
//! [`StatusLine`](runt_core::StatusLine) and the host paints it with whatever
//! cheap text its platform has. That is presentation — the host reads a string
//! and never writes one.

use std::sync::Arc;

use runt_core::{Engine, InputEvent, Key, Sim};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::keyboard::PhysicalKey;
use winit::window::{Window, WindowId};

/// winit → engine translation: the key table, the button indices and the touch
/// virtual stick. Public because it is the host's *interface* to the engine's
/// input vocabulary — another host (or a test) may legitimately want the same
/// mapping without going through [`run_with`].
pub mod input;

/// Gamepads. Nobody delivers a pad as events, so the host polls it once a frame
/// and diffs the result into the same [`InputEvent`]s a keyboard produces —
/// still translation and nothing else (DESIGN §2).
pub mod gamepad;

/// A string key-value store: `localStorage` on web, a file under the config
/// directory natively. Player state, deliberately not sim state.
pub mod storage;

/// The audio pump (DESIGN §8): a cpal stream natively, an `AudioWorklet` on web,
/// and silence when a game does not ask for either.
pub mod audio;

/// Where the content cache's bytes live per platform (DESIGN §6): a directory
/// natively, IndexedDB on web. The engine defines the store; the host picks one.
pub mod cache;

/// The two IndexedDB calls [`cache`] needs, and nothing else.
#[cfg(target_arch = "wasm32")]
pub mod idb;

pub use audio::AudioConfig;

// ---------------------------------------------------------------------------
// What to run
// ---------------------------------------------------------------------------

/// A one-shot hook that runs against the [`Sim`] after `Startup` (so the scene
/// is loaded and its entities exist) and before the first tick.
///
/// This is where a game registers its `FixedSim` systems and inserts its
/// resources. `FnOnce` rather than the plain `fn` pointer the host would prefer:
/// `runt-ball --replay <file>` has to hand the loaded trace to setup, and a
/// function pointer cannot carry it. Boxing costs one allocation per process.
pub type SetupFn = Box<dyn FnOnce(&mut Sim)>;

/// A hook that runs against the whole [`Engine`] once per frame, after the
/// tick(s) that frame ran and **before** the frame is drawn.
///
/// The one thing a game legitimately needs the host's frame loop for: work that
/// takes `&mut Engine` rather than `&mut Sim`. Concretely, offscreen viewports
/// — [`Engine::render_to_texture`](runt_core::Engine::render_to_texture) draws
/// a second `Sim` into a texture that this same frame's HUD then samples — and
/// nothing inside a `FixedSim` system can do it: a system borrows the world,
/// the renderer is not *in* the world (DESIGN §4 keeps render state out of
/// reach of a tick precisely so no fingerprint can depend on it), and a second
/// world cannot be a resource of the first without making one world own the
/// other's clock.
///
/// This is still not game logic in the host (DESIGN §2). The host calls a
/// closure it knows nothing about at a point in the frame only it can name; the
/// closure is the game's, and it lives in the game's crate.
///
/// `FnMut` rather than [`SetupFn`]'s `FnOnce`: it runs every frame, and the
/// state it usually carries — the demo `Sim`, its clock, which tutorial card is
/// open — is state it has to keep between them.
pub type FrameFn = Box<dyn FnMut(&mut Engine)>;

/// Everything the host needs to know about the program it is hosting.
///
/// Deliberately small. If something wants to be in here that is not a title, a
/// scene, a quality tier or a hook, it is probably engine state and belongs
/// behind [`SetupFn`] — or, if it has to happen inside every frame, behind
/// [`FrameFn`].
pub struct RunConfig {
    /// Window title natively; the base of `document.title` on web. A
    /// [`StatusLine`](runt_core::StatusLine) replaces it once a game writes one.
    pub title: String,
    /// The scene RON, normally an `include_str!` so the wasm build needs no
    /// fetch (the same trick `runt-core` plays with `assets/demo.ron`).
    pub scene_ron: String,
    /// Device/LOD multiplier (DESIGN §6, §11). `None` takes the engine default
    /// until §11's probe exists to choose one.
    pub quality: Option<f32>,
    /// Run once against the sim before the first tick.
    pub setup: Option<SetupFn>,
    /// Run every frame against the engine, after that frame's ticks and before
    /// it is drawn (see [`FrameFn`] and [`with_frame`](RunConfig::with_frame)).
    ///
    /// `None` — every program that has no offscreen viewport to draw — costs a
    /// branch on an `Option` per frame and nothing else.
    pub on_frame: Option<FrameFn>,
    /// Run once after the event loop exits, on the sim it left behind.
    ///
    /// Where `runt-ball --record` writes its input trace. **Native only**: a web
    /// page is closed, not exited, and the browser never gives the wasm module
    /// the last word. Anything that must survive on web has to be written as it
    /// happens.
    pub on_exit: Option<SetupFn>,
    /// What to call this program's slice of persistent storage: a directory
    /// under the user's cache directory natively, an IndexedDB database on web.
    ///
    /// `None` derives one from [`title`](RunConfig::title)
    /// ([`cache::slug`]), which is right until two programs pick titles that
    /// squeeze to the same name — then say so explicitly, because the cost of
    /// getting it wrong is two games sharing a cache and each treating the
    /// other's entries as misses forever.
    pub cache_name: Option<String>,
    /// Sound, or `None` for a silent program (DESIGN §8).
    ///
    /// Opt-in per program rather than per build: `runt-native` and the engine
    /// demo leave it `None` and never open a device, `demo/ball` sets it and
    /// gets the same [`AudioEvent`](runt_core::AudioEvent) stream pumped into
    /// whatever the platform has. The engine cannot tell which it got.
    pub audio: Option<AudioConfig>,
    /// Default render scale when the adapter is an integrated (or software)
    /// GPU — the first sliver of DESIGN §11's device probe. Applied **before**
    /// the `setup` hook, so anything that pins a scale explicitly (a settings
    /// file, a `?scale=` query) simply wins by running later. `None` leaves
    /// the engine default (1.0) everywhere.
    ///
    /// A default, not a tier system: the real quality-preset story is game-side
    /// (settings UI) and this field only answers "what should a machine that
    /// never opened the settings get?".
    pub integrated_gpu_scale: Option<f32>,
    /// Whether the host synthesises a [`VirtualStick`](input::VirtualStick) from
    /// the touch screen, on top of the raw
    /// [`InputEvent::Touch`](runt_core::InputEvent::Touch)es it always forwards.
    ///
    /// `true` by default, which is the behaviour every existing program has: one
    /// thumb anywhere on the glass drives
    /// [`Input::drive`](runt_core::Input::drive), and the game needs to know
    /// nothing about fingers. The raw events flow underneath it regardless, and
    /// that is safe precisely because a game which does not read them cannot be
    /// affected by them.
    ///
    /// A game that *does* read them — a floating dpad, a button grid, a camera
    /// drag, all live at once under different fingers — sets this to `false`
    /// ([`without_virtual_stick`](RunConfig::without_virtual_stick)). Otherwise
    /// the finger driving its dpad would also be deflecting the engine's stick,
    /// and anything still reading `drive()` would get double-driven by one
    /// thumb.
    pub virtual_stick: bool,
}

impl RunConfig {
    /// A config for `scene_ron`, with no setup hook.
    pub fn new(title: impl Into<String>, scene_ron: impl Into<String>) -> RunConfig {
        RunConfig {
            title: title.into(),
            scene_ron: scene_ron.into(),
            quality: None,
            setup: None,
            on_frame: None,
            on_exit: None,
            cache_name: None,
            audio: None,
            integrated_gpu_scale: None,
            virtual_stick: true,
        }
    }

    /// The engine's own demo scene — what `runt-native` and the root
    /// `index.html` run. Not a game: no setup hook, because there is no game
    /// logic to register (DESIGN §2).
    pub fn engine_demo() -> RunConfig {
        RunConfig::new("runt", runt_core::scene::DEMO_SCENE_RON)
    }

    pub fn with_quality(mut self, quality: f32) -> RunConfig {
        self.quality = Some(quality);
        self
    }

    /// See [`integrated_gpu_scale`](RunConfig::integrated_gpu_scale).
    pub fn with_integrated_gpu_scale(mut self, scale: f32) -> RunConfig {
        self.integrated_gpu_scale = Some(scale);
        self
    }

    pub fn with_setup(mut self, setup: impl FnOnce(&mut Sim) + 'static) -> RunConfig {
        self.setup = Some(Box::new(setup));
        self
    }

    /// Run `frame` every frame, against the whole [`Engine`], after that
    /// frame's ticks and **before** it is drawn (see [`FrameFn`]).
    ///
    /// The order is the point. A tutorial card's viewport has to be *finished*
    /// when the HUD quad that samples it is drawn, and the HUD is drawn inside
    /// [`Engine::render`](runt_core::Engine::render) — so the offscreen pass
    /// belongs after the tick that decided what the card should show and before
    /// the frame that shows it. Running it after `render` would put a
    /// one-frame-stale picture in every card, forever, and running it before
    /// `update` would show the *previous* tick's world in a card sitting beside
    /// a HUD showing this one's.
    ///
    /// ```no_run
    /// use runt_app::RunConfig;
    /// use runt_core::{RenderTarget, Sim};
    ///
    /// // The tutorial's world: a second `Sim`, ticked and drawn by the hook.
    /// let mut demo = Sim::new();
    /// let card = RenderTarget(0);
    /// let config = RunConfig::engine_demo().with_frame(move |engine| {
    ///     demo.update(engine.tick_count() as f64 * runt_core::TICK_DT);
    ///     // The game's HUD samples `card.handle()` from its own `UiBatch`.
    ///     engine.render_to_texture(card, &mut demo, 320, 180);
    /// });
    /// runt_app::run_with(config);
    /// ```
    pub fn with_frame(mut self, frame: impl FnMut(&mut Engine) + 'static) -> RunConfig {
        self.on_frame = Some(Box::new(frame));
        self
    }

    /// See [`on_exit`](RunConfig::on_exit). Native only; ignored on web.
    pub fn with_on_exit(mut self, on_exit: impl FnOnce(&mut Sim) + 'static) -> RunConfig {
        self.on_exit = Some(Box::new(on_exit));
        self
    }

    /// Play this program's audio through the platform's mixer (DESIGN §8).
    ///
    /// `bank` is a postcard-encoded `runt_audio::PatchBank` — bytes rather than
    /// a type, because this crate's wasm build deliberately does not link the
    /// synthesizer that would understand it.
    pub fn with_audio(mut self, audio: AudioConfig) -> RunConfig {
        self.audio = Some(audio);
        self
    }

    /// Stop synthesising the one-thumb drive stick, leaving only the raw
    /// [`InputEvent::Touch`](runt_core::InputEvent::Touch) stream — for a game
    /// that builds its own touch controls out of
    /// [`Input::touches`](runt_core::Input::touches). See
    /// [`virtual_stick`](RunConfig::virtual_stick) for why this is opt-*out*
    /// rather than opt-in.
    pub fn without_virtual_stick(mut self) -> RunConfig {
        self.virtual_stick = false;
        self
    }

    /// See [`cache_name`](RunConfig::cache_name).
    pub fn with_cache_name(mut self, name: impl Into<String>) -> RunConfig {
        self.cache_name = Some(name.into());
        self
    }

    /// The storage name this config resolves to.
    pub fn cache_name(&self) -> String {
        match &self.cache_name {
            Some(name) => cache::slug(name),
            None => cache::slug(&self.title),
        }
    }

    /// The [`SimConfig`](runt_core::SimConfig) this describes, over the store
    /// the host opened for it (DESIGN §6: persistence is a host opt-in).
    fn sim_config(&self, store: Box<dyn runt_core::CacheStore>) -> runt_core::SimConfig {
        let config = runt_core::SimConfig::default()
            .with_scene(self.scene_ron.clone())
            .with_cache(store);
        match self.quality {
            Some(q) => config.with_quality(q),
            None => config,
        }
    }
}

// ---------------------------------------------------------------------------
// Host: surface + present
// ---------------------------------------------------------------------------

struct Host {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    config: wgpu::SurfaceConfiguration,
    engine: Engine,
    start: web_time::Instant,
    /// winit reports absolute cursor positions; the engine wants deltas.
    last_cursor: Option<(f64, f64)>,
    /// Touch screens as an analog stick (see [`input::VirtualStick`]). One per
    /// window, because the anchor is a property of the surface being touched;
    /// `None` for a game that asked for raw touches only
    /// ([`RunConfig::without_virtual_stick`]).
    stick: Option<input::VirtualStick>,
    /// Gamepads, or `None` where the platform has none to give (see
    /// [`gamepad::Pads`]). Polled once a frame in [`Host::render`].
    pads: Option<gamepad::Pads>,
    /// Turns the polled pad state into events. Lives beside the poller rather
    /// than inside it so both platforms share one definition of "changed".
    pad_diff: gamepad::PadDiffer,
    /// The game's per-frame hook, if it asked for one
    /// ([`RunConfig::with_frame`]). Called in [`Host::render`], between the
    /// tick and the draw.
    on_frame: Option<FrameFn>,
    /// What the program is called when it has nothing to say.
    title: String,
    /// The last [`StatusLine`](runt_core::StatusLine) painted, so an unchanged
    /// line costs nothing per frame.
    shown_status: String,
    /// Where sound goes (DESIGN §8). [`SilentBackend`](runt_core::SilentBackend)
    /// for a program that asked for none — the engine cannot tell the
    /// difference, and neither can a determinism test.
    audio: Box<dyn runt_core::AudioBackend>,
    /// The content cache's storage side (DESIGN §6). The engine holds the
    /// store; the host holds this, which is the only thing that knows how to
    /// get bytes *out* again on a platform where writing is asynchronous.
    cache: cache::HostCache,
}

impl Host {
    async fn new(window: Arc<Window>, run: RunConfig) -> Result<Host, String> {
        let mut size = window.inner_size();
        size.width = size.width.max(1);
        size.height = size.height.max(1);

        // On the web, probe for a real WebGPU adapter and transparently fall
        // back to WebGL2 if none is available (headless, older browser, etc.).
        // Native just uses the default instance.
        let instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        #[cfg(target_arch = "wasm32")]
        let instance = wgpu::util::new_instance_with_webgpu_detection(instance_desc).await;
        #[cfg(not(target_arch = "wasm32"))]
        let instance = wgpu::Instance::new(instance_desc);

        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| format!("create surface: {e}"))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .map_err(|e| {
                format!("no usable GPU adapter (WebGPU and WebGL2 both unavailable): {e}")
            })?;

        // WebGL2-compatible limits so the same code path works on either
        // backend — the descriptor lives in runt-core so every host agrees.
        let (device, queue) = runt_core::request_device(&adapter).await?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // The host is where the persistent half of the content cache is opted
        // into (DESIGN §6): a directory natively, IndexedDB on web. This is the
        // last point at which storage may be *awaited* — after it, the engine's
        // synchronous `CacheStore` is the only view anyone has (see
        // `cache::HostCache`), so the read has to finish here, before the scene
        // load inside `from_config` asks its first question.
        let mut cache = cache::HostCache::open(&run.cache_name()).await;
        let mut engine =
            Engine::from_config(device.clone(), queue, format, run.sim_config(cache.take_store()));

        // Integrated-GPU default render scale, before `setup` so an explicit
        // pin (settings, `?scale=`) wins by running later. `Cpu` is the
        // software-rasterizer case and gets the same mercy.
        if let Some(scale) = run.integrated_gpu_scale {
            use wgpu::DeviceType::{Cpu, IntegratedGpu, Other};
            let ty = adapter.get_info().device_type;
            // `Other` is what the WebGL2 fallback reports — the browser hides
            // the GPU. A WebGPU-capable browser on a discrete card answers
            // honestly and keeps full scale; a hidden GPU errs chunky, which
            // is the right failure mode for the §11 low-end pillar.
            let treat_as_integrated =
                matches!(ty, IntegratedGpu | Cpu) || (cfg!(target_arch = "wasm32") && ty == Other);
            if treat_as_integrated {
                log::info!("integrated/hidden adapter ({ty:?}): default render scale {scale}");
                engine.sim_mut().set_render_scale(scale);
            }
        }

        // The scene exists (`Startup` ran inside `from_config`) and no tick has
        // happened yet: the one moment a game can install `FixedSim` systems
        // *and* see a fully-built world.
        if let Some(setup) = run.setup {
            setup(engine.sim_mut());
        }

        // After `setup`, so a game that builds its bank in the setup hook has
        // already done so — and before the first frame, so no tick's events are
        // dropped for want of a backend.
        let audio = match &run.audio {
            Some(config) => audio::start(config),
            None => Box::new(runt_core::SilentBackend) as Box<dyn runt_core::AudioBackend>,
        };

        let mut host = Host {
            window,
            surface,
            device,
            config,
            engine,
            start: web_time::Instant::now(),
            last_cursor: None,
            stick: run.virtual_stick.then(input::VirtualStick::new),
            pads: gamepad::Pads::new(),
            pad_diff: gamepad::PadDiffer::new(),
            on_frame: run.on_frame,
            title: run.title,
            shown_status: String::new(),
            audio,
            cache,
        };
        host.sync_status();
        Ok(host)
    }

    /// Paint [`StatusLine`](runt_core::StatusLine) wherever this platform has
    /// cheap text, if it changed since the last frame.
    ///
    /// DESIGN §13 leaves HUD text open and the renderer has no glyphs, so the
    /// answer for v0 is the two places every platform already gives us for free:
    /// the window title natively, and on web `document.title` plus the optional
    /// `#runt-status` element a game's `index.html` may declare — §13's
    /// "cheapest candidate: DOM overlay on web, nothing native".
    ///
    /// An empty status line falls back to the configured title, so a scene that
    /// never writes one looks exactly as it did before this existed.
    fn sync_status(&mut self) {
        let status = self.engine.sim().status_line();
        if status == self.shown_status {
            return;
        }
        self.shown_status = status.to_string();

        let text = if self.shown_status.is_empty() {
            self.title.clone()
        } else {
            self.shown_status.clone()
        };
        self.window.set_title(&text);

        #[cfg(target_arch = "wasm32")]
        {
            if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
                doc.set_title(&text);
                // Optional by design: a page with no `#runt-status` still gets
                // the tab title, and the host does not create DOM the page did
                // not ask for.
                if let Some(el) = doc.get_element_by_id("runt-status") {
                    el.set_text_content(Some(&self.shown_status));
                }
            }
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }

    fn render(&mut self) {
        // winit does not resize the web canvas backing store to match its CSS
        // box, so keep the surface synced to the browser viewport each frame.
        #[cfg(target_arch = "wasm32")]
        {
            let ls = browser_logical_size();
            let _ = self.window.request_inner_size(ls);
        }

        use wgpu::CurrentSurfaceTexture as Cst;
        let frame = match self.surface.get_current_texture() {
            Cst::Success(f) | Cst::Suboptimal(f) => f,
            Cst::Outdated | Cst::Lost => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            other => {
                log::warn!("surface unavailable: {other:?}");
                return;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // The pad has no event stream to sit in `handle_input` with the keyboard,
        // so it is read here — *before* `update`, so whatever the player is
        // holding right now lands in the buffer this frame's ticks will consume
        // rather than the next one's.
        self.poll_pads();

        // The host owns the clock; the engine is handed wall time and does its
        // own fixed-tick accounting (DESIGN §4).
        self.engine.update(self.start.elapsed().as_secs_f64());
        // One submit per frame carrying however many ticks that frame ran, in
        // tick order — the tick already batched them (DESIGN §8; see
        // `runt_core::audio` for why the flush sits where it does). Before the
        // render rather than after, so a sound is on its way while the picture
        // is still being drawn.
        self.engine.sim_mut().drain_audio(self.audio.as_mut());

        // The game's own frame work, between the tick and the draw (see
        // `RunConfig::with_frame`). This is where an offscreen viewport is
        // rendered: the tick that decided what the card should show has run,
        // and the HUD quad that samples the card has not been drawn yet.
        //
        // Split borrow rather than `if let Some(f) = &mut self.on_frame` inside
        // a method call: the hook and the engine are sibling fields, and the
        // hook is handed the engine, not the host — a hook cannot reach the
        // surface, the window or the audio device, which is what keeps DESIGN
        // §2's line ("hosts contain no engine logic") pointing the same way
        // through the seam.
        if let Some(on_frame) = self.on_frame.as_mut() {
            on_frame(&mut self.engine);
        }

        self.engine
            .render(&view, self.config.width, self.config.height);

        self.engine.queue().present(frame);
        self.sync_status();

        // Anything the frame generated goes to storage *after* it was
        // presented, never before: the first frame is the slow one (it is the
        // one that ran the bake), and making the player wait on a database to
        // see it would trade the exact thing the cache exists to buy. On web
        // this hands the write to the browser and returns; natively it is
        // nothing at all, the bytes are already on disk.
        //
        // Per frame rather than once, because a cache write is not a start-up
        // event — a level loaded ten minutes in bakes too. It costs one lock
        // and a zero check when there is nothing to say.
        self.cache.flush();
    }

    /// Read every connected pad and push whatever moved since the last frame.
    ///
    /// Cheap and silent when nothing happened: the poll is a state read, and
    /// [`gamepad::PadDiffer`] emits nothing for a pad at rest.
    fn poll_pads(&mut self) {
        let Some(pads) = self.pads.as_mut() else {
            return;
        };
        let snapshot = pads.poll();
        let engine = &mut self.engine;
        self.pad_diff.diff(snapshot, |event| engine.push_input(event));
    }

    /// Translate one winit window event into engine input. Returns `true` if it
    /// was an input event (handled here and nowhere else) — translation is the
    /// host's entire job, the engine never sees a winit type (DESIGN §2).
    fn handle_input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                let PhysicalKey::Code(code) = event.physical_key else {
                    // Unidentified/native keycode: still worth reporting, as
                    // the catch-all.
                    self.engine.push_input(match event.state {
                        ElementState::Pressed => InputEvent::KeyDown(Key::Other),
                        ElementState::Released => InputEvent::KeyUp(Key::Other),
                    });
                    return true;
                };
                let key = input::translate_key(code);
                self.engine.push_input(match event.state {
                    ElementState::Pressed => InputEvent::KeyDown(key),
                    ElementState::Released => InputEvent::KeyUp(key),
                });
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                let now = (position.x, position.y);
                if let Some(prev) = self.last_cursor {
                    self.engine.push_input(InputEvent::MouseMove {
                        dx: (now.0 - prev.0) as f32,
                        dy: (now.1 - prev.1) as f32,
                    });
                }
                self.last_cursor = Some(now);
                true
            }
            // No position to difference against next time the pointer appears.
            WindowEvent::CursorLeft { .. } => {
                self.last_cursor = None;
                true
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.engine.push_input(InputEvent::MouseButton {
                    button: input::translate_button(*button),
                    pressed: *state == ElementState::Pressed,
                });
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let dy = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    // Roughly one notch per 50 px, so both paths land in the
                    // same units on the engine side.
                    MouseScrollDelta::PixelDelta(p) => (p.y / 50.0) as f32,
                };
                self.engine.push_input(InputEvent::Wheel { dy });
                true
            }
            // Touch. winit delivers these on Android, iOS, Web and Windows touch
            // screens, and the translation is identical on all of them because
            // it happens in logical pixels.
            //
            // Every contact is forwarded raw, always; the virtual stick is an
            // *extra* the host may also synthesise from it (see
            // [`input::touch_events`]).
            WindowEvent::Touch(touch) => {
                let scale = self.window.scale_factor().max(f64::MIN_POSITIVE);
                let x = (touch.location.x / scale) as f32;
                let y = (touch.location.y / scale) as f32;
                let (raw, drive) =
                    input::touch_events(self.stick.as_mut(), touch.id, touch.phase, x, y);
                self.engine.push_input(raw);
                if let Some(drive) = drive {
                    self.engine.push_input(drive);
                }
                true
            }
            // Alt-tab, a backgrounded tab, a phone call: the key-up that would
            // normally arrive never will, so tell the engine to let go of
            // everything rather than leaving the ball rolling into the sunset.
            WindowEvent::Focused(false) => {
                if let Some(stick) = self.stick.as_mut() {
                    stick.reset();
                }
                // `FocusLost` already drops every held pad button, centres both
                // sticks, releases both triggers and ends every live touch
                // engine-side, so the differ only has to agree that it did —
                // pushing the releases as well would double every one of them
                // into the trace. Exactly the stance `VirtualStick::reset`
                // takes.
                self.pad_diff.forget();
                self.engine.push_input(InputEvent::FocusLost);
                true
            }
            _ => false,
        }
    }
}

/// Whether this event is the native quit gesture: Escape, pressed, not a repeat.
///
/// Native only, and checked *after* the event has been pushed at the engine, so
/// a recording still contains the keystroke that ended it.
///
/// On the web there is nothing to exit — a page is closed, not quit — and
/// browsers already give Escape a meaning (leaving fullscreen). So the wasm
/// build simply lets the key through as an ordinary [`Key::Escape`].
#[cfg(not(target_arch = "wasm32"))]
fn quit_requested(event: &WindowEvent) -> bool {
    matches!(
        event,
        WindowEvent::KeyboardInput {
            event: winit::event::KeyEvent {
                physical_key: PhysicalKey::Code(winit::keyboard::KeyCode::Escape),
                state: ElementState::Pressed,
                repeat: false,
                ..
            },
            ..
        }
    )
}

// ---------------------------------------------------------------------------
// Application / event loop
// ---------------------------------------------------------------------------

enum UserEvent {
    // Only ever sent from the wasm graphics-init future; natively the host is
    // built inline and this arm is unreachable.
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    HostReady(Host),
}

struct App {
    host: Option<Host>,
    /// Taken by the first `resumed`; `None` afterwards, which is also what
    /// stops a second resume from rebuilding the world (the setup hook is
    /// `FnOnce` and the sim it configured is already running).
    run: Option<RunConfig>,
    /// Held out of `run` so it survives into `run_with`'s tail.
    on_exit: Option<SetupFn>,
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    proxy: Option<EventLoopProxy<UserEvent>>,
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.host.is_some() {
            return;
        }
        let Some(run) = self.run.take() else {
            return; // Already resumed once; graphics init is in flight.
        };
        let attrs = Window::default_attributes().with_title(&run.title);

        #[cfg(target_arch = "wasm32")]
        let attrs = {
            use winit::platform::web::WindowAttributesExtWebSys;
            attrs.with_append(true)
        };

        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));

        #[cfg(not(target_arch = "wasm32"))]
        {
            match pollster::block_on(Host::new(window, run)) {
                Ok(h) => self.host = Some(h),
                Err(e) => {
                    log::error!("graphics init failed: {e}");
                    event_loop.exit();
                }
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            let proxy = self.proxy.take().expect("proxy");
            wasm_bindgen_futures::spawn_local(async move {
                match Host::new(window, run).await {
                    Ok(h) => {
                        let _ = proxy.send_event(UserEvent::HostReady(h));
                    }
                    Err(e) => {
                        log::error!("graphics init failed: {e}");
                        show_fatal(&e);
                    }
                }
            });
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        #[cfg_attr(not(target_arch = "wasm32"), allow(unused_mut))]
        let UserEvent::HostReady(mut h) = event;
        // Apply the real viewport size immediately; the surface was created at
        // whatever tiny default winit gave the fresh canvas.
        #[cfg(target_arch = "wasm32")]
        {
            let ls = browser_logical_size();
            let dpr = web_sys::window()
                .map(|w| w.device_pixel_ratio())
                .unwrap_or(1.0);
            let _ = h.window.request_inner_size(ls);
            h.resize((ls.width * dpr) as u32, (ls.height * dpr) as u32);
        }
        h.window.request_redraw();
        self.host = Some(h);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(h) = self.host.as_mut() else {
            return;
        };
        let handled = h.handle_input(&event);
        // Escape quits, natively. Deliberately after `handle_input`, so the
        // engine has already seen the keystroke: `--record` writes its trace
        // from `on_exit`, and a run that ends on a key the trace does not
        // contain is a run that does not replay. This is also what closes the
        // gap where the only way out was the window's close button.
        #[cfg(not(target_arch = "wasm32"))]
        if quit_requested(&event) {
            log::info!("escape pressed; exiting");
            event_loop.exit();
            return;
        }
        if handled {
            return;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                h.resize(size.width, size.height);
                h.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                h.render();
                h.window.request_redraw();
            }
            _ => {}
        }
    }
}

/// Run the engine's own demo scene. `runt-native` and the root `index.html`.
pub fn run() {
    run_with(RunConfig::engine_demo());
}

/// Open a window (or take over the canvas) and run `config` until it closes.
///
/// The entry point for *any* runt program: the engine demo, `demo/ball`, and
/// whatever comes next all reach the platform through this one function.
pub fn run_with(mut config: RunConfig) {
    let event_loop = EventLoop::<UserEvent>::with_user_event()
        .build()
        .expect("build event loop");
    let proxy = event_loop.create_proxy();
    let mut app = App {
        host: None,
        on_exit: config.on_exit.take(),
        run: Some(config),
        proxy: Some(proxy),
    };
    event_loop.run_app(&mut app).expect("run app");

    // Native only in practice: on web `run_app` diverges into the browser's
    // event loop and this line is never reached, which is exactly what
    // `on_exit`'s docs promise.
    if let (Some(on_exit), Some(host)) = (app.on_exit.take(), app.host.as_mut()) {
        on_exit(host.engine.sim_mut());
    }
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
fn browser_logical_size() -> winit::dpi::LogicalSize<f64> {
    let win = web_sys::window().expect("no window");
    let w = win
        .inner_width()
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(960.0);
    let h = win
        .inner_height()
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(540.0);
    winit::dpi::LogicalSize::new(w.max(1.0), h.max(1.0))
}

#[cfg(target_arch = "wasm32")]
fn show_fatal(msg: &str) {
    if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
        if let Some(body) = doc.body() {
            let el = doc.create_element("div").unwrap();
            el.set_attribute(
                "style",
                "position:fixed;inset:0;display:flex;align-items:center;\
                 justify-content:center;padding:2rem;color:#e6e6e6;\
                 font:14px system-ui;text-align:center;background:#0d0f14",
            )
            .ok();
            el.set_text_content(Some(msg));
            body.append_child(&el).ok();
        }
    }
}

/// Panic hook + console logger. Every wasm entry point wants these and none of
/// them wants to remember the two crate names, so the host owns the boilerplate
/// even though the `#[wasm_bindgen(start)]` itself cannot live here for a game
/// (see [`wasm_start`]).
#[cfg(target_arch = "wasm32")]
pub fn wasm_init() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).ok();
}

/// The engine demo's browser entry point.
///
/// Behind the default `wasm-entry` feature because **a wasm module may have
/// exactly one `#[wasm_bindgen(start)]`**, and it is collected from dependency
/// rlibs too. A game crate that is itself the wasm target therefore depends on
/// `runt-app` with `default-features = false` and declares its own start — see
/// `demo/ball/src/lib.rs`. Building `crates/runt-app` directly (the root
/// `index.html`) keeps the default and gets this one.
#[cfg(all(target_arch = "wasm32", feature = "wasm-entry"))]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn wasm_start() {
    wasm_init();
    run();
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    use runt_core::{RenderTarget, UiBatch, UiQuad};

    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
    const SIZE: u32 = 64;

    /// The default has to stay "stick on", because every program written before
    /// raw touch existed relies on one thumb driving
    /// [`Input::drive`](runt_core::Input::drive) and none of them will be
    /// rewritten to ask for it.
    #[test]
    fn the_virtual_stick_is_on_until_a_game_says_otherwise() {
        assert!(RunConfig::engine_demo().virtual_stick);
        assert!(RunConfig::new("t", "").virtual_stick);
        assert!(
            !RunConfig::engine_demo()
                .without_virtual_stick()
                .virtual_stick
        );
        // …and it composes with the other builders rather than resetting them.
        let config = RunConfig::engine_demo()
            .without_virtual_stick()
            .with_quality(0.5);
        assert!(!config.virtual_stick);
        assert_eq!(config.quality, Some(0.5));
    }

    /// What the frame hook is *for*, driven through the same sequence
    /// [`Host::render`] drives it in: tick → hook → draw.
    ///
    /// The call site itself cannot be tested here — [`Host`] owns a `Window` and
    /// a `Surface` and there is no headless one — so this stands the two claims
    /// that survive without a window: the hook is called once per frame with the
    /// whole engine, and it is called at a point where its offscreen target is
    /// already resident for the draw that follows. The one line of ordering
    /// inside `Host::render` is the part held by review.
    #[test]
    fn the_frame_hook_runs_once_a_frame_between_the_tick_and_the_draw() {
        let mut engine = match pollster::block_on(runt_core::Engine::headless(FORMAT)) {
            Ok(e) => e,
            Err(e) => return eprintln!("SKIP frame hook (no GPU adapter): {e}"),
        };
        let target = engine
            .device()
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("frame hook target"),
                size: wgpu::Extent3d {
                    width: SIZE,
                    height: SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());

        // A HUD in the game's world, so the frame has something to mirror —
        // `Engine::render` copies the batch out of the world, which is what
        // makes `ui_quad_count` a witness to whether the draw has happened yet.
        let card = RenderTarget(0);
        let mut hud = UiBatch::new();
        hud.set_texture(Some(card.handle()));
        hud.textured([0.0, 0.0, 32.0, 32.0], [0.0, 0.0, 1.0, 1.0], [1.0; 4]);
        engine.sim_mut().world_mut().insert_resource(hud);

        // What the hook saw, per call: (its own call index, the engine's tick
        // count, how many HUD quads the *renderer* was holding).
        let seen = Rc::new(RefCell::new(Vec::new()));
        let log = seen.clone();
        // The tutorial world: a second `Sim`, owned by the closure, which is
        // the entire reason the hook is `FnMut` and takes `&mut Engine`.
        let mut demo = runt_core::Sim::new();
        let mut config = RunConfig::engine_demo().with_frame(move |engine| {
            let calls = log.borrow().len();
            let hud = engine.renderer().ui_quad_count();
            log.borrow_mut().push((calls, engine.tick_count(), hud));
            demo.update(calls as f64 * runt_core::TICK_DT);
            engine.render_to_texture(card, &mut demo, 48, 32);
        });

        assert!(
            RunConfig::engine_demo().on_frame.is_none(),
            "a program that never asks for a frame hook must not have one"
        );
        let mut on_frame = config.on_frame.take().expect("with_frame stored the hook");

        // Three frames of `Host::render`'s body, in its order.
        for frame in 0..3u32 {
            engine.update(frame as f64 * 4.0 * runt_core::TICK_DT);
            on_frame(&mut engine);
            engine.render(&target, SIZE, SIZE);
        }

        let seen = seen.borrow();
        assert_eq!(seen.len(), 3, "the hook runs once per frame: {seen:?}");
        // Frame 0 is the one that can tell the order: the renderer is holding
        // no HUD yet, because `Engine::render` has not run. Had the hook been
        // called after the draw, the first thing it saw would have been this
        // frame's quads — and its viewport would have been one frame stale in
        // every card, forever.
        assert_eq!(seen[0].2, 0, "the hook ran after the draw");
        assert!(seen[1].1 > 0, "the hook runs after the tick: {seen:?}");
        assert_eq!(seen[1].2, 1, "later frames see the previous frame's HUD");

        // And the thing the hook exists to do actually landed: the viewport is
        // resident, at its own size, under the handle the HUD names.
        assert_eq!(engine.renderer().render_target_size(card), Some((48, 32)));
        assert!(engine.renderer().textures().contains(card.handle()));
        assert_eq!(engine.renderer().ui_quad_count(), 1);
    }

    /// A quad-and-atlas host still has a whole `UiBatch` to hand over: the
    /// batch's texture runs are opt-in, and the two-line construction below is
    /// the whole migration for a game that used to build one by struct literal.
    #[test]
    fn a_batch_is_built_rather_than_declared() {
        let mut batch = UiBatch::new();
        batch.atlas = Some(runt_core::TextureHandle(1));
        batch.push(UiQuad::solid([0.0, 0.0, 4.0, 4.0], [1.0; 4]));
        assert_eq!(batch.len(), 1);
        assert_eq!(batch.runs().count(), 1, "one atlas is still one run");
    }
}
