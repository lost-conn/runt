//! runt player host — native window + web canvas.
//!
//! Everything here is windowing, surface management and presentation. The
//! engine itself is `runt-core`, which never sees a `Window` or a `Surface`:
//! this host acquires the frame's `TextureView` and hands it to
//! [`runt_core::Renderer::render`].

use std::sync::Arc;

use runt_core::{Engine, InputEvent, Key};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::keyboard::PhysicalKey;
use winit::window::{Window, WindowId};

mod input;

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
}

impl Host {
    async fn new(window: Arc<Window>) -> Result<Host, String> {
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

        let engine = Engine::new(device.clone(), queue, format);

        Ok(Host {
            window,
            surface,
            device,
            config,
            engine,
            start: web_time::Instant::now(),
            last_cursor: None,
        })
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

        // The host owns the clock; the engine is handed wall time and does its
        // own fixed-tick accounting (DESIGN §4).
        self.engine.update(self.start.elapsed().as_secs_f64());
        self.engine
            .render(&view, self.config.width, self.config.height);

        self.engine.queue().present(frame);
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
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Application / event loop
// ---------------------------------------------------------------------------

enum UserEvent {
    HostReady(Host),
}

struct App {
    host: Option<Host>,
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    proxy: Option<EventLoopProxy<UserEvent>>,
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.host.is_some() {
            return;
        }
        let attrs = Window::default_attributes().with_title("runt");

        #[cfg(target_arch = "wasm32")]
        let attrs = {
            use winit::platform::web::WindowAttributesExtWebSys;
            attrs.with_append(true)
        };

        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));

        #[cfg(not(target_arch = "wasm32"))]
        {
            match pollster::block_on(Host::new(window)) {
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
                match Host::new(window).await {
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
        if h.handle_input(&event) {
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

pub fn run() {
    let event_loop = EventLoop::<UserEvent>::with_user_event()
        .build()
        .expect("build event loop");
    let proxy = event_loop.create_proxy();
    let mut app = App {
        host: None,
        proxy: Some(proxy),
    };
    event_loop.run_app(&mut app).expect("run app");
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

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn wasm_start() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).ok();
    run();
}
