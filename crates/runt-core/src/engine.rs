//! [`Engine`] — the whole thing a host talks to: sim + renderer.
//!
//! A host owns a window and a clock and nothing else. It pushes translated
//! input, calls [`Engine::update`] with its wall time, and hands
//! [`Engine::render`] a `TextureView`. Everything between those three calls is
//! engine business.

use glam::Mat4;

use crate::input::InputEvent;
use crate::sim::Sim;
use crate::Renderer;

pub struct Engine {
    sim: Sim,
    renderer: Renderer,
}

impl Engine {
    /// Build on a device/queue the host already owns.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        target_format: wgpu::TextureFormat,
    ) -> Engine {
        Engine {
            sim: Sim::new(),
            renderer: Renderer::new(device, queue, target_format),
        }
    }

    /// Build with no surface and no display handle — tests, bakes, the editor
    /// bridge.
    pub async fn headless(target_format: wgpu::TextureFormat) -> Result<Engine, String> {
        Ok(Engine {
            sim: Sim::new(),
            renderer: Renderer::headless(target_format).await?,
        })
    }

    /// Build with a non-default tick rate (DESIGN §12 step 2's tick-rate
    /// toggle). See [`Sim::with_tick_rate`].
    pub fn with_tick_rate(
        device: wgpu::Device,
        queue: wgpu::Queue,
        target_format: wgpu::TextureFormat,
        hz: f64,
    ) -> Engine {
        Engine {
            sim: Sim::with_tick_rate(hz),
            renderer: Renderer::new(device, queue, target_format),
        }
    }

    // -- host surface -------------------------------------------------------

    /// Buffer a host input event; it is consumed at the next tick boundary.
    pub fn push_input(&mut self, event: InputEvent) {
        self.sim.push_input(event);
    }

    /// Advance the sim to `elapsed_seconds` of host wall time, returning the
    /// number of ticks that ran. The engine never reads a clock itself, so this
    /// value is the only time source in the system (DESIGN §4).
    pub fn update(&mut self, elapsed_seconds: f64) -> u32 {
        self.sim.update(elapsed_seconds)
    }

    /// Draw one frame into `view`, which must be [`target_format`] and
    /// `width` × `height`. Uses the interpolation alpha left by the last
    /// [`update`](Engine::update).
    ///
    /// [`target_format`]: Engine::target_format
    pub fn render(&mut self, view: &wgpu::TextureView, width: u32, height: u32) {
        let model = self.sim.demo_model_matrix();
        self.renderer.render(view, width, height, model);
    }

    /// Draw one frame with an explicit model matrix, bypassing the sim. For the
    /// editor bridge and golden-image tests that want a fixed pose.
    pub fn render_with_model(
        &mut self,
        view: &wgpu::TextureView,
        width: u32,
        height: u32,
        model: Mat4,
    ) {
        self.renderer.render(view, width, height, model);
    }

    // -- accessors ----------------------------------------------------------

    pub fn sim(&self) -> &Sim {
        &self.sim
    }

    pub fn sim_mut(&mut self) -> &mut Sim {
        &mut self.sim
    }

    pub fn renderer(&self) -> &Renderer {
        &self.renderer
    }

    pub fn renderer_mut(&mut self) -> &mut Renderer {
        &mut self.renderer
    }

    /// Interpolation alpha from the last [`update`](Engine::update), in `[0,1)`.
    pub fn alpha(&self) -> f32 {
        self.sim.alpha()
    }

    /// Ticks completed since construction.
    pub fn tick_count(&self) -> u64 {
        self.sim.tick_count()
    }

    pub fn device(&self) -> &wgpu::Device {
        self.renderer.device()
    }

    pub fn queue(&self) -> &wgpu::Queue {
        self.renderer.queue()
    }

    pub fn target_format(&self) -> wgpu::TextureFormat {
        self.renderer.target_format()
    }
}
