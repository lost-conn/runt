//! [`Engine`] — the whole thing a host talks to: sim + renderer.
//!
//! A host owns a window and a clock and nothing else. It pushes translated
//! input, calls [`Engine::update`] with its wall time, and hands
//! [`Engine::render`] a `TextureView`. Everything between those three calls is
//! engine business.

use crate::input::InputEvent;
use crate::sim::{Sim, SimConfig};
use crate::Renderer;

pub struct Engine {
    sim: Sim,
    renderer: Renderer,
    /// A world with no camera draws nothing; say so once rather than every
    /// frame for as long as it stays broken.
    warned_no_camera: bool,
    /// The last wall time [`update`](Engine::update) was given — the render
    /// clock (`FrameUniform::time.x`), forwarded to the renderer at draw time.
    ///
    /// Not the sim's clock: this is the host's raw elapsed seconds, un-quantized
    /// by ticks, so a shader driven from it moves smoothly between them. Nothing
    /// in a `FixedSim` can reach it, which is the property that lets render-side
    /// animation exist at all without putting a replay at risk (DESIGN §4).
    render_seconds: f64,
}

impl Engine {
    /// Build on a device/queue the host already owns.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        target_format: wgpu::TextureFormat,
    ) -> Engine {
        Engine::assembled(Sim::new(), Renderer::new(device, queue, target_format))
    }

    /// Build with no surface and no display handle — tests, bakes, the editor
    /// bridge.
    pub async fn headless(target_format: wgpu::TextureFormat) -> Result<Engine, String> {
        Ok(Engine::assembled(
            Sim::new(),
            Renderer::headless(target_format).await?,
        ))
    }

    /// Build with a non-default tick rate (DESIGN §12 step 2's tick-rate
    /// toggle). See [`Sim::with_tick_rate`].
    pub fn with_tick_rate(
        device: wgpu::Device,
        queue: wgpu::Queue,
        target_format: wgpu::TextureFormat,
        hz: f64,
    ) -> Engine {
        Engine::assembled(
            Sim::with_tick_rate(hz),
            Renderer::new(device, queue, target_format),
        )
    }

    /// Build with an explicit [`SimConfig`] — the quality tier, the cache store
    /// and the scene a host actually wants (DESIGN §6).
    pub fn from_config(
        device: wgpu::Device,
        queue: wgpu::Queue,
        target_format: wgpu::TextureFormat,
        config: SimConfig,
    ) -> Engine {
        Engine::assembled(
            Sim::from_config(config),
            Renderer::new(device, queue, target_format),
        )
    }

    /// As [`headless`](Engine::headless), with an explicit [`SimConfig`].
    pub async fn headless_with_config(
        target_format: wgpu::TextureFormat,
        config: SimConfig,
    ) -> Result<Engine, String> {
        Ok(Engine::assembled(
            Sim::from_config(config),
            Renderer::headless(target_format).await?,
        ))
    }

    /// The one place a `Sim` and a `Renderer` become an `Engine`.
    ///
    /// Every constructor funnels through here so that the load-time work which
    /// needs *both* halves — baking the scene's procedural textures (DESIGN §7)
    /// — cannot be forgotten by whichever constructor a host happens to use.
    fn assembled(sim: Sim, renderer: Renderer) -> Engine {
        let mut engine = Engine {
            sim,
            renderer,
            warned_no_camera: false,
            render_seconds: 0.0,
        };
        engine.bake_scene_textures();
        engine
    }

    /// Bake every texture the loaded scene registered (DESIGN §7).
    ///
    /// > *Baked (baseline): rendered once to an RGBA8 texture at tier-scaled
    /// > resolution at **load time**.* — §7
    ///
    /// Run automatically at construction. Call it again after loading a scene
    /// by hand, so the disk cache is consulted before the first frame — without
    /// it the renderer still bakes lazily on first draw, to the same pixels,
    /// just without the cache and inside a frame.
    pub fn bake_scene_textures(&mut self) {
        let work: Vec<(crate::texture::TextureSpec, u32)> = self
            .sim
            .texture_library()
            .iter()
            .map(|(_, spec, resolution)| (spec.clone(), resolution))
            .collect();
        if work.is_empty() {
            return;
        }
        // Split the borrow by field: the store lives in the sim's `GenCache`
        // and the bake lives in the renderer, and they are disjoint, so no
        // cloning (and no unsafe) is needed to hold both.
        let Engine { sim, renderer, .. } = self;
        let store = sim.cache_store();
        for (spec, resolution) in &work {
            renderer.bake_texture(spec, *resolution, store);
        }
        log::info!(
            "baked {} procedural texture(s), {} resident",
            work.len(),
            renderer.textures().len()
        );
    }

    /// Whether textured draws evaluate their spec per pixel (DESIGN §7's live
    /// path) instead of sampling its bake.
    pub fn live_textures(&self) -> bool {
        self.sim.live_textures()
    }

    /// Switch every textured draw between §7's baked and live paths — v1's perf
    /// gate, held open by hand until §11's probe exists to hold it.
    ///
    /// Cheap enough to call every frame: both variants live in the same
    /// pipeline cache and the bind group does not change, so a flip costs one
    /// pipeline swap the sort order was going to pay anyway. Nothing in the sim
    /// can observe it.
    pub fn set_live_textures(&mut self, live: bool) {
        self.sim.set_live_textures(live);
    }

    /// The fraction of the host's resolution the scene is drawn at
    /// (DESIGN §11). 1.0 unless something set it.
    pub fn render_scale(&self) -> f32 {
        self.sim.render_scale().get()
    }

    /// Draw the scene at `scale` × whatever size [`render`](Engine::render) is
    /// given, then upscale it with a nearest filter — Godot's resolution scale,
    /// and the cheapest lever there is on a device that cannot afford its own
    /// pixel count (0.5 is a quarter of the fragments).
    ///
    /// Clamped into `[RenderScale::MIN, RenderScale::MAX]`; a NaN resolves to
    /// 1.0. At 1.0 the frame is drawn exactly as it was before this existed,
    /// with no internal target and no blit.
    ///
    /// Cheap enough to call every frame, and — like
    /// [`set_live_textures`](Engine::set_live_textures) — invisible to the sim:
    /// no system reads it, so no fingerprint can move when it changes. A game
    /// that would rather bind it to a key writes the
    /// [`RenderScale`](crate::ecs::RenderScale) resource from a `FixedSim`
    /// system instead; this is the host-side door to the same value.
    pub fn set_render_scale(&mut self, scale: f32) {
        self.sim.set_render_scale(scale);
    }

    /// The pixel size the scene is actually drawn at, for a host view of
    /// `width` × `height` — what a status line should report.
    ///
    /// Equal to `(width, height)` at scale 1.0. Pure: it asks nothing of the
    /// GPU and allocates nothing, so a host may call it before the first frame.
    pub fn render_size(&self, width: u32, height: u32) -> (u32, u32) {
        self.sim.render_scale().size(width, height)
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
        if elapsed_seconds.is_finite() {
            self.render_seconds = elapsed_seconds;
        }
        self.sim.update(elapsed_seconds)
    }

    /// Aim the screen-space phase circle every
    /// [`PHASE_CIRCLE`](crate::MaterialVariant::PHASE_CIRCLE) material reads
    /// (DESIGN §5). `center` is NDC, `radius` is in NDC-Y units, `strength` is
    /// `0..1`.
    ///
    /// The host-side door, exactly like
    /// [`set_render_scale`](Engine::set_render_scale): it is a *render* value,
    /// no system reads it, and no fingerprint can move when it changes.
    ///
    /// **Not the normal path for a game.** Projecting a world point needs the
    /// frame's own view-projection and aspect, which do not exist until
    /// [`render`](Engine::render) is running — so a game writes the
    /// [`PhaseFx`](crate::ecs::PhaseFx) resource, in world units, and the frame
    /// resolves it. This is the door for a host or an editor that already has
    /// screen coordinates in hand; a world carrying `PhaseFx` overwrites it on
    /// the next frame.
    pub fn set_phase_fx(&mut self, center: glam::Vec2, radius: f32, strength: f32) {
        self.renderer.set_phase_fx(center, radius, strength);
    }

    /// The phase circle as the next frame will draw it: `(center, radius,
    /// strength)`.
    pub fn phase_fx(&self) -> (glam::Vec2, f32, f32) {
        self.renderer.phase_fx()
    }

    /// Draw one frame into `view`, which must be [`target_format`] and
    /// `width` × `height`. Uses the interpolation alpha left by the last
    /// [`update`](Engine::update).
    ///
    /// The whole frame in order: the sim produces a sorted draw list and the
    /// camera's view-projection for this viewport, then the renderer uploads,
    /// clears and draws it. The host contributes a size and a texture — no view
    /// matrix, no model matrix, no scene knowledge (DESIGN §5).
    ///
    /// [`target_format`]: Engine::target_format
    pub fn render(&mut self, view: &wgpu::TextureView, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        // The host rectangle's aspect, not the internal target's: a scaled frame
        // is stretched back over the full view, so this is the projection that
        // belongs in it either way (see `Renderer::render_scaled`).
        let aspect = width as f32 / height as f32;
        let scale = self.sim.render_scale();
        // The render clock, forwarded rather than measured — the engine owns no
        // clock (DESIGN §4). `as f32` is lossy after a few hours of uptime; a
        // shader animating on a value that has stopped advancing smoothly is a
        // real (and known) limit, and the fix when something needs it is to
        // wrap the seconds, not to hand the GPU an f64 it cannot take.
        self.renderer
            .set_render_clock(self.render_seconds as f32, self.sim.alpha());

        // The HUD the world holds right now (plan D11, `crate::ui`). Mirrored
        // into the renderer every frame — the batch is rebuilt each frame by
        // whoever owns it, so "what the world says" is always the whole truth,
        // and a world that says nothing draws no UI pass at all. A world with
        // no `UiBatch` in it (a hand-built one, an older host) is the same
        // case as an empty batch.
        match self.sim.world().get_resource::<crate::ui::UiBatch>() {
            Some(batch) => self.renderer.set_ui_batch(batch),
            None => self.renderer.set_ui_quads(&[], None),
        }

        // …and the atlas those quads sample, if the game drew one itself
        // (`ui::UiAtlasImage`). Uploaded on the first frame it is present and
        // ignored on every frame after: `upload_ui_atlas` is a hash lookup once
        // the handle is resident, which is what lets this sit in the frame path
        // rather than in a load hook the port would have to reach for.
        if let Some(image) = self.sim.world().get_resource::<crate::ui::UiAtlasImage>() {
            if image.is_valid() {
                self.renderer.upload_ui_atlas(
                    image.handle,
                    image.width,
                    image.height,
                    &image.rgba,
                );
            }
        }

        // The **inbound** half of the UI seam (`ecs::Viewport`): tell the world
        // how big the screen its HUD is being laid out on is. Written here, and
        // therefore read by the *next* tick — one frame stale, which is what a
        // resize costs and what a HUD cannot see.
        let seen = crate::ecs::Viewport::new(width, height);
        if self.sim.world().get_resource::<crate::ecs::Viewport>() != Some(&seen) {
            self.sim.world_mut().insert_resource(seen);
        }

        // The phase circle the world is asking for (D1, `ecs::PhaseFx`),
        // resolved against *this* frame's camera and aspect. Read before
        // `frame_params` only so the borrow of the world ends first; a world
        // that says nothing leaves whatever the host last set alone, which is
        // what keeps `Engine::set_phase_fx` a usable door of its own.
        let phase_fx = self.sim.world().get_resource::<crate::ecs::PhaseFx>().copied();

        let Some(frame) = self.sim.frame_params(aspect) else {
            if !self.warned_no_camera {
                log::warn!("no camera entity in the world; nothing will be drawn");
                self.warned_no_camera = true;
            }
            // Still render, so a host sees an empty sky rather than garbage.
            // With no camera there is no view ray to speak of, so the gradient
            // resolves to a flat horizon-colored frame.
            self.renderer.render_scaled(
                view,
                width,
                height,
                scale,
                &crate::FrameParams::default(),
                &[],
                self.sim.mesh_library(),
                self.sim.texture_library(),
            );
            return;
        };

        if let Some(fx) = phase_fx {
            let (center, radius) = crate::ecs::project_phase_fx(&frame.view_proj, aspect, &fx);
            self.renderer.set_phase_fx(center, radius, fx.strength);
        }

        let draws = self.sim.draw_list();
        self.renderer.render_scaled(
            view,
            width,
            height,
            scale,
            &frame,
            &draws,
            self.sim.mesh_library(),
            self.sim.texture_library(),
        );
    }

    /// Draw **another** [`Sim`]'s scene into an offscreen texture, and return
    /// the handle a UI quad samples it by (see
    /// [`RenderTarget`](crate::RenderTarget)).
    ///
    /// [`render`](Engine::render) for a second world, minus everything that
    /// belongs to the screen rather than to a scene: no HUD is mirrored, no
    /// [`Viewport`](crate::ecs::Viewport) is written back, no
    /// [`PhaseFx`](crate::ecs::PhaseFx) is projected, no render scale is
    /// applied. What is left is the part a viewport is: aspect from the
    /// target's own size, the demo world's camera, the demo world's draw list.
    ///
    /// # Why the demo world is a whole second `Sim`
    ///
    /// It has its own tick clock, its own entities and its own libraries, so a
    /// tutorial card can run a scripted loop while the game is paused — and,
    /// more importantly, nothing it does can reach the game's world. Godot's
    /// `SubViewport` with `own_world_3d` is the same decision. The renderer is
    /// shared, and safely so, because every handle in it is content-addressed:
    /// two worlds that generated the same cube share one upload and two that
    /// did not cannot collide (see [`RenderTarget`](crate::RenderTarget)).
    ///
    /// The demo `Sim` is the caller's to own and to [`update`](Sim::update);
    /// this only draws it. `&mut` because building a draw list caches a query
    /// in the world, which is what makes doing it every frame cheap.
    pub fn render_to_texture(
        &mut self,
        target: crate::RenderTarget,
        sim: &mut Sim,
        width: u32,
        height: u32,
    ) -> crate::texture::TextureHandle {
        let (width, height) = (width.max(1), height.max(1));
        let aspect = width as f32 / height as f32;
        // A world with no camera still draws its sky and nothing else, exactly
        // as `render` does with the game's world: a viewport that is briefly
        // empty beats one full of geometry projected through an identity
        // matrix. No warning, unlike `render` — the demo world is the caller's,
        // and a card that has not spawned its camera yet is a state a tutorial
        // legitimately passes through.
        let (frame, draws) = match sim.frame_params(aspect) {
            Some(frame) => {
                let draws = sim.draw_list();
                (frame, draws)
            }
            None => (crate::FrameParams::default(), Vec::new()),
        };
        self.renderer.render_to_texture(
            target,
            width,
            height,
            &frame,
            &draws,
            sim.mesh_library(),
            sim.texture_library(),
        );
        target.handle()
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
