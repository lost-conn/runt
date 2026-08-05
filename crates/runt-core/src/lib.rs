//! runt engine core — windowless.
//!
//! The core owns a `wgpu::Device`/`Queue` (either handed to it by a host or
//! created headless) and renders into a **caller-provided** `wgpu::TextureView`.
//! It never creates a surface and never presents; that is the host's job.
//! This is what lets the native window, the web canvas, the editor viewport and
//! headless screenshot tests all drive the same engine.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};

pub use runt_mesh as mesh;
use runt_mesh::MeshData;

/// The remappable action layer (DESIGN §4): the engine owns the binding
/// mechanism, each game owns its action vocabulary.
pub mod action;
/// The sim-side audio seam (DESIGN §8). No synthesizer: see the module docs.
pub mod audio;
/// Baking a [`texture::TextureSpec`] to GPU textures (DESIGN §7).
pub mod bake;
pub mod cache;
pub mod camera;
/// Collision v2 (DESIGN §9): capsule character solver, OBBs, layers, queries.
/// Additive — `physics` is untouched by it.
pub mod collide;
pub mod draw;
pub mod ecs;
pub mod engine;
pub mod gen;
pub mod input;
pub mod material;
/// The procedural-noise library, CPU side (DESIGN §7).
pub mod noise;
pub mod physics;
/// Editor-facing reflection (DESIGN §3, §10). Off by default; the wasm player
/// must never pull `bevy_reflect` in.
#[cfg(feature = "reflect")]
pub mod reflect;
pub mod registry;
pub mod scene;
pub mod sim;
pub mod sky;
/// Procedural texture specs and their CPU evaluator (DESIGN §7).
pub mod texture;
pub mod trace;

/// The background pass's WGSL. Standalone (no feature consts), unlike
/// [`material::BASE_SHADER`].
pub const SKY_SHADER: &str = include_str!("sky.wgsl");

/// The render-scale blit's WGSL: one fullscreen triangle, nearest sampler
/// (DESIGN §11). Standalone like [`SKY_SHADER`]; see
/// [`Renderer::render_scaled`].
pub const BLIT_SHADER: &str = include_str!("blit.wgsl");

pub use action::{
    resolve_actions, ActionId, Actions, Bindings, Source, StickDir, DEFAULT_DEADZONE, MAX_ACTIONS,
};
pub use audio::{
    AudioBackend, AudioEvent, AudioOut, Listener, ParamId, PatchId, RecordingBackend, Rolloff,
    SilentBackend, VoiceId,
};
pub use bake::{
    BakeUniform, GpuTexture, TextureBaker, TextureData, TextureRegistry, TextureUniform,
};
pub use cache::{CacheStats, CacheStore, GenCache, NoopCache};
pub use camera::{Camera, FollowCamera};
pub use collide::{
    move_and_slide, CharacterBody, CharacterShape, CollisionLayers, CollisionWorld, Contact,
    ContactKind, MoveResult, ObbCollider, OverlapHit, RayHit, ALL_LAYERS,
};
pub use draw::{DrawItem, FrameParams};
pub use ecs::{
    default_horizon, DemoScene, FixedSim, GeneratorRef, GlobalTransform, Interpolated, Lighting,
    MeshRef, PostSim, QualityTier, RenderScale, Spin, Startup, StatusLine, TerrainSurface,
    TickCount, Transform,
};
pub use engine::Engine;
pub use gen::{GeneratorSpec, Shading};
pub use input::{Input, InputEvent, Key, PadButton, PadStick, PadTrigger};
pub use material::{Material, MaterialVariant};
pub use physics::{
    AabbCollider, Ball, BallController, Grounded, OverlapEvent, RollSpin, SphereCollider, Trigger,
    Velocity,
};
pub use registry::{GpuMesh, MeshHandle, MeshLibrary, MeshRegistry};
pub use runt_mesh::{HeightField, MeshData as Mesh, Quality, TerrainParams, TerrainTint};
pub use scene::{
    load_scene, save_scene, SceneDesc, SceneError, TextureEntry, TEXTURED_SCENE_RON,
};
pub use sim::{Sim, SimConfig, SimSpeed, MAX_ACCUMULATED, TICK_DT};
pub use texture::{
    NoiseSpec, NormalMode, NormalSpec, TextureHandle, TextureLibrary, TextureSpec,
};
pub use noise::{CellReturn, Fractal, Lattice};
pub use trace::{InputTrace, TickEvent};

#[cfg(not(target_arch = "wasm32"))]
pub use cache::NativeDiskCache;

/// The default `env_logger` filter every runt binary installs.
///
/// `info` for our own crates, and everything below wgpu turned down: `wgpu_hal`
/// and `wgpu_core` narrate every Vulkan loader probe and every resource
/// creation at `info`, `naga` announces each module it parses, and `calloop`
/// (winit's Linux event loop) logs at `info` per iteration. A hundred lines of
/// that before the first frame is how a real warning gets missed.
///
/// It is only a *default*: `RUST_LOG` still overrides it wholesale, so
/// `RUST_LOG=wgpu_core=debug` gets the noise back when it is what you want.
/// Validation layers are deliberately left on in debug builds — the filter
/// silences the loader's chatter, not the checks.
pub const DEFAULT_LOG_FILTER: &str = "info,wgpu_hal=warn,wgpu_core=warn,naga=warn,calloop=error";

// ---------------------------------------------------------------------------
// GPU vertex layout
// ---------------------------------------------------------------------------

/// The renderer's interleaved vertex format. `runt-mesh` is struct-of-arrays;
/// this is the only place generation touches the GPU layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 3],
}

impl Vertex {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x3, // position
            1 => Float32x3, // normal
            2 => Float32x2, // uv
            3 => Float32x3, // color
        ],
    };
}

/// Interleave a generated `MeshData` (struct-of-arrays) into [`Vertex`].
pub fn interleave(mesh: &MeshData) -> Vec<Vertex> {
    (0..mesh.positions.len())
        .map(|i| Vertex {
            pos: mesh.positions[i].to_array(),
            normal: mesh.normals[i].to_array(),
            uv: mesh.uvs[i].to_array(),
            color: mesh.colors[i].to_array(),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// GPU uniform blocks
// ---------------------------------------------------------------------------

/// `@group(0)`: constants for the whole frame — camera and light rig.
///
/// Field order is duplicated in `shader.wgsl` and `sky.wgsl`; both restate this
/// block verbatim.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FrameUniform {
    pub view_proj: [[f32; 4]; 4],
    /// The inverse, so the sky pass can rebuild a world-space view ray from a
    /// pixel (see [`sky`]). Inverted once per frame on the CPU rather than per
    /// pixel on the GPU — WGSL has no matrix inverse, and a hand-written one in
    /// the fragment shader would be the most expensive thing in it.
    pub inv_view_proj: [[f32; 4]; 4],
    /// `xyz`: direction towards the key light. `w`: padding (std140 wants the
    /// vec3 padded to 16 bytes anyway, so it may as well be explicit).
    pub light_dir: [f32; 4],
    pub light_color: [f32; 4],
    pub sky_color: [f32; 4],
    pub ground_color: [f32; 4],
    /// The resolved [`Lighting::horizon`] — the sky's middle stop.
    pub horizon_color: [f32; 4],
    /// The screen-space phase circle (DESIGN §5's variant doctrine, the port's
    /// signature effect), read by
    /// [`PHASE_CIRCLE`](MaterialVariant::PHASE_CIRCLE) draws:
    ///
    /// - `xy` — centre in **NDC** (`-1..1`, +Y up),
    /// - `z` — radius in NDC-Y units, i.e. fractions of the half-height, with
    ///   the X offset aspect-corrected so the disc is round on screen,
    /// - `w` — effect strength `0..1`, which drives the edge fringe only.
    ///
    /// Zero (the default) is a circle of no radius: nothing is inside it, so
    /// world geometry is solid and phase geometry is gone. That is both the
    /// resting state and the original's, whose `phase_radius` means the same
    /// thing.
    ///
    /// NDC rather than pixels so the value survives [`RenderScale`]: the frame
    /// is drawn into a smaller target and stretched back over the same
    /// rectangle, and a normalized circle lands in the same place either way.
    pub phase: [f32; 4],
    /// `x` — the **render** clock in seconds; `y` — the interpolation alpha of
    /// the frame being drawn; `zw` — reserved.
    ///
    /// A render-side clock, deliberately: it is the host's wall time, it moves
    /// between ticks, and no system in a `FixedSim` may read it (DESIGN §4).
    /// Animation driven from here is animation that cannot move a replay
    /// fingerprint.
    pub time: [f32; 4],
    /// `xy` — the target's size in pixels; `zw` — its reciprocal.
    ///
    /// The *render* target's, so at [`RenderScale`] below 1.0 this is the
    /// internal target rather than the host's view. That is what makes screen
    /// space mean one thing: a fragment's `position.xy · viewport.zw` is its
    /// place in the frame, `0..1`, whatever resolution the frame was drawn at.
    ///
    /// Not in D1's two vec4s, and here anyway: converting a fragment to NDC
    /// takes the viewport, the phase circle is defined in NDC, and the
    /// alternative was overloading `time.zw` (reserved for a reason) or a
    /// perspective-correct clip-position varying on every variant, paid for by
    /// every draw that has nothing to do with any of this.
    pub viewport: [f32; 4],
}

/// `@group(1)`: one slot per drawn entity, addressed with a dynamic offset.
///
/// Model matrix plus the material's uniform block, exactly as DESIGN §5
/// specifies. 96 bytes; the buffer strides them by the device's
/// `min_uniform_buffer_offset_alignment` (256 under WebGL2 limits).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct InstanceUniform {
    pub model: [[f32; 4]; 4],
    pub base_color: [f32; 4],
    pub params: [f32; 4],
}

/// Instance slots allocated up front; the buffer doubles from here as needed.
const INITIAL_INSTANCE_CAPACITY: u32 = 32;

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// The color the frame is cleared to before the opaque pass.
///
/// Since the sky gradient (§5, [`sky`]) covers every pixel at the head of the
/// pass, this is no longer what an empty frame looks like — it is the value the
/// attachment is initialised to, and nothing but a broken sky pipeline can leave
/// it visible.
pub const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.05,
    g: 0.06,
    b: 0.08,
    a: 1.0,
};

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

/// Draws a sorted [`DrawItem`] list into any `wgpu::TextureView` of
/// `target_format` (DESIGN §5).
///
/// The renderer knows nothing about the ECS. It is handed geometry handles,
/// instance data and one frame block, which is what lets the same code serve the
/// window host, the editor bridge and headless tests — and lets the draw-order
/// rules be tested with no GPU at all.
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    target_format: wgpu::TextureFormat,

    /// One pipeline per shader variant, compiled the first time a draw asks for
    /// it. Never iterated — only looked up — so hashing is harmless (DESIGN §3).
    pipelines: HashMap<MaterialVariant, wgpu::RenderPipeline>,
    pipeline_layout: wgpu::PipelineLayout,
    instance_layout: wgpu::BindGroupLayout,

    frame_buffer: wgpu::Buffer,
    frame_bind_group: wgpu::BindGroup,

    /// [`FrameUniform::phase`] and [`FrameUniform::time`], held between frames.
    ///
    /// Render-side state, kept here rather than in [`FrameParams`] for the same
    /// reason [`RenderScale`] is not a component: nothing in a tick may read
    /// it, so it must not be reachable from the world. `FrameParams` is what
    /// `Extract` produces out of the world; these two are what the *host*
    /// says about the frame it is asking for.
    phase: [f32; 4],
    time: [f32; 4],

    /// The background gradient's pipeline: one fullscreen triangle, `@group(0)`
    /// only, no depth. Not in `pipelines` because it is not a material variant —
    /// there is exactly one sky and it has no key to look up.
    sky_pipeline: wgpu::RenderPipeline,

    /// One uniform buffer for every entity drawn this frame, indexed with
    /// dynamic offsets. `instance_stride` is the device's minimum uniform offset
    /// alignment (256 under WebGL2 limits), not the struct size.
    instance_buffer: wgpu::Buffer,
    instance_bind_group: wgpu::BindGroup,
    instance_stride: u32,
    instance_capacity: u32,
    /// Staging bytes for the instance upload, reused across frames.
    instance_scratch: Vec<u8>,

    meshes: MeshRegistry,

    /// The baked-texture half of §7: the bake pass's pipelines, and handle → GPU
    /// textures. Content-addressed exactly like `meshes`.
    baker: bake::TextureBaker,
    textures: bake::TextureRegistry,

    depth: Option<(u32, u32, wgpu::TextureView)>,

    /// The internal color target a below-native [`RenderScale`] draws into, and
    /// the bind group that reads it back. `None` until the first scaled frame —
    /// a host that never leaves 1.0 allocates nothing and compiles no blit
    /// pipeline (see [`ensure_blit`](Renderer::ensure_blit)).
    offscreen: Option<Offscreen>,
    /// The blit pipeline and its nearest sampler, compiled on first use.
    blit: Option<Blit>,
}

/// The internal color target for a scaled frame, plus the bind group that
/// samples it. Kept whole because the three parts are only ever valid together:
/// the bind group names the view, and the view names the texture.
struct Offscreen {
    width: u32,
    height: u32,
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}

/// The upscale pass: one pipeline, one nearest sampler, one layout.
struct Blit {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl Renderer {
    /// Build on a device/queue the host already owns (window surface, editor,
    /// …). `target_format` must match the views later passed to [`render`].
    ///
    /// [`render`]: Renderer::render
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        target_format: wgpu::TextureFormat,
    ) -> Renderer {
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frame bind layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<FrameUniform>() as u64
                    ),
                },
                count: None,
            }],
        });
        let instance_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instance bind layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    // The whole point: one buffer, one bind group, an offset per
                    // entity. No storage buffers, so this is WebGL2-safe (§11).
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<InstanceUniform>() as u64,
                    ),
                },
                count: None,
            }],
        });
        // Three groups: frame, instance, texture. `max_bind_groups` is 4 under
        // `downlevel_webgl2_defaults`, so this is still the baseline path
        // (DESIGN §11). Group 2 is unconditional — see `TextureRegistry`'s
        // default bind group for why one layout beats two.
        let textures = bake::TextureRegistry::new(&device, &queue);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[
                Some(&frame_layout),
                Some(&instance_layout),
                Some(textures.layout()),
            ],
            immediate_size: 0,
        });

        let frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame uniform"),
            size: std::mem::size_of::<FrameUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame bind group"),
            layout: &frame_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_buffer.as_entire_binding(),
            }],
        });

        // Every dynamic offset must be a multiple of this; the struct is 96
        // bytes but the slots are strided by whatever the device demands.
        let align = device.limits().min_uniform_buffer_offset_alignment.max(1);
        let instance_stride = align_up(std::mem::size_of::<InstanceUniform>() as u32, align);
        let (instance_buffer, instance_bind_group) = create_instance_buffer(
            &device,
            &instance_layout,
            instance_stride,
            INITIAL_INSTANCE_CAPACITY,
        );

        let sky_pipeline = create_sky_pipeline(&device, &frame_layout, target_format);
        let baker = bake::TextureBaker::new(&device);

        Renderer {
            device,
            queue,
            target_format,
            pipelines: HashMap::new(),
            pipeline_layout,
            instance_layout,
            frame_buffer,
            frame_bind_group,
            phase: [0.0; 4],
            time: [0.0; 4],
            sky_pipeline,
            instance_buffer,
            instance_bind_group,
            instance_stride,
            instance_capacity: INITIAL_INSTANCE_CAPACITY,
            instance_scratch: Vec::new(),
            meshes: MeshRegistry::new(),
            baker,
            textures,
            depth: None,
            offscreen: None,
            blit: None,
        }
    }

    /// Create an instance/adapter/device with no surface at all, then build the
    /// renderer for `target_format`. Used by tests, bakes and the editor bridge.
    pub async fn headless(target_format: wgpu::TextureFormat) -> Result<Renderer, String> {
        let (device, queue) = headless_device().await?;
        Ok(Renderer::new(device, queue, target_format))
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn target_format(&self) -> wgpu::TextureFormat {
        self.target_format
    }

    /// The GPU mesh registry — handle → buffers, deduped by content hash.
    pub fn meshes(&self) -> &MeshRegistry {
        &self.meshes
    }

    /// Upload `mesh` and return its handle, reusing existing buffers if that
    /// geometry is already resident.
    pub fn register_mesh(&mut self, mesh: &MeshData) -> MeshHandle {
        self.meshes.register(&self.device, mesh)
    }

    /// The GPU texture registry — handle → baked textures (DESIGN §7).
    pub fn textures(&self) -> &bake::TextureRegistry {
        &self.textures
    }

    /// Bake `spec` at `resolution` if it is not already resident, consulting
    /// (and filling) `store` on the way.
    ///
    /// The one entry point for §7's baked path. Idempotent, content-addressed,
    /// and indistinguishable between a cold bake and a cache hit — which is the
    /// invariant `tests/texture_cache.rs` exists to pin.
    pub fn bake_texture(
        &mut self,
        spec: &texture::TextureSpec,
        resolution: u32,
        store: &dyn CacheStore,
    ) -> texture::TextureHandle {
        self.textures.resolve(
            &self.device,
            &self.queue,
            &self.baker,
            spec,
            resolution,
            store,
        )
    }

    /// Number of shader variants compiled so far.
    pub fn pipeline_count(&self) -> usize {
        self.pipelines.len()
    }

    /// Byte stride between per-entity uniform slots.
    pub fn instance_stride(&self) -> u32 {
        self.instance_stride
    }

    /// Entities the instance buffer can hold before it has to grow.
    pub fn instance_capacity(&self) -> u32 {
        self.instance_capacity
    }

    /// Aim the screen-space phase circle (see [`FrameUniform::phase`]).
    ///
    /// `center` is NDC (`-1..1`, +Y up), `radius` is in NDC-Y units with the X
    /// axis aspect-corrected, and `strength` (`0..1`) drives the edge fringe.
    /// A radius of zero is the resting state: world geometry solid, phase
    /// geometry gone.
    ///
    /// Cheap enough to call every frame — it writes two `vec4`s into the frame
    /// uniform that were already being written — and invisible to the sim.
    pub fn set_phase_fx(&mut self, center: glam::Vec2, radius: f32, strength: f32) {
        self.phase = [center.x, center.y, radius.max(0.0), strength.clamp(0.0, 1.0)];
    }

    /// The phase circle as the next frame will see it: `(center, radius,
    /// strength)`.
    pub fn phase_fx(&self) -> (glam::Vec2, f32, f32) {
        (
            glam::Vec2::new(self.phase[0], self.phase[1]),
            self.phase[2],
            self.phase[3],
        )
    }

    /// Set the render clock (see [`FrameUniform::time`]): host wall seconds and
    /// the interpolation alpha of the frame about to be drawn.
    ///
    /// The renderer has no clock of its own — nothing in runt does (DESIGN §4)
    /// — so this is the same value the host already hands
    /// [`Engine::update`](crate::Engine::update), forwarded rather than
    /// re-measured. A host driving the `Renderer` directly and never calling
    /// this gets a frozen clock at zero, which is exactly what a screenshot
    /// test wants.
    pub fn set_render_clock(&mut self, seconds: f32, alpha: f32) {
        self.time[0] = seconds;
        self.time[1] = alpha;
    }

    /// Compile `variant`'s pipeline if it is not cached yet.
    ///
    /// Variant sources come from one WGSL file plus prepended feature `const`s
    /// (see [`material::variant_source`]), so a new look never means a new
    /// pipeline *shape* — only a new key in this map. Blend mode and depth
    /// state come from the key too, via [`render_state`]: they are the half of
    /// a "look" that no shader branch can express, and hardcoding them here was
    /// the one thing keeping the variant system from covering the whole of §5.
    pub fn ensure_pipeline(&mut self, variant: MaterialVariant) {
        if self.pipelines.contains_key(&variant) {
            return;
        }
        if !variant.unimplemented().is_empty() {
            log::warn!(
                "material variant {:#06b} requests unimplemented features {:#06b}; \
                 they are declared but inert in v1",
                variant.bits(),
                variant.unimplemented().bits()
            );
        }

        let source = material::variant_source(material::BASE_SHADER, variant);
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("shader variant {:#06b}", variant.bits())),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let state = render_state(variant);
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(state.label),
                layout: Some(&self.pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Some(Vertex::LAYOUT)],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.target_format,
                        blend: Some(state.blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(state.depth_write),
                    depth_compare: Some(state.depth_compare),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
        self.pipelines.insert(variant, pipeline);
    }

    /// Draw one frame into `view`, which must be `target_format` and
    /// `width` × `height`.
    ///
    /// Frame anatomy: order the blended draws against this camera (only if
    /// there are any) → upload any newly-referenced geometry → bake any
    /// newly-referenced texture → write the frame uniform → write one instance
    /// slot per draw → compile any missing variant → clear → paint the sky →
    /// walk the sorted list, changing pipeline, texture and vertex buffers only
    /// when the sort key says they changed.
    ///
    /// One loop covers both passes, because the sort put the blended items
    /// last: the opaque half is state-ordered and the tail is back-to-front,
    /// and the pipeline each item names already carries its blend and depth
    /// state (see [`render_state`]). Same encoder, same render pass, same
    /// single submit.
    ///
    /// Eight parameters is past clippy's taste, and a `RenderArgs` struct would
    /// only move the same eight values one level out: the target and its size
    /// come from the host, the frame block and draw list from `Extract`, and the
    /// two libraries are the content side. They have no cohesion to package.
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        view: &wgpu::TextureView,
        width: u32,
        height: u32,
        frame: &FrameParams,
        draws: &[DrawItem],
        library: &MeshLibrary,
        textures: &texture::TextureLibrary,
    ) {
        self.render_scaled(
            view,
            width,
            height,
            RenderScale::default(),
            frame,
            draws,
            library,
            textures,
        );
    }

    /// As [`render`](Renderer::render), drawing the scene at `scale` × the
    /// view's size and blitting it up (DESIGN §11's resolution lever).
    ///
    /// At [`RenderScale::MAX`] this *is* [`render`](Renderer::render): same
    /// encoder, same passes, same one submit, straight into the host's view. No
    /// internal target exists, nothing is sampled, and the pixels are bit for
    /// bit what they were before this method did. That equivalence is the whole
    /// design constraint — the screenshot suite pins it.
    ///
    /// Below it: the depth attachment and the color target are both sized by
    /// [`RenderScale::size`], the entire existing pass sequence runs into that
    /// internal color target, and one extra fullscreen pass copies it to the
    /// real view through a **nearest** sampler. Both passes ride the same
    /// encoder, so a scaled frame is still exactly one submission.
    ///
    /// `aspect` is deliberately *not* recomputed from the scaled size: the frame
    /// is stretched back over the host's rectangle, so the projection that
    /// belongs in it is the host rectangle's. The caller (see
    /// [`Engine::render`](crate::Engine::render)) already built `frame` that way.
    #[allow(clippy::too_many_arguments)]
    pub fn render_scaled(
        &mut self,
        view: &wgpu::TextureView,
        width: u32,
        height: u32,
        scale: RenderScale,
        frame: &FrameParams,
        draws: &[DrawItem],
        library: &MeshLibrary,
        textures: &texture::TextureLibrary,
    ) {
        let (width, height) = (width.max(1), height.max(1));
        let (render_width, render_height) = scale.size(width, height);
        // Not `scale.is_native()`: at a small enough viewport the rounding lands
        // back on the host's own size (a 1-pixel view at 0.5 is still 1 pixel),
        // and blitting a texture onto itself at 1:1 would be pure waste and one
        // more thing to get wrong.
        let scaled = (render_width, render_height) != (width, height);

        self.ensure_depth(render_width, render_height);

        // The blended half of the list has to be ordered by the camera, and the
        // camera is here rather than in `Extract` (see `draw`'s module docs).
        //
        // A frame with nothing blended in it never copies, never re-sorts, and
        // hands the loop below the caller's own slice — which is the whole of
        // "the opaque path is byte-identical to what it was".
        let depth_sorted: Vec<DrawItem>;
        let draws: &[DrawItem] = if draw::has_blended(draws) {
            depth_sorted = {
                let mut items = draws.to_vec();
                draw::sort_draw_list_for_view(&mut items, &frame.view_proj);
                items
            };
            &depth_sorted
        } else {
            draws
        };

        self.upload_missing_meshes(draws, library);
        self.bake_missing_textures(draws, textures);
        self.write_frame_uniform(frame, render_width, render_height);
        self.write_instances(draws);
        for item in draws {
            self.ensure_pipeline(item.variant);
        }
        if scaled {
            self.ensure_blit();
            self.ensure_offscreen(render_width, render_height);
        }

        // The pass writes here; `view` is what the blit writes, if there is one.
        let target = match &self.offscreen {
            Some(off) if scaled => &off.view,
            _ => view,
        };
        let depth_view = &self.depth.as_ref().expect("depth ensured").2;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("opaque forward"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(CLEAR_COLOR),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.frame_bind_group, &[]);

            // The sky goes first, with the depth test off, rather than last at
            // the far plane. Drawing it last would save the one fullscreen
            // overdraw that geometry then covers — but only in a frame that is
            // mostly covered, and it would make the background depend on the
            // depth attachment's clear value and on `LessEqual` semantics for
            // correctness. The frame is being cleared anyway; one guaranteed
            // fullscreen write is the cheap, unconditional version.
            pass.set_pipeline(&self.sky_pipeline);
            pass.draw(0..3, 0..1);

            let mut bound_variant: Option<MaterialVariant> = None;
            let mut bound_mesh: Option<MeshHandle> = None;
            // `None` here means "nothing bound yet", which is distinct from
            // `Some(None)` — the default 1×1 group being bound on purpose.
            let mut bound_texture: Option<Option<texture::TextureHandle>> = None;
            for (slot, item) in draws.iter().enumerate() {
                let Some(gpu) = self.meshes.get(item.mesh) else {
                    continue; // Geometry the library could not supply; warned about above.
                };
                if bound_variant != Some(item.variant) {
                    let pipeline = self
                        .pipelines
                        .get(&item.variant)
                        .expect("variant compiled above");
                    pass.set_pipeline(pipeline);
                    bound_variant = Some(item.variant);
                }
                // Group 2 is in the layout for every variant, so it is bound for
                // every draw — the sort order (texture is the second key)
                // collapses that to one set per texture per frame.
                let wanted = item
                    .texture
                    .filter(|handle| self.textures.contains(*handle));
                if bound_texture != Some(wanted) {
                    let group = match wanted {
                        Some(handle) => {
                            &self.textures.get(handle).expect("filtered above").bind_group
                        }
                        None => self.textures.default_bind_group(),
                    };
                    pass.set_bind_group(2, group, &[]);
                    bound_texture = Some(wanted);
                }
                if bound_mesh != Some(item.mesh) {
                    pass.set_vertex_buffer(0, gpu.vertices.slice(..));
                    pass.set_index_buffer(gpu.indices.slice(..), wgpu::IndexFormat::Uint32);
                    bound_mesh = Some(item.mesh);
                }
                let offset = slot as u32 * self.instance_stride;
                pass.set_bind_group(1, &self.instance_bind_group, &[offset]);
                pass.draw_indexed(0..gpu.index_count, 0, 0..1);
            }
        }

        if scaled {
            let blit = self.blit.as_ref().expect("blit ensured");
            let off = self.offscreen.as_ref().expect("offscreen ensured");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render-scale blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // The triangle covers every pixel of the destination, so
                        // the load only exists to keep the attachment defined —
                        // `Clear` rather than `Load` because loading the host's
                        // previous contents is the one thing that could make a
                        // scaled frame depend on what was on screen before it.
                        load: wgpu::LoadOp::Clear(CLEAR_COLOR),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                // No depth at all: the internal target's depth buffer did its
                // job in the pass above and this is a copy, not a draw.
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&blit.pipeline);
            pass.set_bind_group(0, &off.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    /// The internal target's size, or `None` while the renderer is drawing at
    /// the host's own resolution.
    ///
    /// Introspection for a host's status line and for the tests that pin the
    /// rounding; the allocation is sticky, so this reports what *exists* rather
    /// than what the last frame used.
    pub fn scaled_target_size(&self) -> Option<(u32, u32)> {
        self.offscreen.as_ref().map(|o| (o.width, o.height))
    }

    // -- frame plumbing -----------------------------------------------------

    /// Resolve any handle in the draw list that has no GPU buffers yet.
    ///
    /// Lazy on purpose: geometry is uploaded the first time it is actually
    /// drawn, so a library full of alternate LODs costs nothing until one is
    /// used (DESIGN §6).
    fn upload_missing_meshes(&mut self, draws: &[DrawItem], library: &MeshLibrary) {
        for item in draws {
            if self.meshes.contains(item.mesh) {
                continue;
            }
            match library.get(item.mesh) {
                Some(mesh) => {
                    let uploaded = self.meshes.register(&self.device, mesh);
                    debug_assert_eq!(uploaded, item.mesh, "content hash must round-trip");
                }
                None => log::warn!(
                    "entity {:?} references mesh {:#018x}, which is not in the library",
                    item.entity,
                    item.mesh.0
                ),
            }
        }
    }

    /// Bake any texture in the draw list that is not resident yet.
    ///
    /// The mirror of [`upload_missing_meshes`](Renderer::upload_missing_meshes),
    /// and lazy for the same reason: a library full of alternate textures costs
    /// nothing until one is drawn. No persistent store here — a host that wants
    /// the disk cache consulted calls [`Engine::bake_scene_textures`] at load,
    /// which is where the store lives. Both paths produce identical pixels
    /// (that is what determinism buys), so this one is a correctness fallback
    /// rather than a second policy.
    ///
    /// [`Engine::bake_scene_textures`]: crate::Engine::bake_scene_textures
    fn bake_missing_textures(&mut self, draws: &[DrawItem], library: &texture::TextureLibrary) {
        for item in draws {
            let Some(handle) = item.texture else { continue };
            if self.textures.contains(handle) {
                continue;
            }
            match library.get(handle) {
                Some((spec, resolution)) => {
                    let spec = spec.clone();
                    let baked = self.bake_texture(&spec, resolution, &NoopCache);
                    debug_assert_eq!(baked, handle, "content key must round-trip");
                }
                None => log::warn!(
                    "entity {:?} references texture {:#018x}, which is not in the library",
                    item.entity,
                    handle.0
                ),
            }
        }
    }

    /// `width`/`height` are the **render** target's, not the host view's — see
    /// [`FrameUniform::viewport`].
    fn write_frame_uniform(&mut self, frame: &FrameParams, width: u32, height: u32) {
        let light = frame.lighting;
        let (w, h) = (width.max(1) as f32, height.max(1) as f32);
        let uniform = FrameUniform {
            view_proj: frame.view_proj.to_cols_array_2d(),
            inv_view_proj: frame.view_proj.inverse().to_cols_array_2d(),
            light_dir: light.key_dir.extend(0.0).to_array(),
            light_color: light.key_color.extend(0.0).to_array(),
            sky_color: light.sky_color.extend(0.0).to_array(),
            ground_color: light.ground_color.extend(0.0).to_array(),
            horizon_color: light.horizon().extend(0.0).to_array(),
            phase: self.phase,
            time: self.time,
            viewport: [w, h, 1.0 / w, 1.0 / h],
        };
        self.queue
            .write_buffer(&self.frame_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    /// Pack one aligned instance slot per draw and upload them in one go.
    fn write_instances(&mut self, draws: &[DrawItem]) {
        if draws.is_empty() {
            return;
        }
        self.grow_instances(draws.len() as u32);

        let stride = self.instance_stride as usize;
        self.instance_scratch.clear();
        self.instance_scratch.resize(draws.len() * stride, 0);
        for (slot, item) in draws.iter().enumerate() {
            let uniform = InstanceUniform {
                model: item.model.to_cols_array_2d(),
                base_color: item.base_color.to_array(),
                params: item.params.to_array(),
            };
            let bytes = bytemuck::bytes_of(&uniform);
            let start = slot * stride;
            self.instance_scratch[start..start + bytes.len()].copy_from_slice(bytes);
        }
        self.queue
            .write_buffer(&self.instance_buffer, 0, &self.instance_scratch);
    }

    /// Geometric growth: doubling keeps reallocation amortized O(1) as a scene
    /// fills up, and the bind group is rebuilt with the buffer because it names
    /// it directly.
    fn grow_instances(&mut self, needed: u32) {
        if needed <= self.instance_capacity {
            return;
        }
        let capacity = needed.max(self.instance_capacity.saturating_mul(2));
        let (buffer, bind_group) = create_instance_buffer(
            &self.device,
            &self.instance_layout,
            self.instance_stride,
            capacity,
        );
        log::debug!(
            "instance buffer grew {} → {capacity} slots",
            self.instance_capacity
        );
        self.instance_buffer = buffer;
        self.instance_bind_group = bind_group;
        self.instance_capacity = capacity;
    }

    /// Keep the depth attachment matching the target size, recreating it only
    /// when the size actually changes.
    fn ensure_depth(&mut self, width: u32, height: u32) {
        if !matches!(self.depth, Some((w, h, _)) if w == width && h == height) {
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("depth"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.depth = Some((width, height, view));
        }
    }

    /// Compile the blit pipeline, once, the first time a scaled frame needs it.
    ///
    /// Lazy rather than built in [`new`](Renderer::new) so that the common case
    /// — a host that never leaves native resolution — pays nothing for this
    /// feature: no shader module, no sampler, no pipeline. The sky pipeline is
    /// eager because every frame draws a sky; no frame at 1.0 blits.
    fn ensure_blit(&mut self) {
        if self.blit.is_some() {
            return;
        }
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("blit source layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            // Non-filterable + a non-filtering sampler: the pass
                            // point-samples on purpose, and asking for neither
                            // capability keeps it valid on the narrowest
                            // downlevel adapter (DESIGN §11).
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("blit pipeline layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("render-scale blit"),
                source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
            });
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("render-scale blit"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_blit"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_blit"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.target_format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
        // Nearest on every axis, including between mips (there is one). Chonky
        // pixels are the feature; a `Linear` here would be a blur that costs the
        // same and looks like a mistake.
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("blit nearest"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        log::info!(
            "render scale: blit pipeline compiled (nearest, {:?})",
            self.target_format
        );
        self.blit = Some(Blit {
            pipeline,
            layout,
            sampler,
        });
    }

    /// Keep the internal color target at `width` × `height`, recreating it (and
    /// its bind group, which names its view) only when the size changes.
    ///
    /// The mirror of [`ensure_depth`](Renderer::ensure_depth), and sticky for
    /// the same reason: a host that steps the scale up and down, or drags a
    /// window edge, would otherwise reallocate on the frame it can least afford
    /// to. The size is the *scaled* size, so a resize and a scale change are the
    /// same event as far as this is concerned.
    fn ensure_offscreen(&mut self, width: u32, height: u32) {
        if matches!(&self.offscreen, Some(o) if o.width == width && o.height == height) {
            return;
        }
        let blit = self.blit.as_ref().expect("blit compiled before offscreen");
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("render scale target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // The surface's own format, not a fixed Rgba8: every material
            // pipeline was compiled against `target_format`, and a mismatched
            // attachment is a validation error rather than a conversion.
            format: self.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit source"),
            layout: &blit.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&blit.sampler),
                },
            ],
        });
        log::debug!("render scale: internal target {width}×{height}");
        self.offscreen = Some(Offscreen {
            width,
            height,
            view,
            bind_group,
        });
    }
}

/// The fixed-function half of a variant: what [`Renderer::ensure_pipeline`]
/// builds that is not the shader module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PipelineState {
    pub blend: wgpu::BlendState,
    pub depth_write: bool,
    pub depth_compare: wgpu::CompareFunction,
    /// The pipeline's debug label; also the name a capture tool shows.
    pub label: &'static str,
}

/// Derive a variant's blend and depth state from its key (DESIGN §5).
///
/// A pure function of the bits, and public, so the mapping can be asserted
/// without a GPU — the pipeline cache is keyed on the variant, so this table
/// being right is the difference between "two looks share a pipeline" and "two
/// looks share a pipeline *by accident*".
///
/// | key | blend | depth write | depth test |
/// |---|---|---|---|
/// | *(none of the below)* | replace | yes | `Less` |
/// | [`TRANSPARENT`] | `src·α + dst·(1−α)` | **no** | `Less` |
/// | [`ADDITIVE`] | `src·α + dst` | **no** | `Less` |
/// | `TRANSPARENT \| ADDITIVE` | additive — it wins | **no** | `Less` |
/// | + [`DEPTH_GREATER`] | *(unchanged)* | *(unchanged)* | **`Greater`** |
///
/// Two decisions worth stating out loud:
///
/// - **Additive beats alpha** when a key carries both, the way `LIVE_TEX` beats
///   `TEXTURE` in [`draw::resolve_variant`]: the combination is meaningless
///   rather than illegal, so it resolves the same way everywhere instead of
///   being undefined in one place and rejected in another.
/// - **`DEPTH_GREATER` does not imply a blend.** An opaque `Greater` draw is a
///   perfectly good "fill in what is hidden" pass, and folding the two together
///   would have made the see-through-walls silhouette a special case instead of
///   two composable bits.
///
/// Backface culling stays on for every variant. A camera-facing quad whose CPU
/// basis is built right is wound right, and turning culling off for blended
/// draws would silently double the fill cost of the one population that can
/// least afford it.
///
/// [`TRANSPARENT`]: MaterialVariant::TRANSPARENT
/// [`ADDITIVE`]: MaterialVariant::ADDITIVE
/// [`DEPTH_GREATER`]: MaterialVariant::DEPTH_GREATER
pub fn render_state(variant: MaterialVariant) -> PipelineState {
    let (blend, label) = if variant.contains(MaterialVariant::ADDITIVE) {
        (
            wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                // The destination's alpha is left alone: the attachment may be
                // a surface whose alpha the compositor reads, and a glow has no
                // opinion about how opaque the frame is.
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Zero,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            },
            "forward additive",
        )
    } else if variant.contains(MaterialVariant::TRANSPARENT) {
        (wgpu::BlendState::ALPHA_BLENDING, "forward transparent")
    } else {
        (wgpu::BlendState::REPLACE, "forward opaque")
    };
    PipelineState {
        blend,
        depth_write: !variant.intersects(MaterialVariant::BLENDED),
        depth_compare: if variant.contains(MaterialVariant::DEPTH_GREATER) {
            wgpu::CompareFunction::Greater
        } else {
            wgpu::CompareFunction::Less
        },
        label,
    }
}

/// Compile the background-gradient pipeline (DESIGN §5; see [`sky`]).
///
/// Its own layout, holding `@group(0)` alone: the sky has no material and no
/// per-entity slot, and borrowing the opaque pass's two-group layout would mean
/// binding an instance slot it never reads just to satisfy validation.
fn create_sky_pipeline(
    device: &wgpu::Device,
    frame_layout: &wgpu::BindGroupLayout,
    target_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sky pipeline layout"),
        bind_group_layouts: &[Some(frame_layout)],
        immediate_size: 0,
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sky"),
        source: wgpu::ShaderSource::Wgsl(SKY_SHADER.into()),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sky gradient"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_sky"),
            // No vertex buffer at all: the triangle comes from `vertex_index`.
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_sky"),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            // The oversized triangle's winding is whatever the vertex_index
            // trick produces; there is nothing to cull on a fullscreen pass.
            cull_mode: None,
            ..Default::default()
        },
        // The attachment is shared with the opaque pass, so the format has to
        // match — but the sky neither tests nor writes depth.
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// Round `value` up to a multiple of `align`.
fn align_up(value: u32, align: u32) -> u32 {
    value.div_ceil(align) * align
}

fn create_instance_buffer(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    stride: u32,
    capacity: u32,
) -> (wgpu::Buffer, wgpu::BindGroup) {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("instance uniforms"),
        size: (stride as u64) * (capacity.max(1) as u64),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("instance bind group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            // A window of one struct, slid along the buffer by the dynamic
            // offset — not the whole buffer.
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buffer,
                offset: 0,
                size: wgpu::BufferSize::new(std::mem::size_of::<InstanceUniform>() as u64),
            }),
        }],
    });
    (buffer, bind_group)
}

// ---------------------------------------------------------------------------
// Device creation
// ---------------------------------------------------------------------------

/// The limits every runt device requests: WebGL2-compatible so a single code
/// path is valid on every backend (DESIGN §11).
pub fn required_limits(adapter: &wgpu::Adapter) -> wgpu::Limits {
    wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits())
}

/// Request a device/queue from an adapter with runt's standard descriptor.
pub async fn request_device(adapter: &wgpu::Adapter) -> Result<(wgpu::Device, wgpu::Queue), String> {
    let info = adapter.get_info();
    log::info!(
        "adapter: {} ({:?} via {:?})",
        info.name,
        info.device_type,
        info.backend
    );
    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("device"),
            required_features: wgpu::Features::empty(),
            required_limits: required_limits(adapter),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(|e| format!("request device: {e}"))
}

/// A device with no surface and no display handle.
pub async fn headless_device() -> Result<(wgpu::Device, wgpu::Queue), String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
            ..Default::default()
        })
        .await
        .map_err(|e| format!("no usable GPU adapter for headless rendering: {e}"))?;
    request_device(&adapter).await
}
