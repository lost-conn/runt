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
/// The in-game scene editor's toolkit (DESIGN §10a). Off by default with the
/// rest of the tooling; a wasm player compiles it away entirely.
#[cfg(feature = "editor")]
pub mod editor;
/// The editor's manipulator handles. Feature-gated with [`editor`].
#[cfg(feature = "editor")]
pub mod editor_gizmo;
pub mod engine;
/// Bitmap-font layout over a baked glyph table (DESIGN §10a): the engine owns
/// the code, a game owns the pixels. No typeface here except the optional
/// `default-font` fallback.
pub mod font;
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
/// Live-tunable params — runt's `@export` (DESIGN §3, §10). Off by default with
/// the rest of reflection; a wasm player compiles it away entirely.
#[cfg(feature = "reflect")]
pub mod tweak;
/// The debug overlay that drives [`tweak`]. Feature-gated with it.
#[cfg(feature = "reflect")]
pub mod tweak_panel;
/// Screen-space UI: one instanced quad batch drawn after the frame is finished.
pub mod ui;

/// The background pass's WGSL. Standalone (no feature consts), unlike
/// [`material::BASE_SHADER`].
pub const SKY_SHADER: &str = include_str!("sky.wgsl");

/// The fullscreen pass's WGSL: one triangle, a nearest sampler, and the phase
/// circle's screen effect (DESIGN §11, §5). Standalone like [`SKY_SHADER`]; see
/// [`Renderer::render_scaled`].
pub const BLIT_SHADER: &str = include_str!("blit.wgsl");

/// The screen-space UI pass's WGSL (see [`ui`]). Standalone like
/// [`BLIT_SHADER`]; re-exported from [`ui::UI_SHADER`] so the three standalone
/// shaders sit together.
pub const UI_SHADER: &str = ui::UI_SHADER;

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
pub use draw::{Aabb, DrawItem, DrawStats, FrameParams, Frustum, InstanceRun};
pub use ecs::{
    default_horizon, project_phase_fx, DemoScene, FixedSim, GeneratorRef, GlobalTransform,
    Interpolated, Lighting, MeshRef, PhaseFx, PostSim, QualityTier, RenderScale, Spin, Startup,
    StatusLine, TerrainSurface, TickCount, Transform, Viewport, Visibility, WindowMode,
};
#[cfg(feature = "editor")]
pub use editor::{
    Axis, Drag, DragKind, EditError, EditableScene, EditorState, OpLog, PaletteEntry, Ray, Snap,
    Tool,
};
#[cfg(feature = "editor")]
pub use editor_gizmo::{Gizmo, GizmoMesh, GizmoPart};
pub use engine::Engine;
pub use font::{BitmapFont, FontAsset, FontError, Glyph, Kern};
pub use gen::{GeneratorSpec, Shading};
pub use input::{Input, InputEvent, Key, PadButton, PadStick, PadTrigger, Touch, TouchPhase};
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
pub use ui::{UiAtlasImage, UiBatch, UiPass, UiQuad, UiRun, PREMULTIPLIED_BLEND};
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
    /// The sky pass's two knobs and two reserved slots: `x` — cloud cover
    /// ([`Lighting::clouds`](crate::Lighting::clouds)), `y` — sun-disk size
    /// ([`Lighting::sun`](crate::Lighting::sun)), `zw` — reserved.
    ///
    /// Both are `0` by default, and both are `0`-guarded in the shader, so a
    /// scene that says nothing about weather draws the same three-stop gradient
    /// it always did — which is what `tests/headless_screenshot.rs` holds
    /// against [`sky::gradient`](crate::sky::gradient).
    ///
    /// A block of its own rather than the spare `w` of `light_dir` and
    /// `sky_color`: those are padding because a `vec3` is padded, and a value
    /// hidden in padding is a value nobody finds. `shader.wgsl` restates it too
    /// even though nothing there reads it — one buffer, one layout, three
    /// files (see the block's own note).
    pub sky_params: [f32; 4],
}

/// Per-instance vertex data: one of these per drawn entity, in a vertex buffer
/// on slot 1 (DESIGN §5's "first sanctioned optimization", D3).
///
/// **96 bytes, tightly packed, no alignment tax.** That is the point of the
/// move: the old path put the same three values in a *uniform* buffer and
/// addressed them with a dynamic offset, which the device rounds up to
/// `min_uniform_buffer_offset_alignment` — 256 bytes under WebGL2 limits, so
/// 62% of the buffer was padding and every entity cost a `set_bind_group`.
/// Here the stride is the struct, the buffer is written once per frame, and a
/// run of entities sharing a pipeline, a mesh and a texture is *one*
/// `draw_indexed` over a range of it.
///
/// There is no singleton special case. An entity nothing else matches draws as
/// a run of length one — same pipeline, same buffer, same call shape — which is
/// what keeps this one code path rather than two (and what makes the golden
/// screenshot byte-identical across the change).
///
/// Field order is duplicated in `shader.wgsl`'s `vs_main` signature; see
/// [`InstanceRaw::LAYOUT`] for the attribute numbering.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct InstanceRaw {
    pub model: [[f32; 4]; 4],
    pub color: [f32; 4],
    pub params: [f32; 4],
}

impl InstanceRaw {
    /// Slot 1, stepped per instance, attributes 4–9.
    ///
    /// The mesh stream (slot 0, [`Vertex::LAYOUT`]) owns 0–3, so the two
    /// together declare **10** attributes. `downlevel_webgl2_defaults` grants
    /// `max_vertex_attributes = 16` and `max_vertex_buffers = 8`, so this is
    /// still the baseline path with six attributes and six buffers to spare
    /// (DESIGN §11) — a mat4 costs four locations because a vertex attribute
    /// cannot be wider than a `vec4` on any backend we target.
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &wgpu::vertex_attr_array![
            4 => Float32x4, // model column 0
            5 => Float32x4, // model column 1
            6 => Float32x4, // model column 2
            7 => Float32x4, // model column 3 (translation)
            8 => Float32x4, // base colour
            9 => Float32x4, // material params
        ],
    };

    /// Pack a draw item. Column-major, exactly as `glam` stores a `Mat4` and
    /// exactly as the old uniform wrote it — so the four attributes the vertex
    /// shader reassembles are the same sixteen floats the uniform held, in the
    /// same order, and the multiply that follows is bit for bit the same one.
    pub fn from_item(item: &DrawItem) -> InstanceRaw {
        InstanceRaw {
            model: item.model.to_cols_array_2d(),
            color: item.base_color.to_array(),
            params: item.params.to_array(),
        }
    }
}

/// Instances allocated up front; the buffer doubles from here as needed.
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

    frame_buffer: wgpu::Buffer,
    frame_bind_group: wgpu::BindGroup,
    /// The frame block's layout, kept because the fullscreen pass is compiled
    /// lazily and needs it long after [`new`](Renderer::new) has returned. The
    /// material and sky pipelines take theirs from the same value at
    /// construction, which is what guarantees one buffer serves all three.
    frame_layout: wgpu::BindGroupLayout,

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

    /// One [`InstanceRaw`] per draw item this frame, in list order, as a
    /// **vertex** buffer on slot 1. Written once per frame; every draw is a
    /// range of it.
    instance_buffer: wgpu::Buffer,
    /// Instances the buffer holds before it has to grow. Doubling, sticky.
    instance_capacity: u32,
    /// Staging for the instance upload, reused across frames.
    instance_scratch: Vec<InstanceRaw>,

    /// The visible half of this frame's draw list, and the instanced draws it
    /// coalesces into. Both are per-frame working sets kept between frames so a
    /// steady scene allocates nothing at all.
    visible: Vec<DrawItem>,
    runs: Vec<draw::InstanceRun>,
    stats: draw::DrawStats,

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

    /// Offscreen scene targets, by the caller's own name (see
    /// [`RenderTarget`]). Empty — and therefore free — for every host that
    /// never asks for a second camera.
    scene_targets: HashMap<RenderTarget, SceneTarget>,

    /// The screen-space UI pipeline, compiled the first time a frame actually
    /// has a HUD in it (see [`ensure_ui`](Renderer::ensure_ui)).
    ui: Option<ui::UiPass>,
    /// This frame's HUD, copied out of the world by
    /// [`set_ui_batch`](Renderer::set_ui_batch).
    ///
    /// Render-side state, held here for the same reason [`phase`](Renderer::phase)
    /// is: nothing in a tick may read it. The `Vec` is reused, so a steady HUD
    /// allocates nothing after its first frame.
    ui_quads: Vec<ui::UiQuad>,
    /// The batch's texture runs, copied alongside its quads. Always covers
    /// `ui_quads` exactly; a batch that never switched texture is one run.
    ui_runs: Vec<ui::UiRun>,
    ui_atlas: Option<texture::TextureHandle>,
    /// Physical pixels per logical pixel, from the host
    /// ([`Engine::set_scale_factor`](crate::Engine::set_scale_factor)).
    ///
    /// Read by the UI pass and by nothing else: the scene is drawn in NDC and
    /// does not care how dense the glass is, while a HUD is laid out in pixels
    /// and cares about very little else.
    scale_factor: f32,
}

/// A caller-named offscreen scene target: somewhere to draw a **second camera**
/// — usually a second [`Sim`], with its own world — that a UI quad or a
/// material can then sample.
///
/// The name is the caller's and it is all the identity there is. Handing the
/// same `RenderTarget` to
/// [`render_to_texture`](Renderer::render_to_texture) every frame re-uses one
/// texture; handing a different one allocates a second. The
/// [`TextureHandle`] it resolves to is stable, pure and
/// knowable before the first frame ([`RenderTarget::handle`]), so a game can
/// write it into a UI layout at build time.
///
/// # Why two worlds are safe to share one renderer
///
/// Nothing in the renderer is per-world. Meshes and textures are keyed by
/// *content* — `MeshHandle::of(&mesh)` is the geometry's hash and
/// `TextureHandle` is the spec's — so two `Sim`s that generated the same rock
/// resolve to the same handle and share one upload, and two that generated
/// different rocks cannot collide. The libraries are arguments to each render
/// call rather than state, so "which world is this?" never has to be a question
/// the registries can answer wrongly. The one identifier that is *not* content
/// — this target's own — lives in the reserved half of the handle space
/// ([`TextureHandle::RESERVED_BIT`](texture::TextureHandle::RESERVED_BIT)),
/// where no hash can reach it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RenderTarget(pub u32);

impl RenderTarget {
    /// The handle this target's colour texture is registered under — what a
    /// [`UiQuad`] samples and what a
    /// [`Material`] could name.
    ///
    /// Pure and total: valid to call before the target exists, which is what
    /// lets a HUD be written against a viewport that has not been rendered yet
    /// (it draws the white texel until the first frame lands, exactly like an
    /// atlas that has not been baked).
    pub const fn handle(self) -> texture::TextureHandle {
        texture::TextureHandle::render_target(self.0)
    }
}

/// One offscreen scene target: colour, depth, and a frame block of its own.
///
/// The colour texture itself is not here — it lives in the renderer's
/// [`TextureRegistry`](bake::TextureRegistry) under
/// [`RenderTarget::handle`], which is what makes it samplable by the UI pass
/// and the material path without a second lookup table. This struct holds the
/// parts only the *writing* side needs.
///
/// **The frame uniform is per-target, and that is the whole point.** The
/// renderer has one `frame_buffer` for the main frame; a second camera writing
/// its view-projection into that buffer would be seen by whichever pass
/// executes last, not by the pass it was written for — `queue.write_buffer` is
/// ordered against *submissions*, not against passes inside one. A buffer per
/// target sidesteps the question entirely: each pass binds a block nothing else
/// writes, so the two cameras cannot see each other's matrices however the
/// caller interleaves the calls.
struct SceneTarget {
    width: u32,
    height: u32,
    view: wgpu::TextureView,
    depth: wgpu::TextureView,
    frame_buffer: wgpu::Buffer,
    frame_bind_group: wgpu::BindGroup,
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

/// The fullscreen pass: one pipeline, one nearest sampler, one layout. Serves
/// both the render-scale upscale and the phase circle's screen effect, which
/// are the same fetch (see `blit.wgsl`).
///
/// `layout` is the *source* group's — group 1. Group 0 is the frame block, whose
/// layout the renderer already owns.
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
        // Two groups now: frame at 0 and texture at 2. Group 1 is a **hole** —
        // it held the per-entity uniform until instancing (D3) moved that data
        // into a vertex buffer, and leaving the gap is what lets `@group(2)`
        // keep its number in `shader.wgsl`, in `bake.rs`'s documentation, and
        // in the `TextureRegistry` layout all three pipelines share. Renumbering
        // to close a hole nothing binds would have been churn with a bug in it.
        // `max_bind_groups` is 4 under `downlevel_webgl2_defaults`, so an
        // unused index costs nothing (DESIGN §11).
        let textures = bake::TextureRegistry::new(&device, &queue);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[Some(&frame_layout), None, Some(textures.layout())],
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

        let instance_buffer = create_instance_buffer(&device, INITIAL_INSTANCE_CAPACITY);

        let sky_pipeline = create_sky_pipeline(&device, &frame_layout, target_format);
        let baker = bake::TextureBaker::new(&device);

        Renderer {
            device,
            queue,
            target_format,
            pipelines: HashMap::new(),
            pipeline_layout,
            frame_buffer,
            frame_bind_group,
            frame_layout,
            phase: [0.0; 4],
            time: [0.0; 4],
            sky_pipeline,
            instance_buffer,
            instance_capacity: INITIAL_INSTANCE_CAPACITY,
            instance_scratch: Vec::new(),
            visible: Vec::new(),
            runs: Vec::new(),
            stats: draw::DrawStats::default(),
            meshes: MeshRegistry::new(),
            baker,
            textures,
            depth: None,
            offscreen: None,
            blit: None,
            scene_targets: HashMap::new(),
            ui: None,
            ui_quads: Vec::new(),
            ui_runs: Vec::new(),
            ui_atlas: None,
            scale_factor: 1.0,
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

    /// Make a game-drawn atlas resident under `handle` (see
    /// [`UiAtlasImage`](ui::UiAtlasImage)).
    ///
    /// Idempotent and cheap to call every frame: a resident handle returns
    /// after one hash lookup. Through an [`Engine`] the `UiAtlasImage` resource
    /// is pumped through here once per frame and this is that door — a host
    /// driving the `Renderer` directly calls it itself, or never.
    pub fn upload_ui_atlas(
        &mut self,
        handle: texture::TextureHandle,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) {
        self.textures
            .insert_image(&self.device, &self.queue, handle, width, height, rgba);
    }

    /// Number of shader variants compiled so far.
    pub fn pipeline_count(&self) -> usize {
        self.pipelines.len()
    }

    /// Instances the instance buffer can hold before it has to grow.
    pub fn instance_capacity(&self) -> u32 {
        self.instance_capacity
    }

    /// What the last [`render_scaled`](Renderer::render_scaled) cost: items in,
    /// items culled, instances written, draw calls issued.
    ///
    /// Zeroed by a frame with an empty list, so it always describes the most
    /// recent frame rather than the most recent interesting one.
    pub fn draw_stats(&self) -> draw::DrawStats {
        self.stats
    }

    /// Aim the screen-space phase circle (see [`FrameUniform::phase`]).
    ///
    /// `center` is NDC (`-1..1`, +Y up), `radius` is in NDC-Y units with the X
    /// axis aspect-corrected, and `strength` (`0..1`) drives the edge fringe.
    /// A radius of zero is the resting state: world geometry solid, phase
    /// geometry gone, screen untouched.
    ///
    /// Cheap enough to call every frame — it writes two `vec4`s into the frame
    /// uniform that were already being written — and invisible to the sim. Not
    /// free at the *pass* level, though: a radius above
    /// [`PHASE_MIN_RADIUS`](crate::ecs::PHASE_MIN_RADIUS) turns the screen
    /// effect on, and that makes even a native-resolution frame draw offscreen
    /// and copy (see [`render_scaled`](Renderer::render_scaled)).
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

    /// Hand the next frame its screen-space HUD (DESIGN §13, plan D11; see
    /// [`ui`]).
    ///
    /// Copies rather than borrows, because the batch lives in the world and the
    /// frame is drawn after that borrow ends. The copy is a `memcpy` of
    /// 48 bytes per quad into a `Vec` that keeps its capacity, which is cheaper
    /// than the lifetime it saves.
    ///
    /// Call it every frame: the batch is *replaced*, not accumulated, so a
    /// frame with nothing to draw is passed an empty slice and costs nothing at
    /// all — no pass is encoded and no pipeline is compiled. A host driving the
    /// `Renderer` directly and never calling this never sees the UI path.
    /// Through an [`Engine`], the [`UiBatch`] resource is mirrored here once per
    /// frame and this is that door.
    pub fn set_ui_batch(&mut self, batch: &ui::UiBatch) {
        self.ui_quads.clear();
        self.ui_quads.extend_from_slice(&batch.quads);
        self.ui_runs.clear();
        self.ui_runs.extend(batch.runs());
        self.ui_atlas = batch.atlas;
    }

    /// Physical pixels per logical pixel, for the UI pass.
    ///
    /// The host-side door is [`Engine::set_scale_factor`](crate::Engine::set_scale_factor),
    /// which sets this and the [`Viewport`](crate::ecs::Viewport) the same batch
    /// was laid out against; a caller driving a bare `Renderer` sets it here.
    /// Guarded rather than clamped for that door's reason: a NaN has no frame to
    /// draw, and the last good density beats one.
    pub fn set_scale_factor(&mut self, scale_factor: f32) {
        if scale_factor.is_finite() && scale_factor > 0.0 {
            self.scale_factor = scale_factor;
        }
    }

    /// [`set_ui_batch`](Renderer::set_ui_batch) for a caller that has quads
    /// rather than a [`UiBatch`]: one texture for all of them, which is what
    /// this method took before batches could carry
    /// [runs](ui::UiBatch::set_texture).
    ///
    /// Kept as its own door rather than folded in, because a `&[UiQuad]` is
    /// what a test rig and a hand-rolled host actually hold, and building a
    /// whole batch to hand one over would be ceremony.
    pub fn set_ui_quads(&mut self, quads: &[ui::UiQuad], atlas: Option<texture::TextureHandle>) {
        self.ui_quads.clear();
        self.ui_quads.extend_from_slice(quads);
        self.ui_runs.clear();
        if !quads.is_empty() {
            self.ui_runs.push(ui::UiRun {
                first: 0,
                count: quads.len() as u32,
                texture: None,
            });
        }
        self.ui_atlas = atlas;
    }

    /// Quads the next frame will paint.
    pub fn ui_quad_count(&self) -> usize {
        self.ui_quads.len()
    }

    /// Whether the UI pipeline has been compiled — false until the first frame
    /// with a non-empty batch, which is what "an empty HUD costs nothing" means
    /// concretely.
    pub fn ui_ready(&self) -> bool {
        self.ui.is_some()
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
                    // Slot 0 the mesh, slot 1 the instances. Every variant gets
                    // both, unconditionally: an entity drawn alone is a run of
                    // one, so there is no non-instanced path to keep working.
                    buffers: &[Some(Vertex::LAYOUT), Some(InstanceRaw::LAYOUT)],
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
                    cull_mode: state.cull,
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
    /// Frame anatomy: upload any newly-referenced geometry → drop what this
    /// camera cannot see → order the blended remainder against it (only if
    /// there is any) → coalesce the runs into instanced draws → bake any
    /// newly-referenced texture → write the frame uniform → write one instance
    /// per surviving draw → compile any missing variant → clear → paint the sky
    /// → issue one `draw_indexed` per run.
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
    /// At [`RenderScale::MAX`], with the phase circle at rest, this *is*
    /// [`render`](Renderer::render): same encoder, same passes, same one submit,
    /// straight into the host's view. No internal target exists, nothing is
    /// sampled, and the pixels are bit for bit what they were before this method
    /// did. That equivalence is the whole design constraint — the screenshot
    /// suite pins it.
    ///
    /// Below it: the depth attachment and the color target are both sized by
    /// [`RenderScale::size`], the entire existing pass sequence runs into that
    /// internal color target, and one extra fullscreen pass copies it to the
    /// real view through a **nearest** sampler. Both passes ride the same
    /// encoder, so a scaled frame is still exactly one submission.
    ///
    /// # The phase circle takes the same road
    ///
    /// The screen effect — luminance inverted and desaturated inside the circle
    /// (DESIGN §5, `blit.wgsl`) — needs the *finished* frame as an input, so it
    /// has to run after everything that draws into it and before anything that
    /// must escape it. The fullscreen pass is already exactly that point, so the
    /// effect rides it rather than adding a pass of its own: with the circle on,
    /// one fetch does the copy and the inversion together.
    ///
    /// The price is paid on the other side of the seam. A native-resolution
    /// frame with the circle up can no longer draw straight into the host's
    /// view — there would be nothing to read — so it allocates the internal
    /// target and copies once, which is a full-screen read plus a full-screen
    /// write it did not use to pay. That is the cheapest honest version of a
    /// post-process on a forward renderer with no G-buffer, and it is charged
    /// only while the circle is actually up: at radius zero the old path is
    /// taken unchanged, down to the byte.
    ///
    /// The HUD is unaffected for free, because it is encoded *after* this pass
    /// and straight onto `view`. That is the original's behaviour and it is a
    /// property of the ordering rather than of a flag anyone has to maintain.
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
        // Two reasons to draw offscreen and copy, and only one of them is the
        // resolution. The screen effect has to read the finished frame, so a
        // circle that is actually up forces the detour even at native scale;
        // the same 0.001 the shaders test against decides it, so "off" means
        // the same thing on both sides of the seam.
        let fullscreen = scaled || self.phase[2] > ecs::PHASE_MIN_RADIUS;

        self.ensure_depth(render_width, render_height);

        // Geometry first: the frustum test needs each mesh's object-space box,
        // and that is measured at upload (`MeshRegistry::bounds`). A handle with
        // no bounds yet is kept rather than culled, so this ordering is an
        // accuracy choice, not a correctness one — but culling from the first
        // frame is worth one pass over a list we are about to walk anyway.
        self.upload_missing_meshes(draws, library);

        // The two working sets are moved out of `self` for the duration of the
        // frame. `take` leaves an empty `Vec` behind and costs nothing; putting
        // them back at the end is what keeps their capacity, so a steady scene
        // allocates on neither. It also splits the borrow: filling `visible`
        // reads `self.meshes`, and these being locals is what makes that legal
        // without a clone.
        let mut visible = std::mem::take(&mut self.visible);
        let mut runs = std::mem::take(&mut self.runs);
        self.stats = self.prepare_frame(frame, draws, &mut visible, &mut runs);

        self.bake_missing_textures(&visible, textures);
        self.write_frame_uniform(frame, render_width, render_height);
        self.write_instances(&visible);
        for run in &runs {
            self.ensure_pipeline(run.variant);
        }
        if fullscreen {
            self.ensure_blit();
            self.ensure_offscreen(render_width, render_height);
        }

        // The pass writes here; `view` is what the blit writes, if there is one.
        let target = match &self.offscreen {
            Some(off) if fullscreen => &off.view,
            _ => view,
        };
        let depth_view = &self.depth.as_ref().expect("depth ensured").2;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let issued = self.encode_scene(
            &mut encoder,
            "opaque forward",
            target,
            depth_view,
            &self.frame_bind_group,
            &runs,
        );

        self.stats.draws = issued;
        self.visible = visible;
        self.runs = runs;

        if fullscreen {
            let blit = self.blit.as_ref().expect("blit ensured");
            let off = self.offscreen.as_ref().expect("offscreen ensured");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fullscreen (blit + phase screen)"),
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
            // The same frame block the world pass read: the effect's circle is
            // built from `phase` and `viewport` so it lands exactly where the
            // material shaders' discard boundary did.
            pass.set_bind_group(0, &self.frame_bind_group, &[]);
            pass.set_bind_group(1, &off.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // The HUD, last, over `view` at the host's own resolution — after the
        // blit rather than before it, which is the whole point: the world may be
        // drawn at half scale and chonked back up, and the UI on top of it is
        // still crisp at surface pixels. It rides this encoder, so a frame with
        // a HUD is still exactly one submission.
        //
        // An empty batch skips all of it — no `ensure_ui`, no pass, no draw —
        // so a game with no HUD produces the byte-identical frame it did before
        // this pass existed.
        if !self.ui_quads.is_empty() {
            self.ensure_ui();
            // The batch is in **logical** pixels, so the pass is told the
            // logical size of the surface and the shader's px→NDC divide scales
            // it back out over the physical one. That is what makes a 16-pixel
            // margin a 16-pixel margin on a 2× panel instead of an 8-pixel one,
            // and — the half that is not cosmetic — what keeps a rect the game
            // hit-tests a touch against in the space touches arrive in
            // (`ecs::Viewport`, which is divided by the same factor).
            let ui_size = crate::ecs::Viewport::from_physical(width, height, self.scale_factor);
            // Split the borrow: the pass owns its own buffers and needs the
            // device, the queue and the texture registry, all of which are
            // sibling fields.
            let Renderer {
                ui: Some(ui),
                device,
                queue,
                textures,
                ui_quads,
                ui_runs,
                ui_atlas,
                ..
            } = self
            else {
                unreachable!("ensured above")
            };
            ui.encode(
                device,
                queue,
                &mut encoder,
                view,
                ui_size.width,
                ui_size.height,
                ui_quads,
                ui_runs,
                *ui_atlas,
                textures,
            );
        }

        self.queue.submit(Some(encoder.finish()));
    }

    /// The internal target's size, or `None` while the renderer is drawing at
    /// the host's own resolution.
    ///
    /// Introspection for a host's status line and for the tests that pin the
    /// rounding; the allocation is sticky, so this reports what *exists* rather
    /// than what the last frame used. It is also not only a *scale* target any
    /// more — the phase circle's screen effect allocates one at the host's own
    /// size — so a `Some` equal to the view size means "the fullscreen pass ran
    /// at 1:1", not "the scale is wrong".
    pub fn scaled_target_size(&self) -> Option<(u32, u32)> {
        self.offscreen.as_ref().map(|o| (o.width, o.height))
    }

    // -- offscreen scene targets --------------------------------------------

    /// Draw a **second camera's** scene into `target`'s offscreen texture, and
    /// return the handle a UI quad (or a material) samples it by.
    ///
    /// The scene half of a frame and nothing else: geometry uploaded, textures
    /// baked, culled, sorted, coalesced, sky, one `draw_indexed` per run —
    /// `encode_scene`, the same code the host frame runs. Deliberately *not* in
    /// it:
    ///
    /// - **No UI pass.** A viewport is a picture of a world; the HUD that
    ///   frames it belongs to the host frame, drawn on top of the quad that
    ///   samples this.
    /// - **No render scale and no fullscreen pass.** The caller already chose
    ///   the pixel count — that is what `width`/`height` *are* — so scaling it
    ///   again would be a second opinion about the same number, and there is no
    ///   surface here to stretch back over.
    /// - **No phase circle.** The frame block is written with the circle at
    ///   rest (see `frame_uniform`); a circle aimed at the host's screen has
    ///   no meaning in another world's viewport. If a tutorial card ever needs
    ///   to *demonstrate* phasing, this is where a per-target phase would go.
    ///
    /// # Every frame, or on change
    ///
    /// The target persists and its contents survive until something draws over
    /// them, so a card that only changes when the tutorial step does may render
    /// on change. Re-rendering every frame is the simpler contract and the one
    /// the sizes are sticky for; both are supported because neither costs the
    /// other anything.
    ///
    /// # One submission of its own
    ///
    /// This is its own encoder and its own `submit`, which is also what makes
    /// it safe to share the renderer's *instance* buffer with the host frame:
    /// `queue.write_buffer` lands on the queue timeline in call order, so each
    /// submission executes against the bytes written before it. (The frame
    /// *uniform* does not rely on that — each target owns one; see
    /// `SceneTarget`.)
    ///
    /// Returns this pass's [`DrawStats`] rather than storing them, so
    /// [`draw_stats`](Renderer::draw_stats) keeps meaning "the host frame".
    #[allow(clippy::too_many_arguments)]
    pub fn render_to_texture(
        &mut self,
        target: RenderTarget,
        width: u32,
        height: u32,
        frame: &FrameParams,
        draws: &[DrawItem],
        library: &MeshLibrary,
        textures: &texture::TextureLibrary,
    ) -> draw::DrawStats {
        let (width, height) = (width.max(1), height.max(1));
        self.ensure_render_target(target, width, height);

        // Exactly `render_scaled`'s preamble, in exactly its order: geometry
        // before culling (the frustum test wants measured bounds), textures
        // after culling (nothing baked for a draw that was thrown away).
        self.upload_missing_meshes(draws, library);
        let mut visible = std::mem::take(&mut self.visible);
        let mut runs = std::mem::take(&mut self.runs);
        let mut stats = self.prepare_frame(frame, draws, &mut visible, &mut runs);
        self.bake_missing_textures(&visible, textures);
        self.write_instances(&visible);
        for run in &runs {
            self.ensure_pipeline(run.variant);
        }

        // The target's own frame block, written into the target's own buffer.
        let uniform = self.frame_uniform(frame, width, height, [0.0; 4]);
        let scene = self
            .scene_targets
            .get(&target)
            .expect("ensured at the top of this call");
        self.queue
            .write_buffer(&scene.frame_buffer, 0, bytemuck::bytes_of(&uniform));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("offscreen scene"),
            });
        stats.draws = self.encode_scene(
            &mut encoder,
            "offscreen scene",
            &scene.view,
            &scene.depth,
            &scene.frame_bind_group,
            &runs,
        );
        self.queue.submit(Some(encoder.finish()));

        self.visible = visible;
        self.runs = runs;
        stats
    }

    /// Make sure `target` exists at `width` × `height`, returning the
    /// [`TextureHandle`] it is registered under.
    ///
    /// Idempotent and cheap to call every frame: an existing target of the same
    /// size is one hash lookup. A *different* size recreates the colour and
    /// depth textures and re-registers the handle — sticky otherwise, exactly
    /// like `ensure_offscreen`, and for the same reason: a card being dragged
    /// or animated would otherwise reallocate on the frame it can least afford
    /// to.
    ///
    /// [`render_to_texture`](Renderer::render_to_texture) calls this, so a host
    /// only needs it to allocate a viewport *before* the first frame it draws
    /// one — which is what a HUD that measures its card wants.
    pub fn ensure_render_target(
        &mut self,
        target: RenderTarget,
        width: u32,
        height: u32,
    ) -> texture::TextureHandle {
        let (width, height) = (width.max(1), height.max(1));
        let handle = target.handle();
        if matches!(self.scene_targets.get(&target), Some(t) if t.width == width && t.height == height)
        {
            return handle;
        }

        let color = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // The surface's format, not a fixed Rgba8, for the render-scale
            // target's reason: every material pipeline was compiled against
            // `target_format`, and an attachment of another format is a
            // validation error rather than a conversion. It is also what makes
            // one pipeline cache serve both cameras.
            format: self.target_format,
            // `COPY_SRC` on top of the two the pass needs: a viewport nobody
            // can read back is a viewport nobody can test, and the usage bit is
            // free until something uses it.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = color.create_view(&wgpu::TextureViewDescriptor::default());
        let depth = self
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("scene target depth"),
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
            })
            .create_view(&wgpu::TextureViewDescriptor::default());

        // A frame block of this target's own — see `SceneTarget`.
        let frame_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene target frame uniform"),
            size: std::mem::size_of::<FrameUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene target frame"),
            layout: &self.frame_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_buffer.as_entire_binding(),
            }],
        });

        // The colour texture goes to the registry, which is what makes it
        // samplable by handle from the UI pass and the material path alike.
        // The registry owns it from here; this struct keeps only the parts the
        // *writing* side needs.
        self.textures
            .insert_render_target(&self.device, &self.queue, handle, color);
        log::info!(
            "scene target {} ({:#018x}): {width}×{height}",
            target.0,
            handle.0
        );
        self.scene_targets.insert(
            target,
            SceneTarget {
                width,
                height,
                view,
                depth,
                frame_buffer,
                frame_bind_group,
            },
        );
        handle
    }

    /// The size `target` is allocated at, or `None` if it does not exist.
    pub fn render_target_size(&self, target: RenderTarget) -> Option<(u32, u32)> {
        self.scene_targets.get(&target).map(|t| (t.width, t.height))
    }

    /// Free `target`'s textures and un-register its handle.
    ///
    /// A tutorial that has closed its card and will not reopen it; nothing
    /// calls this on the way out, because dropping the renderer drops
    /// everything anyway. A quad still naming the handle afterwards degrades to
    /// the white texel, like any unbaked atlas.
    pub fn drop_render_target(&mut self, target: RenderTarget) {
        if self.scene_targets.remove(&target).is_some() {
            self.textures.remove(target.handle());
        }
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

    /// The scene, into whatever colour and depth attachments the caller names:
    /// clear → sky → one `draw_indexed` per coalesced run. Returns the number
    /// of calls issued.
    ///
    /// The one place the forward pass is written, so
    /// [`render_scaled`](Renderer::render_scaled) and
    /// [`render_to_texture`](Renderer::render_to_texture) cannot drift apart —
    /// same sort, same instancing, same state-change elision, same sky. What
    /// the caller varies is only *where* it lands and *which frame block* it
    /// reads, which is exactly the difference between two cameras.
    ///
    /// `&self`, deliberately: everything it needs is already resident by the
    /// time it runs (the caller uploaded, baked, wrote and compiled), and
    /// taking the working sets as an argument rather than reading them back out
    /// of `self` is what lets the caller keep holding them.
    fn encode_scene(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        color: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        frame_group: &wgpu::BindGroup,
        runs: &[draw::InstanceRun],
    ) -> u32 {
        let mut issued = 0u32;
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(CLEAR_COLOR),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
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
        pass.set_bind_group(0, frame_group, &[]);

        // The sky goes first, with the depth test off, rather than last at
        // the far plane. Drawing it last would save the one fullscreen
        // overdraw that geometry then covers — but only in a frame that is
        // mostly covered, and it would make the background depend on the
        // depth attachment's clear value and on `LessEqual` semantics for
        // correctness. The frame is being cleared anyway; one guaranteed
        // fullscreen write is the cheap, unconditional version.
        pass.set_pipeline(&self.sky_pipeline);
        pass.draw(0..3, 0..1);

        // Slot 1 is set once for the whole frame. Every material pipeline
        // declares the same instance layout and every draw is a *range* of
        // the one buffer, so there is nothing per-draw to rebind — which is
        // exactly what the dynamic-offset bind group used to cost.
        if !runs.is_empty() {
            pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        }

        let mut bound_variant: Option<MaterialVariant> = None;
        let mut bound_mesh: Option<MeshHandle> = None;
        // `None` here means "nothing bound yet", which is distinct from
        // `Some(None)` — the default 1×1 group being bound on purpose.
        let mut bound_texture: Option<Option<texture::TextureHandle>> = None;
        for run in runs {
            let Some(gpu) = self.meshes.get(run.mesh) else {
                continue; // Geometry the library could not supply; warned about above.
            };
            if bound_variant != Some(run.variant) {
                let pipeline = self
                    .pipelines
                    .get(&run.variant)
                    .expect("variant compiled above");
                pass.set_pipeline(pipeline);
                bound_variant = Some(run.variant);
            }
            // Group 2 is in the layout for every variant, so it is bound for
            // every draw — the sort order (texture is the second key)
            // collapses that to one set per texture per frame.
            let wanted = run
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
            if bound_mesh != Some(run.mesh) {
                pass.set_vertex_buffer(0, gpu.vertices.slice(..));
                pass.set_index_buffer(gpu.indices.slice(..), wgpu::IndexFormat::Uint32);
                bound_mesh = Some(run.mesh);
            }
            pass.draw_indexed(0..gpu.index_count, 0, run.first..run.first + run.count);
            issued += 1;
        }
        issued
    }

    /// `width`/`height` are the **render** target's, not the host view's — see
    /// [`FrameUniform::viewport`].
    fn write_frame_uniform(&mut self, frame: &FrameParams, width: u32, height: u32) {
        let uniform = self.frame_uniform(frame, width, height, self.phase);
        self.queue
            .write_buffer(&self.frame_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    /// The frame block itself, as a value — so the main frame and an offscreen
    /// scene target can each write their own copy into their own buffer.
    ///
    /// `phase` is a parameter rather than `self.phase` because the screen-space
    /// circle belongs to the *screen*: a second camera looking at a second world
    /// has no reason to inherit a circle aimed at the host's view, and
    /// inheriting one would make a demo viewport's geometry vanish because
    /// something happened on the HUD behind it.
    fn frame_uniform(
        &self,
        frame: &FrameParams,
        width: u32,
        height: u32,
        phase: [f32; 4],
    ) -> FrameUniform {
        let light = frame.lighting;
        let (w, h) = (width.max(1) as f32, height.max(1) as f32);
        FrameUniform {
            view_proj: frame.view_proj.to_cols_array_2d(),
            inv_view_proj: frame.view_proj.inverse().to_cols_array_2d(),
            light_dir: light.key_dir.extend(0.0).to_array(),
            light_color: light.key_color.extend(0.0).to_array(),
            sky_color: light.sky_color.extend(0.0).to_array(),
            ground_color: light.ground_color.extend(0.0).to_array(),
            horizon_color: light.horizon().extend(0.0).to_array(),
            phase,
            time: self.time,
            viewport: [w, h, 1.0 / w, 1.0 / h],
            sky_params: [light.clouds, light.sun, 0.0, 0.0],
        }
    }

    /// Turn the caller's sorted list into this frame's *visible* list and the
    /// instanced draws it collapses to (D3, D5).
    ///
    /// Three steps, in the only order that works:
    ///
    /// 1. **Cull.** Every item whose mesh has a measured box and whose box
    ///    cannot reach this camera's frustum is dropped. Conservative — see
    ///    [`Frustum::intersects`](draw::Frustum::intersects) — so this can cost
    ///    a draw call and never a pixel, which is why the golden frame does not
    ///    move when it is switched on.
    /// 2. **Depth-order the blended tail**, if there is one. The cull comes
    ///    first because sorting items that are about to be thrown away is work
    ///    for nothing, and the two commute: culling is order-preserving.
    /// 3. **Coalesce.** Adjacent items agreeing on (variant, mesh, texture)
    ///    become one instanced draw over a contiguous instance range.
    ///
    /// All three are pure functions of (list, camera, mesh bounds), so two
    /// frames from the same world state still produce byte-identical command
    /// streams — the determinism claim survives intact, with fewer commands in
    /// it.
    ///
    /// Returns the frame's [`DrawStats`] with everything but
    /// `draws` filled in — that one is the pass's to count.
    fn prepare_frame(
        &self,
        frame: &FrameParams,
        draws: &[DrawItem],
        visible: &mut Vec<DrawItem>,
        runs: &mut Vec<draw::InstanceRun>,
    ) -> draw::DrawStats {
        visible.clear();
        visible.extend_from_slice(draws);

        let frustum = draw::Frustum::from_view_proj(&frame.view_proj);
        let culled = draw::cull_draw_list(visible, &frustum, |handle| self.meshes.bounds(handle));

        // The blended half has to be ordered by the camera, and the camera is
        // here rather than in `Extract` (see `draw`'s module docs). A frame with
        // nothing blended in it never re-sorts.
        if draw::has_blended(visible) {
            draw::sort_draw_list_for_view(visible, &frame.view_proj);
        }
        draw::coalesce_draws_into(visible, runs);

        draw::DrawStats {
            items: draws.len() as u32,
            culled: culled as u32,
            instances: visible.len() as u32,
            draws: 0,
        }
    }

    /// Pack one [`InstanceRaw`] per visible draw and upload them in one go.
    ///
    /// One `write_buffer` for the whole frame, tightly packed — no 256-byte
    /// striding, no per-entity bind group, and the draw ranges index straight
    /// into it because the order is the list's order.
    fn write_instances(&mut self, draws: &[DrawItem]) {
        if draws.is_empty() {
            return;
        }
        self.grow_instances(draws.len() as u32);
        self.instance_scratch.clear();
        self.instance_scratch
            .extend(draws.iter().map(InstanceRaw::from_item));
        self.queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::cast_slice(&self.instance_scratch),
        );
    }

    /// Geometric growth: doubling keeps reallocation amortized O(1) as a scene
    /// fills up. Nothing names the buffer but the pass, so unlike the uniform
    /// path there is no bind group to rebuild with it.
    fn grow_instances(&mut self, needed: u32) {
        if needed <= self.instance_capacity {
            return;
        }
        let capacity = needed.max(self.instance_capacity.saturating_mul(2));
        log::debug!(
            "instance buffer grew {} → {capacity} instances",
            self.instance_capacity
        );
        self.instance_buffer = create_instance_buffer(&self.device, capacity);
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

    /// Compile the fullscreen pipeline, once, the first time a frame needs it —
    /// because it is scaled, or because the phase circle is on.
    ///
    /// Lazy rather than built in [`new`](Renderer::new) so that the common case
    /// — a host at native resolution with no circle up — pays nothing for
    /// either feature: no shader module, no sampler, no pipeline. The sky
    /// pipeline is eager because every frame draws a sky; plenty of frames
    /// never touch this one.
    fn ensure_blit(&mut self) {
        if self.blit.is_some() {
            return;
        }
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fullscreen source layout"),
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
                label: Some("fullscreen pipeline layout"),
                // Frame at 0, source at 1. The frame block keeps the number it
                // has in every other pipeline in the engine — the alternative,
                // hanging it off group 1 to leave the source where it was, would
                // make this the one shader where `@group(0)` means something
                // else. Group 1 is the hole the material layout leaves, so
                // nothing collides.
                bind_group_layouts: &[Some(&self.frame_layout), Some(&layout)],
                immediate_size: 0,
            });
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fullscreen pass"),
                source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
            });
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("fullscreen pass"),
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
            label: Some("fullscreen nearest"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        log::info!(
            "fullscreen pass: pipeline compiled (nearest, {:?})",
            self.target_format
        );
        self.blit = Some(Blit {
            pipeline,
            layout,
            sampler,
        });
    }

    /// Compile the screen-space UI pipeline, once, the first time a frame has a
    /// non-empty [`UiBatch`] in it.
    ///
    /// Lazy for the blit's reason, one notch stronger: a game with no HUD pays
    /// for no shader module, no sampler, no pipeline, no 1×1 texture and no
    /// instance buffer — and, because the pass is skipped entirely, not one
    /// command in the stream either.
    fn ensure_ui(&mut self) {
        if self.ui.is_none() {
            self.ui = Some(ui::UiPass::new(&self.device, &self.queue, self.target_format));
        }
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
            label: Some("fullscreen source"),
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
    /// Which face to drop, or `None` to draw both
    /// ([`TWO_SIDED`](MaterialVariant::TWO_SIDED)).
    pub cull: Option<wgpu::Face>,
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
/// | key | blend | depth write | depth test | cull |
/// |---|---|---|---|---|
/// | *(none of the below)* | replace | yes | `Less` | back |
/// | [`TRANSPARENT`] | `src·α + dst·(1−α)` | **no** | `Less` | back |
/// | [`ADDITIVE`] | `src·α + dst` | **no** | `Less` | back |
/// | `TRANSPARENT \| ADDITIVE` | additive — it wins | **no** | `Less` | back |
/// | + [`DEPTH_GREATER`] | *(unchanged)* | *(unchanged)* | **`Greater`** | *(unchanged)* |
/// | + [`TWO_SIDED`] | *(unchanged)* | *(unchanged)* | *(unchanged)* | **none** |
///
/// Three decisions worth stating out loud:
///
/// - **Additive beats alpha** when a key carries both, the way `LIVE_TEX` beats
///   `TEXTURE` in [`draw::resolve_variant`]: the combination is meaningless
///   rather than illegal, so it resolves the same way everywhere instead of
///   being undefined in one place and rejected in another.
/// - **`DEPTH_GREATER` does not imply a blend.** An opaque `Greater` draw is a
///   perfectly good "fill in what is hidden" pass, and folding the two together
///   would have made the see-through-walls silhouette a special case instead of
///   two composable bits.
/// - **Backface culling is on unless a key asks for it off.** A camera-facing
///   quad whose CPU basis is built right is wound right, and dropping culling
///   for blended draws wholesale would silently double the fill cost of the one
///   population that can least afford it. [`TWO_SIDED`] is therefore its own
///   opt-in bit rather than a consequence of any other — the surfaces that need
///   it (a pond seen from underneath, a waterfall sheet) are a handful, and
///   they say so.
///
/// [`TRANSPARENT`]: MaterialVariant::TRANSPARENT
/// [`ADDITIVE`]: MaterialVariant::ADDITIVE
/// [`DEPTH_GREATER`]: MaterialVariant::DEPTH_GREATER
/// [`TWO_SIDED`]: MaterialVariant::TWO_SIDED
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
        cull: if variant.contains(MaterialVariant::TWO_SIDED) {
            None
        } else {
            Some(wgpu::Face::Back)
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

/// The per-instance vertex buffer: `capacity` × 96 bytes, no padding.
fn create_instance_buffer(device: &wgpu::Device, capacity: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("instances"),
        size: (std::mem::size_of::<InstanceRaw>() as u64) * (capacity.max(1) as u64),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
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
