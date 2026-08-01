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

pub mod cache;
pub mod camera;
pub mod draw;
pub mod ecs;
pub mod engine;
pub mod gen;
pub mod input;
pub mod material;
pub mod physics;
pub mod registry;
pub mod scene;
pub mod sim;
pub mod trace;

pub use cache::{CacheStats, CacheStore, GenCache, NoopCache};
pub use camera::{Camera, FollowCamera};
pub use draw::{DrawItem, FrameParams};
pub use ecs::{
    DemoScene, FixedSim, GeneratorRef, GlobalTransform, Interpolated, Lighting, MeshRef, PostSim,
    QualityTier, Spin, Startup, StatusLine, TerrainSurface, TickCount, Transform,
};
pub use engine::Engine;
pub use gen::{GeneratorSpec, Shading};
pub use input::{Input, InputEvent, Key};
pub use material::{Material, MaterialVariant};
pub use physics::{
    AabbCollider, Ball, BallController, Grounded, OverlapEvent, RollSpin, SphereCollider, Trigger,
    Velocity,
};
pub use registry::{GpuMesh, MeshHandle, MeshLibrary, MeshRegistry};
pub use runt_mesh::{HeightField, MeshData as Mesh, Quality, TerrainParams};
pub use scene::{load_scene, save_scene, SceneDesc, SceneError};
pub use sim::{Sim, SimConfig, MAX_ACCUMULATED, TICK_DT};
pub use trace::{InputTrace, TickEvent};

#[cfg(not(target_arch = "wasm32"))]
pub use cache::NativeDiskCache;

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
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FrameUniform {
    pub view_proj: [[f32; 4]; 4],
    /// `xyz`: direction towards the key light. `w`: padding (std140 wants the
    /// vec3 padded to 16 bytes anyway, so it may as well be explicit).
    pub light_dir: [f32; 4],
    pub light_color: [f32; 4],
    pub sky_color: [f32; 4],
    pub ground_color: [f32; 4],
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
    depth: Option<(u32, u32, wgpu::TextureView)>,
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
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[Some(&frame_layout), Some(&instance_layout)],
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

        Renderer {
            device,
            queue,
            target_format,
            pipelines: HashMap::new(),
            pipeline_layout,
            instance_layout,
            frame_buffer,
            frame_bind_group,
            instance_buffer,
            instance_bind_group,
            instance_stride,
            instance_capacity: INITIAL_INSTANCE_CAPACITY,
            instance_scratch: Vec::new(),
            meshes: MeshRegistry::new(),
            depth: None,
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

    /// Compile `variant`'s pipeline if it is not cached yet.
    ///
    /// Variant sources come from one WGSL file plus prepended feature `const`s
    /// (see [`material::variant_source`]), so a new look never means a new
    /// pipeline *shape* — only a new key in this map.
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

        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("forward opaque"),
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
                        blend: Some(wgpu::BlendState::REPLACE),
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
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
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
    /// Frame anatomy: upload any newly-referenced geometry → write the frame
    /// uniform → write one instance slot per draw → compile any missing variant
    /// → clear → walk the sorted list, changing pipeline and vertex buffers only
    /// when the sort key says they changed.
    pub fn render(
        &mut self,
        view: &wgpu::TextureView,
        width: u32,
        height: u32,
        frame: &FrameParams,
        draws: &[DrawItem],
        library: &MeshLibrary,
    ) {
        let (width, height) = (width.max(1), height.max(1));
        self.ensure_depth(width, height);

        self.upload_missing_meshes(draws, library);
        self.write_frame_uniform(frame);
        self.write_instances(draws);
        for item in draws {
            self.ensure_pipeline(item.variant);
        }

        let depth_view = &self.depth.as_ref().expect("depth ensured").2;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("opaque forward"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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

            let mut bound_variant: Option<MaterialVariant> = None;
            let mut bound_mesh: Option<MeshHandle> = None;
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
        self.queue.submit(Some(encoder.finish()));
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

    fn write_frame_uniform(&mut self, frame: &FrameParams) {
        let light = frame.lighting;
        let uniform = FrameUniform {
            view_proj: frame.view_proj.to_cols_array_2d(),
            light_dir: light.key_dir.extend(0.0).to_array(),
            light_color: light.key_color.extend(0.0).to_array(),
            sky_color: light.sky_color.extend(0.0).to_array(),
            ground_color: light.ground_color.extend(0.0).to_array(),
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
