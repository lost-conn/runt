//! runt engine core — windowless.
//!
//! The core owns a `wgpu::Device`/`Queue` (either handed to it by a host or
//! created headless) and renders into a **caller-provided** `wgpu::TextureView`.
//! It never creates a surface and never presents; that is the host's job.
//! This is what lets the native window, the web canvas, the editor viewport and
//! headless screenshot tests all drive the same engine.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

pub use runt_mesh as mesh;
use runt_mesh::MeshData;

pub mod ecs;
pub mod engine;
pub mod input;
pub mod scene;
pub mod sim;

pub use ecs::{
    DemoScene, FixedSim, GlobalTransform, Interpolated, PostSim, Spin, Startup, TickCount, Transform,
};
pub use engine::Engine;
pub use input::{Input, InputEvent, Key};
pub use scene::demo_scene;
pub use sim::{Sim, MAX_ACCUMULATED, TICK_DT};

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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Uniforms {
    pub view_proj: [[f32; 4]; 4],
    pub model: [[f32; 4]; 4],
}

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

/// Renders the scene into any `wgpu::TextureView` of `target_format`.
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    target_format: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    index_count: u32,
    ubuf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
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
        let scene = demo_scene();
        scene.validate();
        log::info!(
            "scene: {} verts, {} tris",
            scene.vertex_count(),
            scene.triangle_count()
        );

        let verts = interleave(&scene);
        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vbuf"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ibuf"),
            contents: bytemuck::cast_slice(&scene.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ubuf"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group"),
            layout: &bind_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ubuf.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipeline"),
            layout: Some(&pipeline_layout),
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
                    format: target_format,
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

        Renderer {
            device,
            queue,
            target_format,
            pipeline,
            vbuf,
            ibuf,
            index_count: scene.indices.len() as u32,
            ubuf,
            bind_group,
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

    /// Draw one frame into `view`, which must be `target_format` and
    /// `width` × `height`. `model` is the model matrix for the merged demo
    /// buffer — the sim owns it now, so the renderer neither knows the time nor
    /// the spin rate (DESIGN §3; per-entity draws are step 3).
    pub fn render(&mut self, view: &wgpu::TextureView, width: u32, height: u32, model: Mat4) {
        let (width, height) = (width.max(1), height.max(1));
        self.ensure_depth(width, height);
        let depth_view = &self.depth.as_ref().expect("depth ensured").2;

        let aspect = width as f32 / height as f32;
        let proj = Mat4::perspective_rh(60f32.to_radians(), aspect, 0.1, 100.0);
        let eye = Mat4::look_at_rh(Vec3::new(0.0, 2.4, 6.5), Vec3::ZERO, Vec3::Y);
        let uniforms = Uniforms {
            view_proj: (proj * eye).to_cols_array_2d(),
            model: model.to_cols_array_2d(),
        };
        self.queue
            .write_buffer(&self.ubuf, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main pass"),
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
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.vbuf.slice(..));
            pass.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.index_count, 0, 0..1);
        }
        self.queue.submit(Some(encoder.finish()));
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
