//! Screen-space UI — one instanced quad batch, drawn after everything else
//! (DESIGN §13's open HUD question, plan D11).
//!
//! The engine has no layout engine, no text shaper and no widget tree. It has
//! *quads*: a game builds a [`UiBatch`] of rectangles in logical pixels, and
//! this pass paints them over the finished frame in list order. Everything a
//! HUD is made of — panels, bars, glyphs from an atlas, the fan of slices that
//! makes a ring — is that one primitive. What the engine owns is the part a
//! game cannot write portably: a pipeline that is valid on WebGL2, a screen
//! transform that survives [`RenderScale`](crate::RenderScale), and premultiplied
//! blending that composites the same way on every backend.
//!
//! # Coordinates
//!
//! **Logical pixels, top-left origin, +Y down** — the DOM/HUD convention, so a
//! layout reads the way it is thought about (`rect: [16.0, 16.0, 120.0, 8.0]` is
//! a bar 16 px in from the top-left corner). The shader flips Y into NDC; that
//! flip is the one coordinate subtlety in the whole module and it lives in
//! exactly one line of `ui.wgsl`.
//!
//! "Logical pixel" means *the pixels of the view the host handed
//! [`Engine::render`](crate::Engine::render)* — the surface, *not* the internal
//! render-scale target. The UI pass runs after the blit, so at
//! `RenderScale(0.5)` the world is honest 2×2 chonk and the HUD on top of it is
//! still surface-crisp: a quad edge lands on a surface pixel boundary, never on
//! a blitted block boundary. That is the entire reason this pass is where it is.
//! (A host with a hidpi canvas that wants CSS-pixel layout scales its own rects;
//! the engine deliberately has no second scale factor to get out of sync.)
//!
//! # Premultiplied alpha
//!
//! [`UiQuad::color`] is **premultiplied**: `rgb` is already multiplied through
//! by `a`. The blend is `src·1 + dst·(1−srcα)` on both colour and alpha, which
//! is the composite that stays correct when quads overlap and the only one that
//! gives the same answer for "half-transparent black" as it does for "nothing".
//! [`UiQuad::rgba`] converts a straight-alpha colour, so authoring stays
//! `(1.0, 0.2, 0.2, 0.5)` and the wire format stays premultiplied.
//!
//! The atlas is premultiplied too, and the fragment is `texel · color` — a
//! product of two premultiplied values is premultiplied, so tinting a glyph is a
//! plain multiply with no divide anywhere. A font baked as white-on-transparent
//! must therefore write `rgb = a` (coverage in all four channels), not
//! `rgb = 1`.
//!
//! # Lifetime: rebuilt every frame
//!
//! [`UiBatch`] is **rebuilt from scratch each frame**, not retained and
//! diffed. The game's HUD system clears it and pushes what the HUD looks like
//! *now*; the renderer copies it at draw time and never mutates it. That makes
//! the batch a pure function of world state — no stale-quad class of bug, no
//! ordering hazard between the system that adds a quad and the one that removes
//! it, and an empty batch (a game with no HUD, or a paused frame that draws
//! none) costs literally nothing: no pass is encoded, no pipeline is compiled,
//! no buffer is allocated. The frame is byte-identical to one from an engine
//! without this module in it.
//!
//! # One atlas, unless the batch says otherwise
//!
//! [`UiBatch::atlas`] is still the frame's texture: one handle, set once, that
//! every textured quad samples. That is what a HUD wants — a glyph grid and an
//! icon sheet baked into one image is one bind group and one draw.
//!
//! What a *demo viewport* wants is different: a quad that samples an offscreen
//! scene target (a second world rendered by
//! [`Renderer::render_to_texture`](crate::Renderer::render_to_texture)),
//! sitting in the same painter's order as the panel behind it and the caption
//! over it. So a batch may also carry **texture runs**:
//! [`UiBatch::set_texture`] marks the point in the list where the texture
//! changes, and the pass draws one instanced call per run.
//!
//! Runs are *consecutive*, never sorted. Painter's order is the batch's only
//! ordering rule and grouping quads by texture would quietly break it — a
//! viewport drawn over its own caption is not a cheaper frame, it is a wrong
//! one. The cost of that choice is one draw call per texture change, which a
//! layout controls by pushing its quads in the order it already wanted.
//!
//! A batch that never calls `set_texture` is exactly one run, and the pass
//! encodes exactly the commands it encoded before runs existed.
//!
//! # Rings and arcs are quads too
//!
//! There is deliberately **no arc/ring shader mode**. The plan left it to the
//! implementer, and a second fragment path (SDF ring, `params`-driven sweep)
//! would double this pipeline's surface area to serve one meter. A ring is a fan
//! of small quads built game-side — sixty of them is still one draw call and one
//! `write_buffer`, which is the same cost as one quad plus arithmetic. P7's
//! meter ring builds it that way. Revisit only if the fan looks bad at the sizes
//! the HUD actually uses; the batch format does not change either way.

use std::collections::HashMap;

use bevy_ecs::prelude::Resource;
use bytemuck::{Pod, Zeroable};

use crate::bake::TextureRegistry;
use crate::texture::TextureHandle;

/// The UI pass's WGSL. Standalone (no feature consts), like
/// [`BLIT_SHADER`](crate::BLIT_SHADER).
pub const UI_SHADER: &str = include_str!("ui.wgsl");

// ---------------------------------------------------------------------------
// The batch
// ---------------------------------------------------------------------------

/// One screen-space rectangle: where it goes, what it samples, what colour it
/// is multiplied by.
///
/// 48 bytes, tightly packed, uploaded as a per-instance vertex (the same shape
/// as [`InstanceRaw`](crate::InstanceRaw), for the same reason: a whole HUD is
/// one buffer write and one draw).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct UiQuad {
    /// `x, y, w, h` in logical pixels, top-left origin, +Y down. Negative sizes
    /// are legal and simply wind the quad the other way — culling is off.
    pub rect: [f32; 4],
    /// `u0, v0, u1, v1` in the atlas, `0..1`, or [`SOLID`](UiQuad::SOLID) for a
    /// quad that ignores the atlas entirely.
    pub uv: [f32; 4],
    /// **Premultiplied** RGBA — see the module docs. Multiplied into the atlas
    /// sample, so white is "the atlas, untinted" and
    /// `[0.0, 0.0, 0.0, 0.0]` is invisible.
    pub color: [f32; 4],
}

impl UiQuad {
    /// The `uv` of a quad that ignores the atlas: all-negative, so the value
    /// interpolated across the quad stays negative and the fragment shader can
    /// recognise it with one compare.
    ///
    /// A sentinel rather than a second variant bit or a second draw: solid
    /// quads and atlas quads then live in one batch, in one painter's order,
    /// which is what a panel-behind-its-own-text needs.
    pub const SOLID: [f32; 4] = [-1.0, -1.0, -1.0, -1.0];

    /// A flat rectangle in a straight-alpha colour.
    pub fn solid(rect: [f32; 4], color: [f32; 4]) -> UiQuad {
        UiQuad {
            rect,
            uv: UiQuad::SOLID,
            color: UiQuad::rgba(color),
        }
    }

    /// An atlas region — a glyph, an icon — tinted by a straight-alpha colour.
    /// `uv` is `[u0, v0, u1, v1]`.
    pub fn textured(rect: [f32; 4], uv: [f32; 4], color: [f32; 4]) -> UiQuad {
        UiQuad {
            rect,
            uv,
            color: UiQuad::rgba(color),
        }
    }

    /// Straight alpha → premultiplied. The one conversion in the module, so a
    /// game never has to remember which side of it it is on.
    pub fn rgba(color: [f32; 4]) -> [f32; 4] {
        let a = color[3];
        [color[0] * a, color[1] * a, color[2] * a, a]
    }

    /// Slot 0, stepped per instance, attributes 0–2. The UI pipeline has no
    /// mesh buffer at all — the six vertices come from `vertex_index` — so this
    /// is the only vertex stream it declares.
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<UiQuad>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x4, // rect: x, y, w, h (logical px)
            1 => Float32x4, // uv: u0, v0, u1, v1 (or SOLID)
            2 => Float32x4, // premultiplied colour
        ],
    };
}

/// The frame's HUD, as a world resource: what to draw, and what to sample.
///
/// Filled Extract-side — a `FixedSim` system may write it like any other output
/// resource (the same seam [`StatusLine`](crate::StatusLine) uses), and nothing
/// in the engine reads it back, so no simulation state and no replay
/// fingerprint can depend on what the HUD looks like.
///
/// **Painter's order is `Vec` order**: later quads land on top. There is no
/// z field and no sort — a HUD is authored back-to-front anyway, and a stable
/// explicit order is worth more here than a sort key nobody can see.
///
/// Rebuilt each frame; see the module docs.
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct UiBatch {
    pub quads: Vec<UiQuad>,
    /// The texture every [`textured`](UiQuad::textured) quad in this batch
    /// samples, unless a [run](UiBatch::set_texture) overrides it — one atlas
    /// per batch, set once by the game (usually at load, from the glyph/icon
    /// bake) and left alone.
    ///
    /// `None` binds a 1×1 white texel instead, so solid quads work with zero
    /// setup: a game that only ever draws bars and panels never touches the
    /// texture system at all. A handle the renderer has never baked degrades to
    /// the same white texel rather than to a validation error.
    ///
    /// The handle is an ordinary [`TextureHandle`] — whatever
    /// [`Renderer::bake_texture`](crate::Renderer::bake_texture) or a
    /// `TextureLibrary` entry through
    /// [`Engine::bake_scene_textures`](crate::Engine::bake_scene_textures)
    /// produced. Unlike a draw item's texture it is **not** baked lazily at draw
    /// time: a batch is not a draw list, and a HUD that stalls the first frame
    /// to bake an atlas is exactly the hitch the load-time bake exists to
    /// prevent.
    pub atlas: Option<TextureHandle>,
    /// Where the sampled texture changes, as `(first quad index, texture)`,
    /// ascending. `None` means "back to [`atlas`](UiBatch::atlas)".
    ///
    /// Private because the ascending-and-deduplicated invariant is what makes
    /// [`runs`](UiBatch::runs) a partition of the quad list rather than a set
    /// of overlapping claims about it; [`set_texture`](UiBatch::set_texture) is
    /// the only way to add one. Empty — the case every HUD written before this
    /// existed is in — means one run over everything.
    switches: Vec<(u32, Option<TextureHandle>)>,
}

/// A contiguous stretch of quads sharing one texture: what the pass turns into
/// a single instanced draw.
///
/// `texture` is the run's *override*, not its resolved texture — `None` means
/// the batch's [`atlas`](UiBatch::atlas), which may itself be `None` (the white
/// texel). Resolution happens at encode time, where the registry is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UiRun {
    pub first: u32,
    pub count: u32,
    pub texture: Option<TextureHandle>,
}

impl UiBatch {
    pub fn new() -> UiBatch {
        UiBatch::default()
    }

    /// Drop every quad, keeping the allocation (and the atlas). The first call
    /// of a HUD system's frame.
    ///
    /// The texture runs go with the quads they described — a run list that
    /// outlived its quads would name indices that no longer exist — so a HUD
    /// that samples a second texture re-declares it each frame, exactly like it
    /// re-pushes the quad.
    pub fn clear(&mut self) {
        self.quads.clear();
        self.switches.clear();
    }

    pub fn push(&mut self, quad: UiQuad) {
        self.quads.push(quad);
    }

    /// Every quad pushed *after* this call samples `texture` — `None` meaning
    /// the batch's own [`atlas`](UiBatch::atlas).
    ///
    /// Painter's order is untouched: this marks a boundary in the list, it does
    /// not move anything across one. Calling it twice with the same value, or
    /// before any quad, costs nothing — the switch is folded rather than
    /// recorded, so a layout may bracket every widget with one without turning
    /// a HUD into a hundred draw calls.
    pub fn set_texture(&mut self, texture: Option<TextureHandle>) {
        if self.texture() == texture {
            return;
        }
        let first = self.quads.len() as u32;
        // A switch at the same index as the last one replaces it: nothing was
        // drawn between them, so the earlier value never applied to anything.
        match self.switches.last_mut() {
            Some(last) if last.0 == first => last.1 = texture,
            _ => self.switches.push((first, texture)),
        }
    }

    /// The texture the *next* pushed quad will sample, as an override: `None`
    /// is the batch atlas.
    pub fn texture(&self) -> Option<TextureHandle> {
        self.switches.last().and_then(|(_, texture)| *texture)
    }

    /// The batch as the pass draws it: consecutive runs, in painter's order,
    /// covering every quad exactly once. Empty runs are skipped, so the
    /// iterator is empty for an empty batch and yields exactly one item for the
    /// overwhelmingly common single-texture case.
    pub fn runs(&self) -> impl Iterator<Item = UiRun> + '_ {
        let total = self.quads.len() as u32;
        // The quads pushed before the first `set_texture` — the whole batch
        // when there was never one.
        let lead = match self.switches.first() {
            Some(&(first, _)) => first,
            None => total,
        };
        let switched = self.switches.iter().enumerate();
        let switched = switched.map(move |(i, &(first, texture))| {
            // A run ends where the next switch begins, or at the last quad.
            let end = match self.switches.get(i + 1) {
                Some(&(next, _)) => next,
                None => total,
            };
            UiRun {
                first,
                count: end.saturating_sub(first),
                texture,
            }
        });
        std::iter::once(UiRun {
            first: 0,
            count: lead,
            texture: None,
        })
        .chain(switched)
        .filter(|run| run.count > 0)
    }

    /// [`UiQuad::solid`], pushed.
    pub fn solid(&mut self, rect: [f32; 4], color: [f32; 4]) {
        self.quads.push(UiQuad::solid(rect, color));
    }

    /// [`UiQuad::textured`], pushed.
    pub fn textured(&mut self, rect: [f32; 4], uv: [f32; 4], color: [f32; 4]) {
        self.quads.push(UiQuad::textured(rect, uv, color));
    }

    pub fn len(&self) -> usize {
        self.quads.len()
    }

    pub fn is_empty(&self) -> bool {
        self.quads.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Getting an atlas in
// ---------------------------------------------------------------------------

/// A game-authored atlas, as raw pixels: the one way to make
/// [`UiBatch::atlas`] name something the game drew itself.
///
/// [`TextureLibrary`](crate::texture::TextureLibrary) is the engine's texture
/// door and it takes a [`TextureSpec`](crate::texture::TextureSpec) — a
/// *procedural* material of noise octaves and gradient ramps, which is exactly
/// the right vocabulary for a cliff face and cannot express a bitmap font. A
/// glyph atlas is the opposite kind of content: a few hundred bytes of authored
/// coverage that no generator would ever produce. So this is a second, much
/// smaller door, and it is deliberately UI-shaped rather than a general image
/// loader — the engine still has no file formats, no decoders and no notion of
/// an "asset".
///
/// # Contract
///
/// - `rgba` is `width · height · 4` bytes, row-major, **premultiplied** (see
///   the module docs: a white-on-transparent font writes `rgb = a`).
/// - `handle` is the game's to choose and must be unique to these pixels. The
///   registry is content-addressed everywhere else; here the *game* is the
///   content-addressing, because only it knows what went into the bake.
/// - Uploaded **once**, the first frame it is present and not already resident,
///   and never re-read. Changing the pixels means a new handle.
///
/// Empty (the default) means "no atlas of my own", which is every game that
/// draws only solid quads.
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct UiAtlasImage {
    pub handle: TextureHandle,
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl UiAtlasImage {
    /// Is there anything here worth uploading? Checks the length as well as the
    /// size, so a half-built image is "nothing" rather than a panic in wgpu.
    pub fn is_valid(&self) -> bool {
        self.width > 0
            && self.height > 0
            && self.rgba.len() == (self.width as usize) * (self.height as usize) * 4
    }
}

// ---------------------------------------------------------------------------
// The GPU side
// ---------------------------------------------------------------------------

/// `@group(0)` of the UI pipeline: the surface size, and nothing else.
///
/// Deliberately not [`FrameUniform`](crate::FrameUniform): that block is the
/// *render* target's size (see its `viewport` field), and the UI is drawn after
/// the blit at the host's own resolution. Sharing it would have made the HUD
/// shrink into the corner at any scale below 1.0.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct UiUniform {
    /// `xy` — the surface size in pixels; `zw` — its reciprocal.
    viewport: [f32; 4],
}

/// Quads allocated up front; the buffer doubles from here, like the instance
/// buffer it is modelled on.
const INITIAL_QUAD_CAPACITY: u32 = 64;

/// The UI pass: one pipeline, one uniform, one nearest sampler, one instance
/// buffer, and a bind group per atlas.
///
/// Built lazily by [`Renderer::ensure_ui`](crate::Renderer::ensure_ui) the first
/// frame a batch is non-empty — the blit's precedent, and for the blit's reason:
/// a game with no HUD must not pay for a shader module it never draws.
pub struct UiPass {
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    uniform_group: wgpu::BindGroup,
    /// The `@group(1)` layout: atlas texture + sampler.
    atlas_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// What a batch with no atlas — or with an atlas the registry has never
    /// baked — binds: a 1×1 opaque white texel, so `texel · color` is `color`.
    white_group: wgpu::BindGroup,
    _white: wgpu::Texture,
    /// Handle → its bind group. Built on first use and kept: the registry is
    /// content-addressed, so a handle's texture is never replaced under us.
    atlas_groups: HashMap<TextureHandle, wgpu::BindGroup>,
    /// The last atlas complained about, so a batch naming a texture that never
    /// arrives says so once rather than every frame.
    warned_atlas: Option<TextureHandle>,
    /// Each run's resolved bind-group key, computed before the pass opens
    /// because building a group needs `&mut self` and recording one needs
    /// `&self`. A field rather than a local so a steady HUD allocates nothing.
    resolved: Vec<Option<TextureHandle>>,
    instances: wgpu::Buffer,
    capacity: u32,
}

impl UiPass {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target_format: wgpu::TextureFormat,
    ) -> UiPass {
        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ui frame layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<UiUniform>() as u64
                    ),
                },
                count: None,
            }],
        });
        let atlas_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ui atlas layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        // Non-filterable + a non-filtering sampler, exactly like
                        // the blit: the HUD point-samples on purpose (a bitmap
                        // glyph blurred by a bilinear tap is a bug you can see),
                        // and asking for neither capability keeps the pipeline
                        // valid on the narrowest downlevel adapter.
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
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ui pipeline layout"),
            bind_group_layouts: &[Some(&uniform_layout), Some(&atlas_layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ui"),
            source: wgpu::ShaderSource::Wgsl(UI_SHADER.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ui quads"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_ui"),
                // Slot 0 is the quads. There is no mesh stream: the six corners
                // come from `vertex_index`.
                buffers: &[Some(UiQuad::LAYOUT)],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_ui"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(PREMULTIPLIED_BLEND),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                // No culling. A UI rect has no meaningful facing, and a negative
                // width or a flipped uv is a legitimate thing for a layout to
                // produce — culling would turn it into an invisible quad and a
                // long afternoon.
                cull_mode: None,
                ..Default::default()
            },
            // No depth attachment at all: the pass neither tests nor writes
            // depth, so painter's order is the only thing deciding overlap. It
            // also means the pass can run after the blit, where no depth buffer
            // of the right size need exist.
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ui uniform"),
            size: std::mem::size_of::<UiUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ui frame"),
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });

        // Nearest everywhere and clamped: a HUD samples texel centres inside an
        // atlas region, and a `Repeat` sampler would fetch a neighbouring glyph
        // the moment a uv rounded past its edge.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ui nearest"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let white = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ui white"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            white.as_image_copy(),
            &[255, 255, 255, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let white_group = bind_atlas(device, &atlas_layout, &sampler, &white, "ui white atlas");

        log::info!("ui: pipeline compiled ({target_format:?}, premultiplied, nearest)");
        UiPass {
            pipeline,
            uniform,
            uniform_group,
            atlas_layout,
            sampler,
            white_group,
            _white: white,
            atlas_groups: HashMap::new(),
            warned_atlas: None,
            resolved: Vec::new(),
            instances: create_quad_buffer(device, INITIAL_QUAD_CAPACITY),
            capacity: INITIAL_QUAD_CAPACITY,
        }
    }

    /// Quads the instance buffer holds before it has to grow.
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Upload `quads` and encode the pass over `view`, which must be the
    /// **surface** view (post-blit) at `width` × `height`.
    ///
    /// One `write_buffer` for the whole batch and one `draw` **per texture
    /// run**: there is no per-quad state, so a 200-quad HUD on one atlas is
    /// still one draw call, and a HUD with a demo viewport in the middle of it
    /// is three. Loads rather than clears — this paints *over* the frame that
    /// is already there.
    ///
    /// `runs` must partition `quads` in painter's order (see
    /// [`UiBatch::runs`]); each run's `texture` falls back to `atlas` when it is
    /// `None`. Never called with an empty batch (the caller checks), which is
    /// what makes "no HUD" mean "no pass".
    ///
    /// Eleven parameters is past clippy's taste for the same reason
    /// [`Renderer::render_scaled`](crate::Renderer::render_scaled)'s eight are:
    /// they are the GPU handles, the host's rectangle and the frame's content,
    /// three groups with nothing in common, and a struct would only move the
    /// same values one level out.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        width: u32,
        height: u32,
        quads: &[UiQuad],
        runs: &[UiRun],
        atlas: Option<TextureHandle>,
        textures: &TextureRegistry,
    ) {
        debug_assert!(!quads.is_empty(), "the caller skips the pass when empty");
        debug_assert!(!runs.is_empty(), "a non-empty batch has at least one run");
        let (w, h) = (width.max(1) as f32, height.max(1) as f32);
        queue.write_buffer(
            &self.uniform,
            0,
            bytemuck::bytes_of(&UiUniform {
                viewport: [w, h, 1.0 / w, 1.0 / h],
            }),
        );

        self.grow(device, quads.len() as u32);
        queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(quads));

        // Every run's bind group, resolved before the pass starts, because
        // building one needs `&mut self` and recording it needs `&self`. Moved
        // out of `self` for the duration for the same borrow-splitting reason
        // the renderer moves its working sets out of itself.
        let mut resolved = std::mem::take(&mut self.resolved);
        resolved.clear();
        for run in runs {
            let key = self.ensure_atlas(device, run.texture.or(atlas), textures);
            resolved.push(key);
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ui"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.uniform_group, &[]);
        pass.set_vertex_buffer(0, self.instances.slice(..));
        // Six vertices, the run's instances: one call per texture, and — for
        // the batch that never switched — one call for the whole HUD, which is
        // the command stream this pass encoded before runs existed, byte for
        // byte.
        let mut bound: Option<Option<TextureHandle>> = None;
        for (run, key) in runs.iter().zip(&resolved) {
            if bound != Some(*key) {
                let group = match key {
                    Some(handle) => &self.atlas_groups[handle],
                    None => &self.white_group,
                };
                pass.set_bind_group(1, group, &[]);
                bound = Some(*key);
            }
            pass.draw(0..6, run.first..run.first + run.count);
        }

        drop(pass);
        self.resolved = resolved;
    }

    /// Make sure this batch's atlas has a bind group, returning the key to look
    /// it up with — `None` meaning "bind the white texel".
    ///
    /// A handle the registry has not baked resolves to white rather than
    /// failing: the batch is render-side data a game rebuilt this frame, and a
    /// HUD that briefly draws untinted rectangles while a bake lands is a much
    /// better outcome than a panic.
    fn ensure_atlas(
        &mut self,
        device: &wgpu::Device,
        atlas: Option<TextureHandle>,
        textures: &TextureRegistry,
    ) -> Option<TextureHandle> {
        let handle = atlas?;
        if !self.atlas_groups.contains_key(&handle) {
            let Some(gpu) = textures.get(handle) else {
                // Once per handle, not once per frame: a HUD asking for a
                // texture that never arrives would otherwise narrate it sixty
                // times a second for as long as it stays broken.
                if self.warned_atlas != Some(handle) {
                    log::warn!(
                        "ui batch names atlas {:#018x}, which is not baked; drawing untextured",
                        handle.0
                    );
                    self.warned_atlas = Some(handle);
                }
                return None;
            };
            let group = bind_atlas(
                device,
                &self.atlas_layout,
                &self.sampler,
                &gpu.albedo,
                "ui atlas",
            );
            self.atlas_groups.insert(handle, group);
        }
        Some(handle)
    }

    /// Doubling, sticky — the instance buffer's rule, for the same reason: a
    /// HUD's quad count is stable after the first frame that shows it all.
    fn grow(&mut self, device: &wgpu::Device, needed: u32) {
        if needed <= self.capacity {
            return;
        }
        let capacity = needed.max(self.capacity.saturating_mul(2));
        log::debug!("ui quad buffer grew {} → {capacity}", self.capacity);
        self.instances = create_quad_buffer(device, capacity);
        self.capacity = capacity;
    }
}

/// `src·1 + dst·(1−srcα)`, on colour *and* alpha — the premultiplied composite.
///
/// Not [`BlendState::ALPHA_BLENDING`](wgpu::BlendState::ALPHA_BLENDING), which
/// multiplies the source by its own alpha a second time. The distinction is
/// invisible on a single opaque quad and obvious the moment two translucent ones
/// overlap, which is why the module states the convention rather than leaving it
/// to be inferred.
///
/// The alpha channel is composited rather than left alone (the additive
/// material's choice) because a UI *is* the frame's opacity where it covers it:
/// a host presenting through a compositor that reads alpha should see an opaque
/// HUD panel as opaque.
pub const PREMULTIPLIED_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
};

fn bind_atlas(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    texture: &wgpu::Texture,
    label: &str,
) -> wgpu::BindGroup {
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

/// The per-quad instance buffer: `capacity` × 48 bytes, no padding.
fn create_quad_buffer(device: &wgpu::Device, capacity: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ui quads"),
        size: (std::mem::size_of::<UiQuad>() as u64) * (capacity.max(1) as u64),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn straight_alpha_converts_to_premultiplied() {
        assert_eq!(UiQuad::rgba([1.0, 0.5, 0.0, 1.0]), [1.0, 0.5, 0.0, 1.0]);
        assert_eq!(UiQuad::rgba([1.0, 0.5, 0.0, 0.5]), [0.5, 0.25, 0.0, 0.5]);
        // Fully transparent is the zero vector whatever colour it claims — the
        // property that makes "nothing" and "invisible black" composite alike.
        assert_eq!(UiQuad::rgba([1.0, 1.0, 1.0, 0.0]), [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn a_solid_quad_carries_the_sentinel_and_a_textured_one_does_not() {
        let solid = UiQuad::solid([0.0, 0.0, 8.0, 8.0], [1.0; 4]);
        assert_eq!(solid.uv, UiQuad::SOLID);
        assert!(solid.uv.iter().all(|u| *u < 0.0), "the shader tests uv.x < 0");

        let glyph = UiQuad::textured([0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 0.25, 0.25], [1.0; 4]);
        assert!(glyph.uv[0] >= 0.0);
    }

    #[test]
    fn the_batch_is_painters_order_and_clears_without_losing_its_atlas() {
        let mut batch = UiBatch::new();
        batch.atlas = Some(TextureHandle(7));
        batch.solid([0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]);
        batch.solid([0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(batch.len(), 2);
        // Index order *is* draw order: the green quad is on top.
        assert_eq!(batch.quads[1].color, [0.0, 1.0, 0.0, 1.0]);

        batch.clear();
        assert!(batch.is_empty());
        assert_eq!(batch.atlas, Some(TextureHandle(7)), "the atlas is set once");
    }

    #[test]
    fn a_batch_that_never_switches_texture_is_one_run() {
        let mut batch = UiBatch::new();
        assert_eq!(batch.runs().count(), 0, "an empty batch draws nothing");

        batch.solid([0.0, 0.0, 1.0, 1.0], [1.0; 4]);
        batch.textured([0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0; 4]);
        let runs: Vec<UiRun> = batch.runs().collect();
        assert_eq!(
            runs,
            vec![UiRun {
                first: 0,
                count: 2,
                texture: None
            }],
            "one texture is one draw"
        );
    }

    #[test]
    fn texture_runs_partition_the_batch_in_painters_order() {
        let viewport = TextureHandle::render_target(0);
        let mut batch = UiBatch::new();
        batch.atlas = Some(TextureHandle(7));

        // A panel, a viewport quad, a caption over it: the demo-card layout,
        // which is the whole reason runs exist.
        batch.solid([0.0, 0.0, 100.0, 60.0], [0.0, 0.0, 0.0, 1.0]);
        batch.set_texture(Some(viewport));
        batch.textured([4.0, 4.0, 92.0, 40.0], [0.0, 0.0, 1.0, 1.0], [1.0; 4]);
        batch.set_texture(None);
        batch.textured([8.0, 48.0, 8.0, 8.0], [0.0, 0.0, 0.1, 0.1], [1.0; 4]);
        batch.textured([16.0, 48.0, 8.0, 8.0], [0.1, 0.0, 0.2, 0.1], [1.0; 4]);

        let run = |first, count, texture| UiRun {
            first,
            count,
            texture,
        };
        let runs: Vec<UiRun> = batch.runs().collect();
        let want = vec![run(0, 1, None), run(1, 1, Some(viewport)), run(2, 2, None)];
        assert_eq!(runs, want);
        // A partition: consecutive, gapless, covering every quad exactly once.
        let covered: u32 = runs.iter().map(|r| r.count).sum();
        assert_eq!(covered, batch.len() as u32);
        let ends = |run: &UiRun| run.first + run.count;
        assert!(
            runs.windows(2).all(|w| ends(&w[0]) == w[1].first),
            "the runs leave a gap or overlap: {runs:?}"
        );

        batch.clear();
        assert_eq!(batch.runs().count(), 0, "the runs go with their quads");
        assert_eq!(batch.texture(), None, "…and so does the current texture");
    }

    #[test]
    fn switching_to_the_same_texture_costs_no_run() {
        let handle = TextureHandle::render_target(3);
        let mut batch = UiBatch::new();
        // Bracketing every widget with a `set_texture` is a legitimate way to
        // write a layout, and it must not turn one draw into five.
        batch.set_texture(Some(handle));
        batch.set_texture(Some(handle));
        batch.solid([0.0, 0.0, 1.0, 1.0], [1.0; 4]);
        batch.set_texture(Some(handle));
        batch.solid([0.0, 0.0, 1.0, 1.0], [1.0; 4]);
        assert_eq!(batch.runs().count(), 1);

        // Switches with nothing between them collapse to the last one — the
        // earlier value never applied to a quad.
        batch.set_texture(None);
        batch.set_texture(Some(TextureHandle(9)));
        batch.solid([0.0, 0.0, 1.0, 1.0], [1.0; 4]);
        let runs: Vec<UiRun> = batch.runs().collect();
        assert_eq!(runs.len(), 2, "{runs:?}");
        assert_eq!(runs[1].texture, Some(TextureHandle(9)));
        assert_eq!(runs[1].first, 2);
    }

    #[test]
    fn the_quad_is_the_size_the_layout_claims() {
        assert_eq!(std::mem::size_of::<UiQuad>(), 48);
        assert_eq!(UiQuad::LAYOUT.array_stride, 48);
        assert_eq!(std::mem::size_of::<UiUniform>(), 16);
    }
}
