//! The tweak panel, end to end and on a real GPU (`reflect` feature only).
//!
//! Three claims, and deliberately no fourth:
//!
//! 1. **A registered root becomes rows.** `install` → `register_resource` →
//!    `fields_of` → `panel.rows` is the whole pipeline a game wires up, and it
//!    is asserted through the public API rather than through the module's own
//!    internals.
//! 2. **An open panel puts quads in the batch, and a closed one puts none.**
//!    The batch is the engine's only UI output, so "does it draw" is a count.
//! 3. **Those quads land on the screen.** One frame through the real UI pass,
//!    read back, with the top-left corner — where the panel's rim goes —
//!    different from the same frame without it.
//!
//! There is **no screenshot pin on the text**. The panel draws with the *game's*
//! font ([`PanelFont`]), so a pinned image would be a pin on the test's own
//! stub font; and a debug overlay whose layout cannot be nudged without a golden
//! update is an overlay nobody nudges. The claim worth defending is "it drew",
//! not "it drew these 4 800 pixels".

#![cfg(feature = "reflect")]

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use runt_core::reflect::FieldRange;
use runt_core::tweak::{self, TweakValue};
use runt_core::tweak_panel::{self, PanelFont, TweakPanel};
use runt_core::{
    FrameParams, MeshLibrary, RenderScale, Renderer, TextureLibrary, UiBatch, UiQuad, Viewport,
};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// A stand-in for a game's bitmap font: one solid quad per character, in the
/// cell the real atlas would have put a glyph in.
///
/// Enough to prove the panel asks for text and that the text lands where the
/// layout says, without this file having to carry a font.
struct BlockFont;

const CELL: f32 = 8.0;

impl PanelFont for BlockFont {
    fn width(&self, text: &str, scale: f32) -> f32 {
        text.chars().count() as f32 * CELL * scale
    }

    fn text(&self, batch: &mut UiBatch, x: f32, y: f32, text: &str, scale: f32, color: [f32; 4]) {
        for (i, _) in text.chars().enumerate() {
            batch.push(UiQuad::solid(
                [
                    x + i as f32 * CELL * scale,
                    y,
                    CELL * scale - 1.0,
                    CELL * scale,
                ],
                color,
            ));
        }
    }
}

#[derive(Resource, Reflect, Clone, Copy, Debug, PartialEq)]
struct Weather {
    #[reflect(@FieldRange::new(0.0, 1.0))]
    clouds: f32,
    #[reflect(@FieldRange::new(0.0, 1.0))]
    sun: f32,
    storm: bool,
}

fn world() -> World {
    let mut world = World::new();
    world.insert_resource(Weather {
        clouds: 0.25,
        sun: 0.5,
        storm: false,
    });
    world.insert_resource(Viewport::new(1280, 720));
    world.insert_resource(TweakPanel::new());
    tweak::install(&mut world);
    tweak::register_resource::<Weather>(&mut world, "weather");
    world
}

/// The batch a game's `Set::Ui` would hand the renderer, with the panel in
/// whatever state `open` says.
fn batch_for(world: &mut World, open: bool) -> UiBatch {
    world.resource_mut::<TweakPanel>().set_open(open);
    let fields = tweak::fields_of(world);
    let overrides = world.resource::<tweak::TweakOverrides>().len();
    let mut panel = world.resource::<TweakPanel>().clone();
    let viewport = *world.resource::<Viewport>();
    let mut batch = UiBatch::new();
    tweak_panel::draw(
        &mut panel, &fields, &mut batch, &BlockFont, viewport, overrides,
    );
    // The panel remembers the window it drew, so hand it back rather than
    // dropping it — the touch hit test reads it next tick.
    *world.resource_mut::<TweakPanel>() = panel;
    batch
}

// ---------------------------------------------------------------------------
// 1. Registration produces rows
// ---------------------------------------------------------------------------

#[test]
fn a_registered_resource_becomes_a_group_and_its_fields() {
    let world = world();
    let fields = tweak::fields_of(&world);
    let paths: Vec<&str> = fields.iter().map(|f| f.path.as_str()).collect();
    assert_eq!(paths, ["weather.clouds", "weather.sun", "weather.storm"]);

    let panel = TweakPanel::new();
    // One header plus one row per field: the shape the layout below walks.
    assert_eq!(panel.rows(&fields).len(), fields.len() + 1);
}

// ---------------------------------------------------------------------------
// 2. Open draws, closed does not
// ---------------------------------------------------------------------------

#[test]
fn an_open_panel_fills_the_batch_and_a_closed_one_leaves_it_empty() {
    let mut world = world();
    assert!(
        batch_for(&mut world, false).is_empty(),
        "a closed panel is not a pass"
    );

    let open = batch_for(&mut world, true);
    assert!(
        open.len() > 10,
        "an open panel drew {} quads, which is not a panel",
        open.len()
    );
    // …and it is still one texture run, so the whole overlay is one draw call
    // on top of the game's HUD.
    assert_eq!(open.runs().count(), 1);
}

#[test]
fn an_edited_field_reads_back_changed_and_marks_itself_overridden() {
    let mut world = world();
    tweak::set_and_record(&mut world, "weather.clouds", TweakValue::Float(0.8)).expect("set");
    assert_eq!(world.resource::<Weather>().clouds, 0.8);

    let fields = tweak::fields_of(&world);
    let clouds = fields
        .iter()
        .find(|f| f.path == "weather.clouds")
        .expect("clouds");
    assert_eq!(clouds.value, TweakValue::Float(0.8));
    assert!(clouds.overridden, "an edited field is not marked");
    assert!(
        !fields
            .iter()
            .find(|f| f.path == "weather.sun")
            .expect("sun")
            .overridden,
        "an untouched field was marked"
    );

    // The overridden row is drawn in a different colour, which is the one piece
    // of information the panel encodes as colour rather than as text.
    let batch = batch_for(&mut world, true);
    assert!(
        batch
            .quads
            .iter()
            .any(|q| q.color == UiQuad::rgba(tweak_panel::OVERRIDDEN)),
        "no row drew in the overridden colour"
    );
}

// ---------------------------------------------------------------------------
// 3. …and those quads reach the screen
// ---------------------------------------------------------------------------

/// `copy_texture_to_buffer` wants a 256-byte row stride.
fn align_256(n: u32) -> u32 {
    n.div_ceil(256) * 256
}

fn read(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture, size: u32) -> Vec<u8> {
    let stride = align_256(size * 4);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tweak panel readback"),
        size: (stride * size) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: Some(size),
            },
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll device");
    rx.recv().expect("map").expect("mapped");
    let view = readback.get_mapped_range(..).expect("mapped range");
    let mut out = Vec::with_capacity((size * size * 4) as usize);
    for row in 0..size {
        let start = (row * stride) as usize;
        out.extend_from_slice(&view[start..start + (size * 4) as usize]);
    }
    drop(view);
    readback.unmap();
    out
}

#[test]
fn the_panel_actually_lands_on_the_frame() {
    let Ok(mut renderer) = pollster::block_on(Renderer::headless(FORMAT)) else {
        return eprintln!("SKIP tweak panel GPU test: no adapter");
    };
    let device = renderer.device().clone();
    let queue = renderer.queue().clone();
    let size = 512;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("tweak panel target"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mut world = world();
    world.insert_resource(Viewport::new(size, size));

    let mut frame = |quads: &[UiQuad]| {
        renderer.set_ui_quads(quads, None);
        renderer.render_scaled(
            &view,
            size,
            size,
            RenderScale::new(1.0),
            &FrameParams::default(),
            &[],
            &MeshLibrary::new(),
            &TextureLibrary::new(),
        );
        read(&device, &queue, &texture, size)
    };

    let bare = frame(&[]);
    let panel = batch_for(&mut world, true);
    assert!(!panel.is_empty());
    let painted = frame(&panel.quads);

    assert_ne!(bare, painted, "the panel drew nothing at all");
    // The rim's top-left corner, specifically: an overlay that changed *some*
    // pixel somewhere is not the same claim as one that landed where it said.
    let at = |data: &[u8], x: u32, y: u32| {
        let i = ((y * size + x) * 4) as usize;
        [data[i], data[i + 1], data[i + 2], data[i + 3]]
    };
    let x = tweak_panel::MARGIN as u32 + 2;
    let y = tweak_panel::MARGIN as u32 + 2;
    assert_ne!(
        at(&bare, x, y),
        at(&painted, x, y),
        "nothing was painted where the panel's backdrop goes"
    );
    // …and the frame outside it is untouched, which is what makes this an
    // overlay rather than a takeover.
    assert_eq!(
        at(&bare, size - 4, size - 4),
        at(&painted, size - 4, size - 4),
        "the panel painted outside its own box"
    );
}
