//! Headless render-to-texture smoke test (DESIGN §2, build step 1; §3/§4 for
//! the ECS drive).
//!
//! Proves the core really is windowless: it builds its own device with no
//! surface, ticks the sim to a fixed wall time, renders the demo scene into an
//! offscreen texture, reads the pixels back, and checks that geometry actually
//! landed on screen. This is the foundation for real screenshot/regression
//! tests later.
//!
//! `TIME` is a whole number of 60 Hz ticks (0.7 s = 42 ticks), so the frame is
//! captured at alpha ≈ 0 and the pose matches what the pre-ECS renderer drew
//! from a raw time value.

use runt_core::Engine;

const SIZE: u32 = 512;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const TIME: f64 = 0.7;
/// `TIME` in whole ticks.
const TICKS: u64 = 42;

/// `copy_texture_to_buffer` requires `bytes_per_row` to be a multiple of 256.
fn align_256(n: u32) -> u32 {
    n.div_ceil(256) * 256
}

/// Render one frame headless and return the tightly-packed RGBA8 pixels.
fn render_headless() -> Option<Vec<u8>> {
    let mut engine = match pollster::block_on(Engine::headless(FORMAT)) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("SKIP headless_screenshot: {e}");
            return None;
        }
    };

    let device = engine.device().clone();
    let queue = engine.queue().clone();

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("headless target"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    // The engine takes wall time from the caller and never reads a clock, so a
    // test rendering is exactly reproducible (DESIGN §4). Step it like a host
    // would: jumping straight to 0.7 s would (correctly) hit the 0.25 s
    // spiral-of-death clamp and drop most of the interval.
    // Quarter-tick steps rather than `i * TICK_DT`: feeding times that land
    // exactly on a boundary makes the final tick a coin-toss on f64 rounding.
    // Stopping the instant tick 42 lands leaves alpha at a quarter tick or less.
    engine.update(0.0);
    let mut t = 0.0;
    while engine.tick_count() < TICKS {
        t += runt_core::TICK_DT * 0.25;
        engine.update(t);
    }
    assert_eq!(engine.tick_count(), TICKS, "0.7 s at 60 Hz is 42 ticks");
    assert!(t <= TIME + runt_core::TICK_DT, "reached ~{TIME}s, got {t}");
    assert!(
        engine.alpha() <= 0.26,
        "captured just after a tick, alpha was {}",
        engine.alpha()
    );
    engine.render(&view, SIZE, SIZE);

    let unpadded_row = SIZE * 4;
    let padded_row = align_256(unpadded_row);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (padded_row * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(SIZE),
            },
        },
        wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    // Mapping callbacks only fire while the device is polled.
    let (tx, rx) = std::sync::mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll device");
    rx.recv().expect("map callback").expect("buffer mapped");

    let padded = readback.get_mapped_range(..).expect("mapped range");
    let mut pixels = Vec::with_capacity((unpadded_row * SIZE) as usize);
    for row in 0..SIZE as usize {
        let start = row * padded_row as usize;
        pixels.extend_from_slice(&padded[start..start + unpadded_row as usize]);
    }
    drop(padded);
    readback.unmap();

    Some(pixels)
}

#[test]
fn headless_render_draws_geometry() {
    let Some(pixels) = render_headless() else {
        // No GPU adapter in this environment (CI container without Vulkan/GL);
        // nothing to assert, but the code path above still compiled and ran.
        return;
    };

    assert_eq!(pixels.len(), (SIZE * SIZE * 4) as usize, "full frame read back");

    // runt_core::CLEAR_COLOR in 8-bit, the value every untouched pixel holds.
    let clear = [
        (runt_core::CLEAR_COLOR.r * 255.0).round() as u8,
        (runt_core::CLEAR_COLOR.g * 255.0).round() as u8,
        (runt_core::CLEAR_COLOR.b * 255.0).round() as u8,
    ];
    let tolerance = 2i32;
    let mut drawn = 0usize;
    for px in pixels.chunks_exact(4) {
        let differs = (0..3).any(|c| (px[c] as i32 - clear[c] as i32).abs() > tolerance);
        if differs {
            drawn += 1;
        }
    }

    let total = (SIZE * SIZE) as usize;
    let frac = drawn as f64 / total as f64;

    // A stable fingerprint of the frame — printed with `cargo test -- --nocapture`
    // so a future golden-image test has something to lock onto.
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in &pixels {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    println!("headless frame {SIZE}x{SIZE} t={TIME}: {drawn}/{total} non-clear pixels ({:.1}%), fnv1a=0x{hash:016x}", frac * 100.0);

    assert!(
        frac >= 0.05,
        "expected geometry to cover >=5% of the frame, got {:.2}% ({drawn}/{total})",
        frac * 100.0
    );
}
