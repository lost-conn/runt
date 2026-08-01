//! The CPU bridge, against a real GPU (DESIGN §10).
//!
//! `bridge.rs`'s unit tests prove [`unpad_rows`] against a synthetic buffer.
//! These prove the other half: that the bridge asks wgpu for the *right* stride,
//! at a width where getting it wrong is visible. Every test here uses a width
//! whose natural row length is **not** a multiple of 256, because at 1280 (which
//! is what the editor actually opens at) the padding is zero and a broken
//! implementation would pass.
//!
//! Skipped, not failed, on a machine with no usable adapter — the same
//! convention `runt-core`'s headless screenshot test uses.
//!
//! [`unpad_rows`]: runt_editor_core::bridge::unpad_rows

use runt_editor_core::bridge::{padded_bytes_per_row, FrameBridge};

/// 301 × 4 = 1204 bytes per row, padded to 1280. Every row therefore carries 76
/// bytes of padding, and a bridge that forgot to strip it would shear the image
/// by 19 pixels more on each successive row.
const WIDTH: u32 = 301;
const HEIGHT: u32 = 97;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn gpu() -> Option<Gpu> {
    match pollster::block_on(runt_core::headless_device()) {
        Ok((device, queue)) => Some(Gpu { device, queue }),
        Err(e) => {
            eprintln!("SKIP: no usable GPU adapter ({e})");
            None
        }
    }
}

/// A pattern whose every pixel encodes its own coordinates, so a misread of any
/// kind — wrong stride, wrong origin, transposed axes — shows up as a specific,
/// diagnosable mismatch rather than as "the colours look off".
fn pattern(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.extend_from_slice(&[
                (x % 256) as u8,
                (y % 256) as u8,
                ((x / 256) * 16 + (y / 256)) as u8,
                255,
            ]);
        }
    }
    pixels
}

fn upload(gpu: &Gpu, bridge: &FrameBridge, pixels: &[u8]) {
    let (width, height) = bridge.size();
    gpu.queue.write_texture(
        bridge.texture().as_image_copy(),
        pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            // The *upload* has no 256-byte rule (it is a queue write, not a
            // buffer copy), so this is the tightly-packed stride. The readback
            // is the side that has to pad.
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

#[test]
fn readback_reproduces_a_known_pattern_at_an_unaligned_width() {
    let Some(gpu) = gpu() else { return };

    // Confirm the test is actually testing something.
    assert_ne!(
        padded_bytes_per_row(WIDTH),
        WIDTH * 4,
        "this test is only meaningful at a width that needs padding"
    );

    let mut bridge = FrameBridge::new(&gpu.device, WIDTH, HEIGHT);
    let expected = pattern(WIDTH, HEIGHT);
    upload(&gpu, &bridge, &expected);

    let actual = bridge.read(&gpu.device, &gpu.queue);

    assert_eq!(
        actual.len(),
        (WIDTH * HEIGHT * 4) as usize,
        "the frame handed to SurfaceWriter must be tightly packed"
    );

    // Spot-check the corners and the diagonal first: those give a readable
    // failure. The exhaustive compare below is the real assertion.
    for &(x, y) in &[
        (0u32, 0u32),
        (WIDTH - 1, 0),
        (0, HEIGHT - 1),
        (WIDTH - 1, HEIGHT - 1),
        (WIDTH / 2, HEIGHT / 2),
    ] {
        let i = ((y * WIDTH + x) * 4) as usize;
        assert_eq!(
            &actual[i..i + 4],
            &[(x % 256) as u8, (y % 256) as u8, ((x / 256) * 16 + (y / 256)) as u8, 255],
            "pixel ({x}, {y}) is wrong — row stride handling"
        );
    }

    assert!(
        actual == expected.as_slice(),
        "the read-back frame differs from what was uploaded"
    );
}

/// The failure mode this rules out explicitly: leaving the padding in shifts
/// every row by a growing offset. Row 0 would still be right, which is why a
/// single-pixel check is not enough.
#[test]
fn no_row_is_shifted_by_the_padding() {
    let Some(gpu) = gpu() else { return };

    let mut bridge = FrameBridge::new(&gpu.device, WIDTH, HEIGHT);
    upload(&gpu, &bridge, &pattern(WIDTH, HEIGHT));
    let actual = bridge.read(&gpu.device, &gpu.queue).to_vec();

    for y in 0..HEIGHT {
        // The first pixel of every row must have x == 0.
        let i = ((y * WIDTH) * 4) as usize;
        assert_eq!(
            actual[i], 0,
            "row {y} does not start at x = 0; it is shifted by {} px",
            actual[i]
        );
        assert_eq!(actual[i + 1], (y % 256) as u8, "row {y} has the wrong y");
    }
}

#[test]
fn resizing_rebuilds_the_target_and_the_readback_buffer() {
    let Some(gpu) = gpu() else { return };

    let mut bridge = FrameBridge::new(&gpu.device, WIDTH, HEIGHT);
    assert_eq!(bridge.size(), (WIDTH, HEIGHT));

    // Grow to another unaligned width.
    let (w2, h2) = (517u32, 131u32);
    bridge.resize(&gpu.device, w2, h2);
    assert_eq!(bridge.size(), (w2, h2));

    upload(&gpu, &bridge, &pattern(w2, h2));
    let actual = bridge.read(&gpu.device, &gpu.queue);
    assert_eq!(actual.len(), (w2 * h2 * 4) as usize);
    assert_eq!(&actual[0..4], &[0, 0, 0, 255]);
    let last = ((h2 - 1) * w2 + (w2 - 1)) as usize * 4;
    assert_eq!(actual[last], ((w2 - 1) % 256) as u8);
    assert_eq!(actual[last + 1], ((h2 - 1) % 256) as u8);
}

#[test]
fn resizing_to_the_same_size_is_a_no_op() {
    let Some(gpu) = gpu() else { return };
    let mut bridge = FrameBridge::new(&gpu.device, 64, 64);
    upload(&gpu, &bridge, &pattern(64, 64));
    bridge.read(&gpu.device, &gpu.queue);
    let before = bridge.pixels().to_vec();

    bridge.resize(&gpu.device, 64, 64);
    // The texture was not recreated, so the pattern is still in it.
    let after = bridge.read(&gpu.device, &gpu.queue);
    assert_eq!(after, before.as_slice());
}

#[test]
fn a_zero_size_is_clamped_rather_than_crashing() {
    let Some(gpu) = gpu() else { return };
    // rinch reports (0, 0) for a surface before its first paint, and that value
    // reaches the bridge as an ordinary resize.
    let bridge = FrameBridge::new(&gpu.device, 0, 0);
    assert_eq!(bridge.size(), (1, 1));

    let mut bridge = FrameBridge::new(&gpu.device, 128, 128);
    bridge.resize(&gpu.device, 0, 40);
    assert_eq!(bridge.size(), (1, 40));
}

/// Not an assertion so much as a measurement: DESIGN §10 claims the CPU bridge
/// is "fast enough for an editor", and this is the number that claim rests on.
/// Run with `--nocapture` to see it.
#[test]
fn readback_cost_at_editor_resolution() {
    let Some(gpu) = gpu() else { return };

    for (w, h) in [(640u32, 360u32), (1280, 720), (1920, 1080)] {
        let mut bridge = FrameBridge::new(&gpu.device, w, h);
        upload(&gpu, &bridge, &pattern(w, h));

        // One warm-up: the first readback pays for buffer allocation.
        bridge.read(&gpu.device, &gpu.queue);

        let runs = 20;
        let started = std::time::Instant::now();
        for _ in 0..runs {
            bridge.read(&gpu.device, &gpu.queue);
        }
        let each = started.elapsed().as_secs_f32() * 1000.0 / runs as f32;
        println!(
            "readback {w}x{h}: {each:.2} ms/frame ({:.0} fps ceiling), \
             row {} → {} bytes",
            1000.0 / each,
            w * 4,
            padded_bytes_per_row(w),
        );
        assert!(
            each < 100.0,
            "{w}x{h} readback took {each:.1} ms — the CPU bridge is not viable at this size"
        );
    }
}
