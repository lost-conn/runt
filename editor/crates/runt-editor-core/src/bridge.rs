//! The CPU frame bridge (DESIGN §10).
//!
//! > *v1 uses the **CPU bridge**: runt-core renders on its own wgpu device to an
//! > offscreen texture, reads back RGBA8, and submits via `SurfaceWriter`.
//! > Version-independent, fast enough for an editor.* — DESIGN §10
//!
//! rinch is on a forked wgpu 27; runt-core is on wgpu 30. Two different semver
//! majors means two different `wgpu::Device` types that cannot share so much as
//! a texture handle, so the frame crosses the gap as bytes. This module is the
//! whole of that crossing; nothing else in the editor knows which path is
//! active, and swapping it for `GpuTextureRegistrar` when the versions converge
//! is a change to this file alone.
//!
//! ## The 256-byte row rule
//!
//! `copy_texture_to_buffer` requires `bytes_per_row` to be a multiple of
//! [`wgpu::COPY_BYTES_PER_ROW_ALIGNMENT`] (256). A 1280-wide RGBA8 frame is
//! 5120 bytes per row and needs no padding; a 1277-wide one is 5108 and gets
//! padded to 5120, with 12 bytes of garbage at the end of every row. Handing
//! that straight to `submit_frame` would shear the image progressively down the
//! screen — the classic symptom. [`unpad_rows`] is the fix, and it is a free
//! function precisely so it can be tested against a synthetic buffer with no GPU
//! in sight.
//!
//! ## Cost
//!
//! One `copy_texture_to_buffer`, one map, one memcpy per row. The map is a
//! synchronous stall on the GPU: the frame must be *finished* before its pixels
//! exist, so there is no pipelining here and never will be without a second
//! buffer in flight. At editor resolutions that is a few milliseconds — measured
//! rather than assumed, and reported in [`Stats::readback_ms`].
//!
//! [`Stats::readback_ms`]: crate::protocol::Stats::readback_ms

/// Round `n` up to the next multiple of `align`.
fn align_up(n: u32, align: u32) -> u32 {
    n.div_ceil(align) * align
}

/// The padded row stride a texture of `width` RGBA8 texels needs for
/// `copy_texture_to_buffer`.
pub fn padded_bytes_per_row(width: u32) -> u32 {
    align_up(width * 4, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
}

/// Copy `height` rows of `unpadded_row` bytes out of a buffer whose rows are
/// `padded_row` bytes apart, dropping the padding.
///
/// `out` is cleared and refilled, so one `Vec` serves every frame.
///
/// Panics if `padded` is too short for the geometry described — that is a
/// programming error in the caller's buffer sizing, not a runtime condition, and
/// a silent short read here would be a corrupted frame nobody could explain.
pub fn unpad_rows(
    padded: &[u8],
    padded_row: usize,
    unpadded_row: usize,
    height: usize,
    out: &mut Vec<u8>,
) {
    assert!(
        unpadded_row <= padded_row,
        "unpadded row {unpadded_row} cannot exceed padded row {padded_row}"
    );
    let needed = padded_row * height;
    assert!(
        padded.len() >= needed,
        "readback buffer is {} bytes, need {needed} for {height} rows of {padded_row}",
        padded.len()
    );

    out.clear();
    out.reserve(unpadded_row * height);
    for row in 0..height {
        let start = row * padded_row;
        out.extend_from_slice(&padded[start..start + unpadded_row]);
    }
}

/// An offscreen render target plus the machinery to get its pixels onto the CPU.
///
/// Owns no engine and no device: it borrows both, so a caller can resize it
/// without disturbing anything the engine is holding.
pub struct FrameBridge {
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    /// `padded_bytes_per_row(width) * height` bytes of `COPY_DST | MAP_READ`.
    readback: wgpu::Buffer,
    padded_row: u32,
    /// The tightly-packed frame, reused across frames so a steady state does no
    /// allocation at all.
    pixels: Vec<u8>,
    last_readback_ms: f32,
}

impl FrameBridge {
    /// The format the engine must be built for. `Rgba8Unorm` because
    /// `SurfaceWriter` wants RGBA8 in that byte order and no swizzle should
    /// happen on the CPU.
    pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> FrameBridge {
        let (width, height) = (width.max(1), height.max(1));
        let format = FrameBridge::FORMAT;
        let (texture, view) = create_target(device, format, width, height);
        let padded_row = padded_bytes_per_row(width);
        let readback = create_readback(device, padded_row, height);
        FrameBridge {
            width,
            height,
            format,
            texture,
            view,
            readback,
            padded_row,
            pixels: Vec::new(),
            last_readback_ms: 0.0,
        }
    }

    /// Rebuild the target at a new size, or do nothing if it already matches.
    ///
    /// Idempotent on purpose: the UI sends its layout size every time rinch
    /// measures the surface, which is every frame.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        if width == self.width && height == self.height {
            return;
        }
        let (texture, view) = create_target(device, self.format, width, height);
        self.padded_row = padded_bytes_per_row(width);
        self.readback = create_readback(device, self.padded_row, height);
        self.texture = texture;
        self.view = view;
        self.width = width;
        self.height = height;
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// What [`Engine::render`](runt_core::Engine::render) draws into.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// The target texture itself. Exposed so a test can put a known pattern in
    /// it and prove the row handling on the way out.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// Milliseconds the last [`read`](FrameBridge::read) spent.
    pub fn last_readback_ms(&self) -> f32 {
        self.last_readback_ms
    }

    /// Pull the target's pixels to the CPU, tightly packed, RGBA8.
    ///
    /// Blocks until the GPU has finished the frame — see the module docs. The
    /// returned slice is valid until the next call.
    pub fn read(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> &[u8] {
        let started = std::time::Instant::now();

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runt-editor readback"),
            });
        encoder.copy_texture_to_buffer(
            self.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        // The map callback only runs while the device is polled, so the poll is
        // not optional bookkeeping — it is what makes the map happen.
        let (tx, rx) = std::sync::mpsc::channel();
        self.readback.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        if let Err(e) = device.poll(wgpu::PollType::wait_indefinitely()) {
            log::error!("runt-editor: device poll failed during readback: {e:?}");
            return &self.pixels;
        }
        match rx.recv() {
            Ok(Ok(())) => {}
            other => {
                log::error!("runt-editor: readback buffer did not map: {other:?}");
                return &self.pixels;
            }
        }

        {
            let mapped = self
                .readback
                .get_mapped_range(..)
                .expect("the buffer just mapped successfully");
            unpad_rows(
                &mapped,
                self.padded_row as usize,
                (self.width * 4) as usize,
                self.height as usize,
                &mut self.pixels,
            );
        }
        self.readback.unmap();

        self.last_readback_ms = started.elapsed().as_secs_f32() * 1000.0;
        &self.pixels
    }

    /// The last frame read, without reading a new one.
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }
}

fn create_target(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("runt-editor viewport"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        // `COPY_DST` is not needed by the engine, but it is what lets a test
        // write a known pattern straight into the target.
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn create_readback(device: &wgpu::Device, padded_row: u32, height: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("runt-editor readback"),
        size: (padded_row as u64) * (height as u64),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_stride_is_rounded_up_to_256() {
        // 1280 × 4 = 5120, already aligned: no padding at all.
        assert_eq!(padded_bytes_per_row(1280), 5120);
        // 1277 × 4 = 5108 → 5120, twelve bytes of padding per row.
        assert_eq!(padded_bytes_per_row(1277), 5120);
        // 300 × 4 = 1200 → 1280.
        assert_eq!(padded_bytes_per_row(300), 1280);
        assert_eq!(padded_bytes_per_row(1), 256);
        assert_eq!(padded_bytes_per_row(64), 256);
    }

    /// The failure this guards against is a sheared image: with the padding left
    /// in, row *n* of the output starts `n × padding` bytes late.
    #[test]
    fn unpadding_drops_the_tail_of_every_row() {
        let width = 3usize; // 12 bytes of pixel data
        let unpadded_row = width * 4;
        let padded_row = 16; // 4 bytes of padding per row
        let height = 4usize;

        // Fill each row with its own byte value, then poison the padding.
        let mut padded = vec![0xEEu8; padded_row * height];
        for row in 0..height {
            for byte in 0..unpadded_row {
                padded[row * padded_row + byte] = (row * 100 + byte) as u8;
            }
        }

        let mut out = Vec::new();
        unpad_rows(&padded, padded_row, unpadded_row, height, &mut out);

        assert_eq!(out.len(), unpadded_row * height);
        for row in 0..height {
            for byte in 0..unpadded_row {
                assert_eq!(
                    out[row * unpadded_row + byte],
                    (row * 100 + byte) as u8,
                    "row {row} byte {byte} came from the wrong place"
                );
            }
        }
        assert!(!out.contains(&0xEE), "padding leaked into the frame");
    }

    #[test]
    fn unpadding_is_a_straight_copy_when_nothing_is_padded() {
        let src: Vec<u8> = (0..48u8).collect();
        let mut out = Vec::new();
        unpad_rows(&src, 12, 12, 4, &mut out);
        assert_eq!(out, src);
    }

    #[test]
    fn unpadding_reuses_its_output_buffer() {
        let src = vec![1u8; 64];
        let mut out = vec![9u8; 1000];
        unpad_rows(&src, 16, 8, 4, &mut out);
        assert_eq!(out.len(), 32, "the buffer is cleared, not appended to");
        assert!(out.iter().all(|&b| b == 1));
    }

    #[test]
    #[should_panic(expected = "readback buffer is")]
    fn unpadding_refuses_a_short_buffer() {
        let mut out = Vec::new();
        unpad_rows(&[0u8; 10], 16, 8, 4, &mut out);
    }
}
