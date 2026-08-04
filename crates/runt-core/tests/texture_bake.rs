//! The texture bake pass, on a real GPU (DESIGN §7).
//!
//! The load-bearing test here is `the_bake_matches_the_cpu_twin`: it renders
//! grass with the WGSL library and holds real texels against
//! `runt_core::texture`'s Rust evaluator. That is the same contract
//! `headless_screenshot.rs` enforces between `sky.wgsl` and `sky.rs` — change
//! one copy of the noise and the other goes red — and it is what makes the CPU
//! twin trustworthy enough to build terrain colour on later.
//!
//! Every test skips (loudly) with no adapter, like the rest of the GPU suite.
//!
//! `cargo test -p runt-core --test texture_bake -- --nocapture` prints bake
//! timings at 512/1024/2048 and the measured WGSL-vs-CPU divergence.

use glam::Vec2;
use runt_core::texture::{self, TextureSpec};
use runt_core::{NoopCache, Renderer};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

fn renderer() -> Option<Renderer> {
    match pollster::block_on(Renderer::headless(FORMAT)) {
        Ok(r) => Some(r),
        Err(e) => {
            eprintln!("SKIP (no GPU adapter): {e}");
            None
        }
    }
}

/// Bake `spec` and read both targets back, tightly packed RGBA8.
fn bake(renderer: &mut Renderer, spec: &TextureSpec, resolution: u32) -> (Vec<u8>, Vec<u8>) {
    let handle = renderer.bake_texture(spec, resolution, &NoopCache);
    let gpu = renderer.textures().get(handle).expect("just baked");
    let albedo = runt_core::bake::read_target(
        renderer.device(),
        renderer.queue(),
        &gpu.albedo,
        resolution,
    )
    .expect("read albedo back");
    let normal = runt_core::bake::read_target(
        renderer.device(),
        renderer.queue(),
        &gpu.normal,
        resolution,
    )
    .expect("read normal back");
    (albedo, normal)
}

/// The texel centre a CPU sample has to use to line up with the rasterizer.
fn texel_uv(x: u32, y: u32, resolution: u32) -> Vec2 {
    Vec2::new(
        (x as f32 + 0.5) / resolution as f32,
        (y as f32 + 0.5) / resolution as f32,
    )
}

fn pixel(pixels: &[u8], x: u32, y: u32, resolution: u32) -> [f32; 3] {
    let i = (y as usize * resolution as usize + x as usize) * 4;
    [
        pixels[i] as f32 / 255.0,
        pixels[i + 1] as f32 / 255.0,
        pixels[i + 2] as f32 / 255.0,
    ]
}

/// Albedo divergence between the bake and the CPU twin, as sorted per-texel
/// max-channel errors over a deterministic scatter of the tile.
fn albedo_divergence(spec: &TextureSpec, albedo: &[u8], resolution: u32, samples: u32) -> Vec<f32> {
    let mut errors = Vec::with_capacity(samples as usize);
    for i in 0..samples {
        // Coprime strides, so the scatter walks the whole tile instead of
        // sampling one well-behaved block.
        let x = i.wrapping_mul(97) % resolution;
        let y = i.wrapping_mul(59).wrapping_add(i / resolution * 31) % resolution;
        let want = spec.albedo_at(texel_uv(x, y, resolution));
        let got = pixel(albedo, x, y, resolution);
        errors.push(
            (0..3)
                .map(|c| (got[c] - want[c]).abs())
                .fold(0.0f32, f32::max),
        );
    }
    errors.sort_by(f32::total_cmp);
    errors
}

fn report(label: &str, errors: &[f32]) -> (f32, f32, f32) {
    let median = errors[errors.len() / 2];
    let p99 = errors[errors.len() * 99 / 100];
    let max = *errors.last().expect("sampled something");
    println!(
        "{label}: {} texels — median {:.5} ({:.2} LSB), p99 {:.5}, max {:.5}",
        errors.len(),
        median,
        median * 255.0,
        p99,
        max
    );
    (median, p99, max)
}

/// The tight version: a well-sampled field, where every texel must agree.
///
/// Two octaves put 16 lattice cells across a 256² tile, so no texel is anywhere
/// near a cell boundary by accident and the comparison is a real per-texel
/// equality claim rather than a statistical one. If the port of a hash, of the
/// FCC rounding, of the wrap or of the fBm normalization is wrong *at all*,
/// this is where it shows.
#[test]
fn the_bake_matches_the_cpu_twin() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    const N: u32 = 256;
    let spec = TextureSpec {
        octaves: 2,
        base_resolution: N,
        ..texture::grass()
    };
    let (albedo, _) = bake(&mut renderer, &spec, N);
    let errors = albedo_divergence(&spec, &albedo, N, 4000);
    let (_, p99, max) = report("WGSL vs CPU, 2 octaves", &errors);

    assert!(
        p99 <= 3.0 / 255.0,
        "p99 was {p99} ({:.1} LSB); a well-sampled field must round-trip",
        p99 * 255.0
    );
    assert!(
        max <= 0.10,
        "max was {max}; even a boundary texel cannot be that far off a \
         well-sampled field"
    );
}

/// The authored 5-octave stack, where the top octaves are finer than a texel.
///
/// Octave 4 puts 318 cells across the tile, so adjacent texels land in
/// different cells and a float divergence can flip which one wins —
/// `CellValue` is a *step* function of the nearest cell, not a continuous one.
/// That tail is irreducible in principle, so the assertions below are stated as
/// "almost every texel" rather than "every texel"; in practice it is currently
/// 0.2% of texels and a twentieth of a ramp step, because the divergence that
/// feeds it is a fraction of an ULP.
#[test]
fn the_full_octave_stack_agrees_where_it_is_sampled() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    const N: u32 = 256;
    let spec = TextureSpec {
        base_resolution: N,
        ..texture::grass()
    };
    let (albedo, _) = bake(&mut renderer, &spec, N);
    let errors = albedo_divergence(&spec, &albedo, N, 4000);
    let (_, p99, max) = report("WGSL vs CPU, 5 octaves (authored)", &errors);

    assert!(
        p99 <= 3.0 / 255.0,
        "99% of texels must round-trip: p99 {p99} ({:.1} LSB)",
        p99 * 255.0
    );
    assert!(
        max <= 0.15,
        "max {max}: a sub-texel octave may flip cells, but the field must not \
         be a different field"
    );
    let outliers = errors.iter().filter(|e| **e > 3.0 / 255.0).count();
    println!(
        "  {outliers}/{} texels ({:.2}%) sit on a sub-texel cell boundary",
        errors.len(),
        outliers as f32 * 100.0 / errors.len() as f32
    );
    assert!(
        outliers * 100 < errors.len(),
        "{outliers}/{} texels diverged; that is a different field, not aliasing",
        errors.len()
    );
}

#[test]
fn the_normal_map_matches_the_cpu_twin() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    const N: u32 = 128;
    let spec = TextureSpec {
        base_resolution: N,
        ..texture::rock()
    };
    let (_, normal) = bake(&mut renderer, &spec, N);

    let mut worst = 0.0f32;
    let mut flips = 0usize;
    let mut count = 0usize;
    for y in (0..N).step_by(7) {
        for x in (0..N).step_by(5) {
            let want = spec.packed_normal_at(texel_uv(x, y, N));
            let got = pixel(&normal, x, y, N);
            let error = (0..3)
                .map(|c| (got[c] - want.to_array()[c]).abs())
                .fold(0.0f32, f32::max);
            count += 1;
            if error > 3.0 / 255.0 {
                flips += 1;
            }
            worst = worst.max(error);
        }
    }
    println!("normal WGSL vs CPU: {flips}/{count} outliers, max {worst:.5}");
    assert!(
        flips * 20 < count,
        "{flips}/{count} normal texels diverged, which is a different accumulation"
    );
}

#[test]
fn a_normal_free_spec_bakes_a_flat_normal_map() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    const N: u32 = 64;
    let spec = TextureSpec {
        normal: None,
        base_resolution: N,
        ..texture::grass()
    };
    let (_, normal) = bake(&mut renderer, &spec, N);
    for y in (0..N).step_by(9) {
        for x in (0..N).step_by(11) {
            let got = pixel(&normal, x, y, N);
            assert!(
                (got[0] - 0.5).abs() <= 2.0 / 255.0
                    && (got[1] - 0.5).abs() <= 2.0 / 255.0
                    && got[2] > 0.99,
                "normal-free bake left {got:?} at ({x},{y})"
            );
        }
    }
}

#[test]
fn baking_twice_is_bit_identical() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    const N: u32 = 128;
    let spec = TextureSpec {
        base_resolution: N,
        ..texture::grass()
    };

    let (a_albedo, a_normal) = bake(&mut renderer, &spec, N);
    // A second *renderer*, so this is not testing the registry's memoization —
    // it is testing that the pass itself is a pure function of the spec.
    let Some(mut other) = renderer_or_skip() else {
        return;
    };
    let (b_albedo, b_normal) = bake(&mut other, &spec, N);

    assert_eq!(a_albedo, b_albedo, "albedo bake is not deterministic");
    assert_eq!(a_normal, b_normal, "normal bake is not deterministic");
}

fn renderer_or_skip() -> Option<Renderer> {
    renderer()
}

#[test]
fn the_baked_tile_is_seamless_at_the_wrap() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    const N: u32 = 256;
    let spec = TextureSpec {
        base_resolution: N,
        ..texture::grass()
    };
    let (albedo, _) = bake(&mut renderer, &spec, N);

    // Column 0 and column N-1 are half a texel inside opposite edges, so once
    // the tile repeats they are *neighbours*. The step across that join must
    // therefore be an ordinary neighbour step — no larger than the largest step
    // between two adjacent interior columns anywhere in the texture. A blended
    // or mirrored "seamless" texture fails this; an exactly-wrapped one cannot.
    let step = |ax: u32, bx: u32, y: u32| {
        let a = pixel(&albedo, ax, y, N);
        let b = pixel(&albedo, bx, y, N);
        (0..3).map(|c| (a[c] - b[c]).abs()).fold(0.0f32, f32::max)
    };

    let mut worst_wrap = 0.0f32;
    let mut worst_interior = 0.0f32;
    for y in 0..N {
        worst_wrap = worst_wrap.max(step(0, N - 1, y));
        for x in 1..N {
            worst_interior = worst_interior.max(step(x, x - 1, y));
        }
    }
    println!("worst wrap step {worst_wrap:.4} vs worst interior step {worst_interior:.4}");
    assert!(
        worst_wrap <= worst_interior,
        "the wrap ({worst_wrap}) is a bigger discontinuity than anything inside \
         the tile ({worst_interior}), so the tile is not seamless"
    );
}

#[test]
fn two_qualities_share_an_identity_but_not_a_bake() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let spec = texture::grass();

    // DESIGN §7/§11: the tier picks the *resolution*, and a texture's content is
    // scale-free, so the spec's identity must not move with it.
    let low = spec.resolution(runt_core::Quality(0.25));
    let high = spec.resolution(runt_core::Quality(1.0));
    assert_ne!(low, high, "the tier must actually change the resolution");
    assert_eq!(spec.param_key(), spec.param_key());
    assert_ne!(spec.content_key(low), spec.content_key(high));

    let a = renderer.bake_texture(&spec, low, &NoopCache);
    let b = renderer.bake_texture(&spec, high, &NoopCache);
    assert_ne!(a, b, "two resolutions are two cache entries");
    assert_eq!(renderer.textures().len(), 2);
    // …but the same texture: sample both near the middle of the tile and demand
    // they agree on what colour is there.
    let lo_px = runt_core::bake::read_target(
        renderer.device(),
        renderer.queue(),
        &renderer.textures().get(a).expect("baked").albedo,
        low,
    )
    .expect("readback");
    let hi_px = runt_core::bake::read_target(
        renderer.device(),
        renderer.queue(),
        &renderer.textures().get(b).expect("baked").albedo,
        high,
    )
    .expect("readback");

    let lo = pixel(&lo_px, low / 2, low / 2, low);
    let hi = pixel(&hi_px, high / 2, high / 2, high);
    let error = (0..3).map(|c| (lo[c] - hi[c]).abs()).fold(0.0f32, f32::max);
    println!("centre texel {low}² {lo:?} vs {high}² {hi:?} (Δ {error:.4})");
    assert!(
        error < 0.12,
        "the same point of the same texture reads differently at two \
         resolutions ({error}); the tier is changing content, not detail"
    );
}

#[test]
fn baking_is_idempotent_per_handle() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let spec = texture::grass();
    let a = renderer.bake_texture(&spec, 128, &NoopCache);
    let b = renderer.bake_texture(&spec, 128, &NoopCache);
    assert_eq!(a, b);
    assert_eq!(renderer.textures().len(), 1, "one bake, not two");
}

// ---------------------------------------------------------------------------
// The mip chain
// ---------------------------------------------------------------------------

/// One texel of a level, as floats.
fn texel(pixels: &[u8], x: u32, y: u32, size: u32) -> [f32; 4] {
    let i = (y as usize * size as usize + x as usize) * 4;
    [
        pixels[i] as f32 / 255.0,
        pixels[i + 1] as f32 / 255.0,
        pixels[i + 2] as f32 / 255.0,
        pixels[i + 3] as f32 / 255.0,
    ]
}

#[test]
fn a_bake_carries_a_full_chain_down_to_one_texel() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    // 1×1 is one level, and the registry's default white/flat pair depends on
    // that being true without a special case.
    assert_eq!(runt_core::bake::mip_level_count(1), 1);
    assert_eq!(runt_core::bake::mip_level_count(256), 9);
    assert_eq!(runt_core::bake::mip_level_count(2048), 12);
    assert_eq!(runt_core::bake::mip_size(256, 3), 32);
    assert_eq!(runt_core::bake::mip_size(256, 99), 1, "floored, not zero");

    const N: u32 = 256;
    let spec = TextureSpec {
        base_resolution: N,
        ..texture::rock()
    };
    let handle = renderer.bake_texture(&spec, N, &NoopCache);
    let gpu = renderer.textures().get(handle).expect("just baked");

    for texture in [&gpu.albedo, &gpu.normal] {
        assert_eq!(
            texture.mip_level_count(),
            runt_core::bake::mip_level_count(N),
            "the texture was allocated with the wrong number of levels"
        );
        let chain =
            runt_core::bake::read_chain(renderer.device(), renderer.queue(), texture, N)
                .expect("read the chain back");
        assert_eq!(chain.len(), 9);
        for (level, pixels) in chain.iter().enumerate() {
            let size = runt_core::bake::mip_size(N, level as u32) as usize;
            assert_eq!(pixels.len(), size * size * 4, "level {level} is the wrong size");
        }
        // The last level is one texel and it is *written* — an unwritten level
        // reads back as the clear colour, opaque black, which a real average of
        // this material never is.
        let last = chain.last().expect("a chain");
        assert_eq!(last.len(), 4);
        assert!(
            last[0] != 0 || last[1] != 0 || last[2] != 0,
            "the 1×1 level is black, so the chain stopped early"
        );
    }
}

#[test]
fn every_albedo_mip_is_the_box_average_of_the_level_above() {
    // The claim that makes a mip a mip: level i+1 texel (x,y) is the mean of
    // the 2×2 block at (2x,2y) in level i. If the downsample were sampling the
    // *field* again at half resolution instead of filtering, this would fail —
    // and the shimmer it is supposed to fix would still be there.
    let Some(mut renderer) = renderer() else {
        return;
    };
    const N: u32 = 128;
    let spec = TextureSpec {
        base_resolution: N,
        ..texture::rock()
    };
    let handle = renderer.bake_texture(&spec, N, &NoopCache);
    let gpu = renderer.textures().get(handle).expect("just baked");
    let chain = runt_core::bake::read_chain(renderer.device(), renderer.queue(), &gpu.albedo, N)
        .expect("read the chain back");

    let mut worst = 0.0f32;
    for level in 1..chain.len() {
        let src_size = runt_core::bake::mip_size(N, level as u32 - 1);
        let dst_size = runt_core::bake::mip_size(N, level as u32);
        for y in 0..dst_size {
            for x in 0..dst_size {
                let mut want = [0.0f32; 4];
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let s = texel(&chain[level - 1], x * 2 + dx, y * 2 + dy, src_size);
                    for c in 0..4 {
                        want[c] += s[c] * 0.25;
                    }
                }
                let got = texel(&chain[level], x, y, dst_size);
                for c in 0..4 {
                    worst = worst.max((got[c] - want[c]).abs());
                }
            }
        }
    }
    println!(
        "worst albedo mip deviation from the box average: {worst:.5} ({:.2} LSB)",
        worst * 255.0
    );
    // One LSB of slack and no more: the average of four 8-bit values is exact
    // in fp32 and the only loss is the round back to 8 bits.
    assert!(worst <= 1.5 / 255.0, "mip texels are not box averages ({worst})");
}

#[test]
fn every_normal_mip_is_a_unit_vector_pointing_the_average_way() {
    // Two claims, and the second is why the normal chain is a different shader
    // from the colour chain. (1) Every level is a *unit* normal — averaging
    // packed normals without renormalizing shortens the vector, and a short
    // normal unpacks to a flatter surface, so the crinkle would fade with
    // distance by an amount that varies per texel. (2) It still points where
    // the four it came from pointed, so the flattening that *should* happen
    // (four normals that genuinely disagree average toward flat, and then
    // renormalize to a direction) is not thrown away either.
    let Some(mut renderer) = renderer() else {
        return;
    };
    const N: u32 = 128;
    let spec = TextureSpec {
        base_resolution: N,
        ..texture::rock()
    };
    let handle = renderer.bake_texture(&spec, N, &NoopCache);
    let gpu = renderer.textures().get(handle).expect("just baked");
    let chain = runt_core::bake::read_chain(renderer.device(), renderer.queue(), &gpu.normal, N)
        .expect("read the chain back");

    let unpack =
        |t: [f32; 4]| glam::Vec3::new(t[0] * 2.0 - 1.0, t[1] * 2.0 - 1.0, t[2] * 2.0 - 1.0);

    let mut worst_len = 0.0f32;
    let mut worst_dir = 0.0f32;
    for level in 1..chain.len() {
        let src_size = runt_core::bake::mip_size(N, level as u32 - 1);
        let dst_size = runt_core::bake::mip_size(N, level as u32);
        for y in 0..dst_size {
            for x in 0..dst_size {
                let got = unpack(texel(&chain[level], x, y, dst_size));
                worst_len = worst_len.max((got.length() - 1.0).abs());

                let mut sum = glam::Vec3::ZERO;
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    sum += unpack(texel(&chain[level - 1], x * 2 + dx, y * 2 + dy, src_size));
                }
                if sum.length() > 1e-3 {
                    worst_dir = worst_dir.max(got.angle_between(sum.normalize()));
                }
            }
        }
    }
    println!(
        "normal mips: worst |n|-1 = {worst_len:.4}, worst angle off the mean = {:.2}°",
        worst_dir.to_degrees()
    );
    // 8-bit packing puts a floor under both: a unit vector quantized to
    // 1/255 steps is up to ~0.9% off unit length and ~0.5° off direction.
    assert!(worst_len < 0.02, "a mip normal is not unit length ({worst_len})");
    assert!(
        worst_dir.to_degrees() < 1.5,
        "a mip normal points {:.2}° away from the average of its quad",
        worst_dir.to_degrees()
    );
}

/// Not an assertion so much as a measurement — DESIGN §7 puts the bake at load
/// time, so what matters is that it is *seconds*, not minutes, at the §11 cap.
#[test]
fn bake_timings_at_every_tier() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let spec = texture::grass();
    for resolution in [512u32, 1024, 2048] {
        let mut spec = spec.clone();
        // A distinct seed per size, so nothing is served out of the registry.
        spec.seed_offset = resolution as f32;
        let start = std::time::Instant::now();
        renderer.bake_texture(&spec, resolution, &NoopCache);
        // The bake is queued, not finished, until the device drains.
        renderer
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let elapsed = start.elapsed();
        println!(
            "bake {resolution}²  ({} octaves, FCC 19-cell): {:.1} ms",
            spec.octaves,
            elapsed.as_secs_f64() * 1000.0
        );
        assert!(
            elapsed.as_secs_f64() < 30.0,
            "a {resolution}² bake took {elapsed:?}, which is not load-time work"
        );
    }
}
