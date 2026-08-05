//! §7's **live** path, on a real GPU: the fragment shader evaluating a
//! `TextureSpec` per pixel, held against the same Rust evaluator the bake is.
//!
//! `tests/texture_bake.rs` is the model for this file, and the contract is the
//! stricter one. A baked texel goes through an 8-bit target, a mip chain, a
//! bilinear filter and a triplanar blend before it reaches a pixel, so that
//! test compares *the bake target* to the CPU twin and lets the sampler off.
//! Live has none of that machinery: the number the CPU computes is the number
//! the fragment computes, and the only thing between them is the 8-bit render
//! target. So the tolerance here is one LSB, not a percentile.
//!
//! ## The rig
//!
//! A unit quad in the XY plane at `z = 0`, drawn through an **orthographic**
//! projection with `w = 1` — so the interpolated `world_pos` at a pixel centre
//! is exactly the world point the CPU is asked about, with no perspective
//! division to disagree over. The light rig is flattened to identity (white
//! ambient above and below, black key), which makes `(hemi + key) · albedo`
//! collapse to `albedo` and turns the frame into a direct readout of the ramp.
//!
//! ## The precondition that has to hold
//!
//! Live eval has no mip chain, so it fades octaves out by pixel footprint
//! (`texture::LIVE_LOD_CELL_PIXELS`). A comparison against the CPU twin — which
//! has no camera and therefore no footprint — is only meaningful where *no*
//! octave has faded, so [`the_probe_window_leaves_every_octave_on`] pins the
//! rig's own precondition rather than leaving it to a comment.

use glam::{Mat4, Vec3, Vec4};
use runt_core::draw::{DrawItem, FrameParams};
use runt_core::mesh::MeshData;
use runt_core::registry::{MeshHandle, MeshLibrary};
use runt_core::texture::{self, TextureLibrary, TextureSpec, LIVE_LOD_CELL_PIXELS};
use runt_core::{Lighting, MaterialVariant, NoopCache, Renderer};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const SIZE: u32 = 256;

/// How many world metres the viewport spans, in both axes.
///
/// Small on purpose: the pixel footprint has to stay well under the finest
/// octave's cell so the live octave window leaves every octave at full weight
/// (see the module docs). 2 m over 256 px against `grass`'s 0.087 m finest cell
/// is roughly an order of magnitude of headroom.
const SPAN: f32 = 2.0;

/// The lower-left corner of the viewport in world space.
///
/// The middle of `grass`'s 27.8 m tile, so that the *baked* comparison below
/// (`the_live_field_is_the_baked_field`) samples the tile's interior where the
/// seamless wrap is the identity and the two fields must agree exactly.
fn origin(spec: &TextureSpec) -> f32 {
    spec.tile_meters() * 0.5 - SPAN * 0.5
}

fn renderer() -> Option<Renderer> {
    match pollster::block_on(Renderer::headless(FORMAT)) {
        Ok(r) => Some(r),
        Err(e) => {
            eprintln!("SKIP (no GPU adapter): {e}");
            None
        }
    }
}

/// A light rig that multiplies albedo by exactly one.
///
/// `hemi = mix(ground, sky, 0.5 + 0.5·n.y)`, and the quad's normal is `+Z`, so
/// `n.y = 0` and any sky/ground pair that agrees gives that colour flat. A black
/// key light removes the other term. The frame is then the albedo, unlit, which
/// is what makes a one-LSB comparison against the CPU twin possible at all.
fn flat_lighting() -> Lighting {
    Lighting {
        key_dir: Vec3::Y,
        key_color: Vec3::ZERO,
        sky_color: Vec3::ONE,
        ground_color: Vec3::ONE,
        horizon: None,
        ..Lighting::default()
    }
}

/// A quad spanning `[o, o+SPAN]²` in the XY plane at `z = 0`, facing `+Z`.
///
/// Counter-clockwise seen from `+Z`, which is where the camera is — the opaque
/// pipeline culls back faces and a wound-backwards quad renders as a hole in
/// the sky rather than as a failure anyone can read.
fn quad(o: f32) -> MeshData {
    let (a, b) = (o, o + SPAN);
    MeshData {
        positions: vec![
            Vec3::new(a, a, 0.0),
            Vec3::new(b, a, 0.0),
            Vec3::new(b, b, 0.0),
            Vec3::new(a, b, 0.0),
        ],
        normals: vec![Vec3::Z; 4],
        uvs: vec![glam::Vec2::ZERO; 4],
        colors: vec![Vec3::ONE; 4],
        indices: vec![0, 1, 2, 0, 2, 3],
    }
}

/// Orthographic, axis-aligned, `w = 1` everywhere: pixel centre → world point
/// is an exact affine map in both directions, which is the whole reason this
/// test can make a per-pixel claim.
fn view_proj(o: f32) -> Mat4 {
    // `directx` is the 0..1 depth convention wgpu wants — the same one
    // `runt_core::camera` builds its perspective with.
    glam::camera::rh::proj::directx::orthographic(o, o + SPAN, o, o + SPAN, -1.0, 1.0)
}

/// The world point at the centre of pixel `(x, y)` — the inverse of the map
/// above, spelled out rather than inverted numerically.
fn world_at(o: f32, x: u32, y: u32) -> Vec3 {
    Vec3::new(
        o + (x as f32 + 0.5) / SIZE as f32 * SPAN,
        // Row 0 is the top of the target and `+Y` is up, so v runs backwards.
        o + (1.0 - (y as f32 + 0.5) / SIZE as f32) * SPAN,
        0.0,
    )
}

/// Draw the quad wearing `spec` under `variant`, and read the pixels back.
fn render(renderer: &mut Renderer, spec: &TextureSpec, variant: MaterialVariant) -> Vec<u8> {
    let o = origin(spec);

    let mut library = MeshLibrary::new();
    let mesh: MeshHandle = library.insert(quad(o));

    let mut textures = TextureLibrary::new();
    // 64² is the floor; the live path never samples it, and a full-resolution
    // bake here would be seconds of load-time work for a texture nothing reads.
    let handle = textures.insert(spec.clone(), texture::MIN_RESOLUTION);
    renderer.bake_texture(spec, texture::MIN_RESOLUTION, &NoopCache);

    let draws = [DrawItem {
        entity: bevy_ecs::entity::Entity::from_raw_u32(0).expect("entity 0"),
        variant,
        mesh,
        model: Mat4::IDENTITY,
        base_color: Vec4::ONE,
        params: Vec4::ZERO,
        texture: Some(handle),
    }];

    let target = renderer.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("live probe"),
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

    renderer.render(
        &view,
        SIZE,
        SIZE,
        &FrameParams {
            view_proj: view_proj(o),
            lighting: flat_lighting(),
        },
        &draws,
        &library,
        &textures,
    );

    read_back(renderer, &target)
}

fn read_back(renderer: &Renderer, target: &wgpu::Texture) -> Vec<u8> {
    let (device, queue) = (renderer.device(), renderer.queue());
    let padded = (SIZE * 4).div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("live probe readback"),
        size: (padded * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
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

    let (tx, rx) = std::sync::mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    rx.recv().expect("map callback").expect("mapped");

    let mapped = readback.get_mapped_range(..).expect("mapped range");
    let mut pixels = Vec::with_capacity((SIZE * SIZE * 4) as usize);
    for row in 0..SIZE as usize {
        let start = row * padded as usize;
        pixels.extend_from_slice(&mapped[start..start + (SIZE * 4) as usize]);
    }
    drop(mapped);
    readback.unmap();
    pixels
}

fn pixel(pixels: &[u8], x: u32, y: u32) -> [f32; 3] {
    let i = (y as usize * SIZE as usize + x as usize) * 4;
    [
        pixels[i] as f32 / 255.0,
        pixels[i + 1] as f32 / 255.0,
        pixels[i + 2] as f32 / 255.0,
    ]
}

/// Sorted per-pixel max-channel errors against the CPU twin, over a
/// deterministic scatter — the same shape `texture_bake.rs` reports.
fn divergence(spec: &TextureSpec, pixels: &[u8], samples: u32) -> Vec<f32> {
    let o = origin(spec);
    let mut errors = Vec::with_capacity(samples as usize);
    for i in 0..samples {
        // Coprime strides, so the scatter walks the whole quad.
        let x = i.wrapping_mul(97) % SIZE;
        let y = i.wrapping_mul(59).wrapping_add(i / SIZE * 31) % SIZE;
        let want = spec.live_albedo_at(world_at(o, x, y));
        let got = pixel(pixels, x, y);
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
        "{label}: {} pixels — median {:.5} ({:.2} LSB), p99 {:.5} ({:.2} LSB), max {:.5}",
        errors.len(),
        median,
        median * 255.0,
        p99,
        p99 * 255.0,
        max
    );
    (median, p99, max)
}

// ---------------------------------------------------------------------------
// The precondition
// ---------------------------------------------------------------------------

#[test]
fn the_probe_window_leaves_every_octave_on() {
    // The rig's own assumption, stated as an assertion: at this framing the
    // live octave window has not started to fade anything, so the CPU twin
    // (which has no camera and weights every octave at 1) is the right thing to
    // compare against. If SPAN or SIZE ever moves far enough to break this, the
    // divergence tests would start failing for a reason that has nothing to do
    // with the shader.
    for (name, spec) in [("grass", texture::grass()), ("rock", texture::rock())] {
        let footprint = SPAN / SIZE as f32 * spec.live_cells_per_metre();
        let (min, max) = spec.live_octave_window(footprint, LIVE_LOD_CELL_PIXELS);
        let last = spec.octave_plan().len() as f32 - 1.0;
        println!(
            "{name}: footprint {footprint:.5} cells/px → window ({min:.2}, {max:.2}), \
             {} octaves",
            spec.octave_plan().len()
        );
        assert!(
            min >= last,
            "{name}: octave {last} is already fading at this framing ({min:.2} < {last})"
        );
    }
}

#[test]
fn the_live_field_is_the_baked_tile_where_the_wrap_is_the_identity() {
    // No GPU needed, and it is the claim the A/B toggle rests on: live is not
    // "a similar texture evaluated per pixel", it is *the same texture* with
    // the tile divided out. The bake samples octave i at `uv · span_i`; live
    // samples it at `world · (span_0 · world_scale) · freq_i`; and
    // `uv = world · world_scale`. Where the bake's seamless wrap is the
    // identity the two are the same number — **exactly** the same number, not
    // approximately — and that is only true because `live_seed_offset` folds
    // the offset back into the tile first.
    //
    // `grass` is the material this holds for and `rock` is the one it cannot:
    // rock's coarsest octave puts two cells across its tile, so the lattice
    // neighbourhood spans the whole tile and the wrap is active everywhere.
    // Both are asserted, because "no window" is a fact about that material's
    // tiling and not a hole in this test.
    let spec = texture::grass();
    let (lo, hi) = spec.live_agreement_window();
    assert!(
        hi.x - lo.x > 0.3 && hi.y - lo.y > 0.3,
        "grass's agreement window collapsed to {lo:?}..{hi:?}"
    );

    const N: u32 = 64;
    let mut worst = 0.0f32;
    let mut differing = 0usize;
    for i in 0..N * N {
        let u = lo.x + (i % N) as f32 / (N - 1) as f32 * (hi.x - lo.x);
        let v = lo.y + (i / N) as f32 / (N - 1) as f32 * (hi.y - lo.y);
        let world = Vec3::new(u * spec.tile_meters(), v * spec.tile_meters(), 0.0);
        let baked = spec.albedo_at(glam::Vec2::new(u, v));
        let live = spec.live_albedo_at(world);
        let delta = (baked - live).abs().max_element();
        worst = worst.max(delta);
        if delta > 0.0 {
            differing += 1;
        }
    }
    let total = (N * N) as usize;
    println!(
        "grass: agreement window {:.3}..{:.3} × {:.3}..{:.3} — \
         {differing}/{total} points differ at all, worst {worst:.5}",
        lo.x, hi.x, lo.y, hi.y
    );
    // "Bit-identical almost everywhere", not "close everywhere" — the two are
    // the same arithmetic, so anything but an exact match is a float accident
    // rather than a modelling difference, and there must be very few of them.
    //
    // The accidents belong to the *bake*: it carries the unfolded offset — in
    // the thousands — into every octave and wraps the cell index afterwards, so
    // at octave 4 it takes a `floor` of a number around 25 000, where f32
    // resolves about 0.002 of a cell. A sample that close to a boundary can
    // land either side, and `CellValue` is a step function of which cell won.
    // Folding the offset is what makes the live side the *more* precise of the
    // two; this is measuring how often the bake's precision shows.
    assert!(
        differing * 200 < total,
        "grass: {differing}/{total} points differ; that is a different field, \
         not a boundary tail"
    );
    // A flip at octave 4 moves the fBm by its amplitude over the normalizing
    // sum — a few hundredths of a ramp. A flip at octave 0 would move a fifth
    // of one, and that would mean the *mapping* is wrong rather than the last
    // bit of a hash input.
    assert!(
        worst < 0.05,
        "grass: worst difference {worst} is a coarse-octave flip, not a boundary"
    );

    // rock's window is empty, and the arithmetic says so rather than the test
    // quietly skipping it. A retune that gave rock a denser base octave would
    // open one, and this line is where that would be noticed.
    let (rlo, rhi) = texture::rock().live_agreement_window();
    assert!(
        rhi.x <= rlo.x,
        "rock's base octave got denser; the window is now {rlo:?}..{rhi:?} and \
         this test should start checking it"
    );

    // …and outside the window they diverge, which is not a defect: past it the
    // bake starts its second copy of the tile and the live field simply carries
    // on. If this ever stopped being true the live path would have grown a
    // wrap, and DESIGN §7's whole live/baked distinction with it.
    let repeat = spec.tile_meters();
    let a = spec.live_albedo_at(Vec3::new(repeat * 0.5, repeat * 0.5, 0.0));
    let b = spec.live_albedo_at(Vec3::new(repeat * 1.5, repeat * 0.5, 0.0));
    assert!(
        (a - b).abs().max_element() > 1e-4,
        "the live field repeated at the tile period; it is not unbounded"
    );
}

#[test]
fn both_modes_of_a_material_paint_the_same_colours() {
    // The weaker claim that has to hold for *every* material, including the
    // ones with no agreement window: whatever region of the field live lands
    // on, it is the same field, so it produces the same distribution of
    // colours. A live path that had lost the ramp, drifted off the contrast
    // curve, or normalized its fBm differently would show up as a shifted mean
    // here even where a point-for-point comparison is impossible.
    for (name, spec) in [("grass", texture::grass()), ("rock", texture::rock())] {
        let mean = |mut sample: Box<dyn FnMut(u32) -> Vec3>| {
            let mut sum = Vec3::ZERO;
            let mut lo = Vec3::splat(f32::MAX);
            let mut hi = Vec3::splat(f32::MIN);
            const M: u32 = 4096;
            for i in 0..M {
                let c = sample(i);
                sum += c;
                lo = lo.min(c);
                hi = hi.max(c);
            }
            (sum / M as f32, lo, hi)
        };
        let tile = spec.tile_meters();
        let s = spec.clone();
        let (baked_mean, baked_lo, baked_hi) = mean(Box::new(move |i| {
            s.albedo_at(glam::Vec2::new(
                (i % 64) as f32 / 64.0,
                (i / 64) as f32 / 64.0,
            ))
        }));
        let s = spec.clone();
        let (live_mean, live_lo, live_hi) = mean(Box::new(move |i| {
            // A different region of the field entirely — deliberately nowhere
            // near the tile, so this cannot pass by accidentally overlapping.
            s.live_albedo_at(Vec3::new(
                (i % 64) as f32 / 64.0 * tile * 3.0 + tile * 7.0,
                (i / 64) as f32 / 64.0 * tile * 3.0 - tile * 5.0,
                0.0,
            ))
        }));
        println!(
            "{name}: baked mean {baked_mean:?} live mean {live_mean:?}\n       \
             baked range {baked_lo:?}..{baked_hi:?}\n       \
             live  range {live_lo:?}..{live_hi:?}"
        );
        // The ramp's own span is the scale to judge against — 5% of it is well
        // inside sampling noise for 4096 points and nowhere near a material
        // that has drifted.
        let ramp_span = (baked_hi - baked_lo).max_element().max(1e-3);
        assert!(
            (baked_mean - live_mean).abs().max_element() < ramp_span * 0.05,
            "{name}: the two modes disagree about the material's average colour"
        );

        // Both stay inside the ramp the material authored. Not "the same
        // extremes": a tile is one wrapped period of the field and the live
        // window here is nine of them, so live legitimately reaches further
        // into the ramp's ends — `rock`'s tile holds two cells of its coarsest
        // octave and simply cannot show the tails. What must hold is that
        // neither mode leaves the gradient, which is what the clamp before the
        // lookup is for.
        let stops: Vec<Vec3> = spec.ramp.iter().map(|(_, c)| *c).collect();
        let ramp_lo = stops.iter().copied().fold(Vec3::splat(f32::MAX), Vec3::min);
        let ramp_hi = stops.iter().copied().fold(Vec3::splat(f32::MIN), Vec3::max);
        for (mode, lo, hi) in [
            ("baked", baked_lo, baked_hi),
            ("live", live_lo, live_hi),
        ] {
            assert!(
                lo.cmpge(ramp_lo - 1e-5).all() && hi.cmple(ramp_hi + 1e-5).all(),
                "{name} ({mode}): {lo:?}..{hi:?} leaves the ramp {ramp_lo:?}..{ramp_hi:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The shader against the twin
// ---------------------------------------------------------------------------

#[test]
fn live_eval_matches_the_cpu_twin() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let spec = texture::grass();
    let pixels = render(&mut renderer, &spec, MaterialVariant::LIVE_TEX);
    let errors = divergence(&spec, &pixels, 4000);
    let (median, p99, max) = report("live WGSL vs CPU, 5 octaves (authored)", &errors);

    // Tighter than the bake's budget, and it should be: there is no texel to
    // quantize, no filter to blend and no mip to pick. What is left is the
    // 8-bit target (half an LSB) and the last bits of a float.
    assert!(
        median <= 1.0 / 255.0,
        "median {median} ({:.2} LSB): live eval is not the twin's field",
        median * 255.0
    );
    assert!(
        p99 <= 2.0 / 255.0,
        "p99 {p99} ({:.2} LSB)",
        p99 * 255.0
    );
    // The irreducible tail is the same one the bake has: `CellValue` is a step
    // function of the nearest cell, so a fraction of an ULP either way can flip
    // which cell wins and move a whole ramp step. It is rare and it is not a
    // different field.
    let outliers = errors.iter().filter(|e| **e > 3.0 / 255.0).count();
    println!(
        "  {outliers}/{} pixels ({:.2}%) sit on a cell boundary",
        errors.len(),
        outliers as f32 * 100.0 / errors.len() as f32
    );
    assert!(
        outliers * 100 < errors.len(),
        "{outliers}/{} pixels diverged; that is a different field",
        errors.len()
    );
    assert!(max <= 0.15, "max {max}");
}

#[test]
fn live_eval_matches_the_cpu_twin_on_a_two_octave_spec() {
    // The tight version, mirroring `texture_bake.rs`: two octaves put no
    // feature anywhere near sub-pixel, so this is a per-pixel equality claim
    // rather than a statistical one, and a mis-ported hash or lattice shows up
    // here with nowhere to hide.
    let Some(mut renderer) = renderer() else {
        return;
    };
    let spec = TextureSpec {
        octaves: 2,
        ..texture::grass()
    };
    let pixels = render(&mut renderer, &spec, MaterialVariant::LIVE_TEX);
    let errors = divergence(&spec, &pixels, 4000);
    let (_, p99, max) = report("live WGSL vs CPU, 2 octaves", &errors);
    assert!(p99 <= 2.0 / 255.0, "p99 {p99} ({:.2} LSB)", p99 * 255.0);
    assert!(max <= 0.10, "max {max}");
}

#[test]
fn live_eval_matches_the_cpu_twin_for_rock() {
    // `rock` is the other authored material and it differs in every way that
    // matters to this code path: `ToEdge` normals, a different lattice density,
    // and a lacunarity that quantizes badly enough to bend the octave window.
    let Some(mut renderer) = renderer() else {
        return;
    };
    let spec = texture::rock();
    let pixels = render(&mut renderer, &spec, MaterialVariant::LIVE_TEX);
    let errors = divergence(&spec, &pixels, 4000);
    let (median, p99, _) = report("live WGSL vs CPU, rock", &errors);
    assert!(median <= 1.0 / 255.0, "median {median}");
    assert!(p99 <= 2.0 / 255.0, "p99 {p99}");
}

#[test]
fn the_live_normal_perturbs_without_changing_the_albedo() {
    // `NORMAL_MAP` is the expensive half and it must be *only* the normal: the
    // colour a fragment resolves to cannot depend on whether the crinkle is on.
    // With the flat light rig the normal term is invisible, so the two frames
    // must be byte-identical — and if the live gradient ever leaked into the
    // ramp lookup, this is what would catch it.
    let Some(mut renderer) = renderer() else {
        return;
    };
    let spec = texture::grass();
    let plain = render(&mut renderer, &spec, MaterialVariant::LIVE_TEX);
    let crinkled = render(
        &mut renderer,
        &spec,
        MaterialVariant::LIVE_TEX | MaterialVariant::NORMAL_MAP,
    );
    assert_eq!(plain, crinkled, "the crinkle moved the albedo");
}

#[test]
fn the_live_normal_actually_bends_the_light() {
    // …and the other half: under a light rig that *can* see a normal, turning
    // the crinkle on has to change the frame. A `perturb_normal` that silently
    // returned its input would pass every test above.
    let Some(mut renderer) = renderer() else {
        return;
    };
    let spec = texture::rock();
    let o = origin(&spec);

    let mut library = MeshLibrary::new();
    let mesh = library.insert(quad(o));
    let mut textures = TextureLibrary::new();
    let handle = textures.insert(spec.clone(), texture::MIN_RESOLUTION);
    renderer.bake_texture(&spec, texture::MIN_RESOLUTION, &NoopCache);

    // A key light almost edge-on to the quad, where a small tilt in the normal
    // is a large change in `max(dot(n, l), 0)`.
    let lighting = Lighting {
        key_dir: Vec3::new(0.9, 0.0, 0.44).normalize(),
        key_color: Vec3::ONE,
        sky_color: Vec3::ZERO,
        ground_color: Vec3::ZERO,
        horizon: None,
        ..Lighting::default()
    };

    let shoot = |renderer: &mut Renderer, variant: MaterialVariant| {
        let target = renderer.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("live normal probe"),
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
        renderer.render(
            &view,
            SIZE,
            SIZE,
            &FrameParams {
                view_proj: view_proj(o),
                lighting,
            },
            &[DrawItem {
                entity: bevy_ecs::entity::Entity::from_raw_u32(0).expect("entity 0"),
                variant,
                mesh,
                model: Mat4::IDENTITY,
                base_color: Vec4::ONE,
                params: Vec4::ZERO,
                texture: Some(handle),
            }],
            &library,
            &textures,
        );
        read_back(renderer, &target)
    };

    let flat = shoot(&mut renderer, MaterialVariant::LIVE_TEX);
    let bumpy = shoot(
        &mut renderer,
        MaterialVariant::LIVE_TEX | MaterialVariant::NORMAL_MAP,
    );

    let mut moved = 0usize;
    let mut worst = 0.0f32;
    for y in (0..SIZE).step_by(3) {
        for x in (0..SIZE).step_by(3) {
            let (a, b) = (pixel(&flat, x, y), pixel(&bumpy, x, y));
            let delta = (0..3).map(|c| (a[c] - b[c]).abs()).fold(0.0f32, f32::max);
            worst = worst.max(delta);
            if delta > 2.0 / 255.0 {
                moved += 1;
            }
        }
    }
    let probed = (SIZE as usize / 3 + 1).pow(2);
    println!("live crinkle moved {moved}/{probed} pixels, worst Δ {worst:.4}");
    assert!(
        moved * 4 > probed,
        "only {moved}/{probed} pixels moved; the crinkle is not reaching the light"
    );
    // …and it is a perturbation, not a replacement: a normal that had flipped
    // to face away would black the surface out wholesale.
    assert!(worst < 0.9, "the crinkle is not a perturbation (worst {worst})");
}

// ---------------------------------------------------------------------------
// The octave window
// ---------------------------------------------------------------------------

#[test]
fn the_octave_window_closes_as_the_footprint_grows() {
    // The live path's mip substitute, as arithmetic. A pixel covering a tenth
    // of an octave-0 cell resolves every octave; a pixel covering ten cells
    // resolves none of them, and the window says so rather than evaluating five
    // Voronoi loops of noise the screen cannot show.
    let spec = texture::grass();
    let octaves = spec.octave_plan().len();

    let close = spec.live_octave_window(0.001, LIVE_LOD_CELL_PIXELS);
    assert!(
        close.0 >= octaves as f32,
        "a sub-cell pixel must leave every octave on, got {close:?}"
    );

    let far = spec.live_octave_window(10.0, LIVE_LOD_CELL_PIXELS);
    assert!(
        far.1 <= 0.0,
        "a pixel covering ten cells resolves nothing, got {far:?}"
    );

    // Monotone in between, and the fade is exactly one octave wide so it can
    // never pop.
    let mut previous = f32::INFINITY;
    for step in 0..24 {
        let footprint = 0.001 * 1.6f32.powi(step);
        let (min, max) = spec.live_octave_window(footprint, LIVE_LOD_CELL_PIXELS);
        assert!((max - min - 1.0).abs() < 1e-5, "window {min}..{max}");
        assert!(max <= previous + 1e-4, "the window widened as the pixel grew");
        previous = max;
    }

    // Zero pixels is the "off" sentinel, which is what the bake passes.
    assert_eq!(
        spec.live_octave_window(1.0, 0.0),
        runt_core::noise::OCTAVE_LOD_OFF
    );
}

#[test]
fn a_faded_out_fragment_still_lands_inside_the_ramp() {
    // The fBm normalization divides by the amplitude actually used, so dropping
    // octaves must cost detail and never brightness (`noise::FbmAccum`). At
    // one octave the field is coarser; it is not darker, and it is not outside
    // the ramp. That is what lets the window fade without a visible band.
    let spec = texture::grass();
    let full = spec.live_albedo_at(Vec3::new(13.0, 9.0, 0.0));
    let ramp_min = spec.ramp.iter().map(|(_, c)| c.y).fold(f32::MAX, f32::min);
    let ramp_max = spec.ramp.iter().map(|(_, c)| c.y).fold(0.0f32, f32::max);
    assert!(full.y >= ramp_min - 1e-5 && full.y <= ramp_max + 1e-5, "{full:?}");

    let coarse = TextureSpec {
        octaves: 1,
        ..texture::grass()
    };
    let one = coarse.live_albedo_at(Vec3::new(13.0, 9.0, 0.0));
    assert!(one.y >= ramp_min - 1e-5 && one.y <= ramp_max + 1e-5, "{one:?}");
}
