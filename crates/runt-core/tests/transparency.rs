//! The blended pass (D2): which draws leave the opaque state-sort, what order
//! they end up in, and what the frame looks like once they are drawn.
//!
//! Three claims, in ascending order of how expensive they are to check:
//!
//! 1. **The partition.** A draw carrying `TRANSPARENT` or `ADDITIVE` is drawn
//!    after every opaque draw, whatever its variant bits sort to numerically.
//! 2. **The order is total.** Blended draws go back to front, and any two
//!    identical worlds — including one spawned in the opposite order — produce
//!    the identical list. Depth is a float; the *key* is not.
//! 3. **The pixels.** One transparent, one additive and two phase-circle quads
//!    over an opaque floor, checked against the blend arithmetic done by hand,
//!    plus the phase circle's edge fringe checked against `strength`.
//!
//! **The screen effect is in these frames.** A circle with a radius on it turns
//! on `blit.wgsl`'s luminance inversion (see `tests/phase_screen.rs`), and this
//! rig aims one — so anything the probes read *inside* the circle is the blend
//! result seen through that effect. Rather than dodge it, the expectations
//! compose the two: [`phase_screen_color`] is applied to the colour the blend
//! arithmetic predicts, which incidentally pins the thing neither test could
//! see alone — that the effect's circle and the material shaders' discard
//! circle are the same circle, to the pixel.
//!
//! The pixel values are golden in the sense that matters: they are the exact
//! result of `src·α + dst·(1−α)` (and of `src·α + dst`) on colours chosen so the
//! answer is a round number, not a hash of one machine's rasterizer. A hash
//! would pin the same claim and fail on the next GPU for reasons that are not
//! bugs, so the frame's `fnv1a` is *printed* — the same convention
//! `headless_screenshot` already follows — and the arithmetic is *asserted*.

use bevy_ecs::prelude::*;
use glam::{Mat4, Vec2, Vec3, Vec4};
use runt_core::draw::{build_draw_list, sort_draw_list_for_view, DrawItem};
use runt_core::ecs::{phase_screen_color, PHASE_EDGE};
use runt_core::registry::{MeshHandle, MeshLibrary};
use runt_core::texture::TextureLibrary;
use runt_core::{Material, MaterialVariant, Mesh, MeshRef, Renderer, Transform};

const TRANSPARENT: MaterialVariant = MaterialVariant::TRANSPARENT;
const ADDITIVE: MaterialVariant = MaterialVariant::ADDITIVE;
const UNLIT: MaterialVariant = MaterialVariant::BILLBOARD_UNLIT;
const PHASE: MaterialVariant = MaterialVariant::PHASE_CIRCLE;

// ---------------------------------------------------------------------------
// 1 + 2. The partition and the order — no GPU
// ---------------------------------------------------------------------------

/// A drawable at `z`, so the depth sort has something to disagree about.
fn spawn(world: &mut World, mesh: u64, variant: MaterialVariant, z: f32) -> Entity {
    world
        .spawn((
            MeshRef(MeshHandle(mesh)),
            Material {
                base_color: Vec4::ONE,
                params: Vec4::ZERO,
                texture: None,
                variant,
            },
            Transform::from_translation(Vec3::new(0.0, 0.0, z)),
        ))
        .id()
}

/// A camera at the origin looking down −Z: a point at `z = -d` is `d` away.
fn view_proj() -> Mat4 {
    let view = glam::camera::rh::view::look_at_mat4(Vec3::ZERO, -Vec3::Z, Vec3::Y);
    runt_core::Camera::default().projection(1.0) * view
}

#[test]
fn blended_draws_leave_the_opaque_sort() {
    let mut world = World::new();
    // PHASE_CIRCLE and BILLBOARD_UNLIT are numerically *above* the blend bits
    // and are ordinary opaque looks; if the pass were folded into the variant
    // key they would sort into the blended half and the frame would be wrong in
    // a way no unit test on `variant.bits()` could see.
    let opaque_high = spawn(&mut world, 1, PHASE | UNLIT, -1.0);
    let ghost = spawn(&mut world, 1, TRANSPARENT, -2.0);
    let opaque_low = spawn(&mut world, 1, MaterialVariant::NONE, -3.0);
    let glow = spawn(&mut world, 1, ADDITIVE, -4.0);

    let items = build_draw_list(&mut world, 0.0);
    let order: Vec<Entity> = items.iter().map(|i| i.entity).collect();
    assert_eq!(
        order,
        vec![opaque_low, opaque_high, ghost, glow],
        "opaque first, by state; blended last"
    );
    assert_eq!(
        items.iter().map(DrawItem::pass).collect::<Vec<_>>(),
        vec![0, 0, 1, 1]
    );
    assert!(!items[1].is_blended() && items[2].is_blended());
}

#[test]
fn blended_draws_go_back_to_front() {
    let mut world = World::new();
    let near = spawn(&mut world, 1, TRANSPARENT, -1.0);
    let far = spawn(&mut world, 1, TRANSPARENT, -9.0);
    let middle = spawn(&mut world, 1, ADDITIVE, -5.0);
    let floor = spawn(&mut world, 1, MaterialVariant::NONE, -3.0);

    let mut items = build_draw_list(&mut world, 0.0);
    sort_draw_list_for_view(&mut items, &view_proj());

    let order: Vec<Entity> = items.iter().map(|i| i.entity).collect();
    assert_eq!(
        order,
        vec![floor, far, middle, near],
        "farthest blended draw first, and the opaque one before all of them"
    );

    // The depth order is by distance, not by variant: `middle` is ADDITIVE and
    // sits between two TRANSPARENT draws, so a sort that grouped by pipeline
    // first could not produce this.
    assert_ne!(items[1].variant, items[2].variant);
}

#[test]
fn the_blended_order_is_total_and_spawn_order_free() {
    // The same five draws, spawned in five different orders. This is the
    // determinism claim `draw_list::spawn_order_does_not_change_the_draw_order`
    // makes about the opaque half, asked of the depth-sorted one.
    let build = |rotate: usize| {
        let mut world = World::new();
        let mut specs = vec![
            (1u64, TRANSPARENT, -2.0f32),
            (2, ADDITIVE, -7.0),
            (3, MaterialVariant::NONE, -1.0),
            (4, TRANSPARENT, -6.0),
            (5, ADDITIVE | MaterialVariant::DEPTH_GREATER, -4.0),
        ];
        specs.rotate_left(rotate);
        for (mesh, variant, z) in specs {
            spawn(&mut world, mesh, variant, z);
        }
        let mut items = build_draw_list(&mut world, 0.0);
        sort_draw_list_for_view(&mut items, &view_proj());
        // Meshes, not entities: the entity *ids* legitimately differ between
        // two worlds spawned in different orders. What must not differ is the
        // command stream's content.
        items.iter().map(|i| i.mesh.0).collect::<Vec<_>>()
    };

    let reference = build(0);
    for rotate in 1..5 {
        assert_eq!(build(rotate), reference, "spawn order leaked into the draw order");
    }
    // Opaque first, then 7 units away, then 6, then 4, then 2.
    assert_eq!(reference, vec![3, 2, 4, 5, 1]);

    // Sorting twice changes nothing — the order is a fixed point, which is what
    // lets the renderer sort a list `Extract` already sorted.
    let mut world = World::new();
    spawn(&mut world, 1, TRANSPARENT, -2.0);
    spawn(&mut world, 2, ADDITIVE, -7.0);
    let mut once = build_draw_list(&mut world, 0.0);
    sort_draw_list_for_view(&mut once, &view_proj());
    let mut twice = once.clone();
    sort_draw_list_for_view(&mut twice, &view_proj());
    assert_eq!(once, twice);
}

#[test]
fn an_exact_depth_tie_falls_to_the_entity() {
    // Two blended draws at *the same* distance. There is no right answer, and
    // the sort's job is to have the same wrong one every time: the tie-break is
    // the entity index, so the order is a property of the world rather than of
    // the comparator. (Which also means it is the one case where a differently
    // *spawned* world may draw the pair the other way round — a tie between two
    // co-planar blended surfaces is content's problem, not the sort's.)
    let mut world = World::new();
    let first = spawn(&mut world, 9, TRANSPARENT, -7.0);
    let second = spawn(&mut world, 1, ADDITIVE, -7.0);

    let mut items = build_draw_list(&mut world, 0.0);
    sort_draw_list_for_view(&mut items, &view_proj());
    assert_eq!(items.iter().map(|i| i.entity).collect::<Vec<_>>(), vec![first, second]);
    assert!(first.index_u32() < second.index_u32());

    // Re-sorting from any starting permutation lands in the same place, which
    // is what "total" buys: no pair is ever equal-but-unordered.
    items.reverse();
    sort_draw_list_for_view(&mut items, &view_proj());
    assert_eq!(items.iter().map(|i| i.entity).collect::<Vec<_>>(), vec![first, second]);
}

#[test]
fn a_degenerate_projection_still_orders_deterministically() {
    // The identity view-projection (`FrameParams::default`, the no-camera path)
    // makes every depth equal. That must degrade to "arbitrary but stable" —
    // the entity tie-break — and never to an unstable comparator.
    let mut world = World::new();
    let a = spawn(&mut world, 3, TRANSPARENT, -1.0);
    let b = spawn(&mut world, 1, ADDITIVE, -2.0);
    let c = spawn(&mut world, 2, TRANSPARENT, -3.0);

    let mut items = build_draw_list(&mut world, 0.0);
    sort_draw_list_for_view(&mut items, &Mat4::IDENTITY);
    assert_eq!(
        items.iter().map(|i| i.entity).collect::<Vec<_>>(),
        vec![a, b, c],
        "with no depth to sort by, the entity tie-break is the whole order"
    );
}

// ---------------------------------------------------------------------------
// 3. The pixels
// ---------------------------------------------------------------------------

const WIDTH: u32 = 320;
/// Deliberately not square: the phase circle is aspect-corrected, and on a
/// square target a missing correction looks exactly like a correct one.
const HEIGHT: u32 = 192;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// A unit quad in the XY plane facing +Z, wound counter-clockwise so backface
/// culling (on for every variant) keeps it.
fn quad() -> Mesh {
    Mesh {
        positions: vec![
            Vec3::new(-0.5, -0.5, 0.0),
            Vec3::new(0.5, -0.5, 0.0),
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::new(-0.5, 0.5, 0.0),
        ],
        normals: vec![Vec3::Z; 4],
        uvs: vec![glam::Vec2::ZERO; 4],
        colors: vec![Vec3::ONE; 4],
        indices: vec![0, 1, 2, 0, 2, 3],
    }
}

/// The four colours the frame is made of. Chosen so every blend below lands on
/// a value that can be written down exactly.
const FLOOR: Vec4 = Vec4::new(0.2, 0.2, 0.2, 1.0);
const GHOST: Vec4 = Vec4::new(0.9, 0.1, 0.1, 0.5);
const GLOW: Vec4 = Vec4::new(0.1, 0.6, 0.2, 0.5);
const PHASED: Vec4 = Vec4::new(0.2, 0.4, 0.9, 1.0);
const SOLID: Vec4 = Vec4::new(0.9, 0.8, 0.2, 1.0);

/// Where the two phase quads sit, in world units. The circle is aimed at the
/// first one's centre.
const PHASE_ONLY_AT: Vec3 = Vec3::new(1.5, 1.2, 0.0);
const WORLD_ONLY_AT: Vec3 = Vec3::new(1.5, -1.2, 0.0);
const PHASE_RADIUS: f32 = 0.25;

struct Frame {
    pixels: Vec<u8>,
    view_proj: Mat4,
}

impl Frame {
    fn project(&self, p: Vec3) -> Vec2 {
        let clip = self.view_proj * p.extend(1.0);
        assert!(clip.w > 0.0, "{p:?} is behind the camera");
        clip.truncate().truncate() / clip.w
    }

    /// Mean colour of a 5×5 block around a world point.
    fn sample(&self, p: Vec3) -> Vec3 {
        let ndc = self.project(p);
        let cx = ((ndc.x * 0.5 + 0.5) * WIDTH as f32).round() as i32;
        let cy = ((0.5 - ndc.y * 0.5) * HEIGHT as f32).round() as i32;
        let mut sum = Vec3::ZERO;
        for dy in -2i32..=2 {
            for dx in -2i32..=2 {
                let x = (cx + dx).clamp(0, WIDTH as i32 - 1) as usize;
                let y = (cy + dy).clamp(0, HEIGHT as i32 - 1) as usize;
                let i = (y * WIDTH as usize + x) * 4;
                sum += Vec3::new(
                    self.pixels[i] as f32,
                    self.pixels[i + 1] as f32,
                    self.pixels[i + 2] as f32,
                ) / 255.0;
            }
        }
        sum / 25.0
    }

    /// Whether a world point is inside the phase circle, by the shader's own
    /// arithmetic: aspect-corrected NDC, radius in NDC-Y units.
    fn inside_circle(&self, p: Vec3, center: Vec2) -> bool {
        let mut d = self.project(p) - center;
        d.x *= WIDTH as f32 / HEIGHT as f32;
        d.length() < PHASE_RADIUS
    }

    /// The pixel at `(x, y)`, `0..1`.
    fn pixel(&self, x: u32, y: u32) -> Vec3 {
        let i = ((y * WIDTH + x) * 4) as usize;
        Vec3::new(
            self.pixels[i] as f32,
            self.pixels[i + 1] as f32,
            self.pixels[i + 2] as f32,
        ) / 255.0
    }

    /// Aspect-corrected distance from the circle's centre to a pixel's centre,
    /// in NDC-Y units — the shader's own measure, so a distance compares
    /// directly against [`PHASE_RADIUS`] and [`PHASE_EDGE`].
    fn pixel_distance(x: u32, y: u32, center: Vec2) -> f32 {
        let mut d = Vec2::new(
            (x as f32 + 0.5) / WIDTH as f32 * 2.0 - 1.0,
            1.0 - (y as f32 + 0.5) / HEIGHT as f32 * 2.0,
        ) - center;
        d.x *= WIDTH as f32 / HEIGHT as f32;
        d.length()
    }
}

fn assert_rgb(got: Vec3, want: Vec3, what: &str) {
    let error = (got - want).abs().max_element();
    assert!(
        error <= 3.0 / 255.0,
        "{what}: expected {want:?}, got {got:?} (off by {error})"
    );
}

/// Build the scene, render it once, and read the pixels back.
fn render_blended_frame(strength: f32) -> Option<Frame> {
    let mut renderer = match pollster::block_on(Renderer::headless(FORMAT)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("SKIP transparency (no GPU adapter): {e}");
            return None;
        }
    };

    // One quad mesh, five entities. The handle is the content hash either side,
    // so registering with the renderer and naming it in the world agree without
    // a `MeshLibrary` in the middle.
    let mesh = quad();
    let handle = renderer.register_mesh(&mesh);
    assert_eq!(handle, MeshHandle::of(&mesh));

    let mut world = World::new();
    let mut place = |color: Vec4, variant: MaterialVariant, mode: f32, at: Vec3, scale: f32| {
        world.spawn((
            MeshRef(handle),
            Material {
                base_color: color,
                params: Vec4::new(mode, 0.0, 0.0, 0.0),
                texture: None,
                variant: variant | UNLIT,
            },
            Transform {
                translation: at,
                rotation: glam::Quat::IDENTITY,
                scale: Vec3::splat(scale),
            },
        ));
    };
    // Everything is BILLBOARD_UNLIT so each surface is exactly its base colour
    // and the blends below are arithmetic rather than an estimate of a light
    // rig. Spawn order is deliberately not draw order.
    place(GHOST, TRANSPARENT, 0.0, Vec3::new(-1.5, 1.2, 0.0), 2.0);
    place(FLOOR, MaterialVariant::NONE, 0.0, Vec3::new(0.0, 0.0, -3.0), 40.0);
    place(GLOW, ADDITIVE, 0.0, Vec3::new(-1.5, -1.2, 0.0), 2.0);
    place(PHASED, PHASE, Material::PHASE_ONLY, PHASE_ONLY_AT, 2.0);
    place(SOLID, PHASE, Material::PHASE_WORLD_ONLY, WORLD_ONLY_AT, 2.0);

    let draws = build_draw_list(&mut world, 0.0);
    assert_eq!(draws.len(), 5);

    let aspect = WIDTH as f32 / HEIGHT as f32;
    let view_proj = runt_core::Camera::default().projection(aspect)
        * glam::camera::rh::view::look_at_mat4(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
    let frame = runt_core::FrameParams {
        view_proj,
        ..Default::default()
    };

    // Aim the circle at the phase-only quad's centre.
    let clip = view_proj * PHASE_ONLY_AT.extend(1.0);
    let center = clip.truncate().truncate() / clip.w;
    renderer.set_phase_fx(center, PHASE_RADIUS, strength);

    let target = renderer.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("transparency target"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
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
        WIDTH,
        HEIGHT,
        &frame,
        &draws,
        &MeshLibrary::new(),
        &TextureLibrary::new(),
    );

    let pixels = read_back(&renderer, &target);
    Some(Frame { pixels, view_proj })
}

fn read_back(renderer: &Renderer, target: &wgpu::Texture) -> Vec<u8> {
    let device = renderer.device();
    let unpadded_row = WIDTH * 4;
    let padded_row = unpadded_row.div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (padded_row * HEIGHT) as u64,
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
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(HEIGHT),
            },
        },
        wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
    );
    renderer.queue().submit(Some(encoder.finish()));

    let (tx, rx) = std::sync::mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll device");
    rx.recv().expect("map callback").expect("buffer mapped");

    let padded = readback.get_mapped_range(..).expect("mapped range");
    let mut pixels = Vec::with_capacity((unpadded_row * HEIGHT) as usize);
    for row in 0..HEIGHT as usize {
        let start = row * padded_row as usize;
        pixels.extend_from_slice(&padded[start..start + unpadded_row as usize]);
    }
    drop(padded);
    readback.unmap();
    pixels
}

#[test]
fn one_of_each_new_state_over_an_opaque_floor() {
    // Strength 0: the fringe is a mix towards white, and these probes are
    // reading the blend arithmetic, not the decoration. The fringe has its own
    // test below.
    let Some(frame) = render_blended_frame(0.0) else {
        return;
    };

    let mut hash: u64 = 0xcbf29ce484222325;
    for b in &frame.pixels {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    println!("blended frame {WIDTH}x{HEIGHT}: fnv1a=0x{hash:016x}");

    let floor = FLOOR.truncate();
    // The floor is unlit and covers the frame, so this is the exact value every
    // blend below is measured against.
    assert_rgb(frame.sample(Vec3::new(0.0, 0.0, 0.0)), floor, "floor");

    // Alpha: src·α + dst·(1−α) = 0.5·(0.9,0.1,0.1) + 0.5·(0.2,0.2,0.2).
    assert_rgb(
        frame.sample(Vec3::new(-1.5, 1.2, 0.0)),
        GHOST.truncate() * GHOST.w + floor * (1.0 - GHOST.w),
        "transparent quad over the floor",
    );

    // Additive: src·α + dst = 0.5·(0.1,0.6,0.2) + (0.2,0.2,0.2). Brighter than
    // the floor in every channel, which alpha blending could not manage from a
    // colour this dark.
    let glow = frame.sample(Vec3::new(-1.5, -1.2, 0.0));
    assert_rgb(glow, GLOW.truncate() * GLOW.w + floor, "additive quad");
    assert!(glow.cmpgt(floor).all(), "additive must only ever add");

    // Phase-only (params.x = 1): visible inside the circle, gone outside it.
    //
    // Inside is also where the screen effect is at full strength, so the
    // expectation is the quad's colour *through* it. That the two agree is the
    // coincidence claim: the effect and the discard read the same `phase` out
    // of the same frame block, and a probe deep inside one circle would not sit
    // deep inside a differently placed other one.
    let center = frame.project(PHASE_ONLY_AT);
    assert!(frame.inside_circle(PHASE_ONLY_AT, center));
    assert_rgb(
        frame.sample(PHASE_ONLY_AT),
        phase_screen_color(PHASED.truncate(), 1.0),
        "phase-only quad inside the circle",
    );
    let corner = PHASE_ONLY_AT + Vec3::new(0.8, 0.8, 0.0);
    assert!(
        !frame.inside_circle(corner, center),
        "the probe must be outside the circle for the next assertion to mean anything"
    );
    assert_rgb(frame.sample(corner), floor, "phase-only quad outside the circle");

    // World-only (params.x = 0): the mirror image — solid everywhere the circle
    // is not, and this quad is entirely outside it.
    assert!(!frame.inside_circle(WORLD_ONLY_AT, center));
    assert_rgb(
        frame.sample(WORLD_ONLY_AT),
        SOLID.truncate(),
        "world-only quad outside the circle",
    );
}

/// The fringe is `strength` and nothing else — asked as "what did raising it
/// change, and where".
///
/// It used to be asked as "how red did the circle's column get", which worked
/// while the fringe was the last thing to touch those pixels. It is not any
/// more: `blit.wgsl`'s screen effect now sits between the fringe and the
/// framebuffer, and it *darkens* what the fringe whitened — a whitened edge
/// inverts to a dark one — so brightness on its own no longer reads the fringe
/// at all. What still does, and what the claim was always about, is the
/// difference between two frames that differ in nothing but `strength`.
#[test]
fn the_fringe_is_strength_and_nothing_else() {
    let Some(off) = render_blended_frame(0.0) else {
        return;
    };
    let on = render_blended_frame(1.0).expect("the adapter worked a moment ago");

    let center = off.project(PHASE_ONLY_AT);
    // A pixel either way, so a half-texel disagreement about where a pixel is
    // cannot indict the band it is in.
    let slack = 2.0 / HEIGHT as f32;
    let (mut worst, mut worst_at) = (0.0f32, (0u32, 0u32));
    let mut strayed = 0u32;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let delta = (on.pixel(x, y) - off.pixel(x, y)).abs().max_element();
            if delta <= 1.0 / 255.0 {
                continue;
            }
            if worst < delta {
                worst = delta;
                worst_at = (x, y);
            }
            // Everything `strength` touches is within the fringe band of the
            // circle's edge. This is the "nothing else" half, and it is a
            // *global* claim: one stray pixel anywhere in the frame — a fringe
            // leaking onto the floor, a boundary that moved — fails it.
            if (Frame::pixel_distance(x, y, center) - PHASE_RADIUS).abs() > PHASE_EDGE + slack {
                strayed += 1;
            }
        }
    }
    println!(
        "fringe: peak change {worst:.3} at {worst_at:?}, {strayed} pixels outside the edge band"
    );
    assert_eq!(
        strayed, 0,
        "strength changed pixels away from the circle's edge"
    );
    assert!(
        worst > 0.1,
        "strength 1 should draw a hard fringe, but the frame moved by only {worst}"
    );

    // The fringe is *decoration*: it may not move the boundary. Both frames
    // agree about what exists on either side of it — the phased quad, through
    // the screen effect, inside; the floor, untouched, outside.
    assert_rgb(
        on.sample(PHASE_ONLY_AT),
        phase_screen_color(PHASED.truncate(), 1.0),
        "the middle of the circle is untouched by the fringe",
    );
    assert_rgb(
        on.sample(PHASE_ONLY_AT + Vec3::new(0.8, 0.8, 0.0)),
        FLOOR.truncate(),
        "outside the circle is still discarded",
    );
}
