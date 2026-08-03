//! The game actually draws (DESIGN §2, §5).
//!
//! Everything else in this crate's tests runs on `Sim` — the engine minus the
//! GPU — which is the right way to test rules and determinism and a completely
//! useless way to notice that the level renders as a black screen. So this one
//! goes through the whole path: level RON → generators → cache → `MeshLibrary`
//! → lazy upload → the real `Renderer`, into an offscreen texture with no window
//! anywhere, and then looks at the pixels.
//!
//! It skips silently on a machine with no adapter, exactly as `runt-core`'s
//! `headless_screenshot` does — the point is to catch a broken level or a
//! material mistake in CI where a GPU exists, not to fail where one does not.
//!
//! Coverage is measured against the **sky** rather than against `CLEAR_COLOR`:
//! since the background gradient (DESIGN §5) paints every pixel, "differs from
//! the clear color" is true of a frame with nothing in it. `runt_core::sky` is
//! the CPU twin of the shader, so a pixel that differs from what the sky would
//! have painted there is a pixel some geometry landed on — see
//! `runt-core/tests/headless_screenshot.rs`, which makes the same move and
//! additionally pins the two copies of the gradient to each other.
//!
//! `RUNT_BALL_SHOT=/path/frame.rgba` dumps the raw `W × H × 4` pixels for
//! eyeballing (`magick -size 960x540 -depth 8 rgba:frame.rgba frame.png`).

use glam::{Mat4, Vec2, Vec3};
use runt_core::{Engine, InputEvent, InputTrace, Key, Lighting, SimConfig};

const W: u32 = 960;
const H: u32 = 540;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// Two and a half seconds in: the ball has left the spawn, the camera has
/// settled out of its opening ease, and pickups are mid-bob.
const TICKS: u64 = 150;

/// `copy_texture_to_buffer` wants rows aligned to 256 bytes.
fn align_256(n: u32) -> u32 {
    n.div_ceil(256) * 256
}

struct Frame {
    pixels: Vec<u8>,
    view_proj: Mat4,
    lighting: Lighting,
}

impl Frame {
    fn pixel(&self, x: u32, y: u32) -> [f32; 3] {
        let i = (y as usize * W as usize + x as usize) * 4;
        [
            self.pixels[i] as f32 / 255.0,
            self.pixels[i + 1] as f32 / 255.0,
            self.pixels[i + 2] as f32 / 255.0,
        ]
    }

    /// What the background gradient should be at a pixel.
    fn sky_at(&self, x: u32, y: u32) -> [f32; 3] {
        let ndc = Vec2::new(
            (x as f32 + 0.5) / W as f32 * 2.0 - 1.0,
            1.0 - (y as f32 + 0.5) / H as f32 * 2.0,
        );
        runt_core::sky::color_at(&self.lighting, self.view_proj.inverse(), ndc).to_array()
    }
}

impl Frame {
    /// Mean color of a 5×5 block around where `p` projects to, so one edge pixel
    /// cannot decide a test.
    fn sample(&self, p: Vec3) -> [f32; 3] {
        let clip = self.view_proj * p.extend(1.0);
        assert!(clip.w > 0.0, "{p:?} is behind the camera");
        let ndc = clip.truncate() / clip.w;
        let cx = ((ndc.x * 0.5 + 0.5) * W as f32).round() as i32;
        let cy = ((0.5 - ndc.y * 0.5) * H as f32).round() as i32;

        let mut sum = [0f32; 3];
        let mut n = 0f32;
        for dy in -2i32..=2 {
            for dx in -2i32..=2 {
                let x = (cx + dx).clamp(0, W as i32 - 1) as usize;
                let y = (cy + dy).clamp(0, H as i32 - 1) as usize;
                let i = (y * W as usize + x) * 4;
                for (channel, value) in sum.iter_mut().zip(&self.pixels[i..i + 3]) {
                    *channel += *value as f32 / 255.0;
                }
                n += 1.0;
            }
        }
        [sum[0] / n, sum[1] / n, sum[2] / n]
    }
}

/// Render one frame of the real game, headless.
fn render() -> Option<(Frame, Vec3, Vec3)> {
    let mut engine = match pollster::block_on(Engine::headless_with_config(
        FORMAT,
        SimConfig::default().with_scene(runt_ball::LEVEL1_RON),
    )) {
        Ok(engine) => engine,
        Err(e) => {
            eprintln!("SKIP render: {e}");
            return None;
        }
    };
    runt_ball::game::setup(engine.sim_mut());

    // Roll off the spawn so the frame is a game in progress rather than a still
    // life — and so the follow camera has had to do its job.
    engine
        .sim_mut()
        .play_input_trace(InputTrace::from_pairs([(0, InputEvent::KeyDown(Key::W))]));
    for _ in 0..TICKS {
        engine.sim_mut().tick();
    }

    let state = engine.sim().world().resource::<runt_ball::game::GameState>();
    let player = state.player;
    let ball = engine
        .sim()
        .world()
        .get::<runt_core::Transform>(player)
        .expect("Transform")
        .translation;
    let pickup = engine
        .sim()
        .scene_entity("pickup_11")
        .and_then(|e| engine.sim().world().get::<runt_core::Transform>(e))
        .expect("pickup_11 is still on the field this early")
        .translation;

    let device = engine.device().clone();
    let queue = engine.queue().clone();
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("game frame"),
        size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let params = engine
        .sim_mut()
        .frame_params(W as f32 / H as f32)
        .expect("the level spawns a camera");
    let (view_proj, lighting) = (params.view_proj, params.lighting);
    engine.render(&view, W, H);

    let padded_row = align_256(W * 4);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (padded_row * H) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback"),
    });
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    queue.submit(Some(encoder.finish()));

    let (tx, rx) = std::sync::mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll device");
    rx.recv().expect("map callback").expect("buffer mapped");

    let mapped = readback.get_mapped_range(..).expect("mapped range");
    let mut pixels = Vec::with_capacity((W * H * 4) as usize);
    for row in 0..H as usize {
        let start = row * padded_row as usize;
        pixels.extend_from_slice(&mapped[start..start + (W * 4) as usize]);
    }
    drop(mapped);
    readback.unmap();

    if let Ok(path) = std::env::var("RUNT_BALL_SHOT") {
        std::fs::write(&path, &pixels).expect("write dump");
        println!("wrote {W}x{H} RGBA8 to {path}");
    }

    Some((
        Frame {
            pixels,
            view_proj,
            lighting,
        },
        ball,
        pickup,
    ))
}

#[test]
fn the_level_renders_a_green_world_with_gold_rings_in_it() {
    let Some((frame, ball, pickup)) = render() else {
        return; // No adapter here.
    };

    // Something was drawn, and a lot of it: the terrain patch fills most of the
    // lower frame. A black screen — a broken generator, an empty draw list, a
    // camera facing the void — lands well under this, and so does a frame that
    // is nothing but sky, which is the case the old CLEAR_COLOR version of this
    // check could no longer tell apart.
    let tolerance = 3.0 / 255.0;
    let mut drawn = 0usize;
    for y in 0..H {
        for x in 0..W {
            let got = frame.pixel(x, y);
            let want = frame.sky_at(x, y);
            if (0..3).any(|c| (got[c] - want[c]).abs() > tolerance) {
                drawn += 1;
            }
        }
    }
    let coverage = drawn as f64 / (W * H) as f64;
    println!("game frame: {:.1}% geometry (rest is sky)", coverage * 100.0);
    assert!(
        coverage > 0.4,
        "the level covered only {:.1}% of the frame",
        coverage * 100.0
    );

    // And the sky really is the sky: the top-left corner of this camera's view
    // is above the terrain's horizon at every point of the run.
    let corner = frame.pixel(4, 4);
    let want = frame.sky_at(4, 4);
    assert!(
        (0..3).all(|c| (corner[c] - want[c]).abs() <= tolerance),
        "the background is {corner:?}, the gradient says {want:?}"
    );

    // The ball is red-ish where the sim says it is — which is a claim about the
    // follow camera, the interpolation and the instance offsets all at once.
    let ball_px = frame.sample(ball);
    println!("ball rgb {ball_px:?}");
    assert!(
        ball_px[0] > ball_px[1] + 0.15 && ball_px[0] > ball_px[2] + 0.15,
        "the player ball should read red, got {ball_px:?}"
    );

    // And a pickup is gold, not green: the collectibles have to pop against the
    // ground or the game is unreadable.
    //
    // Probed on the *tube*, `major_radius` out along +X, not at the centre —
    // a torus has a hole in the middle and the middle shows the ground through
    // it. The ring spins about Y and lies in the XZ plane, so this point is on
    // the tube whatever the rotation.
    let ring_px = frame.sample(pickup + Vec3::new(0.5, 0.0, 0.0));
    println!("pickup rgb {ring_px:?}");
    assert!(
        ring_px[0] > 0.35 && ring_px[1] > 0.25 && ring_px[0] > ring_px[2] + 0.2,
        "a pickup should read gold, got {ring_px:?}"
    );
    assert!(
        ring_px[0] > ball_px[2],
        "the gold ring and the red ball must not be the same material"
    );
}
