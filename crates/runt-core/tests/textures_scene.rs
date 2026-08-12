//! Textures end to end: RON → `TextureLibrary` → bake → a drawn frame
//! (DESIGN §7, §5, §6).
//!
//! `texture_bake.rs` proves the pass is right. This proves the *pipeline* is
//! wired: that a scene file can name a texture, that a material can point at
//! one, that the variant bits land where they should, and — the part nothing
//! else covers — that an entity with no texture still renders exactly as it did
//! before any of this existed.
//!
//! The frame check is deliberately a *material* claim rather than a golden
//! image: the ground must read as grass green and the boulders as rock grey,
//! both distinguishable from the sky the CPU model says is behind them. A
//! fingerprint would break on any driver, and a driver change is not a
//! regression in this pipeline.

use glam::{Mat4, Vec2, Vec3};
use runt_core::scene::{self, MaterialDesc, SceneDesc};
use runt_core::texture::{self, TextureLibrary, TextureSpec};
use runt_core::{Engine, Lighting, MaterialVariant, Material, SimConfig};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const SIZE: u32 = 384;

// ---------------------------------------------------------------------------
// The scene file
// ---------------------------------------------------------------------------

#[test]
fn the_textured_scene_says_what_its_own_comments_say() {
    let desc = scene::textured_scene();
    assert_eq!(desc.textures.len(), 2, "the demo scene's two surfaces");

    let by_name = |name: &str| {
        desc.textures
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("assets/textured.ron defines no texture {name:?}"))
            .spec
            .clone()
    };

    // This used to compare the file against `texture::grass()` and
    // `texture::rock()` — two `pub fn`s the engine shipped, on the argument that
    // a scene file naming them made them reference data. They are gone: the
    // engine defines texture *types* and a game defines their uniforms, so the
    // only copy of these numbers on this side of the line is the file itself,
    // and what is worth asserting is that the file agrees with the prose written
    // beside it. A .ron whose comment says `frequency 0.21` and whose value says
    // otherwise is the failure this catches, and it is the one that used to hide
    // behind the two being compared to each other.
    let grass = by_name("grass");
    assert_eq!(grass.frequency, 0.21);
    assert_eq!(grass.octaves, 5);
    assert_eq!(grass.gain, 0.562);
    assert_eq!(grass.world_scale, 0.036);
    assert_eq!(grass.base_resolution, 1024);
    assert_eq!(grass.ramp.len(), 3);
    let gn = grass.normal.expect("the floor is crinkled");
    assert_eq!(gn.edge_width, 0.52);
    assert_eq!(gn.strength, 5.106);

    let rock = by_name("rock");
    assert_eq!(rock.frequency, 0.046);
    assert_eq!(rock.octaves, 5);
    assert_eq!(rock.gain, 0.543);
    assert_eq!(rock.world_scale, 0.025);
    assert_eq!(rock.triplanar_sharpness, 1.0);
    let rn = rock.normal.expect("stone is faceted");
    assert_eq!(rn.edge_width, 0.351);
    assert_eq!(rn.strength, 29.605);

    // …and the two are different surfaces, which is the only thing about them
    // the *engine* has an opinion on: one bake each.
    assert_ne!(grass.param_key(), rock.param_key());
}

#[test]
fn the_scene_round_trips_through_ron_with_its_textures() {
    let desc = scene::textured_scene();
    let ron = ron::ser::to_string_pretty(&desc, ron::ser::PrettyConfig::new().struct_names(false))
        .expect("serialize");
    let back: SceneDesc = ron::from_str(&ron).expect("re-parse what we just wrote");
    assert_eq!(back, desc, "textures did not survive a save/load round trip");

    // And the identity survives with it — a round trip that changed a param key
    // would silently orphan every cached bake.
    for (a, b) in desc.textures.iter().zip(&back.textures) {
        assert_eq!(a.name, b.name);
        assert_eq!(a.spec.param_key(), b.spec.param_key());
    }
}

#[test]
fn a_material_without_a_texture_keeps_the_variant_it_always_had() {
    // The compatibility claim: `MaterialDesc` grew two fields, and neither may
    // move the variant key of a material that does not use them. Anything else
    // re-keys every pipeline cache and every scene file in existence.
    let plain = MaterialDesc::default();
    assert_eq!(plain.texture, None);
    let material = plain.to_material();
    assert_eq!(material.variant, MaterialVariant::VERTEX_COLOR);
    assert_eq!(material.texture, None);
    assert_eq!(material, Material::vertex_colored());

    // `normal_map` defaults to true, and must still be inert without a texture.
    assert!(plain.normal_map);
    assert!(!material.variant.contains(MaterialVariant::NORMAL_MAP));

    // Every entity in the *demo* scene — which this work must not have touched
    // — still resolves to a pre-texture variant.
    for entity in scene::demo_scene().entities {
        let m = entity.material.to_material();
        assert!(
            m.texture.is_none() && !m.variant.contains(MaterialVariant::TEXTURE),
            "demo.ron entity {:?} acquired a texture",
            entity.name
        );
    }
}

#[test]
fn naming_a_texture_sets_the_bits_and_carrying_the_handle() {
    let grass = scene::textured_scene()
        .textures
        .into_iter()
        .find(|t| t.name == "grass")
        .expect("the demo scene's floor")
        .spec;
    let handle = runt_core::TextureHandle(grass.content_key(512));

    let with_normals = MaterialDesc {
        texture: Some("grass".into()),
        vertex_color: false,
        ..MaterialDesc::default()
    }
    .to_material_with(Some(handle));
    assert_eq!(
        with_normals.variant,
        MaterialVariant::TEXTURE | MaterialVariant::NORMAL_MAP
    );
    assert_eq!(with_normals.texture, Some(handle));

    let flat = MaterialDesc {
        texture: Some("grass".into()),
        vertex_color: false,
        normal_map: false,
        ..MaterialDesc::default()
    }
    .to_material_with(Some(handle));
    assert_eq!(flat.variant, MaterialVariant::TEXTURE);
    assert!(!flat.variant.contains(MaterialVariant::NORMAL_MAP));
}

#[test]
fn an_entity_naming_a_missing_texture_is_an_error_not_a_half_scene() {
    let mut world = bevy_ecs::prelude::World::new();
    let src = r#"(
        generators: [ ( name: "c", spec: Cube(size: 1.0) ) ],
        entities: [ ( generator: "c", material: ( texture: Some("nope") ) ) ],
    )"#;
    match scene::load_scene(&mut world, src) {
        Err(runt_core::SceneError::UnknownTexture { entity, name }) => {
            assert_eq!(entity, 0);
            assert_eq!(name, "nope");
        }
        other => panic!("expected UnknownTexture, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Loading into a world
// ---------------------------------------------------------------------------

fn textured_sim() -> runt_core::Sim {
    runt_core::Sim::from_config(SimConfig::default().with_scene(runt_core::TEXTURED_SCENE_RON))
}

#[test]
fn loading_registers_one_bake_per_texture_and_shares_it() {
    let sim = textured_sim();
    let library: &TextureLibrary = sim.texture_library();
    assert_eq!(library.len(), 2, "grass and rock, once each");

    // Three entities wear rock (two boulders and the plinth); all three must
    // carry the *same* handle, or the sharing claim is decoration.
    let mut world_handles = Vec::new();
    for name in ["ground", "boulder", "boulder_twin", "plinth", "marker"] {
        let entity = sim.scene_entity(name).expect("scene entity");
        let material = sim.world().get::<Material>(entity).expect("material");
        world_handles.push((name, material.texture, material.variant));
    }

    let rock: Vec<_> = world_handles
        .iter()
        .filter(|(n, _, _)| n.starts_with("boulder") || *n == "plinth")
        .map(|(_, h, _)| *h)
        .collect();
    assert!(rock.iter().all(|h| h.is_some() && *h == rock[0]));

    let ground = world_handles[0];
    assert!(ground.1.is_some());
    assert_ne!(ground.1, rock[0], "grass and rock are two textures");
    assert!(ground.2.contains(MaterialVariant::TEXTURE));
    assert!(ground.2.contains(MaterialVariant::NORMAL_MAP));

    // The plinth opted out of normals; the boulders did not.
    let plinth = world_handles.iter().find(|(n, _, _)| *n == "plinth").unwrap();
    assert!(plinth.2.contains(MaterialVariant::TEXTURE));
    assert!(!plinth.2.contains(MaterialVariant::NORMAL_MAP));

    // The control entity has neither bit and no handle.
    let marker = world_handles.iter().find(|(n, _, _)| *n == "marker").unwrap();
    assert_eq!(marker.1, None);
    assert!(!marker.2.contains(MaterialVariant::TEXTURE));
}

#[test]
fn the_quality_tier_scales_the_bake_but_not_the_identity() {
    // DESIGN §11: a gate picks data. Two tiers must give two *resolutions* of
    // one texture, not two textures.
    let low = runt_core::Sim::from_config(
        SimConfig::default()
            .with_scene(runt_core::TEXTURED_SCENE_RON)
            .with_quality(0.25),
    );
    let high = textured_sim();

    let spec_of = |sim: &runt_core::Sim, i: usize| {
        let (_, spec, res) = sim.texture_library().iter().nth(i).expect("entry");
        (spec.clone(), res)
    };

    for i in 0..2 {
        let (lo_spec, lo_res) = spec_of(&low, i);
        let (hi_spec, hi_res) = spec_of(&high, i);
        assert_eq!(
            lo_spec.param_key(),
            hi_spec.param_key(),
            "the tier changed the texture's identity"
        );
        assert!(lo_res < hi_res, "the tier did not change the resolution");
        assert!(hi_res <= texture::MAX_RESOLUTION, "DESIGN §11 caps at 2048");
        assert!(lo_res >= texture::MIN_RESOLUTION);
        assert_ne!(
            lo_spec.content_key(lo_res),
            hi_spec.content_key(hi_res),
            "two resolutions must be two cache entries"
        );
    }
}

// ---------------------------------------------------------------------------
// A drawn frame
// ---------------------------------------------------------------------------

struct Frame {
    pixels: Vec<u8>,
    view_proj: Mat4,
    lighting: Lighting,
}

impl Frame {
    fn pixel(&self, x: u32, y: u32) -> [f32; 3] {
        let i = (y as usize * SIZE as usize + x as usize) * 4;
        [
            self.pixels[i] as f32 / 255.0,
            self.pixels[i + 1] as f32 / 255.0,
            self.pixels[i + 2] as f32 / 255.0,
        ]
    }

    fn sky_at(&self, x: u32, y: u32) -> [f32; 3] {
        let ndc = Vec2::new(
            (x as f32 + 0.5) / SIZE as f32 * 2.0 - 1.0,
            1.0 - (y as f32 + 0.5) / SIZE as f32 * 2.0,
        );
        runt_core::sky::color_at(&self.lighting, self.view_proj.inverse(), ndc).to_array()
    }

    /// Mean colour of a 7×7 block around where `p` projects, so one edge pixel
    /// cannot decide anything.
    fn sample(&self, p: Vec3) -> [f32; 3] {
        let clip = self.view_proj * p.extend(1.0);
        assert!(clip.w > 0.0, "{p:?} is behind the camera");
        let ndc = clip.truncate() / clip.w;
        let cx = ((ndc.x * 0.5 + 0.5) * SIZE as f32).round() as i32;
        let cy = ((0.5 - ndc.y * 0.5) * SIZE as f32).round() as i32;
        let mut sum = [0f32; 3];
        let mut n = 0f32;
        for dy in -3i32..=3 {
            for dx in -3i32..=3 {
                let x = (cx + dx).clamp(0, SIZE as i32 - 1) as u32;
                let y = (cy + dy).clamp(0, SIZE as i32 - 1) as u32;
                let px = self.pixel(x, y);
                for c in 0..3 {
                    sum[c] += px[c];
                }
                n += 1.0;
            }
        }
        [sum[0] / n, sum[1] / n, sum[2] / n]
    }
}

fn render_textured() -> Option<Frame> {
    let mut engine = match pollster::block_on(Engine::headless_with_config(
        FORMAT,
        SimConfig::default().with_scene(runt_core::TEXTURED_SCENE_RON),
    )) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("SKIP (no GPU adapter): {e}");
            return None;
        }
    };

    // Construction already baked the scene's textures (that is what DESIGN §7's
    // "at load time" means), so the first frame is a plain draw.
    assert_eq!(
        engine.renderer().textures().len(),
        2,
        "the scene's textures should be resident before the first frame"
    );

    let device = engine.device().clone();
    let queue = engine.queue().clone();
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("textured target"),
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

    engine.update(0.0);
    let frame = engine
        .sim_mut()
        .frame_params(1.0)
        .expect("the scene spawns a camera");
    let (view_proj, lighting) = (frame.view_proj, frame.lighting);
    engine.render(&view, SIZE, SIZE);

    let unpadded = SIZE * 4;
    let padded = unpadded.div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (padded * SIZE) as u64,
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

    let mapped = readback.get_mapped_range(..).expect("range");
    let mut pixels = Vec::with_capacity((unpadded * SIZE) as usize);
    for row in 0..SIZE as usize {
        let start = row * padded as usize;
        pixels.extend_from_slice(&mapped[start..start + unpadded as usize]);
    }
    drop(mapped);
    readback.unmap();

    if let Ok(path) = std::env::var("RUNT_TEXTURED_DUMP") {
        std::fs::write(&path, &pixels).expect("write dump");
        println!("wrote {SIZE}x{SIZE} RGBA8 to {path}");
    }

    Some(Frame {
        pixels,
        view_proj,
        lighting,
    })
}

#[test]
fn the_textured_scene_renders_grass_and_rock() {
    let Some(frame) = render_textured() else {
        return;
    };

    // The ground, well in front of the props. The grass ramp is pure green
    // (red is 0 at every stop), so "green dominates and red is nearly absent"
    // is a claim about the *ramp*, not just about something being drawn.
    let ground = frame.sample(Vec3::new(-0.4, -1.35, 2.6));
    println!("ground probe rgb {ground:?}");
    assert!(
        ground[1] > ground[0] + 0.06 && ground[1] > ground[2] + 0.03,
        "the ground should read as the grass ramp's green, got {ground:?}"
    );

    // The boulder. The rock ramp is near-neutral with a slight red lean, so it
    // must read as grey-brown — and, crucially, not green.
    let boulder = frame.sample(Vec3::new(-2.2, 0.1, 0.4));
    println!("boulder probe rgb {boulder:?}");
    assert!(
        boulder[0] >= boulder[1] && boulder[0] > 0.05,
        "the boulder should read as the rock ramp's grey-brown, got {boulder:?}"
    );
    assert!(
        boulder[1] < ground[1],
        "the boulder ({}) is greener than the ground ({}), so the two textures \
         are not distinguishable",
        boulder[1],
        ground[1]
    );

    // The untextured control is still its own flat yellow — the bit-unset path
    // is untouched by everything above.
    let marker = frame.sample(Vec3::new(0.0, 0.1, 1.6));
    println!("marker probe rgb {marker:?}");
    assert!(
        marker[0] > marker[2] + 0.15 && marker[1] > marker[2] + 0.1,
        "the untextured cone should still be yellow, got {marker:?}"
    );

    // And geometry actually covers the frame, measured against the sky model
    // the same way `headless_screenshot.rs` does it.
    let mut drawn = 0usize;
    for y in 0..SIZE {
        for x in 0..SIZE {
            let got = frame.pixel(x, y);
            let want = frame.sky_at(x, y);
            if (0..3).any(|c| (got[c] - want[c]).abs() > 3.0 / 255.0) {
                drawn += 1;
            }
        }
    }
    let frac = drawn as f64 / (SIZE * SIZE) as f64;
    println!("textured frame: {:.1}% non-sky", frac * 100.0);
    assert!(frac >= 0.20, "only {:.1}% of the frame is geometry", frac * 100.0);
    assert!(frac <= 0.98, "the sky model and the frame disagree everywhere");
}

#[test]
fn the_texture_varies_across_the_surface() {
    // A triplanar sample of a real texture is not a flat colour. This is what
    // separates "the pipeline ran" from "the pipeline bound a 1×1 default and
    // nobody noticed".
    let Some(frame) = render_textured() else {
        return;
    };
    let probes: Vec<[f32; 3]> = [
        Vec3::new(-3.0, -1.5, 3.0),
        Vec3::new(-1.0, -1.4, 3.4),
        Vec3::new(1.2, -1.4, 3.2),
        Vec3::new(3.0, -1.6, 2.8),
        Vec3::new(-2.0, -1.3, 4.2),
        Vec3::new(2.0, -1.5, 4.0),
    ]
    .into_iter()
    .map(|p| frame.sample(p))
    .collect();

    let green: Vec<f32> = probes.iter().map(|p| p[1]).collect();
    let lo = green.iter().cloned().fold(f32::MAX, f32::min);
    let hi = green.iter().cloned().fold(f32::MIN, f32::max);
    println!("ground green across the patch: {green:?}");
    assert!(
        hi - lo > 0.01,
        "the ground is a single flat colour ({lo}..{hi}); the default 1×1 \
         texture is bound, or the triplanar sample is not reaching the world"
    );

    // Variation alone could be lighting. The grass ramp has red = 0 at *every*
    // stop, so a red channel that stays at the floor everywhere is the ramp
    // being sampled — the white 1×1 default would let the key light through as
    // grey and put red level with green.
    for p in &probes {
        assert!(
            p[0] < 0.05 && p[1] > p[0] + 0.15,
            "probe {p:?} is not wearing the grass ramp"
        );
    }
}

#[test]
fn a_second_render_changes_nothing_about_residency() {
    // The bake is load-time work (DESIGN §7): a frame must not be able to
    // trigger one, and a repeated frame must not re-bake.
    let Some(mut engine) = pollster::block_on(Engine::headless_with_config(
        FORMAT,
        SimConfig::default().with_scene(runt_core::TEXTURED_SCENE_RON),
    ))
    .ok() else {
        eprintln!("SKIP (no GPU adapter)");
        return;
    };
    let before = engine.renderer().textures().len();
    assert_eq!(before, 2);

    let device = engine.device().clone();
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    for _ in 0..3 {
        engine.render(&view, 64, 64);
    }
    assert_eq!(engine.renderer().textures().len(), before);
}

#[test]
fn a_sweep_drops_the_bake_a_spec_edit_superseded_and_nothing_else() {
    // The live-authoring lifecycle at the GPU. Editing a spec cannot mutate a
    // bake in place — the handle *is* the params — so an edit leaves the old
    // texture pair resident with nothing pointing at it, and a slider drag is
    // dozens of those. `Engine::sweep_baked_textures` is the reconcile that
    // stops it being a leak, and this pins the two claims that make it safe to
    // call: it drops exactly what the library stopped listing, and it cannot
    // touch a texture it could not re-bake.
    let Some(mut engine) = pollster::block_on(Engine::headless_with_config(
        FORMAT,
        SimConfig::default().with_scene(runt_core::TEXTURED_SCENE_RON),
    ))
    .ok() else {
        eprintln!("SKIP (no GPU adapter)");
        return;
    };
    assert_eq!(engine.renderer().textures().len(), 2, "grass and rock, baked");

    // Nothing has been superseded, so a sweep is a no-op — the registry and the
    // library already agree.
    assert_eq!(engine.sweep_baked_textures(), 0);
    assert_eq!(engine.renderer().textures().len(), 2);

    // An atlas and a render target, neither of which the library knows about.
    // The atlas is the load-bearing one: its handle is content-shaped, so only
    // provenance separates it from a bake.
    let atlas = runt_core::TextureHandle(0x5151_5151);
    let (device, queue) = (engine.device().clone(), engine.queue().clone());
    engine
        .renderer_mut()
        .textures_mut()
        .insert_image(&device, &queue, atlas, 1, 1, &[9, 9, 9, 255]);
    let target_handle = runt_core::TextureHandle::render_target(7);
    let color = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 8,
            height: 8,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    engine
        .renderer_mut()
        .textures_mut()
        .insert_render_target(&device, &queue, target_handle, color);
    assert_eq!(engine.renderer().textures().len(), 4);

    // Now the edit: one spec gains a contrast nudge, so it is a new handle, and
    // the entry it replaced comes out of the library.
    let (old_handle, edited, resolution) = {
        let library = engine.sim().texture_library();
        let (handle, spec, resolution) = library.iter().next().expect("two entries");
        (
            handle,
            TextureSpec {
                contrast: spec.contrast + 0.25,
                ..spec.clone()
            },
            resolution,
        )
    };
    let new_handle = {
        let mut library = engine
            .sim_mut()
            .world_mut()
            .resource_mut::<TextureLibrary>();
        let new_handle = library.insert(edited, resolution);
        assert!(library.remove(old_handle));
        new_handle
    };
    assert_ne!(new_handle, old_handle);

    // The superseded bake is still resident (nothing has swept yet) and the new
    // one is not (nothing has drawn it yet) — which is exactly the state the
    // sweep exists for.
    assert!(engine.renderer().textures().contains(old_handle));
    assert!(!engine.renderer().textures().contains(new_handle));

    assert_eq!(engine.sweep_baked_textures(), 1, "one superseded bake");
    assert!(!engine.renderer().textures().contains(old_handle));
    assert!(
        engine.renderer().textures().contains(atlas),
        "the sweep ate an atlas it cannot rebuild"
    );
    assert!(
        engine.renderer().textures().contains(target_handle),
        "the sweep ate a live render target"
    );
    assert_eq!(engine.renderer().textures().len(), 3);

    // And it is idempotent: the two records agree again.
    assert_eq!(engine.sweep_baked_textures(), 0);
}

#[test]
fn a_swept_texture_rebakes_to_the_same_pixels() {
    // Why being wrong about a sweep costs a fragment pass and never a frame: a
    // dropped bake is a content address that is still true, so resolving it
    // again is byte-identical rather than merely similar.
    let Some(mut engine) = pollster::block_on(Engine::headless_with_config(
        FORMAT,
        SimConfig::default().with_scene(runt_core::TEXTURED_SCENE_RON),
    ))
    .ok() else {
        eprintln!("SKIP (no GPU adapter)");
        return;
    };
    let (handle, spec, resolution) = {
        let library = engine.sim().texture_library();
        let (handle, spec, resolution) = library.iter().next().expect("a texture");
        (handle, spec.clone(), resolution)
    };
    let read = |engine: &Engine| {
        let gpu = engine.renderer().textures().get(handle).expect("resident");
        runt_core::bake::read_target(engine.device(), engine.queue(), &gpu.albedo, resolution)
            .expect("read the albedo back")
    };
    let first = read(&engine);

    // Evict it behind the library's back — the library still lists it, so this
    // is the "swept something that was still wanted" case rather than a
    // reconcile.
    assert!(engine.renderer_mut().textures_mut().remove(handle));
    assert!(!engine.renderer().textures().contains(handle));

    let again = engine
        .renderer_mut()
        .bake_texture(&spec, resolution, &runt_core::NoopCache);
    assert_eq!(again, handle, "the content key did not round-trip");
    assert_eq!(read(&engine), first, "the re-bake is not the same pixels");
}

#[test]
fn the_untextured_demo_scene_still_binds_nothing() {
    // The other half of the compatibility claim, at the GPU: loading the
    // *demo* scene must leave the texture registry empty, so a scene that does
    // not use §7 pays nothing for it existing.
    let Some(engine) = pollster::block_on(Engine::headless(FORMAT)).ok() else {
        eprintln!("SKIP (no GPU adapter)");
        return;
    };
    assert!(engine.sim().texture_library().is_empty());
    assert!(engine.renderer().textures().is_empty());
}

#[test]
fn a_spec_out_of_a_scene_file_bakes_to_the_same_pixels_as_one_built_in_code() {
    // A spec that came through RON and the identical spec built in Rust must be
    // the same *bake* — which catches the failure a structural `assert_eq!`
    // cannot see: a field that serializes lossily, so that two specs comparing
    // equal as values disagree as pixels.
    let a = scene::textured_scene()
        .textures
        .into_iter()
        .find(|t| t.name == "grass")
        .expect("grass")
        .spec;
    // Round-trip it in code rather than comparing to a hand-written twin: the
    // engine no longer ships a spec to compare against, and the claim is about
    // the *encoding*, so re-encoding is the sharper test of it.
    let b: TextureSpec = ron::from_str(&ron::ser::to_string(&a).expect("encode")).expect("decode");
    assert_eq!(a, b, "a spec did not survive its own encoding");
    assert_eq!(a.content_key(512), b.content_key(512));
    assert_eq!(a.octave_plan(), b.octave_plan());
    assert_eq!(a.albedo_at(Vec2::new(0.31, 0.62)), b.albedo_at(Vec2::new(0.31, 0.62)));
}
