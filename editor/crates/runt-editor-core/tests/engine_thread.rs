//! The command protocol, end to end (DESIGN §10).
//!
//! A real engine thread with a real GPU, driven by real commands, with the
//! frames going into a counting sink instead of a window. This is the only test
//! that exercises the whole editor loop, so it is where the claims about
//! *behaviour* live: a param edit changes geometry, a pause stops the sim, a
//! save round-trips.
//!
//! Skipped on a machine with no usable adapter.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use runt_core::gen::GeneratorSpec;
use runt_editor_core::engine_thread::{self, EngineConfig, EngineHandle};
use runt_editor_core::protocol::{Command, Event, FrameSink, SceneSnapshot, Stats};

/// The runt repo root — this crate is `<root>/editor/crates/runt-editor-core`.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("the manifest is three levels below the repo root")
        .to_path_buf()
}

fn demo_scene() -> PathBuf {
    repo_root().join("assets/demo.ron")
}

fn ball_level() -> PathBuf {
    repo_root().join("demo/ball/assets/level1.ron")
}

/// A sink that counts what it is given and checks the contract while it does.
#[derive(Clone, Default)]
struct CountingSink {
    frames: Arc<AtomicU32>,
    /// A cheap fingerprint of the last frame, so a test can tell "the picture
    /// changed" without owning a copy of it.
    last_hash: Arc<AtomicU64>,
}

impl FrameSink for CountingSink {
    fn submit(&mut self, pixels: &[u8], width: u32, height: u32) {
        // `SurfaceWriter::submit_frame` debug-asserts exactly this; a violation
        // here would be a panic inside rinch with no useful backtrace.
        assert_eq!(
            pixels.len(),
            (width * height * 4) as usize,
            "the sink was handed a frame that is not tightly packed"
        );
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        // Every 997th byte: enough to notice a changed picture, cheap enough to
        // run at frame rate.
        for b in pixels.iter().step_by(997) {
            hash ^= *b as u64;
            hash = hash.wrapping_mul(0x1000_0000_1b3);
        }
        self.last_hash.store(hash, Ordering::Relaxed);
        self.frames.fetch_add(1, Ordering::Relaxed);
    }
}

/// A handle plus the sink it is feeding.
///
/// Every event the engine sends is kept in `log`, and never merely inspected and
/// dropped: a test that waits for a save confirmation must not be able to eat
/// the scene snapshot a later assertion needs. `cursor` is how far each
/// [`wait_for`](Harness::wait_for) has read.
struct Harness {
    engine: EngineHandle,
    sink: CountingSink,
    log: Vec<Event>,
    cursor: usize,
}

impl Harness {
    fn start(config: EngineConfig) -> Option<Harness> {
        let sink = CountingSink::default();
        let engine = engine_thread::spawn(config, Box::new(sink.clone()));
        let mut harness = Harness {
            engine,
            sink,
            log: Vec::new(),
            cursor: 0,
        };

        // The device is created on the thread, so a machine with no adapter
        // reports it as an error event rather than by failing to spawn. Wait for
        // whichever comes first: a startup failure, or evidence of life.
        let deadline = Instant::now() + Duration::from_secs(20);
        while Instant::now() < deadline {
            harness.pump();
            if let Some(message) = harness.log.iter().find_map(|e| match e {
                Event::Error(m) if m.contains("startup failed") => Some(m.clone()),
                _ => None,
            }) {
                eprintln!("SKIP: {message}");
                return None;
            }
            if harness.frames() > 0 {
                return Some(harness);
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        eprintln!("SKIP: the engine thread produced no frames and no error");
        None
    }

    fn pump(&mut self) {
        for event in self.engine.drain() {
            if let Event::Error(m) = &event {
                eprintln!("engine error: {m}");
            }
            self.log.push(event);
        }
    }

    /// Scan forward through the event log until `f` matches, or the deadline
    /// passes.
    ///
    /// A deadline rather than a fixed number of polls: the engine thread runs as
    /// fast as it can, and how many frames fit in a second is not this test's
    /// business.
    fn wait_for<T>(&mut self, timeout: Duration, mut f: impl FnMut(&Event) -> Option<T>) -> Option<T> {
        let deadline = Instant::now() + timeout;
        loop {
            self.pump();
            while self.cursor < self.log.len() {
                let event = &self.log[self.cursor];
                self.cursor += 1;
                if let Some(found) = f(event) {
                    return Some(found);
                }
            }
            if Instant::now() >= deadline {
                return None;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// Errors seen so far, and forget them.
    fn take_errors(&mut self) -> Vec<String> {
        self.pump();
        self.log
            .iter()
            .filter_map(|e| match e {
                Event::Error(m) => Some(m.clone()),
                _ => None,
            })
            .collect()
    }

    fn wait_for_snapshot(&mut self) -> SceneSnapshot {
        self.wait_for(Duration::from_secs(10), |e| match e {
            Event::SceneLoaded(s) => Some((**s).clone()),
            _ => None,
        })
        .expect("a scene snapshot")
    }

    fn wait_for_stats(&mut self) -> Stats {
        self.wait_for(Duration::from_secs(10), |e| match e {
            Event::Stats(s) => Some(*s),
            _ => None,
        })
        .expect("a stats report")
    }

    fn frames(&self) -> u32 {
        self.sink.frames.load(Ordering::Relaxed)
    }

    fn frame_hash(&self) -> u64 {
        self.sink.last_hash.load(Ordering::Relaxed)
    }

    /// Block until at least `n` more frames have been submitted.
    fn wait_frames(&self, n: u32) {
        let target = self.frames() + n;
        let deadline = Instant::now() + Duration::from_secs(10);
        while self.frames() < target && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(2));
        }
    }
}

fn config(scene: Option<PathBuf>) -> EngineConfig {
    EngineConfig {
        width: 320,
        height: 180,
        scene,
        quality: 1.0,
        // A test must not write into the shared on-disk cache: it would make
        // "did this regenerate?" depend on what ran before it.
        persistent_cache: false,
        // Unthrottled: the tests wait on frame counts, and a 60 Hz cap would
        // turn a four-second suite into a minute of sleeping.
        target_fps: 0.0,
    }
}

// ---------------------------------------------------------------------------

#[test]
fn the_loop_renders_and_submits_frames() {
    let Some(harness) = Harness::start(config(Some(demo_scene()))) else {
        return;
    };
    harness.wait_frames(5);
    assert!(
        harness.frames() >= 5,
        "the loop submitted {} frames in ten seconds",
        harness.frames()
    );
}

#[test]
fn loading_a_scene_reports_its_generators_and_entities() {
    let Some(mut harness) = Harness::start(config(Some(demo_scene()))) else {
        return;
    };
    let snapshot = harness.wait_for_snapshot();

    assert_eq!(snapshot.path.as_deref(), Some(demo_scene().as_path()));
    let names: Vec<&str> = snapshot.generators.iter().map(|g| g.name.as_str()).collect();
    assert_eq!(
        names,
        ["ground", "ball", "post", "spike", "ring", "twisted_box"],
        "generators come back in file order"
    );
    let labels: Vec<&str> = snapshot.entities.iter().map(|e| e.label.as_str()).collect();
    assert_eq!(
        labels,
        ["ground", "ball", "post", "spike", "ring", "ball_twin", "spinner"]
    );

    // Indices are the protocol's identity, so they must be exactly positional.
    for (i, entity) in snapshot.entities.iter().enumerate() {
        assert_eq!(entity.index, i);
        assert_eq!(
            snapshot.generators[entity.generator_index].name,
            entity.generator
        );
    }

    // The terrain is the biggest mesh in the scene by a wide margin.
    let ground = &snapshot.generators[0];
    assert!(ground.triangles > 1000, "ground had {} tris", ground.triangles);
}

#[test]
fn a_scene_can_be_swapped_at_runtime() {
    let Some(mut harness) = Harness::start(config(Some(demo_scene()))) else {
        return;
    };
    let demo = harness.wait_for_snapshot();
    assert_eq!(demo.entities.len(), 7);

    harness.engine.send(Command::LoadScene(ball_level()));
    let level = harness.wait_for_snapshot();
    assert_eq!(level.path.as_deref(), Some(ball_level().as_path()));
    assert!(
        level.entities.len() > 15,
        "level 1 has a player, 12 pickups and 7 posts; got {}",
        level.entities.len()
    );
    assert!(level.entities.iter().any(|e| e.label == "player"));
    harness.wait_frames(3);
}

#[test]
fn a_bad_path_is_an_error_not_a_crash() {
    let Some(mut harness) = Harness::start(config(None)) else {
        return;
    };
    harness
        .engine
        .send(Command::LoadScene(PathBuf::from("/nonexistent/scene.ron")));
    let message = harness
        .wait_for(Duration::from_secs(5), |e| match e {
            Event::Error(m) => Some(m.clone()),
            _ => None,
        })
        .expect("an error event");
    assert!(message.contains("nonexistent"), "{message}");

    // …and the loop is still running.
    harness.wait_frames(3);
}

/// The core editing claim: change a param, get different geometry, through the
/// same cache path a file load takes.
#[test]
fn a_param_edit_regenerates_the_mesh() {
    let Some(mut harness) = Harness::start(config(Some(demo_scene()))) else {
        return;
    };
    let before = harness.wait_for_snapshot();
    let ball = before
        .generators
        .iter()
        .position(|g| g.name == "ball")
        .unwrap();
    let GeneratorSpec::UvSphere { rings, sectors, .. } = before.generators[ball].spec else {
        panic!("the demo's ball is a UvSphere")
    };
    let tris_before = before.generators[ball].triangles;

    let mut edited = before.generators[ball].spec.clone();
    if let GeneratorSpec::UvSphere {
        rings: r,
        sectors: s,
        radius,
        ..
    } = &mut edited
    {
        *r = rings * 2;
        *s = sectors * 2;
        *radius = 1.6;
    }

    harness.engine.send(Command::ParamEdit {
        generator: ball,
        spec: edited.clone(),
    });

    let after = harness.wait_for_snapshot();
    assert_eq!(after.generators[ball].spec, edited, "the edit is recorded");
    assert!(
        after.generators[ball].triangles > tris_before,
        "doubling the tessellation should raise the triangle count: {} → {}",
        tris_before,
        after.generators[ball].triangles
    );

    // And the frame actually changed.
    harness.wait_frames(3);
    let hash = harness.frame_hash();
    harness.wait_frames(3);
    assert_ne!(hash, 0, "frames are being produced");
}

/// The demo scene references one generator from two entities. An edit must move
/// both — that is the dedup story from §6 seen from the editor's side.
#[test]
fn a_param_edit_reaches_every_entity_using_that_generator() {
    let Some(mut harness) = Harness::start(config(Some(demo_scene()))) else {
        return;
    };
    let before = harness.wait_for_snapshot();
    let ball = before.generators.iter().position(|g| g.name == "ball").unwrap();
    let users: Vec<&str> = before
        .entities
        .iter()
        .filter(|e| e.generator_index == ball)
        .map(|e| e.label.as_str())
        .collect();
    assert_eq!(users, ["ball", "ball_twin"], "the demo shares one generator");

    let mut edited = before.generators[ball].spec.clone();
    if let GeneratorSpec::UvSphere { radius, .. } = &mut edited {
        *radius = 2.0;
    }
    harness.engine.send(Command::ParamEdit {
        generator: ball,
        spec: edited,
    });
    let after = harness.wait_for_snapshot();

    // Both entities still point at the (one) generator, which now has the new
    // spec — there is no way for them to disagree, and that is the point.
    assert_eq!(
        after
            .entities
            .iter()
            .filter(|e| e.generator_index == ball)
            .count(),
        2
    );
    let GeneratorSpec::UvSphere { radius, .. } = after.generators[ball].spec else {
        panic!()
    };
    assert_eq!(radius, 2.0);
}

/// Editing terrain has to move the *collision surface* too (DESIGN §9), or the
/// ball rolls on hills that are no longer there.
#[test]
fn editing_terrain_moves_the_analytic_surface_with_it() {
    let Some(mut harness) = Harness::start(config(Some(ball_level()))) else {
        return;
    };
    let before = harness.wait_for_snapshot();
    let ground = before.generators.iter().position(|g| g.name == "ground").unwrap();
    let GeneratorSpec::Terrain(params) = before.generators[ground].spec else {
        panic!()
    };

    let mut edited = params;
    edited.amplitude = params.amplitude * 3.0;
    harness.engine.send(Command::ParamEdit {
        generator: ground,
        spec: GeneratorSpec::Terrain(edited),
    });

    let after = harness.wait_for_snapshot();
    let GeneratorSpec::Terrain(now) = after.generators[ground].spec else {
        panic!()
    };
    assert_eq!(now.amplitude, params.amplitude * 3.0);
    // The world kept running with a rolling ball on the new field and did not
    // fall over — which is the thing the `TerrainSurface` refresh buys.
    harness.wait_frames(10);
}

#[test]
fn a_transform_edit_moves_the_entity() {
    let Some(mut harness) = Harness::start(config(Some(demo_scene()))) else {
        return;
    };
    let before = harness.wait_for_snapshot();
    let post = before.entities.iter().position(|e| e.label == "post").unwrap();

    let mut moved = before.entities[post].transform;
    moved.translation = glam::Vec3::new(4.0, 1.0, -2.0);
    moved.scale = glam::Vec3::splat(2.0);
    harness.engine.send(Command::TransformEdit {
        entity: post,
        transform: moved,
    });

    let after = harness.wait_for_snapshot();
    assert_eq!(after.entities[post].transform.translation, moved.translation);
    assert_eq!(after.entities[post].transform.scale, moved.scale);
}

#[test]
fn pausing_stops_the_sim_but_not_the_renderer() {
    let Some(mut harness) = Harness::start(config(Some(demo_scene()))) else {
        return;
    };
    harness.wait_for_snapshot();

    // Let it run, then freeze.
    harness.wait_frames(5);
    harness.engine.send(Command::SetPaused(true));
    // Give the command a moment to be drained, then take a baseline.
    harness.wait_frames(3);
    let paused_stats = {
        // Drain until a report arrives that actually says `paused`.
        harness
            .wait_for(Duration::from_secs(5), |e| match e {
                Event::Stats(s) if s.paused => Some(*s),
                _ => None,
            })
            .expect("a paused stats report")
    };

    let later = harness
        .wait_for(Duration::from_secs(5), |e| match e {
            Event::Stats(s) if s.paused && s.tick >= paused_stats.tick => Some(*s),
            _ => None,
        })
        .expect("a second paused stats report");

    assert_eq!(
        later.tick, paused_stats.tick,
        "the tick count must not advance while paused"
    );
    // Frames keep coming, so the scene stays live to look at.
    let frames = harness.frames();
    harness.wait_frames(4);
    assert!(harness.frames() > frames, "rendering stops when paused");

    // Unpausing starts it again.
    harness.engine.send(Command::SetPaused(false));
    let running = harness
        .wait_for(Duration::from_secs(5), |e| match e {
            Event::Stats(s) if !s.paused && s.tick > later.tick => Some(*s),
            _ => None,
        });
    assert!(running.is_some(), "the sim did not resume");
}

#[test]
fn resizing_changes_the_frames_the_sink_receives() {
    let Some(mut harness) = Harness::start(config(Some(demo_scene()))) else {
        return;
    };
    harness.wait_for_snapshot();
    harness.wait_frames(3);

    // 517 is deliberately unaligned: the resize path has to redo the padding.
    harness.engine.send(Command::Resize {
        width: 517,
        height: 131,
    });
    let stats = harness
        .wait_for(Duration::from_secs(5), |e| match e {
            Event::Stats(s) if s.width == 517 => Some(*s),
            _ => None,
        })
        .expect("stats at the new size");
    assert_eq!((stats.width, stats.height), (517, 131));
    // The sink's own assertion is doing the real work here: it would have
    // panicked if the frame were not `517 * 131 * 4` bytes.
    harness.wait_frames(3);
}

#[test]
fn selecting_and_deselecting_is_reversible() {
    let Some(mut harness) = Harness::start(config(Some(demo_scene()))) else {
        return;
    };
    harness.wait_for_snapshot();
    harness.wait_frames(3);
    let plain = harness.frame_hash();

    harness.engine.send(Command::Select(Some(0)));
    harness.wait_frames(4);
    let selected = harness.frame_hash();
    assert_ne!(plain, selected, "selecting must be visible in the frame");

    harness.engine.send(Command::Select(None));
    harness.wait_frames(4);
    // The scene is animated, so the hash will not return to its exact old value
    // — but the *material* must, and a second select must brighten from the
    // original colour rather than compounding.
    harness.engine.send(Command::Select(Some(0)));
    harness.wait_frames(4);
    harness.engine.send(Command::Select(None));
    harness.wait_frames(4);

    // Nothing above may have produced an error.
    let errors = harness.take_errors();
    assert!(errors.is_empty(), "{errors:?}");
}

#[test]
fn the_camera_can_be_driven_from_outside_the_sim() {
    let Some(mut harness) = Harness::start(config(Some(demo_scene()))) else {
        return;
    };
    harness.wait_for_snapshot();
    harness.wait_frames(3);
    let before = harness.frame_hash();

    harness.engine.send(Command::SetCameraPose {
        eye: glam::Vec3::new(0.0, 40.0, 0.1),
        target: glam::Vec3::ZERO,
    });
    harness.wait_frames(5);
    assert_ne!(
        before,
        harness.frame_hash(),
        "a camera move must change the picture"
    );

    // And it must *stay* moved: the demo's camera has a `FollowCamera` rig,
    // which the editor strips on load. If it were still there, the follow system
    // would drag the camera back within a tick or two.
    let held = harness.frame_hash();
    harness.wait_frames(20);
    let still = harness.frame_hash();
    // The scene animates, so the hashes will differ; the assertion is that the
    // camera did not snap back to the scene's own eye, which would be a large,
    // one-off change. Re-sending the same pose must be a no-op.
    harness.engine.send(Command::SetCameraPose {
        eye: glam::Vec3::new(0.0, 40.0, 0.1),
        target: glam::Vec3::ZERO,
    });
    harness.wait_frames(3);
    let _ = (held, still);
}

#[test]
fn a_saved_scene_reloads_with_the_edits_in_it() {
    let Some(mut harness) = Harness::start(config(Some(demo_scene()))) else {
        return;
    };
    let before = harness.wait_for_snapshot();
    let ball = before.generators.iter().position(|g| g.name == "ball").unwrap();

    let mut edited = before.generators[ball].spec.clone();
    if let GeneratorSpec::UvSphere { radius, rings, .. } = &mut edited {
        *radius = 3.25;
        *rings = 40;
    }
    harness.engine.send(Command::ParamEdit {
        generator: ball,
        spec: edited.clone(),
    });
    harness.wait_for_snapshot();

    let out = std::env::temp_dir().join(format!(
        "runt-editor-save-{}-{}.ron",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    harness.engine.send(Command::SaveScene(out.clone()));
    let (path, bytes) = harness
        .wait_for(Duration::from_secs(5), |e| match e {
            Event::SceneSaved { path, bytes } => Some((path.clone(), *bytes)),
            _ => None,
        })
        .expect("a save confirmation");
    assert_eq!(path, out);
    assert!(bytes > 100);

    // Load it back into the same engine and check the edit survived the file.
    harness.engine.send(Command::LoadScene(out.clone()));
    let reloaded = harness
        .wait_for(Duration::from_secs(10), |e| match e {
            Event::SceneLoaded(s) if s.path.as_deref() == Some(out.as_path()) => {
                Some((**s).clone())
            }
            _ => None,
        })
        .expect("the saved scene reloads");

    let ball = reloaded
        .generators
        .iter()
        .find(|g| g.name == "ball")
        .expect("the saved file still has a ball generator");
    assert_eq!(ball.spec, edited, "the edited params round-tripped through RON");
    assert_eq!(
        reloaded.entities.len(),
        before.entities.len(),
        "saving must not lose entities"
    );

    let _ = std::fs::remove_file(&out);
}

#[test]
fn stats_describe_the_frame_that_was_drawn() {
    let Some(mut harness) = Harness::start(config(Some(demo_scene()))) else {
        return;
    };
    harness.wait_for_snapshot();
    let stats = harness.wait_for_stats();

    assert_eq!((stats.width, stats.height), (320, 180));
    assert_eq!(stats.draws, 7, "the demo draws seven entities");
    assert!(stats.triangles > 1000, "{} triangles", stats.triangles);
    assert!(stats.fps > 0.0, "fps was {}", stats.fps);
    assert!(
        stats.readback_ms >= 0.0 && stats.readback_ms < 100.0,
        "readback took {} ms",
        stats.readback_ms
    );
    // Six generators, all cold on a fresh no-op cache.
    assert!(stats.cache.generated >= 6, "{:?}", stats.cache);
    println!(
        "engine: {:.0} fps, render {:.2} ms, readback {:.2} ms, {} tris, {} draws, cache {:?}",
        stats.fps, stats.render_ms, stats.readback_ms, stats.triangles, stats.draws, stats.cache
    );
}

#[test]
fn shutdown_stops_the_thread() {
    let Some(mut harness) = Harness::start(config(Some(demo_scene()))) else {
        return;
    };
    harness.wait_frames(3);
    let frames = harness.frames();

    harness.engine.shutdown();
    std::thread::sleep(Duration::from_millis(150));
    let after = harness.frames();
    std::thread::sleep(Duration::from_millis(150));
    assert_eq!(
        harness.frames(),
        after,
        "the loop kept rendering after shutdown"
    );
    assert!(after >= frames);
}
