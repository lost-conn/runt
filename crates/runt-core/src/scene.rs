//! Scenes as save-as-params: RON in, entities out (DESIGN §6).
//!
//! > *Scene file = save-as-params: RON listing generator invocations (name +
//! > params + quality policy) and entity placements (transform, material,
//! > generator ref).* — DESIGN §6
//!
//! Which is exactly what [`SceneDesc`] is. Two lists and two rigs:
//!
//! ```text
//! generators: [ (name, spec, quality) ]      what to build
//! entities:   [ (generator, transform, …) ]  where to put it
//! camera:     …                              one, per DESIGN §5
//! lighting:   …                              one rig, per DESIGN §5
//! ```
//!
//! The split is the point. *Placement* lives on the entity; *shape* lives in the
//! generator. Two entities naming the same generator get the same content hash,
//! therefore the same [`MeshHandle`](crate::MeshHandle), therefore one pair of
//! GPU buffers — and `assets/demo.ron` proves it on purpose, with two spheres
//! referencing one generator entry.
//!
//! The format is meant to be *typed by a person*. Optional fields carry
//! `#[serde(default)]`, so a placement is usually three lines, and a generator
//! entry reads as the function call it is.
//!
//! ## Loading
//!
//! [`load_scene`] resolves every generator through the [`GenCache`] into the
//! [`MeshLibrary`], then spawns one entity per placement. Nothing here knows how
//! a mesh was obtained — cold generation and a cache hit are indistinguishable
//! by construction (DESIGN §6's "purely an optimization").
//!
//! ## Saving
//!
//! [`save_scene`] serializes the [`LoadedScene`] resource that [`load_scene`]
//! left behind, refreshing any transform the world has since changed. It does
//! **not** reverse-engineer the world: entities spawned outside a scene are not
//! saved in v1, and neither is derived state such as the follow camera's current
//! pose (it is recomputed from its target on load).
//!
//! ## Deviations
//!
//! The scene is embedded with `include_str!` so the wasm build needs no fetch.
//! Loading a scene from a URL (and therefore hot-reloading one) is future work
//! that wants the async story §13 is still holding open.

use bevy_ecs::prelude::*;
use glam::{EulerRot, Quat, Vec3, Vec4};
use runt_mesh::{MeshData, Quality};
use serde::{Deserialize, Serialize};

use crate::cache::GenCache;
use crate::camera::{Camera, FollowCamera};
use crate::ecs::{
    DemoEntity, DemoScene, GeneratorRef, GlobalTransform, Interpolated, Lighting, MeshRef,
    QualityTier, Spin, TerrainSurface, Transform,
};
use crate::gen::GeneratorSpec;
use crate::material::{Material, MaterialVariant};
use crate::registry::{MeshHandle, MeshLibrary};

/// The demo scene, embedded at build time (DESIGN §12 step 4).
pub const DEMO_SCENE_RON: &str = include_str!("../../../assets/demo.ron");

/// The demo's spin rate: 0.4 rad/s about +Y, the rate the pre-ECS renderer
/// hardcoded, kept so screenshots compare like with like.
pub const DEMO_SPIN: f32 = 0.4;

/// Where the demo camera sits once it has settled on its target.
pub const DEMO_EYE: Vec3 = Vec3::new(0.0, 2.4, 6.5);

// ---------------------------------------------------------------------------
// The description
// ---------------------------------------------------------------------------

/// A whole scene, as it appears in a `.ron` file.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SceneDesc {
    /// Generator invocations, each under a name entities refer to.
    #[serde(default)]
    pub generators: Vec<GeneratorEntry>,
    /// Placements. Order is spawn order, and therefore stable.
    #[serde(default)]
    pub entities: Vec<EntityDesc>,
    #[serde(default)]
    pub camera: CameraDesc,
    #[serde(default)]
    pub lighting: LightingDesc,
}

/// One named generator invocation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GeneratorEntry {
    /// What entities call it. Scene-local; unrelated to
    /// [`GeneratorSpec::kind`].
    pub name: String,
    pub spec: GeneratorSpec,
    #[serde(default)]
    pub quality: QualityPolicy,
}

/// How a generator's tessellation responds to the device tier (DESIGN §6:
/// "per-generator overrides allowed").
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum QualityPolicy {
    /// Use the session's [`QualityTier`] as-is. The normal case.
    #[default]
    Inherit,
    /// Tier × this factor — "always half the scene's detail", for background
    /// props that never earn their triangles.
    Scaled(f32),
    /// Ignore the tier. For geometry whose silhouette is the gameplay (a
    /// collectible's ring, a ramp) and must not coarsen on a weak device.
    Fixed(f32),
}

impl QualityPolicy {
    pub fn resolve(self, tier: QualityTier) -> Quality {
        match self {
            QualityPolicy::Inherit => Quality(tier.0),
            QualityPolicy::Scaled(f) => Quality(tier.0 * f),
            QualityPolicy::Fixed(q) => Quality(q),
        }
    }
}

/// One entity placement.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EntityDesc {
    /// Optional handle so other parts of the file (the follow camera) can point
    /// at this entity.
    #[serde(default)]
    pub name: Option<String>,
    /// Which [`GeneratorEntry`] supplies its geometry.
    pub generator: String,
    #[serde(default)]
    pub transform: TransformDesc,
    #[serde(default)]
    pub material: MaterialDesc,
    /// Constant-rate rotation, if any.
    #[serde(default)]
    pub spin: Option<SpinDesc>,
    /// Give it an [`Interpolated`] so the renderer blends its pose between
    /// ticks. Only entities that actually move need it (DESIGN §4); a static
    /// prop with one is just a wasted component. Implied by `spin`.
    #[serde(default)]
    pub interpolated: bool,
}

/// TRS placement. Every field is optional and defaults to identity.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransformDesc {
    #[serde(default)]
    pub translation: Vec3,
    #[serde(default)]
    pub rotation: RotationDesc,
    #[serde(default = "vec3_one")]
    pub scale: Vec3,
}

impl Default for TransformDesc {
    fn default() -> TransformDesc {
        TransformDesc {
            translation: Vec3::ZERO,
            rotation: RotationDesc::default(),
            scale: Vec3::ONE,
        }
    }
}

/// Orientation, in whichever form is easiest to type.
///
/// **Euler is the authoring form**: degrees, applied in glam's `EulerRot::XYZ`
/// order (intrinsic X, then Y, then Z). Quaternions are what a tool writes back
/// — [`save_scene`] uses `Quat` for any transform the world has changed, because
/// converting a live orientation back to Euler is ambiguous (gimbal-equivalent
/// triples) and would silently rewrite a hand-authored file into a different
/// but equal rotation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum RotationDesc {
    #[default]
    Identity,
    /// Degrees, `EulerRot::XYZ`.
    Euler(Vec3),
    /// Raw `(x, y, z, w)`. Normalized on load.
    Quat(Vec4),
}

impl RotationDesc {
    pub fn quat(self) -> Quat {
        match self {
            RotationDesc::Identity => Quat::IDENTITY,
            RotationDesc::Euler(d) => Quat::from_euler(
                EulerRot::XYZ,
                d.x.to_radians(),
                d.y.to_radians(),
                d.z.to_radians(),
            ),
            // A hand-typed quaternion is rarely exactly unit; normalizing is
            // cheaper than letting a slightly-long one scale the whole mesh.
            RotationDesc::Quat(q) => Quat::from_xyzw(q.x, q.y, q.z, q.w).normalize(),
        }
    }
}

impl TransformDesc {
    pub fn to_transform(self) -> Transform {
        Transform {
            translation: self.translation,
            rotation: self.rotation.quat(),
            scale: self.scale,
        }
    }

    /// The description of a live transform, orientation as a quaternion.
    pub fn from_transform(t: &Transform) -> TransformDesc {
        TransformDesc {
            translation: t.translation,
            rotation: if t.rotation == Quat::IDENTITY {
                RotationDesc::Identity
            } else {
                RotationDesc::Quat(Vec4::new(
                    t.rotation.x,
                    t.rotation.y,
                    t.rotation.z,
                    t.rotation.w,
                ))
            },
            scale: t.scale,
        }
    }
}

/// A material, with the variant bitflags spelled out as booleans so a scene file
/// never contains a magic number.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterialDesc {
    #[serde(default = "vec4_one")]
    pub base_color: Vec4,
    /// Reserved uniform slot (ramp threshold/softness as §5's variants land).
    #[serde(default)]
    pub params: Vec4,
    #[serde(default = "yes")]
    pub vertex_color: bool,
    /// Reserved (§7).
    #[serde(default)]
    pub texture: bool,
    /// Reserved (§5).
    #[serde(default)]
    pub ramp: bool,
    /// Reserved (§7).
    #[serde(default)]
    pub live_texture: bool,
}

impl Default for MaterialDesc {
    fn default() -> MaterialDesc {
        MaterialDesc {
            base_color: Vec4::ONE,
            params: Vec4::ZERO,
            vertex_color: true,
            texture: false,
            ramp: false,
            live_texture: false,
        }
    }
}

impl MaterialDesc {
    pub fn to_material(self) -> Material {
        let mut variant = MaterialVariant::NONE;
        for (on, flag) in [
            (self.vertex_color, MaterialVariant::VERTEX_COLOR),
            (self.texture, MaterialVariant::TEXTURE),
            (self.ramp, MaterialVariant::RAMP),
            (self.live_texture, MaterialVariant::LIVE_TEX),
        ] {
            if on {
                variant |= flag;
            }
        }
        Material {
            base_color: self.base_color,
            params: self.params,
            variant,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SpinDesc {
    #[serde(default = "vec3_y")]
    pub axis: Vec3,
    pub rad_per_sec: f32,
}

/// The scene's one camera (DESIGN §5).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CameraDesc {
    pub eye: Vec3,
    /// What it initially looks at. With a `follow`, this is only the starting
    /// aim; the target's position takes over from the first tick.
    #[serde(default)]
    pub target: Vec3,
    #[serde(default = "sixty")]
    pub fov_y_degrees: f32,
    #[serde(default = "near")]
    pub z_near: f32,
    #[serde(default = "far")]
    pub z_far: f32,
    #[serde(default)]
    pub follow: Option<FollowDesc>,
}

impl Default for CameraDesc {
    fn default() -> CameraDesc {
        CameraDesc {
            eye: DEMO_EYE,
            target: Vec3::ZERO,
            fov_y_degrees: 60.0,
            z_near: 0.1,
            z_far: 100.0,
            follow: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FollowDesc {
    /// The `name` of an entity in this scene.
    pub entity: String,
    /// World-space offset from the target to the camera's rest position.
    pub offset: Vec3,
    /// Approach rate, 1/seconds. ~2 is a lazy drift, ~12 is nearly rigid.
    pub stiffness: f32,
}

/// The light rig (DESIGN §5). Mirrors [`Lighting`] field for field.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LightingDesc {
    /// Direction *towards* the key light.
    pub key_dir: Vec3,
    pub key_color: Vec3,
    pub sky_color: Vec3,
    pub ground_color: Vec3,
}

impl Default for LightingDesc {
    fn default() -> LightingDesc {
        LightingDesc::from(Lighting::default())
    }
}

impl From<Lighting> for LightingDesc {
    fn from(l: Lighting) -> LightingDesc {
        LightingDesc {
            key_dir: l.key_dir,
            key_color: l.key_color,
            sky_color: l.sky_color,
            ground_color: l.ground_color,
        }
    }
}

impl LightingDesc {
    pub fn to_lighting(self) -> Lighting {
        Lighting {
            key_dir: self.key_dir,
            key_color: self.key_color,
            sky_color: self.sky_color,
            ground_color: self.ground_color,
        }
    }
}

// serde default helpers — `#[serde(default = "…")]` needs a path, not a literal.
fn vec3_one() -> Vec3 {
    Vec3::ONE
}
fn vec3_y() -> Vec3 {
    Vec3::Y
}
fn vec4_one() -> Vec4 {
    Vec4::ONE
}
fn yes() -> bool {
    true
}
fn sixty() -> f32 {
    60.0
}
fn near() -> f32 {
    0.1
}
fn far() -> f32 {
    100.0
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum SceneError {
    /// The RON did not parse, or did not match the schema.
    Parse(String),
    /// An entity named a generator the file does not define.
    UnknownGenerator { entity: usize, name: String },
    /// The camera's follow target names no entity.
    UnknownFollowTarget(String),
    /// [`save_scene`] on a world that never loaded one.
    NoSceneLoaded,
    Serialize(String),
}

impl std::fmt::Display for SceneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SceneError::Parse(e) => write!(f, "scene parse error: {e}"),
            SceneError::UnknownGenerator { entity, name } => write!(
                f,
                "entity {entity} references generator {name:?}, which the scene does not define"
            ),
            SceneError::UnknownFollowTarget(name) => {
                write!(f, "camera follows entity {name:?}, which the scene does not define")
            }
            SceneError::NoSceneLoaded => write!(f, "no scene has been loaded into this world"),
            SceneError::Serialize(e) => write!(f, "scene serialize error: {e}"),
        }
    }
}

impl std::error::Error for SceneError {}

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// What [`load_scene`] built, kept so [`save_scene`] never has to guess.
///
/// The parallel `Vec`s are index-aligned with `desc.entities`, so save is a zip
/// rather than a search, and the ordering is spawn order — no hash iteration
/// anywhere near it (DESIGN §3).
#[derive(Resource, Clone, Debug)]
pub struct LoadedScene {
    pub desc: SceneDesc,
    /// The entity spawned for `desc.entities[i]`.
    pub spawned: Vec<Entity>,
    /// The camera entity.
    pub camera: Entity,
}

impl LoadedScene {
    /// The entity spawned for the placement named `name`, if any.
    pub fn entity(&self, name: &str) -> Option<Entity> {
        self.desc
            .entities
            .iter()
            .zip(&self.spawned)
            .find(|(desc, _)| desc.name.as_deref() == Some(name))
            .map(|(_, &e)| e)
    }
}

/// The scene a [`Sim`](crate::Sim) was configured to load, consumed by
/// [`load_pending_scene`] during `Startup`.
#[derive(Resource, Clone, Debug, Default)]
pub struct PendingScene(pub Option<String>);

/// What a load did. Useful for tests and for a log line; nothing in the engine
/// branches on it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SceneStats {
    pub generators: usize,
    pub entities: usize,
    /// Distinct mesh handles the generators resolved to. Lower than
    /// `generators` when two entries produce identical geometry.
    pub meshes: usize,
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Parse a scene without touching a world. Cheap enough to use as validation.
pub fn parse_scene(ron_src: &str) -> Result<SceneDesc, SceneError> {
    ron::from_str::<SceneDesc>(ron_src).map_err(|e| SceneError::Parse(e.to_string()))
}

/// The demo scene, parsed. Panics if `assets/demo.ron` does not match the
/// schema — which is the point: it is compiled in, so a broken demo scene should
/// fail loudly at the first test rather than at a customer's page load.
pub fn demo_scene() -> SceneDesc {
    parse_scene(DEMO_SCENE_RON).expect("assets/demo.ron must parse")
}

/// Parse `ron_src` and build it into `world`.
pub fn load_scene(world: &mut World, ron_src: &str) -> Result<SceneStats, SceneError> {
    spawn_scene(world, parse_scene(ron_src)?)
}

/// Build an already-parsed scene into `world`, replacing any scene already
/// loaded there.
///
/// Resolution order matters: generators first (so a failure costs nothing), then
/// entities, then the camera — which needs the entities to exist before it can
/// follow one.
pub fn spawn_scene(world: &mut World, desc: SceneDesc) -> Result<SceneStats, SceneError> {
    world.init_resource::<MeshLibrary>();
    world.init_resource::<GenCache>();
    world.init_resource::<QualityTier>();
    let tier = *world.resource::<QualityTier>();

    // Validate before mutating anything: an entity naming a missing generator
    // should not leave half a scene behind.
    for (i, entity) in desc.entities.iter().enumerate() {
        if !desc.generators.iter().any(|g| g.name == entity.generator) {
            return Err(SceneError::UnknownGenerator {
                entity: i,
                name: entity.generator.clone(),
            });
        }
    }
    if let Some(follow) = &desc.camera.follow {
        if !desc
            .entities
            .iter()
            .any(|e| e.name.as_deref() == Some(follow.entity.as_str()))
        {
            return Err(SceneError::UnknownFollowTarget(follow.entity.clone()));
        }
    }

    despawn_previous_scene(world);

    // --- generators: spec → cache → MeshLibrary ---------------------------
    //
    // Both resources are taken out of the world at once because resolving needs
    // to read one and write the other; `resource_scope` is the sanctioned way
    // to do that without splitting the borrow of `World`.
    let resolved: Vec<(MeshHandle, u64)> =
        world.resource_scope(|world, mut cache: Mut<GenCache>| {
            world.resource_scope(|_world, mut library: Mut<MeshLibrary>| {
                desc.generators
                    .iter()
                    .map(|entry| {
                        let quality = entry.quality.resolve(tier);
                        let handle = cache.resolve(&entry.spec, quality, &mut library);
                        (handle, entry.spec.param_key(quality))
                    })
                    .collect()
            })
        });

    // --- entities ---------------------------------------------------------

    let mut spawned = Vec::with_capacity(desc.entities.len());
    for placement in &desc.entities {
        let index = desc
            .generators
            .iter()
            .position(|g| g.name == placement.generator)
            .expect("validated above");
        let (handle, param_key) = resolved[index];
        let transform = placement.transform.to_transform();

        let mut entity = world.spawn((
            DemoScene,
            MeshRef(handle),
            GeneratorRef {
                name: placement.generator.clone(),
                param_key,
            },
            placement.material.to_material(),
            transform,
            GlobalTransform(transform.matrix()),
        ));

        if let Some(spin) = placement.spin {
            entity.insert(Spin {
                axis: spin.axis,
                rad_per_sec: spin.rad_per_sec,
            });
        }
        // A spinner without `Interpolated` would judder; the file does not get
        // to make that mistake.
        if placement.interpolated || placement.spin.is_some() {
            entity.insert(Interpolated::from(&transform));
        }

        // Terrain carries its analytic field so physics can sample it without
        // ever going near the mesh (DESIGN §9).
        if let GeneratorSpec::Terrain(params) = &desc.generators[index].spec {
            debug_assert!(
                transform.rotation == Quat::IDENTITY && transform.scale == Vec3::ONE,
                "terrain entities must be translation-only; \
                 a rotated or scaled height field is not a height field"
            );
            entity.insert(TerrainSurface::new(params));
        }

        spawned.push(entity.id());
    }

    // --- rigs -------------------------------------------------------------

    world.insert_resource(desc.lighting.to_lighting());

    let camera = spawn_camera(world, &desc, &spawned);

    // The demo handle: whatever the camera watches, else the first spinner.
    let focus = desc
        .camera
        .follow
        .as_ref()
        .and_then(|f| named(&desc, &spawned, &f.entity))
        .or_else(|| {
            desc.entities
                .iter()
                .zip(&spawned)
                .find(|(e, _)| e.spin.is_some())
                .map(|(_, &e)| e)
        });
    if let Some(focus) = focus {
        world.insert_resource(DemoEntity(focus));
    } else {
        world.remove_resource::<DemoEntity>();
    }

    let stats = SceneStats {
        generators: desc.generators.len(),
        entities: spawned.len(),
        meshes: {
            let mut handles: Vec<u64> = resolved.iter().map(|(h, _)| h.0).collect();
            handles.sort_unstable();
            handles.dedup();
            handles.len()
        },
    };
    log::info!(
        "scene: {} generators → {} meshes, {} entities, cache {:?}",
        stats.generators,
        stats.meshes,
        stats.entities,
        world.resource::<GenCache>().stats()
    );

    world.insert_resource(LoadedScene {
        desc,
        spawned,
        camera,
    });
    Ok(stats)
}

fn named(desc: &SceneDesc, spawned: &[Entity], name: &str) -> Option<Entity> {
    desc.entities
        .iter()
        .zip(spawned)
        .find(|(e, _)| e.name.as_deref() == Some(name))
        .map(|(_, &e)| e)
}

fn spawn_camera(world: &mut World, desc: &SceneDesc, spawned: &[Entity]) -> Entity {
    let cam = &desc.camera;
    let pose = Transform::looking_at(cam.eye, cam.target, Vec3::Y);
    let mut entity = world.spawn((
        DemoScene,
        Camera {
            fov_y_rad: cam.fov_y_degrees.to_radians(),
            z_near: cam.z_near,
            z_far: cam.z_far,
        },
        pose,
        GlobalTransform(pose.matrix()),
        Interpolated::from(&pose),
    ));
    if let Some(follow) = &desc.camera.follow {
        let target = named(desc, spawned, &follow.entity).expect("validated above");
        entity.insert(FollowCamera {
            target,
            offset: follow.offset,
            stiffness: follow.stiffness,
        });
    }
    entity.id()
}

/// Remove the entities a previous [`spawn_scene`] created, so loading twice
/// replaces rather than accumulates.
///
/// Only entities this module spawned: anything gameplay put in the world stays,
/// which is the same boundary [`save_scene`] draws.
fn despawn_previous_scene(world: &mut World) {
    let Some(previous) = world.remove_resource::<LoadedScene>() else {
        return;
    };
    for entity in previous.spawned.into_iter().chain([previous.camera]) {
        world.despawn(entity);
    }
    world.remove_resource::<DemoEntity>();
}

/// `Startup`: load the scene the sim was configured with, if any.
pub fn load_pending_scene(world: &mut World) {
    let Some(PendingScene(Some(src))) = world.remove_resource::<PendingScene>() else {
        return;
    };
    if let Err(e) = load_scene(world, &src) {
        // A broken scene must not take the process with it — the host still gets
        // a running engine, just an empty one, and the message says why.
        log::error!("failed to load scene: {e}");
    }
}

// ---------------------------------------------------------------------------
// Saving
// ---------------------------------------------------------------------------

/// Serialize the loaded scene back to RON.
///
/// Transforms are refreshed from the world so that moving something (an editor
/// drag, later) persists; anything untouched keeps the exact form it was
/// authored in, Euler angles included. Comments in the source file are lost —
/// RON round-trips values, not text.
pub fn save_scene(world: &World) -> Result<String, SceneError> {
    let desc = scene_desc(world)?;
    ron::ser::to_string_pretty(&desc, ron_pretty()).map_err(|e| SceneError::Serialize(e.to_string()))
}

/// The loaded scene's description, with live transforms folded back in. The
/// value [`save_scene`] serializes.
pub fn scene_desc(world: &World) -> Result<SceneDesc, SceneError> {
    let loaded = world
        .get_resource::<LoadedScene>()
        .ok_or(SceneError::NoSceneLoaded)?;
    let mut desc = loaded.desc.clone();

    for (placement, &entity) in desc.entities.iter_mut().zip(&loaded.spawned) {
        let Some(live) = world.get::<Transform>(entity) else {
            continue; // Despawned since load; keep what the file said.
        };
        // Only rewrite when it actually moved. Otherwise a save would turn every
        // hand-typed `Euler((90, 0, 0))` into an equal-but-unreadable quaternion.
        if !transform_eq(*live, placement.transform.to_transform()) {
            placement.transform = TransformDesc::from_transform(live);
        }
    }

    if let Some(lighting) = world.get_resource::<Lighting>() {
        desc.lighting = LightingDesc::from(*lighting);
    }
    Ok(desc)
}

/// Component-wise equality with a tolerance a float round-trip cannot exceed.
fn transform_eq(a: Transform, b: Transform) -> bool {
    const EPS: f32 = 1e-6;
    a.translation.abs_diff_eq(b.translation, EPS)
        && a.scale.abs_diff_eq(b.scale, EPS)
        && a.rotation.abs_diff_eq(b.rotation, EPS)
}

/// Formatting that keeps a saved scene as readable as a written one.
fn ron_pretty() -> ron::ser::PrettyConfig {
    ron::ser::PrettyConfig::new()
        .indentor("    ")
        // `Vec3` serializes as a tuple struct; without this every vector would
        // be written as `Vec3(0.0, 1.0, 0.0)` instead of `(0.0, 1.0, 0.0)`.
        .struct_names(false)
        .separate_tuple_members(false)
        .enumerate_arrays(false)
}

// ---------------------------------------------------------------------------
// Demo helpers
// ---------------------------------------------------------------------------

/// A generator from the demo scene, by name. Panics if the demo does not define
/// it — these are test/bench helpers, and a typo should fail immediately.
pub fn demo_generator(name: &str) -> GeneratorSpec {
    demo_scene()
        .generators
        .into_iter()
        .find(|g| g.name == name)
        .unwrap_or_else(|| panic!("assets/demo.ron defines no generator {name:?}"))
        .spec
}

/// A demo generator's mesh at full quality. Reads the scene file rather than
/// restating its params, so these cannot drift away from what actually ships.
pub fn demo_mesh(name: &str) -> MeshData {
    demo_generator(name).generate(Quality::FULL)
}

pub fn ground_mesh() -> MeshData {
    demo_mesh("ground")
}

pub fn ball_mesh() -> MeshData {
    demo_mesh("ball")
}

pub fn post_mesh() -> MeshData {
    demo_mesh("post")
}

pub fn spike_mesh() -> MeshData {
    demo_mesh("spike")
}

pub fn ring_mesh() -> MeshData {
    demo_mesh("ring")
}

pub fn twisted_box_mesh() -> MeshData {
    demo_mesh("twisted_box")
}

/// The demo terrain's analytic field — what step 5's ball will roll on.
pub fn demo_terrain_params() -> runt_mesh::TerrainParams {
    match demo_generator("ground") {
        GeneratorSpec::Terrain(params) => params,
        other => panic!("the demo's ground generator is {}, not Terrain", other.kind()),
    }
}
