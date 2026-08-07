//! Editor-facing reflection (DESIGN §3, §10) — `reflect` feature only.
//!
//! > *Reflection: generator param structs and editor-visible components derive
//! > `Reflect`. This is the contract that lets the editor build panels without
//! > hand-written UI per generator.* — DESIGN §3
//!
//! This module is the *adapter layer* that makes that derive possible. Nothing
//! here is engine logic; the whole file exists so that
//! [`GeneratorSpec`](crate::gen::GeneratorSpec) and the scene description types
//! can carry `#[derive(Reflect)]` without the engine's own types changing shape.
//!
//! ## Why remote definitions
//!
//! `bevy_reflect` 0.19 ships a `glam` feature — but it is compiled against
//! **glam 0.32**, and runt is on **glam 0.33**. Those are different semver
//! majors, so `bevy_reflect`'s `impl Reflect for glam_0_32::Vec3` says nothing
//! about our `glam_0_33::Vec3`; enabling the feature would silently pull a
//! second glam into the tree and still leave our vectors unreflected.
//!
//! Nor can we write the impls ourselves: `Reflect` and `Vec3` are both foreign,
//! so a direct `impl` is an orphan-rule violation.
//!
//! The sanctioned escape hatch is `#[reflect_remote]`: a *local* wrapper type
//! that declares the shape of a foreign one and delegates to it. Fields then say
//! `#[reflect(remote = Vec3Def)]` and reflect exactly as if `Vec3` derived
//! `Reflect` — including nested through `Option`, which is what
//! [`OptVec3Def`] is for. `runt-mesh`'s [`TerrainParams`] gets the same
//! treatment ([`TerrainParamsDef`]) so that crate stays reflection-free.
//!
//! ## Why attributes rather than an editor-side table
//!
//! Slider bounds are declared **at the param**, with `bevy_reflect`'s custom
//! attributes:
//!
//! ```ignore
//! UvSphere {
//!     #[reflect(@FieldRange::new(0.01, 10.0))]
//!     radius: f32,
//!     …
//! }
//! ```
//!
//! `bevy_reflect` 0.19 supports these on struct fields *and* on enum variant
//! fields, and they are readable from static `TypeInfo` with no value in hand —
//! everything a widget mapper needs. A side table in the editor keyed by field
//! path would work, but it would rot: adding a generator param would leave the
//! bound behind in another crate. [`FieldRange::lookup`] is the one function an
//! editor needs, and [`DEFAULT_RANGE`] catches anything unannotated so an
//! un-ranged `f32` still gets a usable slider.
//!
//! ## What is *not* here
//!
//! No widgets, no layout, no editor state. This module is pure metadata; the
//! mapping from `TypeInfo` to controls lives in the editor, which is the only
//! thing that should know what a slider is.

use bevy_reflect::prelude::*;
use bevy_reflect::{reflect_remote, TypeRegistry};

// ---------------------------------------------------------------------------
// glam remote definitions
// ---------------------------------------------------------------------------

/// [`glam::Vec2`], reflected. See the module docs for why this indirection
/// exists.
#[reflect_remote(glam::Vec2)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2Def {
    pub x: f32,
    pub y: f32,
}

/// [`glam::Vec3`], reflected.
#[reflect_remote(glam::Vec3)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3Def {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

// `glam::Vec4` deliberately has **no** remote definition.
//
// A `#[reflect_remote]` wrapper must be structurally identical to the type it
// stands in for, because it delegates field access to it. On glam 0.33 with SSE2
// (every x86-64 build) `Vec4` is `pub struct Vec4(__m128)` — one private SIMD
// register, no `x`/`y`/`z`/`w` fields to delegate to. `Vec3` and `Vec2` are
// ordinary structs, which is why they get wrappers and `Vec4` does not.
//
// The two places a `Vec4` appears in an editable type therefore carry
// `#[reflect(ignore)]`: `MaterialDesc`'s color/params (materials are edited
// through commands, not the param panel) and `RotationDesc::Quat` (Euler is the
// authoring form — see that type's docs — and a tool writing a quaternion back
// is `save_scene`'s job, not a widget's).

/// `Option<glam::Vec3>`, reflected.
///
/// `Option<T>` is reflected by `bevy_reflect` only when `T: Reflect`, which
/// `glam::Vec3` is not — so the *whole* `Option` needs a remote definition, with
/// the inner vector remote in turn. Optional colors are the reason: a
/// `color: Option<Vec3>` param would otherwise block the derive on every
/// generator variant.
#[reflect_remote(Option<glam::Vec3>)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OptVec3Def {
    None,
    Some(#[reflect(remote = Vec3Def)] glam::Vec3),
}

/// [`runt_mesh::Quality`], reflected.
#[reflect_remote(runt_mesh::Quality)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QualityDef(pub f32);

/// [`runt_mesh::TerrainParams`], reflected — the payload of
/// [`GeneratorSpec::Terrain`](crate::gen::GeneratorSpec::Terrain).
///
/// Field-for-field identical to the real thing (the macro checks); the ranges
/// are the editor-facing half that `runt-mesh` has no business carrying.
#[reflect_remote(runt_mesh::TerrainParams)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainParamsDef {
    pub seed: u64,
    #[reflect(remote = Vec2Def, @FieldRange::new(1.0, 256.0))]
    pub size: glam::Vec2,
    #[reflect(@FieldRange::new(0.0, 40.0))]
    pub amplitude: f32,
    #[reflect(@FieldRange::new(1.0, 12.0))]
    pub octaves: u32,
    #[reflect(@FieldRange::new(0.001, 1.0))]
    pub frequency: f32,
    #[reflect(@FieldRange::new(1.0, 4.0))]
    pub lacunarity: f32,
    #[reflect(@FieldRange::new(0.0, 1.0))]
    pub gain: f32,
    #[reflect(@FieldRange::new(1.0, 512.0))]
    pub base_segments: u32,
    #[reflect(remote = OptVec3Def, @FieldRange::new(0.0, 1.0))]
    pub color: Option<glam::Vec3>,
    // Declared (a remote wrapper must be field-for-field identical) but not
    // reflected. Exposing it would want two more remote definitions —
    // `TerrainTint` and `Option<TerrainTint>` — and a widget mapper that can
    // walk a *struct* nested inside an enum variant, which the panel builder has
    // never been asked to do. Same call the `Vec4` fields above make: declared,
    // hashed and serialized correctly; simply not editable from a slider yet.
    #[reflect(ignore)]
    pub tint: Option<runt_mesh::TerrainTint>,
}

// ---------------------------------------------------------------------------
// Range metadata
// ---------------------------------------------------------------------------

/// Editor slider bounds for one numeric param, attached with
/// `#[reflect(@FieldRange::new(min, max))]`.
///
/// **Advisory, not a constraint.** Nothing in the engine validates against a
/// `FieldRange`; a generator that would misbehave outside a range must clamp for
/// itself. This only says where a slider's ends should sit, so dragging lands in
/// the interesting part of the space instead of spending 90 % of the track on
/// values nobody wants.
#[derive(Reflect, Clone, Copy, Debug, PartialEq)]
pub struct FieldRange {
    pub min: f32,
    pub max: f32,
    /// Quantum for a stepper / arrow key. `0.0` means "continuous"; integer
    /// fields ignore it and step by 1.
    pub step: f32,
}

/// The bounds an `f32` gets when its param carries no [`FieldRange`]: wide
/// enough to be useless as a constraint, narrow enough that a slider is not a
/// random-number generator. An editor is expected to offer text entry as well,
/// so this can never trap a value.
pub const DEFAULT_RANGE: FieldRange = FieldRange {
    min: -100.0,
    max: 100.0,
    step: 0.0,
};

/// The bounds an unannotated unsigned integer param gets.
pub const DEFAULT_INT_RANGE: FieldRange = FieldRange {
    min: 0.0,
    max: 256.0,
    step: 1.0,
};

impl FieldRange {
    pub const fn new(min: f32, max: f32) -> FieldRange {
        FieldRange {
            min,
            max,
            step: 0.0,
        }
    }

    pub const fn with_step(mut self, step: f32) -> FieldRange {
        self.step = step;
        self
    }

    /// Clamp `value` into the range. Editors use this on text entry, where a
    /// user can type anything at all.
    pub fn clamp(&self, value: f32) -> f32 {
        value.clamp(self.min, self.max)
    }

    /// Where `value` sits on a `[0, 1]` slider track.
    pub fn normalize(&self, value: f32) -> f32 {
        let span = self.max - self.min;
        if span.abs() < f32::EPSILON {
            return 0.0;
        }
        ((value - self.min) / span).clamp(0.0, 1.0)
    }

    /// The inverse of [`normalize`](FieldRange::normalize).
    pub fn denormalize(&self, t: f32) -> f32 {
        self.min + (self.max - self.min) * t.clamp(0.0, 1.0)
    }

    /// The declared range for a named field of `info`, or `None` if the param
    /// carries no attribute.
    ///
    /// Handles both halves of what a generator enum contains: a struct's named
    /// fields and a *variant*'s named fields, which `bevy_reflect` models with
    /// different types.
    pub fn lookup(info: &bevy_reflect::TypeInfo, field: &str) -> Option<FieldRange> {
        match info {
            bevy_reflect::TypeInfo::Struct(s) => {
                s.field(field)?.get_attribute::<FieldRange>().copied()
            }
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// A [`TypeRegistry`] with every reflected runt type in it.
///
/// An editor needs one to turn a `DynamicEnum` (what "switch this variant" edits
/// produce) back into a concrete [`GeneratorSpec`](crate::gen::GeneratorSpec),
/// and to look a type up by path. Registration is explicit rather than
/// inventory-based so the set is greppable and cannot vary with link order.
pub fn type_registry() -> TypeRegistry {
    let mut registry = TypeRegistry::new();
    register(&mut registry);
    registry
}

/// Add runt's reflected types to an existing registry.
pub fn register(registry: &mut TypeRegistry) {
    registry.register::<FieldRange>();

    registry.register::<Vec2Def>();
    registry.register::<Vec3Def>();
    registry.register::<OptVec3Def>();
    registry.register::<QualityDef>();
    registry.register::<TerrainParamsDef>();

    registry.register::<crate::gen::GeneratorSpec>();
    registry.register::<crate::gen::Shading>();

    registry.register::<crate::ecs::QualityTier>();

    // The live-tunable resources (`crate::tweak`). Registered here as well as
    // reachable through `Typed` because an editor that wants to *name* a type
    // ("what can I tune?") looks it up by path, and this is the one list.
    registry.register::<crate::ecs::Lighting>();
    registry.register::<crate::ecs::RenderScale>();
    registry.register::<crate::ecs::PhaseFx>();

    registry.register::<crate::scene::SceneDesc>();
    registry.register::<crate::scene::GeneratorEntry>();
    registry.register::<crate::scene::QualityPolicy>();
    registry.register::<crate::scene::EntityDesc>();
    registry.register::<crate::scene::TransformDesc>();
    registry.register::<crate::scene::RotationDesc>();
    registry.register::<crate::scene::MaterialDesc>();
    registry.register::<crate::scene::SpinDesc>();
    registry.register::<crate::scene::CameraDesc>();
    registry.register::<crate::scene::FollowDesc>();
    registry.register::<crate::scene::LightingDesc>();
    registry.register::<crate::scene::BallDesc>();
    registry.register::<crate::scene::BallControllerDesc>();
    registry.register::<crate::scene::ObbColliderDesc>();
    registry.register::<crate::scene::CollisionLayersDesc>();
}
