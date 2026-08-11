// The shadow map's depth-only pass (DESIGN §5, §11; see `src/shadow.rs`).
//
// Standalone WGSL, like `sky.wgsl`: no feature consts, because there is
// nothing to vary — every caster takes the same road here. A caster is a
// position and a model matrix and *nothing else*: no normal, no colour, no
// texture, no wave displacement. The vertex buffers still carry all of that
// (the pipeline declares the same two slots every material pipeline does, so
// the meshes and the instance buffer need no second layout), but only
// `@location(0)` and the matrix columns are read, and the rest never leaves
// the fetch stage.
//
// Vertex-wave water is deliberately rigid in here — and, in practice, absent:
// blended draws do not cast (a ghost with a solid shadow reads as a bug), and
// the port's water is transparent. A rigid shadow of an opaque waving surface
// would be the accepted degradation, not a wrong answer.
//
// Casters above the light are **pancaked**, not clipped: the light's near
// plane sits `2·extent` over the box's centre with nothing behind it, so a
// tall enough caster pokes through, and the rasterizer would clip its top —
// its whole silhouette, for a roof seen from a near-vertical light — out of
// the map. Clamping clip z to the near plane instead (safe because the
// projection is orthographic: w is 1, so clip z *is* depth) flattens that
// geometry onto the plane. The flattening writes a too-near depth for any
// triangle crossing the plane, and that is the accepted trade: such a caster
// is between the light and everything the map covers, so any depth ≤ its true
// depth occludes exactly the same receivers. WebGL2 is why it happens here in
// the vertex stage — `DepthClipControl` is the device-feature spelling of the
// same idea, and the floor does not have it.
//
// The fragment stage exists and does nothing. A depth-only pipeline could omit
// it entirely, but an empty entry point costs nothing, keeps the pipeline
// shaped like every other one in the engine, and sidesteps the one downlevel
// translation question a missing fragment stage would pose.

struct ShadowFrame {
    // World → the key light's clip space — `runt_core::shadow::light_view_proj`,
    // the same matrix the main pass samples the map through. One matrix, two
    // consumers, no restatement to drift: the main pass reads its copy from the
    // frame block, this pass from its own little buffer, and both are written
    // from the same `Mat4` in the same frame.
    light_view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> shadow_frame: ShadowFrame;

@vertex
fn vs_shadow(
    // Slot 0 — the mesh. Locations 1–3 (normal, uv, colour) are declared in the
    // buffer layout and simply not read.
    @location(0) pos: vec3<f32>,
    // Slot 1 — the instance's model matrix, columns 4–7 of the same
    // `InstanceRaw` the main pass steps. 8–9 (colour, params) are not read.
    @location(4) m0: vec4<f32>,
    @location(5) m1: vec4<f32>,
    @location(6) m2: vec4<f32>,
    @location(7) m3: vec4<f32>,
) -> @builtin(position) vec4<f32> {
    let model = mat4x4<f32>(m0, m1, m2, m3);
    var clip = shadow_frame.light_view_proj * model * vec4<f32>(pos, 1.0);
    // The pancake (header): a vertex above the light's near plane lands *on*
    // it instead of being clipped away, silhouette intact.
    clip.z = max(clip.z, 0.0);
    return clip;
}

@fragment
fn fs_shadow() {}
