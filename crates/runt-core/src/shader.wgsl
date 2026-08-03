// Stylized forward opaque shading (DESIGN §5): directional key light +
// hemisphere ambient, no PBR anywhere.
//
// This file is NOT standalone WGSL: the `F_*` feature consts it branches on are
// prepended per variant by `runt_core::material::variant_source`. They are
// `const bool`, so every disabled branch folds away at compile time.

// Field order must match `runt_core::FrameUniform`, and `sky.wgsl` restates the
// same block for the background pass.
struct Frame {
    view_proj: mat4x4<f32>,
    // Only the sky pass reads this one; it lives in the shared block because
    // there is exactly one frame uniform and splitting it to save 64 bytes
    // would cost a second bind group.
    inv_view_proj: mat4x4<f32>,
    // xyz: direction *towards* the key light, world space. w: unused.
    light_dir: vec4<f32>,
    light_color: vec4<f32>,
    sky_color: vec4<f32>,
    ground_color: vec4<f32>,
    horizon_color: vec4<f32>,
};
@group(0) @binding(0) var<uniform> frame: Frame;

// Per-entity slot, addressed with a dynamic uniform-buffer offset.
struct Instance {
    model: mat4x4<f32>,
    base_color: vec4<f32>,
    params: vec4<f32>,
};
@group(1) @binding(0) var<uniform> inst: Instance;

struct VSOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
) -> VSOut {
    var out: VSOut;
    out.clip = frame.view_proj * inst.model * vec4<f32>(pos, 1.0);
    // Rotating the normal by the model matrix is exact for the uniform-scale
    // transforms the engine places entities with; non-uniform scale would want
    // an inverse-transpose in the instance block.
    out.normal = (inst.model * vec4<f32>(normal, 0.0)).xyz;
    if (F_VERTEX_COLOR) {
        out.color = color;
    } else {
        out.color = vec3<f32>(1.0, 1.0, 1.0);
    }
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);

    // Hemisphere ambient: sky above, ground below, blended on the normal's Y.
    // Cheap, and it separates silhouettes far better than a constant term.
    let hemi = mix(frame.ground_color.rgb, frame.sky_color.rgb, 0.5 + 0.5 * n.y);
    let key = frame.light_color.rgb * max(dot(n, normalize(frame.light_dir.xyz)), 0.0);

    var albedo = inst.base_color.rgb;
    if (F_VERTEX_COLOR) {
        albedo = albedo * in.color;
    }
    return vec4<f32>((hemi + key) * albedo, inst.base_color.a);
}
