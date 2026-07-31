struct Uniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

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
    out.clip = u.view_proj * u.model * vec4<f32>(pos, 1.0);
    out.normal = (u.model * vec4<f32>(normal, 0.0)).xyz;
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.4, 1.0, 0.6));
    let n = normalize(in.normal);
    let diffuse = max(dot(n, light_dir), 0.0);
    let shade = 0.25 + diffuse * 0.75;
    return vec4<f32>(in.color * shade, 1.0);
}
