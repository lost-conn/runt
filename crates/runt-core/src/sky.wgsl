// The background gradient (DESIGN §5). See `src/sky.rs` for the reasoning and
// for the CPU twin of `gradient` that the screenshot test holds this against.
//
// Standalone WGSL, unlike `shader.wgsl`: the sky has no material and therefore
// no feature consts to prepend. It binds @group(0) only — the frame block — so
// it uses its own pipeline layout and never needs a per-instance slot.

struct Frame {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    light_dir: vec4<f32>,
    light_color: vec4<f32>,
    sky_color: vec4<f32>,
    ground_color: vec4<f32>,
    horizon_color: vec4<f32>,
};
@group(0) @binding(0) var<uniform> frame: Frame;

struct SkyOut {
    @builtin(position) clip: vec4<f32>,
    // Normalized device coordinates, +Y up. Interpolated across the triangle,
    // which is the whole point: it is the per-pixel input to the ray rebuild.
    @location(0) ndc: vec2<f32>,
};

// One oversized triangle covering the viewport, from `vertex_index` alone — no
// vertex buffer, no index buffer, no bind group but the frame's. The three
// corners are (-1,-1), (3,-1), (-1,3), whose intersection with the unit square
// is exactly the screen.
@vertex
fn vs_sky(@builtin(vertex_index) index: u32) -> SkyOut {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    let ndc = uv * 2.0 - vec2<f32>(1.0, 1.0);
    var out: SkyOut;
    // z = 1 is the far plane. The pipeline neither tests nor writes depth (the
    // sky is drawn first, into a freshly cleared frame), so this is only a
    // legal value rather than a load-bearing one.
    out.clip = vec4<f32>(ndc, 1.0, 1.0);
    out.ndc = ndc;
    return out;
}

@fragment
fn fs_sky(in: SkyOut) -> @location(0) vec4<f32> {
    // Unproject the same NDC column at the near and far planes; the difference
    // is the view ray, with no camera pose needed on this side.
    let near = frame.inv_view_proj * vec4<f32>(in.ndc, 0.0, 1.0);
    let far = frame.inv_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);
    let a = near.xyz / near.w;
    let b = far.xyz / far.w;
    let delta = b - a;
    let len = length(delta);
    // A degenerate view-projection (the no-camera path hands us the identity)
    // must produce a colour, not a NaN.
    var dir = vec3<f32>(0.0, 0.0, 1.0);
    if (len > 0.0) {
        dir = delta / len;
    }

    let t = clamp(dir.y, -1.0, 1.0);
    var color: vec3<f32>;
    if (t >= 0.0) {
        color = mix(frame.horizon_color.rgb, frame.sky_color.rgb, pow(t, 0.55));
    } else {
        color = mix(frame.horizon_color.rgb, frame.ground_color.rgb, pow(-t, 0.75));
    }
    return vec4<f32>(color, 1.0);
}
