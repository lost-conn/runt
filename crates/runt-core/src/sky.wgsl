// The background gradient (DESIGN §5). See `src/sky.rs` for the reasoning and
// for the CPU twin of `gradient` that the screenshot test holds this against.
//
// Standalone WGSL, unlike `shader.wgsl`: the sky has no material and therefore
// no feature consts to prepend. It binds @group(0) only — the frame block — so
// it uses its own pipeline layout and never needs a per-instance slot.

// Field order must match `runt_core::FrameUniform`, and `shader.wgsl` restates
// the same block. The sky reads none of the last three — it has no material and
// therefore no phase circle, no clock and no screen-space anything — but the
// block is one buffer with one layout, so all three files move together or the
// uniform is silently misaligned.
struct Frame {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    light_dir: vec4<f32>,
    light_color: vec4<f32>,
    sky_color: vec4<f32>,
    ground_color: vec4<f32>,
    horizon_color: vec4<f32>,
    // xy: phase-circle centre in NDC. z: radius in NDC-Y units. w: strength.
    phase: vec4<f32>,
    // x: render-clock seconds. y: interpolation alpha. zw: reserved.
    time: vec4<f32>,
    // xy: render target size in pixels. zw: its reciprocal.
    viewport: vec4<f32>,
    // x: cloud cover, 0 = no cloud pass. y: sun-disk size as 1 − cos θ, 0 = no
    // disk. zw: reserved.
    sky_params: vec4<f32>,
    // World → the key light's clip space (the shadow map's matrix). The sky
    // neither casts nor receives; restated for the one-buffer reason above.
    light_view_proj: mat4x4<f32>,
    // x: shadow map bound. y: constant bias. z: slope bias. w: reserved.
    shadow_params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> frame: Frame;

// ---------------------------------------------------------------------------
// The cloud layer (`simple_sky.gdshader`, minus its texture)
// ---------------------------------------------------------------------------
//
// The original samples a `NoiseTexture2D`; this evaluates the same *kind* of
// field directly, because a sky that needed a texture would need a bake, a
// handle, a bind group and a second pipeline layout — for two octaves of value
// noise that cost less than the fetch would.
//
// The look's constants are the ones `playground.tscn` authors on its own
// `simple_sky` material, not the shader's defaults — the defaults are a
// scattered-puff sky and the game's is a soft overcast, and it is the game that
// exists. They are here rather than in the frame block for the reason
// `Lighting::clouds` gives: they are one authored weather, and a scene that
// wants another wants another shader, not seven more RON fields.

const CLOUD_SCALE: f32 = 0.6;
// How many lattice cells the *field* has per unit of `cloud_scale`.
//
// Not one of the original's uniforms, and it has to exist: `cloud_scale` maps
// the view direction onto a 0..0.6 patch of a **texture**, whose own frequency
// is a property of the image Godot samples (`terrain_noise.tres`). Evaluating
// a lattice field directly instead means that frequency has to be stated, and
// at 1 the whole sky would land inside a single cell — a flat wash, which is
// what the first attempt at this drew.
const CLOUD_LATTICE: f32 = 9.0;
// Low, so most of the noise range is cloud, and soft, so the edges are haze
// rather than shapes: `cloud_density = 0.197`, `cloud_softness = 0.902`.
const CLOUD_DENSITY: f32 = 0.197;
const CLOUD_SOFTNESS: f32 = 0.902;
const CLOUD_BRIGHTNESS: f32 = 1.0;
// `cloud_speed = 0.01` along `cloud_direction = (1, 0, -0.855)`, normalized.
const CLOUD_SPEED: f32 = 0.01;
const CLOUD_DRIFT: vec2<f32> = vec2<f32>(0.7601, -0.6499);
// How hard the view direction is flattened before it is projected to the cloud
// plane. 1 would be a plane at infinity; `cloud_flatten = 0.104` barely
// flattens at all, which is what keeps the layer reading as a dome overhead
// rather than as a ceiling.
const CLOUD_FLATTEN: f32 = 0.104;
// The band above the horizon the layer fades in over, so the clouds do not
// stack into a hard line where the dome meets the ground.
const CLOUD_HORIZON_FADE: f32 = 0.1;

// Hoskins-family 2D hash. Not a `fract(sin(...))` one: DESIGN §7 forbids those,
// because "highp" is fp24 on cheap mobile parts and the sine destroys them.
fn sky_hash(p_in: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p_in.x, p_in.y, p_in.x) * 0.1031);
    p3 = p3 + vec3<f32>(dot(p3, p3.yzx + vec3<f32>(33.33)));
    return fract((p3.x + p3.y) * p3.z);
}

// Value noise: bilinear over the lattice with a smoothstep fade, which is the
// cheapest field that has no visible grid in it.
fn sky_value_noise(p: vec2<f32>) -> f32 {
    let cell = floor(p);
    let f = p - cell;
    let w = f * f * (3.0 - 2.0 * f);
    let a = sky_hash(cell);
    let b = sky_hash(cell + vec2<f32>(1.0, 0.0));
    let c = sky_hash(cell + vec2<f32>(0.0, 1.0));
    let d = sky_hash(cell + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, w.x), mix(c, d, w.x), w.y);
}

// Two octaves. Three would be prettier and this is a background.
fn sky_clouds(dir: vec3<f32>) -> f32 {
    let flat_dir = normalize(vec3<f32>(dir.x, dir.y * (1.0 - CLOUD_FLATTEN), dir.z));
    let drift = CLOUD_DRIFT * frame.time.x * CLOUD_SPEED;
    let uv = (flat_dir.xz * CLOUD_SCALE + drift) * CLOUD_LATTICE;
    let n = sky_value_noise(uv) * 0.65 + sky_value_noise(uv * 2.17 + vec2<f32>(11.3, 5.7)) * 0.35;
    var alpha = smoothstep(CLOUD_DENSITY, CLOUD_DENSITY + CLOUD_SOFTNESS, n);
    alpha = alpha * smoothstep(0.0, CLOUD_HORIZON_FADE, dir.y);
    return alpha;
}

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

    // Both passes below are skipped outright at zero rather than multiplied by
    // it. That is not a micro-optimization: it is what makes "a scene that says
    // nothing about weather draws exactly the gradient it always did" a fact
    // about the emitted bytes rather than a hope about floating-point identity.
    let cover = frame.sky_params.x;
    if (cover > 0.0) {
        color = mix(color, vec3<f32>(CLOUD_BRIGHTNESS), sky_clouds(dir) * cover);
    }

    let sun_size = frame.sky_params.y;
    if (sun_size > 0.0) {
        // The original's disk: a smoothstep on `dot(view, light)`, blurred over
        // a fixed fraction of the size so a big sun keeps a proportionate edge.
        let blur = sun_size * 0.5;
        let d = dot(dir, normalize(frame.light_dir.xyz));
        let disk = smoothstep(1.0 - sun_size - blur, 1.0 - sun_size, d);
        // Above the horizon only, exactly as `step(0.0, EYEDIR.y)` does: a sun
        // shining up out of the ground half of the gradient reads as a bug.
        color = mix(color, frame.light_color.rgb, disk * step(0.0, dir.y));
    }

    return vec4<f32>(color, 1.0);
}
