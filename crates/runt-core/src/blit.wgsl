// The fullscreen pass: the render-scale blit (DESIGN §11's resolution lever)
// and the phase circle's screen effect, which rides it. See
// `Renderer::render_scaled`.
//
// One fullscreen triangle — the same `vertex_index` trick `sky.wgsl` and
// `bake.wgsl` use, so there is no vertex buffer and no index buffer — sampling
// the internal color target with a **nearest** sampler. Nearest is the whole
// point: a bilinear upscale of a half-resolution frame is a blur, and what that
// feature exists to buy is honest chonky pixels at a quarter of the fragment
// cost.
//
// **Why the screen effect lives here.** Inverting the frame's luminance inside
// the phase circle needs the finished frame as an input, and this is the only
// pass in the engine that has one. Folding it into the material shaders is not
// an option — a fragment cannot read the pixels its neighbours already wrote,
// and the effect has to cover the sky and the gaps between geometry too. So the
// copy and the effect are the same fullscreen fetch: when the circle is off the
// fragment returns the sample untouched, and when it is on it pays a dot
// product and a mix on top of a texture read it was making anyway. The cost of
// the feature is therefore not this shader — it is that a native-resolution
// frame with the circle on now has to be drawn offscreen and copied, which
// `Renderer::render_scaled` explains.
//
// **The HUD is excluded by pass ordering, not by a flag.** `render_scaled`
// encodes the UI pass after this one, straight onto the host's view, so a
// screen-space HUD is never in the texture this samples and can never be
// inverted. That is the original's behaviour (Godot's effect quad sits under
// the CanvasLayer) and it falls out for free.
//
// Standalone WGSL, like `sky.wgsl`: no feature consts, nothing else in scope.
// The sampler is declared non-filtering and the texture non-filterable so the
// pipeline is valid under `downlevel_webgl2_defaults` with no filterable-float
// capability at all.

// Field order must match `runt_core::FrameUniform`, and `shader.wgsl` and
// `sky.wgsl` restate the same block. This pass reads two of the fields —
// `phase` and `viewport` — but the block is one buffer with one layout, so all
// four files move together or the uniform is silently misaligned.
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
    // World → the key light's clip space (the shadow map's matrix). A copy
    // pass reads no light; restated for the one-buffer reason above.
    light_view_proj: mat4x4<f32>,
    // x: shadow map bound. y: constant bias. z: slope bias. w: reserved.
    shadow_params: vec4<f32>,
};

// The frame block keeps its number here for the reason it keeps it everywhere
// else — group 0 is "what this frame is" in `shader.wgsl`, `sky.wgsl` and
// `bake.wgsl`, and a pass that renumbered it would be the one place a reader
// has to check. The source texture takes group 1, which is a hole in the
// material layout and free here.
@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var src: texture_2d<f32>;
@group(1) @binding(1) var src_sampler: sampler;

/// Below this the circle is "off" and this pass is a plain copy.
/// `shader.wgsl` and Godot's `phase_common` use the same 0.001, and the resting
/// state has to be exact or a circle at radius zero tints the whole screen.
const PHASE_MIN_RADIUS: f32 = 0.001;
/// Half-width of the smoothstep at the circle's edge, in NDC-Y units — the same
/// constant, in the same units, that `shader.wgsl` smears its fringe over, so
/// the effect's edge and the geometry's edge are the same edge.
const PHASE_EDGE: f32 = 0.03;
/// Rec. 709 luma, the weights `phase_screen_effect.gdshader` inverts about.
const LUMA: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);
/// How far the inverted colour is pulled towards its own grey. The original's
/// `desaturation = 0.4`: a straight luminance inversion is lurid, and taking
/// most of the way to grey out of it is what makes the inside of the circle
/// read as *another place* rather than as a broken framebuffer.
const PHASE_DESATURATION: f32 = 0.4;

struct BlitOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Corners (0,0), (2,0), (0,2) → NDC (-1,-1), (3,-1), (-1,3): an oversized
// triangle whose intersection with the unit square is exactly the screen.
@vertex
fn vs_blit(@builtin(vertex_index) index: u32) -> BlitOut {
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    let ndc = corner * 2.0 - vec2<f32>(1.0, 1.0);
    var out: BlitOut;
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    // Texture space runs top-down and NDC runs bottom-up, so v is flipped.
    // Getting this backwards renders an upside-down world that still passes
    // every "did anything draw" check, which is why it is spelled out.
    out.uv = vec2<f32>(corner.x, 1.0 - corner.y);
    return out;
}

@fragment
fn fs_blit(in: BlitOut) -> @location(0) vec4<f32> {
    // Alpha comes along for the ride: the internal target is the surface format
    // and the opaque pass wrote 1.0 into it, so the presented frame is opaque
    // whatever the surface's alpha mode does with it.
    let screen = textureSample(src, src_sampler, in.uv);

    // The resting state is an early return rather than a `circle` that happens
    // to be zero: at radius 0 the smoothstep's two edges coincide, and a frame
    // that never leaves the circle off must not depend on what a degenerate
    // smoothstep does on this driver.
    if (frame.phase.z <= PHASE_MIN_RADIUS) {
        return screen;
    }

    // The circle is measured exactly as `shader.wgsl::phase_distance` measures
    // it — NDC with +Y up, X scaled by the aspect ratio — because the two have
    // to be the *same* circle. The material shaders discard against theirs, and
    // an effect whose boundary sat a pixel off would draw a ring of inverted
    // floor around every phase-only surface. The interpolated uv is the same
    // 0..1 the material shaders reach by dividing `@builtin(position)` by
    // `viewport`, so this is scale-independent for the same reason theirs is:
    // this pass runs at the host's resolution while `viewport` describes the
    // (possibly smaller) texture, and only the aspect ratio is read out of it.
    var diff = vec2<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0) - frame.phase.xy;
    diff.x = diff.x * frame.viewport.x * frame.viewport.w;
    let circle = 1.0 - smoothstep(
        frame.phase.z - PHASE_EDGE,
        frame.phase.z + PHASE_EDGE,
        length(diff),
    );

    // `phase_screen_effect.gdshader`, value for value.
    //
    // The inversion is *additive*, not `1 - c`: adding `1 - 2·luma` shifts the
    // whole pixel by however far its luminance is from mid-grey, which keeps
    // the hue — a blue wall stays a blue wall, lit from the other side of the
    // grey axis — where a per-channel complement would turn it orange. That
    // choice is the entire look, so it is copied rather than "cleaned up".
    //
    // Note that `phase.w` is deliberately absent: strength drives the material
    // shaders' edge fringe and nothing else, so the mask alone says how much of
    // the effect a pixel gets. This matches the original, where the effect quad
    // has no strength uniform at all.
    let luma = dot(screen.rgb, LUMA);
    let inverted = screen.rgb + (1.0 - 2.0 * luma);
    let desaturated = mix(inverted, vec3<f32>(dot(inverted, LUMA)), PHASE_DESATURATION);
    return vec4<f32>(mix(screen.rgb, desaturated, circle), screen.a);
}
