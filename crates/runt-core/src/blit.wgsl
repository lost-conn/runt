// The render-scale blit (DESIGN §11's resolution lever). See `Renderer::render_scaled`.
//
// One fullscreen triangle — the same `vertex_index` trick `sky.wgsl` and
// `bake.wgsl` use, so there is no vertex buffer and no index buffer — sampling
// the internal color target with a **nearest** sampler. Nearest is the whole
// point: a bilinear upscale of a half-resolution frame is a blur, and what this
// feature exists to buy is honest chonky pixels at a quarter of the fragment
// cost.
//
// Standalone WGSL, like `sky.wgsl`: no feature consts, its own two-entry bind
// group (texture + sampler), nothing else in scope. The sampler is declared
// non-filtering and the texture non-filterable so the pipeline is valid under
// `downlevel_webgl2_defaults` with no filterable-float capability at all.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;

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
    return textureSample(src, src_sampler, in.uv);
}
