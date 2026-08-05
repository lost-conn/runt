// The screen-space UI pass (DESIGN §13's HUD seam, plan D11). See `ui.rs`.
//
// One instanced draw for a whole batch: six shader-generated vertices per quad
// (`vertex_index` → a unit corner, the same no-vertex-buffer trick `sky.wgsl`
// and `blit.wgsl` use) and one `UiQuad` per instance on buffer slot 0. There is
// no mesh, no index buffer and nothing per-quad to bind.
//
// Standalone WGSL like the sky and the blit: no feature consts, its own two
// bind groups (the viewport uniform, and the atlas + sampler), nothing else in
// scope. The sampler is declared non-filtering and the texture non-filterable,
// so the pipeline is valid under `downlevel_webgl2_defaults` with no
// filterable-float capability at all — and so a glyph atlas comes out crisp
// rather than smeared (DESIGN §11).

struct UiFrame {
    // xy — the *surface* size in pixels (the view the host handed `render`, not
    // the render-scale target: the UI is drawn after the blit). zw — its
    // reciprocal, so the vertex shader multiplies rather than divides.
    viewport: vec4<f32>,
};

@group(0) @binding(0) var<uniform> ui: UiFrame;
@group(1) @binding(0) var atlas: texture_2d<f32>;
@group(1) @binding(1) var atlas_sampler: sampler;

struct Quad {
    // x, y, w, h in logical pixels, top-left origin, +Y **down**.
    @location(0) rect: vec4<f32>,
    // u0, v0, u1, v1 in the atlas, or all-negative for a solid quad.
    @location(1) uv: vec4<f32>,
    // Premultiplied RGBA.
    @location(2) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_ui(@builtin(vertex_index) index: u32, quad: Quad) -> VsOut {
    // Two triangles, (0,0)-(1,0)-(0,1) and (0,1)-(1,0)-(1,1). A function-scope
    // `var` rather than a value array so the runtime index is legal WGSL — it
    // translates to a local array in GLSL-ES 3.00, which is what WebGL2 gets.
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
    );
    let corner = corners[index];

    // Logical pixels → NDC. X maps straight; **Y is flipped**, because UI space
    // is top-left origin with Y down (the DOM/HUD convention, and the one a
    // layout is written in) while NDC is centre origin with Y up.
    let px = quad.rect.xy + corner * quad.rect.zw;
    let ndc = vec2<f32>(
        px.x * ui.viewport.z * 2.0 - 1.0,
        1.0 - px.y * ui.viewport.w * 2.0,
    );

    var out: VsOut;
    // z = 0 and no depth attachment at all: the UI pass neither tests nor
    // writes depth, so painter's order is the *only* thing deciding overlap.
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    out.uv = mix(quad.uv.xy, quad.uv.zw, corner);
    out.color = quad.color;
    return out;
}

@fragment
fn fs_ui(in: VsOut) -> @location(0) vec4<f32> {
    // Sampled unconditionally and *then* discarded, rather than inside an `if`:
    // `textureSample` needs uniform control flow, and a solid quad is the
    // common case, so branching around it would be both illegal and pointless.
    // A solid quad's uv is negative at every corner (it interpolates between
    // two negatives), which is the sentinel — see `UiQuad::SOLID`.
    let texel = textureSample(atlas, atlas_sampler, in.uv);
    let modulator = select(texel, vec4<f32>(1.0, 1.0, 1.0, 1.0), in.uv.x < 0.0);
    // Premultiplied all the way through: the atlas texel is premultiplied, the
    // instance colour is premultiplied, and a product of two premultiplied
    // values is premultiplied. Tinting a glyph is therefore a plain multiply
    // with no divide-by-alpha anywhere in the pipeline.
    return modulator * in.color;
}
