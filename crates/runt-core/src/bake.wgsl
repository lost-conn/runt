// The texture bake pass (DESIGN §7's baked path).
//
// > *Baked (baseline): rendered once to an RGBA8 texture at tier-scaled
// > resolution at load time. This is a fragment pass, not compute — it works
// > fully within WebGL2 limits.* — DESIGN §7
//
// NOT standalone WGSL: `noise.wgsl` is prepended by
// `runt_core::bake::bake_shader_source`. Same trick as the material variant
// preprocessor, one level simpler — a concatenation rather than a feature key,
// because a bake has exactly one shape.
//
// Two entry points share one `bake_sample`: `fs_albedo` writes the ramp-mapped
// colour, `fs_normal` writes the packed Voronoi-boundary normal. They are
// separate passes rather than one MRT pass on purpose — two single-target
// attachments are unambiguously inside `downlevel_webgl2_defaults`, and the
// bake is load-time work that the content cache mostly skips anyway.
//
// The derivative the normal accumulates against is an **analytic texel delta**
// (`bake.normal.z`), not `dFdx`. At bake time there is no screen and no camera;
// there is a tile and a known step across it. See
// `runt_core::texture::NORMAL_REFERENCE_TEXELS` for why that step is a fixed
// fraction of the tile rather than one real texel.

struct Bake {
    // x: lattice, y: cell return type, z: fractal, w: octave count.
    mode: vec4<u32>,
    // x: normal mode, y: ramp stop count, z: noise kind, w: radial sectors.
    counts: vec4<u32>,
    // x: jitter, y: contrast, z: brightness, w: ridged weighted strength.
    shape: vec4<f32>,
    // x: edge width, y: normal strength, z: analytic texel delta, w: unused.
    normal: vec4<f32>,
    // xyz: seed offset, w: unused.
    seed: vec4<f32>,
    // Per octave: x span (cells across the tile), y frequency relative to
    // octave 0, z amplitude, w distance-LOD weight. Planned on the CPU by
    // `TextureSpec::octave_plan` so the shader has no `pow` and no rounding
    // rule it could disagree with the CPU twin about.
    octaves: array<vec4<f32>, 8>,
    // xyz: colour, w: offset along the ramp.
    ramp: array<vec4<f32>, 8>,
};
@group(0) @binding(0) var<uniform> bake: Bake;

struct BakeOut {
    @builtin(position) clip: vec4<f32>,
    // Tile coordinates, 0..1 across the target with (0,0) at the first texel's
    // row. The tile is exactly periodic, so uv 0 and uv 1 are the same texel.
    @location(0) uv: vec2<f32>,
};

// One oversized triangle, from `vertex_index` alone — the same trick the sky
// pass uses, and for the same reason: no vertex buffer, no index buffer.
@vertex
fn vs_bake(@builtin(vertex_index) index: u32) -> BakeOut {
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    let ndc = corner * 2.0 - vec2<f32>(1.0, 1.0);
    var out: BakeOut;
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    // NDC +Y is up, texel row 0 is the top: flip so uv.y grows with the row
    // index and the CPU twin's `(j + 0.5) / n` lands on the same sample.
    out.uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return out;
}

struct BakeResult {
    value: f32,
    dndx: f32,
    dndy: f32,
};

// The whole texture, at one tile coordinate: the fractal scalar plus the
// accumulated Voronoi-boundary gradient. Both come out of the *same* octave
// loop — the feature points the normal needs are already in hand from the
// sample that produced the scalar, so normals cost no extra noise evaluations.
fn bake_sample(uv: vec2<f32>) -> BakeResult {
    let lattice = bake.mode.x;
    let ret = bake.mode.y;
    let fractal = bake.mode.z;
    let count = bake.mode.w;
    let normal_mode = bake.counts.x;
    let kind = bake.counts.z;
    let sectors = f32(bake.counts.w);

    let jitter = bake.shape.x;
    let weighted_strength = bake.shape.w;
    let edge_width = max(bake.normal.x, 1.0e-6);
    let strength = bake.normal.y;
    let delta_texel = bake.normal.z;
    let offset = bake.seed.xyz;

    var accum = fbm_new();
    var dndx = 0.0;
    var dndy = 0.0;

    for (var i = 0u; i < count; i = i + 1u) {
        let o = bake.octaves[i];
        let span = o.x;
        let freq = o.y;
        let amplitude = o.z;
        let w = o.w;

        // The tile maps onto exactly `span` lattice cells, so wrapping the cell
        // index by `span` makes this octave seamless. The seed offset is a
        // constant shift and cannot break that.
        let p = vec3<f32>(
            uv.x * span + offset.x * freq,
            uv.y * span + offset.y * freq,
            offset.z * freq,
        );
        let period = vec3<f32>(span, span, 0.0);
        let cs = noise_field(p, kind, lattice, ret, jitter, sectors * freq, period);

        accum = fbm_push(accum, cs.value, amplitude, w, fractal, weighted_strength);

        if (normal_mode != NORMAL_NONE) {
            // Edge magnitude peaks where the two nearest feature points are
            // equidistant (a cell boundary) and falls to zero well inside a
            // cell, so interiors stay flat.
            let edge_mag = 1.0 - smoothstep(0.0, edge_width, cs.d2 - cs.d1);
            var d = p - cs.f1;                       // to-point: radial
            if (normal_mode == NORMAL_TO_EDGE) {
                d = cs.f2 - cs.f1;                   // to-edge: across the boundary
            }
            let dlen = length(d);
            var dir = vec3<f32>(0.0);
            if (dlen > 1.0e-4) {
                // Negated: `d` points away from the nearest feature point, but
                // dndx/dndy are a height gradient and cell centres are the
                // relief's high points, so uphill is *toward* the feature
                // point. See `TextureSpec::sample_at`'s matching comment in
                // texture.rs for the full argument — `fs_normal` below packs
                // the result the same way `packed_normal_at` does.
                dir = -d / dlen;
            }
            // Derivative-of-fBm: an octave's contribution scales with
            // amplitude * frequency, exactly as in the original.
            let nw = amplitude * freq * edge_mag * w * strength;
            dndx = dndx + nw * dir.x * delta_texel;
            dndy = dndy + nw * dir.y * delta_texel;
        }
    }

    var r: BakeResult;
    r.value = fbm_finish(accum);
    r.dndx = dndx;
    r.dndy = dndy;
    return r;
}

// Contrast/brightness and the ramp are `noise.wgsl`'s, fed from this pass's
// uniform. The live material variant calls the same two functions with
// `@group(2)`'s numbers, which is what makes DESIGN §7's "one source, two
// modes" true of the *colour* and not only of the noise.
fn postprocess(v: f32) -> f32 {
    return tex_postprocess(v, bake.shape.y, bake.shape.z);
}

fn ramp_at(t_in: f32) -> vec3<f32> {
    return ramp_lookup(bake.ramp, bake.counts.y, t_in);
}

@fragment
fn fs_albedo(in: BakeOut) -> @location(0) vec4<f32> {
    let r = bake_sample(in.uv);
    return vec4<f32>(ramp_at(postprocess(r.value)), 1.0);
}

@fragment
fn fs_normal(in: BakeOut) -> @location(0) vec4<f32> {
    let r = bake_sample(in.uv);
    var v = vec3<f32>(-r.dndx, -r.dndy, 0.5);
    let len = length(v);
    if (len > 0.0) {
        v = v / len;
    }
    // Packed to [0,1] the way every tangent-space normal map is; the material
    // shader unpacks with `* 2 - 1`.
    return vec4<f32>(v * 0.5 + vec3<f32>(0.5), 1.0);
}

// ---------------------------------------------------------------------------
// The mip chain
// ---------------------------------------------------------------------------
//
// Level `i+1` is a render pass with level `i` bound as a texture — a downsample
// *chain*, not a compute dispatch, because DESIGN §11's baseline is WebGL2 and
// WebGL2 has neither compute nor storage textures. It is the same shape the
// bake itself uses (one oversized triangle, one colour attachment), pointed at
// a mip view instead of the whole texture.
//
// The filter is an exact 2×2 box: four `textureLoad`s of the source texels the
// destination texel covers, averaged. `textureLoad` (`texelFetch` in GLSL ES
// 3.00, core there) rather than one bilinear tap because a bilinear "box" is
// only as exact as the sampler's filter arithmetic — on a downlevel part that
// is not necessarily fp32 — while four integer fetches are exactly the average
// the CPU can predict, which is what `tests/texture_bake.rs` holds mip 1 to.
//
// A *separate binding* from the bake uniform (`@group(0) @binding(1)`) rather
// than a second shader module: a WGSL module has one global scope, the mip
// pipelines are built against their own bind-group layout, and wgpu validates
// bindings per entry point — so `fs_albedo` never sees this and `fs_mip_*`
// never sees the uniform.
@group(0) @binding(1) var mip_src: texture_2d<f32>;

// The four source texels under one destination texel. `in.clip.xy` is the
// destination fragment centre, and the attachment is the destination mip, so
// truncating it gives the destination texel index directly.
fn mip_quad_origin(clip: vec2<f32>) -> vec2<i32> {
    return vec2<i32>(clip) * 2;
}

@fragment
fn fs_mip_color(in: BakeOut) -> @location(0) vec4<f32> {
    let o = mip_quad_origin(in.clip.xy);
    let a = textureLoad(mip_src, o + vec2<i32>(0, 0), 0);
    let b = textureLoad(mip_src, o + vec2<i32>(1, 0), 0);
    let c = textureLoad(mip_src, o + vec2<i32>(0, 1), 0);
    let d = textureLoad(mip_src, o + vec2<i32>(1, 1), 0);
    return (a + b + c + d) * 0.25;
}

@fragment
fn fs_mip_normal(in: BakeOut) -> @location(0) vec4<f32> {
    let o = mip_quad_origin(in.clip.xy);
    var sum = vec3<f32>(0.0);
    for (var i = 0; i < 4; i = i + 1) {
        let t = o + vec2<i32>(i & 1, i >> 1u);
        sum = sum + (textureLoad(mip_src, t, 0).xyz * 2.0 - vec3<f32>(1.0));
    }
    // Averaging *packed* normals and storing that is the classic bug: four unit
    // vectors that disagree average to something shorter than unit, and once
    // the material shader unpacks it the surface reads as flatter — by a
    // different amount per texel, so the crinkle fades unevenly with distance
    // rather than smoothly. Renormalizing keeps every level a unit normal,
    // which is what the whiteout blend in `shader.wgsl` assumes.
    let len = length(sum);
    var n = vec3<f32>(0.0, 0.0, 1.0);
    if (len > 1.0e-6) {
        n = sum / len;
    }
    return vec4<f32>(n * 0.5 + vec3<f32>(0.5), 1.0);
}
