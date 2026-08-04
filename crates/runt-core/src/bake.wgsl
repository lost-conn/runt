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
    // x: normal mode, y: ramp stop count, zw: unused.
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
        let cs = cellular(p, lattice, ret, jitter, period);

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
                dir = d / dlen;
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

// Contrast, brightness, clamp — in the original's order.
fn postprocess(v: f32) -> f32 {
    let n = clamp((v - 0.5) * bake.shape.y + 0.5, 0.0, 1.0);
    return clamp(n * bake.shape.z, 0.0, 1.0);
}

// The gradient ramp. Linear between stops, held flat outside the ends —
// Godot's `GradientTexture1D` semantics, which is what the authored ramps were
// drawn against.
fn ramp_at(t_in: f32) -> vec3<f32> {
    let t = clamp(t_in, 0.0, 1.0);
    let count = bake.counts.y;
    if (count == 0u) {
        return vec3<f32>(t, t, t);
    }
    if (t <= bake.ramp[0].w) {
        return bake.ramp[0].xyz;
    }
    for (var i = 1u; i < count; i = i + 1u) {
        let a = bake.ramp[i - 1u];
        let b = bake.ramp[i];
        if (t <= b.w) {
            let span = b.w - a.w;
            var f = 0.0;
            if (span > 1.0e-6) {
                f = (t - a.w) / span;
            }
            return mix(a.xyz, b.xyz, f);
        }
    }
    return bake.ramp[count - 1u].xyz;
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
