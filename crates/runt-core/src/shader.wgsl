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

// The baked procedural texture (DESIGN §7). Present in the pipeline layout for
// every variant — an untextured draw binds a 1×1 white albedo and a 1×1 flat
// normal, so the bit-unset path is provably the pre-texture look rather than
// merely intended to be.
@group(2) @binding(0) var t_albedo: texture_2d<f32>;
@group(2) @binding(1) var t_normal: texture_2d<f32>;
@group(2) @binding(2) var t_sampler: sampler;
// x: world units → tile units, y: triplanar sharpness, z: anti-tiling on/off.
// Params of the *texture*, not the instance: they describe how this bake maps
// onto the world, which two entities wearing the same material must agree on.
struct TexParams {
    config: vec4<f32>,
};
@group(2) @binding(3) var<uniform> tex: TexParams;

struct VSOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    // World position, for the triplanar sample. There is no UV in the texture
    // path at all: the original is a world-space shader and CSG-ish generated
    // geometry has no UV worth trusting, so the world *is* the parameterization.
    @location(2) world_pos: vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
) -> VSOut {
    var out: VSOut;
    let world = inst.model * vec4<f32>(pos, 1.0);
    out.clip = frame.view_proj * world;
    out.world_pos = world.xyz;
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

// ---------------------------------------------------------------------------
// Baked-texture sampling (DESIGN §7)
// ---------------------------------------------------------------------------
//
// Everything below samples with **explicit gradients**, taken once per fragment
// from the un-offset plane UV and handed to every tap.
//
// Three things have to be true at once and only `textureSampleGrad` makes them
// so. (1) The bake carries a full mip chain now, so a tap has to pick a level
// or it shimmers. (2) The anti-tiling offsets are a per-virtual-cell hash: two
// neighbouring pixels of a quad can land in different cells, so the *implicit*
// derivative of `uv + o` jumps by a whole tile at every cell boundary, and an
// implicit-LOD `textureSample` would pick the bottom of the chain there —
// a grid of blurred seams, which is worse than the repetition the trick exists
// to hide. (3) `dpdx`/`dpdy` are implicit-derivative built-ins and WGSL only
// allows them in uniform control flow, so they are taken in `fs_main` and
// passed down rather than computed inside the loop.
//
// Downlevel: WGSL `textureSampleGrad` lowers to GLSL `textureGrad`, which is
// core in GLSL ES 3.00 (so, all of WebGL2) and carries no uniformity
// requirement of its own — unlike `texture()`, it is well-defined inside the
// loop. naga's GLSL backend emits it for `SampleLevel::Gradient` and even uses
// it as its *workaround* for backends where `textureLod` misbehaves, so it is
// the better-supported of the two paths, not the riskier one.

// One plane's texture-space derivatives, computed before any offset is applied.
struct Grad {
    dx: vec2<f32>,
    dy: vec2<f32>,
};

fn tap(t: texture_2d<f32>, uv: vec2<f32>, g: Grad) -> vec4<f32> {
    return textureSampleGrad(t, t_sampler, uv, g.dx, g.dy);
}

// A 2D integer-style hash for the anti-tiling offsets. The original used
// `fract(sin(p) * 43758.5453); DESIGN §7 forbids sin-hashes (fp24 "highp" on
// cheap mobile parts destroys them), so this is the same Hoskins family the
// noise library uses.
fn notile_hash2(p_in: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3<f32>(p_in.x, p_in.y, p_in.x) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 = p3 + vec3<f32>(dot(p3, p3.yzx + vec3<f32>(33.33)));
    return fract((p3.xx + p3.yz) * p3.zy);
}

// Inigo Quilez's texture-repetition trick: give each virtual grid cell a random
// offset into the tile and blend the 3×3 neighbourhood with smooth weights, so
// a single bake covering a hillside never reads as wallpaper. ~9 taps.
//
// The tile itself is exactly seamless (see `runt_core::texture`), so this is
// solving *repetition*, not *seams* — two different problems that a blended
// "seamless" texture conflates.
fn notile(t: texture_2d<f32>, uv: vec2<f32>, g: Grad) -> vec4<f32> {
    let cell = floor(uv);
    let f = uv - cell;
    var acc = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var j = -1; j <= 1; j = j + 1) {
        for (var i = -1; i <= 1; i = i + 1) {
            let n = vec2<f32>(f32(i), f32(j));
            let o = notile_hash2(cell + n);
            let r = n - f + o;
            let w = exp(-5.0 * dot(r, r));
            // The *same* gradient for every tap. The offset is a translation in
            // tile space, so it does not change how fast the texture moves
            // under the pixel — only where it is read from.
            acc = acc + w * tap(t, uv + o, g);
            wsum = wsum + w;
        }
    }
    return acc / wsum;
}

fn plane_tap(t: texture_2d<f32>, uv: vec2<f32>, g: Grad, anti_tiling: bool) -> vec4<f32> {
    if (anti_tiling) {
        return notile(t, uv, g);
    }
    return tap(t, uv, g);
}

// Triplanar weights: `pow(abs(N), sharpness)` normalized to sum to 1. The
// degenerate case (a zero normal on broken geometry) falls back to an even
// third rather than dividing by zero.
fn triplanar_blend(n: vec3<f32>, sharpness: f32) -> vec3<f32> {
    let b = pow(abs(n), vec3<f32>(sharpness));
    let s = b.x + b.y + b.z;
    if (s > 1.0e-6) {
        return b / s;
    }
    return vec3<f32>(1.0 / 3.0);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    var n = normalize(in.normal);

    var albedo = inst.base_color.rgb;
    if (F_VERTEX_COLOR) {
        albedo = albedo * in.color;
    }

    if (F_TEXTURE) {
        let world_scale = tex.config.x;
        let sharpness = tex.config.y;
        let anti_tiling = tex.config.z > 0.5;
        let p = in.world_pos * world_scale;
        // The three planes' derivatives, taken here — at the top of the
        // fragment, in uniform control flow — because `dpdx`/`dpdy` may not be
        // called from inside the anti-tiling loop, and because the loop wants
        // the derivative of the *un-offset* UV anyway. `world_scale` is a
        // uniform, so this is the world-space derivative scaled once.
        let g_xy = Grad(dpdx(p.xy), dpdy(p.xy));
        let g_xz = Grad(dpdx(p.xz), dpdy(p.xz));
        let g_yz = Grad(dpdx(p.yz), dpdy(p.yz));
        // The weight-to-plane mapping is the original's: the Z-normal weight
        // drives the XY plane, Y drives XZ, X drives YZ.
        let blend = triplanar_blend(n, sharpness);

        let c_xy = plane_tap(t_albedo, p.xy, g_xy, anti_tiling);
        let c_xz = plane_tap(t_albedo, p.xz, g_xz, anti_tiling);
        let c_yz = plane_tap(t_albedo, p.yz, g_yz, anti_tiling);
        albedo = albedo * (c_xy.rgb * blend.z + c_xz.rgb * blend.y + c_yz.rgb * blend.x);

        if (F_NORMAL_MAP) {
            // Plain taps, no anti-tiling: the crinkle is high-frequency and
            // repetition does not read on it, so it is not worth 3× the taps.
            // Mip-correct all the same — an unmipped normal map is the loudest
            // shimmer on screen, because the lighting amplifies it.
            let n_xy = tap(t_normal, p.xy, g_xy).xy * 2.0 - vec2<f32>(1.0);
            let n_xz = tap(t_normal, p.xz, g_xz).xy * 2.0 - vec2<f32>(1.0);
            let n_yz = tap(t_normal, p.yz, g_yz).xy * 2.0 - vec2<f32>(1.0);
            // "Whiteout" triplanar normal blend: each plane's tangent offsets
            // are applied along that plane's world axes and summed onto the
            // geometric normal. Cheap, stable, and it cannot flip a normal
            // inside out the way a per-plane TBN reconstruction can.
            let perturb = blend.z * vec3<f32>(n_xy.x, n_xy.y, 0.0)
                        + blend.y * vec3<f32>(n_xz.x, 0.0, n_xz.y)
                        + blend.x * vec3<f32>(0.0, n_yz.x, n_yz.y);
            n = normalize(n + perturb);
        }
    }

    // Hemisphere ambient: sky above, ground below, blended on the normal's Y.
    // Cheap, and it separates silhouettes far better than a constant term.
    let hemi = mix(frame.ground_color.rgb, frame.sky_color.rgb, 0.5 + 0.5 * n.y);
    let key = frame.light_color.rgb * max(dot(n, normalize(frame.light_dir.xyz)), 0.0);

    return vec4<f32>((hemi + key) * albedo, inst.base_color.a);
}
