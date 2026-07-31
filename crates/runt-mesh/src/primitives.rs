//! Triangle primitives. All are centered on the origin, wound counter-clockwise
//! when viewed from outside (front faces, matching the renderer's back-face
//! culling), with outward normals, UVs in `0..1`, and white vertex color.
//!
//! Segment counts are explicit; scale them per device/LOD tier with `Quality`.

use std::f32::consts::TAU;

use glam::{Vec2, Vec3};

use super::{MeshData, DEGENERATE_AREA_SQ};

const WHITE: Vec3 = Vec3::ONE;

/// Push a quad as two CCW triangles over four already-added vertices
/// (`a,b,c,d` in CCW order).
fn quad(indices: &mut Vec<u32>, a: u32, b: u32, c: u32, d: u32) {
    indices.extend_from_slice(&[a, b, c, a, c, d]);
}

/// Push a CCW triangle only if it has non-zero area — used at sphere poles
/// where a quad collapses to a single fan triangle.
fn tri_checked(m: &mut MeshData, a: u32, b: u32, c: u32) {
    let (pa, pb, pc) = (
        m.positions[a as usize],
        m.positions[b as usize],
        m.positions[c as usize],
    );
    if (pb - pa).cross(pc - pa).length_squared() > DEGENERATE_AREA_SQ {
        m.indices.extend_from_slice(&[a, b, c]);
    }
}

/// Axis-aligned box of the given full dimensions, centered on the origin.
/// Flat-shaded: 24 vertices, per-face normals.
pub fn box3(dims: Vec3) -> MeshData {
    let h = dims * 0.5;
    // (normal, u-axis, v-axis) per face; corners built as center ± u ± v.
    let faces = [
        (Vec3::Z, Vec3::X, Vec3::Y),
        (Vec3::NEG_Z, Vec3::NEG_X, Vec3::Y),
        (Vec3::X, Vec3::NEG_Z, Vec3::Y),
        (Vec3::NEG_X, Vec3::Z, Vec3::Y),
        (Vec3::Y, Vec3::X, Vec3::NEG_Z),
        (Vec3::NEG_Y, Vec3::X, Vec3::Z),
    ];
    let mut m = MeshData::default();
    for (n, u, v) in faces {
        let center = n * h;
        let uh = u * h;
        let vh = v * h;
        let base = m.positions.len() as u32;
        for (corner, uv) in [
            (center - uh - vh, Vec2::new(0.0, 0.0)),
            (center + uh - vh, Vec2::new(1.0, 0.0)),
            (center + uh + vh, Vec2::new(1.0, 1.0)),
            (center - uh + vh, Vec2::new(0.0, 1.0)),
        ] {
            m.positions.push(corner);
            m.normals.push(n);
            m.uvs.push(uv);
            m.colors.push(WHITE);
        }
        quad(&mut m.indices, base, base + 1, base + 2, base + 3);
    }
    m
}

/// Cube of the given edge length.
pub fn cube(size: f32) -> MeshData {
    box3(Vec3::splat(size))
}

/// Flat grid on the XZ plane, normal +Y, centered, `subdivisions` quads per side
/// (so `subdivisions = 1` is a single quad).
pub fn plane(size: Vec2, subdivisions: u32) -> MeshData {
    let n = subdivisions.max(1);
    let mut m = MeshData::default();
    let half = size * 0.5;
    for j in 0..=n {
        for i in 0..=n {
            let fx = i as f32 / n as f32;
            let fz = j as f32 / n as f32;
            m.positions.push(Vec3::new(
                -half.x + fx * size.x,
                0.0,
                -half.y + fz * size.y,
            ));
            m.normals.push(Vec3::Y);
            m.uvs.push(Vec2::new(fx, fz));
            m.colors.push(WHITE);
        }
    }
    let stride = n + 1;
    for j in 0..n {
        for i in 0..n {
            let a = j * stride + i;
            let b = j * stride + i + 1;
            let c = (j + 1) * stride + i + 1;
            let d = (j + 1) * stride + i;
            // CCW seen from +Y (above).
            quad(&mut m.indices, a, d, c, b);
        }
    }
    m
}

/// Cylinder along the Y axis, centered, with smooth side normals and flat caps.
pub fn cylinder(radius: f32, height: f32, segments: u32) -> MeshData {
    let seg = segments.max(3);
    let hy = height * 0.5;
    let mut m = MeshData::default();

    // Side: two rings of `seg+1` vertices (duplicated seam for clean UVs).
    let side_base = m.positions.len() as u32;
    for i in 0..=seg {
        let a = i as f32 / seg as f32 * TAU;
        let (s, c) = a.sin_cos();
        let dir = Vec3::new(c, 0.0, s);
        let u = i as f32 / seg as f32;
        m.positions.push(dir * radius + Vec3::new(0.0, -hy, 0.0));
        m.normals.push(dir);
        m.uvs.push(Vec2::new(u, 0.0));
        m.colors.push(WHITE);
        m.positions.push(dir * radius + Vec3::new(0.0, hy, 0.0));
        m.normals.push(dir);
        m.uvs.push(Vec2::new(u, 1.0));
        m.colors.push(WHITE);
    }
    for i in 0..seg {
        let b = side_base + i * 2; // b=bottom_i, b+1=top_i, b+2=bottom_i+1, b+3=top_i+1
        // bottom_i, top_i, top_i+1, bottom_i+1 -> CCW outward
        quad(&mut m.indices, b, b + 1, b + 3, b + 2);
    }

    cap(&mut m, radius, hy, seg, true);
    cap(&mut m, radius, -hy, seg, false);
    m
}

/// A flat disc cap at height `y`. `up = true` faces +Y, else -Y.
fn cap(m: &mut MeshData, radius: f32, y: f32, seg: u32, up: bool) {
    let normal = if up { Vec3::Y } else { Vec3::NEG_Y };
    let center = m.positions.len() as u32;
    m.positions.push(Vec3::new(0.0, y, 0.0));
    m.normals.push(normal);
    m.uvs.push(Vec2::new(0.5, 0.5));
    m.colors.push(WHITE);
    let ring = m.positions.len() as u32;
    for i in 0..=seg {
        let a = i as f32 / seg as f32 * TAU;
        let (s, c) = a.sin_cos();
        m.positions.push(Vec3::new(c * radius, y, s * radius));
        m.normals.push(normal);
        m.uvs.push(Vec2::new(c * 0.5 + 0.5, s * 0.5 + 0.5));
        m.colors.push(WHITE);
    }
    for i in 0..seg {
        if up {
            m.indices.extend_from_slice(&[center, ring + i + 1, ring + i]);
        } else {
            m.indices.extend_from_slice(&[center, ring + i, ring + i + 1]);
        }
    }
}

/// Cone along Y: base at `-height/2`, apex at `+height/2`. Smooth side normals.
pub fn cone(radius: f32, height: f32, segments: u32) -> MeshData {
    let seg = segments.max(3);
    let hy = height * 0.5;
    let mut m = MeshData::default();
    let apex = Vec3::new(0.0, hy, 0.0);
    // Side normal tilts by the slope; slant length normalizes it.
    let slant = (radius * radius + height * height).sqrt();
    let ny = radius / slant;
    let nr = height / slant;

    let base = m.positions.len() as u32;
    for i in 0..=seg {
        let a = i as f32 / seg as f32 * TAU;
        let (s, c) = a.sin_cos();
        let dir = Vec3::new(c, 0.0, s);
        let u = i as f32 / seg as f32;
        // base rim vertex
        m.positions.push(dir * radius + Vec3::new(0.0, -hy, 0.0));
        m.normals.push((dir * nr + Vec3::Y * ny).normalize());
        m.uvs.push(Vec2::new(u, 0.0));
        m.colors.push(WHITE);
        // apex vertex (duplicated per segment for its own normal/uv)
        m.positions.push(apex);
        m.normals.push((dir * nr + Vec3::Y * ny).normalize());
        m.uvs.push(Vec2::new(u, 1.0));
        m.colors.push(WHITE);
    }
    for i in 0..seg {
        let b = base + i * 2; // b=rim_i, b+1=apex_i, b+2=rim_i+1
        // rim_i, apex_i, rim_i+1 -> CCW outward
        m.indices.extend_from_slice(&[b, b + 1, b + 2]);
    }
    cap(&mut m, radius, -hy, seg, false);
    m
}

/// UV sphere, `rings` latitudinal bands and `sectors` longitudinal segments.
/// Smooth normals (position / radius).
pub fn uv_sphere(radius: f32, rings: u32, sectors: u32) -> MeshData {
    let rings = rings.max(2);
    let sectors = sectors.max(3);
    let mut m = MeshData::default();
    for r in 0..=rings {
        let v = r as f32 / rings as f32;
        let phi = v * std::f32::consts::PI; // 0..PI, pole to pole
        let (sp, cp) = phi.sin_cos();
        for s in 0..=sectors {
            let u = s as f32 / sectors as f32;
            let theta = u * TAU;
            let (st, ct) = theta.sin_cos();
            let dir = Vec3::new(sp * ct, cp, sp * st);
            m.positions.push(dir * radius);
            m.normals.push(dir);
            m.uvs.push(Vec2::new(u, v));
            m.colors.push(WHITE);
        }
    }
    let stride = sectors + 1;
    for r in 0..rings {
        for s in 0..sectors {
            let a = r * stride + s;
            let b = r * stride + s + 1;
            let c = (r + 1) * stride + s + 1;
            let d = (r + 1) * stride + s;
            // Wound CCW outward (phi increases downward from +Y pole). At the
            // poles one triangle of each quad collapses, so emit checked.
            tri_checked(&mut m, a, b, c);
            tri_checked(&mut m, a, c, d);
        }
    }
    m
}

/// Torus in the XZ plane. `major` is the ring radius, `minor` the tube radius.
pub fn torus(major: f32, minor: f32, major_seg: u32, minor_seg: u32) -> MeshData {
    let maj = major_seg.max(3);
    let min = minor_seg.max(3);
    let mut m = MeshData::default();
    for i in 0..=maj {
        let u = i as f32 / maj as f32;
        let a = u * TAU;
        let (sa, ca) = a.sin_cos();
        let center = Vec3::new(ca * major, 0.0, sa * major);
        for j in 0..=min {
            let v = j as f32 / min as f32;
            let b = v * TAU;
            let (sb, cb) = b.sin_cos();
            // outward radial in the tube's cross-section
            let dir = Vec3::new(ca * cb, sb, sa * cb);
            m.positions.push(center + dir * minor);
            m.normals.push(dir);
            m.uvs.push(Vec2::new(u, v));
            m.colors.push(WHITE);
        }
    }
    let stride = min + 1;
    for i in 0..maj {
        for j in 0..min {
            let a = i * stride + j;
            let b = i * stride + j + 1;
            let c = (i + 1) * stride + j + 1;
            let d = (i + 1) * stride + j;
            quad(&mut m.indices, a, b, c, d);
        }
    }
    m
}
