//! Triangle primitives. All are centered on the origin, wound counter-clockwise
//! when viewed from outside (front faces, matching the renderer's back-face
//! culling), with outward normals, UVs in `0..1`, and white vertex color.
//!
//! Segment counts are explicit; scale them per device/LOD tier with `Quality`.

use std::f32::consts::TAU;

use glam::{Vec2, Vec3};

use super::{MeshData, Quality, DEGENERATE_AREA_SQ};

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

/// Base radial segments around a capsule's Y axis, before [`Quality`] scaling.
const CAPSULE_RADIAL_SEGMENTS: u32 = 16;
/// Base latitude bands per hemispherical cap, before [`Quality`] scaling.
const CAPSULE_CAP_RINGS: u32 = 4;

/// Capsule along the Y axis, centered: a cylinder wall with a hemispherical cap
/// welded to each end.
///
/// `height` is the **total** cap-to-cap extent, Godot's `CapsuleMesh`
/// convention, so the cylindrical mid-section is `height - 2 * radius` and
/// `height == 2 * radius` is a sphere. A shorter `height` is a caller bug rather
/// than a shape; it is clamped up to `2 * radius` (debug builds assert) so a bad
/// number produces a sphere instead of a self-intersecting mesh.
///
/// Caps are UV-sphere style and share their equator ring with the wall, so the
/// surface is one welded strip from pole to pole: normals are analytic and
/// already smooth across both seams, and no `smooth_normals` pass is wanted.
/// UVs are cylindrical — `u` around, `v` along the profile by **arc length**, so
/// a cap covers the fraction of the texture its surface actually occupies.
///
/// Segment counts scale with `quality`: [`CAPSULE_RADIAL_SEGMENTS`] around and
/// [`CAPSULE_CAP_RINGS`] bands per cap.
pub fn capsule(radius: f32, height: f32, quality: Quality) -> MeshData {
    debug_assert!(
        height >= 2.0 * radius,
        "capsule height {height} is shorter than its own caps (2 * radius = {})",
        2.0 * radius
    );
    let radius = radius.max(0.0);
    let height = height.max(2.0 * radius);
    let seg = quality.segs(CAPSULE_RADIAL_SEGMENTS, 3);
    let cap = quality.segs(CAPSULE_CAP_RINGS, 1);

    // Half the cylindrical mid-section: the Y the cap centers sit at.
    let hc = (height - 2.0 * radius) * 0.5;
    // The profile, measured along the surface: a quarter circle, the wall, a
    // quarter circle. `v` is this normalized, so nothing is stretched.
    let quarter = std::f32::consts::FRAC_PI_2 * radius;
    let profile = 2.0 * quarter + (height - 2.0 * radius);
    let along = |d: f32| if profile > 0.0 { d / profile } else { 0.0 };

    // Rows from the +Y pole down: `cap+1` on the top hemisphere (the last of
    // which is the wall's top ring, where the cap normal is already horizontal)
    // and `cap+1` on the bottom, whose first is the wall's bottom ring.
    let mut m = MeshData::default();
    let row = |phi: f32, center_y: f32, v: f32, m: &mut MeshData| {
        let (sp, cp) = phi.sin_cos();
        for s in 0..=seg {
            let u = s as f32 / seg as f32;
            let (st, ct) = (u * TAU).sin_cos();
            let dir = Vec3::new(sp * ct, cp, sp * st);
            m.positions.push(dir * radius + Vec3::new(0.0, center_y, 0.0));
            m.normals.push(dir);
            m.uvs.push(Vec2::new(u, v));
            m.colors.push(WHITE);
        }
    };
    for i in 0..=cap {
        let phi = i as f32 / cap as f32 * std::f32::consts::FRAC_PI_2;
        row(phi, hc, along(phi * radius), &mut m);
    }
    for j in 0..=cap {
        let phi = std::f32::consts::FRAC_PI_2 * (1.0 + j as f32 / cap as f32);
        let arc = quarter + (height - 2.0 * radius) + (phi - std::f32::consts::FRAC_PI_2) * radius;
        row(phi, -hc, along(arc), &mut m);
    }

    let stride = seg + 1;
    let rows = 2 * cap + 1;
    for r in 0..rows {
        for s in 0..seg {
            let a = r * stride + s;
            let b = r * stride + s + 1;
            let c = (r + 1) * stride + s + 1;
            let d = (r + 1) * stride + s;
            // Checked for the same two reasons `uv_sphere` checks: a quad
            // collapses to one triangle at each pole, and the whole wall band
            // collapses when `height == 2 * radius`.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Quality;

    /// A capsule's every vertex is `radius` from the nearer cap center, which is
    /// the shape's whole definition — and its normal points straight out from
    /// that center.
    fn assert_on_the_surface(m: &MeshData, radius: f32, height: f32) {
        let hc = (height - 2.0 * radius) * 0.5;
        for (i, &p) in m.positions.iter().enumerate() {
            let center = Vec3::new(0.0, p.y.clamp(-hc, hc), 0.0);
            let radial = p - center;
            assert!(
                (radial.length() - radius).abs() < 1e-4,
                "vertex {i} at {p:?} is {} from its cap center, not {radius}",
                radial.length()
            );
            let expected = radial.normalize();
            assert!(
                m.normals[i].abs_diff_eq(expected, 1e-4),
                "vertex {i} normal {:?} != outward {expected:?}",
                m.normals[i]
            );
        }
    }

    #[test]
    fn capsule_is_well_formed_and_the_right_size() {
        let m = capsule(0.35, 1.4, Quality::FULL);
        m.validate();
        assert_eq!(m.normals.len(), m.positions.len());
        assert_eq!(m.uvs.len(), m.positions.len());
        assert_eq!(m.colors.len(), m.positions.len());

        // (2 * cap + 2) rows of (seg + 1), welded at both equators.
        let (seg, cap) = (CAPSULE_RADIAL_SEGMENTS, CAPSULE_CAP_RINGS);
        assert_eq!(m.vertex_count() as u32, (2 * cap + 2) * (seg + 1));
        // Every band is two triangles per segment except the two polar ones,
        // where half of each quad collapses.
        assert_eq!(m.triangle_count() as u32, (2 * (2 * cap + 1) - 2) * seg);

        let (min, max) = m.bounds().unwrap();
        assert!((max.y - 0.7).abs() < 1e-5, "half the total height: {max:?}");
        assert!((min.y + 0.7).abs() < 1e-5, "{min:?}");
        assert!((max.x - 0.35).abs() < 1e-5, "widest at the radius: {max:?}");
        assert!((max.z - 0.35).abs() < 1e-5, "{max:?}");
        assert_on_the_surface(&m, 0.35, 1.4);
    }

    #[test]
    fn capsule_winds_outward_and_has_no_open_edge() {
        let m = capsule(0.4, 1.6, Quality::FULL);
        // Convex and centered, so an outward face's geometric normal agrees
        // with its own centroid.
        for t in m.indices.chunks_exact(3) {
            let (a, b, c) = (
                m.positions[t[0] as usize],
                m.positions[t[1] as usize],
                m.positions[t[2] as usize],
            );
            let centroid = (a + b + c) / 3.0;
            assert!(
                (b - a).cross(c - a).dot(centroid) > 0.0,
                "face winds inward at {centroid:?}"
            );
        }

        // Watertight up to the duplicated UV seam: every directed edge, taken
        // between *positions* rather than indices, has exactly one opposite.
        let key = |p: Vec3| {
            let q = |f: f32| (f * 4096.0).round() as i32;
            (q(p.x), q(p.y), q(p.z))
        };
        let mut edges = std::collections::HashMap::new();
        for t in m.indices.chunks_exact(3) {
            for k in 0..3 {
                let a = key(m.positions[t[k] as usize]);
                let b = key(m.positions[t[(k + 1) % 3] as usize]);
                *edges.entry((a, b)).or_insert(0i32) += 1;
            }
        }
        for ((a, b), count) in &edges {
            assert_eq!(*count, 1, "edge {a:?}->{b:?} used {count} times");
            assert_eq!(
                edges.get(&(*b, *a)),
                Some(&1),
                "edge {a:?}->{b:?} has no opposite — the surface is open"
            );
        }
    }

    #[test]
    fn a_capsule_as_tall_as_its_caps_is_a_sphere() {
        // The wall band collapses to zero height; `tri_checked` drops it, and
        // what is left has exactly a sphere's bounds and a sphere's surface.
        let m = capsule(0.5, 1.0, Quality::FULL);
        m.validate();
        let (min, max) = m.bounds().unwrap();
        assert!((max - Vec3::splat(0.5)).length() < 1e-5, "{max:?}");
        assert!((min + Vec3::splat(0.5)).length() < 1e-5, "{min:?}");
        for &p in &m.positions {
            assert!((p.length() - 0.5).abs() < 1e-4, "{p:?} is not on the sphere");
        }
        // Two polar bands lose half their quads; the collapsed wall loses all.
        let (seg, cap) = (CAPSULE_RADIAL_SEGMENTS, CAPSULE_CAP_RINGS);
        assert_eq!(m.triangle_count() as u32, (2 * (2 * cap) - 2) * seg);
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "shorter than its own caps"))]
    fn a_capsule_shorter_than_its_caps_clamps_instead_of_folding() {
        // Debug builds refuse the number; release clamps it. Either way what a
        // shipping build draws is the sphere, not an inside-out pinch.
        let m = capsule(0.5, 0.2, Quality::FULL);
        let (min, max) = m.bounds().unwrap();
        assert!((max - Vec3::splat(0.5)).length() < 1e-5, "{max:?}");
        assert!((min + Vec3::splat(0.5)).length() < 1e-5, "{min:?}");
    }

    #[test]
    fn capsule_quality_scales_both_axes() {
        let half = capsule(0.35, 1.4, Quality(0.5));
        half.validate();
        let (seg, cap) = (
            Quality(0.5).segs(CAPSULE_RADIAL_SEGMENTS, 3),
            Quality(0.5).segs(CAPSULE_CAP_RINGS, 1),
        );
        assert_eq!((seg, cap), (8, 2), "the scaled counts the mesh is built from");
        assert_eq!(half.vertex_count() as u32, (2 * cap + 2) * (seg + 1));
        assert!(half.vertex_count() < capsule(0.35, 1.4, Quality::FULL).vertex_count());
        assert_on_the_surface(&half, 0.35, 1.4);
    }

    #[test]
    fn capsule_uvs_run_the_profile_by_arc_length() {
        let m = capsule(0.5, 3.0, Quality::FULL);
        // A 2 m wall between two quarter-circles of 0.5 * pi/2: the wall is
        // 2 / (2 + pi/2) of the profile, and the caps split the rest evenly.
        let profile = 2.0 + std::f32::consts::PI * 0.5;
        let cap_share = (std::f32::consts::FRAC_PI_2 * 0.5) / profile;
        let mut v: Vec<f32> = m.uvs.iter().map(|uv| uv.y).collect();
        v.sort_by(f32::total_cmp);
        assert!((v[0]).abs() < 1e-6, "the +Y pole is v = 0");
        assert!((v[v.len() - 1] - 1.0).abs() < 1e-6, "the -Y pole is v = 1");
        // The wall's top ring: the only v that should sit at the cap's share.
        let equator = m
            .uvs
            .iter()
            .zip(&m.positions)
            .find(|(_, p)| (p.y - 1.0).abs() < 1e-5)
            .expect("the wall's top ring is at y = height/2 - radius");
        assert!(
            (equator.0.y - cap_share).abs() < 1e-5,
            "v {} at the top of the wall, expected {cap_share}",
            equator.0.y
        );
        for uv in &m.uvs {
            assert!((0.0..=1.0).contains(&uv.x) && (0.0..=1.0).contains(&uv.y), "{uv:?}");
        }
    }
}
