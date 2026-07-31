//! Mesh operations. Each is a pure `fn(MeshData, ...) -> MeshData` so they
//! compose (the fluent methods on `MeshData` are thin sugar over these) and map
//! directly onto editor node-graph nodes later.

use std::collections::HashMap;

use glam::{Mat4, Vec3};

use super::{face_cross, normal_matrix, MeshData, DEGENERATE_AREA_SQ};

/// Apply an affine transform: positions by `m`, normals by its inverse-transpose
/// (so non-uniform scale stays correct), then renormalized.
pub fn transform(mut mesh: MeshData, m: Mat4) -> MeshData {
    let nm = normal_matrix(m);
    for p in &mut mesh.positions {
        *p = m.transform_point3(*p);
    }
    for n in &mut mesh.normals {
        *n = (nm * *n).normalize_or_zero();
    }
    mesh
}

/// Concatenate `b` onto `a` (additive union — no cutting). Missing attributes on
/// either side are filled with defaults so the result stays consistent.
pub fn merge(mut a: MeshData, mut b: MeshData) -> MeshData {
    fill_defaults(&mut a);
    fill_defaults(&mut b);
    let base = a.positions.len() as u32;
    a.positions.append(&mut b.positions);
    a.normals.append(&mut b.normals);
    a.uvs.append(&mut b.uvs);
    a.colors.append(&mut b.colors);
    a.indices.extend(b.indices.iter().map(|i| i + base));
    a
}

/// Set every vertex color.
pub fn set_color(mut mesh: MeshData, color: Vec3) -> MeshData {
    mesh.colors = vec![color; mesh.positions.len()];
    mesh
}

/// Ensure normals/uvs/colors are present (fill with sane defaults if empty).
pub fn fill_defaults(mesh: &mut MeshData) {
    let n = mesh.positions.len();
    if mesh.normals.is_empty() {
        mesh.normals = vec![Vec3::Y; n];
    }
    if mesh.uvs.is_empty() {
        mesh.uvs = vec![glam::Vec2::ZERO; n];
    }
    if mesh.colors.is_empty() {
        mesh.colors = vec![Vec3::ONE; n];
    }
}

/// Faceted normals: expand to one vertex per triangle corner, each face getting
/// its own geometric normal. Triples vertex count; no shared vertices.
pub fn flat_normals(mut mesh: MeshData) -> MeshData {
    fill_defaults(&mut mesh);
    let tri_count = mesh.indices.len() / 3;
    let mut out = MeshData {
        positions: Vec::with_capacity(tri_count * 3),
        normals: Vec::with_capacity(tri_count * 3),
        uvs: Vec::with_capacity(tri_count * 3),
        colors: Vec::with_capacity(tri_count * 3),
        indices: Vec::with_capacity(tri_count * 3),
    };
    for t in mesh.indices.chunks_exact(3) {
        let (i0, i1, i2) = (t[0] as usize, t[1] as usize, t[2] as usize);
        let raw = face_cross(mesh.positions[i0], mesh.positions[i1], mesh.positions[i2]);
        if raw.length_squared() < DEGENERATE_AREA_SQ {
            continue; // drop degenerate (zero-area) triangles, e.g. sphere poles
        }
        let n = raw.normalize();
        for &i in &[i0, i1, i2] {
            out.indices.push(out.positions.len() as u32);
            out.positions.push(mesh.positions[i]);
            out.normals.push(n);
            out.uvs.push(mesh.uvs[i]);
            out.colors.push(mesh.colors[i]);
        }
    }
    out
}

/// Crease-angle normals. Faces sharing a position blend their normals only when
/// within `crease_degrees` of each other; sharper edges stay hard. Positions are
/// welded for adjacency, then corners are re-welded per unique attribute set so
/// hard edges keep distinct normals.
pub fn creased_normals(mut mesh: MeshData, crease_degrees: f32) -> MeshData {
    fill_defaults(&mut mesh);
    let cos_thresh = crease_degrees.to_radians().cos();

    // Weld positions -> cluster id.
    let mut clusters: HashMap<[i64; 3], u32> = HashMap::new();
    let cluster_of: Vec<u32> = mesh
        .positions
        .iter()
        .map(|p| {
            let key = quantize(*p);
            let next = clusters.len() as u32;
            *clusters.entry(key).or_insert(next)
        })
        .collect();

    // Per-triangle raw (area-weighted) and unit face normals.
    let tris: Vec<[usize; 3]> = mesh
        .indices
        .chunks_exact(3)
        .map(|t| [t[0] as usize, t[1] as usize, t[2] as usize])
        .collect();
    let raw: Vec<Vec3> = tris
        .iter()
        .map(|t| face_cross(mesh.positions[t[0]], mesh.positions[t[1]], mesh.positions[t[2]]))
        .collect();
    let unit: Vec<Vec3> = raw.iter().map(|n| n.normalize_or_zero()).collect();

    // cluster id -> triangles incident to it.
    let mut incident: HashMap<u32, Vec<usize>> = HashMap::new();
    for (ti, t) in tris.iter().enumerate() {
        for &v in t {
            incident.entry(cluster_of[v]).or_default().push(ti);
        }
    }

    // Deindex: each corner gets a normal blended from same-cluster faces within
    // the crease threshold of this face, area-weighted.
    let mut out = MeshData::default();
    let mut weld: HashMap<[i64; 8], u32> = HashMap::new();
    out.indices.reserve(mesh.indices.len());
    for (ti, t) in tris.iter().enumerate() {
        if raw[ti].length_squared() < DEGENERATE_AREA_SQ {
            continue; // drop degenerate triangles (e.g. sphere poles)
        }
        let fn_unit = unit[ti];
        for &v in t {
            let mut acc = Vec3::ZERO;
            for &other in &incident[&cluster_of[v]] {
                if unit[other].dot(fn_unit) >= cos_thresh {
                    acc += raw[other];
                }
            }
            let normal = acc.normalize_or_zero();
            let pos = mesh.positions[v];
            let uv = mesh.uvs[v];
            let col = mesh.colors[v];
            let key = weld_key(pos, normal, uv, col);
            let idx = *weld.entry(key).or_insert_with(|| {
                let id = out.positions.len() as u32;
                out.positions.push(pos);
                out.normals.push(normal);
                out.uvs.push(uv);
                out.colors.push(col);
                id
            });
            out.indices.push(idx);
        }
    }
    out
}

/// Twist about `axis` (through origin): rotation angle grows with distance along
/// the axis. Deforms positions only.
pub fn twist(mut mesh: MeshData, radians_per_unit: f32, axis: Vec3) -> MeshData {
    let axis = axis.normalize_or_zero();
    if axis == Vec3::ZERO {
        return mesh;
    }
    for p in &mut mesh.positions {
        let along = p.dot(axis);
        let rot = glam::Quat::from_axis_angle(axis, along * radians_per_unit);
        *p = rot * *p;
    }
    mesh
}

/// Taper along `axis`: cross-sections scale from `1.0` at the min extent to
/// `factor` at the max. Deforms positions only.
pub fn taper(mut mesh: MeshData, factor: f32, axis: Vec3) -> MeshData {
    let axis = axis.normalize_or_zero();
    if axis == Vec3::ZERO || mesh.positions.is_empty() {
        return mesh;
    }
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for p in &mesh.positions {
        let a = p.dot(axis);
        lo = lo.min(a);
        hi = hi.max(a);
    }
    let span = (hi - lo).max(1e-6);
    for p in &mut mesh.positions {
        let a = p.dot(axis);
        let t = (a - lo) / span; // 0..1
        let s = 1.0 + (factor - 1.0) * t;
        let along = a * axis;
        let radial = *p - along;
        *p = along + radial * s;
    }
    mesh
}

fn quantize(p: Vec3) -> [i64; 3] {
    const GRID: f32 = 1.0e5; // weld within ~1e-5
    [
        (p.x * GRID).round() as i64,
        (p.y * GRID).round() as i64,
        (p.z * GRID).round() as i64,
    ]
}

fn weld_key(pos: Vec3, n: Vec3, uv: glam::Vec2, c: Vec3) -> [i64; 8] {
    let qp = 1.0e5;
    let qn = 1.0e4;
    let qu = 1.0e4;
    let qc = 1.0e3;
    [
        (pos.x * qp).round() as i64,
        (pos.y * qp).round() as i64,
        (pos.z * qp).round() as i64,
        ((n.x * qn).round() as i64) * 100_003 + (n.y * qn).round() as i64,
        (n.z * qn).round() as i64,
        (uv.x * qu).round() as i64 * 100_003 + (uv.y * qu).round() as i64,
        (c.x * qc).round() as i64 * 100_003 + (c.y * qc).round() as i64,
        (c.z * qc).round() as i64,
    ]
}
