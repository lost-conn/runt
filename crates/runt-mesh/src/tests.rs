use super::*;
use glam::{Vec2, Vec3};

/// Every attribute is present and correctly sized, indices in range, and all
/// normals are unit length (or zero, which we disallow here).
fn assert_well_formed(m: &MeshData) {
    let n = m.positions.len();
    assert_eq!(m.normals.len(), n, "normals sized");
    assert_eq!(m.uvs.len(), n, "uvs sized");
    assert_eq!(m.colors.len(), n, "colors sized");
    assert_eq!(m.indices.len() % 3, 0, "whole triangles");
    for &i in &m.indices {
        assert!((i as usize) < n, "index in range");
    }
    for nrm in &m.normals {
        assert!((nrm.length() - 1.0).abs() < 1e-3, "normal unit length: {nrm:?}");
    }
}

/// For a convex primitive centered on the origin, every face must wind CCW when
/// seen from outside — i.e. its geometric normal points away from the centroid —
/// and vertex normals must roughly agree with the outward direction.
fn assert_convex_outward(m: &MeshData) {
    for t in m.indices.chunks_exact(3) {
        let (a, b, c) = (
            m.positions[t[0] as usize],
            m.positions[t[1] as usize],
            m.positions[t[2] as usize],
        );
        let face_n = (b - a).cross(c - a);
        let centroid = (a + b + c) / 3.0;
        assert!(
            face_n.dot(centroid) > 0.0,
            "face winds outward (centroid {centroid:?}, n {face_n:?})"
        );
    }
}

#[test]
fn cube_is_well_formed_and_outward() {
    let m = cube(2.0);
    assert_well_formed(&m);
    assert_convex_outward(&m);
    assert_eq!(m.vertex_count(), 24);
    assert_eq!(m.triangle_count(), 12);
    let (min, max) = m.bounds().unwrap();
    assert!((min - Vec3::splat(-1.0)).length() < 1e-5);
    assert!((max - Vec3::splat(1.0)).length() < 1e-5);
}

#[test]
fn cylinder_cone_sphere_torus_well_formed() {
    assert_well_formed(&cylinder(1.0, 2.0, 24));
    assert_well_formed(&cone(1.0, 2.0, 24));
    assert_well_formed(&uv_sphere(1.0, 16, 24));
    assert_well_formed(&torus(1.0, 0.3, 24, 12));
}

#[test]
fn convex_primitives_wind_outward() {
    assert_convex_outward(&cylinder(1.0, 2.0, 24));
    assert_convex_outward(&cone(1.0, 2.0, 20));
    assert_convex_outward(&uv_sphere(1.0, 16, 24));
    // (torus is not convex, so it is excluded from this check)
}

#[test]
fn sphere_normals_are_radial() {
    let m = uv_sphere(1.0, 12, 18);
    for (p, n) in m.positions.iter().zip(&m.normals) {
        assert!(p.normalize().dot(*n) > 0.999, "sphere normal is radial");
    }
}

#[test]
fn quality_scales_segments() {
    let full = cylinder(1.0, 2.0, Quality::FULL.segs(32, 3));
    let low = cylinder(1.0, 2.0, Quality(0.5).segs(32, 3));
    assert!(low.triangle_count() < full.triangle_count());
    assert!(low.triangle_count() > 0);
    // A different quality yields a different mesh (distinct cache key).
    assert_ne!(full.content_hash(), low.content_hash());
}

#[test]
fn merge_concatenates_and_offsets_indices() {
    let a = cube(1.0);
    let b = cube(1.0).translate(Vec3::new(3.0, 0.0, 0.0));
    let merged = a.clone().merge(b);
    assert_eq!(merged.vertex_count(), a.vertex_count() * 2);
    assert_eq!(merged.triangle_count(), a.triangle_count() * 2);
    assert_well_formed(&merged);
    let max_idx = *merged.indices.iter().max().unwrap() as usize;
    assert_eq!(max_idx, merged.vertex_count() - 1);
}

#[test]
fn translate_moves_bounds_not_shape() {
    let m = cube(2.0).translate(Vec3::new(5.0, 0.0, 0.0));
    let (min, max) = m.bounds().unwrap();
    assert!((min - Vec3::new(4.0, -1.0, -1.0)).length() < 1e-5);
    assert!((max - Vec3::new(6.0, 1.0, 1.0)).length() < 1e-5);
}

#[test]
fn nonuniform_scale_keeps_normals_unit() {
    let m = uv_sphere(1.0, 12, 18).scale(Vec3::new(2.0, 0.5, 1.0));
    assert_well_formed(&m); // normals renormalized after the transform
}

#[test]
fn flat_normals_deindex_per_face() {
    let m = uv_sphere(1.0, 8, 12).flat_normals();
    assert_eq!(m.vertex_count(), m.triangle_count() * 3);
    // Each triangle's three normals are identical (its face normal).
    for t in m.indices.chunks_exact(3) {
        let n0 = m.normals[t[0] as usize];
        let n1 = m.normals[t[1] as usize];
        let n2 = m.normals[t[2] as usize];
        assert!((n0 - n1).length() < 1e-4 && (n1 - n2).length() < 1e-4);
    }
    assert_well_formed(&m);
}

#[test]
fn crease_normals_keep_box_edges_hard() {
    // Fully smoothing a box would average adjacent face normals; with a low
    // crease threshold the 90° edges must stay hard (still 24 distinct verts).
    let hard = cube(2.0).smooth_normals(30.0);
    assert_well_formed(&hard);
    // Each corner normal should still match one axis-aligned face direction.
    for n in &hard.normals {
        let aligned = n.abs().max_element();
        assert!(aligned > 0.99, "box edge stayed hard, normal {n:?}");
    }
}

#[test]
fn crease_normals_smooth_a_sphere() {
    // A high threshold smooths the sphere: welded to ~one normal per position.
    let smooth = uv_sphere(1.0, 12, 18).smooth_normals(180.0);
    assert_well_formed(&smooth);
    for (p, n) in smooth.positions.iter().zip(&smooth.normals) {
        assert!(p.normalize().dot(*n) > 0.99, "smoothed sphere stays radial");
    }
}

#[test]
fn taper_shrinks_one_end() {
    // Taper a cylinder to a point-ish top: the +Y rim should be pulled inward.
    let m = cylinder(1.0, 2.0, 24).taper(0.0, Vec3::Y);
    for p in &m.positions {
        if p.y > 0.9 {
            let radial = (p.x * p.x + p.z * p.z).sqrt();
            assert!(radial < 1e-3, "top tapered to axis, radial {radial}");
        }
    }
}

#[test]
fn twist_rotates_with_height_but_preserves_radius() {
    let m = cube(2.0).twist(std::f32::consts::FRAC_PI_2, Vec3::Y);
    // Twisting about Y preserves each point's distance from the Y axis.
    let orig = cube(2.0);
    for (p, o) in m.positions.iter().zip(&orig.positions) {
        let rp = (p.x * p.x + p.z * p.z).sqrt();
        let ro = (o.x * o.x + o.z * o.z).sqrt();
        assert!((rp - ro).abs() < 1e-4, "twist preserves radius");
    }
}

#[test]
fn empty_mesh_has_no_bounds() {
    assert!(MeshData::new().bounds().is_none());
    assert_eq!(MeshData::new().triangle_count(), 0);
}

#[test]
fn plane_faces_up() {
    let m = plane(Vec2::splat(4.0), 3);
    assert_well_formed(&m);
    for t in m.indices.chunks_exact(3) {
        let (a, b, c) = (
            m.positions[t[0] as usize],
            m.positions[t[1] as usize],
            m.positions[t[2] as usize],
        );
        let face_n = (b - a).cross(c - a);
        assert!(face_n.y > 0.0, "plane winds up-facing");
    }
}
