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

// ---------------------------------------------------------------------------
// Height field / terrain (DESIGN §6, §9)
// ---------------------------------------------------------------------------

fn demo_field() -> HeightField {
    HeightField {
        seed: 20260731,
        amplitude: 1.2,
        octaves: 4,
        frequency: 0.055,
        lacunarity: 2.0,
        gain: 0.5,
    }
}

fn demo_params() -> TerrainParams {
    let f = demo_field();
    TerrainParams {
        seed: f.seed,
        size: Vec2::splat(40.0),
        amplitude: f.amplitude,
        octaves: f.octaves,
        frequency: f.frequency,
        lacunarity: f.lacunarity,
        gain: f.gain,
        base_segments: 64,
        color: Some(Vec3::new(0.17, 0.21, 0.18)),
        tint: None,
    }
}

/// A spread of sample points including negative coordinates (where a naive
/// `as i32` lattice index silently folds) and exact lattice hits.
fn sample_points() -> Vec<(f32, f32)> {
    let mut out = Vec::new();
    for i in -7..=7 {
        for j in -7..=7 {
            out.push((i as f32 * 2.7183, j as f32 * -3.1416));
        }
    }
    out.extend([(0.0, 0.0), (1.0 / 0.055, 0.0), (-1.0 / 0.055, 2.0 / 0.055)]);
    out
}

#[test]
fn the_field_is_a_pure_function_of_its_params() {
    let f = demo_field();
    for (x, z) in sample_points() {
        // Bit-exact, not approximately: `h` is what physics integrates against,
        // and "almost the same height" is how a replay diverges.
        assert_eq!(
            f.height(x, z).to_bits(),
            f.height(x, z).to_bits(),
            "h({x}, {z}) must be reproducible"
        );
        assert_eq!(demo_field().height(x, z).to_bits(), f.height(x, z).to_bits());
    }
}

#[test]
fn a_different_seed_is_a_different_surface() {
    let a = demo_field();
    let b = HeightField { seed: a.seed + 1, ..a };
    let differing = sample_points()
        .into_iter()
        .filter(|&(x, z)| (a.height(x, z) - b.height(x, z)).abs() > 1e-4)
        .count();
    assert!(
        differing > 200,
        "one seed apart should change nearly every sample, changed {differing}"
    );
}

#[test]
fn the_gradient_is_the_derivative_of_the_height() {
    // The gradient is analytic (a closed form of the bilinear patch), so this
    // is a genuine cross-check of two independent computations rather than a
    // tautology — physics reads the gradient and the mesh reads the heights,
    // and they have to describe one surface.
    let f = demo_field();
    let h = 1.0e-2_f32;
    for (x, z) in sample_points() {
        let g = f.gradient(x, z);
        let fd_x = (f.height(x + h, z) - f.height(x - h, z)) / (2.0 * h);
        let fd_z = (f.height(x, z + h) - f.height(x, z - h)) / (2.0 * h);
        // Central differences are second-order accurate; the surface's own
        // curvature sets the floor on how close they can get.
        assert!(
            (g.x - fd_x).abs() < 2.0e-3,
            "d/dx at ({x}, {z}): analytic {} vs finite {fd_x}",
            g.x
        );
        assert!(
            (g.y - fd_z).abs() < 2.0e-3,
            "d/dz at ({x}, {z}): analytic {} vs finite {fd_z}",
            g.y
        );
    }
}

#[test]
fn the_normal_agrees_with_the_gradient() {
    let f = demo_field();
    for (x, z) in sample_points() {
        let n = f.normal(x, z);
        assert!((n.length() - 1.0).abs() < 1e-5, "unit normal");
        assert!(n.y > 0.0, "a height field never overhangs");
        // Tangent along X is (1, h_x, 0); the normal must be perpendicular.
        let g = f.gradient(x, z);
        assert!(n.dot(Vec3::new(1.0, g.x, 0.0)).abs() < 1e-4);
        assert!(n.dot(Vec3::new(0.0, g.y, 1.0)).abs() < 1e-4);
    }
}

#[test]
fn height_stays_inside_the_amplitude() {
    let f = demo_field();
    for (x, z) in sample_points() {
        assert!(
            f.height(x, z).abs() <= f.amplitude + 1e-4,
            "octave weights are normalized, so amplitude is a real bound"
        );
    }
}

#[test]
fn the_field_does_not_depend_on_tessellation() {
    // DESIGN §9's load-bearing property: the mesh is a *view* of the field, so
    // visual LOD cannot move the surface physics feels. Meshing at three
    // qualities must place every shared vertex at the same height.
    let params = demo_params();
    let coarse = terrain(&params, Quality(0.25));
    let fine = terrain(&params, Quality(1.0));
    assert!(coarse.positions.len() < fine.positions.len(), "quality did something");

    let field = params.field();
    for m in [&coarse, &fine] {
        for p in &m.positions {
            assert_eq!(
                p.y.to_bits(),
                field.height(p.x, p.z).to_bits(),
                "vertex {p:?} must sit exactly on h(x, z)"
            );
        }
    }

    // And the field itself, asked directly, ignores quality entirely — there is
    // no quality argument to `height` at all, which is the design, not an
    // accident of these params.
    for (x, z) in sample_points() {
        assert_eq!(
            params.field().height(x, z).to_bits(),
            demo_params().field().height(x, z).to_bits()
        );
    }
}

#[test]
fn terrain_normals_come_from_the_field_not_the_facets() {
    let params = TerrainParams { base_segments: 8, ..demo_params() };
    let m = terrain(&params, Quality::FULL);
    assert_well_formed(&m);
    let field = params.field();
    for (p, n) in m.positions.iter().zip(&m.normals) {
        let want = field.normal(p.x, p.z);
        assert!(
            n.abs_diff_eq(want, 1e-5),
            "normal at {p:?} is {n:?}, field says {want:?}"
        );
    }

    // Face-averaged normals would be a *different* answer at this tessellation,
    // which is the whole reason we do not use them: prove the two disagree so
    // this test cannot pass vacuously.
    let faceted = m.clone().flat_normals();
    let drift = faceted
        .normals
        .iter()
        .zip(&faceted.positions)
        .map(|(n, p)| n.angle_between(field.normal(p.x, p.z)))
        .fold(0.0f32, f32::max);
    assert!(drift > 0.01, "flat normals should visibly differ, max drift {drift}");
}

#[test]
fn terrain_is_a_well_formed_upward_grid() {
    let params = TerrainParams { base_segments: 6, ..demo_params() };
    let m = terrain(&params, Quality::FULL);
    assert_well_formed(&m);

    for t in m.indices.chunks_exact(3) {
        let (a, b, c) = (
            m.positions[t[0] as usize],
            m.positions[t[1] as usize],
            m.positions[t[2] as usize],
        );
        assert!((b - a).cross(c - a).y > 0.0, "terrain winds up-facing");
    }

    let (min, max) = m.bounds().expect("terrain has vertices");
    assert!((min.x + 20.0).abs() < 1e-4 && (max.x - 20.0).abs() < 1e-4, "size honored on X");
    assert!((min.z + 20.0).abs() < 1e-4 && (max.z - 20.0).abs() < 1e-4, "size honored on Z");
    for uv in &m.uvs {
        assert!((0.0..=1.0).contains(&uv.x) && (0.0..=1.0).contains(&uv.y));
    }
    for c in &m.colors {
        assert_eq!(*c, Vec3::new(0.17, 0.21, 0.18));
    }
}

#[test]
fn terrain_quality_scales_segments_with_a_floor() {
    let params = TerrainParams { base_segments: 64, ..demo_params() };
    assert_eq!(params.segments(Quality::FULL), 64);
    assert_eq!(params.segments(Quality(0.5)), 32);
    // DESIGN §11: scale down, never fail. An absurd tier still meshes.
    assert_eq!(params.segments(Quality(0.0)), 1);
    assert!(!terrain(&params, Quality(0.0)).is_empty());
}

// --- tints (DESIGN §5's vertex-color × albedo look) ------------------------

fn demo_tint() -> TerrainTint {
    TerrainTint {
        low_color: Vec3::new(0.20, 0.40, 0.18),
        high_color: Vec3::new(0.70, 0.62, 0.44),
        steep_color: Vec3::new(0.40, 0.36, 0.32),
        steep_start_deg: 20.0,
        steep_full_deg: 42.0,
    }
}

#[test]
fn an_absent_tint_generates_the_mesh_byte_for_byte() {
    // The compatibility claim in one assertion: every scene written before tints
    // existed parses to `tint: None` and must produce the *same* geometry, down
    // to the content hash the mesh registry and the on-disk cache are keyed on.
    let plain = demo_params();
    assert_eq!(plain.tint, None, "the default really is None");
    let a = terrain(&plain, Quality::FULL);
    let b = terrain(&TerrainParams { ..plain }, Quality::FULL);
    assert_eq!(a, b);
    assert_eq!(a.content_hash(), b.content_hash());
    // The flat `color` is still exactly what lands on every vertex.
    for c in &a.colors {
        assert_eq!(*c, Vec3::new(0.17, 0.21, 0.18));
    }

    // …and a tint really does change it, so the above is not vacuous.
    let tinted = terrain(
        &TerrainParams { tint: Some(demo_tint()), ..plain },
        Quality::FULL,
    );
    assert_eq!(tinted.positions, a.positions, "a tint is colour, not shape");
    assert_ne!(tinted.colors, a.colors);
    assert_ne!(tinted.content_hash(), a.content_hash());
}

#[test]
fn a_tint_is_a_property_of_the_field_not_the_tessellation() {
    // The same rule DESIGN §9 puts on collision, applied to colour: a vertex at
    // a given (x, z) is the same colour whichever quality meshed it. Segment
    // counts of 8 and 32 share every 4th lattice line, so the coarse mesh's
    // vertices all reappear in the fine one.
    let params = TerrainParams {
        base_segments: 32,
        tint: Some(demo_tint()),
        ..demo_params()
    };
    let coarse = terrain(&params, Quality(0.25));
    let fine = terrain(&params, Quality(1.0));
    assert_eq!(coarse.positions.len(), 9 * 9);
    assert_eq!(fine.positions.len(), 33 * 33);

    let mut matched = 0;
    for (p, c) in coarse.positions.iter().zip(&coarse.colors) {
        let (q, d) = fine
            .positions
            .iter()
            .zip(&fine.colors)
            .find(|(q, _)| q.x == p.x && q.z == p.z)
            .expect("every coarse lattice point is also a fine one");
        assert_eq!(q.y.to_bits(), p.y.to_bits());
        assert_eq!(d.to_array(), c.to_array(), "colour drifted with quality at {p:?}");
        matched += 1;
    }
    assert_eq!(matched, 81, "the coarse lattice was not fully covered");
}

#[test]
fn a_tint_reads_height_then_slope() {
    let tint = demo_tint();
    let amplitude = 2.0;
    let flat = Vec2::ZERO;

    // Height band: the ends of the amplitude are the authored colours exactly,
    // and it is monotonic in between.
    assert!(tint.sample(-amplitude, flat, amplitude).abs_diff_eq(tint.low_color, 1e-6));
    assert!(tint.sample(amplitude, flat, amplitude).abs_diff_eq(tint.high_color, 1e-6));
    let mid = tint.sample(0.0, flat, amplitude);
    assert!(mid.abs_diff_eq((tint.low_color + tint.high_color) * 0.5, 1e-6));

    // Past the amplitude the band clamps rather than extrapolating past the
    // authored colours.
    assert!(tint
        .sample(10.0 * amplitude, flat, amplitude)
        .abs_diff_eq(tint.high_color, 1e-6));

    // Slope band: flat ground is untouched, a wall is fully `steep_color`, and
    // an angle inside the band is somewhere between the two.
    let wall = Vec2::new((60f32).to_radians().tan(), 0.0);
    assert!(tint.sample(0.0, wall, amplitude).abs_diff_eq(tint.steep_color, 1e-6));
    let leaning = Vec2::new((31f32).to_radians().tan(), 0.0);
    let partial = tint.sample(0.0, leaning, amplitude);
    assert!(partial.distance(mid) > 1e-3 && partial.distance(tint.steep_color) > 1e-3);

    // The slope is read from |∇h|, so which way the hill faces cannot matter.
    let mirrored = tint.sample(0.0, Vec2::new(0.0, -leaning.x), amplitude);
    assert!(partial.abs_diff_eq(mirrored, 1e-6));

    // A zero-amplitude field has no band; it must not divide by zero.
    assert!(tint.sample(0.0, flat, 0.0).is_finite());
}

#[test]
fn the_lattice_hash_is_not_diagonally_symmetric() {
    // A symmetric coordinate combine hashes (a, b) and (b, a) alike and shows up
    // as a mirror ridge across the diagonal — cheap to write, hard to unsee.
    let f = HeightField { octaves: 1, frequency: 1.0, ..demo_field() };
    let mirrored = (1..40)
        .map(|i| {
            let (x, z) = (i as f32 * 0.37, i as f32 * 1.61);
            (f.height(x, z) - f.height(z, x)).abs()
        })
        .filter(|d| *d < 1e-6)
        .count();
    assert!(mirrored <= 1, "h(x,z) must not mirror h(z,x) ({mirrored} matches)");
}
