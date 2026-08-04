//! Sweeps of a 2D cross-section into a solid — the three `CSGPolygon3D` modes
//! the Godot level converter has to reproduce:
//!
//! | Godot `mode` | here |
//! |---|---|
//! | 0 `MODE_DEPTH` | [`extrude`] |
//! | 1 `MODE_SPIN`  | [`lathe`] |
//! | 2 `MODE_PATH`  | [`path_extrude`] |
//!
//! All three are pure `fn(params) -> MeshData`, share one winding rule, and end
//! by handing the raw mesh to a normals op rather than writing normals by hand:
//! caps are planar so their faceted normal *is* the analytic one, and side walls
//! are quads whose faceted normal is exactly right for a hard-edged solid. Where
//! a smooth result is wanted the caller says so (`smooth: bool`, which selects
//! [`ops::creased_normals`] at a 30° crease) instead of the generator guessing.
//!
//! ## Winding rule
//!
//! Every sweep normalizes its cross-section to counter-clockwise
//! ([`poly2d::signed_area`] > 0) first, so the outward side of the polygon is
//! always its left-hand side and the side walls come out front-facing under the
//! renderer's back-face culling. Which *direction* the body extends then decides
//! the cap winding: the cap at the start of the sweep faces **against** the
//! sweep direction, the cap at the end faces **with** it.
//!
//! ## Godot sign conventions (read before trusting these)
//!
//! Godot's `csg_shape.cpp` builds `MODE_DEPTH` from `z = 0` to `z = -depth`:
//! the cross-section sits on the XY plane, its front face looks down `+Z`, and
//! the body grows toward `-Z`. That is what [`extrude`] encodes. `MODE_SPIN`
//! revolves about `+Y` with the polygon's `x` as radius, using the basis
//! `(cos a, 0, -sin a)` — so the sweep also starts off toward `-Z`, which is why
//! [`lathe`] and [`extrude`] share a cap rule.
//!
//! These signs are transcribed from Godot's documented/observed behaviour, not
//! measured — this crate has no headless Godot to diff against. The empirical
//! check is the converter's A/B screenshot pass (M7). If a converted brush comes
//! out inside-out or mirrored in Z, the fix is one sign here, not a redesign:
//! flip the `-depth` in [`extrude`] / the `-sin` in [`lathe`] and swap the two
//! `flip` arguments to [`push_cap`] alongside it.
//!
//! ## UVs
//!
//! Side walls get `u` = normalized distance around the cross-section's
//! perimeter, `v` = normalized progress along the sweep (depth, angle, or arc
//! length). Caps get the raw polygon coordinates, matching Godot. Nothing here
//! tries to be a good atlas layout: the consumers are world-space triplanar
//! materials, which ignore UVs entirely.

use glam::{Vec2, Vec3};

use super::{ops, poly2d, MeshData, PathFrame};

const WHITE: Vec3 = Vec3::ONE;

/// `CSGPolygon3D` mode 0 (Depth): extrude `poly` along `-Z`.
///
/// The cross-section lies on the XY plane at `z = 0` (front cap, facing `+Z`)
/// and the body runs to `z = -depth` (back cap, facing `-Z`). See the module
/// docs on the sign convention and how to flip it.
///
/// Returns a flat-shaded solid: hard edges everywhere, which is what a box-like
/// CSG brush wants. Call [`MeshData::smooth_normals`] on the result if a
/// particular brush was authored with `smooth_faces = true`.
pub fn extrude(poly: &[Vec2], depth: f32) -> MeshData {
    let ccw = ccw_copy(poly);
    if ccw.len() < 3 {
        return MeshData::default();
    }
    let tris = poly2d::triangulate(&ccw);
    let u = perimeter_params(&ccw);

    let mut m = MeshData::default();
    let front = push_ring(&mut m, &ccw, &u, 0.0, |p| p.extend(0.0));
    let back = push_ring(&mut m, &ccw, &u, 1.0, |p| p.extend(-depth));
    // Sweep runs 0 -> -depth, so `front` is the start ring.
    push_wall(&mut m, ccw.len(), front, back, false);

    // Caps carry their own vertices: their UVs are polygon-space, not
    // perimeter-space, and they must not weld into the wall ring's seam.
    push_cap(&mut m, &ccw, &tris, |p| p.extend(0.0), false);
    push_cap(&mut m, &ccw, &tris, |p| p.extend(-depth), true);

    ops::flat_normals(m)
}

/// `CSGPolygon3D` mode 1 (Spin): revolve `poly` about the local `Y` axis.
///
/// The cross-section's `x` is a radius from the Y axis and its `y` is height,
/// swept through `degrees` in `sides` steps. A full revolution (|degrees| = 360)
/// closes the surface and emits no caps; a partial one is closed by a flat cap
/// at each end angle.
///
/// `smooth` picks crease-angle normals (Godot's `smooth_faces`) over faceted
/// ones — the playground's `Summit` brushes set it, and a 13-sided revolve looks
/// like a cone rather than a gem with it on.
///
/// Vertices with `x == 0` sit on the axis and their quads collapse; the normals
/// op drops those degenerate triangles, so an axis-touching profile still yields
/// a clean solid.
pub fn lathe(poly: &[Vec2], degrees: f32, sides: u32, smooth: bool) -> MeshData {
    let ccw = ccw_copy(poly);
    if ccw.len() < 3 {
        return MeshData::default();
    }
    let sides = sides.max(3); // Godot clamps spin_sides to >= 3
    let total = degrees.to_radians();
    let full = (degrees.abs() - 360.0).abs() <= 1.0e-3;
    let n = ccw.len();
    let u = perimeter_params(&ccw);

    let mut m = MeshData::default();
    let mut rings = Vec::with_capacity(sides as usize + 1);
    for j in 0..=sides {
        let f = j as f32 / sides as f32;
        let (sa, ca) = (total * f).sin_cos();
        rings.push(push_ring(&mut m, &ccw, &u, f, |p| {
            Vec3::new(p.x * ca, p.y, -p.x * sa)
        }));
    }
    for w in rings.windows(2) {
        push_wall(&mut m, n, w[0], w[1], false);
    }

    if !full {
        let tris = poly2d::triangulate(&ccw);
        let (se, ce) = total.sin_cos();
        push_cap(&mut m, &ccw, &tris, |p| p.extend(0.0), false);
        push_cap(
            &mut m,
            &ccw,
            &tris,
            |p| Vec3::new(p.x * ce, p.y, -p.x * se),
            true,
        );
    }

    finish(m, smooth)
}

/// `CSGPolygon3D` mode 2 (Path): sweep `poly` along a sequence of oriented
/// frames (see [`crate::curve::Curve3::frames`]).
///
/// The cross-section is placed in each frame's `(binormal, normal)` plane —
/// polygon `x` runs across the path, polygon `y` runs "up" it — matching Godot's
/// path-follow orientation. Because the frame basis is right-handed with
/// `binormal × normal = tangent`, the sweep direction is `+tangent`, the
/// opposite of [`extrude`]'s; the wall winding and cap flips are mirrored to
/// match.
///
/// `joined` welds the last ring back onto the first (Godot's `path_joined`),
/// giving a closed loop with no caps — the playground's cliff bands are built
/// this way. Otherwise both ends are capped with the triangulated cross-section.
pub fn path_extrude(poly: &[Vec2], frames: &[PathFrame], joined: bool, smooth: bool) -> MeshData {
    let ccw = ccw_copy(poly);
    if ccw.len() < 3 || frames.len() < 2 {
        return MeshData::default();
    }
    let n = ccw.len();
    let u = perimeter_params(&ccw);
    let v = arc_params(frames, joined);

    let mut m = MeshData::default();
    let mut rings = Vec::with_capacity(frames.len() + 1);
    for (i, f) in frames.iter().enumerate() {
        rings.push(push_ring(&mut m, &ccw, &u, v[i], |p| place(*f, p)));
    }
    if joined {
        // Repeat frame 0 as a terminating ring rather than wrapping the index:
        // it keeps the wall loop uniform and gives the seam a v = 1 row instead
        // of a UV discontinuity. The duplicate welds by position, so the surface
        // is still closed.
        let f0 = frames[0];
        rings.push(push_ring(&mut m, &ccw, &u, 1.0, |p| place(f0, p)));
    }
    for w in rings.windows(2) {
        push_wall(&mut m, n, w[0], w[1], true);
    }

    if !joined {
        let tris = poly2d::triangulate(&ccw);
        let (first, last) = (frames[0], frames[frames.len() - 1]);
        push_cap(&mut m, &ccw, &tris, |p| place(first, p), true);
        push_cap(&mut m, &ccw, &tris, |p| place(last, p), false);
    }

    finish(m, smooth)
}

// --- shared machinery -------------------------------------------------------

/// Place a cross-section point into a frame: `x -> binormal`, `y -> normal`.
fn place(f: PathFrame, p: Vec2) -> Vec3 {
    f.pos + f.binormal * p.x + f.normal * p.y
}

/// The cross-section wound counter-clockwise, whatever the caller drew.
/// Degenerate (zero-area) input is passed through untouched — there is no
/// meaningful outward side to pick.
fn ccw_copy(poly: &[Vec2]) -> Vec<Vec2> {
    let mut out = poly.to_vec();
    if poly2d::signed_area(poly) < 0.0 {
        out.reverse();
    }
    out
}

/// Cumulative perimeter fraction per cross-section vertex, `len + 1` long with a
/// trailing `1.0` for the duplicated seam vertex.
fn perimeter_params(poly: &[Vec2]) -> Vec<f32> {
    let n = poly.len();
    let mut acc = vec![0.0; n + 1];
    for i in 0..n {
        acc[i + 1] = acc[i] + (poly[(i + 1) % n] - poly[i]).length();
    }
    let total = acc[n];
    if total > 0.0 {
        for a in &mut acc {
            *a /= total;
        }
    }
    acc
}

/// Normalized sweep progress per frame, by arc length along the frame
/// positions. `joined` counts the closing leg in the total so the seam lands
/// near `v = 1` instead of overshooting it.
fn arc_params(frames: &[PathFrame], joined: bool) -> Vec<f32> {
    let n = frames.len();
    let mut acc = vec![0.0; n];
    for i in 1..n {
        acc[i] = acc[i - 1] + (frames[i].pos - frames[i - 1].pos).length();
    }
    let mut total = acc[n - 1];
    if joined {
        total += (frames[0].pos - frames[n - 1].pos).length();
    }
    if total > 0.0 {
        for a in &mut acc {
            *a /= total;
        }
    }
    acc
}

/// Push one swept ring: `poly.len() + 1` vertices (the last duplicates the
/// first so the `u` seam is not stitched back to 0). Returns its base index.
fn push_ring(
    m: &mut MeshData,
    poly: &[Vec2],
    u: &[f32],
    v: f32,
    place: impl Fn(Vec2) -> Vec3,
) -> u32 {
    let base = m.positions.len() as u32;
    for i in 0..=poly.len() {
        m.positions.push(place(poly[i % poly.len()]));
        m.uvs.push(Vec2::new(u[i], v));
        m.colors.push(WHITE);
    }
    base
}

/// Stitch two rings of `n + 1` vertices into a wall.
///
/// `along_plus` says the sweep runs *toward* the second ring in the same
/// direction the cross-section's front faces (the path case); `false` means it
/// runs away from it (depth and spin). The two need opposite windings for the
/// wall to face outward.
fn push_wall(m: &mut MeshData, n: usize, a: u32, b: u32, along_plus: bool) {
    for i in 0..n as u32 {
        if along_plus {
            quad(&mut m.indices, a + i, a + i + 1, b + i + 1, b + i);
        } else {
            quad(&mut m.indices, a + i, b + i, b + i + 1, a + i + 1);
        }
    }
}

/// Push a triangulated cap with its own vertices. `flip` reverses the winding
/// (i.e. the cap faces the other way along the sweep).
fn push_cap(
    m: &mut MeshData,
    poly: &[Vec2],
    tris: &[[u32; 3]],
    place: impl Fn(Vec2) -> Vec3,
    flip: bool,
) {
    let base = m.positions.len() as u32;
    for &p in poly {
        m.positions.push(place(p));
        m.uvs.push(p); // Godot writes the raw polygon coords here
        m.colors.push(WHITE);
    }
    for t in tris {
        if flip {
            m.indices
                .extend_from_slice(&[base + t[0], base + t[2], base + t[1]]);
        } else {
            m.indices
                .extend_from_slice(&[base + t[0], base + t[1], base + t[2]]);
        }
    }
}

/// Two CCW triangles over four already-pushed vertices given in CCW order.
fn quad(indices: &mut Vec<u32>, a: u32, b: u32, c: u32, d: u32) {
    indices.extend_from_slice(&[a, b, c, a, c, d]);
}

/// Normals are always computed from the finished geometry, never authored:
/// planar caps and quad walls both get exactly the right faceted normal, and
/// `smooth` is purely a shading choice the caller inherits from Godot's
/// `smooth_faces`. The 30° crease keeps cap-to-wall edges hard either way.
fn finish(m: MeshData, smooth: bool) -> MeshData {
    if smooth {
        ops::creased_normals(m, 30.0)
    } else {
        ops::flat_normals(m)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::{
        curve::{Curve3, CurvePoint},
        Quality,
    };

    const SQUARE: [Vec2; 4] = [
        Vec2::new(-0.5, -0.5),
        Vec2::new(0.5, -0.5),
        Vec2::new(0.5, 0.5),
        Vec2::new(-0.5, 0.5),
    ];

    /// A CCW rectangle profile from radius `r0` to `r1`, height `2 * hy`.
    fn tube_profile(r0: f32, r1: f32, hy: f32) -> [Vec2; 4] {
        [
            Vec2::new(r0, -hy),
            Vec2::new(r1, -hy),
            Vec2::new(r1, hy),
            Vec2::new(r0, hy),
        ]
    }

    /// Every undirected edge is shared by exactly two triangles.
    ///
    /// Adjacency is taken over *welded positions*, not indices: both normals ops
    /// de-index the mesh, and the sweeps deliberately duplicate the UV seam
    /// vertex, so index identity says nothing about whether the surface closes.
    fn is_closed_surface(m: &MeshData) -> bool {
        let mut ids: HashMap<[i64; 3], u32> = HashMap::new();
        let vid: Vec<u32> = m
            .positions
            .iter()
            .map(|p| {
                let key = [
                    (p.x * 1.0e4).round() as i64,
                    (p.y * 1.0e4).round() as i64,
                    (p.z * 1.0e4).round() as i64,
                ];
                let next = ids.len() as u32;
                *ids.entry(key).or_insert(next)
            })
            .collect();

        let mut counts: HashMap<(u32, u32), u32> = HashMap::new();
        for t in m.indices.chunks_exact(3) {
            let v = [vid[t[0] as usize], vid[t[1] as usize], vid[t[2] as usize]];
            for k in 0..3 {
                let (x, y) = (v[k], v[(k + 1) % 3]);
                *counts.entry((x.min(y), x.max(y))).or_insert(0) += 1;
            }
        }
        counts.values().all(|&c| c == 2)
    }

    /// Does any triangle's plane put the mesh centroid on its front side?
    /// (Sanity check that faces point outward on a convex solid.)
    fn all_faces_outward(m: &MeshData) -> bool {
        let c: Vec3 = m.positions.iter().copied().sum::<Vec3>() / m.positions.len() as f32;
        m.triangles().all(|[a, b, t]| {
            let n = (b - a).cross(t - a);
            n.dot(a - c) > 0.0
        })
    }

    #[test]
    fn extrude_square_is_a_closed_box() {
        let m = extrude(&SQUARE, 2.0);
        m.validate();
        // 4 wall quads (8 tris) + 2 caps (2 tris each) = 12 triangles.
        assert_eq!(m.triangle_count(), 12);
        assert_eq!(m.vertex_count(), 36); // flat_normals de-indexes
        assert_eq!(m.colors, vec![Vec3::ONE; 36]);
        assert!(is_closed_surface(&m), "extrusion is not watertight");
        assert!(all_faces_outward(&m), "extrusion has inward-facing faces");

        let (lo, hi) = m.bounds().unwrap();
        assert!((lo - Vec3::new(-0.5, -0.5, -2.0)).length() < 1e-5, "{lo:?}");
        assert!((hi - Vec3::new(0.5, 0.5, 0.0)).length() < 1e-5, "{hi:?}");
    }

    #[test]
    fn extrude_depth_runs_toward_negative_z_with_the_front_cap_at_zero() {
        // The Godot MODE_DEPTH convention this crate encodes (see module docs).
        let m = extrude(&SQUARE, 3.0);
        let front = m
            .triangles()
            .zip(m.indices.chunks_exact(3))
            .find(|([a, b, c], _)| a.z == 0.0 && b.z == 0.0 && c.z == 0.0)
            .expect("a cap at z = 0");
        let [a, b, c] = front.0;
        let n = (b - a).cross(c - a).normalize();
        assert!((n - Vec3::Z).length() < 1e-5, "front cap faces {n:?}, want +Z");
        assert!(m.positions.iter().all(|p| p.z <= 1e-6));
    }

    #[test]
    fn extrude_normalizes_clockwise_input() {
        let mut cw = SQUARE.to_vec();
        cw.reverse();
        let m = extrude(&cw, 2.0);
        assert_eq!(m.triangle_count(), 12);
        assert!(is_closed_surface(&m));
        assert!(all_faces_outward(&m), "CW input produced an inside-out solid");
    }

    #[test]
    fn extrude_of_a_concave_profile_stays_closed() {
        let l = [
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 2.0),
            Vec2::new(0.0, 2.0),
        ];
        let m = extrude(&l, 1.0);
        m.validate();
        // 6 wall quads (12 tris) + 2 caps of 4 tris = 20.
        assert_eq!(m.triangle_count(), 20);
        assert!(is_closed_surface(&m), "L-extrusion is not watertight");
    }

    #[test]
    fn extrude_rejects_degenerate_input() {
        assert!(extrude(&[], 1.0).is_empty());
        assert!(extrude(&SQUARE[..2], 1.0).is_empty());
    }

    #[test]
    fn full_lathe_matches_a_cylinder_shell() {
        const SIDES: u32 = 12;
        let m = lathe(&tube_profile(1.0, 2.0, 1.0), 360.0, SIDES, false);
        m.validate();
        // 4 profile edges x SIDES steps x 2 tris, and no caps on a full turn.
        assert_eq!(m.triangle_count(), 4 * SIDES as usize * 2);
        assert_eq!(m.vertex_count(), m.triangle_count() * 3);
        assert!(is_closed_surface(&m), "full revolution is not watertight");

        let (lo, hi) = m.bounds().unwrap();
        assert!((lo - Vec3::new(-2.0, -1.0, -2.0)).length() < 1e-4, "{lo:?}");
        assert!((hi - Vec3::new(2.0, 1.0, 2.0)).length() < 1e-4, "{hi:?}");
        // Every vertex sits on one of the two radii (a shell, not a disc).
        for p in &m.positions {
            let r = Vec2::new(p.x, p.z).length();
            assert!(
                (r - 1.0).abs() < 1e-4 || (r - 2.0).abs() < 1e-4,
                "off-shell radius {r}"
            );
        }
    }

    #[test]
    fn partial_lathe_gets_flat_end_caps() {
        const SIDES: u32 = 8;
        let profile = tube_profile(1.0, 2.0, 1.0);
        let m = lathe(&profile, 90.0, SIDES, false);
        m.validate();
        // Walls as before, plus a 2-triangle cap at each end angle.
        assert_eq!(m.triangle_count(), 4 * SIDES as usize * 2 + 4);
        assert!(is_closed_surface(&m), "partial revolution is not watertight");

        // The start cap sits on the XY plane facing +Z (against the sweep).
        let start = m
            .triangles()
            .find(|[a, b, c]| a.z.abs() < 1e-6 && b.z.abs() < 1e-6 && c.z.abs() < 1e-6)
            .expect("a cap at angle 0");
        let n = (start[1] - start[0]).cross(start[2] - start[0]).normalize();
        assert!((n - Vec3::Z).length() < 1e-4, "start cap faces {n:?}");
        // A quarter turn from +X toward -Z stays in x >= 0, z <= 0.
        assert!(m.positions.iter().all(|p| p.x >= -1e-4 && p.z <= 1e-4));
    }

    #[test]
    fn lathe_smooth_flag_selects_crease_normals() {
        let profile = tube_profile(1.0, 2.0, 1.0);
        let flat = lathe(&profile, 360.0, 16, false);
        let smooth = lathe(&profile, 360.0, 16, true);
        assert_eq!(flat.triangle_count(), smooth.triangle_count());
        // Creasing welds the co-planar corners the faceted build kept apart.
        assert!(smooth.vertex_count() < flat.vertex_count());
        // The outer wall is genuinely rounded now: its normal has a tangential
        // component instead of matching the flat facet exactly.
        let curved = smooth.positions.iter().zip(&smooth.normals).any(|(p, n)| {
            let radial = Vec3::new(p.x, 0.0, p.z).normalize_or_zero();
            n.dot(radial) > 0.9 && n.dot(radial) < 0.9999
        });
        assert!(curved, "smooth lathe produced no rounded normals");
    }

    #[test]
    fn lathe_profile_touching_the_axis_drops_collapsed_quads() {
        // x = 0 vertices lie on the spin axis; their quads have zero area.
        let m = lathe(&tube_profile(0.0, 2.0, 1.0), 360.0, 10, false);
        m.validate();
        assert!(m.triangle_count() > 0);
        assert!(m.triangles().all(|[a, b, c]| (b - a).cross(c - a).length() > 1e-6));
    }

    #[test]
    fn path_extrude_joined_on_a_closed_square_path_is_watertight() {
        let path = Curve3 {
            points: vec![
                CurvePoint::at(Vec3::new(-3.0, 0.0, -3.0)),
                CurvePoint::at(Vec3::new(3.0, 0.0, -3.0)),
                CurvePoint::at(Vec3::new(3.0, 0.0, 3.0)),
                CurvePoint::at(Vec3::new(-3.0, 0.0, 3.0)),
            ],
            closed: true,
        };
        let frames = path.frames(2.0, Quality::FULL);
        assert_eq!(frames.len(), 12);

        let m = path_extrude(&SQUARE, &frames, true, false);
        m.validate();
        // 12 rings + the repeated seam ring = 12 wall bands x 4 quads x 2 tris.
        assert_eq!(m.triangle_count(), 12 * 4 * 2);
        assert!(is_closed_surface(&m), "joined sweep is not watertight");
        // No caps: the tube is hollow along the path, so its bounds are the
        // path's box grown by the half-section.
        let (lo, hi) = m.bounds().unwrap();
        assert!((lo - Vec3::new(-3.5, -0.5, -3.5)).length() < 1e-4, "{lo:?}");
        assert!((hi - Vec3::new(3.5, 0.5, 3.5)).length() < 1e-4, "{hi:?}");
    }

    #[test]
    fn path_extrude_open_caps_both_ends() {
        let path = Curve3 {
            points: vec![
                CurvePoint::at(Vec3::ZERO),
                CurvePoint::at(Vec3::new(0.0, 0.0, 4.0)),
            ],
            closed: false,
        };
        let frames = path.frames(1.0, Quality::FULL);
        assert_eq!(frames.len(), 5);

        let m = path_extrude(&SQUARE, &frames, false, false);
        m.validate();
        // 4 wall bands x 4 quads x 2 tris, plus 2 tris per cap.
        assert_eq!(m.triangle_count(), 4 * 4 * 2 + 4);
        assert!(is_closed_surface(&m), "capped sweep is not watertight");
        assert!(all_faces_outward(&m), "capped sweep has inward-facing faces");
    }

    #[test]
    fn path_extrude_places_polygon_x_across_and_y_up() {
        // Assert the mapping itself rather than a hard-coded basis: which world
        // axis seeds the frame is `curve`'s business, this is `extrude`'s.
        let path = Curve3 {
            points: vec![
                CurvePoint::at(Vec3::ZERO),
                CurvePoint::at(Vec3::new(0.0, 0.0, 4.0)),
            ],
            closed: false,
        };
        let f = path.frames(4.0, Quality::FULL)[0];
        // Polygon y maps to the frame normal, x to the binormal, both
        // perpendicular to the direction of travel.
        assert!(f.tangent.dot(Vec3::Z) > 0.999);
        assert!(f.normal.dot(f.tangent).abs() < 1e-5);
        let p = place(f, Vec2::new(1.0, 2.0));
        assert!((p - (f.binormal * 1.0 + f.normal * 2.0)).length() < 1e-5);
    }

    #[test]
    fn path_extrude_rejects_short_input() {
        let f = [PathFrame {
            pos: Vec3::ZERO,
            tangent: Vec3::Z,
            normal: Vec3::Y,
            binormal: Vec3::X,
        }];
        assert!(path_extrude(&SQUARE, &f, false, false).is_empty());
        assert!(path_extrude(&SQUARE, &[], true, false).is_empty());
    }

    #[test]
    fn uvs_span_the_unit_square() {
        for m in [
            extrude(&SQUARE, 2.0),
            lathe(&tube_profile(1.0, 2.0, 1.0), 360.0, 8, false),
        ] {
            assert_eq!(m.uvs.len(), m.positions.len());
            assert!(m.uvs.iter().all(|uv| uv.is_finite()));
        }
        // Wall UVs specifically: u and v both reach 0 and 1 on a lathe (which
        // has no caps to pollute the range).
        let m = lathe(&tube_profile(1.0, 2.0, 1.0), 360.0, 8, false);
        let (umin, umax) = m.uvs.iter().fold((f32::MAX, f32::MIN), |(a, b), uv| {
            (a.min(uv.x), b.max(uv.x))
        });
        let (vmin, vmax) = m.uvs.iter().fold((f32::MAX, f32::MIN), |(a, b), uv| {
            (a.min(uv.y), b.max(uv.y))
        });
        assert!(umin.abs() < 1e-6 && (umax - 1.0).abs() < 1e-6);
        assert!(vmin.abs() < 1e-6 && (vmax - 1.0).abs() < 1e-6);
    }

    /// Content-hash pins for the two canonical brushes the Godot converter
    /// leans on. A change to winding, vertex order, UVs, or the normals op moves
    /// them. That is allowed — but it has to be a deliberate edit to these
    /// numbers, because the hash is also the content-cache key (DESIGN §6) and
    /// changing it silently invalidates every cached level.
    ///
    /// These pin *this crate's* output, not a portable digest:
    /// `MeshData::content_hash` is built on `DefaultHasher`, whose algorithm the
    /// standard library is free to change between Rust releases. A toolchain
    /// bump moving both numbers at once is the expected signature of that, and
    /// is not a regression in the geometry.
    #[test]
    fn canonical_meshes_hash_stably() {
        assert_eq!(extrude(&SQUARE, 2.0).content_hash(), 10999788032361020439);
        assert_eq!(
            lathe(&tube_profile(1.0, 2.0, 1.0), 360.0, 12, false).content_hash(),
            1136293487338084893
        );
    }

    #[test]
    fn generation_is_deterministic() {
        let a = extrude(&SQUARE, 2.0);
        let b = lathe(&tube_profile(1.0, 2.0, 1.0), 200.0, 9, true);
        for _ in 0..4 {
            assert_eq!(extrude(&SQUARE, 2.0), a);
            assert_eq!(lathe(&tube_profile(1.0, 2.0, 1.0), 200.0, 9, true), b);
        }
    }
}
