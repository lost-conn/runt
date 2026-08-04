//! 2D polygon utilities: signed area and ear-clipping triangulation.
//!
//! This exists to serve the Godot level converter: `CSGPolygon3D` hands us a
//! `PackedVector2Array` cross-section that has to become caps for an extrusion,
//! a lathe, or a path sweep (see [`crate::extrude`]). The polygons in question
//! are hand-drawn 4–7-gons, so the priority is *determinism and never hanging*,
//! not asymptotic speed — O(n²) ear clipping is the right shape of algorithm at
//! n < 10 and it has no allocation-order or hash-iteration nondeterminism.
//!
//! Everything here is pure `fn(&[Vec2]) -> …`.

use glam::Vec2;

/// Shoelace signed area. Positive when `poly` winds counter-clockwise in a
/// right-handed XY frame (+X right, +Y up), negative when clockwise, and zero
/// for a degenerate (collinear or empty) polygon.
///
/// Callers use the *sign* far more than the magnitude — it is how winding gets
/// normalized before triangulation and before building extrusion side walls, so
/// that outward faces come out outward regardless of how the polygon was drawn
/// in the Godot editor.
pub fn signed_area(poly: &[Vec2]) -> f32 {
    if poly.len() < 3 {
        return 0.0;
    }
    let mut acc = 0.0;
    let mut prev = poly[poly.len() - 1];
    for &p in poly {
        acc += prev.x * p.y - p.x * prev.y;
        prev = p;
    }
    acc * 0.5
}

/// Ear-clipping triangulation of a **simple** polygon.
///
/// Returns triangles as index triples **into the caller's slice**, always wound
/// counter-clockwise (positive area) in the polygon's own XY plane — even when
/// the input was clockwise. Winding is normalized by walking a reversed *index
/// list*, never by reordering points, so the returned indices always address
/// `poly` directly and a caller can share one vertex ring between the cap
/// triangles and the side walls.
///
/// Determinism (DESIGN's core doctrine — same inputs, same mesh, on every
/// device) is bought by two rules:
///
/// - The lowest-index valid ear is always the one clipped. No "best" ear
///   heuristic, no sorting by angle, no floating-point tie-break that could
///   land differently under a different `fma` contraction.
/// - No adaptive refinement or randomized restart anywhere.
///
/// ## Tolerances
///
/// Collinearity and containment tests are area-like (cross products), so they
/// scale as *length²*. The threshold is derived from the polygon's own bounding
/// box (`1e-6 * extent`, squared into cross-product units) rather than being an
/// absolute epsilon: the same polygon authored at 0.1 units and at 200 units —
/// both occur in the playground scene — must triangulate identically.
///
/// Exactly-collinear vertices are stripped up front rather than clipped as
/// zero-area ears, so the output contains no degenerate triangles.
///
/// ## Degenerate input
///
/// Self-intersecting or otherwise non-simple polygons have no valid ear at some
/// point. Rather than loop forever, the search is retried with progressively
/// relaxed convexity/containment rules and, if all of those fail, the remaining
/// ring is closed with a naive fan. That is visually wrong for a genuinely
/// broken polygon but is bounded, allocation-free of surprises, and never
/// panics — a converter should import a mangled brush, not abort the level.
pub fn triangulate(poly: &[Vec2]) -> Vec<[u32; 3]> {
    let n = poly.len();
    if n < 3 {
        return Vec::new();
    }

    let eps_cross = cross_epsilon(poly);

    // Work counter-clockwise while keeping indices into the caller's slice.
    let mut work: Vec<u32> = (0..n as u32).collect();
    if signed_area(poly) < 0.0 {
        work.reverse();
    }
    strip_collinear(poly, &mut work, eps_cross);

    let mut out = Vec::with_capacity(work.len().saturating_sub(2));
    // Relaxation ladder: 0 strict, 1 allow exactly-touching convexity,
    // 2 drop the containment test, 3 both. Past that, fan and bail.
    let mut relax = 0u32;
    const MAX_RELAX: u32 = 3;
    while work.len() > 3 {
        let mut clipped = false;
        for k in 0..work.len() {
            if is_ear(poly, &work, k, eps_cross, relax) {
                let m = work.len();
                out.push([work[(k + m - 1) % m], work[k], work[(k + 1) % m]]);
                work.remove(k);
                clipped = true;
                break; // lowest-index valid ear wins, always
            }
        }
        if !clipped {
            if relax >= MAX_RELAX {
                for i in 1..work.len() - 1 {
                    out.push([work[0], work[i], work[i + 1]]);
                }
                return out;
            }
            relax += 1;
        }
    }
    if work.len() == 3 {
        out.push([work[0], work[1], work[2]]);
    }
    out
}

/// Cross-product-space epsilon scaled by the polygon's bounding box, so the
/// notion of "collinear" is scale-invariant. Cross products of edge vectors
/// scale as length², hence `extent²`.
fn cross_epsilon(poly: &[Vec2]) -> f32 {
    let mut lo = poly[0];
    let mut hi = poly[0];
    for &p in poly {
        lo = lo.min(p);
        hi = hi.max(p);
    }
    let extent = (hi - lo).max_element().max(1e-12);
    1.0e-6 * extent * extent
}

/// Drop vertices whose two incident edges are collinear (this also catches
/// repeated points and needle spikes, whose turn cross product is ~0). Never
/// reduces the ring below a triangle.
fn strip_collinear(poly: &[Vec2], work: &mut Vec<u32>, eps_cross: f32) {
    let mut i = 0;
    while work.len() > 3 && i < work.len() {
        let m = work.len();
        let a = poly[work[(i + m - 1) % m] as usize];
        let b = poly[work[i] as usize];
        let c = poly[work[(i + 1) % m] as usize];
        if (b - a).perp_dot(c - b).abs() <= eps_cross {
            work.remove(i);
            // The previous vertex may have become collinear; re-test it.
            i = i.saturating_sub(1);
        } else {
            i += 1;
        }
    }
}

/// Is `work[k]` the tip of a clippable ear? `relax` loosens the two tests in
/// turn (see [`triangulate`]).
fn is_ear(poly: &[Vec2], work: &[u32], k: usize, eps_cross: f32, relax: u32) -> bool {
    let m = work.len();
    let (ia, ib, ic) = (work[(k + m - 1) % m], work[k], work[(k + 1) % m]);
    let (a, b, c) = (poly[ia as usize], poly[ib as usize], poly[ic as usize]);

    // Convex in a CCW ring means a left turn.
    let convex_min = if relax.is_multiple_of(2) { eps_cross } else { 0.0 };
    if turn_at(poly, work, k) <= convex_min {
        return false;
    }
    if relax >= 2 {
        return true;
    }
    // …and no *reflex* vertex of the ring may sit in the closed triangle.
    //
    // Two details that a naive "no vertex strictly inside" test gets wrong, both
    // of which the playground's L-shaped profiles hit:
    //
    // - The triangle must be *closed*. A reflex vertex lying exactly on the ear's
    //   base is not interior, but clipping the ear anyway cuts across the
    //   polygon boundary and leaves a zero-area remainder that has no ears at
    //   all — the same polygon then triangulates differently depending only on
    //   which winding it arrived in.
    // - Only reflex vertices can block. Convex ones may legitimately touch the
    //   ear's edges, and rejecting on those stalls otherwise-fine polygons.
    !(0..m).any(|j| {
        let iv = work[j];
        iv != ia
            && iv != ib
            && iv != ic
            && turn_at(poly, work, j) <= eps_cross
            && point_in_closed_tri(poly[iv as usize], a, b, c, eps_cross)
    })
}

/// Signed turn at `work[k]`: positive is a left (convex, for a CCW ring) corner.
fn turn_at(poly: &[Vec2], work: &[u32], k: usize) -> f32 {
    let m = work.len();
    let a = poly[work[(k + m - 1) % m] as usize];
    let b = poly[work[k] as usize];
    let c = poly[work[(k + 1) % m] as usize];
    (b - a).perp_dot(c - b)
}

/// Inside-or-on-the-boundary test for a CCW triangle.
fn point_in_closed_tri(p: Vec2, a: Vec2, b: Vec2, c: Vec2, eps_cross: f32) -> bool {
    (b - a).perp_dot(p - a) >= -eps_cross
        && (c - b).perp_dot(p - b) >= -eps_cross
        && (a - c).perp_dot(p - c) >= -eps_cross
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sum of the triangles' areas, in the polygon's plane.
    fn tri_area_sum(poly: &[Vec2], tris: &[[u32; 3]]) -> f32 {
        tris.iter()
            .map(|t| {
                let (a, b, c) = (
                    poly[t[0] as usize],
                    poly[t[1] as usize],
                    poly[t[2] as usize],
                );
                (b - a).perp_dot(c - a) * 0.5
            })
            .sum()
    }

    const SQUARE: [Vec2; 4] = [
        Vec2::new(0.0, 0.0),
        Vec2::new(2.0, 0.0),
        Vec2::new(2.0, 2.0),
        Vec2::new(0.0, 2.0),
    ];

    /// CCW L, area 3, no collinear vertices.
    const L_HEX: [Vec2; 6] = [
        Vec2::new(0.0, 0.0),
        Vec2::new(2.0, 0.0),
        Vec2::new(2.0, 1.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(1.0, 2.0),
        Vec2::new(0.0, 2.0),
    ];

    #[test]
    fn signed_area_sign_follows_winding() {
        assert!((signed_area(&SQUARE) - 4.0).abs() < 1e-6);
        let mut cw = SQUARE.to_vec();
        cw.reverse();
        assert!((signed_area(&cw) + 4.0).abs() < 1e-6);
        assert_eq!(signed_area(&SQUARE[..2]), 0.0);
    }

    #[test]
    fn quad_is_two_triangles_preserving_area() {
        let tris = triangulate(&SQUARE);
        assert_eq!(tris.len(), 2);
        assert!((tri_area_sum(&SQUARE, &tris) - 4.0).abs() < 1e-5);
    }

    #[test]
    fn l_hexagon_is_four_triangles_with_the_right_area() {
        let tris = triangulate(&L_HEX);
        assert_eq!(tris.len(), 4);
        assert!((tri_area_sum(&L_HEX, &tris) - 3.0).abs() < 1e-5);
    }

    #[test]
    fn clockwise_input_yields_ccw_triangles_indexing_the_original() {
        let mut cw = L_HEX.to_vec();
        cw.reverse();
        assert!(signed_area(&cw) < 0.0);
        let tris = triangulate(&cw);
        assert_eq!(tris.len(), 4);
        // Every emitted triangle is CCW (positive area) …
        for t in &tris {
            let (a, b, c) = (
                cw[t[0] as usize],
                cw[t[1] as usize],
                cw[t[2] as usize],
            );
            assert!((b - a).perp_dot(c - a) > 0.0, "triangle {t:?} is not CCW");
        }
        // … and the total still covers the polygon.
        assert!((tri_area_sum(&cw, &tris) - 3.0).abs() < 1e-5);
        // Indices address the caller's slice, so they stay in range.
        assert!(tris.iter().flatten().all(|&i| (i as usize) < cw.len()));
    }

    #[test]
    fn collinear_vertices_are_stripped_not_emitted_as_slivers() {
        // A square with a redundant midpoint on the bottom edge.
        let poly = [
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(0.0, 2.0),
        ];
        let tris = triangulate(&poly);
        assert_eq!(tris.len(), 2, "the collinear vertex should not add a sliver");
        assert!((tri_area_sum(&poly, &tris) - 4.0).abs() < 1e-5);
    }

    #[test]
    fn scale_invariance() {
        let big: Vec<Vec2> = L_HEX.iter().map(|p| *p * 200.0).collect();
        let small: Vec<Vec2> = L_HEX.iter().map(|p| *p * 0.05).collect();
        assert_eq!(triangulate(&big), triangulate(&L_HEX));
        assert_eq!(triangulate(&small), triangulate(&L_HEX));
    }

    #[test]
    fn deterministic_across_runs() {
        let a = triangulate(&L_HEX);
        for _ in 0..8 {
            assert_eq!(triangulate(&L_HEX), a);
        }
    }

    #[test]
    fn degenerate_input_terminates_without_panicking() {
        // Bowtie: self-intersecting, no valid ear decomposition.
        let bowtie = [
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(0.0, 2.0),
        ];
        let tris = triangulate(&bowtie);
        assert!(!tris.is_empty());
        assert!(tris.iter().flatten().all(|&i| (i as usize) < bowtie.len()));

        // All points identical, and fewer than three points.
        assert!(triangulate(&[Vec2::ZERO; 5]).len() <= 3);
        assert!(triangulate(&[Vec2::ZERO, Vec2::X]).is_empty());
        assert!(triangulate(&[]).is_empty());
    }
}
