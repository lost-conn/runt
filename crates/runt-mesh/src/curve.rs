//! A port of Godot's `Curve3D` / `Path3D` sampling, reduced to what a level
//! converter needs: turn a list of bezier control points into a deterministic
//! sequence of oriented frames that [`crate::extrude::path_extrude`] can sweep a
//! cross-section along (`CSGPolygon3D` mode 2, "Path").
//!
//! ## Why this is not Godot's baker
//!
//! Godot bakes a curve by walking each segment with an *adaptive* subdivision
//! that stops when the chord error falls under a tolerance. That is a loop whose
//! iteration count depends on floating-point comparisons, so two machines with
//! different `fma` contraction can bake different vertex counts from the same
//! scene. This crate's doctrine is that a mesh is a pure function of its params,
//! so subdivision here is a *closed form*: build one fixed 8-chord polyline per
//! segment ([`ArcTable`]), divide its length by the requested interval, round
//! up, and scale by [`Quality`]. That same table is then inverted to place the
//! samples by arc length rather than by bezier parameter — Godot's baked curves
//! are arc-length parameterized, and without it `interval` would not mean world
//! units. Same inputs, same vertex count, same positions, everywhere. The cost
//! is a fixed sub-percent spacing error on wildly curved segments, which is
//! invisible for level geometry.
//!
//! ## Frames
//!
//! Orientation uses **rotation-minimizing frames** (parallel transport): each
//! frame's normal is the previous frame's normal carried along by the minimal
//! rotation between the two tangents. The alternative — Frenet frames from the
//! curvature vector — flips 180° at inflection points and is undefined on
//! straight sections, which is exactly what a cliff-face sweep is made of.
//!
//! The seed normal is derived from the first tangent alone (the least-aligned
//! world axis), so it is deterministic rather than an author-supplied hint.
//!
//! Note that parallel transport around a *closed* curve does not generally
//! return to its starting normal (the holonomy of a non-planar loop is
//! non-zero). For the planar paths in the source scene it does, and a residual
//! twist at the seam is a texture artifact rather than a hole, so no closing
//! correction is applied.

use glam::{Quat, Vec3};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::Quality;

/// Hard ceiling on samples per segment, so a pathological
/// `interval` (or a curve authored in millimetres) cannot allocate unbounded.
const MAX_SUBDIVS: u32 = 4096;

/// One control point of a [`Curve3`].
///
/// Handles are stored **relative to `pos`**, matching Godot's `Curve3D`
/// `in`/`out` convention, so a converter can copy the scene values through
/// unchanged. `tilt` is a roll about the tangent in radians (Godot stores
/// degrees on `Path3D`'s inspector but radians in the resource).
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CurvePoint {
    pub pos: Vec3,
    pub in_handle: Vec3,
    pub out_handle: Vec3,
    pub tilt: f32,
}

impl CurvePoint {
    /// A corner point: zero handles (so both adjacent segments are straight
    /// lines) and no tilt.
    pub fn at(pos: Vec3) -> Self {
        Self {
            pos,
            in_handle: Vec3::ZERO,
            out_handle: Vec3::ZERO,
            tilt: 0.0,
        }
    }
}

/// A cubic-bezier polyline in 3D. `closed` adds a final segment from the last
/// point back to the first (Godot's `Curve3D` closes when the endpoints
/// coincide; the converter decides and sets this flag explicitly).
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Curve3 {
    pub points: Vec<CurvePoint>,
    pub closed: bool,
}

/// A position on the curve plus an orthonormal right-handed basis.
///
/// `binormal × normal = tangent`, i.e. the triple `(binormal, normal, tangent)`
/// behaves like `(X, Y, Z)`. A cross-section polygon laid out with `x → binormal`
/// and `y → normal` therefore keeps its counter-clockwise winding facing along
/// `+tangent`, which is what makes swept side walls come out facing outward.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathFrame {
    pub pos: Vec3,
    pub tangent: Vec3,
    pub normal: Vec3,
    pub binormal: Vec3,
}

impl Curve3 {
    /// Sample the curve into frames roughly `interval` world units apart.
    ///
    /// `interval` is a target, not a guarantee: each segment is divided into a
    /// whole number of steps, so spacing varies segment to segment (this matches
    /// `CSGPolygon3D`'s `path_interval` in `PATH_INTERVAL_DISTANCE` mode).
    /// `quality` scales the step count per LOD tier via [`Quality::segs`], with
    /// a floor of one step per segment.
    ///
    /// The returned sequence contains no duplicate endpoints: for an open curve
    /// it runs from the first point to the last inclusive; for a closed curve it
    /// stops one step short of wrapping, so a consumer joins the last frame back
    /// to the first itself.
    pub fn frames(&self, interval: f32, quality: Quality) -> Vec<PathFrame> {
        let n = self.points.len();
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            let p = self.points[0];
            return vec![frame_from(p.pos, Vec3::Z, seed_normal(Vec3::Z), p.tilt)];
        }
        let interval = interval.max(1.0e-4);

        // --- sample positions, raw tangents and tilts -----------------------
        let seg_count = if self.closed { n } else { n - 1 };
        let mut pos: Vec<Vec3> = Vec::new();
        let mut tan: Vec<Vec3> = Vec::new();
        let mut tilt: Vec<f32> = Vec::new();
        for s in 0..seg_count {
            let a = self.points[s];
            let b = self.points[(s + 1) % n];
            let (p0, c0, c1, p1) = handles(a, b);
            let arc = ArcTable::build(p0, c0, c1, p1);
            let subdivs = quality
                .segs(steps_for(arc.length(), interval), 1)
                .min(MAX_SUBDIVS);
            for j in 0..subdivs {
                let t = arc.t_at(j as f32 / subdivs as f32);
                pos.push(bezier(p0, c0, c1, p1, t));
                tan.push(bezier_tangent(p0, c0, c1, p1, t));
                tilt.push(a.tilt + (b.tilt - a.tilt) * t);
            }
        }
        if !self.closed {
            // Open curves include the final control point; closed ones would
            // duplicate frame 0 there.
            let a = self.points[n - 2];
            let b = self.points[n - 1];
            let (p0, c0, c1, p1) = handles(a, b);
            pos.push(p1);
            tan.push(bezier_tangent(p0, c0, c1, p1, 1.0));
            tilt.push(b.tilt);
        }

        // --- normalize tangents, reusing the previous one where degenerate --
        // Repeated control points (common in hand-authored Godot curves) give a
        // zero derivative; carrying the last good direction keeps the frame
        // basis orthonormal instead of collapsing the ring to a point.
        let seed_tan = tan
            .iter()
            .find(|t| t.length_squared() > 1.0e-12)
            .map(|t| t.normalize())
            .unwrap_or(Vec3::Z);
        let mut prev = seed_tan;
        for t in &mut tan {
            if t.length_squared() > 1.0e-12 {
                prev = t.normalize();
            }
            *t = prev;
        }

        // --- parallel transport ---------------------------------------------
        let mut normal = seed_normal(tan[0]);
        let mut out = Vec::with_capacity(pos.len());
        for i in 0..pos.len() {
            if i > 0 {
                normal = min_rotation(tan[i - 1], tan[i]) * normal;
            }
            // Re-project every step: transport accumulates drift out of the
            // plane perpendicular to the tangent otherwise.
            normal = (normal - tan[i] * normal.dot(tan[i])).normalize_or_zero();
            if normal == Vec3::ZERO {
                normal = seed_normal(tan[i]);
            }
            out.push(frame_from(pos[i], tan[i], normal, tilt[i]));
        }
        out
    }

    /// Total length of the 8-sample-per-segment estimate polyline — the same
    /// number [`Curve3::frames`] divides by `interval`. Useful to a converter
    /// sizing UVs without re-sampling.
    pub fn approx_length(&self) -> f32 {
        let n = self.points.len();
        if n < 2 {
            return 0.0;
        }
        let seg_count = if self.closed { n } else { n - 1 };
        (0..seg_count)
            .map(|s| {
                let (p0, c0, c1, p1) = handles(self.points[s], self.points[(s + 1) % n]);
                ArcTable::build(p0, c0, c1, p1).length()
            })
            .sum()
    }
}

/// Fixed-resolution arc-length table for one bezier segment: `SAMPLES` chords,
/// cumulative length at each knot.
///
/// It does double duty. Its total is the segment's length estimate (how many
/// steps the segment earns), and inverting it turns an arc-length fraction back
/// into a bezier parameter — which is what makes `interval` mean *world units*
/// the way Godot's `path_interval` does. Sampling uniformly in `t` instead would
/// bunch rings toward the ends of every zero-handle segment, because a cubic
/// with coincident handles eases in and out along its own chord.
///
/// The resolution is a constant and the inversion is a bounded linear scan, so
/// this stays a pure function of the control points — no error-driven
/// refinement, which is the property that lets two machines agree on a mesh.
/// The price is a fixed sub-percent spacing wobble on strongly curved segments,
/// which is below what any level geometry notices.
struct ArcTable {
    cum: [f32; SAMPLES + 1],
}

const SAMPLES: usize = 8;

impl ArcTable {
    fn build(p0: Vec3, c0: Vec3, c1: Vec3, p1: Vec3) -> Self {
        let mut cum = [0.0f32; SAMPLES + 1];
        let mut prev = p0;
        for k in 1..=SAMPLES {
            let p = bezier(p0, c0, c1, p1, k as f32 / SAMPLES as f32);
            cum[k] = cum[k - 1] + (p - prev).length();
            prev = p;
        }
        Self { cum }
    }

    fn length(&self) -> f32 {
        self.cum[SAMPLES]
    }

    /// Bezier parameter at arc-length fraction `f` (clamped to `0..=1`).
    fn t_at(&self, f: f32) -> f32 {
        let total = self.length();
        if total <= 0.0 || !total.is_finite() {
            return f.clamp(0.0, 1.0);
        }
        let s = f.clamp(0.0, 1.0) * total;
        let mut k = 0;
        while k + 1 < SAMPLES && self.cum[k + 1] < s {
            k += 1;
        }
        let span = self.cum[k + 1] - self.cum[k];
        let local = if span > 0.0 { (s - self.cum[k]) / span } else { 0.0 };
        (k as f32 + local) / SAMPLES as f32
    }
}

/// Bezier control quadruple for the segment `a -> b`, with Godot's relative
/// handle convention resolved to absolute control points.
fn handles(a: CurvePoint, b: CurvePoint) -> (Vec3, Vec3, Vec3, Vec3) {
    (a.pos, a.pos + a.out_handle, b.pos + b.in_handle, b.pos)
}

/// Build a frame, applying `tilt` as a roll about the tangent.
///
/// The roll is applied here, to the *output* only — the caller keeps carrying
/// the untilted normal — so an author's tilt never accumulates into the
/// transport chain and cannot spiral a long path.
fn frame_from(pos: Vec3, tangent: Vec3, normal: Vec3, tilt: f32) -> PathFrame {
    let binormal = normal.cross(tangent);
    if tilt == 0.0 {
        return PathFrame {
            pos,
            tangent,
            normal,
            binormal,
        };
    }
    let roll = Quat::from_axis_angle(tangent, tilt);
    PathFrame {
        pos,
        tangent,
        normal: roll * normal,
        binormal: roll * binormal,
    }
}

fn steps_for(est: f32, interval: f32) -> u32 {
    if !est.is_finite() || est <= 0.0 {
        return 1;
    }
    ((est / interval).ceil().max(1.0) as u32).min(MAX_SUBDIVS)
}

fn bezier(p0: Vec3, c0: Vec3, c1: Vec3, p1: Vec3, t: f32) -> Vec3 {
    let u = 1.0 - t;
    p0 * (u * u * u) + c0 * (3.0 * u * u * t) + c1 * (3.0 * u * t * t) + p1 * (t * t * t)
}

fn bezier_tangent(p0: Vec3, c0: Vec3, c1: Vec3, p1: Vec3, t: f32) -> Vec3 {
    let u = 1.0 - t;
    (c0 - p0) * (3.0 * u * u) + (c1 - c0) * (6.0 * u * t) + (p1 - c1) * (3.0 * t * t)
}

/// The world axis least aligned with `t`, ties broken X < Y < Z. Deterministic
/// by construction — no "pick any perpendicular" branch.
fn least_aligned_axis(t: Vec3) -> Vec3 {
    let a = t.abs();
    if a.x <= a.y && a.x <= a.z {
        Vec3::X
    } else if a.y <= a.z {
        Vec3::Y
    } else {
        Vec3::Z
    }
}

/// Seed normal for the transport chain: the least-aligned world axis projected
/// perpendicular to the tangent.
fn seed_normal(t: Vec3) -> Vec3 {
    let axis = least_aligned_axis(t);
    (axis - t * t.dot(axis)).normalize_or(Vec3::Y)
}

/// Minimal rotation taking unit `a` onto unit `b`. Hand-rolled rather than
/// `Quat::from_rotation_arc` so the antiparallel case picks a *documented*
/// deterministic axis instead of an implementation-defined one.
fn min_rotation(a: Vec3, b: Vec3) -> Quat {
    let d = a.dot(b).clamp(-1.0, 1.0);
    if d > 1.0 - 1.0e-7 {
        return Quat::IDENTITY;
    }
    if d < -1.0 + 1.0e-7 {
        let axis = a.cross(least_aligned_axis(a)).normalize_or(Vec3::Y);
        return Quat::from_axis_angle(axis, std::f32::consts::PI);
    }
    Quat::from_axis_angle(a.cross(b).normalize(), d.acos())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, PI};

    fn line(from: Vec3, to: Vec3) -> Curve3 {
        Curve3 {
            points: vec![CurvePoint::at(from), CurvePoint::at(to)],
            closed: false,
        }
    }

    /// A closed square path in the XZ plane, side `2 * half`.
    fn square_path(half: f32) -> Curve3 {
        Curve3 {
            points: vec![
                CurvePoint::at(Vec3::new(-half, 0.0, -half)),
                CurvePoint::at(Vec3::new(half, 0.0, -half)),
                CurvePoint::at(Vec3::new(half, 0.0, half)),
                CurvePoint::at(Vec3::new(-half, 0.0, half)),
            ],
            closed: true,
        }
    }

    #[test]
    fn straight_line_gives_constant_frames() {
        let f = line(Vec3::ZERO, Vec3::new(6.0, 0.0, 0.0)).frames(2.0, Quality::FULL);
        assert_eq!(f.len(), 4, "3 steps + the terminating point");
        for w in f.windows(2) {
            assert!((w[0].tangent - w[1].tangent).length() < 1e-6);
            assert!((w[0].normal - w[1].normal).length() < 1e-6);
            assert!((w[0].binormal - w[1].binormal).length() < 1e-6);
        }
        // Tangent +X -> least-aligned axis is +Y (tie X-before-Z on |dot| = 0).
        assert!((f[0].tangent - Vec3::X).length() < 1e-6);
        assert!((f[0].normal - Vec3::Y).length() < 1e-6);
        assert!((f[0].binormal - Vec3::NEG_Z).length() < 1e-6);
        // Evenly spaced along the line: the arc-length table makes `interval`
        // mean world units even though a zero-handle cubic eases in `t`. The
        // slack is the table's fixed resolution, not an adaptive tolerance.
        for (i, fr) in f.iter().enumerate() {
            assert!((fr.pos.x - i as f32 * 2.0).abs() < 0.02, "{:?}", fr.pos);
        }
    }

    #[test]
    fn frames_form_a_right_handed_orthonormal_basis() {
        let mut c = square_path(5.0);
        c.points[1].out_handle = Vec3::new(0.0, 3.0, 2.0);
        c.points[2].in_handle = Vec3::new(-2.0, 1.0, 0.0);
        for f in c.frames(1.0, Quality::FULL) {
            assert!((f.tangent.length() - 1.0).abs() < 1e-4);
            assert!((f.normal.length() - 1.0).abs() < 1e-4);
            assert!((f.binormal.length() - 1.0).abs() < 1e-4);
            assert!(f.tangent.dot(f.normal).abs() < 1e-4);
            assert!(f.tangent.dot(f.binormal).abs() < 1e-4);
            assert!(f.normal.dot(f.binormal).abs() < 1e-4);
            // binormal x normal = tangent  (i.e. X x Y = Z)
            assert!((f.binormal.cross(f.normal) - f.tangent).length() < 1e-4);
        }
    }

    #[test]
    fn tilt_rolls_the_normal_about_the_tangent() {
        let mut c = line(Vec3::ZERO, Vec3::new(4.0, 0.0, 0.0));
        for p in &mut c.points {
            p.tilt = FRAC_PI_2;
        }
        let f = c.frames(2.0, Quality::FULL);
        // Untilted the frame is (t=+X, n=+Y, b=-Z); rolling +90° about +X takes
        // +Y to +Z.
        for fr in &f {
            assert!((fr.normal - Vec3::Z).length() < 1e-5, "normal {:?}", fr.normal);
            assert!((fr.binormal - Vec3::Y).length() < 1e-5);
            assert!((fr.tangent - Vec3::X).length() < 1e-6);
        }
        // Tilt is interpolated along the segment, not snapped per point.
        let mut ramp = line(Vec3::ZERO, Vec3::new(4.0, 0.0, 0.0));
        ramp.points[1].tilt = PI;
        let r = ramp.frames(1.0, Quality::FULL);
        assert!(r[0].normal.dot(Vec3::Y) > 0.999);
        assert!(r[r.len() - 1].normal.dot(Vec3::Y) < -0.999);
    }

    #[test]
    fn closed_curve_stops_one_step_short_of_wrapping() {
        let f = square_path(3.0).frames(2.0, Quality::FULL);
        // Four 6-unit sides, 3 steps each, no duplicated seam point.
        assert_eq!(f.len(), 12);
        assert!((f[0].pos - Vec3::new(-3.0, 0.0, -3.0)).length() < 1e-5);
        assert!((f[f.len() - 1].pos - Vec3::new(-3.0, 0.0, -1.0)).length() < 0.02);
    }

    #[test]
    fn planar_closed_path_transports_without_twist() {
        // Every tangent lies in XZ, so every transport rotation is about Y and
        // the seed normal +Y survives the loop exactly.
        for f in square_path(4.0).frames(1.5, Quality::FULL) {
            assert!((f.normal - Vec3::Y).length() < 1e-5);
            assert!(f.tangent.y.abs() < 1e-6);
        }
    }

    #[test]
    fn quality_scales_step_count_deterministically() {
        let c = line(Vec3::ZERO, Vec3::new(8.0, 0.0, 0.0));
        assert_eq!(c.frames(1.0, Quality::FULL).len(), 9);
        assert_eq!(c.frames(1.0, Quality(0.5)).len(), 5);
        assert_eq!(c.frames(1.0, Quality(0.25)).len(), 3);
        // Floor of one step per segment however low the tier goes.
        assert_eq!(c.frames(1.0, Quality(0.0)).len(), 2);
        // Repeatable.
        let a: Vec<_> = c.frames(1.3, Quality(0.7)).iter().map(|f| f.pos).collect();
        for _ in 0..4 {
            let b: Vec<_> = c.frames(1.3, Quality(0.7)).iter().map(|f| f.pos).collect();
            assert_eq!(a, b);
        }
    }

    #[test]
    fn repeated_points_reuse_the_previous_tangent() {
        let c = Curve3 {
            points: vec![
                CurvePoint::at(Vec3::ZERO),
                CurvePoint::at(Vec3::new(4.0, 0.0, 0.0)),
                CurvePoint::at(Vec3::new(4.0, 0.0, 0.0)), // duplicate
                CurvePoint::at(Vec3::new(8.0, 0.0, 0.0)),
            ],
            closed: false,
        };
        let f = c.frames(2.0, Quality::FULL);
        assert!(!f.is_empty());
        for fr in &f {
            assert!((fr.tangent.length() - 1.0).abs() < 1e-4, "{:?}", fr.tangent);
            assert!(fr.normal.is_finite());
            assert!(fr.binormal.is_finite());
        }
    }

    #[test]
    fn curved_segment_subdivides_by_estimated_length() {
        // A bulging segment is longer than its chord, so it earns more steps.
        let chord = line(Vec3::ZERO, Vec3::new(4.0, 0.0, 0.0));
        let mut bulge = chord.clone();
        bulge.points[0].out_handle = Vec3::new(0.0, 6.0, 0.0);
        bulge.points[1].in_handle = Vec3::new(0.0, 6.0, 0.0);
        assert!(bulge.approx_length() > chord.approx_length() * 1.5);
        assert!(bulge.frames(1.0, Quality::FULL).len() > chord.frames(1.0, Quality::FULL).len());
    }

    #[test]
    fn empty_and_single_point_curves_are_safe() {
        assert!(Curve3::default().frames(1.0, Quality::FULL).is_empty());
        let one = Curve3 {
            points: vec![CurvePoint::at(Vec3::ONE)],
            closed: false,
        };
        let f = one.frames(1.0, Quality::FULL);
        assert_eq!(f.len(), 1);
        assert!((f[0].tangent.length() - 1.0).abs() < 1e-6);
    }
}
