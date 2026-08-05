//! Mesh operations. Each is a pure `fn(MeshData, ...) -> MeshData` so they
//! compose (the fluent methods on `MeshData` are thin sugar over these) and map
//! directly onto editor node-graph nodes later.
//!
//! ## Cleanup ops
//!
//! [`weld`], [`cull_slivers`] and [`decimate`] are the *render-side* cleanup
//! pass: they run on baked geometry (CSG level output above all) to buy back
//! the vertices and triangles a boolean pipeline necessarily wastes. They are
//! not run on collision geometry — the trimesh collider is built from the
//! unmodified boolean result, so nothing here can move a surface the player
//! stands on.
//!
//! All three are deterministic in the DESIGN §3/§4 sense: same input bytes give
//! the same output bytes on every platform. The two ingredients are (a) keys
//! that are quantized integers rather than floats, so no comparison can be
//! order-sensitive, and (b) no `HashMap`/`HashSet` *iteration* ever reaching the
//! output — maps are looked up, never walked. Any change to these ops that
//! alters output bytes must bump [`MESH_PIPELINE_VERSION`](crate::MESH_PIPELINE_VERSION),
//! because that constant salts the content cache's keys.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

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

// ---------------------------------------------------------------------------
// Cleanup: weld
// ---------------------------------------------------------------------------

/// Merge vertices that land in the same `eps`-sized cell of **all four**
/// attributes — position, normal, uv, colour — and rebuild the index buffer
/// around the survivors.
///
/// `eps` is one absolute tolerance applied to every attribute: world units for
/// positions, unit-vector components for normals, uv units for uvs, `0..1` for
/// colours. Requiring all four to agree is what makes this crease-safe — the two
/// corners a hard edge produced share a position but carry different normals, so
/// they never merge and the edge stays hard. [`creased_normals`] already welds
/// its own output on the same principle, so welding after it with a small `eps`
/// is a no-op rather than a re-smoothing.
///
/// Keying is a quantized grid (`round(v / eps)`), not true eps-ball clustering:
/// two vertices `eps/2` apart can still straddle a cell boundary and survive
/// separately. That is the deliberate trade — a grid cell is a pure function of
/// the vertex alone, so the result never depends on visit order, insertion
/// order, or how a spatial index happened to balance. Within a cell the
/// first-occurring vertex (lowest input index) wins and keeps its attributes.
///
/// Triangles that degenerate as a consequence (two corners merged into one, or
/// zero area afterwards) are dropped, and vertices left unreferenced are
/// compacted away.
///
/// The headline use is CSG output: [`Csg::into_mesh`](crate::Csg::into_mesh)
/// fan-triangulates every polygon into its own private vertices, so a level bake
/// arrives with roughly 3× the vertices it needs.
pub fn weld(mut mesh: MeshData, eps: f32) -> MeshData {
    fill_defaults(&mut mesh);
    let inv = 1.0 / (eps.abs() as f64).max(1.0e-9);
    let g = |v: f32| (v as f64 * inv).round() as i64;

    let mut cells: HashMap<[i64; 11], u32> = HashMap::new();
    let mut merged = MeshData {
        indices: Vec::with_capacity(mesh.indices.len()),
        ..MeshData::default()
    };
    let remap: Vec<u32> = (0..mesh.positions.len())
        .map(|i| {
            let (p, n) = (mesh.positions[i], mesh.normals[i]);
            let (uv, c) = (mesh.uvs[i], mesh.colors[i]);
            let key = [
                g(p.x),
                g(p.y),
                g(p.z),
                g(n.x),
                g(n.y),
                g(n.z),
                g(uv.x),
                g(uv.y),
                g(c.x),
                g(c.y),
                g(c.z),
            ];
            *cells.entry(key).or_insert_with(|| {
                let id = merged.positions.len() as u32;
                merged.positions.push(p);
                merged.normals.push(n);
                merged.uvs.push(uv);
                merged.colors.push(c);
                id
            })
        })
        .collect();

    for t in mesh.indices.chunks_exact(3) {
        let v = [
            remap[t[0] as usize],
            remap[t[1] as usize],
            remap[t[2] as usize],
        ];
        if v[0] == v[1] || v[1] == v[2] || v[0] == v[2] {
            continue; // the weld pinched this triangle shut
        }
        let cross = face_cross(
            merged.positions[v[0] as usize],
            merged.positions[v[1] as usize],
            merged.positions[v[2] as usize],
        );
        if cross.length_squared() < DEGENERATE_AREA_SQ {
            continue;
        }
        merged.indices.extend_from_slice(&v);
    }

    compact(merged)
}

// ---------------------------------------------------------------------------
// Cleanup: cull_slivers
// ---------------------------------------------------------------------------

/// Conservative default for [`cull_slivers`]' area gate, in square world units
/// (metres² for this engine): one square millimetre.
pub const SLIVER_MIN_AREA: f32 = 1.0e-6;

/// Conservative default for [`cull_slivers`]' aspect gate. `1.0` is equilateral;
/// `40.0` is a needle roughly 46× longer than it is wide.
pub const SLIVER_MAX_ASPECT: f32 = 40.0;

/// Drop triangles that are **both** smaller than `min_area` **and** more
/// needle-shaped than `max_aspect`.
///
/// The aspect metric is
///
/// ```text
/// aspect = longest_edge² / (4 · area / √3)
/// ```
///
/// i.e. the triangle's longest edge measured against the edge an *equilateral*
/// triangle of the same area would have. It is `1.0` for an equilateral
/// triangle, `~1.73` for the right isoceles halves a subdivided quad is made of,
/// and grows without bound as a triangle degenerates towards a segment. A
/// zero-area triangle is treated as infinitely bad and is always culled.
///
/// Both gates must fire, which is what keeps this from opening holes. A long
/// thin wall trim is needle-shaped but has real area, so it survives; a
/// legitimately tiny triangle (a detail facet on a small prop) is compact, so it
/// survives too. Only geometry that is simultaneously microscopic and
/// degenerate — the seam debris a BSP boolean leaves where two brush faces are
/// nearly, but not exactly, coincident — matches both, and by construction its
/// screen contribution is under a pixel at any camera distance where the rest of
/// the surface is still one.
///
/// [`SLIVER_MIN_AREA`] and [`SLIVER_MAX_ASPECT`] are the recommended settings.
/// Vertices left unreferenced are compacted away.
pub fn cull_slivers(mut mesh: MeshData, min_area: f32, max_aspect: f32) -> MeshData {
    fill_defaults(&mut mesh);
    let mut out = MeshData {
        indices: Vec::with_capacity(mesh.indices.len()),
        ..MeshData::default()
    };
    out.positions = mesh.positions;
    out.normals = mesh.normals;
    out.uvs = mesh.uvs;
    out.colors = mesh.colors;

    for t in mesh.indices.chunks_exact(3) {
        let (a, b, c) = (
            out.positions[t[0] as usize],
            out.positions[t[1] as usize],
            out.positions[t[2] as usize],
        );
        if is_sliver(a, b, c, min_area, max_aspect) {
            continue;
        }
        out.indices.extend_from_slice(t);
    }

    compact(out)
}

/// The [`cull_slivers`] predicate, split out so the doc'd metric has exactly one
/// implementation.
fn is_sliver(a: Vec3, b: Vec3, c: Vec3, min_area: f32, max_aspect: f32) -> bool {
    let area = face_cross(a, b, c).length() * 0.5;
    if area >= min_area {
        return false;
    }
    if area <= 0.0 {
        return true; // infinitely bad aspect
    }
    let longest = (b - a)
        .length_squared()
        .max((c - b).length_squared())
        .max((a - c).length_squared());
    let equilateral = 4.0 * area / 3.0f32.sqrt();
    longest / equilateral > max_aspect
}

// ---------------------------------------------------------------------------
// Cleanup: decimate (quadric error metric edge collapse)
// ---------------------------------------------------------------------------

/// Reduce triangle count by quadric-error-metric (QEM) half-edge collapse,
/// stopping once every remaining collapse would move the surface further than
/// `max_error` world units.
///
/// This is a solid basic Garland–Heckbert: each vertex accumulates the sum of
/// the fundamental error quadrics of its incident faces, and the cost of
/// collapsing `u` into `v` is `sqrt((Q_u + Q_v) evaluated at pos(v))` — the
/// RMS distance from `v`'s position to the planes `u` used to lie on. Placement
/// is *subset*, never optimal: the surviving vertex keeps its own position and
/// its own attributes verbatim. No 3×3 solve, no interpolated normals or UVs,
/// and every position in the output is a position from the input.
///
/// # Determinism
///
/// The hard requirement (DESIGN §3/§4). Three things buy it:
///
/// - **Integer priorities.** The heap is keyed on `(round(error · 1e7),
///   from_vertex, to_vertex)`, compared lexicographically. Error never reaches a
///   float comparison, so two collapses that are equally good are ordered by
///   vertex index, and *only* by vertex index. Duplicate entries for one
///   directed edge are interchangeable by construction, so `BinaryHeap`'s
///   unspecified tie order cannot be observed.
/// - **Index-keyed state.** Adjacency, quadrics, liveness and version stamps are
///   all `Vec`s indexed by vertex or triangle id. The only `HashMap` here is the
///   one [`weld`]-style edge tally used to classify boundaries, and it is looked
///   up, never iterated; edge classification itself runs over a *sorted* `Vec`.
/// - **f64 arithmetic.** Quadrics accumulate in `f64` in a fixed order, using
///   only `+`/`-`/`*` and `sqrt`, all of which IEEE-754 specifies exactly. No
///   transcendentals, no fast-math.
///
/// # What is preserved
///
/// A vertex is **locked** (may never be collapsed away; other vertices may still
/// collapse *into* it) if it touches an edge that is not shared by exactly two
/// triangles. That single rule covers both requirements:
///
/// - **Open borders.** A mesh boundary edge has one incident triangle, so the
///   whole border polyline is locked and comes through untouched — a decimated
///   plane keeps its four corners exactly.
/// - **Attribute seams.** Two vertices at one position carrying different
///   normals/UVs/colours are *different* vertices, so the edge between the
///   triangles on either side of the seam is two distinct one-triangle edges.
///   Seams are therefore boundaries in index space and are locked by the same
///   test. This is why [`weld`] must run *before* `decimate`: welding decides
///   what is a seam, and everything welded flat becomes free to simplify.
/// - Non-manifold edges (3+ triangles) lock too, so pathological input is
///   simplified around rather than corrupted.
///
/// Each accepted collapse additionally requires the edge to have exactly two
/// incident triangles, the two endpoints to share exactly two neighbours (the
/// link condition, which prevents pinching a tube shut), and no incident
/// triangle to flip its normal or go degenerate. Failing any of these skips the
/// collapse rather than aborting.
///
/// # Cost
///
/// Roughly linear in triangles with a log factor from the heap; a ~100k-triangle
/// CSG level bake decimates in well under a second in release. Bake-time only —
/// like everything in this crate it is never a per-frame operation.
pub fn decimate(mut mesh: MeshData, max_error: f32) -> MeshData {
    fill_defaults(&mut mesh);
    if mesh.indices.len() < 3 || !max_error.is_finite() || max_error <= 0.0 {
        return mesh;
    }

    let vcount = mesh.positions.len();

    // --- triangles (dropping anything already degenerate) ------------------
    let mut tris: Vec<[u32; 3]> = Vec::with_capacity(mesh.triangle_count());
    for t in mesh.indices.chunks_exact(3) {
        let v = [t[0], t[1], t[2]];
        if v[0] == v[1] || v[1] == v[2] || v[0] == v[2] {
            continue;
        }
        let cross = face_cross(
            mesh.positions[v[0] as usize],
            mesh.positions[v[1] as usize],
            mesh.positions[v[2] as usize],
        );
        if cross.length_squared() < DEGENERATE_AREA_SQ {
            continue;
        }
        tris.push(v);
    }
    if tris.is_empty() {
        return MeshData::default();
    }
    let mut tri_alive = vec![true; tris.len()];

    // --- locked vertices: anything touching a non-2-manifold edge ----------
    // Classified over a sorted Vec of undirected edges (no map iteration).
    let mut edges: Vec<(u32, u32)> = Vec::with_capacity(tris.len() * 3);
    for t in &tris {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            edges.push((a.min(b), a.max(b)));
        }
    }
    edges.sort_unstable();
    let mut locked = vec![false; vcount];
    let mut i = 0;
    while i < edges.len() {
        let mut j = i + 1;
        while j < edges.len() && edges[j] == edges[i] {
            j += 1;
        }
        if j - i != 2 {
            locked[edges[i].0 as usize] = true;
            locked[edges[i].1 as usize] = true;
        }
        i = j;
    }
    drop(edges);

    // --- per-vertex state --------------------------------------------------
    let mut vert_tris: Vec<Vec<u32>> = vec![Vec::new(); vcount];
    for (ti, t) in tris.iter().enumerate() {
        for &v in t {
            vert_tris[v as usize].push(ti as u32);
        }
    }
    let mut alive: Vec<bool> = vert_tris.iter().map(|t| !t.is_empty()).collect();
    let mut version = vec![0u32; vcount];

    let mut quadrics = vec![Quadric::ZERO; vcount];
    for t in &tris {
        let q = Quadric::from_face(
            mesh.positions[t[0] as usize],
            mesh.positions[t[1] as usize],
            mesh.positions[t[2] as usize],
        );
        for &v in t {
            quadrics[v as usize] = quadrics[v as usize].plus(&q);
        }
    }

    // --- seed the queue ----------------------------------------------------
    let mut heap: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
    let mut state = Collapser {
        positions: &mesh.positions,
        tris: &mut tris,
        tri_alive: &mut tri_alive,
        vert_tris: &mut vert_tris,
        alive: &mut alive,
        locked: &locked,
        quadrics: &mut quadrics,
        version: &mut version,
    };
    for t in 0..state.tris.len() {
        let tri = state.tris[t];
        for k in 0..3 {
            let (a, b) = (tri[k], tri[(k + 1) % 3]);
            state.offer(a, b, &mut heap);
            state.offer(b, a, &mut heap);
        }
    }

    // --- collapse ----------------------------------------------------------
    let stop = quantize_error(max_error as f64);
    while let Some(Reverse(c)) = heap.pop() {
        if c.key > stop {
            break; // every remaining candidate costs at least this much
        }
        if state.version[c.from as usize] != c.from_version
            || state.version[c.to as usize] != c.to_version
        {
            continue; // stale: a neighbouring collapse moved the goalposts
        }
        if !state.legal(c.from, c.to) {
            continue;
        }
        state.collapse(c.from, c.to);
        // Re-offer everything in the new one-ring: quadrics changed at `to`, and
        // legality may have changed for the edges opposite it.
        let ring = state.vert_tris[c.to as usize].clone();
        for ti in ring {
            let tri = state.tris[ti as usize];
            for k in 0..3 {
                let (a, b) = (tri[k], tri[(k + 1) % 3]);
                state.offer(a, b, &mut heap);
                state.offer(b, a, &mut heap);
            }
        }
    }

    // --- rebuild -----------------------------------------------------------
    let mut out = MeshData::default();
    let mut remap = vec![u32::MAX; vcount];
    for (ti, t) in tris.iter().enumerate() {
        if !tri_alive[ti] {
            continue;
        }
        for &v in t {
            let slot = &mut remap[v as usize];
            if *slot == u32::MAX {
                *slot = out.positions.len() as u32;
                out.positions.push(mesh.positions[v as usize]);
                out.normals.push(mesh.normals[v as usize]);
                out.uvs.push(mesh.uvs[v as usize]);
                out.colors.push(mesh.colors[v as usize]);
            }
            out.indices.push(*slot);
        }
    }
    out
}

/// Error grid for the collapse queue's integer key: 1e-7 world units, i.e. a
/// tenth of a micron. Fine enough that no visually distinct collapse ties, coarse
/// enough that the last bits of an `f64` cannot reorder the queue.
const ERROR_GRID: f64 = 1.0e7;

fn quantize_error(distance: f64) -> i64 {
    (distance.clamp(0.0, 1.0e11) * ERROR_GRID).round() as i64
}

/// A queued half-edge collapse `from -> to`. The derived `Ord` is exactly the
/// documented tie-break: quantized error first, then the lower source index,
/// then the target index. The version stamps ride along so a stale entry is
/// recognised on pop instead of needing deletion from the heap.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Candidate {
    key: i64,
    from: u32,
    to: u32,
    from_version: u32,
    to_version: u32,
}

/// The mutable half of [`decimate`], grouped so the legality checks and the
/// collapse itself can share it without threading eight arguments each.
struct Collapser<'a> {
    positions: &'a [Vec3],
    tris: &'a mut Vec<[u32; 3]>,
    tri_alive: &'a mut Vec<bool>,
    vert_tris: &'a mut Vec<Vec<u32>>,
    alive: &'a mut Vec<bool>,
    locked: &'a [bool],
    quadrics: &'a mut Vec<Quadric>,
    version: &'a mut Vec<u32>,
}

impl Collapser<'_> {
    /// The vertices sharing a live triangle with `v`, ascending and deduped.
    fn neighbours(&self, v: u32) -> Vec<u32> {
        let mut out = Vec::with_capacity(8);
        for &ti in &self.vert_tris[v as usize] {
            if !self.tri_alive[ti as usize] {
                continue;
            }
            for &x in &self.tris[ti as usize] {
                if x != v {
                    out.push(x);
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Live triangles containing both `a` and `b`.
    fn shared_tris(&self, a: u32, b: u32) -> usize {
        self.vert_tris[a as usize]
            .iter()
            .filter(|&&ti| self.tri_alive[ti as usize] && self.tris[ti as usize].contains(&b))
            .count()
    }

    /// May `from` be collapsed onto `to`? See [`decimate`]'s docs for the rules.
    fn legal(&self, from: u32, to: u32) -> bool {
        if from == to
            || !self.alive[from as usize]
            || !self.alive[to as usize]
            || self.locked[from as usize]
        {
            return false;
        }
        if self.shared_tris(from, to) != 2 {
            return false;
        }
        // Link condition: exactly the two triangles' opposite corners in common.
        let (na, nb) = (self.neighbours(from), self.neighbours(to));
        let mut common = 0;
        let (mut i, mut j) = (0, 0);
        while i < na.len() && j < nb.len() {
            match na[i].cmp(&nb[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    common += 1;
                    i += 1;
                    j += 1;
                }
            }
        }
        if common != 2 {
            return false;
        }
        // No incident triangle may fold over or collapse to a segment.
        let target = self.positions[to as usize];
        for &ti in &self.vert_tris[from as usize] {
            if !self.tri_alive[ti as usize] {
                continue;
            }
            let t = self.tris[ti as usize];
            if t.contains(&to) {
                continue; // this one disappears
            }
            let p = |v: u32| {
                if v == from {
                    target
                } else {
                    self.positions[v as usize]
                }
            };
            let before = face_cross(
                self.positions[t[0] as usize],
                self.positions[t[1] as usize],
                self.positions[t[2] as usize],
            );
            let after = face_cross(p(t[0]), p(t[1]), p(t[2]));
            if after.length_squared() < DEGENERATE_AREA_SQ || after.dot(before) <= 0.0 {
                return false;
            }
        }
        true
    }

    /// Cost of `from -> to`: RMS distance from `to`'s position to the planes the
    /// two vertices between them are responsible for.
    fn cost(&self, from: u32, to: u32) -> f64 {
        let q = self.quadrics[from as usize].plus(&self.quadrics[to as usize]);
        q.error(self.positions[to as usize]).max(0.0).sqrt()
    }

    /// Queue `from -> to` if it is currently legal.
    fn offer(&self, from: u32, to: u32, heap: &mut BinaryHeap<Reverse<Candidate>>) {
        if !self.legal(from, to) {
            return;
        }
        heap.push(Reverse(Candidate {
            key: quantize_error(self.cost(from, to)),
            from,
            to,
            from_version: self.version[from as usize],
            to_version: self.version[to as usize],
        }));
    }

    fn collapse(&mut self, from: u32, to: u32) {
        self.alive[from as usize] = false;
        let moved = std::mem::take(&mut self.vert_tris[from as usize]);
        for ti in moved {
            if !self.tri_alive[ti as usize] {
                continue;
            }
            let t = &mut self.tris[ti as usize];
            if t.contains(&to) {
                self.tri_alive[ti as usize] = false;
                continue;
            }
            for slot in t.iter_mut() {
                if *slot == from {
                    *slot = to;
                }
            }
            self.vert_tris[to as usize].push(ti);
        }
        let tri_alive = &self.tri_alive;
        self.vert_tris[to as usize].retain(|&ti| tri_alive[ti as usize]);
        self.quadrics[to as usize] = self.quadrics[to as usize].plus(&self.quadrics[from as usize]);
        self.version[from as usize] += 1;
        self.version[to as usize] += 1;
    }
}

/// A Garland–Heckbert fundamental error quadric: the symmetric 4×4 `K_p` stored
/// as its ten distinct entries, in `f64` so that summing a few dozen of them
/// does not lose the small residuals the ordering depends on.
#[derive(Clone, Copy)]
struct Quadric([f64; 10]);

impl Quadric {
    const ZERO: Quadric = Quadric([0.0; 10]);

    /// `K_p` for the plane through `a`, `b`, `c`, area-weighted (the classic
    /// weighting: a big face should dominate a sliver sharing the vertex).
    fn from_face(a: Vec3, b: Vec3, c: Vec3) -> Quadric {
        let cross = face_cross(a, b, c);
        let len = cross.length() as f64;
        if len <= 0.0 {
            return Quadric::ZERO;
        }
        let (x, y, z) = (
            cross.x as f64 / len,
            cross.y as f64 / len,
            cross.z as f64 / len,
        );
        let d = -(x * a.x as f64 + y * a.y as f64 + z * a.z as f64);
        let w = len * 0.5; // twice-area / 2 = area
        Quadric([
            w * x * x,
            w * x * y,
            w * x * z,
            w * x * d,
            w * y * y,
            w * y * z,
            w * y * d,
            w * z * z,
            w * z * d,
            w * d * d,
        ])
    }

    fn plus(&self, other: &Quadric) -> Quadric {
        let mut out = self.0;
        for (o, b) in out.iter_mut().zip(&other.0) {
            *o += *b;
        }
        Quadric(out)
    }

    /// `pᵀ Q p` for `p = (x, y, z, 1)`.
    fn error(&self, p: Vec3) -> f64 {
        let (x, y, z) = (p.x as f64, p.y as f64, p.z as f64);
        let q = &self.0;
        q[0] * x * x
            + 2.0 * q[1] * x * y
            + 2.0 * q[2] * x * z
            + 2.0 * q[3] * x
            + q[4] * y * y
            + 2.0 * q[5] * y * z
            + 2.0 * q[6] * y
            + q[7] * z * z
            + 2.0 * q[8] * z
            + q[9]
    }
}

/// Drop vertices no triangle references, keeping the order of the survivors.
fn compact(mesh: MeshData) -> MeshData {
    let mut used = vec![false; mesh.positions.len()];
    for &i in &mesh.indices {
        used[i as usize] = true;
    }
    if used.iter().all(|&u| u) {
        return mesh;
    }
    let mut remap = vec![0u32; mesh.positions.len()];
    let mut out = MeshData {
        indices: mesh.indices,
        ..MeshData::default()
    };
    for (i, _) in used.iter().enumerate().filter(|(_, &u)| u) {
        remap[i] = out.positions.len() as u32;
        out.positions.push(mesh.positions[i]);
        out.normals.push(mesh.normals[i]);
        out.uvs.push(mesh.uvs[i]);
        out.colors.push(mesh.colors[i]);
    }
    for i in &mut out.indices {
        *i = remap[*i as usize];
    }
    out
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{primitives, Csg};
    use glam::Vec2;

    /// Every undirected edge shared by exactly two triangles, over positions
    /// welded on a 1e-4 grid — the same closed-surface test `csg` and `extrude`
    /// use, repeated here so a cleanup op cannot quietly open a hole.
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
                if x == y {
                    continue;
                }
                *counts.entry((x.min(y), x.max(y))).or_insert(0) += 1;
            }
        }
        counts.values().all(|&c| c == 2)
    }

    fn assert_well_formed(m: &MeshData) {
        let n = m.positions.len();
        assert_eq!(m.normals.len(), n, "normals sized");
        assert_eq!(m.uvs.len(), n, "uvs sized");
        assert_eq!(m.colors.len(), n, "colors sized");
        assert_eq!(m.indices.len() % 3, 0, "whole triangles");
        for &i in &m.indices {
            assert!((i as usize) < n, "index in range");
        }
    }

    /// The two operands of the canonical union: `[-1, 1]³` and `[0, 2]³`.
    fn corner_union() -> MeshData {
        Csg::from_mesh(&primitives::cube(2.0), 0)
            .union(Csg::from_mesh(
                &primitives::cube(2.0).translate(Vec3::splat(1.0)),
                1,
            ))
            .into_mesh()
    }

    // --- weld --------------------------------------------------------------

    #[test]
    fn weld_after_creased_normals_changes_nothing() {
        // `creased_normals` already welds on all four attributes; a finer weld
        // on top of it must find nothing left to merge. If this ever fails, the
        // two ops disagree about what "the same vertex" means, and the crease
        // story is the first casualty.
        for m in [
            primitives::cube(2.0).smooth_normals(30.0),
            primitives::uv_sphere(1.0, 12, 18).smooth_normals(180.0),
            primitives::torus(1.0, 0.3, 16, 10).smooth_normals(45.0),
        ] {
            let before = m.clone();
            let after = m.weld(1.0e-6);
            assert_eq!(after.vertex_count(), before.vertex_count(), "no verts merged");
            assert_eq!(after.triangle_count(), before.triangle_count(), "no tris lost");
            assert_eq!(after.content_hash(), before.content_hash(), "identical bytes");
        }
    }

    #[test]
    fn weld_keeps_a_hard_edge_hard() {
        // A cube's 24 corners are 8 positions × 3 face normals. Welding on
        // position alone would leave 8 and round the cube off; welding on all
        // four attributes must leave all 24.
        let cube = primitives::cube(2.0);
        assert_eq!(cube.vertex_count(), 24);
        let welded = cube.weld(1.0e-4);
        assert_eq!(welded.vertex_count(), 24, "creased corners are not the same vertex");
        for n in &welded.normals {
            assert!(n.abs().max_element() > 0.99, "normal stayed axis-aligned: {n:?}");
        }
    }

    #[test]
    fn weld_collapses_the_csg_fan() {
        // `into_mesh` gives every polygon its own copy of every corner. Welding
        // is the whole reason a level bake is affordable, so assert the size of
        // the win, not just its direction.
        let raw = corner_union();
        assert!(is_closed_surface(&raw), "the union starts watertight");

        let welded = raw.clone().weld(1.0e-4);
        assert_well_formed(&welded);
        welded.validate();
        assert!(
            welded.vertex_count() * 5 < raw.vertex_count() * 3,
            "expected >40% fewer vertices, {} -> {}",
            raw.vertex_count(),
            welded.vertex_count()
        );
        assert_eq!(
            welded.triangle_count(),
            raw.triangle_count(),
            "a union has no degenerate faces to drop"
        );
        assert!(is_closed_surface(&welded), "welding opened the surface");
        assert_eq!(welded.bounds(), raw.bounds(), "welding is not a deformation");

        // Determinism: the grid key depends on the vertex, never on visit order.
        assert_eq!(corner_union().weld(1.0e-4).content_hash(), welded.content_hash());
    }

    #[test]
    fn weld_drops_the_triangles_it_pinches_shut() {
        // Two corners a hair apart, welded together: the triangle has to go, and
        // the vertex it left behind with it.
        let mut m = MeshData {
            positions: vec![
                Vec3::ZERO,
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(1.0, 0.0, 1.0e-7),
            ],
            indices: vec![0, 1, 2, 1, 3, 2],
            ..MeshData::default()
        };
        fill_defaults(&mut m);
        let welded = weld(m, 1.0e-4);
        assert_eq!(welded.vertex_count(), 3, "the near-duplicate merged");
        assert_eq!(welded.triangle_count(), 1, "the pinched triangle went away");
    }

    // --- cull_slivers ------------------------------------------------------

    #[test]
    fn cull_slivers_leaves_a_clean_grid_alone() {
        // The op must be a no-op on sane geometry — not "nearly", byte-for-byte,
        // because it runs unconditionally in the level bake.
        for m in [
            primitives::plane(Vec2::splat(4.0), 8),
            primitives::cube(2.0),
            primitives::uv_sphere(1.0, 16, 24).smooth_normals(180.0),
            corner_union().weld(1.0e-4),
        ] {
            let culled = m.clone().cull_slivers(SLIVER_MIN_AREA, SLIVER_MAX_ASPECT);
            assert_eq!(culled.triangle_count(), m.triangle_count(), "no triangle lost");
            assert_eq!(culled.content_hash(), m.content_hash(), "identical bytes");
        }
    }

    #[test]
    fn cull_slivers_needs_both_gates() {
        let tiny_needle = [
            Vec3::ZERO,
            Vec3::new(1.0e-3, 0.0, 0.0),
            Vec3::new(0.5e-3, 0.0, 1.0e-8),
        ];
        let long_needle = [Vec3::ZERO, Vec3::new(4.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.01)];
        let tiny_compact = [
            Vec3::ZERO,
            Vec3::new(1.0e-3, 0.0, 0.0),
            Vec3::new(0.5e-3, 0.0, 0.87e-3),
        ];
        let assert_sliver = |t: [Vec3; 3], want: bool, what: &str| {
            assert_eq!(
                is_sliver(t[0], t[1], t[2], SLIVER_MIN_AREA, SLIVER_MAX_ASPECT),
                want,
                "{what}"
            );
        };
        assert_sliver(tiny_needle, true, "microscopic and degenerate: cull");
        assert_sliver(long_needle, false, "a needle with real area is wall trim: keep");
        assert_sliver(tiny_compact, false, "a tiny but well-shaped facet: keep");
        // Exactly zero area is the worst case and must not divide by zero.
        assert_sliver([Vec3::ZERO, Vec3::X, Vec3::X * 2.0], true, "collinear: cull");

        // …and the op actually removes it, taking the orphaned vertex with it.
        let mut m = MeshData {
            positions: vec![
                Vec3::ZERO,
                Vec3::X,
                Vec3::Z,
                tiny_needle[1] + Vec3::Z * 2.0,
                tiny_needle[2] + Vec3::Z * 2.0,
                Vec3::Z * 2.0,
            ],
            indices: vec![0, 1, 2, 5, 3, 4],
            ..MeshData::default()
        };
        fill_defaults(&mut m);
        let culled = m.cull_slivers(SLIVER_MIN_AREA, SLIVER_MAX_ASPECT);
        assert_eq!(culled.triangle_count(), 1);
        assert_eq!(culled.vertex_count(), 3, "orphaned vertices compacted away");
    }

    // --- decimate ----------------------------------------------------------

    #[test]
    fn decimate_is_deterministic() {
        // DESIGN §3/§4's hard requirement, and the reason the queue is keyed on
        // integers: identical input must give identical output *bytes*, not a
        // similar-looking mesh, or the content cache and the level bake
        // disagree about what a level is.
        let source = primitives::uv_sphere(1.0, 24, 32).weld(1.0e-5);
        let first = source.clone().decimate(0.01);
        for _ in 0..4 {
            let again = source.clone().decimate(0.01);
            assert_eq!(again.content_hash(), first.content_hash());
            assert_eq!(again, first, "bit-identical, not merely equivalent");
        }
        assert!(first.triangle_count() < source.triangle_count(), "it did something");
    }

    /// Content-hash pin for one canonical decimation, the same tripwire the
    /// sweeps and booleans carry: it moves if the quadrics, the tie-break, the
    /// legality rules or the output vertex ordering change. That is allowed —
    /// but it has to be a deliberate edit to this number *and* a bump of
    /// [`crate::MESH_PIPELINE_VERSION`], because a silent move invalidates every
    /// cached level while the cache goes on serving the old bytes.
    ///
    /// Same caveat as `csg` and `extrude`: `MeshData::content_hash` is built on
    /// `DefaultHasher`, whose algorithm the standard library may change between
    /// Rust releases. A toolchain bump moving this number is expected and is not
    /// a geometry regression — an unexplained move on a fixed toolchain is.
    #[test]
    fn canonical_decimation_hashes_stably() {
        let m = primitives::uv_sphere(1.0, 24, 32).weld(1.0e-5).decimate(0.01);
        assert_eq!(m.content_hash(), 3386611370233868525);
    }

    #[test]
    fn decimate_pays_for_itself_inside_the_error_budget() {
        const MAX_ERROR: f32 = 0.02;
        let source = primitives::uv_sphere(1.0, 32, 32);
        let out = source.clone().decimate(MAX_ERROR);
        assert_well_formed(&out);
        out.validate();

        let kept = out.triangle_count() as f32 / source.triangle_count() as f32;
        assert!(
            kept <= 0.70,
            "wanted >=30% fewer triangles, kept {kept:.3} ({} -> {})",
            source.triangle_count(),
            out.triangle_count()
        );

        // The error budget is a real bound on how far the silhouette moved.
        let (lo0, hi0) = source.bounds().unwrap();
        let (lo1, hi1) = out.bounds().unwrap();
        let drift = (lo1 - lo0).abs().max((hi1 - hi0).abs()).max_element();
        assert!(drift < MAX_ERROR, "bounds moved {drift}, budget {MAX_ERROR}");

        // Every surviving vertex is an *original* vertex (subset placement), so
        // decimation can never invent a position off the sphere.
        for p in &out.positions {
            assert!((p.length() - 1.0).abs() < 1.0e-5, "vertex left the sphere: {p:?}");
        }
    }

    #[test]
    fn decimate_preserves_an_open_border() {
        // A plane is flat, so QEM says every interior collapse is free and the
        // op will simplify as hard as the legality rules let it. What must not
        // move is the border: its edges have one triangle each, so every border
        // vertex is locked — corners included.
        let source = primitives::plane(Vec2::splat(4.0), 8);
        let out = source.clone().decimate(1.0);
        assert_well_formed(&out);
        assert!(out.triangle_count() < source.triangle_count(), "it did something");
        assert_eq!(out.bounds(), source.bounds(), "the outline is untouched");

        for corner in [
            Vec3::new(-2.0, 0.0, -2.0),
            Vec3::new(2.0, 0.0, -2.0),
            Vec3::new(-2.0, 0.0, 2.0),
            Vec3::new(2.0, 0.0, 2.0),
        ] {
            assert!(
                out.positions.iter().any(|p| p.distance(corner) < 1.0e-5),
                "corner {corner:?} was decimated away"
            );
        }
        // Every border vertex, in fact — 8 subdivisions = 32 of them.
        let on_border = |p: &Vec3| p.x.abs() > 2.0 - 1e-5 || p.z.abs() > 2.0 - 1e-5;
        assert_eq!(
            out.positions.iter().filter(|p| on_border(p)).count(),
            32,
            "the whole border polyline survives"
        );
        // …and the surface stays flat and single-sided.
        for [a, b, c] in out.triangles() {
            assert!(face_cross(a, b, c).y > 0.0, "a triangle flipped");
        }
    }

    #[test]
    fn decimate_leaves_a_creased_solid_alone() {
        // Every vertex of a welded cube is an attribute seam, and every seam is
        // a boundary in index space, so there is nothing legal to collapse. The
        // op must notice rather than round the corners off.
        let cube = primitives::cube(2.0);
        let out = cube.clone().decimate(10.0);
        assert_eq!(out.content_hash(), cube.content_hash(), "a cube is irreducible");
    }

    #[test]
    fn decimate_keeps_a_boolean_result_closed() {
        // The point of locking seams: a welded CSG solid is closed as a surface
        // but riddled with attribute seams, and every crease is a boundary in
        // index space. Simplifying the flat interiors between them must not open
        // the solid up — a hole in a level is a hole the player falls through.
        let solid = Csg::from_mesh(&primitives::cube(4.0), 0)
            .subtract(Csg::from_mesh(
                &primitives::cube(2.0).translate(Vec3::new(1.0, 1.0, 1.0)),
                1,
            ))
            .into_mesh()
            .weld(1.0e-4);
        assert!(is_closed_surface(&solid), "the notched box starts watertight");

        let out = solid.clone().decimate(0.05);
        assert_well_formed(&out);
        out.validate();
        assert!(out.triangle_count() < solid.triangle_count(), "it did something");
        assert!(is_closed_surface(&out), "decimation opened the solid");
        assert_eq!(out.bounds(), solid.bounds(), "the silhouette is untouched");
    }

    #[test]
    fn degenerate_input_does_not_panic() {
        assert!(weld(MeshData::default(), 1.0e-4).is_empty());
        assert!(cull_slivers(MeshData::default(), SLIVER_MIN_AREA, SLIVER_MAX_ASPECT).is_empty());
        assert!(decimate(MeshData::default(), 0.1).is_empty());
        // A zero/negative/NaN budget is a no-op, not a wipe.
        let cube = primitives::cube(1.0);
        assert_eq!(cube.clone().decimate(0.0), cube);
        assert_eq!(cube.clone().decimate(f32::NAN), cube);
    }

    // --- measurement (not a gate) ------------------------------------------

    /// Wall-clock and triangle counts for the cleanup pass on a level-sized CSG
    /// bake. Not an assertion — timings are not a property of the geometry — so
    /// it is `#[ignore]`d and meant to be run deliberately:
    ///
    /// ```text
    /// cargo test -p runt-mesh --release -- --ignored --nocapture bake
    /// ```
    #[test]
    #[ignore = "measurement, not a gate; run with --release --nocapture"]
    fn measure_cleanup_on_a_level_sized_bake() {
        use std::time::Instant;

        let t0 = Instant::now();
        // Shaped like the playground's brushwork, scaled up until the output is
        // level-sized: a big slab, a grid of shafts punched through it on a
        // diagonal (so every cut crosses several faces), a mound unioned on top
        // and a bowl carved back out of it.
        let slab = primitives::cube(1.0).scale(Vec3::new(40.0, 4.0, 40.0));
        let mut solid = Csg::from_mesh(&slab, 0);
        for i in 0..6 {
            for j in 0..6 {
                let at = Vec3::new(-15.0 + i as f32 * 6.0, 0.0, -15.0 + j as f32 * 6.0);
                let shaft = primitives::cube(1.0)
                    .scale(Vec3::new(2.5, 8.0, 2.5))
                    .rotate(glam::Quat::from_rotation_y(0.5))
                    .translate(at);
                solid = solid.subtract(Csg::from_mesh(&shaft, 1));
            }
        }
        let mound = primitives::uv_sphere(9.0, 32, 40).translate(Vec3::new(0.0, 1.0, 0.0));
        solid = solid.union(Csg::from_mesh(&mound, 2));
        let bowl = primitives::uv_sphere(6.0, 32, 40).translate(Vec3::new(6.0, 6.0, -6.0));
        solid = solid.subtract(Csg::from_mesh(&bowl, 3));
        let arch = primitives::torus(10.0, 1.5, 32, 16)
            .rotate(glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2))
            .translate(Vec3::new(-8.0, 2.0, 8.0));
        solid = solid.union(Csg::from_mesh(&arch, 4));
        let raw = solid.into_mesh();
        let t_csg = t0.elapsed();

        let t1 = Instant::now();
        let welded = raw.clone().weld(1.0e-4);
        let t_weld = t1.elapsed();

        let t2 = Instant::now();
        let culled = welded
            .clone()
            .cull_slivers(SLIVER_MIN_AREA, SLIVER_MAX_ASPECT);
        let t_cull = t2.elapsed();

        let t3 = Instant::now();
        let decimated = culled.clone().decimate(0.02);
        let t_decimate = t3.elapsed();

        let pct = |a: usize, b: usize| 100.0 - (b as f32 / a as f32) * 100.0;
        println!("\n--- cleanup on a level-sized CSG bake ---");
        println!(
            "csg bake      {:>8.1?}  {:>7} tris  {:>7} verts",
            t_csg,
            raw.triangle_count(),
            raw.vertex_count()
        );
        println!(
            "weld          {:>8.1?}  {:>7} tris  {:>7} verts  (-{:.1}% verts)",
            t_weld,
            welded.triangle_count(),
            welded.vertex_count(),
            pct(raw.vertex_count(), welded.vertex_count())
        );
        println!(
            "cull_slivers  {:>8.1?}  {:>7} tris  {:>7} verts  (-{:.1}% tris)",
            t_cull,
            culled.triangle_count(),
            culled.vertex_count(),
            pct(welded.triangle_count(), culled.triangle_count())
        );
        println!(
            "decimate      {:>8.1?}  {:>7} tris  {:>7} verts  (-{:.1}% tris)",
            t_decimate,
            decimated.triangle_count(),
            decimated.vertex_count(),
            pct(culled.triangle_count(), decimated.triangle_count())
        );
        println!(
            "cleanup total {:>8.1?}  {:.1}% of the raw triangles, {:.1}% of the raw vertices",
            t_weld + t_cull + t_decimate,
            100.0 - pct(raw.triangle_count(), decimated.triangle_count()),
            100.0 - pct(raw.vertex_count(), decimated.vertex_count())
        );

        println!("\n--- decimate budget sweep (from the welded+culled mesh) ---");
        for budget in [0.002f32, 0.005, 0.01, 0.02, 0.05] {
            let t = Instant::now();
            let d = culled.clone().decimate(budget);
            println!(
                "max_error {budget:>5}  {:>8.1?}  {:>7} tris  (-{:.1}%)",
                t.elapsed(),
                d.triangle_count(),
                pct(culled.triangle_count(), d.triangle_count())
            );
        }
        println!();
    }
}
