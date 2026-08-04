//! Constructive solid geometry: union, subtract, intersect over [`MeshData`].
//!
//! This is a faithful port of Evan Wallace's `csg.js` BSP algorithm, with the
//! per-triangle **material id** that the Godot level converter needs riding
//! along on every polygon. The playground level is ~113 CSG nodes (27 of them
//! non-union), authored as boxes, extrusions and lathes that cut into each
//! other; the bake produces one render mesh *per material* plus one merged mesh
//! for the trimesh collider, which is exactly the two shapes
//! [`Csg::into_meshes`] and [`Csg::into_mesh`] return.
//!
//! ## The algorithm
//!
//! A solid is a bag of convex polygons. Each boolean builds a BSP tree per
//! operand, clips each tree's polygons against the other, and re-builds one
//! tree from the survivors — `union`/`subtract`/`intersect` differ only in
//! where the `invert()` calls go (see the three methods; the sequences are
//! transcribed from `csg.js` unchanged). The splitter for a node is always the
//! **first polygon's plane**, never a "best split" heuristic: it costs some
//! balance but it is the only choice that cannot introduce a tie-break.
//!
//! ## Determinism (DESIGN's core doctrine)
//!
//! Nothing on the algorithm path uses a `HashMap`/`HashSet`, sorts by a float,
//! or depends on allocation addresses. Output polygon order is the input order
//! transformed by a fixed recursion: pre-order (node, front subtree, back
//! subtree) over a tree whose shape is a pure function of the input order.
//! [`Csg::into_meshes`] groups by material through a sorted `Vec` of ids, not a
//! map, so its output order is ascending-by-id on every platform. Same inputs
//! give the same `MeshData`, byte for byte, everywhere —
//! [`MeshData::content_hash`] is the cross-platform tripwire for that claim.
//!
//! ## Recursion
//!
//! `csg.js` recurses in `build`, `clipPolygons` and `allPolygons`. Tree depth
//! is bounded by the number of *distinct* splitting planes on a root-to-leaf
//! path, which for a pathological splitter chain approaches the polygon count —
//! and a `path_extrude`d cliff band is already ~5k triangles, deep enough that
//! a native stack is not obviously safe and a wasm stack (1 MB by default)
//! certainly is not. So the tree lives in a flat arena (`Vec<Node>` with index
//! children) and every traversal is an explicit work stack:
//!
//! - `build` / `clip_polygons`: `Vec<(node, polygons)>` work stack.
//! - `invert`: a *linear* pass over the arena — inverting flips every node the
//!   same way, so no traversal is needed at all.
//! - `clip_to`: linear pass, each node's polygon list is independent.
//! - `into_polygons` (`csg.js`'s `allPolygons`): explicit pre-order stack,
//!   matching its `polygons ++ front ++ back` concatenation order.
//!
//! ## Cost
//!
//! Fragmentation, not triangle count, is what hurts: every polygon of each
//! operand is classified against the other's tree, and splits multiply. Two
//! 4.7k-triangle spheres overlapping halfway subtract in ~1 s (release) and
//! blow 9.4k input triangles up to 68k output ones. Level brushes are boxes and
//! low-side lathes — tens to low hundreds of polygons — so a whole-level bake
//! is comfortably in bake-time budget, but this is not a per-frame operation
//! and high-poly operands (spheres, smooth sweeps) should be kept out of it.
//!
//! ## Epsilon story
//!
//! Two tolerances, and they live at different places on purpose:
//!
//! - **1e-4 position grid, applied once at the boundary** ([`Csg::from_mesh`]).
//!   Snapping inputs first means two brushes authored to meet at `x = 3.0`
//!   actually meet, instead of meeting at `3.0` and `2.9999998` and generating
//!   a sliver wall. Everything downstream then works on already-snapped data.
//! - **1e-5 plane classification epsilon** (`EPSILON`), the csg.js value. It is
//!   deliberately *smaller* than the input grid: a vertex that survived
//!   snapping is either exactly on a plane or a full grid step off it.
//!
//! Split fragments inherit their parent polygon's plane rather than recomputing
//! it from the fragment's first three vertices (a deliberate deviation from
//! `csg.js`, which happily produces a `NaN` normal for a sliver fragment). The
//! fragment is coplanar with its parent by construction, so the inherited plane
//! is the more accurate one as well as the safer one.
//!
//! ## Empty operands
//!
//! An empty solid is short-circuited out of every operation (`∅ ∪ x = x`,
//! `x - ∅ = x`, `∅ - x = ∅`, `∅ ∩ x = ∅`). `csg.js` does not do this and gets
//! `∅ - x` wrong as a result: a BSP with no plane clips nothing, so the tool
//! survives whole and comes back *inverted*. A converted brush that degenerates
//! to nothing has to disappear, not turn into an inside-out solid swallowing
//! the level.
//!
//! ## Material semantics
//!
//! A polygon keeps its own material through every clip and split. Godot's
//! behaviour — *the cut surface of a subtraction shows the tool's material* —
//! is not special-cased here, it falls out: the polygons that survive to line
//! the cavity are the tool's own polygons, inverted. See
//! `subtract_cut_walls_carry_the_tool_material`.
//!
//! ## Accepted defects
//!
//! - **T-junctions and slivers at seams.** A polygon split by a plane whose
//!   neighbour across an edge is not split by that same plane leaves a hairline
//!   crack. Both consumers tolerate it: shading is world-space triplanar (no
//!   UV seams to tear) and collision is a trimesh (no manifold requirement).
//!   Fixing it properly means a vertex-welding/T-junction pass, which is a
//!   later milestone if artefacts actually show up on screen.
//! - **No exact arithmetic.** `f32` planes, no adaptive predicates. Two solids
//!   sharing a face plane exactly are handled (coplanar routing), two sharing
//!   it to within a hair are not guaranteed to be.
//! - **No vertex welding on output.** Every polygon fans into its own vertices.
//!   The renderer does not need shared vertices; call
//!   [`MeshData::smooth_normals`] if a welded, creased result is wanted.

use glam::{Vec2, Vec3};

use super::{face_cross, MeshData, DEGENERATE_AREA_SQ};

/// Plane classification tolerance — the `csg.js` value. See the module docs on
/// why it is smaller than the input quantization grid.
const EPSILON: f32 = 1.0e-5;

/// Input positions are snapped to this grid (1e-4) before anything else runs.
const QUANT: f32 = 1.0e4;

// Vertex-vs-plane classes. Bitwise-or'd into a polygon class, which is why
// `FRONT | BACK == SPANNING`.
const COPLANAR: u8 = 0;
const FRONT: u8 = 1;
const BACK: u8 = 2;
const SPANNING: u8 = 3;

/// A polygon corner. Every attribute is interpolated when a polygon is split,
/// so cut faces keep usable normals/UVs/colors instead of defaults.
#[derive(Clone, Copy, Debug)]
struct CsgVertex {
    pos: Vec3,
    normal: Vec3,
    uv: Vec2,
    color: Vec3,
}

impl CsgVertex {
    fn flip(&mut self) {
        self.normal = -self.normal;
    }

    /// Linear blend of every attribute. The normal is renormalized (with the
    /// `a` side as fallback if the blend cancels) — `csg.js` leaves it raw, but
    /// a short normal reads as a dark seam under lighting.
    fn lerp(&self, other: &CsgVertex, t: f32) -> CsgVertex {
        let n = self.normal.lerp(other.normal, t);
        CsgVertex {
            pos: self.pos.lerp(other.pos, t),
            normal: if n.length_squared() > 1.0e-12 {
                n.normalize()
            } else {
                self.normal
            },
            uv: self.uv.lerp(other.uv, t),
            color: self.color.lerp(other.color, t),
        }
    }
}

/// An oriented plane: points `p` with `n · p == w` lie on it, `n · p > w` in
/// front of it.
#[derive(Clone, Copy, Debug)]
struct Plane {
    n: Vec3,
    w: f32,
}

impl Plane {
    /// The plane of a triangle, or `None` if it is degenerate (zero area).
    fn from_points(a: Vec3, b: Vec3, c: Vec3) -> Option<Plane> {
        let cross = face_cross(a, b, c);
        if cross.length_squared() < DEGENERATE_AREA_SQ {
            return None;
        }
        let n = cross.normalize();
        Some(Plane { n, w: n.dot(a) })
    }

    fn flip(&mut self) {
        self.n = -self.n;
        self.w = -self.w;
    }

    /// Signed distance of `p` from the plane.
    fn distance(&self, p: Vec3) -> f32 {
        self.n.dot(p) - self.w
    }

    /// Classify/split `poly` against this plane. Consumes the polygon: the
    /// non-spanning cases hand it straight back, so only a genuine split
    /// allocates.
    fn split_polygon(&self, poly: CsgPolygon) -> Split {
        let mut poly_class = COPLANAR;
        let mut classes = Vec::with_capacity(poly.verts.len());
        for v in &poly.verts {
            let d = self.distance(v.pos);
            let c = if d < -EPSILON {
                BACK
            } else if d > EPSILON {
                FRONT
            } else {
                COPLANAR
            };
            poly_class |= c;
            classes.push(c);
        }

        match poly_class {
            COPLANAR => {
                if self.n.dot(poly.plane.n) > 0.0 {
                    Split::CoplanarFront(poly)
                } else {
                    Split::CoplanarBack(poly)
                }
            }
            FRONT => Split::Front(poly),
            BACK => Split::Back(poly),
            _ => {
                let n = poly.verts.len();
                let mut f: Vec<CsgVertex> = Vec::with_capacity(n + 1);
                let mut b: Vec<CsgVertex> = Vec::with_capacity(n + 1);
                for i in 0..n {
                    let j = (i + 1) % n;
                    let (ci, cj) = (classes[i], classes[j]);
                    let (vi, vj) = (poly.verts[i], poly.verts[j]);
                    if ci != BACK {
                        f.push(vi);
                    }
                    if ci != FRONT {
                        b.push(vi);
                    }
                    if (ci | cj) == SPANNING {
                        // One endpoint is strictly in front, the other strictly
                        // behind, so the denominator is at least 2 * EPSILON.
                        let t = (self.w - self.n.dot(vi.pos)) / self.n.dot(vj.pos - vi.pos);
                        let v = vi.lerp(&vj, t);
                        f.push(v);
                        b.push(v);
                    }
                }
                // Fragments inherit the parent plane and material. A fragment
                // with fewer than three corners is a degenerate sliver and is
                // dropped (as in csg.js).
                let keep = |verts: Vec<CsgVertex>| {
                    if verts.len() < 3 {
                        return None;
                    }
                    Some(CsgPolygon {
                        verts,
                        plane: poly.plane,
                        material: poly.material,
                    })
                };
                Split::Spanning {
                    front: keep(f),
                    back: keep(b),
                }
            }
        }
    }
}

/// The result of [`Plane::split_polygon`]. `csg.js` writes into four output
/// lists, two of which the caller aliases; Rust can't alias `&mut`, so the
/// classification comes back as a value and each caller routes it.
enum Split {
    CoplanarFront(CsgPolygon),
    CoplanarBack(CsgPolygon),
    Front(CsgPolygon),
    Back(CsgPolygon),
    Spanning {
        front: Option<CsgPolygon>,
        back: Option<CsgPolygon>,
    },
}

/// A convex, planar polygon with a material id. Polygons start as triangles
/// (from [`Csg::from_mesh`]) and only ever get *smaller* through splitting, so
/// convexity is preserved and fan triangulation on the way out is valid.
#[derive(Clone, Debug)]
struct CsgPolygon {
    verts: Vec<CsgVertex>,
    plane: Plane,
    material: u32,
}

impl CsgPolygon {
    fn flip(&mut self) {
        self.verts.reverse();
        for v in &mut self.verts {
            v.flip();
        }
        self.plane.flip();
    }
}

// --- the BSP tree -----------------------------------------------------------

#[derive(Clone, Debug, Default)]
struct Node {
    plane: Option<Plane>,
    front: Option<u32>,
    back: Option<u32>,
    polygons: Vec<CsgPolygon>,
}

/// A BSP tree in a flat arena — node 0 is always the root. See the module docs
/// on why this is an arena and not `Box`ed children.
#[derive(Clone, Debug)]
struct Bsp {
    nodes: Vec<Node>,
}

impl Bsp {
    fn new(polygons: Vec<CsgPolygon>) -> Bsp {
        let mut bsp = Bsp {
            nodes: vec![Node::default()],
        };
        bsp.build(polygons);
        bsp
    }

    fn alloc(&mut self) -> u32 {
        self.nodes.push(Node::default());
        (self.nodes.len() - 1) as u32
    }

    /// Add polygons to the tree, splitting them into the existing structure.
    /// Iterative equivalent of `csg.js`'s `Node.build`.
    fn build(&mut self, polygons: Vec<CsgPolygon>) {
        let mut stack: Vec<(u32, Vec<CsgPolygon>)> = vec![(0, polygons)];
        while let Some((idx, polys)) = stack.pop() {
            if polys.is_empty() {
                continue;
            }
            let i = idx as usize;
            let plane = match self.nodes[i].plane {
                Some(p) => p,
                None => {
                    // csg.js's splitter rule: the first polygon's plane.
                    let p = polys[0].plane;
                    self.nodes[i].plane = Some(p);
                    p
                }
            };

            let mut own = std::mem::take(&mut self.nodes[i].polygons);
            let mut front = Vec::new();
            let mut back = Vec::new();
            for p in polys {
                match plane.split_polygon(p) {
                    // Coplanar polygons (either facing) live at this node.
                    Split::CoplanarFront(p) | Split::CoplanarBack(p) => own.push(p),
                    Split::Front(p) => front.push(p),
                    Split::Back(p) => back.push(p),
                    Split::Spanning { front: f, back: b } => {
                        if let Some(f) = f {
                            front.push(f);
                        }
                        if let Some(b) = b {
                            back.push(b);
                        }
                    }
                }
            }
            self.nodes[i].polygons = own;

            // Push back first so the front subtree is processed first, matching
            // the recursive version's node-allocation order.
            if !back.is_empty() {
                let child = match self.nodes[i].back {
                    Some(c) => c,
                    None => {
                        let c = self.alloc();
                        self.nodes[i].back = Some(c);
                        c
                    }
                };
                stack.push((child, back));
            }
            if !front.is_empty() {
                let child = match self.nodes[i].front {
                    Some(c) => c,
                    None => {
                        let c = self.alloc();
                        self.nodes[i].front = Some(c);
                        c
                    }
                };
                stack.push((child, front));
            }
        }
    }

    /// Flip the solid the tree represents (inside becomes outside). Recursion
    /// would visit every node once and do the same three things; a linear pass
    /// over the arena is the same operation without the traversal.
    fn invert(&mut self) {
        for node in &mut self.nodes {
            for p in &mut node.polygons {
                p.flip();
            }
            if let Some(plane) = &mut node.plane {
                plane.flip();
            }
            std::mem::swap(&mut node.front, &mut node.back);
        }
    }

    /// Remove the parts of `polygons` that are inside this solid.
    ///
    /// The output order matches the recursive `front ++ back` concatenation:
    /// the work stack pops the front subtree before the back subtree, and
    /// polygons that reach a front leaf are emitted the moment they get there —
    /// which is always before anything from a back subtree pushed alongside or
    /// above them.
    fn clip_polygons(&self, polygons: Vec<CsgPolygon>) -> Vec<CsgPolygon> {
        let mut out = Vec::with_capacity(polygons.len());
        let mut stack: Vec<(u32, Vec<CsgPolygon>)> = vec![(0, polygons)];
        while let Some((idx, polys)) = stack.pop() {
            if polys.is_empty() {
                continue;
            }
            let node = &self.nodes[idx as usize];
            let Some(plane) = node.plane else {
                // An unbuilt node bounds nothing: everything survives.
                out.extend(polys);
                continue;
            };

            let mut front = Vec::new();
            let mut back = Vec::new();
            for p in polys {
                match plane.split_polygon(p) {
                    // Coplanar polygons follow the side they face.
                    Split::CoplanarFront(p) | Split::Front(p) => front.push(p),
                    Split::CoplanarBack(p) | Split::Back(p) => back.push(p),
                    Split::Spanning { front: f, back: b } => {
                        if let Some(f) = f {
                            front.push(f);
                        }
                        if let Some(b) = b {
                            back.push(b);
                        }
                    }
                }
            }
            // No back child means "behind this plane is solid": drop `back`.
            if let Some(b) = node.back {
                stack.push((b, back));
            }
            match node.front {
                Some(f) => stack.push((f, front)),
                None => out.extend(front),
            }
        }
        out
    }

    /// Remove every polygon of this tree that is inside `other`.
    fn clip_to(&mut self, other: &Bsp) {
        for i in 0..self.nodes.len() {
            let polys = std::mem::take(&mut self.nodes[i].polygons);
            self.nodes[i].polygons = other.clip_polygons(polys);
        }
    }

    /// Node indices in pre-order (node, front subtree, back subtree).
    fn preorder(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.nodes.len());
        let mut stack = vec![0u32];
        while let Some(i) = stack.pop() {
            out.push(i);
            let node = &self.nodes[i as usize];
            if let Some(b) = node.back {
                stack.push(b);
            }
            if let Some(f) = node.front {
                stack.push(f);
            }
        }
        out
    }

    /// `csg.js`'s `allPolygons`, but draining rather than cloning: every call
    /// site is the last thing done with the tree.
    fn into_polygons(mut self) -> Vec<CsgPolygon> {
        let order = self.preorder();
        let mut out = Vec::new();
        for i in order {
            out.append(&mut self.nodes[i as usize].polygons);
        }
        out
    }
}

// --- the public solid -------------------------------------------------------

/// A solid, as a bag of convex polygons with per-polygon material ids.
///
/// Build one per source mesh with [`Csg::from_mesh`], combine with
/// [`union`](Csg::union) / [`subtract`](Csg::subtract) /
/// [`intersect`](Csg::intersect), then bake with [`into_meshes`](Csg::into_meshes)
/// (render, split per material) or [`into_mesh`](Csg::into_mesh) (collision,
/// everything merged).
///
/// Operands are expected to be closed solids. A non-closed input is not
/// rejected — it just produces whatever the BSP makes of it, which is rarely
/// what anyone wanted.
#[derive(Clone, Debug, Default)]
pub struct Csg {
    polygons: Vec<CsgPolygon>,
}

impl Csg {
    /// Convert a triangle mesh into a solid, tagging every triangle `material`.
    ///
    /// Positions are snapped to a 1e-4 grid *first* (see the module docs on the
    /// epsilon story), then triangles that are degenerate after snapping —
    /// cross product shorter than `sqrt(DEGENERATE_AREA_SQ)`, e.g. sphere poles
    /// or a lathe profile touching its axis — are dropped. Missing attributes
    /// are filled: normals from the face, UVs zero, colors white.
    pub fn from_mesh(mesh: &MeshData, material: u32) -> Csg {
        let snap = |p: Vec3| {
            Vec3::new(
                (p.x * QUANT).round() / QUANT,
                (p.y * QUANT).round() / QUANT,
                (p.z * QUANT).round() / QUANT,
            )
        };
        let pos: Vec<Vec3> = mesh.positions.iter().map(|p| snap(*p)).collect();

        let mut polygons = Vec::with_capacity(mesh.triangle_count());
        for t in mesh.indices.chunks_exact(3) {
            let idx = [t[0] as usize, t[1] as usize, t[2] as usize];
            let (a, b, c) = (pos[idx[0]], pos[idx[1]], pos[idx[2]]);
            let Some(plane) = Plane::from_points(a, b, c) else {
                continue; // degenerate after quantization
            };
            let verts = idx
                .iter()
                .map(|&i| CsgVertex {
                    pos: pos[i],
                    normal: *mesh.normals.get(i).unwrap_or(&plane.n),
                    uv: *mesh.uvs.get(i).unwrap_or(&Vec2::ZERO),
                    color: *mesh.colors.get(i).unwrap_or(&Vec3::ONE),
                })
                .collect();
            polygons.push(CsgPolygon {
                verts,
                plane,
                material,
            });
        }
        Csg { polygons }
    }

    /// Everything in either solid. (`csg.js` `union`, unchanged.)
    pub fn union(self, other: Csg) -> Csg {
        if self.polygons.is_empty() {
            return other;
        }
        if other.polygons.is_empty() {
            return self;
        }
        let mut a = Bsp::new(self.polygons);
        let mut b = Bsp::new(other.polygons);
        a.clip_to(&b);
        b.clip_to(&a);
        b.invert();
        b.clip_to(&a);
        b.invert();
        a.build(b.into_polygons());
        Csg {
            polygons: a.into_polygons(),
        }
    }

    /// This solid with `other` carved out of it. (`csg.js` `subtract`.)
    ///
    /// The walls of the cavity are `other`'s polygons, inverted — so they carry
    /// `other`'s material, which is Godot's behaviour and the reason the level
    /// converter can give a subtraction tool a material at all.
    pub fn subtract(self, other: Csg) -> Csg {
        if self.polygons.is_empty() || other.polygons.is_empty() {
            return self;
        }
        let mut a = Bsp::new(self.polygons);
        let mut b = Bsp::new(other.polygons);
        a.invert();
        a.clip_to(&b);
        b.clip_to(&a);
        b.invert();
        b.clip_to(&a);
        b.invert();
        a.build(b.into_polygons());
        a.invert();
        Csg {
            polygons: a.into_polygons(),
        }
    }

    /// Only what is inside both solids. (`csg.js` `intersect`.)
    pub fn intersect(self, other: Csg) -> Csg {
        if self.polygons.is_empty() || other.polygons.is_empty() {
            return Csg::default();
        }
        let mut a = Bsp::new(self.polygons);
        let mut b = Bsp::new(other.polygons);
        a.invert();
        b.clip_to(&a);
        b.invert();
        a.clip_to(&b);
        b.clip_to(&a);
        a.build(b.into_polygons());
        a.invert();
        Csg {
            polygons: a.into_polygons(),
        }
    }

    /// How many polygons the solid currently holds. A rough cost signal (each
    /// becomes `verts - 2` triangles) and the cheapest "did this op vanish?"
    /// check.
    pub fn polygon_count(&self) -> usize {
        self.polygons.len()
    }

    /// One mesh per material, ascending by material id — the render-side bake.
    ///
    /// Materials with no surviving polygons simply do not appear, so an empty
    /// solid returns an empty `Vec` rather than a set of empty meshes.
    pub fn into_meshes(self) -> Vec<(u32, MeshData)> {
        let mut mats: Vec<u32> = self.polygons.iter().map(|p| p.material).collect();
        mats.sort_unstable();
        mats.dedup();

        mats.into_iter()
            .map(|mat| {
                let mesh = build_mesh(self.polygons.iter().filter(|p| p.material == mat));
                (mat, mesh)
            })
            .collect()
    }

    /// Everything as one mesh, materials ignored — the collision-side bake
    /// (a trimesh collider does not care which surface it came from).
    pub fn into_mesh(self) -> MeshData {
        build_mesh(self.polygons.iter())
    }
}

/// Fan-triangulate polygons into a mesh. Valid because BSP polygons are convex:
/// they start as triangles and splitting only ever shrinks them.
fn build_mesh<'a>(polys: impl Iterator<Item = &'a CsgPolygon>) -> MeshData {
    let mut m = MeshData::default();
    for poly in polys {
        let base = m.positions.len() as u32;
        for v in &poly.verts {
            m.positions.push(v.pos);
            m.normals.push(v.normal);
            m.uvs.push(v.uv);
            m.colors.push(v.color);
        }
        for i in 1..poly.verts.len() as u32 - 1 {
            m.indices.extend_from_slice(&[base, base + i, base + i + 1]);
        }
    }
    m.validate();
    m
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::{extrude::lathe, primitives};

    /// Two unit-ish boxes overlapping in one corner octant: `a` spans
    /// `[-1, 1]³`, `b` spans `[0, 2]³`, so the overlap is exactly `[0, 1]³`.
    fn corner_pair() -> (MeshData, MeshData) {
        (
            primitives::cube(2.0),
            primitives::cube(2.0).translate(Vec3::splat(1.0)),
        )
    }

    /// Every undirected edge is shared by exactly two triangles, over *welded
    /// positions* (CSG output is never index-shared).
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
                    continue; // degenerate sliver edge, not a surface boundary
                }
                *counts.entry((x.min(y), x.max(y))).or_insert(0) += 1;
            }
        }
        counts.values().all(|&c| c == 2)
    }

    fn assert_bounds(m: &MeshData, lo: Vec3, hi: Vec3) {
        let (l, h) = m.bounds().expect("mesh has vertices");
        assert!((l - lo).length() < 1e-4, "min {l:?}, want {lo:?}");
        assert!((h - hi).length() < 1e-4, "max {h:?}, want {hi:?}");
    }

    /// Triangle centroids of a mesh.
    fn centroids(m: &MeshData) -> Vec<Vec3> {
        m.triangles().map(|[a, b, c]| (a + b + c) / 3.0).collect()
    }

    #[test]
    fn union_of_overlapping_cubes_is_a_closed_solid() {
        let (a, b) = corner_pair();
        let solid = Csg::from_mesh(&a, 0).union(Csg::from_mesh(&b, 0));
        assert!(solid.polygon_count() > 12);
        let m = solid.into_mesh();
        m.validate();
        assert!(is_closed_surface(&m), "union is not watertight");
        // The union spans both boxes.
        assert_bounds(&m, Vec3::splat(-1.0), Vec3::splat(2.0));
        // Nothing survives strictly inside the other solid: no triangle
        // centroid may land in the open overlap box.
        assert!(centroids(&m)
            .iter()
            .all(|c| !(c.min_element() > 0.01 && c.max_element() < 0.99)));
    }

    #[test]
    fn union_of_disjoint_cubes_keeps_both_intact() {
        let a = primitives::cube(1.0);
        let b = primitives::cube(1.0).translate(Vec3::new(10.0, 0.0, 0.0));
        let solid = Csg::from_mesh(&a, 0).union(Csg::from_mesh(&b, 0));
        // Nothing to clip: both boxes' 12 triangles come through unsplit.
        assert_eq!(solid.polygon_count(), 24);
        let m = solid.into_mesh();
        assert_eq!(m.triangle_count(), 24);
        assert!(is_closed_surface(&m));
    }

    #[test]
    fn subtract_carves_a_corner_notch() {
        let (a, b) = corner_pair();
        let solid = Csg::from_mesh(&a, 0).subtract(Csg::from_mesh(&b, 1));
        let m = solid.clone().into_mesh();
        m.validate();
        // Carving a corner off does not change the box's extent.
        assert_bounds(&m, Vec3::splat(-1.0), Vec3::splat(1.0));
        assert!(is_closed_surface(&m), "notched box is not watertight");
        // The removed octant is genuinely gone: no surface inside it.
        assert!(centroids(&m)
            .iter()
            .all(|c| !(c.min_element() > 0.01 && c.max_element() < 0.99)));
    }

    #[test]
    fn subtract_cut_walls_carry_the_tool_material() {
        // Godot shows the *tool's* material on a cut surface; here that is not
        // a special case, it is where the surviving polygons came from.
        let (a, b) = corner_pair();
        let meshes = Csg::from_mesh(&a, 0)
            .subtract(Csg::from_mesh(&b, 1))
            .into_meshes();
        assert_eq!(meshes.iter().map(|(m, _)| *m).collect::<Vec<_>>(), [0, 1]);

        // The three walls of the notch lie on x = 0, y = 0 and z = 0 inside the
        // carved octant. Every triangle on them must be the tool's.
        for axis in 0..3 {
            let mut wall_tris = 0;
            for (mat, mesh) in &meshes {
                for c in centroids(mesh) {
                    let others: Vec<f32> = (0..3).filter(|&i| i != axis).map(|i| c[i]).collect();
                    let on_wall = c[axis].abs() < 1e-4
                        && others.iter().all(|&v| (0.05..0.95).contains(&v));
                    if on_wall {
                        wall_tris += 1;
                        assert_eq!(*mat, 1, "notch wall on axis {axis} has material {mat}");
                    }
                }
            }
            assert!(wall_tris > 0, "no notch wall found on axis {axis}");
        }
    }

    #[test]
    fn intersect_keeps_only_the_overlap_box() {
        let (a, b) = corner_pair();
        let m = Csg::from_mesh(&a, 0)
            .intersect(Csg::from_mesh(&b, 0))
            .into_mesh();
        m.validate();
        assert_bounds(&m, Vec3::ZERO, Vec3::ONE);
        assert!(is_closed_surface(&m), "intersection is not watertight");
    }

    #[test]
    fn intersect_of_disjoint_solids_is_empty() {
        let a = primitives::cube(1.0);
        let b = primitives::cube(1.0).translate(Vec3::new(10.0, 0.0, 0.0));
        let solid = Csg::from_mesh(&a, 0).intersect(Csg::from_mesh(&b, 1));
        assert_eq!(solid.polygon_count(), 0);
        assert!(solid.clone().into_meshes().is_empty());
        let m = solid.into_mesh();
        assert!(m.is_empty());
        m.validate();
    }

    #[test]
    fn into_meshes_splits_per_material_ascending() {
        let (a, b) = corner_pair();
        // Deliberately out of order: 7 unions into 3.
        let meshes = Csg::from_mesh(&a, 7)
            .union(Csg::from_mesh(&b, 3))
            .into_meshes();
        assert_eq!(meshes.len(), 2);
        assert_eq!(meshes[0].0, 3);
        assert_eq!(meshes[1].0, 7);
        for (_, m) in &meshes {
            m.validate();
            assert!(!m.is_empty());
            assert_eq!(m.normals.len(), m.positions.len());
            assert_eq!(m.uvs.len(), m.positions.len());
            assert_eq!(m.colors.len(), m.positions.len());
        }
        // The split is a partition of the merged bake.
        let merged = Csg::from_mesh(&a, 7)
            .union(Csg::from_mesh(&b, 3))
            .into_mesh();
        let split: usize = meshes.iter().map(|(_, m)| m.triangle_count()).sum();
        assert_eq!(split, merged.triangle_count());
    }

    #[test]
    fn empty_operands_are_absorbed_not_inverted() {
        // The deviation from csg.js documented in the module header: without
        // the short-circuits, `∅ - cube` comes back as an inside-out cube.
        let empty = || Csg::from_mesh(&MeshData::default(), 0);
        let cube = || Csg::from_mesh(&primitives::cube(1.0), 0);

        assert_eq!(empty().union(cube()).polygon_count(), 12);
        assert_eq!(cube().union(empty()).polygon_count(), 12);
        assert_eq!(cube().subtract(empty()).polygon_count(), 12);
        assert_eq!(empty().subtract(cube()).polygon_count(), 0);
        assert_eq!(cube().intersect(empty()).polygon_count(), 0);
        assert_eq!(empty().intersect(cube()).polygon_count(), 0);
        assert_eq!(empty().union(empty()).polygon_count(), 0);
        assert!(empty().union(empty()).into_mesh().is_empty());
    }

    #[test]
    fn from_mesh_drops_degenerate_triangles() {
        let mut m = primitives::cube(1.0);
        // A triangle whose corners collapse onto one another after snapping.
        let base = m.positions.len() as u32;
        for _ in 0..3 {
            m.positions.push(Vec3::new(5.0, 5.0, 5.000_001));
            m.normals.push(Vec3::Y);
            m.uvs.push(Vec2::ZERO);
            m.colors.push(Vec3::ONE);
        }
        m.indices.extend_from_slice(&[base, base + 1, base + 2]);
        assert_eq!(Csg::from_mesh(&m, 0).polygon_count(), 12);
    }

    #[test]
    fn lathe_minus_cylinder_bore_is_sane() {
        // A solid disc (profile touches the spin axis) with a bore drilled
        // through it — the M4 sweeps feeding M5, which is the whole point.
        let profile = [
            Vec2::new(0.0, -0.5),
            Vec2::new(2.0, -0.5),
            Vec2::new(2.0, 0.5),
            Vec2::new(0.0, 0.5),
        ];
        let disc = lathe(&profile, 360.0, 16, false);
        let drill = primitives::cylinder(0.5, 4.0, 12);
        let solid = Csg::from_mesh(&disc, 0).subtract(Csg::from_mesh(&drill, 1));
        assert!(solid.polygon_count() > disc.triangle_count());

        let m = solid.clone().into_mesh();
        m.validate();
        assert!(m.triangle_count() > 0);
        // The bore is open top to bottom: nothing left near the axis.
        let near_axis = centroids(&m)
            .iter()
            .filter(|c| Vec2::new(c.x, c.z).length() < 0.4 && c.y.abs() < 0.49)
            .count();
        assert_eq!(near_axis, 0, "material left inside the bore");
        // The bore wall is the drill's, so both materials survive.
        assert_eq!(
            solid.into_meshes().iter().map(|(m, _)| *m).collect::<Vec<_>>(),
            [0, 1]
        );
    }

    #[test]
    fn booleans_are_deterministic() {
        let (a, b) = corner_pair();
        let once = Csg::from_mesh(&a, 0)
            .subtract(Csg::from_mesh(&b, 1))
            .into_mesh();
        for _ in 0..4 {
            let again = Csg::from_mesh(&a, 0)
                .subtract(Csg::from_mesh(&b, 1))
                .into_mesh();
            assert_eq!(once.content_hash(), again.content_hash());
            assert_eq!(once, again);
        }
    }

    /// Content-hash pin for one canonical boolean, the same tripwire the sweeps
    /// carry (see `extrude`'s `canonical_meshes_hash_stably`): it moves if
    /// polygon order, split order, or the fan triangulation changes, and since
    /// the hash is also the content-cache key (DESIGN §6) that has to be a
    /// deliberate edit rather than a silent one.
    ///
    /// Same caveat as M4: `MeshData::content_hash` is built on `DefaultHasher`,
    /// whose algorithm the standard library may change between Rust releases.
    /// A toolchain bump moving this number is expected and is not a geometry
    /// regression — an unexplained move on a fixed toolchain is.
    #[test]
    fn canonical_boolean_hashes_stably() {
        let (a, b) = corner_pair();
        let m = Csg::from_mesh(&a, 0)
            .subtract(Csg::from_mesh(&b, 1))
            .into_mesh();
        assert_eq!(m.content_hash(), 9101101746590643101);
    }
}
