//! Mesh storage, CPU and GPU side (DESIGN §5, §6).
//!
//! Everything is keyed by [`MeshData::content_hash`] — determinism paying rent.
//! Two entities whose generators produced byte-identical meshes get the same
//! [`MeshHandle`] and therefore share one pair of GPU buffers, without anyone
//! having to notice the duplication.
//!
//! Two layers, deliberately separate:
//!
//! - [`MeshLibrary`] is a world *resource*: handle → [`MeshData`], GPU-free, so
//!   the sim (and its tests) can talk about meshes with no adapter in sight.
//! - [`MeshRegistry`] lives in the renderer: handle → GPU buffers, filled in
//!   lazily from the library the first time a handle is actually drawn.
//!
//! Step 4 (§6) replaces the "library is populated at spawn time" half with a
//! generator registry + content cache; the handle type and the GPU half do not
//! have to change for that.

use std::collections::HashMap;

use bevy_ecs::prelude::Resource;
use runt_mesh::MeshData;
use wgpu::util::DeviceExt;

use crate::draw::Aabb;
use crate::interleave;

/// A content-addressed mesh key: literally `MeshData::content_hash()`.
///
/// Not an index and not a generation — equality means "the same geometry", full
/// stop, which is what makes dedup free and caching possible.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MeshHandle(pub u64);

impl MeshHandle {
    /// The handle `mesh` will always have, computed without storing anything.
    pub fn of(mesh: &MeshData) -> MeshHandle {
        MeshHandle(mesh.content_hash())
    }
}

/// Uploaded geometry: one vertex buffer, one index buffer, one draw range —
/// and the object-space box the frustum test asks about (DESIGN §5, D5).
pub struct GpuMesh {
    pub vertices: wgpu::Buffer,
    pub indices: wgpu::Buffer,
    pub index_count: u32,
    pub vertex_count: u32,
    /// Object-space bounds, measured once at upload. `None` for geometry with
    /// no vertices, which the culler reads as "keep it" rather than "cull it".
    ///
    /// Here rather than recomputed per frame because it is O(vertices) and the
    /// answer is immutable: the handle *is* the content hash, so two meshes
    /// with the same handle have the same box by definition.
    pub bounds: Option<Aabb>,
}

/// Handle → GPU buffers. Owned by the [`Renderer`](crate::Renderer).
#[derive(Default)]
pub struct MeshRegistry {
    meshes: HashMap<MeshHandle, GpuMesh>,
}

impl MeshRegistry {
    pub fn new() -> MeshRegistry {
        MeshRegistry::default()
    }

    /// Upload `mesh` and return its handle — or return the existing handle
    /// untouched if geometry with that content hash is already resident.
    ///
    /// Idempotent by construction: calling this every frame for every draw
    /// would be correct, just wasteful, which is why the renderer only calls it
    /// for handles it has never seen.
    pub fn register(&mut self, device: &wgpu::Device, mesh: &MeshData) -> MeshHandle {
        let handle = MeshHandle::of(mesh);
        if self.meshes.contains_key(&handle) {
            return handle;
        }
        mesh.validate();

        let verts = interleave(mesh);
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh vertices"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        self.meshes.insert(
            handle,
            GpuMesh {
                vertices,
                indices,
                index_count: mesh.indices.len() as u32,
                vertex_count: verts.len() as u32,
                bounds: Aabb::of_mesh(mesh),
            },
        );
        handle
    }

    pub fn get(&self, handle: MeshHandle) -> Option<&GpuMesh> {
        self.meshes.get(&handle)
    }

    /// The object-space bounds of a resident mesh — the culler's whole input
    /// (see [`crate::draw::cull_draw_list`]).
    ///
    /// `None` for a handle that is not resident *and* for one whose geometry is
    /// empty. The culler treats both the same way, and it is the same answer:
    /// there is nothing here that can be proven off screen.
    pub fn bounds(&self, handle: MeshHandle) -> Option<Aabb> {
        self.meshes.get(&handle)?.bounds
    }

    pub fn contains(&self, handle: MeshHandle) -> bool {
        self.meshes.contains_key(&handle)
    }

    /// Number of distinct meshes resident on the GPU.
    pub fn len(&self) -> usize {
        self.meshes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.meshes.is_empty()
    }
}

/// Handle → [`MeshData`], as a world resource. GPU-free.
///
/// This is where generated geometry lands at scene-build time; the renderer
/// pulls from it on demand. Iteration order is never part of anything the sim
/// computes (DESIGN §3): lookups are by handle only.
#[derive(Resource, Default)]
pub struct MeshLibrary {
    meshes: HashMap<MeshHandle, MeshData>,
}

impl MeshLibrary {
    pub fn new() -> MeshLibrary {
        MeshLibrary::default()
    }

    /// Store `mesh` and return its handle, dropping it if identical geometry is
    /// already stored. Same dedup rule as [`MeshRegistry::register`], one level
    /// up the pipeline.
    pub fn insert(&mut self, mesh: MeshData) -> MeshHandle {
        let handle = MeshHandle::of(&mesh);
        self.meshes.entry(handle).or_insert(mesh);
        handle
    }

    pub fn get(&self, handle: MeshHandle) -> Option<&MeshData> {
        self.meshes.get(&handle)
    }

    pub fn contains(&self, handle: MeshHandle) -> bool {
        self.meshes.contains_key(&handle)
    }

    pub fn len(&self) -> usize {
        self.meshes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.meshes.is_empty()
    }
}
