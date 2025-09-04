# Copyright 2025 Intrinsic Innovation LLC

"""
Watertight-ify a mesh using:

  • Open3D  : fast voxelisation (CPU-parallel C++)
  • PyMCubes: fast marching-cubes extraction (C/OpenMP)
  • Trimesh : final mesh export to GLB

Stdout → JSON:
    {"mesh":"<base64-glb>",
     "stats":{"triangles":…, "vertices":…, "watertight":…, "pitch":…}}

"""

import argparse
import numpy as np
import trimesh, open3d as o3d, mcubes
import io
from typing import List, Dict
from .utils import encode_mesh_bytes

def trimesh_to_o3d(tm: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    """Lightweight converter; keeps vertices + faces 1-for-1."""
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(tm.vertices)
    m.triangles = o3d.utility.Vector3iVector(tm.faces)
    return m

def voxelise_open3d(o3_mesh: o3d.geometry.TriangleMesh, pitch: float):
    """
    Returns (dense_volume_uint8, origin_world_xyz).
    `dense_volume` is a NumPy array with 1 = occupied.
    """
    vg = o3d.geometry.VoxelGrid.create_from_triangle_mesh(o3_mesh, voxel_size=pitch)
    voxels = vg.get_voxels()
    if not voxels:
        raise RuntimeError("Voxeliser produced an empty grid.")

    idx = np.asarray([v.grid_index for v in voxels], dtype=np.int32)  # (N,3)
    dims = idx.max(axis=0) + 1
    vol = np.zeros(dims, dtype=np.uint8)
    vol[idx[:, 0], idx[:, 1], idx[:, 2]] = 1

    origin = np.asarray(vg.origin) + pitch * 0.5  # shift to voxel-centre coords
    return vol, origin


def run_voxel_watertight_from_args(args: argparse.Namespace) -> Dict:
    """
    Main function to run the voxelization.

    Args:
        args: Command line arguments containing input mesh path and optional pitch.

    Returns:
        A dictionary with the base64-encoded mesh and statistics."""
    mesh_o3d = o3d.io.read_triangle_mesh(args.input)
    bbox_min = mesh_o3d.get_min_bound()
    bbox_max = mesh_o3d.get_max_bound()
    bbox_size = np.asarray(bbox_max) - np.asarray(bbox_min)

    pitch = args.pitch or (bbox_size.min() / 100.0)

    # 1) Voxelise
    volume, origin = voxelise_open3d(mesh_o3d, pitch)

    # 2) Marching Cubes
    verts, faces = mcubes.marching_cubes(volume, 0.5)
    verts = verts * pitch + origin
    mesh_out = trimesh.Trimesh(verts, faces, process=False)

    # 3) Export to GLB (in memory)
    buf = io.BytesIO()
    mesh_out.export(file_obj=buf, file_type="glb")
    glb_bytes = buf.getvalue()

    return {
        "mesh": encode_mesh_bytes(glb_bytes),
        "stats": {
            "triangles": int(mesh_out.faces.shape[0]),
            "vertices": int(mesh_out.vertices.shape[0]),
            "watertight": bool(mesh_out.is_watertight),
            "pitch": pitch,
        },
    }
