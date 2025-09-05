# Copyright 2025 Intrinsic Innovation LLC

"""
Decimate a mesh to a target number of triangles using Open3D.
"""

import base64
from typing import List, Dict

import numpy as np
import open3d as o3d
import trimesh

from .utils import encode_mesh_bytes

def trimesh_to_o3d(tm: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    """Convert a Trimesh object to an Open3D TriangleMesh."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(tm.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(tm.faces)
    return mesh


def o3d_to_trimesh(m: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    """Convert an Open3D TriangleMesh to a Trimesh object."""
    return trimesh.Trimesh(
        vertices=np.asarray(m.vertices),
        faces=np.asarray(m.triangles),
        process=False,
    )


def run_decimate_from_args(args) -> Dict[str, str]:
    """
    Simplify a mesh to a target number of triangles.
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
            - input: Path to the input mesh file.
            - triangles: Target number of triangles for the output mesh.
    Returns:
        Dict[str, str]: A dictionary containing the base64-encoded GLB mesh.
    """
    assert args.triangles > 0, "Target number of triangles must be greater than 0."

    mesh_in = o3d.io.read_triangle_mesh(args.input)

    simplified = mesh_in.simplify_quadric_decimation(
        target_number_of_triangles=args.triangles
    )
    simplified.remove_unreferenced_vertices()
    simplified.remove_duplicated_vertices()
    simplified.remove_duplicated_triangles()
    simplified.remove_degenerate_triangles()

    tm_out = o3d_to_trimesh(simplified)
    glb_bytes = tm_out.export(file_type="glb")

    return {
        "mesh": encode_mesh_bytes(glb_bytes),
    }
