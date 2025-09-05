# Copyright 2025 Intrinsic Innovation LLC

"""
Check if a mesh is watertight using trimesh.
"""
import trimesh
from typing import Dict


def run_is_mesh_watertight(mesh_path: str) -> Dict[str, bool]:
    """
    Check if a mesh is watertight.
    Args:
        mesh_path (str): Path to the mesh file.
    Returns:
        Dict[str, bool]: Dictionary with key "watertight" indicating if the mesh is watertight.
    """
    mesh = trimesh.load(mesh_path, force="mesh")
    return {"watertight": bool(getattr(mesh, "is_watertight", False))}
