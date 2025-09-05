# Copyright 2025 Intrinsic Innovation LLC

"""
Test for voxelization functionality in geometry module.
"""
import argparse
import base64
import trimesh
from pathlib import Path

from .voxelize import run_voxel_watertight_from_args


def create_test_mesh(file_path: Path):
    """Create and export a basic watertight mesh (a cube)."""
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    mesh.export(file_path)


def test_run_voxel_watertight_from_argv(tmp_path):
    mesh_path = tmp_path / "cube.obj"
    create_test_mesh(mesh_path)
    args = argparse.Namespace(input=str(mesh_path), pitch=None)
    result = run_voxel_watertight_from_args(args)

    # Validate structure
    assert isinstance(result, dict)
    assert "mesh" in result
    assert "stats" in result

    # Validate base64 output
    mesh_b64 = result["mesh"]
    assert isinstance(mesh_b64, str)
    decoded = base64.b64decode(mesh_b64)
    assert len(decoded) > 0

    # Validate stats
    stats = result["stats"]
    assert isinstance(stats, dict)
    assert "triangles" in stats
    assert "vertices" in stats
    assert "watertight" in stats
    assert "pitch" in stats
    assert stats["triangles"] > 100_000 and stats["triangles"] < 120_000
    assert stats["vertices"] > 0
    assert isinstance(stats["watertight"], bool)
    assert stats["watertight"] is True
