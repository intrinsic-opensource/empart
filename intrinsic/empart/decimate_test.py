# Copyright 2025 Intrinsic Innovation LLC

"""
Tests for the mesh decimation functionality.
"""
import argparse
import base64
import pytest
import trimesh
import io
from .decimate import run_decimate_from_args


def create_dummy_mesh(path: str):
    """Creates and writes a simple sphere mesh to `path` in .ply format."""
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    sphere.export(path)


@pytest.mark.parametrize("triangle_target", [100, 500])
def test_run_decimate_from_argv(tmp_path, triangle_target):
    # Prepare test input
    mesh_path = tmp_path / "input_mesh.glb"
    create_dummy_mesh(str(mesh_path))

    args = argparse.Namespace(             
        min_val=0.0,
        max_val=0.1,
        input=str(mesh_path),
        triangles=triangle_target,
    )
    # Run decimation
    result = run_decimate_from_args(args)

    # Assertions
    assert isinstance(result, dict)
    assert "mesh" in result
    encoded = result["mesh"]
    assert isinstance(encoded, str)

    # Try decoding the base64 to ensure it's valid
    glb_bytes = base64.b64decode(encoded)
    assert len(glb_bytes) > 0

    # Optionally, verify it's a valid GLB by loading it with trimesh
    loaded = trimesh.load(io.BytesIO(glb_bytes), file_type="glb", force="mesh")
    assert isinstance(loaded, trimesh.Trimesh)
    assert len(loaded.faces) == triangle_target
