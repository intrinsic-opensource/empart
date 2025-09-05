# Copyright 2025 Intrinsic Innovation LLC

"""
Tests for the watertight mesh checking functionality.
"""
import trimesh
from pathlib import Path
from .check_watertight import run_is_mesh_watertight


def test_watertight_mesh(tmp_path: Path):
    # Create a watertight sphere mesh
    mesh = trimesh.creation.icosphere()
    file_path = tmp_path / "sphere.ply"
    mesh.export(file_path)

    result = run_is_mesh_watertight(str(file_path))
    assert isinstance(result, dict)
    assert result["watertight"] is True


def test_non_watertight_mesh(tmp_path: Path):
    # Create a watertight box and remove one face
    mesh = trimesh.creation.box()
    mesh.faces = mesh.faces[:-1]  # remove a face to introduce a hole
    file_path = tmp_path / "box_with_hole.ply"
    mesh.export(file_path)

    result = run_is_mesh_watertight(str(file_path))
    assert isinstance(result, dict)
    assert result["watertight"] is False
