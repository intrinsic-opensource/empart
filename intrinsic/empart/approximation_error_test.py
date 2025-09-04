# Copyright 2025 Intrinsic Innovation LLC

"""
Unit tests for approximation_error.run_approx_err_from_args.

True mesh:  unit sphere  (r = 1.00)
Approx mesh: slightly larger sphere (r = 1.01)
"""
from __future__ import annotations

import argparse
import base64
from io import BytesIO
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import trimesh

from .approximation_error import run_approx_err_from_args


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _write_sphere(path: Path, radius: float) -> None:
    """
    Create a tiny icosphere with the given radius and save it.
    Using subdivisions=0 keeps vertex count low → tests stay fast.
    """
    mesh = trimesh.creation.icosphere(subdivisions=0, radius=radius)
    mesh.export(path)


@pytest.fixture
def sample_meshes(tmp_path: Path) -> Tuple[Path, Path]:
    """
    Returns (true_mesh_path, approx_mesh_path) with sphere vs. larger sphere.
    """
    true_path = tmp_path / "true.glb"
    approx_path = tmp_path / "approx.glb"

    _write_sphere(true_path, radius=1.0)    # TRUE mesh  (smaller)
    _write_sphere(approx_path, radius=1.01)  # APPROX mesh (larger)

    return true_path, approx_path


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
def test_run_success(sample_meshes: Tuple[Path, Path]) -> None:
    """
    End-to-end run: real sampling & KD-Tree; assert valid JSON + mesh output.
    """
    
    true_path, approx_path = sample_meshes
    args = argparse.Namespace(
        display_true_error=True,
        region=None,
        min_val=0.0,
        max_val=0.1,
        paths=[str(true_path), str(approx_path)],
    )
    result = run_approx_err_from_args(args)

    # ---- JSON structure ------------------------------------------------------
    assert isinstance(result, dict)
    assert list(result) == ["error_meshes"]
    assert len(result["error_meshes"]) == 1

    # ---- Decode and sanity-check PLY -----------------------------------------
    mesh_bytes = base64.b64decode(result["error_meshes"][0])
    assert mesh_bytes.startswith(b"ply")

    mesh = trimesh.load(BytesIO(mesh_bytes), file_type="ply", force="mesh")

    assert mesh.visual.kind == "vertex"
    # All vertices in true mesh were coloured, so count must match
    assert mesh.vertices.shape[0] == mesh.visual.vertex_colors.shape[0]


def test_min_ge_max_raises(sample_meshes: Tuple[Path, Path]) -> None:
    """<min_val> must be strictly less than <max_val>."""
    true_path, approx_path = sample_meshes
    bad_args = argparse.Namespace(
        min_val=0.05,
        max_val=0.05,
        paths=[str(true_path), str(approx_path)],
    )
    with pytest.raises(ValueError, match="strictly less"):
        run_approx_err_from_args(bad_args)


def test_missing_approximation_raises(sample_meshes: Tuple[Path, Path]) -> None:
    """Supplying fewer than two mesh paths should raise ValueError."""
    (true_path, _) = sample_meshes
    bad_args = argparse.Namespace(
        min_val=0.0,
        max_val=0.1,
        paths=[str(true_path)],           # only TRUE mesh, no approximation
    )
    with pytest.raises(ValueError, match="Need at least one true mesh and one approximated mesh"):
        run_approx_err_from_args(bad_args)
