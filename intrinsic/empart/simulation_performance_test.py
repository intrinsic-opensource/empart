# Copyright 2025 Intrinsic Innovation LLC

from pathlib import Path
import pytest
import trimesh

# Skip the whole file if MuJoCo isn't installed in the test environment
pytest.importorskip("mujoco")

from .simulation_performance import run_helper


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------


def make_cube_glb(path: Path, size: float = 0.01) -> None:
    """Create a unit cube and export it as a .glb file on disk."""
    cube = trimesh.creation.box(extents=(size, size, size))
    cube.export(path)  # format inferred from suffix


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_run_helper_real_sim(tmp_path):
    """
    End-to-end check on the *library* API (run_helper):
    • build cube → GLB
    • call run_helper
    • verify returned dict contains sane metrics
    """
    cube_glb = tmp_path / "cube.glb"
    make_cube_glb(cube_glb)

    out_dir = tmp_path / "pkg_out"
    metrics = run_helper(
        glbs=[str(cube_glb)],
        out=str(out_dir),
        scale=1.0,
        validate=False,  # speed: skip validation pass
    )

    # It should return a dict with these keys:
    expected_keys = {
        "sim_elapsed",
        "real_elapsed",
        "rt_factor",
        "num_contacts",
        "num_constraints",
    }
    assert isinstance(metrics, dict), "run_helper must return a dict"
    assert expected_keys <= set(
        metrics.keys()
    ), "missing one or more expected metric keys"

    # Basic sanity: simulation time should be small but > 0, and at least one contact occurred
    assert 0.01 < metrics["sim_elapsed"] < 0.1
    assert round(metrics["num_contacts"]) >= 1
