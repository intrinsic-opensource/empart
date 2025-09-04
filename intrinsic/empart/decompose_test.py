# Copyright 2025 Intrinsic Innovation LLC

"""
Test cases for the decompose module.
"""

import argparse
from pathlib import Path
from typing import List
import trimesh

import pytest

from .decompose import run_decompose_from_args

TEST_GLB = Path(__file__).resolve().parents[2] / "test" / "data" / "models" / "motor.glb"


def _extract_volumes(result) -> list[float]:
    """Flatten all 'volume' entries into a deterministic list."""
    vols = [float(h["volume"]) for h in result["meshes"]["non_select_obj"]]
    vols.extend(
        float(o["volume"])
        for key in sorted(result["meshes"]["select_objs"])
        for o in result["meshes"]["select_objs"][key]
    )
    return vols


# --------------------------------------------------------------------------- #
# local helper: build the Namespace exactly like the CLI                      #
# --------------------------------------------------------------------------- #
def _parse(argv: List[str]) -> argparse.Namespace:
    """
    Re-use the CLI schema so the tests stay in lock-step with
    whatever the real parser accepts.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("input_glb")
    parser.add_argument("non_select_convex_hull_max", type=float)
    parser.add_argument("--method", choices=["vhacd", "coacd"], default="vhacd")
    parser.add_argument("--include_raw", action="store_true")
    parser.add_argument("resolution", type=int, default=50000)
    parser.add_argument("boxes", nargs="*", type=float)

    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# single-box test                                                             #
# --------------------------------------------------------------------------- #
def test_single_box():
    raw = "1 50000 -0.665 0.212 -0.051 -0.335 0.267 0.049 0".split()
    ns = _parse([str(TEST_GLB), *raw])  # ← build Namespace

    result = run_decompose_from_args(ns, as_meshes=True)

    assert result["hulls"] == 5


# --------------------------------------------------------------------------- #
# full motor test                                                            #
# --------------------------------------------------------------------------- #
def test_full_motor():
    raw = (
        "5 50000 -0.665 0.212 -0.051 -0.335 0.267 0.049 1 -0.190 0.010 0.176 -0.138 0.046 0.247 3 0.152 0.012 0.174 0.232 0.046 0.247 3 -0.193 0.009 -0.247 -0.126 0.046 -0.181 5 0.141 0.008 -0.247 0.249 0.046 -0.179 5"
    ).split()
    ns = _parse([str(TEST_GLB), *raw])

    result = run_decompose_from_args(ns, as_meshes=True)
    assert result["hulls"] == 50


def test_single_hull():
    raw = ("1 50000").split()
    ns = _parse([str(TEST_GLB), *raw])
    test_tri = trimesh.load(TEST_GLB, force="mesh")
    result = run_decompose_from_args(ns, as_meshes=True)
    convex_hull = result["meshes"]["non_select_obj"][0]["mesh"]

    num_leftover = test_tri.difference(convex_hull, engine="manifold").vertices.shape[0]
    num_target = test_tri.difference(
        test_tri.convex_hull, engine="manifold"
    ).vertices.shape[0]

    assert num_leftover <= 20
