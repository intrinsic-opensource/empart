#!/usr/bin/env python3
# Copyright 2025 Intrinsic Innovation LLC

"""
Mesh processing CLI.

This CLI provides various mesh processing functionalities such as:
    • Decimation
    • Approximation error calculation
    • Simulation performance evaluation
    • Voxelization and watertight conversion
    • Decomposition of meshes into convex parts
    • Watertightness checking
"""
import argparse, json, logging, os, sys, open3d as o3d
from typing import Dict

from .decompose import run_decompose_from_args
from .simulation_performance import run_perf_from_args
from .approximation_error import run_approx_err_from_args
from .decimate import run_decimate_from_args
from .check_watertight import run_is_mesh_watertight
from .voxelize import run_voxel_watertight_from_args

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
logging.getLogger("trimesh").setLevel(logging.ERROR)
os.environ["PYTHONWARNINGS"] = "ignore"


def _json(result: Dict):
    """Print JSON or bail with a non-zero code."""
    try:
        print(json.dumps(result))
    except Exception as exc:  # pragma: no cover
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)


def _decimate(ns: argparse.Namespace):
    _json(run_decimate_from_args(ns))


def _approx_error(ns: argparse.Namespace):
    _json(run_approx_err_from_args(ns))


def _sim_perf(ns: argparse.Namespace):
    _json(run_perf_from_args(ns))


def _voxelize(ns: argparse.Namespace):
    _json(run_voxel_watertight_from_args(ns))


def _decompose(ns: argparse.Namespace):
    _json(run_decompose_from_args(ns, as_meshes=False))


def _check_watertight(ns: argparse.Namespace):
    _json(run_is_mesh_watertight(ns.input))


# ── CLI builder ───────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mesh processing CLI")
    sp = p.add_subparsers(dest="cmd", required=True)

    # decimate ------------------------------------------------------------------
    d = sp.add_parser("decimate", help="Simplify a mesh to a target triangle count")
    d.add_argument("input"), d.add_argument("triangles", type=int)
    d.set_defaults(func=_decimate)

    # approximation_error -------------------------------------------------------
    a = sp.add_parser(
        "approximation_error",
        help="Colour true mesh by per-vertex error and emit base-64 JSON",
    )
    a.add_argument("--display_true_error", action="store_true", default=False)
    a.add_argument(
        "--region",
        metavar=("MINX", "MINY", "MINZ", "MAXX", "MAXY", "MAXZ"),
        nargs=6,
        type=float,
        action="append",
        default=[],
        help="One AABB per flag: MINX MINY MINZ MAXX MAXY MAXZ. Repeat --region to add more.",
    )
    a.add_argument("min_val", type=float), a.add_argument("max_val", type=float)
    a.add_argument("paths", nargs="+")
    a.set_defaults(func=_approx_error)

    # simulation_performance ----------------------------------------------------
    s = sp.add_parser("simulation_performance", help="Evaluate simulation performance")
    s.add_argument("glbs", nargs="+")
    s.add_argument("-o", "--out", default="mujoco_pkg")
    s.add_argument("-m", "--meshdir")
    s.add_argument("-n", "--name", default="generated_model")
    s.add_argument("--scale", type=float, default=1.0)
    s.add_argument("--validate", action="store_true")
    s.set_defaults(func=_sim_perf)

    # voxelize ------------------------------------------------------------------
    v = sp.add_parser("voxelize", help="Voxelize a mesh and make it watertight")
    v.add_argument("input"), v.add_argument("--pitch", type=float)
    v.set_defaults(func=_voxelize)

    # decompose -----------------------------------------------------------------
    c = sp.add_parser("decompose", help="Slice and convex-decompose a .glb mesh")
    c.add_argument("input_glb", help="Input .glb file")
    c.add_argument(
        "non_select_convex_hull_max",
        type=int,
        help="Max number of convex hulls for non-selected region",
    )
    c.add_argument(
        "--method",
        choices=["vhacd", "coacd"],
        default="vhacd",
        help="Decomposition method",
    )
    c.add_argument(
        "--include_raw", action="store_true", help="Include raw mesh in output"
    )
    c.add_argument(
        "--resolution",
        type=int,
        default=50000,
        help="Resolution used for decomposition (default: 50000)",
    )
    c.add_argument(
        "boxes",
        nargs="*",
        type=float,
        help="List of box parameters for decomposition (min_x, min_y, min_z, max_x, max_y, max_z, max_hulls)",
    )
    c.set_defaults(func=_decompose)

    # check_watertight ----------------------------------------------------------
    w = sp.add_parser("check_watertight", help="Return JSON {watertight: bool}")
    w.add_argument("input")
    w.set_defaults(func=_check_watertight)

    return p


# ── main ──────────────────────────────────────────────────────────────────────
def main(argv: list[str] = None) -> None:
    ns = build_parser().parse_args(argv)
    ns.func(ns)  # parsed Namespace handed straight in


if __name__ == "__main__":  # pragma: no cover
    main()
