# Copyright 2025 Intrinsic Innovation LLC

"""
evaluate_convex_decompositions.py

This script provides a full evaluation and benchmarking pipeline for comparing uniform convex decomposition (vhacd/coacd)
to our method. It decomposes input `.glb` meshes into convex parts, computes geometric error metrics 
(e.g., Hausdorff-like distances), performs physical simulation-based 
robustness testing, and saves colored error visualizations and structured 
JSON results.

Key Features:
- Per-box and per-part convex decomposition using VHACD or CoACD.
- Bidirectional surface sampling for Hausdorff-like error analysis.
- Export of GLB and OBJ meshes with color-coded geometric errors.
- Interactive 3D visualizations using Plotly.
- Automated simulation experiments with varying decomposition parameters.
- CLI interface for reproducible, logged evaluation experiments.

Example usage (motor test):
    python testing/evaluate_convex_decompositions.py testing/data/models/motor.glb 1 -0.501 0.228 -0.031 -0.339 0.281 \
        0.031 1 -0.182 -0.008 0.171 -0.148 0.062 0.215 3 0.141 -0.010 0.177 0.227 0.064 0.220 4 0.138 \
            -0.013 -0.220 0.217 0.067 -0.182 3 -0.195 -0.005 -0.217 -0.148 0.058 -0.175 3 0.185 0.416 \
                -0.243 0.238 0.495 -0.169 3 -0.204 0.408 0.156 -0.115 0.501 0.243 3 --exp_name motor_test --method vhacd
"""

import argparse
import coacd
from pathlib import Path
from typing import List, Tuple, Dict, Any
from datetime import datetime
import json

import trimesh
from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import plotly.graph_objects as go

import sys

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from intrinsic.empart.decompose import (
    decompose_mesh,
    _create_cuboid_trimesh,
    _vhacd_params,
    _parse_boxes,
)
from intrinsic.empart.simulation_performance import package_and_simulate
import tempfile
import time
from manifold3d import Manifold, Mesh, OpType
from intrinsic.empart.utils import manifold_to_trimesh


def safe_sample_surface(mesh: trimesh.Trimesh, n_samples: int):
    """Safely sample *n_samples* points on *mesh*'s surface.

    If the mesh has no vertices (or n_samples == 0), return empty arrays so
    downstream code can gracefully skip expensive calculations without
    crashing.  The caller *must* check the returned array lengths before using
    them.
    """
    if mesh.vertices.shape[0] == 0 or n_samples == 0:
        return np.empty((0, 3)), np.empty((0,))

    points, face_indices = trimesh.sample.sample_surface(mesh, n_samples)

    return points, face_indices


def save_error_color(
    true_mesh_path: Path,
    approx_mesh_path: Path,
    output_path: Path,
    colormap_name: str = "bwr",
    n_samples: int = 200_00,
    return_distances: bool = False,
):
    """
    Colors the true mesh's vertices based on distance to the approximate mesh surface
    and optionally returns the sampled points and distances.

    Args:
        true_mesh_path (Path or ndarray): Path to the ground-truth mesh or sampled points.
        approx_mesh_path (Path or ndarray): Path to the approximate mesh or sampled points.
        output_path (Path): File path to save the colored true mesh.
        colormap_name (str): Matplotlib colormap to use (default: "bwr").
        n_samples (int): Number of surface samples (default: 20000).
        return_distances (bool): If True, returns sampled points and distance values.

    Returns:
        Tuple[ndarray, ndarray] | Tuple[None, None]:
            Sampled true points and distances if `return_distances` is True;
            otherwise, (None, None).
    """
    # Check if true_mesh_path is a string
    if isinstance(true_mesh_path, Path):
        mesh_true = trimesh.load(true_mesh_path, force="mesh")
        mesh_approx = trimesh.load(approx_mesh_path, force="mesh")

        sampled_pts, _ = safe_sample_surface(mesh_approx, n_samples)
        sampled_true, _ = safe_sample_surface(mesh_true, n_samples)
    else:
        sampled_true = true_mesh_path
        sampled_pts = approx_mesh_path

    tree_approx = cKDTree(sampled_pts)
    if sampled_pts.shape[0] == 0 or sampled_true.shape[0] == 0:
        return None, None
    distances, _ = tree_approx.query(sampled_true)

    # normalise distances → [0,1]
    min_val, max_val = distances.min(), distances.max()
    errors_norm = (np.clip(distances, min_val, max_val) - min_val) / (
        max_val - min_val + 1e-9
    )
    colours_rgba = plt.get_cmap(colormap_name)(errors_norm)
    colours_rgb = (colours_rgba[:, :3] * 255).astype(np.uint8)

    if isinstance(true_mesh_path, Path):
        mesh_true.visual.vertex_colors = colours_rgb
        trimesh.exchange.export.export_mesh(mesh_true, output_path)

    if return_distances:
        return sampled_true, distances
    return None, None


def _make_interactive_html(
    points: np.ndarray, dists: np.ndarray, html_path: Path, title: str
):
    """Write a Plotly scatter3d with hoverable error values."""
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(
                    size=2, color=dists, colorscale="RdBu", colorbar=dict(title="Error")
                ),
                text=[f"{d:.4f}" for d in dists],
                hovertemplate="Error: %{text}<extra></extra>",
            )
        ]
    )
    fig.update_layout(scene_aspectmode="data", title=title)
    fig.write_html(html_path)


def _render_and_save_errors(
    true_mesh: trimesh.Trimesh,
    approx_mesh: trimesh.Trimesh,
    exp_dir: Path,
    tag: str,
    n_samples: int = 2_000,
):
    """
    Generates and saves bidirectional geometric error visualizations between a
    ground-truth mesh and an approximated mesh using color-coded OBJ and interactive HTML.

    Args:
        true_mesh (trimesh.Trimesh | np.ndarray): Ground-truth mesh or pre-sampled points.
        approx_mesh (trimesh.Trimesh | np.ndarray): Approximate mesh or pre-sampled points.
        exp_dir (Path): Directory to save output files (meshes, visualizations).
        tag (str): Identifier prefix for naming output files.
        n_samples (int): Number of surface samples for distance evaluation (default: 2000).

    Returns:
        - Colored OBJ meshes showing per-vertex error (true→approx and approx→true).
        - Interactive Plotly HTML files visualizing the same directional errors.
    """
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Check if true_mesh is a mesh or nd array
    if not isinstance(true_mesh, np.ndarray) and (
        not isinstance(approx_mesh, np.ndarray)
    ):
        true_path = exp_dir / f"{tag}_true.glb"
        approx_path = exp_dir / f"{tag}_approx.glb"
        true_mesh.export(true_path)
        approx_mesh.export(approx_path)
    else:
        true_path = true_mesh
        approx_path = approx_mesh
    # true → approx ---------------------------------------------------------
    vertices, dists = save_error_color(
        true_path,
        approx_path,
        exp_dir / f"{tag}_err_true2approx.obj",
        n_samples=n_samples,
        return_distances=True,
    )
    if vertices is not None:
        _make_interactive_html(
            vertices, dists, exp_dir / f"{tag}_true2approx.html", f"{tag}: true→approx"
        )

    # approx → true ---------------------------------------------------------
    vertices, dists = save_error_color(
        approx_path,
        true_path,
        exp_dir / f"{tag}_err_approx2true.obj",
        n_samples=n_samples,
        return_distances=True,
    )

    if vertices is not None:
        _make_interactive_html(
            vertices, dists, exp_dir / f"{tag}_approx2true.html", f"{tag}: approx→true"
        )


def err_analysis(
    mesh_true_samples: np.array,
    mesh_approx_samples: np.array,
    return_details: bool = False,
    box: Dict[str, Any] = None,
):
    """
    Computes a bidirectional Hausdorff-like distance between sampled surfaces of
    a ground-truth and an approximate mesh, optionally within a specified bounding box.

    Args:
        mesh_true_samples (np.array): Tuple of (points, _) sampled from the true mesh surface.
        mesh_approx_samples (np.array): Tuple of (points, _) from the approximate mesh.
        return_details (bool): If True, includes metadata in the result.
        box (dict, optional): Axis-aligned bounding box.

    Returns:
        - If return_details is True:
            Tuple[float, dict]: Maximum distance and direction ('true→approx' or 'approx→true').
        - If return_details is False:
            Tuple[float, np.ndarray, np.ndarray]: Maximum distance and the original sampled points.
        - None if either point set is empty.
    """

    sampled_approx, _ = mesh_approx_samples
    sampled_true, _ = mesh_true_samples
    if sampled_approx.shape[0] > 0 and sampled_true.shape[0] > 0:

        if box is not None:
            # Filter out all points that are not inside the box
            min_box = np.array([box["min"]["x"], box["min"]["y"], box["min"]["z"]])
            max_box = np.array([box["max"]["x"], box["max"]["y"], box["max"]["z"]])
            mask = np.all((sampled_true >= min_box) & (sampled_true <= max_box), axis=1)
            sampled_true_filtered = sampled_true[mask]
            mask = np.all(
                (sampled_approx >= min_box) & (sampled_approx <= max_box), axis=1
            )
            sampled_approx_filtered = sampled_approx[mask]

        if sampled_approx.shape[0] == 0 or sampled_true.shape[0] == 0:
            return None

        tree_approx = cKDTree(sampled_approx)  #
        tree_true = cKDTree(sampled_true)

        d_true2appx, _ = tree_approx.query(sampled_true_filtered)
        d_appx2true, _ = tree_true.query(sampled_approx_filtered)

        # pick the larger of the two directions
        if d_true2appx.max() >= d_appx2true.max():
            max_val = float(d_true2appx.max())
            details = {
                "direction": "true→approx",
            }
        else:
            max_val = float(d_appx2true.max())
            details = {
                "direction": "approx→true",
            }

        return (
            (max_val, details) if return_details else max_val,
            sampled_true,
            sampled_approx,
        )
    return None


def eval_our_cut(
    boxes: List[float], input_glb: Path, coarse: float, method: str, exp_dir: Path
):
    """
    Applies custom convex decomposition to an input mesh using predefined box regions,
    computes per-part geometric errors, and generates visualizations.

    Args:
        boxes (List[float]): Flattened list of 7-tuples (xmin, ymin, zmin, xmax, ymax, zmax, coarse)
                             defining axis-aligned box regions for decomposition.
        input_glb (Path): Path to the input mesh (.glb) file.
        coarse (float): Global coarse decomposition parameter.
        method (str): Decomposition method ('vhacd' or 'coacd').
        exp_dir (Path): Directory where outputs (meshes, visualizations) will be saved.

    Returns:
        Tuple[
            float,                   # Mean of maximum errors per part (Hausdorff-like)
            trimesh.Trimesh,         # Concatenated mesh of all approximated regions
            int,                     # Number of convex hulls from decomposition
            List[trimesh.Trimesh],   # List of all individual parts
            Dict[str, Any]           # Log of per-part and overall maximum error
        ]
    """
    box_specs: List[Tuple[float, float, float, float, float, float, float]] = [
        tuple(boxes[i * 7 : (i + 1) * 7])  # type: ignore[arg-type]
        for i in range(len(boxes) // 7)
    ]

    slice_result = decompose_mesh(
        str(input_glb),
        coarse,
        boxes=box_specs,
        method=method,
        include_raw=True,
        as_meshes=True,
        resolution=50_000,  # VHACD resolution
    )

    mean_errs: list[float] = []
    all_approx_meshes: list[trimesh.Trimesh] = []
    all_parts: list[trimesh.Trimesh] = []
    part_err_logs: list[dict[str, Any]] = []
    overall_max = None
    boxes = _parse_boxes(boxes)

    # ---- iterate over sub‑objects returned by slicer ----------------------
    for mesh_id in slice_result["meshes"]["select_objs"]:
        raw_mesh: trimesh.Trimesh = slice_result["meshes"]["select_objs"][mesh_id][0][
            "raw_mesh"
        ]
        parts = [
            p["mesh"]
            for p in slice_result["meshes"]["select_objs"][mesh_id]
            if "mesh" in p
        ]
        all_parts.extend(parts)

        parts_cat = colorize_and_concatenate_parts(parts)
        colored_parts_cat = trimesh.util.concatenate(parts)

        all_approx_meshes.append(colored_parts_cat)

        approx_points = safe_sample_surface(parts_cat, 100_000)
        raw_mesh_points = safe_sample_surface(raw_mesh, 100_000)

        tup, filter_true, filter_approx = err_analysis(
            raw_mesh_points,  # true mesh
            approx_points,  # approx mesh
            return_details=True,
            box=boxes[int(mesh_id)],
        )
        err_val, details = tup

        if err_val is not None:
            part_err_logs.append(
                {
                    "part_id": mesh_id,
                    "max_error": err_val,
                    **details,
                }
            )
            mean_errs.append(err_val)  # <- now a straight list of per-part maxima

            # keep track of the global worst case
            if overall_max is None or err_val > overall_max["max_error"]:
                overall_max = {**part_err_logs[-1]}

        # Save coloured error meshes (optional)
        _render_and_save_errors(
            filter_true[
                np.random.choice(filter_true.shape[0], size=1000, replace=False)
            ],
            filter_approx[
                np.random.choice(filter_approx.shape[0], size=1000, replace=False)
            ],
            exp_dir,
            tag=f"our_slice_{mesh_id}",
        )
    # ---- non‑selected objects --------------------------------------------
    for it in slice_result["meshes"]["non_select_obj"]:
        all_approx_meshes.append(it["mesh"])
        all_parts.append(it["mesh"])

    valid_errs_overall = [e for e in mean_errs if e is not None]
    total_err = float(np.mean(valid_errs_overall)) if valid_errs_overall else None

    return (
        total_err,
        trimesh.util.concatenate(all_approx_meshes),
        slice_result["hulls"],
        all_parts,
        {
            "per_part": part_err_logs,
            "overall": overall_max,
        },
    )


def colorize_and_concatenate_parts(trimesh_parts, bool_union=True):
    """
    Takes a list of trimesh objects, assigns a random color to each part,
    and returns a single concatenated mesh. If bool_union is True, performs a union operation.

    Args:
        trimesh_parts (List[trimesh.Trimesh]): List of individual mesh parts.

    Returns:
        trimesh.Trimesh: A single mesh with random per-part coloring.
    """
    randomized_parts = []

    for part in trimesh_parts:
        part = part.copy()  # Avoid modifying the original mesh
        num_faces = len(part.faces)

        # Random RGB color (0–255)
        color = np.random.randint(0, 256, size=(1, 3), dtype=np.uint8)
        # Repeat for all faces and add alpha channel
        face_colors = np.hstack(
            [
                np.tile(color, (num_faces, 1)),
                255 * np.ones((num_faces, 1), dtype=np.uint8),
            ]
        )

        part.visual.face_colors = face_colors
        randomized_parts.append(part)

    if bool_union:
        all_m = [
            Manifold(
                Mesh(
                    np.array(m.vertices.astype("float32")),
                    np.array(m.faces.astype("int32")),
                )
            )
            for m in randomized_parts
        ]
        mesh_approx = manifold_to_trimesh(Manifold.batch_boolean(all_m, OpType.Add))
        assert mesh_approx.is_watertight
        return mesh_approx
    else:
        return trimesh.util.concatenate(randomized_parts)


def eval_vhacd_cut(
    args_boxes: List[float],
    input_glb: Path,
    hulls: int,
    resolution: int,
    exp_dir: Path,
    save_error=False,
    is_coacd=False,
):
    """
    Applies global convex decomposition (VHACD or CoACD) to a mesh and computes
    per-region geometric error metrics using Hausdorff-like distances.

    Args:
        args_boxes (List[float]): Flattened list of 7-tuples defining evaluation boxes
                                  (xmin, ymin, zmin, xmax, ymax, zmax, coarse).
        input_glb (Path): Path to the input .glb mesh file.
        hulls (int): Maximum number of convex hulls for decomposition.
        resolution (int): Voxel resolution for decomposition (affects quality).
        exp_dir (Path): Directory to save error visualizations (if enabled).
        save_error (bool): If True, saves colored OBJ and HTML error visualizations.
        is_coacd (bool): If True, uses CoACD instead of VHACD for decomposition.

    Returns:
        Tuple[
            float,                  # Mean of max errors per box
            trimesh.Trimesh,        # Colored, concatenated approximation mesh
            List[trimesh.Trimesh],  # Individual convex parts
            Dict[str, Any]          # Error log: per-box and global max error
        ]
    """
    true_mesh = trimesh.load(input_glb, force="mesh")

    if is_coacd:
        cm = coacd.Mesh(true_mesh.vertices, true_mesh.faces)
        parts = coacd.run_coacd(
            cm,
            max_convex_hull=_vhacd_params(hulls, resolution)["maxConvexHulls"],
            threshold=0.01,
            mcts_nodes=100,
            mcts_max_depth=2,
            merge=True,
        )
        trimesh_parts = [trimesh.Trimesh(p[0], p[1], process=False) for p in parts]
    else:
        outputs = trimesh.decomposition.convex_decomposition(
            true_mesh, **_vhacd_params(hulls, resolution=resolution)
        )
        trimesh_parts = [
            trimesh.Trimesh(p["vertices"], p["faces"], process=False) for p in outputs
        ]
    colored_mesh_approx = colorize_and_concatenate_parts(
        trimesh_parts, bool_union=False
    )
    mesh_approx = colorize_and_concatenate_parts(trimesh_parts)

    # ---- box‑by‑box error -------------------------------------------------
    boxes: List[Dict[str, Any]] = []
    for i in range(len(args_boxes) // 7):
        xmin, ymin, zmin, xmax, ymax, zmax, c = args_boxes[i * 7 : (i + 1) * 7]
        boxes.append(
            {
                "min": {"x": xmin, "y": ymin, "z": zmin},
                "max": {"x": xmax, "y": ymax, "z": zmax},
                "coarse": c,
            }
        )

    cuboids = [
        _create_cuboid_trimesh(
            b["min"]["x"],
            b["min"]["y"],
            b["min"]["z"],
            b["max"]["x"],
            b["max"]["y"],
            b["max"]["z"],
        )
        for b in boxes
    ]
    errs: list[float] = []
    slice_err_logs: list[dict[str, Any]] = []
    overall_max: dict[str, Any] = None

    true_mesh_samples = safe_sample_surface(true_mesh, 100_000)
    approx_mesh_samples = safe_sample_surface(mesh_approx, 100_000)
    for i, cuboid in enumerate(cuboids):

        tup, _, _ = err_analysis(
            true_mesh_samples, approx_mesh_samples, box=boxes[i], return_details=True
        )
        err_val, details = tup
        if err_val is not None:
            errs.append(err_val)

            # enrich with slice index for easier tracking
            slice_info = {
                "slice_id": i,
                "max_error": err_val,
                **details,  # 'direction', 'coords'
            }
            slice_err_logs.append(slice_info)

            if overall_max is None or err_val > overall_max["max_error"]:
                overall_max = slice_info

            if save_error:
                _render_and_save_errors(
                    true_mesh_samples[0],
                    approx_mesh_samples[0],
                    exp_dir,
                    tag=f"vhacd_{hulls}h_slice_{i}",
                )

    total_err = float(np.mean(errs)) if errs else None
    err_log = {"per_slice": slice_err_logs, "overall": overall_max}
    return total_err, colored_mesh_approx, trimesh_parts, err_log


def binary_search_hull_count(
    argss: argparse.Namespace, our_err: float, *, low: int, high: int
):
    """
    Performs a binary search to find the number of convex hulls for VHACD decomposition
    that yields an error closest to the target `our_err`. In case of a tie, prefers
    the smaller hull count.

    Args:
        argss (argparse.Namespace): Parsed CLI arguments including mesh path, boxes, and output directory.
        our_err (float): Target error value to match.
        low (int): Lower bound of the hull count search range.
        high (int): Upper bound of the hull count search range.

    Returns:
        Tuple[
            int,                   # Selected hull count with closest error
            float,                 # Corresponding VHACD error
            trimesh.Trimesh,       # Combined mesh of the best decomposition
            List[trimesh.Trimesh], # Individual convex parts
            Dict[str, Any]         # Error log with per-box and overall info
        ]
    """

    best_hull_count: int | None = None
    best_err: float = float("inf")
    best_mesh: trimesh.Trimesh | None = None
    best_parts: list[trimesh.Trimesh] | None = None
    resolution = 100_000  # resolution for the VHACD voxelization
    while low <= high:
        mid = (low + high) // 2
        vhacd_err, approx_mesh, trimesh_parts, vhacd_err_log = eval_vhacd_cut(
            argss.boxes,
            argss.input_glb,
            mid,
            resolution=resolution,
            exp_dir=argss.exp_dir,
        )

        print(
            f"Hull count {mid:>5}: vhacd_err={vhacd_err:.8f}, " f"target={our_err:.8f}"
        )

        # ---- 1. prefer-lower tie-breaker -----------------------------
        closer = abs(vhacd_err - our_err) < abs(best_err - our_err)
        tied = abs(vhacd_err - our_err) == abs(best_err - our_err)

        if closer or (tied and (best_hull_count is None or mid < best_hull_count)):
            best_err = vhacd_err
            best_hull_count = mid
            best_mesh = approx_mesh
            best_parts = trimesh_parts

        # ---- 2. classic binary-search step ---------------------------
        if vhacd_err < our_err:
            high = mid - 1  # search lower hull counts
        else:
            low = mid + 1  # search higher hull counts

    # ---- 3. final report (no extra eval call needed) -----------------
    print(
        f"Closest hull count: {best_hull_count}  "
        f"vhacd_err={best_err:.8f} vs target={our_err:.8f}"
    )
    best_err, best_mesh, best_parts, vhacd_err_log = eval_vhacd_cut(
        argss.boxes,
        argss.input_glb,
        best_hull_count,
        resolution=resolution,
        exp_dir=argss.exp_dir,
        save_error=True,
    )
    return best_hull_count, best_err, best_mesh, best_parts, vhacd_err_log


def run_global_convex_method(argss: argparse.Namespace, resolution, num_parts):
    """
    Executes a global convex decomposition (VHACD or CoACD) on the input mesh and
    returns the geometric error and resulting parts.

    Args:
        argss (argparse.Namespace): Parsed arguments containing mesh path, method, and box config.
        resolution (int): Voxel resolution for the decomposition.
        num_parts (int): Maximum number of convex parts (hulls) to generate.

    Returns:
        Tuple[
            float,                  # Mean max Hausdorff-like error across regions
            trimesh.Trimesh,        # Colored combined approximation mesh
            List[trimesh.Trimesh],  # List of individual convex parts
            Dict[str, Any]          # Error log (per-box and overall stats)
        ]
    """
    vhacd_err, approx_mesh, trimesh_parts, vhacd_err_log = eval_vhacd_cut(
        argss.boxes,
        argss.input_glb,
        num_parts,
        resolution=resolution,
        exp_dir=argss.exp_dir,
        save_error=True,
        is_coacd=argss.method == "coacd",  # Use CoACD if specified
    )

    return vhacd_err, approx_mesh, trimesh_parts, vhacd_err_log


def specify_hulls(num_hulls: int, boxes: List[float]) -> List[float]:
    """
    Replaces the per-box hull count (coarseness value) in a flattened list of box specs
    with a uniform value across all boxes.

    Args:
        num_hulls (int): Number of convex hulls to assign per box.
        boxes (List[float]): Flattened list of box definitions in the format
                             [xmin, ymin, zmin, xmax, ymax, zmax, coarse, ...].

    Returns:
        List[float]: Modified list with updated hull count for each box.
    """
    out = list(boxes)
    for i in range(6, len(out), 7):
        out[i] = num_hulls
    return out


def save_parts_to_glbs(
    parts: List[trimesh.Trimesh], out_dir: Path, prefix: str
) -> List[Path]:
    """
    Saves a list of mesh parts to individual `.glb` files in the specified directory,
    using a consistent filename prefix.

    Args:
        parts (List[trimesh.Trimesh]): List of mesh parts to export.
        out_dir (Path): Directory to save the GLB files.
        prefix (str): Filename prefix for each part (e.g., "part" → part_000.glb, part_001.glb).

    Returns:
        List[Path]: List of file paths to the saved `.glb` files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    glb_paths: list[Path] = []
    for idx, mesh in enumerate(parts):
        path = out_dir / f"{prefix}_{idx:03d}.glb"
        mesh.export(path, file_type="glb")
        glb_paths.append(path)
    return glb_paths


def run_parts_experiment(
    all_parts: List[trimesh.Trimesh],
    vhacd_parts: List[trimesh.Trimesh],
    *,
    scale: float = 1.0,
    duration: float = 0.1,
    render: bool = False,
    validate: bool = False,
    num_instances: int = 10,
    num_layers: int = 5,
    apply_random_forces: bool = True,
    random_force_mag: float = 25.0,
    base_exp_dir: str = None,
):
    with tempfile.TemporaryDirectory(prefix="parts_exp_") as tmp_root_str:
        tmp_root = Path(tmp_root_str)
        all_glbs = save_parts_to_glbs(all_parts, tmp_root / "all_glbs", "all")
        vhacd_glbs = save_parts_to_glbs(vhacd_parts, tmp_root / "best_glbs", "best")
        timestamp = datetime.now().strftime("%d_%H_%M_%S")
        all_metrics = package_and_simulate(
            all_glbs,
            model_name="our_model",
            scale=scale,
            duration=duration,
            render=render,
            validate=validate,
            video_name=f"exp_vids/vid_ours_{timestamp}.mp4",
            num_instances=num_instances,
            num_layers=num_layers,
            apply_random_forces=apply_random_forces,
            random_force_mag=random_force_mag,
            save_mesh=base_exp_dir,
        )

        best_metrics = package_and_simulate(
            vhacd_glbs,
            model_name="vhacd_model",
            scale=scale,
            duration=duration,
            render=render,
            validate=validate,
            video_name=f"exp_vids/vid_vhacd_{timestamp}.mp4",
            num_instances=num_instances,
            num_layers=num_layers,
            apply_random_forces=apply_random_forces,
            random_force_mag=random_force_mag,
            save_mesh=base_exp_dir,
        )
    return all_metrics, best_metrics


def run_parts_experiment_single(
    all_parts: List[trimesh.Trimesh],
    *,
    scale: float = 1.0,
    duration: float = 0.1,
    render: bool = False,
    validate: bool = False,
    num_instances: int = 10,
    num_layers: int = 5,
    apply_random_forces: bool = True,
    random_force_mag: float = 25.0,
    base_exp_dir: str = None,
    model_name="our_model",
):
    """
    Runs a physics-based simulation using a single set of mesh parts and returns
    performance metrics such as stability, contact count, and runtime.

    Args:
        all_parts (List[trimesh.Trimesh]): Mesh parts to be simulated.
        scale (float): Scale factor for object instances (default: 1.0).
        duration (float): Duration of the simulation in seconds (default: 0.1).
        render (bool): If True, renders and saves a simulation video.
        validate (bool): If True, performs post-simulation validation checks.
        num_instances (int): Number of repeated mesh instances in the simulation (default: 10).
        num_layers (int): Number of stacked object layers (default: 5).
        apply_random_forces (bool): If True, applies randomized forces during simulation.
        random_force_mag (float): Magnitude of random forces (default: 25.0).
        base_exp_dir (str or None): Directory for saving output data.
        model_name (str): Name of the model used for logging and output identification.

    Retrns:
        dict: A dictionary of simulation metrics (e.g., timing, contacts, constraints).
    """
    with tempfile.TemporaryDirectory(prefix="parts_exp_") as tmp_root_str:
        tmp_root = Path(tmp_root_str)
        all_glbs = save_parts_to_glbs(all_parts, tmp_root / "all_glbs", "all")
        timestamp = datetime.now().strftime("%d_%H_%M_%S")
        all_metrics = package_and_simulate(
            all_glbs,
            model_name=model_name,
            scale=scale,
            duration=duration,
            render=render,
            validate=validate,
            video_name=f"exp_vids/vid_ours_{timestamp}.mp4",
            num_instances=num_instances,
            num_layers=num_layers,
            apply_random_forces=apply_random_forces,
            random_force_mag=random_force_mag,
            save_mesh=base_exp_dir,
        )

    return all_metrics


def eval_independent(argss):
    """
    Runs independent evaluation of convex decomposition methods (VHACD/CoACD vs. our method)
    over varying hull counts. For each configuration, it performs decomposition, geometric error analysis,
    physics simulation, and stores metrics and visual results.

    Args:
        argss (argparse.Namespace): Parsed CLI arguments including:
            - input_glb: Path to input mesh
            - boxes: Box specifications
            - method: Decomposition method ('vhacd' or 'coacd')
            - exp_dir: Root directory for saving results

    Process:
        - For each global hull count (e.g., 10 to 10,000):
            - Run decomposition using VHACD or CoACD
            - Evaluate geometric accuracy and simulate part stability
            - Save combined mesh and JSON results

        - For each local (per-region) hull count:
            - Run custom decomposition using pre-defined boxes
            - Evaluate and simulate
            - Save results and output meshes

    Returns:
        - Saves per-configuration results as JSON files in `exp_dir`
        - Exports combined meshes and simulation metadata for comparison
    """
    base_exp_dir = argss.exp_dir  # keep a handle on the root
    for in_num_hulls in [100, 1000, 10000]:  # 3, 12, 2
        point_dir = base_exp_dir / f"global_method_data_point_{in_num_hulls:02d}"
        point_dir.mkdir(parents=True, exist_ok=True)
        argss.exp_dir = point_dir
        vhacd_err, approx_mesh, trimesh_parts, vhacd_err_log = run_global_convex_method(
            argss, resolution=500_000, num_parts=in_num_hulls
        )

        global_metrics = run_parts_experiment_single(
            trimesh_parts,
            model_name="global_method",
            scale=1.0,
            duration=0.25,
            render=False,
            validate=False,
            num_instances=5,
            num_layers=5,
            apply_random_forces=True,
            random_force_mag=30,
            base_exp_dir=argss.exp_dir,
        )

        approx_mesh.export(point_dir / "global_combined.obj")
        result = {
            "err": vhacd_err,
            "in_num_hulls": in_num_hulls,
            "num_hulls": len(trimesh_parts),
            "rt_factor": global_metrics.get("rt_factor"),
            "real_elapsed": global_metrics.get("real_elapsed"),
            "sim_elapsed": global_metrics.get("sim_elapsed"),
            "contacts": global_metrics.get("num_contacts"),
            "constraints": global_metrics.get("num_constraints"),
            "hausdorff_log": vhacd_err_log,
        }
        result_file = base_exp_dir / "result.json"
        if result_file.exists():
            with result_file.open("r") as fp:
                results = json.load(fp)
        else:
            results = []

        results.append(result)

        with result_file.open("w") as fp:
            json.dump(results, fp, indent=2)

    # Eval Our Method
    for num_hulls_per_region in range(1, 15, 3):  # 3, 12, 2
        point_dir = base_exp_dir / f"our_data_point_{num_hulls_per_region:02d}"
        point_dir.mkdir(parents=True, exist_ok=True)
        argss.exp_dir = point_dir
        our_err, our_mesh, num_hulls, our_parts, our_err_log = eval_our_cut(
            specify_hulls(num_hulls_per_region, argss.boxes),
            argss.input_glb,
            1,
            argss.method,
            exp_dir=argss.exp_dir,
        )
        our_metrics = run_parts_experiment_single(
            our_parts,
            model_name="our_model",
            scale=1.0,
            duration=0.25,
            render=False,
            validate=False,
            num_instances=5,
            num_layers=5,
            apply_random_forces=True,
            random_force_mag=30,
            base_exp_dir=argss.exp_dir,
        )
        our_mesh.export(point_dir / "ours_combined.obj")
        result = {
            "in_num_hulls": num_hulls_per_region,
            "method_err": our_err,
            "num_hulls": num_hulls,
            "rt_factor": our_metrics.get("rt_factor"),
            "real_elapsed": our_metrics.get("real_elapsed"),
            "sim_elapsed": our_metrics.get("sim_elapsed"),
            "contacts": our_metrics.get("num_contacts"),
            "constraints": our_metrics.get("num_constraints"),
            "hausdorff_log": our_err_log,
        }
        result_file = base_exp_dir / "result.json"
        if result_file.exists():
            with result_file.open("r") as fp:
                results = json.load(fp)
        else:
            results = []

        results.append(result)

        with result_file.open("w") as fp:
            json.dump(results, fp, indent=2)


def eval_all(argss: argparse.Namespace):
    """
    Runs a full evaluation loop comparing the custom per-region convex decomposition
    method against VHACD, using binary search to find the best VHACD hull count that
    matches the custom method's error.

    Args:
        argss (argparse.Namespace): Parsed command-line arguments containing:
            - input_glb: Path to input mesh (.glb)
            - boxes: Flattened list of region boxes
            - method: Decomposition method (e.g., "vhacd" or "coacd")
            - exp_dir: Output directory
            - exp_name: Experiment identifier string

    Process:
        - Iterates over increasing numbers of hulls per region.
        - For each setting:
            - Runs the custom method and computes error.
            - Uses binary search to find the VHACD hull count that most closely matches that error.
            - Runs physical simulations for both methods.
            - Records and exports:
                - Error metrics (Hausdorff-like)
                - Runtime and contact statistics
                - Combined mesh files (.obj)
                - JSON logs with per-iteration results

    Returns:
        - Saves individual results in per-config folders.
        - Consolidates all results into a master JSON file in `exp_dir`.
    """
    results: list[dict[str, Any]] = []
    base_exp_dir = argss.exp_dir  # keep a handle on the root
    start_time = time.time()
    for num_hulls_per_region in range(1, 6, 1):  # 3, 12, 2
        point_dir = base_exp_dir / f"data_point_{num_hulls_per_region:02d}"
        point_dir.mkdir(parents=True, exist_ok=True)
        argss.exp_dir = point_dir
        our_err, our_mesh, num_hulls, our_parts, our_err_log = eval_our_cut(
            specify_hulls(num_hulls_per_region, argss.boxes),
            argss.input_glb,
            1,
            argss.method,
            exp_dir=argss.exp_dir,
        )
        best_hull_count, best_err, best_mesh, best_parts, vhacd_err_log = (
            binary_search_hull_count(argss, our_err, low=1, high=500)
        )

        our_metrics, vhacd_metrics = run_parts_experiment(
            our_parts,
            best_parts,
            scale=1.0,
            duration=0.25,
            render=False,
            validate=False,
            num_instances=5,
            num_layers=5,
            apply_random_forces=True,
            random_force_mag=30,
            base_exp_dir=argss.exp_dir,
        )
        try:
            if our_mesh is not None:
                our_mesh.export(point_dir / "ours_combined.obj")
            if best_mesh is not None:
                best_mesh.export(point_dir / "vhacd_combined.obj")
        except Exception as e:
            print(f"[warn] failed to save combined meshes: {e}")
        result = {
            "num_hulls_input_our_per_region": num_hulls_per_region,
            "our_method_err": our_err,
            "our_method_num_hulls": num_hulls,
            "best_hull_count_vhacd": best_hull_count,
            "best_err_vhacd": best_err,
            "our_rt_factor": our_metrics.get("rt_factor"),
            "our_real_elapsed": our_metrics.get("real_elapsed"),
            "our_sim_elapsed": our_metrics.get("sim_elapsed"),
            "our_contacts": our_metrics.get("num_contacts"),
            "our_constraints": our_metrics.get("num_constraints"),
            "vhacd_rt_factor": vhacd_metrics.get("rt_factor"),
            "vhacd_real_elapsed": vhacd_metrics.get("real_elapsed"),
            "vhacd_sim_elapsed": vhacd_metrics.get("sim_elapsed"),
            "vhacd_contacts": vhacd_metrics.get("num_contacts"),
            "vhacd_num_constraints": vhacd_metrics.get("num_constraints"),
            "hausdorff_log_ours": our_err_log,
            "hausdorff_log_vhacd": vhacd_err_log,
        }
        with (point_dir / "result.json").open("w") as fp:
            json.dump(result, fp, indent=2)
        results.append(result)
        print(result)

    # ---- save JSON inside experiment folder -------------------------------
    json_path = argss.exp_dir / f"experiment_results_{argss.exp_name}.json"
    with json_path.open("w") as fp:
        json.dump(results, fp, indent=2)
    print(f"Results written to {json_path} in {(time.time() - start_time)/1000}")


def _parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for mesh slicing, convex decomposition, and benchmarking.

    RETURNS:
        argparse.Namespace: An object containing all parsed arguments including:
            - input_glb (Path): Path to the ground-truth mesh (.glb file).
            - global_coarse (float): Coarseness value for global convex decomposition.
            - method (str): Decomposition method to use ("vhacd" or "coacd").
            - out (Path or None): Optional path to write the JSON report.
            - exp_name (str): Experiment name used to create output directory.
            - boxes (List[float]): List of box definitions (groups of 7 floats).
    """
    p = argparse.ArgumentParser(
        description="Slice a mesh into axis‑aligned boxes, compute per‑box errors, and benchmark convex decompositions.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    p.add_argument("input_glb", type=Path, help="Path to the *ground‑truth* .glb")
    p.add_argument(
        "global_coarse",
        type=float,
        help="Global convex‑decomposition coarseness (0 → disabled)",
    )
    p.add_argument(
        "--method",
        choices=["vhacd", "coacd"],
        default="vhacd",
        help="Convex‑decomposition backend (default: vhacd)",
    )
    p.add_argument(
        "--out",
        type=Path,
        metavar="FILE",
        help="Optional path to write the JSON report (overridden by --exp_name)",
    )
    p.add_argument(
        "--exp_name",
        type=str,
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Experiment name; directory experiments_log/<exp_name>/ will be created (default: timestamp)",
    )
    p.add_argument(
        "boxes",
        nargs="*",
        type=float,
        help="zero or more 7‑number groups (xmin ymin zmin xmax ymax zmax coarse)",
    )

    return p.parse_intermixed_args()


# ──────────────────────────────────────────────────────────────
# 8) Entry‑point ------------------------------------------------------------
# ──────────────────────────────────────────────────────────────
def main():
    argss = _parse_args()

    # create experiments_log/<exp_name>/ -----------------------
    argss.exp_dir = Path("experiments_log") / argss.exp_name
    argss.exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment directory: {argss.exp_dir}")

    eval_independent(argss)


if __name__ == "__main__":
    main()
