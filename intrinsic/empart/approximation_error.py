# Copyright 2025 Intrinsic Innovation LLC

"""
Calculates approximation error between a true mesh and one or more approximated meshes.

The error is visualized by coloring the vertices of the true mesh based on the distance to the closest point on the approximated mesh.
The resulting mesh can be saved to a file or returned as a Base64-encoded string for JSON transport.
"""

import argparse

import matplotlib
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from manifold3d import Manifold, Mesh, OpType
from typing import List, Dict, Optional, Union
from .utils import encode_mesh_bytes, manifold_to_trimesh, trimesh_to_manifold
from matplotlib.colors import LinearSegmentedColormap
from manifold3d import Manifold, Mesh, OpType
from trimesh.points import PointCloud


def _regions_to_dicts(region_list):
    """
    Convert a list of regions in the format [minx,miny,minz,maxx,maxy,maxz, ...] to a list of dicts.

    Args:
        region_list: List of lists or tuples, each containing 6 float values representing the min
        and max coordinates of a 3D bounding box.
    Returns:
        List of dicts with 'min' and 'max' keys, each containing 'x', 'y', and 'z' coordinates.
    """
    filtered = []
    for vals in region_list:
        mn = vals[:3]
        mx = vals[3:]
        filtered.append({
            "min": {"x": mn[0], "y": mn[1], "z": mn[2]},
            "max": {"x": mx[0], "y": mx[1], "z": mx[2]},
        })
    return filtered

def _boxes_union_from_regions(filtered_regions):
    """
    Create one mesh that is the union of all region boxes using the manifold engine.

    Args:
        filtered_regions: list[{"min": {"x":..,"y":..,"z":..},
                                "max": {"x":..,"y":..,"z":..}}]
    
    Returns:
        trimesh.Trimesh or None if no regions.
    """
    if not filtered_regions:
        return None
    filtered_regions = _regions_to_dicts(filtered_regions)
    boxes = []
    for region in filtered_regions:
        min_coords = np.array([region["min"]["x"], region["min"]["y"], region["min"]["z"]], dtype=float)
        max_coords = np.array([region["max"]["x"], region["max"]["y"], region["max"]["z"]], dtype=float)
        extents = max_coords - min_coords
        center = (min_coords + max_coords) / 2.0
        T = trimesh.transformations.translation_matrix(center)
        box = trimesh.creation.box(extents=extents, transform=T)
        boxes.append(box)

    # If only one box, return it directly
    if len(boxes) == 1:
        return boxes[0]

    # Union them via manifold
    try:
        union = trimesh.boolean.union(boxes, engine="manifold")
    except Exception:
        # optional fallback
        union = trimesh.boolean.union(boxes, engine="blender")
    return union

def _transfer_colors(src: trimesh.Trimesh, dst: trimesh.Trimesh):
    """
    Copy colors from src -> dst even after topology changes.
    Supports both 'face' and 'vertex' visuals

    Args:
        src (trimesh.Trimesh): Source mesh with colors.
        dst (trimesh.Trimesh): Destination mesh to transfer colors to.
    
    Returns:
        None: The function modifies dst in place.
    """

    if src.visual.kind == 'face' and hasattr(src.visual, 'face_colors'):
        # map each dst face center to the nearest src face center
        src_centers = src.triangles_center
        dst_centers = dst.triangles_center
        tree = cKDTree(src_centers)
        _, nn = tree.query(dst_centers, k=1)
        dst.visual.face_colors = src.visual.face_colors[nn]

    elif src.visual.kind == 'vertex' and hasattr(src.visual, 'vertex_colors'):
        # map each dst vertex to the nearest src vertex
        tree = cKDTree(src.vertices)
        _, nn = tree.query(dst.vertices, k=1)
        dst.visual.vertex_colors = src.visual.vertex_colors[nn]

    else:
        # No color data on src; nothing to do.
        pass

def cut_regions_out_of_true(mesh_true, filtered_regions):
    """
    Cut the true mesh by the filtered regions, returning a new mesh.

    Args:
        mesh_true (trimesh.Trimesh): The true mesh to be cut.
        filtered_regions (List[Dict[str, Dict[str, float]]]): List of regions to cut out.
    
    Returns:
        trimesh.Trimesh: The cut mesh.
    """
    if not filtered_regions:
        return mesh_true

    cut_volume = _boxes_union_from_regions(filtered_regions)
    if cut_volume is None:
        return mesh_true
    try:
        # manifold boolean intersection
        mesh_true_cut = manifold_to_trimesh(Manifold.batch_boolean([trimesh_to_manifold(mesh_true), trimesh_to_manifold(cut_volume)], OpType.Intersect))
        
        _transfer_colors(mesh_true, mesh_true_cut)

    except Exception:
        # fallback engine
        mesh_true_cut = trimesh.boolean.intersection([mesh_true, cut_volume], engine="blender")

        _transfer_colors(mesh_true, mesh_true_cut)

    return mesh_true_cut

def save_error_colored_mesh_sampled_to_vertices_on_true(
    true_mesh_path: str,
    approx_mesh_paths: List[str],
    output_path: Optional[str],
    colormap_name: str = "bwr",
    n_samples: int = 20000,
    min_val: float = 0.0,
    max_val: Optional[float] = 0.05,
    filtered_regions: Optional[List[Dict[str, Dict[str, float]]]] = None,
) -> Optional[Union[bytes, str]]:
    """
    Color the vertices of the true mesh based on the distance to the closest point on the approximated mesh.

    Args:
        true_mesh_path (str): Path to the true mesh file.
        approx_mesh_paths (List[str]): List of paths to the approximated mesh files.
        output_path (Optional[str]): Path to save the colored mesh. If None, returns the mesh as bytes.
        colormap_name (str): Name of the colormap to use for coloring.
        n_samples (int): Number of points to sample on the surface of the approximated mesh.
        min_val (float): Minimum value for the colormap normalization.
        max_val (float): Maximum value for the colormap normalization.
        filtered_regions (Optional[List[Dict[str, Dict[str, float]]]]): List of regions to filter the error visualization.

    Returns:
        Optional[Union[bytes, str]]: If output_path is None, returns the mesh as bytes. Otherwise, saves the mesh to the specified path.
    """

    mesh_true = trimesh.load(true_mesh_path, force="mesh")

    approx_meshes = [trimesh.load(p, force="mesh") for p in approx_mesh_paths]

    all_m = [
        Manifold(
            Mesh(
                np.array(m.vertices.astype("float32")),
                np.array(m.faces.astype("int32")),
            )
        )
        for m in approx_meshes
    ]
    mesh_approx = manifold_to_trimesh(Manifold.batch_boolean(all_m, OpType.Add))
    assert mesh_approx.is_watertight

    # Sample points on the surface of the APPROX mesh
    sampled_points, _ = trimesh.sample.sample_surface(mesh_approx, n_samples)

    # Build KD-Tree on APPROX mesh sampled points
    tree_approx = cKDTree(sampled_points)

    # Compute distances from TRUE mesh vertices to the sampled APPROX surface
    distances, _ = tree_approx.query(mesh_true.vertices)

    if max_val is not None:
        distances_clamped = np.clip(distances, min_val, max_val)
    else:
        max_val = np.percentile(distances, 97)
        distances_clamped = np.clip(distances, 0, max_val)

    if max_val > min_val:
        errors_norm = (distances_clamped - min_val) / (max_val - min_val)
    else:
        errors_norm = np.zeros_like(distances_clamped)

    cmap = matplotlib.colormaps[colormap_name]
    colors_rgba = cmap(errors_norm)  # (N,4) array in [0,1]
    colors_rgb = (colors_rgba[:, :3] * 255).astype(np.uint8)

    # Assign vertex colors to TRUE mesh
    mesh_true.visual.vertex_colors = colors_rgb
    # U the filtered regions to cut out the boxes fm the true mesh
    mesh_true = cut_regions_out_of_true(mesh_true, filtered_regions)
    # Export the colored TRUE mesh
    if output_path is not None:
        mesh_true.export(output_path)
        return None
    else:
        return mesh_true.export(file_type="ply")


def save_error_colored_mesh_sampled_to_vertices_on_approx(
    true_mesh_path: str,
    approx_mesh_paths: List[str],
    output_path: Optional[str],
    colormap_name: str = "bwr",
    n_samples: int = 20000,
    min_val: float = 0.0,
    max_val: Optional[float] = 0.05,
) -> Optional[Union[bytes, str]]:
    """
    Color the vertices of the approx mesh based on the distance to the closest point on the true mesh.

    Args:
        true_mesh_path (str): Path to the true mesh file.
        approx_mesh_paths (List[str]): List of paths to the approximated mesh files.
        output_path (Optional[str]): Path to save the colored mesh. If None, returns the mesh as bytes.
        colormap_name (str): Name of the colormap to use for coloring.
        n_samples (int): Number of points to sample on the surface of the approximated mesh.
        min_val (float): Minimum value for the colormap normalization.
        max_val (float): Maximum value for the colormap normalization.

    Returns:
        Optional[Union[bytes, str]]: If output_path is None, returns the mesh as bytes. Otherwise, saves the mesh to the specified path.
    """

    mesh_true = trimesh.load(true_mesh_path, force="mesh")

    approx_meshes = [trimesh.load(p, force="mesh") for p in approx_mesh_paths]

    all_m = [
        Manifold(
            Mesh(
                np.array(m.vertices.astype("float32")),
                np.array(m.faces.astype("int32")),
            )
        )
        for m in approx_meshes
    ]
    mesh_approx = manifold_to_trimesh(Manifold.batch_boolean(all_m, OpType.Add))
    assert mesh_approx.is_watertight

    points, _ = trimesh.sample.sample_surface(mesh_approx, count=100000)

    # Build KD-Tree on APPROX mesh sampled points
    tree_true = cKDTree(trimesh.sample.sample_surface(mesh_true, count=1000000)[0])

    # Compute distances from TRUE mesh vertices to the sampled APPROX surface
    distances, _ = tree_true.query(points)
    
 
    if max_val is not None:
        distances_clamped = np.clip(distances, min_val, max_val)
    else:
        max_val = np.percentile(distances, 97)
        distances_clamped = np.clip(distances, 0, max_val)

    if max_val > min_val:
        errors_norm = (distances_clamped - min_val) / (max_val - min_val)
    else:
        errors_norm = np.zeros_like(distances_clamped)

    cmap = matplotlib.colormaps[colormap_name]
    colors_rgba = cmap(errors_norm)  # (N,4) array in [0,1]
    colors_rgb = (colors_rgba[:, :3] * 255).astype(np.uint8)

    pc = PointCloud(points, colors=colors_rgb)
 
    if output_path is not None:
        pc.export(output_path)
        return None
    else:
        return pc.export(file_type="ply")

def run_approx_err_from_args(args: argparse.Namespace) -> Dict[str, List[str]]:
    """Run the approximation error calculation from command line arguments.

    Args:
        Argv (List[str]): Command line arguments in the following order:
            display_true_error (bool) - Whether to display the true mesh error or approx mesh error.
            filtered_regions (List[Dict[str, Dict[str, float]]]): List of regions to filter the error calculation.
            min_val (str)          - Minimum threshold value.
            max_val (str)          - Maximum threshold value.
            true_mesh_path (str)   - Path to the ground truth mesh file.
            approx_mesh_paths (str)- Path(s) to predicted/approximate mesh file(s).

    Returns:
        Dict[str, List[str]]: Dictionary containing the Base64-encoded mesh.
    """

    if args.min_val >= args.max_val:
        raise ValueError("<min> must be strictly less than <max>")

    if len(args.paths) < 2:
        raise ValueError("Need at least one true mesh and one approximated mesh")

    true_mesh_path = args.paths[0]
    approx_paths = args.paths[1:]

    if "white_red" not in matplotlib.colormaps:
        white_red = LinearSegmentedColormap.from_list(
            "white_red", ["#ffffff", "#ff0000"], N=256
        )
        matplotlib.colormaps.register(cmap=white_red)
    
    if args.display_true_error:
        mesh_bytes = save_error_colored_mesh_sampled_to_vertices_on_true(
            true_mesh_path=true_mesh_path,
            approx_mesh_paths=approx_paths,
            output_path=None,
            colormap_name="white_red",  # bwr
            n_samples=20_000,
            min_val=args.min_val,
            max_val=args.max_val,
            filtered_regions=args.region if args.region else None,
        )
    else:
        mesh_bytes = save_error_colored_mesh_sampled_to_vertices_on_approx(
            true_mesh_path=true_mesh_path,
            approx_mesh_paths=approx_paths,
            output_path=None,
            colormap_name="white_red",  # bwr
            n_samples=20_000,
            min_val=args.min_val,
            max_val=args.max_val,
        )

    return {"error_meshes": [encode_mesh_bytes(mesh_bytes)]}
