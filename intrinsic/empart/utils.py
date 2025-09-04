# Copyright 2025 Intrinsic Innovation LLC

"""
Utility functions for geometry operations, including mesh encoding,
bounding box calculations, and convex hull merging.
"""
import numpy as np
import base64
from manifold3d import Manifold, Mesh  # type: ignore
import trimesh
import heapq
from typing import Any, List, Tuple


def encode_mesh_bytes(mesh_bytes: bytes) -> str:
    """Return a Base64-ASCII string suitable for JSON transport."""
    return base64.b64encode(mesh_bytes).decode()


def bounding_box_planes(
    box: dict[str, dict[str, float]],
) -> List[Tuple[np.ndarray[Any, Any], float]]:
    """Return a list of (normal, origin_offset) pairs for *box*.

    The *box* argument is a mapping with keys ``min`` and ``max`` whose values
    are sub-mappings providing ``x``, ``y`` and ``z`` floats (same schema used
    to create the cuboid Manifolds).  Normals point *outward* from the box.
    """
    xmin, ymin, zmin = box["min"]["x"], box["min"]["y"], box["min"]["z"]
    xmax, ymax, zmax = box["max"]["x"], box["max"]["y"], box["max"]["z"]

    return [
        (np.array([1.0, 0.0, 0.0]), xmin),  # plane x = xmin (+X inward)
        (np.array([-1.0, 0.0, 0.0]), -xmax),  # plane x = xmax (−X inward)
        (np.array([0.0, 1.0, 0.0]), ymin),  # plane y = ymin (+Y inward)
        (np.array([0.0, -1.0, 0.0]), -ymax),  # plane y = ymax (−Y inward)
        (np.array([0.0, 0.0, 1.0]), zmin),  # plane z = zmin (+Z inward)
        (np.array([0.0, 0.0, -1.0]), -zmax),  # plane z = zmax (−Z inward)
    ]


def manifold_to_trimesh(man:  Manifold) -> trimesh.Trimesh:
    """Convert a Manifold to a trimesh.Trimesh object."""
    mesh_data = man.to_mesh()
    vertices = np.asarray(mesh_data.vert_properties, dtype=np.float32)
    faces = np.asarray(mesh_data.tri_verts, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh


def trimesh_to_manifold(mesh: trimesh.Trimesh) -> Manifold:
    """
    Convert a trimesh.Trimesh object to a Manifold.
    """
    if not mesh.is_watertight:
        raise ValueError("Mesh must be watertight to be converted to Manifold.")

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    # Construct the Mesh object
    m = Mesh(vertices, faces)


    # Return the Manifold object
    return Manifold(m)

def is_aabb_touch(
    min1: np.ndarray, max1: np.ndarray, min2: np.ndarray, max2: np.ndarray, tol: float
) -> bool:
    """True if two AABBs (min/max) overlap within tol."""
    return not (
        max1[0] + tol < min2[0]
        or max2[0] + tol < min1[0]
        or max1[1] + tol < min2[1]
        or max2[1] + tol < min1[1]
        or max1[2] + tol < min2[2]
        or max2[2] + tol < min1[2]
    )


def compute_convex_union(h1: Manifold, h2: Manifold) -> Tuple[Manifold, float]:
    """Return (hull_mesh, hull_volume) of the convex hull of all vertices."""
    hull = Manifold.hull(h1 + h2)
    return hull, hull.volume()


def make_aabb_pair(bounds: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a sequence of bounds, return the min and max corners of the AABB.
    
    Args:
      bounds : List[float] or np.ndarray
        The bounds can be given in two formats:
        - As a 2D array with shape (2, 3) where the first row is the min corner and the second row is the max corner.
        - As a flat sequence of six floats representing minx, miny, minz, maxx, maxy, maxz.
        
    Returns:
        (min_corner, max_corner) as two length-3 numpy arrays.
    """
    arr = np.array(bounds, dtype=float)
    if arr.ndim == 2 and arr.shape == (2, 3):
        return arr[0], arr[1]
    elif arr.ndim == 1 and arr.size == 6:
        return arr[:3], arr[3:]
    else:
        raise ValueError(f"Unexpected bounds shape: {arr.shape}")


def merge_convex_neighbors(hulls: List[Manifold], tol: float = 1e-2) -> List[Manifold]:
    """
    Merge convex hulls that are neighbors (AABB-touching) into larger convex hulls.

    Args:
        hulls : Hulls to merge (List[Manifold])
        tol   : maximum allowed extra volume to still call the result 'convex'.

    Returns:
        Merged: The merged convex pieces (List[Manifold])
      
    """
    # 1) Initialize nodes with AABBs, volumes, meshes
    nodes = []
    for h in hulls:
        vol = h.volume()
        mn, mx = make_aabb_pair(Manifold.bounding_box(h))  # (2,3)
        nodes.append({"mani": h, "vol": vol, "min": mn, "max": mx})

    num_nodes = len(nodes)
    # 2) Build initial adjacency and heap of (cost, i, j, hull_mesh)
    volume_tol = tol
    heap = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if not is_aabb_touch(
                nodes[i]["min"], nodes[i]["max"], nodes[j]["min"], nodes[j]["max"], tol
            ):
                continue
            hull_mesh, hull_vol = compute_convex_union(
                nodes[i]["mani"], nodes[j]["mani"]
            )
            cost = hull_vol - (nodes[i]["vol"] + nodes[j]["vol"])

            if cost <= volume_tol:
                heapq.heappush(heap, (cost, i, j, hull_mesh, hull_vol))

    alive = [True] * num_nodes
    # 3) Process heap
    while heap:
        cost, i, j, hull_mesh, hull_vol = heapq.heappop(heap)
        if not (alive[i] and alive[j]):
            continue  # one was merged already
        # 4) Merge i,j → new node k
        k = len(nodes)
        minx, maxx = make_aabb_pair(Manifold.bounding_box(hull_mesh))
        nodes.append({"mani": hull_mesh, "vol": hull_vol, "min": minx, "max": maxx})
        alive.append(True)
        alive[i] = alive[j] = False

        # 5) For each surviving node, if AABB-touch new, compute cost and push
        for l, nodel in enumerate(nodes[:-1]):
            if not alive[l]:
                continue
            if not is_aabb_touch(
                nodes[k]["min"], nodes[k]["max"], nodel["min"], nodel["max"], tol
            ):
                continue
            hm, hv = compute_convex_union(nodes[k]["mani"], nodel["mani"])
            c2 = hv - (nodes[k]["vol"] + nodel["vol"])
            if c2 <= volume_tol:
                heapq.heappush(heap, (c2, k, l, hm, hv))

    # 6) Return only the still-alive hulls
    res = [node["mani"] for idx, node in enumerate(nodes) if alive[idx]]
    return res
