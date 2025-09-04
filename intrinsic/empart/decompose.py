# Copyright 2025 Intrinsic Innovation LLC

"""
Approximately decompose the input mesh into a set of convex hulls. A list of bounding boxes (in mesh
coordinates) can be specified to tune the approximation coarseness in different regions of mesh
independently.

The main entry‑point methods are `run_decompose_from_args` and `decompose_mesh`.
Note that the algorithms in this module can handle both watertight and non‑watertight meshes, although
performance is significantly better with watertight meshes.
See `approximation_error.py` for methods to compute the approximation error.
Also see `simulation_performance.py` for methods to compute performance metrics for the output geometry
set.

──────────────────────────────────────────────────────────────────────────────────────────────────────
Algorithm overview
──────────────────────────────────────────────────────────────────────────────────────────────────────
1. **Load & Pre‑process**
   - Load the *.glb* file via **trimesh** and record whether the mesh is watertight.
   - Convert vertices/faces to NumPy for fast downstream operations.

2. **Parse Region Parameters**
   - Convert every bounding‑box specification (either a 7‑float tuple or a `RegionParams` instance)
     into a canonical `dict` containing
       `{min:{x,y,z}, max:{x,y,z}, maxConvexHulls}`.

3. **Carve‑out Select Regions**
   - Build Manifold or Trimesh cuboids from each region and perform a boolean *difference* with the
     original mesh.  The carved‑out parts ("select regions") are processed independently in step 5.

4. **Global (Non‑Select) Decomposition**
   - If `non_select_convex_hull_max > 0` run **VHACD** or **CoACD** once on the remainder to obtain up
     to that many convex hulls.  Otherwise keep the raw remainder mesh.
   - When the remainder is split into multiple hulls, merge touching neighbours to reduce count.

5. **Per‑Box Processing** (runs in parallel when using VHACD)
   For each region box:
   a. Intersect the original mesh with the cuboid (fast Manifold batch boolean when watertight).
   b. If `maxConvexHulls > 0` decompose the intersection with the chosen method; otherwise keep it
      as a single raw mesh chunk.
   c. Color each output hull for easy visual debugging and package metadata (`origin`, `box_index`,
      `volume`, etc.).

6. **Post‑processing & Merge**
   - Optionally split and merge hulls across region boundaries to avoid duplicates and ensure strict
     convexity (uses face‑plane splitting and adjacency merging heuristics).

7. **Assemble Final Result**
   - Return a dictionary:
       {
         "hulls": <total convex hull count>,
         "meshes": {
             "non_select_obj": [ <global meshes> ],
             "select_objs":    { <box‑idx>: [ <per‑box meshes> ], ... }
         }
       }
   - Each mesh entry can contain either an in‑memory `trimesh.Trimesh` (when `as_meshes=True`) or a
     base‑64 encoded GLB blob.

8. **Performance Notes**
   - Manifold boolean operations are ~5× faster than Trimesh‑Blender for watertight meshes,
     hence they are used whenever possible.
   - The per‑box stage is parallel and is executed via a `ProcessPoolExecutor` when
     VHACD is selected; CoACD performs its own internal parallelism so we keep it in‑process.

"""

import argparse
import random
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple
import dataclasses

import numpy as np
import trimesh
from trimesh.interfaces.blender import boolean as blender_boolean
from trimesh.visual import ColorVisuals

import coacd

coacd.set_log_level("error")
from manifold3d import Manifold, Mesh, OpType

from .utils import (
    bounding_box_planes,
    merge_convex_neighbors,
    encode_mesh_bytes,
)


_INPUT_MANIFOLD = None


@dataclasses.dataclass
class RegionParams:
    """Parameters for a particular region of the mesh, specified as a bounding box"""

    xmin: float
    ymin: float
    zmin: float
    xmax: float
    ymax: float
    zmax: float
    maxConvexHulls: float


def _random_rgba() -> List[int]:
    """Return a random, fully‑opaque RGBA colour as a list of four ints [0‑255]."""
    return [random.randint(0, 255) for _ in range(3)] + [255]


def _mesh_to_b64(mesh: trimesh.Trimesh) -> str:
    glb_bytes = mesh.export(file_type="glb")  # always in-memory
    return encode_mesh_bytes(glb_bytes)


def _init_worker(verts: np.ndarray, faces: np.ndarray, watertight: bool):
    """Initialise a worker by stashing the *input* mesh (as a Manifold **or** Trimesh) in a global variable.
    This is called once per process in the ProcessPoolExecutor so each worker re‑uses the same copy.
    """
    global _INPUT_MANIFOLD
    if watertight:
        _INPUT_MANIFOLD = Manifold(
            Mesh(verts.astype(np.float32), faces.astype(np.uint32))
        )
    else:
        _INPUT_MANIFOLD = trimesh.Trimesh(verts, faces, process=False)


def _create_cuboid_manifold(xmin, ymin, zmin, xmax, ymax, zmax):
    w, h, d = xmax - xmin, ymax - ymin, zmax - zmin
    return Manifold.cube((w, h, d)).translate((xmin, ymin, zmin))


def _create_cuboid_trimesh(xmin, ymin, zmin, xmax, ymax, zmax):
    box = trimesh.creation.box(extents=(xmax - xmin, ymax - ymin, zmax - zmin))
    box.apply_translation([(xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2])
    return box


def _vhacd_params(maxConvexHulls: float, resolution=50000) -> Dict[str, Any]:
    return {
        "maxConvexHulls": int(maxConvexHulls),
        "resolution": resolution,
        "minimumVolumePercentErrorAllowed": 0.001,
        "maxRecursionDepth": 40,
        "maxNumVerticesPerCH": 128,
        "minEdgeLength": 0,
        "findBestPlane": True,
        "shrinkWrap": False,
    }


def _fully_inside_cube(mani: Manifold, cube: Manifold, rel_tol=1e-6, abs_tol=1e-12):
    if mani.volume() - cube.volume() > abs_tol:
        return False
    outside = mani - cube
    return outside.volume() < max(abs_tol, rel_tol * cube.volume())

def _process_box(
    box_index: int,
    box: Dict[str, Any],
    watertight: bool,
    decomp_method: str,
    *,
    as_meshes: bool,
    include_raw: bool = False,
    resolution: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Boolean‑intersect the global mesh with the specified `box`, optionally convex‑decompose it, and return a
    list of JSON‑structured dictionaries describing the result. When `as_meshes` is *True* the 'mesh' field holds
    the in‑memory `trimesh.Trimesh`; otherwise it contains a base‑64 GLB export (original behaviour).
    """

    im = _INPUT_MANIFOLD  # aliased purely for brevity

    if watertight:
        cub = _create_cuboid_manifold(
            box["min"]["x"],
            box["min"]["y"],
            box["min"]["z"],
            box["max"]["x"],
            box["max"]["y"],
            box["max"]["z"],
        )
        inter = Manifold.batch_boolean([cub, im], OpType.Intersect)
        verts = np.asarray(inter.to_mesh().vert_properties, dtype=np.float32)
        faces = np.asarray(inter.to_mesh().tri_verts, dtype=np.int32)
    else:
        cub = _create_cuboid_trimesh(
            box["min"]["x"],
            box["min"]["y"],
            box["min"]["z"],
            box["max"]["x"],
            box["max"]["y"],
            box["max"]["z"],
        )
        inter = blender_boolean(
            meshes=[cub, im],
            operation="intersection",
            use_exact=True,
            use_self=True,
            debug=False,
        )

        verts = inter.vertices
        faces = np.asarray(inter.faces, dtype=np.int32)

    maxConvexHulls = box["maxConvexHulls"]
    glb_list = []
    if include_raw:
        raw_mesh = trimesh.Trimesh(verts, faces.astype(np.int64), process=False)
        glb_list.append(
            {
                "raw_mesh": raw_mesh if as_meshes else _mesh_to_b64(raw_mesh),
                "convex": False,
                "origin": "per-box",
                "box_index": box_index,
                "volume": float(raw_mesh.volume),
            }
        )

    if (
        maxConvexHulls > 0
    ):  # ------------------------ convex‑decompose the per‑box chunk ------------------------

        outputs = _run_global_decomposition(
            trimesh.Trimesh(verts, faces), decomp_method, maxConvexHulls, resolution
        )

        for m in outputs:
            mesh = trimesh.Trimesh(m["vertices"], m["faces"], process=False)
            if mesh.visual is None:  # ensure non-None
                mesh.visual = ColorVisuals(mesh)
            mesh.visual.face_colors = np.tile(_random_rgba(), (len(mesh.faces), 1))
            glb_list.append(
                {
                    "mesh": mesh if as_meshes else _mesh_to_b64(mesh),
                    "convex": True,
                    "origin": "per-box",
                    "box_index": box_index,
                    "volume": float(mesh.volume),
                }
            )

        hulls = len(outputs)
    else:  # ------------------------------ maxConvexHulls == 0 → keep the raw intersection ------------------------------
        mesh = trimesh.Trimesh(verts, faces.astype(np.int64), process=False)
        glb_list.append(
            {
                "mesh": mesh if as_meshes else _mesh_to_b64(mesh),
                "convex": False,
                "origin": "per-box",
                "box_index": box_index,
                "volume": float(mesh.volume),
            }
        )
        hulls = 0

    return glb_list, hulls


def run_decompose_from_args(
    args: argparse.Namespace, *, as_meshes: bool = True
) -> Dict[str, Any]:
    """Run the mesh boolean operations and convex decomposition pipeline using command‑line arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
            - input_glb: Path to the input *.glb* file.
            - non_select_convex_hull_max: maxConvexHulls for the global convex decomposition (0 disables it).
            - method: Decomposition method, either 'vhacd' or 'coacd'.
            - boxes: List of bounding boxes specified as a list of floats (xmin, ymin, zmin, xmax, ymax, zmax, maxConvexHulls).
        as_meshes (bool):
            If True, the 'mesh' fields contain `trimesh.Trimesh` instances.
            If False, the 'mesh' fields contain base‑64 encoded GLB bytes
    Returns:
        Dict[str, Any]: A dictionary containing the number of hulls and the meshes [{"mesh": ..., "convex": ...}, ...].

    Example
    -------
    >>> result = run_decompose_from_args([
    ...     "input.glb", "0.5", "--method", "coacd",  # global args
    ...     "0", "0", "0", "1", "1", "1", "0.25",          # box #0
    ...     "1", "0", "0", "2", "1", "1", "0.0"            # box #1
    ... ])
    """
    return _run_pipeline(args, as_meshes=as_meshes)


def decompose_mesh(
    input_glb: str,
    non_select_convex_hull_max: float,
    *,
    boxes: List[RegionParams] = [],
    method: str = "vhacd",
    as_meshes: bool = True,
    include_raw: bool = False,
    resolution: int = 50000,
) -> Dict[str, Any]:
    """Convenience wrapper around :func:`run_decompose_from_args` that uses **typed** parameters instead of raw strings.

    Ags:
        input_glb:
            Path to the input *.glb*.
        non_select_convex_hull_max:
            maxConvexHulls for the *global* convex decomposition (``0`` disables it).
        boxes:
            Optional list of RegionParams to specify desired per-region parameters.
        method:
            ``'vhacd'`` (default) or ``'coacd'``.
        as_meshes:
            ``True`` → the 'mesh' fields contain **`trimesh.Trimesh`** instances.
            ``False`` → the 'mesh' fields contain base‑64 encoded GLB bytes (original behaviour).
        include_raw:
            If True, include the raw intersection mesh in the output.
        resolution:
            Resolution for the VHACD decomposition (default: 50000).

    Returns:
        Dict[str, Any]: A dictionary containing the number of hulls and the meshes [{"mesh": ..., "convex": ...}, ...].
    """
    args = argparse.Namespace(
        input_glb=input_glb,
        non_select_convex_hull_max=non_select_convex_hull_max,
        boxes=[str(v) for b in boxes for v in b],
        method=method,
        include_raw=include_raw,
        resolution=resolution,
    )
    return run_decompose_from_args(args, as_meshes=as_meshes)


# --------------------------------------------------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------------------------------------------------


def _parse_boxes(raw_boxes: List[float]) -> List[Dict[str, Any]]:
    """Convert flat argv list → list of typed dicts (xmin … maxConvexHulls)."""
    if len(raw_boxes) % 7 != 0:
        raise ValueError(
            "Each bounding‑box requires exactly 7 numbers (xmin ymin zmin xmax ymax zmax maxConvexHulls)"
        )

    return [
        {
            "min": {"x": float(xmin), "y": float(ymin), "z": float(zmin)},
            "max": {"x": float(xmax), "y": float(ymax), "z": float(zmax)},
            "maxConvexHulls": int(float(maxConvexHulls)),
        }
        for xmin, ymin, zmin, xmax, ymax, zmax, maxConvexHulls in zip(
            *[iter(raw_boxes)] * 7
        )
    ]


def _create_cuboids(
    boxes: List[Dict[str, Any]], *, watertight: bool
) -> Tuple[List["Manifold"], List["trimesh.Trimesh"]]:
    """Return both Manifold and trimesh cuboids, depending on watertightness."""
    manifolds = [
        _create_cuboid_manifold(
            b["min"]["x"],
            b["min"]["y"],
            b["min"]["z"],
            b["max"]["x"],
            b["max"]["y"],
            b["max"]["z"],
        )
        for b in boxes
    ]

    if watertight:
        return manifolds, []

    meshes = [
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
    return manifolds, meshes


def _subtract_cuboids(
    input_mani: "Manifold",
    cuboids: List["Manifold"],
    input_tm: trimesh.Trimesh,
    cuboid_meshes: List[trimesh.Trimesh],
) -> trimesh.Trimesh:
    """Boolean‑subtract *cuboids* from the original mesh and return the remainder."""
    if not cuboids:
        return input_tm
    # If the input mesh is watertight, use Manifold batch boolean
    # (which is ~5x faster than trimesh's blender_boolean).
    if input_tm.is_watertight:
        non_select = Manifold.batch_boolean([input_mani, *cuboids], OpType.Subtract)
        return trimesh.Trimesh(
            np.asarray(non_select.to_mesh().vert_properties, dtype=np.float32),
            np.asarray(non_select.to_mesh().tri_verts, dtype=np.int64),
            process=False,
        )
    else:
        return blender_boolean(
            meshes=[input_tm, *cuboid_meshes],
            operation="difference",
            use_exact=False,
            use_self=False,
            debug=False,
        )


def _run_global_decomposition(
    non_select_mesh: trimesh.Trimesh, method: str, maxConvexHulls: int, resolution: int
) -> List[Dict[str, Any]]:
    """Run VHACD / CoACD

    Args:
        non_select_mesh (trimesh.Trimesh): The mesh to decompose.
        method (str): The decomposition method, either 'vhacd' or 'coacd'.
        maxConvexHulls (int): The max number of convex hulls for decomposition.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the decomposed meshes.
    """
    if method == "vhacd":
        outputs = trimesh.decomposition.convex_decomposition(
            non_select_mesh, **_vhacd_params(maxConvexHulls, resolution)
        )

        # Adapt to the common {vertices, faces} dict format
        return [
            {"vertices": tm["vertices"], "faces": tm["faces"]} for tm in outputs  # type: ignore
        ]

    # CoACD path
    cm = coacd.Mesh(non_select_mesh.vertices, non_select_mesh.faces)

    parts = coacd.run_coacd(
        cm,
        max_convex_hull=maxConvexHulls,
        threshold=0.01,
        mcts_nodes=100,
        mcts_max_depth=2,
        merge=True,
        decimate=True,
        max_ch_vertex=128
    )
    results = []
    for p in parts:
        mesh_part = trimesh.Trimesh(p[0], p[1], process=False)
        results.append(
            {
                "vertices": mesh_part.vertices,
                "faces": mesh_part.faces,
            }
        )
    return results

def _assemble_final_groups(
    global_outputs: List[Dict[str, Any]],
    cuboids: List["Manifold"],
    boxes: List[Dict[str, Any]],
    *,
    as_meshes: bool,
) -> Tuple[List[Dict[str, Any]], int]:
    """Build final *non‑select* groups (with carve‑out + merge when boxes exist)."""
    final_convex_groups: List[List["Manifold"]] = []

    if not cuboids:
        # Easy path – every hull is kept as‑is
        for op in global_outputs:
            pts, fcs = op["vertices"], op["faces"]
            final_convex_groups.append(
                [Manifold(Mesh(pts.astype(np.float32), fcs.astype(np.uint32)))]
            )
    else:
        for i,op in enumerate(global_outputs):
            pts, fcs = op["vertices"], op["faces"]
            h_mani = Manifold(Mesh(pts.astype(np.float32), fcs.astype(np.uint32)))            
            
            tol = h_mani.volume() * 1e-10
            work = [h_mani]

            # Rank cuboids by intersection volume
            cube_score = [
                (
                    Manifold.batch_boolean([h_mani, cube], OpType.Intersect).volume(),
                    idx,
                )
                for idx, cube in enumerate(cuboids)
            ]
            cuboids_sorted = [
                (cuboids[i], i) for _, i in sorted(cube_score, reverse=True)
            ]

            # Split along each cuboid's bounding planes
            for cube, cube_idx in cuboids_sorted:
                for normal, offset in bounding_box_planes(boxes[cube_idx]):                
                    nxt: List["Manifold"] = []
                    for piece in work:
                        if abs((piece - cube).volume() - piece.volume()) == 0:
                            nxt.append(piece)
                            continue
                        neg, pos = piece.split_by_plane(normal, float(offset))

                        if neg.volume() > tol:
                            nxt.append(neg)
                        if pos.volume() > tol:
                            nxt.append(pos)
                    work = nxt
                work = [f for f in work if not _fully_inside_cube(f, cube)]

            work = merge_convex_neighbors(work, tol=0)
            final_convex_groups.append(work)

        # Merge across groups once more
        # TODO: This extra merge causes constraint violations in some cases, so disable for now
        # merged = merge_convex_neighbors(
        #     [h for grp in final_convex_groups for h in grp], tol=1e-8
        # )
        # final_convex_groups = [merged]

    # ───────────────────────────  To trimesh + colour  ───────────────────────
    glb_groups: List[Dict[str, Any]] = []
    for grp in final_convex_groups:
        for m in grp:
            tm_conv = trimesh.Trimesh(
                np.asarray(m.to_mesh().vert_properties, dtype=np.float32),
                np.asarray(m.to_mesh().tri_verts, dtype=np.int64),
                process=False,
            )
            if tm_conv.visual is None:
                tm_conv.visual = ColorVisuals(tm_conv)
            tm_conv.visual.face_colors = np.tile(
                _random_rgba(), (len(tm_conv.faces), 1)
            )
            glb_groups.append(
                {
                    "mesh": tm_conv if as_meshes else _mesh_to_b64(tm_conv),
                    "convex": True,
                    "origin": "global",
                    "box_index": None,
                    "volume": tm_conv.volume,
                }
            )

    return glb_groups, len(glb_groups)


def _wrap_non_select(mesh: trimesh.Trimesh, *, as_meshes: bool) -> List[Dict[str, Any]]:
    """Return the *non‑select* mesh wrapped in the usual metadata structure."""
    grey = [200, 200, 200, 255]
    if mesh.visual is None:
        mesh.visual = ColorVisuals(mesh)
    mesh.visual.vertex_colors = np.tile(grey, (len(mesh.vertices), 1))
    return [
        {
            "mesh": mesh if as_meshes else _mesh_to_b64(mesh),
            "convex": False,
            "origin": "global",
            "box_index": None,
            "volume": float(mesh.volume),
        }
    ]


def _decompose_boxes(
    args: argparse.Namespace,
    boxes: List[Dict[str, Any]],
    verts_full: np.ndarray,
    faces_full: np.ndarray,
    watertight: bool,
    *,
    as_meshes: bool,
    method: str,
    resolution: int,
) -> Tuple[Dict[str, Any], int]:
    """Run the *per‑box* decomposition stage (possibly in parallel)."""
    select_objs: Dict[str, Any] = {}
    total_hulls = 0

    if not boxes:
        return select_objs, 0

    if method == "vhacd":
        with ProcessPoolExecutor(
            initializer=_init_worker, initargs=(verts_full, faces_full, watertight)
        ) as exe:
            futs = {
                exe.submit(
                    _process_box,
                    idx,
                    b,
                    watertight,
                    method,
                    as_meshes=as_meshes,
                    include_raw=args.include_raw,
                    resolution=resolution,
                ): idx
                for idx, b in enumerate(boxes)
            }
            for fut in as_completed(futs):
                idx = futs[fut]
                objs, nhulls = fut.result()
                select_objs[str(idx)] = objs
                total_hulls += nhulls
    else:
        _init_worker(verts_full, faces_full, watertight)
        for idx, b in enumerate(boxes):
            objs, nhulls = _process_box(
                idx,
                b,
                watertight,
                method,
                as_meshes=as_meshes,
                include_raw=args.include_raw,
                resolution=resolution,
            )
            select_objs[str(idx)] = objs
            total_hulls += nhulls

    return select_objs, total_hulls


def _run_pipeline(args: argparse.Namespace, *, as_meshes: bool) -> Dict[str, Any]:
    """Entry-point performing convex decomposition with optional boxed carve-outs."""
    decomp_method = args.method
    # ───────────────────────────  Load + preprocess  ──────────────────────────
    tm = trimesh.load(args.input_glb, force="mesh")

    watertight = tm.is_watertight
    verts_full = tm.vertices.astype(np.float32)
    faces_full = tm.faces.astype(np.uint32)
    input_mani = Manifold(Mesh(verts_full, faces_full))
    boxes = _parse_boxes(args.boxes)

    cuboids, cuboids_trimesh = _create_cuboids(boxes, watertight=watertight)

    non_select_mesh = _subtract_cuboids(input_mani, cuboids, tm, cuboids_trimesh)

    # ─────────────────────────────  Global stage  ─────────────────────────────
    if args.non_select_convex_hull_max > 0:
        with tempfile.TemporaryDirectory():
            global_outputs = _run_global_decomposition(
                non_select_mesh,
                decomp_method,
                args.non_select_convex_hull_max,
                args.resolution,
            )
        glb_groups, glb_hulls = _assemble_final_groups(
            global_outputs, cuboids, boxes, as_meshes=as_meshes
        )
    else:
        glb_groups = _wrap_non_select(non_select_mesh, as_meshes=as_meshes)
        glb_hulls = 0
    # ─────────────────────────────  Per‑box stage  ────────────────────────────
    select_objs, box_hulls = _decompose_boxes(
        args,
        boxes,
        verts_full,
        faces_full,
        watertight,
        as_meshes=as_meshes,
        method=decomp_method,
        resolution=args.resolution,
    )

    # ─────────────────────────────  Assemble result  ──────────────────────────
    return {
        "hulls": glb_hulls + box_hulls,
        "meshes": {
            "non_select_obj": glb_groups,
            "select_objs": select_objs,
        },
    }
