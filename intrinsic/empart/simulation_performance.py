# Copyright 2025 Intrinsic Innovation LLC

"""Each simulation consists of instantiating N objects in a uniformly spaced
grid inside a closed box such that they are no contacts initially, and then
applying randomized forces to the objects over a period of T seconds. The
passed meshes are used to construct a set of collision geometries for each
object in the simulation. Both the number of objects N and the time for which
the simulation is run T can be passed as arguments to the public methods.

The returned metrics can be used to evaluate how performant a particular
approximation of the original geometry of the object is in sim in a
task-agnostic manner. (See `decompose.py` for methods to generate an approximate
decomposition of a mesh).

`run_perf_from_args` is the main entry point in the public API.
"""

# ─── Standard library ─────────────────────────────────────────────────────────
import argparse
import math
import os
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional

# ─── Third-party packages ─────────────────────────────────────────────────────
import imageio
import mujoco
import numpy as np
import trimesh


def load_glb_and_export_obj(
    glb_path: Path, out_dir: Path, scale: float = 1.0
) -> List[Tuple[str, str]]:
    """
    Load a GLB as a trimesh.Scene, export *each* node’s mesh (baked with its node‐transform)
    as its own .obj, and return [(mesh_name, obj_filename), …].
    """
    scene = trimesh.load(glb_path, force="scene")
    mesh_entries = []

    if not isinstance(scene, trimesh.Scene):
        # fallback to single‐mesh path
        mesh = scene
        if scale != 1.0:
            mesh.apply_scale(scale)
        mesh_name = glb_path.stem
        obj_filename = f"{mesh_name}.obj"
        mesh.export(out_dir / obj_filename)
        return [(mesh_name, obj_filename)]

    # For each geometry in the scene, grab its node‐transform and bake it
    for node_name, geom in scene.geometry.items():
        # node_name is the key, geom is a Trimesh
        mesh = geom.copy()

        # scene.graph stores transforms; get the 4×4 matrix
        transform, _ = scene.graph.get(node_name)
        mesh.apply_transform(transform)

        if scale != 1.0:
            mesh.apply_scale(scale)

        mesh_name = f"{glb_path.stem}_{node_name}"
        obj_filename = f"{mesh_name}.obj"
        mesh.export(out_dir / obj_filename)
        mesh_entries.append((mesh_name, obj_filename))
    return mesh_entries


def _compute_object_bounds(mins: List[np.ndarray], maxs: List[np.ndarray]):
    """Return (width, depth, height, all_mins, all_maxs)."""
    all_mins = np.vstack(mins).min(axis=0)
    all_maxs = np.vstack(maxs).max(axis=0)
    dims = all_maxs - all_mins
    return (*dims, all_mins, all_maxs)  # width, depth, height, …


def _grid_params(width: float, depth: float, n_inst: int):
    """XY–plane grid size and spacing."""
    grid_n = math.ceil(math.sqrt(n_inst))
    max_dim = max(width, depth)
    margin_xy = 0.15 * max_dim  # 10 % clearance
    spacing_xy = max_dim + margin_xy
    half_int = (grid_n * spacing_xy) / 2.0
    return grid_n, spacing_xy, half_int


def _height_params(height: float, layers: int):
    """Vertical spacing for the bin and object stack."""
    margin_z = 0.07 * height
    wall_height = height * layers + margin_z
    spawn_base = wall_height  # initial lift above walls
    layer_spacing = height + margin_z
    wall_thick = 0.03
    return wall_height, spawn_base, layer_spacing, wall_thick


# --------------------------------------------------------------------------- #
# ──────────────────────────  XML construction helpers  ────────────────────── #
# --------------------------------------------------------------------------- #
def _create_root(model_name: str, meshdir: str) -> ET.Element:
    mj = ET.Element("mujoco", model=model_name)

    visual = ET.SubElement(mj, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")

    ET.SubElement(mj, "compiler", angle="degree", coordinate="local", meshdir=meshdir)
    ET.SubElement(mj, "option", timestep="0.001", gravity="0 0 -3")
    return mj


def _add_assets(parent: ET.Element, mesh_entries: List[Tuple[str, str]]):
    asset = ET.SubElement(parent, "asset")
    for mesh_name, obj_file in mesh_entries:
        ET.SubElement(asset, "mesh", name=mesh_name, file=obj_file)


def _add_ground_and_camera(world: ET.Element):
    ET.SubElement(
        world,
        "camera",
        name="cam_front",
        pos="0.213 -12.195 4.932",
        xyaxes="1.000 0.017 -0.000 -0.002 0.098 0.995",
    )
    ET.SubElement(
        world,
        "geom",
        type="plane",
        size="5 5 0.1",
        rgba="0.9 0.9 0.9 1",
        friction="1 0.00 0.000",
    )


def _build_bin(
    world: ET.Element, half_int: float, wall_height: float, wall_thick: float
):
    bin_body = ET.SubElement(world, "body", name="bin", pos="0 0 0")
    ET.SubElement(bin_body, "joint", type="free")

    # four side walls
    for dx, dy in [(0, half_int), (0, -half_int), (half_int, 0), (-half_int, 0)]:
        size = (
            f"{half_int} {wall_thick} {wall_height}"
            if dx == 0
            else f"{wall_thick} {half_int} {wall_height}"
        )
        ET.SubElement(
            bin_body,
            "geom",
            type="box",
            size=size,
            pos=f"{dx} {dy} {wall_height}",
            rgba="0.7 0.7 0.7 0.1",
            friction="1 0.00 0.000",
        )

    # ceiling (prevents objects escaping upward)
    ET.SubElement(
        bin_body,
        "geom",
        type="box",
        size=f"{half_int} {half_int} {wall_thick}",
        pos=f"0 0 {2*wall_height}",
        rgba="0.7 0.7 0.7 0.1",
        friction="1 0.00 0.000",
    )


def _spawn_objects(
    world: ET.Element,
    mesh_entries: List[Tuple[str, str]],
    grid_n: int,
    spacing_xy: float,
    half_int: float,
    num_layers: int,
    spawn_base: float,
    layer_spacing: float,
):
    idx = 0
    for layer in range(num_layers):
        z = spawn_base + layer * layer_spacing
        for i in range(grid_n):
            for j in range(grid_n):
                x = -half_int + spacing_xy / 2 + j * spacing_xy
                y = -half_int + spacing_xy / 2 + i * spacing_xy

                body = ET.SubElement(
                    world,
                    "body",
                    name=f"object_inst{idx}",
                    pos=f"{x:.3f} {y:.3f} {z:.3f}",
                )
                ET.SubElement(
                    body,
                    "inertial",
                    pos="0 0 0",
                    mass="0.1",
                    diaginertia="0.0001 0.0001 0.0001",
                )
                ET.SubElement(body, "joint", type="free")

                # add every provided mesh as a geom (matches original loop)
                for mesh_name, _ in mesh_entries:
                    ET.SubElement(
                        body,
                        "geom",
                        type="mesh",
                        mesh=mesh_name,
                        condim="3",
                        friction="1 0.00 0.000",
                        rgba="1 1 1 1",
                        contype="1",
                        conaffinity="1",
                        group="1",
                    )
                idx += 1


def _validate_xml(xml_bytes: bytes):
    """Run MuJoCo’s built‑in validation in a temp file (optional)."""
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        tmp.write(xml_bytes)
        tmp.flush()
        mujoco.MjModel.from_xml_path(tmp.name)
    Path(tmp.name).unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# ────────────────────────────────  Public API  ────────────────────────────── #
# --------------------------------------------------------------------------- #
def build_mujoco_xml(
    mesh_entries: List[Tuple[str, str]],
    mins: List[np.ndarray],
    maxs: List[np.ndarray],
    meshdir: str,
    model_name: str,
    num_instances: int = 5,  # kept for backward‑compat (still ignored)
    num_layers: int = 3,
    validate: bool = False,
) -> str:
    """
    Assemble a MuJoCo XML scene file and return it as a UTF‑8 string.

    The behaviour is identical to the original monolithic version; it is now
    divided up for readability and easier unit‑testing.
    """
    # 1) Bounds and dimensions ------------------------------------------------
    obj_w, obj_d, obj_h, *_ = _compute_object_bounds(mins, maxs)

    # 2) XY grid layout -------------------------------------------------------
    grid_n, spacing_xy, half_int = _grid_params(obj_w, obj_d, num_instances)

    # 3) Z‑axis layout --------------------------------------------------------
    wall_h, spawn_base, layer_spacing, wall_thick = _height_params(obj_h, num_layers)

    # 4) Build the XML tree ---------------------------------------------------
    mj = _create_root(model_name, meshdir)
    _add_assets(mj, mesh_entries)

    world = ET.SubElement(mj, "worldbody")
    _add_ground_and_camera(world)
    _build_bin(world, half_int, wall_h, wall_thick)
    _spawn_objects(
        world,
        mesh_entries,
        grid_n,
        spacing_xy,
        half_int,
        num_layers,
        spawn_base,
        layer_spacing,
    )

    # 5) Serialize (+ optional validation) ------------------------------------
    xml_bytes = ET.tostring(mj, encoding="utf-8")
    if validate:
        _validate_xml(xml_bytes)

    return xml_bytes.decode("utf-8")


def write_package(
    glb_paths: List[Path],
    output_dir: Path,
    meshdir: str,
    model_name: str,
    scale: float,
    validate: bool,
) -> Path:
    # Determine where to write OBJ meshes
    meshes_dir = Path(meshdir)
    if not meshes_dir.is_absolute():
        meshes_dir = output_dir / meshdir
    meshes_dir.mkdir(parents=True, exist_ok=True)

    mesh_entries = [
        entry
        for p in glb_paths
        for entry in load_glb_and_export_obj(p, meshes_dir, scale)
    ]

    mins, maxs = [], []
    for _, obj_file in mesh_entries:
        mesh = trimesh.load(Path(meshdir) / obj_file, force="mesh")
        b0, b1 = mesh.bounds  # [[minx,miny,minz],[maxx,maxy,maxz]]
        mins.append(b0)
        maxs.append(b1)

    xml_string = build_mujoco_xml(
        mesh_entries,
        mins=mins,
        maxs=maxs,
        meshdir=meshdir,
        model_name=model_name,
        validate=validate,
    )

    xml_path = output_dir / f"{model_name}.xml"
    xml_path.write_text(xml_string)

    return xml_path


def run(
    xml_source: str,
    out_video: str = "vid.mp4",
    *,
    duration: float = 0.1,
    fps: int = 60,
    render: bool = True,
    from_string: bool = False,
    apply_random_forces: bool = True,
    random_force_mag: float = 1.0,
) -> Tuple[float, float, float, int, int]:
    """
    Simulate a MuJoCo model, applying random forces if requested.

    Args:
        xml_source: XML string or path to the MuJoCo model.
        out_video: Path to save the rendered video.
        duration: Duration of the simulation in seconds.
        fps: Frames per second for the video.
        render: Whether to render the simulation.
        from_string: If True, treat xml_source as a string; otherwise, as a file path.
        apply_random_forces: If True, apply random forces to dynamic objects.
        random_force_mag: Magnitude of the random forces applied to dynamic objects.

    Returns:
        A tuple containing:
        - sim_elapsed: Simulation time elapsed.
        - real_elapsed: Real time elapsed.
        - rt_factor: Real-time factor (simulated time / real time).
        - contact_total: Average number of contacts per step.
        - constraints_total: Average number of constraints per step.
    """
    if from_string is None:
        from_string = Path(xml_source).is_file() is False

    model = (
        mujoco.MjModel.from_xml_string(strip_bad_meshes(xml_source))
        if from_string
        else mujoco.MjModel.from_xml_path(strip_bad_meshes(xml_source))
    )
    data = mujoco.MjData(model)

    # Identify dynamic object bodies once
    if apply_random_forces:
        # model.body_names is a tuple of body name strings
        object_body_ids = []
        bin_id = None
        for bid in range(model.nbody):
            # mj_id2name returns a bytes or str name
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if name is not None and name.startswith("object_inst"):
                object_body_ids.append(bid)

    if render:
        viewer = mujoco.Renderer(model, 480 * 2, 640 * 2, max_geom=300_000)
        viewer.update_scene(data, "cam_front")
        viewer.render()
        writer = imageio.get_writer(out_video, fps=fps, codec="libx264")

    n_steps = int(duration / model.opt.timestep)
    real_start = time.perf_counter()
    sim_start = data.time
    contact_total, constraints_total = 0, 0
    if apply_random_forces:
        for bid in object_body_ids:
            data.xfrc_applied[bid] = np.concatenate(
                (np.random.uniform(-random_force_mag, random_force_mag, 3), np.zeros(3))
            )

    for n_i in range(n_steps):
        mujoco.mj_step(model, data, nstep=8)

        contact_total += data.ncon
        constraints_total += data.nefc
        if render:
            viewer.update_scene(data, "cam_front")
            writer.append_data(viewer.render())
        if apply_random_forces:
            for bid in object_body_ids:
                data.xfrc_applied[bid] = np.concatenate(
                    (
                        np.random.uniform(-random_force_mag, random_force_mag, 3),
                        np.zeros(3),
                    )
                )
    contact_total /= 8 * n_steps
    constraints_total /= 8 * n_steps

    if render:
        writer.close()

    real_elapsed = time.perf_counter() - real_start
    sim_elapsed = data.timer[
        mujoco.mjtTimer.mjTIMER_POS_COLLISION
    ].duration  
    rt_factor = (data.time - sim_start) / real_elapsed

    return sim_elapsed, real_elapsed, rt_factor, contact_total, constraints_total


def benchmark(xml:str, out_video : str, duration : float, fps : int, render: bool, from_string: bool, apply_random_forces: bool, random_force_mag: float, runs: int=3):
    """
    Run the simulation multiple times and average the results.

    Args:
        xml (str): XML string describing the simulation model.
        out_video (str): Path to output video file.
        duration (float): Simulation duration in seconds.
        fps (int): Frames per second for the output video.
        render (bool): Whether to render frames during simulation.
        from_string (bool): Whether the xml should be used or not
        apply_random_forces (bool): Whether to apply random forces.
        random_force_mag (float): Magnitude of random forces to apply.
        runs (int, optional): Number of trials to average over. Defaults to 10.

    Returns:
        dict: Mean values of each metric with keys:
            - 'sim_elapsed' (float): Average simulated time elapsed.
            - 'real_elapsed' (float): Average real wall‑clock time elapsed.
            - 'rt_factor' (float): Average real‑time factor.
            - 'contact_total' (float): Average total contacts.
            - 'constraints_total' (float): Average total constraints.
    """
    # Collect metrics from each run into an (runs × 5) array
    data = np.array([
        run(xml,
            out_video=out_video,
            duration=duration,
            fps=fps,
            render=render,
            from_string=from_string,
            apply_random_forces=apply_random_forces,
            random_force_mag=random_force_mag)
        for _ in range(runs)
    ])
    # Compute column‑wise means and return as a dict
    return tuple(data.mean(axis=0))
# ────────────────────────────────────────────────────────────────────────────
#  Public API: no-filesystem «package + simulate»
# ────────────────────────────────────────────────────────────────────────────
def package_and_simulate(
    glb_paths: List[Path],
    *,
    model_name: str = "generated_model",
    scale: float = 1.0,
    validate: bool = False,
    duration: float = 0.1,
    render: bool = False,
    video_name: str = "vid.mp4",
    fps: int = 60,
    return_xml: bool = False,
    num_instances: int = 5,
    num_layers: int = 3,
    apply_random_forces=True,
    random_force_mag=25,
    save_mesh=None,
):
    """
    Convert GLBs → OBJ meshes → MuJoCo XML (kept in-memory) and simulate.

    Nothing is left on disk unless you set `render=True` – then a video is
    written into the same temporary directory that is deleted when the
    function returns.

    Args:
        glb_paths : List[Path]
            List of GLB file paths to convert and simulate.
        model_name : str
            Name of the MuJoCo model to be generated.
        scale : float
            Scale factor for the meshes.
        validate : bool
            Whether to validate the generated XML model.
        duration : float
            Duration of the simulation in seconds.
        render : bool
            Whether to render the simulation and save a video.
        video_name : str
            Name of the output video file.
        fps : int
            Frames per second for the rendered video.
        return_xml : bool
            If True, return the generated MuJoCo XML string for the mesh object model along with metrics.
        num_instances : int
            Number of object instances to simulate.
        num_layers : int
            Number of layers of objects in the simulation.
        apply_random_forces : bool
            If True, apply random forces to dynamic objects during simulation.
        random_force_mag : float
            Magnitude of the random forces applied to dynamic objects.
        save_mesh: str or None
            If provided, save the combined trimesh object to this directory as an OBJ file.
    Returns
    -------
    metrics : dict
        { sim_elapsed, real_elapsed, rt_factor }
    xml_string : str   (only if return_xml=True)
    """
    # Resolve all GLB paths now (safer before entering tmpdir)
    glb_paths = [Path(p).expanduser().resolve() for p in glb_paths]
    if save_mesh:
        tri = trimesh.util.concatenate([trimesh.load(pa) for pa in glb_paths])
        tri.export(f"{save_mesh}/{model_name}_{len(glb_paths)}.obj")
    with tempfile.TemporaryDirectory(prefix="mujoco_tmp_") as tmp_root:
        tmp_root = Path(tmp_root)
        meshes_dir = tmp_root / "meshes"
        meshes_dir.mkdir(parents=True, exist_ok=True)

        # 1. Convert GLB → per-node OBJ
        mesh_entries = []
        for p in glb_paths:
            mesh_entries.extend(load_glb_and_export_obj(p, meshes_dir, scale))

        mins, maxs = [], []
        for _, obj_file in mesh_entries:
            mesh = trimesh.load(Path(meshes_dir) / obj_file, force="mesh")
            b0, b1 = mesh.bounds  # [[minx,miny,minz],[maxx,maxy,maxz]]
            mins.append(b0)
            maxs.append(b1)
            
        # 2. Build XML *string*
        xml_string = build_mujoco_xml(
            mesh_entries,
            meshdir=str(meshes_dir),
            mins=mins,
            maxs=maxs,
            model_name=model_name,
            num_instances=num_instances,
            num_layers=num_layers,
            validate=validate,
        )

        # 3. Simulate (optionally record video inside tmp_root)
        video_path = str(video_name)
        
        sim_elapsed, real_elapsed, rt_factor, contact_total, constraints_total = benchmark(
            xml_string,
            out_video=video_path,
            duration=duration,
            fps=fps,
            render=render,
            from_string=True,  # xml_string is a string, not a file
            apply_random_forces=apply_random_forces,
            random_force_mag=random_force_mag,
        )

    metrics = {
        "sim_elapsed": sim_elapsed,
        "real_elapsed": real_elapsed,
        "rt_factor": rt_factor,
        "num_contacts": contact_total,
        "num_constraints": constraints_total,
    }
    return (metrics, xml_string) if return_xml else metrics


def strip_bad_meshes(xml_input: str) -> str:
    is_file = os.path.isfile(xml_input)

    if is_file:
        tree = ET.parse(xml_input)
    else:
        tree = ET.ElementTree(ET.fromstring(xml_input))

    root = tree.getroot()
    asset = root.find("asset")
    bad = []

    xml_str = ET.tostring(root, encoding="unicode")

    while True:
        try:
            mujoco.MjModel.from_xml_string(xml_str)
            break  # success
        except ValueError as err:
            if "qhull error" not in str(err):
                raise  # something else is broken

            mesh_name = str(err).split("'")[1]
            bad.append(mesh_name)

            # Remove the mesh from <asset>
            for m in asset.findall("mesh"):
                if m.get("name") == mesh_name:
                    asset.remove(m)

            # Remove any geom referencing the mesh
            for body in root.findall(".//body"):
                for g in list(body.findall("geom")):
                    if g.get("mesh") == mesh_name:
                        body.remove(g)

            xml_str = ET.tostring(root, encoding="unicode")
            tree = ET.ElementTree(ET.fromstring(xml_str))  # reparse for next iteration

    if is_file:
        cleaned_path = xml_input.replace(".xml", "_cleaned.xml")
        with open(cleaned_path, "w") as f:
            f.write(xml_str)
        return cleaned_path
    else:
        return xml_str


def run_helper(
    glbs: List[str],
    out: str = "mujoco_pkg",
    meshdir: Optional[str] = None,
    name: str = "generated_model",
    scale: float = 1.0,
    validate: bool = False,
) -> str:
    """
    Runs a simulation with the given meshes and returns the path to the metrics-JSON that was written.

    Args:
        glbs: List of GLB file paths to convert and simulate.
        out: Output directory for the generated MuJoCo package.
        meshdir: Directory to store the OBJ meshes; defaults to the parent of the first GLB.
        name: Name of the MuJoCo model to be generated.
        scale: Scale factor for the meshes.
        validate: Whether to validate the generated XML model.

    Returns:
        metrics: Dictionary containing simulation metrics.

    """
    out_dir = Path(out).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    glb_paths = [Path(p).expanduser().resolve() for p in glbs]

    # Default meshdir → parent of the first GLB
    meshdir = meshdir or str(glb_paths[0].parent)

    xml_path = write_package(glb_paths, out_dir, meshdir, name, scale, validate)

    sim_elapsed, real_elapsed, rt_factor, contact_total, constraints_total = run(
        str(xml_path),
        "vid.mp4",
        render=False,
        apply_random_forces=True,
        random_force_mag=25,
    )

    metrics = {
        "sim_elapsed": sim_elapsed,
        "real_elapsed": real_elapsed,
        "rt_factor": rt_factor,
        "num_contacts": contact_total,
        "num_constraints": constraints_total,
    }

    return metrics


def run_perf_from_args(args: argparse.Namespace) -> None:
    json_dumped = run_helper(
        glbs=args.glbs,
        out=args.out,
        meshdir=args.meshdir,
        name=args.name,
        scale=args.scale,
        validate=args.validate,
    )
    return json_dumped
