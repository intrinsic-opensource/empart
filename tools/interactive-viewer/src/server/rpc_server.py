# Copyright 2025 Intrinsic Innovation LLC

# ────────────────────────────────────────────────────────────────
# rpc_server.py  (filled arg_builders)
# ────────────────────────────────────────────────────────────────
# Lightweight RPC server that listens on stdin for JSON-RPC messages,
# dispatches geometry processing tasks via `intrinsic.empart` module,
# and returns results over stdout.
#
# Features:
# • Maps method names to CLI-style arguments using `_arg_builders`
# • Handles tasks like voxelization, decimation, decomposition, etc.
# • Emulates `python -m intrinsic.empart ...` execution
# • Captures and returns stdout output as structured JSON
# • Prints command invocations and cwd to stderr for traceability
# 
# Designed for integration with a Node.js RPC client via subprocess streaming.

import os
import pathlib
import sys, json, traceback, runpy
from io import StringIO

project_root = pathlib.Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))
# switch working directory so that imports resolve relative to project root
os.chdir(str(project_root))
# print current working directory to stderr for debugging
sys.stderr.write(f"rpc_server cwd: {os.getcwd()}\n")
# Build argv the same way the old spawn() calls did
_arg_builders = {
    # 1. Voxelize ---------------------------------------------------
    "voxelize": lambda a: (
        ["voxelize", a["file"]]
        + (["--pitch", str(a["pitch"])] if a.get("pitch") is not None else [])
    ),
    # 2. Watertight check ------------------------------------------
    "check_watertight": lambda a: ["check_watertight", a["file"]],
    # 3. Decimate ---------------------------------------------------
    "decimate": lambda a: ["decimate", a["file"], str(a["target_tris"])],
    # 4. Approximation error ---------------------------------------
    "approximation_error": lambda a: (
        ["approximation_error"]
        + (["--display_true_error"] if a.get("display_true_error", True) else [])
        + [
        x
        for r in (a.get("filtered_regions") or [])
        for x in (
            "--region",
            str(r["min"]["x"]), str(r["min"]["y"]), str(r["min"]["z"]),
            str(r["max"]["x"]), str(r["max"]["y"]), str(r["max"]["z"]),
        )
    ]
        + (
            [str(a["min"]), str(a["max"])]
            if a.get("min") is not None and a.get("max") is not None
            else []
        )
        + [a["true_mesh"]]
        + a.get("non_select", [])
        + a.get("select", []) 
        
    ),
    # 5. Decompose --------------------------------------------------
    "decompose": lambda a: (
        ["decompose", a["file"], str(a.get("coarseness", 0))]
        + [
            str(c)
            for box in a.get("bounding_boxes", [])
            for c in (
                box["min"]["x"],
                box["min"]["y"],
                box["min"]["z"],
                box["max"]["x"],
                box["max"]["y"],
                box["max"]["z"],
                box.get("coarseness", 0),
            )
        ] + ["--method", a.get("algorithm", "vhacd")] + ["--resolution", str(50000)]   
          ),
    # 6. Simulation performance ------------------------------------
    "simulation_performance": lambda a: (
        ["simulation_performance"]
        + a["files"]
        + ["-o", a["out_dir"], "-n", a["model_name"]]
        + list(map(str, a.get("extra_args", [])))
    ),
}


def rpc_loop():
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue

        try:
            msg = json.loads(raw)
            method = msg["method"]
            args = msg.get("args", {})
            req_id = msg["id"]

            if method not in _arg_builders:
                raise ValueError(f"Unknown method `{method}`")

            argv = _arg_builders[method](args)
            sys.stderr.write(
                "python -m intrinsic.empart "
                + " ".join(argv)
                + "\n"
            )
            # capture stdout from intrinsic.empart
            old_argv, old_stdout = sys.argv, sys.stdout
            buf = StringIO()
            # emulate `python -m intrinsic.empart ...` by setting argv[0] to module name
            sys.argv = ["intrinsic.empart", *argv]
            sys.stdout = buf

            runpy.run_module("intrinsic.empart", run_name="__main__")

            result = json.loads(buf.getvalue())
            resp = {"id": req_id, "result": result}

        except Exception as e:
            resp = {
                "id": msg.get("id"),
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "trace": traceback.format_exc(),
                },
            }
        finally:
            sys.argv, sys.stdout = old_argv, old_stdout

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    rpc_loop()
