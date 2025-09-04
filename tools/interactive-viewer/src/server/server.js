// Copyright 2025 Intrinsic Innovation LLC

// ────────────────────────────────────────────────────────────────
// server.js
// ────────────────────────────────────────────────────────────────
// Node.js + Express server that exposes a geometry-processing API via REST,
// using a persistent Python RPC client (PyRpcClient) to handle heavy operations.
//
// Key Features:
// • Accepts 3D model data (GLB or OBJ) via HTTP POST requests
// • Delegates tasks like voxelization, decimation, watertightness checks,
//   bounding-box decomposition, and MuJoCo packaging to Python
// • Compresses responses (gzip) and manages temporary files for each request
// • Exposes endpoints like:
//     - /api/voxelize
//     - /api/check-watertight
//     - /api/decimate
//     - /api/run-error
//     - /api/decompose
//     - /api/mujoco-perf
//
// Usage:
//   - Start the server: `node server.js`
//   - Python processes are kept hot via PyRpcClient for performance
//   - On shutdown, Python subprocess is gracefully terminated

const express = require("express");
const cors = require("cors");
const path = require("path");
const fs = require("fs");
const fsP = fs.promises;
const os = require("os");
const zlib = require("zlib");
const getRawBody = require("raw-body");
const { program } = require("commander");

program
  .option(
    "-t, --tmp-base <path>",
    "base directory for temporary working files",
    os.tmpdir(),
  )
  .option("-p, --port <number>", "port to run the server on", 5005)
  .parse(process.argv);

const { tmpBase, port } = program.opts();
const TMP_BASE = path.resolve(tmpBase);

const mkdtemp = (prefix) => fs.mkdtempSync(path.join(TMP_BASE, prefix));
console.log(`Using temporary files in: ${TMP_BASE}`);

const PyRpcClient = require("./py-rpc-client");
const py = new PyRpcClient("python3"); // spawn once, stays hot

const app = express();

/* ─────────────────────────────
 *  Global middleware & static
 * ────────────────────────────*/
app.use(cors());
app.options("*", cors());
app.use(express.json({ limit: "50mb" }));
app.use("/scripts", express.static(path.join(__dirname, "scripts")));

/* ─────────────────────────────
 *  Helpers
 * ────────────────────────────*/
const asyncHandler = (fn) => (req, res, next) =>
  Promise.resolve(fn(req, res, next)).catch(next);

const readBody = async (req, limit = "50mb") => {
  const raw = await getRawBody(req, { limit });
  const buf =
    req.headers["x-content-encoding"] === "gzip" ? zlib.gunzipSync(raw) : raw;
  try {
    return JSON.parse(buf.toString("utf8"));
  } catch {
    throw { status: 400, message: "Invalid JSON" };
  }
};

const tmpModel = (data, fmt = "obj") => {
  const dir = mkdtemp("mdl-");
  const ext = fmt === "glb" ? "glb" : "obj";
  const file = path.join(dir, `input.${ext}`);
  const buf = fmt === "glb" ? Buffer.from(data, "base64") : data;
  fs.writeFileSync(file, buf, fmt === "glb" ? undefined : "utf8");
  return { dir, file };
};

const compress = (res, payload) => {
  zlib.gzip(Buffer.from(JSON.stringify(payload)), (e, gz) => {
    if (e) return res.status(500).json({ error: "Compression failed" });
    res.setHeader("X-Content-Encoding", "gzip");
    res.send(gz);
  });
};

const cleanup = (dir) =>
  dir && fs.rmSync(dir, { recursive: true, force: true });

/* ─────────────────────────────
 *  Routes (now using py.call)
 * ────────────────────────────*/

// 1. Voxelize -------------------------------------------------------------
app.post(
  "/api/voxelize",
  asyncHandler(async (req, res) => {
    const { glb_blob, pitch } = await readBody(req);
    if (!glb_blob?.trim())
      throw { status: 400, message: "`glb_blob` required" };

    const pitchNum = pitch == null ? null : Number(pitch);
    if (pitchNum != null && !Number.isFinite(pitchNum))
      throw { status: 400, message: "`pitch` must be finite" };

    const { dir, file } = tmpModel(glb_blob, "glb");
    try {
      const result = await py.call("voxelize", {
        file,
        ...(pitchNum != null ? { pitch: pitchNum } : {}),
      });
      compress(res, result);
      console.log("Voxelization complete");
    } finally {
      cleanup(dir);
    }
  }),
);

// 2. Watertightness -------------------------------------------------------
app.post(
  "/api/check-watertight",
  asyncHandler(async (req, res) => {
    const { glb_blob, obj_text } = await readBody(req);
    if (!glb_blob && !obj_text)
      throw { status: 400, message: "Supply `glb_blob` or `obj_text`" };
    const isGLB = Boolean(glb_blob?.trim());
    const { dir, file } = tmpModel(
      isGLB ? glb_blob : obj_text,
      isGLB ? "glb" : "obj",
    );
    try {
      const result = await py.call("check_watertight", { file });
      compress(res, result);
      console.log("Watertightness check complete");
    } finally {
      cleanup(dir);
    }
  }),
);

// 3. Decimate -------------------------------------------------------------
app.post(
  "/api/decimate",
  asyncHandler(async (req, res) => {
    const { glb_blob, target_tris } = await readBody(req);
    if (!glb_blob?.trim())
      throw { status: 400, message: "`glb_blob` required" };
    const tris = Number.parseInt(target_tris, 10);
    if (!Number.isFinite(tris) || tris <= 0)
      throw { status: 400, message: "`target_tris` must be positive int" };

    const { dir, file } = tmpModel(glb_blob, "glb");
    try {
      const result = await py.call("decimate", { file, target_tris: tris });
      compress(res, result);
      console.log("GLB decimation complete");
    } finally {
      cleanup(dir);
    }
  }),
);

// 4. Approximation error visualisation -----------------------------------
app.post(
  "/api/run-error",
  asyncHandler(async (req, res) => {
    const body = await readBody(req);
    const {
      approx_non_select = [],
      approx_select = [],
      true_mesh,
      err_range,
      display_true_error,
      filtered_regions,
    } = body;
    if (!true_mesh)
      throw { status: 400, message: "true_mesh must be provided" };
    if (!Array.isArray(approx_non_select) || !Array.isArray(approx_select))
      throw { status: 400, message: "approx arrays must be arrays" };
    if (approx_non_select.length === 0 && approx_select.length === 0)
      throw {
        status: 400,
        message: "At least one approximated mesh is required",
      };

    const tempDir = mkdtemp("err-");
    try {
      const meshPath = (name) => path.join(tempDir, name);
      const writeB64 = (b64, name) =>
        fs.writeFileSync(meshPath(name), Buffer.from(b64, "base64"));

      writeB64(true_mesh, "true.glb");
      approx_non_select.forEach((b, i) => writeB64(b, `non_${i}.glb`));
      approx_select.forEach((b, i) => writeB64(b, `sel_${i}.glb`));

      const result = await py.call("approximation_error", {
        min: err_range?.min,
        max: err_range?.max,
        true_mesh: meshPath("true.glb"),
        non_select: approx_non_select.map((_, i) => meshPath(`non_${i}.glb`)),
        select: approx_select.map((_, i) => meshPath(`sel_${i}.glb`)),
        display_true_error: display_true_error,
        filtered_regions: filtered_regions || [],
      });
      compress(res, { error_meshes: result.error_meshes });
      console.log("Approximation error visualisation complete");
    } finally {
      cleanup(tempDir);
    }
  }),
);

// 5. Bounding‑box decomposition ------------------------------------------
app.post(
  "/api/decompose",
  asyncHandler(async (req, res) => {
    const {
      input_data,
      bounding_boxes = [],
      coarseness = 0,
      algorithm = "vhacd",
    } = await readBody(req);
    if (!input_data?.trim())
      throw { status: 400, message: "`input_data` required" };
    const { dir, file } = tmpModel(input_data, "glb");
    try {
      const result = await py.call("decompose", {
        file,
        coarseness,
        bounding_boxes,
        algorithm,
      });
      compress(res, { meshes: result.meshes, hull_count: result.hulls });
      console.log("Decomposition complete");
    } finally {
      cleanup(dir);
    }
  }),
);

// 6. MuJoCo packaging -----------------------------------------------------
app.post(
  "/api/mujoco-perf",
  asyncHandler(async (req, res) => {
    const {
      glb_blobs,
      modelName = "generated_model",
      extra_args = [],
    } = await readBody(req);
    if (!Array.isArray(glb_blobs) || glb_blobs.length === 0)
      throw { status: 400, message: "glb_blobs must be non-empty array" };

    const tempDir = mkdtemp("glb-pkg-");
    try {
      const outDir = path.join(tempDir, "out");
      fs.mkdirSync(outDir);
      const glbPaths = glb_blobs.map((b64, i) => {
        const p = path.join(tempDir, `mesh_${i}.glb`);
        fs.writeFileSync(p, Buffer.from(b64, "base64"));
        return p;
      });
      const result = await py.call("simulation_performance", {
        files: glbPaths,
        out_dir: outDir,
        model_name: modelName,
        extra_args,
      });
      compress(res, result);
      console.log("Performance check done");
    } finally {
      cleanup(tempDir);
    }
  }),
);

/* ─────────────────────────────
 *  Centralised error handler
 * ────────────────────────────*/
app.use((err, req, res, _next) => {
  // Log full error for debugging
  console.error("Error encountered:", err);
  const { status = 500, message = "Internal server error" } = err || {};
  // Include Python traceback if available
  const stack = err.trace || err.stack || null;
  res.status(status).json({ error: { message, stack } });
});

/* ─────────────────────────────
 *  Start server & tidy shutdown
 * ────────────────────────────*/
const PORT = Number(port) || 5005;
app.listen(PORT, () => console.log(`Server listening on ${PORT}`));

process.on("SIGINT", async () => {
  console.log("Shutting down…");
  await py.end();
  process.exit();
});
