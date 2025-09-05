// Copyright 2025 Intrinsic Innovation LLC

// ---------------------------------------------------------------------------
// hooks/useBackend.js
// ---------------------------------------------------------------------------
// Centralised backend I/O.
// Exposes three async callbacks:
//   - decompose()        → POST /api/decompose
//   - runError()         → POST /api/run-error
//   - perfCheck()        → POST /api/mujoco-perf (MuJoCo rtf check)
// -----------------------------------------------------------------------------

import { useCallback } from "react";
import pako from "pako";
import { blobUrlToBase64, decodeBlobUrl } from "../utils";

/* ------------------------------------------------------------ helpers */

const API_HOST = process.env.REACT_APP_API_HOST;
const API_PORT = process.env.REACT_APP_API_PORT;
const API_BASE = `${API_HOST}:${API_PORT}`;

/** Build an AABB region object the back‑end expects */
function boxesToAABBs(indices, boxes) {
  return indices.map((i) => {
    const b = boxes[i];
    return {
      min: {
        x: +(b.center.x - b.size.x / 2).toFixed(6),
        y: +(b.center.y - b.size.y / 2).toFixed(6),
        z: +(b.center.z - b.size.z / 2).toFixed(6),
      },
      max: {
        x: +(b.center.x + b.size.x / 2).toFixed(6),
        y: +(b.center.y + b.size.y / 2).toFixed(6),
        z: +(b.center.z + b.size.z / 2).toFixed(6),
      },
    };
  });
}

/* ------------------------------------------------------------ hook */

export default function useBackend({
  /* —— read‑only inputs —— */
  meshes,
  boundingBoxes,
  originalUrl,
  errorBoxFilter,
  boxToHullMap,
  defaultCoarseness,
  algorithm,
  errorRange, // { min, max }
  /* —— setters (write‑backs) —— */
  setMeshes,
  setHullVolumes,
  setStatsApprox,
  setBoxToHullMap,
  setMeshLabels,
  setErrsForward,
  setErrsReverse,
  setShowErrorForward,
  setShowErrorReverse,
  setTimings,
  setIsLoading,
  setShowApproximation,
  setShowOriginal,
  /* optional helper from caller */
  countMeshStatsForUrls,
}) {
  /* -------------------------------------------------------- /decompose */

  const decompose = useCallback(async () => {
    if (!originalUrl) return;
    setIsLoading(true);
    const t0 = performance.now();

    try {
      /* 1️⃣  build request */
      const bbPayload = boundingBoxes.map((b) => ({
        min: {
          x: (b.center.x - b.size.x / 2).toFixed(3),
          y: (b.center.y - b.size.y / 2).toFixed(3),
          z: (b.center.z - b.size.z / 2).toFixed(3),
        },
        max: {
          x: (b.center.x + b.size.x / 2).toFixed(3),
          y: (b.center.y + b.size.y / 2).toFixed(3),
          z: (b.center.z + b.size.z / 2).toFixed(3),
        },
        coarseness: b.coarseness,
      }));

      const glbB64 = await blobUrlToBase64(originalUrl);
      const res = await fetch(`${API_BASE}/api/decompose`, {
        method: "POST",
        mode: "cors",
        headers: { "X-Content-Encoding": "gzip" },
        body: new Blob(
          [
            pako.gzip(
              JSON.stringify({
                input_data: glbB64,
                format: "glb",
                bounding_boxes: bbPayload,
                coarseness: defaultCoarseness,
                algorithm: algorithm,
              }),
            ),
          ],
          { type: "application/gzip" },
        ),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      /* unpack */
      const buf = await res.arrayBuffer();
      const { meshes: payload } = JSON.parse(
        pako.ungzip(new Uint8Array(buf), { to: "string" }),
      );
      const { non_select_obj: nonSelectGroups, select_objs: selectGroups } =
        payload;

      /* ---------- non‑selected ---------- */
      const nonSelectUrl = [];
      const nonSelectVolumes = [];

      for (const { mesh, volume } of nonSelectGroups) {
        nonSelectUrl.push(decodeBlobUrl(mesh, "model/gltf-binary"));
        nonSelectVolumes.push(volume);
      }

      const box_indices = Object.values(nonSelectGroups)
        .flat()
        .map(({ box_index }) => box_index);
      const convex_labels = Object.values(nonSelectGroups)
        .flat()
        .map(({ convex }) => convex);

      /* ---------- selected ---------- */
      const nested = Object.values(selectGroups);
      const flatList = nested.flat(); // exactly one extra array layer
      const selectUrls = [];
      const selectVolumes = [];

      for (const { mesh, volume } of flatList) {
        selectUrls.push(decodeBlobUrl(mesh, "model/gltf-binary"));
        selectVolumes.push(volume);
      }

      const select_convex_labels = flatList.map(({ convex }) => convex);
      const select_box_indices = flatList.map(({ box_index }) => box_index);

      /* ---------- write results to state ---------- */
      setMeshes({ nonSelect: nonSelectUrl, select: selectUrls });
      setHullVolumes({ nonSelect: nonSelectVolumes, select: selectVolumes });
      setMeshLabels({
        nonSelectConvex: convex_labels,
        nonSelectBox: box_indices,
        selectConvex: select_convex_labels,
        selectBox: select_box_indices,
      });

      /* ---------- box → hull map (unchanged) ---------- */
      const newMap = { nonSelect: {}, select: {} };

      box_indices.forEach((boxIdx, hullIdx) => {
        if (!newMap.nonSelect[boxIdx]) newMap.nonSelect[boxIdx] = [];
        newMap.nonSelect[boxIdx].push(hullIdx);
      });
      select_box_indices.forEach((boxIdx, hullIdx) => {
        if (!newMap.select[boxIdx]) newMap.select[boxIdx] = [];
        newMap.select[boxIdx].push(hullIdx);
      });
      setBoxToHullMap(newMap);

      /* ---------- optional stats ---------- */
      if (countMeshStatsForUrls) {
        const { vertices, faces } = await countMeshStatsForUrls([
          ...nonSelectUrl,
          ...selectUrls,
        ]);
        setStatsApprox((s) => ({
          ...s,
          vertices,
          faces,
          hulls: nonSelectUrl.length + selectUrls.length,
        }));
      }
      /* UI state */
      setShowApproximation(true);
      setShowOriginal(false);
      setShowErrorForward(false);
      setShowErrorReverse(false);
      setTimings((t) => ({
        ...t,
        process: +(performance.now() - t0).toFixed(0),
      }));
    } catch (err) {
      console.error("[useBackend] decompose error:", err);
    } finally {
      setIsLoading(false);
    }
  }, [
    algorithm,
    boundingBoxes,
    defaultCoarseness,
    originalUrl,
    /* setters */
    setBoxToHullMap,
    setHullVolumes,
    setIsLoading,
    setMeshes,
    setMeshLabels,
    setStatsApprox,
    setTimings,
    setShowApproximation,
    setShowOriginal,
    countMeshStatsForUrls,
  ]);

  /* ---------------------------------------------------------- /runError */

  const runError = useCallback(
    async (displayTrueError = false, onlySelectedBoxes = true) => {
      if (!meshes.nonSelect.length) return;
      setIsLoading(true);
      const t0 = performance.now();
      try {
        /* which hull indices to send */
        const nsIdx = [];
        const selIdx = [];
        if (onlySelectedBoxes && errorBoxFilter.size) {
          errorBoxFilter.forEach((boxIdx) => {
            (boxToHullMap.nonSelect[boxIdx] || []).forEach((i) =>
              nsIdx.push(i),
            );
            (boxToHullMap.select[boxIdx] || []).forEach((i) => selIdx.push(i));
          });
        } else {
          nsIdx.push(...meshes.nonSelect.keys());
          selIdx.push(...meshes.select.keys());
        }

        const approx_non_select = await Promise.all(
          nsIdx.map((i) => blobUrlToBase64(meshes.nonSelect[i])),
        );
        const approx_select = await Promise.all(
          selIdx.map((i) => blobUrlToBase64(meshes.select[i])),
        );
        const true_mesh = await blobUrlToBase64(originalUrl);

        const filtered_regions =
          onlySelectedBoxes && errorBoxFilter.size
            ? boxesToAABBs([...errorBoxFilter], boundingBoxes)
            : [];

        const raw = JSON.stringify({
          approx_non_select,
          approx_select,
          true_mesh,
          err_range: errorRange,
          display_true_error: !!displayTrueError,
          filtered_regions,
        });

        const res = await fetch(`${API_BASE}/api/run-error`, {
          method: "POST",
          mode: "cors",
          headers: { "X-Content-Encoding": "gzip" },
          body: new Blob([pako.gzip(raw)], { type: "application/gzip" }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const buf = await res.arrayBuffer();
        const { error_meshes } = JSON.parse(
          pako.ungzip(new Uint8Array(buf), { to: "string" }),
        );

        const [nonSelectErr, selectErr] = error_meshes.map((b64) =>
          decodeBlobUrl(b64, "application/octet-stream"),
        );
        const errSet = {
          nonSelectErr: nonSelectErr || null,
          selectErrs: selectErr ? [selectErr] : [],
        };

        if (displayTrueError) {
          setErrsForward(errSet);
          setShowErrorForward(true);
          setShowErrorReverse(false);
          setShowOriginal(false);
          // setShowApproximation(false)
        } else {
          setErrsReverse(errSet);
          setShowErrorReverse(true);
          setShowErrorForward(false);
          setShowOriginal(false);
          // setShowApproximation(false)
        }

        setTimings((t) => ({
          ...t,
          runError: +(performance.now() - t0).toFixed(0),
        }));
        if (window.location.search.includes("e2e")) {
          console.e2e(`[E2E] Process Error completed`);
        }
      } catch (e) {
        console.error("[useBackend] runError error:", e);
      } finally {
        setIsLoading(false);
      }
    },
    [
      meshes,
      boundingBoxes,
      originalUrl,
      errorRange,
      errorBoxFilter,
      boxToHullMap,
      setErrsForward,
      setErrsReverse,
      setShowErrorForward,
      setShowErrorReverse,
      setShowApproximation,
      setTimings,
      setIsLoading,
    ],
  );

  /* --------------------------------------------------------- /perfCheck */

  const perfCheck = useCallback(async () => {
    if (!meshes.nonSelect.length && !meshes.select.length) return;
    setIsLoading(true);
    const t0 = performance.now();
    try {
      const allUrls = [...meshes.nonSelect, ...meshes.select];
      const glb_blobs = await Promise.all(allUrls.map(blobUrlToBase64));

      const raw = JSON.stringify({
        glb_blobs,
        modelName: "generated_model",
      });

      const res = await fetch(`${API_BASE}/api/mujoco-perf`, {
        method: "POST",
        mode: "cors",
        headers: { "X-Content-Encoding": "gzip" },
        body: new Blob([pako.gzip(raw)], { type: "application/gzip" }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const buf = await res.arrayBuffer();
      const { rt_factor } = JSON.parse(
        pako.ungzip(new Uint8Array(buf), { to: "string" }),
      );

      setStatsApprox((s) => ({ ...s, rt_factor }));
      setTimings((t) => ({ ...t, perf: +(performance.now() - t0).toFixed(0) }));
    } catch (e) {
      console.error("[useBackend] perfCheck error:", e);
    } finally {
      setIsLoading(false);
    }
  }, [meshes, setStatsApprox, setTimings, setIsLoading]);

  /* --------------------------------------------------- public interface */

  return {
    decompose,
    runError,
    perfCheck,
  };
}
