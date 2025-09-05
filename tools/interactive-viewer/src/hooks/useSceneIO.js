// Copyright 2025 Intrinsic Innovation LLC

// -----------------------------------------------------------------------------
// useSceneIO.js – full import / export helper
// -----------------------------------------------------------------------------
// Keeps <App/> tidy by isolating every costly file‑IO task.  Exposes:
//   • exportSceneState()   → downloads `scene-with-state.gltf`
//   • importSceneState(f)  → hydrates React state from that file
//   • exportMeshesZip()    → downloads `meshes_export.zip`
// -----------------------------------------------------------------------------

import { useCallback } from "react";
import JSZip from "jszip";
import * as THREE from "three";
import { GLTFExporter } from "three/examples/jsm/exporters/GLTFExporter.js";

import {
  blobUrlToBase64,
  decodeBlobUrl,
  downloadBlobWithPrompt,
} from "../utils";

export default function useSceneIO(params) {
  /* ──────────────────────────── props in ─────────────────────────── */
  const {
    // ── mesh & label data ──
    meshes,
    meshLabels,
    boundingBoxes,
    hiddenHulls,

    // ── UI state flags / scalars ──
    showOriginal,
    showApproximation,
    showErrorForward,
    showErrorReverse,
    showBoxes,
    showCanvasSliders,
    wireframeMode,
    boxMode,
    isEditing,
    errorRange, // { min, max }
    defaultSize,
    defaultCoarseness,
    meshDimensions,
    statsOrig,
    statsApprox,
    timings,
    selectedHulls,
    watertight,

    // ── setters used only by *import* ──
    setOriginalUrl,
    checkWatertight,
    countMeshStats,
    setStatsOrig,
    setMeshes,
    setBoundingBoxes,
    setShowOriginal,
    setShowApproximation,
    setShowErrorForward,
    setShowErrorReverse,
    setShowBoxes,
    setShowCanvasSliders,
    setWireframeMode,
    setBoxMode,
    setIsEditing,
    setErrorRange, // preferred
    setErrorMin, // legacy support
    setErrorMax,
    setDefaultSize,
    setDefaultCoarseness,
    setMeshDimensions,
    setMeshLabels,
    setStatsApprox,
    setHiddenHulls,
    setSelectedHulls,
    setTimings,
    setErrsForward,
    setErrsReverse,
    setIsLoading,

    // ── raw URLs / error sets ──
    originalUrl,
    errsForward,
    errsReverse,
    setBoxToHullMap,
    boxToHullMap,
  } = params;

  /* ───────────────────────── helper utils ───────────────────────── */
  const maybeBase64 = async (url) => (url ? await blobUrlToBase64(url) : null);
  const encodeErrSet = async (set = { nonSelectErr: null, selectErrs: [] }) => {
    const arr = [];
    if (set.nonSelectErr) arr.push(await blobUrlToBase64(set.nonSelectErr));
    for (const u of set.selectErrs) arr.push(await blobUrlToBase64(u));
    return arr;
  };
  const decodeErrSet = (arr = []) => {
    const nonSelectErr = arr[0]
      ? decodeBlobUrl(arr[0], "application/octet-stream")
      : null;
    const selectErrs = arr
      .slice(1)
      .map((b64) => decodeBlobUrl(b64, "application/octet-stream"));
    return { nonSelectErr, selectErrs };
  };

  /* ──────────────────────── EXPORT: scene state ──────────────────────── */
  const exportSceneState = useCallback(async () => {
    try {
      setIsLoading?.(true);

      // encode originals / approximations / errors
      const originalMeshB64 = await maybeBase64(originalUrl);
      const errorMeshesForwardB64 = await encodeErrSet(errsForward);
      const errorMeshesReverseB64 = await encodeErrSet(errsReverse);
      const approxNonSelectB64 = await Promise.all(
        (meshes.nonSelect || []).map(blobUrlToBase64),
      );
      const approxSelectB64 = await Promise.all(
        (meshes.select || []).map(blobUrlToBase64),
      );

      // snapshot of *all* serialisable UI + mesh data
      const snapshot = {
        boundingBoxes: boundingBoxes.map((b) => ({
          center: [b.center.x, b.center.y, b.center.z],
          size: b.size,
          coarseness: b.coarseness,
        })),
        showOriginal,
        showApproximation,
        showErrorForward,
        showErrorReverse,
        showBoxes,
        showCanvasSliders,
        wireframeMode,
        boxMode,
        isEditing,
        errorMin: errorRange.min,
        errorMax: errorRange.max,
        defaultSize,
        defaultCoarseness,
        meshDimensions,
        meshLabels,
        statsOrig,
        statsApprox,
        timings,
        hiddenHulls: Array.from(hiddenHulls),
        selectedHulls,
        watertight,
        // raw mesh blobs
        originalMeshB64,
        errorMeshesForwardB64,
        errorMeshesReverseB64,
        approxNonSelectB64,
        approxSelectB64,
      };

      // store snapshot in .userData of a dummy Scene and export
      const scene = new THREE.Scene();
      scene.userData = snapshot;
      new GLTFExporter().parse(
        scene,
        (gltfJson) => {
          const blob = new Blob([JSON.stringify(gltfJson, null, 2)], {
            type: "application/json",
          });
          downloadBlobWithPrompt(blob, "scene-with-state.gltf");
        },
        { binary: false, embedImages: true, embedBuffers: true },
      );
    } finally {
      setIsLoading?.(false);
    }
  }, [
    meshes,
    boundingBoxes,
    hiddenHulls,
    showOriginal,
    showApproximation,
    showErrorForward,
    showErrorReverse,
    showBoxes,
    showCanvasSliders,
    wireframeMode,
    boxMode,
    isEditing,
    errorRange,
    defaultSize,
    defaultCoarseness,
    meshDimensions,
    meshLabels,
    statsOrig,
    statsApprox,
    timings,
    selectedHulls,
    watertight,
    originalUrl,
    errsForward,
    errsReverse,
  ]);

  /* ──────────────────────── IMPORT: scene state ──────────────────────── */
  const importSceneState = useCallback(
    async (fileOrEvent) => {
      if (!fileOrEvent) return;
      setIsLoading?.(true);
      try {
        const file =
          fileOrEvent instanceof File
            ? fileOrEvent
            : fileOrEvent?.target?.files?.[0];
        const text = await file.text();
        const gltfJson = JSON.parse(text);
        const s = gltfJson.scenes?.[0]?.extras ?? {};

        /* ——— Restore meshes ——— */
        if (s.originalMeshB64) {
          const url = decodeBlobUrl(s.originalMeshB64, "model/gltf-binary");
          setOriginalUrl?.(url);
          checkWatertight?.(url);
          countMeshStats?.(url).then(setStatsOrig);
        }

        // error sets (forward / reverse)
        const forwardArr = s.errorMeshesForwardB64 || [];
        const reverseArr = s.errorMeshesReverseB64 || [];
        // legacy single‑array fallback
        if (!forwardArr.length && s.errorMeshesB64?.length)
          forwardArr.push(...s.errorMeshesB64);

        setErrsForward?.(decodeErrSet(forwardArr));
        setErrsReverse?.(decodeErrSet(reverseArr));
        setShowErrorForward?.(!!forwardArr.length);
        setShowErrorReverse?.(!!reverseArr.length);

        // approximated hulls
        const decodedNonSelect = (s.approxNonSelectB64 || []).map((b64) =>
          decodeBlobUrl(b64, "model/gltf-binary"),
        );
        const decodedSelect = (s.approxSelectB64 || []).map((b64) =>
          decodeBlobUrl(b64, "model/gltf-binary"),
        );
        setMeshes?.({ nonSelect: decodedNonSelect, select: decodedSelect });
        setBoundingBoxes?.(
          (s.boundingBoxes || []).map((b) => ({
            center: new THREE.Vector3(...b.center),
            size: Array.isArray(b.size)
              ? { x: b.size[0], y: b.size[1], z: b.size[2] }
              : b.size,
            coarseness: b.coarseness,
          })),
        );

        // refresh approx stats once loaded
        if (countMeshStats && setStatsApprox) {
          Promise.all(
            [...decodedNonSelect, ...decodedSelect].map(countMeshStats),
          ).then((arr) => {
            const { vertices, faces } = arr.reduce(
              (tot, s) => ({
                vertices: tot.vertices + (s.vertices ?? 0),
                faces: tot.faces + (s.faces ?? 0),
              }),
              { vertices: 0, faces: 0 },
            );
            setStatsApprox((prev) => ({ ...prev, vertices, faces }));
          });
        }

        /* ——— Restore UI state ——— */

        setShowOriginal?.(s.showOriginal ?? true);
        setShowApproximation?.(s.showApproximation ?? false);
        setShowBoxes?.(s.showBoxes ?? true);
        setShowCanvasSliders?.(s.showCanvasSliders ?? false);
        setWireframeMode?.(s.wireframeMode ?? false);
        setBoxMode?.(s.boxMode ?? "center");
        setIsEditing?.(s.isEditing ?? true);
        if (setErrorRange) {
          setErrorRange({ min: s.errorMin ?? 0, max: s.errorMax ?? 0.01 });
        } else {
          setErrorMin?.(s.errorMin ?? 0);
          setErrorMax?.(s.errorMax ?? 0.01);
        }
        setDefaultSize?.(s.defaultSize ?? { x: 0.1, y: 0.1, z: 0.1 });
        setDefaultCoarseness?.(s.defaultCoarseness ?? 0);
        setMeshDimensions?.(s.meshDimensions ?? { x: 1, y: 1, z: 1 });
        setMeshLabels?.(
          s.meshLabels ?? {
            nonSelectConvex: [],
            nonSelectBox: [],
            selectConvex: [],
            selectBox: [],
          },
        );
        setStatsApprox?.(
          s.statsApprox ?? {
            vertices: null,
            faces: null,
            hulls: null,
            rt_factor: null,
          },
        );
        setHiddenHulls?.(new Set(s.hiddenHulls ?? []));
        setSelectedHulls?.(s.selectedHulls ?? []);
        setTimings?.(
          s.timings ?? { process: null, runError: null, perf: null },
        );
        if (setBoxToHullMap && s.meshLabels) {
          const newMap = { nonSelect: {}, select: {} };
          (s.meshLabels.nonSelectBox || []).forEach((boxIdx, hullIdx) => {
            if (boxIdx == null) return;
            if (!newMap.nonSelect[boxIdx]) newMap.nonSelect[boxIdx] = [];
            newMap.nonSelect[boxIdx].push(hullIdx);
          });
          (s.meshLabels.selectBox || []).forEach((boxIdx, hullIdx) => {
            if (boxIdx == null) return;
            if (!newMap.select[boxIdx]) newMap.select[boxIdx] = [];
            newMap.select[boxIdx].push(hullIdx);
          });
          setBoxToHullMap(newMap);
        }
      } catch (err) {
        console.error("[useSceneIO] importSceneState error:", err);
      } finally {
        setIsLoading?.(false);
      }
    },
    [
      decodeBlobUrl,
      decodeErrSet,
      countMeshStats,
      setIsLoading,
      setOriginalUrl,
      checkWatertight,
      setStatsOrig,
      setErrsForward,
      setErrsReverse,
      setMeshes,
      setStatsApprox,
      setBoundingBoxes,
      setShowOriginal,
      setShowApproximation,
      setShowBoxes,
      setShowCanvasSliders,
      setWireframeMode,
      setBoxMode,
      setIsEditing,
      setErrorRange,
      setErrorMin,
      setErrorMax,
      setDefaultSize,
      setDefaultCoarseness,
      setMeshDimensions,
      setMeshLabels,
      setHiddenHulls,
      setSelectedHulls,
      setTimings,
      setBoxToHullMap,
    ],
  );

  /* ──────────────────────── EXPORT: meshes zip ──────────────────────── */
  const exportMeshesZip = useCallback(async () => {
    if (!meshes.nonSelect.length && !meshes.select.length) return;

    const zip = new JSZip();
    const metadata = { nonSelect: [], select: [] };

    const addGroup = async (urls = [], convexArr = [], boxArr = [], group) => {
      for (let i = 0; i < urls.length; i++) {
        const resp = await fetch(urls[i]);
        if (!resp.ok) throw new Error(`mesh fetch failed (${resp.status})`);
        const buf = await resp.arrayBuffer();
        const fname = `${group}_${i}_convex${convexArr[i]}_box${boxArr[i]}.glb`;
        zip.file(fname, buf);
        metadata[group].push({
          filename: fname,
          convex_label: convexArr[i],
          box_index: boxArr[i],
        });
      }
    };

    await addGroup(
      meshes.nonSelect,
      meshLabels.nonSelectConvex,
      meshLabels.nonSelectBox,
      "nonSelect",
    );
    await addGroup(
      meshes.select,
      meshLabels.selectConvex,
      meshLabels.selectBox,
      "select",
    );

    zip.file("meshes_metadata.json", JSON.stringify(metadata, null, 2));
    downloadBlobWithPrompt(
      await zip.generateAsync({ type: "blob" }),
      "meshes_export.zip",
    );
  }, [meshes, meshLabels]);

  return { exportSceneState, importSceneState, exportMeshesZip };
}
