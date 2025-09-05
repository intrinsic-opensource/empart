// Copyright 2025 Intrinsic Innovation LLC

// src/hooks/useOriginalModel.js
// ------------------------------------------------------------
// React hook for managing the user's original mesh model and related metadata.
//
// Features:
// • Loads the default or user-supplied GLB file into the Three.js scene
// • Computes mesh stats (vertices, faces, bounding box)
// • Integrates with server APIs to check watertightness, voxelize, and decimate
// • Tracks loading state, file URL, and geometry properties for downstream use
// ------------------------------------------------------------

import { useCallback, useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import pako from "pako";
import { blobUrlToBase64, decodeBlobUrl } from "../utils";

// Environment vars (same you had in App.jsx)
const API_HOST = process.env.REACT_APP_API_HOST;
const API_PORT = process.env.REACT_APP_API_PORT;
const API_BASE = `${API_HOST}:${API_PORT}`;

export default function useOriginalModel({
  sceneRef,
  defaultPath = "/models/motor.glb",
}) {
  /* ──────── state ──────── */
  const [originalUrl, setOriginalUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [statsOrig, setStatsOrig] = useState({
    vertices: null,
    faces: null,
    hulls: null,
    rt_factor: null,
  });
  const [meshDimensions, setMeshDimensions] = useState({ x: 1, y: 1, z: 1 });
  const [watertight, setWatertight] = useState(null);

  /* ──────── helpers ──────── */
  const countMeshStats = useCallback(
    (url) =>
      new Promise((resolve, reject) => {
        const loader = new GLTFLoader();
        loader.load(
          url,
          (gltf) => {
            let vertices = 0;
            let faces = 0;
            gltf.scene.traverse((c) => {
              if (c.isMesh) {
                const g = c.geometry;
                const v = g.attributes.position.count;
                vertices += v;
                faces += g.index ? g.index.count / 3 : v / 3;
              }
            });
            resolve({ vertices, faces });
          },
          undefined,
          reject,
        );
      }),
    [],
  );

  /* derive bounding‑box size whenever the mesh changes */
  useEffect(() => {
    if (!originalUrl) return;
    const loader = new GLTFLoader();
    loader.load(originalUrl, (gltf) => {
      const bbox = new THREE.Box3().setFromObject(gltf.scene);
      const size = bbox.getSize(new THREE.Vector3());
      setMeshDimensions({ x: size.x, y: size.y, z: size.z });
    });
  }, [originalUrl]);

  /* -----------------------------------------------------------------------
   * REST helpers (voxelise / decimate / watertight / …)
   * -------------------------------------------------------------------- */

  /** Query the server to check whether `url` is watertight */
  const checkWatertight = useCallback(async (url) => {
    try {
      const b64 = await blobUrlToBase64(url);
      const body = pako.gzip(JSON.stringify({ glb_blob: b64 }));
      const res = await fetch(`${API_BASE}/api/check-watertight`, {
        method: "POST",
        mode: "cors",
        headers: { "X-Content-Encoding": "gzip" },
        body,
      });
      if (!res.ok) throw new Error(`Server ${res.status}`);
      const buf = await res.arrayBuffer();
      const json = JSON.parse(
        pako.ungzip(new Uint8Array(buf), { to: "string" }),
      );
      setWatertight(Boolean(json.watertight));
    } catch (err) {
      console.warn("Watertight check failed:", err);
      setWatertight(null);
    }
  }, []);

  /** Ask the backend to voxelise → watertight‑ify the mesh */
  const handleWatertight = useCallback(
    async (pitch) => {
      if (!originalUrl) return;
      setIsLoading(true);
      try {
        const glbB64 = await blobUrlToBase64(originalUrl);
        const payload = { glb_blob: glbB64 };
        if (Number.isFinite(pitch)) payload.pitch = pitch;

        const res = await fetch(`${API_BASE}/api/voxelize`, {
          method: "POST",
          mode: "cors",
          headers: { "X-Content-Encoding": "gzip" },
          body: pako.gzip(JSON.stringify(payload)),
        });
        if (!res.ok) throw new Error(`Server ${res.status}`);

        const buf = await res.arrayBuffer();
        const out = JSON.parse(
          pako.ungzip(new Uint8Array(buf), { to: "string" }),
        );
        const newUrl = decodeBlobUrl(out.mesh, "model/gltf-binary");

        setOriginalUrl(newUrl);
        await checkWatertight(newUrl);

        if (out.stats?.vertices) {
          setStatsOrig({
            vertices: out.stats.vertices,
            faces: out.stats.triangles,
            hulls: null,
            rt_factor: null,
          });
        } else {
          countMeshStats(newUrl).then(setStatsOrig);
        }
      } catch (err) {
        console.error("Watertight:", err);
      } finally {
        setIsLoading(false);
      }
    },
    [originalUrl, checkWatertight, countMeshStats],
  );

  /** Ask the backend to decimate the mesh to `targetTris` triangles */
  const handleDecimate = useCallback(
    async (targetTris) => {
      if (!originalUrl) return;
      setIsLoading(true);
      try {
        const glbB64 = await blobUrlToBase64(originalUrl);
        const payload = { glb_blob: glbB64, target_tris: targetTris };
        const res = await fetch(`${API_BASE}/api/decimate`, {
          method: "POST",
          mode: "cors",
          headers: { "X-Content-Encoding": "gzip" },
          body: new Blob([pako.gzip(JSON.stringify(payload))], {
            type: "application/gzip",
          }),
        });
        if (!res.ok) throw new Error(`Server ${res.status}`);

        const buf = await res.arrayBuffer();
        const out = JSON.parse(
          pako.ungzip(new Uint8Array(buf), { to: "string" }),
        );
        const newUrl = decodeBlobUrl(out.mesh, "model/gltf-binary");

        setOriginalUrl(newUrl);
        await checkWatertight(newUrl);

        if (out.stats?.vertices) {
          setStatsOrig({
            vertices: out.stats.vertices,
            faces: out.stats.triangles,
            hulls: null,
            rt_factor: null,
          });
        } else {
          countMeshStats(newUrl).then(setStatsOrig);
        }
      } catch (err) {
        console.error("handleDecimate:", err);
      } finally {
        setIsLoading(false);
      }
    },
    [originalUrl, checkWatertight, countMeshStats],
  );

  /* -----------------------------------------------------------------------
   * User‑facing helpers
   * -------------------------------------------------------------------- */

  /** Open a “Choose File” <input> and load whatever the user picked */
  const handleFileInput = useCallback(
    (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        const url = URL.createObjectURL(file);
        setOriginalUrl(url); // overwrite current mesh
        checkWatertight(url);
        countMeshStats(url).then(setStatsOrig);
      };
      reader.readAsText(file);
    },
    [checkWatertight, countMeshStats],
  );

  /** (Re)load the default GLB shipped with the app */
  const loadDefault = useCallback(async () => {
    const resp = await fetch(defaultPath);
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    setOriginalUrl(url);
    await checkWatertight(url);
    countMeshStats(url).then(setStatsOrig);
  }, [defaultPath, checkWatertight, countMeshStats]);

  // Run once at mount—mimics your previous `useEffect`
  useEffect(() => {
    loadDefault();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  /* expose everything the component cared about */
  return {
    /* state */
    originalUrl,
    isLoading,
    setIsLoading,
    statsOrig,
    setStatsOrig,
    meshDimensions,
    setMeshDimensions,
    watertight,

    /* setters / helpers */
    setOriginalUrl,
    countMeshStats,
    checkWatertight,
    handleWatertight,
    handleDecimate,
    loadDefault,
    handleFileInput,
  };
}
