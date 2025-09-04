// Copyright 2025 Intrinsic Innovation LLC

// ---------------------------------------------------------------------------
// hooks/useMeshes.js
// ---------------------------------------------------------------------------
// React hook for managing 3D hull meshes and their metadata in a Three.js scene.
//
// Features:
// • Tracks selectable mesh groups and their corresponding volumes
// • Handles hull selection, hiding, deletion, and merging via ConvexGeometry
// • Exports merged hulls to GLB blobs for use across the app
// • Exposes mesh state and actions for integration with panels and canvas
import { useState, useCallback, useRef, useEffect } from "react";
import * as THREE from "three";
import { ConvexGeometry } from "three/examples/jsm/geometries/ConvexGeometry.js";
import { GLTFExporter } from "three/examples/jsm/exporters/GLTFExporter.js";

import { computeMergedHullVolume, pointsFromMesh } from "../utils/volume";

// Generate the canonical mesh name used throughout the app
const meshName = (group, idx) => `hull-${group}-${idx}`;

/**
 * useMeshes
 * ---------
 * All mesh‑related state and actions (load, hide, select, merge, delete).
 * Call it once in <App/> and pass the returned props to the panels / scene.
 */
export default function useMeshes(sceneRef) {
  /* ────────────────────────────────────────────
     Reactive state
     ──────────────────────────────────────────── */
  const [meshes, setMeshes] = useState({ nonSelect: [], select: [] });
  const [hullVolumes, setHullVolumes] = useState({ nonSelect: [], select: [] });
  const [selectedHulls, setSelectedHulls] = useState([]); // [{meshKey,group,idx,stats}]
  const [hiddenHulls, setHiddenHulls] = useState(new Set());

  // Handy ref to avoid stale closures inside callbacks
  const selectedHullsRef = useRef(selectedHulls);
  useEffect(() => {
    selectedHullsRef.current = selectedHulls;
  }, [selectedHulls]);

  /* ────────────────────────────────────────────
     Selection helpers
     ──────────────────────────────────────────── */
  const toggleHull = useCallback((meshKey, group, idx, stats) => {
    const alreadySelected = selectedHullsRef.current.some(
      (h) => h.meshKey === meshKey,
    );
    if (window.location.search.includes("e2e")) {
      console.e2e(
        `[E2E] ${alreadySelected ? "Deselected" : "Selected"} hull → ` +
          `hull-${group}-${idx}, vol=${stats.volume.toFixed(3)}`,
      );
    }
    setSelectedHulls((cur) =>
      cur.some((h) => h.meshKey === meshKey)
        ? cur.filter((h) => h.meshKey !== meshKey)
        : [...cur, { meshKey, group, idx, stats }],
    );
  }, []);

  const clearHullSelections = useCallback(() => setSelectedHulls([]), []);

  /* ────────────────────────────────────────────
     Delete a hull entirely
     ──────────────────────────────────────────── */
  const deleteHull = useCallback((group, idx, meshKey) => {
    setSelectedHulls([]);

    setMeshes((m) => ({
      ...m,
      [group]: m[group].filter((_, i) => i !== idx),
    }));

    setHullVolumes((v) => ({
      ...v,
      [group]: v[group].filter((_, i) => i !== idx),
    }));

    setHiddenHulls((set) => {
      const next = new Set(set);
      next.delete(meshKey);
      return next;
    });
  }, []);

  /* ────────────────────────────────────────────
     Merge currently‑selected hulls
     ──────────────────────────────────────────── */
  const handleMergeHulls = useCallback(() => {
    const picks = selectedHullsRef.current;
    if (picks.length < 2 || !sceneRef.current) return;

    // 1) THREE.Mesh instances of the picks
    const meshesToMerge = picks
      .map((h) => sceneRef.current.getObjectByName(meshName(h.group, h.idx)))
      .filter(Boolean);
    if (meshesToMerge.length < 2) return;

    // 2) New convex geometry
    const mergedVolume = computeMergedHullVolume(meshesToMerge);
    const pts = meshesToMerge.flatMap(pointsFromMesh);
    const geom = new ConvexGeometry(
      pts.map(([x, y, z]) => new THREE.Vector3(x, y, z)),
    );
    geom.computeVertexNormals();

    // 3) Material with random hue for easy visual distinction
    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color().setHSL(Math.random(), 0.8, 0.5),
      transparent: true,
      opacity: 0.8,
      flatShading: true,
      roughness: 0.9,
      metalness: 0.1,
    });

    // 4) Wrap in a mesh + export to GLB (so rest of app keeps using URLs)
    const tempMesh = new THREE.Mesh(geom, material);
    new GLTFExporter().parse(
      tempMesh,
      (gltf) => {
        const blob = new Blob([gltf], { type: "model/gltf-binary" });
        const url = URL.createObjectURL(blob);

        // Strip originals & push the merged version
        setMeshes((prev) => {
          const next = {
            nonSelect: prev.nonSelect.filter(
              (_, i) =>
                !picks.some((h) => h.group === "nonSelect" && h.idx === i),
            ),
            select: prev.select.filter(
              (_, i) => !picks.some((h) => h.group === "select" && h.idx === i),
            ),
          };
          next.nonSelect = [...next.nonSelect, url];
          return next;
        });

        setHullVolumes((prev) => {
          const next = {
            nonSelect: prev.nonSelect.filter(
              (_, i) =>
                !picks.some((h) => h.group === "nonSelect" && h.idx === i),
            ),
            select: prev.select.filter(
              (_, i) => !picks.some((h) => h.group === "select" && h.idx === i),
            ),
          };
          next.nonSelect = [...next.nonSelect, mergedVolume];
          return next;
        });

        clearHullSelections();
      },
      (err) => console.error("GLTF export failed", err),
      { binary: true, embedBuffers: true, embedImages: true },
    );
  }, [sceneRef, clearHullSelections]);

  /* ────────────────────────────────────────────
     API surface
     ──────────────────────────────────────────── */
  return {
    // state
    meshes,
    hullVolumes,
    selectedHulls,
    hiddenHulls,

    // actions
    setMeshes,
    setHullVolumes,
    toggleHull,
    deleteHull,
    clearHullSelections,
    handleMergeHulls,
    setHiddenHulls,
    setSelectedHulls,
  };
}
